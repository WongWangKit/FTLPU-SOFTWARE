// Tile-level software pipeline for the native planar adjacent-pair RoPE.
// SXM serializes ingress(i+1) and egress(i-1) while VXM computes tile i.
// All bounds below use real SR distances, coarse-domain retirement and the
// four-superlane drain; neither an elastic SXM nor interleaved MEM macros
// are assumed.
void emitPipelinedPlanarRotary(I ni) {
  auto &node=planner.nodes[ni];const auto &words=rotaryWords.at(ni);
  const I rows=node.inputs[0].shape[0],blocks=node.outputs.size();
  V tasks[2];for(I b=0;b<blocks;++b)tasks[words[b*rows].side].push_back(b);
  if(tasks[0].size()!=tasks[1].size())throw std::runtime_error("unbalanced planar rotary tiles");
  // Even outputs, then odd outputs: two ascending X scans feed consecutive
  // products, and strided stores restore the original feature order.
  const I tiles=tasks[0].size(),pitch=2,tileCompute=32*pitch;
  const I begin=s.cursor,drain=s.target.throughput().tile_rows;
  auto home=[&](mlir::DictionaryAttr p,I side){return s.change(p,J{{"hemisphere",side?"west":"east"}});};
  auto rd=[&](I slice){return *s.target.transport_latency(fc::target::StreamEndpoint::Mem,
      fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,slice);};
  auto wr=[&](I slice){return *s.target.transport_latency(fc::target::StreamEndpoint::SxmResult,
      fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,slice);};
  // VXM uses local 0..1 for X and 4..7 for FP32 coefficients. Local 2..3
  // are free on both hemispheres, including while the SXM output traverses
  // MEM on its way towards VXM. VXM results travel in the other direction.
  auto sxmStream=[](I side){return 34+side*16;};
  auto transpose=[&](I cycle,I side) {
    s.emit("sxm",J{{"cycle",cycle},{"hemisphere",side},{"opcode","transpose"},
        {"source_streams",s.array({16,17})},{"destination_streams",s.array({sxmStream(side),sxmStream(side)+1})},
        {"permute_map",s.array(range(32))},{"weight_layout","wide_matrix"},
        {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
  };
  llvm::json::Array report;
  for(I token=0;token<rows;token+=32) {
    V input(tiles),compute(tiles),output(tiles),ready(tiles);
    I nextSxm=s.cursor+32,end=s.cursor;
    // Both halves use the same VXM configuration. Plan their common cycle
    // from the slowest byte-plane route before emitting any operation.
    auto scheduleOutput=[&](I slot) {
      I capture=nextSxm;const auto y=node.workspaces[2+slot%2];
      for(I plane=0;plane<2;++plane) {
        const I slice=slices(y)[plane];
        const I lastWrite=compute[slot]+4+slice/s.group+31*pitch;
        capture=std::max(capture,lastWrite+drain+rd(slice));
      }
      output[slot]=capture;nextSxm=capture+32;ready[slot]=capture;
      for(I side=0;side<2;++side) {
        const I destSide=1-side,block=tasks[side][slot];
        auto src=home(y,destSide),dest=node.out[block];
        transpose(capture,destSide);
        for(I byte=0;byte<2;++byte) {
          s.mem(capture-rd(slices(src)[byte]),src,byte,"read",field(src,"base_row"),16+byte,{32,1,1},{1,0,0});
          const I write=capture+35+wr(slices(dest)[byte]);
          s.mem(write,dest,byte,"write",field(dest,"base_row")+token,sxmStream(destSide)+byte,{32,1,1},{1,0,0});
          ready[slot]=std::max(ready[slot],write+32+drain);
        }
      }
      end=std::max(end,ready[slot]);
    };
    for(I slot=0;slot<tiles;++slot) {
      auto x=node.workspaces[slot%2],y=node.workspaces[2+slot%2];
      I capture=nextSxm;
      if(slot>=2)for(I plane=0;plane<2;++plane) {
        const I slice=slices(x)[plane];
        const I lastRead=compute[slot-2]+31*pitch-slice/s.group;
        capture=std::max(capture,lastRead+drain-35-wr(slice));
      }
      input[slot]=capture;nextSxm=capture+32;
      I start=slot?compute[slot-1]+tileCompute:s.cursor;
      for(I plane=0;plane<2;++plane) {
        const I slice=slices(x)[plane];
        const I lastWrite=capture+35+wr(slice)+31;
        start=std::max(start,lastWrite+drain+1+slice/s.group);
        if(slot>=2) {
          const I ys=slices(y)[plane];
          const I lastRead=output[slot-2]-rd(ys)+31;
          start=std::max(start,lastRead+drain-4-ys/s.group);
        }
      }
      compute[slot]=start;
      for(I side=0;side<2;++side) {
        const I block=tasks[side][slot];const auto src=words[block*rows+token];
        auto dest=home(x,side);transpose(capture,side);
        for(I r=0;r<32;++r)if(words[block*rows+token+r].p!=src.p || words[block*rows+token+r].address!=src.address+r)
          throw std::runtime_error("non-affine pipelined rotary token tile");
        for(I byte=0;byte<2;++byte) {
          s.mem(capture-rd(slices(src.p)[byte]),src.p,byte,"read",src.address,16+byte,{32,1,1},{1,0,0});
          s.mem(capture+35+wr(slices(dest)[byte]),dest,byte,"write",field(dest,"base_row"),sxmStream(side)+byte,
              {32,1,1},{1,0,0});
        }
      }
      if(slot)scheduleOutput(slot-1);
    }
    scheduleOutput(tiles-1);
    // A single C0 configuration with FU repetition spans each contiguous
    // tile run. A new configuration is needed only after a dependency gap.
    for(I first=0;first<tiles;) {
      I stop=first+1;while(stop<tiles && compute[stop]==compute[first]+(stop-first)*tileCompute)++stop;
      s.emit("vxm",J{{"cycle",compute[first]-1},{"queue",0},{"opcode","multiply"},{"chain_depth",2},
          {"lhs_kind","stream_bf16"},{"lhs_index",0},{"lhs_immediate",0.0},
          {"rhs_kind","stream_f32"},{"rhs_index",4},{"rhs_immediate",0.0},
          {"lhs_stream_source","local"},{"rhs_stream_source","local"},
          {"input_hemisphere","east"},{"output_hemisphere","west"},{"cast_target","fp32"},{"output_stream",-1},
          {"repeat_count",2},{"repeat_interval",1},{"wave_count",(stop-first)*32},{"wave_interval",pitch}});
      first=stop;
    }
    for(I slot=0;slot<tiles;++slot) {
      const I start=compute[slot];
      for(I d=0;d<32;++d)for(I term=0;term<2;++term)
        s.emit("vxm",J{{"cycle",start-1+d*pitch+term},{"queue",1},{"opcode","add"},{"chain_depth",2},
            {"lhs_kind","previous"},{"lhs_index",0},{"lhs_immediate",0.0},
            {"rhs_kind","accumulator"},{"rhs_index",0},{"rhs_immediate",0.0},
            {"input_hemisphere","east"},{"output_hemisphere","west"},
            {"cast_target",term?"bf16":"fp32"},{"output_stream",0},
            {"accumulator_reset",term==0},{"accumulator_write",true},{"accumulator_emit",term==1},
            {"repeat_count",1},{"repeat_interval",1}});
      for(I side=0;side<2;++side) {
        const I block=tasks[side][slot],coefBlock=block*32%node.planarRotaryDim;
        auto x=home(node.workspaces[slot%2],side),coef=home(node.in[0],side),y=home(node.workspaces[2+slot%2],1-side);
        for(I byte=0;byte<2;++byte)
          s.mem(start-1-slices(x)[byte]/s.group,x,byte,"read",field(x,"base_row"),32+side*16+byte,
              {32,2,1},{1,0,0},{1,32,1});
        for(I byte=0;byte<4;++byte)
          s.mem(start-1-slices(coef)[byte]/s.group,coef,byte,"read",field(coef,"base_row")+(token/32)*2*node.planarRotaryDim+2*coefBlock,
              32+side*16+4+byte,{64,1,1},{1,0,0});
        for(I byte=0;byte<2;++byte)
          s.mem(start+4+slices(y)[byte]/s.group,y,byte,"write",field(y,"base_row"),side*8+byte,
              {16,2,1},{2,1,0},{pitch,16*pitch,1});
      }
      report.push_back(J{{"token_tile",token/32},{"slot",slot},{"buffer",slot%2},
          {"input_sxm",input[slot]},{"vxm_start",start},{"vxm_end",start+tileCompute},
          {"output_sxm",output[slot]},{"output_ready",ready[slot]}});
    }
    s.cursor=end;
    s.emit("timeline",J{{"name","imem.drain."+node.name},{"start",end-1},{"end",end}});
  }
  s.emit("timeline",J{{"name",node.name},{"start",begin},{"end",s.cursor}});
  llvm::json::Array buffers;
  for(I i=0;i<4;++i)buffers.push_back(J{{"role",i<2?"input":"output"},{"slot",i%2},
      {"slices",s.array(slices(node.workspaces[i]))},{"bank",field(node.workspaces[i],"bank")},
      {"base_row",field(node.workspaces[i],"base_row")},{"words_per_plane",32}});
  nodeRanges.push_back(J{{"node",ni},{"name",node.name},{"native_planar_rotary",true},{"tile_pipeline",true},
      {"start",begin},{"end",s.cursor},{"tokens",rows},{"rotary_dim",node.planarRotaryDim},
      {"intermediate_product_mem_writes",0},{"accumulation","fp32"},{"output","bf16_rne"},
      {"feature_order","even_then_odd_per_tile"},{"x_read_domains_per_tile_per_plane",1},
      {"feature_interval",pitch},{"tile_compute_interval",tileCompute},{"buffers",std::move(buffers)},
      {"tiles",std::move(report)},{"transpose_input_vectors",blocks*rows},{"transpose_output_vectors",blocks*rows}});
}
