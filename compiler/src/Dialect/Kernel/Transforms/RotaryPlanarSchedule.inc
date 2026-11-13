// Native adjacent-pair RoPE. A tile is transposed once in each direction;
// the two products never leave the FP32 VXM accumulator. All storage is
// ordinary planar, and every MEM domain runs to completion on its ICU.
struct RotaryWord { mlir::DictionaryAttr p; I address, side; };
std::map<I,std::vector<RotaryWord>> rotaryWords;
bool rotaryTileOverlap() const {
  const auto policy=planner.policies.getAs<mlir::BoolAttr>("ftlpu.rope_tile_overlap");
  return policy && policy.getValue();
}

void planPlanarRotary(I ni) {
  auto &node=planner.nodes[ni];const auto &value=node.inputs[0];
  const I rows=value.shape[0],blocks=value.shape[1]/32;
  std::vector<mlir::DictionaryAttr> busy;
  auto &words=rotaryWords[ni];
  for(I block=0;block<blocks;++block)for(I row=0;row<rows;++row) {
    const I at=row*value.shape[1]+block*32;const auto ref=value.refs.at(at);
    auto p=cache.at({ref.node,ref.output});
    const auto &shape=planner.nodes[ref.node].outputs[ref.output].shape;
    bool valid=shape.size()==2 && shape[1]%32==0 && ref.index%32==0 && slices(p).size()==2 && !p.get("storage_shape");
    for(I lane=0;valid && lane<32;++lane) {
      const auto r=value.refs[at+lane];valid=r.node==ref.node && r.output==ref.output && r.index==ref.index+lane;
    }
    if(!valid)throw std::runtime_error("planar rotary requires contiguous BF16 producer words");
    const I column=ref.index%shape[1];I side,address;
    if(text(p,"kind")=="fp16_hemisphere_pair_planar") {
      side=column/32%2;address=field(p,"base_row")+(column/64)*shape[0]+ref.index/shape[1];
    } else if(text(p,"kind")=="fp16_pair_planar") {
      side=text(p,"hemisphere")=="both"?block%2:I(text(p,"hemisphere")=="west");
      address=field(p,"base_row")+(column/32)*shape[0]+ref.index/shape[1];
    } else throw std::runtime_error("unsupported native planar rotary input layout");
    p=s.change(p,J{{"kind","fp16_pair_planar"},{"hemisphere",side?"west":"east"}});
    words.push_back({p,address,side});if(!llvm::is_contained(busy,p))busy.push_back(p);
  }
  const auto coefRef=node.inputs[1].refs.front();
  auto coefficient=cache.at({coefRef.node,coefRef.output});busy.push_back(coefficient);
  node.in={coefficient};
  const auto readQueues=busy;
  for(I block=0;block<blocks;++block) {
    const I side=words[block*rows].side;
    for(I row=0;row<rows;++row)if(words[block*rows+row].side!=side)
      throw std::runtime_error("rotary word must retain one hemisphere across its token tile");
    // During tile overlap, final SXM writes also coexist with producer and
    // coefficient reads. Separate ICU queues, not merely row ranges.
    node.out.push_back(allocate({ni,block},node.outputs[block],side?"east":"west",1,{},
        rotaryTileOverlap()?readQueues:std::vector<mlir::DictionaryAttr>{}));
    busy.push_back(node.out.back());
  }
  Tensor scratch;scratch.shape={rotaryTileOverlap()?32:blocks*32,32};scratch.dtype="bf16";
  if(rotaryTileOverlap()) {
    // X_A, X_B, Y_A, Y_B: one 32x32 BF16 tile in each hemisphere.
    for(I slot=0;slot<4;++slot) {
      const Key key{ni,-1-slot};last[key]=ni;
      auto p=allocate(key,scratch,"both",1,{},busy);
      node.workspaces.push_back(p);busy.push_back(p);
    }
    return;
  }
  last[{ni,-1}]=last[{ni,-2}]=ni;
  auto x=allocate({ni,-1},scratch,"both",1,{},busy);busy.push_back(x);
  auto y=allocate({ni,-2},scratch,"both",1,{},busy);
  node.workspaces={x,y};
}

void emitPlanarRotary(I ni) {
  if(rotaryTileOverlap()) { emitPipelinedPlanarRotary(ni); return; }
  auto &node=planner.nodes[ni];auto &words=rotaryWords.at(ni);
  const I rows=node.inputs[0].shape[0],blocks=node.outputs.size();
  V tasks[2];for(I block=0;block<blocks;++block)tasks[words[block*rows].side].push_back(block);
  const I tiles=std::max(tasks[0].size(),tasks[1].size());
  if(tasks[0].size()!=tasks[1].size())throw std::runtime_error("native rotary needs balanced hemisphere tiles");
  // Even outputs followed by odd outputs use two ascending scans of X.
  // One MEM domain feeds the consecutive two-product VXM chain per tile.
  const I begin=s.cursor,pitch=2;
  auto sideHome=[&](mlir::DictionaryAttr p,I side){return s.change(p,J{{"hemisphere",side?"west":"east"}});};
  auto transpose=[&](I cycle,I side) {
    s.emit("sxm",J{{"cycle",cycle},{"hemisphere",side},{"opcode","transpose"},
        {"source_streams",s.array({16,17})},{"destination_streams",s.array({32,33})},
        {"permute_map",s.array(range(32))},{"weight_layout","wide_matrix"},
        {"repeat_count",1},{"repeat_interval",32},{"wave_count",1},{"wave_interval",32}});
  };
  auto rdLatency=[&](I slice){return *s.target.transport_latency(fc::target::StreamEndpoint::Mem,
      fc::target::StreamEndpoint::SxmInput,fc::target::StreamDirection::East,slice);};
  auto wrLatency=[&](I slice){return *s.target.transport_latency(fc::target::StreamEndpoint::SxmResult,
      fc::target::StreamEndpoint::Mem,fc::target::StreamDirection::West,slice);};
  for(I token=0;token<rows;token+=32) {
    const I ingress=s.cursor+32;I end=ingress;
    for(I side=0;side<2;++side)for(I slot=0;slot<tiles;++slot) {
      const I block=tasks[side][slot],capture=ingress+slot*32;
      const auto src=words[block*rows+token];auto x=sideHome(node.workspaces[0],side);
      transpose(capture,side);
      for(I byte=0;byte<2;++byte) {
        // Prove the whole producer row domain before forming a Read.
        for(I r=0;r<32;++r)if(words[block*rows+token+r].p!=src.p || words[block*rows+token+r].address!=src.address+r)
          throw std::runtime_error("non-affine native rotary token tile");
        s.mem(capture-rdLatency(slices(src.p)[byte]),src.p,byte,"read",src.address,16+byte,{32,1,1},{1,0,0});
        const I write=capture+35+wrLatency(slices(x)[byte]);
        s.mem(write,x,byte,"write",field(x,"base_row")+slot*32,32+byte,{32,1,1},{1,0,0});
        end=std::max(end,write+36);
      }
    }
    const I start=end+16;
    // Both hemispheres share a configuration, but consume their own words.
    J mul{{"cycle",start-1},{"queue",0},{"opcode","multiply"},{"chain_depth",2},
        {"lhs_kind","stream_bf16"},{"lhs_index",0},{"lhs_immediate",0.0},
        {"rhs_kind","stream_f32"},{"rhs_index",4},{"rhs_immediate",0.0},
        {"lhs_stream_source","local"},{"rhs_stream_source","local"},
        {"input_hemisphere","east"},{"output_hemisphere","west"},
        {"cast_target","fp32"},{"output_stream",-1},
        {"repeat_count",2},{"repeat_interval",1},{"wave_count",tiles*32},{"wave_interval",pitch}};
    s.emit("vxm",std::move(mul));
    for(I f=0;f<tiles*32;++f)for(I term=0;term<2;++term) {
      s.emit("vxm",J{{"cycle",start-1+f*pitch+term},{"queue",1},{"opcode","add"},{"chain_depth",2},
          {"lhs_kind","previous"},{"lhs_index",0},{"lhs_immediate",0.0},
          {"rhs_kind","accumulator"},{"rhs_index",0},{"rhs_immediate",0.0},
          {"input_hemisphere","east"},{"output_hemisphere","west"},
          {"cast_target",term?"bf16":"fp32"},{"output_stream",0},
          {"accumulator_reset",term==0},{"accumulator_write",true},{"accumulator_emit",term==1},
          {"repeat_count",1},{"repeat_interval",1}});
    }
    end=start;
    for(I side=0;side<2;++side) {
      auto x=sideHome(node.workspaces[0],side),coef=sideHome(node.in[0],side),y=sideHome(node.workspaces[1],1-side);
      for(I slot=0;slot<tiles;++slot) {
        const I block=tasks[side][slot],coefficientBlock=(block*32)%node.planarRotaryDim;
        for(I byte=0;byte<2;++byte)
          s.mem(start+slot*32*pitch-1-slices(x)[byte]/s.group,x,byte,"read",field(x,"base_row")+slot*32,32+side*16+byte,
              {32,2,1},{1,0,0},{1,32,1});
        for(I byte=0;byte<4;++byte)
          s.mem(start+slot*32*pitch-1-slices(coef)[byte]/s.group,coef,byte,"read",
              field(coef,"base_row")+(token/32)*(2*node.planarRotaryDim)+2*coefficientBlock,32+side*16+4+byte,
              {64,1,1},{1,0,0});
        for(I byte=0;byte<2;++byte) {
          const I write=start+slot*32*pitch+4+slices(y)[byte]/s.group;
          s.mem(write,y,byte,"write",field(y,"base_row")+slot*32,side*8+byte,{16,2,1},{2,1,0},{pitch,16*pitch,1});
          end=std::max(end,write+31*pitch+5);
        }
      }
    }
    const I egress=end+16;
    for(I side=0;side<2;++side)for(I slot=0;slot<tiles;++slot) {
      const I block=tasks[side][slot],destSide=1-side,capture=egress+slot*32;
      auto y=sideHome(node.workspaces[1],destSide),out=node.out[block];
      transpose(capture,destSide);
      for(I byte=0;byte<2;++byte) {
        s.mem(capture-rdLatency(slices(y)[byte]),y,byte,"read",field(y,"base_row")+slot*32,16+byte,{32,1,1},{1,0,0});
        const I write=capture+35+wrLatency(slices(out)[byte]);
        s.mem(write,out,byte,"write",field(out,"base_row")+token,32+byte,{32,1,1},{1,0,0});
        end=std::max(end,write+36);
      }
    }
    s.cursor=end;
    s.emit("timeline",J{{"name","imem.drain."+node.name},{"start",end-1},{"end",end}});
  }
  s.emit("timeline",J{{"name",node.name},{"start",begin},{"end",s.cursor}});
  nodeRanges.push_back(J{{"node",ni},{"name",node.name},{"native_planar_rotary",true},
      {"start",begin},{"end",s.cursor},{"tokens",rows},{"rotary_dim",node.planarRotaryDim},
      {"intermediate_product_mem_writes",0},{"accumulation","fp32"},{"output","bf16_rne"},
      {"feature_order","even_then_odd_per_tile"},{"x_read_domains_per_tile_per_plane",1},
      {"feature_interval",pitch},{"transpose_input_vectors",blocks*rows},{"transpose_output_vectors",blocks*rows}});
}
