// A 32-key Flash attention tile is one scheduling unit. The transpose of
// PV, V^T P, keeps query rows in lanes through the FP32 denominator epilogue.
// No expanded coefficient tensor or intermediate layout-copy kernel exists.

static mlir::FailureOr<mlir::Value> emitFlashTileCore(
    mlir::IRRewriter& r, AttentionTaskGraph op,
    const target::LPUTargetModel& target, int64_t outputIndex)
{
    using namespace attention_detail;
    using E=target::StreamEndpoint;
    using D=target::StreamDirection;
    const auto loc=op.getLoc();
    const auto plan=op.getMemoryPlan();
    const auto home=[&](llvm::StringRef name) {return plan.getAs<mlir::DictionaryAttr>(name);};
    const auto number=[](mlir::DictionaryAttr p,llvm::StringRef key) {return p.getAs<mlir::IntegerAttr>(key).getInt();};
    const auto id=[&](mlir::DictionaryAttr p,int byte) {return llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[byte]).getInt();};
    const auto route=[&](E from,E to,D direction,int64_t slice=0) {
        auto latency=target.transport_latency(from,to,direction,slice);
        if(!latency)throw std::runtime_error("Flash tile requires a missing hardware route");
        return *latency;
    };
    const int64_t width=op.getHeadDim(), blocks=width/32, group=op.getQueryHeads()/op.getKvHeads();
    const int64_t queryRows=op.getSeqLen()*group, tiles=queryRows/32, pairs=op.getKvHeads()/2;
    const bool hasSink=op.core->getNumOperands()==4;
    const auto order=op.config().getAs<mlir::BoolAttr>("query_head_major");
    const bool headMajor=order && order.getValue();
    const int64_t pipeline=target.mxm_first_result_latency();
    const int64_t period=std::max<int64_t>(32,target.throughput().mxm0_accumulator_latency);
    const int64_t sxmLatency=32+target.throughput().tile_rows-1;
    const int64_t toWeight=route(E::SxmResult,E::MxmWeight,D::East);
    const int64_t toSxm=route(E::VxmResult,E::SxmInput,D::East);
    const int64_t returning=route(E::MxmResult,E::Mem,D::West)
        +route(E::Mem,E::VxmInput,D::West);
    llvm::SmallVector<int64_t> identity;for(int i=0;i<32;++i)identity.push_back(i);
    r.setInsertionPoint(op.core);
    for(int i=0;i<3;++i) {
        auto value=op.core->getOperand(i);
        const char* names[]={"query","key","value"};
        createBinding(r,loc,value,llvm::cast<mlir::BlockArgument>(value).getArgNumber(),"input","activation",
            llvm::cast<mlir::RankedTensorType>(value.getType()),home(names[i]),names[i]);
    }
    if(hasSink) {
        auto value=op.core->getOperand(3);
        createBinding(r,loc,value,llvm::cast<mlir::BlockArgument>(value).getArgNumber(),"input","constant",
            llvm::cast<mlir::RankedTensorType>(value.getType()),home("sink"),"attention.sink");
    }
    if(op.getCausal())createBinding(r,loc,{},0,"internal","constant",
        mlir::RankedTensorType::get({31,32},r.getBF16Type()),home("causal_mask"),"flash.causal_mask","causal_mask",r.getDictionaryAttr({}));
    const auto mem=[&](mlir::DictionaryAttr p,int byte,int side,int64_t cycle,llvm::StringRef opcode,
                       int64_t address,int stream,int64_t count,int64_t stride=1,
                       int64_t waves=1,int64_t interval=1,int64_t waveStride=0) {
        direct_domain_detail::emitMem3D(r,loc,target,cycle,side*target.memory().slices_per_hemisphere+id(p,byte),
            opcode,number(p,"base_row")+address,stream,count,1,stride,waves,interval,waveStride,1,1,0,-1,number(p,"bank"));
    };
    const auto alu=[&](int64_t begin,int col,llvm::StringRef kind,llvm::StringRef lhs,int li,
                       llvm::StringRef rhs,int ri,float immediate,llvm::StringRef format,int output,
                       int64_t count,int depth,bool reset=false,bool accumulate=false,bool emit=true) {
        auto x=ffn_detail::create_vxm(r,loc,op.getInput(),op.getInput(),op.getInput().getType(),
            begin-1,col,kind,lhs,li,0.f,rhs,ri,immediate,format,output,count,1,"east","west",-1,
            reset,accumulate,emit,false,depth);
        if(lhs.starts_with("stream_"))x->setAttr("lhs_stream_source",r.getStringAttr("local"));
        if(rhs.starts_with("stream_"))x->setAttr("rhs_stream_source",r.getStringAttr("local"));
        return x;
    };
    const auto readVxm=[&](mlir::DictionaryAttr p,int planes,int64_t begin,int64_t row,int port,int64_t count,int64_t stride=1) {
        for(int side=0;side<2;++side)for(int byte=0;byte<planes;++byte)
            mem(p,byte,side,begin-route(E::Mem,E::VxmInput,D::West,id(p,byte)),"read",row,32+side*16+port+byte,count,stride);
    };
    const auto writeVxm=[&](mlir::DictionaryAttr p,int planes,int64_t output,int64_t row,int port,int64_t count) {
        int64_t end=0;
        for(int side=0;side<2;++side)for(int byte=0;byte<planes;++byte) {
            const auto at=output+route(E::VxmResult,E::Mem,D::East,id(p,byte));
            mem(p,byte,side,at,"write",row,(1-side)*8+port+byte,count);
            end=std::max(end,at+count);
        }
        return end;
    };
    const auto readSink=[&](int64_t begin,int64_t pair,int64_t tile) {
        auto p=home("sink");
        // QK's westbound result is on the opposite physical MEM side from
        // its query owner (as is flash_value). Ordinary mirrored constants
        // retain both head rows; select by that logical owner.
        for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte)
            mem(p,byte,side,begin-route(E::Mem,E::VxmInput,D::West,id(p,byte)),
                "read",2*(pair*tiles+tile)+(1-side),32+side*16+byte,1);
    };
    // Track the whole instruction domains, not individual FU issues. A queue
    // never switches to another macro while a current macro remains active.
    int64_t mxmFree=96,loadFree=96,sxmFree=96,vxmFree=96,outputEnd=0;
    int64_t lastWeightBuffer=0;
    const auto qk=[&](int64_t pair,int64_t tile) {
        auto q=home("query"),k=home("key"),score=home(tile%2?"score_alt":"score");
        // K Q^T consumes each query vector as one MXM weight column. Q is
        // already [query,K32] at ingress: a transpose would reverse the
        // mathematical orientation, as well as add a needless SXM pass.
        // LOAD and COMPUTE have independent ICUs. Fill the other weight
        // buffer during the last 32 compute beats of the preceding tile.
        const int64_t buffer=1-lastWeightBuffer;
        const int64_t load=std::max(loadFree,mxmFree-32),compute=load+32;
        for(int side=0;side<2;++side) {
            for(int byte=0;byte<2;++byte) {
                mem(q,byte,side,load-route(E::Mem,E::MxmWeight,D::East,id(q,byte)),"read",
                    pair*queryRows*blocks+tile*32,18+byte,32,1,blocks,period,queryRows);
                mem(k,byte,side,compute-route(E::Mem,E::MxmActivation,D::East,id(k,byte)),"read",
                    pair*width,4+byte,32,1,blocks,period,32);
            }
            MxmDomain3D iw;iw.repeat_count=32;iw.repeat_weight_column_stride=1;
            iw.wave_count=blocks;iw.wave_interval=period;iw.weight_buffer_mode="toggle_dim1";
            emitMxm3D(r,loc,load,side,"iw",buffer,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",iw,18);
            auto cm=iw;cm.repeat_weight_column_stride=0;cm.terminal_dimension=1;
            cm.terminal_accumulator_destination="stream";cm.terminal_accumulator_clear=true;cm.terminal_accumulator_output_format="bf16";
            emitMxm3D(r,loc,compute,side,"compute",buffer,0,4,24,0,1,"sram",false,"block",0,"bf16","direct16","",cm);
            for(int byte=0;byte<2;++byte)
                mem(score,byte,side,compute+(blocks-1)*period+pipeline+route(E::MxmResult,E::Mem,D::West,id(score,byte)),
                    "write_tap",0,56+byte,32);
        }
        loadFree=load+(blocks-1)*period+32;
        mxmFree=compute+(blocks-1)*period+32;
        lastWeightBuffer=buffer^((blocks-1)%2);
        const int64_t ready=mxmFree+pipeline+route(E::MxmResult,E::Mem,D::West,id(score,1));
        createTimeline(r,loc,"flash.qk.continuous_k",load,ready);
        return ready;
    };
    const auto maskHead=[&](int64_t begin,int64_t tile,int depth) {
        auto mask=home("causal_mask");
        // Group adjacent keys with identical masking mode into one domain.
        for(int64_t key=0;key<32;) {
            const auto threshold=[&](int64_t k){return op.getCausal()
                ?(headMajor?k-op.getPositionOffset():(k-op.getPositionOffset())*group-tile*32):int64_t{0};};
            const int64_t t=threshold(key);const int mode=t<=0?0:t>=32?2:1;
            int64_t end=key+1;
            if(mode!=1)while(end<32 && (threshold(end)<=0?0:threshold(end)>=32?2:1)==mode)++end;
            alu(begin+key,0,"add","stream_bf16",0,mode==1?"stream_bf16":"immediate",mode==1?2:0,mode==2?-1.e9f:0.f,"fp32",-1,end-key,depth);
            if(mode==1)readVxm(mask,2,begin+key,t-1,2,1);
            key=end;
        }
    };
    const auto prepareSoftmax=[&](int64_t pair,int64_t tile,int64_t ready) {
        const int64_t row=0;
        auto raw=home(tile%2?"score_alt":"score"),scaled=home("scaled");
        int64_t begin=std::max(vxmFree,ready)+8;
        const int64_t start=begin;
        readVxm(raw,2,begin,row,0,32);
        const auto scale=op.config().getAs<mlir::FloatAttr>("scale").getValueAsDouble();
        alu(begin,0,"multiply","stream_bf16",0,"immediate",0,scale,"fp32",-1,32,2);
        alu(begin,1,"bypass","previous",0,"immediate",0,0.f,"bf16",0,32,2);
        // BF16 scale remains a storage boundary, exactly as in the original
        // Flash arithmetic. Max consumes masked values; unmasked values keep
        // their original BF16 bits, masked values underflow to zero at exp.
        begin=writeVxm(scaled,2,begin+2,row,0,32)+8;
        if(hasSink) {
            // Include the sink in the stable maximum. Each query lane owns
            // its head's FP32 scalar, including distinct east/west heads.
            readSink(begin,pair,tile);
            alu(begin,0,"bypass","stream_f32",0,"immediate",0,0.f,"fp32",-1,1,2);
            alu(begin,1,"add","previous",0,"accumulator",0,0.f,"fp32",-1,1,2,true,true,false);
            begin+=4;
        }
        readVxm(scaled,2,begin,row,0,32);maskHead(begin,tile,2);
        if(hasSink)alu(begin,1,"max","previous",0,"accumulator",0,0.f,"fp32",-1,32,2,false,true,false);
        else {
            alu(begin,1,"add","previous",0,"accumulator",0,0.f,"fp32",-1,1,2,true,true);
            alu(begin+1,1,"max","previous",0,"accumulator",0,0.f,"fp32",-1,31,2,false,true);
        }
        vxmFree=begin+32+4;
        createTimeline(r,loc,"flash.softmax.scale_max",start,vxmFree);
    };
    const auto softmax=[&](int64_t pair,int64_t tile) {
        const int64_t row=0,scalar=0;
        auto scaled=home("scaled"),ex=home("exp"),inverse=home("inverse");
        int64_t begin=vxmFree;
        const int64_t start=begin;
        readVxm(scaled,2,begin,row,0,32);maskHead(begin,tile,8);
        alu(begin,1,"subtract","previous",0,"accumulator",0,0.f,"fp32",-1,32,8);
        for(int col=2;col<8;++col)
            alu(begin,col,col==5?"exp":"bypass","previous",0,"immediate",0,0.f,"fp32",col==7?6:-1,32,8);
        begin=writeVxm(ex,4,begin+11,row,6,32)+8;
        if(hasSink) {
            readSink(begin,pair,tile);
            alu(begin,0,"bypass","stream_f32",0,"immediate",0,0.f,"fp32",-1,1,8);
            alu(begin,1,"subtract","previous",0,"accumulator",0,0.f,"fp32",-1,1,8);
            for(int col=2;col<8;++col)
                alu(begin,col,col==5?"exp":"bypass","previous",0,"immediate",0,0.f,"fp32",col==7?6:-1,1,8);
            // Reuse the eventual inverse scalar home; sink exp contributes
            // only to the denominator and never to the value product.
            begin=writeVxm(inverse,4,begin+11,0,6,1)+8;
            readVxm(inverse,4,begin,0,0,1);
            alu(begin,2,"bypass","stream_f32",0,"immediate",0,0.f,"fp32",-1,1,2);
            alu(begin,3,"add","previous",0,"accumulator",0,0.f,"fp32",-1,1,2,true,true,false);
            begin+=4;
        }
        // PV uses buffer zero. It can be filled while the preceding QK
        // computes on buffer one; otherwise wait for that reader to finish.
        const int64_t earliestLoad=std::max(loadFree,mxmFree-(lastWeightBuffer==0?0:32));
        begin=std::max({begin,sxmFree-toSxm-1,earliestLoad-toSxm-1-sxmLatency-toWeight});
        // C0 and C2 share the existing fixed FP32 input bundle. Read each
        // exponent vector once: C0/C1 casts P to BF16 for SXM while C2/C3
        // accumulates the original FP32 values in the original key order.
        // P is unnormalized, so its capture/load need not wait for 1/sum.
        readVxm(ex,4,begin,row,0,32);
        alu(begin,0,"bypass","stream_f32",0,"immediate",0,0.f,"fp32",-1,32,2);
        alu(begin,1,"bypass","previous",0,"immediate",0,0.f,"bf16",0,32,2);
        alu(begin,2,"bypass","stream_f32",0,"immediate",0,0.f,"fp32",-1,32,2);
        alu(begin,3,"add","previous",0,"accumulator",0,0.f,"fp32",-1,1,2,!hasSink,true,false);
        alu(begin+1,3,"add","previous",0,"accumulator",0,0.f,"fp32",-1,30,2,false,true,false);
        alu(begin+31,3,"add","previous",0,"accumulator",0,0.f,"fp32",-1,1,2,false,true,true);
        // Arm the existing C3->C2 feedback after C2's 32-input domain.
        // Only the terminal sum returns; no partial sum uses MEM or SR.
        alu(begin+32,2,"bypass","feedback",0,"immediate",0,0.f,"fp32",-1,1,2);
        alu(begin+32,3,"reciprocal","previous",0,"immediate",0,0.f,"fp32",2,1,2);
        const int64_t inverseReady=writeVxm(inverse,4,begin+38,scalar,2,1);
        const int64_t capture=begin+1+toSxm,load=capture+sxmLatency+toWeight;
        for(int side=0;side<2;++side) {
            emitSxm(r,loc,capture,side,"transpose",{(1-side)*8,(1-side)*8+1},{18,19},identity,"wide_matrix");
            MxmDomain3D iw;iw.repeat_count=32;iw.repeat_weight_column_stride=1;
            emitMxm3D(r,loc,load,side,"iw",0,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",iw,18);
        }
        sxmFree=capture+sxmLatency+32;
        loadFree=load+32;
        vxmFree=begin+40;
        createTimeline(r,loc,"flash.softmax.sum_p_shared_read",begin,vxmFree);
        createTimeline(r,loc,"flash.softmax.p_to_sxm_mxm",start,load+32);
        return std::max(load+32,inverseReady);
    };
    const auto pv=[&](int64_t pair,int64_t tile,int64_t ready) {
        auto v=home("value"),inverse=home("inverse"),output=home("context");
        const int64_t compute=std::max({mxmFree,ready,vxmFree-pipeline-returning});
        const int64_t arrival=compute+pipeline+returning;
        for(int side=0;side<2;++side) {
            for(int byte=0;byte<2;++byte)
                mem(v,byte,side,compute-route(E::Mem,E::MxmActivation,D::East,id(v,byte)),"read",pair*width,4+byte,width);
            MxmDomain3D cm;cm.repeat_count=32;cm.wave_count=blocks;cm.wave_interval=32;
            emitMxm3D(r,loc,compute,side,"compute",0,0,4,side*16,0,1,"stream",true,"block",0,"bf16","direct16","fp32",cm);
        }
        readVxm(inverse,4,arrival,0,4,width,0);
        alu(arrival,0,"multiply","stream_f32",0,"stream_f32",4,0.f,"fp32",-1,width,2);
        alu(arrival,1,"bypass","previous",0,"immediate",0,0.f,"bf16",0,width,2);
        // Restore feature lanes directly after the normalization chain. This
        // final transpose streams into the consumer's planar context home;
        // no intermediate context is written to MEM and read back.
        const int64_t capture=arrival+2+toSxm;
        if(capture<sxmFree)throw std::runtime_error("Flash PV transpose overlaps the active SXM domain");
        for(int side=0;side<2;++side) {
            emitSxm(r,loc,capture,side,"transpose",{(1-side)*8,(1-side)*8+1},{60,61},identity,"wide_matrix",-1,-1,-1,blocks,32);
            for(int byte=0;byte<2;++byte) {
                const auto at=capture+sxmLatency+route(E::SxmResult,E::Mem,D::West,id(output,byte));
                mem(output,byte,side,at,"write",(pair*tiles+tile)*width,60+byte,width);
                outputEnd=std::max(outputEnd,at+width);
            }
        }
        sxmFree=capture+sxmLatency+width;
        // The following exponent pass changes depth-2 to depth-8. Drain the
        // mirrored chain's control/data slots before admitting that Bundle.
        vxmFree=arrival+width+2+target.throughput().vxm_alus/2;
        mxmFree=compute+width;
        lastWeightBuffer=0;
        createTimeline(r,loc,"flash.pv.mxm_vxm_direct",compute,outputEnd);
    };
    for(int64_t pair=0;pair<pairs;++pair) {
        std::vector<int64_t> ready(tiles);
        ready[0]=qk(pair,0);
        for(int64_t tile=0;tile<tiles;++tile) {
            // One-tile lookahead: QK for the following tile uses MXM while
            // VXM processes this tile. Data homes remain live until PV ends.
            if(tile+1<tiles)ready[tile+1]=qk(pair,tile+1);
            if(tile==0)prepareSoftmax(pair,tile,ready[tile]);
            const auto weightsReady=softmax(pair,tile);
            // While P crosses SXM and fills the weight buffer, prepare the
            // following tile's scale/max. C1 retains that max across the PV
            // epilogue, whose bypass never writes its accumulator. The next
            // exponent chain consumes it without a max spill/reload.
            if(tile+1<tiles)prepareSoftmax(pair,tile+1,ready[tile+1]);
            pv(pair,tile,weightsReady);
        }
    }
    createTimeline(r,loc,"flash.tile_pipeline",96,outputEnd+target.streams().system_register_columns);
    auto result=createBinding(r,loc,{},outputIndex,"output","result",
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType()),home("context"),"attention.context");
    result->setAttr("ready_cycle",r.getI64IntegerAttr(outputEnd));
    op.core->getParentOfType<mlir::func::FuncOp>()->setAttr("ftlpu.attention_core_schedule",r.getStringAttr("flash_tile_pipeline"));
    return result.getValue();
}
