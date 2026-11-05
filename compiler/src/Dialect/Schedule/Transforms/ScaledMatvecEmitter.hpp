// Included in VectorAluScheduleEmitter's private namespace. This primitive
// retains a scaled matrix while streaming BF16 values into MXM IW;
// the contraction never reloads the scaled matrix from MEM.
// Its optional scaled-residual epilogue consumes the returning MXM stream.
mlir::LogicalResult planScaledMatvecMemory(mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    mlir::IRRewriter r(function.getContext());
    for (auto op : function.getOps<kernel::ScaledMatvecOp>()) {
        if (target.memory().bytes_per_word!=32 || target.throughput().mxm_weight_buffers<2)
            return op.emitError("streamed scaled_matvec requires 32-lane vectors and two MXM weight buffers");
        auto m=llvm::cast<mlir::RankedTensorType>(op.getMatrix().getType());
        auto d=llvm::cast<mlir::RankedTensorType>(op.getScale().getType());
        auto k=llvm::cast<mlir::RankedTensorType>(op.getVector().getType());
        auto p=llvm::cast<mlir::RankedTensorType>(op.getProduct().getType());
        if(m.getRank()!=2 || d.getRank()!=2 || k.getRank()!=2 || p.getRank()!=2
            || (!m.getElementType().isF32() && !m.getElementType().isBF16())
            || d.getElementType()!=m.getElementType() || p.getElementType()!=m.getElementType()
            || !k.getElementType().isBF16()
            || op.getScaled().getType()!=m || k.getDimSize(0)<=0
            || k.getDimSize(0)>32 || d.getDimSize(0)!=1
            || d.getDimSize(1)>32 || d.getDimSize(1)<static_cast<int64_t>(op.getGroupOffset())+k.getDimSize(0)
            || m.getDimSize(1)!=k.getDimSize(1)
            || m.getDimSize(0)%k.getDimSize(0) || m.getDimSize(1)%32
            || p.getDimSize(0)!=k.getDimSize(0)
            || p.getDimSize(1)!=m.getDimSize(0)/k.getDimSize(0) || p.getDimSize(1)%32)
            return op.emitError("scaled_matvec needs matching FP32 or BF16 matrix/scale/product, BF16 vector, K/V multiples of 32");
        const auto fp32Policy=function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
        const bool full=fp32Policy && fp32Policy.getValue() && m.getElementType().isF32();
        if(op->hasAttr("matrix_placement")) continue;
        op->setAttr("matrix_placement",placement(r,m,0,"both",1,true,32));
        op->setAttr("scale_placement",placement(r,d,4,"both",1,true,32));
        op->setAttr("vector_placement",placement(r,k,8,"both",1,true,32));
        op->setAttr("scaled_placement",placement(r,m,12,"both",1,true,32));
        op->setAttr("product_placement",placement(r,p,16,"both",1,true,32));
        auto broadcast=mlir::RankedTensorType::get({d.getDimSize(1),32},d.getElementType());
        op->setAttr("scale_table",placement(r,broadcast,20,"both",1,true,32));
        if(!op.getEpilogue().empty()) {
            op->setAttr("value_placement",placement(r,llvm::cast<mlir::RankedTensorType>(op.getEpilogue()[0].getType()),24,"both",1,true,32));
            op->setAttr("coefficient_placement",placement(r,d,full?28:26,"both",1,true,32));
            op->setAttr("correction_placement",placement(r,p,36,"both",1,true,32));
            op->setAttr("coefficient_table",placement(r,broadcast,40,"both",1,true,32));
            if(op.getCorrection().size()>=2) {
                op->setAttr("updated_placement",placement(r,m,44,"both",1,true,32));
                // Two head-sized buffers, on independent MEM ICU banks.
                auto rows=mlir::RankedTensorType::get({p.getDimSize(1),32},full?r.getF32Type():r.getBF16Type());
                op->setAttr("correction_table",placement(r,rows,full?48:50,"both",1,true,32));
                op->setAttr("correction_table_alt",placement(r,rows,full?48:50,"both",0,true,32));
                if(op.getEpilogue().size()==3) {
                    op->setAttr("query_placement",placement(r,k,full?32:30,"both",1,true,32));
                    op->setAttr("readout_placement",placement(r,p,44,"both",1,true,32));
                }
            }
        }
    }
    return mlir::success();
}

mlir::LogicalResult lowerScaledMatvecSchedules(mlir::IRRewriter& r,
    mlir::func::FuncOp function,const target::LPUTargetModel& target)
{
    llvm::SmallVector<kernel::ScaledMatvecOp> ops;
    function.walk([&](kernel::ScaledMatvecOp op){ops.push_back(op);});
    for(auto op:ops) {
        r.setInsertionPoint(op);
        const auto loc=op.getLoc();
        auto m=llvm::cast<mlir::RankedTensorType>(op.getMatrix().getType());
        auto k=llvm::cast<mlir::RankedTensorType>(op.getVector().getType());
        // BF16 storage uses the FU's RNE output conversion. Opt-in full FP32
        // ingress preserves all State/gate/delta bytes. MXM alone consumes
        // the high BF16 pair; the retained FP32 State is never narrowed.
        const bool rne=m.getElementType().isBF16();
        const int64_t high=rne?0:2, bytes=rne?2:4;
        const auto fp32Policy=function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
        const bool full=fp32Policy && fp32Policy.getValue() && !rne;
        const int64_t inputBytes=full?4:2, inputHigh=full?0:high;
        const llvm::StringRef inputKind=full?"stream_f32":"stream_bf16";
        const llvm::StringRef outputFormat=rne?"bf16":"fp32";
        const int64_t heads=k.getDimSize(0),kd=k.getDimSize(1),vd=m.getDimSize(0)/heads;
        const int64_t scaleHeads=llvm::cast<mlir::RankedTensorType>(op.getScale().getType()).getDimSize(1);
        auto get=[&](llvm::StringRef name){return op->getAttrOfType<mlir::DictionaryAttr>(name);};
        auto matrix=get("matrix_placement"),scale=get("scale_placement"),key=get("vector_placement");
        auto saved=get("scaled_placement"),product=get("product_placement"),table=get("scale_table");
        const bool epilogue=!op.getEpilogue().empty();
        const bool readout=op.getEpilogue().size()==3;
        if(epilogue && full && !llvm::cast<mlir::RankedTensorType>(op.getEpilogue()[0].getType()).getElementType().isF32())
            return op.emitError("full FP32 scaled_matvec requires FP32 value on the independent C4 RHS port");
        if(epilogue && !full && !llvm::cast<mlir::RankedTensorType>(op.getEpilogue()[0].getType()).getElementType().isBF16())
            return op.emitError("FP32 epilogue value requires vector_fp32_ingress");
        const int64_t depth=epilogue?4:8, tailStream=epilogue?2:6;
        auto value=get("value_placement"),coefficient=get("coefficient_placement");
        auto correction=get("correction_placement"),coefficientTable=get("coefficient_table");
        auto id=[&](mlir::DictionaryAttr p,int64_t byte){return llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[byte]).getInt();};
        auto base=[&](mlir::DictionaryAttr p){return p.getAs<mlir::IntegerAttr>("base_row").getInt();};
        auto bank=[&](mlir::DictionaryAttr p){return p.getAs<mlir::IntegerAttr>("bank").getInt();};
        auto mem=[&](mlir::DictionaryAttr p,int64_t byte,int64_t side,int64_t cycle,
                     llvm::StringRef opcode,int64_t address,int64_t stream,
                     int64_t count=1,int64_t stride=1,int64_t interval=1) {
            direct_domain_detail::emitMem3D(r,loc,target,cycle,
                side*target.memory().slices_per_hemisphere+id(p,byte),opcode,base(p)+address,
                stream,count,interval,stride,1,1,0,1,1,0,-1,bank(p));
        };
        auto bind=[&](mlir::Value value,mlir::Type type,int64_t index,
                      llvm::StringRef access,mlir::DictionaryAttr p) {
            mlir::OperationState b(loc,BindingOp::getOperationName());
            if(value)b.addOperands(value);b.addTypes(type);
            b.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(index)),
                r.getNamedAttr("access",r.getStringAttr(access)),
                r.getNamedAttr("role",r.getStringAttr("activation")),
                r.getNamedAttr("bytes",r.getI64IntegerAttr(llvm::cast<mlir::RankedTensorType>(type).getNumElements()*(llvm::cast<mlir::RankedTensorType>(type).getElementType().isF32()?4:2))),
                r.getNamedAttr("placement",p)});
            return llvm::cast<BindingOp>(r.create(b)).getResult();
        };
        for(unsigned i=0;i<op->getNumOperands();++i) if(auto arg=llvm::dyn_cast<mlir::BlockArgument>(op->getOperand(i)))
            bind(arg,arg.getType(),arg.getArgNumber(),"input",i==0?matrix:i==1?scale:i==2?key:i==3?value:i==4?coefficient:get("query_placement"));
        int64_t cursor=96;
        // A packed head vector is repeated 32 times into wide SXM, producing
        // one all-lanes coefficient vector per head. BF16 storage is already
        // RNE-converted; legacy FP32 storage supplies only its high planes.
        // Wide SXM transposes two byte planes at a time, so full FP32
        // broadcasts use separate low/high-pair passes.
        const auto wide=function->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
        if(!wide || !wide.getValue())return op.emitError("scaled_matvec streaming requires wide SXM");
        for(int factor=0;factor<(epilogue?2:1);++factor) {
          auto factorInput=factor?coefficient:scale;
          auto factorTable=factor?coefficientTable:table;
          for(int pair=0;pair<inputBytes/2;++pair) {
          const int64_t factorCycle=cursor+80*(factor*(inputBytes/2)+pair);
          for(int64_t side=0;side<2;++side) {
            llvm::SmallVector<mlir::Attribute> src,dst,map;
            for(int b=pair*2;b<pair*2+2;++b){src.push_back(r.getI64IntegerAttr(16+inputBytes*factor+b));dst.push_back(r.getI64IntegerAttr(32+side*16+b));}
            for(int i=0;i<32;++i)map.push_back(r.getI64IntegerAttr(i));
            mlir::OperationState sx(loc,SxmOp::getOperationName());
            sx.addAttributes({r.getNamedAttr("cycle",r.getI64IntegerAttr(factorCycle)),
                r.getNamedAttr("hemisphere",r.getI64IntegerAttr(side)),
                r.getNamedAttr("opcode",r.getStringAttr("transpose")),
                r.getNamedAttr("source_streams",r.getArrayAttr(src)),
                r.getNamedAttr("destination_streams",r.getArrayAttr(dst)),
                r.getNamedAttr("permute_map",r.getArrayAttr(map)),
                r.getNamedAttr("weight_layout",r.getStringAttr("wide_matrix")),
                r.getNamedAttr("repeat_count",r.getI64IntegerAttr(1)),
                r.getNamedAttr("repeat_interval",r.getI64IntegerAttr(32))});
            r.create(sx);
            for(int b=pair*2;b<pair*2+2;++b) {
                const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(factorInput,b+inputHigh));
                const auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(factorTable,b+inputHigh));
                mem(factorInput,b+inputHigh,side,factorCycle-rd,"read",0,16+inputBytes*factor+b,32,0);
                mem(factorTable,b+inputHigh,side,factorCycle+35+wr,"write",0,32+side*16+b,scaleHeads);
            }
        }
        }
        }
        cursor+=(epilogue?160:80)*(inputBytes/2)+80;
        const int64_t begin=cursor;
        // Each 32x32 matrix block is generated once. Its output is saved
        // while the BF16 pair directly feeds the MXM weight port.
        // The contraction never reads 'saved'; only the optional later
        // rank-one update consumes that retained matrix.
        for(int64_t h=0;h<heads;++h)for(int64_t vb=0;vb<vd/32;++vb)
            for(int64_t kb=0;kb<kd/32;++kb) {
                const int64_t start=cursor+32;
                const int64_t weightBuffer=((h*(vd/32)+vb)*(kd/32)+kb)%2;
                const int64_t address=kb*heads*vd+h*vd+vb*32;
                for(int64_t side=0;side<2;++side)for(int b=0;b<inputBytes;++b) {
                    mem(matrix,b+inputHigh,side,start-1-id(matrix,b+inputHigh)/4,"read",address,32+side*16+b,32);
                    mem(table,b+inputHigh,side,start-1-id(table,b+inputHigh)/4,"read",h+op.getGroupOffset(),32+side*16+inputBytes+b,32,0);
                }
                mlir::Value current=op.getMatrix();
                for(int col=0;col<depth;++col) {
                    auto alu=ffn_detail::create_vxm(r,loc,current,current,m,start-1,col,
                        col?"bypass":"multiply",col?"previous":inputKind,0,0.f,
                        col?"immediate":inputKind,col?0:inputBytes,0.f,col==depth-1?outputFormat:"fp32",
                        col==depth-1?tailStream:-1,32,1,"east","west",-1,false,false,true,false,depth);
                    current=alu.getResult();
                }
                for(int64_t side=0;side<2;++side) {
                    for(int b=0;b<bytes;++b)
                        mem(saved,b,side,start+depth+1+id(saved,b)/4,"write_tap",address,(1-side)*8+tailStream+b,32);
                    attention_detail::MxmDomain3D load;
                    load.repeat_count=32;load.repeat_weight_column_stride=1;
                    const auto vxmToMxm=*target.transport_latency(target::StreamEndpoint::VxmResult,
                        target::StreamEndpoint::Mem,target::StreamDirection::East,0)
                        + *target.transport_latency(target::StreamEndpoint::Mem,
                        target::StreamEndpoint::MxmWeight,target::StreamDirection::East,0);
                    attention_detail::emitMxm3D(r,loc,start+depth+vxmToMxm,
                        side*target.throughput().mxms_per_hemisphere,"iw",weightBuffer,0,
                        0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",load,
                        (1-side)*8+tailStream+high);
                    // Consume the completed tile immediately after its last
                    // column arrives. The next tile loads the other buffer.
                    const int64_t compute=start+depth+vxmToMxm+32;
                    for(int b=0;b<2;++b) {
                        const auto latency=*target.transport_latency(target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmActivation,target::StreamDirection::East,id(key,b));
                        mem(key,b,side,compute-latency,"read",kb*heads+h,b);
                    }
                    const bool last=kb==kd/32-1;
                    // W0-W3 (BF16) or W0-W7 (FP32) carry continuous
                    // state/coefficient inputs.
                    // Without an epilogue, KS uses W24-W27. With one, place
                    // the operand on each hemisphere's fixed C4 LHS port.
                    attention_detail::emitMxm(r,loc,compute,
                        side*target.throughput().mxms_per_hemisphere,"compute",weightBuffer,0,0,epilogue?8-inputHigh+side*16:24,1,1,
                        0,1,last?"stream":"sram",last,"block",0,"bf16","direct16",last?outputFormat:"");
                    if(last)for(int b=0;b<bytes;++b) {
                        const auto latency=*target.transport_latency(target::StreamEndpoint::MxmResult,
                            target::StreamEndpoint::Mem,target::StreamDirection::West,id(product,b));
                        mem(product,b,side,compute+target.mxm_first_result_latency()+latency,
                            epilogue?"write_tap":"write",vb*heads+h,32+(epilogue?8-inputHigh+side*16:24)+b);
                    }
                }
                if(epilogue) {
                    // C4-C7 are a separate depth-four chain, mirrored by the
                    // shared ICU. Capture beta once per head into C5's local
                    // scalar register, while C0-C3 keep streaming state tiles.
                    auto chain=[&](int64_t cycle,bool capture) {
                        mlir::Value current=op.getMatrix();
                        for(int col=4;col<8;++col) {
                            auto alu=ffn_detail::create_vxm(r,loc,current,current,m,cycle,col,
                                capture?"bypass":col==4?"subtract":col==5?"multiply":col==6?"negate":"bypass",
                                col==4?inputKind:"previous",col==4?8:0,0.f,
                                !capture && col==4?inputKind:!capture && col==5?"accumulator":"immediate",
                                !capture && col==4?8+inputBytes:0,0.f,col==7?outputFormat:"fp32",!capture && col==7?6:-1,
                                1,1,"east","west",-1,false,false,true,capture && col==5,4);
                            current=alu.getResult();
                        }
                    };
                    if(vb==0 && kb==0) {
                        // Leave the preceding head's final multiply time to
                        // consume its beta before replacing the local scalar.
                        const bool preview=start+63<begin+31+heads*(vd/32)*(kd/32)*32;
                        for(int64_t side=0;side<2;++side)for(int b=0;b<inputBytes;++b)
                            mem(coefficientTable,b+inputHigh,side,start+63-preview-id(coefficientTable,b+inputHigh)/4,
                                "read",h+op.getGroupOffset(),32+side*16+8+b,1+preview,0);
                        chain(start+63,true);
                    }
                    if(kb==kd/32-1) {
                        const auto forward=*target.transport_latency(target::StreamEndpoint::VxmResult,
                            target::StreamEndpoint::Mem,target::StreamDirection::East,0)
                            + *target.transport_latency(target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmWeight,target::StreamDirection::East,0);
                        const auto returning=*target.transport_latency(target::StreamEndpoint::MxmResult,
                            target::StreamEndpoint::Mem,target::StreamDirection::West,0)
                            + *target.transport_latency(target::StreamEndpoint::Mem,
                            target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
                        const int64_t arrival=start+depth+forward+32+target.mxm_first_result_latency()+returning;
                        const bool preview=arrival-1<begin+31+heads*(vd/32)*(kd/32)*32;
                        // The shared Bundle includes the newly decoding C4
                        // ports one cycle before C4 executes. C0 is already
                        // active, so supply that preview beat as well. It is
                        // discarded by C0; only the following real MXM result
                        // is consumed by C4. This avoids an entire KS spill.
                        for(int64_t side=0;side<2;++side)for(int b=0;b<inputBytes;++b) {
                            mem(value,b,side,arrival-1-preview-id(value,b)/4,"read",vb*heads+h,32+side*16+8+inputBytes+b,1+preview,0);
                            if(preview)mem(coefficientTable,b+inputHigh,side,arrival-2-id(coefficientTable,b+inputHigh)/4,
                                "read",h+op.getGroupOffset(),32+side*16+8+b);
                        }
                        chain(arrival-1,false);
                        for(int64_t side=0;side<2;++side)for(int b=0;b<bytes;++b)
                            mem(correction,b,side,arrival+5+id(correction,b)/4,"write",vb*heads+h,(1-side)*8+6+b);
                    }
                }
                // Alternate physical IW buffers across every head/value/key
                // tile, including an odd number of key blocks. The next
                // 32-column domain starts on the very next cycle.
                cursor=start;
            }
        cursor+=160; // Drain final compute, ACC output and distributed MEM writes.
        mlir::OperationState timeline(loc,TimelineOp::getOperationName());
        timeline.addAttributes({r.getNamedAttr("name",r.getStringAttr("scaled_matvec.vxm_to_mxm_and_mem")),
            r.getNamedAttr("start",r.getI64IntegerAttr(begin)),r.getNamedAttr("end",r.getI64IntegerAttr(cursor))});
        r.create(timeline);
        // Optional rank-one update. Keep the matrix in its native [H,V,K]
        // layout. Only the small correction vector is expanded by wide SXM;
        // no outer-product intermediate is ever materialized in MEM.
        if(op.getCorrection().size()>=2) {
            auto updated=get("updated_placement");
            auto buffer=[&](int64_t h){return get(h%2?"correction_table_alt":"correction_table");};
            const int64_t updateBegin=cursor;
            auto broadcastHead=[&](int64_t h,int64_t launch) {
              auto expanded=buffer(h);
              for(int64_t vb=0;vb<vd/32;++vb) {
                for(int pair=0;pair<inputBytes/2;++pair) {
                const int64_t t=launch+(vb*(inputBytes/2)+pair)*80;
                for(int64_t side=0;side<2;++side) {
                    llvm::SmallVector<mlir::Attribute> src,dst,map;
                    for(int b=pair*2;b<pair*2+2;++b) {
                        src.push_back(r.getI64IntegerAttr(16+b));
                        // Reserve separate SR bytes from the active binary
                        // VXM chain: W0..W7 are occupied in FP32 mode.
                        dst.push_back(r.getI64IntegerAttr(32+side*16+(full?8:4)+b));
                    }
                    for(int i=0;i<32;++i)map.push_back(r.getI64IntegerAttr(i));
                    mlir::OperationState sx(loc,SxmOp::getOperationName());
                    sx.addAttributes({r.getNamedAttr("cycle",r.getI64IntegerAttr(t)),
                        r.getNamedAttr("hemisphere",r.getI64IntegerAttr(side)),
                        r.getNamedAttr("opcode",r.getStringAttr("transpose")),
                        r.getNamedAttr("source_streams",r.getArrayAttr(src)),
                        r.getNamedAttr("destination_streams",r.getArrayAttr(dst)),
                        r.getNamedAttr("permute_map",r.getArrayAttr(map)),
                        r.getNamedAttr("weight_layout",r.getStringAttr("wide_matrix")),
                        r.getNamedAttr("repeat_count",r.getI64IntegerAttr(1)),
                        r.getNamedAttr("repeat_interval",r.getI64IntegerAttr(32))});
                    r.create(sx);
                    for(int b=pair*2;b<pair*2+2;++b) {
                        auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(correction,b+inputHigh));
                        auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(expanded,b));
                        mem(correction,b+inputHigh,side,t-rd,"read",vb*heads+h,16+b,32,0);
                        mem(expanded,b,side,t+35+wr,"write",vb*32,32+side*16+(full?8:4)+b,32);
                    }
                }
                }
              }
              // Include the final transpose output, MEM commit, and route
              // to VXM before allowing a read from this head's buffer.
              return launch+(vd/32)*80*(inputBytes/2)+64;
            };
            cursor=broadcastHead(0,cursor);
            auto updateChain=[&](int64_t issue,bool capture,int64_t count) {
                mlir::Value current=op.getMatrix();
                for(int col=0;col<4;++col) {
                    auto alu=ffn_detail::create_vxm(r,loc,current,current,m,issue,col,
                        capture?"bypass":col==1?"multiply":col==2?"add":"bypass",
                        col?"previous":capture?"stream_bf16":inputKind,0,0.f,
                        capture?"immediate":col==0?inputKind:col==1?"accumulator":col==2?"auxiliary":"immediate",
                        !capture && col==0?inputBytes:0,0.f,col==3?outputFormat:"fp32",!capture && col==3?2:-1,
                        count,1,"east","west",-1,false,false,true,capture && col==1,4);
                    current=alu.getResult();
                }
            };
            for(int64_t h=0;h<heads;++h) {
              auto expanded=buffer(h);
              // At this boundary bank(h+1) is either unused or was fully
              // consumed by head h-1. Broadcast while head h computes.
              const int64_t nextReady=h+1<heads?broadcastHead(h+1,cursor):0;
              // A readout completes one value tile's K reduction before
              // advancing to the next tile, using the existing MXM ACC.
              for(int64_t vb=0;vb<(readout?vd/32:1);++vb)
              for(int64_t kb=0;kb<kd/32;++kb) {
                // Each lane retains its own K element in C1's local scalar.
                // Reuse this vector across every value row, with contiguous
                // correction/S reads and updated-S writes. No per-row capture.
                const int64_t capture=cursor, start=capture+1;
                const int64_t vectors=readout?32:vd;
                const int64_t valueOffset=readout?vb*32:0;
                const int64_t address=kb*heads*vd+h*vd+valueOffset;
                updateChain(capture-1,true,1);
                updateChain(start-1,false,vectors);
                for(int64_t side=0;side<2;++side) {
                    for(int b=0;b<2;++b) {
                        mem(key,b,side,capture-1-id(key,b)/4,"read",kb*heads+h,32+side*16+b);
                    }
                    for(int b=0;b<inputBytes;++b) {
                        mem(expanded,b,side,start-1-id(expanded,b)/4,"read",valueOffset,32+side*16+b,vectors);
                        mem(saved,b+inputHigh,side,start-1-id(saved,b+inputHigh)/4,"read",address,32+side*16+inputBytes+b,vectors);
                    }
                    for(int b=0;b<bytes;++b)
                        mem(updated,b,side,start+5+id(updated,b)/4,readout?"write_tap":"write",address,(1-side)*8+2+b,vectors);
                    if(readout) {
                        auto query=get("query_placement"),result=get("readout_placement");
                        const int64_t weightBuffer=((h*(vd/32)+vb)*(kd/32)+kb)%2;
                        const auto forward=*target.transport_latency(target::StreamEndpoint::VxmResult,target::StreamEndpoint::Mem,target::StreamDirection::East,0)
                            + *target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmWeight,target::StreamDirection::East,0);
                        const int64_t loadCycle=start+4+forward,compute=loadCycle+32;
                        attention_detail::MxmDomain3D domain;
                        domain.repeat_count=32;domain.repeat_weight_column_stride=1;
                        attention_detail::emitMxm3D(r,loc,loadCycle,side*target.throughput().mxms_per_hemisphere,
                            "iw",weightBuffer,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",domain,(1-side)*8+2+high);
                        for(int b=0;b<2;++b) {
                            const auto latency=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmActivation,target::StreamDirection::East,id(query,b));
                            mem(query,b,side,compute-latency,"read",kb*heads+h,b);
                        }
                        const bool last=kb==kd/32-1;
                        // Full-width SXM broadcasts use W24..W27 on the
                        // mirrored side; keep readout on its own W28..W31.
                        attention_detail::emitMxm(r,loc,compute,side*target.throughput().mxms_per_hemisphere,
                            "compute",weightBuffer,0,0,full?28:24,1,1,0,1,last?"stream":"sram",last,"block",0,"bf16","direct16",last?outputFormat:"");
                        if(last)for(int b=0;b<bytes;++b) {
                            const auto latency=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(result,b));
                            mem(result,b,side,compute+target.mxm_first_result_latency()+latency,"write",vb*heads+h,32+(full?28:24)+b);
                        }
                    }
                }
                // Multiply has a two-cycle result path; leave one beat before
                // switching C1 to the one-cycle scalar-capture bypass.
                cursor=start+vectors+1;
              }
              cursor=std::max(cursor,nextReady);
            }
            cursor+=readout?160:64;
            mlir::OperationState updateTimeline(loc,TimelineOp::getOperationName());
            updateTimeline.addAttributes({r.getNamedAttr("name",r.getStringAttr("scaled_matvec.rank_one_update")),
                r.getNamedAttr("head_count",r.getI64IntegerAttr(heads)),
                r.getNamedAttr("value_width",r.getI64IntegerAttr(vd)),
                r.getNamedAttr("key_width",r.getI64IntegerAttr(kd)),
                r.getNamedAttr("start",r.getI64IntegerAttr(updateBegin)),r.getNamedAttr("end",r.getI64IntegerAttr(cursor))});
            r.create(updateTimeline);
        }
        auto a=bind({},op.getScaled().getType(),0,"output",saved);
        auto b=bind({},op.getProduct().getType(),1,"output",product);
        if(epilogue) {
            auto c=bind({},op.getCorrection()[0].getType(),2,"output",correction);
            if(op.getCorrection().size()>=2) {
                auto d=bind({},op.getCorrection()[1].getType(),3,"output",get("updated_placement"));
                if(readout) {
                    auto e=bind({},op.getCorrection()[2].getType(),4,"output",get("readout_placement"));
                    r.replaceOp(op,{a,b,c,d,e});
                } else r.replaceOp(op,{a,b,c,d});
            } else r.replaceOp(op,{a,b,c});
        } else r.replaceOp(op,{a,b});
    }
    return mlir::success();
}
