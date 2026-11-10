// Generic head scaling and unit-lower forward substitution. No host arithmetic.
// A single FP32 coefficient row is expanded through two wide-SXM byte tiles.
mlir::LogicalResult emitUnitLowerSolve(mlir::IRRewriter& r,kernel::VectorAluOp op,
                                     const target::LPUTargetModel& target) {
    r.setInsertionPoint(op);const auto loc=op.getLoc();
    const bool blocked=op.getKind()=="unit_lower_solve_blocked";
    const bool solve=op.getKind()!="head_scale";
    const auto overlapFeedback=op->getAttrOfType<mlir::BoolAttr>("solve.projection_overlap");
    const int64_t feedbackStream=overlapFeedback && overlapFeedback.getValue()?12:28;
    const int64_t heads=solve?op.getLhs().getType().getDimSize(0)/32:op.getRhs().getType().getDimSize(1);
    const int64_t dim=op.getResult().getType().getDimSize(1),blocks=dim/32,half=heads/2,rows=half*32;
    auto get=[&](llvm::StringRef n){return op->getAttrOfType<mlir::DictionaryAttr>(n);};
    auto lhs=get("ftlpu.vector_lhs_placement"),rhs=get("ftlpu.vector_rhs_placement"),out=get("ftlpu.vector_result_placement");
    auto table=get("solve.coefficients"),feedback=get("solve.feedback");
    const auto wide=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    if(!wide || !wide.getValue() || target.memory().bytes_per_word!=32)
        return op.emitError("unit_lower_solve/head_scale require wide SXM and 32-lane vectors");
    const auto field=[](mlir::DictionaryAttr p,llvm::StringRef n){return p.getAs<mlir::IntegerAttr>(n).getInt();};
    const auto id=[](mlir::DictionaryAttr p,int64_t b){return llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[b]).getInt();};
    const auto partition=[](mlir::DictionaryAttr p){return p.getAs<mlir::StringAttr>("kind").getValue().starts_with("token_partition");};
    const auto mem=[&](mlir::DictionaryAttr p,int64_t byte,int64_t side,int64_t cycle,
                      llvm::StringRef opcode,int64_t address,int64_t stream,int64_t count=1,int64_t stride=1) {
        direct_domain_detail::emitMem3D(r,loc,target,cycle,side*target.memory().slices_per_hemisphere+id(p,byte),
            opcode,field(p,"base_row")+address,stream,count,1,stride,1,1,0,1,1,0,-1,field(p,"bank"));
    };
    const auto timeline=[&](llvm::StringRef name,int64_t start,int64_t end) {
        mlir::OperationState s(loc,TimelineOp::getOperationName());s.addAttributes({r.getNamedAttr("name",r.getStringAttr(name)),
            r.getNamedAttr("start",r.getI64IntegerAttr(start)),r.getNamedAttr("end",r.getI64IntegerAttr(end))});r.create(s);
    };
    const auto alu=[&](int64_t at,int col,llvm::StringRef opcode,llvm::StringRef a,int ai,
                       llvm::StringRef b,int bi,float immediate,int64_t count,bool reset=false,bool acc=false,bool emit=true,int output=-1) {
        auto v=ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),at,col,opcode,a,ai,0.f,b,bi,immediate,
            "fp32",output,count,1,"east","west",-1,reset,acc,emit,false,solve?4:2);
        if(a.starts_with("stream_"))v->setAttr("lhs_stream_source",r.getStringAttr("local"));
        if(b.starts_with("stream_"))v->setAttr("rhs_stream_source",r.getStringAttr("local"));
    };
    llvm::SmallVector<int64_t> map;for(int i=0;i<32;++i)map.push_back(i);
    int64_t cursor=64;
    if(!solve) {
        auto words=op->getAttrOfType<mlir::ArrayAttr>("solve.value_words");
        // The two SXMs split FP32 byte planes instead of repeating the same
        // transpose on both hemispheres. Each result is written locally and
        // returned through the passive SR bridge to the other hemisphere.
        // Alternate coefficient banks so the next token can be broadcast
        // while VXM consumes the current token, without interleaving macros
        // in one MEM ICU queue.
        auto alternate=get("solve.coefficients_alternate");
        if(target.memory().banks_per_slice<2 || !alternate)
            return op.emitError("head_scale requires two planned coefficient banks");
        const int64_t work=half*blocks;
        int64_t ready=0,period=std::max<int64_t>(32,work);
        int64_t writeOffset[2][4]{};
        for(int side=0;side<2;++side)for(int b=0;b<2;++b) {
            const int plane=side*2+b;
            const auto local=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(table,plane));
            const auto cross=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0)
                +*target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(table,plane));
            writeOffset[side][plane]=35+local;
            writeOffset[1-side][plane]=35+cross;
        }
        for(int side=0;side<2;++side)for(int b=0;b<4;++b)
            ready=std::max(ready,writeOffset[side][b]+heads+1+id(table,b)/4);
        // A bank is reused two tokens later. Its previous READ3D must finish
        // before its next WRITE3D begins, including asymmetric SR distances.
        for(int side=0;side<2;++side)for(int b=0;b<4;++b)
            period=std::max(period,(ready-1-id(table,b)/4+work-writeOffset[side][b]+1)/2);
        const auto firstCapture=cursor+32;
        for(int64_t t=0;t<32;++t) {
            const auto coefficients=t%2?alternate:table;
            const int64_t capture=firstCapture+t*period,start=capture+ready;
            for(int side=0;side<2;++side) {
                attention_detail::emitSxm(r,loc,capture,side,"transpose",{16,17},{56,57},map,"wide_matrix",-1,-1,-1,1,32);
                for(int b=0;b<2;++b) {
                    const auto plane=side*2+b;
                    const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(rhs,plane));
                    mem(rhs,plane,side,capture-rd,"read",t,16+b,32,0);
                    // Preserve the local result for its passive cross-side
                    // consumer; an ordinary consuming write would cut it off.
                    mem(coefficients,plane,side,capture+writeOffset[side][plane],"write_tap",0,56+b,heads);
                    mem(coefficients,plane,1-side,capture+writeOffset[1-side][plane],"write",0,24+b,heads);
                }
            }
            timeline("head_scale.beta_broadcast",capture,start);
            for(int64_t pair=0;pair<half;++pair) {
                for(int side=0;side<2;++side) {
                    const int64_t h=2*pair+side;
                    // Each head is one affine feature-block domain. Resolve
                    // producer addresses before issuing anything; no V copy.
                    auto p=lhs;int64_t address=h*blocks*32+t,stride=32;
                    if(words) {
                        auto w=llvm::cast<mlir::DictionaryAttr>(words[t*heads*blocks+h*blocks]);
                        p=w.getAs<mlir::DictionaryAttr>("placement");address=field(w,"address")-field(p,"base_row");
                        if(blocks>1) {
                            auto next=llvm::cast<mlir::DictionaryAttr>(words[t*heads*blocks+h*blocks+1]);
                            stride=field(next,"address")-field(w,"address");
                        }
                        if(p.getAs<mlir::StringAttr>("hemisphere").getValue()!=(side?"west":"east"))
                            return op.emitError("native head_scale ownership h=") << h << " side=" << side << " source=" << p;
                    }
                    bool affine=true;
                    for(int64_t block=0;block<blocks;++block) {
                        if(words) {
                            auto w=llvm::cast<mlir::DictionaryAttr>(words[t*heads*blocks+h*blocks+block]);
                            auto native=w.getAs<mlir::DictionaryAttr>("placement");
                            if(native.getAs<mlir::StringAttr>("hemisphere").getValue()!=(side?"west":"east"))
                                return op.emitError("head_scale feature block crosses hemisphere ownership");
                            affine &= native==p
                                && field(w,"address")==field(p,"base_row")+address+block*stride;
                        }
                    }
                    const auto headStart=start+pair*blocks;
                    if(affine) {
                        for(int b=0;b<2;++b)mem(p,b,side,headStart-1-id(p,b)/4,"read",address,32+side*16+b,blocks,stride);
                    } else {
                        for(int64_t block=0;block<blocks;++block) {
                            auto w=llvm::cast<mlir::DictionaryAttr>(words[t*heads*blocks+h*blocks+block]);
                            auto native=w.getAs<mlir::DictionaryAttr>("placement");
                            for(int b=0;b<2;++b)mem(native,b,side,headStart+block-1-id(native,b)/4,"read",field(w,"address")-field(native,"base_row"),32+side*16+b);
                        }
                    }
                }
            }
            for(int side=0;side<2;++side)for(int b=0;b<4;++b) {
                direct_domain_detail::emitMem3D(r,loc,target,start-1-id(coefficients,b)/4,
                    side*target.memory().slices_per_hemisphere+id(coefficients,b),"read",
                    field(coefficients,"base_row")+side,32+side*16+4+b,
                    blocks,1,0,half,blocks,2,1,1,0,-1,field(coefficients,"bank"));
                direct_domain_detail::emitMem3D(r,loc,target,start+3+id(out,b)/4,
                    (1-side)*target.memory().slices_per_hemisphere+id(out,b),"write",
                    field(out,"base_row")+t,side*8+b,
                    blocks,1,rows,half,blocks,32,1,1,0,-1,field(out,"bank"));
            }
            alu(start-1,0,"multiply","stream_bf16",0,"stream_f32",4,0,work);
            alu(start-1,1,"bypass","previous",0,"immediate",0,0,work,false,false,true,0);
            cursor=start+work+3+id(out,3)/4+16;
            timeline("head_scale.beta_v",start,start+work);
        }
        // No reload point inside this overlapped producer/consumer pipeline.
        timeline("imem.drain.head_scale.complete",cursor-17,cursor-16);
    } else {
        auto alternate=get("solve.coefficients_alternate");
        if(target.memory().banks_per_slice<2 || !alternate)
            return op.emitError("unit_lower_solve requires two planned coefficient banks");
        const auto coefficients=[&](int64_t row) {return row%2?alternate:table;};
        auto weightTile=get("solve_block.weight_tile"),crossProduct=get("solve_block.product");
        if(blocked && (!weightTile || !crossProduct || target.throughput().mxms_per_hemisphere!=1))
            return op.emitError("blocked unit-lower solve requires one MXM per hemisphere and planned tile/product storage");
        if(blocked) {
            // MXM consumes a full K=32 tile. Initialize only the future D1
            // output rows; they serve as the zero-padded lower half of D0.
            // The solve later overwrites these rows with the actual answer.
            const int64_t count=half*blocks*16,start=cursor;
            alu(start-1,0,"multiply","immediate",0,"immediate",0,0,count);
            alu(start-1,1,"bypass","previous",0,"immediate",0,0,count);
            alu(start-1,2,"bypass","previous",0,"immediate",0,0,count);
            alu(start-1,3,"bypass","previous",0,"immediate",0,0,count,false,false,true,2);
            for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte)
                direct_domain_detail::emitMem3D(r,loc,target,start+5+id(out,byte)/4,
                    side*target.memory().slices_per_hemisphere+id(out,byte),"write",
                    field(out,"base_row")+16,(1-side)*8+2+byte,
                    16,1,1,half*blocks,16,32,1,1,0,-1,field(out,"bank"));
            cursor=start+count+5+id(out,3)/4+32;
            timeline("unit_lower_solve_blocked.zero_padding",start,cursor-32);
        }
        const auto updateBlock=[&](int64_t pair) {
            // L and the completed D0 output share head ownership. Transpose
            // D0 directly from its final home, including the initialized zeros.
            // One BF16 product per V tile; no split components or inverse.
            const auto begin=cursor;
            for(int64_t block=0;block<blocks;++block) {
                const auto capture=cursor+32;int64_t transposed=capture;
                for(int side=0;side<2;++side) {
                    attention_detail::emitSxm(r,loc,capture,side,"transpose",{16,17},{56,57},map,"wide_matrix",-1,-1,-1,1,32);
                    for(int byte=0;byte<2;++byte) {
                        const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(out,byte+2));
                        const auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(weightTile,byte));
                        mem(out,byte+2,side,capture-rd,"read",block*rows+pair*32,16+byte,32);
                        mem(weightTile,byte,side,capture+35+wr,"write",0,56+byte,32);
                        transposed=std::max(transposed,capture+35+wr+32);
                    }
                }
                const auto load=transposed+32,compute=load+32;
                for(int side=0;side<2;++side) {
                    for(int byte=0;byte<2;++byte) {
                        const auto iw=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmWeight,target::StreamDirection::East,id(weightTile,byte));
                        const auto ar=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmActivation,target::StreamDirection::East,id(lhs,byte+2));
                        mem(weightTile,byte,side,load-iw,"read",0,16+byte,32);
                        mem(lhs,byte+2,side,compute-ar,"read",(partition(lhs)?pair:side*half+pair)*32+16,4+byte,16);
                    }
                    attention_detail::MxmDomain3D domain;domain.repeat_count=32;domain.repeat_weight_column_stride=1;
                    attention_detail::emitMxm3D(r,loc,load,side,"iw",0,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",domain,16);
                    attention_detail::emitMxm(r,loc,compute,side,"compute",0,0,4,24,
                        16,1,0,1,"stream",true,"block",0,"bf16","direct16","fp32");
                }
                const auto returning=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,0)
                    +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
                const auto arrival=compute+target.mxm_first_result_latency()+returning;
                int64_t end=arrival;
                for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte) {
                    const auto wr=*target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(crossProduct,byte));
                    mem(crossProduct,byte,1-side,arrival+wr,"write",block*16,24+byte,16);
                    end=std::max(end,arrival+wr+16);
                }
                cursor=end+32;
                timeline("imem.drain.unit_lower_solve_blocked.product"+std::to_string(pair*blocks+block),end,cursor-16);
            }
            timeline("unit_lower_solve_blocked.L10_D0",begin,cursor);
        };
        // Leave room for raw configuration framing, waits and queue setup.
        const int64_t reloadRows=std::max<int64_t>(1,std::min<int64_t>(8,
            target.icu_memory().vxm_imem_depth/(32*blocks)));
        // Prefetch one independent L row while VXM solves the previous row.
        // Both byte-pair tiles stay on their owning head hemisphere, and
        // cross passive SR into the consumer's coefficient bank.
        for(int64_t pair=0;pair<half;++pair) {
            auto epochBegin=cursor;
            int64_t ready[32]{},bankFree[2]{},nextCapture=cursor+32;
            int64_t coefficientDrain=cursor;
            const auto prefetch=[&](int64_t row) {
                const int64_t first=blocked && row>=16?16:0;
                auto dst=coefficients(row);
                int64_t capture=nextCapture;
                for(int byte=0;byte<4;++byte) {
                    const auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0)
                        +*target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(dst,byte));
                    capture=std::max(capture,bankFree[row%2]-(byte/2)*32-35-wr);
                }
                int64_t firstRead=capture;
                for(int pass=0;pass<2;++pass)for(int side=0;side<2;++side) {
                    const auto at=capture+pass*32;
                    attention_detail::emitSxm(r,loc,at,side,"transpose",{16,17},{56,57},map,"wide_matrix",-1,-1,-1,1,32);
                    for(int b=0;b<2;++b) {
                        const int byte=pass*2+b;
                        const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(lhs,byte));
                        const auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0)
                            +*target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(dst,byte));
                        firstRead=std::min(firstRead,at-rd);
                        mem(lhs,byte,side,at-rd,"read",(partition(lhs)?pair:side*half+pair)*32+row,16+b,32,0);
                        mem(dst,byte,1-side,at+35+wr+first,"write",0,24+b,row-first);
                        // The whole WRITE3D retires before this bank's READ3D.
                        ready[row]=std::max(ready[row],at+35+wr+row+id(dst,byte)/4+target.throughput().tile_rows-1);
                        coefficientDrain=std::max(coefficientDrain,at+70+wr);
                    }
                }
                nextCapture=capture+64;
                timeline("unit_lower_solve.coefficient_row",firstRead,ready[row]);
            };
            for(int64_t i=0;i<32;++i) {
                const int64_t n=blocked?i%16:i;
                if(blocked && i==16) {
                    updateBlock(pair);nextCapture=cursor+32;epochBegin=cursor;
                }
                if(n && i%reloadRows==0)prefetch(i);
                auto coeff=coefficients(i);
                const int64_t begin=std::max(cursor,ready[i]);
                // C3 has up to three three-word accumulator configurations.
                // Nine cycles sustain their fetch rate; longer dot products
                // run back-to-back without draining the MEM/VXM pipeline.
                // A one-cycle RHS Subtract must not enter immediately after
                // the previous vector's two-cycle Multiply (output collision).
                const int64_t interval=std::max<int64_t>(n+1+(blocked && i>=16?1:0),9);
                int64_t drain=begin,lastOut=begin;
                for(int64_t block=0;block<blocks;++block) {
                    const int64_t start=begin+block*interval;
                    for(int side=0;side<2;++side)for(int b=0;b<4;++b) {
                        // L is transposed on its owning side, then crosses SR
                        // to the opposite VXM. The RHS must name that same
                        // head: east VXM consumes the second logical half,
                        // west VXM the first. Native partitioned RHS storage
                        // already encodes this ownership; mirrored FP32 storage
                        // needs the opposite half in its row address.
                        const auto address=block*(partition(rhs)?rows:heads*32)+(partition(rhs)?pair:(1-side)*half+pair)*32+i;
                        mem(rhs,b,side,start-1-id(rhs,b)/4,"read",address,32+side*16+b);
                        if(blocked && i>=16)
                            mem(crossProduct,b,side,start-1-id(crossProduct,b)/4,"read",block*16+n,32+side*16+4+b);
                        if(n) {
                            mem(feedback,b,side,start-id(feedback,b)/4,"read",block*32,32+side*16+b,n);
                            mem(coeff,b,side,start-id(coeff,b)/4,"read",0,32+side*16+4+b,n);
                            bankFree[i%2]=std::max(bankFree[i%2],start-id(coeff,b)/4+n);
                        }
                        // Subtract is one cycle; Multiply is two. For the
                        // first row of D1 no later dot-product term delays it.
                        const int64_t write=start+n+5+id(out,b)/4-(blocked && i==16?1:0);
                        mem(out,b,1-side,write,"write",block*rows+pair*32+i,side*8+2+b);
                        lastOut=std::max(lastOut,write+1);
                        drain=std::max(drain,write+4);
                    }
                    if(blocked && i>=16)alu(start-1,0,"subtract","stream_f32",0,"stream_f32",4,0,1);
                    else alu(start-1,0,"multiply","stream_f32",0,"immediate",0,1.f,1);
                    if(n)alu(start,0,"multiply","stream_f32",0,"stream_f32",4,0,n);
                    alu(start-1,1,"bypass","previous",0,"immediate",0,0,1);
                    if(n)alu(start,1,"negate","previous",0,"immediate",0,0,n);
                    alu(start-1,2,"bypass","previous",0,"immediate",0,0,n+1);
                    alu(start-1,3,"add","previous",0,"accumulator",0,0,1,true,true,n==0,n==0?2:-1);
                    if(n>1)alu(start,3,"add","previous",0,"accumulator",0,0,n-1,false,true,false);
                    if(n)alu(start+(n>1?1:0),3,"add","previous",0,"accumulator",0,0,1,false,true,true,2);
                }
                timeline("unit_lower_solve.forward_row",begin,drain);
                cursor=std::max(drain,begin+blocks*interval);
                if(i<31 && (!blocked || i!=15)) {
                    // All output writes retire before the passive return READ.
                    // Feedback writes retire before next-row feedback READs.
                    const auto copy=lastOut+target.throughput().tile_rows-1;int64_t end=copy;
                    for(int side=0;side<2;++side)for(int b=0;b<4;++b) {
                        const auto delay=id(out,b)/4+id(feedback,b)/4+2;
                        // Keep this return off SXM's concurrent W24/E24 pair.
                        mem(out,b,1-side,copy,"read",pair*32+i,32+feedbackStream+b,blocks,rows);
                        mem(feedback,b,side,copy+delay,"write",n,feedbackStream+b,blocks,32);
                        end=std::max(end,copy+delay+blocks);
                        cursor=std::max(cursor,copy+delay+blocks+id(feedback,b)/4+target.throughput().tile_rows-1);
                    }
                    timeline("unit_lower_solve.feedback_passive_sr",copy,end);
                    if((i+1)%reloadRows)prefetch(i+1);
                }
                if((i+1)%reloadRows==0 || i==31) {
                    // Bound raw accumulator configurations to finite iMEM;
                    // never cut through an overlapped coefficient prefetch.
                    cursor=std::max(cursor,coefficientDrain)+16;
                    timeline("imem.drain.unit_lower_solve.rows"+std::to_string(pair*32+i),epochBegin,cursor);
                    cursor+=32;nextCapture=cursor+32;epochBegin=cursor;
                }
            }
        }
    }
    timeline(blocked?"unit_lower_solve_blocked.mixed":solve?"unit_lower_solve.fp32":"head_scale.fp32",32,cursor);
    mlir::OperationState binding(loc,BindingOp::getOperationName());binding.addTypes(op.getResult().getType());
    binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),r.getNamedAttr("access",r.getStringAttr("output")),
        r.getNamedAttr("role",r.getStringAttr("result")),r.getNamedAttr("bytes",r.getI64IntegerAttr(op.getResult().getType().getNumElements()*4)),r.getNamedAttr("placement",out)});
    r.replaceOp(op,llvm::cast<BindingOp>(r.create(binding)).getValue());return mlir::success();
}
