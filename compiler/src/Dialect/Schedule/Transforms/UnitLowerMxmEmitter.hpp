// Chunk32 polynomial inverse and head-wise matmul. All dynamic arithmetic is
// emitted as real MXM/VXM instructions; the second inverse operand is immutable I.
mlir::LogicalResult emitUnitLowerMxm(mlir::IRRewriter& r,kernel::VectorAluOp op,
                                   const target::LPUTargetModel& target,
                                   kernel::VectorAluOp epilogue = {},
                                   kernel::VectorAluOp cast = {}) {
    r.setInsertionPoint(op);const auto loc=op.getLoc();
    const bool inverse=op.getKind()=="unit_lower_inverse";
    const auto precision=op->getAttrOfType<mlir::StringAttr>("mxm_precision");
    const bool singleBf16=!inverse && precision && precision.getValue()=="bf16";
    const int64_t heads=op.getLhs().getType().getDimSize(0)/32,half=heads/2,rows=half*32;
    const int64_t blocks=op.getResult().getType().getDimSize(1)/32;
    auto get=[&](llvm::StringRef n){return op->getAttrOfType<mlir::DictionaryAttr>(n);};
    auto tail=cast?cast:epilogue?epilogue:op;
    auto lhs=get("ftlpu.vector_lhs_placement"),rhs=get("ftlpu.vector_rhs_placement");
    auto out=tail->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement");
    auto prior=epilogue?epilogue->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement"):mlir::DictionaryAttr{};
    if(epilogue && (!singleBf16 || epilogue.getKind()!="add" || epilogue.getUseImmediate() || epilogue.getReduce() ||
        epilogue.getLhs()!=op.getResult() || !op.getResult().hasOneUse() ||
        epilogue.getRhs().getType()!=op.getResult().getType() || epilogue.getResult().getType()!=op.getResult().getType()))
        return op.emitError("head_matmul epilogue requires one FP32 matrix add after a BF16 product");
    if(cast && (!epilogue || cast.getKind()!="multiply" || !cast.getUseImmediate() || cast.getReduce() ||
        cast.getImmediate().convertToFloat()!=1.f || cast.getLhs()!=epilogue.getResult() ||
        !cast.getResult().getType().getElementType().isBF16() || cast.getResult().getType().getShape()!=op.getResult().getType().getShape() ||
        !llvm::all_of(epilogue.getResult().getUsers(),[&](mlir::Operation *user){return user==cast.getOperation();})))
        return op.emitError("head_matmul cast requires a sole immediate-one BF16 consumer of the FP32 add");
    auto tile=get("solve_mxm.weight_tile");
    const auto wide=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    if(!wide || !wide.getValue() || target.memory().bytes_per_word!=32 ||
       target.memory().slices_per_hemisphere<52 || target.memory().banks_per_slice<2 ||
       target.throughput().mxms_per_hemisphere!=1)
        return op.emitError("unit_lower_inverse/head_matmul require wide SXM, 32 lanes, two MEM banks and one MXM per hemisphere");
    const auto field=[](mlir::DictionaryAttr p,llvm::StringRef n){return p.getAs<mlir::IntegerAttr>(n).getInt();};
    const auto id=[](mlir::DictionaryAttr p,int64_t b){return llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[b]).getInt();};
    const auto partition=[](mlir::DictionaryAttr p){return p.getAs<mlir::StringAttr>("kind").getValue().starts_with("token_partition");};
    const auto address=[&](mlir::DictionaryAttr p,int64_t pair,int side,int64_t block=0) {
        return block*(partition(p)?rows:heads*32)+(partition(p)?pair:side*half+pair)*32;
    };
    const auto mem=[&](mlir::DictionaryAttr p,int b,int side,int64_t cycle,llvm::StringRef opcode,
                       int64_t offset,int stream,int64_t count=32) {
        direct_domain_detail::emitMem3D(r,loc,target,cycle,side*target.memory().slices_per_hemisphere+id(p,b),
            opcode,field(p,"base_row")+offset,stream,count,1,1,1,1,0,1,1,0,-1,field(p,"bank"));
    };
    const auto timeline=[&](llvm::StringRef name,int64_t start,int64_t end) {
        mlir::OperationState s(loc,TimelineOp::getOperationName());s.addAttributes({r.getNamedAttr("name",r.getStringAttr(name)),
            r.getNamedAttr("start",r.getI64IntegerAttr(start)),r.getNamedAttr("end",r.getI64IntegerAttr(end))});r.create(s);
    };
    const auto alu=[&](int64_t at,llvm::StringRef opcode) {
        auto v=ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),at,0,opcode,
            "stream_f32",0,0.f,"stream_f32",4,0.f,"fp32",-1,32,1,"east","west",-1,false,false,true,false,2);
        v->setAttr("lhs_stream_source",r.getStringAttr("local"));v->setAttr("rhs_stream_source",r.getStringAttr("local"));
        ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),at,1,"bypass",
            "previous",0,0.f,"immediate",0,0.f,"fp32",0,32,1,"east","west",-1,false,false,true,false,2);
    };
    llvm::SmallVector<int64_t> map;for(int i=0;i<32;++i)map.push_back(i);
    int64_t cursor=64;int drainIndex=0;
    // Matrices are row vectors. A single 32x32 BF16 weight tile is transposed
    // for column-load IW. FP32 workspaces remain intact; MXM explicitly consumes
    // their upper two planes. Six partial products approximate FP32 products
    // using three BF16 components; all partials stay in MXM FP32 ACC.
    const auto partialProduct=[&](mlir::DictionaryAttr a,mlir::DictionaryAttr b,mlir::DictionaryAttr dst,
                           int64_t pair,int64_t block,bool externalA,bool externalB,bool externalDst,
                           mlir::DictionaryAttr add,llvm::StringRef name,bool final,bool newWeights) {
        const int64_t begin=cursor,capture=cursor+32;
        int64_t transposed=capture;
        if(newWeights)for(int side=0;side<2;++side) {
            attention_detail::emitSxm(r,loc,capture,side,"transpose",{16,17},{56,57},map,"wide_matrix",-1,-1,-1,1,32);
            for(int byte=0;byte<2;++byte) {
                auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(b,byte+2));
                auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(tile,byte));
                mem(b,byte+2,side,capture-rd,"read",externalB?address(b,pair,side,block):0,16+byte);
                mem(tile,byte,side,capture+35+wr,"write",0,56+byte);
                transposed=std::max(transposed,capture+35+wr+32);
            }
        }
        const int64_t load=transposed+32,compute=newWeights?load+32:cursor;
        for(int side=0;side<2;++side) {
            for(int byte=0;byte<2;++byte) {
                auto iw=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmWeight,target::StreamDirection::East,id(tile,byte));
                auto ar=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmActivation,target::StreamDirection::East,id(a,byte+2));
                if(newWeights)mem(tile,byte,side,load-iw,"read",0,16+byte);
                mem(a,byte+2,side,compute-ar,"read",externalA?address(a,pair,side):0,4+byte);
            }
            attention_detail::MxmDomain3D domain;domain.repeat_count=32;domain.repeat_weight_column_stride=1;
            if(newWeights)attention_detail::emitMxm3D(r,loc,load,side,"iw",0,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",domain,16);
            attention_detail::emitMxm(r,loc,compute,side,"compute",0,0,4,add?side*16:24,
                32,1,0,1,final?"stream":"sram",final,"block",0,"bf16","direct16",final?"fp32":"");
        }
        const auto returning=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,0)
            +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
        const int64_t arrival=compute+target.mxm_first_result_latency()+returning;
        if(!final) { cursor=compute+32; return; }
        if(add)alu(arrival-1,"add");
        int64_t end=arrival;
        for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte) {
            if(add)mem(add,byte,side,arrival-1-id(add,byte)/4,"read",0,32+side*16+4+byte);
            const auto wr=add?2+id(dst,byte)/4:
                *target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(dst,byte));
            mem(dst,byte,1-side,arrival+wr,"write",externalDst?block*rows+pair*32:0,add?side*8+byte:24+byte);
            end=std::max(end,arrival+wr+32);
        }
        timeline(name,begin,end);cursor=end+32;
        timeline("imem.drain.solve_mxm."+std::to_string(drainIndex++),end,cursor-16);
    };
    struct Matrix { mlir::DictionaryAttr p; bool external; };
    // x and BF16(x) use the same four incoming planes: the BF16 RHS aliases
    // planes 2/3 of the FP32 LHS. One MEM read supplies both operands.
    const auto residual=[&](Matrix src,mlir::DictionaryAttr dst,int64_t pair,int64_t block) {
        auto bridge=get("solve_mxm.residual_bridge");
        const auto start=cursor;int64_t end=start;
        auto v=ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),start-1,0,"subtract",
            "stream_f32",0,0.f,"stream_bf16",2,0.f,"fp32",-1,32,1,"east","west",-1,false,false,true,false,2);
        v->setAttr("lhs_stream_source",r.getStringAttr("local"));v->setAttr("rhs_stream_source",r.getStringAttr("local"));
        ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),start-1,1,"bypass",
            "previous",0,0.f,"immediate",0,0.f,"fp32",0,32,1,"east","west",-1,false,false,true,false,2);
        for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte) {
            mem(src.p,byte,side,start-1-id(src.p,byte)/4,"read",src.external?address(src.p,pair,side,block):0,32+side*16+byte);
            mem(bridge,byte,1-side,start+2+id(bridge,byte)/4,"write",0,side*8+byte);
            end=std::max(end,start+2+id(bridge,byte)/4+32);
        }
        const auto copy=end+16;
        for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte) {
            const auto delay=id(bridge,byte)/4+id(dst,byte)/4+2;
            mem(bridge,byte,1-side,copy,"read",0,56+byte);
            mem(dst,byte,side,copy+delay,"write",0,24+byte);
            end=std::max(end,copy+delay+32);
        }
        timeline("solve_mxm.split_fp32",start,end);cursor=end+32;
    };
    const auto product=[&](mlir::DictionaryAttr a,mlir::DictionaryAttr b,mlir::DictionaryAttr dst,
                           int64_t pair,int64_t block,bool externalA,bool externalB,bool externalDst,
                           mlir::DictionaryAttr add,llvm::StringRef name) {
        const auto begin=cursor;
        Matrix av[3]={{a,externalA},{get("solve_mxm.a1"),false},{get("solve_mxm.a2"),false}};
        Matrix bv[3]={{b,externalB},{get("solve_mxm.b1"),false},{get("solve_mxm.b2"),false}};
        residual(av[0],av[1].p,pair,0);residual(av[1],av[2].p,pair,0);
        if(a==b && externalA==externalB && block==0) { bv[1]=av[1];bv[2]=av[2]; }
        else {residual(bv[0],bv[1].p,pair,block);residual(bv[1],bv[2].p,pair,0);}
        // Terms with total component degree <= 2. Reuse each weight component
        // for all matching activation components; high*high is accumulated last.
        const int terms[6][2]={{0,2},{1,1},{0,1},{2,0},{1,0},{0,0}};
        for(int term=0;term<6;++term) {
            auto x=av[terms[term][0]],y=bv[terms[term][1]];
            partialProduct(x.p,y.p,dst,pair,block,x.external,y.external,externalDst,add,name,term==5,term==0 || terms[term][1]!=terms[term-1][1]);
        }
        timeline("solve_mxm.compensated_product",begin,cursor-32);
    };
    if(singleBf16) {
        // Native BF16 operands already occupy two byte planes. FP32 operands
        // retain the existing upper-plane BF16 interpretation. This changes
        // only the MEM source addresses, not MXM arithmetic or accumulation.
        const int lhsPlane=op.getLhs().getType().getElementType().isBF16()?0:2;
        const int rhsPlane=op.getRhs().getType().getElementType().isBF16()?0:2;
        const auto eastFirst=[](mlir::DictionaryAttr p) {
            return p.getAs<mlir::StringAttr>("kind").getValue()!="token_partition_planar";
        };
        const auto mirrored=[](mlir::DictionaryAttr p) {
            const auto kind=p.getAs<mlir::StringAttr>("kind").getValue();
            return p.getAs<mlir::StringAttr>("hemisphere").getValue()=="both" &&
                (p.get("mxm_lhs_head_rows") || kind=="fp16_pair_planar" || kind=="fp32_quad_planar");
        };
        // Replicated A/B contain both logical head halves on either side.
        // Select the local half already held by a partitioned prior sum;
        // this avoids mirroring that sum between successive K reductions.
        const bool selectPriorHalf=epilogue && prior && partition(prior) && mirrored(lhs) && mirrored(rhs);
        const bool inputEastFirst=selectPriorHalf?eastFirst(prior):eastFirst(lhs);
        if(epilogue && (!prior || prior.getAs<mlir::ArrayAttr>("slices").size()!=4 ||
            prior.getAs<mlir::StringAttr>("hemisphere").getValue()!="both" ||
            eastFirst(lhs)!=eastFirst(rhs) || (partition(prior) && eastFirst(prior)!=inputEastFirst) ||
            (!partition(prior) && prior.getAs<mlir::StringAttr>("kind").getValue()!="fp32_quad_planar")))
            return op.emitError("MXM add needs a mirrored FP32 prior or a partition matching the MXM input head order");
        if(target.throughput().mxm_weight_buffers<2 || !target.supports_mxm_weight_activation_overlap())
            return op.emitError("streaming head_matmul requires two overlapping MXM weight buffers");
        const auto sxmToWeight=target.transport_latency(target::StreamEndpoint::SxmResult,
            target::StreamEndpoint::MxmWeight,target::StreamDirection::East,0);
        if(!sxmToWeight)return op.emitError("head_matmul requires SXM-to-MXM weight routing");
        const int64_t capture=96,load=capture+35+*sxmToWeight,compute=load+32,jobs=half*blocks;
        const auto domainMem=[&](mlir::DictionaryAttr p,int byte,int side,int64_t at,
                                llvm::StringRef opcode,int stream,int64_t blockStride,int logicalSide=-1) {
            direct_domain_detail::emitMem3D(r,loc,target,at,
                side*target.memory().slices_per_hemisphere+id(p,byte),opcode,
                field(p,"base_row")+(partition(p)?0:(logicalSide<0?side:logicalSide)*rows),stream,
                32,1,1,blocks,32,blockStride,half,blocks*32,32,-1,field(p,"bank"));
        };
        const auto returning=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,0)
            +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
        const int64_t arrival=compute+target.mxm_first_result_latency()+returning;
        for(int side=0;side<2;++side) {
            // Input ports 16/17 continue past SXM while the next tile arrives.
            // Use different East output ports 18/19 to avoid fabric collisions.
            attention_detail::emitSxm(r,loc,capture,side,"transpose",{16,17},{18,19},map,"wide_matrix",-1,-1,-1,jobs,32);
            for(int byte=0;byte<2;++byte) {
                const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(rhs,byte+rhsPlane));
                const int logicalSide=inputEastFirst?side:1-side;
                domainMem(rhs,byte+rhsPlane,side,capture-rd,"read",16+byte,partition(rhs)?rows:heads*32,
                    selectPriorHalf?logicalSide:side);
                const int lhsHead=selectPriorHalf?logicalSide:side;
                if(auto headRows=lhs.getAs<mlir::ArrayAttr>("mxm_lhs_head_rows")) {
                    const int64_t base=llvm::cast<mlir::IntegerAttr>(headRows[lhsHead]).getInt();
                    // Residue j supplies tokens j, j+8, j+16, j+24. Each
                    // subsequent output-column tile consumes that same input.
                    for(int residue=0;residue<8;++residue) {
                        const auto slice=id(lhs,2*residue+byte);
                        const auto delay=target.transport_latency(target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmActivation,target::StreamDirection::East,slice);
                        if(!delay)return op.emitError("missing distributed MEM-to-MXM route");
                        direct_domain_detail::emitMem3D(r,loc,target,compute+residue-*delay,
                            side*target.memory().slices_per_hemisphere+slice,"read",base,
                            4+byte,4,8,1,blocks,32,0,1,1,0,-1,field(lhs,"bank"));
                    }
                } else {
                    const auto ar=*target.transport_latency(target::StreamEndpoint::Mem,
                        target::StreamEndpoint::MxmActivation,target::StreamDirection::East,id(lhs,byte+lhsPlane));
                    domainMem(lhs,byte+lhsPlane,side,compute-ar,"read",4+byte,0,lhsHead);
                }
            }
            attention_detail::MxmDomain3D iw;
            iw.repeat_count=32;iw.repeat_weight_column_stride=1;
            iw.wave_count=jobs;iw.wave_interval=32;iw.weight_buffer_mode="toggle_dim1";
            attention_detail::emitMxm3D(r,loc,load,side,"iw",0,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",iw,18);
            auto cm=iw;cm.repeat_weight_column_stride=0;
            attention_detail::emitMxm3D(r,loc,compute,side,"compute",0,0,4,epilogue?side*16:24,0,1,"stream",true,"block",0,"bf16","direct16","fp32",cm);
            for(int byte=0;byte<4;++byte) {
                if(epilogue) {
                    const int logicalSide=inputEastFirst?side:1-side;
                    direct_domain_detail::emitMem3D(r,loc,target,arrival-1-id(prior,byte)/4,
                        side*target.memory().slices_per_hemisphere+id(prior,byte),"read",
                        field(prior,"base_row")+(partition(prior)?0:logicalSide*rows),32+side*16+4+byte,
                        32,1,1,blocks,32,partition(prior)?rows:heads*32,half,blocks*32,32,-1,field(prior,"bank"));
                }
                if(cast && byte>=2)continue;
                const auto wr=epilogue?2+int(bool(cast))+id(out,byte)/4:
                    *target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(out,byte));
                domainMem(out,byte,1-side,arrival+wr,"write",epilogue?side*8+byte:24+byte,rows);
                cursor=std::max(cursor,arrival+wr+jobs*32+32);
            }
        }
        if(epilogue) {
            auto add=ffn_detail::create_vxm(r,loc,op.getLhs(),epilogue.getRhs(),epilogue.getResult().getType(),arrival-1,0,"add",
                "stream_f32",0,0.f,"stream_f32",4,0.f,"fp32",-1,jobs*32,1,"east","west",-1,false,false,true,false,2);
            add->setAttr("lhs_stream_source",r.getStringAttr("local"));add->setAttr("rhs_stream_source",r.getStringAttr("local"));
            ffn_detail::create_vxm(r,loc,op.getLhs(),epilogue.getRhs(),tail.getResult().getType(),arrival-1,1,cast?"multiply":"bypass",
                "previous",0,0.f,"immediate",0,cast?1.f:0.f,cast?"bf16":"fp32",0,jobs*32,1,"east","west",-1,false,false,true,false,2);
        }
        timeline("head_matmul.sxm_to_mxm",capture,cursor);
    } else for(int64_t pair=0;pair<half;++pair) {
        if(inverse) {
            auto t0=get("solve_mxm.t0"),t1=get("solve_mxm.t1"),p0=get("solve_mxm.p0"),p1=get("solve_mxm.p1");
            const auto start=cursor;int64_t end=start;
            for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte) {
                mem(rhs,byte,side,start-1-id(rhs,byte)/4,"read",address(rhs,pair,side),32+side*16+byte);
                mem(lhs,byte,side,start-1-id(lhs,byte)/4,"read",address(lhs,pair,side),32+side*16+4+byte);
                mem(t0,byte,1-side,start+2+id(t0,byte)/4,"write",0,side*8+byte);
                end=std::max(end,start+2+id(t0,byte)/4+32);
            }
            alu(start-1,"subtract");timeline("unit_lower_inverse.I_minus_L",start,end);cursor=end+32;
            product(lhs,lhs,p0,pair,0,true,true,false,{},"unit_lower_inverse.L2");
            for(int round=0;round<4;++round) {
                const bool last=round==3;
                product(p0,t0,last?out:t1,pair,0,false,false,last,t0,"unit_lower_inverse.update_T");
                if(!last)product(p0,p0,p1,pair,0,false,false,false,{},"unit_lower_inverse.square_P");
                std::swap(t0,t1);std::swap(p0,p1);
            }
        } else for(int64_t block=0;block<blocks;++block) {
            product(lhs,rhs,out,pair,block,true,true,true,{},"head_matmul.TB");
        }
    }
    timeline(inverse?"unit_lower_inverse.mxm":"head_matmul.mxm",32,cursor);
    mlir::OperationState binding(loc,BindingOp::getOperationName());binding.addTypes(tail.getResult().getType());
    binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),r.getNamedAttr("access",r.getStringAttr("output")),
        r.getNamedAttr("role",r.getStringAttr("result")),r.getNamedAttr("bytes",r.getI64IntegerAttr(tail.getResult().getType().getNumElements()*(cast?2:4))),r.getNamedAttr("placement",out)});
    r.replaceOp(tail,llvm::cast<BindingOp>(r.create(binding)).getValue());
    if(cast)r.eraseOp(epilogue);
    if(epilogue)r.eraseOp(op);
    return mlir::success();
}
