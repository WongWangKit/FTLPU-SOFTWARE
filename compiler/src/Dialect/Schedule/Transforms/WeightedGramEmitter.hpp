// Generic per-head Gram matrix with a pointwise weight epilogue.
// Included in VectorAluScheduleEmitter's private namespace.
mlir::LogicalResult emitWeightedGram(mlir::IRRewriter& r,kernel::VectorAluOp op,
                                    const target::LPUTargetModel& target)
{
    r.setInsertionPoint(op);
    const auto loc=op.getLoc();
    auto key=op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_lhs_placement");
    auto weights=op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement");
    auto out=op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement");
    auto table=op->getAttrOfType<mlir::DictionaryAttr>("weighted_gram.r_table");
    const bool nativeR=op.getRhs().getType().getElementType().isBF16();
    if(nativeR)table=weights;
    const int64_t heads=nativeR?op.getRhs().getType().getDimSize(0)/32:op.getRhs().getType().getDimSize(1);
    const int64_t dim=op.getLhs().getType().getDimSize(1)/heads;
    const auto wide=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    if(!wide || !wide.getValue() || target.memory().bytes_per_word!=32 ||
        target.memory().slices_per_hemisphere<50 || target.throughput().mxm_weight_buffers<2 ||
        target.throughput().mxms_per_hemisphere!=1 || !table)
        return op.emitError("weighted_gram requires wide SXM and planned 32-lane storage");
    const auto field=[](mlir::DictionaryAttr p,llvm::StringRef name) {return p.getAs<mlir::IntegerAttr>(name).getInt();};
    const auto id=[](mlir::DictionaryAttr p,int64_t b) {return llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[b]).getInt();};
    const bool striped=static_cast<bool>(key.getAs<mlir::IntegerAttr>("bank_tile_rows"));
    const bool distributed=key.getAs<mlir::ArrayAttr>("slices").size()==16;
    const auto mem=[&](mlir::DictionaryAttr p,int64_t byte,int64_t side,int64_t cycle,
                       llvm::StringRef opcode,int64_t address,int64_t stream,
                       int64_t count,int64_t stride=1,int64_t interval=1,int64_t bankOverride=-1) {
        direct_domain_detail::emitMem3D(r,loc,target,cycle,side*target.memory().slices_per_hemisphere+id(p,byte),
            opcode,field(p,"base_row")+address,stream,count,interval,stride,1,1,0,1,1,0,-1,bankOverride<0?field(p,"bank"):bankOverride);
    };
    const auto timeline=[&](llvm::StringRef name,int64_t start,int64_t end) {
        mlir::OperationState state(loc,TimelineOp::getOperationName());
        state.addAttributes({r.getNamedAttr("name",r.getStringAttr(name)),
            r.getNamedAttr("start",r.getI64IntegerAttr(start)),r.getNamedAttr("end",r.getI64IntegerAttr(end))});
        r.create(state);
    };
    // R is [i,j,head], while each MXM result vector spans j for one head.
    // Transpose each i tile once; never expand a scalar coefficient table.
    const int64_t capture=96;
    if(!nativeR) for(int64_t side=0;side<2;++side) {
        llvm::SmallVector<int64_t> map;
        for(int64_t i=0;i<32;++i)map.push_back(i);
        attention_detail::emitSxm(r,loc,capture,side,"transpose",{16,17},
            {32+side*16+4,33+side*16+4},map,"wide_matrix",-1,-1,-1,32,32);
        for(int64_t b=0;b<2;++b) {
            const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(weights,b+2));
            const auto wr=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,id(table,b));
            mem(weights,b+2,side,capture-rd,"read",0,16+b,1024);
            mem(table,b,side,capture+35+wr,"write",0,32+side*16+4+b,1024);
        }
    }
    int64_t cursor=nativeR?32:capture+1024+96;
    if(striped && (!distributed || field(key,"bank_tile_rows")!=4 || dim!=128 ||
        !target.supports_mxm_weight_activation_overlap()))
        return op.emitError("striped Gram requires D128, distributed16 and weight/activation overlap");
    if(!nativeR) timeline("weighted_gram.r_head_transpose",capture,cursor);
    const int64_t begin=cursor;
    int64_t drain=cursor;
    const auto readKey=[&](int64_t side,int64_t h,int64_t kb,int64_t at,bool weight) {
        for(int64_t p=0;p<(distributed?8:1);++p)for(int64_t b=0;b<2;++b) {
            const int64_t byte=distributed?2*p+b:b;
            const auto endpoint=weight?target::StreamEndpoint::MxmWeight:target::StreamEndpoint::MxmActivation;
            const auto latency=*target.transport_latency(target::StreamEndpoint::Mem,endpoint,target::StreamDirection::East,id(key,byte));
            const int64_t address=distributed?(h*(dim/32)+kb)*4:(h*(dim/32)+kb)*32;
            mem(key,byte,side,at+(distributed?p:0)-latency,"read",address,
                (weight?16:4)+b,distributed?4:32,1,distributed?8:1,striped?(field(key,"bank")^(kb%2)):-1);
        }
    };
    for(int64_t pair=0;pair<heads/2;++pair) {
        int64_t lastCompute=0;
        for(int64_t kb=0;kb<dim/32;++kb) {
            const int64_t load=cursor+32,compute=load+32,buffer=kb%2;
            const bool last=kb+1==dim/32;
            for(int64_t side=0;side<2;++side) {
                const int64_t h=pair*2+1-side,queue=side*target.throughput().mxms_per_hemisphere;
                readKey(side,h,kb,load,true);
                attention_detail::MxmDomain3D domain;
                domain.repeat_count=32;domain.repeat_weight_column_stride=1;
                attention_detail::emitMxm3D(r,loc,load,queue,"iw",buffer,0,0,0,0,1,
                    "stream",true,"vector",0,"bf16","direct16","",domain,16);
                readKey(side,h,kb,compute,false);
                attention_detail::emitMxm(r,loc,compute,queue,"compute",buffer,0,4,side*16,
                    32,1,0,1,last?"stream":"sram",last,"block",0,"bf16","direct16",last?"bf16":"");
            }
            lastCompute=compute;
            // Alternating K banks permit next-tile IW while this tile computes.
            // Each bank still retires its complete weight-read domain before
            // its activation-read domain. Single-bank inputs retain serial IW.
            cursor=striped?load:compute;
        }
        const auto returning=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,0)
            +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
        const int64_t arrival=lastCompute+target.mxm_first_result_latency()+returning;
        // MXM's BF16 output enters fixed C0 LHS; R occupies fixed C0 RHS.
        // The Gram matrix is never written to/read from SRAM.
        // C1 has fixed output ports 0..3 (8..11 on the other hemisphere).
        // Next-head activation reads use selectable MXM ports 4/5;
        // weights use 16/17, so all three streams remain independent.
        ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),
            arrival-1,0,"multiply","stream_bf16",0,0.f,"stream_bf16",2,0.f,"fp32",-1,
            32,1,"east","west",-1,false,false,true,false,2);
        ffn_detail::create_vxm(r,loc,op.getLhs(),op.getRhs(),op.getResult().getType(),
            arrival-1,1,"bypass","previous",0,0.f,"immediate",0,0.f,"fp32",0,
            32,1,"east","west",-1,false,false,true,false,2);
        for(int64_t side=0;side<2;++side) {
            const int64_t h=pair*2+1-side;
            for(int64_t b=0;b<2;++b)
                mem(table,b,side,arrival-1-id(table,b)/4,"read",h,32+side*16+2+b,32,nativeR?heads:32);
            for(int64_t b=0;b<4;++b)
                mem(out,b,1-side,arrival+3+id(out,b)/4,"write",pair*32,side*8+b,32);
        }
        drain=std::max(drain,arrival+3+id(out,3)/4+32+32);
        if(!striped) cursor=drain;
    }
    timeline("weighted_gram.mxm_to_vxm",begin,drain);
    mlir::OperationState binding(loc,BindingOp::getOperationName());
    binding.addTypes(op.getResult().getType());
    binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),
        r.getNamedAttr("access",r.getStringAttr("output")),r.getNamedAttr("role",r.getStringAttr("result")),
        r.getNamedAttr("bytes",r.getI64IntegerAttr(op.getResult().getType().getNumElements()*4)),
        r.getNamedAttr("placement",out)});
    r.replaceOp(op,llvm::cast<BindingOp>(r.create(binding)).getValue());
    return mlir::success();
}
