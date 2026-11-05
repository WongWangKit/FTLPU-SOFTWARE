// Generic per-head weighted transpose/contraction. The weight is the last
// row of an existing head matrix; no model arithmetic is delegated to CModel.
mlir::LogicalResult emitHeadWeightedProduct(mlir::IRRewriter& r,kernel::VectorAluOp scale,
    kernel::VectorAluOp product,const target::LPUTargetModel& target,kernel::VectorAluOp outputProduct={}) {
    r.setInsertionPoint(scale);const auto loc=scale.getLoc();
    const int64_t heads=scale.getRhs().getType().getDimSize(0)/32,half=heads/2;
    const int64_t kd=scale.getLhs().getType().getDimSize(1)/heads,vd=product.getResult().getType().getDimSize(1);
    auto key=scale->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_lhs_placement");
    auto weight=scale->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement");
    auto value=product->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement");
    auto out=product->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement");
    auto outputLhs=outputProduct?outputProduct->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_lhs_placement"):mlir::DictionaryAttr{};
    auto outputHome=outputProduct?outputProduct->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement"):mlir::DictionaryAttr{};
    const auto field=[](mlir::DictionaryAttr p,llvm::StringRef n){return p.getAs<mlir::IntegerAttr>(n).getInt();};
    const auto id=[](mlir::DictionaryAttr p,int b){return llvm::cast<mlir::IntegerAttr>(p.getAs<mlir::ArrayAttr>("slices")[b]).getInt();};
    const auto kind=[](mlir::DictionaryAttr p){return p.getAs<mlir::StringAttr>("kind").getValue();};
    const auto wide=scale->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
    const bool distributed=key.getAs<mlir::ArrayAttr>("slices").size()==16;
    if(!wide || !wide.getValue() || target.memory().bytes_per_word!=32 ||
       kind(value)!="token_partition_east_first_planar" || kind(out)!="token_partition_planar" ||
       kind(weight)!="fp16_pair_planar" || (distributed && kd!=128))
        return scale.emitError("weighted head product requires wide SXM and native planar/head-partition homes");
    if(outputProduct && (kind(outputLhs)!="token_partition_east_first_planar" || kind(outputHome)!="token_partition_planar" ||
        outputProduct.getRhs()!=product.getRhs() || outputProduct.getLhs().getType().getDimSize(0)!=heads*32))
        return outputProduct.emitError("shared-weight product requires matching native head groups and the same D value");
    const auto mem=[&](mlir::DictionaryAttr p,int byte,int side,int64_t at,llvm::StringRef opcode,
                       int64_t addr,int stream,int64_t count,int64_t stride=1,int64_t interval=1,int bank=-1) {
        direct_domain_detail::emitMem3D(r,loc,target,at,side*target.memory().slices_per_hemisphere+id(p,byte),opcode,
            field(p,"base_row")+addr,stream,count,interval,stride,1,1,0,1,1,0,-1,bank<0?field(p,"bank"):bank);
    };
    llvm::SmallVector<int64_t> map;for(int i=0;i<32;++i)map.push_back(i);
    const auto sxmToVxm=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
    const auto vxmToMxm=*target.transport_latency(target::StreamEndpoint::VxmResult,target::StreamEndpoint::Mem,target::StreamDirection::East,0)
        +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmActivation,target::StreamDirection::East,0);
    int64_t cursor=96,end=96;
    const auto emitOutput=[&](int64_t pair,int64_t vb,int64_t buffer,int64_t outputCompute) {
        if(outputProduct)for(int side=0;side<2;++side) {
            for(int b=0;b<2;++b) {
                const auto ar=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::MxmActivation,target::StreamDirection::East,id(outputLhs,b+2));
                mem(outputLhs,b+2,side,outputCompute-ar,"read",pair*32,4+b,32);
            }
            attention_detail::MxmDomain3D cm;cm.repeat_count=32;
            attention_detail::emitMxm3D(r,loc,outputCompute,side,"compute",buffer,0,4,24,0,1,"stream",true,"block",0,"bf16","direct16","fp32",cm);
            for(int b=0;b<4;++b) {
                const auto back=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,0)
                    +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0)
                    +*target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(outputHome,b));
                const auto at=outputCompute+target.mxm_first_result_latency()+back;
                mem(outputHome,b,1-side,at,"write",vb*half*32+pair*32,24+b,32);
                end=std::max(end,at+32);
            }
        }
    };
    // Batch two D tiles in the physical weight buffers. Then stream both K
    // contractions consecutively, amortizing the SXM direction-change drain.
    // O brackets the State products: O0, State0, State1, O1. The next D0 can
    // load into buffer 0 while O1 still consumes buffer 1; no MEM tile cache.
    const int64_t capacity=target.throughput().mxm_weight_buffers>=2 &&
        target.supports_mxm_weight_activation_overlap()?2:1;
    const int64_t jobs=half*(vd/32),sxmDrain=70;
    for(int64_t first=0;first<jobs;first+=capacity) {
        const int64_t batch=std::min(capacity,jobs-first);
        const int64_t firstK=cursor+(batch-1)*32+sxmDrain;
        for(int64_t lane=0;lane<batch;++lane) {
            const int64_t job=first+lane,pair=job/(vd/32),vb=job%(vd/32);
            const int64_t captureD=cursor+lane*32,load=captureD+36;
            for(int side=0;side<2;++side) {
                attention_detail::emitSxm(r,loc,captureD,side,"transpose",{16,17},{18,19},map,"wide_matrix",-1,-1,-1,1,32);
                for(int b=0;b<2;++b) {
                    const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(value,b+2));
                    mem(value,b+2,side,captureD-rd,"read",vb*half*32+pair*32,16+b,32);
                }
                attention_detail::MxmDomain3D iw;iw.repeat_count=32;iw.repeat_weight_column_stride=1;
                attention_detail::emitMxm3D(r,loc,load,side,"iw",lane,0,0,0,0,1,"stream",true,"vector",0,"bf16","direct16","",iw,18);
            }
        }
        for(int64_t lane=0;lane<batch;++lane) {
            const int64_t job=first+lane,pair=job/(vd/32),vb=job%(vd/32);
            const int64_t captureK=firstK+lane*kd,start=captureK+35+sxmToVxm;
            const int64_t compute=start+2+vxmToMxm;
            if(lane==0)emitOutput(pair,vb,lane,compute-32);
            for(int side=0;side<2;++side) {
                attention_detail::emitSxm(r,loc,captureK,side,"transpose",{16,17},{32+side*16,33+side*16},map,"wide_matrix",-1,-1,-1,kd/32,32);
                const int64_t h=pair*2+1-side;
                for(int64_t kb=0;kb<kd/32;++kb)for(int p=0;p<(distributed?8:1);++p)for(int b=0;b<2;++b) {
                    const int byte=distributed?p*2+b:b;
                    const auto rd=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,id(key,byte));
                    const auto striped=key.getAs<mlir::IntegerAttr>("bank_tile_rows");
                    mem(key,byte,side,captureK+kb*32+(distributed?p:0)-rd,"read",
                        (h*(kd/32)+kb)*(distributed?4:32),16+b,distributed?4:32,1,distributed?8:1,
                        striped && striped.getInt()?field(key,"bank")^(kb%2):-1);
                }
                for(int b=0;b<2;++b)
                    mem(weight,b,side,start-1-id(weight,b)/4,"read",31*heads+h,32+side*16+2+b,kd,0);
                // K's SXM vector spans the 32 contraction tokens. R's last row
                // therefore multiplies lane-wise; no scalar broadcast or spill.
                attention_detail::MxmDomain3D cm;cm.repeat_count=32;cm.wave_count=kd/32;cm.wave_interval=32;
                attention_detail::emitMxm3D(r,loc,compute,1-side,"compute",lane,0,side*8,24,0,1,"stream",true,"block",0,"bf16","direct16","fp32",cm);
                for(int b=0;b<4;++b) {
                    // Passive central return keeps the final West-first head home.
                    const auto back=*target.transport_latency(target::StreamEndpoint::MxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,0)
                        +*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::VxmInput,target::StreamDirection::West,0)
                        +*target.transport_latency(target::StreamEndpoint::VxmBridgeResult,target::StreamEndpoint::Mem,target::StreamDirection::East,id(out,b));
                    const auto at=compute+target.mxm_first_result_latency()+back;
                    mem(out,b,side,at,"write",vb*half*kd+pair*kd,24+b,kd);
                    end=std::max(end,at+kd);
                }
            }
            auto v=ffn_detail::create_vxm(r,loc,scale.getLhs(),scale.getRhs(),scale.getResult().getType(),start-1,0,"multiply",
                "stream_bf16",0,0.f,"stream_bf16",2,0.f,"fp32",-1,kd,1,"east","west",-1,false,false,true,false,2);
            v->setAttr("lhs_stream_source",r.getStringAttr("local"));v->setAttr("rhs_stream_source",r.getStringAttr("local"));
            ffn_detail::create_vxm(r,loc,scale.getLhs(),scale.getRhs(),scale.getResult().getType(),start-1,1,"bypass",
                "previous",0,0.f,"immediate",0,0.f,"bf16",0,kd,1,"east","west",-1,false,false,true,false,2);
            if(lane!=0)emitOutput(pair,vb,lane,compute+kd);
        }
        // Only SXM direction ownership gates the next capture. Let prior
        // VXM/MXM work and final MEM writes drain concurrently.
        cursor=firstK+batch*kd-32+sxmDrain;
    }
    cursor=std::max(cursor,end+32);
    mlir::OperationState tl(loc,TimelineOp::getOperationName());tl.addAttributes({r.getNamedAttr("name",r.getStringAttr(outputProduct?"head_products.shared_weight_tile":"head_weighted_product.streaming")),
        r.getNamedAttr("start",r.getI64IntegerAttr(64)),r.getNamedAttr("end",r.getI64IntegerAttr(cursor))});r.create(tl);
    const auto bindOutput=[&](kernel::VectorAluOp op,mlir::DictionaryAttr home) {
        int64_t index=-1;
        op->getParentOfType<mlir::func::FuncOp>().walk([&](mlir::func::ReturnOp ret) {
            for(auto [i,v]:llvm::enumerate(ret.getOperands())) if(v==op.getResult())index=i;
        });
        mlir::OperationState bind(loc,BindingOp::getOperationName());bind.addTypes(op.getResult().getType());
        bind.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(index)),r.getNamedAttr("access",r.getStringAttr("output")),
            r.getNamedAttr("role",r.getStringAttr("result")),r.getNamedAttr("bytes",r.getI64IntegerAttr(op.getResult().getType().getNumElements()*4)),r.getNamedAttr("placement",home)});
        r.replaceOp(op,llvm::cast<BindingOp>(r.create(bind)).getValue());
    };
    if(outputProduct)bindOutput(outputProduct,outputHome);
    bindOutput(product,out);r.eraseOp(scale);return mlir::success();
}
