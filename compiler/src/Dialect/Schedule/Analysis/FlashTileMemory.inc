// One complete 32-key Flash tile family. All words are ordinary planar;
// the layout records mathematical ownership, not a distributed16 swizzle.
mlir::FailureOr<mlir::DictionaryAttr> planFlashTileMemory(
    mlir::OpBuilder& b, mlir::DictionaryAttr config,
    const target::LPUTargetModel& target)
{
    const auto integer=[&](llvm::StringRef name) {return config.getAs<mlir::IntegerAttr>(name).getInt();};
    const int64_t q=integer("seq_len"), h=integer("query_heads"), kh=integer("kv_heads"), d=integer("head_dim");
    const int64_t rows=q*(h/kh), pairs=kh/2, tiles=rows/32;
    if(target.throughput().mxm_rows!=32 || target.throughput().mxms_per_hemisphere!=1
        || target.memory().hemispheres!=2 || target.memory().banks_per_slice<2
        || target.memory().slices_per_hemisphere<24 || kh%2 || rows%32
        || integer("kv_seq_len")!=32 || integer("actual_kv_len")!=32 || d%32
        || target.throughput().mxm_weight_buffers<2
        || !target.supports_mxm_weight_activation_overlap())return mlir::failure();
    mlir::NamedAttrList plan;
    const auto add=[&](llvm::StringRef name,int64_t first,int64_t planes,int64_t bank,int64_t count,llvm::StringRef role={}) {
        if(count>target.memory().sram_depth_rows)throw std::runtime_error("Flash tile planar allocation exceeds SRAM depth");
        llvm::SmallVector<int64_t> slices;
        for(int64_t i=0;i<planes;++i)slices.push_back(first+i);
        mlir::NamedAttrList p({
            b.getNamedAttr("kind",b.getStringAttr(planes==4?"fp32_quad_planar":"fp16_pair_planar")),
            b.getNamedAttr("hemisphere",b.getStringAttr("both")),
            b.getNamedAttr("slices",b.getI64ArrayAttr(slices)),
            b.getNamedAttr("bank",b.getI64IntegerAttr(bank)),
            b.getNamedAttr("base_row",b.getI64IntegerAttr(0)),
            b.getNamedAttr("instruction_count",b.getI64IntegerAttr(count)),
            b.getNamedAttr("address_stride",b.getI64IntegerAttr(1))});
        if(!role.empty()) {
            p.set("dense_attention_role",b.getStringAttr(role));
            p.set("heads",b.getI64IntegerAttr(role=="flash_query" || role=="flash_context"?h:kh));
            p.set("head_dim",b.getI64IntegerAttr(d));
            p.set("token_stride",b.getI64IntegerAttr(role=="flash_query" || role=="flash_context"?q:32));
            p.set("query_heads_per_kv",b.getI64IntegerAttr(h/kh));
            p.set("flash_query_rows",b.getI64IntegerAttr(rows));
            if(auto headMajor=config.getAs<mlir::BoolAttr>("query_head_major"))
                p.set("query_head_major",headMajor);
        }
        if(name=="causal_mask")p.set("kind",b.getStringAttr("fp16_causal_mask_tile"));
        plan.set(name,p.getDictionary(b.getContext()));
    };
    add("query",0,2,0,pairs*rows*(d/32),"flash_query");
    add("key",4,2,0,pairs*d,"flash_key");
    // V^T P keeps the query axis in lanes, so the denominator can feed the
    // PV epilogue directly. C2C ingress places V at this final planar home.
    add("value",8,2,0,pairs*d,"flash_value");
    // Bounded live tiles: the following QK writes the other ICU pair while
    // softmax reads the current one. No full QK or P matrix is materialized.
    add("score",0,2,1,32);
    add("score_alt",2,2,1,32);
    add("scaled",4,2,1,32);
    add("causal_mask",10,2,0,31);
    add("exp",12,4,0,32);
    add("inverse",16,4,0,1);
    add("context",20,2,1,pairs*tiles*d,"flash_context");
    if(auto sink=config.getAs<mlir::BoolAttr>("has_sink_tensor");sink && sink.getValue())
        add("sink",24,4,0,2*pairs*tiles);
    return plan.getDictionary(b.getContext());
}
