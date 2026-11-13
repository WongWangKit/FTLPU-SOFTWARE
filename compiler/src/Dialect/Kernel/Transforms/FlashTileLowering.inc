// Select by shape and target, never by model/test name. Multi-KV-tile online
// attention retains the existing general path until it has its own tile DAG.
bool nativeFlashTileEligible(const Tensor& q,const Tensor& k,const Tensor& v,I position) {
    const auto wide=policies.getAs<mlir::BoolAttr>("ftlpu.sxm_wide");
    return wide && wide.getValue() && position==0 && q.shape[1]==32 && k.shape[1]==32
        && k.shape==v.shape && q.shape[3]==k.shape[3]
        && q.dtype=="bf16" && k.dtype=="bf16" && v.dtype=="bf16"
        && k.shape[2]%2==0 && q.shape[2]%k.shape[2]==0
        && (q.shape[3]==32 || q.shape[3]==64 || q.shape[3]==128 || q.shape[3]==512)
        && q.shape[0]*q.shape[2]*q.shape[3]/2 <= target.memory().sram_depth_rows
        && target.throughput().mxms_per_hemisphere==1
        && target.throughput().mxm_weight_buffers>=2
        && target.supports_mxm_weight_activation_overlap();
}
Tensor lowerNativeFlashTile(const std::string& name,const Tensor& q,const Tensor& k,
    const Tensor& v,const Tensor& output,float scale,bool causal,I position,
    mlir::DenseF32ArrayAttr sinks={}) {
    const auto ty=[&](const Tensor& x) {std::string t;llvm::raw_string_ostream os(t);type(&context,x).print(os);return t;};
    std::string literal;llvm::raw_string_ostream stream(literal);
    mlir::FloatAttr::get(mlir::Float32Type::get(&context),scale).print(stream);
    const I batch=q.shape[0],qh=q.shape[2],kvh=k.shape[2],d=q.shape[3];
    const I h=batch*qh,kh=batch*kvh;
    const auto config="{seq_len = 32 : i64, kv_seq_len = 32 : i64, actual_kv_len = 32 : i64, current_len = 32 : i64, "
        "position_offset = 0 : i64, hidden = "+std::to_string(h*d)+" : i64, query_heads = "+std::to_string(h)+
        " : i64, kv_heads = "+std::to_string(kh)+" : i64, head_dim = "+std::to_string(d)+
        " : i64, causal = "+(causal?"true":"false")+", scale = "+literal+", attention_strategy = \"flash_tile\", query_head_major = "+(d>128?"true":"false")+", has_sink_tensor = "+(sinks?"true":"false")+"}";
    // Requests become independent head groups of one native scheduling unit.
    // This is only an index view: no request can see another request's KV.
    const auto pack=[&](const Tensor& x) {
        V indices;const I columns=x.shape[2]*d;
        for(I t=0;t<32;++t)for(I b=0;b<batch;++b) {
            auto row=range(columns,(b*32+t)*columns);
            indices.insert(indices.end(),row.begin(),row.end());
        }
        return view(x,indices,{32,batch*columns});
    };
    auto a=pack(q),key=pack(k),value=pack(v);
    Tensor sink;std::vector<Tensor> inputs{a,key,value};
    if(sinks) {
      const I group=qh/kvh,tiles=group;
      std::vector<float> values;
      for(I pair=0;pair<kh/2;++pair)for(I tile=0;tile<tiles;++tile)
        for(I side=0;side<2;++side)for(I lane=0;lane<32;++lane) {
          const I head=((2*pair+side)%kvh)*group+(d>128?tile:(tile*32+lane)%group);
          values.push_back(sinks.asArrayRef()[head]);
        }
      sink=deltaConstant({kh*tiles,32},values);inputs.push_back(sink);
    }
    Tensor result;result.shape=a.shape;result.dtype="bf16";
    const auto source="module { func.func @stage(%q: "+ty(a)+", %k: "+ty(key)+", %v: "+ty(value)+(sinks?", %s: "+ty(sink):"")+") -> "+ty(result)+
        " { %r = ftlpu.kernel.attention_core %q, %k, %v "+(sinks?"sink %s ":"")+"{config = "+config+"} : ("+ty(a)+", "+ty(key)+", "+ty(value)+(sinks?", "+ty(sink):"")+") -> "+ty(result)+
        " return %r : "+ty(result)+" } }";
    result=add(name+".tiles",source,inputs,{result})[0];
    nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
    V order;
    for(I b=0;b<batch;++b)for(I t=0;t<32;++t) {
        auto row=range(qh*d,(t*batch+b)*qh*d);order.insert(order.end(),row.begin(),row.end());
    }
    return view(result,order,output.shape);
}
