// Included in Planner. Preserve the native dense attention subgraph as one
// scheduling unit per independent request; batching never aliases KV state.
Tensor lowerDenseAttention(const std::string &name, const Tensor &query,
    const Tensor &key, const Tensor &value, const Tensor &output,
    float scale, bool causal, I position,
    mlir::DenseF32ArrayAttr sinks = {}, I window = 0) {
  const I batches=query.shape[0], tokens=query.shape[1], heads=query.shape[2];
  const I width=query.shape[3], kvTokens=key.shape[1], kvHeads=key.shape[2];
  if(std::getenv("FTLPU_GRAPH_PROGRESS")) llvm::errs()<<"native dense attention: D="<<width<<" heads="<<heads<<" sinks="<<(sinks?sinks.size():0)<<"\n";
  const I tile=target.throughput().mxm_rows;
  if(tile!=32 || (tokens!=1 && tokens%tile) || position%tile ||
      (width!=32 && width!=64 && width!=128 && width!=512))
    throw std::runtime_error("native dense attention supports Tq=1 or 32-aligned Tq, aligned positions, head width 32/64/128/512");
  if(window && position+tokens>window)
    throw std::runtime_error("native window attention currently requires the query prefix to fit its window");
  // Bound the wide-head scheduling unit by the native planar Q/context homes.
  // Keep multiple Q heads per hemisphere so a shared KV tile stays resident
  // through the core's QK/softmax/PV lookahead instead of restarting per pair.
  const auto wide=policies.getAs<mlir::BoolAttr>("ftlpu.sxm_wide");
  const bool tileGeometry=wide && wide.getValue() && tokens==32 && kvTokens==32 && position==0;
  const I wideHeadCapacity=tileGeometry
      ?std::max<I>(2,2*(target.memory().sram_depth_rows/(batches*width))):2;
  if(width>128 && (heads>wideHeadCapacity || (kvHeads>1 && kvHeads%2))) {
    std::vector<Tensor> parts; V order;
    const I perKv=heads/kvHeads;
    for(I h=0;h<heads;) {
      const I count=std::min<I>(wideHeadCapacity,perKv-h%perKv), kh=h/perKv;
      V qi,ki;
      for(I b=0;b<batches;++b)for(I t=0;t<tokens;++t)for(I j=0;j<count*width;++j)
        qi.push_back(((b*tokens+t)*heads+h)*width+j);
      for(I b=0;b<batches;++b)for(I t=0;t<kvTokens;++t)for(I d=0;d<width;++d)
        ki.push_back(((b*kvTokens+t)*kvHeads+kh)*width+d);
      auto q=view(query,qi,{batches,tokens,count,width});
      auto k=view(key,ki,{batches,kvTokens,1,width});
      auto v=view(value,ki,k.shape); auto o=output;o.shape=q.shape;
      const auto subSinks=sinks?mlir::DenseF32ArrayAttr::get(&context,sinks.asArrayRef().slice(h,count)):mlir::DenseF32ArrayAttr{};
      parts.push_back(lowerDenseAttention(name+".heads"+std::to_string(h),q,k,v,o,scale,causal,position,subSinks,window));
      parts.back().shape={batches*tokens,count*width}; h+=count;
    }
    auto result=concat(parts,1);result.shape=output.shape;return result;
  }
  const auto ty=[&](const Tensor &t) {std::string str; llvm::raw_string_ostream os(str); type(&context,t).print(os); return str;};
  // The shared planar tile core can split a single KV group across two
  // hemispheres by assigning a distinct query head to each. The KV views
  // alias the same tensor; neither queries nor arithmetic are duplicated.
  if(width>128 && heads%2==0 && kvHeads==1 && tokens==32 && kvTokens==32 && position==0) {
    V indices;
    for(I b=0;b<batches;++b)for(I t=0;t<kvTokens;++t)for(I side=0;side<2;++side)
      for(I d=0;d<width;++d)indices.push_back((b*kvTokens+t)*width+d);
    auto k=view(key,indices,{batches,kvTokens,2,width});
    auto v=view(value,indices,k.shape);
    if(nativeFlashTileEligible(query,k,v,position))
      return lowerNativeFlashTile(name,query,k,v,output,scale,causal,position,sinks);
  }
  if(width>128 && nativeFlashTileEligible(query,key,value,position))
    return lowerNativeFlashTile(name,query,key,value,output,scale,causal,position,sinks);
  std::string literal;llvm::raw_string_ostream os(literal);
  mlir::FloatAttr::get(mlir::Float32Type::get(&context),scale).print(os);
  const I paddedKv=(kvTokens+tile-1)/tile*tile;
  std::string sinkConfig;
  if(sinks) {
    std::ostringstream stream;stream<<", attention_sinks = array<f32: "<<std::scientific<<std::setprecision(9);
    for(I i=0;i<sinks.size();++i) {if(i)stream<<", ";stream<<sinks.asArrayRef()[i];}
    stream<<">";sinkConfig=stream.str();
  }
  const auto config="{seq_len = "+std::to_string(tokens)+" : i64, kv_seq_len = "+std::to_string(paddedKv)+
      " : i64, actual_kv_len = "+std::to_string(kvTokens)+" : i64, current_len = "+std::to_string(tokens)+
      " : i64, position_offset = "+std::to_string(position)+" : i64, hidden = "+std::to_string(heads*width)+
      " : i64, query_heads = "+std::to_string(heads)+" : i64, kv_heads = "+std::to_string(kvHeads)+
      " : i64, head_dim = "+std::to_string(width)+" : i64, kv_cache_resident_tokens = "+std::to_string(paddedKv)+
      " : i64, causal = "+(causal?"true":"false")+", scale = "+literal+sinkConfig+"}";
  if(std::getenv("FTLPU_GRAPH_PROGRESS")) llvm::errs()<<"native dense config: "<<config<<"\n";
  std::vector<Tensor> results;
  const auto padKv=[&](Tensor tensor, const std::string &label) {
    if(kvTokens==paddedKv)return tensor;
    if(tensor.external()) {
      tensor.shape[0]=paddedKv;
      tensor.data.resize(tensor.size()*tensor.bytes(),0);
      tensor.sourceContiguous=false;
      return tensor;
    }
    Tensor zero;zero.shape={paddedKv-kvTokens,kvHeads*width};
    zero.dtype="bf16";zero.constant=deltaConstantId++;
    zero.data.resize(zero.size()*zero.bytes(),0);
    return concat({tensor,join(zero,label+".padding")});
  };
  for(I batch=0;batch<batches;++batch) {
    auto q=view(query,range(tokens*heads*width,batch*tokens*heads*width),{tokens,heads*width});
    auto k=view(key,range(kvTokens*kvHeads*width,batch*kvTokens*kvHeads*width),{kvTokens,kvHeads*width});
    auto v=view(value,range(kvTokens*kvHeads*width,batch*kvTokens*kvHeads*width),{kvTokens,kvHeads*width});
    k=padKv(std::move(k),name+".request"+std::to_string(batch)+".key");
    v=padKv(std::move(v),name+".request"+std::to_string(batch)+".value");
    Tensor out;out.shape=q.shape;out.dtype=output.dtype;
    const auto qt=ty(q),kt=ty(k),vt=ty(v),ot=ty(out);
    const auto source="module { func.func @stage(%q: "+qt+", %k: "+kt+", %v: "+vt+") -> "+ot+
        " { %r = ftlpu.kernel.attention_core %q, %k, %v {config = "+config+"} : ("+qt+", "+kt+", "+vt+") -> "+ot+
        " return %r : "+ot+" } }";
    results.push_back(add(name+".request"+std::to_string(batch),source,{q,k,v},{out})[0]);
    // Each independent request has completely drained its native stages.
    // Use the existing physical iMEM segmentation protocol for large batches.
    nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
  }
  auto result=concat(results);result.shape=output.shape;return result;
}
