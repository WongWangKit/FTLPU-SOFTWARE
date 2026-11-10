// Included inside Planner. Semantic tiling precedes physical allocation and
// raw-macro scheduling; no host-computed scores/probabilities are consumed.
Tensor flashUnary(const std::string &name, const Tensor &a,
                  const std::string &kind, float immediate,
                  bool reduce=false, bool maximum=false, const std::string &dtype="f32") {
  Tensor out; out.shape=reduce?V{a.shape[0],1}:a.shape; out.dtype=dtype;
  const auto ty=[&](const Tensor &t) {
    std::string s; llvm::raw_string_ostream os(s); type(&context,t).print(os); return s;
  };
  std::string imm; llvm::raw_string_ostream stream(imm);
  mlir::FloatAttr::get(mlir::Float32Type::get(&context),immediate).print(stream);
  const auto at=ty(a),ot=ty(out);
  const auto source="module { func.func @stage(%a: "+at+") -> "+ot+
      " attributes {ftlpu.vector_bf16_spill = false, ftlpu.vector_scalar_layout = true} { %r = ftlpu.kernel.vector_alu %a, %a {kind = \""+
      kind+"\", use_immediate = true, immediate = "+imm+", reduce = "+(reduce?"true":"false")+
      (maximum?", reduction_kind = \"max\", reduction_seed_first = true":"")+
      "} : ("+at+", "+at+") -> "+ot+" return %r : "+ot+" } }";
  return add(name,source,{a},{out})[0];
}

Tensor flashBinary(const std::string &name, const Tensor &a, const Tensor &b,
                   const std::string &kind, const std::string &dtype="f32",
                   bool packedUpdate=false) {
  Tensor out;out.shape=a.shape;out.dtype=dtype;
  const auto ty=[&](const Tensor &t) {std::string s;llvm::raw_string_ostream os(s);type(&context,t).print(os);return s;};
  const auto at=ty(a),bt=ty(b),ot=ty(out);
  const std::string layout=packedUpdate?"ftlpu.vector_packed_update":"ftlpu.vector_scalar_layout";
  const auto source="module { func.func @stage(%a: "+at+", %b: "+bt+") -> "+ot+
      " attributes {ftlpu.vector_bf16_spill = false, "+layout+" = true} { %r = ftlpu.kernel.vector_alu %a, %b {kind = \""+
      kind+"\", use_immediate = false, immediate = 0.0 : f32, reduce = false} : ("+at+", "+bt+") -> "+ot+
      " return %r : "+ot+" } }";
  return add(name,source,{a,b},{out})[0];
}

// Time-major words carry independent query rows in lanes. An axis-0
// reduction consumes successive keys in order without exchanging lanes.
Tensor flashTimeOp(const std::string &name, const Tensor &x, const std::string &kind,
                   const std::string &dtype, const Tensor *rhs=nullptr,
                   float imm=0.f, const char *reduction=nullptr) {
  Tensor out;out.shape=x.shape;out.dtype=dtype;
  if(reduction)out.shape[0]=1;
  const auto ty=[&](const Tensor &t){std::string s;llvm::raw_string_ostream os(s);type(&context,t).print(os);return s;};
  std::string literal;llvm::raw_string_ostream os(literal);
  mlir::FloatAttr::get(mlir::Float32Type::get(&context),imm).print(os);
  const auto xt=ty(x),rt=rhs?ty(*rhs):xt,ot=ty(out);
  const std::string reduceAttrs=!reduction?"":std::string(", reduction_axis = 0 : i64, reduction_kind = \"")+reduction+
      (llvm::StringRef(reduction)=="max"?"\", reduction_seed_first = true":"\"");
  const auto source="module { func.func @stage(%x: "+xt+(rhs?", %y: "+rt:"")+") -> "+ot+
      " attributes {ftlpu.vector_bf16_spill = "+(reduction?"false":"true")+
      (rhs?", ftlpu.vector_independent_ports = true":"")+"} { %r = ftlpu.kernel.vector_alu %x, "+
      (rhs?"%y":"%x")+" {kind = \""+kind+"\", use_immediate = "+(rhs?"false":"true")+
      ", immediate = "+literal+", reduce = "+(reduction?"true":"false")+reduceAttrs+
      "} : ("+xt+", "+rt+") -> "+ot+" return %r : "+ot+" } }";
  return add(name,source,rhs?std::vector<Tensor>{x,*rhs}:std::vector<Tensor>{x},{out})[0];
}

// Both operations retain FP32 results. The public vector chain selects
// compatible ALUs, removing a MEM round trip without moving BF16 boundaries.
Tensor flashBinaryThenUnary(const std::string &name, const Tensor &x, const Tensor &y,
    const std::string &binary, const std::string &unary) {
  Tensor out;out.shape=x.shape;out.dtype="f32";
  const auto ty=[&](const Tensor &t){std::string s;llvm::raw_string_ostream os(s);type(&context,t).print(os);return s;};
  const auto xt=ty(x),yt=ty(y),ot=ty(out);
  const auto source="module { func.func @stage(%x: "+xt+", %y: "+yt+") -> "+ot+
      " attributes {ftlpu.vector_bf16_spill = true, ftlpu.vector_independent_ports = true} { "
      "%shift = ftlpu.kernel.vector_alu %x, %y {kind = \""+binary+"\", use_immediate = false, "
      "immediate = 0.0 : f32, reduce = false} : ("+xt+", "+yt+") -> "+ot+
      " %r = ftlpu.kernel.vector_alu %shift, %shift {kind = \""+unary+"\", use_immediate = true, "
      "immediate = 0.0 : f32, reduce = false} : ("+ot+", "+ot+") -> "+ot+
      " return %r : "+ot+" } }";
  return add(name,source,{x,y},{out})[0];
}

Tensor flashExpDifference(const std::string &name, const Tensor &x, const Tensor &y) {
  return flashBinaryThenUnary(name,x,y,"subtract","exp");
}

// A value-channel word shares one query's coefficient across all its lanes.
Tensor flashRowCoefficients(const std::string &name, Tensor x, I live, I tile,
                            const std::string &dtype="f32") {
  x.shape={2*tile,1};
  auto scalar=flashUnary(name,x,"multiply",1.f,false,false,dtype);
  V ids; for(I s=0;s<2;++s)for(I r=0;r<live;++r)ids.push_back(s*tile+r);
  return view(scalar,ids,{2*live,1});
}

Tensor lowerFlashAttention(const std::string &name, const Tensor &q,
    const Tensor &k, const Tensor &v, const Tensor &out,
    float scale, bool causal, I position) {
  const I batch=q.shape[0], qt=q.shape[1], qh=q.shape[2], kd=q.shape[3];
  const I kt=k.shape[1], kh=k.shape[2], vd=v.shape[3], group=qh/kh;
  const I tile=target.throughput().mxm_rows;
  if(tile!=32) throw std::runtime_error("flash attention currently requires a 32-lane MXM target");
  const I groups=batch*kh, queryRows=qt*group;
  std::vector<std::vector<Tensor>> grouped(groups);
  for(I pair=0;pair<groups;pair+=2) {
    for(I qr=0;qr<queryRows;qr+=tile) {
      const I live=std::min<I>(tile,queryRows-qr);
      V qi;
      for(I side=0;side<2;++side) {
        const I g=std::min<I>(pair+side,groups-1), b=g/kh, h=g%kh;
        for(I r=0;r<tile;++r) {
          const I x=qr+std::min(r,live-1);
          for(I d=0;d<kd;++d) qi.push_back(((b*qt+x/group)*qh+h*group+x%group)*kd+d);
        }
      }
      const auto query=view(q,qi,{2,tile,kd});
      // The same prepared Q serves every KV tile in this query block.
      // Keep its lifetime explicit so the memory planner protects the home.
      ActivationRhsTiles sharedQuery;
      Tensor maximum, denominator, numerator, inverse;
      for(I first=0;first<kt;first+=tile) {
        if(causal && first>position+(qr+live-1)/group) break;
        const auto prefix=name+".group"+std::to_string(pair)+".q"+
            std::to_string(qr)+".kv"+std::to_string(first);
        const I keys=std::min(tile,kt-first);
        V ki,vi; std::vector<float> mask;
        for(I side=0;side<2;++side) {
          const I g=std::min<I>(pair+side,groups-1), b=g/kh, h=g%kh;
          for(I r=0;r<tile;++r) {
            const I token=std::min<I>(first+r,kt-1);
            for(I d=0;d<kd;++d) ki.push_back(((b*kt+token)*kh+h)*kd+d);
            for(I d=0;d<vd;++d) vi.push_back(((b*kt+token)*kh+h)*vd+d);
          }
        }
        // K Q^T has the same dot products and reduction order as Q K^T.
        // Each packed word is already one key across 32 query rows.
        Tensor scoreType;scoreType.shape={2,tile,tile};scoreType.dtype="bf16";
        auto dots=lowerActivationBatchMatmul(prefix+".kqt",view(k,ki,{2,tile,kd}),
            query,scoreType,true,nullptr,&sharedQuery,true);
        V timeOrder;
        for(I t=0;t<keys;++t)for(I side=0;side<2;++side)for(I r=0;r<tile;++r) {
          timeOrder.push_back((side*tile+t)*tile+r);
          const I queryToken=(qr+std::min(r,live-1))/group;
          mask.push_back(!causal || first+t<=position+queryToken?0.f:-1.e30f);
        }
        auto score=flashTimeOp(prefix+".scale",view(dots,timeOrder,{keys,2*tile}),"multiply","bf16",nullptr,scale);
        if(llvm::any_of(mask,[](float x){return x!=0.f;})) {
          const auto m=deltaConstant(score.shape,mask);
          score=flashTimeOp(prefix+".mask",score,"add","bf16",&m);
        }
        auto blockMax=flashTimeOp(prefix+".row_max",score,"multiply","f32",nullptr,1.f,"max");
        auto nextMax=first?flashTimeOp(prefix+".max_update",maximum,"max","f32",&blockMax):blockMax;
        auto probability=flashExpDifference(prefix+".shift_exp",score,nextMax);
        // A one-key tile already is its FP32 column sum; reuse the producer.
        auto blockSum=keys==1?probability:
            flashTimeOp(prefix+".row_sum",probability,"multiply","f32",nullptr,1.f,"add");
        Tensor product;
        if (keys == 1) {
          // A single key contributes the outer product P[:,0] * V[0,:].
          // Round P once here, at the same BF16 boundary used by the MXM.
          auto coefficients=flashRowCoefficients(prefix+".pv.probability_rows",probability,live,tile,"bf16");
          V valueRows;
          for (I side=0; side<2; ++side) {
            const auto row=vi.begin()+side*tile*vd;
            for (I r=0; r<live; ++r)
              valueRows.insert(valueRows.end(),row,row+vd);
          }
          product=flashBinary(prefix+".pv.outer",view(v,valueRows,{2*live,vd}),coefficients,
                              "multiply","f32",true);
        } else {
          auto packed=flashTimeOp(prefix+".probability_bf16",probability,"multiply","bf16",nullptr,1.f);
          if(keys<tile) {
            // Only PV needs padding; do not feed padded keys into m or l.
            auto zero=flashTimeOp(prefix+".zero_probability",sliceRows(packed,0,1),"multiply","bf16",nullptr,0.f);
            V indices;for(I t=keys;t<tile;++t)for(I c=0;c<2*tile;++c)indices.push_back(c);
            packed=concat({packed,view(zero,indices,{tile-keys,2*tile})});
          }
          V pt; for(I s=0;s<2;++s)for(I key=0;key<tile;++key)for(I row=0;row<tile;++row)
            pt.push_back(key*2*tile+s*tile+row);
          Tensor productType;productType.shape={2,live,vd};productType.dtype="f32";
          product=lowerTimeMajorProbabilityValue(prefix+".pv",view(packed,pt,{2,tile,tile}),
              view(v,vi,{2,tile,vd}),live,productType);
          product.shape={2*live,vd};
        }
        if(first) {
          auto alpha=flashExpDifference(prefix+".rescale",maximum,nextMax);
          denominator=flashTimeOp(prefix+".denominator_scale",denominator,"multiply","f32",&alpha);
          const bool last=first+tile>=kt || (causal && first+tile>position+(qr+live-1)/group);
          if(last)
            inverse=flashBinaryThenUnary(prefix+".denominator_normalizer",denominator,blockSum,"add","reciprocal");
          else
            denominator=flashTimeOp(prefix+".denominator_add",denominator,"add","f32",&blockSum);
          // Four FP32 byte planes retain independent value channels in the
          // lanes. Only alpha is broadcast across a 32-channel word.
          numerator=flashBinary(prefix+".numerator_scale",numerator,flashRowCoefficients(prefix+".alpha_rows",alpha,live,tile),"multiply","f32",true);
          numerator=flashBinary(prefix+".numerator_add",numerator,product,"add","f32",true);
        } else { denominator=blockSum; numerator=product; }
        maximum=nextMax;
        // FP32 m/l/U are materialized; no live FU context crosses a reload.
        nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
      }
      const auto prefix=name+".group"+std::to_string(pair)+".q"+std::to_string(qr);
      if(inverse.shape.empty())inverse=flashTimeOp(prefix+".inverse",denominator,"reciprocal","f32");
      // Normalize in the same value-lane layout and write final BF16 words
      // directly. This also applies when a single KV tile supplied U.
      auto result=flashBinary(prefix+".normalize",numerator,flashRowCoefficients(prefix+".inverse_rows",inverse,live,tile),"multiply","bf16",true);
      nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
      for(I side=0;side<2 && pair+side<groups;++side)
        grouped[pair+side].push_back(sliceRows(result,side*live,live));
    }
  }
  std::vector<Tensor> matrices;for(auto &parts:grouped)matrices.push_back(concat(parts));
  auto groupedOutput=concat(matrices); V order;
  for(I b=0;b<batch;++b)for(I t=0;t<qt;++t)for(I h=0;h<qh;++h)for(I d=0;d<vd;++d)
    order.push_back(((b*kh+h/group)*queryRows+t*group+h%group)*vd+d);
  return view(groupedOutput,order,out.shape);
}
