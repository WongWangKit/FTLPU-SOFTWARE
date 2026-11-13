// Reusable hyper-connection equations. A vector lane owns one token; the
// stream/head axes are addressed as vector rows, never cross-lane arithmetic.
std::map<V,Tensor> mhcInputBlocks;
Tensor mhcTokenBlock(const std::string &name, Tensor input, I token, I column, I width) {
  input.shape={input.shape[0],input.size()/input.shape[0]};
  const V key{input.constant,input.sourceOffset,input.shape[0],input.shape[1],token,column,width};
  const bool cache=input.external() && input.sourceContiguous;
  if(cache && mhcInputBlocks.count(key))return mhcInputBlocks.at(key);
  V order;
  for(I d=0;d<width;++d)for(I t=0;t<32;++t)order.push_back((token+t)*input.shape[1]+column+d);
  // This immutable block feeds both fixed VXM datapaths repeatedly. Give
  // C2C/its producer the final mirrored planar home once, rather than
  // copying a West-only staging image before every mixes/pre/post consumer.
  auto result=join(view(input,order,{width,32}),name,true);
  if(cache)mhcInputBlocks.emplace(key,result);
  return result;
}
Tensor mhcAlu(const std::string &name,Tensor a,Tensor b,const std::string &kind,
              bool immediate=false,float value=0.f,const std::string &dtype="f32",bool reduce=false,
              const std::vector<std::pair<std::string,float>> &tail={}) {
  // Keep a coefficient row compact: the native READ3D repeats its address.
  // Expanding [1,T32] here would materialize an entire hidden-width table.
  const bool rowBroadcast=b.shape==V{1,a.shape[1]};
  if(!immediate && (!rowBroadcast || (a.dtype=="f32" && b.dtype=="bf16")))
    b=deltaBroadcast(b,a.shape);
  // Narrow LHS + FP32 RHS has disjoint fixed ports. The reverse orientation
  // aliases the RHS bytes, so commute finite elementwise operations instead
  // of inserting a stored cast or relaxing the hardware port contract.
  if(!immediate && a.dtype=="f32" && b.dtype=="bf16" &&
     (kind=="add" || kind=="multiply" || kind=="max"))std::swap(a,b);
  const bool same=!a.external() && !b.external() && a.shape==b.shape && a.dtype==b.dtype &&
      a.refs.size()==b.refs.size() && std::equal(a.refs.begin(),a.refs.end(),b.refs.begin(),
          [](const Ref &x,const Ref &y){return x.node==y.node && x.output==y.output && x.index==y.index;});
  const bool single=immediate || same;
  Tensor result;result.dtype=dtype;result.shape=reduce?V{1,a.shape[1]}:a.shape;
  mlir::OpBuilder builder(&context);const auto loc=builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module=mlir::ModuleOp::create(loc);
  llvm::SmallVector<mlir::Type> args{type(&context,a)};
  if(!single)args.push_back(type(&context,b));
  auto fn=mlir::func::FuncOp::create(loc,"hc_arithmetic",builder.getFunctionType(args,{type(&context,result)}));
  fn->setAttr("ftlpu.vector_bf16_spill",builder.getBoolAttr(true));
  fn->setAttr("ftlpu.vector_independent_ports",builder.getBoolAttr(true));
  fn->setAttr("ftlpu.tight_serial_schedule",builder.getUnitAttr());
  module->push_back(fn);builder.setInsertionPointToStart(fn.addEntryBlock());
  mlir::OperationState op(loc,fc::kernel::VectorAluOp::getOperationName());
  auto intermediate=result;if(!tail.empty())intermediate.dtype="f32";
  op.addOperands({fn.getArgument(0),fn.getArgument(single?0:1)});op.addTypes(type(&context,intermediate));
  op.addAttribute("kind",builder.getStringAttr(kind));op.addAttribute("use_immediate",builder.getBoolAttr(immediate));
  op.addAttribute("immediate",builder.getF32FloatAttr(value));op.addAttribute("reduce",builder.getBoolAttr(reduce));
  op.addAttribute("ftlpu.vector_require_fp32",builder.getUnitAttr());
  if(reduce) {
    op.addAttribute("reduction_axis",builder.getI64IntegerAttr(0));
    op.addAttribute("reduction_kind",builder.getStringAttr("add"));
  }
  auto current=builder.create(op)->getResult(0);
  for(size_t i=0;i<tail.size();++i) {
    mlir::OperationState next(loc,fc::kernel::VectorAluOp::getOperationName());
    next.addOperands({current,current});
    next.addTypes(type(&context,i+1==tail.size()?result:intermediate));
    next.addAttribute("kind",builder.getStringAttr(tail[i].first));
    next.addAttribute("use_immediate",builder.getBoolAttr(true));
    next.addAttribute("immediate",builder.getF32FloatAttr(tail[i].second));
    next.addAttribute("reduce",builder.getBoolAttr(false));
    next.addAttribute("ftlpu.vector_require_fp32",builder.getUnitAttr());
    current=builder.create(next)->getResult(0);
  }
  builder.create<mlir::func::ReturnOp>(loc,current);
  std::string source;llvm::raw_string_ostream os(source);module->print(os);
  return add(name,source,single?std::vector<Tensor>{a}:std::vector<Tensor>{a,b},{result})[0];
}
Tensor mhcUnary(const std::string &name,const Tensor &a,const std::string &kind,float value=0.f,
                const std::string &dtype="f32") {
  return mhcAlu(name,a,a,kind,true,value,dtype);
}
Tensor mhcScaleAdd(const std::string &name,Tensor x,Tensor coefficient,Tensor partial,const std::string &dtype="f32",bool addThenScale=false) {
  if(!addThenScale)coefficient=deltaBroadcast(coefficient,{1,x.shape[1]});
  if(partial.shape!=x.shape && partial.shape!=V{1,x.shape[1]})partial=deltaBroadcast(partial,x.shape);
  auto product=x;product.dtype="f32";auto result=x;result.dtype=dtype;
  const auto str=[&](const Tensor &t){std::string text;llvm::raw_string_ostream os(text);type(&context,t).print(os);return text;};
  const auto xt=str(x),ct=str(coefficient),st=str(partial),pt=str(product),rt=str(result);
  const auto source="module { func.func @stage(%x: "+xt+", %c: "+ct+", %s: "+st+") -> "+rt+
      " attributes {ftlpu.vector_bf16_spill=true, ftlpu.vector_independent_ports=true, ftlpu.tight_serial_schedule} { "
      "%p = ftlpu.kernel.vector_alu %x, %c {kind=\""+(addThenScale?"add":"multiply")+"\", use_immediate=false, immediate=0.0:f32, reduce=false, ftlpu.vector_require_fp32} : ("+xt+", "+ct+") -> "+pt+
      " %y = ftlpu.kernel.vector_alu %p, %s {kind=\""+(addThenScale?"multiply":"add")+"\", use_immediate=false, immediate=0.0:f32, reduce=false, ftlpu.vector_require_fp32} : ("+pt+", "+st+") -> "+rt+
      " return %y : "+rt+" } }";
  return add(name,source,{x,coefficient,partial},{result})[0];
}
Tensor mhcSquareSum(const std::string &name,const Tensor &x) {
  auto square=x;square.dtype="f32";auto result=square;result.shape={1,x.shape[1]};
  const auto str=[&](const Tensor &t){std::string text;llvm::raw_string_ostream os(text);type(&context,t).print(os);return text;};
  const auto xt=str(x),pt=str(square),rt=str(result);
  const auto source="module { func.func @stage(%x: "+xt+") -> "+rt+
      " attributes {ftlpu.vector_bf16_spill=true, ftlpu.vector_independent_ports=true, ftlpu.tight_serial_schedule} { "
      "%p = ftlpu.kernel.vector_alu %x, %x {kind=\"multiply\", use_immediate=false, immediate=0.0:f32, reduce=false, ftlpu.vector_require_fp32} : ("+xt+", "+xt+") -> "+pt+
      " %s = ftlpu.kernel.vector_alu %p, %p {kind=\"multiply\", use_immediate=true, immediate=1.0:f32, reduce=true, reduction_axis=0:i64, reduction_kind=\"add\", ftlpu.vector_require_fp32} : ("+pt+", "+pt+") -> "+rt+
      " return %s : "+rt+" } }";
  return add(name,source,{x},{result})[0];
}
Tensor mhcSum(const std::string &name,const std::vector<Tensor> &parts,const std::string &kind="add",
              const std::vector<std::pair<std::string,float>> &tail={}) {
  if(parts.size()>=2 && parts.size()<=4) {
    auto result=parts[0];result.dtype="f32";
    mlir::OpBuilder b(&context);const auto loc=b.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module=mlir::ModuleOp::create(loc);
    llvm::SmallVector<mlir::Type> types;for(auto &p:parts)types.push_back(type(&context,p));
    auto fn=mlir::func::FuncOp::create(loc,"pointwise_fold",b.getFunctionType(types,{type(&context,result)}));
    fn->setAttr("ftlpu.vector_bf16_spill",b.getBoolAttr(true));
    fn->setAttr("ftlpu.vector_independent_ports",b.getBoolAttr(true));
    fn->setAttr("ftlpu.tight_serial_schedule",b.getUnitAttr());
    module->push_back(fn);b.setInsertionPointToStart(fn.addEntryBlock());
    mlir::Value value=fn.getArgument(0);
    const auto emit=[&](llvm::StringRef k,mlir::Value rhs,bool immediate,float number) {
      mlir::OperationState op(loc,fc::kernel::VectorAluOp::getOperationName());
      op.addOperands({value,rhs});op.addTypes(type(&context,result));
      op.addAttribute("kind",b.getStringAttr(k));op.addAttribute("use_immediate",b.getBoolAttr(immediate));
      op.addAttribute("immediate",b.getF32FloatAttr(number));op.addAttribute("reduce",b.getBoolAttr(false));
      op.addAttribute("ftlpu.vector_require_fp32",b.getUnitAttr());value=b.create(op)->getResult(0);
    };
    for(size_t i=1;i<parts.size();++i)emit(kind,fn.getArgument(i),false,0.f);
    for(auto &entry:tail)emit(entry.first,value,true,entry.second);
    b.create<mlir::func::ReturnOp>(loc,value);
    std::string text;llvm::raw_string_ostream os(text);module->print(os);
    return add(name,text,parts,{result})[0];
  }
  auto sum=parts.at(0);
  for(I i=1;i<I(parts.size());++i)sum=mhcAlu(name+std::to_string(i),sum,parts[i],kind);
  for(size_t i=0;i<tail.size();++i)sum=mhcUnary(name+".tail"+std::to_string(i),sum,tail[i].first,tail[i].second);
  return sum;
}
void mhcDrain() { nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context)); }
Tensor mhcCoefficient(const Tensor &table,I token,I field,I fields) {
  V order;for(I t=0;t<32;++t)order.push_back((token+t)*fields+field);
  return view(table,order,{1,32});
}
Tensor mhcMatrixReduce(const std::string &name,const Tensor &matrix,I h,bool rows,
                       const std::string &kind="add",const std::vector<std::pair<std::string,float>> &tail={}) {
  std::vector<Tensor> parts;
  for(I k=0;k<h;++k) {
    V order;for(I i=0;i<h;++i)for(I t=0;t<32;++t)order.push_back(((rows?i*h+k:k*h+i)*32)+t);
    parts.push_back(view(matrix,order,{h,32}));
  }
  return mhcSum(name,parts,kind,tail);
}
Tensor mhcMatrixBroadcast(const Tensor &values,I h,bool rows) {
  V order;for(I i=0;i<h;++i)for(I j=0;j<h;++j)for(I t=0;t<32;++t)
    order.push_back((rows?i:j)*32+t);
  return view(values,order,{h*h,32});
}
Tensor mhcNormalize(const std::string &name,const Tensor &matrix,I h,bool rows,float epsilon,const std::string &dtype="f32") {
  auto inverse=mhcMatrixReduce(name+".inverse",matrix,h,rows,"add",{{"add",epsilon},{"reciprocal",0.f}});
  auto result=mhcAlu(name+".normalize",matrix,mhcMatrixBroadcast(inverse,h,rows),"multiply",false,0.f,dtype);
  mhcDrain();return result;
}
Tensor mhcProjection(const std::string &name,const Tensor &weight,
                     const std::vector<Tensor> &tokenColumns,const Tensor &inverse) {
  // W[F,K] * X^T[K,T32] produces coefficient rows with token lanes directly.
  // Reuse the native BF16 MXM core, including its streamed FP32 add epilogue.
  // When possible split K across the hemispheres instead of duplicating GEMM.
  const I fields=weight.shape[0],k=weight.shape[1];
  const I partitions=k%64==0?2:1,localK=k/partitions;
  V order;
  for(I side=0;side<partitions;++side)for(I field=0;field<fields;++field)
    for(I column=0;column<localK;++column)
      order.push_back(field*k+side*localK+column);
  auto w=view(weight,order,{partitions,fields,localK});
  auto x=concat(tokenColumns);x.shape={partitions,localK,32};
  Tensor partial;partial.dtype="f32";partial.shape={partitions,fields,32};
  auto products=lowerActivationBatchMatmul(name,w,x,partial,false,nullptr,nullptr,true,true,true);
  products.shape={partitions*fields,32};
  auto sum=sliceRows(products,0,fields);
  if(partitions==2)
    return mhcScaleAdd(name+".combine_and_normalize",sum,sliceRows(products,fields,fields),inverse,"bf16",true);
  return mhcAlu(name+".normalize",sum,inverse,"multiply",false,0.f,"bf16");
}
std::vector<Tensor> lowerHcMixes(const std::string &name,const std::vector<Tensor> &inputs,
                                I iterations,float epsilon,float normEpsilon) {
  const auto &input=inputs[0],&weight=inputs[1];
  const I tokens=input.shape[0],h=input.shape[1],d=input.shape[2],fields=h*(h+2);
  const I block=std::min<I>(d,std::min<I>(target.memory().words_per_bank,
      target.memory().words_per_bank*32/fields)/32*32);
  std::vector<Tensor> preTiles,postTiles,combTiles,projectionTiles;
  for(I token=0;token<tokens;token+=32) {
    const auto prefix=name+".tokens"+std::to_string(token);
    std::vector<Tensor> xBlocks,normParts;
    for(I stream=0;stream<h;++stream)for(I first=0;first<d;first+=block) {
      const I width=std::min(block,d-first),idx=xBlocks.size();
      auto x=mhcTokenBlock(prefix+".input"+std::to_string(idx),input,token,stream*d+first,width);
      xBlocks.push_back(x);
      normParts.push_back(mhcSquareSum(prefix+".square_sum"+std::to_string(idx),x));
    }
    auto inverse=mhcSum(prefix+".inverse_rms",normParts,"add",
        {{"multiply",1.f/float(h*d)},{"add",normEpsilon},{"rsqrt",0.f}});
    auto normalized=mhcProjection(prefix+".projection_mxm",weight,xBlocks,inverse);
    mhcDrain();
    std::vector<Tensor> mixes;
    for(I field=0;field<fields;++field)mixes.push_back(sliceRows(normalized,field,1));
    projectionTiles.push_back(deltaTranspose(normalized));
    std::vector<Tensor> pre,post,logits;
    for(I field=0;field<fields;++field) {
      const I scale=field<h?0:field<2*h?1:2;
      auto scalar=inputs[2];scalar.shape={1,3};scalar=sliceColumns(scalar,scale,1);
      auto base=inputs[3];base.shape={1,fields};base=sliceColumns(base,field,1);
      auto affine=mhcScaleAdd(prefix+".affine"+std::to_string(field),mixes[field],scalar,base);
      if(field>=2*h) {logits.push_back(affine);continue;}
      auto gate=mhcAlu(prefix+".gate"+std::to_string(field),affine,affine,"sigmoid",true,0.f,"bf16",false,
          {{field<h?"add":"multiply",field<h?epsilon:2.f}});
      if(field<h)pre.push_back(gate);else post.push_back(gate);
    }
    auto matrix=concat(logits);
    auto maximum=mhcMatrixReduce(prefix+".rowmax",matrix,h,true,"max");
    auto exponential=mhcAlu(prefix+".exp",matrix,mhcMatrixBroadcast(maximum,h,true),"subtract",false,0.f,"f32",false,{{"exp",0.f}});
    auto reciprocal=mhcMatrixReduce(prefix+".softmax_inverse",exponential,h,true,"add",{{"reciprocal",0.f}});
    matrix=mhcAlu(prefix+".softmax",exponential,mhcMatrixBroadcast(reciprocal,h,true),"multiply",false,0.f,"f32",false,{{"add",epsilon}});
    matrix=mhcNormalize(prefix+".sinkhorn0.column",matrix,h,false,epsilon,iterations==1?"bf16":"f32");
    for(I i=1;i<iterations;++i) {
      matrix=mhcNormalize(prefix+".sinkhorn"+std::to_string(i)+".row",matrix,h,true,epsilon);
      matrix=mhcNormalize(prefix+".sinkhorn"+std::to_string(i)+".column",matrix,h,false,epsilon,i+1==iterations?"bf16":"f32");
    }
    preTiles.push_back(deltaTranspose(concat(pre)));postTiles.push_back(deltaTranspose(concat(post)));
    combTiles.push_back(deltaTranspose(matrix));mhcDrain();
  }
  auto comb=concat(combTiles);comb.shape={tokens,h,h};
  return {concat(preTiles),concat(postTiles),comb,concat(projectionTiles)};
}
Tensor lowerHcPre(const std::string &name,const Tensor &input,const Tensor &pre) {
  const I tokens=input.shape[0],h=input.shape[1],d=input.shape[2];
  const I block=std::min<I>(d,target.memory().words_per_bank/32*32);
  std::vector<Tensor> tiles;
  for(I token=0;token<tokens;token+=32) {
    std::vector<Tensor> pieces;
    for(I first=0;first<d;first+=block) {
      const I width=std::min(block,d-first);const auto prefix=name+".t"+std::to_string(token)+".d"+std::to_string(first);
      Tensor sum;
      for(I stream=0;stream<h;++stream) {
        auto x=mhcTokenBlock(prefix+".input"+std::to_string(stream),input,token,stream*d+first,width);
        auto c=mhcCoefficient(pre,token,stream,h);
        sum=stream?mhcScaleAdd(prefix+".weighted_add"+std::to_string(stream),x,c,sum,stream+1==h?"bf16":"f32"):
            mhcAlu(prefix+".multiply0",x,c,"multiply",false,0.f,h==1?"bf16":"f32");
      }
      pieces.push_back(deltaTranspose(sum));mhcDrain();
    }
    tiles.push_back(concat(pieces,1));
  }
  return concat(tiles);
}
Tensor lowerHcPost(const std::string &name,const std::vector<Tensor> &inputs) {
  const auto &residual=inputs[1];const I tokens=residual.shape[0],h=residual.shape[1],d=residual.shape[2];
  const I block=std::min<I>(d,target.memory().words_per_bank/32*32);
  std::vector<Tensor> tiles;
  for(I token=0;token<tokens;token+=32) {
    std::vector<Tensor> streams;
    for(I output=0;output<h;++output) {
      std::vector<Tensor> pieces;
      for(I first=0;first<d;first+=block) {
        const I width=std::min(block,d-first);
        const auto prefix=name+".t"+std::to_string(token)+".h"+std::to_string(output)+".d"+std::to_string(first);
        Tensor sum;
        for(I stream=0;stream<h;++stream) {
          auto x=mhcTokenBlock(prefix+".residual"+std::to_string(stream),residual,token,stream*d+first,width);
          auto c=mhcCoefficient(inputs[3],token,stream*h+output,h*h);
          sum=stream?mhcScaleAdd(prefix+".weighted_add"+std::to_string(stream),x,c,sum):
              mhcAlu(prefix+".residual_mul0",x,c,"multiply");
        }
        auto x=mhcTokenBlock(prefix+".sublayer",inputs[0],token,first,width);
        auto result=mhcScaleAdd(prefix+".result",x,mhcCoefficient(inputs[2],token,output,h),sum,"bf16");
        pieces.push_back(deltaTranspose(result));mhcDrain();
      }
      streams.push_back(concat(pieces,1));
    }
    tiles.push_back(concat(streams,1));
  }
  auto result=concat(tiles);result.shape=residual.shape;return result;
}
