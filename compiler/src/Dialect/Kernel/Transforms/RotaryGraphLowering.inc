// Standalone adjacent-pair RoPE. Constants encode only positional mathematics;
// all input-dependent permutation and arithmetic execute on the common units.
Tensor lowerInterleavedRotary(const std::string &name, const Tensor &input,
    I rotaryDim, I rotaryOffset, float theta, bool inverse, I position) {
  const I tokens=input.shape[0],heads=input.shape[1],width=input.shape[2];
  if(rotaryDim%32 || tokens%32 || rotaryOffset<0 || rotaryOffset+rotaryDim>width)
    throw std::runtime_error("standalone interleaved RoPE requires aligned tokens/rotary dimension and a valid rotary region");
  auto resident=input;
  const auto wide=policies.getAs<mlir::BoolAttr>("ftlpu.sxm_wide");
  const bool native=wide && wide.getValue() && input.dtype=="bf16" && rotaryDim>0 && rotaryDim<=256
      && rotaryDim%64==0 && rotaryOffset%32==0 && width%32==0;
  if(resident.external()) {resident.shape={tokens,heads*width};resident=join(resident,name+".input",native);resident.shape=input.shape;}
  if(native) {
    // Within each 32-feature tile, compute even outputs and then odd outputs.
    // Both halves read X in ascending pair order. Arrange the immutable
    // coefficients in that execution order; no input-dependent copy is needed.
    std::vector<float> coefficients;
    for(I block=0;block<rotaryDim;block+=32)for(I parity=0;parity<2;++parity)
    for(I pair=0;pair<16;++pair)for(I term=0;term<2;++term)for(I t=0;t<tokens;++t) {
      const float angle=float(position+t)/std::pow(theta,float(block+2*pair)/rotaryDim);
      const bool cosine=term==parity;
      coefficients.push_back(cosine ? std::cos(angle)
          : (inverse?-1.f:1.f)*(parity?1.f:-1.f)*std::sin(angle));
    }
    auto table=join(deltaConstant({2*rotaryDim,tokens},coefficients),name+".paired_coefficients",true);
    std::vector<Tensor> perHead;
    // At most 512 single-word accumulator configurations per hemisphere
    // group leave room for the real iMEM reload/sync controls.
    const I headBatch=512/rotaryDim;
    for(I h0=0;h0<heads;h0+=headBatch) {
      const I count=std::min(headBatch,heads-h0);std::vector<Tensor> rows,tails;
      for(I h=h0;h<h0+count;++h) {
        V ids;for(I t=0;t<tokens;++t)for(I d=0;d<width;++d)ids.push_back((t*heads+h)*width+d);
        rows.push_back(view(resident,ids,{tokens,width}));
        tails.push_back(sliceColumns(rows.back(),rotaryOffset,rotaryDim));
      }
      Node n;n.name=name+".paired_heads"+std::to_string(h0);n.planarRotaryDim=rotaryDim;
      n.inputs={concat(tails,1),table};
      for(I block=0;block<count*rotaryDim/32;++block)n.outputs.push_back(output(nodes.size(),block,{tokens,32},"bf16"));
      auto results=n.outputs;nodes.push_back(std::move(n));
      for(I h=0;h<count;++h) {
        std::vector<Tensor> parts;
        if(rotaryOffset)parts.push_back(sliceColumns(rows[h],0,rotaryOffset));
        for(I block=0;block<rotaryDim/32;++block)parts.push_back(results[h*rotaryDim/32+block]);
        if(rotaryOffset+rotaryDim<width)parts.push_back(sliceColumns(rows[h],rotaryOffset+rotaryDim,width-rotaryOffset-rotaryDim));
        perHead.push_back(concat(parts,1));
      }
    }
    auto result=concat(perHead,1);result.shape=input.shape;return result;
  }
  Tensor rotation;rotation.shape={1,rotaryDim,rotaryDim};rotation.dtype="bf16";
  rotation.constant=deltaConstantId++;rotation.data.resize(rotation.size()*2,0);
  for(I d=0;d<rotaryDim;d+=2) {
    uint16_t plus=0x3f80,minus=0xbf80;
    std::memcpy(rotation.data.data()+2*(d*rotaryDim+d+1),&plus,2);
    std::memcpy(rotation.data.data()+2*((d+1)*rotaryDim+d),&minus,2);
  }
  std::vector<float> cosines,sines;
  for(I t=0;t<tokens;++t)for(I d=0;d<rotaryDim;++d) {
    const float angle=float(position+t)/std::pow(theta,float(d/2*2)/rotaryDim);
    cosines.push_back(std::cos(angle));sines.push_back((inverse?-1.f:1.f)*std::sin(angle));
  }
  auto cosine=join(deltaConstant({tokens,rotaryDim},cosines),name+".cosine_table");
  auto sine=join(deltaConstant({tokens,rotaryDim},sines),name+".sine_table");
  const auto binary=[&](const std::string &label,const Tensor &a,const Tensor &b,
                        const std::string &kind,const std::string &dtype) {
    auto result=deltaAlu(label,a,b,kind,false,dtype,true,true);
    // Both operands arrive on the fixed VXM ports in the same cycle.
    // The common allocator must assign distinct MEM ICU queues to them.
    nodes.back().function->setAttr("ftlpu.vector_independent_ports",mlir::BoolAttr::get(&context,true));
    return result;
  };
  std::vector<Tensor> perHead;
  for(I h=0;h<heads;++h) {
    V ids;for(I t=0;t<tokens;++t)for(I d=0;d<width;++d)ids.push_back((t*heads+h)*width+d);
    auto row=view(resident,ids,{tokens,width});auto tail=sliceColumns(row,rotaryOffset,rotaryDim);
    auto lhs=tail;lhs.shape={1,tokens,rotaryDim};auto output=lhs;
    auto rotated=lowerActivationBatchMatmul(name+".head"+std::to_string(h)+".pair_rotation",lhs,rotation,output,false);
    rotated.shape=tail.shape;
    const auto a=binary(name+".cos"+std::to_string(h),tail,cosine,"multiply","f32");
    const auto b=binary(name+".sin"+std::to_string(h),rotated,sine,"multiply","f32");
    auto r=binary(name+".sum"+std::to_string(h),a,b,"add","bf16");
    std::vector<Tensor> parts;
    if(rotaryOffset)parts.push_back(sliceColumns(row,0,rotaryOffset));
    parts.push_back(r);
    if(rotaryOffset+rotaryDim<width)parts.push_back(sliceColumns(row,rotaryOffset+rotaryDim,width-rotaryOffset-rotaryDim));
    perHead.push_back(concat(parts,1));
    nodes.back().function->setAttr("ftlpu.imem_drain_after",mlir::UnitAttr::get(&context));
  }
  auto result=concat(perHead,1);result.shape=input.shape;return result;
}
