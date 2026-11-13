// Direct planar FP32 matrix / scalar-row multiplication, retaining the
// existing FP32 arithmetic and final BF16 conversion boundary.
mlir::LogicalResult emitNativePackedRowScale(mlir::IRRewriter &r,
    mlir::func::FuncOp function,const target::LPUTargetModel &target) {
  auto words=function->getAttrOfType<mlir::ArrayAttr>("native_row_scale_words");
  auto factors=function->getAttrOfType<mlir::ArrayAttr>("native_row_scale_coefficients");
  auto ops=function.getOps<kernel::VectorAluOp>();
  auto op=*ops.begin();
  const auto type=op.getResult().getType();
  const int64_t count=type.getNumElements()/64;
  auto out=op->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement");
  const auto field=[](mlir::DictionaryAttr d,llvm::StringRef key) {
    return d.getAs<mlir::IntegerAttr>(key).getInt();
  };
  const auto sideOf=[](mlir::DictionaryAttr p) {
    return p.getAs<mlir::StringAttr>("hemisphere").getValue()=="west"?1:0;
  };
  if(!factors || words.size()!=2*count || factors.size()!=words.size())
    return function.emitError("invalid native row-scale word plan");
  llvm::SmallVector<mlir::DictionaryAttr> sources[2][2];
  for(int64_t i=0;i<2*count;++i) {
    auto a=llvm::cast<mlir::DictionaryAttr>(words[i]), b=llvm::cast<mlir::DictionaryAttr>(factors[i]);
    const auto side=sideOf(a.getAs<mlir::DictionaryAttr>("placement"));
    if(side!=sideOf(b.getAs<mlir::DictionaryAttr>("placement")))
      return function.emitError("row-scale operands must share source hemisphere");
    sources[0][side].push_back(a);sources[1][side].push_back(b);
  }
  r.setInsertionPoint(op);
  const int64_t start=96;
  for(int operand=0;operand<2;++operand)for(int side=0;side<2;++side) {
    const auto &input=sources[operand][side];
    if(input.size()!=count) return function.emitError("row-scale partitions must be balanced");
    // Each retained producer contributes a closed affine READ3D run. Runs
    // join at the VXM input, without a temporary matrix or inactive ALUs.
    for(int64_t first=0;first<count;) {
      const auto home=input[first].getAs<mlir::DictionaryAttr>("placement");
      const auto address=field(input[first],"address");
      int64_t n=1,stride=0;
      if(first+1<count && input[first+1].getAs<mlir::DictionaryAttr>("placement")==home)
        stride=field(input[first+1],"address")-address;
      while(first+n<count && input[first+n].getAs<mlir::DictionaryAttr>("placement")==home &&
          field(input[first+n],"address")==address+n*stride) ++n;
      auto ids=home.getAs<mlir::ArrayAttr>("slices");
      for(int byte=0;byte<4;++byte) {
        const int64_t slice=llvm::cast<mlir::IntegerAttr>(ids[byte]).getInt();
        const auto route=*target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,target::StreamDirection::West,slice);
        direct_domain_detail::emitMem3D(r,op.getLoc(),target,start+first-route,
            side*target.memory().slices_per_hemisphere+slice,"read",address,
            32+side*16+operand*4+byte,n,1,stride,1,1,0,1,1,0,-1,field(home,"bank"));
      }
      first+=n;
    }
  }
  mlir::Value current=op.getLhs();
  for(int col=0;col<8;++col) {
    auto v=ffn_detail::create_vxm(r,op.getLoc(),current,current,type,start-1,col,
        col?"bypass":"multiply",col?"previous":"stream_f32",0,0.f,
        col?"immediate":"stream_f32",col?0:4,0.f,col==7?"bf16":"fp32",col==7?6:-1,
        count,1,"east","west",-1,false,false,true,false,8);
    if(!col) {
      v->setAttr("lhs_stream_source",r.getStringAttr("local"));
      v->setAttr("rhs_stream_source",r.getStringAttr("local"));
    }
    current=v.getResult();
  }
  const int64_t latency=9;
  auto ids=out.getAs<mlir::ArrayAttr>("slices");
  int64_t end=start+latency+count;
  for(int side=0;side<2;++side)for(int byte=0;byte<2;++byte) {
    const int64_t slice=llvm::cast<mlir::IntegerAttr>(ids[byte]).getInt();
    const auto route=*target.transport_latency(target::StreamEndpoint::VxmResult,
        target::StreamEndpoint::Mem,target::StreamDirection::East,slice);
    const int64_t at=start+latency-1+route;
    direct_domain_detail::emitMem3D(r,op.getLoc(),target,at,
        side*target.memory().slices_per_hemisphere+slice,"write",field(out,"base_row"),
        (1-side)*8+6+byte,count,1,1,1,1,0,1,1,0,-1,field(out,"bank"));
    end=std::max(end,at+count+3);
  }
  mlir::OperationState timeline(op.getLoc(),TimelineOp::getOperationName());
  timeline.addAttributes({r.getNamedAttr("name",r.getStringAttr("vector.native_row_scale")),
      r.getNamedAttr("start",r.getI64IntegerAttr(start-1)),r.getNamedAttr("end",r.getI64IntegerAttr(end))});
  r.create(timeline);
  mlir::OperationState binding(op.getLoc(),BindingOp::getOperationName());
  binding.addTypes(type);
  binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),
      r.getNamedAttr("access",r.getStringAttr("output")),r.getNamedAttr("role",r.getStringAttr("result")),
      r.getNamedAttr("bytes",r.getI64IntegerAttr(type.getNumElements()*2)),r.getNamedAttr("placement",out)});
  r.replaceOp(op,llvm::cast<BindingOp>(r.create(binding)).getValue());
  return mlir::success();
}
