// Generic FP32-preserving planar arithmetic chains. These matches operate on
// the logical SSA graph, before any instruction or placement trace is emitted.
bool hasPlanarArithmeticContract(kernel::VectorAluOp op) {
  auto fn=op->getParentOfType<mlir::func::FuncOp>();
  return fn && fn->hasAttr("ftlpu.vector_bf16_spill");
}
bool isPlanarArithmeticType(mlir::RankedTensorType type) {
  return type.getRank()==2 && type.getDimSize(0)>0 && type.getDimSize(1)>0 &&
      (type.getElementType().isF32() || type.getElementType().isBF16());
}
bool matchScaleAdd(kernel::VectorAluOp mul,kernel::VectorAluOp add) {
  if(!mul || !add || mul.getKind()!="multiply" || mul.getUseImmediate() || mul.getReduce() ||
      add.getKind()!="add" || add.getUseImmediate() || add.getReduce() ||
      add.getLhs()!=mul.getResult() || !mul.getResult().hasOneUse() ||
      !mul.getResult().getType().getElementType().isF32() || !hasPlanarArithmeticContract(mul))return false;
  const auto x=mul.getLhs().getType(),c=mul.getRhs().getType(),s=add.getRhs().getType();
  return isPlanarArithmeticType(x) && isPlanarArithmeticType(c) && isPlanarArithmeticType(s) &&
      isPlanarArithmeticType(add.getResult().getType()) && x.getDimSize(1)%32==0 &&
      c.getShape()==llvm::ArrayRef<int64_t>({1,x.getDimSize(1)}) &&
      s.getDimSize(1)==x.getDimSize(1) && (s.getDimSize(0)==1 || s.getDimSize(0)==x.getDimSize(0)) &&
      !(x.getElementType().isF32() && !s.getElementType().isF32());
}
bool matchSquareSum(kernel::VectorAluOp square,kernel::VectorAluOp sum) {
  if(!square || !sum || square.getKind()!="multiply" || square.getUseImmediate() || square.getReduce() ||
      square.getLhs()!=square.getRhs() || !reduceAlongRows(sum) ||
      !llvm::all_of(square.getResult().getUsers(),[&](auto *user){return user==sum.getOperation();}) ||
      sum.getLhs()!=square.getResult() || sum.getKind()!="multiply" || !sum.getUseImmediate() ||
      sum.getImmediate().convertToFloat()!=1.f || !hasPlanarArithmeticContract(square) ||
      !isPlanarArithmeticType(square.getLhs().getType()) || square.getLhs().getType().getDimSize(1)%32)return false;
  const auto kind=sum->getAttrOfType<mlir::StringAttr>("reduction_kind");
  return kind && kind.getValue()=="add" && square.getResult().getType().getElementType().isF32() &&
      sum.getResult().getType().getElementType().isF32();
}

bool matchAddScale(kernel::VectorAluOp add,kernel::VectorAluOp mul) {
  if(!add || !mul || add.getKind()!="add" || add.getUseImmediate() || add.getReduce() ||
      mul.getKind()!="multiply" || mul.getUseImmediate() || mul.getReduce() ||
      mul.getLhs()!=add.getResult() || !add.getResult().hasOneUse() ||
      !add.getResult().getType().getElementType().isF32() || !hasPlanarArithmeticContract(add))return false;
  const auto x=add.getLhs().getType(),y=add.getRhs().getType(),c=mul.getRhs().getType();
  return isPlanarArithmeticType(x) && isPlanarArithmeticType(y) && isPlanarArithmeticType(c) &&
      isPlanarArithmeticType(mul.getResult().getType()) && x.getDimSize(1)%32==0 && y.getShape()==x.getShape() &&
      c.getShape()==llvm::ArrayRef<int64_t>({1,x.getDimSize(1)}) &&
      !(x.getElementType().isF32() && !y.getElementType().isF32());
}

struct PointwiseFoldPlan {
  llvm::SmallVector<int64_t> columns;
  llvm::SmallVector<mlir::Value> inputs;
};
PointwiseFoldPlan matchPointwiseFold(llvm::ArrayRef<kernel::VectorAluOp> ops) {
  PointwiseFoldPlan plan;
  auto head=ops.front();const auto ty=head.getLhs().getType();
  if((head.getKind()!="add" && head.getKind()!="max") || head.getUseImmediate() || head.getReduce() ||
      !hasPlanarArithmeticContract(head) || !isPlanarArithmeticType(ty) ||
      !ty.getElementType().isF32() || ty!=head.getRhs().getType() ||
      ty.getDimSize(0)>32 || ty.getDimSize(1)%32)return plan;
  plan.columns={0};plan.inputs={head.getLhs(),head.getRhs()};
  size_t i=1;
  for(;i<ops.size() && plan.inputs.size()<4;++i) {
    auto next=ops[i],prior=ops[i-1];
    if(next.getKind()!=head.getKind() || next.getUseImmediate() || next.getReduce() ||
        next.getLhs()!=prior.getResult() || !prior.getResult().hasOneUse() ||
        next.getRhs().getType()!=ty || next.getRhs().getDefiningOp<kernel::VectorAluOp>())break;
    plan.columns.push_back(plan.inputs.size()==2?1:3);plan.inputs.push_back(next.getRhs());
  }
  if(plan.inputs.size()<3)return {};
  for(;i<ops.size();++i) {
    auto next=ops[i],prior=ops[i-1];
    if(!next.getUseImmediate() || next.getReduce() || next.getLhs()!=prior.getResult() ||
        !llvm::all_of(prior.getResult().getUsers(),[&](auto *user){return user==next.getOperation();}) ||
        !prior.getResult().getType().getElementType().isF32())break;
    auto k=next.getKind();
    if(k!="add" && k!="multiply" && k!="reciprocal" && k!="rsqrt")break;
    int64_t col=plan.columns.back()+1;
    if(k=="reciprocal" || k=="rsqrt")while(col%4!=3)++col;
    if(col>=8)break;
    plan.columns.push_back(col);
  }
  return plan;
}

mlir::LogicalResult emitPointwiseFold(mlir::IRRewriter &r,
    llvm::ArrayRef<kernel::VectorAluOp> chain,const target::LPUTargetModel &target) {
  auto plan=matchPointwiseFold(chain);auto head=chain.front(),tail=chain.back();const auto loc=head.getLoc();
  const auto ty=head.getLhs().getType(),out=tail.getResult().getType();
  llvm::SmallVector<mlir::DictionaryAttr> homes;
  homes.push_back(head->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_lhs_placement"));
  homes.push_back(head->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement"));
  for(size_t i=2;i<plan.inputs.size();++i)homes.push_back(chain[i-1]->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_rhs_placement"));
  auto output=tail->getAttrOfType<mlir::DictionaryAttr>("ftlpu.vector_result_placement");auto os=storageOf(output);
  if(!os.both || llvm::any_of(homes,[](auto p){return !storageOf(p).both;}))return head.emitError("fold chain requires mirrored planar operands");
  r.setInsertionPoint(head);
  for(size_t i=0;i<plan.inputs.size();++i)if(auto arg=llvm::dyn_cast<mlir::BlockArgument>(plan.inputs[i])) {
    mlir::OperationState binding(loc,BindingOp::getOperationName());binding.addOperands(arg);binding.addTypes(ty);
    binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(arg.getArgNumber())),r.getNamedAttr("access",r.getStringAttr("input")),
        r.getNamedAttr("role",r.getStringAttr("activation")),r.getNamedAttr("bytes",r.getI64IntegerAttr(ty.getNumElements()*4)),r.getNamedAttr("placement",homes[i])});r.create(binding);
  }
  const auto read=[&](int input,int64_t row,int64_t word,int64_t at,int port) {
    auto st=storageOf(homes[input]);const int64_t address=st.base+(st.access?st.access->address(row,word):word*ty.getDimSize(0)+row);
    for(int side=0;side<2;++side)for(int byte=0;byte<4;++byte) {
      auto slice=st.slice+byte;
      direct_domain_detail::emitMem3D(r,loc,target,at-1-slice/4,side*target.memory().slices_per_hemisphere+slice,
          "read",address,32+side*16+port+byte,1,1,0,1,1,0,1,1,0,-1,st.bank);
    }
  };
  const auto emit=[&](int64_t at,int col,llvm::StringRef kind,llvm::StringRef lhs,llvm::StringRef rhs,float immediate,
      bool capture=false,bool output=false) {
    auto v=ffn_detail::create_vxm(r,loc,head.getLhs(),head.getRhs(),out,at-1,col,kind,lhs,0,0.f,rhs,rhs=="stream_f32"?4:0,immediate,
        output && out.getElementType().isBF16()?"bf16":"fp32",output?6:-1,1,1,"east","west",-1,false,false,true,capture,8);
    if(lhs=="stream_f32")v->setAttr("lhs_stream_source",r.getStringAttr("local"));
    if(rhs=="stream_f32")v->setAttr("rhs_stream_source",r.getStringAttr("local"));
  };
  int64_t cursor=96;
  for(int64_t word=0;word<ty.getDimSize(1)/32;++word)for(int64_t row=0;row<ty.getDimSize(0);++row) {
    for(size_t input=2;input<plan.inputs.size();++input) {
      read(input,row,word,cursor,0);
      for(int col=0;col<8;++col)emit(cursor,col,"bypass",col?"previous":"stream_f32","immediate",0.f,col==(input==2?1:3));
      ++cursor;
    }
    read(0,row,word,cursor,0);read(1,row,word,cursor,4);
    int64_t latency=8;
    for(int col=0;col<8;++col) {
      auto found=llvm::find(plan.columns,col);const bool active=found!=plan.columns.end();
      auto member=active?chain[found-plan.columns.begin()]:head;
      auto kind=active?member.getKind():llvm::StringRef("bypass");
      auto rhs=col==0?llvm::StringRef("stream_f32"):active && !member.getUseImmediate()?llvm::StringRef("accumulator"):llvm::StringRef("immediate");
      emit(cursor,col,kind,col?"previous":"stream_f32",rhs,active?member.getImmediate().convertToFloat():0.f,false,col==7);
      latency+=kind=="reciprocal" || kind=="rsqrt"?4:kind=="multiply"?1:0;
    }
    for(int side=0;side<2;++side)for(int byte=0;byte<(out.getElementType().isF32()?4:2);++byte) {
      auto slice=os.slice+byte;
      direct_domain_detail::emitMem3D(r,loc,target,cursor+latency+slice/4,side*target.memory().slices_per_hemisphere+slice,"write",
          os.base+word*ty.getDimSize(0)+row,(1-side)*8+6+byte,1,1,0,1,1,0,1,1,0,-1,os.bank);
    }
    cursor+=latency+(os.slice+3)/4+16;
  }
  mlir::OperationState timeline(loc,TimelineOp::getOperationName());
  timeline.addAttributes({r.getNamedAttr("name",r.getStringAttr("vector.fold_chain")),r.getNamedAttr("start",r.getI64IntegerAttr(96)),r.getNamedAttr("end",r.getI64IntegerAttr(cursor))});r.create(timeline);
  mlir::OperationState binding(loc,BindingOp::getOperationName());binding.addTypes(out);
  binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),r.getNamedAttr("access",r.getStringAttr("output")),
      r.getNamedAttr("role",r.getStringAttr("result")),r.getNamedAttr("bytes",r.getI64IntegerAttr(out.getNumElements()*(out.getElementType().isF32()?4:2))),r.getNamedAttr("placement",output)});
  auto result=llvm::cast<BindingOp>(r.create(binding));r.replaceOp(tail,result.getValue());
  for(int i=int(chain.size())-2;i>=0;--i)r.eraseOp(chain[i]);
  return mlir::success();
}

mlir::LogicalResult emitPlanarArithmeticFusion(mlir::IRRewriter &r,
    llvm::ArrayRef<kernel::VectorAluOp> chain,const target::LPUTargetModel &target,bool squareSum) {
  auto head=chain.front(),tail=chain.back();const auto loc=head.getLoc();
  const bool addScale=matchAddScale(head,tail);
  auto x=head.getLhs().getType(),out=tail.getResult().getType();
  const int64_t rows=x.getDimSize(0),words=x.getDimSize(1)/32;
  if(!words)return head.emitError("planar arithmetic fusion requires full vector words");
  const auto home=[](kernel::VectorAluOp op,llvm::StringRef attr){return op->getAttrOfType<mlir::DictionaryAttr>(attr);};
  const auto xp=home(head,"ftlpu.vector_lhs_placement"),cp=home(addScale?tail:head,"ftlpu.vector_rhs_placement"),
      yp=home(addScale?head:tail,"ftlpu.vector_rhs_placement"),op=home(tail,"ftlpu.vector_result_placement");
  const auto coefficientType=(addScale?tail:head).getRhs().getType();
  const auto secondType=(addScale?head:tail).getRhs().getType();
  auto xs=storageOf(xp),cs=storageOf(cp),ys=storageOf(yp),os=storageOf(op);
  if(!xs.both || !os.both || (!squareSum && (!cs.both || !ys.both)))
    return head.emitError("planar arithmetic fusion requires mirrored fixed-port storage");
  r.setInsertionPoint(head);
  llvm::SmallVector<mlir::Value> bound;
  for(auto alu:chain)for(auto pair:{std::make_pair(alu.getLhs(),home(alu,"ftlpu.vector_lhs_placement")),
                                  std::make_pair(alu.getRhs(),home(alu,"ftlpu.vector_rhs_placement"))}) {
    auto arg=llvm::dyn_cast<mlir::BlockArgument>(pair.first);
    if(!arg || llvm::is_contained(bound,pair.first))continue;
    bound.push_back(pair.first);
    mlir::OperationState binding(loc,BindingOp::getOperationName());
    binding.addOperands(pair.first);binding.addTypes(pair.first.getType());
    binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(arg.getArgNumber())),r.getNamedAttr("access",r.getStringAttr("input")),
        r.getNamedAttr("role",r.getStringAttr("activation")),r.getNamedAttr("name",r.getStringAttr("planar_fusion")),
        r.getNamedAttr("bytes",r.getI64IntegerAttr(llvm::cast<mlir::RankedTensorType>(pair.first.getType()).getNumElements()*(llvm::cast<mlir::RankedTensorType>(pair.first.getType()).getElementType().isF32()?4:2))),
        r.getNamedAttr("placement",pair.second)});
    r.create(binding);
  }
  const auto read=[&](Storage st,mlir::RankedTensorType ty,int64_t word,int64_t at,int64_t port,int64_t count) {
    analysis::PlanarReadPlan access=st.access.value_or(analysis::PlanarReadPlan{ty.getDimSize(0),ty.getDimSize(1)/32,0,ty.getDimSize(0),1,0,ty.getDimSize(0)});
    access.offset+=word*access.columnStride;access.columns=1;
    if(ty.getDimSize(0)==1) {access.rows=count;access.rowChunk=count;access.rowStride=access.groupStride=0;}
    for(int side=0;side<2;++side)for(int byte=0;byte<(ty.getElementType().isF32()?4:2);++byte) {
      const auto slice=st.slice+byte;
      emitPlanarReadDomain(r,loc,target,at-1-slice/4,side*target.memory().slices_per_hemisphere+slice,
          st.base,32+side*16+port+byte,st.bank,access,1);
    }
  };
  const auto alu=[&](int64_t at,int col,llvm::StringRef kind,llvm::StringRef lhs,llvm::StringRef rhs,
      int port,int64_t count,int depth,bool reset=false,bool write=false,bool emit=true,bool capture=false,llvm::StringRef format="fp32",int output=-1) {
    auto v=ffn_detail::create_vxm(r,loc,head.getLhs(),head.getRhs(),out,at,col,kind,lhs,0,0.f,rhs,port,0.f,format,output,
        count,1,"east","west",-1,reset,write,emit,capture,depth);
    if(lhs.starts_with("stream"))v->setAttr("lhs_stream_source",r.getStringAttr("local"));
    if(rhs.starts_with("stream"))v->setAttr("rhs_stream_source",r.getStringAttr("local"));
  };
  int64_t cursor=96;
  for(int64_t word=0;word<words;++word) {
    int64_t last=0;
    if(squareSum) {
      const int64_t begin=cursor;
      read(xs,x,word,begin,0,rows);
      alu(begin-1,0,"bypass",x.getElementType().isF32()?"stream_f32":"stream_bf16","immediate",0,rows,4);
      alu(begin-1,1,"multiply","previous","original",0,rows,4);
      alu(begin-1,2,"bypass","previous","immediate",0,rows,4);
      alu(begin-1,3,"add","previous","accumulator",0,1,4,true,true,rows==1,false,"fp32",2);
      if(rows>2)alu(begin,3,"add","previous","accumulator",0,rows-2,4,false,true,false,false,"fp32",2);
      if(rows>1)alu(begin+(rows>2?1:0),3,"add","previous","accumulator",0,1,4,false,true,true,false,"fp32",2);
      last=begin+rows+5;
    } else {
      const int64_t capture=cursor,begin=capture+1;
      read(cs,coefficientType,word,capture,0,1);
      for(int col=0;col<4;++col)alu(capture-1,col,"bypass",col?"previous":coefficientType.getElementType().isF32()?"stream_f32":"stream_bf16","immediate",0,1,4,false,false,true,col==1);
      read(xs,x,word,begin,0,rows);
      const bool full=secondType.getElementType().isF32();
      read(ys,secondType,word,begin,full?4:2,rows);
      alu(begin-1,0,addScale?"add":"bypass",x.getElementType().isF32()?"stream_f32":"stream_bf16",full?"stream_f32":"stream_bf16",full?4:2,rows,4);
      alu(begin-1,1,"multiply","previous","accumulator",0,rows,4);
      alu(begin-1,2,addScale?"bypass":"add","previous",addScale?"immediate":"auxiliary",0,rows,4);
      alu(begin-1,3,"bypass","previous","immediate",0,rows,4,false,false,true,false,out.getElementType().isF32()?"fp32":"bf16",2);
      last=begin+rows+5;
    }
    const int64_t count=squareSum?1:rows,first=last-count;
    for(int side=0;side<2;++side)for(int byte=0;byte<(out.getElementType().isF32()?4:2);++byte) {
      const auto slice=os.slice+byte;
      direct_domain_detail::emitMem3D(r,loc,target,first+slice/4,side*target.memory().slices_per_hemisphere+slice,"write",
          os.base+word*count,(1-side)*8+2+byte,count,1,1,1,1,0,1,1,0,-1,os.bank);
    }
    cursor=last+(os.slice+3)/4+16;
  }
  mlir::OperationState timeline(loc,TimelineOp::getOperationName());
  timeline.addAttributes({r.getNamedAttr("name",r.getStringAttr(squareSum?"vector.square_sum_chain":addScale?"vector.add_scale_chain":"vector.scale_add_chain")),
      r.getNamedAttr("start",r.getI64IntegerAttr(96)),r.getNamedAttr("end",r.getI64IntegerAttr(cursor))});r.create(timeline);
  mlir::OperationState binding(loc,BindingOp::getOperationName());binding.addTypes(out);
  binding.addAttributes({r.getNamedAttr("index",r.getI64IntegerAttr(0)),r.getNamedAttr("access",r.getStringAttr("output")),
      r.getNamedAttr("role",r.getStringAttr("result")),r.getNamedAttr("bytes",r.getI64IntegerAttr(out.getNumElements()*(out.getElementType().isF32()?4:2))),r.getNamedAttr("placement",op)});
  auto result=llvm::cast<BindingOp>(r.create(binding));r.replaceOp(tail,result.getValue());r.eraseOp(head);
  return mlir::success();
}
