// Delay terminal FP32 pointwise work until an independent later projection.
// Plan the complete group before liveness and allocation; only exports may
// consume a delayed value, so the stable move preserves every SSA dependency.
void planProjectionVectorOverlap(Planner &planner, std::vector<Tensor> &finals) {
  auto enabled=planner.policies.getAs<mlir::BoolAttr>("ftlpu.projection_vector_overlap");
  if(!enabled || !enabled.getValue())return;
  const auto &target=planner.target;
  const auto &hw=target.throughput();
  if(target.memory().hemispheres!=2 || hw.mxms_per_hemisphere!=1 || hw.mxm_rows!=32 ||
     hw.mxm_prefill_weight_streams!=1 || hw.mxm_weight_buffers!=2 ||
     !target.supports_mxm_weight_activation_overlap() || target.streams().streams_per_direction!=32)
    return;
  auto &nodes=planner.nodes;
  const I count=nodes.size();
  std::set<I> consumed, exported;
  for(const auto &node:nodes)for(const auto &input:node.inputs)
    for(const auto &ref:input.refs)consumed.insert(ref.node);
  for(const auto &output:finals)for(const auto &ref:output.refs)exported.insert(ref.node);
  V terminal;
  for(I i=0;i<count;++i) {
    auto &node=nodes[i];
    if(!node.module || node.parallel>=0 || node.projectionSolvePeer>=0 ||
       node.projectionConvGroup>=0 || consumed.count(i) || !exported.count(i) ||
       node.outputs.size()!=1 || node.outputs[0].dtype!="f32" ||
       node.outputs[0].shape.size()!=2 || node.outputs[0].shape[1]%32 ||
       !node.function->hasAttr("ftlpu.vector_bf16_spill"))continue;
    bool pointwise=true, arithmetic=false;
    for(auto &operation:node.function.front()) {
      if(llvm::isa<mlir::func::ReturnOp>(operation))continue;
      auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(operation);
      if(!op || op.getReduce() || (op.getKind()!="multiply" &&
          op.getKind()!="add" && op.getKind()!="subtract")) {pointwise=false;break;}
      arithmetic=true;
    }
    if(pointwise && arithmetic)terminal.push_back(i);
  }
  if(terminal.empty())return;
  I projection=-1;
  for(I i=count-1;i>terminal.front();--i) {
    auto &node=nodes[i];
    if(!node.module || node.parallel>=0 || node.projectionSolvePeer>=0 ||
       node.projectionConvGroup>=0)continue;
    auto ops=node.function.getOps<fc::tensor::ProjectionTaskOp>();
    if(std::distance(ops.begin(),ops.end())!=1)continue;
    auto op=*ops.begin();
    if(op.getKind()!="linear" || node.inputs.size()!=2 || node.outputs.size()!=1 ||
       node.inputs[0].shape.size()!=2 || node.inputs[1].shape.size()!=2 ||
       node.inputs[0].shape[0]!=32 || node.inputs[0].shape[1]%64 ||
       node.inputs[0].dtype!="bf16" || node.inputs[1].dtype!="i8" ||
       node.outputs[0].shape[1]%64 || !node.inputs[1].external())continue;
    if(auto replicated=op.getConfig().getAs<mlir::BoolAttr>("replicated_result");
       replicated && replicated.getValue())continue;
    auto policy=fc::target::mxm_execution_policy_from_operation(op.getOperation());
    if(mlir::failed(policy))continue;
    auto strategy=fc::target::plan_mxm_execution_strategy(
        {32,node.outputs[0].shape[1],node.inputs[0].shape[1],true,true,true,true},target,*policy);
    if(mlir::succeeded(strategy) && strategy->uses_local_dequant()) {projection=i;break;}
  }
  if(projection<0)return;
  V moved;
  for(I i:terminal)if(i<projection)moved.push_back(i);
  if(moved.empty())return;
  V order;
  for(I i=0;i<count;++i) {
    if(llvm::is_contained(moved,i))continue;
    order.push_back(i);
    if(i==projection)order.insert(order.end(),moved.begin(),moved.end());
  }
  V remap(count);
  for(I i=0;i<count;++i)remap[order[i]]=i;
  const I leader=remap[projection];
  auto op=*nodes[projection].function.getOps<fc::tensor::ProjectionTaskOp>().begin();
  mlir::NamedAttrList config(op.getConfig());
  config.set("projection_vector_overlap",mlir::BoolAttr::get(&planner.context,true));
  op->setAttr("config",config.getDictionary(&planner.context));
  auto remapTensor=[&](Tensor &tensor) {
    for(auto &ref:tensor.refs)ref.node=remap.at(ref.node);
  };
  std::vector<Node> reordered;
  reordered.reserve(count);
  for(I old:order) {
    auto &node=nodes[old];
    for(auto &input:node.inputs)remapTensor(input);
    for(auto &output:node.outputs)remapTensor(output);
    if(node.projectionSolvePeer>=0)node.projectionSolvePeer=remap.at(node.projectionSolvePeer);
    if(node.projectionConvGroup>=0)node.projectionConvGroup=remap.at(node.projectionConvGroup);
    if(old==projection || llvm::is_contained(moved,old)) {
      node.projectionConvGroup=leader;
      node.projectionVectorMember=true;
    }
    reordered.push_back(std::move(node));
  }
  for(auto &output:finals)remapTensor(output);
  nodes=std::move(reordered);
}
