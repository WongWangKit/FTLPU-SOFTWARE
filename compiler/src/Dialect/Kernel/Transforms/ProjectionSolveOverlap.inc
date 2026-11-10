// Scheduling policy for two independent native operators. Called before graph
// liveness/allocation: moving a projection after memory planning is not safe.
void planProjectionSolveOverlap(Planner& planner, std::vector<Tensor>& finals) {
  auto enabled=planner.policies.getAs<mlir::BoolAttr>("ftlpu.projection_solve_overlap");
  if(!enabled || !enabled.getValue()) return;
  const auto& target=planner.target;
  if(target.memory().hemispheres!=2 || target.throughput().mxms_per_hemisphere!=1
      || target.streams().streams_per_direction<32
      || target.throughput().mxm_prefill_weight_streams!=1) return;
  mlir::Builder b(&planner.context);
  for(I solveIndex=0;solveIndex<I(planner.nodes.size());++solveIndex) {
    auto& solve=planner.nodes[solveIndex];
    if(!solve.module || solve.parallel>=0 || solve.projectionSolvePeer>=0) continue;
    auto vectors=solve.function.getOps<fc::kernel::VectorAluOp>();
    if(std::distance(vectors.begin(),vectors.end())!=1
        || (*vectors.begin()).getKind()!="unit_lower_solve") continue;
    I candidate=-1;
    for(I next=solveIndex+1;next<I(planner.nodes.size());++next) {
      auto& projection=planner.nodes[next];
      if(!projection.module || projection.parallel>=0 || projection.projectionSolvePeer>=0) continue;
      auto ops=projection.function.getOps<fc::tensor::ProjectionTaskOp>();
      if(std::distance(ops.begin(),ops.end())!=1 || (*ops.begin()).getKind()!="linear") continue;
      auto op=*ops.begin();
      const auto input=llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
      const auto weight=llvm::cast<mlir::RankedTensorType>(op.getWeight().getType());
      if(input.getDimSize(0)!=32 || input.getDimSize(1)%32
          || !weight.getElementType().isInteger(8)
          || projection.outputs.size()!=1 || projection.outputs[0].shape[1]%64) continue;
      auto policy=fc::target::mxm_execution_policy_from_operation(op.getOperation());
      if(mlir::failed(policy))continue;
      auto execution=fc::target::plan_mxm_execution_strategy(
          {32,projection.outputs[0].shape[1],input.getDimSize(1),
           input.getElementType().isBF16(),true,true,true},target,*policy);
      if(mlir::failed(execution) || !execution->uses_local_dequant())continue;
      auto config=op.getConfig();
      if(auto replicated=config.getAs<mlir::BoolAttr>("replicated_result");replicated && replicated.getValue()) continue;
      // No transitive dependence can cross this boundary: every nonconstant
      // input must already be produced before the solve starts.
      bool independent=true;
      for(const auto& input:projection.inputs)for(const auto& ref:input.refs)
        independent &= ref.node<solveIndex;
      if(!independent) continue;
      candidate=next;break;
    }
    if(candidate<0) continue;
    // Stable move retains all other topological relationships. Remap logical
    // views and exported tensors as well as ordinary producer edges.
    std::rotate(planner.nodes.begin()+solveIndex,planner.nodes.begin()+candidate,
                planner.nodes.begin()+candidate+1);
    const auto remap=[&](Tensor& tensor) {
      for(auto& ref:tensor.refs) {
        if(ref.node==candidate)ref.node=solveIndex;
        else if(ref.node>=solveIndex && ref.node<candidate)++ref.node;
      }
    };
    for(auto& node:planner.nodes) {
      for(auto& input:node.inputs)remap(input);
      for(auto& output:node.outputs)remap(output);
    }
    for(auto& output:finals)remap(output);
    auto& projection=planner.nodes[solveIndex];
    auto& pairedSolve=planner.nodes[solveIndex+1];
    projection.projectionSolvePeer=solveIndex+1;
    pairedSolve.projectionSolvePeer=solveIndex;
    auto op=*projection.function.getOps<fc::tensor::ProjectionTaskOp>().begin();
    mlir::NamedAttrList config(op.getConfig());
    config.set("projection_solve_overlap",b.getBoolAttr(true));
    op->setAttr("config",config.getDictionary(&planner.context));
    pairedSolve.function.walk([&](fc::kernel::VectorAluOp vector) {
      vector->setAttr("solve.projection_overlap",b.getBoolAttr(true));
    });
    ++solveIndex;
  }
}
