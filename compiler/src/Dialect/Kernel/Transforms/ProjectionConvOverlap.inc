// Pipeline already produced projection pages through their depthwise consumers.
// Once a complete channel page is ready, the causal conv is independent of
// every later output column of the projection, including across token tiles.
void planProjectionConvOverlap(Planner &planner, std::vector<Tensor> &finals) {
  auto enabled=planner.policies.getAs<mlir::BoolAttr>("ftlpu.projection_conv_overlap");
  if(!enabled || !enabled.getValue())return;
  const auto &target=planner.target;
  const auto &hw=target.throughput();
  if(target.memory().hemispheres!=2 || hw.mxms_per_hemisphere!=1 || hw.mxm_rows!=32 ||
     hw.mxm_prefill_weight_streams!=1 || hw.mxm_weight_buffers!=2 ||
     !target.supports_mxm_weight_activation_overlap() || target.streams().streams_per_direction!=32 ||
     !fc::target::GraphWeightPaging::resolve(target,planner.policies).overlap)return;

  auto &nodes=planner.nodes;
  const I count=nodes.size();
  // Assign pairs as units so their two hemisphere programs retain the same
  // start. Require contiguous projection pages followed by native conv pairs.
  std::map<I,V> after;
  std::map<I,I> earlyIngress;
  const auto ingress=[&](I id) {
    const auto &n=nodes[id];
    return !n.module && n.inputs.size()==1 && n.inputs[0].external();
  };
  std::set<I> moved;
  for(I first=0;first<count;) {
    const I family=nodes[first].projectionPageFamily;
    if(family<0) {++first;continue;}
    I end=first+1;
    while(end<count && nodes[end].projectionPageFamily==family)++end;
    for(I scan=end;end-first>=2 && scan+1<count;) {
      const I pair=scan;
      if(ingress(pair)) {++scan;continue;}
      scan+=2;
      auto &left=nodes[pair], &right=nodes[pair+1];
      if(left.parallel<0 || left.parallel!=right.parallel ||
         (pair+2<count && nodes[pair+2].parallel==left.parallel))break;
      I producer=-1;
      bool eligible=true;
      std::set<I> history;
      for(auto *conv:{&left,&right}) {
        if(!conv->module || !conv->function->hasAttr("ftlpu.depthwise_conv_paired") ||
           conv->function.getOps<fc::tensor::DepthwiseConv1DTaskOp>().empty() ||
           conv->inputs.size()!=2 || !conv->inputs[1].external() ||
           conv->inputs[0].refs.empty()) {eligible=false;break;}
        const auto& window=conv->inputs[0];
        const I owner=window.refs.back().node;
        if(producer<0)producer=owner;
        if(producer!=owner) {eligible=false;break;}
        for(I j=0;j<window.size();++j) {
          const auto ref=window.refs[j];
          if(ref.node==producer && ref.output==0)continue;
          // A decode tap may come from external history. It must already
          // exist independently of the projection; never hoist a computed
          // join or a consumer of any of the pages under consideration.
          if(window.shape.size()==2 && window.shape[0]==4 &&
             j<3*window.shape[1] && ref.output==0 && ingress(ref.node)) {
            history.insert(ref.node);
            continue;
          }
          eligible=false;break;
        }
      }
      if(!eligible || producer<first || producer+1>=end)continue;
      const I next=producer+1;
      auto &page=nodes[next];
      if(page.projectionSolvePeer>=0 || page.parallel>=0)continue;
      auto ops=page.function.getOps<fc::tensor::ProjectionTaskOp>();
      if(std::distance(ops.begin(),ops.end())!=1)continue;
      auto op=*ops.begin();
      const auto &input=page.inputs[0], &weights=page.inputs[1];
      const I m=input.shape[0], k=input.shape[1], n=weights.shape[1];
      if(op.getKind()!="linear" || input.dtype!="bf16" || weights.dtype!="i8" ||
         m<=0 || m%32 || k%64 || n%64 || !input.external() || !weights.external())continue;
      if(auto replicated=op.getConfig().getAs<mlir::BoolAttr>("replicated_result");
         replicated && replicated.getValue())continue;
      auto policy=fc::target::mxm_execution_policy_from_operation(op.getOperation());
      if(mlir::failed(policy))continue;
      auto strategy=fc::target::plan_mxm_execution_strategy({m,n,k,true,true,true,true},target,*policy);
      if(mlir::failed(strategy) || !strategy->uses_local_dequant())continue;
      for(I binding:history)if(binding>next) {
        auto it=earlyIngress.emplace(binding,next).first;
        it->second=std::min(it->second,next);
      }
      after[next].push_back(pair);
      after[next].push_back(pair+1);
      moved.insert(pair);
      moved.insert(pair+1);
    }
    first=end;
  }
  if(after.empty())return;

  V order;
  std::map<I,V> before;
  for(auto entry:earlyIngress) {
    before[entry.second].push_back(entry.first);
    moved.insert(entry.first);
  }
  for(I i=0;i<count;++i) {
    if(moved.count(i))continue;
    if(auto it=before.find(i);it!=before.end())
      order.insert(order.end(),it->second.begin(),it->second.end());
    const I leader=order.size();
    order.push_back(i);
    auto follow=after.find(i);
    if(follow==after.end())continue;
    nodes[i].projectionConvGroup=leader;
    auto op=*nodes[i].function.getOps<fc::tensor::ProjectionTaskOp>().begin();
    mlir::NamedAttrList config(op.getConfig());
    config.set("projection_conv_overlap",mlir::BoolAttr::get(&planner.context,true));
    op->setAttr("config",config.getDictionary(&planner.context));
    for(I child:follow->second) {
      nodes[child].projectionConvGroup=leader;
      order.push_back(child);
    }
  }
  V remap(count);
  for(I i=0;i<count;++i)remap[order[i]]=i;
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
    reordered.push_back(std::move(node));
  }
  for(auto &output:finals)remapTensor(output);
  nodes=std::move(reordered);
}
