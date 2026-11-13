// Included inside the whole-graph memory planner. These decisions are made
// from logical tensor references and allocation contracts, before ICU/FU
// emission. Consumers declare support; unsupported views keep a named copy.
mlir::DictionaryAttr withPlanarRead(mlir::DictionaryAttr home,
                                  const fc::analysis::PlanarReadPlan &plan) {
  mlir::NamedAttrList attrs(home);
  attrs.set("planar_read",fc::analysis::encodePlanarRead(s.b,plan));
  return attrs.getDictionary(&s.context);
}

bool acceptsPlanarRead(const Node &node, I operand) {
  if(operand<0 || operand>=I(node.inputSlots.size()) || node.inputSlots[operand].empty())return false;
  const auto update=node.function->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_packed_update");
  if(update && update.getValue())return true;
  // The fixed-port packed chain already emits READ3D from an affine planar
  // descriptor. A slice or repeated coefficient row need not be copied into
  // a new allocation simply because it uses the ordinary pointwise path.
  if(node.function->hasAttr("ftlpu.vector_bf16_spill") &&
     node.function->hasAttr("ftlpu.vector_independent_ports") &&
     llvm::all_of(node.inputSlots[operand],[](const auto &slot) {
       auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(slot.op);
       if(!op || op.getReduce())return false;
       const auto k=op.getKind();
       return k=="add" || k=="subtract" || k=="multiply" || k=="max" ||
           k=="negate" || k=="exp" || k=="reciprocal" || k=="rsqrt" || k=="log" || k=="sigmoid";
     }))return true;
  return operand<2 && node.inputSlots[operand].size()==1 &&
      llvm::isa<fc::kernel::VectorAluOp>(node.inputSlots[operand][0].op) &&
      llvm::cast<fc::kernel::VectorAluOp>(node.inputSlots[operand][0].op).getKind()=="head_matmul" &&
      node.inputs[operand].shape.size()==2 && node.inputs[operand].shape[0]==64;
}

std::optional<fc::analysis::PlanarReadPlan> analyzePlanarView(
    const Tensor &value,mlir::DictionaryAttr home,std::string &reason) {
  const auto fail=[&](const char *why)->std::optional<fc::analysis::PlanarReadPlan> {
    reason=why;return std::nullopt;
  };
  if(value.refs.empty() || value.shape.size()!=2)return fail("no_matrix_view");
  const auto first=value.refs.front();const Key source{first.node,first.output};
  const auto &producer=planner.nodes[source.first].outputs[source.second];
  const auto kind=text(home,"kind");
  const bool scalarWords=kind=="fp16_scalar_broadcast_planar" || kind=="fp32_scalar_broadcast_planar";
  const bool packedWords=kind=="fp16_pair_planar" || kind=="fp32_quad_planar";
  if((!scalarWords && !packedWords) || text(home,"hemisphere")!="both")return fail("ownership_or_storage_format");
  if(slices(home).size()!=size_t(value.bytes()) || value.dtype!=producer.dtype)
    return fail("byte_plane_or_precision_change");
  if(producer.shape.size()!=2 || home.get("storage_shape") || home.get("storage_offset"))
    return fail("physical_shape_view_requires_legalization");
  const I lanes=scalarWords?1:32;
  if(value.shape[1]%lanes || producer.shape[1]%lanes)return fail("partial_word_or_lane_change");
  const auto sourceView=fc::analysis::decodePlanarRead(home);
  V offsets;
  const I rows=value.shape[0],columns=value.shape[1]/lanes;
  for(I c=0;c<columns;++c)for(I r=0;r<rows;++r) {
    const auto base=value.refs[r*value.shape[1]+c*lanes];
    if(base.node!=source.first || base.output!=source.second)return fail("multiple_allocations");
    if(base.index<0 || base.index+lanes>producer.size() || base.index%producer.shape[1]%lanes)
      return fail("partial_word_or_lane_change");
    for(I lane=0;lane<lanes;++lane) {
      const auto ref=value.refs[r*value.shape[1]+c*lanes+lane];
      if(ref.node!=base.node || ref.output!=base.output || ref.index!=base.index+lane)
        return fail("lane_permutation");
    }
    const I sr=base.index/producer.shape[1],sc=base.index%producer.shape[1]/lanes;
    offsets.push_back(sourceView?sourceView->address(sr,sc):sc*producer.shape[0]+sr);
  }
  auto access=fc::analysis::planPlanarRead(rows,columns,offsets);
  if(!access)return fail("not_one_affine_3d_domain");
  return access;
}

bool planSegmentedMxmRhs(I ni,I operand) {
  auto &node=planner.nodes[ni];
  if(operand!=1 || node.inputSlots[operand].size()!=1)return false;
  auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(node.inputSlots[operand][0].op);
  if(!op || op.getKind()!="head_matmul" || op.getLhs().getType().getDimSize(1)<=32)return false;
  const auto precision=op->getAttrOfType<mlir::StringAttr>("mxm_precision");
  const I k=op.getLhs().getType().getDimSize(1);
  const auto &value=node.inputs[operand];
  if(!precision || precision.getValue()!="bf16" || value.dtype!="bf16" ||
      value.shape!=V{2*k,32} || I(value.refs.size())!=value.size())return false;
  llvm::SmallVector<mlir::Attribute> segments;
  for(I row=0;row<2*k;) {
    const auto first=value.refs[row*32];const Key source{first.node,first.output};
    auto home=cache.at(source);
    const auto &producer=planner.nodes[source.first].outputs[source.second];
    if(producer.dtype!="bf16" || producer.shape.size()!=2 || producer.shape[1]!=32 ||
        text(home,"kind")!="fp16_pair_planar" || text(home,"hemisphere")!="both" ||
        slices(home).size()!=2 || home.get("storage_shape") || home.get("storage_offset") ||
        home.get("planar_read") || first.index%32 || first.index<0)return false;
    for(I other=0;other<I(node.in.size());++other)
      if(other!=operand && overlap(home,node.in[other]))return false;
    for(auto p:node.out)if(overlap(home,p))return false;
    for(auto p:node.workspaces)if(overlap(home,p))return false;
    // Do not cross a logical hemisphere boundary, even if it happens to
    // adjoin the same allocation. Each SXM consumes its own complete K32s.
    const I limit=(row/k+1)*k;
    I count=0;
    while(row+count<limit) {
      bool contiguous=true;
      for(I lane=0;lane<32;++lane) {
        const auto ref=value.refs[(row+count)*32+lane];
        if(ref.node!=source.first || ref.output!=source.second || ref.index!=first.index+count*32+lane) {
          contiguous=false;break;
        }
      }
      if(!contiguous)break;
      ++count;
    }
    if(!count || row%32 || count%32 || first.index+count*32>producer.size())return false;
    segments.push_back(s.change(home,J{{"base_row",field(home,"base_row")+first.index/32},
        {"logical_begin",row},{"word_count",count},{"instruction_count",count}}));
    row+=count;
  }
  mlir::NamedAttrList attrs(llvm::cast<mlir::DictionaryAttr>(segments.front()));
  attrs.set("mxm_rhs_segments",s.b.getArrayAttr(segments));
  auto placement=attrs.getDictionary(&s.context);
  node.in[operand]=placement;
  for(auto &slot:node.inputSlots[operand])slot.set(placement);
  node.stagedInputs[operand]=true;
  node.accessPlans.push_back(J{{"input",operand},{"decision","segmented_native_mxm_read"},
      {"segments",I(segments.size())},{"logical_words",2*k}});
  return true;
}

bool planPlanarInput(I ni,I operand) {
  auto &node=planner.nodes[ni];
  if(!acceptsPlanarRead(node,operand))return false;
  const auto &value=node.inputs[operand];
  const auto fail=[&](const std::string &reason) {
    node.accessPlans.push_back(J{{"input",operand},{"decision","materialize_or_legacy"},{"reason",reason}});
    return false;
  };
  if(value.refs.empty())return fail("no_matrix_view");
  const auto first=value.refs.front();const Key source{first.node,first.output};
  const auto home=cache.at(source),expected=node.in[operand];
  if(text(home,"kind")!=text(expected,"kind") || slices(home).size()!=slices(expected).size())
    return fail("consumer_storage_contract");
  std::string reason;
  auto access=analyzePlanarView(value,home,reason);
  if(!access)return fail(reason);
  // Two independent fixed-port reads cannot occupy the same MEM ICU, even
  // if their source addresses differ. Native matmul's LHS/RHS overlap too.
  const auto captured=[&](I arg) {
    const auto args=node.function->getAttrOfType<mlir::ArrayAttr>("ftlpu.vector_capture_arguments");
    return args && llvm::any_of(args,[&](auto a){return llvm::cast<mlir::IntegerAttr>(a).getInt()==arg;});
  };
  for(I other=0;other<I(node.in.size());++other)
    if(other!=operand && !captured(operand) && !captured(other) && overlap(home,node.in[other]))return fail("mem_icu_queue_conflict");
  auto placement=withPlanarRead(home,*access);
  node.in[operand]=placement;
  for(auto &slot:node.inputSlots[operand])slot.set(placement);
  node.stagedInputs[operand]=true;
  node.accessPlans.push_back(J{{"input",operand},{"decision","address_view"},
      {"source_node",source.first},{"source_output",source.second},
      {"logical_words",access->rows*access->columns},{"row_chunk",access->rowChunk},
      {"row_stride",access->rowStride},{"group_stride",access->groupStride},
      {"column_stride",access->columnStride}});
  return true;
}

bool planarJoinConsumers(I ni) {
  auto lastUse=last.find({ni,0});
  if(lastUse==last.end())return false;
  for(I user=ni+1;user<=lastUse->second;++user) {
    if(user>=I(planner.nodes.size()))return false;
    const auto &consumer=planner.nodes[user];
    for(I i=0;i<I(consumer.inputs.size());++i) {
      const auto &input=consumer.inputs[i];
      if(llvm::none_of(input.refs,[&](const Ref &r){return r.node==ni;}))continue;
      if(!consumer.module || !acceptsPlanarRead(consumer,i))return false;
    }
  }
  return true;
}

bool planDirectPlanarJoin(I ni,const std::vector<mlir::DictionaryAttr>& otherQueues) {
  auto &node=planner.nodes[ni];const auto &value=node.inputs[0];
  if(value.refs.empty() || !planarJoinConsumers(ni))return false;
  const auto first=value.refs.front();const Key source{first.node,first.output},result{ni,0};
  const auto home=cache.at(source);
  if(text(home,"kind")!=(value.dtype=="bf16"?"fp16_pair_planar":"fp32_quad_planar")) {
    node.viewReason="join_requires_packed_words";return false;
  }
  auto access=analyzePlanarView(value,home,node.viewReason);
  if(!access)return false;
  if(llvm::any_of(otherQueues,[&](auto p){return overlap(p,home);})) {
    node.viewReason="mem_icu_queue_conflict";return false;
  }
  // Retaining the original footprint also retains all its aliases. Dynamic
  // allocations see live[result]; fixed native workspaces need this check.
  const I begin=field(home,"base_row"),end=begin+field(home,"instruction_count");
  for(auto side:{"east","west"})for(I slice:slices(home)) {
    const auto &ranges=field(home,"bank")==0?nativeBank0[{side,slice}]:nativeTimed[{side,slice}];
    for(const auto &range:ranges)
      if(range.node>ni && range.node<=last.at(result) && begin<range.end && range.begin<end) {
        node.viewReason="retained_view_lifetime_conflict";return false;
      }
  }
  cache[result]=live[result]=withPlanarRead(home,*access);
  node.addressOnlyView=true;node.viewReason="direct_planar_address_view";
  return true;
}

// A join is a request for an operand view, not an unconditional expanded
// buffer. For matrix consumers, repeated logical rows can share storage.
bool planCompactPlanarJoin(I ni,const std::vector<mlir::DictionaryAttr>& otherQueues) {
  auto &node=planner.nodes[ni];const auto &value=node.inputs[0];
  node.viewReason="consumer_requires_materialized_layout";
  if(!node.matmulLhsJoin || value.dtype!="bf16" || value.shape!=V{64,32} ||
     value.refs.size()!=2048)return false;
  // The compact view is exposed only to consumers with the access-plan
  // contract. Unsupported/export consumers must retain normal storage.
  if(!planarJoinConsumers(ni))return false;
  std::map<std::vector<std::tuple<I,I,I>>,I> unique;
  V selected,offsets;
  for(I row=0;row<64;++row) {
    std::vector<std::tuple<I,I,I>> key;
    for(I lane=0;lane<32;++lane) {
      const auto ref=value.refs[row*32+lane];key.emplace_back(ref.node,ref.output,ref.index);
    }
    auto [entry,inserted]=unique.emplace(key,I(unique.size()));
    if(inserted)for(I lane=0;lane<32;++lane)selected.push_back(row*32+lane);
    offsets.push_back(entry->second);
  }
  if(unique.size()==64){node.viewReason="no_repeated_rows";return false;}
  const auto access=fc::analysis::planPlanarRead(64,1,offsets);
  if(!access){node.viewReason="not_one_affine_3d_domain";return false;}
  node.materializedView=view(value,selected,{I(unique.size()),32});
  auto home=allocate({ni,0},*node.materializedView,"both",1,{},otherQueues);
  cache[{ni,0}]=live[{ni,0}]=withPlanarRead(home,*access);
  node.viewReason="compact_unique_rows_with_affine_read";
  return true;
}
