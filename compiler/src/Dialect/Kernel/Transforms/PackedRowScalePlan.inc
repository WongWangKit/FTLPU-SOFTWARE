// Included inside the graph memory planner. A packed matrix times an FP32
// row coefficient can read piecewise planar producer homes directly. Each
// hemisphere owns different rows; neither operand is replicated or repacked.
bool planNativePackedRowScale(Node &node) {
  if(!node.function->hasAttr("ftlpu.vector_packed_update") || node.inputs.size()!=2 ||
     node.outputs.size()!=1 || node.inputs[0].external() || node.inputs[1].external()) return false;
  const auto fp32=(*node.module)->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_fp32_ingress");
  if(!fp32 || !fp32.getValue()) return false;
  const auto &x=node.inputs[0], &coefficient=node.inputs[1], &result=node.outputs[0];
  if(x.shape.size()!=2 || x.shape[0]%2 || x.shape[1]%32 || x.dtype!="f32" ||
     coefficient.shape!=V{x.shape[0],1} || coefficient.dtype!="f32" ||
     result.shape!=x.shape || result.dtype!="bf16" || node.inputSlots[0].size()!=1) return false;
  auto op=llvm::dyn_cast<fc::kernel::VectorAluOp>(node.inputSlots[0][0].op);
  if(!op || op.getKind()!="multiply" || op.getReduce() || op.getUseImmediate() ||
     op.getLhs()!=node.function.getArgument(0) || op.getRhs()!=node.function.getArgument(1)) return false;
  const I rows=x.shape[0], blocks=x.shape[1]/32;
  llvm::SmallVector<mlir::Attribute> words, factors;
  V rowSides,rowIndices; I localRows[2]={0,0};
  std::vector<mlir::DictionaryAttr> homes;
  for(I block=0;block<blocks;++block) for(I row=0;row<rows;++row) {
    const auto ref=x.refs[row*x.shape[1]+block*32];
    auto home=cache.at({ref.node,ref.output});
    const auto &shape=planner.nodes[ref.node].outputs[ref.output].shape;
    if(shape.size()!=2 || shape[0]%2 || shape[1]%32 || ref.index%32 ||
       slices(home).size()!=4 || home.get("storage_shape")) return false;
    for(I lane=0;lane<32;++lane) {
      const auto r=x.refs[row*x.shape[1]+block*32+lane];
      if(r.node!=ref.node || r.output!=ref.output || r.index!=ref.index+lane) return false;
    }
    const I sourceRow=ref.index/shape[1], column=ref.index%shape[1]/32;
    const auto kind=text(home,"kind");
    I side=0, offset=0;
    if(kind=="token_partition_planar" || kind=="token_partition_east_first_planar") {
      if(home.get("partition_row_sides")) return false;
      const I half=shape[0]/2;
      side=(sourceRow<half) != (kind=="token_partition_east_first_planar");
      offset=column*half+sourceRow%half;
    } else return false; // Mirrored packed updates keep the established planar legalization.
    if(block==0) { rowSides.push_back(1-side);rowIndices.push_back(localRows[side]++); }
    else if(rowSides[row]!=1-side) return false;
    const auto factor=coefficient.refs[row];
    auto scalarHome=cache.at({factor.node,factor.output});
    if(text(scalarHome,"kind")!="fp32_scalar_broadcast_planar" ||
       text(scalarHome,"hemisphere")!="both" || slices(scalarHome).size()!=4 ||
       scalarHome.get("storage_shape") || overlap(home,scalarHome)) return false;
    // All source queues are reserved through the output drain, including
    // producer pieces whose SRAM rows happen to be disjoint.
    for(auto p:{home,scalarHome})
      if(std::find(homes.begin(),homes.end(),p)==homes.end()) homes.push_back(p);
    home=s.change(home,J{{"kind","fp32_quad_planar"},{"hemisphere",side?"west":"east"}});
    scalarHome=s.change(scalarHome,J{{"hemisphere",side?"west":"east"}});
    const auto descriptor=[&](mlir::DictionaryAttr p,I address) {
      return s.b.getDictionaryAttr({s.b.getNamedAttr("placement",p),
          s.b.getNamedAttr("address",s.b.getI64IntegerAttr(address))});
    };
    words.push_back(descriptor(home,field(home,"base_row")+offset));
    factors.push_back(descriptor(scalarHome,field(scalarHome,"base_row")+factor.index));
  }
  if(localRows[0]!=rows/2 || localRows[1]!=rows/2) return false;
  node.function->setAttr("native_row_scale_words",s.b.getArrayAttr(words));
  node.function->setAttr("native_row_scale_coefficients",s.b.getArrayAttr(factors));
  node.out[0]=s.change(node.out[0],J{{"kind","token_partition_planar"},{"hemisphere","both"},
      {"partition_row_sides",s.array(rowSides)},{"partition_row_indices",s.array(rowIndices)},
      {"partition_storage_rows",rows/2}});
  node.stagedInputs[0]=node.stagedInputs[1]=true;
  node.workspaces.insert(node.workspaces.end(),homes.begin(),homes.end());
  return true;
}
