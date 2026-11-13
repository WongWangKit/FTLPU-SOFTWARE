// Shared logical projection planning. Quantization is an input policy; weight
// residency splits N only. Every native projection retains the complete K
// reduction in the existing MXM accumulator, including block-scaled FP8.
Tensor lowerBlockQuantizedActivation(const std::string &name,const Tensor &input,
                                    bool materializeResult=false) {
  const I tile=target.throughput().mxm_rows;
  auto x=input.external()?join(input,name+".input"):input;
  std::vector<Tensor> quantized,packed;
  // The scalar fallback replicates each logical element across lanes. Bound
  // its live representation by one bank's row capacity. The packed strategy
  // instead streams channels through the accumulator, one token per lane.
  const I groupsPerBatch=std::max<I>(1,target.memory().words_per_bank/(x.shape[0]*tile));
  const I groups=x.shape[1]/tile;
  const auto rowPartition=policies.getAs<mlir::BoolAttr>("ftlpu.vector_bf16_token_halves");
  // Row partitioning assumes rows are tokens. After transposition they are
  // channels, so retain the scalar strategy when that policy is requested.
  const bool tokenLanes=x.shape[0]==tile && !(rowPartition && rowPartition.getValue());
  for(I group=0;group<x.shape[1]/tile;++group) {
    auto part=sliceColumns(x,group*tile,tile);
    if(tokenLanes)part=deltaTranspose(part);
    mlir::OpBuilder b(&context);auto loc=b.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module=mlir::ModuleOp::create(loc);
    auto ty=type(&context,part);
    auto fn=mlir::func::FuncOp::create(loc,"activation_quantization",b.getFunctionType({ty},{ty}));
    module->push_back(fn);b.setInsertionPointToStart(fn.addEntryBlock());
    mlir::OperationState state(loc,fc::kernel::BlockFp8QdqOp::getOperationName());
    state.addOperands(fn.getArgument(0));state.addTypes(ty);
    auto *op=b.create(state);b.create<mlir::func::ReturnOp>(loc,op->getResults());
    if(tokenLanes) {
      // A lane owns one token and streams its K32 channels in order. The
      // existing axis-zero maximum gives 32 independent group scales; all
      // subsequent FP32 arithmetic retains those token lanes.
      fn->setAttr("ftlpu.vector_bf16_spill",b.getBoolAttr(true));
      mlir::IRRewriter rewriter(&context);
      fc::kernel::expandBlockFp8Arithmetic(op,fn.getArgument(0),true,rewriter,true);
    }
    std::string source;llvm::raw_string_ostream os(source);module->print(os);
    auto value=add(name+".qdq"+std::to_string(group),source,{part},{part})[0];
    quantized.push_back(tokenLanes?deltaTranspose(value):value);
    nodes.back().function->setAttr("ftlpu.imem_drain_after",b.getUnitAttr());
    if(!tokenLanes && groups>groupsPerBatch && (I(quantized.size())==groupsPerBatch || group+1==groups)) {
      packed.push_back(join(concat(quantized,1),name+".packed"+std::to_string(packed.size())));
      quantized.clear();
    }
  }
  auto result=concat(packed.empty()?quantized:packed,1);
  // A standalone QDQ can feed consumers with scatter-only native layouts
  // (e.g. attention K and V). Restore feature lanes once, shared by all such
  // consumers, using the ordinary wide-SXM transpose mover. Projection input
  // preparation can consume the token-lane views directly and skips this.
  return tokenLanes && materializeResult?join(result,name+".token_major"):result;
}
Tensor lowerProjection(const std::string &name, const Tensor &input,
                       const Tensor &weight, const Tensor &output,
                       mlir::DictionaryAttr attributes,
                       bool quantizeActivation = false) {
  const I tile = target.throughput().mxm_rows;
  const I m = input.shape[0], k = input.shape[1], n = weight.shape[1];
  const bool fp8 = weight.dtype == "f8E4M3FN";
  if (tile != 32 || m <= 0 || n <= 0 || k <= 0 || k % tile
      || (m > tile && m % tile)
      || (fp8 && (m % tile || n % (2 * tile))))
    throw std::runtime_error("shared projection requires supported aligned dimensions");
  const auto paging = fc::target::GraphWeightPaging::resolve(target, policies);
  // Use the same slot capacity even when prefetch overlap is disabled: the
  // storage ABI and the native page traversal remain identical in both modes.
  const I columnsPerItem = target.memory().hemispheres * tile;
  const fc::target::weight_pages::Geometry geometry{
      paging.rows, target.memory().banks_per_slice, 1};
  const fc::target::weight_pages::ItemLayout layout{
      (k / tile) * target.throughput().tile_rows};
  const I columnsPerPage = columnsPerItem *
      fc::target::weight_pages::itemsPerGroup(geometry, layout);
  if (!columnsPerPage)
    throw std::runtime_error("one full-K projection tile exceeds weight page capacity");
  const auto blockScales = attributes.getAs<mlir::DenseF32ArrayAttr>("rhs_block_scales");
  if (blockScales && blockScales.size() != (n / tile) * (k / tile))
    throw std::runtime_error("projection block scale shape mismatch");
  auto resident = quantizeActivation && input.external() ? join(input, name + ".input") : input;
  // Preserve the existing INT8 multi-token-tile schedule and weight reuse.
  // FP8's currently validated native ingress uses one M32 tile.
  const I rowCapacity = fp8 ? tile
      : target.throughput().mxm_accumulator_blocks * tile;
  std::vector<Tensor> rows;
  for (I first = 0; first < m; first += rowCapacity) {
    const I rowCount = std::min(rowCapacity, m - first);
    auto x = sliceRows(resident, first, rowCount);
    if (quantizeActivation) {
      x=lowerBlockQuantizedActivation(name+".row"+std::to_string(first),x);
    }
    std::vector<Tensor> columns;
    const I pageFamily = nodes.size();
    const auto headPolicy = policies.getAs<mlir::BoolAttr>("ftlpu.l2_head_partition");
    for (I column = 0; column < n; column += columnsPerPage) {
      const I width = std::min(columnsPerPage,
          ((n - column + columnsPerItem - 1) / columnsPerItem) * columnsPerItem);
      auto w = sliceColumns(weight, column, width, true);
      const bool partitionHeads = distributedConvOutput && headPolicy && headPolicy.getValue()
          && n % 256 == 0 && width % 256 == 0;
      const bool decodeHeads = headPolicy && headPolicy.getValue() && decodeHeadWidth > 0
          && rowCount == 1 && n % (2 * decodeHeadWidth) == 0
          && width % (2 * decodeHeadWidth) == 0;
      // Preserve the established paired-head/decode storage ABI. This is a
      // view of immutable weight bytes, not a device MEM-to-MEM transpose.
      const auto logicalColumn = [&](I c) {
        if (decodeHeads)
          return (c / tile % 2) * (width / 2) + (c / (2 * tile)) * tile + c % tile;
        if (partitionHeads) {
          const I word = c / tile, localWord = word / 2;
          return ((localWord / 4) * 2 + word % 2) * 128
              + (localWord % 4) * tile + c % tile;
        }
        return c;
      };
      if (decodeHeads || partitionHeads) {
        V order;
        for (I row = 0; row < k; ++row)
          for (I c = 0; c < width; ++c)
            order.push_back(row * width + logicalColumn(c));
        w = view(w, order, w.shape);
      }
      Tensor result = output; result.shape = {rowCount, width};
      mlir::OpBuilder b(&context); auto loc = b.getUnknownLoc();
      mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
      auto fn = mlir::func::FuncOp::create(loc, "projection", b.getFunctionType(
          {type(&context, x), type(&context, w)}, {type(&context, result)}));
      module->push_back(fn); b.setInsertionPointToStart(fn.addEntryBlock());
      mlir::NamedAttrList config(attributes);
      config.set("m", b.getI64IntegerAttr(rowCount));
      config.set("n", b.getI64IntegerAttr(width));
      config.set("k", b.getI64IntegerAttr(k));
      if (blockScales) {
        llvm::SmallVector<float> pageScales;
        for (I block = 0; block < width / tile; ++block) {
          const I original = (column + logicalColumn(block * tile)) / tile;
          const auto source = blockScales.asArrayRef().slice(original * (k / tile), k / tile);
          pageScales.append(source.begin(), source.end());
        }
        config.set("rhs_block_scales", b.getDenseF32ArrayAttr(pageScales));
      }
      mlir::OperationState state(loc, fc::kernel::MatmulOp::getOperationName());
      state.addOperands(fn.getArguments()); state.addTypes(type(&context, result));
      state.addAttributes(config);
      auto *op = b.create(state);
      b.create<mlir::func::ReturnOp>(loc, op->getResults());
      std::string source; llvm::raw_string_ostream os(source); module->print(os);
      const auto rowName = m <= rowCapacity ? name : name + ".row" + std::to_string(first);
      auto value = add(rowName + ".page" + std::to_string(column / columnsPerPage),
          source, {x, w}, {result})[0];
      nodes.back().projectionPageFamily = pageFamily;
      // Existing INT8 projection barriers and overlap policy are unchanged.
      if (fp8) nodes.back().function->setAttr("ftlpu.imem_drain_after", b.getUnitAttr());
      if (decodeHeads || partitionHeads) {
        V order;
        for (I row = 0; row < rowCount; ++row)
          for (I c = 0; c < width; ++c) {
            if (decodeHeads)
              order.push_back(row * width + (c % (width / 2) / tile) * (2 * tile)
                  + (c / (width / 2)) * tile + c % tile);
            else {
              const I head = c / 128, within = c % 128;
              order.push_back(row * width + (((head / 2) * 4 + within / tile) * 2
                  + head % 2) * tile + within % tile);
            }
          }
        value = view(value, order, value.shape);
      }
      columns.push_back(sliceColumns(value, 0, std::min(width, n - column)));
    }
    rows.push_back(concat(columns, 1));
  }
  return concat(rows);
}
