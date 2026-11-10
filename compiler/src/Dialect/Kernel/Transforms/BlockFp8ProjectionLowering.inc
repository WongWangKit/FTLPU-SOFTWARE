// Members of the connected graph planner. Tile selection and the K-order
// reduction are compiler work; checkpoint byte tensors remain encoded.
Tensor lowerBlockFp8Projection(const std::string &name, const Tensor &x,
                              const Tensor &weights, const Tensor &scales) {
  if (!weights.external() || !scales.external())
    throw std::runtime_error("block FP8 projection requires constant weights and scales");
  for (auto code : scales.data)
    if (code == 255) throw std::runtime_error("UE8M0 NaN scale is invalid");
  const I m = x.shape[0], k = x.shape[1], n = weights.shape[0];
  const I groups = k / 32;
  const auto typeText = [&](const Tensor &t) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    type(&context, t).print(stream);
    return text;
  };
  // Keep a compact ingress image instead of uploading one scalar-broadcast
  // image per K group. Consume each QDQ tile across N before moving on, so
  // only the output accumulators survive the K loop.
  const auto residentX = x.external() ? join(x, name + ".input") : x;
  std::vector<Tensor> tokenTiles;
  for (I row = 0; row < m; row += 32) {
    std::vector<Tensor> sums(n / 64);
    for (I group = 0; group < groups; ++group) {
      auto tile = sliceColumns(sliceRows(residentX, row, 32), group * 32, 32);
      const auto ty = typeText(tile);
      const auto source = "module { func.func @stage(%x: " + ty + ") -> " + ty +
          " { %y = ftlpu.kernel.block_fp8_qdq %x : (" + ty + ") -> " + ty +
          " return %y : " + ty + " } }";
      auto quantized = add(name + ".row" + std::to_string(row) + ".qdq" +
          std::to_string(group), source, {tile}, {tile})[0];
      nodes.back().function->setAttr("ftlpu.imem_drain_after", mlir::UnitAttr::get(&context));
      for (I column = 0; column < n; column += 64) {
        auto &sum = sums[column / 64];
        auto w = deltaTranspose(sliceColumns(sliceRows(weights, column, 64), group * 32, 32));
        Tensor partial;
        partial.shape = {32, 64};
        partial.dtype = "f32";
        const auto a = typeText(quantized), b = typeText(w), c = typeText(partial);
        const auto scaleText = [&](I side) {
          const auto code = scales.data[(column / 32 + side) * groups + group];
          std::ostringstream text;
          text << std::scientific << std::setprecision(9) << std::ldexp(1.0f, int(code) - 127);
          return text.str();
        };
        const auto source = "module { func.func @stage(%x: " + a + ", %w: " + b + ") -> " + c +
            " { %y = \"ftlpu.kernel.matmul\"(%x, %w) {m=32:i64,k=32:i64,n=64:i64,unit=\"MXM\","
            "rhs_scale=" + scaleText(0) + ":f32,rhs_scale_west=" + scaleText(1) +
            ":f32} : (" + a + ", " + b + ") -> " + c + " return %y : " + c + " } }";
        const auto prefix = name + ".row" + std::to_string(row) + ".n" +
            std::to_string(column) + ".k" + std::to_string(group * 32);
        auto product = add(prefix, source, {quantized, w}, {partial})[0];
        sum = group ? deltaAlu(prefix + ".accumulate", sum, product, "add", false, "f32", true, true) : product;
        nodes.back().function->setAttr("ftlpu.imem_drain_after", mlir::UnitAttr::get(&context));
      }
    }
    std::vector<Tensor> outputTiles;
    for (I column = 0; column < n; column += 64) {
      auto rounded = deltaAlu(name + ".round.n" + std::to_string(column), sums[column / 64],
          deltaConstant({1, 1}, {1.f}), "multiply", false, "bf16", true, true);
      outputTiles.push_back(std::move(rounded));
    }
    tokenTiles.push_back(concat(outputTiles, 1));
  }
  return concat(tokenTiles);
}
