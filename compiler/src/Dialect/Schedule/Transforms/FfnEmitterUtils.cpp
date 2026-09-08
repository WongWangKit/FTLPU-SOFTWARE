#include "FfnEmitterUtils.hpp"

#include <algorithm>
#include <optional>

namespace ftlpu::compiler::schedule::ffn_detail {

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute attribute : placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(attribute).getInt());
    return result;
}

int64_t get_base_row(mlir::DictionaryAttr placement)
{
    return placement.getAs<mlir::IntegerAttr>("base_row").getInt();
}

mlir::FailureOr<PagedWeightPagePlacement> resolve_page_placement(
    mlir::DictionaryAttr placement, int64_t page)
{
    if (!placement || page < 0) return mlir::failure();
    const auto integerOr = [&](llvm::StringRef name, int64_t fallback) {
        const auto value = placement.getAs<mlir::IntegerAttr>(name);
        return value ? value.getInt() : fallback;
    };
    const auto arrayValue = [&](llvm::StringRef name)
        -> std::optional<int64_t> {
        const auto values = placement.getAs<mlir::ArrayAttr>(name);
        if (!values || page >= static_cast<int64_t>(values.size()))
            return std::nullopt;
        const auto value = llvm::dyn_cast<mlir::IntegerAttr>(values[page]);
        if (!value) return std::nullopt;
        return value.getInt();
    };
    if (placement.get("page_banks")) {
        const auto bank = arrayValue("page_banks");
        const auto groupBase = arrayValue("page_slice_group_bases");
        const auto groupCount = arrayValue("page_slice_group_counts");
        const auto baseRow = arrayValue("page_base_rows");
        const auto rowCount = arrayValue("page_row_counts");
        if (!bank || !groupBase || !groupCount || !baseRow || !rowCount
            || *groupCount <= 0 || *rowCount <= 0)
            return mlir::failure();
        return PagedWeightPagePlacement {
            *bank, *groupBase, *groupCount, *baseRow, *rowCount};
    }
    const int64_t bankCount = std::max<int64_t>(
        1, integerOr("page_bank_count", 1));
    return PagedWeightPagePlacement {
        (integerOr("bank", 0) + page) % bankCount,
        integerOr("page_role_group_base", 0),
        std::max<int64_t>(1,
            integerOr("page_role_group_count", 1)),
        0,
        std::max<int64_t>(1,
            integerOr("instruction_count", 1)),
    };
}

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t baseRow, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind)
{
    return schedule_placement(builder, slices, baseRow, count, stride,
        hemisphere, kind, 0);
}

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t baseRow, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind,
    int64_t bank)
{
    llvm::SmallVector<mlir::Attribute> sliceAttrs;
    for (int64_t slice : slices)
        sliceAttrs.push_back(builder.getI64IntegerAttr(slice));
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr(
            "hemisphere", builder.getStringAttr(hemisphere)),
        builder.getNamedAttr("slices", builder.getArrayAttr(sliceAttrs)),
        builder.getNamedAttr("base_row", builder.getI64IntegerAttr(baseRow)),
        builder.getNamedAttr("bank", builder.getI64IntegerAttr(bank)),
        builder.getNamedAttr(
            "instruction_count", builder.getI64IntegerAttr(count)),
        builder.getNamedAttr(
            "address_stride", builder.getI64IntegerAttr(stride)),
    });
}

VxmOp create_vxm(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Value lhsValue, mlir::Value rhsValue, mlir::Type resultType,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere,
    int64_t scaleBinding, bool accumulatorReset,
    bool accumulatorWrite, bool accumulatorEmit,
    bool localScalarWrite, int64_t chainDepth)
{
    mlir::OperationState state(location, VxmOp::getOperationName());
    state.addOperands({lhsValue, rhsValue});
    state.addTypes(resultType);
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("chain_depth",
            rewriter.getI64IntegerAttr(chainDepth)),
        rewriter.getNamedAttr("lhs_kind", rewriter.getStringAttr(lhsKind)),
        rewriter.getNamedAttr(
            "lhs_index", rewriter.getI64IntegerAttr(lhsIndex)),
        rewriter.getNamedAttr(
            "lhs_immediate", rewriter.getF32FloatAttr(lhsImmediate)),
        rewriter.getNamedAttr("rhs_kind", rewriter.getStringAttr(rhsKind)),
        rewriter.getNamedAttr(
            "rhs_index", rewriter.getI64IntegerAttr(rhsIndex)),
        rewriter.getNamedAttr(
            "rhs_immediate", rewriter.getF32FloatAttr(rhsImmediate)),
        rewriter.getNamedAttr(
            "cast_target", rewriter.getStringAttr(castTarget)),
        rewriter.getNamedAttr(
            "output_stream", rewriter.getI64IntegerAttr(outputStream)),
        rewriter.getNamedAttr(
            "repeat_count", rewriter.getI64IntegerAttr(repeatCount)),
        rewriter.getNamedAttr(
            "repeat_interval", rewriter.getI64IntegerAttr(repeatInterval)),
        rewriter.getNamedAttr(
            "input_hemisphere", rewriter.getStringAttr(inputHemisphere)),
        rewriter.getNamedAttr(
            "output_hemisphere", rewriter.getStringAttr(outputHemisphere)),
    });
    if (scaleBinding >= 0)
        state.addAttribute(
            "scale_binding", rewriter.getI64IntegerAttr(scaleBinding));
    if (accumulatorReset)
        state.addAttribute("accumulator_reset", rewriter.getBoolAttr(true));
    if (accumulatorWrite)
        state.addAttribute("accumulator_write", rewriter.getBoolAttr(true));
    if (!accumulatorEmit)
        state.addAttribute("accumulator_emit", rewriter.getBoolAttr(false));
    if (localScalarWrite)
        state.addAttribute("local_scalar_write", rewriter.getBoolAttr(true));
    return llvm::cast<VxmOp>(rewriter.create(state));
}

llvm::StringRef hemisphere_name(int64_t hemisphere)
{
    return hemisphere == 0 ? "east" : "west";
}

} // namespace ftlpu::compiler::schedule::ffn_detail
