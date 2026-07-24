#include "FfnEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute attribute : placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(attribute).getInt());
    return result;
}

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t baseRow, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind)
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
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere)
{
    mlir::OperationState state(location, VxmOp::getOperationName());
    state.addOperands({lhsValue, rhsValue});
    state.addTypes(resultType);
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
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
    return llvm::cast<VxmOp>(rewriter.create(state));
}

llvm::StringRef hemisphere_name(int64_t hemisphere)
{
    return hemisphere == 0 ? "east" : "west";
}

} // namespace ftlpu::compiler::schedule::ffn_detail
