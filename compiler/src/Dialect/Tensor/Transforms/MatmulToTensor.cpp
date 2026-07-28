#include "KernelToTensorLowering.hpp"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_matmul(kernel::MatmulOp op,
    FunctionMemoryPlanner& planner,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter)
{
    const auto lhs = allocate_value(op.getLhs(), PlacementKind::Activation);
    const auto rhs = allocate_value(op.getRhs(), PlacementKind::Weight);
    const auto result =
        planner.allocate(op.getResult(), PlacementKind::Result);
    if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(result)) {
        op.emitError("cannot allocate static tensor storage in the east MEM hemisphere");
        return mlir::failure();
    }

    const mlir::Value old_result = op.getResult();
    rewriter.setInsertionPoint(op);
    auto lowered = rewriter.create<tensor::MatmulOp>(op.getLoc(), op.getLhs(), op.getRhs(),
        op.getResult().getType(), op.getM(), op.getN(), op.getK(), op.getUnitAttr(),
        op.getRhsScaleAttr(),
        make_address_attr(rewriter, *lhs), make_placement_attr(rewriter, *lhs),
        make_address_attr(rewriter, *rhs), make_placement_attr(rewriter, *rhs),
        make_address_attr(rewriter, *result), make_placement_attr(rewriter, *result),
        lhs->bytes, rhs->bytes, result->bytes);
    planner.replace_value(old_result, lowered.getResult());
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
