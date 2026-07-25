#include "KernelToTensorLowering.hpp"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_swiglu(kernel::SwigluOp op,
    EastMemoryAllocator& allocator, AllocateValueFn allocate_value,
    mlir::IRRewriter& rewriter)
{
    const auto input = allocate_value(op.getInput(), PlacementKind::Activation);
    const auto gate = allocate_value(op.getGateWeight(), PlacementKind::Weight);
    const auto up = allocate_value(op.getUpWeight(), PlacementKind::Weight);
    const auto result_bytes = get_static_tensor_bytes(op.getResult().getType());
    const auto result = mlir::succeeded(result_bytes)
        ? allocator.allocate(PlacementKind::VxmResult, *result_bytes)
        : mlir::FailureOr<Allocation>(mlir::failure());
    if (mlir::failed(input) || mlir::failed(gate) || mlir::failed(up)
        || mlir::failed(result)) {
        op.emitError("cannot allocate dual-MXM SwiGLU storage");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op);
    mlir::OperationState state(op.getLoc(), tensor::SwigluOp::getOperationName());
    state.addOperands({op.getInput(), op.getGateWeight(), op.getUpWeight()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        rewriter.getNamedAttr("m", op.getMAttr()),
        rewriter.getNamedAttr("n", op.getNAttr()),
        rewriter.getNamedAttr("k", op.getKAttr()),
        rewriter.getNamedAttr("gate_scale", op.getGateScaleAttr()),
        rewriter.getNamedAttr("up_scale", op.getUpScaleAttr()),
        rewriter.getNamedAttr("output_scale", op.getOutputScaleAttr()),
        rewriter.getNamedAttr("output_zero_point", op.getOutputZeroPointAttr()),
        rewriter.getNamedAttr("input_address", make_address_attr(rewriter, *input)),
        rewriter.getNamedAttr("input_placement", make_placement_attr(rewriter, *input)),
        rewriter.getNamedAttr("gate_weight_address", make_address_attr(rewriter, *gate)),
        rewriter.getNamedAttr("gate_weight_placement", make_placement_attr(rewriter, *gate)),
        rewriter.getNamedAttr("up_weight_address", make_address_attr(rewriter, *up)),
        rewriter.getNamedAttr("up_weight_placement", make_placement_attr(rewriter, *up)),
        rewriter.getNamedAttr("result_address", make_address_attr(rewriter, *result)),
        rewriter.getNamedAttr("result_placement", make_placement_attr(rewriter, *result)),
        rewriter.getNamedAttr("input_bytes", rewriter.getI64IntegerAttr(input->bytes)),
        rewriter.getNamedAttr("gate_weight_bytes", rewriter.getI64IntegerAttr(gate->bytes)),
        rewriter.getNamedAttr("up_weight_bytes", rewriter.getI64IntegerAttr(up->bytes)),
        rewriter.getNamedAttr("result_bytes", rewriter.getI64IntegerAttr(result->bytes)),
    });
    auto lowered = llvm::cast<tensor::SwigluOp>(rewriter.create(state));
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
