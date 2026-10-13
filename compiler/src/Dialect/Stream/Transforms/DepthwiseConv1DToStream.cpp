#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {
namespace {

int64_t firstSlice(mlir::ArrayAttr allocations)
{
    const auto allocation = llvm::cast<mlir::DictionaryAttr>(allocations[0]);
    const auto placement =
        allocation.getAs<mlir::DictionaryAttr>("placement");
    return llvm::cast<mlir::IntegerAttr>(
        placement.getAs<mlir::ArrayAttr>("slices")[0]).getInt();
}

} // namespace

mlir::LogicalResult lower_depthwise_conv1d(
    tensor::DepthwiseConv1DTaskOp op, LoweringContext& context)
{
    const int64_t begin = context.stage;
    auto input = context.allocator.allocate(
        target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
        target::StreamDirection::West, firstSlice(op.getInputAllocations()),
        begin, begin + 4, 8);
    auto weight = context.allocator.allocate(
        target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
        target::StreamDirection::West, firstSlice(op.getWeightAllocations()),
        begin, begin + 4, 8);
    auto output = context.allocator.allocate(
        target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
        target::StreamDirection::East, firstSlice(op.getResultAllocations()),
        begin, begin + 4, 2);
    if (mlir::failed(input) || mlir::failed(weight)
        || mlir::failed(output))
        return op.emitError("cannot allocate depthwise Conv1D stream routes");

    context.rewriter.setInsertionPoint(op);
    mlir::OperationState state(
        op.getLoc(), stream::DepthwiseConv1DTaskOp::getOperationName());
    state.addOperands({op.getInput(), op.getWeight()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        context.rewriter.getNamedAttr("kernel_size", op.getKernelSizeAttr()),
        context.rewriter.getNamedAttr("causal", op.getCausalAttr()),
        context.rewriter.getNamedAttr("activation", op.getActivationAttr()),
        context.rewriter.getNamedAttr("input_stream_base",
            context.rewriter.getI64IntegerAttr(input->stream_base)),
        context.rewriter.getNamedAttr("weight_stream_base",
            context.rewriter.getI64IntegerAttr(weight->stream_base)),
        context.rewriter.getNamedAttr("output_stream_base",
            context.rewriter.getI64IntegerAttr(output->stream_base)),
        context.rewriter.getNamedAttr(
            "input_allocations", op.getInputAllocations()),
        context.rewriter.getNamedAttr(
            "weight_allocations", op.getWeightAllocations()),
        context.rewriter.getNamedAttr(
            "scratch_allocations", op.getScratchAllocations()),
        context.rewriter.getNamedAttr(
            "result_allocations", op.getResultAllocations()),
        context.rewriter.getNamedAttr("config", op.getConfig()),
    });
    auto lowered = llvm::cast<stream::DepthwiseConv1DTaskOp>(
        context.rewriter.create(state));
    context.rewriter.replaceOp(op, lowered.getResult());
    context.stage += 4;
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
