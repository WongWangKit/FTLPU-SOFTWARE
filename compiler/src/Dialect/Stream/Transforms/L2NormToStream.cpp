#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {
mlir::LogicalResult lower_l2_norm(tensor::L2NormTaskOp op, LoweringContext& context)
{
    auto input = get_task_allocation(op.getInputAllocations(), 0);
    auto output = get_task_allocation(op.getResultAllocations(), 0);
    auto feedbackInput = get_task_allocation(op.getScratchAllocations(), 0);
    auto feedbackOutput = get_task_allocation(op.getScratchAllocations(), 1);
    if (mlir::failed(input) || mlir::failed(output)
        || mlir::failed(feedbackInput) || mlir::failed(feedbackOutput))
        return op.emitError("requires complete L2Norm input and result allocations");
    llvm::SmallVector<mlir::Attribute> routes;
    const auto route = [&](llvm::StringRef role, const TaskAllocation& allocation,
                           target::StreamEndpoint source, target::StreamEndpoint destination,
                           target::StreamDirection direction, int64_t begin) {
        const auto slice = llvm::cast<mlir::IntegerAttr>(
            allocation.placement.getAs<mlir::ArrayAttr>("slices")[0]).getInt();
        auto binding = context.allocator.allocate(source, destination, direction,
            slice, begin, begin + 2);
        auto latency = context.target.transport_latency(source, destination, direction, slice);
        if (mlir::failed(binding) || !latency) return false;
        auto& b = context.rewriter;
        routes.push_back(b.getDictionaryAttr({
            b.getNamedAttr("role", b.getStringAttr(role)),
            b.getNamedAttr("source", b.getStringAttr(target::LPUTargetModel::endpoint_name(source))),
            b.getNamedAttr("destination", b.getStringAttr(target::LPUTargetModel::endpoint_name(destination))),
            b.getNamedAttr("direction", b.getStringAttr(target::LPUTargetModel::direction_name(direction))),
            b.getNamedAttr("stream_base", b.getI64IntegerAttr(binding->stream_base)),
            b.getNamedAttr("stream_count", b.getI64IntegerAttr(binding->stream_count)),
            b.getNamedAttr("transport_latency", b.getI64IntegerAttr(*latency)),
        }));
        return true;
    };
    if (!route("transpose_input", *input, target::StreamEndpoint::Mem,
            target::StreamEndpoint::SxmInput, target::StreamDirection::East, context.stage)
        || !route("feedback_input_write", *feedbackInput, target::StreamEndpoint::SxmResult,
            target::StreamEndpoint::Mem, target::StreamDirection::West, context.stage + 2)
        || !route("feedback_input", *feedbackInput, target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput, target::StreamDirection::West, context.stage + 4)
        || !route("normalized", *feedbackOutput, target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem, target::StreamDirection::East, context.stage + 6)
        || !route("restore_input", *feedbackOutput, target::StreamEndpoint::Mem,
            target::StreamEndpoint::SxmInput, target::StreamDirection::East, context.stage + 8)
        || !route("result", *output, target::StreamEndpoint::SxmResult,
            target::StreamEndpoint::Mem, target::StreamDirection::West, context.stage + 10))
        return op.emitError("cannot allocate L2Norm stream routes");
    auto& b = context.rewriter;
    b.setInsertionPoint(op);
    mlir::OperationState state(op.getLoc(), stream::L2NormTaskOp::getOperationName());
    state.addOperands(op.getInput());
    state.addTypes(op.getResult().getType());
    state.addAttributes(op->getAttrs());
    state.addAttribute("routes", b.getArrayAttr(routes));
    auto lowered = llvm::cast<stream::L2NormTaskOp>(b.create(state));
    b.replaceOp(op, lowered.getResult());
    context.stage += 12;
    return mlir::success();
}
} // namespace ftlpu::compiler::stream_lowering
