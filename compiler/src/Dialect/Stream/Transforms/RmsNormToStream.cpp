#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {

mlir::LogicalResult lower_rms_norm(
    tensor::RmsNormTaskOp op, LoweringContext& context)
{
    auto input = get_task_allocation(op.getInputAllocations(), 0);
    auto weight = get_task_allocation(op.getWeightAllocations(), 0);
    auto square = get_task_allocation(op.getScratchAllocations(), 0);
    auto factor = get_task_allocation(op.getScratchAllocations(), 1);
    const auto strategy =
        op.getConfig().getAs<mlir::StringAttr>("strategy");
    const bool feedback =
        strategy && strategy.getValue() == "vxm_feedback";
    auto normalized = feedback
        ? get_task_allocation(op.getScratchAllocations(), 2)
        : mlir::FailureOr<TaskAllocation>(mlir::failure());
    auto result = get_task_allocation(op.getResultAllocations(), 0);
    if (mlir::failed(input) || mlir::failed(weight)
        || mlir::failed(square) || mlir::failed(factor)
        || (feedback && mlir::failed(normalized))
        || mlir::failed(result)) {
        op.emitError("requires complete RMSNorm physical allocations");
        return mlir::failure();
    }

    const auto firstSlice = [](mlir::DictionaryAttr placement) {
        return llvm::cast<mlir::IntegerAttr>(
            placement.getAs<mlir::ArrayAttr>("slices")[0]).getInt();
    };
    llvm::SmallVector<mlir::Attribute> routes;
    int64_t stage = context.stage;
    const auto addRoute =
        [&](llvm::StringRef phase, llvm::StringRef role,
            target::StreamEndpoint source,
            target::StreamEndpoint destination,
            target::StreamDirection direction,
            const TaskAllocation& allocation,
            int64_t begin, int64_t end) -> bool {
        const int64_t slice = firstSlice(allocation.placement);
        auto binding = context.allocator.allocate(source, destination,
            direction, slice, begin, end);
        auto latency = context.target.transport_latency(
            source, destination, direction, slice);
        if (mlir::failed(binding) || !latency) return false;
        routes.push_back(context.rewriter.getDictionaryAttr({
            context.rewriter.getNamedAttr(
                "phase", context.rewriter.getStringAttr(phase)),
            context.rewriter.getNamedAttr(
                "role", context.rewriter.getStringAttr(role)),
            context.rewriter.getNamedAttr("source",
                context.rewriter.getStringAttr(
                    target::LPUTargetModel::endpoint_name(source))),
            context.rewriter.getNamedAttr("destination",
                context.rewriter.getStringAttr(
                    target::LPUTargetModel::endpoint_name(destination))),
            context.rewriter.getNamedAttr("direction",
                context.rewriter.getStringAttr(
                    target::LPUTargetModel::direction_name(direction))),
            context.rewriter.getNamedAttr("stream_base",
                context.rewriter.getI64IntegerAttr(binding->stream_base)),
            context.rewriter.getNamedAttr("stream_count",
                context.rewriter.getI64IntegerAttr(binding->stream_count)),
            context.rewriter.getNamedAttr("register_id",
                context.rewriter.getI64IntegerAttr(binding->register_id)),
            context.rewriter.getNamedAttr("producer_stage",
                context.rewriter.getI64IntegerAttr(begin)),
            context.rewriter.getNamedAttr("consumer_stage",
                context.rewriter.getI64IntegerAttr(end)),
            context.rewriter.getNamedAttr("transport_latency",
                context.rewriter.getI64IntegerAttr(*latency)),
        }));
        return true;
    };

    bool valid = true;
    int64_t stageCount = 14;
    if (feedback) {
        valid &= addRoute("transpose_input", "input",
            target::StreamEndpoint::Mem, target::StreamEndpoint::SxmInput,
            target::StreamDirection::East, *input, stage, stage + 2);
        valid &= addRoute("transpose_input", "feedback_input",
            target::StreamEndpoint::SxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::West, *square, stage + 2, stage + 4);
        valid &= addRoute("transpose_weight", "weight",
            target::StreamEndpoint::Mem, target::StreamEndpoint::SxmInput,
            target::StreamDirection::East, *weight, stage + 4, stage + 6);
        valid &= addRoute("transpose_weight", "feedback_weight",
            target::StreamEndpoint::SxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::West, *factor, stage + 6, stage + 8);
        valid &= addRoute("feedback", "input",
            target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, *square, stage + 8, stage + 10);
        valid &= addRoute("feedback", "weight",
            target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, *factor, stage + 8, stage + 10);
        valid &= addRoute("feedback", "normalized",
            target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::East, *normalized,
            stage + 10, stage + 12);
        valid &= addRoute("restore_layout", "normalized",
            target::StreamEndpoint::Mem, target::StreamEndpoint::SxmInput,
            target::StreamDirection::East, *normalized,
            stage + 12, stage + 14);
        valid &= addRoute("restore_layout", "result",
            target::StreamEndpoint::SxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::West, *result,
            stage + 14, stage + 16);
        stageCount = 16;
    } else {
        valid &= addRoute("square", "input",
            target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, *input, stage, stage + 2);
        valid &= addRoute("square", "square_write",
            target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::East, *square, stage + 2, stage + 4);
        valid &= addRoute("reduce", "square_read",
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmActivation,
            target::StreamDirection::East, *square, stage + 4, stage + 6);
        valid &= addRoute("factor", "reduction_result",
            target::StreamEndpoint::MxmResult,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, *factor, stage + 6, stage + 8);
        valid &= addRoute("factor", "factor_write",
            target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::East, *factor, stage + 8, stage + 10);
        valid &= addRoute("scale", "input",
            target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, *input, stage + 10, stage + 12);
        valid &= addRoute("scale", "weight",
            target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, *weight, stage + 10, stage + 12);
        valid &= addRoute("scale", "result",
            target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
            target::StreamDirection::East, *result, stage + 12, stage + 14);
    }
    if (!valid) {
        op.emitError("cannot allocate RMSNorm stream routes");
        return mlir::failure();
    }

    context.rewriter.setInsertionPoint(op);
    mlir::OperationState state(
        op.getLoc(), stream::RmsNormTaskOp::getOperationName());
    state.addOperands({op.getInput(), op.getWeight()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        context.rewriter.getNamedAttr("axis", op.getAxisAttr()),
        context.rewriter.getNamedAttr(
            "epsilon", op.getEpsilonAttr()),
        context.rewriter.getNamedAttr(
            "routes", context.rewriter.getArrayAttr(routes)),
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
    auto lowered =
        llvm::cast<stream::RmsNormTaskOp>(
            context.rewriter.create(state));
    context.rewriter.replaceOp(op, lowered.getResult());
    context.stage += stageCount;
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
