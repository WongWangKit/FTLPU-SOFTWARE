#include <algorithm>

#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"

#include "FfnStageEmitter.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace {

void shiftProjectionTimeline(
    ftlpu::compiler::schedule::FfnProjectionTimeline& timeline,
    int64_t offset)
{
    timeline.initial_compute_cycle += offset;
    timeline.final_projection_cycle += offset;
    for (auto& block : timeline.blocks) {
        block.weight_compute_cycle += offset;
        block.dequant_start += offset;
        for (auto& tile : block.tiles)
            tile.compute_cycle += offset;
    }
}

void emitAccumulatorClearPrelude(
    ftlpu::compiler::schedule::ffn_detail::FfnEmissionContext& context,
    int64_t rows)
{
    auto& rewriter = context.rewriter;
    rewriter.setInsertionPoint(context.ffn.getOperation());
    const auto& throughput = context.target.throughput();
    const int64_t unitCount =
        context.target.memory().hemispheres
        * throughput.mxms_per_hemisphere;
    const auto inputType = llvm::cast<mlir::RankedTensorType>(
        context.ffn.getActivation().getType());
    const llvm::StringRef dataFormat =
        ftlpu::compiler::lpu_16bit_data_format(inputType.getElementType());
    for (int64_t unit = 0; unit < unitCount; ++unit) {
        const int64_t outputStream =
            (unit % throughput.mxms_per_hemisphere)
            * throughput.mxm_result_streams;
        for (int64_t offset = 0; offset < rows; ++offset) {
            mlir::OperationState state(
                context.ffn.getLoc(),
                ftlpu::compiler::schedule::MxmIssueOp::
                    getOperationName());
            state.addAttributes({
                rewriter.getNamedAttr("cycle",
                    rewriter.getI64IntegerAttr(offset)),
                rewriter.getNamedAttr("unit_id",
                    rewriter.getI64IntegerAttr(unit)),
                rewriter.getNamedAttr("opcode",
                    rewriter.getStringAttr("accumulator_read")),
                rewriter.getNamedAttr("weight_buffer",
                    rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("weight_column",
                    rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("activation_stream_base",
                    rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("output_stream_base",
                    rewriter.getI64IntegerAttr(outputStream)),
                rewriter.getNamedAttr("repeat_count",
                    rewriter.getI64IntegerAttr(1)),
                rewriter.getNamedAttr("repeat_interval",
                    rewriter.getI64IntegerAttr(1)),
                rewriter.getNamedAttr("accumulator_address",
                    rewriter.getI64IntegerAttr(offset)),
                rewriter.getNamedAttr("accumulator_row_stride",
                    rewriter.getI64IntegerAttr(1)),
                rewriter.getNamedAttr("accumulator_destination",
                    rewriter.getStringAttr("sram")),
                rewriter.getNamedAttr("accumulator_clear",
                    rewriter.getBoolAttr(true)),
                rewriter.getNamedAttr("data_format",
                    rewriter.getStringAttr(dataFormat)),
            });
            rewriter.create(state);
        }
    }
}

} // namespace

namespace ftlpu::compiler {

mlir::FailureOr<mlir::Value> schedule::lowerFfnSchedule(
    mlir::IRRewriter& rewriter,
    schedule::PrimitiveFfnSchedulePlan& plan,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target)
{
    auto context = schedule::ffn_detail::createFfnEmissionContext(
        rewriter, plan, strategy, target);
    if (mlir::failed(context)) {
        plan.getOperation()->emitError(
            "failed to create the FFN emission context");
        return mlir::failure();
    }
    if ((*context)->block8_projection) {
        auto swish =
            schedule::ffn_detail::emitFfnBlock8ProjectionAndSwish(
                **context);
        if (mlir::failed(swish)) {
            plan.getOperation()->emitError(
                "failed to emit the Block8 FFN projection and Swish");
            return mlir::failure();
        }
        auto result = schedule::ffn_detail::emitFfnDownProjection(
            **context, *swish);
        if (mlir::failed(result))
            plan.getOperation()->emitError(
                "failed to emit the Block8 FFN down projection");
        return result;
    }

    const int64_t accumulatorRows =
        (*context)->down_accumulator_base + (*context)->m();
    emitAccumulatorClearPrelude(**context, accumulatorRows);
    const int64_t clearCycles = accumulatorRows
        + target.throughput().accumulator_read_to_vxm_latency + 1;
    shiftProjectionTimeline(
        (*context)->projection_timeline, clearCycles);

    auto projection =
        schedule::ffn_detail::emitFfnProjection(**context);
    if (mlir::failed(projection)) {
        plan.getOperation()->emitError(
            "failed to emit the FFN projection schedule");
        return mlir::failure();
    }
    auto swish = schedule::ffn_detail::emitFfnSwish(
        **context, std::move(*projection));
    if (mlir::failed(swish)) {
        plan.getOperation()->emitError(
            "failed to emit the FFN Swish schedule");
        return mlir::failure();
    }
    auto result = schedule::ffn_detail::emitFfnDownProjection(
        **context, *swish);
    if (mlir::failed(result))
        plan.getOperation()->emitError(
            "failed to emit the FFN down-projection schedule");
    return result;
}

} // namespace ftlpu::compiler
