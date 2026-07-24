#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"

#include "FfnStageEmitter.hpp"

namespace ftlpu::compiler {

mlir::FailureOr<mlir::Value> schedule::lowerFfnSchedule(
    mlir::IRRewriter& rewriter,
    schedule::PrimitiveFfnSchedulePlan& plan,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target)
{
    auto context = schedule::ffn_detail::createFfnEmissionContext(
        rewriter, plan, strategy, target);
    if (mlir::failed(context)) return mlir::failure();

    auto projection =
        schedule::ffn_detail::emitFfnProjection(**context);
    if (mlir::failed(projection)) return mlir::failure();

    auto swish = schedule::ffn_detail::emitFfnSwish(
        **context, std::move(*projection));
    if (mlir::failed(swish)) return mlir::failure();

    return schedule::ffn_detail::emitFfnDownProjection(
        **context, *swish);
}

} // namespace ftlpu::compiler
