// Keep this translation unit rebuilt with target topology ABI changes.
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

class LowerStreamToSchedulePass final
    : public mlir::PassWrapper<LowerStreamToSchedulePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStreamToSchedulePass)

    LowerStreamToSchedulePass() = default;
    explicit LowerStreamToSchedulePass(FfnScheduleStrategy strategy)
        : ffn_strategy_(strategy)
    {
    }

    llvm::StringRef getArgument() const final { return "ftlpu-stream-to-schedule"; }
    llvm::StringRef getDescription() const final
    {
        return "Schedules LPU stream routes at exact CModel issue cycles";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError("cycle scheduling currently requires a single-block function");
            signalPassFailure();
            return;
        }

        mlir::IRRewriter rewriter(&getContext());
        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        auto primitive_ffns =
            schedule::collectPrimitiveFfnSchedulePlans(function);
        if (mlir::failed(primitive_ffns)) {
            signalPassFailure();
            return;
        }
        for (schedule::PrimitiveFfnSchedulePlan& ffn : *primitive_ffns) {
            rewriter.setInsertionPoint(ffn.add);
            auto result = schedule::lowerFfnSchedule(
                rewriter, ffn, ffn_strategy_, target);
            if (mlir::failed(result)) {
                ffn.add.emitError(
                    "failed to schedule a primitive W8A16 FFN graph");
                signalPassFailure();
                return;
            }
            rewriter.replaceOp(ffn.add, *result);
            rewriter.eraseOp(ffn.down1);
            rewriter.eraseOp(ffn.down0);
            rewriter.eraseOp(ffn.hidden1_route);
            rewriter.eraseOp(ffn.hidden0_route);
            rewriter.eraseOp(ffn.multiply);
            rewriter.eraseOp(ffn.swish);
            rewriter.eraseOp(ffn.up);
            rewriter.eraseOp(ffn.gate);
        }

        if (mlir::failed(
                schedule::lowerAttentionSchedules(rewriter, function, target))) {
            signalPassFailure();
            return;
        }

        schedule::ResourceScheduler scheduler;
        if (mlir::failed(schedule::lowerSwigluSchedules(
                rewriter, function, target, scheduler))
            || mlir::failed(schedule::lowerMatmulSchedules(
                rewriter, function, target, scheduler))) {
            signalPassFailure();
            return;
        }
    }

private:
    FfnScheduleStrategy ffn_strategy_ = FfnScheduleStrategy::Tail;
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass(
    FfnScheduleStrategy ffn_strategy)
{
    return std::make_unique<LowerStreamToSchedulePass>(ffn_strategy);
}

} // namespace ftlpu::compiler
