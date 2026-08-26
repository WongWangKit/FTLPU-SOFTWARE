// Keep this translation unit rebuilt with target topology ABI changes.
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>

namespace ftlpu::compiler {
namespace {

bool isScheduleOperation(mlir::Operation& operation)
{
    return operation.getName().getDialectNamespace() == "ftlpu.schedule";
}

int64_t integerAttribute(
    mlir::Operation& operation, llvm::StringRef name, int64_t fallback)
{
    if (auto value = operation.getAttrOfType<mlir::IntegerAttr>(name))
        return value.getInt();
    return fallback;
}

void shiftIntegerAttribute(mlir::Operation& operation,
    llvm::StringRef name, int64_t offset)
{
    auto value = operation.getAttrOfType<mlir::IntegerAttr>(name);
    if (!value) return;
    operation.setAttr(name, mlir::IntegerAttr::get(
        value.getType(), value.getInt() + offset));
}

void assignMxmDataFormats(mlir::func::FuncOp function)
{
    function.walk([&](schedule::MxmComputeOp op) {
        const auto activationType =
            llvm::cast<mlir::RankedTensorType>(
                op.getActivation().getType());
        op.setDataFormat(
            lpu_16bit_data_format(
                activationType.getElementType()));
    });
}

void sequentializeScheduleStages(mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    llvm::SmallVector<llvm::SmallVector<mlir::Operation*>> stages;
    llvm::SmallVector<mlir::Operation*> current;
    std::optional<mlir::Location> currentLocation;
    for (mlir::Operation& operation : function.getBody().front()) {
        if (!isScheduleOperation(operation)) {
            if (!current.empty()) {
                stages.push_back(std::move(current));
                current.clear();
                currentLocation.reset();
            }
            continue;
        }
        if (!current.empty() && operation.getLoc() != *currentLocation) {
            stages.push_back(std::move(current));
            current.clear();
        }
        if (current.empty()) currentLocation = operation.getLoc();
        current.push_back(&operation);
    }
    if (!current.empty()) stages.push_back(std::move(current));

    int64_t cursor = 0;
    const int64_t streamDrainCycles =
        target.streams().system_register_columns;
    for (auto& stage : stages) {
        int64_t first = std::numeric_limits<int64_t>::max();
        int64_t end = 0;
        for (mlir::Operation* operation : stage) {
            if (auto cycle =
                    operation->getAttrOfType<mlir::IntegerAttr>("cycle")) {
                first = std::min(first, cycle.getInt());
                const int64_t duration = std::max<int64_t>(
                    1, integerAttribute(*operation, "duration",
                        integerAttribute(*operation, "repeat_count", 1)
                            * integerAttribute(
                                *operation, "repeat_interval", 1)));
                end = std::max(end, cycle.getInt() + duration);
            }
            if (auto resultCycle = operation->getAttrOfType<
                    mlir::IntegerAttr>("result_cycle")) {
                first = std::min(first, resultCycle.getInt());
                end = std::max(end, resultCycle.getInt()
                    + integerAttribute(
                        *operation, "result_duration", 1));
            }
            if (auto start =
                    operation->getAttrOfType<mlir::IntegerAttr>("start"))
                first = std::min(first, start.getInt());
            if (auto timelineEnd =
                    operation->getAttrOfType<mlir::IntegerAttr>("end"))
                end = std::max(end, timelineEnd.getInt());
        }
        if (first == std::numeric_limits<int64_t>::max()) continue;
        const int64_t offset = cursor - first;
        for (mlir::Operation* operation : stage) {
            shiftIntegerAttribute(*operation, "cycle", offset);
            shiftIntegerAttribute(*operation, "result_cycle", offset);
            shiftIntegerAttribute(*operation, "start", offset);
            shiftIntegerAttribute(*operation, "end", offset);
        }
        // A stage can finish issuing while its final vector beat is still
        // moving through passive stream-register links. Keep the next stage
        // from injecting a different producer onto those links until the
        // longest possible on-chip route has drained.
        cursor += std::max<int64_t>(1, end - first) + streamDrainCycles;
    }

    llvm::SmallDenseSet<mlir::Value> returnedValues;
    function.walk([&](mlir::func::ReturnOp op) {
        for (mlir::Value value : op.getOperands())
            returnedValues.insert(value);
    });
    function.walk([&](schedule::BindingOp binding) {
        if (binding.getAccess() == "output"
            && !returnedValues.contains(binding.getValue()))
            binding->setAttr(
                "access", mlir::StringAttr::get(
                    function.getContext(), "internal"));
    });
    int64_t outputIndex = 0;
    function.walk([&](schedule::BindingOp binding) {
        if (binding.getAccess() == "output")
            binding->setAttr("index", mlir::IntegerAttr::get(
                binding.getIndexAttr().getType(), outputIndex++));
    });
}

class LowerStreamToSchedulePass final
    : public mlir::PassWrapper<LowerStreamToSchedulePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStreamToSchedulePass)

    LowerStreamToSchedulePass() = default;
    explicit LowerStreamToSchedulePass(FfnScheduleStrategy ffnStrategy,
        AttentionScheduleStrategy attentionStrategy, bool stageTiming)
        : ffn_strategy_(ffnStrategy)
        , attention_strategy_(attentionStrategy)
        , stage_timing_(stageTiming)
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
        const auto passStart = std::chrono::steady_clock::now();
        auto lastStage = passStart;
        const auto reportStage = [&](llvm::StringRef name) {
            if (!stage_timing_) return;
            const auto now = std::chrono::steady_clock::now();
            const double stageSeconds =
                std::chrono::duration<double>(now - lastStage).count();
            const double totalSeconds =
                std::chrono::duration<double>(now - passStart).count();
            llvm::errs() << "[ftlpu-stage-timing] " << name << ": "
                         << llvm::format("%.3f", stageSeconds)
                         << " s (total "
                         << llvm::format("%.3f", totalSeconds) << " s)\n";
            lastStage = now;
        };
        auto primitive_ffns =
            schedule::collectPrimitiveFfnSchedulePlans(function);
        if (mlir::failed(primitive_ffns)) {
            signalPassFailure();
            return;
        }
        reportStage("primitive-ffn-plan");
        int64_t ffnIndex = 0;
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
            if (stage_timing_)
                reportStage("primitive-ffn-lower-"
                    + std::to_string(ffnIndex));
            ++ffnIndex;
        }
        if (mlir::failed(
                schedule::lowerAttentionSchedules(
                    rewriter, function, target, attention_strategy_))) {
            signalPassFailure();
            return;
        }
        reportStage("attention");
        if (mlir::failed(
                schedule::lowerRmsNormSchedules(rewriter, function, target))) {
            signalPassFailure();
            return;
        }
        reportStage("rmsnorm");
        if (mlir::failed(schedule::lowerLinearProjectionSchedules(
                rewriter, function, target))) {
            signalPassFailure();
            return;
        }
        reportStage("linear-projection");
        if (mlir::failed(schedule::lowerElementwiseSchedules(
                rewriter, function, target))) {
            signalPassFailure();
            return;
        }
        reportStage("elementwise");
        schedule::ResourceScheduler scheduler;
        schedule::StreamFabricScheduler streamScheduler(
            target.streams().system_register_columns,
            target.streams().streams_per_direction);
        if (mlir::failed(schedule::lowerSwigluSchedules(
                rewriter, function, target, scheduler, streamScheduler))
            || mlir::failed(schedule::lowerMatmulSchedules(
                rewriter, function, target, scheduler, streamScheduler))) {
            signalPassFailure();
            return;
        }
        reportStage("generic-kernels");

        assignMxmDataFormats(function);
        sequentializeScheduleStages(function, target);
        reportStage("finalize");
    }

private:
    FfnScheduleStrategy ffn_strategy_ = FfnScheduleStrategy::Tail;
    AttentionScheduleStrategy attention_strategy_ =
        AttentionScheduleStrategy::Tail;
    bool stage_timing_ = false;
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass(
    FfnScheduleStrategy ffn_strategy,
    AttentionScheduleStrategy attention_strategy, bool stage_timing)
{
    return std::make_unique<LowerStreamToSchedulePass>(
        ffn_strategy, attention_strategy, stage_timing);
}

} // namespace ftlpu::compiler
