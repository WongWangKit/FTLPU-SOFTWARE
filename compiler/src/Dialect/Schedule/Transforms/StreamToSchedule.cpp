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

#include <algorithm>
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

void sequentializeScheduleStages(mlir::func::FuncOp function)
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
        cursor += std::max<int64_t>(1, end - first);
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

        if (mlir::failed(
                schedule::lowerRmsNormSchedules(rewriter, function, target))) {
            signalPassFailure();
            return;
        }

        if (mlir::failed(schedule::lowerElementwiseSchedules(
                rewriter, function, target))) {
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

        sequentializeScheduleStages(function);
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
