#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ftlpu::compiler {
namespace {

class ScheduleVerifier {
public:
    ScheduleVerifier(mlir::func::FuncOp function,
        const target::LPUTargetModel& target)
        : function_(function), resources_model_(target) {}

    mlir::LogicalResult run()
    {
        mlir::WalkResult result = function_.walk([&](mlir::Operation* operation) {
            return verify(operation);
        });
        return result.wasInterrupted() ? mlir::failure() : mlir::success();
    }

private:
    mlir::WalkResult reserveMemPlacement(mlir::Operation* operation,
        mlir::DictionaryAttr placement, int64_t cycle, int64_t duration,
        llvm::StringRef port)
    {
        const auto hemisphereAttr =
            placement.getAs<mlir::StringAttr>("hemisphere");
        const auto slicesAttr = placement.getAs<mlir::ArrayAttr>("slices");
        if (!hemisphereAttr || !slicesAttr)
            return mlir::WalkResult::advance();
        const int64_t bank = placement.getAs<mlir::IntegerAttr>("bank")
            ? placement.getAs<mlir::IntegerAttr>("bank").getInt()
            : 0;
        llvm::SmallVector<int64_t, 2> hemispheres;
        if (hemisphereAttr.getValue() == "both")
            hemispheres = {0, 1};
        else
            hemispheres.push_back(
                hemisphereAttr.getValue() == "west" ? 1 : 0);
        for (int64_t hemisphere : hemispheres) {
            for (mlir::Attribute sliceAttr : slicesAttr) {
                const int64_t slice =
                    llvm::cast<mlir::IntegerAttr>(sliceAttr).getInt();
                auto queueResult = reserve(operation,
                    resources_model_.mem_icu(hemisphere, slice, bank),
                    cycle, duration, 1);
                if (queueResult.wasInterrupted()) return queueResult;
                const auto portResource = port == "read"
                    ? resources_model_.mem_read_port(
                        hemisphere, slice, bank)
                    : resources_model_.mem_write_port(
                        hemisphere, slice, bank);
                auto portResult = reserve(operation, portResource,
                    cycle, duration, 1);
                if (portResult.wasInterrupted()) return portResult;
            }
        }
        return mlir::WalkResult::advance();
    }

    mlir::WalkResult reserve(mlir::Operation* operation,
        schedule::PhysicalResourceId resource,
        int64_t start, int64_t repeatCount = 1, int64_t repeatInterval = 1)
    {
        auto& reservations = resources_[resource];
        for (int64_t repeat = 0; repeat < repeatCount; ++repeat) {
            const int64_t cycle = start + repeat * repeatInterval;
            auto [position, inserted] =
                reservations.try_emplace(cycle, operation);
            if (!inserted) {
                operation->emitError()
                    << "resource '"
                    << schedule::physical_resource_name(resource)
                    << "' overlaps at cycle "
                    << cycle << " with " << position->second->getName()
                    << "; current attributes "
                    << operation->getAttrDictionary()
                    << "; existing attributes "
                    << position->second->getAttrDictionary();
                return mlir::WalkResult::interrupt();
            }
        }
        return mlir::WalkResult::advance();
    }

    mlir::WalkResult verify(mlir::Operation* operation)
    {
        if (auto op = llvm::dyn_cast<schedule::MemReadOp>(operation)) {
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                auto result = reserveMemPlacement(operation,
                    op.getPlacement(),
                    op.getCycle()
                        + wave * op.getWaveInterval().value_or(1),
                    op.getDuration(), "read");
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MemWriteOp>(operation)) {
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                auto result = reserveMemPlacement(operation,
                    op.getPlacement(),
                    op.getCycle()
                        + wave * op.getWaveInterval().value_or(1),
                    op.getDuration(), "write");
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MemTransferOp>(operation)) {
            const int64_t bank = op.getBank().value_or(0);
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                const int64_t cycle = op.getCycle()
                    + wave * op.getWaveInterval().value_or(1);
                auto queueResult = reserve(operation,
                    resources_model_.mem_icu(
                        op.getHemisphere(), op.getSlice(), bank),
                    cycle, op.getRepeatCount(), op.getRepeatInterval());
                if (queueResult.wasInterrupted()) return queueResult;
                if (op.getOpcode() == "read"
                    || op.getOpcode() == "read_write") {
                    auto result = reserve(operation,
                        resources_model_.mem_read_port(
                            op.getHemisphere(), op.getSlice(), bank),
                        cycle, op.getRepeatCount(),
                        op.getRepeatInterval());
                    if (result.wasInterrupted()) return result;
                }
                if (op.getOpcode() == "write"
                    || op.getOpcode() == "write_tap"
                    || op.getOpcode() == "read_write") {
                    auto result = reserve(operation,
                        resources_model_.mem_write_port(
                            op.getHemisphere(), op.getSlice(), bank),
                        cycle, op.getRepeatCount(),
                        op.getRepeatInterval());
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmIssueOp>(operation)) {
            const int64_t waveCount = op.getWaveCount().value_or(1);
            const int64_t waveInterval = op.getWaveInterval().value_or(1);
            const int64_t groupCount = op.getGroupCount().value_or(1);
            const int64_t groupInterval = op.getGroupInterval().value_or(1);
            for (int64_t group = 0; group < groupCount; ++group) {
                for (int64_t wave = 0; wave < waveCount; ++wave) {
                    const bool load = op.getOpcode() == "iw"
                        || op.getOpcode() == "decode_load_activation";
                    auto result = reserve(operation,
                        resources_model_.queue_resource(load
                                ? resources_model_.mxm_load_queue(op.getUnitId())
                                : resources_model_.mxm_compute_queue(op.getUnitId())),
                        op.getCycle() + group * groupInterval
                            + wave * waveInterval,
                        op.getRepeatCount(), op.getRepeatInterval());
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmDequantOp>(operation)) {
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                auto result = reserve(operation,
                    resources_model_.queue_resource(
                        resources_model_.mxm_dequant_queue(op.getUnitId())),
                    op.getCycle()
                        + wave * op.getWaveInterval().value_or(1),
                    op.getRepeatCount(), op.getRepeatInterval());
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::VxmOp>(operation))
            return reserve(operation, resources_model_.queue_resource(
                    resources_model_.vxm_queue(op.getQueue())),
                op.getCycle(), 1, 1);
        if (auto op = llvm::dyn_cast<schedule::SxmOp>(operation))
            return reserve(operation, resources_model_.queue_resource(
                    resources_model_.sxm_queue(
                        op.getOpcode() == "transpose", op.getHemisphere())),
                op.getCycle(), op.getRepeatCount().value_or(1),
                op.getRepeatInterval().value_or(1));
        if (auto op = llvm::dyn_cast<schedule::MxmLoadOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                auto result = reserve(operation,
                    resources_model_.queue_resource(
                        resources_model_.mxm_load_queue(op.getUnitId())),
                    op.getCycle()
                        + group * op.getGroupInterval().value_or(1),
                    op.getDuration(), 1);
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmComputeOp>(operation)) {
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                auto result = reserve(operation,
                    resources_model_.queue_resource(
                        resources_model_.mxm_compute_queue(op.getUnitId())),
                    op.getCycle()
                        + wave * op.getWaveInterval().value_or(1),
                    op.getDuration(), 1);
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        return mlir::WalkResult::advance();
    }

    mlir::func::FuncOp function_;
    schedule::LPUResourceModel resources_model_;
    std::unordered_map<schedule::PhysicalResourceId,
        std::unordered_map<int64_t, mlir::Operation*>,
        schedule::PhysicalResourceIdHash> resources_;
};

class VerifySchedulePass final
    : public mlir::PassWrapper<VerifySchedulePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySchedulePass)

    llvm::StringRef getArgument() const final { return "ftlpu-verify-schedule"; }
    llvm::StringRef getDescription() const final
    {
        return "Verifies exact-cycle LPU resource exclusivity before command lowering";
    }

    void runOnOperation() final
    {
        if (mlir::failed(mlir::verify(getOperation()))) {
            signalPassFailure();
            return;
        }
        auto targetModel = target::LPUTargetModel::from_operation(getOperation());
        if (mlir::failed(targetModel)) {
            signalPassFailure();
            return;
        }
        ScheduleVerifier verifier(getOperation(), *targetModel);
        if (mlir::failed(verifier.run())) signalPassFailure();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_verify_schedule_pass()
{
    return std::make_unique<VerifySchedulePass>();
}

} // namespace ftlpu::compiler
