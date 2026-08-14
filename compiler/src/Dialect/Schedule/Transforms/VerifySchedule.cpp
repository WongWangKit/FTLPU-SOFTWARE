#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    explicit ScheduleVerifier(mlir::func::FuncOp function) : function_(function) {}

    mlir::LogicalResult run()
    {
        mlir::WalkResult result = function_.walk([&](mlir::Operation* operation) {
            return verify(operation);
        });
        return result.wasInterrupted() ? mlir::failure() : mlir::success();
    }

private:
    mlir::WalkResult reserve(mlir::Operation* operation, std::string resource,
        int64_t start, int64_t repeatCount = 1, int64_t repeatInterval = 1)
    {
        auto& reservations = resources_[resource];
        for (int64_t repeat = 0; repeat < repeatCount; ++repeat) {
            const int64_t cycle = start + repeat * repeatInterval;
            auto [position, inserted] =
                reservations.try_emplace(cycle, operation);
            if (!inserted) {
                operation->emitError()
                    << "resource '" << resource << "' overlaps at cycle "
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
        if (auto op = llvm::dyn_cast<schedule::MemTransferOp>(operation)) {
            const std::string base = "mem."
                + std::to_string(op.getHemisphere()) + "."
                + std::to_string(op.getSlice()) + ".";
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                const int64_t cycle = op.getCycle()
                    + wave * op.getWaveInterval().value_or(1);
                if (op.getOpcode() == "read"
                    || op.getOpcode() == "read_write") {
                    auto result = reserve(operation, base + "read",
                        cycle, op.getRepeatCount(),
                        op.getRepeatInterval());
                    if (result.wasInterrupted()) return result;
                }
                if (op.getOpcode() == "write"
                    || op.getOpcode() == "write_tap"
                    || op.getOpcode() == "read_write") {
                    auto result = reserve(operation, base + "write",
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
                    auto result = reserve(operation,
                        "mxm." + op.getOpcode().str() + "."
                            + std::to_string(op.getUnitId()),
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
                    "mxm.dequant." + std::to_string(op.getUnitId()),
                    op.getCycle()
                        + wave * op.getWaveInterval().value_or(1),
                    op.getRepeatCount(), op.getRepeatInterval());
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::VxmOp>(operation))
            return reserve(operation, "vxm." + std::to_string(op.getQueue()),
                op.getCycle(), 1, 1);
        if (auto op = llvm::dyn_cast<schedule::SxmOp>(operation))
            return reserve(operation, "sxm." + op.getOpcode().str() + "."
                    + std::to_string(op.getHemisphere()),
                op.getCycle(), op.getRepeatCount().value_or(1),
                op.getRepeatInterval().value_or(1));
        if (auto op = llvm::dyn_cast<schedule::MxmLoadOp>(operation))
            return reserve(operation, "mxm.iw." + std::to_string(op.getUnitId()),
                op.getCycle(), op.getDuration(), 1);
        if (auto op = llvm::dyn_cast<schedule::MxmComputeOp>(operation))
            return reserve(operation, "mxm.compute." + std::to_string(op.getUnitId()),
                op.getCycle(), op.getDuration(), 1);
        return mlir::WalkResult::advance();
    }

    mlir::func::FuncOp function_;
    std::unordered_map<std::string,
        std::unordered_map<int64_t, mlir::Operation*>> resources_;
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
        ScheduleVerifier verifier(getOperation());
        if (mlir::failed(verifier.run())) signalPassFailure();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_verify_schedule_pass()
{
    return std::make_unique<VerifySchedulePass>();
}

} // namespace ftlpu::compiler
