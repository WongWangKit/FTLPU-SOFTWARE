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

struct Reservation {
    int64_t start;
    int64_t end;
    mlir::Operation* owner;
};

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
            const int64_t end = cycle + 1;
            for (const Reservation& existing : reservations) {
                if (cycle < existing.end && end > existing.start) {
                    operation->emitError()
                        << "resource '" << resource << "' overlaps at cycle "
                        << cycle << " with " << existing.owner->getName()
                        << "; current attributes " << operation->getAttrDictionary()
                        << "; existing attributes "
                        << existing.owner->getAttrDictionary();
                    return mlir::WalkResult::interrupt();
                }
            }
            reservations.push_back({cycle, end, operation});
        }
        return mlir::WalkResult::advance();
    }

    mlir::WalkResult verify(mlir::Operation* operation)
    {
        if (auto op = llvm::dyn_cast<schedule::MemTransferOp>(operation))
            return reserve(operation, "mem."
                    + std::to_string(op.getHemisphere()) + "."
                    + std::to_string(op.getSlice()),
                op.getCycle(), op.getRepeatCount(), op.getRepeatInterval());
        if (auto op = llvm::dyn_cast<schedule::MxmIssueOp>(operation))
            return reserve(operation, "mxm." + op.getOpcode().str() + "."
                    + std::to_string(op.getUnitId()),
                op.getCycle(), op.getRepeatCount(), op.getRepeatInterval());
        if (auto op = llvm::dyn_cast<schedule::VxmOp>(operation))
            return reserve(operation, "vxm." + std::to_string(op.getQueue()),
                op.getCycle(), op.getRepeatCount(), op.getRepeatInterval());
        if (auto op = llvm::dyn_cast<schedule::SxmOp>(operation))
            return reserve(operation, "sxm." + op.getOpcode().str() + "."
                    + std::to_string(op.getHemisphere()),
                op.getCycle());
        if (auto op = llvm::dyn_cast<schedule::MxmLoadOp>(operation))
            return reserve(operation, "mxm.iw." + std::to_string(op.getUnitId()),
                op.getCycle(), op.getDuration(), 1);
        if (auto op = llvm::dyn_cast<schedule::MxmComputeOp>(operation))
            return reserve(operation, "mxm.compute." + std::to_string(op.getUnitId()),
                op.getCycle(), op.getDuration(), 1);
        return mlir::WalkResult::advance();
    }

    mlir::func::FuncOp function_;
    std::unordered_map<std::string, std::vector<Reservation>> resources_;
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
