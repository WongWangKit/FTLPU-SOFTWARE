#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
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
    explicit ScheduleVerifier(mlir::func::FuncOp function) : function_(function) {}

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
                const std::string base = "mem."
                    + std::to_string(hemisphere) + "."
                    + std::to_string(slice) + ".bank"
                    + std::to_string(bank) + ".";
                auto queueResult = reserve(operation,
                    base + "icu",
                    cycle, duration, 1);
                if (queueResult.wasInterrupted()) return queueResult;
                auto portResult = reserve(operation, base + port.str(),
                    cycle, duration, 1);
                if (portResult.wasInterrupted()) return portResult;
            }
        }
        return mlir::WalkResult::advance();
    }

    mlir::WalkResult reserve(mlir::Operation* operation, std::string resource,
        int64_t start, int64_t repeatCount = 1, int64_t repeatInterval = 1)
    {
        auto& reservations = resources_[resource];
        for (int64_t repeat = 0; repeat < repeatCount; ++repeat) {
            const int64_t cycle = start + repeat * repeatInterval;
            for (const auto& interval : intervals_[resource]) {
                if (cycle >= interval.start && cycle < interval.end) {
                    operation->emitError()
                        << "resource '" << resource << "' overlaps at cycle "
                        << cycle << " with " << interval.operation->getName();
                    return mlir::WalkResult::interrupt();
                }
            }
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

    mlir::WalkResult reserveInterval(mlir::Operation* operation,
        const std::string& resource, int64_t start, int64_t end)
    {
        for (const auto& interval : intervals_[resource]) {
            if (start < interval.end && interval.start < end) {
                operation->emitError()
                    << "resource '" << resource << "' ICU contexts overlap ["
                    << start << ", " << end << ") with "
                    << interval.operation->getName() << " ["
                    << interval.start << ", " << interval.end << ")";
                return mlir::WalkResult::interrupt();
            }
        }
        for (const auto& [cycle, owner] : resources_[resource]) {
            if (cycle >= start && cycle < end) {
                operation->emitError()
                    << "resource '" << resource << "' overlaps at cycle "
                    << cycle << " with " << owner->getName();
                return mlir::WalkResult::interrupt();
            }
        }
        intervals_[resource].push_back({start, end, operation});
        return mlir::WalkResult::advance();
    }

    mlir::WalkResult verify(mlir::Operation* operation)
    {
        if (auto op = llvm::dyn_cast<schedule::MemReadOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                for (int64_t wave = 0;
                     wave < op.getWaveCount().value_or(1); ++wave) {
                    auto result = reserveMemPlacement(operation,
                        op.getPlacement(), op.getCycle()
                            + group * op.getGroupInterval().value_or(1)
                            + wave * op.getWaveInterval().value_or(1),
                        op.getDuration(), "read");
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MemWriteOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                for (int64_t wave = 0;
                     wave < op.getWaveCount().value_or(1); ++wave) {
                    auto result = reserveMemPlacement(operation,
                        op.getPlacement(), op.getCycle()
                            + group * op.getGroupInterval().value_or(1)
                            + wave * op.getWaveInterval().value_or(1),
                        op.getDuration(), "write");
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MemTransferOp>(operation)) {
            const int64_t bank = op.getBank().value_or(0);
            const std::string base = "mem."
                + std::to_string(op.getHemisphere()) + "."
                + std::to_string(op.getSlice()) + ".bank"
                + std::to_string(bank) + ".";
            const int64_t lastCycle = op.getCycle()
                + (op.getGroupCount().value_or(1) - 1)
                    * op.getGroupInterval().value_or(1)
                + (op.getWaveCount().value_or(1) - 1)
                    * op.getWaveInterval().value_or(1)
                + (op.getRepeatCount() - 1) * op.getRepeatInterval();
            auto contextResult = reserveInterval(operation, base + "icu",
                op.getCycle(), lastCycle + 1);
            if (contextResult.wasInterrupted()) return contextResult;
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
              for (int64_t wave = 0;
                   wave < op.getWaveCount().value_or(1); ++wave) {
                const int64_t cycle = op.getCycle()
                    + group * op.getGroupInterval().value_or(1)
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
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmIssueOp>(operation)) {
            const int64_t waveCount = op.getWaveCount().value_or(1);
            const int64_t waveInterval = op.getWaveInterval().value_or(1);
            const int64_t groupCount = op.getGroupCount().value_or(1);
            const int64_t groupInterval = op.getGroupInterval().value_or(1);
            const std::string resource = op.getOpcode() == "iw"
                ? "mxm.iw." + std::to_string(op.getUnitId())
                : "mxm.compute." + std::to_string(op.getUnitId());
            for (int64_t group = 0; group < groupCount; ++group) {
                for (int64_t wave = 0; wave < waveCount; ++wave) {
                    auto result = reserve(operation,
                        resource,
                        op.getCycle() + group * groupInterval
                            + wave * waveInterval,
                        op.getRepeatCount(), op.getRepeatInterval());
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmDequantOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                for (int64_t wave = 0;
                     wave < op.getWaveCount().value_or(1); ++wave) {
                    auto result = reserve(operation,
                        "mxm.dequant." + std::to_string(op.getUnitId()),
                        op.getCycle()
                            + group * op.getGroupInterval().value_or(1)
                            + wave * op.getWaveInterval().value_or(1),
                        op.getRepeatCount(), op.getRepeatInterval());
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MemWriteRead2DOp>(operation)) {
            const std::string base = "mem."
                + std::to_string(op.getHemisphere()) + "."
                + std::to_string(op.getSlice()) + ".bank"
                + std::to_string(op.getBank()) + ".";
            const int64_t writeLast =
                (op.getCount0() - 1) * op.getWriteCycleStride0()
                + (op.getCount1() - 1) * op.getWriteCycleStride1();
            const int64_t readLast = op.getReadStartOffset()
                + (op.getCount0() - 1) * op.getReadCycleStride0()
                + (op.getCount1() - 1) * op.getReadCycleStride1();
            return reserveInterval(operation, base + "icu", op.getCycle(),
                op.getCycle() + std::max(writeLast, readLast) + 1);
        }
        if (auto op = llvm::dyn_cast<schedule::VxmOp>(operation)) {
            // A contiguous VXM repeat is carried by the compact functional
            // instruction itself.  The macro ICU launches that body once and
            // immediately frees its sole RUN_2D context; the VXM-local config
            // FIFO then executes the body repeat.  Sparse repeats remain ICU
            // loop launches, so their context stays live through every hole.
            const int64_t repeatCount = op.getRepeatCount();
            const int64_t repeatInterval = op.getRepeatInterval();
            const int64_t waveCount = op.getWaveCount().value_or(1);
            const int64_t waveInterval = op.getWaveInterval().value_or(1);
            const int64_t outerRepeatCount =
                repeatInterval == 1 ? 1 : repeatCount;
            const int64_t finalCycle = op.getCycle()
                + (outerRepeatCount - 1) * repeatInterval
                + (waveCount - 1) * waveInterval;
            auto result = reserve(operation,
                "vxm." + std::to_string(op.getQueue()), op.getCycle(),
                finalCycle - op.getCycle() + 1, 1);
            if (result.wasInterrupted()) return result;
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::SxmOp>(operation)) {
            for (int64_t wave = 0;
                 wave < op.getWaveCount().value_or(1); ++wave) {
                auto result = reserve(operation,
                    "sxm." + op.getOpcode().str() + "."
                        + std::to_string(op.getHemisphere()),
                    op.getCycle() + wave * op.getWaveInterval().value_or(1),
                    op.getRepeatCount().value_or(1),
                    op.getRepeatInterval().value_or(1));
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmLoadOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                auto result = reserve(operation,
                    "mxm.iw." + std::to_string(op.getUnitId()),
                    op.getCycle()
                        + group * op.getGroupInterval().value_or(1),
                    op.getDuration(), 1);
                if (result.wasInterrupted()) return result;
            }
            return mlir::WalkResult::advance();
        }
        if (auto op = llvm::dyn_cast<schedule::MxmComputeOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                for (int64_t wave = 0;
                     wave < op.getWaveCount().value_or(1); ++wave) {
                    auto result = reserve(operation,
                        "mxm.compute." + std::to_string(op.getUnitId()),
                        op.getCycle()
                            + group * op.getGroupInterval().value_or(1)
                            + wave * op.getWaveInterval().value_or(1),
                        op.getDuration(), 1);
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        if (auto op =
                llvm::dyn_cast<schedule::MxmAccumulatorReadOp>(operation)) {
            for (int64_t group = 0;
                 group < op.getGroupCount().value_or(1); ++group) {
                for (int64_t wave = 0;
                     wave < op.getWaveCount().value_or(1); ++wave) {
                    auto result = reserve(operation,
                        "mxm.compute." + std::to_string(op.getUnitId()),
                        op.getCycle()
                            + group * op.getGroupInterval().value_or(1)
                            + wave * op.getWaveInterval().value_or(1),
                        op.getRepeatCount().value_or(1),
                        op.getRepeatInterval().value_or(1));
                    if (result.wasInterrupted()) return result;
                }
            }
            return mlir::WalkResult::advance();
        }
        return mlir::WalkResult::advance();
    }

    mlir::func::FuncOp function_;
    std::unordered_map<std::string,
        std::unordered_map<int64_t, mlir::Operation*>> resources_;
    struct Interval {
        int64_t start;
        int64_t end;
        mlir::Operation* operation;
    };
    std::unordered_map<std::string, std::vector<Interval>> intervals_;
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
