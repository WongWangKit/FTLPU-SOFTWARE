#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>

namespace ftlpu::compiler {
namespace {

constexpr int64_t kMaxRepeatCount = 1023;
constexpr int64_t kMaxRepeatInterval = 255;
constexpr int64_t kMinRepeatStride = -2048;
constexpr int64_t kMaxRepeatStride = 2047;

int64_t optional_integer(
    mlir::Operation* operation, llvm::StringRef name, int64_t fallback)
{
    if (auto value = operation->getAttrOfType<mlir::IntegerAttr>(name))
        return value.getInt();
    return fallback;
}

bool same_mem_body(
    schedule::MemTransferOp lhs, schedule::MemTransferOp rhs)
{
    return lhs.getHemisphere() == rhs.getHemisphere()
        && lhs.getSlice() == rhs.getSlice()
        && lhs.getOpcode() == rhs.getOpcode()
        && lhs.getPackedStream() == rhs.getPackedStream()
        && lhs.getRepeatCount() == rhs.getRepeatCount()
        && lhs.getRepeatInterval() == rhs.getRepeatInterval()
        && lhs.getAddressStride() == rhs.getAddressStride()
        && lhs.getAccumulatorDestination()
            == rhs.getAccumulatorDestination()
        && optional_integer(lhs, "wave_count", 1) == 1
        && optional_integer(rhs, "wave_count", 1) == 1;
}

mlir::DictionaryAttr placement_without_slices(
    mlir::DictionaryAttr placement, mlir::MLIRContext* context)
{
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (mlir::NamedAttribute attribute : placement)
        if (attribute.getName().strref() != "slices")
            attributes.push_back(attribute);
    return mlir::DictionaryAttr::get(context, attributes);
}

std::optional<int64_t> single_slice(schedule::MemReadOp read)
{
    auto slices = read.getPlacement().getAs<mlir::ArrayAttr>("slices");
    if (!slices || slices.size() != 1 || read.getStreamCount() != 1)
        return std::nullopt;
    return llvm::cast<mlir::IntegerAttr>(slices[0]).getInt();
}

bool same_read_body(schedule::MemReadOp lhs, schedule::MemReadOp rhs,
    mlir::MLIRContext* context)
{
    return lhs.getInput() == rhs.getInput()
        && lhs.getDuration() == rhs.getDuration()
        && lhs.getDirection() == rhs.getDirection()
        && lhs.getRole() == rhs.getRole()
        && lhs.getAddress() == rhs.getAddress()
        && lhs.getOutput().getType() == rhs.getOutput().getType()
        && placement_without_slices(lhs.getPlacement(), context)
            == placement_without_slices(rhs.getPlacement(), context);
}

bool same_vxm_body(schedule::VxmOp lhs, schedule::VxmOp rhs)
{
    for (llvm::StringRef name : {"queue", "opcode", "lhs_kind",
             "lhs_index", "lhs_immediate", "rhs_kind", "rhs_index",
             "rhs_immediate", "cast_target", "output_stream",
             "input_hemisphere", "output_hemisphere"})
        if (lhs->getAttr(name) != rhs->getAttr(name))
            return false;
    return lhs.getRepeatCount() == 1 && rhs.getRepeatCount() == 1
        && lhs.getResult().getType() == rhs.getResult().getType();
}

class CompressSchedulePass final
    : public mlir::PassWrapper<CompressSchedulePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompressSchedulePass)

    llvm::StringRef getArgument() const final
    {
        return "ftlpu-compress-schedule";
    }
    llvm::StringRef getDescription() const final
    {
        return "Compresses repeated schedule issues into nested wave/repeat form";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (function->hasAttr("ftlpu.schedule.compressed"))
            return;
        auto target = target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target)) {
            signalPassFailure();
            return;
        }

        mlir::Builder builder(&getContext());
        // Round one forms hardware-sized repeat chunks. Round two combines
        // equal chunks into an outer wave without flattening the ICU repeat.
        for (int round = 0; round < 2; ++round) {
            llvm::SmallVector<schedule::MemTransferOp> transfers;
            function.walk([&](schedule::MemTransferOp op) {
                transfers.push_back(op);
            });
            llvm::sort(transfers,
                [&](schedule::MemTransferOp lhs,
                    schedule::MemTransferOp rhs) {
                const int64_t lhsQueue = lhs.getHemisphere()
                        * target->memory().slices_per_hemisphere
                    + lhs.getSlice();
                const int64_t rhsQueue = rhs.getHemisphere()
                        * target->memory().slices_per_hemisphere
                    + rhs.getSlice();
                return lhsQueue != rhsQueue
                    ? lhsQueue < rhsQueue
                    : lhs.getCycle() < rhs.getCycle();
                });

            llvm::SmallVector<schedule::MemTransferOp> toErase;
            for (std::size_t index = 0; index < transfers.size();) {
            schedule::MemTransferOp first = transfers[index];
            std::size_t end = index + 1;
            if (end >= transfers.size()
                || !same_mem_body(first, transfers[end])) {
                index = end;
                continue;
            }

            const int64_t waveInterval =
                transfers[end].getCycle() - first.getCycle();
            const int64_t waveAddressStride =
                transfers[end].getAddress() - first.getAddress();
            if (waveInterval <= 0) {
                index = end;
                continue;
            }
            ++end;
            while (end < transfers.size()) {
                schedule::MemTransferOp next = transfers[end];
                const int64_t wave =
                    static_cast<int64_t>(end - index);
                if (!same_mem_body(first, next)
                    || next.getCycle()
                        != first.getCycle() + wave * waveInterval
                    || next.getAddress()
                        != first.getAddress()
                            + wave * waveAddressStride)
                    break;
                ++end;
            }

            const bool canUseHardwareRepeat =
                first.getRepeatCount() == 1
                && waveInterval <= kMaxRepeatInterval
                && waveAddressStride >= kMinRepeatStride
                && waveAddressStride <= kMaxRepeatStride;
            if (canUseHardwareRepeat)
                end = std::min(end,
                    index + static_cast<std::size_t>(
                                kMaxRepeatCount));
            const int64_t count = static_cast<int64_t>(end - index);
            const bool useHardwareRepeat = canUseHardwareRepeat;
            if (useHardwareRepeat) {
                first->setAttr(
                    "repeat_count", builder.getI64IntegerAttr(count));
                first->setAttr("repeat_interval",
                    builder.getI64IntegerAttr(waveInterval));
                first->setAttr("address_stride",
                    builder.getI64IntegerAttr(waveAddressStride));
            } else {
                first->setAttr(
                    "wave_count", builder.getI64IntegerAttr(count));
                first->setAttr("wave_interval",
                    builder.getI64IntegerAttr(waveInterval));
                first->setAttr("wave_address_stride",
                    builder.getI64IntegerAttr(waveAddressStride));
            }
            for (std::size_t erase = index + 1; erase < end; ++erase)
                toErase.push_back(transfers[erase]);
            index = end;
            }
            for (schedule::MemTransferOp operation : toErase)
                operation.erase();
        }

        compressMemReads(function, *target, builder);
        compressVxm(function, builder);
        function->setAttr(
            "ftlpu.schedule.compressed", builder.getUnitAttr());
    }

private:
    void compressMemReads(mlir::func::FuncOp function,
        const target::LPUTargetModel& target, mlir::Builder& builder)
    {
        llvm::SmallVector<schedule::MemReadOp> reads;
        function.walk(
            [&](schedule::MemReadOp op) { reads.push_back(op); });
        llvm::SmallVector<schedule::MemReadOp> toErase;
        for (std::size_t index = 0; index < reads.size();) {
            schedule::MemReadOp first = reads[index];
            auto firstSlice = single_slice(first);
            if (!firstSlice) {
                ++index;
                continue;
            }
            const auto direction = first.getDirection() == "west"
                ? target::StreamDirection::West
                : target::StreamDirection::East;
            const auto endpoint =
                first.getRole() == "weight"
                ? target::StreamEndpoint::MxmWeight
                : first.getRole() == "activation"
                ? target::StreamEndpoint::MxmActivation
                : target::StreamEndpoint::VxmInput;
            const auto firstLatency = target.transport_latency(
                target::StreamEndpoint::Mem, endpoint, direction,
                *firstSlice);
            if (!firstLatency) {
                ++index;
                continue;
            }
            const int64_t productionCycle =
                first.getCycle() + *firstLatency;
            llvm::SmallVector<int64_t> slices {*firstSlice};
            llvm::SmallVector<int64_t> registerIds {
                static_cast<int64_t>(first.getRegisterId())};
            int64_t minCycle = static_cast<int64_t>(first.getCycle());
            int64_t totalBytes = static_cast<int64_t>(first.getBytes());
            std::size_t end = index + 1;
            while (end < reads.size()
                && slices.size()
                    < static_cast<std::size_t>(
                        target.streams().streams_per_direction)) {
                schedule::MemReadOp next = reads[end];
                auto slice = single_slice(next);
                if (!slice || !same_read_body(first, next, &getContext())
                    || next.getStreamBase()
                        != first.getStreamBase()
                            + static_cast<int64_t>(slices.size()))
                    break;
                auto latency = target.transport_latency(
                    target::StreamEndpoint::Mem, endpoint, direction,
                    *slice);
                if (!latency
                    || next.getCycle() + *latency != productionCycle)
                    break;
                slices.push_back(*slice);
                registerIds.push_back(next.getRegisterId());
                minCycle = std::min(
                    minCycle, static_cast<int64_t>(next.getCycle()));
                totalBytes += next.getBytes();
                ++end;
            }
            if (end == index + 1) {
                ++index;
                continue;
            }

            llvm::SmallVector<mlir::Attribute> sliceAttrs;
            llvm::SmallVector<mlir::Attribute> registerAttrs;
            for (int64_t slice : slices)
                sliceAttrs.push_back(builder.getI64IntegerAttr(slice));
            for (int64_t registerId : registerIds)
                registerAttrs.push_back(
                    builder.getI64IntegerAttr(registerId));
            mlir::NamedAttrList placement(first.getPlacement());
            placement.set("slices", builder.getArrayAttr(sliceAttrs));
            first->setAttr("placement",
                placement.getDictionary(&getContext()));
            first->setAttr(
                "cycle", builder.getI64IntegerAttr(minCycle));
            first->setAttr("stream_count",
                builder.getI64IntegerAttr(slices.size()));
            first->setAttr(
                "bytes", builder.getI64IntegerAttr(totalBytes));
            first->setAttr(
                "register_ids", builder.getArrayAttr(registerAttrs));
            for (std::size_t erase = index + 1; erase < end; ++erase) {
                reads[erase].getOutput().replaceAllUsesWith(
                    first.getOutput());
                toErase.push_back(reads[erase]);
            }
            index = end;
        }
        for (schedule::MemReadOp operation : toErase)
            operation.erase();
    }

    void compressVxm(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        llvm::SmallVector<schedule::VxmOp> operations;
        function.walk(
            [&](schedule::VxmOp op) { operations.push_back(op); });
        llvm::sort(operations,
            [](schedule::VxmOp lhs, schedule::VxmOp rhs) {
                return lhs.getQueue() != rhs.getQueue()
                    ? lhs.getQueue() < rhs.getQueue()
                    : lhs.getCycle() < rhs.getCycle();
            });
        llvm::SmallVector<schedule::VxmOp> toErase;
        for (std::size_t index = 0; index < operations.size();) {
            schedule::VxmOp first = operations[index];
            std::size_t end = index + 1;
            if (end >= operations.size()
                || !same_vxm_body(first, operations[end])) {
                ++index;
                continue;
            }
            const int64_t interval =
                operations[end].getCycle() - first.getCycle();
            if (interval <= 0 || interval > kMaxRepeatInterval) {
                ++index;
                continue;
            }
            ++end;
            while (end < operations.size()
                && static_cast<int64_t>(end - index)
                    < kMaxRepeatCount) {
                const int64_t repeat =
                    static_cast<int64_t>(end - index);
                if (!same_vxm_body(first, operations[end])
                    || operations[end].getCycle()
                        != first.getCycle() + repeat * interval)
                    break;
                ++end;
            }

            schedule::VxmOp representative = first;
            for (std::size_t candidate = index + 1;
                 candidate < end; ++candidate)
                if (operations[candidate]->isBeforeInBlock(
                        representative))
                    representative = operations[candidate];
            representative->setAttr(
                "cycle", builder.getI64IntegerAttr(first.getCycle()));
            representative->setAttr("repeat_count",
                builder.getI64IntegerAttr(end - index));
            representative->setAttr(
                "repeat_interval", builder.getI64IntegerAttr(interval));
            for (std::size_t erase = index; erase < end; ++erase) {
                if (operations[erase] == representative)
                    continue;
                operations[erase].getResult().replaceAllUsesWith(
                    representative.getResult());
                toErase.push_back(operations[erase]);
            }
            index = end;
        }
        for (schedule::VxmOp operation : toErase)
            operation.erase();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_compress_schedule_pass()
{
    return std::make_unique<CompressSchedulePass>();
}

} // namespace ftlpu::compiler
