#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <map>
#include <tuple>

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
        && lhs.getBank().value_or(0) == rhs.getBank().value_or(0)
        && lhs.getAddressBinding() == rhs.getAddressBinding()
        && lhs.getWeightPage() == rhs.getWeightPage()
        && lhs.getLogicalBaseRow() == rhs.getLogicalBaseRow()
        && lhs.getOpcode() == rhs.getOpcode()
        && lhs.getPackedStream() == rhs.getPackedStream()
        && lhs.getRepeatCount() == rhs.getRepeatCount()
        && lhs.getRepeatInterval() == rhs.getRepeatInterval()
        && lhs.getAddressStride() == rhs.getAddressStride()
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
             "lhs_index", "lhs_immediate", "lhs_stream_source",
             "rhs_kind", "rhs_index", "rhs_stream_source",
             "rhs_immediate", "cast_target", "output_stream",
               "input_hemisphere", "output_hemisphere", "scale_binding",
               "chain_depth",
               "accumulator_reset", "accumulator_write", "accumulator_emit",
             "local_scalar_write"})
        if (lhs->getAttr(name) != rhs->getAttr(name))
            return false;
    return lhs.getRepeatCount() == 1 && rhs.getRepeatCount() == 1
        && lhs.getResult().getType() == rhs.getResult().getType();
}

bool same_sxm_body(schedule::SxmOp lhs, schedule::SxmOp rhs)
{
    for (llvm::StringRef name : {"hemisphere", "opcode",
               "source_streams", "destination_streams", "permute_map",
               "weight_layout", "output_row", "input_row", "output_tile"})
        if (lhs->getAttr(name) != rhs->getAttr(name))
            return false;
    return lhs.getRepeatCount().value_or(1) == 1
        && rhs.getRepeatCount().value_or(1) == 1;
}

const void* attribute_pointer(mlir::Attribute attribute)
{
    return attribute ? attribute.getAsOpaquePointer() : nullptr;
}

mlir::DictionaryAttr operation_pattern_attributes(
    mlir::Operation* operation, mlir::MLIRContext* context,
    llvm::ArrayRef<llvm::StringRef> excluded)
{
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (mlir::NamedAttribute attribute : operation->getAttrs())
        if (!llvm::is_contained(
                excluded, attribute.getName().strref()))
            attributes.push_back(attribute);
    return mlir::DictionaryAttr::get(context, attributes);
}

mlir::DictionaryAttr placement_without_wave_base(
    mlir::DictionaryAttr placement, mlir::MLIRContext* context)
{
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (mlir::NamedAttribute attribute : placement) {
        const llvm::StringRef name = attribute.getName().strref();
        if (name != "base_row" && name != "logical_base_row")
            attributes.push_back(attribute);
    }
    return mlir::DictionaryAttr::get(context, attributes);
}

int64_t placement_base_row(mlir::DictionaryAttr placement)
{
    if (auto value = placement.getAs<mlir::IntegerAttr>("base_row"))
        return value.getInt();
    return 0;
}

bool same_temporal_read_body(schedule::MemReadOp lhs,
    schedule::MemReadOp rhs, mlir::MLIRContext* context)
{
    return lhs.getInput() == rhs.getInput()
        && lhs.getDuration() == rhs.getDuration()
        && lhs.getStreamBase() == rhs.getStreamBase()
        && lhs.getStreamCount() == rhs.getStreamCount()
        && lhs.getRegisterId() == rhs.getRegisterId()
        && lhs.getRegisterIds() == rhs.getRegisterIds()
        && lhs.getDirection() == rhs.getDirection()
        && lhs.getRole() == rhs.getRole()
        && lhs.getAddress() == rhs.getAddress()
        && lhs.getBytes() == rhs.getBytes()
        && lhs.getOutput().getType() == rhs.getOutput().getType()
        && placement_without_wave_base(lhs.getPlacement(), context)
            == placement_without_wave_base(rhs.getPlacement(), context);
}

std::size_t temporal_read_hash(
    schedule::MemReadOp op, mlir::MLIRContext* context)
{
    return static_cast<std::size_t>(llvm::hash_combine(
        op.getInput().getAsOpaquePointer(), op.getDuration(),
        op.getStreamBase(), op.getStreamCount(), op.getRegisterId(),
        attribute_pointer(op->getAttr("register_ids")),
        attribute_pointer(op.getDirectionAttr()),
        attribute_pointer(op.getRoleAttr()),
        op.getAddress().getAsOpaquePointer(), op.getBytes(),
        op.getOutput().getType().getAsOpaquePointer(),
        placement_without_wave_base(op.getPlacement(), context)
            .getAsOpaquePointer()));
}

bool same_temporal_write_body(schedule::MemWriteOp lhs,
    schedule::MemWriteOp rhs, mlir::MLIRContext* context)
{
    return lhs.getDuration() == rhs.getDuration()
        && lhs.getStreamBase() == rhs.getStreamBase()
        && lhs.getStreamCount() == rhs.getStreamCount()
        && lhs.getRegisterId() == rhs.getRegisterId()
        && lhs.getDirection() == rhs.getDirection()
        && lhs.getAddress() == rhs.getAddress()
        && lhs.getBytes() == rhs.getBytes()
        && lhs.getOutput().getType() == rhs.getOutput().getType()
        && placement_without_wave_base(lhs.getPlacement(), context)
            == placement_without_wave_base(rhs.getPlacement(), context);
}

std::size_t temporal_write_hash(
    schedule::MemWriteOp op, mlir::MLIRContext* context)
{
    return static_cast<std::size_t>(llvm::hash_combine(
        op.getDuration(), op.getStreamBase(), op.getStreamCount(),
        op.getRegisterId(), attribute_pointer(op.getDirectionAttr()),
        op.getAddress().getAsOpaquePointer(), op.getBytes(),
        op.getOutput().getType().getAsOpaquePointer(),
        placement_without_wave_base(op.getPlacement(), context)
            .getAsOpaquePointer()));
}

bool same_mxm_attributes(schedule::MxmIssueOp lhs,
    schedule::MxmIssueOp rhs, bool ignoreWeightColumn,
    bool ignoreGroup)
{
    for (mlir::NamedAttribute attribute : lhs->getAttrs()) {
        const llvm::StringRef name = attribute.getName().strref();
        if (name == "cycle"
            || (ignoreWeightColumn && name == "weight_column")
            || (ignoreGroup
                && (name == "group_count"
                    || name == "group_interval")))
            continue;
        if (rhs->getAttr(attribute.getName()) != attribute.getValue())
            return false;
    }
    for (mlir::NamedAttribute attribute : rhs->getAttrs()) {
        const llvm::StringRef name = attribute.getName().strref();
        if (name == "cycle"
            || (ignoreWeightColumn && name == "weight_column")
            || (ignoreGroup
                && (name == "group_count"
                    || name == "group_interval")))
            continue;
        if (!lhs->hasAttr(attribute.getName())) return false;
    }
    return true;
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
                if (lhsQueue != rhsQueue) return lhsQueue < rhsQueue;
                const int64_t lhsBank = lhs.getBank().value_or(0);
                const int64_t rhsBank = rhs.getBank().value_or(0);
                if (lhsBank != rhsBank) return lhsBank < rhsBank;
                const int64_t lhsBinding =
                    lhs.getAddressBinding().value_or(-1);
                const int64_t rhsBinding =
                    rhs.getAddressBinding().value_or(-1);
                return lhsBinding != rhsBinding
                    ? lhsBinding < rhsBinding
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
        compressMemReadWaves(function, builder);
        compressMemWriteWaves(function, builder);
        compressMxmLoadGroups(function, builder);
        compressMxmDequantWaves(function, builder);
        compressMxmComputeWaves(function, builder);
        compressMxmIssues(function, *target, builder);
        compressVxm(function, builder);
        compressSxm(function, builder);
        function->setAttr(
            "ftlpu.schedule.compressed", builder.getUnitAttr());
    }

private:
    void compressMemReadWaves(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        struct ReadGroup {
            schedule::MemReadOp exemplar;
            llvm::SmallVector<schedule::MemReadOp> operations;
        };
        std::map<std::size_t, llvm::SmallVector<ReadGroup>> buckets;
        function.walk([&](schedule::MemReadOp op) {
            if (op.getWaveCount().value_or(1) != 1) return;
            auto& groups = buckets[temporal_read_hash(op, &getContext())];
            auto position = llvm::find_if(groups, [&](const ReadGroup& group) {
                return same_temporal_read_body(
                    group.exemplar, op, &getContext());
            });
            if (position == groups.end()) {
                groups.push_back(ReadGroup {op, {op}});
            } else {
                position->operations.push_back(op);
            }
        });

        llvm::SmallVector<schedule::MemReadOp> toErase;
        for (auto& [hash, groups] : buckets) {
            (void)hash;
            for (ReadGroup& group : groups) {
                llvm::sort(group.operations,
                    [](schedule::MemReadOp lhs,
                        schedule::MemReadOp rhs) {
                        if (lhs.getCycle() != rhs.getCycle())
                            return lhs.getCycle() < rhs.getCycle();
                        return placement_base_row(lhs.getPlacement())
                            < placement_base_row(rhs.getPlacement());
                    });

                llvm::SmallVector<schedule::MemReadOp> unique;
                for (schedule::MemReadOp op : group.operations) {
                    if (!unique.empty()
                        && unique.back().getCycle() == op.getCycle()
                        && placement_base_row(unique.back().getPlacement())
                            == placement_base_row(op.getPlacement())) {
                        op.getOutput().replaceAllUsesWith(
                            unique.back().getOutput());
                        toErase.push_back(op);
                        continue;
                    }
                    unique.push_back(op);
                }
                if (unique.size() < 2) continue;

                std::map<std::pair<int64_t, int64_t>, std::size_t> lookup;
                for (std::size_t index = 0; index < unique.size(); ++index)
                    lookup.emplace(
                        std::pair {static_cast<int64_t>(unique[index].getCycle()),
                            placement_base_row(unique[index].getPlacement())},
                        index);
                llvm::SmallVector<bool> used(unique.size(), false);
                for (std::size_t seed = 0; seed < unique.size(); ++seed) {
                    if (used[seed]) continue;
                    llvm::SmallVector<std::size_t> best {seed};
                    const int64_t seedCycle = unique[seed].getCycle();
                    const int64_t seedAddress =
                        placement_base_row(unique[seed].getPlacement());
                    const std::size_t candidateEnd = std::min(
                        unique.size(), seed + std::size_t {17});
                    for (std::size_t candidate = seed + 1;
                         candidate < candidateEnd; ++candidate) {
                        if (used[candidate]) continue;
                        const int64_t interval =
                            unique[candidate].getCycle() - seedCycle;
                        const int64_t stride =
                            placement_base_row(unique[candidate].getPlacement())
                            - seedAddress;
                        if (interval <= 0 || interval > 65535
                            || stride < -32768 || stride > 32767)
                            continue;
                        llvm::SmallVector<std::size_t> sequence {seed};
                        for (int64_t wave = 1; wave < 1023; ++wave) {
                            auto found = lookup.find(
                                {seedCycle + wave * interval,
                                    seedAddress + wave * stride});
                            if (found == lookup.end() || used[found->second])
                                break;
                            sequence.push_back(found->second);
                        }
                        if (sequence.size() > best.size())
                            best = std::move(sequence);
                    }
                    if (best.size() < 2) {
                        used[seed] = true;
                        continue;
                    }

                    schedule::MemReadOp representative = unique[best.front()];
                    for (std::size_t index : best)
                        if (unique[index]->isBeforeInBlock(representative))
                            representative = unique[index];
                    mlir::NamedAttrList placement(
                        unique[best.front()].getPlacement());
                    representative->setAttr("cycle",
                        builder.getI64IntegerAttr(seedCycle));
                    representative->setAttr("placement",
                        placement.getDictionary(&getContext()));
                    representative->setAttr("wave_count",
                        builder.getI64IntegerAttr(best.size()));
                    representative->setAttr("wave_interval",
                        builder.getI64IntegerAttr(
                            unique[best[1]].getCycle() - seedCycle));
                    representative->setAttr("wave_address_stride",
                        builder.getI64IntegerAttr(
                            placement_base_row(unique[best[1]].getPlacement())
                            - seedAddress));
                    for (std::size_t index : best) {
                        used[index] = true;
                        if (unique[index] == representative) continue;
                        unique[index].getOutput().replaceAllUsesWith(
                            representative.getOutput());
                        toErase.push_back(unique[index]);
                    }
                }
            }
        }
        for (schedule::MemReadOp operation : toErase)
            operation.erase();
    }

    void compressMemWriteWaves(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        struct WriteGroup {
            schedule::MemWriteOp exemplar;
            llvm::SmallVector<schedule::MemWriteOp> operations;
        };
        std::map<std::size_t, llvm::SmallVector<WriteGroup>> buckets;
        function.walk([&](schedule::MemWriteOp op) {
            if (op.getWaveCount().value_or(1) != 1) return;
            auto& groups = buckets[temporal_write_hash(op, &getContext())];
            auto position = llvm::find_if(groups,
                [&](const WriteGroup& group) {
                    return same_temporal_write_body(
                        group.exemplar, op, &getContext());
                });
            if (position == groups.end())
                groups.push_back(WriteGroup {op, {op}});
            else
                position->operations.push_back(op);
        });

        llvm::SmallVector<schedule::MemWriteOp> toErase;
        for (auto& [hash, groups] : buckets) {
            (void)hash;
            for (WriteGroup& group : groups) {
                llvm::sort(group.operations,
                    [](schedule::MemWriteOp lhs,
                        schedule::MemWriteOp rhs) {
                        if (lhs.getCycle() != rhs.getCycle())
                            return lhs.getCycle() < rhs.getCycle();
                        return placement_base_row(lhs.getPlacement())
                            < placement_base_row(rhs.getPlacement());
                    });
                llvm::SmallVector<schedule::MemWriteOp> unique;
                for (schedule::MemWriteOp op : group.operations) {
                    if (!unique.empty()
                        && unique.back().getCycle() == op.getCycle()
                        && placement_base_row(unique.back().getPlacement())
                            == placement_base_row(op.getPlacement())) {
                        op.getOutput().replaceAllUsesWith(
                            unique.back().getOutput());
                        toErase.push_back(op);
                        continue;
                    }
                    unique.push_back(op);
                }
                if (unique.size() < 2) continue;

                std::map<std::pair<int64_t, int64_t>, std::size_t> lookup;
                for (std::size_t index = 0; index < unique.size(); ++index)
                    lookup.emplace(
                        std::pair {static_cast<int64_t>(unique[index].getCycle()),
                            placement_base_row(unique[index].getPlacement())},
                        index);
                llvm::SmallVector<bool> used(unique.size(), false);
                for (std::size_t seed = 0; seed < unique.size(); ++seed) {
                    if (used[seed]) continue;
                    llvm::SmallVector<std::size_t> best {seed};
                    const int64_t seedCycle = unique[seed].getCycle();
                    const int64_t seedAddress =
                        placement_base_row(unique[seed].getPlacement());
                    const std::size_t candidateEnd = std::min(
                        unique.size(), seed + std::size_t {17});
                    for (std::size_t candidate = seed + 1;
                         candidate < candidateEnd; ++candidate) {
                        if (used[candidate]) continue;
                        const int64_t interval =
                            unique[candidate].getCycle() - seedCycle;
                        const int64_t stride = placement_base_row(
                            unique[candidate].getPlacement()) - seedAddress;
                        if (interval <= 0 || interval > 65535
                            || stride < -32768 || stride > 32767)
                            continue;
                        llvm::SmallVector<std::size_t> sequence {seed};
                        for (int64_t wave = 1; wave < 1023; ++wave) {
                            auto found = lookup.find(
                                {seedCycle + wave * interval,
                                    seedAddress + wave * stride});
                            if (found == lookup.end() || used[found->second])
                                break;
                            sequence.push_back(found->second);
                        }
                        if (sequence.size() > best.size())
                            best = std::move(sequence);
                    }
                    if (best.size() < 2) {
                        used[seed] = true;
                        continue;
                    }
                    schedule::MemWriteOp representative = unique[best.front()];
                    for (std::size_t index : best)
                        if (unique[index]->isBeforeInBlock(representative))
                            representative = unique[index];
                    representative->setAttr("cycle",
                        builder.getI64IntegerAttr(seedCycle));
                    representative->setAttr("placement",
                        unique[best.front()].getPlacement());
                    representative->setAttr("wave_count",
                        builder.getI64IntegerAttr(best.size()));
                    representative->setAttr("wave_interval",
                        builder.getI64IntegerAttr(
                            unique[best[1]].getCycle() - seedCycle));
                    representative->setAttr("wave_address_stride",
                        builder.getI64IntegerAttr(placement_base_row(
                            unique[best[1]].getPlacement()) - seedAddress));
                    for (std::size_t index : best) {
                        used[index] = true;
                        if (unique[index] == representative) continue;
                        unique[index].getOutput().replaceAllUsesWith(
                            representative.getOutput());
                        toErase.push_back(unique[index]);
                    }
                }
            }
        }
        for (schedule::MemWriteOp operation : toErase)
            operation.erase();
    }

    void compressMxmLoadGroups(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        struct Group {
            llvm::SmallVector<schedule::MxmLoadOp> operations;
        };
        std::map<std::pair<const void*, const void*>, Group> groups;
        function.walk([&](schedule::MxmLoadOp op) {
            if (op.getGroupCount().value_or(1) != 1) return;
            auto attributes = operation_pattern_attributes(op,
                &getContext(), {"cycle", "group_count", "group_interval"});
            groups[{op.getInput().getAsOpaquePointer(),
                attributes.getAsOpaquePointer()}]
                .operations.push_back(op);
        });
        llvm::SmallVector<schedule::MxmLoadOp> toErase;
        for (auto& [key, group] : groups) {
            (void)key;
            auto& operations = group.operations;
            llvm::sort(operations,
                [](schedule::MxmLoadOp lhs, schedule::MxmLoadOp rhs) {
                    return lhs.getCycle() < rhs.getCycle();
                });
            std::map<int64_t, std::size_t> lookup;
            for (std::size_t index = 0; index < operations.size(); ++index)
                lookup.emplace(operations[index].getCycle(), index);
            llvm::SmallVector<bool> used(operations.size(), false);
            for (std::size_t seed = 0; seed < operations.size(); ++seed) {
                if (used[seed]) continue;
                llvm::SmallVector<std::size_t> best {seed};
                const int64_t seedCycle = operations[seed].getCycle();
                const std::size_t candidateEnd = std::min(
                    operations.size(), seed + std::size_t {17});
                for (std::size_t candidate = seed + 1;
                     candidate < candidateEnd; ++candidate) {
                    if (used[candidate]) continue;
                    const int64_t interval =
                        operations[candidate].getCycle() - seedCycle;
                    if (interval <= 0 || interval > 65535) continue;
                    llvm::SmallVector<std::size_t> sequence {seed};
                    for (int64_t groupIndex = 1;
                         groupIndex < 1023; ++groupIndex) {
                        auto found = lookup.find(
                            seedCycle + groupIndex * interval);
                        if (found == lookup.end() || used[found->second])
                            break;
                        sequence.push_back(found->second);
                    }
                    if (sequence.size() > best.size())
                        best = std::move(sequence);
                }
                if (best.size() < 2) {
                    used[seed] = true;
                    continue;
                }
                schedule::MxmLoadOp representative = operations[best.front()];
                for (std::size_t index : best)
                    if (operations[index]->isBeforeInBlock(representative))
                        representative = operations[index];
                representative->setAttr("cycle",
                    builder.getI64IntegerAttr(seedCycle));
                representative->setAttr("group_count",
                    builder.getI64IntegerAttr(best.size()));
                representative->setAttr("group_interval",
                    builder.getI64IntegerAttr(
                        operations[best[1]].getCycle() - seedCycle));
                for (std::size_t index : best) {
                    used[index] = true;
                    if (operations[index] == representative) continue;
                    operations[index].getOutput().replaceAllUsesWith(
                        representative.getOutput());
                    toErase.push_back(operations[index]);
                }
            }
        }
        for (schedule::MxmLoadOp operation : toErase)
            operation.erase();
    }

    void compressMxmDequantWaves(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        struct Group {
            llvm::SmallVector<schedule::MxmDequantOp> operations;
        };
        std::map<const void*, Group> groups;
        function.walk([&](schedule::MxmDequantOp op) {
            if (op.getWaveCount().value_or(1) != 1) return;
            auto attributes = operation_pattern_attributes(op,
                &getContext(), {"cycle", "wave_count", "wave_interval"});
            groups[attributes.getAsOpaquePointer()].operations.push_back(op);
        });
        llvm::SmallVector<schedule::MxmDequantOp> toErase;
        for (auto& [key, group] : groups) {
            (void)key;
            auto& operations = group.operations;
            llvm::sort(operations,
                [](schedule::MxmDequantOp lhs,
                    schedule::MxmDequantOp rhs) {
                    return lhs.getCycle() < rhs.getCycle();
                });
            std::map<int64_t, std::size_t> lookup;
            for (std::size_t index = 0; index < operations.size(); ++index)
                lookup.emplace(operations[index].getCycle(), index);
            llvm::SmallVector<bool> used(operations.size(), false);
            for (std::size_t seed = 0; seed < operations.size(); ++seed) {
                if (used[seed]) continue;
                llvm::SmallVector<std::size_t> best {seed};
                const int64_t seedCycle = operations[seed].getCycle();
                const std::size_t candidateEnd = std::min(
                    operations.size(), seed + std::size_t {17});
                for (std::size_t candidate = seed + 1;
                     candidate < candidateEnd; ++candidate) {
                    if (used[candidate]) continue;
                    const int64_t interval =
                        operations[candidate].getCycle() - seedCycle;
                    if (interval <= 0 || interval > 65535) continue;
                    llvm::SmallVector<std::size_t> sequence {seed};
                    for (int64_t wave = 1; wave < 1023; ++wave) {
                        auto found = lookup.find(seedCycle + wave * interval);
                        if (found == lookup.end() || used[found->second])
                            break;
                        sequence.push_back(found->second);
                    }
                    if (sequence.size() > best.size())
                        best = std::move(sequence);
                }
                if (best.size() < 2) {
                    used[seed] = true;
                    continue;
                }
                schedule::MxmDequantOp representative =
                    operations[best.front()];
                representative->setAttr("wave_count",
                    builder.getI64IntegerAttr(best.size()));
                representative->setAttr("wave_interval",
                    builder.getI64IntegerAttr(
                        operations[best[1]].getCycle() - seedCycle));
                for (std::size_t index : best) {
                    used[index] = true;
                    if (index != best.front())
                        toErase.push_back(operations[index]);
                }
            }
        }
        for (schedule::MxmDequantOp operation : toErase)
            operation.erase();
    }

    void compressMxmComputeWaves(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        struct ComputePair {
            schedule::MxmComputeOp compute;
            schedule::MxmAccumulateOp accumulator;
        };
        using GroupKey = std::tuple<const void*, const void*, const void*,
            const void*, int64_t, int64_t>;
        std::map<GroupKey, llvm::SmallVector<ComputePair>> groups;
        function.walk([&](schedule::MxmComputeOp compute) {
            if (compute.getWaveCount().value_or(1) != 1) return;
            schedule::MxmAccumulateOp accumulator;
            for (mlir::Operation* user : compute.getResult().getUsers()) {
                auto candidate =
                    llvm::dyn_cast<schedule::MxmAccumulateOp>(user);
                if (!candidate || accumulator) return;
                accumulator = candidate;
            }
            if (!accumulator
                || accumulator.getWaveCount().value_or(1) != 1)
                return;
            auto computeAttributes = operation_pattern_attributes(compute,
                &getContext(), {"cycle", "result_cycle", "wave_count",
                    "wave_interval", "wave_accumulator_address_stride"});
            auto accumulatorAttributes = operation_pattern_attributes(
                accumulator, &getContext(), {"cycle", "accumulator_address",
                    "wave_count", "wave_interval",
                    "wave_accumulator_address_stride"});
            groups[{compute.getActivation().getAsOpaquePointer(),
                compute.getWeight().getAsOpaquePointer(),
                computeAttributes.getAsOpaquePointer(),
                accumulatorAttributes.getAsOpaquePointer(),
                compute.getResultCycle() - compute.getCycle(),
                accumulator.getCycle() - compute.getCycle()}]
                .push_back({compute, accumulator});
        });

        llvm::SmallVector<schedule::MxmAccumulateOp> accumulatorsToErase;
        llvm::SmallVector<schedule::MxmComputeOp> computesToErase;
        for (auto& [key, pairs] : groups) {
            (void)key;
            llvm::sort(pairs, [](ComputePair lhs, ComputePair rhs) {
                if (lhs.compute.getCycle() != rhs.compute.getCycle())
                    return lhs.compute.getCycle() < rhs.compute.getCycle();
                return lhs.accumulator.getAccumulatorAddress()
                    < rhs.accumulator.getAccumulatorAddress();
            });
            std::map<std::pair<int64_t, int64_t>, std::size_t> lookup;
            for (std::size_t index = 0; index < pairs.size(); ++index)
                lookup.emplace(
                    std::pair {
                        static_cast<int64_t>(pairs[index].compute.getCycle()),
                        static_cast<int64_t>(pairs[index].accumulator
                                .getAccumulatorAddress())},
                    index);
            llvm::SmallVector<bool> used(pairs.size(), false);
            for (std::size_t seed = 0; seed < pairs.size(); ++seed) {
                if (used[seed]) continue;
                llvm::SmallVector<std::size_t> best {seed};
                const int64_t seedCycle = pairs[seed].compute.getCycle();
                const int64_t seedAddress =
                    pairs[seed].accumulator.getAccumulatorAddress();
                const std::size_t candidateEnd =
                    std::min(pairs.size(), seed + std::size_t {17});
                for (std::size_t candidate = seed + 1;
                     candidate < candidateEnd; ++candidate) {
                    if (used[candidate]) continue;
                    const int64_t interval =
                        pairs[candidate].compute.getCycle() - seedCycle;
                    const int64_t stride = pairs[candidate].accumulator
                            .getAccumulatorAddress()
                        - seedAddress;
                    if (interval <= 0 || interval > 65535) continue;
                    llvm::SmallVector<std::size_t> sequence {seed};
                    for (int64_t wave = 1; wave < 1023; ++wave) {
                        auto found = lookup.find(
                            {seedCycle + wave * interval,
                                seedAddress + wave * stride});
                        if (found == lookup.end() || used[found->second])
                            break;
                        sequence.push_back(found->second);
                    }
                    if (sequence.size() > best.size())
                        best = std::move(sequence);
                }
                if (best.size() < 2) {
                    used[seed] = true;
                    continue;
                }

                std::size_t representativeIndex = best.front();
                for (std::size_t index : best)
                    if (pairs[index].compute->isBeforeInBlock(
                            pairs[representativeIndex].compute))
                        representativeIndex = index;
                auto representative = pairs[representativeIndex];
                const int64_t interval =
                    pairs[best[1]].compute.getCycle() - seedCycle;
                const int64_t stride = pairs[best[1]].accumulator
                        .getAccumulatorAddress()
                    - seedAddress;
                const int64_t resultOffset =
                    representative.compute.getResultCycle()
                    - representative.compute.getCycle();
                const int64_t accumulatorOffset =
                    representative.accumulator.getCycle()
                    - representative.compute.getCycle();
                representative.compute->setAttr(
                    "cycle", builder.getI64IntegerAttr(seedCycle));
                representative.compute->setAttr("result_cycle",
                    builder.getI64IntegerAttr(seedCycle + resultOffset));
                representative.compute->setAttr("wave_count",
                    builder.getI64IntegerAttr(best.size()));
                representative.compute->setAttr("wave_interval",
                    builder.getI64IntegerAttr(interval));
                representative.compute->setAttr(
                    "wave_accumulator_address_stride",
                    builder.getI64IntegerAttr(stride));
                representative.accumulator->setAttr(
                    "cycle", builder.getI64IntegerAttr(
                        seedCycle + accumulatorOffset));
                representative.accumulator->setAttr("accumulator_address",
                    builder.getI64IntegerAttr(seedAddress));
                representative.accumulator->setAttr("wave_count",
                    builder.getI64IntegerAttr(best.size()));
                representative.accumulator->setAttr("wave_interval",
                    builder.getI64IntegerAttr(interval));
                representative.accumulator->setAttr(
                    "wave_accumulator_address_stride",
                    builder.getI64IntegerAttr(stride));

                for (std::size_t index : best) {
                    used[index] = true;
                    if (index == representativeIndex) continue;
                    pairs[index].accumulator.getOutput().replaceAllUsesWith(
                        representative.accumulator.getOutput());
                    accumulatorsToErase.push_back(pairs[index].accumulator);
                    computesToErase.push_back(pairs[index].compute);
                }
            }
        }
        for (schedule::MxmAccumulateOp operation : accumulatorsToErase)
            operation.erase();
        for (schedule::MxmComputeOp operation : computesToErase)
            operation.erase();
    }

    void compressMxmIssues(mlir::func::FuncOp function,
        const target::LPUTargetModel& target, mlir::Builder& builder)
    {
        llvm::SmallVector<schedule::MxmIssueOp> operations;
        function.walk(
            [&](schedule::MxmIssueOp op) { operations.push_back(op); });
        llvm::sort(operations,
            [](schedule::MxmIssueOp lhs, schedule::MxmIssueOp rhs) {
                if (lhs.getUnitId() != rhs.getUnitId())
                    return lhs.getUnitId() < rhs.getUnitId();
                if (lhs.getOpcode() != rhs.getOpcode())
                    return lhs.getOpcode() < rhs.getOpcode();
                return lhs.getCycle() < rhs.getCycle();
            });

        // Form an IW column wave. Other MXM fields stay invariant and the
        // hardware induction variable changes only the weight column.
        llvm::SmallVector<schedule::MxmIssueOp> erased;
        for (std::size_t index = 0; index < operations.size();) {
            schedule::MxmIssueOp first = operations[index];
            if (first.getOpcode() != "iw"
                || first.getWaveCount().value_or(1) != 1
                || first.getGroupCount().value_or(1) != 1
                || index + 1 >= operations.size()) {
                ++index;
                continue;
            }
            schedule::MxmIssueOp second = operations[index + 1];
            if (!same_mxm_attributes(first, second, true, false)) {
                ++index;
                continue;
            }
            const int64_t interval = second.getCycle() - first.getCycle();
            const int64_t stride =
                second.getWeightColumn() - first.getWeightColumn();
            if (interval <= 0 || interval > kMaxRepeatInterval
                || stride == 0) {
                ++index;
                continue;
            }
            std::size_t end = index + 2;
            while (end < operations.size()
                && static_cast<int64_t>(end - index) < kMaxRepeatCount) {
                const int64_t wave = static_cast<int64_t>(end - index);
                schedule::MxmIssueOp next = operations[end];
                if (!same_mxm_attributes(first, next, true, false)
                    || next.getCycle() != first.getCycle() + wave * interval
                    || next.getWeightColumn()
                        != first.getWeightColumn() + wave * stride)
                    break;
                ++end;
            }
            const int64_t count = static_cast<int64_t>(end - index);
            const int64_t finalColumn = first.getWeightColumn()
                + (count - 1) * stride;
            if (finalColumn < 0
                || finalColumn >= target.throughput().tile_rows) {
                ++index;
                continue;
            }
            first->setAttr(
                "wave_count", builder.getI64IntegerAttr(count));
            first->setAttr(
                "wave_interval", builder.getI64IntegerAttr(interval));
            first->setAttr("wave_weight_column_stride",
                builder.getI64IntegerAttr(stride));
            for (std::size_t erase = index + 1; erase < end; ++erase)
                erased.push_back(operations[erase]);
            index = end;
        }
        for (schedule::MxmIssueOp operation : erased)
            operation.erase();

        // Group equal issue bodies at a regular interval. A wave plus an
        // existing inner repeat would require three hardware loop dimensions.
        operations.clear();
        function.walk(
            [&](schedule::MxmIssueOp op) { operations.push_back(op); });
        llvm::sort(operations,
            [](schedule::MxmIssueOp lhs, schedule::MxmIssueOp rhs) {
                if (lhs.getUnitId() != rhs.getUnitId())
                    return lhs.getUnitId() < rhs.getUnitId();
                if (lhs.getOpcode() != rhs.getOpcode())
                    return lhs.getOpcode() < rhs.getOpcode();
                return lhs.getCycle() < rhs.getCycle();
            });
        erased.clear();
        for (std::size_t index = 0; index < operations.size();) {
            schedule::MxmIssueOp first = operations[index];
            if (first.getGroupCount().value_or(1) != 1
                || (first.getWaveCount().value_or(1) > 1
                    && first.getRepeatCount() > 1)
                || index + 1 >= operations.size()) {
                ++index;
                continue;
            }
            schedule::MxmIssueOp second = operations[index + 1];
            if (!same_mxm_attributes(first, second, false, true)) {
                ++index;
                continue;
            }
            const int64_t interval = second.getCycle() - first.getCycle();
            if (interval <= 0) {
                ++index;
                continue;
            }
            std::size_t end = index + 2;
            while (end < operations.size()
                && static_cast<int64_t>(end - index) < kMaxRepeatCount) {
                const int64_t group = static_cast<int64_t>(end - index);
                schedule::MxmIssueOp next = operations[end];
                if (!same_mxm_attributes(first, next, false, true)
                    || next.getCycle()
                        != first.getCycle() + group * interval)
                    break;
                ++end;
            }
            first->setAttr("group_count",
                builder.getI64IntegerAttr(end - index));
            first->setAttr(
                "group_interval", builder.getI64IntegerAttr(interval));
            for (std::size_t erase = index + 1; erase < end; ++erase)
                erased.push_back(operations[erase]);
            index = end;
        }
        for (schedule::MxmIssueOp operation : erased)
            operation.erase();
    }

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
              // VXM repeat_count is a datapath hold count, not an ICU
              // issue-loop with a programmable interval.
              if (interval != 1) {
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

    void compressSxm(
        mlir::func::FuncOp function, mlir::Builder& builder)
    {
        llvm::SmallVector<schedule::SxmOp> operations;
        function.walk(
            [&](schedule::SxmOp op) { operations.push_back(op); });
        llvm::sort(operations,
            [](schedule::SxmOp lhs, schedule::SxmOp rhs) {
                if (lhs.getHemisphere() != rhs.getHemisphere())
                    return lhs.getHemisphere() < rhs.getHemisphere();
                if (lhs.getOpcode() != rhs.getOpcode())
                    return lhs.getOpcode() < rhs.getOpcode();
                return lhs.getCycle() < rhs.getCycle();
            });

        llvm::SmallVector<schedule::SxmOp> toErase;
        for (std::size_t index = 0; index < operations.size();) {
            schedule::SxmOp first = operations[index];
            std::size_t end = index + 1;
            if (end >= operations.size()
                || !same_sxm_body(first, operations[end])) {
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
                if (!same_sxm_body(first, operations[end])
                    || operations[end].getCycle()
                        != first.getCycle() + repeat * interval)
                    break;
                ++end;
            }
            first->setAttr("repeat_count",
                builder.getI64IntegerAttr(end - index));
            first->setAttr(
                "repeat_interval", builder.getI64IntegerAttr(interval));
            for (std::size_t erase = index + 1; erase < end; ++erase)
                toErase.push_back(operations[erase]);
            index = end;
        }
        for (schedule::SxmOp operation : toErase)
            operation.erase();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_compress_schedule_pass()
{
    return std::make_unique<CompressSchedulePass>();
}

} // namespace ftlpu::compiler
