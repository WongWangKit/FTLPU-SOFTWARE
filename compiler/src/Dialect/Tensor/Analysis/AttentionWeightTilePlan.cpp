#include "ftlpu/compiler/Dialect/Tensor/Analysis/attention_weight_tile_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::tensor {
namespace {
int64_t divideCeil(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}
} // namespace

mlir::FailureOr<AttentionWeightTilePlan> planAttentionWeightTiles(
    int64_t hidden, int64_t queryHeads, int64_t kvHeads,
    int64_t headDim, int64_t initialBank,
    const target::LPUTargetModel& target)
{
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const auto& streams = target.streams();
    const int64_t tile = throughput.mxm_rows;
    const int64_t loadSlices = memory.w8a16_weight_slice_count;
    const auto storage = target.weight_storage_slices();
    if (hidden <= 0 || queryHeads <= 0 || kvHeads <= 0
        || headDim <= 0 || hidden % tile != 0
        || headDim % tile != 0 || memory.banks_per_slice < 2
        || initialBank < 0 || initialBank >= memory.banks_per_slice
        || throughput.mxms_per_hemisphere != 1
        || loadSlices <= 0 || storage.size() < 2 * loadSlices
        || storage.size() % loadSlices != 0
        || streams.c2c_streams_per_direction <= 0)
        return mlir::failure();

    const int64_t groups = storage.size() / loadSlices;
    const int64_t projectionReductionBlocks = hidden / tile;
    const int64_t outputReductionBlocks = queryHeads * headDim / tile;
    const int64_t projectionRowsPerItem = projectionReductionBlocks * 8;
    const int64_t outputRowsPerItem = outputReductionBlocks * 4;
    const int64_t projectionItemsPerGroup =
        memory.sram_depth_rows / projectionRowsPerItem;
    // W8 output reads use a doubled physical reduction stride while adjacent
    // logical output groups begin at the compact row stride.  Reserve one
    // compact item of tail headroom so the final doubled reduction remains
    // inside the bank (for example, 20 rather than 21 Phi output groups in
    // an 8192-row slot).
    const int64_t outputItemsPerGroup = std::max<int64_t>(
        1, memory.sram_depth_rows / outputRowsPerItem - 1);
    if (projectionItemsPerGroup <= 0 || outputItemsPerGroup <= 0)
        return mlir::failure();

    const int64_t queryItems = divideCeil(queryHeads * headDim, 4 * tile);
    const int64_t keyItems = divideCeil(kvHeads * headDim, 4 * tile);
    const int64_t valueItems = keyItems;
    const int64_t outputItems = hidden
        / (memory.hemispheres * tile);
    const int64_t queryGroups = divideCeil(
        queryItems, projectionItemsPerGroup);
    const int64_t queryTail = queryItems % projectionItemsPerGroup;
    const bool keyFitsQueryTail = queryTail != 0
        && queryTail + keyItems <= projectionItemsPerGroup;
    const int64_t keyGroup = keyFitsQueryTail
        ? queryGroups - 1 : queryGroups;
    const int64_t keyBase = keyFitsQueryTail
        ? queryTail * projectionRowsPerItem : 0;
    const int64_t keyGroups = keyFitsQueryTail
        ? 1 : divideCeil(keyItems, projectionItemsPerGroup);
    const int64_t valueGroups = divideCeil(
        valueItems, projectionItemsPerGroup);
    const int64_t valueGroup = keyGroup + keyGroups;
    const int64_t projectionGroupsUsed = valueGroup
        + valueGroups;
    const int64_t outputGroups = divideCeil(
        outputItems, outputItemsPerGroup);
    const bool disjointResidency =
        projectionGroupsUsed + outputGroups <= groups;
    if (!disjointResidency
        && std::max({queryGroups, keyGroups, valueGroups, outputGroups})
            > groups)
        return mlir::failure();

    const auto transferCycles = [&](int64_t rows, int64_t columns) {
        return target.external_read_transfer_cycles(rows * columns);
    };
    const auto placement = [&](AttentionWeightTileKind kind,
                               int64_t bank, int64_t baseRow,
                               int64_t groupBegin, int64_t groupCount,
                               int64_t itemsPerGroup, int64_t itemCount,
                               int64_t rowsPerItem, int64_t rows,
                               int64_t columns) {
        const int64_t residentRows =
            std::min(itemsPerGroup, itemCount) * rowsPerItem;
        return AttentionWeightTilePlacement {kind, bank, 0, baseRow,
            std::min(memory.sram_depth_rows - baseRow,
                residentRows),
            groupBegin, groupCount, itemsPerGroup, itemCount,
            rowsPerItem, transferCycles(rows, columns)};
    };

    AttentionWeightTilePlan result;
    result.bank_rows = memory.sram_depth_rows;
    result.load_slice_count = loadSlices;
    result.storage_slice_count = storage.size();
    result.slice_group_count = groups;
    // Prefer one disjoint attention epoch when it fits.  Wider dense models
    // (for example Phi-4 Mini) need more slice groups than exist in one SRAM
    // bank. Alternate Q -> V -> K -> O across two banks instead: K may
    // replace Q while V computes, and O may replace V while K computes.
    // Page-ready events protect consumers; the compiler still describes
    // each bank's physical release so ingress never overwrites live weights.
    result.transfer_phase_count = disjointResidency ? 2 : 4;
    result.bank_level_ping_pong = !disjointResidency;
    const int64_t alternateBank = (initialBank + 1) % memory.banks_per_slice;
    result.storage_slices.assign(storage.begin(), storage.end());
    const int64_t queryGroupBegin = 0;
    const int64_t keyGroupBegin = disjointResidency ? keyGroup : 0;
    const int64_t keyRowBase = disjointResidency ? keyBase : 0;
    const int64_t valueGroupBegin = disjointResidency ? valueGroup : 0;
    const int64_t outputGroupBegin =
        disjointResidency ? projectionGroupsUsed : 0;
    result.placements = {
        placement(AttentionWeightTileKind::Query, initialBank, 0,
            queryGroupBegin, queryGroups, projectionItemsPerGroup, queryItems,
            projectionRowsPerItem, hidden, queryHeads * headDim),
        placement(AttentionWeightTileKind::Key, initialBank, keyRowBase,
            keyGroupBegin, keyGroups, projectionItemsPerGroup, keyItems,
            projectionRowsPerItem, hidden, kvHeads * headDim),
        placement(AttentionWeightTileKind::Value,
            disjointResidency ? initialBank : alternateBank, 0,
            valueGroupBegin, valueGroups,
            projectionItemsPerGroup, valueItems,
            projectionRowsPerItem, hidden, kvHeads * headDim),
        // When the whole epoch fits, O occupies a disjoint group in its
        // bank. Otherwise O reuses V's bank while K consumes the other bank.
        placement(AttentionWeightTileKind::Output,
            disjointResidency ? initialBank : alternateBank, 0,
            outputGroupBegin, outputGroups,
            outputItemsPerGroup, outputItems,
            outputRowsPerItem, queryHeads * headDim, hidden),
    };
    return result;
}

} // namespace ftlpu::compiler::tensor
