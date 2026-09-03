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
        || queryHeads * headDim != hidden
        || headDim % tile != 0 || memory.banks_per_slice < 2
        || initialBank < 0 || initialBank >= memory.banks_per_slice
        || throughput.mxms_per_hemisphere != 1
        || loadSlices <= 0 || storage.size() < 2 * loadSlices
        || storage.size() % loadSlices != 0
        || streams.c2c_streams_per_direction <= 0)
        return mlir::failure();

    const int64_t groups = storage.size() / loadSlices;
    const int64_t reductionBlocks = hidden / tile;
    const int64_t projectionRowsPerItem = reductionBlocks * 8;
    const int64_t outputRowsPerItem = reductionBlocks * 4;
    const int64_t projectionItemsPerGroup =
        memory.sram_depth_rows / projectionRowsPerItem;
    const int64_t outputItemsPerGroup =
        memory.sram_depth_rows / outputRowsPerItem;
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
    const int64_t valueGroup = keyGroup + 1;
    const int64_t projectionGroupsUsed = valueGroup
        + divideCeil(valueItems, projectionItemsPerGroup);
    const int64_t outputGroups = divideCeil(
        outputItems, outputItemsPerGroup);
    if (projectionGroupsUsed > groups || outputGroups > groups)
        return mlir::failure();

    const auto transferCycles = [&](int64_t columns) {
        return target.external_read_transfer_cycles(hidden * columns);
    };
    const auto placement = [&](AttentionWeightTileKind kind,
                               int64_t bank, int64_t baseRow,
                               int64_t groupBegin, int64_t groupCount,
                               int64_t itemsPerGroup, int64_t itemCount,
                               int64_t rowsPerItem, int64_t columns) {
        const int64_t residentRows =
            std::min(itemsPerGroup, itemCount) * rowsPerItem;
        return AttentionWeightTilePlacement {kind, bank, 0, baseRow,
            std::min(memory.sram_depth_rows - baseRow,
                residentRows),
            groupBegin, groupCount, itemsPerGroup, itemCount,
            rowsPerItem, transferCycles(columns)};
    };

    AttentionWeightTilePlan result;
    result.bank_rows = memory.sram_depth_rows;
    result.load_slice_count = loadSlices;
    result.storage_slice_count = storage.size();
    result.slice_group_count = groups;
    result.transfer_phase_count = 2;
    result.storage_slices.assign(storage.begin(), storage.end());
    result.placements = {
        placement(AttentionWeightTileKind::Query, initialBank, 0,
            0, queryGroups, projectionItemsPerGroup, queryItems,
            projectionRowsPerItem, queryHeads * headDim),
        placement(AttentionWeightTileKind::Key, initialBank, keyBase,
            keyGroup, 1, projectionItemsPerGroup, keyItems,
            projectionRowsPerItem, kvHeads * headDim),
        placement(AttentionWeightTileKind::Value, initialBank, 0,
            valueGroup, divideCeil(valueItems, projectionItemsPerGroup),
            projectionItemsPerGroup, valueItems,
            projectionRowsPerItem, kvHeads * headDim),
        placement(AttentionWeightTileKind::Output,
            (initialBank + 1) % memory.banks_per_slice, 0,
            0, outputGroups, outputItemsPerGroup, outputItems,
            outputRowsPerItem, hidden),
    };
    return result;
}

} // namespace ftlpu::compiler::tensor
