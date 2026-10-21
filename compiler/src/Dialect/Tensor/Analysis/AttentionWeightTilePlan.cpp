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
        || (throughput.mxms_per_hemisphere != 1
            && throughput.mxms_per_hemisphere != 2)
        || loadSlices <= 0 || storage.size() < 2 * loadSlices
        || storage.size() % loadSlices != 0
        || streams.c2c_streams_per_direction <= 0)
        return mlir::failure();

    const int64_t groups = storage.size() / loadSlices;
    const int64_t projectionReductionBlocks = hidden / tile;
    const int64_t outputReductionBlocks = queryHeads * headDim / tile;
    // Each hemisphere stores two MXM output slots.  A 16-row MXM needs two
    // weight pulses per slot, so all four rows can be packed without the
    // four unused rows inherited from the 32-row layout.
    const int64_t projectionRowsPerItem =
        projectionReductionBlocks * 2 * (tile / 8);
    const int64_t outputRowsPerItem = outputReductionBlocks
        * throughput.tile_rows;
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
    const bool multiPage =
        std::max({queryGroups, keyGroups, valueGroups, outputGroups})
            > groups;

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
    if (!multiPage) {
        for (std::size_t kind = 0; kind < result.placements.size(); ++kind)
            result.pages[kind].push_back(result.placements[kind]);
        return result;
    }

    // An individual projection can exceed the entire bank on small SRAM
    // targets. Page it by output item, never by an arbitrary row boundary:
    // all reductions for an item remain resident while that item computes.
    // Q -> V -> K -> O is the schedule's consumption order. Alternating
    // banks across *pages* leaves the next transfer a different destination
    // from the currently consumed page.
    const std::array<AttentionWeightTileKind, 4> order = {
        AttentionWeightTileKind::Query, AttentionWeightTileKind::Value,
        AttentionWeightTileKind::Key, AttentionWeightTileKind::Output};
    int64_t physicalPage = 0;
    result.transfer_phase_count = 0;
    result.bank_level_ping_pong = true;
    for (AttentionWeightTileKind kind : order) {
        const std::size_t index = static_cast<std::size_t>(kind);
        const auto& whole = result.placements[index];
        const int64_t itemsPerPage = groups * whole.items_per_slice_group;
        if (itemsPerPage <= 0) return mlir::failure();
        for (int64_t begin = 0; begin < whole.item_count;
             begin += itemsPerPage) {
            const int64_t count = std::min(
                itemsPerPage, whole.item_count - begin);
            AttentionWeightTilePlacement page = whole;
            page.bank = (initialBank + physicalPage)
                % memory.banks_per_slice;
            page.page_index = static_cast<int64_t>(result.pages[index].size());
            page.base_row = 0;
            page.slice_group_begin = 0;
            page.slice_group_count = divideCeil(
                count, whole.items_per_slice_group);
            page.item_begin = begin;
            page.item_count = count;
            page.rows = std::min(count, whole.items_per_slice_group)
                * whole.rows_per_item;
            // The byte count is proportional to the output items in this
            // page; the final short page must not charge a full projection.
            const int64_t totalColumns = kind == AttentionWeightTileKind::Output
                ? hidden : kind == AttentionWeightTileKind::Query
                    ? queryHeads * headDim : kvHeads * headDim;
            const int64_t reductionRows = kind == AttentionWeightTileKind::Output
                ? queryHeads * headDim : hidden;
            page.transfer_cycles = transferCycles(reductionRows,
                divideCeil(totalColumns * count, whole.item_count));
            if (page.rows > memory.sram_depth_rows
                || page.slice_group_count > groups)
                return mlir::failure();
            result.pages[index].push_back(page);
            ++physicalPage;
            ++result.transfer_phase_count;
        }
        result.placements[index] = result.pages[index].front();
    }
    return result;
}

} // namespace ftlpu::compiler::tensor
