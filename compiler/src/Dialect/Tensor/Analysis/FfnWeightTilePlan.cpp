#include "ftlpu/compiler/Dialect/Tensor/Analysis/ffn_weight_tile_plan.hpp"
#include "ftlpu/compiler/Target/weight_page_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::tensor {
namespace {
int64_t divideCeil(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}
} // namespace

mlir::FailureOr<FfnWeightTilePlan> planFfnWeightTiles(
    FfnWeightShape shape, const target::LPUTargetModel& target,
    int64_t initialBank, bool streamingDownDoubleBuffer)
{
    const auto& memory = target.memory();
    const auto& streams = target.streams();
    const auto& throughput = target.throughput();
    const int64_t tile = throughput.mxm_rows;
    const int64_t rowsPerWeightTile = throughput.tile_rows;
    const int64_t weightSlices = memory.w8a16_weight_slice_count;
    const int64_t storageSlices =
        static_cast<int64_t>(target.weight_storage_slices().size());
    if (!target.supports_w8a16_ffn_shape(
            shape.m, shape.k, shape.hidden, shape.n)
        || memory.banks_per_slice < 2 || initialBank < 0
        || initialBank >= memory.banks_per_slice
        || memory.sram_depth_rows <= 0
        || memory.bytes_per_word <= 0 || memory.hemispheres <= 0
        || tile <= 0 || rowsPerWeightTile <= 0 || weightSlices <= 0
        || storageSlices < 2 * weightSlices
        || storageSlices % weightSlices != 0
        || streams.c2c_streams_per_direction <= 0
        || throughput.mxms_per_hemisphere < 1
        || throughput.mxms_per_hemisphere > 2)
        return mlir::failure();

    const int64_t reductionBlocks = shape.k / tile;
    const int64_t projectionRowsPerWave =
        reductionBlocks * rowsPerWeightTile;
    // With two MXMs per hemisphere, each hemisphere currently owns a full
    // Gate/Up output copy; its page sequence must cover every physical pair.
    const int64_t projectionWaves = shape.hidden
        / ((throughput.mxms_per_hemisphere == 1
               ? memory.hemispheres : 1) * tile);
    const int64_t sliceGroupCount = storageSlices / weightSlices;
    if (sliceGroupCount % 2 != 0) return mlir::failure();
    const target::weight_pages::Geometry pageGeometry{
        memory.sram_depth_rows, memory.banks_per_slice, sliceGroupCount};
    const int64_t splitProjectionGroupsPerRole = sliceGroupCount / 2;
    // Two adjacent 64-column projection waves share one doubled physical
    // reduction-row stripe. Keep the per-group capacity even and leave one
    // complete pair below the raw SRAM limit; an odd tail still occupies the
    // full doubled stripe and therefore does not save rows.
    const target::weight_pages::ItemLayout projectionLayout{
        projectionRowsPerWave, 2, projectionRowsPerWave};
    const int64_t projectionWavesPerGroup =
        target::weight_pages::itemsPerGroup(pageGeometry, projectionLayout);
    if (projectionWavesPerGroup <= 0) return mlir::failure();
    const bool bankLevelPingPong = projectionWaves
        <= sliceGroupCount * projectionWavesPerGroup;
    const int64_t projectionGroupsPerRole = bankLevelPingPong
        ? sliceGroupCount : splitProjectionGroupsPerRole;
    const int64_t projectionWavesPerPage =
        projectionGroupsPerRole * projectionWavesPerGroup;
    if (projectionWavesPerPage <= 0) return mlir::failure();

    FfnWeightTilePlan result;
    result.bank_rows = memory.sram_depth_rows;
    result.bank_bytes = memory.sram_depth_rows * memory.bytes_per_word;
    result.weight_load_slice_count = weightSlices;
    result.weight_storage_slice_count = storageSlices;
    result.slice_group_count = sliceGroupCount;
    result.projection_wave_count = projectionWaves;
    result.projection_waves_per_page = projectionWavesPerPage;
    result.projection_slice_groups_per_role = projectionGroupsPerRole;
    result.projection_gate_slice_group_base = 0;
    result.projection_up_slice_group_base = bankLevelPingPong
        ? 0 : projectionGroupsPerRole;
    result.projection_waves_per_slice_group = projectionWavesPerGroup;
    result.bank_level_ping_pong = bankLevelPingPong;
    const auto appendPage = [&](FfnWeightTilePage page,
                                int64_t bindingPageIndex) {
        page.index = static_cast<int64_t>(result.pages.size());
        if (page.bank < 0)
            page.bank =
                (initialBank + bindingPageIndex) % memory.banks_per_slice;
        for (FfnWeightTileSpan& span : page.spans)
            if (span.bank < 0) span.bank = page.bank;
        page.transfer_cycles = target.external_read_transfer_cycles(
            page.transfer_vectors * memory.bytes_per_word);
        result.pages.push_back(std::move(page));
    };

    const auto projectionPages = target::weight_pages::splitItems(
        projectionWaves, projectionGroupsPerRole, pageGeometry, projectionLayout);
    if (projectionPages.empty()) return mlir::failure();
    for (const auto& itemPage : projectionPages) {
        const int64_t wave = itemPage.item_begin;
        const int64_t count = itemPage.item_count;
        const int64_t rows = itemPage.rows_per_group;
        FfnWeightTilePage page {-1, -1, 0, rows,
            memory.hemispheres * weightSlices * count
                * projectionRowsPerWave * 2,
            0, {}};
        page.spans.push_back({FfnWeightTileKind::Gate,
            bankLevelPingPong ? initialBank : -1, 0, 0,
            projectionGroupsPerRole, projectionWavesPerGroup, wave,
            count, 0, reductionBlocks, 1, rows});
        page.spans.push_back({FfnWeightTileKind::Up,
            bankLevelPingPong
                ? (initialBank + 1) % memory.banks_per_slice : -1,
            0, result.projection_up_slice_group_base,
            projectionGroupsPerRole,
            projectionWavesPerGroup, wave, count, 0, reductionBlocks,
            1, rows});
        appendPage(std::move(page), wave / projectionWavesPerPage);
    }

    constexpr int64_t downOutputBlocksPerHemisphere = 2;
    const int64_t downColumnsPerWave = memory.hemispheres
        * downOutputBlocksPerHemisphere * tile;
    const int64_t downWaves = divideCeil(shape.n, downColumnsPerWave);
    const int64_t downReductionBlocks = shape.hidden / tile;
    const target::weight_pages::ItemLayout downReductionLayout{
        downOutputBlocksPerHemisphere * rowsPerWeightTile};
    const int64_t downReductionsPerGroup =
        target::weight_pages::itemsPerGroup(pageGeometry, downReductionLayout);
    const int64_t downReductionsPerPage =
        sliceGroupCount * downReductionsPerGroup;
    if (downReductionsPerPage <= 0) return mlir::failure();
    result.down_wave_count = downWaves;
    result.down_reduction_blocks_per_page = downReductionsPerPage;
    result.down_reduction_blocks_per_slice_group =
        downReductionsPerGroup;
    result.down_items_per_page = downReductionsPerPage;
    result.down_items_per_slice_group = downReductionsPerGroup;
    result.down_output_waves_per_page = 1;

    // A Down output wave is independently consumed, so keep it as a transfer
    // chunk while packing several chunks into disjoint rows of one physical
    // slice group. This decouples transfer readiness from SRAM residency: for
    // the 8192-row Qwen target, three waves fit per group and all twelve waves
    // fit in one bank without overwriting each other.
    const int64_t downRowsPerWave = downReductionBlocks
        * downOutputBlocksPerHemisphere * rowsPerWeightTile;
    const bool compactDownWaves = target::weight_pages::itemsPerGroup(
        pageGeometry, {downRowsPerWave}) > 0;
    // Keep one logical Down output wave per runtime page. Several waves may
    // remain resident simultaneously in disjoint slice-group/row slots, but
    // the page identity must stay aligned with the compute timeline and the
    // per-page synchronization ABI. Packing several output waves into one
    // logical page loses that boundary and can release the tail waves before
    // their result writes drain.
    int64_t downPageIndex = 0;
    const auto reductionPages = target::weight_pages::splitItems(
        downReductionBlocks, sliceGroupCount, pageGeometry, downReductionLayout);
    if (reductionPages.empty()) return mlir::failure();
    for (int64_t wave = 0; wave < downWaves; ++wave) {
        for (const auto& itemPage : reductionPages) {
            const int64_t reduction = itemPage.item_begin;
            const int64_t count = itemPage.item_count;
            const int64_t usedGroups = itemPage.group_count;
            int64_t bank = -1;
            int64_t group = 0;
            int64_t baseRow = 0;
            if (compactDownWaves && reduction == 0
                && count == downReductionBlocks) {
                // Distinct consumer pages can share a bank while retaining
                // distinct rows. Only the streaming schedule requests bank
                // alternation; both policies fill every slot before reuse.
                const auto slot = target::weight_pages::residentSlot(
                    downPageIndex, downRowsPerWave, pageGeometry, initialBank,
                    streamingDownDoubleBuffer && !bankLevelPingPong);
                if (!slot) return mlir::failure();
                bank = slot->bank;
                group = slot->slice_group_begin;
                baseRow = slot->base_row;
            }
            const int64_t rows = compactDownWaves
                    && reduction == 0 && count == downReductionBlocks
                ? downRowsPerWave
                : itemPage.rows_per_group;
            FfnWeightTilePage page {-1, bank, baseRow, rows,
                memory.hemispheres * weightSlices * rows, 0, {}};
            page.spans.push_back({FfnWeightTileKind::Down,
                bankLevelPingPong ? initialBank : bank,
                baseRow, group,
                compactDownWaves ? 1 : usedGroups,
                downReductionsPerGroup, wave, 1,
                reduction, count, downOutputBlocksPerHemisphere, rows});
            appendPage(std::move(page), downPageIndex++);
        }
    }
    result.minimum_hidden_slices = divideCeil(
        shape.m * shape.hidden * 2, result.bank_bytes);
    return result;
}

} // namespace ftlpu::compiler::tensor
