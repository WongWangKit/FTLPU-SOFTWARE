#include "ftlpu/compiler/Dialect/Tensor/Analysis/ffn_weight_tile_plan.hpp"

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
        || throughput.mxms_per_hemisphere != 1)
        return mlir::failure();

    const int64_t reductionBlocks = shape.k / tile;
    const int64_t projectionRowsPerWave =
        reductionBlocks * rowsPerWeightTile;
    const int64_t projectionWaves =
        shape.hidden / (memory.hemispheres * tile);
    const int64_t sliceGroupCount = storageSlices / weightSlices;
    if (sliceGroupCount % 2 != 0) return mlir::failure();
    const int64_t splitProjectionGroupsPerRole = sliceGroupCount / 2;
    const int64_t projectionWavesPerGroup =
        memory.sram_depth_rows / projectionRowsPerWave;
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

    for (int64_t wave = 0; wave < projectionWaves;
         wave += projectionWavesPerPage) {
        const int64_t count = std::min(
            projectionWavesPerPage, projectionWaves - wave);
        const int64_t rows = std::min(count, projectionWavesPerGroup)
            * projectionRowsPerWave;
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
    const int64_t downReductionsPerGroup = memory.sram_depth_rows
        / (downOutputBlocksPerHemisphere * rowsPerWeightTile);
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
    const bool compactDownWaves = downRowsPerWave > 0
        && downRowsPerWave <= memory.sram_depth_rows;
    const int64_t downWavesPerGroup = compactDownWaves
        ? memory.sram_depth_rows / downRowsPerWave : 0;
    const int64_t downWavesPerBank =
        downWavesPerGroup * sliceGroupCount;
    const bool groupedDownPages = bankLevelPingPong && compactDownWaves
        && downWavesPerGroup > 0 && downWaves <= downWavesPerBank;
    if (groupedDownPages) {
        result.down_output_waves_per_page = downWavesPerGroup;
        result.down_items_per_page =
            downWavesPerGroup * downReductionBlocks;
        result.down_items_per_slice_group = result.down_items_per_page;
        int64_t bindingPageIndex = 0;
        for (int64_t wave = 0; wave < downWaves;
             wave += downWavesPerGroup) {
            const int64_t waveCount = std::min(
                downWavesPerGroup, downWaves - wave);
            const int64_t group = wave / downWavesPerGroup;
            const int64_t rows = waveCount * downRowsPerWave;
            FfnWeightTilePage page {-1, initialBank, 0, rows,
                memory.hemispheres * weightSlices * rows, 0, {}};
            page.spans.push_back({FfnWeightTileKind::Down,
                initialBank, 0, group, 1,
                waveCount * downReductionBlocks, wave, waveCount,
                0, downReductionBlocks, downOutputBlocksPerHemisphere,
                rows});
            appendPage(std::move(page), bindingPageIndex++);
        }
        result.minimum_hidden_slices = divideCeil(
            shape.m * shape.hidden * 2, result.bank_bytes);
        return result;
    }
    int64_t downPageIndex = 0;
    for (int64_t wave = 0; wave < downWaves; ++wave) {
        for (int64_t reduction = 0; reduction < downReductionBlocks;
             reduction += downReductionsPerPage) {
            const int64_t count = std::min(
                downReductionsPerPage, downReductionBlocks - reduction);
            const int64_t usedGroups = divideCeil(
                count, downReductionsPerGroup);
            int64_t bank = -1;
            int64_t group = 0;
            int64_t baseRow = 0;
            if (compactDownWaves && reduction == 0
                && count == downReductionBlocks) {
                if (streamingDownDoubleBuffer && !bankLevelPingPong) {
                    // Alternate banks for consecutive consumer tiles, but do
                    // not throw away the remaining SRAM in either bank. Fill
                    // every disjoint (slice-group,row-slot) before wrapping
                    // and reusing a physical region. This preserves ping-pong
                    // overlap while making page boundaries reflect actual
                    // SRAM capacity instead of one logical output wave.
                    const int64_t bankSlot = downPageIndex
                        % memory.banks_per_slice;
                    const int64_t slotInBank =
                        (downPageIndex / memory.banks_per_slice)
                        % downWavesPerBank;
                    bank = (initialBank + bankSlot)
                        % memory.banks_per_slice;
                    group = slotInBank / downWavesPerGroup;
                    baseRow = (slotInBank % downWavesPerGroup)
                        * downRowsPerWave;
                } else {
                    const int64_t residentSlot = wave
                        % (downWavesPerBank * memory.banks_per_slice);
                    const int64_t bankSlot = residentSlot / downWavesPerBank;
                    const int64_t slotInBank = residentSlot % downWavesPerBank;
                    bank = (initialBank + bankSlot) % memory.banks_per_slice;
                    group = slotInBank / downWavesPerGroup;
                    baseRow = (slotInBank % downWavesPerGroup)
                        * downRowsPerWave;
                }
            }
            const int64_t rows = compactDownWaves
                    && reduction == 0 && count == downReductionBlocks
                ? downRowsPerWave
                : std::min(count, downReductionsPerGroup)
                    * downOutputBlocksPerHemisphere * rowsPerWeightTile;
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
