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
    FfnWeightShape shape, const target::LPUTargetModel& target)
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
        || memory.banks_per_slice < 2 || memory.sram_depth_rows <= 0
        || memory.bytes_per_word <= 0 || memory.hemispheres <= 0
        || tile <= 0 || rowsPerWeightTile <= 0 || weightSlices <= 0
        || storageSlices < 2 * weightSlices
        || storageSlices % weightSlices != 0
        || streams.c2c_streams_per_direction <= 0
        || throughput.mxms_per_hemisphere != 1
        || target.supports_mxm_block8_compute())
        return mlir::failure();

    const int64_t reductionBlocks = shape.k / tile;
    const int64_t projectionRowsPerWave =
        reductionBlocks * rowsPerWeightTile;
    const int64_t projectionWaves =
        shape.hidden / (memory.hemispheres * tile);
    const int64_t sliceGroupCount = storageSlices / weightSlices;
    if (sliceGroupCount % 2 != 0) return mlir::failure();
    const int64_t projectionGroupsPerRole = sliceGroupCount / 2;
    const int64_t projectionWavesPerGroup =
        memory.sram_depth_rows / projectionRowsPerWave;
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
    result.projection_waves_per_slice_group = projectionWavesPerGroup;
    const int64_t c2cLanes = memory.hemispheres
        * streams.c2c_streams_per_direction;
    const auto appendPage = [&](FfnWeightTilePage page) {
        page.index = static_cast<int64_t>(result.pages.size());
        page.bank = page.index % memory.banks_per_slice;
        page.transfer_cycles = divideCeil(page.transfer_vectors, c2cLanes);
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
        page.spans.push_back({FfnWeightTileKind::Gate, 0, 0,
            projectionGroupsPerRole, projectionWavesPerGroup, wave,
            count, 0, reductionBlocks, 1, rows});
        page.spans.push_back({FfnWeightTileKind::Up, 0,
            projectionGroupsPerRole, projectionGroupsPerRole,
            projectionWavesPerGroup, wave, count, 0, reductionBlocks,
            1, rows});
        appendPage(std::move(page));
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
    for (int64_t wave = 0; wave < downWaves; ++wave) {
        for (int64_t reduction = 0; reduction < downReductionBlocks;
             reduction += downReductionsPerPage) {
            const int64_t count = std::min(
                downReductionsPerPage, downReductionBlocks - reduction);
            const int64_t rows = std::min(count, downReductionsPerGroup)
                * downOutputBlocksPerHemisphere * rowsPerWeightTile;
            FfnWeightTilePage page {-1, -1, 0, rows,
                memory.hemispheres * weightSlices * rows, 0, {}};
            page.spans.push_back({FfnWeightTileKind::Down, 0, 0,
                sliceGroupCount, downReductionsPerGroup, wave, 1,
                reduction, count, downOutputBlocksPerHemisphere, rows});
            appendPage(std::move(page));
        }
    }
    result.minimum_hidden_slices = divideCeil(
        shape.m * shape.hidden * 2, result.bank_bytes);
    return result;
}

} // namespace ftlpu::compiler::tensor
