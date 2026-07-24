#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

mlir::FailureOr<FfnDownProjectionTimeline> planFfnDownProjectionTimeline(
    FfnScheduleShape shape, const FfnProjectionTimeline& projection,
    int64_t lastSwishCycle, llvm::ArrayRef<int64_t> weightSlices,
    llvm::ArrayRef<int64_t> hiddenSlices,
    llvm::ArrayRef<int64_t> resultSlices,
    const target::LPUTargetModel& target)
{
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = throughput.mxm_rows;
    if (lastSwishCycle < 0 || hiddenSlices.empty() || resultSlices.empty()
        || projection.blocks.empty())
        return mlir::failure();

    const auto westLatency = [&](int64_t slice) {
        return slice / target.streams().mem_slices_per_register_group + 2;
    };
    FfnDownProjectionTimeline result;
    result.phase_start = lastSwishCycle + 1
        + throughput.swiglu_write_latency
        + westLatency(hiddenSlices.back()) + 1
        + throughput.accumulator_to_vxm_latency;
    result.pair_transition_interval =
        2 * tile + throughput.accumulator_to_vxm_latency;
    for (int64_t resultSlice : resultSlices) {
        if (std::find(weightSlices.begin(), weightSlices.end(), resultSlice)
            == weightSlices.end())
            continue;
        const int64_t lastWriteEnd =
            throughput.accumulator_to_vxm_latency + tile
            + resultSlice
                / target.streams().mem_slices_per_register_group
            + 1;
        result.pair_transition_interval =
            std::max(result.pair_transition_interval,
                lastWriteEnd + tile + westLatency(resultSlice));
    }

    result.reduction_block_count = shape.hidden / tile;
    const int64_t columnsPerHemisphere =
        throughput.mxms_per_hemisphere * tile;
    result.columns_per_wave =
        memory.hemispheres * columnsPerHemisphere;
    result.wave_count =
        (shape.n + result.columns_per_wave - 1) / result.columns_per_wave;
    int64_t computeCycle =
        result.phase_start + projection.initial_compute_cycle;

    for (int64_t wave = 0; wave < result.wave_count; ++wave) {
        const int64_t activeHemispheres = std::min<int64_t>(
            memory.hemispheres,
            (shape.n - wave * result.columns_per_wave
                + columnsPerHemisphere - 1)
                / columnsPerHemisphere);
        for (int64_t reduction = 0;
             reduction < result.reduction_block_count; ++reduction) {
            FfnDownBlockSchedule block;
            block.index = static_cast<int64_t>(result.blocks.size());
            block.output_wave = wave;
            block.reduction_block = reduction;
            block.active_hemispheres = activeHemispheres;
            block.weight_compute_cycle = computeCycle;
            block.dequant_start = computeCycle - tile;
            block.weight_buffer = block.index % 2;
            block.final_reduction =
                reduction + 1 == result.reduction_block_count;

            for (int64_t mTile = 0;
                 mTile < projection.m_tile_count; ++mTile) {
                FfnDownTileSchedule tileSchedule;
                tileSchedule.m_tile = mTile;
                tileSchedule.compute_cycle = block.weight_compute_cycle
                    + mTile * projection.pipelined_block_interval;
                tileSchedule.prefetch_next_weight = !block.final_reduction
                    && projection.m_tile_count > 1
                    && mTile + 1 == projection.m_tile_count;
                if (!tileSchedule.prefetch_next_weight) {
                    tileSchedule.segments.push_back({tile, 0});
                } else {
                    tileSchedule.segments.push_back(
                        {throughput.vxm_weight_to_iw_latency, 0});
                    for (int64_t unit = 0;
                         unit < activeHemispheres
                                 * throughput.mxms_per_hemisphere;
                         ++unit) {
                        tileSchedule.segments.push_back(
                            {projection.weight_load_cycles,
                                unit % throughput.mxms_per_hemisphere == 0
                                    ? target.streams()
                                          .streams_per_direction
                                          / 2
                                    : 0});
                    }
                    const int64_t routedRows =
                        throughput.vxm_weight_to_iw_latency
                        + activeHemispheres
                            * throughput.mxms_per_hemisphere
                            * projection.weight_load_cycles;
                    if (routedRows < tile)
                        tileSchedule.segments.push_back(
                            {tile - routedRows, 0});
                }
                block.tiles.push_back(std::move(tileSchedule));
            }
            result.blocks.push_back(std::move(block));

            if (result.blocks.back().final_reduction) {
                if (wave + 1 < result.wave_count)
                    computeCycle += (projection.m_tile_count - 1)
                            * projection.pipelined_block_interval
                        + result.pair_transition_interval;
            } else {
                computeCycle += projection.weight_block_interval
                    + (projection.m_tile_count > 1 ? 0 : tile);
            }
        }
    }
    return result;
}

} // namespace ftlpu::compiler::schedule
