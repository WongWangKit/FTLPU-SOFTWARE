#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

mlir::FailureOr<FfnProjectionTimeline> planFfnProjectionTimeline(
    FfnScheduleShape shape, llvm::ArrayRef<int64_t> weightSlices,
    const target::LPUTargetModel& target)
{
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = throughput.mxm_rows;
    if (!target.supports_w8a16_ffn_shape(
            shape.m, shape.k, shape.hidden, shape.n)
        || weightSlices.empty() || tile <= 0
        || throughput.lanes_per_tile <= 0)
        return mlir::failure();

    const auto westLatency = [&](int64_t slice) {
        return slice / target.streams().mem_slices_per_register_group + 2;
    };
    int64_t maxWeightLatency = 0;
    for (int64_t slice : weightSlices)
        maxWeightLatency = std::max(maxWeightLatency, westLatency(slice));

    FfnProjectionTimeline result;
    result.weight_load_cycles = tile / throughput.lanes_per_tile;
    result.pipelined_block_interval = target.mxm_block_issue_interval();
    result.initial_compute_cycle = maxWeightLatency + 1 + tile;
    result.pair_count = shape.hidden / (memory.hemispheres * tile);
    result.m_tile_count = shape.m / tile;
    result.weight_block_interval =
        result.m_tile_count * result.pipelined_block_interval;
    const int64_t reductionBlocks = shape.k / tile;
    const int64_t totalBlocks = result.pair_count * reductionBlocks;

    for (int64_t pair = 0; pair < result.pair_count; ++pair) {
        for (int64_t reduction = 0; reduction < reductionBlocks;
             ++reduction) {
            FfnProjectionBlockSchedule block;
            block.index = static_cast<int64_t>(result.blocks.size());
            block.pair = pair;
            block.reduction_block = reduction;
            block.weight_compute_cycle = result.initial_compute_cycle
                + block.index * result.weight_block_interval;
            block.dequant_start = block.weight_compute_cycle - tile;
            block.weight_buffer = block.index % 2;
            block.final_reduction = reduction + 1 == reductionBlocks;
            const bool hasNextWeight = block.index + 1 < totalBlocks;

            for (int64_t mTile = 0; mTile < result.m_tile_count; ++mTile) {
                FfnProjectionTileSchedule tileSchedule;
                tileSchedule.m_tile = mTile;
                tileSchedule.compute_cycle = block.weight_compute_cycle
                    + mTile * result.pipelined_block_interval;
                tileSchedule.prefetch_next_weight = hasNextWeight
                    && mTile + 1 == result.m_tile_count;
                tileSchedule.hemisphere_segments.resize(
                    static_cast<std::size_t>(memory.hemispheres));

                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    auto& segments = tileSchedule.hemisphere_segments[
                        static_cast<std::size_t>(hemisphere)];
                    const int64_t nextWeightDistance =
                        tileSchedule.prefetch_next_weight
                        ? result.weight_block_interval
                            - mTile * result.pipelined_block_interval
                        : 2 * tile;
                    const int64_t switchRow = nextWeightDistance - tile
                        + throughput.vxm_weight_to_iw_latency
                        + hemisphere * throughput.mxms_per_hemisphere
                            * result.weight_load_cycles;
                    if (!tileSchedule.prefetch_next_weight
                        || switchRow >= tile) {
                        segments.push_back({tile, 0});
                        continue;
                    }
                    if (switchRow > 0)
                        segments.push_back({switchRow, 0});
                    const int64_t switchedRows =
                        std::min(result.weight_load_cycles, tile - switchRow);
                    segments.push_back(
                        {switchedRows,
                            throughput.mxm_load_streams_per_cycle});
                    if (switchRow + switchedRows < tile)
                        segments.push_back(
                            {tile - switchRow - switchedRows, 0});
                }
                block.tiles.push_back(std::move(tileSchedule));
            }
            result.blocks.push_back(std::move(block));
        }
    }

    if (result.blocks.empty()) return mlir::failure();
    result.final_projection_cycle = result.initial_compute_cycle
        + (totalBlocks - 1) * result.weight_block_interval
        + (result.m_tile_count - 1) * result.pipelined_block_interval;
    const int64_t gateAccumulatorSlice = memory.accumulator_slice_base;
    const int64_t upAccumulatorSlice = memory.accumulator_slice_base
        + memory.accumulator_slices_per_mxm;
    result.accumulator_queue_release = std::max(
        throughput.mxm0_accumulator_latency + tile
            + westLatency(gateAccumulatorSlice),
        throughput.mxm1_accumulator_latency + tile
            + westLatency(upAccumulatorSlice));
    return result;
}

} // namespace ftlpu::compiler::schedule
