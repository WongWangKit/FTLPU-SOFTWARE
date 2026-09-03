#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

mlir::FailureOr<FfnDownProjectionTimeline> planFfnDownProjectionTimeline(
    FfnScheduleShape shape, const FfnProjectionTimeline& projection,
    int64_t lastSwishCycle, llvm::ArrayRef<int64_t> weightSlices,
    llvm::ArrayRef<int64_t> hiddenSlices,
    llvm::ArrayRef<int64_t> resultSlices,
    const target::LPUTargetModel& target,
    int64_t reductionsPerWeightPage)
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
    const auto weightLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem,
                   target::StreamEndpoint::MxmWeight,
                   target::StreamDirection::East, slice)
            .value_or(slice
                    / target.streams().mem_slices_per_register_group
                + 2);
    };
    int64_t maxWeightLatency = 0;
    for (int64_t slice : weightSlices)
        maxWeightLatency = std::max(maxWeightLatency, weightLatency(slice));
    int64_t maxResultLatency = 0;
    for (int64_t slice : resultSlices) {
        const auto latency = target.transport_latency(
            target::StreamEndpoint::MxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::West, slice);
        if (!latency) return mlir::failure();
        maxResultLatency = std::max(maxResultLatency, *latency);
    }
    FfnDownProjectionTimeline result;
    result.phase_start = lastSwishCycle + 1
        + throughput.swiglu_write_latency
        + westLatency(hiddenSlices.back()) + 1
        + throughput.accumulator_to_vxm_latency;
    result.reduction_interval = projection.weight_block_interval;
    // Down0/Down1 alternate the two physical weight buffers. While one
    // buffer computes, the other accepts the following reduction's weight,
    // so a reduction occupies exactly the logical projection issue slots.
    result.pair_transition_interval = result.reduction_interval;
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
    result.pair_transition_interval = std::max(
        result.pair_transition_interval, result.reduction_interval);

    result.reduction_block_count = shape.hidden / tile;
    // Vector FFN keeps two logical 32-column output slots per hemisphere.
    // A dual-MXM target executes them in parallel; a single-MXM target
    // time-multiplexes both slots through its two weight buffers.
    const int64_t columnsPerHemisphere =
        std::max<int64_t>(2, throughput.mxms_per_hemisphere) * tile;
    result.columns_per_wave =
        memory.hemispheres * columnsPerHemisphere;
    result.wave_count =
        (shape.n + result.columns_per_wave - 1) / result.columns_per_wave;
    result.output_stream_base = target.streams().streams_per_direction
        - 2 * throughput.mxm_result_streams;
    result.first_accumulator_stream =
        target.streams().streams_per_direction;
    result.second_accumulator_stream = result.first_accumulator_stream
        + throughput.mxm_result_streams;
    result.vxm_queues_per_hemisphere =
        throughput.vxm_alus / memory.hemispheres;
    if (result.output_stream_base < 0
        || result.vxm_queues_per_hemisphere <= 0)
        return mlir::failure();
    int64_t computeCycle =
        result.phase_start + projection.initial_compute_cycle;
    const int64_t pagesPerWave = reductionsPerWeightPage > 0
        ? (result.reduction_block_count + reductionsPerWeightPage - 1)
            / reductionsPerWeightPage
        : 0;
    int64_t previousPage = -1;
    int64_t previousPageDrainEnd = 0;

    for (int64_t wave = 0; wave < result.wave_count; ++wave) {
        const int64_t activeHemispheres = std::min<int64_t>(
            memory.hemispheres,
            (shape.n - wave * result.columns_per_wave
                + columnsPerHemisphere - 1)
                / columnsPerHemisphere);
        for (int64_t reduction = 0;
             reduction < result.reduction_block_count; ++reduction) {
            const int64_t page = pagesPerWave > 0
                ? wave * pagesPerWave
                    + reduction / reductionsPerWeightPage
                : -1;
            if (page >= 0 && previousPage >= 0 && page != previousPage) {
                // A runtime page miss may hold every compute-side ICU queue
                // while C2C/MEM ingress continues. Move the next page's first
                // weight read beyond the preceding page's final in-flight
                // result so that this coarse wait point is pipeline-safe.
                computeCycle = std::max(computeCycle,
                    previousPageDrainEnd + maxWeightLatency + tile);
                previousPageDrainEnd = 0;
            }
            previousPage = page;

            FfnDownBlockSchedule block;
            block.index = static_cast<int64_t>(result.blocks.size());
            block.output_wave = wave;
            block.reduction_block = reduction;
            block.active_hemispheres = activeHemispheres;
            block.weight_compute_cycle = computeCycle;
            block.dequant_start = computeCycle - tile;
            if (throughput.mxms_per_hemisphere == 1
                && target.supports_mxm_local_dequant()
                && !result.blocks.empty()) {
                const int64_t previousLastCompute =
                    result.blocks.back().tiles.back().compute_cycle;
                block.dequant_start = std::max(block.dequant_start,
                    previousLastCompute
                        + target.mxm_result_window_cycles(tile));
            }
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
                } else if (target.supports_mxm_local_dequant()) {
                    // Local INT8 weight input owns E0..E15 while the next
                    // weight wave propagates to the MXMs. Keep the complete
                    // activation wave on the disjoint upper stream bank.
                    tileSchedule.segments.push_back(
                        {tile, throughput.mxm_load_streams_per_cycle});
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

            const auto& emittedBlock = result.blocks.back();
            const int64_t lastTileCompute =
                emittedBlock.tiles.back().compute_cycle;
            const int64_t lastLogicalCompute = lastTileCompute
                + (throughput.mxms_per_hemisphere == 1
                        ? projection.projection_slot_interval : 0);
            int64_t blockDrainEnd = lastLogicalCompute
                + target.mxm_result_window_cycles(tile);
            if (emittedBlock.final_reduction) {
                blockDrainEnd = std::max(blockDrainEnd,
                    lastLogicalCompute
                        + target.mxm_first_result_latency()
                        + maxResultLatency + tile);
            }
            blockDrainEnd += target.streams().system_register_columns;
            previousPageDrainEnd = std::max(
                previousPageDrainEnd, blockDrainEnd);

            if (result.blocks.back().final_reduction) {
                if (wave + 1 < result.wave_count)
                    computeCycle += (projection.m_tile_count - 1)
                            * projection.pipelined_block_interval
                        + result.pair_transition_interval;
            } else {
                computeCycle += result.reduction_interval;
            }
        }
    }
    return result;
}

} // namespace ftlpu::compiler::schedule
