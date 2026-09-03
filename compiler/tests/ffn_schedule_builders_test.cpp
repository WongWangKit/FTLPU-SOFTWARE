#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_swish_planner.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main() try
{
    using namespace ftlpu::compiler;
    target::LPUTargetModel target;

    auto plan = schedule::buildFfnTaskPlan({128, 576, 1536, 576}, target);
    require(plan.tasks.size() == 4, "FFN plan must contain four stages");
    require(mlir::succeeded(plan.tasks.validate()), "FFN task DAG is invalid");
    require(plan.tasks.task(plan.task_ids.weight_load).stage
            == schedule::ScheduleStage::FfnWeightLoad,
        "weight-load stage mismatch");
    require(plan.tasks.task(plan.task_ids.swish).kind
            == schedule::ScheduleTaskKind::VxmCompute,
        "Swish must be assigned to VXM");
    schedule::ResourceScheduler resources;
    require(mlir::succeeded(plan.tasks.schedule(resources)),
        "FFN task DAG failed to schedule");

    schedule::ResourceScheduler repeatedWave;
    repeatedWave.reserve_at(32, {{"mem.0", 0, 4}});
    repeatedWave.reserve_at(606, {{"mem.0", 0, 4}});
    require(repeatedWave.minimum_non_overlapping_shift(576) == 578,
        "repeat wave interval did not account for a trailing MEM write");

    schedule::FfnSwishScheduleRequest swish;
    swish.tasks = {{10, 0}, {10, 1}};
    swish.dequant_windows = {{0, 8}};
    swish.tile_rows = 32;
    auto cycles = schedule::planFfnSwishCycles(swish, target);
    require(mlir::succeeded(cycles), "FFN Swish planner failed");
    require((*cycles)[0] == 10 && (*cycles)[1] == 47,
        "FFN Swish planner did not serialize the shared VXM pipeline");

    const auto& memory = target.memory();
    llvm::SmallVector<int64_t> weightSlices;
    for (int64_t index = 0; index < memory.w8a16_weight_slice_count; ++index)
        weightSlices.push_back(memory.w8a16_weight_slice_base
            + index * memory.w8a16_weight_slice_stride);
    auto projection = schedule::planFfnProjectionTimeline(
        {128, 576, 1536, 576}, weightSlices, target);
    require(mlir::succeeded(projection), "FFN projection timeline failed");
    require(projection->blocks.size() == 24 * 18,
        "FFN projection timeline has the wrong block count");
    auto replicatedProjection = schedule::planFfnProjectionTimeline(
        {128, 576, 1536, 576}, weightSlices, target, false, true);
    require(mlir::succeeded(replicatedProjection),
        "replicated FFN projection timeline failed");
    require(replicatedProjection->blocks.size() == 48 * 18,
        "replicated FFN projection must schedule every output block");
    require(projection->blocks.front().weight_buffer == 0
            && projection->blocks[1].weight_buffer == 1,
        "FFN projection weight buffers must ping-pong");
    require(projection->blocks.front().dequant_start
            + target.throughput().mxm_rows
            == projection->blocks.front().weight_compute_cycle,
        "FFN projection dequant lead is incorrect");
    const int64_t expectedProjectionSlot =
        projection->m_tile_count
        * projection->pipelined_block_interval;
    require(projection->projection_slot_interval
            == expectedProjectionSlot,
        "FFN projection slot does not match back-to-back MXM issue");
    auto localDequantProjection = schedule::planFfnProjectionTimeline(
        {32, 576, 1536, 576}, weightSlices, target, true);
    require(mlir::succeeded(localDequantProjection),
        "local-dequant FFN projection timeline failed");
    const auto& firstLocalBlock = localDequantProjection->blocks.front();
    const auto& reusedLocalBlock =
        localDequantProjection->blocks[
            target.throughput().mxm_weight_buffers];
    require(reusedLocalBlock.dequant_start
            >= firstLocalBlock.tiles.back().compute_cycle
                + target.throughput().mxm_rows
                + target.mxm_first_result_latency(),
        "local-dequant FFN reused a live MXM weight buffer");
    auto singleMxmThroughput = target.throughput();
    singleMxmThroughput.mxms_per_hemisphere = 1;
    singleMxmThroughput.mxm_weight_buffers = 2;
    singleMxmThroughput.mxm_local_dequant_enabled = 1;
    target::LPUTargetModel singleMxmTarget(
        target.memory(), target.streams(), singleMxmThroughput);
    auto singleMxmProjection = schedule::planFfnProjectionTimeline(
        {32, 576, 1536, 576}, weightSlices, singleMxmTarget, true);
    require(mlir::succeeded(singleMxmProjection)
            && singleMxmProjection->blocks.size() > 1,
        "single-MXM local-dequant FFN projection timeline failed");
    require(singleMxmProjection->blocks[1].dequant_start
            >= singleMxmProjection->blocks[0].tiles.back().compute_cycle
                + singleMxmTarget.mxm_result_window_cycles(
                    singleMxmThroughput.mxm_rows),
        "single-MXM FFN overwrote a fixed projection weight buffer");
    require(singleMxmProjection->blocks[1].weight_compute_cycle
            - singleMxmProjection->blocks[0].weight_compute_cycle
            == singleMxmProjection->weight_block_interval,
        "single-MXM FFN delayed compute instead of retiming weight load");
    auto residencyFirstProjection = schedule::planFfnProjectionTimeline(
        {32, 576, 1536, 576}, weightSlices, singleMxmTarget, true,
        false, schedule::FfnProjectionOrder::UpThenGate);
    require(mlir::succeeded(residencyFirstProjection),
        "residency-first FFN projection timeline failed");
    require(residencyFirstProjection->projection_order
                == schedule::FfnProjectionOrder::UpThenGate
            && residencyFirstProjection->second_projection_offset > 0,
        "residency-first FFN did not serialize its projections");
    require(residencyFirstProjection->blocks[1].weight_compute_cycle
            - residencyFirstProjection->blocks[0].weight_compute_cycle
            == residencyFirstProjection->projection_slot_interval,
        "residency-first FFN left a bubble between projection blocks");
    for (std::size_t index = 1;
         index < residencyFirstProjection->blocks.size(); ++index)
        require(residencyFirstProjection->blocks[index].weight_compute_cycle
                - residencyFirstProjection->blocks[index - 1]
                      .weight_compute_cycle
                == residencyFirstProjection->projection_block_interval,
            "residency-first FFN left an output-block boundary bubble");
    const auto& finalFirstProjectionBlock =
        residencyFirstProjection->blocks.back();
    require(residencyFirstProjection->initial_compute_cycle
                + residencyFirstProjection->second_projection_offset
            == finalFirstProjectionBlock.tiles.back().compute_cycle
                + residencyFirstProjection->projection_slot_interval,
        "residency-first FFN left a bubble between projection phases");
    require(residencyFirstProjection->blocks[0].weight_buffer == 0
            && residencyFirstProjection->blocks[1].weight_buffer == 1,
        "residency-first FFN did not make both weight buffers available");
    require(residencyFirstProjection->final_projection_cycle
            > residencyFirstProjection->second_projection_offset,
        "residency-first FFN final cycle does not include both phases");
    for (const auto& block : projection->blocks) {
        for (const auto& tile : block.tiles) {
            for (const auto& hemisphere : tile.hemisphere_segments) {
                int64_t rows = 0;
                for (const auto& segment : hemisphere)
                    rows += segment.rows;
                require(rows == target.throughput().mxm_rows,
                    "FFN projection segments do not cover one MXM tile");
            }
        }
    }

    llvm::SmallVector<int64_t> hiddenSlices {
        40, 41, 42, 43};
    llvm::SmallVector<int64_t> resultSlices {
        36, 37, 38, 39};
    auto down = schedule::planFfnDownProjectionTimeline(
        {128, 576, 1536, 576}, *projection, 1000, weightSlices,
        hiddenSlices, resultSlices, target, 0);
    require(mlir::succeeded(down), "FFN down timeline failed");
    const int64_t logicalOutputSlotsPerHemisphere =
        std::max<int64_t>(2,
            target.throughput().mxms_per_hemisphere);
    const int64_t expectedDownWaves =
        (576 + target.memory().hemispheres
                    * logicalOutputSlotsPerHemisphere
                    * target.throughput().mxm_rows
                - 1)
        / (target.memory().hemispheres
            * logicalOutputSlotsPerHemisphere
            * target.throughput().mxm_rows);
    require(down->wave_count == expectedDownWaves
            && down->blocks.size()
                == static_cast<std::size_t>(expectedDownWaves * 48),
        "FFN down timeline has the wrong wave or block count");
    require(down->output_stream_base == 24
            && down->first_accumulator_stream == 32
            && down->second_accumulator_stream == 36
            && down->vxm_queues_per_hemisphere == 8,
        "FFN down timeline changed the default stream layout");
    require(down->blocks.front().weight_compute_cycle
            == down->phase_start + projection->initial_compute_cycle,
        "FFN down timeline has the wrong phase offset: compute="
            + std::to_string(
                down->blocks.front().weight_compute_cycle)
            + " phase=" + std::to_string(down->phase_start)
            + " initial="
            + std::to_string(projection->initial_compute_cycle));
    const int64_t expectedDownReductionInterval =
        projection->weight_block_interval;
    require(down->reduction_interval == expectedDownReductionInterval
            && down->pair_transition_interval
                >= down->reduction_interval,
        "FFN down timeline reuses a live MXM weight buffer");
    bool sawLocalDequantPrefetch = false;
    for (const auto& block : down->blocks) {
        for (const auto& tile : block.tiles) {
            int64_t rows = 0;
            for (const auto& segment : tile.segments)
                rows += segment.rows;
            require(rows == target.throughput().mxm_rows,
                "FFN down segments do not cover one MXM tile");
            if (!tile.prefetch_next_weight) continue;
            sawLocalDequantPrefetch = true;
            require(tile.segments.size() == 1
                    && tile.segments.front().rows
                        == target.throughput().mxm_rows
                    && tile.segments.front().stream_base
                        == target.throughput()
                               .mxm_load_streams_per_cycle,
                "local-dequant FFN down activation overlaps weight streams");
        }
    }
    require(sawLocalDequantPrefetch,
        "FFN down timeline did not exercise weight prefetch");

    auto singleMxmDown = schedule::planFfnDownProjectionTimeline(
        {32, 576, 1536, 576}, *singleMxmProjection, 1000,
        weightSlices, hiddenSlices, resultSlices, singleMxmTarget, 0);
    require(mlir::succeeded(singleMxmDown)
            && singleMxmDown->blocks.size() > 1,
        "single-MXM local-dequant FFN down timeline failed");
    require(singleMxmDown->blocks[1].dequant_start
            >= singleMxmDown->blocks[0].tiles.back().compute_cycle
                + singleMxmTarget.mxm_result_window_cycles(
                    singleMxmThroughput.mxm_rows),
        "single-MXM FFN down overwrote a fixed weight buffer");
    require(singleMxmDown->blocks[1].weight_compute_cycle
            - singleMxmDown->blocks[0].weight_compute_cycle
            == singleMxmDown->reduction_interval,
        "single-MXM FFN down delayed compute instead of retiming weight load");

    auto pagedSingleMxmDown = schedule::planFfnDownProjectionTimeline(
        {32, 576, 1536, 576}, *singleMxmProjection, 1000,
        weightSlices, hiddenSlices, resultSlices, singleMxmTarget, 48);
    require(mlir::succeeded(pagedSingleMxmDown)
            && pagedSingleMxmDown->wave_count > 1,
        "paged single-MXM FFN down timeline failed");
    const auto& previousPageLast = pagedSingleMxmDown->blocks[47];
    const auto& nextPageFirst = pagedSingleMxmDown->blocks[48];
    int64_t maxWeightLatency = 0;
    for (int64_t slice : weightSlices) {
        maxWeightLatency = std::max(maxWeightLatency,
            singleMxmTarget.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, slice).value());
    }
    int64_t maxResultLatency = 0;
    for (int64_t slice : resultSlices) {
        maxResultLatency = std::max(maxResultLatency,
            singleMxmTarget.transport_latency(
                target::StreamEndpoint::MxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::West, slice).value());
    }
    const int64_t previousLastCompute =
        previousPageLast.tiles.back().compute_cycle
        + singleMxmProjection->projection_slot_interval;
    const int64_t previousDrainEnd = previousLastCompute
        + singleMxmTarget.mxm_first_result_latency()
        + maxResultLatency + singleMxmThroughput.mxm_rows
        + singleMxmTarget.streams().system_register_columns;
    require(nextPageFirst.dequant_start - maxWeightLatency
            >= previousDrainEnd,
        "paged FFN down lacks a pipeline-safe runtime wait boundary: read="
            + std::to_string(
                nextPageFirst.dequant_start - maxWeightLatency)
            + " drain=" + std::to_string(previousDrainEnd));

    auto exploredStreams = target.streams();
    exploredStreams.streams_per_direction = 40;
    exploredStreams.encoded_streams = 80;
    target::LPUTargetModel exploredTarget(
        target.memory(), exploredStreams, target.throughput());
    auto exploredProjection = schedule::planFfnProjectionTimeline(
        {128, 576, 1536, 576}, weightSlices, exploredTarget);
    require(mlir::succeeded(exploredProjection),
        "40-stream FFN projection timeline failed");
    auto exploredDown = schedule::planFfnDownProjectionTimeline(
        {128, 576, 1536, 576}, *exploredProjection, 1000, weightSlices,
        hiddenSlices, resultSlices, exploredTarget, 0);
    require(mlir::succeeded(exploredDown),
        "40-stream FFN down timeline failed");
    require(exploredDown->output_stream_base == 32
            && exploredDown->first_accumulator_stream == 40
            && exploredDown->second_accumulator_stream == 44,
        "FFN down stream layout did not follow the 40-stream target");
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ffn_schedule_builders_test failed: " << ex.what()
              << '\n';
    return 1;
}
