#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_block8_projection_planner.hpp"
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
    target::ThroughputModel block8Throughput;
    block8Throughput.mxm_block_compute_enabled = 1;
    target::LPUTargetModel target(
        target::MemoryTopology {}, target::StreamTopology {}, block8Throughput);
    const auto distributedActivationSlices =
        target.mxm_distributed_activation_slices();
    for (int64_t tempSlice :
        target.memory().w8a16_fused_gate_temp_slices) {
        require(std::find(distributedActivationSlices.begin(),
                    distributedActivationSlices.end(), tempSlice)
                == distributedActivationSlices.end(),
            "gate temporary slice overlaps distributed activation");
    }
    for (int64_t tempSlice :
        target.memory().w8a16_fused_up_temp_slices) {
        require(std::find(distributedActivationSlices.begin(),
                    distributedActivationSlices.end(), tempSlice)
                == distributedActivationSlices.end(),
            "up temporary slice overlaps distributed activation");
    }
    const auto derivedHiddenSlices = target.ffn_hidden_slices();
    require(derivedHiddenSlices.size()
            == static_cast<std::size_t>(
                target.throughput().mxm_activation_streams),
        "target did not allocate the complete fused hidden layout");
    for (int64_t hiddenSlice : derivedHiddenSlices) {
        require(std::find(distributedActivationSlices.begin(),
                    distributedActivationSlices.end(), hiddenSlice)
                == distributedActivationSlices.end(),
            "fused hidden slice overlaps distributed activation");
    }
    auto conflictingMemory = target.memory();
    conflictingMemory.w8a16_fused_gate_temp_slices[0] =
        distributedActivationSlices.front();
    target::LPUTargetModel conflictingTarget(
        conflictingMemory, target.streams(), target.throughput());
    std::string validationError;
    require(mlir::failed(conflictingTarget.validate(&validationError)),
        "target validation accepted a fused temporary activation hazard");
    require(validationError.find("overlap") != std::string::npos,
        "unexpected target validation error: " + validationError);

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

    auto block8Projection = schedule::planFfnBlock8ProjectionSchedule(
        4, 4, 0,
        target.ffn_projection_weight_slices(
            target::FfnProjectionKind::Gate),
        target.ffn_projection_weight_slices(
            target::FfnProjectionKind::Up),
        target.ffn_block8_input_slices(), target);
    require(mlir::succeeded(block8Projection),
        "Block8 projection planner failed");
    require(block8Projection->reductions.size() == 4
            && block8Projection->end_cycle == 68,
        "Block8 projection planner has the wrong extent");
    const int64_t expectedLoadCycles[] = {0, 4, 32, 36};
    const int64_t expectedComputeCycles[][4] = {
        {4, 12, 20, 28},
        {8, 16, 24, 32},
        {36, 44, 52, 60},
        {40, 48, 56, 64},
    };
    for (int64_t reduction = 0; reduction < 4; ++reduction) {
        const auto& scheduled = block8Projection->reductions[reduction];
        require(scheduled.reduction == reduction
                && scheduled.weight_buffer == reduction % 2
                && scheduled.load_cycle == expectedLoadCycles[reduction],
            "Block8 projection load/buffer plan is incorrect");
        require(std::equal(scheduled.compute_cycles.begin(),
                    scheduled.compute_cycles.end(),
                    expectedComputeCycles[reduction]),
            "Block8 projection compute interleave is incorrect");
    }
    const auto gateWeightSlices = target.ffn_projection_weight_slices(
        target::FfnProjectionKind::Gate);
    const auto upWeightSlices = target.ffn_projection_weight_slices(
        target::FfnProjectionKind::Up);
    const auto block8InputSlices = target.ffn_block8_input_slices();
    for (int64_t slice : gateWeightSlices) {
        require(std::find(upWeightSlices.begin(), upWeightSlices.end(), slice)
                    == upWeightSlices.end()
                && std::find(block8InputSlices.begin(),
                       block8InputSlices.end(), slice)
                    == block8InputSlices.end(),
            "Gate weight slice is not independent");
    }
    for (int64_t slice : upWeightSlices)
        require(std::find(block8InputSlices.begin(),
                    block8InputSlices.end(), slice)
                == block8InputSlices.end(),
            "Up weight slice overlaps Block8 activation");

    require(block8Projection->reductions[0].activation_stream_bases[0] == 16
            && block8Projection->reductions[1]
                    .activation_stream_bases[3] == 16
            && block8Projection->reductions[2]
                    .activation_stream_bases[0] == 16,
        "Block8 projection did not move overlapping activation traffic");

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
        (projection->m_tile_count - 1)
            * projection->pipelined_block_interval
        + target.mxm_first_result_latency()
        + target.mxm_result_window_cycles(
            target.throughput().mxm_rows);
    require(projection->projection_slot_interval
            == expectedProjectionSlot,
        "FFN projection slot does not include the MXM result drain");
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
        hiddenSlices, resultSlices, target);
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
        "FFN down timeline has the wrong phase offset");
    const int64_t expectedDownReductionInterval =
        target.throughput().mxms_per_hemisphere == 1
        ? std::max(projection->weight_block_interval,
              projection->projection_slot_interval
                  + 2 * target.throughput().mxm_rows
                  - projection->weight_load_cycles
                  + target.mxm_first_result_latency())
        : projection->weight_block_interval;
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
        hiddenSlices, resultSlices, exploredTarget);
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
