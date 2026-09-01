#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

AttentionStagePlan planAttentionStages(AttentionStageShape shape,
    const target::LPUTargetModel& target)
{
    AttentionStagePlan result;
    const AttentionProjectionPlanner projectionPlanner(
        {shape.sequence_length, shape.hidden, shape.query_heads,
            shape.kv_heads, shape.head_dim},
        target);
    result.projection_work = projectionPlanner.work();

    const AttentionWorkPlanner workPlanner(
        {shape.sequence_length, shape.query_heads, shape.kv_heads,
            shape.head_dim},
        target);
    result.qk_waves = workPlanner.qk_waves();
    result.pv_waves = workPlanner.pv_waves();
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t tokenBlocks = shape.sequence_length / tile;
    const int64_t headBlocks = shape.head_dim / tile;
    const int64_t qkIssueCount = tokenBlocks * headBlocks;
    const int64_t qkWaveComputeCycles =
        (qkIssueCount - 1) * target.mxm_block_issue_interval() + tile;
    int64_t scoreDrainCycles = target.mxm_first_result_latency();
    int64_t queryIwReadLead = 0;
    for (int64_t slice = 0;
         slice < target.memory().slices_per_hemisphere; ++slice) {
        const auto latency = target.transport_latency(
            target::StreamEndpoint::MxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::West, slice);
        if (latency)
            scoreDrainCycles = std::max(
                scoreDrainCycles,
                target.mxm_first_result_latency() + *latency);
        const auto iwLatency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        if (iwLatency)
            queryIwReadLead = std::max(queryIwReadLead, *iwLatency);
    }
    result.qk_iw_to_compute_cycles =
        target.throughput().qk_iw_to_compute_latency;
    const int64_t firstIwOffset =
        target.throughput().mxm_earliest_iw_cycle
        + *target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, 0);
    const int64_t computeEnd = firstIwOffset
        + result.qk_iw_to_compute_cycles + qkWaveComputeCycles;
    const bool supportsWavefront =
        target.supports_mxm_weight_activation_overlap()
        && target.throughput().mxm_weight_buffers >= 2;
    if (!supportsWavefront) {
        result.qk_wave_interval = computeEnd - firstIwOffset;
    } else if (target.throughput().mxms_per_hemisphere == 1) {
        // Query IW uses the alternate physical buffer while the current wave
        // computes. The final partial streams scores west, so neither the ACC
        // rows nor the east-side weight path need a drain bubble before the
        // next wave starts computing.
        result.qk_wave_interval = qkWaveComputeCycles;
    } else {
        // A dual-MXM QK wave uses E16..E31 both for MXM1 weights and for
        // activation traffic. The next wave can start loading only after the
        // previous activation packet diagonal has cleared those SR links.
        const int64_t iwPhasesPerReduction = tile / 8;
        const int64_t localMxmPreloadOffset =
            (target.throughput().mxms_per_hemisphere - 1)
            * headBlocks * iwPhasesPerReduction;
        const int64_t streamReuseGap = std::max<int64_t>(0,
            result.qk_iw_to_compute_cycles - localMxmPreloadOffset);
        result.qk_wave_interval = qkWaveComputeCycles + streamReuseGap;
    }
    result.qk_wave_duration = computeEnd
        + std::max(target.throughput().mxm0_accumulator_latency,
            target.throughput().mxm1_accumulator_latency)
        + tile + scoreDrainCycles;

    result.task_ids.projection =
        AttentionProjectionStagePlanner().append(result.tasks, shape, target);
    result.task_ids.rope = AttentionRopeStagePlanner().append(
        result.tasks, shape, result.task_ids.projection, target);
    result.task_ids.softmax = AttentionSoftmaxStagePlanner().append(
        result.tasks, shape, result.task_ids.rope, target);
    result.task_ids.pv = AttentionPvStagePlanner().append(
        result.tasks, shape, result.task_ids.softmax, target);
    result.task_ids.output_projection =
        AttentionOutputProjectionStagePlanner().append(
            result.tasks, shape, result.task_ids.pv, target);
    return result;
}

} // namespace ftlpu::compiler::schedule
