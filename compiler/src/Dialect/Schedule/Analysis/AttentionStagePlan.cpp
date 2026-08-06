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
    result.qk_iw_to_compute_cycles =
        target.throughput().qk_iw_to_compute_latency;
    const int64_t firstIwOffset =
        target.throughput().mxm_earliest_iw_cycle
        + *target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, 0);
    const int64_t computeEnd = firstIwOffset
        + result.qk_iw_to_compute_cycles + qkWaveComputeCycles;
    result.qk_wave_interval =
        target.supports_mxm_weight_activation_overlap()
            && target.throughput().mxm_weight_buffers >= 2
        ? qkWaveComputeCycles
        : computeEnd - firstIwOffset;
    result.qk_wave_duration = computeEnd
        + std::max(target.throughput().mxm0_accumulator_latency,
            target.throughput().mxm1_accumulator_latency);

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
