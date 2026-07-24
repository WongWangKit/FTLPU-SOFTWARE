#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

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
