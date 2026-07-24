#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"

namespace ftlpu::compiler::schedule {

FfnTaskPlan buildFfnTaskPlan(FfnScheduleShape shape,
    const target::LPUTargetModel& target)
{
    FfnTaskPlan result;
    result.task_ids.weight_load =
        FfnWeightLoadScheduleBuilder().append(result.tasks, shape, target);
    result.task_ids.projection = FfnProjectionScheduleBuilder().append(
        result.tasks, shape, result.task_ids.weight_load, target);
    result.task_ids.swish = FfnSwishScheduleBuilder().append(
        result.tasks, shape, result.task_ids.projection, target);
    result.task_ids.down_projection =
        FfnDownProjectionScheduleBuilder().append(
            result.tasks, shape, result.task_ids.swish, target);
    return result;
}

} // namespace ftlpu::compiler::schedule
