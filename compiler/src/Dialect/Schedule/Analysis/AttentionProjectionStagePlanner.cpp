#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId AttentionProjectionStagePlanner::append(SchedulePlan& plan,
    AttentionStageShape shape, const target::LPUTargetModel& target) const
{
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t heads = shape.query_heads + 2 * shape.kv_heads;
    const int64_t work = heads * (shape.hidden / tile)
        * (shape.sequence_length / tile);
    return plan.addTask("attention.projection", ScheduleTaskKind::MxmCompute,
        ScheduleStage::Projection, 0, std::max<int64_t>(1, work));
}

} // namespace ftlpu::compiler::schedule
