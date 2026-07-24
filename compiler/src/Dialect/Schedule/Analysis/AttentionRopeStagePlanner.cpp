#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId AttentionRopeStagePlanner::append(SchedulePlan& plan,
    AttentionStageShape shape, ScheduleTaskId projection,
    const target::LPUTargetModel& target) const
{
    const int64_t work = (shape.query_heads + shape.kv_heads)
        * shape.sequence_length;
    const auto id = plan.addTask("attention.rope", ScheduleTaskKind::VxmCompute,
        ScheduleStage::Rope, 0, std::max<int64_t>(1, work));
    (void)plan.addDependency(projection, id,
        target.throughput().accumulator_to_vxm_latency);
    return id;
}

} // namespace ftlpu::compiler::schedule
