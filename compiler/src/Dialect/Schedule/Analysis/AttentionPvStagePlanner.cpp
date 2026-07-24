#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId AttentionPvStagePlanner::append(SchedulePlan& plan,
    AttentionStageShape shape, ScheduleTaskId softmax,
    const target::LPUTargetModel& target) const
{
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t work = shape.query_heads
        * (shape.sequence_length / tile) * (shape.head_dim / tile);
    const auto id = plan.addTask("attention.pv", ScheduleTaskKind::MxmCompute,
        ScheduleStage::Pv, 0, std::max<int64_t>(1, work));
    (void)plan.addDependency(softmax, id,
        target.throughput().mem_to_mxm_latency);
    return id;
}

} // namespace ftlpu::compiler::schedule
