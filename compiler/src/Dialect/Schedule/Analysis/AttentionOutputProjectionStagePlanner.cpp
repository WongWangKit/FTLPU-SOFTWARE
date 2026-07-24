#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId AttentionOutputProjectionStagePlanner::append(
    SchedulePlan& plan, AttentionStageShape shape, ScheduleTaskId pv,
    const target::LPUTargetModel& target) const
{
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t work = (shape.sequence_length / tile)
        * (shape.hidden / tile);
    const auto id = plan.addTask("attention.output_projection",
        ScheduleTaskKind::MxmCompute, ScheduleStage::OutputProjection,
        0, std::max<int64_t>(1, work));
    (void)plan.addDependency(pv, id,
        target.throughput().mxm1_accumulator_latency);
    return id;
}

} // namespace ftlpu::compiler::schedule
