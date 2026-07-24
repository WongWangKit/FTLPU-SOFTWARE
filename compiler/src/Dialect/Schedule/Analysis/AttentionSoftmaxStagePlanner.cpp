#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId AttentionSoftmaxStagePlanner::append(SchedulePlan& plan,
    AttentionStageShape shape, ScheduleTaskId rope,
    const target::LPUTargetModel& target) const
{
    const int64_t work = shape.query_heads * shape.sequence_length;
    const auto id = plan.addTask("attention.softmax",
        ScheduleTaskKind::VxmCompute, ScheduleStage::Softmax,
        0, std::max<int64_t>(1, work));
    (void)plan.addDependency(rope, id,
        target.throughput().mxm0_accumulator_latency);
    return id;
}

} // namespace ftlpu::compiler::schedule
