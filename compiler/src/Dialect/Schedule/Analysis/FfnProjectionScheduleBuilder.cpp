#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId FfnProjectionScheduleBuilder::append(SchedulePlan& plan,
    FfnScheduleShape shape, ScheduleTaskId weightLoad,
    const target::LPUTargetModel& target) const
{
    const int64_t tile = target.throughput().mxm_rows;
    const auto blocks = [tile](int64_t value) {
        return std::max<int64_t>(1, (value + tile - 1) / tile);
    };
    const int64_t work =
        2 * blocks(shape.m) * blocks(shape.hidden) * blocks(shape.k);
    const auto id = plan.addTask("ffn.projection",
        ScheduleTaskKind::MxmCompute, ScheduleStage::FfnProjection,
        0, work);
    (void)plan.addDependency(weightLoad, id,
        target.throughput().vxm_weight_to_iw_latency);
    return id;
}

} // namespace ftlpu::compiler::schedule
