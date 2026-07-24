#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId FfnDownProjectionScheduleBuilder::append(SchedulePlan& plan,
    FfnScheduleShape shape, ScheduleTaskId swish,
    const target::LPUTargetModel& target) const
{
    const int64_t tile = target.throughput().mxm_rows;
    const auto blocks = [tile](int64_t value) {
        return std::max<int64_t>(1, (value + tile - 1) / tile);
    };
    const int64_t work =
        blocks(shape.m) * blocks(shape.n) * blocks(shape.hidden);
    const auto id = plan.addTask("ffn.down_projection",
        ScheduleTaskKind::MxmCompute, ScheduleStage::FfnDownProjection,
        0, work);
    (void)plan.addDependency(swish, id,
        target.throughput().swiglu_write_latency);
    return id;
}

} // namespace ftlpu::compiler::schedule
