#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId FfnWeightLoadScheduleBuilder::append(SchedulePlan& plan,
    FfnScheduleShape shape, const target::LPUTargetModel& target) const
{
    const int64_t tile = target.throughput().mxm_rows;
    const auto blocks = [tile](int64_t value) {
        return std::max<int64_t>(1, (value + tile - 1) / tile);
    };
    const int64_t work = 2 * blocks(shape.hidden) * blocks(shape.k)
        + blocks(shape.n) * blocks(shape.hidden);
    return plan.addTask("ffn.weight_load", ScheduleTaskKind::MxmLoad,
        ScheduleStage::FfnWeightLoad, 0, work);
}

} // namespace ftlpu::compiler::schedule
