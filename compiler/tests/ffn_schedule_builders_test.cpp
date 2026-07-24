#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_swish_planner.hpp"

#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
{
    using namespace ftlpu::compiler;
    target::LPUTargetModel target;
    auto plan = schedule::buildFfnTaskPlan({128, 576, 1536, 576}, target);
    require(plan.tasks.size() == 4, "FFN plan must contain four stages");
    require(mlir::succeeded(plan.tasks.validate()), "FFN task DAG is invalid");
    require(plan.tasks.task(plan.task_ids.weight_load).stage
            == schedule::ScheduleStage::FfnWeightLoad,
        "weight-load stage mismatch");
    require(plan.tasks.task(plan.task_ids.swish).kind
            == schedule::ScheduleTaskKind::VxmCompute,
        "Swish must be assigned to VXM");
    schedule::ResourceScheduler resources;
    require(mlir::succeeded(plan.tasks.schedule(resources)),
        "FFN task DAG failed to schedule");

    schedule::FfnSwishScheduleRequest swish;
    swish.tasks = {{10, 0}, {10, 1}};
    swish.dequant_windows = {{0, 8}};
    swish.tile_rows = 32;
    auto cycles = schedule::planFfnSwishCycles(swish, target);
    require(mlir::succeeded(cycles), "FFN Swish planner failed");
    require((*cycles)[0] == 10 && (*cycles)[1] == 47,
        "FFN Swish planner did not serialize the shared VXM pipeline");
}
