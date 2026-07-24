#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

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
    auto plan = schedule::planAttentionStages({128, 576, 9, 3, 64}, target);
    require(plan.tasks.size() == 5, "attention plan must contain five stages");
    require(mlir::succeeded(plan.tasks.validate()), "attention task DAG is invalid");
    require(!plan.projection_work.empty(), "projection work was not planned");
    require(!plan.qk_waves.empty() && plan.qk_waves.size() == plan.pv_waves.size(),
        "QK/PV work waves were not planned consistently");
    require(plan.tasks.task(plan.task_ids.rope).stage
            == schedule::ScheduleStage::Rope,
        "RoPE task has the wrong stage");
    schedule::ResourceScheduler resources;
    require(mlir::succeeded(plan.tasks.schedule(resources)),
        "attention stage DAG failed to schedule");
}
