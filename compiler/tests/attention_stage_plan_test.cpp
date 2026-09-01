#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include <cstdio>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (condition) return;
    std::fprintf(stderr, "attention_stage_plan_test: %s\n", message);
    std::fflush(stderr);
    throw std::logic_error(message);
}

} // namespace

int main()
{
    using namespace ftlpu::compiler;
    target::ThroughputModel singleMxmThroughput;
    singleMxmThroughput.mxms_per_hemisphere = 1;
    target::LPUTargetModel target(target::MemoryTopology {},
        target::StreamTopology {}, singleMxmThroughput);
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

    require(plan.qk_wave_interval == 256,
        "single-MXM QK waves should overlap without a refill gap");

    target::ThroughputModel dualMxmThroughput;
    dualMxmThroughput.mxms_per_hemisphere = 2;
    target::LPUTargetModel dualMxmTarget(
        target::MemoryTopology {}, target::StreamTopology {},
        dualMxmThroughput);
    const auto dualMxmPlan = schedule::planAttentionStages(
        {128, 576, 9, 3, 64}, dualMxmTarget);
    require(dualMxmPlan.qk_wave_interval == 272,
        "dual-MXM QK waves must include the physical SR reuse gap");

    dualMxmThroughput.mxm_weight_activation_overlap_enabled = 0;
    target::LPUTargetModel serialTarget(
        target::MemoryTopology {}, target::StreamTopology {},
        dualMxmThroughput);
    const auto serialPlan = schedule::planAttentionStages(
        {128, 576, 9, 3, 64}, serialTarget);
    require(serialPlan.qk_wave_interval == 280,
        "QK waves without weight/activation overlap require full refill");
}
