#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
{
    using namespace ftlpu::compiler::schedule;
    SchedulePlan plan;
    const ResourceWindow mxm[] {{"MXM.0.compute", 0, 4}};
    const auto projection = plan.addTask("projection", ScheduleTaskKind::MxmCompute,
        ScheduleStage::Projection, 0, 4, mxm);
    const auto rope = plan.addTask("rope", ScheduleTaskKind::VxmCompute,
        ScheduleStage::Rope, 0, 2, {});
    const auto softmax = plan.addTask("softmax", ScheduleTaskKind::VxmCompute,
        ScheduleStage::Softmax, 0, 3, {});
    require(mlir::succeeded(plan.addDependency(projection, rope, 2)),
        "failed to add projection-to-RoPE dependency");
    require(mlir::succeeded(plan.addDependency(rope, softmax, 1)),
        "failed to add RoPE-to-softmax dependency");
    require(mlir::succeeded(plan.validate()), "valid task DAG was rejected");

    ResourceScheduler resources;
    auto assignment = plan.schedule(resources);
    require(mlir::succeeded(assignment), "task DAG failed to schedule");
    require((*assignment)[projection].cycle == 0, "projection cycle mismatch");
    require((*assignment)[rope].cycle == 6, "transport latency was ignored");
    require((*assignment)[softmax].cycle == 9, "consumer dependency was ignored");
    require(plan.findTask("rope") == rope, "stable task lookup failed");

    SchedulePlan invalid;
    invalid.addTask("duplicate", ScheduleTaskKind::Barrier,
        ScheduleStage::Generic, 0, 1);
    invalid.addTask("duplicate", ScheduleTaskKind::Barrier,
        ScheduleStage::Generic, 0, 1);
    require(mlir::failed(invalid.validate()), "duplicate task names were accepted");
}
