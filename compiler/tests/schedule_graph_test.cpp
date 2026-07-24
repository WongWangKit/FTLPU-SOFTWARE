#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_graph.hpp"

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
    ScheduleGraph graph;
    const ResourceWindow shared[] {{"vxm.alu0", 0, 4}};
    const ResourceWindow independent[] {{"mem.h0.s0", 0, 2}};
    const auto first = graph.add_node("first", 0, 4, shared);
    const auto second = graph.add_node("second", 0, 4, shared);
    const auto consumer = graph.add_node("consumer", 0, 2, independent);
    require(mlir::succeeded(graph.add_dependency(first, consumer, 2)),
        "failed to add dependency");
    ResourceScheduler resources;
    auto result = graph.schedule(resources);
    require(mlir::succeeded(result), "valid schedule graph failed");
    require((*result)[first].cycle == 0, "first node was not scheduled at zero");
    require((*result)[second].cycle == 4, "shared resource was not serialized");
    require((*result)[consumer].cycle == 6,
        "producer-to-consumer latency was not honored");

    ScheduleGraph cycle;
    const auto lhs = cycle.add_node("lhs", 0, 1, {});
    const auto rhs = cycle.add_node("rhs", 0, 1, {});
    require(mlir::succeeded(cycle.add_dependency(lhs, rhs)), "failed to add edge");
    require(mlir::succeeded(cycle.add_dependency(rhs, lhs)), "failed to add edge");
    ResourceScheduler cycleResources;
    require(mlir::failed(cycle.schedule(cycleResources)),
        "dependency cycle was not rejected");
}
