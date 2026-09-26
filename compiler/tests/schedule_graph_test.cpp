#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_graph.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"

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

    ResourceScheduler transactional;
    const ResourceWindow conflictingBundle[] {
        {"mem.h0.s3.b0.icu", 3, 1},
        {"mem.h0.s3.b0.icu", 3, 1},
    };
    require(transactional.has_internal_conflict(conflictingBundle),
        "bundle-internal ICU conflict was not detected");
    require(!transactional.try_reserve_at(100, conflictingBundle),
        "bundle-internal ICU conflict was committed");
    const ResourceWindow nestedConflict[] {
        {"mem.h0.s3.b0.icu", 0, 10},
        {"mem.h0.s3.b0.icu", 1, 1},
        {"mem.h0.s3.b0.icu", 3, 1},
    };
    require(transactional.has_internal_conflict(nestedConflict),
        "nested bundle-internal conflict was not detected");
    const ResourceWindow legalBundle[] {
        {"mem.h0.s3.b0.icu", 3, 1},
        {"mem.h0.s3.b0.read", 3, 1},
    };
    require(transactional.try_reserve_at(100, legalBundle),
        "legal transactional bundle was rejected");
    require(!transactional.try_reserve_at(100, legalBundle),
        "existing ICU reservation was ignored");

    bool fixedCycleRejected = false;
    try {
        transactional.reserve_at(200, conflictingBundle);
    } catch (const std::invalid_argument&) {
        fixedCycleRejected = true;
    }
    require(fixedCycleRejected,
        "fixed-cycle API silently committed an internal conflict");

    ResourceScheduler blocked;
    blocked.block_at(10, {{"external.busy", 0, 4}});
    blocked.block_at(12, {{"external.busy", 0, 4}});
    blocked.block_at(13, {{"external.busy", 0, 1}});
    require(blocked.find_earliest(10, {{"external.busy", 0, 1}}) == 16,
        "overlapping external busy windows were not treated as a union");

    const ftlpu::compiler::target::LPUTargetModel target;
    const LPUResourceModel model(target);
    require(model.mem_queue(0, 0, 0).index == 0,
        "east MEM queue origin changed");
    require(model.mem_queue(1, 0, 0).index
            == target.memory().slices_per_hemisphere
                * target.memory().banks_per_slice,
        "west MEM queue flattening is inconsistent");
    require(model.mem_icu(0, 0, 0)
            == model.queue_resource(model.mem_queue(0, 0, 0)),
        "MEM scheduler resource and command queue identity diverged");
    require(model.queue_resource(model.mxm_load_queue(0))
            != model.queue_resource(model.mxm_compute_queue(0)),
        "MXM load and compute queues were aliased");
}
