#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"
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

    const auto& memory = target.memory();
    llvm::SmallVector<int64_t> weightSlices;
    for (int64_t index = 0; index < memory.w8a16_weight_slice_count; ++index)
        weightSlices.push_back(index * memory.w8a16_weight_slice_stride);
    auto projection = schedule::planFfnProjectionTimeline(
        {128, 576, 1536, 576}, weightSlices, target);
    require(mlir::succeeded(projection), "FFN projection timeline failed");
    require(projection->blocks.size() == 24 * 18,
        "FFN projection timeline has the wrong block count");
    require(projection->blocks.front().weight_buffer == 0
            && projection->blocks[1].weight_buffer == 1,
        "FFN projection weight buffers must ping-pong");
    require(projection->blocks.front().dequant_start
            + target.throughput().mxm_rows
            == projection->blocks.front().weight_compute_cycle,
        "FFN projection dequant lead is incorrect");
    for (const auto& block : projection->blocks) {
        for (const auto& tile : block.tiles) {
            for (const auto& hemisphere : tile.hemisphere_segments) {
                int64_t rows = 0;
                for (const auto& segment : hemisphere)
                    rows += segment.rows;
                require(rows == target.throughput().mxm_rows,
                    "FFN projection segments do not cover one MXM tile");
            }
        }
    }

    llvm::SmallVector<int64_t> hiddenSlices {
        40, 41, 42, 43};
    llvm::SmallVector<int64_t> resultSlices {
        36, 37, 38, 39};
    auto down = schedule::planFfnDownProjectionTimeline(
        {128, 576, 1536, 576}, *projection, 1000, weightSlices,
        hiddenSlices, resultSlices, target);
    require(mlir::succeeded(down), "FFN down timeline failed");
    require(down->wave_count == 5 && down->blocks.size() == 5 * 48,
        "FFN down timeline has the wrong wave or block count");
    require(down->blocks.front().weight_compute_cycle
            == down->phase_start + projection->initial_compute_cycle,
        "FFN down timeline has the wrong phase offset");
    for (const auto& block : down->blocks) {
        for (const auto& tile : block.tiles) {
            int64_t rows = 0;
            for (const auto& segment : tile.segments)
                rows += segment.rows;
            require(rows == target.throughput().mxm_rows,
                "FFN down segments do not cover one MXM tile");
        }
    }
}
