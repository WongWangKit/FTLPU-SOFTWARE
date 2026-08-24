#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_weight_tile_planner.hpp"

#include <string>

namespace ftlpu::compiler::schedule {
FfnWeightTileTaskPlan buildFfnWeightTileTaskPlan(
    const FfnWeightTilePlan& tiles, FfnScheduleShape shape,
    const target::LPUTargetModel& target)
{
    FfnWeightTileTaskPlan result;
    const auto& throughput = target.throughput();
    const int64_t mTiles = shape.m / throughput.mxm_rows;
    const int64_t issueInterval = target.mxm_block_issue_interval();
    for (const FfnWeightTilePage& page : tiles.pages) {
        const std::string suffix = std::to_string(page.index);
        const auto prefetch = result.tasks.addTask(
            "ffn.weight_page." + suffix + ".prefetch",
            ScheduleTaskKind::C2cPrefetch,
            ScheduleStage::FfnWeightLoad, 0, page.transfer_cycles,
            {{"c2c.weight.east", 0, page.transfer_cycles},
             {"c2c.weight.west", 0, page.transfer_cycles},
             {"mem.weight.bank." + std::to_string(page.bank),
                 0, page.transfer_cycles}});

        int64_t computeDuration = 0;
        for (const FfnWeightTileSpan& span : page.spans) {
            const int64_t projectionMultiplier =
                span.kind == FfnWeightTileKind::Down ? 1 : 2;
            computeDuration += span.output_wave_count
                * span.reduction_block_count * mTiles * issueInterval
                * span.output_blocks_per_hemisphere
                * projectionMultiplier;
        }
        computeDuration = std::max<int64_t>(1, computeDuration);
        const auto compute = result.tasks.addTask(
            "ffn.weight_page." + suffix + ".compute",
            ScheduleTaskKind::MxmCompute,
            page.spans.front().kind == FfnWeightTileKind::Down
                ? ScheduleStage::FfnDownProjection
                : ScheduleStage::FfnProjection,
            0, computeDuration,
            {{"mxm.east.0", 0, computeDuration},
             {"mxm.west.0", 0, computeDuration},
             {"mem.weight.bank." + std::to_string(page.bank),
                 0, computeDuration}});
        (void)result.tasks.addDependency(prefetch, compute);
        if (!result.page_tasks.empty()) {
            const auto previous = result.page_tasks.back();
            (void)result.tasks.addDependency(previous.prefetch, prefetch);
            (void)result.tasks.addDependency(previous.compute, compute);
        }
        if (result.page_tasks.size() >= 2) {
            const auto previousSameBank =
                result.page_tasks[result.page_tasks.size() - 2];
            (void)result.tasks.addDependency(
                previousSameBank.compute, prefetch);
        }
        result.page_tasks.push_back({prefetch, compute});
    }
    return result;
}

} // namespace ftlpu::compiler::schedule
