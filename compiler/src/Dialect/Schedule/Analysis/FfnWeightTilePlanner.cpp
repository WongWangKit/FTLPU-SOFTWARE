#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_weight_tile_planner.hpp"

#include <set>
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
    using BankGroup = std::pair<int64_t, int64_t>;
    const auto occupiedGroups = [](const FfnWeightTilePage& page) {
        std::set<BankGroup> groups;
        for (const FfnWeightTileSpan& span : page.spans)
            for (int64_t group = span.slice_group_begin;
                 group < span.slice_group_begin + span.slice_group_count;
                 ++group)
                groups.insert({span.bank, group});
        return groups;
    };
    const auto appendMemResources = [&](llvm::SmallVectorImpl<ResourceWindow>& out,
                                        const FfnWeightTilePage& page,
                                        int64_t duration) {
        for (const auto [bank, group] : occupiedGroups(page))
            out.push_back({"mem.weight.bank." + std::to_string(bank)
                    + ".group." + std::to_string(group),
                0, duration});
    };
    for (const FfnWeightTilePage& page : tiles.pages) {
        const std::string suffix = std::to_string(page.index);
        llvm::SmallVector<ResourceWindow, 8> prefetchResources {
            {"c2c.weight.east", 0, page.transfer_cycles},
            {"c2c.weight.west", 0, page.transfer_cycles},
        };
        appendMemResources(prefetchResources, page, page.transfer_cycles);
        const auto prefetch = result.tasks.addTask(
            "ffn.weight_page." + suffix + ".prefetch",
            ScheduleTaskKind::C2cPrefetch,
            ScheduleStage::FfnWeightLoad, 0, page.transfer_cycles,
            prefetchResources);

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
        llvm::SmallVector<ResourceWindow, 8> computeResources {
            {"mxm.east.0", 0, computeDuration},
            {"mxm.west.0", 0, computeDuration},
        };
        appendMemResources(computeResources, page, computeDuration);
        const auto compute = result.tasks.addTask(
            "ffn.weight_page." + suffix + ".compute",
            ScheduleTaskKind::MxmCompute,
            page.spans.front().kind == FfnWeightTileKind::Down
                ? ScheduleStage::FfnDownProjection
                : ScheduleStage::FfnProjection,
            0, computeDuration, computeResources);
        (void)result.tasks.addDependency(prefetch, compute);
        if (!result.page_tasks.empty()) {
            const auto previous = result.page_tasks.back();
            (void)result.tasks.addDependency(previous.prefetch, prefetch);
            (void)result.tasks.addDependency(previous.compute, compute);
        }
        for (std::size_t previous = result.page_tasks.size();
             previous-- > 0;) {
            const auto previousGroups = occupiedGroups(tiles.pages[previous]);
            const auto currentGroups = occupiedGroups(page);
            bool sharesQueue = false;
            for (const auto& group : currentGroups)
                if (previousGroups.contains(group)) {
                    sharesQueue = true;
                    break;
                }
            if (!sharesQueue) continue;
            (void)result.tasks.addDependency(
                result.page_tasks[previous].compute, prefetch);
            break;
        }
        result.page_tasks.push_back({prefetch, compute});
    }
    return result;
}

} // namespace ftlpu::compiler::schedule
