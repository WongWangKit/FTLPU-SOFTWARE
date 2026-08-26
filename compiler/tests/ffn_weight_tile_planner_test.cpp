#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_weight_tile_planner.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace ftlpu::compiler;

int main()
try {
    target::MemoryTopology memory;
    memory.words_per_bank = 2048;
    memory.sram_depth_rows = 2048;
    memory.bytes_per_word = 32;
    memory.dedicated_slice_roles = 1;
    memory.w8a16_weight_slice_base = 20;
    target::ThroughputModel throughput;
    throughput.mxms_per_hemisphere = 1;
    throughput.mxm_block_compute_enabled = 0;
    target::LPUTargetModel target(memory, {}, throughput);

    auto plan = tensor::planFfnWeightTiles(
        {32, 1536, 8960, 1536}, target);
    if (mlir::failed(plan))
        throw std::logic_error("Qwen FFN weight tiling failed");
    if (plan->bank_bytes != 64 * 1024
        || plan->projection_wave_count != 140
        || plan->weight_storage_slice_count != 32
        || plan->slice_group_count != 4
        || plan->projection_waves_per_page != 20
        || plan->projection_waves_per_slice_group != 10
        || plan->down_wave_count != 12
        || plan->down_reduction_blocks_per_page != 1024
        || plan->minimum_hidden_slices != 9
        || plan->pages.size() != 19)
        throw std::logic_error("unexpected Qwen FFN tile geometry");

    const auto projectionPages = std::count_if(
        plan->pages.begin(), plan->pages.end(), [](const auto& page) {
            return page.spans.size() == 2;
        });
    const auto downPages = plan->pages.size() - projectionPages;
    if (projectionPages != 7 || downPages != 12)
        throw std::logic_error("unexpected Qwen FFN page count");
    for (std::size_t index = 0; index < plan->pages.size(); ++index) {
        const auto& page = plan->pages[index];
        const auto bindingPage = index < projectionPages
            ? index : index - projectionPages;
        if (page.bank != static_cast<int64_t>(bindingPage % 2)
            || page.rows_per_slice <= 0
            || page.rows_per_slice > plan->bank_rows
            || page.transfer_cycles <= 0)
            throw std::logic_error("invalid FFN weight page");
    }
    if (plan->pages.front().rows_per_slice != 1920
        || plan->pages.front().spans[0].page_base_row != 0
        || plan->pages.front().spans[0].slice_group_count != 2
        || plan->pages.front().spans[1].slice_group_begin != 2
        || plan->pages[7].rows_per_slice != 2048
        || plan->pages[7].spans[0].reduction_block_count != 280)
        throw std::logic_error("Down K split does not honor 64 KiB bank");
    auto bankOnePlan = tensor::planFfnWeightTiles(
        {32, 1536, 8960, 1536}, target, 1);
    if (mlir::failed(bankOnePlan)
        || bankOnePlan->pages[0].bank != 1
        || bankOnePlan->pages[1].bank != 0
        || bankOnePlan->pages[projectionPages].bank != 1)
        throw std::logic_error("FFN page banks ignore the initial bank");
    auto tasks = schedule::buildFfnWeightTileTaskPlan(
        *plan, {32, 1536, 8960, 1536}, target);
    schedule::ResourceScheduler resources;
    auto assignment = tasks.tasks.schedule(resources);
    if (mlir::failed(assignment)
        || tasks.page_tasks.size() != plan->pages.size())
        throw std::logic_error("cannot schedule FFN weight pages");
    for (std::size_t index = 1; index < tasks.page_tasks.size(); ++index) {
        const auto& previous = (*assignment)[
            tasks.page_tasks[index - 1].compute];
        const auto& prefetch = (*assignment)[
            tasks.page_tasks[index].prefetch];
        const auto& compute = (*assignment)[
            tasks.page_tasks[index].compute];
        if (compute.cycle < previous.end_cycle)
            throw std::logic_error("FFN compute pages overlap one MXM");
        for (std::size_t previous = index; previous-- > 0;) {
            if (plan->pages[previous].bank != plan->pages[index].bank)
                continue;
            const auto& sameBank = (*assignment)[
                tasks.page_tasks[previous].compute];
            if (prefetch.cycle < sameBank.end_cycle)
                throw std::logic_error("C2C overwrites a live weight bank");
            break;
        }
    }

    std::cout << "ffn_weight_tile_planner_test passed: projection_pages="
              << projectionPages << " down_pages=" << downPages
              << " hidden_slices=" << plan->minimum_hidden_slices << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ffn_weight_tile_planner_test failed: "
              << error.what() << '\n';
    return 1;
}
