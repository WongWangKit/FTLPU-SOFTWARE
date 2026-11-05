#include "ftlpu/compiler/Target/weight_page_plan.hpp"

#include <iostream>
#include <stdexcept>

using namespace ftlpu::compiler::target::weight_pages;

static void require(bool condition, const char* message) {
    if (!condition) throw std::logic_error(message);
}

int main()
try {
    const Geometry geometry{8192, 2, 4};
    const ItemLayout pairedWaves{192, 2, 192};
    require(itemsPerGroup(geometry, pairedWaves) == 40,
            "paired reduction stripes must leave tail headroom");
    const auto complete = splitItems(140, 4, geometry, pairedWaves);
    require(complete.size() == 1 && complete[0].item_count == 140
        && complete[0].group_count == 4 && complete[0].rows_per_group == 7680,
        "a complete projection that fits must remain one page");
    const auto split = splitItems(140, 2, geometry, pairedWaves);
    require(split.size() == 2 && split[0].item_count == 80
        && split[1].item_begin == 80 && split[1].item_count == 60,
        "split must preserve whole reduction items and complete coverage");
    const auto tail = splitItems(3, 1, geometry, pairedWaves);
    require(tail.size() == 1 && tail[0].rows_per_group == 768,
            "an odd tail still occupies an aligned physical stripe");
    require(itemsPerGroup(geometry, {384, 1, 384}) == 20,
            "output reduction tail headroom changes capacity");
    require(itemsPerGroup({32, 2, 4}, {64}) == 0
        && splitItems(1, 1, {32, 2, 4}, {64}).empty()
        && itemsPerGroup({64, 2, 4}, {64, 1, 64}) == 0,
        "insufficient SRAM must fail instead of forcing one item");

    // Independent output transfer pages stay resident in distinct row slots.
    for (int64_t i = 0; i < 12; ++i) {
        const auto slot = residentSlot(i, 2240, geometry, 1);
        require(slot && slot->bank == 1 && slot->slice_group_begin == i / 3
            && slot->base_row == (i % 3) * 2240,
            "disjoint resident pages must not force bank rotation");
    }
    require(residentSlot(12, 2240, geometry, 1)->bank == 0,
            "bank changes only after resident slots fill");
    for (int64_t i = 0; i < 24; ++i) {
        const auto slot = residentSlot(i, 2240, geometry, 1, true);
        require(slot && slot->bank == (1 + i) % 2
            && slot->slice_group_begin == (i / 2) / 3
            && slot->base_row == ((i / 2) % 3) * 2240,
            "explicit ping-pong must retain all row slots in both banks");
    }

    SlotPlanner planner(geometry);
    require(planner.reserve({0, 0, 4, 0, 4096}, {0, 1}, true)
        && planner.reserve({1, 0, 4, 0, 4096}, {0, 1}, true),
        "pin activation/constant regions outside the weight slots");
    const auto first = planner.allocate(1, 4096, {0, 10}, 0, 4096);
    const auto second = planner.allocate(1, 4096, {0, 10}, 0, 4096);
    const auto opposite = planner.allocate(1, 4096, {0, 10}, 1, 4096);
    require(first && *first == Slot{0, 0, 1, 4096, 4096}
        && second && *second == Slot{0, 1, 1, 4096, 4096}
        && opposite && *opposite == Slot{1, 0, 1, 4096, 4096},
        "allocate disjoint slots and honor the preferred overlap bank");
    require(!planner.reserve(*first, {9, 12}), "a live page cannot be overwritten");
    const auto reused = planner.allocate(1, 4096, {10, 20}, 0, 4096);
    require(reused && *reused == *first && planner.earliestReuse(*first, 0) == 20,
        "a half-open lifetime releases its physical slot at the boundary");
    require(!planner.allocate(1, 4097, {20, 30}, 0, 4096)
        && !planner.reserve({0, 0, 1, 0, 1}, {100, 101})
        && !planner.earliestReuse({0, 0, 1, 0, 1}, 100),
        "pinned reservations remain protected after their nominal lifetime");
    require(!planner.reserve({2, 0, 1, 0, 1}, {0, 1})
        && !planner.reserve({0, 4, 1, 0, 1}, {0, 1})
        && !planner.reserve({0, 0, 1, 8191, 2}, {0, 1}),
        "out-of-range bank/group/row must be rejected");

    SlotPlanner aligned({128, 1, 1});
    require(aligned.reserve({0, 0, 1, 0, 3}, {0, 20}, true), "pin unaligned prefix");
    const auto alignedSlot = aligned.allocate(1, 32, {0, 10}, 0, 0, 16);
    require(alignedSlot && alignedSlot->base_row == 16,
            "allocation must honor physical row alignment after reserved ranges");
    require(!aligned.reserve({0, 0, 1, 16, 32}, {20, 30}, true),
            "new pinned storage cannot retroactively overlap a finite reservation");

    std::cout << "weight_page_plan_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "weight_page_plan_test failed: " << error.what() << '\n';
    return 1;
}
