#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"

#include <iostream>
#include <stdexcept>

int main()
{
    using namespace ftlpu::compiler;
    const target::LPUTargetModel target;
    const schedule::AttentionWorkPlanner planner({128, 9, 3, 64}, target);
    if (planner.query_block_count() != 4 || planner.qk_waves().size() != 12
        || planner.pv_waves().size() != 12)
        throw std::logic_error("unexpected generic GQA wave count");
    int64_t work_items = 0;
    for (const auto& wave : planner.qk_waves()) {
        if (wave.slots.size() != 4) throw std::logic_error("unexpected LPU slot count");
        for (std::size_t slot = 0; slot < wave.slots.size(); ++slot) {
            const auto& work = wave.slots[slot];
            if (!work) continue;
            ++work_items;
            if (work->kv_head != work->query_head / 3)
                throw std::logic_error("GQA affinity was not preserved");
            if (slot != static_cast<std::size_t>(work->hemisphere * 2 + work->local_mxm))
                throw std::logic_error("wave violates physical MXM slot ownership");
        }
    }
    if (work_items != 9 * 4) throw std::logic_error("missing QK work items");
    std::cout << "attention_work_planner_test passed\n";
    return 0;
}
