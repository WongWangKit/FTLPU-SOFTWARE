#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"

#include <iostream>
#include <stdexcept>

int main()
{
    using namespace ftlpu::compiler;
    const target::LPUTargetModel target;
    const schedule::AttentionProjectionPlanner planner({128, 576, 9, 3, 64}, target);
    int64_t query = 0;
    int64_t key = 0;
    int64_t value = 0;
    for (const auto& work : planner.work()) {
        if (work.projection == schedule::AttentionProjection::Query) ++query;
        if (work.projection == schedule::AttentionProjection::Key) ++key;
        if (work.projection == schedule::AttentionProjection::Value) ++value;
        if (work.final_reduction != (work.reduction_block == 17))
            throw std::logic_error("final reduction marker is incorrect");
    }
    if (query != 9 * 18 * 4 || key != 3 * 18 * 4 || value != 3 * 18 * 4)
        throw std::logic_error("projection planner omitted work items");
    std::cout << "attention_projection_planner_test passed\n";
}
