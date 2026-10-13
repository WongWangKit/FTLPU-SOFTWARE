#include "ftlpu/software/runtime/moe_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ftlpu::software::runtime;

int main()
try {
    // Qwen3-30B-A3B router geometry. The test keeps real hidden/expert/Top-K
    // dimensions while using deterministic synthetic activations and weights.
    constexpr std::size_t tokens = 32;
    constexpr std::size_t hidden = 2048;
    constexpr std::size_t experts = 128;
    constexpr std::size_t topK = 8;
    std::vector<float> activation(tokens * hidden);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t feature = 0; feature < hidden; ++feature)
            activation[token * hidden + feature] =
                static_cast<float>(static_cast<int>(
                    (token * 17 + feature * 13) % 31) - 15) / 32.0f;
    std::vector<float> router(experts * hidden, 0.0f);
    for (std::size_t expert = 0; expert < experts; ++expert) {
        router[expert * hidden + (expert * 11 + 3) % hidden] =
            (expert & 1) ? -0.75f : 0.75f;
        router[expert * hidden + (expert * 19 + 7) % hidden] =
            static_cast<float>((expert % 7) + 1) / 16.0f;
    }
    const MoeRouterConfig config{tokens, hidden, experts, topK, true};
    const MoeDispatchPlan plan =
        plan_qwen3_moe_routes(activation, router, config);
    constexpr float routerScale = 1.0f / 16.0f;
    std::vector<std::int8_t> quantizedRouter(router.size());
    std::vector<float> dequantizedRouter(router.size());
    for (std::size_t index = 0; index < router.size(); ++index) {
        quantizedRouter[index] = static_cast<std::int8_t>(
            std::lround(router[index] / routerScale));
        dequantizedRouter[index] =
            static_cast<float>(quantizedRouter[index]) * routerScale;
    }
    const MoeDispatchPlan quantizedPlan = plan_qwen3_moe_routes_int8(
        activation, quantizedRouter, routerScale, config);
    const MoeDispatchPlan dequantizedPlan = plan_qwen3_moe_routes(
        activation, dequantizedRouter, config);
    if (quantizedPlan.routes.size() != dequantizedPlan.routes.size())
        throw std::logic_error("Qwen3 MoE INT8 router changed route geometry");
    for (std::size_t index = 0; index < quantizedPlan.routes.size(); ++index) {
        if (quantizedPlan.routes[index].expert
                != dequantizedPlan.routes[index].expert
            || std::fabs(quantizedPlan.routes[index].weight
                    - dequantizedPlan.routes[index].weight) > 1.0e-6f)
            throw std::logic_error(
                "Qwen3 MoE INT8 router disagrees with dequantized reference");
    }
    if (plan.routes.size() != tokens * topK
        || plan.assignments.size() != tokens * topK
        || plan.expert_offsets.size() != experts + 1)
        throw std::logic_error("Qwen3 MoE route plan has the wrong geometry");
    for (std::size_t token = 0; token < tokens; ++token) {
        float sum = 0.0f;
        std::vector<std::uint32_t> selected;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            const auto route = plan.routes[token * topK + slot];
            sum += route.weight;
            selected.push_back(route.expert);
        }
        std::ranges::sort(selected);
        if (std::ranges::adjacent_find(selected) != selected.end())
            throw std::logic_error("Qwen3 MoE selected one expert twice");
        if (std::fabs(sum - 1.0f) > 1.0e-5f)
            throw std::logic_error("Qwen3 MoE normalized Top-K sum is not one");
    }
    const auto packed = dispatch_moe_tokens(activation, plan);
    for (std::size_t row = 0; row < plan.assignments.size(); ++row)
        for (std::size_t feature : {std::size_t{0}, std::size_t{317}, hidden - 1})
            if (packed[row * hidden + feature]
                != activation[plan.assignments[row].token * hidden + feature])
                throw std::logic_error("Qwen3 MoE dispatch changed a token value");

    std::vector<float> expertOutput(packed.size());
    for (std::size_t row = 0; row < plan.assignments.size(); ++row) {
        const float bias = static_cast<float>(plan.assignments[row].expert) / 128.0f;
        for (std::size_t feature = 0; feature < hidden; ++feature)
            expertOutput[row * hidden + feature] =
                packed[row * hidden + feature] + bias;
    }
    const auto combined = combine_moe_outputs(expertOutput, plan);
    for (std::size_t token = 0; token < tokens; ++token) {
        float expectedBias = 0.0f;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            const auto route = plan.routes[token * topK + slot];
            expectedBias += route.weight * static_cast<float>(route.expert) / 128.0f;
        }
        for (std::size_t feature : {std::size_t{0}, std::size_t{317}, hidden - 1}) {
            const float expected = activation[token * hidden + feature] + expectedBias;
            if (std::fabs(combined[token * hidden + feature] - expected) > 2.0e-6f)
                throw std::logic_error("Qwen3 MoE weighted combine mismatch");
        }
    }
    const auto pages = plan_moe_expert_pages(plan, 2, 0, 2);
    if (pages.size() != plan.active_experts.size())
        throw std::logic_error("Qwen3 MoE page plan lost an active expert");
    std::size_t pagedAssignments = 0;
    for (std::size_t index = 0; index < pages.size(); ++index) {
        if (pages[index].bank != index % 2
            || pages[index].residency_slot != index % 2)
            throw std::logic_error("Qwen3 MoE pages do not ping-pong banks");
        pagedAssignments += pages[index].assignment_count;
    }
    if (pagedAssignments != tokens * topK)
        throw std::logic_error("Qwen3 MoE page plan lost assignments");
    std::cout << "Qwen3-30B-A3B router passed: tokens=" << tokens
              << " experts=" << experts << " top_k=" << topK
              << " active_experts=" << pages.size()
              << " assignments=" << plan.assignments.size() << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "qwen3_moe_router_test failed: " << exception.what() << '\n';
    return 1;
}
