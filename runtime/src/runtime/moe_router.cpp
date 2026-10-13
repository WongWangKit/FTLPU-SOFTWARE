#include "ftlpu/software/runtime/moe_router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

void validate_config(const MoeRouterConfig& config)
{
    if (config.token_count == 0 || config.hidden_size == 0
        || config.num_experts == 0 || config.experts_per_token == 0
        || config.experts_per_token > config.num_experts)
        throw std::invalid_argument("invalid MoE router geometry");
    if (config.num_experts > std::numeric_limits<std::uint32_t>::max()
        || config.experts_per_token
            > std::numeric_limits<std::uint16_t>::max())
        throw std::invalid_argument("MoE router geometry exceeds plan fields");
}

} // namespace

template <typename Weight>
MoeDispatchPlan plan_routes(std::span<const float> hidden_states,
    std::span<const Weight> router_weight, float routerScale,
    const MoeRouterConfig& config)
{
    validate_config(config);
    if (!std::isfinite(routerScale) || routerScale <= 0.0f)
        throw std::invalid_argument("MoE router scale must be finite and positive");
    if (hidden_states.size() != config.token_count * config.hidden_size)
        throw std::invalid_argument("MoE hidden-state size does not match geometry");
    if (router_weight.size() != config.num_experts * config.hidden_size)
        throw std::invalid_argument("MoE router-weight size does not match geometry");

    MoeDispatchPlan plan;
    plan.config = config;
    plan.routes.reserve(config.token_count * config.experts_per_token);
    std::vector<bool> active(config.num_experts, false);
    std::vector<float> probability(config.num_experts);
    std::vector<std::size_t> order(config.num_experts);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t token = 0; token < config.token_count; ++token) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t expert = 0; expert < config.num_experts; ++expert) {
            float logit = 0.0f;
            for (std::size_t feature = 0; feature < config.hidden_size; ++feature)
                logit = std::fma(
                    hidden_states[token * config.hidden_size + feature],
                    static_cast<float>(router_weight[
                        expert * config.hidden_size + feature]) * routerScale,
                    logit);
            if (!std::isfinite(logit))
                throw std::invalid_argument("MoE router produced a non-finite logit");
            probability[expert] = logit;
            maximum = std::max(maximum, logit);
        }
        float denominator = 0.0f;
        for (float& value : probability) {
            value = std::exp(value - maximum);
            denominator += value;
        }
        for (float& value : probability) value /= denominator;
        std::partial_sort(order.begin(),
            order.begin() + static_cast<std::ptrdiff_t>(config.experts_per_token),
            order.end(), [&](std::size_t lhs, std::size_t rhs) {
                if (probability[lhs] != probability[rhs])
                    return probability[lhs] > probability[rhs];
                return lhs < rhs;
            });
        float selectedSum = 0.0f;
        for (std::size_t slot = 0; slot < config.experts_per_token; ++slot)
            selectedSum += probability[order[slot]];
        for (std::size_t slot = 0; slot < config.experts_per_token; ++slot) {
            const auto expert = order[slot];
            const float weight = config.norm_topk_prob
                ? probability[expert] / selectedSum : probability[expert];
            plan.routes.push_back({static_cast<std::uint32_t>(expert), weight});
            if (!active[expert]) {
                active[expert] = true;
                plan.active_experts.push_back(static_cast<std::uint32_t>(expert));
            }
        }
    }

    plan.expert_offsets.resize(config.num_experts + 1, 0);
    for (const MoeRoute& route : plan.routes)
        ++plan.expert_offsets[route.expert + 1];
    std::partial_sum(plan.expert_offsets.begin(), plan.expert_offsets.end(),
        plan.expert_offsets.begin());
    plan.assignments.resize(plan.routes.size());
    auto cursor = plan.expert_offsets;
    for (std::size_t token = 0; token < config.token_count; ++token) {
        for (std::size_t slot = 0; slot < config.experts_per_token; ++slot) {
            const MoeRoute& route =
                plan.routes[token * config.experts_per_token + slot];
            plan.assignments[cursor[route.expert]++] = {
                static_cast<std::uint32_t>(token),
                static_cast<std::uint16_t>(slot), route.expert, route.weight};
        }
    }
    return plan;
}

MoeDispatchPlan plan_qwen3_moe_routes(std::span<const float> hidden_states,
    std::span<const float> router_weight, const MoeRouterConfig& config)
{
    return plan_routes(hidden_states, router_weight, 1.0f, config);
}

MoeDispatchPlan plan_qwen3_moe_routes_int8(
    std::span<const float> hidden_states,
    std::span<const std::int8_t> router_weight, float router_scale,
    const MoeRouterConfig& config)
{
    return plan_routes(hidden_states, router_weight, router_scale, config);
}

std::vector<float> dispatch_moe_tokens(std::span<const float> hidden_states,
    const MoeDispatchPlan& plan)
{
    if (hidden_states.size()
        != plan.config.token_count * plan.config.hidden_size)
        throw std::invalid_argument("MoE dispatch input size does not match plan");
    std::vector<float> packed(
        plan.assignments.size() * plan.config.hidden_size);
    for (std::size_t row = 0; row < plan.assignments.size(); ++row) {
        const auto token = plan.assignments[row].token;
        std::copy_n(hidden_states.begin()
                + static_cast<std::ptrdiff_t>(token * plan.config.hidden_size),
            plan.config.hidden_size,
            packed.begin() + static_cast<std::ptrdiff_t>(row * plan.config.hidden_size));
    }
    return packed;
}

std::vector<float> combine_moe_outputs(std::span<const float> expert_outputs,
    const MoeDispatchPlan& plan)
{
    if (expert_outputs.size()
        != plan.assignments.size() * plan.config.hidden_size)
        throw std::invalid_argument("MoE combine input size does not match plan");
    std::vector<float> result(
        plan.config.token_count * plan.config.hidden_size, 0.0f);
    for (std::size_t row = 0; row < plan.assignments.size(); ++row) {
        const MoeAssignment& assignment = plan.assignments[row];
        for (std::size_t feature = 0; feature < plan.config.hidden_size; ++feature)
            result[assignment.token * plan.config.hidden_size + feature]
                = std::fma(assignment.weight,
                    expert_outputs[row * plan.config.hidden_size + feature],
                    result[assignment.token * plan.config.hidden_size + feature]);
    }
    return result;
}

std::vector<MoeExpertPageRequest> plan_moe_expert_pages(
    const MoeDispatchPlan& plan, std::size_t residency_slots,
    std::uint16_t first_bank, std::uint16_t bank_count)
{
    if (residency_slots == 0 || bank_count == 0)
        throw std::invalid_argument("MoE expert paging requires slots and banks");
    std::vector<MoeExpertPageRequest> requests;
    requests.reserve(plan.active_experts.size());
    for (std::size_t index = 0; index < plan.active_experts.size(); ++index) {
        const auto expert = plan.active_experts[index];
        requests.push_back({expert,
            static_cast<std::uint16_t>(index % residency_slots),
            static_cast<std::uint16_t>((first_bank + index) % bank_count),
            plan.expert_offsets[expert + 1] - plan.expert_offsets[expert]});
    }
    return requests;
}

} // namespace ftlpu::software::runtime
