#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ftlpu::software::runtime {

struct MoeRouterConfig {
    std::size_t token_count{0};
    std::size_t hidden_size{0};
    std::size_t num_experts{0};
    std::size_t experts_per_token{0};
    bool norm_topk_prob{true};
};

struct MoeRoute {
    std::uint32_t expert{0};
    float weight{0.0f};
};

struct MoeAssignment {
    std::uint32_t token{0};
    std::uint16_t slot{0};
    std::uint32_t expert{0};
    float weight{0.0f};
};

struct MoeDispatchPlan {
    MoeRouterConfig config{};
    // Token-major [token, top-k] routes.
    std::vector<MoeRoute> routes{};
    // Expert-major assignment list and its CSR offsets.
    std::vector<MoeAssignment> assignments{};
    std::vector<std::size_t> expert_offsets{};
    // Active experts in first-token encounter order. This is the order used
    // by the dynamic C2C expert-page planner.
    std::vector<std::uint32_t> active_experts{};
};

struct MoeExpertPageRequest {
    std::uint32_t expert{0};
    std::uint16_t residency_slot{0};
    std::uint16_t bank{0};
    std::size_t assignment_count{0};
};

// Qwen3 MoE routing: FP32 linear projection, softmax over every expert,
// deterministic Top-K, and optional normalization of the selected weights.
MoeDispatchPlan plan_qwen3_moe_routes(std::span<const float> hidden_states,
    std::span<const float> router_weight, const MoeRouterConfig& config);

// Same routing path for the W8A16 importer output. Router weights use one
// symmetric per-tensor scale, matching router.i8.bin and metadata.scales.router.
MoeDispatchPlan plan_qwen3_moe_routes_int8(
    std::span<const float> hidden_states,
    std::span<const std::int8_t> router_weight, float router_scale,
    const MoeRouterConfig& config);

// Packs source tokens into the expert-major assignment order. The returned
// matrix has [assignments, hidden_size] row-major layout.
std::vector<float> dispatch_moe_tokens(std::span<const float> hidden_states,
    const MoeDispatchPlan& plan);

// Multiplies expert-major output rows by route weights and scatter-adds them
// back into [tokens, hidden_size] order.
std::vector<float> combine_moe_outputs(std::span<const float> expert_outputs,
    const MoeDispatchPlan& plan);

// Assigns active expert weight bundles (Gate, Up, and Down) to a bounded set
// of SRAM residency slots. Consecutive bundles ping-pong across physical
// banks; page-ready synchronization remains the responsibility of the
// ordinary C2C pager.
std::vector<MoeExpertPageRequest> plan_moe_expert_pages(
    const MoeDispatchPlan& plan, std::size_t residency_slots = 2,
    std::uint16_t first_bank = 0, std::uint16_t bank_count = 2);

} // namespace ftlpu::software::runtime
