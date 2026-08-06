#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_route_plan.hpp"

#include <algorithm>

namespace ftlpu::compiler::stream {

void StreamRoutePlan::add(llvm::StringRef phase, llvm::StringRef role,
    target::StreamEndpoint source, target::StreamEndpoint destination,
    target::StreamDirection direction, llvm::StringRef buffer,
    int64_t producerOffset, int64_t consumerOffset)
{
    routes_.push_back({phase.str(), role.str(), source, destination,
        direction, buffer.str(), producerOffset, consumerOffset});
    duration_ = std::max(duration_, consumerOffset + 1);
}

bool StreamRoutePlan::valid() const
{
    if (routes_.empty() || duration_ <= 0) return false;
    for (const RouteLifetime& route : routes_)
        if (route.phase.empty() || route.role.empty() || route.buffer.empty()
            || route.producer_offset < 0
            || route.consumer_offset <= route.producer_offset)
            return false;
    return true;
}

StreamRoutePlan plan_attention_routes()
{
    using E = target::StreamEndpoint;
    using D = target::StreamDirection;
    StreamRoutePlan plan;
    const auto add = [&](llvm::StringRef phase, llvm::StringRef role,
                         E source, E destination, D direction,
                         llvm::StringRef buffer, int64_t begin, int64_t end) {
        plan.add(phase, role, source, destination, direction,
            buffer, begin, end);
    };
    add("qkv", "query_weight", E::Mem, E::VxmInput, D::East,
        "query_weight", 0, 2);
    add("qkv", "query_weight_dequant", E::VxmResult, E::MxmWeight,
        D::East, "query_weight", 2, 4);
    add("qkv", "activation", E::Mem, E::MxmActivation, D::East,
        "input", 4, 8);
    add("qkv", "qkv_result", E::MxmResult, E::Mem, D::West,
        "query", 8, 10);
    add("qkv", "key_weight", E::Mem, E::VxmInput, D::East,
        "key_weight", 10, 12);
    add("qkv", "key_weight_dequant", E::VxmResult, E::MxmWeight,
        D::East, "key_weight", 12, 14);
    add("qkv", "key_activation", E::Mem, E::MxmActivation, D::East,
        "input", 14, 18);
    add("qkv", "key_result", E::MxmResult, E::Mem, D::West,
        "key", 18, 20);
    add("qkv", "value_weight", E::Mem, E::VxmInput, D::East,
        "value_weight", 20, 22);
    add("qkv", "value_weight_dequant", E::VxmResult, E::MxmWeight,
        D::East, "value_weight", 22, 24);
    add("qkv", "value_activation", E::Mem, E::MxmActivation, D::East,
        "input", 24, 28);
    add("qkv", "value_result", E::MxmResult, E::Mem, D::West,
        "value", 28, 30);
    add("rope", "qk_to_vxm", E::Mem, E::VxmInput, D::East,
        "query", 30, 32);
    add("qk", "query_activation", E::Mem, E::MxmActivation, D::East,
        "query", 32, 36);
    add("qk", "key_weight", E::Mem, E::MxmWeight, D::East,
        "key", 32, 36);
    add("qk", "score_result", E::MxmResult, E::Mem, D::West,
        "score", 36, 38);
    add("softmax", "score_to_vxm", E::Mem, E::VxmInput, D::East,
        "score", 38, 41);
    add("pv", "probability_activation", E::Mem, E::MxmActivation,
        D::East, "probability_diagonal", 41, 45);
    add("pv", "value_weight", E::Mem, E::MxmWeight, D::East,
        "value", 41, 45);
    add("pv", "context_result", E::MxmResult, E::Mem, D::West,
        "context", 45, 47);
    add("o_proj", "context_activation", E::Mem, E::MxmActivation,
        D::East, "context", 47, 51);
    add("o_proj", "output_weight", E::Mem, E::VxmInput, D::East,
        "output_weight", 47, 49);
    add("o_proj", "output_weight_dequant", E::VxmResult, E::MxmWeight,
        D::East, "output_weight", 49, 51);
    add("o_proj", "result", E::MxmResult, E::Mem, D::West,
        "result", 51, 53);
    return plan;
}

} // namespace ftlpu::compiler::stream
