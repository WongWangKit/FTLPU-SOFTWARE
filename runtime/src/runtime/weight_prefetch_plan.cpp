#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <ranges>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

constexpr std::uint64_t kTransportGuardCycles = 64;

const BinaryBinding& find_paged_weight(
    const BinaryProgram& program, std::uint32_t index)
{
    for (const BinaryBinding& binding : program.bindings)
        if (binding.access == BindingAccess::Input
            && binding.index == index && binding.paged_weight)
            return binding;
    throw std::logic_error(
        "weight-page use references a missing paged input binding");
}

std::uint64_t page_byte_size(
    const BinaryBinding& binding, std::uint32_t page)
{
    if (binding.page_count == 0 || page >= binding.page_count)
        throw std::logic_error("weight-page index is outside its binding");
    const auto quotient = binding.byte_size / binding.page_count;
    const auto remainder = binding.byte_size % binding.page_count;
    return quotient + (page < remainder ? 1 : 0);
}

void add_binding_bytes(WeightPrefetchPlan& plan,
    const BinaryBinding& binding, std::uint64_t bytes)
{
    const bool east = (binding.hemisphere_mask & 1u) != 0;
    const bool west = (binding.hemisphere_mask & 2u) != 0;
    const std::uint64_t sideCount = static_cast<std::uint64_t>(east)
        + static_cast<std::uint64_t>(west);
    if (sideCount == 0)
        throw std::logic_error(
            "paged weight binding has an empty hemisphere mask");
    const auto quotient = bytes / sideCount;
    const auto remainder = bytes % sideCount;
    if (east) plan.bytes[0] += quotient + (remainder != 0 ? 1 : 0);
    if (west) plan.bytes[1] += quotient;
}

} // namespace

std::vector<WeightPrefetchPlan> plan_weight_prefetches(
    const BinaryProgram& program)
{
    if (program.weight_page_uses.empty()) return {};
    const std::uint64_t lanes =
        program.hardware.c2c_streams_per_direction;
    const std::uint64_t bytesPerLane =
        program.hardware.c2c_bytes_per_stream_per_cycle;
    if (lanes == 0 || bytesPerLane == 0)
        throw std::logic_error(
            "paged weights require non-zero C2C bandwidth");
    if (lanes > std::numeric_limits<std::uint64_t>::max() / bytesPerLane)
        throw std::overflow_error("C2C bandwidth overflows uint64_t");
    const std::uint64_t bandwidth = lanes * bytesPerLane;
    const std::uint64_t clockMhz = program.hardware.lpu_clock_mhz;
    const std::uint64_t ddrBandwidth =
        program.hardware.ddr_peak_bandwidth_mbytes_per_second;
    const std::uint64_t ddrEfficiency =
        program.hardware.ddr_scheduling_efficiency_percent;
    if (clockMhz == 0 || ddrBandwidth == 0 || ddrEfficiency == 0
        || ddrEfficiency > 100)
        throw std::logic_error(
            "paged weights require a valid external-memory bandwidth model");

    std::vector<std::size_t> ordered(program.weight_page_uses.size());
    for (std::size_t index = 0; index < ordered.size(); ++index)
        ordered[index] = index;
    std::ranges::sort(ordered, [&](std::size_t lhs, std::size_t rhs) {
        const auto& left = program.weight_page_uses[lhs];
        const auto& right = program.weight_page_uses[rhs];
        return left.ready_cycle != right.ready_cycle
            ? left.ready_cycle < right.ready_cycle
            : left.binding_index != right.binding_index
            ? left.binding_index < right.binding_index
            : left.page_index < right.page_index;
    });

    std::vector<WeightPrefetchPlan> plans;
    for (const std::size_t useIndex : ordered) {
        const auto& use = program.weight_page_uses[useIndex];
        auto found = std::ranges::find_if(plans,
            [&](const WeightPrefetchPlan& candidate) {
                return candidate.page_index == use.page_index
                    && candidate.bank == use.bank
                    && use.ready_cycle <= candidate.release_cycle
                    && candidate.ready_cycle <= use.release_cycle;
            });
        if (found == plans.end()) {
            plans.push_back(WeightPrefetchPlan {
                use.page_index, use.bank, 0, 0, use.ready_cycle,
                use.release_cycle, {}, {useIndex}});
            found = std::prev(plans.end());
        } else {
            found->ready_cycle =
                std::min(found->ready_cycle, use.ready_cycle);
            found->release_cycle =
                std::max(found->release_cycle, use.release_cycle);
            found->use_indices.push_back(useIndex);
        }
        const auto& binding = find_paged_weight(program, use.binding_index);
        add_binding_bytes(
            *found, binding, page_byte_size(binding, use.page_index));
    }

    std::ranges::sort(plans, [](const auto& lhs, const auto& rhs) {
        return lhs.ready_cycle < rhs.ready_cycle;
    });
    std::uint64_t fabricCursor = 0;
    std::map<std::uint16_t, std::uint64_t> bankReleaseCycles;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        auto& plan = plans[index];
        const auto sideBytes = std::max(plan.bytes[0], plan.bytes[1]);
        const auto c2cCycles = (sideBytes + bandwidth - 1) / bandwidth;
        const auto totalBytes = plan.bytes[0] + plan.bytes[1];
        if (totalBytes >
            std::numeric_limits<std::uint64_t>::max() / clockMhz / 100)
            throw std::overflow_error(
                "DDR weight-page duration overflows uint64_t");
        const auto effectiveBandwidth = ddrBandwidth * ddrEfficiency;
        const auto ddrCycles =
            (totalBytes * clockMhz * 100 + effectiveBandwidth - 1)
            / effectiveBandwidth;
        const auto duration = std::max(c2cCycles, ddrCycles)
            + program.hardware.ddr_read_latency_cycles
            + program.hardware.ddr_read_latency_jitter_cycles
            + kTransportGuardCycles;
        const auto lead = duration;
        const auto deadlineStart =
            plan.ready_cycle > lead ? plan.ready_cycle - lead : 0;
        // Only the leading page is known to be safe before any paged weight
        // consumer exists. Later pages retain the compiler's just-in-time
        // lifetime order to avoid overwriting a still-live physical page.
        const auto previousBankRelease = bankReleaseCycles.find(plan.bank);
        const std::uint64_t reusableCycle =
            previousBankRelease == bankReleaseCycles.end()
            ? 0 : previousBankRelease->second;
        plan.start_cycle = index == 0 ? 0
            : std::max({fabricCursor, deadlineStart, reusableCycle});
        plan.transfer_end_cycle = plan.start_cycle + duration;
        if (plan.transfer_end_cycle > plan.ready_cycle)
            throw std::logic_error(
                "C2C weight page cannot meet its first-consumer deadline");
        fabricCursor = plan.transfer_end_cycle;
        bankReleaseCycles[plan.bank] = std::max(
            bankReleaseCycles[plan.bank], plan.release_cycle);
    }
    return plans;
}

} // namespace ftlpu::software::runtime
