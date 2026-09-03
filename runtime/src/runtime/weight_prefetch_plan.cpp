#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include <algorithm>
#include <limits>
#include <ranges>
#include <sstream>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

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

std::vector<WeightResidencyRegion> binding_regions(
    const BinaryBinding& binding, std::uint16_t bank)
{
    std::vector<std::uint16_t> slices;
    if (!binding.page_storage_slices.empty()) {
        const std::size_t groupWidth = binding.slices.empty()
            ? 8 : binding.slices.size();
        const std::size_t begin = std::min<std::size_t>(
            binding.page_storage_slices.size(),
            static_cast<std::size_t>(binding.page_role_group_base)
                * groupWidth);
        const std::size_t end = std::min<std::size_t>(
            binding.page_storage_slices.size(),
            static_cast<std::size_t>(binding.page_role_group_base
                + binding.page_role_group_count) * groupWidth);
        slices.assign(binding.page_storage_slices.begin() + begin,
            binding.page_storage_slices.begin() + end);
    } else {
        slices = binding.slices;
    }

    const std::uint32_t rowBegin = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(std::numeric_limits<std::uint32_t>::max(),
            static_cast<std::uint64_t>(std::max<int64_t>(0,
                binding.base_row))));
    const std::uint64_t rowSpan = std::max<std::uint64_t>(1,
        static_cast<std::uint64_t>(binding.instruction_count)
            * std::max<std::uint32_t>(1, binding.address_stride));
    const std::uint64_t rowEnd = std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(),
        static_cast<std::uint64_t>(rowBegin) + rowSpan);
    std::vector<WeightResidencyRegion> regions;
    if (slices.empty()) {
        regions.push_back({bank, binding.hemisphere_mask, 0,
            rowBegin, static_cast<std::uint32_t>(rowEnd), true});
        return regions;
    }
    std::ranges::sort(slices);
    const auto unique = std::ranges::unique(slices);
    slices.erase(unique.begin(), unique.end());
    regions.reserve(slices.size());
    for (const std::uint16_t slice : slices)
        regions.push_back({bank, binding.hemisphere_mask, slice,
            rowBegin, static_cast<std::uint32_t>(rowEnd), false});
    return regions;
}

bool regions_overlap(const WeightResidencyRegion& lhs,
    const WeightResidencyRegion& rhs)
{
    return lhs.bank == rhs.bank
        && (lhs.hemisphere_mask & rhs.hemisphere_mask) != 0
        && (lhs.wildcard_slice || rhs.wildcard_slice
            || lhs.slice == rhs.slice)
        && lhs.row_begin < rhs.row_end && rhs.row_begin < lhs.row_end;
}

bool plans_overlap(
    const WeightPrefetchPlan& lhs, const WeightPrefetchPlan& rhs)
{
    return std::ranges::any_of(lhs.regions, [&](const auto& left) {
        return std::ranges::any_of(rhs.regions, [&](const auto& right) {
            return regions_overlap(left, right);
        });
    });
}

} // namespace

std::vector<WeightPrefetchPlan> plan_weight_prefetches(
    const BinaryProgram& program)
{
    return plan_weight_prefetches(program, program.hardware);
}

std::vector<WeightPrefetchPlan> plan_weight_prefetches(
    const BinaryProgram& program,
    const ExecutableHardwareConfig& runtimeHardware)
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
    const std::uint64_t clockMhz = runtimeHardware.lpu_clock_mhz;
    const std::uint64_t ddrBandwidth =
        runtimeHardware.ddr_peak_bandwidth_mbytes_per_second;
    const std::uint64_t ddrEfficiency =
        runtimeHardware.ddr_scheduling_efficiency_percent;
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
                return !candidate.use_indices.empty()
                    && program.weight_page_uses[
                        candidate.use_indices.front()].binding_index
                        == use.binding_index
                    && candidate.page_index == use.page_index
                    && candidate.bank == use.bank
                    && use.ready_cycle <= candidate.release_cycle
                    && candidate.ready_cycle <= use.release_cycle;
            });
        if (found == plans.end()) {
            WeightPrefetchPlan plan;
            plan.page_index = use.page_index;
            plan.bank = use.bank;
            plan.ready_cycle = use.ready_cycle;
            plan.release_cycle = use.release_cycle;
            plan.use_indices.push_back(useIndex);
            plan.regions = binding_regions(
                find_paged_weight(program, use.binding_index), use.bank);
            plans.push_back(std::move(plan));
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

    schedule_weight_prefetches(program, plans, runtimeHardware);
    return plans;
}

void schedule_weight_prefetches(const BinaryProgram& program,
    std::vector<WeightPrefetchPlan>& plans)
{
    schedule_weight_prefetches(program, plans, program.hardware);
}

void schedule_weight_prefetches(const BinaryProgram& program,
    std::vector<WeightPrefetchPlan>& plans,
    const ExecutableHardwareConfig& runtimeHardware)
{
    std::ranges::sort(plans, [](const auto& lhs, const auto& rhs) {
        return lhs.ready_cycle < rhs.ready_cycle;
    });
    for (auto& plan : plans) {
        plan.pre_execution = false;
        plan.start_cycle = 0;
        plan.transfer_end_cycle = 0;
    }
    const std::uint64_t lanes =
        program.hardware.c2c_streams_per_direction;
    const std::uint64_t bytesPerLane =
        program.hardware.c2c_bytes_per_stream_per_cycle;
    if (lanes == 0 || bytesPerLane == 0)
        throw std::logic_error(
            "paged weights require non-zero C2C bandwidth");
    const std::uint64_t bandwidth = lanes * bytesPerLane;
    const std::uint64_t clockMhz = runtimeHardware.lpu_clock_mhz;
    const std::uint64_t ddrBandwidth =
        runtimeHardware.ddr_peak_bandwidth_mbytes_per_second;
    const std::uint64_t ddrEfficiency =
        runtimeHardware.ddr_scheduling_efficiency_percent;
    if (clockMhz == 0 || ddrBandwidth == 0 || ddrEfficiency == 0
        || ddrEfficiency > 100)
        throw std::logic_error(
            "paged weights require a valid runtime external-memory model");

    std::vector<std::uint64_t> durations(plans.size());
    std::vector<std::uint64_t> reusableCycles(plans.size());
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
        const auto queueDrain =
            (static_cast<std::uint64_t>(
                 runtimeHardware.ddr_request_queue_depth)
                + lanes - 1)
            / lanes;
        const auto transportGuard = queueDrain
            + hw::kMemEastBoundaryStreamRegisterColumn
            + hw::kTileRows + lanes;
        durations[index] = std::max(c2cCycles, ddrCycles)
            + runtimeHardware.ddr_read_latency_cycles
            + runtimeHardware.ddr_read_latency_jitter_cycles
            + transportGuard;
        // The compiler's dedicated-slice layout keeps activation scratch out
        // of the weight residency regions represented here. Consequently,
        // physically disjoint pages may be loaded before execution; pages that
        // share a bank/slice/row range retain their release-ordered JIT load.
        bool canPreload = true;
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (!plans_overlap(plan, plans[previous])) continue;
            // DDR latency is intentionally nondeterministic. With no staging
            // buffer between DDR and the shared C2C/MEM path, launching before
            // the old page's release could let an early response overwrite
            // live SRAM. The event gate therefore opens no earlier than the
            // compiler-provided physical release cycle.
            const auto safeDmaStart = plans[previous].release_cycle;
            reusableCycles[index] = std::max(
                reusableCycles[index], safeDmaStart);
            if (plans[previous].pre_execution) canPreload = false;
        }
        plan.pre_execution = canPreload;
        if (plan.pre_execution)
            plan.transfer_end_cycle = durations[index];
    }

    std::uint64_t nextQueueCursor = 0;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        auto& plan = plans[index];
        if (plan.pre_execution) continue;
        // Once every overlapping SRAM region has been released, retaining
        // the transfer until just before its consumer only reduces the time
        // available to absorb DDR latency and jitter. Launch at the earliest
        // physically safe cycle; page-ready synchronization still protects
        // the consumer when runtime bandwidth is lower than planned.
        plan.start_cycle = std::max(
            reusableCycles[index], nextQueueCursor);
        plan.transfer_end_cycle = plan.start_cycle + durations[index];
        std::array<std::uint64_t, hw::kHemispheres> segmentCounts{};
        for (const auto& region : plan.regions)
            for (std::size_t side = 0; side < hw::kHemispheres; ++side)
                if ((region.hemisphere_mask & (1u << side)) != 0)
                    ++segmentCounts[side];
        // One queue cycle releases WAIT_EVENT, followed by one issue per
        // segment on the busiest hemisphere.
        nextQueueCursor = plan.start_cycle + 1
            + *std::max_element(segmentCounts.begin(), segmentCounts.end());
    }
}

} // namespace ftlpu::software::runtime
