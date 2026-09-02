#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu::software::runtime {

struct WeightResidencyRegion {
    std::uint16_t bank{0};
    std::uint16_t hemisphere_mask{0};
    std::uint16_t slice{0};
    std::uint32_t row_begin{0};
    std::uint32_t row_end{0};
    bool wildcard_slice{false};
};

struct WeightPrefetchPlan {
    std::uint32_t page_index{0};
    std::uint16_t bank{0};
    bool pre_execution{false};
    std::uint64_t start_cycle{0};
    std::uint64_t transfer_end_cycle{0};
    std::uint64_t ready_cycle{0};
    std::uint64_t release_cycle{0};
    std::array<std::uint64_t, 2> bytes{};
    std::vector<std::size_t> use_indices{};
    std::vector<WeightResidencyRegion> regions{};
};

// Builds the executable-local C2C schedule from compiler-provided weight
// residency intervals. Pages whose physical regions can coexist are loaded
// through C2C before the ICU executable starts. Later pages receive an
// earliest-safe launch hint derived from physical residency lifetimes. The
// runtime uses page-ready events for correctness; this plan never acts as a
// hardware deadline.
std::vector<WeightPrefetchPlan> plan_weight_prefetches(
    const BinaryProgram& program);
std::vector<WeightPrefetchPlan> plan_weight_prefetches(
    const BinaryProgram& program,
    const ExecutableHardwareConfig& runtime_hardware);

// Recomputes timing after a packer replaces logical page byte estimates with
// exact physical bytes. Plan order, regions, and use membership are retained.
void schedule_weight_prefetches(const BinaryProgram& program,
    std::vector<WeightPrefetchPlan>& plans);
void schedule_weight_prefetches(const BinaryProgram& program,
    std::vector<WeightPrefetchPlan>& plans,
    const ExecutableHardwareConfig& runtime_hardware);

} // namespace ftlpu::software::runtime
