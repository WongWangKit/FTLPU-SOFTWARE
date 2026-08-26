#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu::software::runtime {

struct WeightPrefetchPlan {
    std::uint32_t page_index{0};
    std::uint16_t bank{0};
    std::uint64_t start_cycle{0};
    std::uint64_t transfer_end_cycle{0};
    std::uint64_t ready_cycle{0};
    std::uint64_t release_cycle{0};
    std::array<std::uint64_t, 2> bytes{};
    std::vector<std::size_t> use_indices{};
};

// Builds the executable-local C2C schedule from compiler-provided weight
// residency intervals. The first page starts with the executable so that its
// transfer overlaps leading layout/RMSNorm work; later pages are deadline
// scheduled with a small transport guard.
std::vector<WeightPrefetchPlan> plan_weight_prefetches(
    const BinaryProgram& program);

} // namespace ftlpu::software::runtime
