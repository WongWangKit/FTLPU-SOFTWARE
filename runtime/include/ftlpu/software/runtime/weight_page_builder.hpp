#pragma once

#include "ftlpu/software/runtime/model_package.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace ftlpu::software::runtime {

struct PackedWeightSegment {
    std::uint64_t byte_offset{0};
    std::uint16_t hemisphere{0};
    std::uint16_t slice{0};
    std::uint32_t base_row{0};
    std::uint32_t vector_count{0};
};

struct PackedWeightImage {
    std::vector<std::uint8_t> data{};
    std::vector<PackedWeightSegment> segments{};
};

// Converts one logical runtime tensor into the exact 32-byte SRAM vectors
// consumed by its executable binding. This is the offline counterpart of a
// host upload and is the only representation accepted by C2C weight pages.
PackedWeightImage pack_weight_binding(
    const BinaryBinding& binding, std::span<const std::uint8_t> logical_data,
    const ExecutableHardwareConfig& hardware);

struct WeightPageBuildOptions {
    std::uint16_t first_bank{0};
    bool remove_logical_weights{true};
};

// Rewrites weight inputs to target-packed tensors, creates one alternating
// bank page per invocation, and rejects physical overlap between page-resident
// weights. Executables must already target the selected bank.
void build_weight_pages(ModelPackage& package,
    const WeightPageBuildOptions& options = {});

} // namespace ftlpu::software::runtime
