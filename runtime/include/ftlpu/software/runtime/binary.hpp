#pragma once

#include "ftlpu/software/runtime/icu_program.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ftlpu::software::runtime {

enum class BindingAccess : std::uint16_t {
    Input = 0,
    Output = 1,
    Internal = 2,
};

enum class BindingElementType : std::uint16_t {
    I8 = 1,
    I32 = 2,
    F16 = 3,
    F32 = 4,
};

enum class BindingLayout : std::uint16_t {
    Vector = 1,
    MxmWeightStriped = 2,
    Int32BytePlanar = 3,
    Fp16BytePlanar = 4,
    Fp16MxmActivationPlanar = 5,
    W8A16MxmWeightStriped = 6,
    Fp16PairPlanar = 7,
    W8A16AttentionWeightStriped = 8,
    W8A16MxmWeightWaveStriped = 9,
    Fp32CausalMaskTile = 10,
};

struct BinaryBinding {
    std::uint32_t index{0};
    BindingAccess access{BindingAccess::Input};
    BindingElementType element_type{BindingElementType::I8};
    BindingLayout layout{BindingLayout::Vector};
    std::uint64_t byte_size{0};
    std::int64_t base_row{0};
    std::int64_t instruction_count{0};
    std::int64_t address_stride{0};
    std::vector<std::uint64_t> shape{};
    std::vector<std::uint16_t> slices{};
    // Bit 0 selects east and bit 1 selects west. Inputs may be replicated to both.
    std::uint16_t hemisphere_mask{1};
};

struct BinaryProgram {
    std::size_t max_cycle{0};
    std::vector<QueueProgram> queues{};
    std::vector<BinaryBinding> bindings{};
};

void write_binary_program(const BinaryProgram& program, const std::filesystem::path& path);
BinaryProgram read_binary_program(const std::filesystem::path& path);

} // namespace ftlpu::software::runtime
