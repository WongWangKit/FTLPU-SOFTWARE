#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/device_backend.hpp"

#include <cstddef>
#include <vector>

namespace ftlpu::software::runtime {

struct AdaptedExecutable {
    DeviceProgram device_program{};
    std::vector<BinaryBinding> bindings{};
};

AdaptedExecutable adapt_binary_program(
    const BinaryProgram& binary,
    const TargetDescription& target,
    std::size_t drain_cycles);

} // namespace ftlpu::software::runtime
