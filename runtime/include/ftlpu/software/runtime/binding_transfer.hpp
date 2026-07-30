#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/target_description.hpp"

#include "ftlpu/dma/descriptor.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace ftlpu::software::runtime {

struct BindingTransfer {
    MemGlobalAddress24 address{};
    DmaPurpose purpose{DmaPurpose::General};
    std::vector<std::uint8_t> bytes{};
};

std::vector<BindingTransfer> pack_binding(
    const BinaryBinding& binding,
    std::span<const std::uint8_t> logical_bytes,
    const TargetDescription& target);

std::vector<std::uint8_t> unpack_binding(
    const BinaryBinding& binding,
    const std::vector<BindingTransfer>& transfers,
    const TargetDescription& target);

} // namespace ftlpu::software::runtime
