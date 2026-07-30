#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/target_description.hpp"

#include "ftlpu/dma/descriptor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ftlpu::software::runtime {

struct BindingTransfer {
    MemGlobalAddress24 address{};
    DmaPurpose purpose{DmaPurpose::General};
    std::vector<std::uint8_t> bytes{};
};

// A DMA-addressable, contiguous physical region occupied by one binding.
// byte_size is always a whole number of TargetDescription::vector_bytes.
struct BindingRegion {
    MemGlobalAddress24 address{};
    DmaPurpose purpose{DmaPurpose::General};
    std::size_t byte_size{0};
};

std::vector<BindingRegion> plan_binding_regions(
    const BinaryBinding& binding,
    const TargetDescription& target);

std::vector<BindingTransfer> pack_binding(
    const BinaryBinding& binding,
    std::span<const std::uint8_t> logical_bytes,
    const TargetDescription& target);

std::vector<std::uint8_t> unpack_binding(
    const BinaryBinding& binding,
    const std::vector<BindingTransfer>& transfers,
    const TargetDescription& target);

} // namespace ftlpu::software::runtime
