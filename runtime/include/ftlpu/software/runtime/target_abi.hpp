#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstdint>
#include <string_view>

namespace ftlpu::software::runtime {

inline constexpr std::string_view kLpu32StreamTargetName = "lpu_32stream_v1";

class TargetAbiHasher {
public:
    constexpr void add(std::int64_t value)
    {
        const auto bits = static_cast<std::uint64_t>(value);
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value_ ^= static_cast<std::uint8_t>(bits >> shift);
            value_ *= 1099511628211ULL;
        }
    }

    constexpr std::uint64_t value() const { return value_; }

private:
    std::uint64_t value_ = 14695981039346656037ULL;
};

// This list is the command ABI contract for the CModel-backed 32-stream LPU.
// LPUTargetModel::abi_fingerprint() hashes fields in exactly this order.
constexpr std::uint64_t lpu_32stream_target_abi(
    std::int64_t mxms_per_hemisphere = hw::kMxmsPerHemisphere)
{
    TargetAbiHasher hash;
    hash.add(7); // Hardware ABI schema version.

    hash.add(hw::kHemispheres);
    hash.add(hw::kMemSliceColumns);
    hash.add(16);
    hash.add(4096);
    hash.add(16);
    hash.add(hw::kSramDepthRows);
    hash.add(1); // One SRAM read port per slice.
    hash.add(1); // One SRAM write port per slice.

    hash.add(hw::kStreamsPerDirection);
    hash.add(hw::kStreams);
    hash.add(hw::kMemBoundaryStreamRegisterColumns);
    hash.add(hw::kSystemStreamRegisterColumns);
    hash.add(hw::kMemSlicesPerGroup);

    hash.add(hw::kTileRows);
    hash.add(hw::kLanesPerTile);
    hash.add(hw::kMemReadBytesPerCycle);
    hash.add(hw::kMemWriteBytesPerCycle);
    hash.add(hw::kMxmRows);
    hash.add(hw::kMxmColumns);
    hash.add(hw::kMxmLoadStreamsPerCycle);
    hash.add(hw::kMxmInt8LoadStreamsPerCycle);
    hash.add(hw::kMxmLoadBytesPerCycle);
    hash.add(4);
    hash.add(4);
    hash.add(4);
    hash.add(hw::kMxmBlockRows);
    hash.add(1); // MXM-local INT8 dequant is enabled.
    hash.add(1); // Block8 compute is enabled.
    hash.add(1); // Weight/activation transport overlap is enabled.
    hash.add(4); // Wavefront weight-load to compute latency.
    hash.add(8); // Interval between 32-row Block8 groups.
    hash.add(2);
    hash.add(24);
    hash.add(mxms_per_hemisphere);
    hash.add(2);
    hash.add(16);
    hash.add(hw::kMxmBoundaryStreamRegisterColumn + 2);
    hash.add(hw::kMemGroups + 1);
    hash.add(hw::kMemGroups + 2);
    hash.add(6);
    hash.add(5);
    hash.add(hw::kMxmBoundaryStreamRegisterColumn + 4);
    hash.add(hw::kMxmBoundaryStreamRegisterColumn + 1);
    hash.add(13);
    return hash.value();
}

inline constexpr std::uint64_t kLpu32StreamTargetAbi =
    lpu_32stream_target_abi();

} // namespace ftlpu::software::runtime
