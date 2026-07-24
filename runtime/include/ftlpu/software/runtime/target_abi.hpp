#pragma once

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
constexpr std::uint64_t lpu_32stream_target_abi()
{
    TargetAbiHasher hash;
    hash.add(1); // ABI schema version.

    hash.add(2);
    hash.add(44);
    hash.add(2);
    hash.add(4096);
    hash.add(16);
    hash.add(36);
    hash.add(4);
    hash.add(8);
    hash.add(4);
    hash.add(32);
    hash.add(21);
    hash.add(24);
    hash.add(1600);
    for (const auto value : {1, 5, 9, 13}) hash.add(value);
    for (const auto value : {2, 6, 10, 14}) hash.add(value);

    hash.add(32);
    hash.add(64);
    hash.add(12);
    hash.add(13);
    hash.add(4);

    hash.add(4);
    hash.add(8);
    hash.add(8);
    hash.add(8);
    hash.add(32);
    hash.add(32);
    hash.add(16);
    hash.add(128);
    hash.add(4);
    hash.add(4);
    hash.add(4);
    hash.add(2);
    hash.add(2);
    hash.add(2);
    hash.add(16);
    hash.add(14);
    hash.add(12);
    hash.add(13);
    hash.add(6);
    hash.add(5);
    hash.add(16);
    hash.add(13);
    return hash.value();
}

inline constexpr std::uint64_t kLpu32StreamTargetAbi =
    lpu_32stream_target_abi();

} // namespace ftlpu::software::runtime
