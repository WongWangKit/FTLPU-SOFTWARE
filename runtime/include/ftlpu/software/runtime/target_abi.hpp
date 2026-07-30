#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ftlpu::software::runtime {

inline constexpr std::string_view kLpu32StreamTargetName = "lpu_32stream_v1";

struct ExecutableTargetIdentity {
    std::string name{};
    std::uint64_t abi{0};

    friend bool operator==(
        const ExecutableTargetIdentity&,
        const ExecutableTargetIdentity&) = default;
};

// Values shared by compiler scheduling and backend execution. This schema is
// deliberately distinct from a backend implementation/version fingerprint.
struct ExecutableTargetParameters {
    std::int64_t hemisphere_count{0};
    std::int64_t mem_slices_per_hemisphere{0};
    std::int64_t sram_banks_per_tile{0};
    std::int64_t sram_rows_per_bank{0};
    std::int64_t sram_word_bytes{0};
    std::int64_t tile_rows{0};
    std::int64_t lanes_per_tile{0};
    std::int64_t vector_bytes{0};
    std::int64_t streams_per_direction{0};
    std::int64_t mem_read_bytes_per_cycle{0};
    std::int64_t mem_write_bytes_per_cycle{0};
    std::int64_t mxm_count{0};
    std::int64_t mxm_rows{0};
    std::int64_t mxm_columns{0};
    std::int64_t mxm_load_streams_per_cycle{0};
    std::int64_t mxm_load_bytes_per_cycle{0};
    std::int64_t vxm_alu_count{0};
};

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

constexpr std::uint64_t executable_target_abi(
    const ExecutableTargetParameters& target)
{
    TargetAbiHasher hash;
    hash.add(2); // Executable ABI schema version.
    hash.add(target.hemisphere_count);
    hash.add(target.mem_slices_per_hemisphere);
    hash.add(target.sram_banks_per_tile);
    hash.add(target.sram_rows_per_bank);
    hash.add(target.sram_word_bytes);
    hash.add(target.tile_rows);
    hash.add(target.lanes_per_tile);
    hash.add(target.vector_bytes);
    hash.add(target.streams_per_direction);
    hash.add(target.mem_read_bytes_per_cycle);
    hash.add(target.mem_write_bytes_per_cycle);
    hash.add(target.mxm_count);
    hash.add(target.mxm_rows);
    hash.add(target.mxm_columns);
    hash.add(target.mxm_load_streams_per_cycle);
    hash.add(target.mxm_load_bytes_per_cycle);
    hash.add(target.vxm_alu_count);
    return hash.value();
}

// Frozen legacy identity retained only for reading/testing older binaries.
// New compiler emissions use executable_target_abi() above.
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
    hash.add(24);
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
    hash.add(13);
    return hash.value();
}

inline constexpr std::uint64_t kLpu32StreamTargetAbi =
    lpu_32stream_target_abi();

} // namespace ftlpu::software::runtime
