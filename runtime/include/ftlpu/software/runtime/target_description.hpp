#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ftlpu::software::runtime {

// Runtime-visible hardware contract.  Values come from the selected backend;
// ModelSession and future BinaryProgram adapters must not duplicate CModel
// compile-time constants.
struct TargetDescription {
    std::string name{};
    std::uint64_t abi_fingerprint{0};

    std::size_t tile_rows{0};
    std::size_t lanes_per_tile{0};
    std::size_t vector_bytes{0};

    std::size_t hemisphere_count{0};
    std::size_t mem_slices_per_hemisphere{0};
    std::size_t sram_banks_per_tile{0};
    std::size_t sram_rows_per_bank{0};
    std::size_t sram_word_bytes{0};
    std::size_t global_memory_address_bits{0};

    std::size_t instruction_packet_bytes{0};
    std::size_t ifetch_block_bytes{0};
    std::size_t ifetch_packets{0};

    std::size_t mxm_count{0};
    std::size_t mxm_rows{0};
    std::size_t mxm_columns{0};
    std::size_t mxm_reduction{0};
    std::size_t mxm_weight_bytes_per_value{0};
    std::size_t mxm_activation_bytes_per_value{0};

    std::size_t vxm_alu_count{0};
    std::size_t sxm_count{0};
};

} // namespace ftlpu::software::runtime