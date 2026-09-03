#pragma once

#include "ftlpu/software/runtime/icu_program.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu::software::runtime {

// Physical MEM Macro format version 1. Templates are carried inline per run;
// only the seven most frequent (start-cycle, address) deltas are kept as
// queue-local state. The payload is a packed bitstream and may cross i-MEM
// word boundaries.
inline constexpr std::uint32_t kMemMacroBitstreamVersion = 1;
inline constexpr std::size_t kMemMacroDeltaDictionarySize = 7;

struct MemMacroDelta {
    std::uint32_t start_cycle{0};
    std::int32_t address{0};
};

struct MemMacroBitstreamStats {
    std::size_t command_count{0};
    std::size_t run_count{0};
    std::size_t compact_template_runs{0};
    std::size_t extended_template_runs{0};
    std::size_t dictionary_transitions{0};
    std::size_t escaped_transitions{0};
    std::size_t wide_escaped_transitions{0};
    std::uint64_t stream_bits{0};
    std::uint64_t dictionary_bits{0};

    std::uint64_t physical_bits() const
    {
        return stream_bits + dictionary_bits;
    }
};

struct MemMacroBitstream {
    std::uint32_t version{kMemMacroBitstreamVersion};
    std::uint32_t command_count{0};
    std::array<MemMacroDelta, kMemMacroDeltaDictionarySize> deltas{};
    std::uint8_t delta_count{0};
    std::vector<std::uint8_t> bytes{};
    std::uint64_t bit_count{0};
    MemMacroBitstreamStats stats{};
};

// Encodes one all-Macro MEM queue. Unsupported/large template fields use an
// explicit extended template, while large deltas use a 32+32-bit escape.
MemMacroBitstream encode_mem_macro_bitstream(const QueueProgram& queue);

// Decodes the physical image back to semantic QueueCommands. The queue kind
// and index are supplied by the containing queue descriptor, not stored in
// i-MEM.
QueueProgram decode_mem_macro_bitstream(const MemMacroBitstream& image,
    std::size_t queue_index);

} // namespace ftlpu::software::runtime
