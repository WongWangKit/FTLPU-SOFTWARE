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
inline constexpr std::uint32_t kMxmMacroBitstreamVersion = 1;
inline constexpr std::size_t kMxmMacroDeltaDictionarySize = 7;
inline constexpr std::uint32_t kMemIcuImemWordBits = 96;
inline constexpr std::uint32_t kMxmIcuImemWordBits = 128;

// Physical queue control word stored in the low bits of i-MEM word 0. The
// remaining bits in the 96/128-bit word are reserved and must be zero. Queue
// kind and codec version are fixed by the owning ICU and QueueMode.
enum class IcuQueueMode : std::uint8_t {
    Native = 0,
    Macro = 1,
    Reserved2 = 2,
    Reserved3 = 3,
};

inline constexpr unsigned kIcuQueueControlValidBit = 0;
inline constexpr unsigned kIcuQueueControlEnableBit = 1;
inline constexpr unsigned kIcuQueueControlModeShift = 2;
inline constexpr unsigned kIcuQueueControlModeBits = 2;
inline constexpr unsigned kIcuQueueControlCommandCountShift = 4;
inline constexpr unsigned kIcuQueueControlCommandCountBits = 24;
inline constexpr std::uint32_t kIcuQueueControlMaxCommandCount =
    (std::uint32_t{1} << kIcuQueueControlCommandCountBits) - 1;

struct IcuQueueControl {
    bool valid{false};
    bool enable{false};
    IcuQueueMode mode{IcuQueueMode::Native};
    std::uint32_t command_count{0};
};

std::uint32_t encode_icu_queue_control(const IcuQueueControl& control);
IcuQueueControl decode_icu_queue_control(std::uint32_t word);

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

PackedMacroImemImage pack_mem_macro_imem(const MemMacroBitstream& image,
                                         bool enable = true,
                                         bool valid = true);
QueueProgram decode_mem_macro_imem(const PackedMacroImemImage& image,
                                   std::size_t queue_index);

struct MxmMacroDelta {
    std::uint32_t start_cycle{0};
    std::int32_t operand{0};
};

using MxmMacroBitstreamStats = MemMacroBitstreamStats;

struct MxmMacroBitstream {
    std::uint32_t version{kMxmMacroBitstreamVersion};
    QueueKind kind{QueueKind::MxmLoad};
    std::uint32_t command_count{0};
    std::array<MxmMacroDelta, kMxmMacroDeltaDictionarySize> deltas{};
    std::uint8_t delta_count{0};
    std::vector<std::uint8_t> bytes{};
    std::uint64_t bit_count{0};
    MxmMacroBitstreamStats stats{};
};

// Encodes one all-Macro MXM queue. Queue kind is out-of-band QueueMode state;
// templates and the seven common (start-cycle, induced-operand) deltas are
// packed across fixed-width 128-bit i-MEM fetch words.
MxmMacroBitstream encode_mxm_macro_bitstream(const QueueProgram& queue);

QueueProgram decode_mxm_macro_bitstream(const MxmMacroBitstream& image,
                                        std::size_t queue_index);

PackedMacroImemImage pack_mxm_macro_imem(const MxmMacroBitstream& image,
                                         bool enable = true,
                                         bool valid = true);
QueueProgram decode_mxm_macro_imem(const PackedMacroImemImage& image,
                                   QueueKind kind,
                                   std::size_t queue_index);

} // namespace ftlpu::software::runtime
