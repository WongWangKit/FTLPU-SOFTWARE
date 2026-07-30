#pragma once

#include "ftlpu/core/instruction_codec.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu::software::runtime {

enum class QueueKind : std::uint16_t {
    Mem = 0,
    MxmLoad = 1,
    MxmCompute = 2,
    Vxm = 4,
    SxmTranspose = 5,
    SxmPermute = 6,
};

enum class InstructionKind : std::uint16_t {
    None = 0,
    Mem = 1,
    Mxm = 2,
    Vxm = 3,
    Sxm = 4,
};

struct QueueCommand {
    isa::EncodedIcuCommand command{0};
    InstructionKind instruction_kind{InstructionKind::None};
    std::uint16_t word_count{0};
    std::array<std::uint32_t, 4> words{};
    // SXM carries variable stream lists and a lane map in this trailing
    // payload. Other instruction kinds use only the fixed words above.
    std::vector<std::uint32_t> extension_words{};
};

struct QueueProgram {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::vector<QueueCommand> commands{};
};

constexpr const char* queue_kind_name(QueueKind kind) noexcept
{
    switch (kind) {
    case QueueKind::Mem: return "MEM";
    case QueueKind::MxmLoad: return "MXM load";
    case QueueKind::MxmCompute: return "MXM compute";
    case QueueKind::Vxm: return "VXM";
    case QueueKind::SxmTranspose: return "SXM transpose";
    case QueueKind::SxmPermute: return "SXM permute";
    }
    return "unknown";
}

} // namespace ftlpu::software::runtime
