#pragma once

#include "ftlpu/software/runtime/icu_program.hpp"

#include <cstddef>
#include <vector>

namespace ftlpu::software::runtime {

struct Fu3dMacroLoweringResult {
    QueueProgram queue;
    // Source command index -> output Macro indices, after chronological sort.
    // Packet continuation words and NOPs have no output indices. A caller
    // must copy each header relocation to every index in its corresponding row.
    std::vector<std::vector<std::size_t>> command_indices;
};

// Converts a complete native MEM/MXM queue to all-Macro commands, retaining
// exact absolute issue cycles and operands. Work stays in coarse affine domains:
// the implementation never constructs the expanded functional issue stream.
// It splits buffer parity, terminal modes and blocked addresses before reducing
// domains to at most two axes. No transformation combines source packets, so
// relocation provenance remains unambiguous.
//
// Only NOP control records are accepted. Event waits, synchronized transfers,
// native4 decode and unsupported records fail explicitly. Descriptor budgets,
// fixed Active Context fields and arithmetic bounds are checked before output.
Fu3dMacroLoweringResult lower_fu_3d_queue_to_macro(
    const QueueProgram& queue, std::size_t max_descriptors = 65536);

} // namespace ftlpu::software::runtime
