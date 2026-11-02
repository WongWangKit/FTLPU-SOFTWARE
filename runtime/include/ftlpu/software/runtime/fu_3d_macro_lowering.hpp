#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace ftlpu::software::runtime {

struct Fu3dMacroLoweringResult {
    QueueProgram queue;
    BinaryQueueControlPartition controls;
    QueueProgram transport;
    std::vector<std::optional<std::size_t>> transport_command_indices;
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
// With externalize_controls, SYNC/WAIT_EVENT/NOTIFY are retained in a
// queue-local sideband plan, before Macro folding. Otherwise only NOPs are accepted.
// Synchronized transfers,
// native4 decode and unsupported records fail explicitly. Descriptor budgets,
// fixed Active Context fields and arithmetic bounds are checked before output.
Fu3dMacroLoweringResult lower_fu_3d_queue_to_macro(
    const QueueProgram& queue, std::size_t max_descriptors = 65536,
    bool externalize_controls = false);

// Restores a compact linker recipe; callers must relink and encode before execution.
BinaryProgram native_queue_link_input(const BinaryProgram& program);

// Final encoding pass over an already linked Native synchronization plan.
// Native transport packets remain byte-for-byte identical; relocations are
// mapped to their final computation or transport store.
void externalize_native_queue_controls(BinaryProgram& program, bool encode_macro = true,
    const std::function<void(QueueSyncPlan&)>& link_releases = {});

} // namespace ftlpu::software::runtime
