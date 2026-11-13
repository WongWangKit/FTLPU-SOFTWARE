#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace ftlpu::software::runtime {

struct MacroQueueLinkResult {
    QueueProgram queue;
    BinaryQueueControlPartition controls;
    QueueProgram transport;
    std::vector<std::optional<std::size_t>> transport_command_indices;
    // Source command index -> output Macro indices, after chronological sort.
    // Packet continuation words and NOPs have no output indices. A caller
    // must copy each header relocation to every index in its corresponding row.
    std::vector<std::vector<std::size_t>> command_indices;
};

// Links compiler-generated Macro descriptors without changing their issue domains.
// NOPs provide control-boundary coordinates; SYNC/WAIT_EVENT/NOTIFY and
// synchronized MEM transfers are placed in the existing bounded QCTL/transport
// storage. Native FU templates and retired loop records are rejected.
MacroQueueLinkResult link_macro_queue_controls(
    const QueueProgram& queue, std::size_t max_descriptors = 65536,
    bool externalize_controls = false);

// Restores a compact linker recipe; callers must relink and encode before execution.
BinaryProgram macro_queue_link_input(const BinaryProgram& program);

// Final encoding pass over an already linked Native synchronization plan.
// Native transport packets remain byte-for-byte identical; relocations are
// mapped to their final computation or transport store.
void link_queue_controls(BinaryProgram& program, bool encode_macro = true,
    const std::function<void(QueueSyncPlan&)>& link_releases = {});

} // namespace ftlpu::software::runtime
