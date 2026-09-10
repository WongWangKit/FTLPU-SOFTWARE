#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu::software::runtime {

// Capacity under the current CModel abstraction: every decoded QueueCommand,
// including NOP and control commands, consumes exactly one physical i-MEM
// slot in its queue. This deliberately does not claim a final RTL bitstream
// encoding for Macro or variable-width SXM instructions.
struct CmodelAbstractImemQueue {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::uint32_t slot_bits{0};
    std::uint32_t depth{0};
    std::size_t used_slots{0};
    std::size_t instruction_entries{0};
    std::size_t nop_entries{0};
    std::size_t repeat_entries{0};
    std::size_t repeat_2d_entries{0};
    std::size_t macro_entries{0};
    std::size_t coarse_program_entries{0};
    std::uint64_t expanded_work{0};

    bool overflow() const { return used_slots > depth; }
    std::size_t overflow_slots() const
    {
        return overflow() ? used_slots - depth : 0;
    }
    std::uint64_t used_bits() const
    {
        return static_cast<std::uint64_t>(used_slots) * slot_bits;
    }
};

struct CmodelAbstractImemReport {
    std::vector<CmodelAbstractImemQueue> queues;
    std::uint64_t used_bits{0};
    std::uint64_t active_queue_capacity_bits{0};
    std::uint64_t expanded_work{0};
    std::size_t used_slots{0};
    std::size_t encoded_work_entries{0};
    std::size_t overflow_queues{0};

    bool fits() const { return overflow_queues == 0; }
};

CmodelAbstractImemReport analyze_cmodel_abstract_imem(
    const BinaryProgram& program);

// Target-facing storage estimate. MEM all-Macro queues use the inline-template
// Delta-RLE v1 codec. MEM/MXM STREAM_ND commands use the fixed 320-bit packet
// decoded by the local ICU; each packet occupies an integral number of that
// queue's physical i-MEM slots.
struct PhysicalImemQueue : CmodelAbstractImemQueue {
    std::uint64_t physical_bits{0};
    std::size_t physical_slots{0};
    bool mem_delta_rle{false};
    std::size_t stream_nd_packets{0};
    MemMacroBitstreamStats macro_codec{};
    std::size_t peak_macro_contexts{0};
    std::size_t macro_context_capacity{0};
    std::uint32_t macro_context_bits{0};

    bool physical_overflow() const { return physical_slots > depth; }
    bool macro_context_overflow() const
    {
        return peak_macro_contexts > macro_context_capacity;
    }
    bool deployment_overflow() const
    {
        return physical_overflow() || macro_context_overflow();
    }
};

struct PhysicalImemReport {
    std::vector<PhysicalImemQueue> queues;
    std::uint64_t used_bits{0};
    std::uint64_t active_queue_capacity_bits{0};
    std::size_t used_slots{0};
    std::size_t overflow_queues{0};
    std::size_t macro_context_overflow_queues{0};
    std::size_t mem_delta_rle_queues{0};
    std::size_t stream_nd_packets{0};
    std::uint64_t peak_macro_context_bits{0};
    std::uint64_t provisioned_macro_context_bits{0};

    bool fits() const { return overflow_queues == 0; }
};

PhysicalImemReport analyze_physical_imem(const BinaryProgram& program);

} // namespace ftlpu::software::runtime
