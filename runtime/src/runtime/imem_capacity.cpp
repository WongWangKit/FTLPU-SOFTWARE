#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <limits>
#include <algorithm>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

std::pair<std::uint32_t, std::uint32_t> geometry_for(
    QueueKind kind, const ExecutableHardwareConfig& hardware)
{
    switch (kind) {
    case QueueKind::Mem:
        return {hardware.icu_mem_instruction_bits,
            hardware.icu_mem_imem_depth};
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return {hardware.icu_mxm_instruction_bits,
            hardware.icu_mxm_imem_depth};
    case QueueKind::Vxm:
        return {hardware.icu_vxm_instruction_bits,
            hardware.icu_vxm_imem_depth};
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return {hardware.icu_sxm_instruction_bits,
            hardware.icu_sxm_imem_depth};
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return {static_cast<std::uint32_t>(hw::kIcuC2cInstructionBits),
            static_cast<std::uint32_t>(hw::kIcuC2cImemDepth)};
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

std::size_t fu_3d_packet_words(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return isa::EncodedMemIcu3DPacket::kWordCount;
    case QueueKind::MxmLoad:
        return isa::EncodedMxmLoadIcu3DPacket::kWordCount;
    case QueueKind::MxmCompute:
        return isa::EncodedMxmComputeIcu3DPacket::kWordCount;
    case QueueKind::MxmDequant:
        return isa::EncodedMxmDequantIcu3DPacket::kWordCount;
    case QueueKind::Vxm:
        return isa::EncodedVxmIcuRun2DPacket::kWordCount;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return isa::EncodedSxmIcuRun2DPacket::kWordCount;
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        break;
    }
    throw std::invalid_argument(
        "FU 3-D packet is unsupported for this physical ICU queue");
}

bool is_mem_write_read_2d_header(
    const QueueProgram& queue, std::size_t index)
{
    if (queue.kind != QueueKind::Mem
        || !is_fu_3d_raw_packet_header(queue.commands.at(index)))
        return false;
    const auto& header = queue.commands[index];
    return ((header.words[2] >> 24) & 0xfU) == 7
        && ((header.words[0] >> 4) & 0x3U) == 3;
}

MemIcuWriteRead2DInstruction decode_mem_write_read_2d(
    const QueueProgram& queue, std::size_t index)
{
    if (!is_mem_write_read_2d_header(queue, index)
        || index + isa::EncodedMemIcuWriteRead2DPacket::kWordCount
            > queue.commands.size())
        throw std::logic_error("truncated MEM WRITE_READ_2D packet");
    isa::EncodedMemIcuWriteRead2DPacket packet{};
    for (std::size_t word = 0; word < packet.words.size(); ++word) {
        const auto& command = queue.commands[index + word];
        if (command.instruction_kind != InstructionKind::Mem
            || command.word_count != packet.kLanesPerWord)
            throw std::logic_error("invalid MEM WRITE_READ_2D packet word");
        packet.words[word].lanes = {
            command.words[0], command.words[1], command.words[2]};
    }
    return isa::decode_mem_icu_write_read_2d_instruction(packet);
}

void checked_add(std::uint64_t& destination, std::uint64_t value)
{
    if (value > std::numeric_limits<std::uint64_t>::max() - destination)
        throw std::overflow_error("i-MEM capacity statistic overflow");
    destination += value;
}

std::uint64_t stream_points(const IcuStreamNdSchedule& schedule)
{
    std::uint64_t points = 1;
    for (std::size_t dimension = 0; dimension < schedule.rank;
         ++dimension) {
        if (schedule.counts[dimension]
            > std::numeric_limits<std::uint64_t>::max() / points)
            throw std::overflow_error("STREAM_ND work statistic overflow");
        points *= schedule.counts[dimension];
    }
    return points;
}

std::uint64_t loop_3d_points(const IcuLoop3D& loop)
{
    std::uint64_t points = 1;
    for (const auto count : loop.counts) {
        if (count > std::numeric_limits<std::uint64_t>::max() / points)
            throw std::overflow_error("FU 3-D work statistic overflow");
        points *= count;
    }
    return points;
}

std::uint64_t synchronized_mem_points(const QueueCommand& header)
{
    if (!is_mem_synchronized_raw_packet_header(header))
        throw std::logic_error(
            "queue command is not a synchronized MEM header");
    return static_cast<std::uint64_t>((header.words[0] >> 2) & 0xffffU)
        + 1;
}

using ContextInterval = std::pair<std::uint64_t, std::uint64_t>;

std::size_t peak_contexts(std::vector<ContextInterval> intervals)
{
    std::sort(intervals.begin(), intervals.end());
    std::vector<std::uint64_t> activeEnds;
    std::size_t peak = 0;
    for (const auto [start, end] : intervals) {
        activeEnds.erase(std::remove_if(activeEnds.begin(), activeEnds.end(),
            [start](std::uint64_t activeEnd) { return activeEnd < start; }),
            activeEnds.end());
        activeEnds.push_back(end);
        peak = std::max(peak, activeEnds.size());
    }
    return peak;
}

ContextInterval loop_3d_interval(
    const IcuLoop3D& loop, std::uint64_t queueCursor = 0)
{
    const auto start = queueCursor
        + static_cast<std::uint64_t>(loop.start_cycle)
        + static_cast<std::uint64_t>(loop.wait_cycle);
    std::uint64_t end = start;
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension)
        end += static_cast<std::uint64_t>(loop.counts[dimension] - 1)
            * loop.cycle_strides[dimension];
    return {start, end};
}

ContextInterval stream_interval(const IcuStreamNdSchedule& schedule,
    std::size_t cycleOffset = 0)
{
    std::uint64_t start = schedule.start_cycle + cycleOffset;
    std::uint64_t end = start;
    for (std::size_t dimension = 0;
         dimension < schedule.rank; ++dimension)
        end += static_cast<std::uint64_t>(schedule.counts[dimension] - 1)
            * schedule.cycle_strides[dimension];
    return {start, end};
}

ContextInterval macro_interval(const IcuMacroSchedule& schedule)
{
    const auto end = static_cast<std::uint64_t>(schedule.start_cycle)
        + static_cast<std::uint64_t>(schedule.inner_count - 1)
            * schedule.inner_interval
        + static_cast<std::uint64_t>(schedule.outer_count - 1)
            * schedule.outer_interval;
    return {schedule.start_cycle, end};
}

std::size_t peak_macro_contexts(const QueueProgram& queue)
{
    std::vector<ContextInterval> intervals;
    intervals.reserve(queue.commands.size());
    const auto addStream = [&](const IcuStreamNdSchedule& schedule,
                               std::size_t cycleOffset = 0) {
        intervals.push_back(stream_interval(schedule, cycleOffset));
    };
    for (std::size_t index = 0; index < queue.commands.size(); ++index) {
        const auto& command = queue.commands[index];
        if (is_mem_synchronized_raw_packet_header(command)) {
            static_cast<void>(decode_mem_synchronized_icu_packet(
                queue, index));
            index += InstructionControlUnit::MemIcu::
                synchronized_packet_word_count - 1;
            continue;
        }
        if (is_mem_synchronized_raw_word_command(command)
            && isa::decode_icu_command_opcode(command.command)
                == isa::IcuCommandOpcode::Instruction)
            throw std::logic_error(
                "orphan MEM_WRITE_SYNC continuation word in binary queue");
        if (is_fu_3d_raw_packet_header(command)) {
            // Raw packets use the dedicated FU 3-D context counted below,
            // not the legacy Macro/STREAM_ND descriptor context.
            if (is_mem_write_read_2d_header(queue, index))
                static_cast<void>(decode_mem_write_read_2d(queue, index));
            else
                static_cast<void>(decode_fu_3d_raw_packet_loop(queue, index));
            index += fu_3d_raw_packet_word_count(queue.kind) - 1;
            continue;
        }
        if (is_fu_3d_raw_word_command(command))
            throw std::logic_error(
                "orphan FU 3-D continuation word in binary queue");
        if (is_mem_slice_program_command(command)) {
            const auto program = decode_mem_slice_program_command(command);
            // Each body entry becomes an independently active N-D context in
            // the local ICU calendar. Counting the parent as one context
            // hides the principal hardware cost of MEM_SLICE_PROGRAM.
            for (const auto& body : program.body)
                addStream(program.schedule, body.cycle_offset);
            continue;
        }
        if (is_mem_stream_nd_command(command)) {
            addStream(decode_mem_stream_nd_command(command));
            continue;
        }
        if (is_mxm_stream_nd_command(command)) {
            addStream(decode_mxm_stream_nd_command(command));
            continue;
        }
        if (!is_macro_schedule_command(command)) continue;
        intervals.push_back(
            macro_interval(decode_macro_schedule_command(command)));
    }
    return peak_contexts(std::move(intervals));
}

bool has_fu_3d_context(QueueKind kind)
{
    return kind == QueueKind::Mem || kind == QueueKind::MxmLoad
        || kind == QueueKind::MxmCompute
        || kind == QueueKind::MxmDequant || kind == QueueKind::Vxm
        || kind == QueueKind::SxmTranspose
        || kind == QueueKind::SxmPermute;
}

std::size_t peak_fu_3d_contexts(const QueueProgram& queue)
{
    if (!has_fu_3d_context(queue.kind)) return 0;

    std::vector<ContextInterval> intervals;
    intervals.reserve(queue.commands.size());
    std::uint64_t cursor = 0;
    std::uint64_t previousIssueCycle = 0;
    bool hasPreviousIssue = false;
    for (std::size_t index = 0; index < queue.commands.size(); ++index) {
        const auto& command = queue.commands[index];
        if (is_mem_synchronized_raw_packet_header(command)) {
            static_cast<void>(decode_mem_synchronized_icu_packet(
                queue, index));
            index += InstructionControlUnit::MemIcu::
                synchronized_packet_word_count - 1;
            continue;
        }
        if (is_mem_synchronized_raw_word_command(command)
            && isa::decode_icu_command_opcode(command.command)
                == isa::IcuCommandOpcode::Instruction)
            throw std::logic_error(
                "orphan MEM_WRITE_SYNC continuation word in binary queue");
        if (is_fu_3d_raw_packet_header(command)) {
            const auto interval = is_mem_write_read_2d_header(queue, index)
                ? ContextInterval {cursor, cursor +
                    detail::mem_icu_write_read_2d_last_issue_cycle(
                        decode_mem_write_read_2d(queue, index))}
                : loop_3d_interval(
                    decode_fu_3d_raw_packet_loop(queue, index), cursor);
            intervals.push_back(interval);
            previousIssueCycle = interval.second;
            hasPreviousIssue = true;
            cursor = interval.second + 1;
            index += fu_3d_raw_packet_word_count(queue.kind) - 1;
            continue;
        }
        if (is_fu_3d_raw_word_command(command))
            throw std::logic_error(
                "orphan FU 3-D continuation word in binary queue");
        if (is_mem_stream_nd_command(command)) {
            const auto interval = stream_interval(
                decode_mem_stream_nd_command(command));
            intervals.push_back(interval);
            cursor = std::max(cursor, interval.second + 1);
            continue;
        }
        if (is_mxm_stream_nd_command(command)) {
            const auto interval = stream_interval(
                decode_mxm_stream_nd_command(command));
            intervals.push_back(interval);
            cursor = std::max(cursor, interval.second + 1);
            continue;
        }
        if (is_vxm_stream_nd_command(command)) {
            const auto interval = stream_interval(
                decode_vxm_stream_nd_command(command).schedule);
            intervals.push_back(interval);
            cursor = std::max(cursor, interval.second + 1);
            continue;
        }
        if (is_sxm_tile_program_command(command)) {
            const auto interval = stream_interval(
                decode_sxm_tile_program_command(command).schedule);
            intervals.push_back(interval);
            cursor = std::max(cursor, interval.second + 1);
            continue;
        }
        if (is_macro_schedule_command(command)) {
            const auto interval =
                macro_interval(decode_macro_schedule_command(command));
            intervals.push_back(interval);
            cursor = std::max(cursor, interval.second + 1);
            continue;
        }
        if (is_repeat_2d_command(command)) {
            if (!hasPreviousIssue)
                throw std::logic_error(
                    "Repeat2D without a preceding instruction in i-MEM capacity analysis");
            const auto repeat = decode_repeat_2d_command(command);
            cursor = previousIssueCycle
                + static_cast<std::uint64_t>(repeat.outer_count - 1)
                    * repeat.outer_interval
                + static_cast<std::uint64_t>(repeat.inner_count - 1)
                    * repeat.inner_interval
                + 1;
            continue;
        }

        switch (isa::decode_icu_command_opcode(command.command)) {
        case isa::IcuCommandOpcode::Instruction:
            previousIssueCycle = cursor;
            hasPreviousIssue = true;
            ++cursor;
            break;
        case isa::IcuCommandOpcode::Nop:
            cursor += isa::decode_icu_nop_cycles(command.command);
            break;
        case isa::IcuCommandOpcode::Repeat: {
            if (!hasPreviousIssue)
                throw std::logic_error(
                    "Repeat without a preceding instruction in i-MEM capacity analysis");
            const auto repeat = isa::decode_icu_repeat(command.command);
            if (repeat.count != 0)
                cursor = previousIssueCycle
                    + static_cast<std::uint64_t>(repeat.count)
                        * repeat.interval
                    + 1;
            break;
        }
        default:
            break;
        }
    }
    return peak_contexts(std::move(intervals));
}

std::size_t macro_context_capacity_for(QueueKind kind,
    const ExecutableHardwareConfig& hardware)
{
    switch (kind) {
    case QueueKind::Mem:
        return hardware.icu_mem_macro_contexts;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return hardware.icu_mxm_macro_contexts;
    case QueueKind::Vxm:
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return 0;
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

std::uint32_t macro_context_bits_for(QueueKind kind,
    const ExecutableHardwareConfig& hardware)
{
    switch (kind) {
    case QueueKind::Mem:
        return hardware.icu_mem_macro_context_bits;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return hardware.icu_mxm_macro_context_bits;
    case QueueKind::Vxm:
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return 0;
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

std::size_t fu_3d_context_capacity_for(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return hw::kIcuMem3DContextDepth;
    case QueueKind::MxmLoad: return hw::kIcuMxmLoad3DContextDepth;
    case QueueKind::MxmCompute: return hw::kIcuMxmCompute3DContextDepth;
    case QueueKind::MxmDequant: return hw::kIcuMxmDequant3DContextDepth;
    case QueueKind::Vxm:
        return hw::kIcuVxmRun2DContextDepth;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return hw::kIcuSxmRun2DContextDepth;
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return 0;
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

std::uint32_t fu_3d_context_bits_for(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return static_cast<std::uint32_t>(hw::kIcuMem3DContextBits);
    case QueueKind::MxmLoad:
        return static_cast<std::uint32_t>(hw::kIcuMxmLoad3DContextBits);
    case QueueKind::MxmCompute:
        return static_cast<std::uint32_t>(hw::kIcuMxmCompute3DContextBits);
    case QueueKind::MxmDequant:
        return static_cast<std::uint32_t>(hw::kIcuMxmDequant3DContextBits);
    case QueueKind::Vxm:
        return static_cast<std::uint32_t>(hw::kIcuVxmRun2DContextBits);
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return static_cast<std::uint32_t>(
            hw::kIcuSxmRun2DContextBits);
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return 0;
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

} // namespace

CmodelAbstractImemReport analyze_cmodel_abstract_imem(
    const BinaryProgram& program)
{
    CmodelAbstractImemReport report;
    report.queues.reserve(program.queues.size());
    for (const auto& queue : program.queues) {
        CmodelAbstractImemQueue capacity;
        capacity.kind = queue.kind;
        capacity.index = queue.index;
        const auto [slotBits, depth] = geometry_for(queue.kind,
            program.hardware);
        capacity.slot_bits = slotBits;
        capacity.depth = depth;
        capacity.used_slots = queue.commands.size();

        for (std::size_t index = 0;
             index < queue.commands.size(); ++index) {
            const auto& command = queue.commands[index];
            if (is_mem_synchronized_raw_packet_header(command)) {
                static_cast<void>(decode_mem_synchronized_icu_packet(
                    queue, index));
                ++capacity.coarse_program_entries;
                checked_add(capacity.expanded_work,
                    synchronized_mem_points(command));
                index += InstructionControlUnit::MemIcu::
                    synchronized_packet_word_count - 1;
                continue;
            }
            if (is_mem_synchronized_raw_word_command(command)
                && isa::decode_icu_command_opcode(command.command)
                    == isa::IcuCommandOpcode::Instruction)
                throw std::logic_error(
                    "orphan MEM_WRITE_SYNC continuation word in binary queue");
            if (queue.kind == QueueKind::C2cDma
                && command.instruction_kind == InstructionKind::C2cDma) {
                static_cast<void>(decode_c2c_dma_icu_packet(queue, index));
                ++capacity.instruction_entries;
                checked_add(capacity.expanded_work, 1);
                index += C2cDmaIcuPacket::kWordCount - 1;
                continue;
            }
            if (is_fu_3d_raw_packet_header(command)) {
                ++capacity.coarse_program_entries;
                if (is_mem_write_read_2d_header(queue, index)) {
                    const auto instruction =
                        decode_mem_write_read_2d(queue, index);
                    checked_add(capacity.expanded_work,
                        2ULL * instruction.counts[0]
                            * instruction.counts[1]);
                } else {
                    checked_add(capacity.expanded_work,
                        loop_3d_points(
                            decode_fu_3d_raw_packet_loop(queue, index)));
                }
                index += fu_3d_raw_packet_word_count(queue.kind) - 1;
                continue;
            }
            if (is_fu_3d_raw_word_command(command))
                throw std::logic_error(
                    "orphan FU 3-D continuation word in binary queue");
            if (is_mem_slice_program_command(command)) {
                ++capacity.coarse_program_entries;
                const auto program =
                    decode_mem_slice_program_command(command);
                const auto points = stream_points(program.schedule);
                if (program.body.size()
                    > std::numeric_limits<std::uint64_t>::max() / points)
                    throw std::overflow_error(
                        "MEM_SLICE_PROGRAM work statistic overflow");
                checked_add(capacity.expanded_work,
                    points * program.body.size());
                continue;
            }
            if (is_vxm_stream_nd_command(command)) {
                ++capacity.coarse_program_entries;
                checked_add(capacity.expanded_work,
                    stream_points(
                        decode_vxm_stream_nd_command(command).schedule));
                continue;
            }
            if (is_sxm_tile_program_command(command)) {
                ++capacity.coarse_program_entries;
                checked_add(capacity.expanded_work,
                    stream_points(
                        decode_sxm_tile_program_command(command).schedule));
                continue;
            }
            if (is_mem_stream_nd_command(command)
                || is_mxm_stream_nd_command(command)) {
                ++capacity.coarse_program_entries;
                checked_add(capacity.expanded_work,
                    stream_points(is_mem_stream_nd_command(command)
                            ? decode_mem_stream_nd_command(command)
                            : decode_mxm_stream_nd_command(command)));
                continue;
            }
            if (is_macro_schedule_command(command)) {
                ++capacity.macro_entries;
                const auto macro = decode_macro_schedule_command(command);
                checked_add(capacity.expanded_work,
                    static_cast<std::uint64_t>(macro.inner_count)
                        * macro.outer_count);
                continue;
            }
            if (is_icu_control_raw_word_command(command)) {
                const auto control =
                    decode_icu_control_raw_word(command);
                switch (control.opcode) {
                case IcuControlOpcode::Nop:
                    ++capacity.nop_entries;
                    break;
                case IcuControlOpcode::Repeat:
                    ++capacity.repeat_entries;
                    checked_add(capacity.expanded_work, control.count);
                    break;
                case IcuControlOpcode::Repeat2D: {
                    ++capacity.repeat_2d_entries;
                    const auto repetitions =
                        static_cast<std::uint64_t>(
                            control.repeat_2d.inner_count)
                        * control.repeat_2d.outer_count;
                    if (repetitions != 0)
                        checked_add(capacity.expanded_work,
                            repetitions - 1);
                    break;
                }
                case IcuControlOpcode::Sync:
                case IcuControlOpcode::Notify:
                case IcuControlOpcode::WaitEvent:
                    break;
                }
                continue;
            }
            if (is_repeat_2d_command(command)) {
                ++capacity.repeat_2d_entries;
                const auto repeat = decode_repeat_2d_command(command);
                const auto repetitions =
                    static_cast<std::uint64_t>(repeat.inner_count)
                    * repeat.outer_count;
                // Repeat2D replays the preceding functional instruction; its
                // first issue is counted by that instruction entry.
                if (repetitions != 0)
                    checked_add(capacity.expanded_work, repetitions - 1);
                continue;
            }
            switch (isa::decode_icu_command_opcode(command.command)) {
            case isa::IcuCommandOpcode::Instruction:
                ++capacity.instruction_entries;
                checked_add(capacity.expanded_work, 1);
                break;
            case isa::IcuCommandOpcode::Nop:
                ++capacity.nop_entries;
                break;
            case isa::IcuCommandOpcode::Repeat: {
                ++capacity.repeat_entries;
                const auto repeat = isa::decode_icu_repeat(command.command);
                checked_add(capacity.expanded_work, repeat.count);
                break;
            }
            default:
                break;
            }
        }

        const auto encodedWork = capacity.instruction_entries
            + capacity.repeat_entries + capacity.repeat_2d_entries
            + capacity.macro_entries
            + capacity.coarse_program_entries;
        report.used_slots += capacity.used_slots;
        report.encoded_work_entries += encodedWork;
        checked_add(report.expanded_work, capacity.expanded_work);
        checked_add(report.used_bits, capacity.used_bits());
        checked_add(report.active_queue_capacity_bits,
            static_cast<std::uint64_t>(capacity.depth)
                * capacity.slot_bits);
        if (capacity.overflow()) ++report.overflow_queues;
        report.queues.push_back(capacity);
    }
    return report;
}

PhysicalImemReport analyze_physical_imem(const BinaryProgram& program)
{
    const auto abstract = analyze_cmodel_abstract_imem(program);
    PhysicalImemReport report;
    report.queues.reserve(program.queues.size());
    for (std::size_t i = 0; i < program.queues.size(); ++i) {
        const auto& queue = program.queues[i];
        PhysicalImemQueue physical;
        static_cast<CmodelAbstractImemQueue&>(physical) = abstract.queues[i];
        physical.peak_macro_contexts = peak_macro_contexts(queue);
        physical.macro_context_capacity = macro_context_capacity_for(
            queue.kind, program.hardware);
        physical.macro_context_bits = macro_context_bits_for(
            queue.kind, program.hardware);
        physical.peak_fu_3d_contexts = peak_fu_3d_contexts(queue);
        physical.fu_3d_context_capacity =
            fu_3d_context_capacity_for(queue.kind);
        physical.fu_3d_context_bits = fu_3d_context_bits_for(queue.kind);
        if (physical.slot_bits == 0)
            throw std::invalid_argument(
                "target i-MEM slot width must be non-zero");
        for (const auto& command : queue.commands) {
            const auto streamNd = is_mem_stream_nd_command(command)
                || is_mxm_stream_nd_command(command)
                || is_vxm_stream_nd_command(command)
                || is_sxm_tile_program_command(command);
            const auto legacyFuMacro =
                is_macro_schedule_command(command)
                && (queue.kind == QueueKind::Mem
                    || queue.kind == QueueKind::MxmLoad
                    || queue.kind == QueueKind::MxmCompute
                    || queue.kind == QueueKind::MxmDequant);
            if (streamNd || legacyFuMacro) {
                if (streamNd) ++physical.stream_nd_packets;
                const auto packetWords =
                    fu_3d_packet_words(queue.kind);
                checked_add(physical.physical_bits,
                    static_cast<std::uint64_t>(packetWords)
                        * physical.slot_bits);
                physical.physical_slots += packetWords;
            } else {
                physical.physical_bits += physical.slot_bits;
                ++physical.physical_slots;
            }
        }
        report.stream_nd_packets += physical.stream_nd_packets;
        checked_add(report.used_bits, physical.physical_bits);
        report.used_slots += physical.physical_slots;
        checked_add(report.peak_macro_context_bits,
            static_cast<std::uint64_t>(physical.peak_macro_contexts)
                * physical.macro_context_bits);
        checked_add(report.provisioned_macro_context_bits,
            static_cast<std::uint64_t>(physical.macro_context_capacity)
                * physical.macro_context_bits);
        checked_add(report.peak_fu_3d_context_bits,
            static_cast<std::uint64_t>(physical.peak_fu_3d_contexts)
                * physical.fu_3d_context_bits);
        checked_add(report.provisioned_fu_3d_context_bits,
            static_cast<std::uint64_t>(physical.fu_3d_context_capacity)
                * physical.fu_3d_context_bits);
        checked_add(report.active_queue_capacity_bits,
            static_cast<std::uint64_t>(physical.depth) * physical.slot_bits);
        if (physical.macro_context_overflow())
            ++report.macro_context_overflow_queues;
        if (physical.fu_3d_context_overflow())
            ++report.fu_3d_context_overflow_queues;
        if (physical.deployment_overflow()) ++report.overflow_queues;
        report.queues.push_back(std::move(physical));
    }
    return report;
}

} // namespace ftlpu::software::runtime
