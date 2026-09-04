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
    }
    throw std::invalid_argument("unknown ICU queue kind");
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

std::size_t peak_macro_contexts(const QueueProgram& queue)
{
    std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
    intervals.reserve(queue.commands.size());
    for (const auto& command : queue.commands) {
        if (is_mem_slice_program_command(command)) {
            const auto program = decode_mem_slice_program_command(command);
            std::uint64_t end = program.schedule.start_cycle;
            for (std::size_t dimension = 0;
                 dimension < program.schedule.rank; ++dimension)
                end += static_cast<std::uint64_t>(
                           program.schedule.counts[dimension] - 1)
                    * program.schedule.cycle_strides[dimension];
            std::uint64_t start =
                std::numeric_limits<std::uint64_t>::max();
            std::uint64_t final = 0;
            for (const auto& body : program.body) {
                start = std::min(start,
                    static_cast<std::uint64_t>(
                        program.schedule.start_cycle)
                        + body.cycle_offset);
                final = std::max(final, end + body.cycle_offset);
            }
            intervals.push_back({start, final});
            continue;
        }
        if (!is_macro_schedule_command(command)) continue;
        const auto schedule = decode_macro_schedule_command(command);
        const auto end = static_cast<std::uint64_t>(schedule.start_cycle)
            + static_cast<std::uint64_t>(schedule.inner_count - 1)
                * schedule.inner_interval
            + static_cast<std::uint64_t>(schedule.outer_count - 1)
                * schedule.outer_interval;
        intervals.push_back({schedule.start_cycle, end});
    }
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

bool is_all_macro_mem(const QueueProgram& queue)
{
    return queue.kind == QueueKind::Mem && !queue.commands.empty()
        && std::all_of(queue.commands.begin(), queue.commands.end(),
            [](const QueueCommand& command) {
                return is_macro_schedule_command(command)
                    && command.instruction_kind == InstructionKind::Mem;
            });
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

        for (const auto& command : queue.commands) {
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
            + capacity.loop_entries + capacity.macro_entries
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
        if (is_all_macro_mem(queue)) {
            if (program.hardware.icu_macro_encoding_version
                != kMemMacroBitstreamVersion)
                throw std::invalid_argument(
                    "target does not support MEM Macro bitstream v1");
            const auto image = encode_mem_macro_bitstream(queue);
            physical.mem_delta_rle = true;
            physical.macro_codec = image.stats;
            physical.physical_bits = image.stats.physical_bits();
            ++report.mem_delta_rle_queues;
        } else {
            physical.physical_bits = static_cast<std::uint64_t>(
                physical.used_slots) * physical.slot_bits;
        }
        physical.physical_slots = physical.slot_bits == 0 ? 0
            : static_cast<std::size_t>((physical.physical_bits
                  + physical.slot_bits - 1) / physical.slot_bits);
        checked_add(report.used_bits, physical.physical_bits);
        report.used_slots += physical.physical_slots;
        checked_add(report.peak_macro_context_bits,
            static_cast<std::uint64_t>(physical.peak_macro_contexts)
                * physical.macro_context_bits);
        checked_add(report.provisioned_macro_context_bits,
            static_cast<std::uint64_t>(physical.macro_context_capacity)
                * physical.macro_context_bits);
        checked_add(report.active_queue_capacity_bits,
            static_cast<std::uint64_t>(physical.depth) * physical.slot_bits);
        if (physical.macro_context_overflow())
            ++report.macro_context_overflow_queues;
        if (physical.deployment_overflow()) ++report.overflow_queues;
        report.queues.push_back(std::move(physical));
    }
    return report;
}

} // namespace ftlpu::software::runtime
