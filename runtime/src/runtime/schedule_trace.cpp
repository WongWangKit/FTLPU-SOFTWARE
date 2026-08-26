#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include "ftlpu/core/instruction_codec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {
namespace {

std::string csv_field(const std::string& value)
{
    std::string result = "\"";
    for (const char ch : value) {
        if (ch == '"') result += '"';
        result += ch;
    }
    return result + '"';
}

std::string stream_name(std::size_t packed)
{
    const auto stream = StreamId::from_packed(packed);
    return std::string(stream.direction() == StreamDirection::East ? "E" : "W")
        + std::to_string(stream.index());
}

const char* mem_opcode_name(MemOpcode opcode)
{
    switch (opcode) {
    case MemOpcode::Read: return "Read";
    case MemOpcode::Write: return "Write";
    case MemOpcode::Gather: return "Gather";
    case MemOpcode::Scatter: return "Scatter";
    }
    return "Unknown";
}

struct EventDescription {
    std::string resource;
    std::string detail;
};

EventDescription describe(const QueueProgram& queue, const QueueCommand& command,
    std::int64_t address_delta, std::size_t mxms_per_hemisphere,
    IcuInductionTarget induction_target = IcuInductionTarget::None)
{
    const auto east = queue.index < InstructionControlUnit::kMemQueuesPerHemisphere;
    switch (queue.kind) {
    case QueueKind::Mem: {
        if (induction_target != IcuInductionTarget::None
            && induction_target != IcuInductionTarget::MemAddress)
            throw std::logic_error(
                "runtime trace has a non-MEM induction target on a MEM queue");
        const std::size_t localQueue =
            queue.index % InstructionControlUnit::kMemQueuesPerHemisphere;
        const std::size_t slice = localQueue / hw::kMemBanksPerSlice;
        const std::size_t bank = localQueue % hw::kMemBanksPerSlice;
        const auto encoded = static_cast<isa::EncodedMemInstruction>(command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
        auto instruction = isa::decode_mem_instruction(encoded);
        const auto address =
            static_cast<std::int64_t>(instruction.address) + address_delta;
        if (address < 0)
            throw std::logic_error(
                "runtime trace MEM induction underflows the address");
        std::ostringstream detail;
        detail << "slice=" << slice << " bank=" << bank
               << " addr=" << address << " stream=" << stream_name(instruction.stream);
        const char* opcodeName = instruction.preserve_stream
            ? "WriteTap"
            : mem_opcode_name(instruction.opcode);
        return {std::string("MEM.") + (east ? "E." : "W.")
                + opcodeName, detail.str()};
    }
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: {
        const auto encoded =
            static_cast<isa::EncodedMxmInstruction>(command.words[0])
            | (static_cast<isa::EncodedMxmInstruction>(command.words[1])
                << 32);
        auto instruction = isa::decode_mxm_instruction(encoded);
        auto target = induction_target;
        // Repeat and Loop predate explicit induction targets. Preserve their
        // opcode-selected MXM stride semantics while honoring the typed target
        // carried by Repeat2D and macro schedule commands.
        if (target == IcuInductionTarget::None && address_delta != 0) {
            target = instruction.opcode == MxmControlOpcode::IW
                ? IcuInductionTarget::MxmWeightColumn
                : IcuInductionTarget::MxmAccumulatorAddress;
        }
        if (target == IcuInductionTarget::MxmWeightColumn) {
            if (instruction.opcode != MxmControlOpcode::IW)
                throw std::logic_error(
                    "runtime trace MXM column induction requires IW");
            const auto column =
                static_cast<std::int64_t>(instruction.weight_column)
                + address_delta;
            if (column < 0
                || column >= static_cast<std::int64_t>(hw::kMxmColumns))
                throw std::logic_error(
                    "runtime trace MXM column induction is out of range");
            instruction.weight_column = static_cast<std::size_t>(column);
        } else if (target == IcuInductionTarget::MxmAccumulatorAddress) {
            if (instruction.opcode != MxmControlOpcode::Compute
                && instruction.opcode != MxmControlOpcode::AccumulatorRead)
                throw std::logic_error(
                    "runtime trace MXM accumulator induction requires compute or accumulator-read");
            const auto address =
                static_cast<std::int64_t>(instruction.accumulator_address)
                + address_delta;
            if (address < 0)
                throw std::logic_error(
                    "runtime trace MXM accumulator induction underflows the address");
            instruction.accumulator_address =
                static_cast<std::size_t>(address);
        } else if (target != IcuInductionTarget::None) {
            throw std::logic_error(
                "runtime trace has a non-MXM induction target on an MXM queue");
        }
        const auto per_hemisphere = mxms_per_hemisphere;
        const auto side = queue.index < per_hemisphere ? "E" : "W";
        std::ostringstream detail;
        if (instruction.opcode == MxmControlOpcode::IW) {
            detail << "IW buffer=" << instruction.weight_buffer
                   << " column=" << instruction.weight_column;
        } else if (instruction.opcode == MxmControlOpcode::Compute) {
            detail << "Compute buffer=" << instruction.weight_buffer
                   << " act=" << stream_name(instruction.activation_stream_base)
                   << " out=" << stream_name(instruction.stream_base)
                   << " acc=" << instruction.accumulator_address
                   << " stride=" << instruction.accumulator_row_stride
                   << " format="
                   << mxm_data_format_name(instruction.data_format)
                   << (instruction.accumulator_destination
                               == MxmAccumulatorDestination::Stream
                           ? " dst=stream"
                           : " dst=sram");
            if (instruction.accumulator_destination
                == MxmAccumulatorDestination::Stream)
                detail << " acc_output="
                       << (instruction.accumulator_output_format
                                   == MxmAccumulatorOutputFormat::BFloat16
                               ? "bf16"
                               : "fp32");
        } else {
            detail << "AccumulatorRead acc=" << instruction.accumulator_address
                   << " out=" << stream_name(instruction.stream_base)
                   << (instruction.accumulator_clear ? " clear" : " keep");
        }
        return {std::string("MXM.") + side + std::to_string(queue.index % per_hemisphere)
                + (queue.kind == QueueKind::MxmLoad ? ".Load" : ".Compute"), detail.str()};
    }
    case QueueKind::MxmDequant: {
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct an MXM dequant instruction");
        const auto instruction =
            isa::decode_mxm_dequant_instruction(
                static_cast<isa::EncodedMxmDequantInstruction>(
                    command.words[0]));
        const auto perHemisphere = mxms_per_hemisphere;
        const auto side =
            queue.index < perHemisphere ? "E" : "W";
        std::ostringstream detail;
        detail << "scale=" << instruction.scale();
        return {std::string("MXM.") + side
                + std::to_string(queue.index % perHemisphere)
                + ".Dequant",
            detail.str()};
    }
    case QueueKind::Vxm: {
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct a VXM instruction");
        const auto decoded = isa::decode_vxm_instruction(queue.index,
            isa::EncodedVxmInstruction {
                static_cast<std::uint64_t>(command.words[0])
                    | (static_cast<std::uint64_t>(command.words[1]) << 32),
                command.words[2]});
        const auto& instruction = decoded.instruction;
        std::ostringstream detail;
        detail << VxmLane::operation_name(instruction.operation)
               << " depth=" << static_cast<std::size_t>(decoded.chain_depth)
               << " repeat=" << instruction.repeat_count;
        if (instruction.output_stream)
            detail << " -> S" << *instruction.output_stream;
        return {"VXM.C" + std::to_string(queue.index), detail.str()};
    }
    case QueueKind::SxmTranspose:
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct an SXM transpose instruction");
        return {std::string("SXM.") + (queue.index == 0 ? "E.Transpose" : "W.Transpose"),
            "transpose"};
    case QueueKind::SxmPermute:
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct an SXM permute instruction");
        return {std::string("SXM.") + (queue.index == 0 ? "E.Permute" : "W.Permute"),
            "permute"};
    }
    throw std::logic_error("unknown ICU queue kind in schedule trace");
}

void write_event(std::ostream& output, std::int64_t start, std::int64_t end,
    const EventDescription& event, std::size_t count = 1, std::size_t interval = 1,
    std::int64_t stride = 0)
{
    auto detail = event.detail;
    if (count > 1) {
        detail += " count=" + std::to_string(count)
            + " interval=" + std::to_string(interval)
            + " stride=" + std::to_string(stride);
    }
    output << start << ',' << end << ',' << csv_field(event.resource) << ','
           << csv_field(detail) << '\n';
}

const BinaryBinding& find_paged_weight(
    const BinaryProgram& program, std::uint32_t index)
{
    const auto found = std::ranges::find_if(program.bindings,
        [index](const BinaryBinding& binding) {
            return binding.access == BindingAccess::Input
                && binding.index == index && binding.paged_weight;
        });
    if (found == program.bindings.end())
        throw std::logic_error(
            "weight-page trace references a missing paged binding");
    return *found;
}

void write_weight_prefetches(
    std::ostream& output, const BinaryProgram& program)
{
    if (program.weight_page_uses.empty()) return;
    const std::uint64_t lanes =
        program.hardware.c2c_streams_per_direction;
    const std::uint64_t bytesPerLane =
        program.hardware.c2c_bytes_per_stream_per_cycle;
    if (lanes == 0 || bytesPerLane == 0)
        throw std::logic_error(
            "paged weight trace requires non-zero C2C bandwidth");

    const std::uint64_t bandwidth = lanes * bytesPerLane;
    for (const auto& prefetch : plan_weight_prefetches(program)) {
        std::vector<std::string> bindings;
        for (const std::size_t useIndex : prefetch.use_indices) {
            const auto& use = program.weight_page_uses[useIndex];
            const auto& binding =
                find_paged_weight(program, use.binding_index);
            bindings.push_back(binding.name.empty()
                ? std::to_string(binding.index) : binding.name);
        }
        for (std::size_t side = 0; side < prefetch.bytes.size(); ++side) {
            if (prefetch.bytes[side] == 0) continue;
            std::ostringstream detail;
            detail << "page=" << prefetch.page_index
                   << " bank=" << prefetch.bank << " bindings=";
            for (std::size_t index = 0; index < bindings.size(); ++index) {
                if (index != 0) detail << '+';
                detail << bindings[index];
            }
            detail << " bytes=" << prefetch.bytes[side]
                   << " lanes=" << lanes
                   << " bandwidth=" << bandwidth
                   << "B/cycle deadline=" << prefetch.ready_cycle
                   << " scheduled=true";
            write_event(output,
                static_cast<std::int64_t>(prefetch.start_cycle),
                static_cast<std::int64_t>(prefetch.transfer_end_cycle),
                {std::string("C2C.") + (side == 0 ? "E" : "W")
                        + ".Prefetch",
                    detail.str()});
        }
    }
}

} // namespace

void write_schedule_trace_csv(const BinaryProgram& program, const std::filesystem::path& path)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open runtime schedule trace: " + path.string());
    output << "start,end,resource,detail\n";
    write_weight_prefetches(output, program);

    for (const auto& queue : program.queues) {
        std::size_t cursor = 0;
        std::size_t previous_cycle = 0;
        const QueueCommand* previous = nullptr;
        std::deque<std::pair<const QueueCommand*, std::size_t>> history;
        for (const auto& command : queue.commands) {
            if (is_macro_schedule_command(command)) {
                const auto macro = decode_macro_schedule_command(command);
                for (std::size_t outer = 0;
                     outer < macro.outer_count; ++outer) {
                    for (std::size_t inner = 0;
                         inner < macro.inner_count; ++inner) {
                        const auto cycle = macro.start_cycle
                            + outer * macro.outer_interval
                            + inner * macro.inner_interval;
                        const auto delta =
                            static_cast<std::int64_t>(outer)
                                * macro.outer_stride
                            + static_cast<std::int64_t>(inner)
                                * macro.inner_stride;
                        write_event(output, cycle, cycle + 1,
                            describe(queue, command, delta,
                                program.hardware.mxms_per_hemisphere,
                                macro.induction_target));
                    }
                }
                cursor = std::max(cursor, macro.start_cycle
                    + (macro.outer_count - 1) * macro.outer_interval
                    + (macro.inner_count - 1) * macro.inner_interval + 1);
                continue;
            }
            if (is_repeat_2d_command(command)) {
                if (!previous)
                    throw std::logic_error(
                        "runtime trace found Repeat2D without instruction");
                const auto repeat = decode_repeat_2d_command(command);
                for (std::size_t outer = 0;
                     outer < repeat.outer_count; ++outer) {
                    for (std::size_t inner = 0;
                         inner < repeat.inner_count; ++inner) {
                        if (outer == 0 && inner == 0) continue;
                        const auto cycle = previous_cycle
                            + outer * repeat.outer_interval
                            + inner * repeat.inner_interval;
                        const auto delta =
                            static_cast<std::int64_t>(outer)
                                * repeat.outer_stride
                            + static_cast<std::int64_t>(inner)
                                * repeat.inner_stride;
                        write_event(output, cycle, cycle + 1,
                            describe(queue, *previous, delta,
                                program.hardware.mxms_per_hemisphere,
                                repeat.induction_target));
                    }
                }
                cursor = previous_cycle
                    + (repeat.outer_count - 1) * repeat.outer_interval
                    + (repeat.inner_count - 1) * repeat.inner_interval + 1;
                continue;
            }
            const auto opcode = isa::decode_icu_command_opcode(command.command);
            if (opcode == isa::IcuCommandOpcode::Nop) {
                cursor += isa::decode_icu_nop_cycles(command.command);
                continue;
            }
            if (opcode == isa::IcuCommandOpcode::Repeat) {
                if (!previous) throw std::logic_error("runtime trace found Repeat without instruction");
                const auto repeat = isa::decode_icu_repeat(command.command);
                if (repeat.count != 0) {
                    const auto first = previous_cycle + repeat.interval;
                    const auto last = previous_cycle + repeat.count * repeat.interval;
                    if (repeat.interval == 1) {
                        write_event(output, first, last + 1,
                            describe(queue, *previous, repeat.address_stride,
                                program.hardware.mxms_per_hemisphere), repeat.count,
                            repeat.interval, repeat.address_stride);
                    } else {
                        for (std::size_t index = 1; index <= repeat.count; ++index) {
                            write_event(output, previous_cycle + index * repeat.interval,
                                previous_cycle + index * repeat.interval + 1,
                                describe(queue, *previous,
                                    static_cast<std::int64_t>(index)
                                        * repeat.address_stride,
                                    program.hardware.mxms_per_hemisphere));
                        }
                    }
                    cursor = last + 1;
                }
                continue;
            }
            if (opcode == isa::IcuCommandOpcode::Loop) {
                const auto loop = isa::decode_icu_loop(command.command);
                if (loop.window_size > history.size())
                    throw std::logic_error(
                        "runtime trace found Loop with an invalid window");
                const auto first = history.end() - loop.window_size;
                for (std::size_t round = 1; round <= loop.count; ++round) {
                    std::size_t offset = 0;
                    for (auto item = first; item != history.end();
                         ++item, ++offset) {
                        const auto cycle = cursor
                            + (round - 1) * loop.interval + offset;
                        write_event(output, cycle, cycle + 1,
                            describe(queue, *item->first,
                                static_cast<std::int64_t>(round)
                                    * loop.address_stride,
                                program.hardware.mxms_per_hemisphere));
                    }
                }
                cursor += (loop.count - 1) * loop.interval
                    + loop.window_size;
                continue;
            }
            if (opcode != isa::IcuCommandOpcode::Instruction)
                throw std::logic_error("runtime trace found unsupported ICU command");
            const auto event = describe(
                queue, command, 0, program.hardware.mxms_per_hemisphere);
            write_event(output, cursor, cursor + 1, event);
            previous = &command;
            previous_cycle = cursor;
            history.emplace_back(&command, cursor);
            if (history.size() > 63) history.pop_front();
            ++cursor;
        }
    }
}

} // namespace ftlpu::software::runtime
