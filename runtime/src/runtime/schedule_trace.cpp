#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/core/instruction_codec.hpp"

#include <fstream>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>

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
    std::int64_t address_delta, std::size_t mxms_per_hemisphere)
{
    const auto east = queue.index < InstructionControlUnit::kMemQueuesPerHemisphere;
    switch (queue.kind) {
    case QueueKind::Mem: {
        const std::size_t localQueue =
            queue.index % InstructionControlUnit::kMemQueuesPerHemisphere;
        const std::size_t slice = localQueue / hw::kMemBanksPerSlice;
        const std::size_t bank = localQueue % hw::kMemBanksPerSlice;
        const auto encoded = static_cast<isa::EncodedMemInstruction>(command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
        auto instruction = isa::decode_mem_instruction(encoded);
        const auto address = static_cast<std::int64_t>(instruction.address) + address_delta;
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
        const auto instruction = isa::decode_mxm_instruction(encoded);
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
        return {std::string("SXM.") + (queue.index == 0 ? "E.Transpose" : "W.Transpose"),
            "transpose"};
    case QueueKind::SxmPermute:
        return {std::string("SXM.") + (queue.index == 0 ? "E.Permute" : "W.Permute"),
            "permute"};
    }
    throw std::logic_error("unknown ICU queue kind in schedule trace");
}

void write_event(std::ostream& output, std::size_t start, std::size_t end,
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

} // namespace

void write_schedule_trace_csv(const BinaryProgram& program, const std::filesystem::path& path)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open runtime schedule trace: " + path.string());
    output << "start,end,resource,detail\n";

    for (const auto& queue : program.queues) {
        std::size_t cursor = 0;
        std::size_t previous_cycle = 0;
        const QueueCommand* previous = nullptr;
        std::deque<std::pair<const QueueCommand*, std::size_t>> history;
        for (const auto& command : queue.commands) {
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
                                program.hardware.mxms_per_hemisphere));
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
