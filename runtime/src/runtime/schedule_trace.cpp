#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/core/instruction_codec.hpp"

#include <fstream>
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
    case MemOpcode::ReadWrite: return "ReadWrite";
    case MemOpcode::Gather: return "Gather";
    case MemOpcode::Scatter: return "Scatter";
    }
    return "Unknown";
}

const char* vxm_opcode_name(VxmAluOpcode opcode)
{
    switch (opcode) {
    case VxmAluOpcode::Pass: return "pass";
    case VxmAluOpcode::Add: return "add";
    case VxmAluOpcode::Subtract: return "subtract";
    case VxmAluOpcode::Multiply: return "multiply";
    case VxmAluOpcode::Divide: return "divide";
    case VxmAluOpcode::Negate: return "negate";
    case VxmAluOpcode::Abs: return "abs";
    case VxmAluOpcode::Min: return "min";
    case VxmAluOpcode::Max: return "max";
    case VxmAluOpcode::Clamp: return "clamp";
    case VxmAluOpcode::Square: return "square";
    case VxmAluOpcode::Sqrt: return "sqrt";
    case VxmAluOpcode::Exp: return "exp";
    case VxmAluOpcode::Log: return "log";
    case VxmAluOpcode::Relu: return "relu";
    case VxmAluOpcode::Cast: return "cast";
    }
    return "unknown";
}

struct EventDescription {
    std::string resource;
    std::string detail;
};

EventDescription describe(const QueueProgram& queue, const QueueCommand& command,
    std::int64_t address_delta)
{
    const auto east = queue.index < InstructionControlUnit::kMemQueuesPerHemisphere;
    switch (queue.kind) {
    case QueueKind::Mem: {
        const auto encoded = static_cast<isa::EncodedMemInstruction>(command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
        auto instruction = isa::decode_mem_instruction(encoded);
        const auto address = static_cast<std::int64_t>(instruction.address) + address_delta;
        std::ostringstream detail;
        detail << "slice=" << queue.index % InstructionControlUnit::kMemQueuesPerHemisphere
               << " addr=" << address << " stream=" << stream_name(instruction.stream);
        return {std::string("MEM.") + (east ? "E." : "W.")
                + mem_opcode_name(instruction.opcode), detail.str()};
    }
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: {
        const auto encoded =
            static_cast<isa::EncodedMxmInstruction>(command.words[0])
            | (static_cast<isa::EncodedMxmInstruction>(command.words[1])
                << 32);
        const auto instruction = isa::decode_mxm_instruction(encoded);
        const auto per_hemisphere = InstructionControlUnit::kMxmQueues / 2;
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
                   << (instruction.accumulator_destination
                               == MxmAccumulatorDestination::Stream
                           ? " dst=stream"
                           : " dst=sram");
        } else {
            detail << "AccumulatorRead acc=" << instruction.accumulator_address
                   << " out=" << stream_name(instruction.stream_base)
                   << (instruction.accumulator_clear ? " clear" : " keep");
        }
        return {std::string("MXM.") + side + std::to_string(queue.index % per_hemisphere)
                + (queue.kind == QueueKind::MxmLoad ? ".Load" : ".Compute"), detail.str()};
    }
    case QueueKind::Vxm: {
        const auto instruction = isa::decode_vxm_instruction(
            isa::EncodedVxmInstruction {command.words});
        std::ostringstream detail;
        if (instruction.opcode == VxmAluOpcode::Pass
            && instruction.output_stream
            && instruction.cast_target == VxmCastTarget::Float16) {
            detail << "FP32->FP16 output cast";
        } else {
            detail << vxm_opcode_name(instruction.opcode);
        }
        if (instruction.output_stream) detail << " -> " << stream_name(*instruction.output_stream);
        return {"VXM.ALU" + std::to_string(queue.index), detail.str()};
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
        for (const auto& command : queue.commands) {
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
                            describe(queue, *previous, repeat.address_stride), repeat.count,
                            repeat.interval, repeat.address_stride);
                    } else {
                        for (std::size_t index = 1; index <= repeat.count; ++index) {
                            write_event(output, previous_cycle + index * repeat.interval,
                                previous_cycle + index * repeat.interval + 1,
                                describe(queue, *previous,
                                    static_cast<std::int64_t>(index) * repeat.address_stride));
                        }
                    }
                    cursor = last + 1;
                }
                continue;
            }
            if (opcode != isa::IcuCommandOpcode::Instruction)
                throw std::logic_error("runtime trace found unsupported ICU command");
            const auto event = describe(queue, command, 0);
            write_event(output, cursor, cursor + 1, event);
            previous = &command;
            previous_cycle = cursor;
            ++cursor;
        }
    }
}

} // namespace ftlpu::software::runtime
