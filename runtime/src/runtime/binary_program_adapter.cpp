#include "ftlpu/software/runtime/binary_program_adapter.hpp"

#include "ftlpu/core/instruction_packet.hpp"
#include "ftlpu/program/program_image.hpp"
#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ftlpu::software::runtime {
namespace {

void require_binary_target(
    const BinaryProgram& binary,
    const TargetDescription& target)
{
    if (binary.target_name.empty() || binary.target_abi == 0) {
        throw std::invalid_argument(
            "BinaryProgram has no executable target identity");
    }
    if (target.executable_target.name.empty()
        || target.executable_target.abi == 0) {
        throw std::invalid_argument(
            "DeviceBackend does not advertise an executable target identity");
    }
    if (binary.target_name != target.executable_target.name
        || binary.target_abi != target.executable_target.abi) {
        std::ostringstream message;
        message << "BinaryProgram target mismatch: binary '"
                << binary.target_name << "' ABI 0x" << std::hex
                << binary.target_abi << ", backend requires '"
                << target.executable_target.name << "' ABI 0x"
                << target.executable_target.abi;
        throw std::invalid_argument(message.str());
    }
}

IcuLocation queue_location(
    const QueueProgram& queue,
    const TargetDescription& target)
{
    switch (queue.kind) {
    case QueueKind::Mem: {
        const auto queue_count =
            target.hemisphere_count * target.mem_slices_per_hemisphere;
        if (queue.index >= queue_count) {
            throw std::out_of_range(
                "BinaryProgram MEM queue index is outside the target");
        }
        const auto hemisphere =
            queue.index / target.mem_slices_per_hemisphere;
        const auto slice =
            queue.index % target.mem_slices_per_hemisphere;
        return IcuLocation::Mem(
            static_cast<Hemisphere>(hemisphere), slice);
    }
    case QueueKind::MxmLoad:
        if (queue.index >= target.mxm_count) {
            throw std::out_of_range(
                "BinaryProgram MXM load queue index is outside the target");
        }
        return IcuLocation::MxmLoad(queue.index);
    case QueueKind::MxmCompute:
        if (queue.index >= target.mxm_count) {
            throw std::out_of_range(
                "BinaryProgram MXM compute queue index is outside the target");
        }
        return IcuLocation::MxmCompute(queue.index);
    case QueueKind::Vxm:
        if (queue.index >= target.vxm_alu_count) {
            throw std::out_of_range(
                "BinaryProgram VXM queue index is outside the target");
        }
        return IcuLocation::Vxm(queue.index);
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        if (queue.index >= target.hemisphere_count
            || queue.index >= target.sxm_count) {
            throw std::out_of_range(
                "BinaryProgram SXM queue index is outside the target");
        }
        return IcuLocation::Sxm(
            static_cast<Hemisphere>(queue.index));
    }
    throw std::invalid_argument("BinaryProgram has an unknown queue kind");
}

SxmInstruction decode_sxm_command(const QueueCommand& command)
{
    if (command.instruction_kind != InstructionKind::Sxm
        || command.word_count != 4
        || command.words[3] != 0
        || command.extension_words.size()
            < 2 + SxmInstruction::kTotalLanes) {
        throw std::invalid_argument(
            "SXM queue command has an invalid serialized payload");
    }
    SxmInstruction instruction{};
    instruction.opcode = static_cast<SxmOpcode>(command.words[0]);
    instruction.shift_source =
        static_cast<SxmShiftSource>(command.words[1]);
    instruction.shift_distance = command.words[2];
    const auto source_count =
        static_cast<std::size_t>(command.extension_words[0]);
    const auto destination_count =
        static_cast<std::size_t>(command.extension_words[1]);
    const auto map_begin = 2 + source_count + destination_count;
    if (command.extension_words.size()
        != map_begin + SxmInstruction::kTotalLanes) {
        throw std::invalid_argument(
            "SXM queue command has malformed stream lists");
    }
    for (std::size_t index = 0; index < source_count; ++index) {
        instruction.src_streams.push_back(
            SxmStreamId {command.extension_words[2 + index]});
    }
    for (std::size_t index = 0; index < destination_count; ++index) {
        instruction.dst_streams.push_back(SxmStreamId {
            command.extension_words[2 + source_count + index]});
    }
    for (std::size_t lane = 0;
         lane < SxmInstruction::kTotalLanes;
         ++lane) {
        const auto value = command.extension_words[map_begin + lane];
        instruction.permute_map[lane] =
            value == UINT32_MAX ? SxmInstruction::kZeroFill : value;
    }
    return instruction;
}

isa::EncodedInstructionPacket lower_command(
    const QueueProgram& queue,
    const QueueCommand& command)
{
    const auto opcode =
        isa::decode_icu_command_opcode(command.command);
    if (opcode == isa::IcuCommandOpcode::Nop
        || opcode == isa::IcuCommandOpcode::Repeat) {
        if (command.instruction_kind != InstructionKind::None
            || command.word_count != 0
            || !command.extension_words.empty()) {
            throw std::invalid_argument(
                "ICU control command unexpectedly carries instruction words");
        }
        if ((queue.kind == QueueKind::SxmTranspose
                || queue.kind == QueueKind::SxmPermute)
            && opcode == isa::IcuCommandOpcode::Repeat) {
            throw std::invalid_argument(
                "SXM queues do not support Repeat commands");
        }
        return isa::encode_packet(
            isa::decode_icu_control_instruction(command.command));
    }
    if (opcode != isa::IcuCommandOpcode::Instruction) {
        throw std::invalid_argument(
            "BinaryProgram queue contains an unsupported ICU control opcode");
    }

    switch (queue.kind) {
    case QueueKind::Mem: {
        if (command.instruction_kind != InstructionKind::Mem) {
            throw std::invalid_argument(
                "MEM queue command has the wrong instruction kind");
        }
        if (command.word_count == 1
            || (command.word_count == 2 && command.words[1] == 0)) {
            return isa::encode_packet(
                isa::decode_mem_instruction(command.words[0]));
        }
        if (command.word_count == 3) {
            return isa::encode_packet(
                isa::decode_extended_mem_instruction(
                    isa::EncodedExtendedMemInstruction {{
                        command.words[0],
                        command.words[1],
                        command.words[2],
                    }}));
        }
        throw std::invalid_argument(
            "MEM queue command has an invalid word count");
    }
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: {
        if (command.instruction_kind != InstructionKind::Mxm
            || (command.word_count != 1
                && !(command.word_count == 2
                    && command.words[1] == 0))) {
            throw std::invalid_argument(
                "MXM queue command has an invalid serialized payload");
        }
        const auto instruction =
            isa::decode_mxm_instruction(command.words[0]);
        const auto load =
            instruction.opcode == MxmControlOpcode::IW
            || instruction.opcode == MxmControlOpcode::LoadScales;
        if ((queue.kind == QueueKind::MxmLoad) != load) {
            throw std::invalid_argument(
                "MXM instruction opcode does not match its queue");
        }
        return isa::encode_packet(instruction);
    }
    case QueueKind::Vxm: {
        if (command.instruction_kind != InstructionKind::Vxm
            || (command.word_count != 3
                && !(command.word_count == 4
                    && command.words[3] == 0))) {
            throw std::invalid_argument(
                "VXM queue command has an invalid serialized payload");
        }
        return isa::encode_packet(
            isa::decode_vxm_instruction(isa::EncodedVxmInstruction {{
                command.words[0],
                command.words[1],
                command.words[2],
            }}));
    }
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: {
        const auto instruction = decode_sxm_command(command);
        const auto expected = queue.kind == QueueKind::SxmTranspose
            ? SxmOpcode::Transpose : SxmOpcode::Permute;
        if (instruction.opcode != expected) {
            throw std::invalid_argument(
                "SXM instruction opcode does not match its queue");
        }
        return isa::encode_packet(instruction);
    }
    }
    throw std::invalid_argument("BinaryProgram has an unknown queue kind");
}

} // namespace

AdaptedExecutable adapt_binary_program(
    const BinaryProgram& binary,
    const TargetDescription& target,
    std::size_t drain_cycles)
{
    require_binary_target(binary, target);
    if (target.instruction_packet_bytes
            != sizeof(isa::EncodedInstructionPacket)
        || target.ifetch_packets == 0) {
        throw std::invalid_argument(
            "DeviceBackend target has incompatible instruction geometry");
    }
    if (binary.max_cycle
        > std::numeric_limits<std::size_t>::max() - 1 - drain_cycles) {
        throw std::overflow_error(
            "BinaryProgram execution cycle count overflows size_t");
    }

    ProgramImage image(ProgramImageHeader {
        ProgramImageHeader::kMagic,
        1,
        binary.target_name,
        "BinaryProgram lowered through DeviceBackend",
    });
    std::vector<std::array<bool, 2>> sxm_kinds(
        target.hemisphere_count);
    for (const auto& queue : binary.queues) {
        const auto location = queue_location(queue, target);
        if (queue.commands.empty()) {
            continue;
        }
        if (queue.kind == QueueKind::SxmTranspose
            || queue.kind == QueueKind::SxmPermute) {
            const auto kind =
                queue.kind == QueueKind::SxmTranspose ? 0u : 1u;
            sxm_kinds[queue.index][kind] = true;
            if (sxm_kinds[queue.index][0]
                && sxm_kinds[queue.index][1]) {
                throw std::invalid_argument(
                    "one hemisphere contains both SXM transpose and "
                    "SXM permute queues");
            }
        }

        ProgramSection section {
            location,
            {},
            0,
            std::string(queue_kind_name(queue.kind))
                + " queue " + std::to_string(queue.index),
        };
        section.packets.reserve(queue.commands.size());
        for (const auto& command : queue.commands) {
            section.packets.push_back(lower_command(queue, command));
        }
        if (section.packets.size() > target.ifetch_packets) {
            std::ostringstream message;
            message << queue_kind_name(queue.kind) << " queue "
                    << queue.index << " contains "
                    << section.packets.size()
                    << " packets, exceeding IFetch capacity "
                    << target.ifetch_packets;
            throw std::length_error(message.str());
        }
        image.add_section(std::move(section));
    }
    if (image.sections().empty()) {
        throw std::invalid_argument(
            "BinaryProgram has no non-empty queue sections");
    }

    return AdaptedExecutable {
        DeviceProgram {
            std::move(image),
            binary.max_cycle + 1 + drain_cycles,
        },
        binary.bindings,
    };
}

} // namespace ftlpu::software::runtime
