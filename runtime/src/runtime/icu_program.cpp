#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ftlpu::software::runtime {

namespace {

void validate_fu_3d_context_capacity(
    const std::vector<QueueProgram>& queues)
{
    BinaryProgram probe;
    probe.queues = queues;
    const auto report = analyze_physical_imem(probe);
    for (const auto& queue : report.queues) {
        if (!queue.fu_3d_context_overflow()) continue;
        std::ostringstream message;
        message << "binary FU 3-D context capacity exceeded: resource="
                << queue_kind_name(queue.kind)
                << " queue=" << queue.index
                << " peak=" << queue.peak_fu_3d_contexts
                << " capacity=" << queue.fu_3d_context_capacity
                << "; lower overlapping work into one FU-specific 3-D instruction";
        throw StaticScheduleError(message.str());
    }
}

constexpr isa::EncodedIcuCommand kInstructionCommand =
    static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);

QueueCommand encode_mem_command(const MemInstruction& instruction)
{
    const auto encoded = isa::encode_mem_instruction(instruction);
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    };
}

QueueCommand encode_mxm_command(const MxmControlInstruction& instruction)
{
    const auto encoded = isa::encode_mxm_instruction(instruction);
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    };
}

QueueCommand encode_mxm_dequant_command(
    const MxmDequantInstruction& instruction)
{
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::MxmDequant,
        1,
        {
            static_cast<std::uint32_t>(
                isa::encode_mxm_dequant_instruction(instruction)),
            0,
            0,
            0,
        },
    };
}

QueueCommand encode_vxm_command(std::size_t queue,
    VxmChainDepth depth, const VxmLaneAluInstruction& instruction)
{
    const auto encoded = isa::encode_vxm_instruction(queue, depth, instruction);
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Vxm,
        3,
        {
            static_cast<std::uint32_t>(encoded.control),
            static_cast<std::uint32_t>(encoded.control >> 32),
            encoded.immediate_bits,
            0,
        },
    };
}

QueueCommand encode_sxm_command(const SxmInstruction& instruction)
{
    QueueCommand command {kInstructionCommand, InstructionKind::Sxm, 4, {}};
    command.words[0] = static_cast<std::uint32_t>(instruction.opcode);
    command.words[1] = static_cast<std::uint32_t>(instruction.shift_source);
    command.words[2] = static_cast<std::uint32_t>(instruction.shift_distance);
    command.words[3] =
        (instruction.output_row == SxmInstruction::kAllOutputRows ? 0xffu
            : static_cast<std::uint32_t>(instruction.output_row))
        | ((instruction.input_row == SxmInstruction::kAllInputRows ? 0xffu
            : static_cast<std::uint32_t>(instruction.input_row)) << 8)
        | ((instruction.output_tile == SxmInstruction::kAllOutputTiles ? 0xffu
            : static_cast<std::uint32_t>(instruction.output_tile)) << 16);
    command.extension_words.push_back(static_cast<std::uint32_t>(instruction.src_streams.size()));
    command.extension_words.push_back(static_cast<std::uint32_t>(instruction.dst_streams.size()));
    for (const auto stream : instruction.src_streams)
        command.extension_words.push_back(static_cast<std::uint32_t>(stream.stream));
    for (const auto stream : instruction.dst_streams)
        command.extension_words.push_back(static_cast<std::uint32_t>(stream.stream));
    for (const auto lane : instruction.permute_map)
        command.extension_words.push_back(lane == SxmInstruction::kZeroFill
            ? UINT32_MAX : static_cast<std::uint32_t>(lane));
    return command;
}

SxmInstruction decode_sxm_command(const QueueCommand& command)
{
    if (command.instruction_kind != InstructionKind::Sxm || command.word_count != 4
        || command.extension_words.size() < 2 + SxmInstruction::kTotalLanes)
        throw std::logic_error("SXM queue command has an invalid variable payload");
    SxmInstruction instruction {};
    instruction.opcode = static_cast<SxmOpcode>(command.words[0]);
    instruction.shift_source = static_cast<SxmShiftSource>(command.words[1]);
    instruction.shift_distance = command.words[2];
    const auto output_row = command.words[3] & 0xffu;
    const auto input_row = (command.words[3] >> 8) & 0xffu;
    const auto output_tile = (command.words[3] >> 16) & 0xffu;
    instruction.output_row = output_row == 0xffu
        ? SxmInstruction::kAllOutputRows : output_row;
    instruction.input_row = input_row == 0xffu
        ? SxmInstruction::kAllInputRows : input_row;
    instruction.output_tile = output_tile == 0xffu
        ? SxmInstruction::kAllOutputTiles : output_tile;
    const auto src_count = command.extension_words[0];
    const auto dst_count = command.extension_words[1];
    const std::size_t map_begin = 2 + src_count + dst_count;
    if (command.extension_words.size() != map_begin + SxmInstruction::kTotalLanes)
        throw std::logic_error("SXM queue command has malformed stream lists");
    for (std::size_t index = 0; index < src_count; ++index)
        instruction.src_streams.push_back(SxmStreamId {command.extension_words[2 + index]});
    for (std::size_t index = 0; index < dst_count; ++index)
        instruction.dst_streams.push_back(SxmStreamId {command.extension_words[2 + src_count + index]});
    for (std::size_t lane = 0; lane < SxmInstruction::kTotalLanes; ++lane) {
        const auto value = command.extension_words[map_begin + lane];
        instruction.permute_map[lane] = value == UINT32_MAX ? SxmInstruction::kZeroFill : value;
    }
    return instruction;
}

void validate_mxm_queue_opcode(
    QueueKind kind,
    std::size_t mxm,
    const MxmControlInstruction& instruction)
{
    if (kind == QueueKind::MxmLoad && instruction.opcode != MxmControlOpcode::IW) {
        throw std::logic_error("MXM load queue only accepts IW instructions");
    }
    if (kind == QueueKind::MxmCompute
        && instruction.opcode != MxmControlOpcode::Compute
        && instruction.opcode != MxmControlOpcode::AccumulatorRead) {
        throw std::logic_error(
            "MXM compute queue only accepts Compute or AccumulatorRead instructions");
    }
    (void)mxm;
}

std::string queue_name(QueueKind kind, std::size_t index)
{
    std::ostringstream os;
    os << queue_kind_name(kind) << index;
    return os.str();
}

void validate_queue_index(QueueKind kind, std::size_t index)
{
    if (kind == QueueKind::Mem && index >= InstructionControlUnit::kMemQueues) {
        throw std::out_of_range("binary MEM queue index is outside the CModel ICU range");
    }
    if ((kind == QueueKind::MxmLoad || kind == QueueKind::MxmCompute
            || kind == QueueKind::MxmDequant)
        && index >= InstructionControlUnit::kMxmQueues) {
        throw std::out_of_range("binary MXM queue index is outside the CModel ICU range");
    }
    if (kind == QueueKind::Vxm && index >= InstructionControlUnit::kVxmQueues) {
        throw std::out_of_range("binary VXM queue index is outside the CModel ICU range");
    }
    if ((kind == QueueKind::SxmTranspose || kind == QueueKind::SxmPermute)
        && index >= hw::kHemispheres)
        throw std::out_of_range("binary SXM hemisphere index is outside the CModel ICU range");
    if ((kind == QueueKind::C2cDma || kind == QueueKind::C2cTx
            || kind == QueueKind::C2cRx)
        && index >= hw::kHemispheres)
        throw std::out_of_range(
            "binary C2C hemisphere index is outside the CModel ICU range");
}

bool is_mxm_queue(QueueKind kind)
{
    return kind == QueueKind::MxmLoad
        || kind == QueueKind::MxmCompute
        || kind == QueueKind::MxmDequant;
}

InstructionKind instruction_kind_for_raw_queue(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return InstructionKind::Mem;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
        return InstructionKind::Mxm;
    case QueueKind::MxmDequant:
        return InstructionKind::MxmDequant;
    case QueueKind::Vxm:
        return InstructionKind::Vxm;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return InstructionKind::Sxm;
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        break;
    }
    throw std::logic_error(
        "raw FU loop words are unsupported on this ICU queue");
}

std::size_t raw_3d_packet_word_count(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return isa::EncodedMemIcu3DPacket::kWordCount;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return isa::EncodedMxmLoadIcu3DPacket::kWordCount;
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
    throw std::logic_error(
        "raw FU loop packets are unsupported on this ICU queue");
}

bool begins_raw_3d_packet(
    QueueKind kind, const QueueCommand& command)
{
    if (kind != QueueKind::Mem && !is_mxm_queue(kind)
        && kind != QueueKind::Vxm
        && kind != QueueKind::SxmTranspose
        && kind != QueueKind::SxmPermute)
        return false;
    return is_fu_3d_raw_packet_header(command);
}

IcuLocation queue_location(QueueKind kind, std::size_t queueIndex)
{
    switch (kind) {
    case QueueKind::Mem: {
        const auto hemisphere = static_cast<Hemisphere>(queueIndex
            / InstructionControlUnit::kMemQueuesPerHemisphere);
        const auto local = queueIndex
            % InstructionControlUnit::kMemQueuesPerHemisphere;
        return IcuLocation::Mem(hemisphere,
            local / hw::kMemBanksPerSlice,
            local % hw::kMemBanksPerSlice);
    }
    case QueueKind::MxmLoad:
        return IcuLocation::MxmLoad(queueIndex);
    case QueueKind::MxmCompute:
        return IcuLocation::MxmCompute(queueIndex);
    case QueueKind::MxmDequant:
        return IcuLocation::MxmDequant(queueIndex);
    case QueueKind::Vxm:
        return IcuLocation::Vxm(queueIndex);
    case QueueKind::SxmTranspose:
        return IcuLocation::Sxm(
            static_cast<Hemisphere>(queueIndex), 0);
    case QueueKind::SxmPermute:
        return IcuLocation::Sxm(
            static_cast<Hemisphere>(queueIndex), 1);
    case QueueKind::C2cDma:
        return IcuLocation::C2cDma(
            static_cast<Hemisphere>(queueIndex));
    case QueueKind::C2cTx:
        return IcuLocation::C2cTx(
            static_cast<Hemisphere>(queueIndex));
    case QueueKind::C2cRx:
        return IcuLocation::C2cRx(
            static_cast<Hemisphere>(queueIndex));
    }
    throw std::logic_error("unknown ICU queue kind");
}

template <typename Packet>
Packet read_raw_3d_packet(const QueueProgram& queue,
    std::size_t commandIndex)
{
    constexpr auto physicalWordCount = Packet::kWordCount;
    constexpr auto lanesPerWord = Packet::kLanesPerWord;
    if (commandIndex > queue.commands.size()
        || physicalWordCount > queue.commands.size() - commandIndex)
        throw std::logic_error("truncated FU 3-D raw packet in binary queue");
    const auto expectedKind = instruction_kind_for_raw_queue(queue.kind);
    Packet packet {};
    for (std::size_t wordIndex = 0;
         wordIndex < physicalWordCount; ++wordIndex) {
        const auto& command = queue.commands[commandIndex + wordIndex];
        if (command.instruction_kind != expectedKind
            || command.word_count != lanesPerWord
            || !command.extension_words.empty()
            || command.command != command.words[0]
            || isa::decode_icu_command_opcode(command.command)
                != isa::IcuCommandOpcode::Extended
            || ((command.words[0] >> 2)
                    & (expectedKind == InstructionKind::Sxm
                            ? 0x7U : 0x3U)) != wordIndex)
            throw std::logic_error(
                "malformed FU 3-D raw packet word in binary queue");
        for (std::size_t lane = 0; lane < lanesPerWord; ++lane)
            packet.words[wordIndex].lanes[lane] = command.words[lane];
    }
    return packet;
}

std::size_t physical_queue_index(QueueKind kind, std::size_t logical_index,
    std::size_t logical_mxms_per_hemisphere)
{
    if (!is_mxm_queue(kind)) return logical_index;
    if (logical_mxms_per_hemisphere == 0
        || logical_mxms_per_hemisphere > hw::kMxmsPerHemisphere)
        throw std::out_of_range(
            "binary MXM topology cannot be mapped onto the CModel");
    const std::size_t logical_mxm_count =
        hw::kHemispheres * logical_mxms_per_hemisphere;
    if (logical_index >= logical_mxm_count)
        throw std::out_of_range(
            "binary MXM queue index is outside its logical topology");
    const std::size_t hemisphere =
        logical_index / logical_mxms_per_hemisphere;
    const std::size_t local_mxm =
        logical_index % logical_mxms_per_hemisphere;
    return hemisphere * hw::kMxmsPerHemisphere + local_mxm;
}

} // namespace

QueueCommand encode_icu_control_raw_word(
    const IcuControlInstruction& instruction)
{
    const auto word =
        InstructionControlUnit::MemIcu::encode_control_raw_word(
            instruction);
    QueueCommand command;
    command.command = word.lanes[0];
    command.instruction_kind = InstructionKind::None;
    command.word_count = 3;
    std::copy(word.lanes.begin(), word.lanes.end(),
        command.words.begin());
    return command;
}

bool is_icu_control_raw_word_command(
    const QueueCommand& command) noexcept
{
    if (command.instruction_kind != InstructionKind::None
        || command.word_count != 3
        || !command.extension_words.empty()
        || command.command != command.words[0])
        return false;

    const auto opcode =
        isa::decode_icu_command_opcode(command.command);
    if (opcode == isa::IcuCommandOpcode::Nop
        || opcode == isa::IcuCommandOpcode::Repeat)
        return true;
    if (opcode != isa::IcuCommandOpcode::Extended) return false;
    const auto subtype = (command.words[2] >> 24) & 0xfU;
    return subtype == 1 || subtype == 3 || subtype == 4
        || subtype == 5;
}

IcuControlInstruction decode_icu_control_raw_word(
    const QueueCommand& command)
{
    if (!is_icu_control_raw_word_command(command))
        throw std::logic_error(
            "queue command is not a physical ICU control word");
    isa::EncodedMemIcu3DWord word {{
        command.words[0], command.words[1], command.words[2]}};
    return InstructionControlUnit::MemIcu::decode_control_raw_word(word);
}

std::array<QueueCommand,
    InstructionControlUnit::MemIcu::synchronized_packet_word_count>
encode_mem_synchronized_icu_packet(
    const InstructionControlUnit::MemIcu::EncodedSynchronizedPacket& packet)
{
    std::array<QueueCommand,
        InstructionControlUnit::MemIcu::synchronized_packet_word_count>
        commands {};
    for (std::size_t index = 0; index < commands.size(); ++index) {
        auto& command = commands[index];
        command.command = packet[index].lanes[0];
        command.instruction_kind = InstructionKind::Mem;
        command.word_count = 3;
        std::copy(packet[index].lanes.begin(), packet[index].lanes.end(),
            command.words.begin());
    }
    return commands;
}

bool is_mem_synchronized_raw_word_command(
    const QueueCommand& command) noexcept
{
    if (command.instruction_kind != InstructionKind::Mem
        || command.word_count != 3
        || !command.extension_words.empty()
        || command.command != command.words[0])
        return false;
    const auto opcode =
        isa::decode_icu_command_opcode(command.command);
    return opcode == isa::IcuCommandOpcode::Instruction
        || (opcode == isa::IcuCommandOpcode::Extended
            && ((command.words[2] >> 24) & 0xfU) == 6);
}

bool is_mem_synchronized_raw_packet_header(
    const QueueCommand& command) noexcept
{
    return is_mem_synchronized_raw_word_command(command)
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && ((command.words[2] >> 24) & 0xfU) == 6;
}

InstructionControlUnit::MemIcu::EncodedSynchronizedPacket
decode_mem_synchronized_icu_packet(
    const QueueProgram& queue, std::size_t commandIndex)
{
    constexpr auto kWordCount =
        InstructionControlUnit::MemIcu::synchronized_packet_word_count;
    if (queue.kind != QueueKind::Mem
        || commandIndex > queue.commands.size()
        || kWordCount > queue.commands.size() - commandIndex
        || !is_mem_synchronized_raw_packet_header(
            queue.commands[commandIndex]))
        throw std::logic_error(
            "truncated or malformed MEM_WRITE_SYNC raw packet");

    const auto& continuation = queue.commands[commandIndex + 1];
    if (!is_mem_synchronized_raw_word_command(continuation)
        || isa::decode_icu_command_opcode(continuation.command)
            != isa::IcuCommandOpcode::Instruction)
        throw std::logic_error(
            "MEM_WRITE_SYNC continuation is not a native MEM word");

    InstructionControlUnit::MemIcu::EncodedSynchronizedPacket packet {};
    for (std::size_t word = 0; word < kWordCount; ++word) {
        for (std::size_t lane = 0; lane < 3; ++lane)
            packet[word].lanes[lane] =
                queue.commands[commandIndex + word].words[lane];
    }

    // Reuse the CModel's hardware decoder to validate reserved fields and
    // that the native template is a MEM Write before returning the packet.
    InstructionControlUnit::MemIcu validator;
    validator.push_encoded_synchronized_packet(packet);
    return packet;
}

bool is_fu_3d_raw_word_command(const QueueCommand& command) noexcept
{
    const bool supportedPair =
        (command.instruction_kind == InstructionKind::Mem
            && command.word_count
                == isa::EncodedMemIcu3DPacket::kLanesPerWord)
        || ((command.instruction_kind == InstructionKind::Mxm
                || command.instruction_kind
                    == InstructionKind::MxmDequant)
            && command.word_count
                == isa::EncodedMxmLoadIcu3DPacket::kLanesPerWord)
        || (command.instruction_kind == InstructionKind::Vxm
            && command.word_count
                == isa::EncodedVxmIcuRun2DPacket::kLanesPerWord)
        || (command.instruction_kind == InstructionKind::Sxm
            && command.word_count
                == isa::EncodedSxmIcuRun2DPacket::kLanesPerWord);
    if (!supportedPair) return false;
    if (!command.extension_words.empty()
        || command.command != command.words[0]
        || isa::decode_icu_command_opcode(command.command)
            != isa::IcuCommandOpcode::Extended)
        return false;
    const auto wordIndex = (command.words[0] >> 2)
        & (command.instruction_kind == InstructionKind::Sxm ? 0x7U : 0x3U);
    // Extended subtype lives at physical [91:88] of packet word zero.
    const auto subtype = (command.words[2] >> 24) & 0xfU;
    return wordIndex != 0 || subtype == 2
        || (command.instruction_kind == InstructionKind::Mem
            && subtype == 7 && ((command.words[0] >> 4) & 0x3U) == 3);
}

bool is_mem_write_read_2d_raw_packet_header(
    const QueueCommand& command) noexcept
{
    return command.instruction_kind == InstructionKind::Mem
        && is_fu_3d_raw_packet_header(command)
        && ((command.words[2] >> 24) & 0xfU) == 7
        && ((command.words[0] >> 4) & 0x3U) == 3;
}

MemIcuWriteRead2DInstruction decode_mem_write_read_2d_raw_packet(
    const QueueProgram& queue, std::size_t commandIndex)
{
    if (queue.kind != QueueKind::Mem
        || commandIndex >= queue.commands.size()
        || !is_mem_write_read_2d_raw_packet_header(
            queue.commands[commandIndex]))
        throw std::logic_error(
            "expected WRITE_READ_2D raw packet header in MEM queue");
    return isa::decode_mem_icu_write_read_2d_instruction(
        read_raw_3d_packet<isa::EncodedMemIcuWriteRead2DPacket>(
            queue, commandIndex));
}

bool is_fu_3d_raw_packet_header(
    const QueueCommand& command) noexcept
{
    return is_fu_3d_raw_word_command(command)
        && ((command.words[0] >> 2)
            & (command.instruction_kind == InstructionKind::Sxm
                    ? 0x7U : 0x3U)) == 0;
}

std::size_t fu_3d_raw_packet_word_count(QueueKind kind)
{
    return raw_3d_packet_word_count(kind);
}

IcuLoop3D decode_fu_3d_raw_packet_loop(
    const QueueProgram& queue, std::size_t commandIndex)
{
    switch (queue.kind) {
    case QueueKind::Mem:
        return isa::decode_mem_icu_3d_instruction(
            read_raw_3d_packet<isa::EncodedMemIcu3DPacket>(
                queue, commandIndex)).loop;
    case QueueKind::MxmLoad:
        return isa::decode_mxm_load_icu_3d_instruction(
            read_raw_3d_packet<isa::EncodedMxmLoadIcu3DPacket>(
                queue, commandIndex)).loop;
    case QueueKind::MxmDequant:
        return isa::decode_mxm_dequant_icu_3d_instruction(
            read_raw_3d_packet<isa::EncodedMxmDequantIcu3DPacket>(
                queue, commandIndex)).loop;
    case QueueKind::MxmCompute:
        return isa::decode_mxm_compute_icu_3d_instruction(
            read_raw_3d_packet<isa::EncodedMxmComputeIcu3DPacket>(
                queue, commandIndex)).loop;
    case QueueKind::Vxm:
        return isa::decode_vxm_icu_run_2d_instruction(
            read_raw_3d_packet<isa::EncodedVxmIcuRun2DPacket>(
                queue, commandIndex)).loop;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return isa::decode_sxm_icu_run_2d_instruction(
            read_raw_3d_packet<isa::EncodedSxmIcuRun2DPacket>(
                queue, commandIndex)).loop;
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        break;
    }
    throw std::logic_error("unknown ICU queue kind");
}

const char* queue_kind_name(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return "mem";
    case QueueKind::MxmLoad:
        return "mxm_load";
    case QueueKind::MxmCompute:
        return "mxm_compute";
    case QueueKind::MxmDequant:
        return "mxm_dequant";
    case QueueKind::Vxm:
        return "vxm";
    case QueueKind::SxmTranspose:
        return "sxm_transpose";
    case QueueKind::SxmPermute:
        return "sxm_permute";
    case QueueKind::C2cDma:
        return "c2c_dma";
    case QueueKind::C2cTx:
        return "c2c_tx";
    case QueueKind::C2cRx:
        return "c2c_rx";
    }
    return "unknown";
}

void IcuProgram::emit_mem(std::size_t cycle, std::size_t column, MemInstruction instruction)
{
    check_mem_column(column);
    mem_[column].push_back(ScheduledInstruction<MemInstruction> {cycle, instruction});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_mxm_load(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction)
{
    check_mxm(mxm);
    validate_mxm_queue_opcode(QueueKind::MxmLoad, mxm, instruction);
    mxm_load_[mxm].push_back(ScheduledInstruction<MxmControlInstruction> {cycle, instruction});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_mxm_compute(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction)
{
    check_mxm(mxm);
    validate_mxm_queue_opcode(QueueKind::MxmCompute, mxm, instruction);
    mxm_compute_[mxm].push_back(ScheduledInstruction<MxmControlInstruction> {cycle, instruction});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_mxm_dequant(
    std::size_t cycle,
    std::size_t mxm,
    MxmDequantInstruction instruction)
{
    check_mxm(mxm);
    mxm_dequant_[mxm].push_back(
        ScheduledInstruction<MxmDequantInstruction> {cycle, instruction});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_vxm(std::size_t cycle, std::size_t alu,
    VxmChainDepth depth, VxmLaneAluInstruction instruction)
{
    check_vxm_alu(alu);
    vxm_[alu].push_back(ScheduledInstruction<VxmInstruction> {
        cycle, VxmInstruction {depth, std::move(instruction)}});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_vxm(std::size_t cycle, std::size_t alu,
    VxmLaneAluInstruction instruction)
{
    emit_vxm(cycle, alu, VxmChainDepth::Eight, std::move(instruction));
}

void IcuProgram::emit_sxm_transpose(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction)
{
    if (instruction.opcode != SxmOpcode::Transpose)
        throw std::invalid_argument("SXM transpose queue only accepts Transpose instructions");
    sxm_transpose_[hemisphere_index(hemisphere)].push_back({cycle, std::move(instruction)});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_sxm_permute(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction)
{
    if (instruction.opcode != SxmOpcode::Permute)
        throw std::invalid_argument("SXM permute queue only accepts Permute instructions");
    sxm_permute_[hemisphere_index(hemisphere)].push_back({cycle, std::move(instruction)});
    last_cycle_ = std::max(last_cycle_, cycle);
}

std::vector<QueueProgram> IcuProgram::encode_queues() const
{
    auto queues = std::vector<QueueProgram> {};

    for (std::size_t column = 0; column < mem_.size(); ++column) {
        queues.push_back(QueueProgram {
            QueueKind::Mem,
            column,
            encode_scheduled_queue(mem_[column], queue_name(QueueKind::Mem, column), encode_mem_command),
        });
    }

    for (std::size_t mxm = 0; mxm < mxm_load_.size(); ++mxm) {
        queues.push_back(QueueProgram {
            QueueKind::MxmLoad,
            mxm,
            encode_scheduled_queue(mxm_load_[mxm], queue_name(QueueKind::MxmLoad, mxm), encode_mxm_command),
        });
        queues.push_back(QueueProgram {
            QueueKind::MxmCompute,
            mxm,
            encode_scheduled_queue(mxm_compute_[mxm], queue_name(QueueKind::MxmCompute, mxm), encode_mxm_command),
        });
        queues.push_back(QueueProgram {
            QueueKind::MxmDequant,
            mxm,
            encode_scheduled_queue(
                mxm_dequant_[mxm],
                queue_name(QueueKind::MxmDequant, mxm),
                encode_mxm_dequant_command),
        });
    }

    for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
        queues.push_back(QueueProgram {
            QueueKind::Vxm,
            alu,
            encode_scheduled_queue(vxm_[alu], queue_name(QueueKind::Vxm, alu),
                [alu](const VxmInstruction& item) {
                    return encode_vxm_command(
                        alu, item.depth, item.instruction);
                }),
        });
    }
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
        queues.push_back(QueueProgram {QueueKind::SxmTranspose, hemisphere,
            encode_scheduled_queue(sxm_transpose_[hemisphere],
                queue_name(QueueKind::SxmTranspose, hemisphere), encode_sxm_command)});
        queues.push_back(QueueProgram {QueueKind::SxmPermute, hemisphere,
            encode_scheduled_queue(sxm_permute_[hemisphere],
                queue_name(QueueKind::SxmPermute, hemisphere), encode_sxm_command)});
    }

    return queues;
}

void IcuProgram::load_into(InstructionControlUnit& icu) const
{
    for (std::size_t column = 0; column < mem_.size(); ++column) {
        load_scheduled_queue(
            mem_[column],
            queue_name(QueueKind::Mem, column),
            [&](std::size_t cycles) { icu.enqueue_mem_nop(column, cycles); },
            [&](const MemInstruction& instruction) { icu.enqueue_mem(column, instruction); });
    }

    for (std::size_t mxm = 0; mxm < mxm_load_.size(); ++mxm) {
        load_scheduled_queue(
            mxm_load_[mxm],
            queue_name(QueueKind::MxmLoad, mxm),
            [&](std::size_t cycles) { icu.enqueue_mxm_load_nop(mxm, cycles); },
            [&](const MxmControlInstruction& instruction) { icu.enqueue_mxm(mxm, instruction); });
        load_scheduled_queue(
            mxm_compute_[mxm],
            queue_name(QueueKind::MxmCompute, mxm),
            [&](std::size_t cycles) { icu.enqueue_mxm_compute_nop(mxm, cycles); },
            [&](const MxmControlInstruction& instruction) { icu.enqueue_mxm(mxm, instruction); });
        load_scheduled_queue(
            mxm_dequant_[mxm],
            queue_name(QueueKind::MxmDequant, mxm),
            [&](std::size_t cycles) {
                icu.enqueue_mxm_dequant_nop(mxm, cycles);
            },
            [&](const MxmDequantInstruction& instruction) {
                icu.enqueue_mxm_dequant(mxm, instruction);
            });
    }

    for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
        load_scheduled_queue(
            vxm_[alu],
            queue_name(QueueKind::Vxm, alu),
            [&](std::size_t cycles) { icu.enqueue_vxm_nop(alu, cycles); },
            [&](const VxmInstruction& item) {
                icu.enqueue_vxm(alu, item.depth, item.instruction);
            });
    }
    for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
        const auto side = static_cast<Hemisphere>(hemisphere);
        load_scheduled_queue(sxm_transpose_[hemisphere],
            queue_name(QueueKind::SxmTranspose, hemisphere),
            [&](std::size_t cycles) { icu.enqueue_sxm_transpose_nop(side, cycles); },
            [&](const SxmInstruction& instruction) { icu.enqueue_sxm_transpose(side, instruction); });
        load_scheduled_queue(sxm_permute_[hemisphere],
            queue_name(QueueKind::SxmPermute, hemisphere),
            [&](std::size_t cycles) { icu.enqueue_sxm_permute_nop(side, cycles); },
            [&](const SxmInstruction& instruction) { icu.enqueue_sxm_permute(side, instruction); });
    }
}

std::size_t IcuProgram::last_cycle() const
{
    return last_cycle_;
}

bool IcuProgram::empty() const
{
    for (const auto& queue : mem_) {
        if (!queue.empty()) {
            return false;
        }
    }
    for (const auto& queue : mxm_load_) {
        if (!queue.empty()) {
            return false;
        }
    }
    for (const auto& queue : mxm_compute_) {
        if (!queue.empty()) {
            return false;
        }
    }
    for (const auto& queue : mxm_dequant_) {
        if (!queue.empty()) return false;
    }
    for (const auto& queue : sxm_transpose_) if (!queue.empty()) return false;
    for (const auto& queue : sxm_permute_) if (!queue.empty()) return false;
    for (const auto& queue : vxm_) {
        if (!queue.empty()) {
            return false;
        }
    }
    return true;
}

void load_queue_programs_into_icu(const std::vector<QueueProgram>& queues,
    InstructionControlUnit& icu,
    std::size_t logical_mxms_per_hemisphere)
{
    validate_fu_3d_context_capacity(queues);
    for (const auto& queue : queues) {
        if (queue.commands.empty()) continue;
        const std::size_t queue_index = physical_queue_index(
            queue.kind, queue.index, logical_mxms_per_hemisphere);
        validate_queue_index(queue.kind, queue_index);
        for (std::size_t command_index = 0; command_index < queue.commands.size(); ++command_index) {
            const auto& command = queue.commands[command_index];
            if (command.instruction_kind
                    == InstructionKind::C2cEndpoint) {
                if (queue.kind != QueueKind::C2cTx
                    && queue.kind != QueueKind::C2cRx)
                    throw std::logic_error(
                        "C2C endpoint word targets a non-endpoint queue");
                const auto word = decode_c2c_raw_word(
                    command, InstructionKind::C2cEndpoint);
                const auto side = static_cast<Hemisphere>(queue_index);
                if (queue.kind == QueueKind::C2cTx) {
                    const auto decoded = C2cIcuPacketCodec::decode_tx(word);
                    if (decoded.endpoint_hemisphere != side)
                        throw std::logic_error(
                            "C2C TX packet hemisphere does not match its queue");
                    icu.c2c_tx_iq(side)
                        .push_encoded_c2c_endpoint_packet(word);
                } else {
                    const auto decoded = C2cIcuPacketCodec::decode_rx(word);
                    if (decoded.endpoint_hemisphere != side)
                        throw std::logic_error(
                            "C2C RX packet hemisphere does not match its queue");
                    icu.c2c_rx_iq(side)
                        .push_encoded_c2c_endpoint_packet(word);
                }
                continue;
            }
            if (command.instruction_kind == InstructionKind::C2cDma) {
                if (queue.kind != QueueKind::C2cDma)
                    throw std::logic_error(
                        "C2C DMA word targets a non-DMA queue");
                const auto packet = decode_c2c_dma_icu_packet(
                    queue, command_index);
                const auto decoded = C2cIcuPacketCodec::decode_dma(packet);
                const auto side = static_cast<Hemisphere>(queue_index);
                if (decoded.endpoint_hemisphere != side)
                    throw std::logic_error(
                        "C2C DMA packet hemisphere does not match its queue");
                icu.c2c_dma_iq(side).push_encoded_c2c_dma_packet(packet);
                command_index += C2cDmaIcuPacket::kWordCount - 1;
                continue;
            }
            if (is_mem_synchronized_raw_packet_header(command)) {
                if (queue.kind != QueueKind::Mem)
                    throw std::logic_error(
                        "MEM_WRITE_SYNC must target a MEM queue");
                icu.mem_iq(queue_index)
                    .push_encoded_synchronized_packet(
                        decode_mem_synchronized_icu_packet(
                            queue, command_index));
                command_index +=
                    InstructionControlUnit::MemIcu::
                        synchronized_packet_word_count - 1;
                continue;
            }
            if (is_mem_synchronized_raw_word_command(command)
                && isa::decode_icu_command_opcode(command.command)
                    == isa::IcuCommandOpcode::Instruction)
                throw std::logic_error(
                    "orphan MEM_WRITE_SYNC continuation word in binary queue");
            if (is_icu_control_raw_word_command(command)) {
                icu.enqueue_control(queue_location(queue.kind, queue_index),
                    decode_icu_control_raw_word(command));
                continue;
            }
            if (begins_raw_3d_packet(queue.kind, command)) {
                switch (queue.kind) {
                case QueueKind::Mem: {
                    if (is_mem_write_read_2d_raw_packet_header(command)) {
                        icu.mem_iq(queue_index)
                            .push_encoded_mem_write_read_2d_packet(
                                read_raw_3d_packet<
                                    isa::EncodedMemIcuWriteRead2DPacket>(
                                    queue, command_index));
                    } else {
                        const auto packet = read_raw_3d_packet<
                            isa::EncodedMemIcu3DPacket>(
                            queue, command_index);
                        icu.mem_iq(queue_index).push_encoded_3d_packet(packet);
                    }
                    break;
                }
                case QueueKind::MxmLoad:
                    icu.mxm_load_iq(queue_index).push_encoded_3d_packet(
                        read_raw_3d_packet<
                            isa::EncodedMxmLoadIcu3DPacket>(
                            queue, command_index));
                    break;
                case QueueKind::MxmDequant:
                    icu.mxm_dequant_iq(queue_index)
                        .push_encoded_3d_packet(
                            read_raw_3d_packet<
                                isa::EncodedMxmDequantIcu3DPacket>(
                                queue, command_index));
                    break;
                case QueueKind::MxmCompute:
                    icu.mxm_compute_iq(queue_index)
                        .push_encoded_3d_packet(
                            read_raw_3d_packet<
                                isa::EncodedMxmComputeIcu3DPacket>(
                                queue, command_index));
                    break;
                case QueueKind::Vxm:
                    icu.vxm_iq(queue_index).push_encoded_3d_packet(
                        read_raw_3d_packet<
                            isa::EncodedVxmIcuRun2DPacket>(
                            queue, command_index));
                    break;
                case QueueKind::SxmTranspose:
                    icu.sxm_transpose_iq(
                        static_cast<Hemisphere>(queue_index))
                        .push_encoded_3d_packet(
                            read_raw_3d_packet<
                                isa::EncodedSxmIcuRun2DPacket>(
                                queue, command_index));
                    break;
                case QueueKind::SxmPermute:
                    icu.sxm_permute_iq(
                        static_cast<Hemisphere>(queue_index))
                        .push_encoded_3d_packet(
                            read_raw_3d_packet<
                                isa::EncodedSxmIcuRun2DPacket>(
                                queue, command_index));
                    break;
                case QueueKind::C2cDma:
                case QueueKind::C2cTx:
                case QueueKind::C2cRx:
                    throw std::logic_error(
                        "C2C queues do not carry FU 3-D packets");
                }
                command_index += raw_3d_packet_word_count(queue.kind) - 1;
                continue;
            }
            if (is_vxm_stream_nd_command(command)) {
                if (queue.kind != QueueKind::Vxm)
                    throw std::logic_error(
                        "VXM_STREAM_ND must target a VXM queue");
                const auto descriptor =
                    decode_vxm_stream_nd_command(command);
                const VxmCompactInstruction instruction {
                    static_cast<std::uint64_t>(
                        descriptor.instruction.words[0])
                        | (static_cast<std::uint64_t>(
                               descriptor.instruction.words[1])
                            << 32),
                    descriptor.instruction.words[2]};
                icu.enqueue_vxm_stream_nd(queue_index,
                    descriptor.schedule, instruction);
                continue;
            }
            if (is_sxm_tile_program_command(command)) {
                if (queue.kind != QueueKind::SxmTranspose
                    && queue.kind != QueueKind::SxmPermute)
                    throw std::logic_error(
                        "SXM_TILE_PROGRAM must target an SXM queue");
                const auto descriptor =
                    decode_sxm_tile_program_command(command);
                auto instruction = decode_sxm_command(
                    descriptor.instruction);
                const auto side = static_cast<Hemisphere>(queue_index);
                if (queue.kind == QueueKind::SxmTranspose) {
                    if (instruction.opcode != SxmOpcode::Transpose)
                        throw std::logic_error(
                            "SXM transpose tile program carries a non-transpose instruction");
                    icu.enqueue_sxm_transpose_tile_program(side,
                        descriptor.schedule, std::move(instruction));
                } else {
                    if (instruction.opcode != SxmOpcode::Permute)
                        throw std::logic_error(
                            "SXM permute tile program carries a non-permute instruction");
                    icu.enqueue_sxm_permute_tile_program(side,
                        descriptor.schedule, std::move(instruction));
                }
                continue;
            }
            if (is_mem_slice_program_command(command)) {
                if (queue.kind != QueueKind::Mem)
                    throw std::logic_error(
                        "MEM_SLICE_PROGRAM must target a MEM queue");
                icu.enqueue_mem_slice_program(queue_index,
                    decode_mem_slice_program_command(command));
                continue;
            }
            if (is_mem_stream_nd_command(command)) {
                if (queue.kind != QueueKind::Mem
                    || command.word_count < 1 || command.word_count > 2)
                    throw std::logic_error(
                        "MEM_STREAM_ND must target a MEM queue and carry one or two MEM words");
                const auto encoded =
                    static_cast<isa::EncodedMemInstruction>(command.words[0])
                    | (static_cast<isa::EncodedMemInstruction>(
                           command.words[1])
                        << 32);
                icu.enqueue_mem_stream_nd(queue_index,
                    decode_mem_stream_nd_command(command),
                    isa::decode_mem_instruction(encoded));
                continue;
            }
            if (is_mxm_stream_nd_command(command)) {
                const auto schedule = decode_mxm_stream_nd_command(command);
                try {
                    if (queue.kind == QueueKind::MxmDequant) {
                        if (command.instruction_kind
                                != InstructionKind::MxmDequant
                            || command.word_count != 1)
                            throw std::logic_error(
                                "MXM dequant STREAM_ND must carry one scale word");
                        icu.enqueue_mxm_dequant_stream_nd(queue_index,
                            schedule,
                            isa::decode_mxm_dequant_instruction(
                                static_cast<
                                    isa::EncodedMxmDequantInstruction>(
                                    command.words[0])));
                    } else {
                        if ((queue.kind != QueueKind::MxmLoad
                                && queue.kind != QueueKind::MxmCompute)
                            || command.instruction_kind
                                != InstructionKind::Mxm
                            || command.word_count < 1
                            || command.word_count > 2)
                            throw std::logic_error(
                                "MXM_STREAM_ND must target an MXM queue and carry one or two MXM words");
                        const auto encoded =
                            static_cast<isa::EncodedMxmInstruction>(
                                command.words[0])
                            | (static_cast<isa::EncodedMxmInstruction>(
                                   command.words[1])
                                << 32);
                        const auto instruction =
                            isa::decode_mxm_instruction(encoded);
                        validate_mxm_queue_opcode(
                            queue.kind, queue_index, instruction);
                        if (queue.kind == QueueKind::MxmLoad)
                            icu.enqueue_mxm_load_stream_nd(queue_index,
                                schedule, instruction);
                        else
                            icu.enqueue_mxm_compute_stream_nd(queue_index,
                                schedule, instruction);
                    }
                } catch (const std::exception& error) {
                    std::ostringstream message;
                    message << error.what() << "; queue_kind="
                            << static_cast<int>(queue.kind)
                            << ", queue_index=" << queue.index
                            << ", command_index=" << command_index
                            << ", rank=" << schedule.rank << ", counts=";
                    for (std::size_t dimension = 0;
                         dimension < schedule.rank; ++dimension) {
                        if (dimension != 0) message << 'x';
                        message << schedule.counts[dimension];
                    }
                    message << ", cycle_strides=";
                    for (std::size_t dimension = 0;
                         dimension < schedule.rank; ++dimension) {
                        if (dimension != 0) message << ',';
                        message << schedule.cycle_strides[dimension];
                    }
                    throw std::logic_error(message.str());
                }
                continue;
            }
            if (is_macro_schedule_command(command)) {
                const auto schedule = decode_macro_schedule_command(command);
                switch (queue.kind) {
                case QueueKind::Mem: {
                    if (command.instruction_kind != InstructionKind::Mem
                        || command.word_count < 1 || command.word_count > 2)
                        throw std::logic_error(
                            "MEM macro must carry one or two MEM instruction words");
                    const auto encoded =
                        static_cast<isa::EncodedMemInstruction>(command.words[0])
                        | (static_cast<isa::EncodedMemInstruction>(
                               command.words[1])
                            << 32);
                    icu.enqueue_mem_macro(queue_index, schedule,
                        isa::decode_mem_instruction(encoded));
                    break;
                }
                case QueueKind::MxmLoad:
                case QueueKind::MxmCompute: {
                    if (command.instruction_kind != InstructionKind::Mxm
                        || command.word_count < 1 || command.word_count > 2)
                        throw std::logic_error(
                            "MXM macro must carry one or two MXM instruction words");
                    const auto encoded =
                        static_cast<isa::EncodedMxmInstruction>(command.words[0])
                        | (static_cast<isa::EncodedMxmInstruction>(
                               command.words[1])
                            << 32);
                    const auto instruction =
                        isa::decode_mxm_instruction(encoded);
                    validate_mxm_queue_opcode(
                        queue.kind, queue_index, instruction);
                    if (queue.kind == QueueKind::MxmLoad)
                        icu.enqueue_mxm_load_macro(
                            queue_index, schedule, instruction);
                    else
                        icu.enqueue_mxm_compute_macro(
                            queue_index, schedule, instruction);
                    break;
                }
                case QueueKind::MxmDequant:
                    if (command.instruction_kind
                            != InstructionKind::MxmDequant
                        || command.word_count != 1)
                        throw std::logic_error(
                            "MXM dequant macro must carry one scale word");
                    icu.enqueue_mxm_dequant_macro(queue_index, schedule,
                        isa::decode_mxm_dequant_instruction(
                            static_cast<isa::EncodedMxmDequantInstruction>(
                                command.words[0])));
                    break;
                case QueueKind::Vxm:
                case QueueKind::SxmTranspose:
                case QueueKind::SxmPermute:
                case QueueKind::C2cDma:
                case QueueKind::C2cTx:
                case QueueKind::C2cRx:
                    throw std::logic_error(
                        "ICU macro v1 supports MEM and MXM queues only");
                }
                continue;
            }
            if (is_repeat_2d_command(command)) {
                icu.enqueue_control(queue_location(queue.kind, queue_index),
                    IcuControlInstruction::Repeat2D(
                        decode_repeat_2d_command(command)));
                continue;
            }
            const auto opcode = isa::decode_icu_command_opcode(command.command);
            if (opcode == isa::IcuCommandOpcode::Nop) {
                const auto cycles = isa::decode_icu_nop_cycles(command.command);
                switch (queue.kind) {
                case QueueKind::Mem:
                    icu.enqueue_mem_nop(queue_index, cycles);
                    break;
                case QueueKind::MxmLoad:
                    icu.enqueue_mxm_load_nop(queue_index, cycles);
                    break;
                case QueueKind::MxmCompute:
                    icu.enqueue_mxm_compute_nop(queue_index, cycles);
                    break;
                case QueueKind::MxmDequant:
                    icu.enqueue_mxm_dequant_nop(queue_index, cycles);
                    break;
                case QueueKind::Vxm:
                    icu.enqueue_vxm_nop(queue_index, cycles);
                    break;
                case QueueKind::SxmTranspose:
                    icu.enqueue_sxm_transpose_nop(static_cast<Hemisphere>(queue_index), cycles);
                    break;
                case QueueKind::SxmPermute:
                    icu.enqueue_sxm_permute_nop(static_cast<Hemisphere>(queue_index), cycles);
                    break;
                case QueueKind::C2cDma:
                    icu.enqueue_c2c_dma_nop(
                        static_cast<Hemisphere>(queue_index), cycles);
                    break;
                case QueueKind::C2cTx:
                    icu.enqueue_c2c_tx_nop(
                        static_cast<Hemisphere>(queue_index), cycles);
                    break;
                case QueueKind::C2cRx:
                    icu.enqueue_c2c_rx_nop(
                        static_cast<Hemisphere>(queue_index), cycles);
                    break;
                }
                continue;
            }

            if (opcode == isa::IcuCommandOpcode::Repeat) {
                const auto repeat = isa::decode_icu_repeat(command.command);
                switch (queue.kind) {
                case QueueKind::Mem:
                    icu.enqueue_mem_repeat(queue_index, repeat.count, repeat.interval, repeat.address_stride);
                    break;
                case QueueKind::MxmLoad:
                    icu.enqueue_mxm_load_repeat(queue_index, repeat.count, repeat.interval);
                    break;
                case QueueKind::MxmCompute:
                    icu.enqueue_mxm_compute_repeat(queue_index, repeat.count, repeat.interval);
                    break;
                case QueueKind::MxmDequant:
                    icu.enqueue_mxm_dequant_repeat(
                        queue_index, repeat.count, repeat.interval);
                    break;
                case QueueKind::Vxm:
                    icu.enqueue_vxm_repeat(queue_index, repeat.count, repeat.interval);
                    break;
                case QueueKind::SxmTranspose:
                    icu.enqueue_sxm_transpose_repeat(
                        static_cast<Hemisphere>(queue_index),
                        repeat.count, repeat.interval);
                    break;
                case QueueKind::SxmPermute:
                    icu.enqueue_sxm_permute_repeat(
                        static_cast<Hemisphere>(queue_index),
                        repeat.count, repeat.interval);
                    break;
                case QueueKind::C2cDma:
                    icu.enqueue_control(
                        IcuLocation::C2cDma(
                            static_cast<Hemisphere>(queue_index)),
                        IcuControlInstruction::Repeat(repeat.count,
                            repeat.interval, repeat.address_stride));
                    break;
                case QueueKind::C2cTx:
                    icu.enqueue_control(
                        IcuLocation::C2cTx(
                            static_cast<Hemisphere>(queue_index)),
                        IcuControlInstruction::Repeat(repeat.count,
                            repeat.interval, repeat.address_stride));
                    break;
                case QueueKind::C2cRx:
                    icu.enqueue_control(
                        IcuLocation::C2cRx(
                            static_cast<Hemisphere>(queue_index)),
                        IcuControlInstruction::Repeat(repeat.count,
                            repeat.interval, repeat.address_stride));
                    break;
                }
                continue;
            }

            if (opcode != isa::IcuCommandOpcode::Instruction) {
                throw std::logic_error("unsupported ICU command opcode in binary queue");
            }

            switch (queue.kind) {
            case QueueKind::Mem: {
                if (command.instruction_kind != InstructionKind::Mem
                    || command.word_count < 1 || command.word_count > 2) {
                    throw std::logic_error("MEM queue command must carry one or two MEM instruction words");
                }
                const auto encoded = static_cast<isa::EncodedMemInstruction>(command.words[0])
                    | (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
                icu.enqueue_mem(queue_index, isa::decode_mem_instruction(encoded));
                break;
            }
            case QueueKind::MxmLoad:
            case QueueKind::MxmCompute: {
                if (command.instruction_kind != InstructionKind::Mxm
                    || command.word_count < 1 || command.word_count > 2) {
                    throw std::logic_error(
                        "MXM queue command must carry one or two MXM instruction words");
                }
                const auto encoded =
                    static_cast<isa::EncodedMxmInstruction>(command.words[0])
                    | (static_cast<isa::EncodedMxmInstruction>(
                           command.words[1])
                        << 32);
                MxmControlInstruction instruction;
                try {
                    instruction = isa::decode_mxm_instruction(encoded);
                } catch (const std::exception& error) {
                    std::ostringstream message;
                    message << error.what() << "; queue_kind="
                            << static_cast<int>(queue.kind)
                            << ", queue_index=" << queue.index
                            << ", command_index=" << command_index
                            << ", encoded=0x" << std::hex << encoded;
                    throw std::logic_error(message.str());
                }
                validate_mxm_queue_opcode(queue.kind, queue_index, instruction);
                icu.enqueue_mxm(queue_index, instruction);
                break;
            }
            case QueueKind::MxmDequant:
                if (command.instruction_kind
                        != InstructionKind::MxmDequant
                    || command.word_count != 1)
                    throw std::logic_error(
                        "MXM dequant queue command must carry one scale word");
                icu.enqueue_mxm_dequant(
                    queue_index,
                    isa::decode_mxm_dequant_instruction(
                        static_cast<isa::EncodedMxmDequantInstruction>(
                            command.words[0])));
                break;
            case QueueKind::Vxm:
                if (command.instruction_kind != InstructionKind::Vxm
                    || command.word_count != 3) {
                    throw std::logic_error(
                        "VXM queue command must carry one 96-bit compact packet");
                }
                icu.enqueue_vxm(queue_index, VxmCompactInstruction {
                    static_cast<std::uint64_t>(command.words[0])
                        | (static_cast<std::uint64_t>(command.words[1]) << 32),
                    command.words[2]});
                break;
            case QueueKind::SxmTranspose: {
                const auto instruction = decode_sxm_command(command);
                if (instruction.opcode != SxmOpcode::Transpose)
                    throw std::logic_error("SXM transpose queue received a non-transpose instruction");
                icu.enqueue_sxm_transpose(static_cast<Hemisphere>(queue_index), std::move(instruction));
                break;
            }
            case QueueKind::SxmPermute: {
                const auto instruction = decode_sxm_command(command);
                if (instruction.opcode != SxmOpcode::Permute)
                    throw std::logic_error("SXM permute queue received a non-permute instruction");
                icu.enqueue_sxm_permute(static_cast<Hemisphere>(queue_index), std::move(instruction));
                break;
            }
            case QueueKind::C2cDma:
            case QueueKind::C2cTx:
            case QueueKind::C2cRx:
                throw std::logic_error(
                    "C2C functional commands must use fixed raw packets");
            }
        }
    }
}

void IcuProgram::check_mem_column(std::size_t column) const
{
    if (column >= InstructionControlUnit::kMemQueues) {
        throw std::out_of_range("MEM queue index is outside the CModel ICU range");
    }
}

void IcuProgram::check_mxm(std::size_t mxm) const
{
    if (mxm >= InstructionControlUnit::kMxmQueues) {
        throw std::out_of_range("MXM queue index is outside the CModel ICU range");
    }
}

void IcuProgram::check_vxm_alu(std::size_t alu) const
{
    if (alu >= InstructionControlUnit::kVxmQueues) {
        throw std::out_of_range("VXM ALU queue index is outside the CModel ICU range");
    }
}

template <typename Instruction, typename EncodeFn>
std::vector<QueueCommand> IcuProgram::encode_scheduled_queue(
    std::vector<ScheduledInstruction<Instruction>> events,
    const std::string& queue_name,
    EncodeFn encode)
{
    std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });

    auto commands = std::vector<QueueCommand> {};
    auto cursor = std::size_t {0};
    for (const auto& event : events) {
        if (event.cycle < cursor) {
            std::ostringstream os;
            os << "two instructions target one ICU queue cycle"
               << " queue=" << queue_name
               << " event_cycle=" << event.cycle
               << " cursor=" << cursor;
            throw std::logic_error(os.str());
        }

        const auto gap = event.cycle - cursor;
        if (gap != 0) {
            commands.push_back(QueueCommand {
                isa::encode_icu_nop(gap),
                InstructionKind::None,
                0,
                {},
            });
        }
        commands.push_back(encode(event.instruction));
        cursor = event.cycle + 1;
    }
    return commands;
}

template <typename Instruction, typename NopFn, typename EmitFn>
void IcuProgram::load_scheduled_queue(
    std::vector<ScheduledInstruction<Instruction>> events,
    const std::string& queue_name,
    NopFn nop,
    EmitFn emit)
{
    std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });

    auto cursor = std::size_t {0};
    for (const auto& event : events) {
        if (event.cycle < cursor) {
            std::ostringstream os;
            os << "two instructions target one ICU queue cycle"
               << " queue=" << queue_name
               << " event_cycle=" << event.cycle
               << " cursor=" << cursor;
            throw std::logic_error(os.str());
        }
        nop(event.cycle - cursor);
        emit(event.instruction);
        cursor = event.cycle + 1;
    }
}

} // namespace ftlpu::software::runtime
