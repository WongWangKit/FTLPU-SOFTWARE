#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/icu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

enum class QueueKind : std::uint16_t {
    Mem = 0,
    MxmLoad = 1,
    MxmCompute = 2,
    MxmDequant = 3,
    Vxm = 4,
    SxmTranspose = 5,
    SxmPermute = 6,
};

enum class InstructionKind : std::uint16_t {
    None = 0,
    Mem = 1,
    Mxm = 2,
    Vxm = 3,
    Sxm = 4,
    MxmDequant = 5,
};

struct QueueCommand {
    isa::EncodedIcuCommand command{0};
    InstructionKind instruction_kind{InstructionKind::None};
    std::uint16_t word_count{0};
    std::array<std::uint32_t, 4> words{};
    // MEM/MXM/VXM fit in the fixed header. Extended ICU descriptors and SXM's
    // variable stream/lane maps use this trailing payload.
    std::vector<std::uint32_t> extension_words{};
};

inline constexpr std::uint32_t kIcuMacroScheduleMagic = 0x4d414352u;
inline constexpr std::uint32_t kIcuMemStreamNdMagic = 0x4d534e44u;
inline constexpr std::uint32_t kIcuMemSliceProgramMagic = 0x4d535047u;
inline constexpr std::uint32_t kIcuMxmStreamNdMagic = 0x4d584e44u;
inline constexpr std::uint32_t kIcuVxmStreamNdMagic = 0x56534e44u;
inline constexpr std::uint32_t kIcuSxmTileProgramMagic = 0x53545047u;

inline void validate_stream_nd_iteration_space(
    const IcuStreamNdSchedule& schedule, const char* descriptor)
{
    if (schedule.rank == 0
        || schedule.rank > IcuStreamNdSchedule::kMaxRank)
        throw std::invalid_argument(
            std::string(descriptor) + " rank must be between one and three");
    std::size_t lowerSpan = 0;
    for (std::size_t dimension = 0; dimension < schedule.rank;
         ++dimension) {
        const auto count = schedule.counts[dimension];
        const auto stride = schedule.cycle_strides[dimension];
        if (count == 0 || stride == 0)
            throw std::invalid_argument(std::string(descriptor)
                + " counts and cycle strides must be non-zero");
        if (dimension != 0 && count > 1 && stride <= lowerSpan)
            throw std::invalid_argument(std::string(descriptor)
                + " dimensions overlap in issue time");
        const auto steps = count - 1;
        if (steps != 0
            && stride > (std::numeric_limits<std::size_t>::max()
                    - lowerSpan)
                / steps)
            throw std::overflow_error(
                std::string(descriptor) + " cycle span overflows");
        lowerSpan += steps * stride;
    }
    if (schedule.start_cycle
        > std::numeric_limits<std::size_t>::max() - lowerSpan)
        throw std::overflow_error(
            std::string(descriptor) + " final issue cycle overflows");
}

inline bool is_mem_stream_nd_command(const QueueCommand& command)
{
    return command.instruction_kind == InstructionKind::Mem
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && command.extension_words.size() == 12
        && command.extension_words[0] == kIcuMemStreamNdMagic;
}

inline IcuMemStreamNdSchedule decode_mem_stream_nd_command(
    const QueueCommand& command)
{
    if (!is_mem_stream_nd_command(command))
        throw std::logic_error(
            "queue command is not MEM_STREAM_ND");
    IcuMemStreamNdSchedule schedule;
    schedule.start_cycle = command.extension_words[1];
    schedule.rank = command.extension_words[2];
    for (std::size_t dimension = 0;
         dimension < IcuMemStreamNdSchedule::kMaxRank; ++dimension) {
        const auto offset = 3 + dimension * 3;
        schedule.counts[dimension] = command.extension_words[offset];
        schedule.cycle_strides[dimension] =
            command.extension_words[offset + 1];
        schedule.operand_strides[dimension] =
            static_cast<std::int32_t>(
                command.extension_words[offset + 2]);
    }
    schedule.induction_target = IcuInductionTarget::MemAddress;
    validate_stream_nd_iteration_space(schedule, "MEM_STREAM_ND");
    return schedule;
}

inline QueueCommand encode_mem_stream_nd_command(
    QueueCommand instruction, const IcuMemStreamNdSchedule& schedule)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    const auto fits_i32 = [](std::int64_t value) {
        return value >= std::numeric_limits<std::int32_t>::min()
            && value <= std::numeric_limits<std::int32_t>::max();
    };
    if (instruction.instruction_kind != InstructionKind::Mem
        || isa::decode_icu_command_opcode(instruction.command)
            != isa::IcuCommandOpcode::Instruction
        || !instruction.extension_words.empty())
        throw std::invalid_argument(
            "MEM_STREAM_ND requires one native MEM instruction");
    if (schedule.rank == 0
        || schedule.rank > IcuMemStreamNdSchedule::kMaxRank
        || !fits_u32(schedule.start_cycle))
        throw std::invalid_argument(
            "MEM_STREAM_ND header does not fit the binary descriptor");
    validate_stream_nd_iteration_space(schedule, "MEM_STREAM_ND");
    for (std::size_t dimension = 0;
         dimension < IcuMemStreamNdSchedule::kMaxRank; ++dimension) {
        if (schedule.counts[dimension] == 0
            || schedule.cycle_strides[dimension] == 0
            || !fits_u32(schedule.counts[dimension])
            || !fits_u32(schedule.cycle_strides[dimension])
            || !fits_i32(schedule.operand_strides[dimension]))
            throw std::invalid_argument(
                "MEM_STREAM_ND dimension does not fit the binary descriptor");
    }

    instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    instruction.extension_words = {
        kIcuMemStreamNdMagic,
        static_cast<std::uint32_t>(schedule.start_cycle),
        static_cast<std::uint32_t>(schedule.rank),
    };
    for (std::size_t dimension = 0;
         dimension < IcuMemStreamNdSchedule::kMaxRank; ++dimension) {
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(schedule.counts[dimension]));
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(
                schedule.cycle_strides[dimension]));
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(
                schedule.operand_strides[dimension]));
    }
    return instruction;
}

inline bool is_mem_slice_program_command(const QueueCommand& command)
{
    if (command.instruction_kind != InstructionKind::Mem
        || command.word_count != 0
        || isa::decode_icu_command_opcode(command.command)
            != isa::IcuCommandOpcode::Extended
        || command.extension_words.size() < 17
        || command.extension_words[0] != kIcuMemSliceProgramMagic
        || command.extension_words[1] != 1)
        return false;
    const auto bodyCount = command.extension_words[4];
    return bodyCount != 0
        && bodyCount <= IcuMemSliceProgram::kMaxBodyEntries
        && command.extension_words.size() == 11 + bodyCount * 6;
}

inline IcuMemSliceProgram decode_mem_slice_program_command(
    const QueueCommand& command)
{
    if (!is_mem_slice_program_command(command))
        throw std::logic_error(
            "queue command is not MEM_SLICE_PROGRAM");

    IcuMemSliceProgram program;
    program.schedule.start_cycle = command.extension_words[2];
    program.schedule.rank = command.extension_words[3];
    for (std::size_t dimension = 0;
         dimension < IcuStreamNdSchedule::kMaxRank; ++dimension) {
        const auto offset = 5 + dimension * 2;
        program.schedule.counts[dimension] =
            command.extension_words[offset];
        program.schedule.cycle_strides[dimension] =
            command.extension_words[offset + 1];
    }
    program.schedule.operand_strides = {0, 0, 0};
    program.schedule.induction_target = IcuInductionTarget::None;
    validate_stream_nd_iteration_space(
        program.schedule, "MEM_SLICE_PROGRAM");

    const auto bodyCount = command.extension_words[4];
    program.body.reserve(bodyCount);
    for (std::size_t index = 0; index < bodyCount; ++index) {
        const auto offset = 11 + index * 6;
        const auto encoded =
            static_cast<isa::EncodedMemInstruction>(
                command.extension_words[offset + 4])
            | (static_cast<isa::EncodedMemInstruction>(
                   command.extension_words[offset + 5])
                << 32);
        IcuMemSliceProgramEntry entry;
        entry.cycle_offset = command.extension_words[offset];
        for (std::size_t dimension = 0;
             dimension < IcuStreamNdSchedule::kMaxRank; ++dimension)
            entry.operand_strides[dimension] =
                static_cast<std::int32_t>(
                    command.extension_words[offset + 1 + dimension]);
        entry.instruction = isa::decode_mem_instruction(encoded);

        auto bodySchedule = program.schedule;
        if (entry.cycle_offset
            > std::numeric_limits<std::size_t>::max()
                - bodySchedule.start_cycle)
            throw std::overflow_error(
                "MEM_SLICE_PROGRAM body start cycle overflows");
        bodySchedule.start_cycle += entry.cycle_offset;
        bodySchedule.operand_strides = entry.operand_strides;
        bodySchedule.induction_target = IcuInductionTarget::MemAddress;
        validate_stream_nd_iteration_space(
            bodySchedule, "MEM_SLICE_PROGRAM body");
        program.body.push_back(std::move(entry));
    }
    return program;
}

inline QueueCommand encode_mem_slice_program_command(
    const IcuMemSliceProgram& program)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    const auto fits_i32 = [](std::int64_t value) {
        return value >= std::numeric_limits<std::int32_t>::min()
            && value <= std::numeric_limits<std::int32_t>::max();
    };
    if (program.body.empty()
        || program.body.size() > IcuMemSliceProgram::kMaxBodyEntries)
        throw std::invalid_argument(
            "MEM_SLICE_PROGRAM body must contain between one and sixteen entries");
    if (!fits_u32(program.schedule.start_cycle)
        || program.schedule.rank == 0
        || program.schedule.rank > IcuStreamNdSchedule::kMaxRank
        || program.schedule.induction_target != IcuInductionTarget::None
        || std::any_of(program.schedule.operand_strides.begin(),
            program.schedule.operand_strides.end(),
            [](std::int64_t stride) { return stride != 0; }))
        throw std::invalid_argument(
            "MEM_SLICE_PROGRAM has an invalid launch domain");
    validate_stream_nd_iteration_space(
        program.schedule, "MEM_SLICE_PROGRAM");

    QueueCommand command;
    command.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    command.instruction_kind = InstructionKind::Mem;
    command.extension_words = {
        kIcuMemSliceProgramMagic,
        1,
        static_cast<std::uint32_t>(program.schedule.start_cycle),
        static_cast<std::uint32_t>(program.schedule.rank),
        static_cast<std::uint32_t>(program.body.size()),
    };
    for (std::size_t dimension = 0;
         dimension < IcuStreamNdSchedule::kMaxRank; ++dimension) {
        if (program.schedule.counts[dimension] == 0
            || program.schedule.cycle_strides[dimension] == 0
            || !fits_u32(program.schedule.counts[dimension])
            || !fits_u32(program.schedule.cycle_strides[dimension]))
            throw std::invalid_argument(
                "MEM_SLICE_PROGRAM dimension does not fit the binary descriptor");
        command.extension_words.push_back(
            static_cast<std::uint32_t>(
                program.schedule.counts[dimension]));
        command.extension_words.push_back(
            static_cast<std::uint32_t>(
                program.schedule.cycle_strides[dimension]));
    }
    for (const auto& entry : program.body) {
        if (!fits_u32(entry.cycle_offset)
            || std::any_of(entry.operand_strides.begin(),
                entry.operand_strides.end(),
                [&](std::int64_t stride) { return !fits_i32(stride); }))
            throw std::invalid_argument(
                "MEM_SLICE_PROGRAM body does not fit the binary descriptor");
        const auto encoded = isa::encode_mem_instruction(entry.instruction);
        command.extension_words.push_back(
            static_cast<std::uint32_t>(entry.cycle_offset));
        for (const auto stride : entry.operand_strides)
            command.extension_words.push_back(
                static_cast<std::uint32_t>(stride));
        command.extension_words.push_back(
            static_cast<std::uint32_t>(encoded));
        command.extension_words.push_back(
            static_cast<std::uint32_t>(encoded >> 32));
    }
    return command;
}

inline bool is_mxm_stream_nd_command(const QueueCommand& command)
{
    return (command.instruction_kind == InstructionKind::Mxm
            || command.instruction_kind
                == InstructionKind::MxmDequant)
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && command.extension_words.size() == 13
        && command.extension_words[0] == kIcuMxmStreamNdMagic;
}

inline IcuMxmStreamNdSchedule decode_mxm_stream_nd_command(
    const QueueCommand& command)
{
    if (!is_mxm_stream_nd_command(command))
        throw std::logic_error(
            "queue command is not MXM_STREAM_ND");
    IcuMxmStreamNdSchedule schedule;
    schedule.start_cycle = command.extension_words[1];
    schedule.rank = command.extension_words[2];
    schedule.induction_target = static_cast<IcuInductionTarget>(
        command.extension_words[3]);
    for (std::size_t dimension = 0;
         dimension < IcuMxmStreamNdSchedule::kMaxRank; ++dimension) {
        const auto offset = 4 + dimension * 3;
        schedule.counts[dimension] = command.extension_words[offset];
        schedule.cycle_strides[dimension] =
            command.extension_words[offset + 1];
        schedule.operand_strides[dimension] =
            static_cast<std::int32_t>(
                command.extension_words[offset + 2]);
    }
    if (schedule.induction_target
        > IcuInductionTarget::MxmAccumulatorAddress)
        throw std::invalid_argument(
            "MXM_STREAM_ND has an invalid induction target");
    validate_stream_nd_iteration_space(schedule, "MXM_STREAM_ND");
    return schedule;
}

inline QueueCommand encode_mxm_stream_nd_command(
    QueueCommand instruction, const IcuMxmStreamNdSchedule& schedule)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    const auto fits_i32 = [](std::int64_t value) {
        return value >= std::numeric_limits<std::int32_t>::min()
            && value <= std::numeric_limits<std::int32_t>::max();
    };
    if ((instruction.instruction_kind != InstructionKind::Mxm
            && instruction.instruction_kind
                != InstructionKind::MxmDequant)
        || isa::decode_icu_command_opcode(instruction.command)
            != isa::IcuCommandOpcode::Instruction
        || !instruction.extension_words.empty())
        throw std::invalid_argument(
            "MXM_STREAM_ND requires one native MXM instruction");
    if (schedule.rank == 0
        || schedule.rank > IcuMxmStreamNdSchedule::kMaxRank
        || !fits_u32(schedule.start_cycle)
        || schedule.induction_target
            > IcuInductionTarget::MxmAccumulatorAddress)
        throw std::invalid_argument(
            "MXM_STREAM_ND header does not fit the binary descriptor");
    validate_stream_nd_iteration_space(schedule, "MXM_STREAM_ND");
    bool hasOperandInduction = false;
    for (std::size_t dimension = 0;
         dimension < IcuMxmStreamNdSchedule::kMaxRank; ++dimension) {
        if (schedule.counts[dimension] == 0
            || schedule.cycle_strides[dimension] == 0
            || !fits_u32(schedule.counts[dimension])
            || !fits_u32(schedule.cycle_strides[dimension])
            || !fits_i32(schedule.operand_strides[dimension]))
            throw std::invalid_argument(
                "MXM_STREAM_ND dimension does not fit the binary descriptor");
        if (dimension < schedule.rank
            && schedule.operand_strides[dimension] != 0)
            hasOperandInduction = true;
    }
    if (hasOperandInduction
        && schedule.induction_target == IcuInductionTarget::None)
        throw std::invalid_argument(
            "MXM_STREAM_ND operand stride requires an induction target");
    if (instruction.instruction_kind == InstructionKind::MxmDequant
        && (hasOperandInduction
            || schedule.induction_target != IcuInductionTarget::None))
        throw std::invalid_argument(
            "MXM dequant STREAM_ND cannot induce an operand");

    instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    instruction.extension_words = {
        kIcuMxmStreamNdMagic,
        static_cast<std::uint32_t>(schedule.start_cycle),
        static_cast<std::uint32_t>(schedule.rank),
        static_cast<std::uint32_t>(schedule.induction_target),
    };
    for (std::size_t dimension = 0;
         dimension < IcuMxmStreamNdSchedule::kMaxRank; ++dimension) {
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(schedule.counts[dimension]));
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(
                schedule.cycle_strides[dimension]));
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(
                schedule.operand_strides[dimension]));
    }
    return instruction;
}

struct VxmStreamNdDescriptor {
    IcuVxmStreamNdSchedule schedule{};
    QueueCommand instruction{};
};

inline bool is_vxm_stream_nd_command(const QueueCommand& command)
{
    return command.instruction_kind == InstructionKind::Vxm
        && command.word_count == 3
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && command.extension_words.size() == 9
        && command.extension_words[0] == kIcuVxmStreamNdMagic;
}

inline VxmStreamNdDescriptor decode_vxm_stream_nd_command(
    const QueueCommand& command)
{
    if (!is_vxm_stream_nd_command(command))
        throw std::logic_error("queue command is not VXM_STREAM_ND");
    VxmStreamNdDescriptor descriptor;
    auto& schedule = descriptor.schedule;
    schedule.start_cycle = command.extension_words[1];
    schedule.rank = command.extension_words[2];
    for (std::size_t dimension = 0;
         dimension < IcuVxmStreamNdSchedule::kMaxRank; ++dimension) {
        const auto offset = 3 + dimension * 2;
        schedule.counts[dimension] = command.extension_words[offset];
        schedule.cycle_strides[dimension] =
            command.extension_words[offset + 1];
        schedule.operand_strides[dimension] = 0;
    }
    schedule.induction_target = IcuInductionTarget::None;
    validate_stream_nd_iteration_space(schedule, "VXM_STREAM_ND");
    descriptor.instruction = command;
    descriptor.instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Instruction);
    descriptor.instruction.extension_words.clear();
    return descriptor;
}

inline QueueCommand encode_vxm_stream_nd_command(
    QueueCommand instruction, const IcuVxmStreamNdSchedule& schedule)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    if (instruction.instruction_kind != InstructionKind::Vxm
        || instruction.word_count != 3
        || isa::decode_icu_command_opcode(instruction.command)
            != isa::IcuCommandOpcode::Instruction
        || !instruction.extension_words.empty())
        throw std::invalid_argument(
            "VXM_STREAM_ND requires one native 96-bit VXM packet");
    if (schedule.rank == 0
        || schedule.rank > IcuVxmStreamNdSchedule::kMaxRank
        || !fits_u32(schedule.start_cycle)
        || schedule.induction_target != IcuInductionTarget::None)
        throw std::invalid_argument(
            "VXM_STREAM_ND header does not fit the binary descriptor");
    validate_stream_nd_iteration_space(schedule, "VXM_STREAM_ND");

    instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    instruction.extension_words = {
        kIcuVxmStreamNdMagic,
        static_cast<std::uint32_t>(schedule.start_cycle),
        static_cast<std::uint32_t>(schedule.rank),
    };
    for (std::size_t dimension = 0;
         dimension < IcuVxmStreamNdSchedule::kMaxRank; ++dimension) {
        if (schedule.counts[dimension] == 0
            || schedule.cycle_strides[dimension] == 0
            || !fits_u32(schedule.counts[dimension])
            || !fits_u32(schedule.cycle_strides[dimension])
            || schedule.operand_strides[dimension] != 0)
            throw std::invalid_argument(
                "VXM_STREAM_ND dimension does not fit the binary descriptor");
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(schedule.counts[dimension]));
        instruction.extension_words.push_back(
            static_cast<std::uint32_t>(
                schedule.cycle_strides[dimension]));
    }
    return instruction;
}

struct SxmTileProgramDescriptor {
    IcuSxmTileProgramSchedule schedule{};
    QueueCommand instruction{};
};

inline bool is_sxm_tile_program_command(const QueueCommand& command)
{
    return command.instruction_kind == InstructionKind::Sxm
        && command.word_count == 4
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && command.extension_words.size()
            >= 9 + 2 + SxmInstruction::kTotalLanes
        && command.extension_words[0] == kIcuSxmTileProgramMagic;
}

inline SxmTileProgramDescriptor decode_sxm_tile_program_command(
    const QueueCommand& command)
{
    if (!is_sxm_tile_program_command(command))
        throw std::logic_error(
            "queue command is not SXM_TILE_PROGRAM");
    SxmTileProgramDescriptor descriptor;
    auto& schedule = descriptor.schedule;
    schedule.start_cycle = command.extension_words[1];
    schedule.rank = command.extension_words[2];
    for (std::size_t dimension = 0;
         dimension < IcuSxmTileProgramSchedule::kMaxRank; ++dimension) {
        const auto offset = 3 + dimension * 2;
        schedule.counts[dimension] = command.extension_words[offset];
        schedule.cycle_strides[dimension] =
            command.extension_words[offset + 1];
        schedule.operand_strides[dimension] = 0;
    }
    schedule.induction_target = IcuInductionTarget::None;
    validate_stream_nd_iteration_space(schedule, "SXM_TILE_PROGRAM");
    descriptor.instruction = command;
    descriptor.instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Instruction);
    descriptor.instruction.extension_words.erase(
        descriptor.instruction.extension_words.begin(),
        descriptor.instruction.extension_words.begin() + 9);
    return descriptor;
}

inline QueueCommand encode_sxm_tile_program_command(
    QueueCommand instruction, const IcuSxmTileProgramSchedule& schedule)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    if (instruction.instruction_kind != InstructionKind::Sxm
        || instruction.word_count != 4
        || isa::decode_icu_command_opcode(instruction.command)
            != isa::IcuCommandOpcode::Instruction
        || instruction.extension_words.size()
            < 2 + SxmInstruction::kTotalLanes)
        throw std::invalid_argument(
            "SXM_TILE_PROGRAM requires one native SXM instruction");
    if (schedule.rank == 0
        || schedule.rank > IcuSxmTileProgramSchedule::kMaxRank
        || !fits_u32(schedule.start_cycle)
        || schedule.induction_target != IcuInductionTarget::None)
        throw std::invalid_argument(
            "SXM_TILE_PROGRAM header does not fit the binary descriptor");
    validate_stream_nd_iteration_space(schedule, "SXM_TILE_PROGRAM");

    std::vector<std::uint32_t> payload {
        kIcuSxmTileProgramMagic,
        static_cast<std::uint32_t>(schedule.start_cycle),
        static_cast<std::uint32_t>(schedule.rank),
    };
    for (std::size_t dimension = 0;
         dimension < IcuSxmTileProgramSchedule::kMaxRank; ++dimension) {
        if (schedule.counts[dimension] == 0
            || schedule.cycle_strides[dimension] == 0
            || !fits_u32(schedule.counts[dimension])
            || !fits_u32(schedule.cycle_strides[dimension])
            || schedule.operand_strides[dimension] != 0)
            throw std::invalid_argument(
                "SXM_TILE_PROGRAM dimension does not fit the binary descriptor");
        payload.push_back(
            static_cast<std::uint32_t>(schedule.counts[dimension]));
        payload.push_back(static_cast<std::uint32_t>(
            schedule.cycle_strides[dimension]));
    }
    payload.insert(payload.end(), instruction.extension_words.begin(),
        instruction.extension_words.end());
    instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    instruction.extension_words = std::move(payload);
    return instruction;
}

inline bool is_macro_schedule_command(const QueueCommand& command)
{
    return command.instruction_kind != InstructionKind::None
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && command.extension_words.size() == 9
        && command.extension_words[0] == kIcuMacroScheduleMagic;
}

inline IcuMacroSchedule decode_macro_schedule_command(
    const QueueCommand& command)
{
    if (!is_macro_schedule_command(command))
        throw std::logic_error("queue command is not an ICU macro schedule");
    return IcuMacroSchedule {
        command.extension_words[1],
        command.extension_words[2],
        command.extension_words[3],
        static_cast<std::int32_t>(command.extension_words[4]),
        command.extension_words[5],
        command.extension_words[6],
        static_cast<std::int32_t>(command.extension_words[7]),
        static_cast<IcuInductionTarget>(command.extension_words[8]),
    };
}

inline QueueCommand encode_macro_schedule_command(
    QueueCommand instruction, const IcuMacroSchedule& schedule)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    const auto fits_i32 = [](std::int64_t value) {
        return value >= std::numeric_limits<std::int32_t>::min()
            && value <= std::numeric_limits<std::int32_t>::max();
    };
    if (instruction.instruction_kind == InstructionKind::None
        || isa::decode_icu_command_opcode(instruction.command)
            != isa::IcuCommandOpcode::Instruction
        || !instruction.extension_words.empty())
        throw std::invalid_argument(
            "ICU macro requires one native functional instruction");
    if (schedule.inner_count == 0 || schedule.outer_count == 0
        || schedule.inner_interval == 0 || schedule.outer_interval == 0
        || !fits_u32(schedule.start_cycle)
        || !fits_u32(schedule.inner_count)
        || !fits_u32(schedule.inner_interval)
        || !fits_u32(schedule.outer_count)
        || !fits_u32(schedule.outer_interval)
        || !fits_i32(schedule.inner_stride)
        || !fits_i32(schedule.outer_stride))
        throw std::invalid_argument(
            "ICU macro schedule does not fit the binary descriptor");
    if (static_cast<std::uint8_t>(schedule.induction_target)
        > static_cast<std::uint8_t>(
            IcuInductionTarget::MxmAccumulatorAddress))
        throw std::invalid_argument("ICU macro has an invalid induction target");

    instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    instruction.extension_words = {
        kIcuMacroScheduleMagic,
        static_cast<std::uint32_t>(schedule.start_cycle),
        static_cast<std::uint32_t>(schedule.inner_count),
        static_cast<std::uint32_t>(schedule.inner_interval),
        static_cast<std::uint32_t>(schedule.inner_stride),
        static_cast<std::uint32_t>(schedule.outer_count),
        static_cast<std::uint32_t>(schedule.outer_interval),
        static_cast<std::uint32_t>(schedule.outer_stride),
        static_cast<std::uint32_t>(schedule.induction_target),
    };
    return instruction;
}

inline bool is_repeat_2d_command(const QueueCommand& command)
{
    return command.instruction_kind == InstructionKind::None
        && command.word_count == 3
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended;
}

inline IcuRepeat2D decode_repeat_2d_command(const QueueCommand& command)
{
    if (!is_repeat_2d_command(command))
        throw std::logic_error("queue command is not ICU Repeat2D");
    return isa::decode_icu_repeat_2d(
        isa::EncodedIcuRepeat2D {{
            command.words[0], command.words[1], command.words[2]}});
}

struct QueueProgram {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::vector<QueueCommand> commands{};
};

class IcuProgram {
public:
    void emit_mem(std::size_t cycle, std::size_t column, MemInstruction instruction);
    void emit_mxm_load(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction);
    void emit_mxm_dequant(std::size_t cycle, std::size_t mxm, MxmDequantInstruction instruction);
    void emit_mxm_compute(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction);
    void emit_vxm(std::size_t cycle, std::size_t alu,
        VxmChainDepth depth, VxmLaneAluInstruction instruction);
    void emit_vxm(std::size_t cycle, std::size_t alu, VxmLaneAluInstruction instruction);
    void emit_sxm_transpose(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction);
    void emit_sxm_permute(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction);

    std::vector<QueueProgram> encode_queues() const;
    void load_into(InstructionControlUnit& icu) const;

    std::size_t last_cycle() const;
    bool empty() const;

private:
    template <typename Instruction>
    struct ScheduledInstruction {
        std::size_t cycle{0};
        Instruction instruction{};
    };

    using MemQueue = std::vector<ScheduledInstruction<MemInstruction>>;
    using MxmQueue = std::vector<ScheduledInstruction<MxmControlInstruction>>;
    using MxmDequantQueue =
        std::vector<ScheduledInstruction<MxmDequantInstruction>>;
    struct VxmInstruction {
        VxmChainDepth depth{VxmChainDepth::Eight};
        VxmLaneAluInstruction instruction{};
    };
    using VxmQueue = std::vector<ScheduledInstruction<VxmInstruction>>;
    using SxmQueue = std::vector<ScheduledInstruction<SxmInstruction>>;

    void check_mem_column(std::size_t column) const;
    void check_mxm(std::size_t mxm) const;
    void check_vxm_alu(std::size_t alu) const;

    template <typename Instruction, typename EncodeFn>
    static std::vector<QueueCommand> encode_scheduled_queue(
        std::vector<ScheduledInstruction<Instruction>> events,
        const std::string& queue_name,
        EncodeFn encode);

    template <typename Instruction, typename NopFn, typename EmitFn>
    static void load_scheduled_queue(
        std::vector<ScheduledInstruction<Instruction>> events,
        const std::string& queue_name,
        NopFn nop,
        EmitFn emit);

    std::array<MemQueue, InstructionControlUnit::kMemQueues> mem_{};
    std::array<MxmQueue, InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<MxmDequantQueue, InstructionControlUnit::kMxmQueues>
        mxm_dequant_{};
    std::array<MxmQueue, InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<VxmQueue, InstructionControlUnit::kVxmQueues> vxm_{};
    std::array<SxmQueue, hw::kHemispheres> sxm_transpose_{};
    std::array<SxmQueue, hw::kHemispheres> sxm_permute_{};
    std::size_t last_cycle_{0};
};

const char* queue_kind_name(QueueKind kind);
void load_queue_programs_into_icu(const std::vector<QueueProgram>& queues,
    InstructionControlUnit& icu,
    std::size_t logical_mxms_per_hemisphere = hw::kMxmsPerHemisphere);

} // namespace ftlpu::software::runtime
