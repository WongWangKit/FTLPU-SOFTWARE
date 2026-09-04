#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ftlpu::software::runtime {

namespace {

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
}

bool is_mxm_queue(QueueKind kind)
{
    return kind == QueueKind::MxmLoad
        || kind == QueueKind::MxmCompute
        || kind == QueueKind::MxmDequant;
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
    for (const auto& source_queue : queues) {
        QueueProgram decoded_physical_queue;
        const QueueProgram* queue_pointer = &source_queue;
        if (source_queue.kind == QueueKind::Mem
            && !source_queue.commands.empty()
            && std::all_of(source_queue.commands.begin(),
                source_queue.commands.end(), [](const QueueCommand& command) {
                    return is_macro_schedule_command(command)
                        && command.instruction_kind == InstructionKind::Mem;
                })) {
            // Exercise the exact target bitstream decoder on the CModel path;
            // the distributed ICU then models the decoded finite Macro
            // contexts and issue timing.
            decoded_physical_queue = decode_mem_macro_bitstream(
                encode_mem_macro_bitstream(source_queue), source_queue.index);
            queue_pointer = &decoded_physical_queue;
        }
        const auto& queue = *queue_pointer;
        if (queue.commands.empty()) continue;
        const std::size_t queue_index = physical_queue_index(
            queue.kind, queue.index, logical_mxms_per_hemisphere);
        validate_queue_index(queue.kind, queue_index);
        for (std::size_t command_index = 0; command_index < queue.commands.size(); ++command_index) {
            const auto& command = queue.commands[command_index];
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
                            icu.enqueue_mxm_load_stream_nd(
                                queue_index, schedule, instruction);
                        else
                            icu.enqueue_mxm_compute_stream_nd(
                                queue_index, schedule, instruction);
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
                    throw std::logic_error(
                        "ICU macro v1 supports MEM and MXM queues only");
                }
                continue;
            }
            if (is_repeat_2d_command(command)) {
                IcuLocation location;
                switch (queue.kind) {
                case QueueKind::Mem:
                    location = IcuLocation::Mem(
                        static_cast<Hemisphere>(queue_index
                            / InstructionControlUnit::kMemQueuesPerHemisphere),
                        (queue_index
                            % InstructionControlUnit::kMemQueuesPerHemisphere)
                            / hw::kMemBanksPerSlice,
                        queue_index % hw::kMemBanksPerSlice);
                    break;
                case QueueKind::MxmLoad:
                    location = IcuLocation::MxmLoad(queue_index);
                    break;
                case QueueKind::MxmCompute:
                    location = IcuLocation::MxmCompute(queue_index);
                    break;
                case QueueKind::MxmDequant:
                    location = IcuLocation::MxmDequant(queue_index);
                    break;
                case QueueKind::Vxm:
                    location = IcuLocation::Vxm(queue_index);
                    break;
                case QueueKind::SxmTranspose:
                    location = IcuLocation::Sxm(
                        static_cast<Hemisphere>(queue_index), 0);
                    break;
                case QueueKind::SxmPermute:
                    location = IcuLocation::Sxm(
                        static_cast<Hemisphere>(queue_index), 1);
                    break;
                }
                icu.enqueue_control(location, IcuControlInstruction::Repeat2D(
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
