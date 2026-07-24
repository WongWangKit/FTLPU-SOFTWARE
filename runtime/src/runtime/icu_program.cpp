#include "ftlpu/software/runtime/icu_program.hpp"

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

QueueCommand encode_vxm_command(const VxmLaneAluInstruction& instruction)
{
    const auto encoded = isa::encode_vxm_instruction(instruction);
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Vxm,
        4,
        encoded.words,
    };
}

QueueCommand encode_sxm_command(const SxmInstruction& instruction)
{
    QueueCommand command {kInstructionCommand, InstructionKind::Sxm, 4, {}};
    command.words[0] = static_cast<std::uint32_t>(instruction.opcode);
    command.words[1] = static_cast<std::uint32_t>(instruction.shift_source);
    command.words[2] = static_cast<std::uint32_t>(instruction.shift_distance);
    command.words[3] = static_cast<std::uint32_t>(instruction.weight_layout);
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
    instruction.weight_layout = static_cast<SxmWeightLayout>(command.words[3]);
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
    if ((kind == QueueKind::MxmLoad || kind == QueueKind::MxmCompute)
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

void IcuProgram::emit_vxm(std::size_t cycle, std::size_t alu, VxmLaneAluInstruction instruction)
{
    check_vxm_alu(alu);
    vxm_[alu].push_back(ScheduledInstruction<VxmLaneAluInstruction> {cycle, instruction});
    last_cycle_ = std::max(last_cycle_, cycle);
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
    }

    for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
        queues.push_back(QueueProgram {
            QueueKind::Vxm,
            alu,
            encode_scheduled_queue(vxm_[alu], queue_name(QueueKind::Vxm, alu), encode_vxm_command),
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
    }

    for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
        load_scheduled_queue(
            vxm_[alu],
            queue_name(QueueKind::Vxm, alu),
            [&](std::size_t cycles) { icu.enqueue_vxm_nop(alu, cycles); },
            [&](const VxmLaneAluInstruction& instruction) { icu.enqueue_vxm(alu, instruction); });
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
    for (const auto& queue : sxm_transpose_) if (!queue.empty()) return false;
    for (const auto& queue : sxm_permute_) if (!queue.empty()) return false;
    for (const auto& queue : vxm_) {
        if (!queue.empty()) {
            return false;
        }
    }
    return true;
}

void load_queue_programs_into_icu(const std::vector<QueueProgram>& queues, InstructionControlUnit& icu)
{
    for (const auto& queue : queues) {
        validate_queue_index(queue.kind, queue.index);
        for (std::size_t command_index = 0; command_index < queue.commands.size(); ++command_index) {
            const auto& command = queue.commands[command_index];
            const auto opcode = isa::decode_icu_command_opcode(command.command);
            if (opcode == isa::IcuCommandOpcode::Nop) {
                const auto cycles = isa::decode_icu_nop_cycles(command.command);
                switch (queue.kind) {
                case QueueKind::Mem:
                    icu.enqueue_mem_nop(queue.index, cycles);
                    break;
                case QueueKind::MxmLoad:
                    icu.enqueue_mxm_load_nop(queue.index, cycles);
                    break;
                case QueueKind::MxmCompute:
                    icu.enqueue_mxm_compute_nop(queue.index, cycles);
                    break;
                case QueueKind::Vxm:
                    icu.enqueue_vxm_nop(queue.index, cycles);
                    break;
                case QueueKind::SxmTranspose:
                    icu.enqueue_sxm_transpose_nop(static_cast<Hemisphere>(queue.index), cycles);
                    break;
                case QueueKind::SxmPermute:
                    icu.enqueue_sxm_permute_nop(static_cast<Hemisphere>(queue.index), cycles);
                    break;
                }
                continue;
            }

            if (opcode == isa::IcuCommandOpcode::Repeat) {
                const auto repeat = isa::decode_icu_repeat(command.command);
                switch (queue.kind) {
                case QueueKind::Mem:
                    icu.enqueue_mem_repeat(queue.index, repeat.count, repeat.interval, repeat.address_stride);
                    break;
                case QueueKind::MxmLoad:
                    icu.enqueue_mxm_load_repeat(queue.index, repeat.count, repeat.interval);
                    break;
                case QueueKind::MxmCompute:
                    icu.enqueue_mxm_compute_repeat(queue.index, repeat.count, repeat.interval);
                    break;
                case QueueKind::Vxm:
                    icu.enqueue_vxm_repeat(queue.index, repeat.count, repeat.interval);
                    break;
                case QueueKind::SxmTranspose:
                case QueueKind::SxmPermute:
                    throw std::logic_error("SXM queues do not support repeat commands");
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
                icu.enqueue_mem(queue.index, isa::decode_mem_instruction(encoded));
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
                const auto instruction =
                    isa::decode_mxm_instruction(encoded);
                validate_mxm_queue_opcode(queue.kind, queue.index, instruction);
                icu.enqueue_mxm(queue.index, instruction);
                break;
            }
            case QueueKind::Vxm:
                if (command.instruction_kind != InstructionKind::Vxm || command.word_count != 4) {
                    throw std::logic_error("VXM queue command does not carry four VXM instruction words");
                }
                icu.enqueue_vxm(queue.index, isa::decode_vxm_instruction(isa::EncodedVxmInstruction {command.words}));
                break;
            case QueueKind::SxmTranspose: {
                const auto instruction = decode_sxm_command(command);
                if (instruction.opcode != SxmOpcode::Transpose)
                    throw std::logic_error("SXM transpose queue received a non-transpose instruction");
                icu.enqueue_sxm_transpose(static_cast<Hemisphere>(queue.index), std::move(instruction));
                break;
            }
            case QueueKind::SxmPermute: {
                const auto instruction = decode_sxm_command(command);
                if (instruction.opcode != SxmOpcode::Permute)
                    throw std::logic_error("SXM permute queue received a non-permute instruction");
                icu.enqueue_sxm_permute(static_cast<Hemisphere>(queue.index), std::move(instruction));
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
