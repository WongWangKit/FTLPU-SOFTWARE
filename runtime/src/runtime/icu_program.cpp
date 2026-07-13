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
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Mem,
        1,
        {isa::encode_mem_instruction(instruction), 0, 0},
    };
}

QueueCommand encode_mxm_command(const MxmControlInstruction& instruction)
{
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Mxm,
        1,
        {isa::encode_mxm_instruction(instruction), 0, 0},
    };
}

QueueCommand encode_vxm_command(const VxmLaneAluInstruction& instruction)
{
    const auto encoded = isa::encode_vxm_instruction(instruction);
    return QueueCommand {
        kInstructionCommand,
        InstructionKind::Vxm,
        3,
        encoded.words,
    };
}

void validate_mxm_queue_opcode(
    QueueKind kind,
    std::size_t mxm,
    const MxmControlInstruction& instruction)
{
    if (kind == QueueKind::MxmLoad && instruction.opcode != MxmControlOpcode::IW) {
        throw std::logic_error("MXM load queue only accepts IW instructions");
    }
    if (kind == QueueKind::MxmCompute && instruction.opcode != MxmControlOpcode::Compute) {
        throw std::logic_error("MXM compute queue only accepts Compute instructions");
    }
    if (kind == QueueKind::MxmOutput && instruction.opcode != MxmControlOpcode::Output) {
        throw std::logic_error("MXM output queue only accepts Output instructions");
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
    if ((kind == QueueKind::MxmLoad || kind == QueueKind::MxmCompute || kind == QueueKind::MxmOutput)
        && index >= InstructionControlUnit::kMxmQueues) {
        throw std::out_of_range("binary MXM queue index is outside the CModel ICU range");
    }
    if (kind == QueueKind::Vxm && index >= InstructionControlUnit::kVxmQueues) {
        throw std::out_of_range("binary VXM queue index is outside the CModel ICU range");
    }
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
    case QueueKind::MxmOutput:
        return "mxm_output";
    case QueueKind::Vxm:
        return "vxm";
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

void IcuProgram::emit_mxm_output(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction)
{
    check_mxm(mxm);
    validate_mxm_queue_opcode(QueueKind::MxmOutput, mxm, instruction);
    mxm_output_[mxm].push_back(ScheduledInstruction<MxmControlInstruction> {cycle, instruction});
    last_cycle_ = std::max(last_cycle_, cycle);
}

void IcuProgram::emit_vxm(std::size_t cycle, std::size_t alu, VxmLaneAluInstruction instruction)
{
    check_vxm_alu(alu);
    vxm_[alu].push_back(ScheduledInstruction<VxmLaneAluInstruction> {cycle, instruction});
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
            QueueKind::MxmOutput,
            mxm,
            encode_scheduled_queue(mxm_output_[mxm], queue_name(QueueKind::MxmOutput, mxm), encode_mxm_command),
        });
    }

    for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
        queues.push_back(QueueProgram {
            QueueKind::Vxm,
            alu,
            encode_scheduled_queue(vxm_[alu], queue_name(QueueKind::Vxm, alu), encode_vxm_command),
        });
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
            mxm_output_[mxm],
            queue_name(QueueKind::MxmOutput, mxm),
            [&](std::size_t cycles) { icu.enqueue_mxm_output_nop(mxm, cycles); },
            [&](const MxmControlInstruction& instruction) { icu.enqueue_mxm(mxm, instruction); });
    }

    for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
        load_scheduled_queue(
            vxm_[alu],
            queue_name(QueueKind::Vxm, alu),
            [&](std::size_t cycles) { icu.enqueue_vxm_nop(alu, cycles); },
            [&](const VxmLaneAluInstruction& instruction) { icu.enqueue_vxm(alu, instruction); });
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
    for (const auto& queue : mxm_output_) {
        if (!queue.empty()) {
            return false;
        }
    }
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
        for (const auto& command : queue.commands) {
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
                case QueueKind::MxmOutput:
                    icu.enqueue_mxm_output_nop(queue.index, cycles);
                    break;
                case QueueKind::Vxm:
                    icu.enqueue_vxm_nop(queue.index, cycles);
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
                case QueueKind::MxmOutput:
                    icu.enqueue_mxm_output_repeat(queue.index, repeat.count, repeat.interval);
                    break;
                case QueueKind::Vxm:
                    icu.enqueue_vxm_repeat(queue.index, repeat.count, repeat.interval);
                    break;
                }
                continue;
            }

            if (opcode != isa::IcuCommandOpcode::Instruction) {
                throw std::logic_error("unsupported ICU command opcode in binary queue");
            }

            switch (queue.kind) {
            case QueueKind::Mem:
                if (command.instruction_kind != InstructionKind::Mem || command.word_count != 1) {
                    throw std::logic_error("MEM queue command does not carry one MEM instruction word");
                }
                icu.enqueue_mem(queue.index, isa::decode_mem_instruction(command.words[0]));
                break;
            case QueueKind::MxmLoad:
            case QueueKind::MxmCompute:
            case QueueKind::MxmOutput: {
                if (command.instruction_kind != InstructionKind::Mxm || command.word_count != 1) {
                    throw std::logic_error("MXM queue command does not carry one MXM instruction word");
                }
                const auto instruction = isa::decode_mxm_instruction(command.words[0]);
                validate_mxm_queue_opcode(queue.kind, queue.index, instruction);
                icu.enqueue_mxm(queue.index, instruction);
                break;
            }
            case QueueKind::Vxm:
                if (command.instruction_kind != InstructionKind::Vxm || command.word_count != 3) {
                    throw std::logic_error("VXM queue command does not carry three VXM instruction words");
                }
                icu.enqueue_vxm(queue.index, isa::decode_vxm_instruction(isa::EncodedVxmInstruction {command.words}));
                break;
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
