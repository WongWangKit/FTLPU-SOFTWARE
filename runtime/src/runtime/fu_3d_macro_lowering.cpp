#include "ftlpu/software/runtime/fu_3d_macro_lowering.hpp"

#include "ftlpu/icu/active_macro_context.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

constexpr auto kInstruction =
    static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);

std::size_t checked_add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Macro coarse-domain cycle overflow");
    return a + b;
}

std::size_t checked_multiply(std::size_t a, std::size_t b)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Macro coarse-domain count overflow");
    return a * b;
}

std::int64_t checked_operand_add(std::int64_t a, std::int64_t b)
{
    if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b)
        || (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))
        throw std::overflow_error("Macro coarse-domain operand overflow");
    return a + b;
}

std::int64_t checked_operand_multiply(std::int64_t a, std::size_t b)
{
    if (b > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("Macro coarse-domain operand count overflow");
    const auto n = static_cast<std::int64_t>(b);
    if ((a > 0 && n > std::numeric_limits<std::int64_t>::max() / a)
        || (a < 0 && a != -1
            && n > std::numeric_limits<std::int64_t>::min() / a))
        throw std::overflow_error("Macro coarse-domain operand multiply overflow");
    return a * n;
}

InstructionKind instruction_kind(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return InstructionKind::Mem;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: return InstructionKind::Mxm;
    case QueueKind::MxmDequant: return InstructionKind::MxmDequant;
    default:
        throw std::invalid_argument("Macro coarse lowering requires a MEM/MXM queue");
    }
}

std::uint64_t native_bits(const QueueCommand& command)
{
    return static_cast<std::uint64_t>(command.words[0])
        | (static_cast<std::uint64_t>(command.words[1]) << 32);
}

QueueCommand native_command(const MemInstruction& instruction)
{
    const auto bits = isa::encode_mem_instruction(instruction);
    return {kInstruction, InstructionKind::Mem, 2,
        {static_cast<std::uint32_t>(bits),
            static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

QueueCommand native_command(const MxmControlInstruction& instruction)
{
    const auto bits = isa::encode_mxm_instruction(instruction);
    return {kInstruction, InstructionKind::Mxm, 2,
        {static_cast<std::uint32_t>(bits),
            static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

QueueCommand native_command(const MxmDequantInstruction& instruction)
{
    return {kInstruction, InstructionKind::MxmDequant, 1,
        {static_cast<std::uint32_t>(
             isa::encode_mxm_dequant_instruction(instruction)), 0, 0, 0}};
}

QueueCommand induce(const QueueCommand& base, IcuInductionTarget target,
    std::int64_t delta)
{
    switch (base.instruction_kind) {
    case InstructionKind::Mem:
        return native_command(detail::apply_icu_repeat_2d_stride(
            isa::decode_mem_instruction(native_bits(base)), target, delta));
    case InstructionKind::Mxm:
        return native_command(detail::apply_icu_repeat_2d_stride(
            isa::decode_mxm_instruction(native_bits(base)), target, delta));
    case InstructionKind::MxmDequant:
        if (target == IcuInductionTarget::None && delta == 0) return base;
        break;
    default: break;
    }
    throw std::invalid_argument("invalid Macro coarse-domain induction target");
}

void validate_context(const QueueCommand& native,
    const IcuMacroSchedule& schedule)
{
    if (native.instruction_kind == InstructionKind::Mem) {
        static_cast<void>(IcuActiveMacroContextCodec256<MemInstruction>::
            pack_initial(schedule, isa::decode_mem_instruction(native_bits(native))));
    } else if (native.instruction_kind == InstructionKind::Mxm) {
        static_cast<void>(IcuActiveMacroContextCodec256<MxmControlInstruction>::
            pack_initial(schedule, isa::decode_mxm_instruction(native_bits(native))));
    } else {
        static_cast<void>(IcuActiveMacroContextCodec256<MxmDequantInstruction>::
            pack_initial(schedule, isa::decode_mxm_dequant_instruction(native.words[0])));
    }
}

template <typename Packet>
Packet read_packet(const QueueProgram& queue, std::size_t pc)
{
    const auto count = fu_3d_raw_packet_word_count(queue.kind, queue.commands.at(pc));
    if (count > Packet::kWordCount || count > queue.commands.size() - pc)
        throw std::invalid_argument("truncated Macro source FU packet");
    Packet packet {};
    for (std::size_t word = 0; word < count; ++word) {
        const auto& command = queue.commands[pc + word];
        if (command.instruction_kind != instruction_kind(queue.kind)
            || command.word_count != Packet::kLanesPerWord
            || !command.extension_words.empty()
            || command.command != command.words[0]
            || isa::decode_icu_command_opcode(command.command)
                != isa::IcuCommandOpcode::Extended
            || ((command.words[0] >> 2) & 3U) != word)
            throw std::invalid_argument("malformed Macro source FU packet word");
        for (std::size_t lane = 0; lane < Packet::kLanesPerWord; ++lane)
            packet.words[word].lanes[lane] = command.words[lane];
        for (std::size_t lane = Packet::kLanesPerWord;
             lane < command.words.size(); ++lane)
            if (command.words[lane] != 0)
                throw std::invalid_argument("nonzero padding in Macro source packet");
    }
    return packet;
}

struct Axis {
    std::size_t count;
    std::size_t cycle_stride;
    std::int64_t operand_stride;
};

struct Domain {
    std::size_t start;
    QueueCommand native;
    IcuInductionTarget target;
    std::vector<Axis> axes;
};

struct Box {
    IcuCoordinate3D origin {};
    std::array<std::size_t, 3> counts {1, 1, 1};
    std::array<std::size_t, 3> steps {1, 1, 1};
};

std::size_t packet_end(const IcuLoop3D& loop, std::size_t cursor)
{
    auto end = checked_add(cursor, loop.wait_cycle);
    for (std::size_t axis = 0; axis < 3; ++axis)
        end = checked_add(end,
            checked_multiply(loop.counts[axis] - 1, loop.cycle_strides[axis]));
    return checked_add(end, 1);
}

std::vector<Box> split_boxes(const IcuLoop3D& loop,
    MxmIcuBufferMode bufferMode,
    std::size_t terminalDimension = IcuLoop3D::kDimensions)
{
    Box initial;
    initial.counts = loop.counts;
    std::vector<Box> boxes {initial};
    if (terminalDimension < 3 && loop.counts[terminalDimension] > 1) {
        boxes[0].counts[terminalDimension] -= 1;
        auto terminal = initial;
        terminal.origin.index[terminalDimension] = loop.counts[terminalDimension] - 1;
        terminal.counts[terminalDimension] = 1;
        boxes.push_back(terminal);
    }
    if (bufferMode == MxmIcuBufferMode::Fixed) return boxes;
    const auto axis = static_cast<std::size_t>(bufferMode) - 1;
    if (axis >= 3) throw std::invalid_argument("invalid MXM buffer parity dimension");
    std::vector<Box> parity;
    for (const auto& box : boxes) {
        for (std::size_t residue = 0;
             residue < std::min<std::size_t>(box.counts[axis], 2); ++residue) {
            auto part = box;
            part.origin.index[axis] += residue * box.steps[axis];
            part.counts[axis] = (box.counts[axis] - residue + 1) / 2;
            part.steps[axis] *= 2;
            parity.push_back(part);
        }
    }
    return parity;
}

template <typename Instruction, typename Native>
Domain box_domain(const Instruction& instruction, const Box& box,
    std::size_t cursor, const std::array<std::int64_t, 3>& operandStrides,
    IcuInductionTarget target, Native expand)
{
    Domain domain {checked_add(cursor,
                       detail::icu_loop_3d_issue_cycle(instruction.loop, box.origin)),
        native_command(expand(instruction, box.origin)), target, {}};
    for (std::size_t axis = 0; axis < 3; ++axis)
        domain.axes.push_back({box.counts[axis],
            checked_multiply(instruction.loop.cycle_strides[axis], box.steps[axis]),
            checked_operand_multiply(operandStrides[axis], box.steps[axis])});
    return domain;
}

void flatten_axes(Domain& domain)
{
    std::erase_if(domain.axes, [](const Axis& axis) { return axis.count == 1; });
    std::sort(domain.axes.begin(), domain.axes.end(),
        [](const Axis& a, const Axis& b) { return a.cycle_stride < b.cycle_stride; });
    // Adjacent affine axes can be flattened algebraically. Keep counts within
    // the physical Active Context field instead of creating oversized loops.
    for (std::size_t i = 1; i < domain.axes.size();) {
        auto& inner = domain.axes[i - 1];
        const auto outer = domain.axes[i];
        if (inner.count <= 65536 / outer.count
            && outer.cycle_stride == checked_multiply(inner.count, inner.cycle_stride)
            && outer.operand_stride
                == checked_operand_multiply(inner.operand_stride, inner.count)) {
            inner.count *= outer.count;
            domain.axes.erase(domain.axes.begin() + i);
        } else {
            ++i;
        }
    }
}

struct Pending {
    std::size_t source;
    QueueCommand command;
};

IcuMacroSchedule make_schedule(const Domain& domain,
    const std::vector<std::size_t>& keep)
{
    IcuMacroSchedule schedule;
    schedule.start_cycle = domain.start;
    schedule.induction_target = domain.target;
    if (!keep.empty()) {
        const auto& axis = domain.axes[keep[0]];
        schedule.inner_count = axis.count;
        schedule.inner_interval = axis.cycle_stride;
        schedule.inner_stride = axis.operand_stride;
    }
    if (keep.size() > 1) {
        const auto& axis = domain.axes[keep[1]];
        schedule.outer_count = axis.count;
        schedule.outer_interval = axis.cycle_stride;
        schedule.outer_stride = axis.operand_stride;
    }
    return schedule;
}

void emit_domain(Domain domain, std::size_t source,
    std::size_t budget, std::vector<Pending>& output)
{
    flatten_axes(domain);
    struct Plan { std::vector<std::size_t> keep; std::size_t entries; };
    std::vector<Plan> plans;
    const auto addPlan = [&](std::vector<std::size_t> keep) {
        auto entries = std::size_t {1};
        for (std::size_t axis = 0; axis < domain.axes.size(); ++axis) {
            if (std::find(keep.begin(), keep.end(), axis) == keep.end()) {
                if (domain.axes[axis].count > budget / entries) return;
                entries *= domain.axes[axis].count;
            }
        }
        plans.push_back({std::move(keep), entries});
    };
    for (std::size_t a = 0; a < domain.axes.size(); ++a)
        for (std::size_t b = a + 1; b < domain.axes.size(); ++b) addPlan({a, b});
    for (std::size_t a = 0; a < domain.axes.size(); ++a) addPlan({a});
    addPlan({});
    std::stable_sort(plans.begin(), plans.end(),
        [](const Plan& a, const Plan& b) { return a.entries < b.entries; });

    const Plan* selected = nullptr;
    for (const auto& plan : plans) {
        try {
            validate_context(domain.native, make_schedule(domain, plan.keep));
            selected = &plan;
            break;
        } catch (const std::logic_error&) {
            // Another axis pair may fit the real row-transition field even
            // when the largest-work pair does not. No work points are expanded.
        }
    }
    if (!selected)
        throw std::invalid_argument(
            "Macro coarse domain exceeds the descriptor budget or Active Context fields");
    std::vector<std::size_t> drop;
    for (std::size_t axis = 0; axis < domain.axes.size(); ++axis)
        if (std::find(selected->keep.begin(), selected->keep.end(), axis)
            == selected->keep.end()) drop.push_back(axis);
    auto schedule = make_schedule(domain, selected->keep);
    const auto emit = [&](auto&& self, std::size_t position,
                          std::size_t cycle, std::int64_t operand) -> void {
        if (position == drop.size()) {
            schedule.start_cycle = cycle;
            auto native = induce(domain.native, domain.target, operand);
            validate_context(native, schedule);
            output.push_back({source, encode_macro_schedule_command(std::move(native), schedule)});
            return;
        }
        const auto& axis = domain.axes[drop[position]];
        for (std::size_t index = 0; index < axis.count; ++index)
            self(self, position + 1,
                checked_add(cycle, checked_multiply(index, axis.cycle_stride)),
                checked_operand_add(operand, checked_operand_multiply(axis.operand_stride, index)));
    };
    emit(emit, 0, domain.start, 0);
}

QueueCommand decode_native(const QueueCommand& command, QueueKind kind)
{
    if (command.instruction_kind != instruction_kind(kind)
        || !command.extension_words.empty() || command.word_count == 0)
        throw std::invalid_argument("malformed native Macro source instruction");
    auto bits = native_bits(command);
    if (command.word_count > 2) {
        const auto width = kind == QueueKind::Mem ? 3U : 4U;
        if (command.word_count != width || command.command != command.words[0]
            || (command.words[2] >> 2) != 0 || command.words[3] != 0)
            throw std::invalid_argument("unsupported native Macro source word envelope");
        bits = (bits >> 2) | (static_cast<std::uint64_t>(command.words[2] & 3U) << 62);
    } else if (command.command != kInstruction
        || command.words[2] != 0 || command.words[3] != 0
        || (command.word_count == 1 && command.words[1] != 0)) {
        throw std::invalid_argument("noncanonical native Macro source instruction");
    }
    if (kind == QueueKind::Mem) return native_command(isa::decode_mem_instruction(bits));
    if (kind == QueueKind::MxmDequant) {
        if ((bits >> 16) != 0)
            throw std::invalid_argument("invalid native dequant scale bits");
        return native_command(isa::decode_mxm_dequant_instruction(bits));
    }
    auto instruction = isa::decode_mxm_instruction(bits);
    if (instruction.opcode == MxmControlOpcode::Decode)
        throw std::invalid_argument("native4 decode is not supported by coarse Macro lowering");
    if ((kind == QueueKind::MxmLoad && instruction.opcode != MxmControlOpcode::IW)
        || (kind == QueueKind::MxmCompute
            && instruction.opcode != MxmControlOpcode::Compute
            && instruction.opcode != MxmControlOpcode::AccumulatorRead))
        throw std::invalid_argument("native opcode does not match the Macro queue role");
    return native_command(instruction);
}

} // namespace

Fu3dMacroLoweringResult lower_fu_3d_queue_to_macro(
    const QueueProgram& queue, std::size_t max_descriptors)
{
    static_cast<void>(instruction_kind(queue.kind));
    Fu3dMacroLoweringResult result;
    result.queue.kind = queue.kind;
    result.queue.index = queue.index;
    result.command_indices.resize(queue.commands.size());
    std::vector<Pending> output;
    std::size_t cursor = 0;
    const auto emit = [&](Domain domain, std::size_t pc) {
        if (output.size() >= max_descriptors)
            throw std::invalid_argument("Macro coarse lowering descriptor budget exceeded");
        emit_domain(std::move(domain), pc, max_descriptors - output.size(), output);
    };
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (is_mem_synchronized_raw_packet_header(command))
            throw std::invalid_argument("synchronized MEM transfers require the native ICU path");
        if (is_icu_control_raw_word_command(command)) {
            const auto control = decode_icu_control_raw_word(command);
            if (control.opcode != IcuControlOpcode::Nop)
                throw std::invalid_argument("event/SYNC control requires the native ICU path");
            cursor = checked_add(cursor, control.count);
            ++pc;
            continue;
        }
        const auto opcode = isa::decode_icu_command_opcode(command.command);
        if (opcode == isa::IcuCommandOpcode::Nop) {
            if (command.instruction_kind != InstructionKind::None
                || command.word_count != 0 || !command.extension_words.empty())
                throw std::invalid_argument("malformed Macro source NOP");
            cursor = checked_add(cursor, isa::decode_icu_nop_cycles(command.command));
            ++pc;
            continue;
        }
        if (is_fu_3d_raw_packet_header(command)) {
            if (is_mem_write_read_2d_raw_packet_header(command))
                throw std::invalid_argument("WRITE_READ_2D is not supported by coarse Macro lowering");
            IcuLoop3D loop;
            if (queue.kind == QueueKind::Mem) {
                const auto instruction = isa::decode_mem_icu_3d_instruction(
                    read_packet<isa::EncodedMemIcu3DPacket>(queue, pc));
                loop = instruction.loop;
                const auto& address = instruction.address;
                const auto group = address.outer_group_size;
                const auto addMem = [&](std::size_t outerBase,
                                        std::size_t remainderCount,
                                        std::size_t groupCount) {
                    const IcuCoordinate3D origin {{0, 0, outerBase}};
                    Domain domain {checked_add(cursor,
                                       detail::icu_loop_3d_issue_cycle(loop, origin)),
                        native_command(detail::expand_mem_icu_instruction(instruction, origin)),
                        IcuInductionTarget::MemAddress,
                        {{loop.counts[0], loop.cycle_strides[0], address.inner_stride},
                            {loop.counts[1], loop.cycle_strides[1], address.middle_stride},
                            {remainderCount, loop.cycle_strides[2], address.outer_inner_stride},
                            {groupCount, checked_multiply(group, loop.cycle_strides[2]), address.outer_group_stride}}};
                    emit(std::move(domain), pc);
                };
                if (group == 1 || address.outer_group_stride
                        == checked_operand_multiply(address.outer_inner_stride, group)) {
                    Box box;
                    box.counts = loop.counts;
                    emit(box_domain(instruction, box, cursor,
                             {address.inner_stride, address.middle_stride,
                                 group == 1 ? address.outer_group_stride : address.outer_inner_stride},
                             IcuInductionTarget::MemAddress, detail::expand_mem_icu_instruction), pc);
                } else {
                    const auto groups = loop.counts[2] / group;
                    const auto tail = loop.counts[2] % group;
                    if (groups != 0) addMem(0, group, groups);
                    if (tail != 0) addMem(groups * group, tail, 1);
                }
            } else if (queue.kind == QueueKind::MxmLoad) {
                const auto instruction = isa::decode_mxm_load_icu_3d_instruction(
                    read_packet<isa::EncodedMxmLoadIcu3DPacket>(queue, pc));
                if (instruction.opcode != MxmLoadIcuOpcode::Load3D)
                    throw std::invalid_argument("native4 decode is not supported by coarse Macro lowering");
                loop = instruction.loop;
                for (const auto& box : split_boxes(loop, instruction.weight_buffer_mode))
                    emit(box_domain(instruction, box, cursor, instruction.weight_column_strides,
                             IcuInductionTarget::MxmWeightColumn, detail::expand_mxm_load_icu_instruction), pc);
            } else if (queue.kind == QueueKind::MxmCompute) {
                const auto instruction = isa::decode_mxm_compute_icu_3d_instruction(
                    read_packet<isa::EncodedMxmComputeIcu3DPacket>(queue, pc));
                if (instruction.opcode == MxmComputeIcuOpcode::DecodeStreamCompute3D)
                    throw std::invalid_argument("native4 decode is not supported by coarse Macro lowering");
                loop = instruction.loop;
                const auto terminal = instruction.opcode == MxmComputeIcuOpcode::Compute3D
                    ? instruction.terminal_dimension : IcuLoop3D::kDimensions;
                for (const auto& box : split_boxes(loop, instruction.weight_buffer_mode, terminal))
                    emit(box_domain(instruction, box, cursor, instruction.accumulator_address_strides,
                             IcuInductionTarget::MxmAccumulatorAddress, detail::expand_mxm_compute_icu_instruction), pc);
            } else {
                const auto instruction = isa::decode_mxm_dequant_icu_3d_instruction(
                    read_packet<isa::EncodedMxmDequantIcu3DPacket>(queue, pc));
                loop = instruction.loop;
                Box box;
                box.counts = loop.counts;
                emit(box_domain(instruction, box, cursor, {0, 0, 0}, IcuInductionTarget::None,
                         [](const auto& value, const auto&) { return value.instruction; }), pc);
            }
            cursor = packet_end(loop, cursor);
            pc += fu_3d_raw_packet_word_count(queue.kind, command);
            continue;
        }
        if (opcode != isa::IcuCommandOpcode::Instruction)
            throw std::invalid_argument("unsupported control or orphan FU packet in Macro source queue");
        emit({cursor, decode_native(command, queue.kind), IcuInductionTarget::None, {}}, pc);
        cursor = checked_add(cursor, 1);
        ++pc;
    }
    std::stable_sort(output.begin(), output.end(), [](const Pending& a, const Pending& b) {
        return decode_macro_schedule_command(a.command).start_cycle
            < decode_macro_schedule_command(b.command).start_cycle;
    });
    result.queue.commands.reserve(output.size());
    for (auto& pending : output) {
        result.command_indices[pending.source].push_back(result.queue.commands.size());
        result.queue.commands.push_back(std::move(pending.command));
    }
    return result;
}

} // namespace ftlpu::software::runtime
