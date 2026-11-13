#include "ftlpu/software/runtime/issue_inspector.hpp"

#include "ftlpu/icu/distributed_queue.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace ftlpu::software::runtime {
namespace {

using QueueId = std::pair<QueueKind, std::size_t>;
struct IssueMap {
    std::map<std::size_t, QueueCommand> entries;
    std::size_t* remaining{nullptr};
};

constexpr isa::EncodedIcuCommand kInstructionCommand =
    static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);

std::uint64_t words64(const QueueCommand& command)
{
    return static_cast<std::uint64_t>(command.words[0])
        | (static_cast<std::uint64_t>(command.words[1]) << 32);
}

QueueCommand canonical_instruction(QueueCommand command,
    IcuInductionTarget target = IcuInductionTarget::None,
    std::int64_t delta = 0)
{
    if (is_macro_schedule_command(command))
        command.extension_words.erase(command.extension_words.begin(), command.extension_words.begin()+9);
    command.command = kInstructionCommand;
    switch (command.instruction_kind) {
    case InstructionKind::Mem: {
        auto instruction = isa::decode_mem_instruction(words64(command));
        instruction = detail::apply_macro_induction(
            instruction, target, delta);
        const auto encoded = isa::encode_mem_instruction(instruction);
        command.word_count = 2;
        command.words = {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0};
        command.extension_words.clear();
        return command;
    }
    case InstructionKind::Mxm: {
        auto instruction = isa::decode_mxm_instruction(words64(command));
        instruction = detail::apply_macro_induction(
            instruction, target, delta);
        const auto encoded = isa::encode_mxm_instruction(instruction);
        command.word_count = 2;
        command.words = {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0};
        command.extension_words.clear();
        return command;
    }
    case InstructionKind::MxmDequant:
        if (target != IcuInductionTarget::None || delta != 0)
            throw std::invalid_argument(
                "logical inspector cannot induce an MXM dequant instruction");
        command.word_count = 1;
        command.words[1] = command.words[2] = command.words[3] = 0;
        command.extension_words.clear();
        return command;
    case InstructionKind::Vxm:
        if (target != IcuInductionTarget::None || delta != 0)
            throw std::invalid_argument(
                "logical inspector cannot induce a VXM instruction");
        command.word_count = 3;
        command.words[3] = 0;
        command.extension_words.clear();
        return command;
    case InstructionKind::Sxm:
        if (target != IcuInductionTarget::None || delta != 0)
            throw std::invalid_argument(
                "logical inspector cannot induce an SXM instruction");
        command.word_count = 4;
        return command;
    case InstructionKind::None:
        break;
    }
    throw std::invalid_argument(
        "logical inspector found a functional command without an instruction kind");
}

QueueCommand repeat_instruction(
    const QueueCommand& base, std::int64_t stride, std::size_t index)
{
    switch (base.instruction_kind) {
    case InstructionKind::Mem: {
        auto instruction = isa::decode_mem_instruction(words64(base));
        instruction = detail::apply_icu_repeat_stride(
            instruction, stride, index);
        const auto encoded = isa::encode_mem_instruction(instruction);
        auto result = base;
        result.words = {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0};
        return result;
    }
    case InstructionKind::Mxm: {
        auto instruction = isa::decode_mxm_instruction(words64(base));
        instruction = detail::apply_icu_repeat_stride(
            instruction, stride, index);
        const auto encoded = isa::encode_mxm_instruction(instruction);
        auto result = base;
        result.words = {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0};
        return result;
    }
    default:
        if (stride != 0)
            throw std::invalid_argument(
                "logical inspector found a non-zero Repeat stride on an unsupported queue");
        return base;
    }
}

bool same_instruction(const QueueCommand& left, const QueueCommand& right)
{
    return left.instruction_kind == right.instruction_kind
        && left.word_count == right.word_count
        && left.words == right.words
        && left.extension_words == right.extension_words;
}

void check_cycle(const BinaryProgram& program, std::size_t cycle,
    QueueKind kind, std::size_t index)
{
    if (cycle > program.max_cycle)
        throw std::runtime_error("logical issue exceeds program max_cycle: queue="
            + std::string(queue_kind_name(kind)) + "["
            + std::to_string(index) + "] cycle=" + std::to_string(cycle)
            + " max_cycle=" + std::to_string(program.max_cycle));
}

void insert_issue(IssueMap& issues, const BinaryProgram& program,
    QueueKind kind, std::size_t index, std::size_t cycle,
    QueueCommand instruction)
{
    check_cycle(program, cycle, kind, index);
    if (!issues.remaining || *issues.remaining == 0)
        throw LogicalIssueAnalysisLimit();
    const auto [position, inserted] = issues.entries.emplace(
        cycle, std::move(instruction));
    if (inserted) --*issues.remaining;
    if (!inserted)
        throw std::runtime_error("multiple functional issues on one ICU queue cycle: queue="
            + std::string(queue_kind_name(kind)) + "["
            + std::to_string(index) + "] cycle=" + std::to_string(cycle));
}

std::size_t checked_cycle_add(std::size_t base, std::size_t delta)
{
    if (delta > std::numeric_limits<std::size_t>::max() - base)
        throw std::overflow_error("logical inspector cycle overflows size_t");
    return base + delta;
}

InstructionKind instruction_kind_for(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return InstructionKind::Mem;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: return InstructionKind::Mxm;
    case QueueKind::MxmDequant: return InstructionKind::MxmDequant;
    case QueueKind::Vxm: return InstructionKind::Vxm;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: return InstructionKind::Sxm;
    default: throw std::runtime_error(
        "logical inspector cannot map C2C transport to functional FU issues");
    }
}

QueueCommand native_instruction(const MemInstruction& instruction)
{
    const auto word = isa::encode_mem_instruction(instruction);
    return {kInstructionCommand, InstructionKind::Mem, 2,
        {static_cast<std::uint32_t>(word), static_cast<std::uint32_t>(word >> 32), 0, 0}};
}

QueueCommand native_instruction(const MxmControlInstruction& instruction)
{
    const auto word = isa::encode_mxm_instruction(instruction);
    return {kInstructionCommand, InstructionKind::Mxm, 2,
        {static_cast<std::uint32_t>(word), static_cast<std::uint32_t>(word >> 32), 0, 0}};
}

QueueCommand native_instruction(const MxmDequantInstruction& instruction)
{
    return {kInstructionCommand, InstructionKind::MxmDequant, 1,
        {static_cast<std::uint32_t>(isa::encode_mxm_dequant_instruction(instruction)), 0, 0, 0}};
}

QueueCommand native_instruction(const VxmCompactInstruction& instruction)
{
    return {kInstructionCommand, InstructionKind::Vxm, 3,
        {static_cast<std::uint32_t>(instruction.control),
            static_cast<std::uint32_t>(instruction.control >> 32),
            instruction.immediate_bits, 0}};
}

QueueCommand native_instruction(const SxmInstruction& instruction)
{
    QueueCommand command{kInstructionCommand, InstructionKind::Sxm, 4,
        {static_cast<std::uint32_t>(instruction.opcode),
            static_cast<std::uint32_t>(instruction.shift_source),
            static_cast<std::uint32_t>(instruction.shift_distance),
            (instruction.output_row == SxmInstruction::kAllOutputRows ? 0xffu
                : static_cast<std::uint32_t>(instruction.output_row))
                | ((instruction.input_row == SxmInstruction::kAllInputRows ? 0xffu
                    : static_cast<std::uint32_t>(instruction.input_row)) << 8)
                | ((instruction.output_tile == SxmInstruction::kAllOutputTiles ? 0xffu
                    : static_cast<std::uint32_t>(instruction.output_tile)) << 16)}};
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

QueueCommand canonical_queue_instruction(const QueueProgram& queue, const QueueCommand& command)
{
    // Native MEM/MXM physical words carry a two-bit ICU envelope; semantic
    // native words and VXM/SXM compact instructions do not carry that envelope.
    const auto physicalWidth = queue.kind == QueueKind::Mem ? 3u : 4u;
    if ((queue.kind == QueueKind::Mem || queue.kind == QueueKind::MxmLoad
            || queue.kind == QueueKind::MxmCompute || queue.kind == QueueKind::MxmDequant)
        && command.word_count == physicalWidth && command.extension_words.empty()) {
        if (command.instruction_kind != instruction_kind_for(queue.kind)
            || command.command != command.words[0] || (command.words[2] >> 2) != 0
            || command.words[3] != 0)
            throw std::runtime_error("logical inspector found an invalid native FU envelope");
        const auto bits = (words64(command) >> 2)
            | (static_cast<std::uint64_t>(command.words[2] & 3u) << 62);
        if (queue.kind == QueueKind::MxmDequant && (bits >> 16) != 0)
            throw std::runtime_error("logical inspector found reserved native dequant bits");
        QueueCommand native{kInstructionCommand, command.instruction_kind, 2,
            {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32), 0, 0}};
        return canonical_instruction(std::move(native));
    }
    return canonical_instruction(command);
}

void expand_queue(const BinaryProgram& program, const QueueProgram& queue,
    IssueMap& issues)
{
    std::size_t cursor = 0;
    std::size_t previousCycle = 0;
    std::optional<QueueCommand> previous;
    if (!queue.commands.empty() && (queue.kind == QueueKind::C2cDma
            || queue.kind == QueueKind::C2cRx || queue.kind == QueueKind::C2cTx))
        throw std::runtime_error("logical inspector cannot map C2C transport to functional FU issues");
    for (std::size_t commandIndex = 0; commandIndex < queue.commands.size(); ++commandIndex) {
        const auto& encodedCommand = queue.commands[commandIndex];
        if (is_mem_synchronized_raw_packet_header(encodedCommand))
            throw std::runtime_error("logical inspector cannot map runtime-dependent MEM synchronization");
        if (is_icu_control_raw_word_command(encodedCommand)) {
            const auto control = decode_icu_control_raw_word(encodedCommand);
            if (control.opcode == IcuControlOpcode::Nop) {
                cursor = checked_cycle_add(cursor, control.count);
                continue;
            }
            if (control.opcode != IcuControlOpcode::Repeat)
                throw std::runtime_error("logical inspector cannot map runtime-dependent ICU control");
            if (!previous)
                throw std::runtime_error("logical inspector found Repeat without a preceding instruction");
            for (std::size_t index = 1; index <= control.count; ++index)
                insert_issue(issues, program, queue.kind, queue.index,
                    checked_cycle_add(previousCycle, index * control.interval),
                    repeat_instruction(*previous, control.address_stride, index));
            if (control.count != 0)
                cursor = checked_cycle_add(previousCycle, control.count * control.interval + 1);
            continue;
        }
        if (is_macro_schedule_command(encodedCommand)) {
            const auto macro = decode_macro_schedule_command(encodedCommand);
            for (std::size_t outer = 0; outer < macro.outer_count; ++outer)
                for (std::size_t inner = 0; inner < macro.inner_count;
                     ++inner) {
                    const auto cycle = macro.start_cycle
                        + outer * macro.outer_interval
                        + inner * macro.inner_interval;
                    const auto delta = static_cast<std::int64_t>(outer)
                            * macro.outer_stride
                        + static_cast<std::int64_t>(inner)
                            * macro.inner_stride;
                    insert_issue(issues, program, queue.kind, queue.index,
                        cycle, canonical_instruction(encodedCommand,
                            macro.induction_target, delta));
                }
            continue;
        }


        const auto opcode =
            isa::decode_icu_command_opcode(encodedCommand.command);
        if (opcode == isa::IcuCommandOpcode::Nop) {
            cursor = checked_cycle_add(cursor, isa::decode_icu_nop_cycles(encodedCommand.command));
            continue;
        }
        if (opcode == isa::IcuCommandOpcode::Repeat) {
            if (!previous)
                throw std::runtime_error(
                    "logical inspector found Repeat without a preceding instruction");
            const auto repeat = isa::decode_icu_repeat(encodedCommand.command);
            for (std::size_t index = 1; index <= repeat.count; ++index)
                insert_issue(issues, program, queue.kind, queue.index,
                    previousCycle + index * repeat.interval,
                    repeat_instruction(*previous,
                        repeat.address_stride, index));
            if (repeat.count != 0)
                cursor = previousCycle + repeat.count * repeat.interval + 1;
            continue;
        }
        if (opcode != isa::IcuCommandOpcode::Instruction)
            throw std::runtime_error(
                "logical inspector found an unsupported ICU command");
        previous = canonical_queue_instruction(queue, encodedCommand);
        previousCycle = cursor;
        insert_issue(issues, program, queue.kind, queue.index,
            cursor, *previous);
        ++cursor;
    }
}

const LogicalQueueTimeline* find_queue(
    const LogicalIssueProgram& program, QueueId id)
{
    const auto found = std::lower_bound(program.queues.begin(),
        program.queues.end(), id, [](const auto& queue, QueueId key) {
            return std::tie(queue.kind, queue.index)
                < std::tie(key.first, key.second);
        });
    if (found == program.queues.end()
        || QueueId {found->kind, found->index} != id)
        return nullptr;
    return &*found;
}

} // namespace

std::optional<std::vector<MemIssueRun>> inspect_mem_macro_runs(
    const BinaryProgram& program, const QueueProgram& queue, std::size_t& work_left)
{
    if (queue.kind != QueueKind::Mem ||
        !std::ranges::all_of(queue.commands, [](const QueueCommand& command) {
            return command.instruction_kind == InstructionKind::Mem &&
                is_macro_schedule_command(command);
        }))
        return std::nullopt;

    std::vector<MemIssueRun> runs;
    for (const auto& command : queue.commands) {
        const auto m = decode_macro_schedule_command(command);
        const auto mem = isa::decode_mem_instruction(words64(command));
        if (mem.opcode != MemOpcode::Read && mem.opcode != MemOpcode::Write)
            throw std::invalid_argument("overlap proof requires direct MEM addresses");
        if (!m.inner_count || !m.outer_count || !m.inner_interval || !m.outer_interval)
            throw std::invalid_argument("invalid MEM macro iteration space");

        // Check the cycle bound first. The 32-bit cycle horizon and positive
        // intervals also bound (outer + inner), so signed address deltas fit
        // int64_t even for opposite-signed 32-bit induction strides.
        const auto last_outer = m.outer_count - 1;
        const auto last_inner = m.inner_count - 1;
        const auto last = checked_cycle_add(checked_cycle_add(m.start_cycle,
            last_outer * m.outer_interval), last_inner * m.inner_interval);
        check_cycle(program, last, queue.kind, queue.index);
        if (last > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("overlap proof exceeds finite cycle bound");
        const auto outer_delta = static_cast<std::int64_t>(last_outer) * m.outer_stride;
        const auto inner_delta = static_cast<std::int64_t>(last_inner) * m.inner_stride;
        // Every address lies between these two affine corner extrema. Use
        // the exact same induction and native encoder as pointwise inspection.
        for (const auto delta : {
                std::min<std::int64_t>(0, outer_delta) + std::min<std::int64_t>(0, inner_delta),
                std::max<std::int64_t>(0, outer_delta) + std::max<std::int64_t>(0, inner_delta)})
            (void)canonical_instruction(command, m.induction_target, delta);

        const auto append = [&](std::size_t first, std::size_t end) {
            if (work_left == 0) throw LogicalIssueAnalysisLimit();
            --work_left;
            runs.push_back({static_cast<std::uint32_t>(first),
                static_cast<std::uint32_t>(end), mem.stream});
        };
        for (std::size_t row = 0; row < m.outer_count; ++row) {
            const auto begin = m.start_cycle + row * m.outer_interval;
            if (m.inner_interval == 1) {
                append(begin, begin + last_inner);
            } else {
                for (std::size_t col = 0; col < m.inner_count; ++col) {
                    const auto cycle = begin + col * m.inner_interval;
                    append(cycle, cycle);
                }
            }
        }
    }
    // Rows or packets may be emitted out of time order. A queue has one issue
    // port even when colliding packets select different SRs or addresses.
    std::ranges::sort(runs, {}, &MemIssueRun::first);
    for (std::size_t i = 1; i < runs.size(); ++i)
        if (runs[i].first <= runs[i - 1].last)
            throw std::runtime_error("multiple functional issues on one MEM queue cycle: queue="
                + std::to_string(queue.index) + " cycle=" + std::to_string(runs[i].first));
    return runs;
}

LogicalIssueProgram inspect_logical_issues(const BinaryProgram& program, std::size_t max_issues)
{
    std::map<QueueId, IssueMap> expanded;
    for (const auto& queue : program.queues) {
        const QueueId id {queue.kind, queue.index};
        if (expanded.contains(id))
            throw std::runtime_error("binary contains a duplicate ICU queue: "
                + std::string(queue_kind_name(queue.kind)) + "["
                + std::to_string(queue.index) + "]");
        expanded[id].remaining = &max_issues;
        expand_queue(program, queue, expanded[id]);
    }

    LogicalIssueProgram result;
    result.max_cycle = program.max_cycle;
    result.queues.reserve(expanded.size());
    for (auto& [id, issues] : expanded) {
        LogicalQueueTimeline timeline;
        timeline.kind = id.first;
        timeline.index = id.second;
        timeline.issues.reserve(issues.entries.size());
        for (auto& [cycle, instruction] : issues.entries)
            timeline.issues.push_back(
                LogicalIssue {cycle, std::move(instruction)});
        const auto horizon = program.max_cycle
                == std::numeric_limits<std::size_t>::max()
            ? program.max_cycle
            : program.max_cycle + 1;
        timeline.logical_nop_cycles = horizon - timeline.issues.size();
        result.functional_issues += timeline.issues.size();
        result.logical_nop_cycles += timeline.logical_nop_cycles;
        result.queues.push_back(std::move(timeline));
    }
    return result;
}

LogicalIssueComparison compare_logical_issues(
    const BinaryProgram& left, const BinaryProgram& right)
{
    const auto leftIssues = inspect_logical_issues(left);
    const auto rightIssues = inspect_logical_issues(right);
    LogicalIssueComparison result;
    result.same_target = left.target_name == right.target_name
        && left.target_abi == right.target_abi;
    result.same_horizon = left.max_cycle == right.max_cycle;

    std::vector<QueueId> ids;
    for (const auto& queue : leftIssues.queues)
        ids.emplace_back(queue.kind, queue.index);
    for (const auto& queue : rightIssues.queues)
        ids.emplace_back(queue.kind, queue.index);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    for (const auto id : ids) {
        const auto* leftQueue = find_queue(leftIssues, id);
        const auto* rightQueue = find_queue(rightIssues, id);
        const std::vector<LogicalIssue> empty;
        const auto& lhs = leftQueue ? leftQueue->issues : empty;
        const auto& rhs = rightQueue ? rightQueue->issues : empty;
        std::size_t leftIndex = 0;
        std::size_t rightIndex = 0;
        while (leftIndex < lhs.size() || rightIndex < rhs.size()) {
            const auto leftCycle = leftIndex < lhs.size()
                ? lhs[leftIndex].cycle
                : std::numeric_limits<std::size_t>::max();
            const auto rightCycle = rightIndex < rhs.size()
                ? rhs[rightIndex].cycle
                : std::numeric_limits<std::size_t>::max();
            if (leftCycle == rightCycle
                && same_instruction(lhs[leftIndex].instruction,
                    rhs[rightIndex].instruction)) {
                ++leftIndex;
                ++rightIndex;
                continue;
            }
            LogicalIssueMismatch mismatch;
            mismatch.kind = id.first;
            mismatch.index = id.second;
            mismatch.cycle = std::min(leftCycle, rightCycle);
            if (leftCycle == mismatch.cycle)
                mismatch.left = lhs[leftIndex].instruction;
            if (rightCycle == mismatch.cycle)
                mismatch.right = rhs[rightIndex].instruction;
            mismatch.reason = leftCycle == rightCycle
                ? "functional instruction differs"
                : "functional issue versus logical NOP";
            if (!result.first_mismatch
                || std::tie(mismatch.cycle, mismatch.kind, mismatch.index)
                    < std::tie(result.first_mismatch->cycle,
                        result.first_mismatch->kind,
                        result.first_mismatch->index))
                result.first_mismatch = std::move(mismatch);
            break;
        }
    }

    if (!result.first_mismatch && !result.same_horizon) {
        LogicalIssueMismatch mismatch;
        mismatch.cycle = std::min(left.max_cycle, right.max_cycle) + 1;
        mismatch.reason = "program max_cycle differs";
        result.first_mismatch = std::move(mismatch);
    }
    if (!result.first_mismatch && !result.same_target) {
        LogicalIssueMismatch mismatch;
        mismatch.reason = "program target or target ABI differs";
        result.first_mismatch = std::move(mismatch);
    }
    result.equivalent = result.same_target && result.same_horizon
        && !result.first_mismatch;
    return result;
}

std::string describe_logical_instruction(const QueueCommand& instruction)
{
    std::ostringstream output;
    output << "kind=" << static_cast<unsigned>(instruction.instruction_kind)
           << " words=";
    output << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < instruction.word_count; ++index) {
        if (index != 0) output << ':';
        output << "0x" << std::setw(8) << instruction.words[index];
    }
    if (!instruction.extension_words.empty()) {
        output << " extension=";
        for (std::size_t index = 0;
             index < instruction.extension_words.size(); ++index) {
            if (index != 0) output << ':';
            output << "0x" << std::setw(8)
                   << instruction.extension_words[index];
        }
    }
    return output.str();
}

} // namespace ftlpu::software::runtime
