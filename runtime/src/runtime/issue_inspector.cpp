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
using IssueMap = std::map<std::size_t, QueueCommand>;

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
    command.command = kInstructionCommand;
    switch (command.instruction_kind) {
    case InstructionKind::Mem: {
        auto instruction = isa::decode_mem_instruction(words64(command));
        instruction = detail::apply_icu_repeat_2d_stride(
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
        instruction = detail::apply_icu_repeat_2d_stride(
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
    const auto [position, inserted] = issues.emplace(
        cycle, std::move(instruction));
    if (!inserted)
        throw std::runtime_error("multiple functional issues on one ICU queue cycle: queue="
            + std::string(queue_kind_name(kind)) + "["
            + std::to_string(index) + "] cycle=" + std::to_string(cycle));
}

template <typename Emit>
void for_each_stream_point(const IcuStreamNdSchedule& schedule, Emit emit)
{
    const auto depthCount = schedule.rank > 2
        ? schedule.counts[2] : std::size_t {1};
    const auto outerCount = schedule.rank > 1
        ? schedule.counts[1] : std::size_t {1};
    for (std::size_t depth = 0; depth < depthCount; ++depth)
        for (std::size_t outer = 0; outer < outerCount; ++outer)
            for (std::size_t inner = 0; inner < schedule.counts[0]; ++inner) {
                const std::array<std::size_t, 3> coordinates {
                    inner, outer, depth};
                std::size_t cycle = schedule.start_cycle;
                std::int64_t delta = 0;
                for (std::size_t dimension = 0;
                     dimension < schedule.rank; ++dimension) {
                    cycle += coordinates[dimension]
                        * schedule.cycle_strides[dimension];
                    delta += static_cast<std::int64_t>(
                                 coordinates[dimension])
                        * schedule.operand_strides[dimension];
                }
                emit(cycle, delta);
            }
}

void expand_queue(const BinaryProgram& program, const QueueProgram& queue,
    IssueMap& issues)
{
    std::size_t cursor = 0;
    std::size_t previousCycle = 0;
    std::optional<QueueCommand> previous;
    for (const auto& encodedCommand : queue.commands) {
        if (is_mem_slice_program_command(encodedCommand)) {
            const auto descriptor =
                decode_mem_slice_program_command(encodedCommand);
            for (const auto& body : descriptor.body) {
                const auto encoded = isa::encode_mem_instruction(
                    body.instruction);
                QueueCommand native {kInstructionCommand,
                    InstructionKind::Mem, 2,
                    {static_cast<std::uint32_t>(encoded),
                        static_cast<std::uint32_t>(encoded >> 32), 0, 0}};
                auto schedule = descriptor.schedule;
                schedule.start_cycle += body.cycle_offset;
                schedule.operand_strides = body.operand_strides;
                schedule.induction_target = IcuInductionTarget::MemAddress;
                for_each_stream_point(schedule,
                    [&](std::size_t cycle, std::int64_t delta) {
                        insert_issue(issues, program, queue.kind, queue.index,
                            cycle, canonical_instruction(
                                native, schedule.induction_target, delta));
                    });
            }
            continue;
        }

        if (is_vxm_stream_nd_command(encodedCommand)) {
            const auto descriptor =
                decode_vxm_stream_nd_command(encodedCommand);
            for_each_stream_point(descriptor.schedule,
                [&](std::size_t cycle, std::int64_t delta) {
                    insert_issue(issues, program, queue.kind, queue.index,
                        cycle, canonical_instruction(descriptor.instruction,
                            descriptor.schedule.induction_target, delta));
                });
            continue;
        }
        if (is_sxm_tile_program_command(encodedCommand)) {
            const auto descriptor =
                decode_sxm_tile_program_command(encodedCommand);
            for_each_stream_point(descriptor.schedule,
                [&](std::size_t cycle, std::int64_t delta) {
                    insert_issue(issues, program, queue.kind, queue.index,
                        cycle, canonical_instruction(descriptor.instruction,
                            descriptor.schedule.induction_target, delta));
                });
            continue;
        }
        if (is_mem_stream_nd_command(encodedCommand)
            || is_mxm_stream_nd_command(encodedCommand)) {
            const auto schedule = is_mem_stream_nd_command(encodedCommand)
                ? decode_mem_stream_nd_command(encodedCommand)
                : decode_mxm_stream_nd_command(encodedCommand);
            for_each_stream_point(schedule,
                [&](std::size_t cycle, std::int64_t delta) {
                    insert_issue(issues, program, queue.kind, queue.index,
                        cycle, canonical_instruction(encodedCommand,
                            schedule.induction_target, delta));
                });
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
        if (is_repeat_2d_command(encodedCommand)) {
            if (!previous)
                throw std::runtime_error(
                    "logical inspector found Repeat2D without a preceding instruction");
            const auto repeat = decode_repeat_2d_command(encodedCommand);
            for (std::size_t outer = 0; outer < repeat.outer_count; ++outer)
                for (std::size_t inner = 0; inner < repeat.inner_count;
                     ++inner) {
                    if (outer == 0 && inner == 0) continue;
                    const auto cycle = previousCycle
                        + outer * repeat.outer_interval
                        + inner * repeat.inner_interval;
                    const auto delta = static_cast<std::int64_t>(outer)
                            * repeat.outer_stride
                        + static_cast<std::int64_t>(inner)
                            * repeat.inner_stride;
                    insert_issue(issues, program, queue.kind, queue.index,
                        cycle, canonical_instruction(*previous,
                            repeat.induction_target, delta));
                }
            cursor = previousCycle
                + (repeat.outer_count - 1) * repeat.outer_interval
                + (repeat.inner_count - 1) * repeat.inner_interval + 1;
            continue;
        }

        const auto opcode =
            isa::decode_icu_command_opcode(encodedCommand.command);
        if (opcode == isa::IcuCommandOpcode::Nop) {
            cursor += isa::decode_icu_nop_cycles(encodedCommand.command);
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
        previous = canonical_instruction(encodedCommand);
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

LogicalIssueProgram inspect_logical_issues(const BinaryProgram& program)
{
    std::map<QueueId, IssueMap> expanded;
    for (const auto& queue : program.queues) {
        const QueueId id {queue.kind, queue.index};
        if (expanded.contains(id))
            throw std::runtime_error("binary contains a duplicate ICU queue: "
                + std::string(queue_kind_name(queue.kind)) + "["
                + std::to_string(queue.index) + "]");
        expand_queue(program, queue, expanded[id]);
    }

    LogicalIssueProgram result;
    result.max_cycle = program.max_cycle;
    result.queues.reserve(expanded.size());
    for (auto& [id, issues] : expanded) {
        LogicalQueueTimeline timeline;
        timeline.kind = id.first;
        timeline.index = id.second;
        timeline.issues.reserve(issues.size());
        for (auto& [cycle, instruction] : issues)
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
