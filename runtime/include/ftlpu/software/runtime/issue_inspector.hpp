#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

struct LogicalIssue {
    std::size_t cycle{0};
    QueueCommand instruction{};
};

struct LogicalQueueTimeline {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::vector<LogicalIssue> issues{};
    std::size_t logical_nop_cycles{0};
};

struct LogicalIssueProgram {
    std::size_t max_cycle{0};
    std::vector<LogicalQueueTimeline> queues{};
    std::size_t functional_issues{0};
    std::size_t logical_nop_cycles{0};
};

// Expands native, control-compressed, macro, and coarse descriptor queues into
// the same sparse per-cycle functional issue representation. Every cycle not
// present in a queue timeline through max_cycle is a logical NOP/idle cycle.
LogicalIssueProgram inspect_logical_issues(const BinaryProgram& program);

struct LogicalIssueMismatch {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::size_t cycle{0};
    std::optional<QueueCommand> left{};
    std::optional<QueueCommand> right{};
    std::string reason{};
};

struct LogicalIssueComparison {
    bool equivalent{false};
    bool same_target{false};
    bool same_horizon{false};
    std::optional<LogicalIssueMismatch> first_mismatch{};
};

LogicalIssueComparison compare_logical_issues(
    const BinaryProgram& left, const BinaryProgram& right);

std::string describe_logical_instruction(const QueueCommand& instruction);

} // namespace ftlpu::software::runtime
