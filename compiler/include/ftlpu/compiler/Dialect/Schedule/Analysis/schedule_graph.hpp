#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ftlpu::compiler::schedule {

using ScheduleNodeId = std::size_t;

struct ScheduleDependency {
    ScheduleNodeId predecessor;
    int64_t latency;
};

struct ScheduleNode {
    std::string name;
    int64_t earliest_cycle;
    int64_t duration;
    llvm::SmallVector<ResourceWindow, 8> resources;
    llvm::SmallVector<ScheduleDependency, 4> dependencies;
};

struct ScheduledNode {
    int64_t cycle;
    int64_t end_cycle;
};

// A small target-independent list scheduler. Nodes describe relative resource
// windows, while dependencies describe producer-to-consumer latency.
class ScheduleGraph {
public:
    ScheduleNodeId add_node(std::string name, int64_t earliest_cycle,
        int64_t duration, llvm::ArrayRef<ResourceWindow> resources);
    mlir::LogicalResult add_dependency(ScheduleNodeId predecessor,
        ScheduleNodeId successor, int64_t latency = 0);
    mlir::FailureOr<std::vector<ScheduledNode>> schedule(
        ResourceScheduler& resources) const;

    const ScheduleNode& node(ScheduleNodeId id) const { return nodes_.at(id); }
    std::size_t size() const { return nodes_.size(); }

private:
    std::vector<ScheduleNode> nodes_;
};

} // namespace ftlpu::compiler::schedule
