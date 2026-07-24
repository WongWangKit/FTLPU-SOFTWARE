#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_graph.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleNodeId ScheduleGraph::add_node(std::string name, int64_t earliest_cycle,
    int64_t duration, llvm::ArrayRef<ResourceWindow> resources)
{
    ScheduleNode node;
    node.name = std::move(name);
    node.earliest_cycle = earliest_cycle;
    node.duration = duration;
    node.resources.append(resources.begin(), resources.end());
    nodes_.push_back(std::move(node));
    return nodes_.size() - 1;
}

mlir::LogicalResult ScheduleGraph::add_dependency(ScheduleNodeId predecessor,
    ScheduleNodeId successor, int64_t latency)
{
    if (predecessor >= nodes_.size() || successor >= nodes_.size()
        || predecessor == successor || latency < 0)
        return mlir::failure();
    nodes_[successor].dependencies.push_back({predecessor, latency});
    return mlir::success();
}

mlir::FailureOr<std::vector<ScheduledNode>> ScheduleGraph::schedule(
    ResourceScheduler& resources) const
{
    std::vector<ScheduledNode> result(nodes_.size(), {-1, -1});
    std::vector<bool> scheduled(nodes_.size(), false);
    std::size_t remaining = nodes_.size();
    while (remaining != 0) {
        bool progress = false;
        for (ScheduleNodeId id = 0; id < nodes_.size(); ++id) {
            if (scheduled[id]) continue;
            const ScheduleNode& current = nodes_[id];
            int64_t earliest = current.earliest_cycle;
            bool ready = true;
            for (const ScheduleDependency& dependency : current.dependencies) {
                if (!scheduled[dependency.predecessor]) {
                    ready = false;
                    break;
                }
                earliest = std::max(earliest,
                    result[dependency.predecessor].end_cycle + dependency.latency);
            }
            if (!ready || current.duration <= 0) continue;
            const int64_t cycle = resources.reserve(earliest, current.resources);
            result[id] = {cycle, cycle + current.duration};
            scheduled[id] = true;
            --remaining;
            progress = true;
        }
        if (!progress) return mlir::failure();
    }
    return result;
}

} // namespace ftlpu::compiler::schedule
