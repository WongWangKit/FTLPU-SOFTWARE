#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

#include "llvm/ADT/StringSet.h"

#include <algorithm>

namespace ftlpu::compiler::schedule {

ScheduleTaskId SchedulePlan::addTask(std::string name, ScheduleTaskKind kind,
    ScheduleStage stage, int64_t earliestCycle, int64_t duration,
    llvm::ArrayRef<ResourceWindow> resources)
{
    ScheduleTask task {std::move(name), kind, stage, earliestCycle, duration};
    task.resources.append(resources.begin(), resources.end());
    tasks_.push_back(std::move(task));
    return tasks_.size() - 1;
}

mlir::LogicalResult SchedulePlan::addDependency(ScheduleTaskId predecessor,
    ScheduleTaskId successor, int64_t latency)
{
    if (predecessor >= tasks_.size() || successor >= tasks_.size()
        || predecessor == successor || latency < 0)
        return mlir::failure();
    tasks_[successor].dependencies.push_back({predecessor, latency});
    return mlir::success();
}

mlir::LogicalResult SchedulePlan::validate() const
{
    llvm::StringSet<> names;
    for (ScheduleTaskId id = 0; id < tasks_.size(); ++id) {
        const ScheduleTask& task = tasks_[id];
        if (task.name.empty() || task.earliest_cycle < 0 || task.duration <= 0
            || !names.insert(task.name).second)
            return mlir::failure();
        for (const ScheduleTaskDependency& dependency : task.dependencies) {
            if (dependency.predecessor >= tasks_.size()
                || dependency.predecessor == id || dependency.latency < 0)
                return mlir::failure();
        }
    }

    std::vector<bool> visited(tasks_.size(), false);
    std::vector<bool> active(tasks_.size(), false);
    const auto visit = [&](auto&& self, ScheduleTaskId id) -> bool {
        if (active[id]) return false;
        if (visited[id]) return true;
        active[id] = true;
        for (const ScheduleTaskDependency& dependency : tasks_[id].dependencies)
            if (!self(self, dependency.predecessor)) return false;
        active[id] = false;
        visited[id] = true;
        return true;
    };
    for (ScheduleTaskId id = 0; id < tasks_.size(); ++id)
        if (!visit(visit, id)) return mlir::failure();
    return mlir::success();
}

mlir::FailureOr<ScheduleAssignment> SchedulePlan::schedule(
    ResourceScheduler& resources) const
{
    if (mlir::failed(validate())) return mlir::failure();
    ScheduleAssignment assignment;
    assignment.tasks.assign(tasks_.size(), {-1, -1});
    std::vector<bool> scheduled(tasks_.size(), false);
    std::size_t remaining = tasks_.size();
    while (remaining != 0) {
        bool progress = false;
        for (ScheduleTaskId id = 0; id < tasks_.size(); ++id) {
            if (scheduled[id]) continue;
            const ScheduleTask& current = tasks_[id];
            int64_t earliest = current.earliest_cycle;
            bool ready = true;
            for (const ScheduleTaskDependency& dependency : current.dependencies) {
                if (!scheduled[dependency.predecessor]) {
                    ready = false;
                    break;
                }
                earliest = std::max(earliest,
                    assignment[dependency.predecessor].end_cycle
                        + dependency.latency);
            }
            if (!ready) continue;
            const int64_t cycle = resources.reserve(earliest, current.resources);
            assignment.tasks[id] = {cycle, cycle + current.duration};
            scheduled[id] = true;
            --remaining;
            progress = true;
        }
        if (!progress) return mlir::failure();
    }
    return assignment;
}

std::optional<ScheduleTaskId> SchedulePlan::findTask(llvm::StringRef name) const
{
    for (ScheduleTaskId id = 0; id < tasks_.size(); ++id)
        if (tasks_[id].name == name) return id;
    return std::nullopt;
}

} // namespace ftlpu::compiler::schedule
