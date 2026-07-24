#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ftlpu::compiler::schedule {

using ScheduleTaskId = std::size_t;

enum class ScheduleTaskKind : std::uint8_t {
    MemRead,
    MxmLoad,
    MxmCompute,
    VxmCompute,
    SxmCompute,
    MemWrite,
    Barrier,
};

enum class ScheduleStage : std::uint8_t {
    Generic,
    Projection,
    Rope,
    Softmax,
    Pv,
    OutputProjection,
    FfnWeightLoad,
    FfnProjection,
    FfnSwish,
    FfnDownProjection,
};

struct ScheduleTaskDependency {
    ScheduleTaskId predecessor;
    int64_t latency;
};

struct ScheduleTask {
    std::string name;
    ScheduleTaskKind kind;
    ScheduleStage stage;
    int64_t earliest_cycle;
    int64_t duration;
    llvm::SmallVector<ResourceWindow, 8> resources;
    llvm::SmallVector<ScheduleTaskDependency, 4> dependencies;
};

struct ScheduledTask {
    int64_t cycle;
    int64_t end_cycle;
};

struct ScheduleAssignment {
    std::vector<ScheduledTask> tasks;

    const ScheduledTask& operator[](ScheduleTaskId id) const
    {
        return tasks.at(id);
    }
};

class SchedulePlan {
public:
    ScheduleTaskId addTask(std::string name, ScheduleTaskKind kind,
        ScheduleStage stage, int64_t earliestCycle, int64_t duration,
        llvm::ArrayRef<ResourceWindow> resources = {});

    mlir::LogicalResult addDependency(ScheduleTaskId predecessor,
        ScheduleTaskId successor, int64_t latency = 0);

    mlir::LogicalResult validate() const;

    mlir::FailureOr<ScheduleAssignment> schedule(
        ResourceScheduler& resources) const;

    const ScheduleTask& task(ScheduleTaskId id) const { return tasks_.at(id); }
    llvm::ArrayRef<ScheduleTask> tasks() const { return tasks_; }
    std::optional<ScheduleTaskId> findTask(llvm::StringRef name) const;
    std::size_t size() const { return tasks_.size(); }

private:
    std::vector<ScheduleTask> tasks_;
};

} // namespace ftlpu::compiler::schedule
