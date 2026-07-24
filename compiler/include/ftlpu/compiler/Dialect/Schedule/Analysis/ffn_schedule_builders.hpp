#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

namespace ftlpu::compiler::schedule {

struct FfnScheduleShape {
    int64_t m;
    int64_t k;
    int64_t hidden;
    int64_t n;
};

struct FfnStageTaskIds {
    ScheduleTaskId weight_load;
    ScheduleTaskId projection;
    ScheduleTaskId swish;
    ScheduleTaskId down_projection;
};

struct FfnTaskPlan {
    SchedulePlan tasks;
    FfnStageTaskIds task_ids;
};

class FfnWeightLoadScheduleBuilder {
public:
    ScheduleTaskId append(SchedulePlan& plan, FfnScheduleShape shape,
        const target::LPUTargetModel& target) const;
};

class FfnProjectionScheduleBuilder {
public:
    ScheduleTaskId append(SchedulePlan& plan, FfnScheduleShape shape,
        ScheduleTaskId weightLoad,
        const target::LPUTargetModel& target) const;
};

class FfnSwishScheduleBuilder {
public:
    ScheduleTaskId append(SchedulePlan& plan, FfnScheduleShape shape,
        ScheduleTaskId projection,
        const target::LPUTargetModel& target) const;
};

class FfnDownProjectionScheduleBuilder {
public:
    ScheduleTaskId append(SchedulePlan& plan, FfnScheduleShape shape,
        ScheduleTaskId swish, const target::LPUTargetModel& target) const;
};

FfnTaskPlan buildFfnTaskPlan(FfnScheduleShape shape,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
