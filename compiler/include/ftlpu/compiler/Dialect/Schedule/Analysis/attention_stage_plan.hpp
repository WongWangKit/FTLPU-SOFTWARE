#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

namespace ftlpu::compiler::schedule {

struct AttentionStageShape {
    int64_t sequence_length;
    int64_t hidden;
    int64_t query_heads;
    int64_t kv_heads;
    int64_t head_dim;
};

struct AttentionStageTaskIds {
    ScheduleTaskId projection;
    ScheduleTaskId rope;
    ScheduleTaskId softmax;
    ScheduleTaskId pv;
    ScheduleTaskId output_projection;
};

struct AttentionStagePlan {
    SchedulePlan tasks;
    AttentionStageTaskIds task_ids;
    std::vector<AttentionProjectionWork> projection_work;
    std::vector<AttentionWorkWave> qk_waves;
    std::vector<AttentionWorkWave> pv_waves;
};

class AttentionProjectionStagePlanner {
public:
    ScheduleTaskId append(SchedulePlan& plan, AttentionStageShape shape,
        const target::LPUTargetModel& target) const;
};

class AttentionRopeStagePlanner {
public:
    ScheduleTaskId append(SchedulePlan& plan, AttentionStageShape shape,
        ScheduleTaskId projection,
        const target::LPUTargetModel& target) const;
};

class AttentionSoftmaxStagePlanner {
public:
    ScheduleTaskId append(SchedulePlan& plan, AttentionStageShape shape,
        ScheduleTaskId rope, const target::LPUTargetModel& target) const;
};

class AttentionPvStagePlanner {
public:
    ScheduleTaskId append(SchedulePlan& plan, AttentionStageShape shape,
        ScheduleTaskId softmax,
        const target::LPUTargetModel& target) const;
};

class AttentionOutputProjectionStagePlanner {
public:
    ScheduleTaskId append(SchedulePlan& plan, AttentionStageShape shape,
        ScheduleTaskId pv, const target::LPUTargetModel& target) const;
};

AttentionStagePlan planAttentionStages(AttentionStageShape shape,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
