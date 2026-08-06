#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_task_graph.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace ftlpu::compiler::schedule {

struct AttentionSoftmaxSchedule {
    std::vector<std::array<std::optional<int64_t>, 2>> wave_cycles;
    int64_t work_interval = 0;
    int64_t end_cycle = 0;
};

mlir::FailureOr<AttentionSoftmaxSchedule> planAttentionSoftmax(
    const AttentionTaskGraph& graph,
    const std::vector<AttentionWorkWave>& waves,
    int64_t qkStart, int64_t qkEnd, int64_t qkWaveInterval,
    int64_t qkIwToComputeCycles, bool fused,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
