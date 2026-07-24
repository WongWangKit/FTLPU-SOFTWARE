#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace ftlpu::compiler::schedule {

struct AttentionSoftmaxSchedule {
    std::vector<std::array<std::optional<int64_t>, 2>> wave_cycles;
    int64_t end_cycle = 0;
};

mlir::FailureOr<AttentionSoftmaxSchedule> planAttentionSoftmax(
    stream::AttentionOp op, const std::vector<AttentionWorkWave>& waves,
    int64_t qkEnd, const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
