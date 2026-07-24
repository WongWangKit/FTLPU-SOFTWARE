#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::schedule {

mlir::FailureOr<mlir::Value> lowerFfnSchedule(
    mlir::IRRewriter& rewriter, PrimitiveFfnSchedulePlan& plan,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
