#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/stream_fabric_scheduler.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::schedule {

mlir::LogicalResult lowerSwigluSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target,
    ResourceScheduler& scheduler, StreamFabricScheduler& stream_scheduler);

mlir::LogicalResult lowerMatmulSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target,
    ResourceScheduler& scheduler, StreamFabricScheduler& stream_scheduler);

mlir::LogicalResult lowerLinearProjectionSchedules(
    mlir::IRRewriter& rewriter, mlir::func::FuncOp function,
    const target::LPUTargetModel& target);

mlir::LogicalResult lowerRmsNormSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target);

// Emit the VXM feedback implementation used by fused operations that already
// own their input, gamma, and scratch placements.
int64_t emitVxmFeedbackRmsNorm(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::Value input, mlir::Value weight,
    mlir::RankedTensorType inputType, double epsilon,
    const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr weightPlacement,
    mlir::DictionaryAttr outputPlacement, int64_t start);

mlir::LogicalResult lowerElementwiseSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
