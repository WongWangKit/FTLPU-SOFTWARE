#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"

#include <utility>

namespace ftlpu::compiler::schedule::ffn_detail {

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement);

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t baseRow, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind);

VxmOp create_vxm(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Value lhsValue, mlir::Value rhsValue, mlir::Type resultType,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere);

llvm::StringRef hemisphere_name(int64_t hemisphere);

std::pair<VxmOp, VxmOp> emitFfnSwishAlu(
    mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Type resultType, mlir::Value gateValue, mlir::Value upValue,
    const target::LPUTargetModel& target, FfnScheduleStrategy strategy,
    int64_t cycle, int64_t hemisphere);

MxmLoadOp emitFfnWeightTile(mlir::IRRewriter& rewriter,
    mlir::Location location, stream::RouteOp rawRoute,
    mlir::Type dequantizedType, llvm::ArrayRef<int64_t> weightSlices,
    const target::LPUTargetModel& target, float scale, int64_t startCycle,
    int64_t baseRow, int64_t hemisphere, int64_t localMxm,
    int64_t unit, int64_t weightBuffer);

mlir::Value emitFfnSwishRow(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& plan, const target::LPUTargetModel& target,
    FfnScheduleStrategy strategy, llvm::ArrayRef<int64_t> hiddenSlices,
    mlir::Value gateValue, mlir::Value upValue, int64_t cycle,
    int64_t mTile, int64_t pair, int64_t row, int64_t hemisphere);

} // namespace ftlpu::compiler::schedule::ffn_detail
