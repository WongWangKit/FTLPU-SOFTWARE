#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_swish_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <array>
#include <memory>
#include <vector>

namespace ftlpu::compiler::schedule::ffn_detail {

struct FfnEmissionContext {
    mlir::IRRewriter& rewriter;
    PrimitiveFfnSchedulePlan& ffn;
    FfnScheduleStrategy strategy;
    const target::LPUTargetModel& target;

    stream::RouteOp activation_route;
    stream::RouteOp gate_route;
    stream::RouteOp up_route;
    stream::RouteOp down_route;
    stream::RouteOp gate_raw;
    stream::RouteOp up_raw;
    stream::RouteOp down_raw;

    llvm::SmallVector<int64_t> weight_slices;
    llvm::SmallVector<int64_t> activation_slices;
    llvm::SmallVector<int64_t> hidden_slices;
    llvm::SmallVector<int64_t> result_slices;
    llvm::SmallVector<int64_t> gate_acc_slices;
    llvm::SmallVector<int64_t> up_acc_slices;

    FfnProjectionTimeline projection_timeline;
    mlir::RankedTensorType projection_type;
    int64_t activation_latency;
    int64_t down_accumulator_base;

    int64_t m() const { return static_cast<int64_t>(ffn.getM()); }
    int64_t k() const { return static_cast<int64_t>(ffn.getK()); }
    int64_t hidden() const { return static_cast<int64_t>(ffn.getHidden()); }
    int64_t n() const { return static_cast<int64_t>(ffn.getN()); }
    int64_t tile() const { return target.throughput().mxm_rows; }

    int64_t westLatency(int64_t slice) const;
    int64_t eastMxmLatency(int64_t slice) const;
    llvm::StringRef hemisphereName(int64_t hemisphere) const;

    schedule::MemReadOp emitSliceRead(mlir::Value value,
        stream::RouteOp route, int64_t cycle, int64_t slice,
        int64_t base, int64_t count, int64_t stride, int64_t stream,
        llvm::StringRef direction, llvm::StringRef role,
        llvm::StringRef hemisphere);
};

struct CompletedProjectionTile {
    int64_t pair;
    int64_t m_tile;
    int64_t hemisphere;
    int64_t compute_cycle;
    int64_t deferred_ready_cycle;
    schedule::MemAccumulateOp gate;
    schedule::MemAccumulateOp up;
    mlir::Value gate_temp;
    mlir::Value up_temp;
};

struct FfnProjectionEmission {
    llvm::SmallVector<CompletedProjectionTile> completed_tiles;
    std::array<std::vector<FfnCycleWindow>, 2> temp_mem_busy_windows;
};

struct FfnSwishEmission {
    mlir::Value hidden;
    int64_t last_cycle;
};

mlir::FailureOr<std::unique_ptr<FfnEmissionContext>>
createFfnEmissionContext(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& ffn, FfnScheduleStrategy strategy,
    const target::LPUTargetModel& target);

mlir::FailureOr<FfnProjectionEmission> emitFfnProjection(
    FfnEmissionContext& context);

mlir::FailureOr<FfnSwishEmission> emitFfnSwish(
    FfnEmissionContext& context, FfnProjectionEmission emission);

mlir::FailureOr<mlir::Value> emitFfnDownProjection(
    FfnEmissionContext& context, const FfnSwishEmission& swish);

} // namespace ftlpu::compiler::schedule::ffn_detail
