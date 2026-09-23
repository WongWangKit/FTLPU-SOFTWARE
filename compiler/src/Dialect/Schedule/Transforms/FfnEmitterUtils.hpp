#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>
#include <utility>

namespace ftlpu::compiler::schedule::ffn_detail {

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement);

int64_t get_base_row(mlir::DictionaryAttr placement);

struct PagedWeightPagePlacement {
    int64_t bank;
    int64_t slice_group_base;
    int64_t slice_group_count;
    int64_t base_row;
    int64_t row_count;
};

mlir::FailureOr<PagedWeightPagePlacement> resolve_page_placement(
    mlir::DictionaryAttr placement, int64_t page);

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t baseRow, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind);

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t baseRow, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind,
    int64_t bank);

struct FfnLoopDomain3D {
    int64_t wave_count{1};
    int64_t wave_interval{1};
    int64_t wave_address_stride{0};
    int64_t group_count{1};
    int64_t group_interval{1};
    int64_t group_address_stride{0};
    bool blocked_outer_address{false};
    int64_t outer_group_size{1};
    int64_t outer_inner_stride{0};
    int64_t outer_group_stride{0};
};

struct FfnMxmDomain3D {
    int64_t repeat_count{1};
    int64_t repeat_interval{1};
    int64_t repeat_weight_column_stride{0};
    int64_t repeat_accumulator_address_stride{0};
    int64_t wave_count{1};
    int64_t wave_interval{1};
    int64_t wave_weight_column_stride{0};
    int64_t wave_accumulator_address_stride{0};
    int64_t group_count{1};
    int64_t group_interval{1};
    int64_t group_weight_column_stride{0};
    int64_t group_accumulator_address_stride{0};
    llvm::StringRef weight_buffer_mode{"fixed"};
    int64_t terminal_dimension{-1};
    llvm::StringRef terminal_accumulator_destination;
    std::optional<bool> terminal_accumulator_clear;
    llvm::StringRef terminal_accumulator_output_format;
};

MxmIssueOp emitFfnMxmIssue3D(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t unit,
    llvm::StringRef opcode, int64_t weightBuffer, int64_t weightColumn,
    int64_t activationStream, int64_t outputStream,
    int64_t accumulatorAddress, int64_t accumulatorRowStride,
    llvm::StringRef accumulatorDestination, bool accumulatorClear,
    llvm::StringRef dataFormat, llvm::StringRef accumulatorOutputFormat,
    const FfnMxmDomain3D& domain);

// Emits one hardware-shaped MEM loop domain.  The caller owns the operator
// loop nest and supplies its outer dimensions directly; this helper never
// searches previously emitted operations or reconstructs loops afterwards.
MemReadOp emitFfnMemRead3D(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::Value input, int64_t cycle,
    int64_t duration, int64_t streamBase, int64_t streamCount,
    int64_t registerId, mlir::StringAttr direction, mlir::StringAttr role,
    mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
    int64_t bytes, FfnLoopDomain3D domain = {});

void setFfnLoopDomain3D(mlir::Operation* operation,
    mlir::OpBuilder& builder, FfnLoopDomain3D domain);

MemTransferOp emitFfnMemTransfer3D(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t hemisphere,
    int64_t slice, llvm::StringRef opcode, int64_t address,
    int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t addressStride, int64_t bank, FfnLoopDomain3D domain = {},
    int64_t addressBinding = -1,
    llvm::StringRef addressBindingAccess = {}, int64_t weightPage = -1,
    int64_t logicalBaseRow = -1);

VxmOp create_vxm(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Value lhsValue, mlir::Value rhsValue, mlir::Type resultType,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere,
    int64_t scaleBinding = -1, bool accumulatorReset = false,
    bool accumulatorWrite = false, bool accumulatorEmit = true,
    bool localScalarWrite = false, int64_t chainDepth = 8);

llvm::StringRef hemisphere_name(int64_t hemisphere);

std::pair<VxmOp, VxmOp> emitFfnSwishAlu(
    mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Type resultType, mlir::Value gateValue, mlir::Value upValue,
    const target::LPUTargetModel& target, FfnScheduleStrategy strategy,
    int64_t cycle, int64_t hemisphere, int64_t outputStream,
    int64_t repeatCount = 1, int64_t repeatInterval = 1);

mlir::Value emitFfnSwishResultTile(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& plan, const target::LPUTargetModel& target,
    llvm::ArrayRef<int64_t> hiddenSlices, mlir::Value output,
    int64_t inputCycle, int64_t mTile, int64_t pair,
    int64_t sourceHemisphere, int64_t rowCount,
    bool mirroredBroadcast = false,
    FfnLoopDomain3D outerDomain = {},
    bool directParityDomain = false);

MxmLoadOp emitFfnWeightTile(mlir::IRRewriter& rewriter,
    mlir::Location location, stream::RouteOp rawRoute,
    mlir::Type dequantizedType, llvm::ArrayRef<int64_t> weightSlices,
    const target::LPUTargetModel& target, float scale, int64_t startCycle,
    int64_t baseRow, int64_t hemisphere, int64_t localMxm,
    int64_t unit, int64_t weightBuffer, bool localDequant,
    int64_t bank = 0, int64_t pageIndex = -1,
    int64_t logicalBaseRow = -1,
    mlir::DictionaryAttr bindingPlacement = {},
    FfnLoopDomain3D domain = {},
    llvm::StringRef weightBufferMode = "fixed",
    std::optional<FfnLoopDomain3D> mxmControlDomain = std::nullopt,
    bool emitMxmControl = true);

} // namespace ftlpu::compiler::schedule::ffn_detail
