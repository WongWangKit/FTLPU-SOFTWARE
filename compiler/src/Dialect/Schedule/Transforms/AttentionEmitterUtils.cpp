#include "AttentionEmitterUtils.hpp"

#include "llvm/ADT/SmallVector.h"

namespace ftlpu::compiler::schedule::attention_detail {

void emitMem(mlir::IRRewriter &rewriter, mlir::Location location, int64_t cycle,
             int64_t queue, llvm::StringRef opcode, int64_t address,
             int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
             int64_t addressStride, llvm::StringRef destination,
             int64_t addressBinding, int64_t bank, int64_t weightPage,
             int64_t logicalBaseRow) {
  const target::LPUTargetModel target;
  mlir::OperationState state(location, MemTransferOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("hemisphere",
                            rewriter.getI64IntegerAttr(
                                queue / target.memory().slices_per_hemisphere)),
      rewriter.getNamedAttr("slice",
                            rewriter.getI64IntegerAttr(
                                queue % target.memory().slices_per_hemisphere)),
      rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
      rewriter.getNamedAttr("address", rewriter.getI64IntegerAttr(address)),
      rewriter.getNamedAttr("packed_stream",
                            rewriter.getI64IntegerAttr(packedStream)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
      rewriter.getNamedAttr("address_stride",
                            rewriter.getI64IntegerAttr(addressStride)),
  });
  if (addressBinding >= 0)
    state.addAttribute("address_binding",
                       rewriter.getI64IntegerAttr(addressBinding));
  if (bank >= 0)
    state.addAttribute("bank", rewriter.getI64IntegerAttr(bank));
  if (weightPage >= 0)
    state.addAttribute("weight_page", rewriter.getI64IntegerAttr(weightPage));
  if (logicalBaseRow >= 0)
    state.addAttribute("logical_base_row",
                       rewriter.getI64IntegerAttr(logicalBaseRow));
  rewriter.create(state);
}

void emitMemWave(mlir::IRRewriter &rewriter, mlir::Location location,
                 int64_t cycle, int64_t queue, llvm::StringRef opcode,
                 int64_t address, int64_t packedStream, int64_t repeatCount,
                 int64_t repeatInterval, int64_t addressStride,
                 llvm::StringRef destination, int64_t addressBinding,
                 int64_t waveCount, int64_t waveInterval,
                 int64_t waveAddressStride) {
  emitMemWave(rewriter, location, cycle, queue, opcode, address, packedStream,
              repeatCount, repeatInterval, addressStride, destination,
              addressBinding, waveCount, waveInterval, waveAddressStride, -1);
}

void emitMemWave(mlir::IRRewriter &rewriter, mlir::Location location,
                 int64_t cycle, int64_t queue, llvm::StringRef opcode,
                 int64_t address, int64_t packedStream, int64_t repeatCount,
                 int64_t repeatInterval, int64_t addressStride,
                 llvm::StringRef destination, int64_t addressBinding,
                 int64_t waveCount, int64_t waveInterval,
                 int64_t waveAddressStride, int64_t bank) {
  const target::LPUTargetModel target;
  mlir::OperationState state(location, MemTransferOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("hemisphere",
                            rewriter.getI64IntegerAttr(
                                queue / target.memory().slices_per_hemisphere)),
      rewriter.getNamedAttr("slice",
                            rewriter.getI64IntegerAttr(
                                queue % target.memory().slices_per_hemisphere)),
      rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
      rewriter.getNamedAttr("address", rewriter.getI64IntegerAttr(address)),
      rewriter.getNamedAttr("packed_stream",
                            rewriter.getI64IntegerAttr(packedStream)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
      rewriter.getNamedAttr("address_stride",
                            rewriter.getI64IntegerAttr(addressStride)),
      rewriter.getNamedAttr("wave_count",
                            rewriter.getI64IntegerAttr(waveCount)),
      rewriter.getNamedAttr("wave_interval",
                            rewriter.getI64IntegerAttr(waveInterval)),
      rewriter.getNamedAttr("wave_address_stride",
                            rewriter.getI64IntegerAttr(waveAddressStride)),
  });
  if (addressBinding >= 0)
    state.addAttribute("address_binding",
                       rewriter.getI64IntegerAttr(addressBinding));
  if (bank >= 0)
    state.addAttribute("bank", rewriter.getI64IntegerAttr(bank));
  rewriter.create(state);
}

void emitMxmWave(mlir::IRRewriter &rewriter, mlir::Location location,
                 int64_t cycle, int64_t queue, llvm::StringRef opcode,
                 int64_t weightBuffer, int64_t weightColumn,
                 int64_t activationStream, int64_t outputStream,
                 int64_t repeatCount, int64_t repeatInterval,
                 int64_t accumulatorAddress, int64_t accumulatorRowStride,
                 llvm::StringRef accumulatorDestination, bool accumulatorClear,
                 llvm::StringRef weightLoadMode, int64_t weightInnerColumn,
                 llvm::StringRef dataFormat, llvm::StringRef weightInputMode,
                 llvm::StringRef computeMode,
                 llvm::StringRef accumulatorOutputFormat, int64_t waveCount,
                 int64_t waveInterval, int64_t waveWeightColumnStride,
                 int64_t groupCount, int64_t groupInterval,
                 int64_t waveAccumulatorAddressStride,
                 int64_t weightStreamBase) {
  mlir::OperationState state(location, MxmIssueOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(queue)),
      rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
      rewriter.getNamedAttr("weight_buffer",
                            rewriter.getI64IntegerAttr(weightBuffer)),
      rewriter.getNamedAttr("weight_column",
                            rewriter.getI64IntegerAttr(weightColumn)),
      rewriter.getNamedAttr("activation_stream_base",
                            rewriter.getI64IntegerAttr(activationStream)),
      rewriter.getNamedAttr("output_stream_base",
                            rewriter.getI64IntegerAttr(outputStream)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
      rewriter.getNamedAttr("accumulator_address",
                            rewriter.getI64IntegerAttr(accumulatorAddress)),
      rewriter.getNamedAttr("accumulator_row_stride",
                            rewriter.getI64IntegerAttr(accumulatorRowStride)),
      rewriter.getNamedAttr("accumulator_destination",
                            rewriter.getStringAttr(accumulatorDestination)),
      rewriter.getNamedAttr("accumulator_clear",
                            rewriter.getBoolAttr(accumulatorClear)),
      rewriter.getNamedAttr("data_format", rewriter.getStringAttr(dataFormat)),
      rewriter.getNamedAttr("weight_load_mode",
                            rewriter.getStringAttr(weightLoadMode)),
      rewriter.getNamedAttr("weight_inner_column",
                            rewriter.getI64IntegerAttr(weightInnerColumn)),
  });
  if (!weightInputMode.empty())
    state.addAttribute("weight_input_mode",
                       rewriter.getStringAttr(weightInputMode));
  if (weightStreamBase >= 0)
    state.addAttribute("weight_stream_base",
                       rewriter.getI64IntegerAttr(weightStreamBase));
  if (!computeMode.empty())
    state.addAttribute("compute_mode", rewriter.getStringAttr(computeMode));
  if (!accumulatorOutputFormat.empty())
    state.addAttribute("accumulator_output_format",
                       rewriter.getStringAttr(accumulatorOutputFormat));
  if (waveCount != 1 || waveInterval != 1 || waveWeightColumnStride != 0 ||
      waveAccumulatorAddressStride != 0) {
    state.addAttribute("wave_count", rewriter.getI64IntegerAttr(waveCount));
    state.addAttribute("wave_interval",
                       rewriter.getI64IntegerAttr(waveInterval));
    state.addAttribute("wave_weight_column_stride",
                       rewriter.getI64IntegerAttr(waveWeightColumnStride));
    state.addAttribute(
        "wave_accumulator_address_stride",
        rewriter.getI64IntegerAttr(waveAccumulatorAddressStride));
  }
  if (groupCount != 1 || groupInterval != 1) {
    state.addAttribute("group_count", rewriter.getI64IntegerAttr(groupCount));
    state.addAttribute("group_interval",
                       rewriter.getI64IntegerAttr(groupInterval));
  }
  rewriter.create(state);
}

// Preserve the helper ABI for incremental builds and out-of-tree emitters.
void emitMxmWave(mlir::IRRewriter &rewriter, mlir::Location location,
                 int64_t cycle, int64_t queue, llvm::StringRef opcode,
                 int64_t weightBuffer, int64_t weightColumn,
                 int64_t activationStream, int64_t outputStream,
                 int64_t repeatCount, int64_t repeatInterval,
                 int64_t accumulatorAddress, int64_t accumulatorRowStride,
                 llvm::StringRef accumulatorDestination, bool accumulatorClear,
                 llvm::StringRef weightLoadMode, int64_t weightInnerColumn,
                 llvm::StringRef dataFormat, llvm::StringRef weightInputMode,
                 llvm::StringRef computeMode,
                 llvm::StringRef accumulatorOutputFormat, int64_t waveCount,
                 int64_t waveInterval, int64_t waveWeightColumnStride) {
  emitMxmWave(rewriter, location, cycle, queue, opcode, weightBuffer,
              weightColumn, activationStream, outputStream, repeatCount,
              repeatInterval, accumulatorAddress, accumulatorRowStride,
              accumulatorDestination, accumulatorClear, weightLoadMode,
              weightInnerColumn, dataFormat, weightInputMode, computeMode,
              accumulatorOutputFormat, waveCount, waveInterval,
              waveWeightColumnStride, 1, 1, 0, -1);
}

void emitMxm(mlir::IRRewriter &rewriter, mlir::Location location, int64_t cycle,
             int64_t queue, llvm::StringRef opcode, int64_t weightBuffer,
             int64_t weightColumn, int64_t activationStream,
             int64_t outputStream, int64_t repeatCount, int64_t repeatInterval,
             int64_t accumulatorAddress, int64_t accumulatorRowStride,
             llvm::StringRef accumulatorDestination, bool accumulatorClear,
             llvm::StringRef weightLoadMode, int64_t weightInnerColumn,
             llvm::StringRef dataFormat, llvm::StringRef weightInputMode,
             llvm::StringRef computeMode,
             llvm::StringRef accumulatorOutputFormat,
             int64_t weightStreamBase) {
  emitMxmWave(rewriter, location, cycle, queue, opcode, weightBuffer,
              weightColumn, activationStream, outputStream, repeatCount,
              repeatInterval, accumulatorAddress, accumulatorRowStride,
              accumulatorDestination, accumulatorClear, weightLoadMode,
              weightInnerColumn, dataFormat, weightInputMode, computeMode,
              accumulatorOutputFormat, 1, 1, 0, 1, 1, 0, weightStreamBase);
}

void emitMxmDequant(mlir::IRRewriter &rewriter, mlir::Location location,
                    int64_t cycle, int64_t unitId, float scale,
                    int64_t repeatCount, int64_t repeatInterval,
                    int64_t scaleBinding) {
  mlir::OperationState state(location, MxmDequantOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(unitId)),
      rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
  });
  if (scaleBinding >= 0)
    state.addAttribute("scale_binding",
                       rewriter.getI64IntegerAttr(scaleBinding));
  rewriter.create(state);
}

void emitMxmDequantWave(mlir::IRRewriter &rewriter, mlir::Location location,
                        int64_t cycle, int64_t unitId, float scale,
                        int64_t repeatCount, int64_t repeatInterval,
                        int64_t waveCount, int64_t waveInterval,
                        int64_t scaleBinding) {
  mlir::OperationState state(location, MxmDequantOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(unitId)),
      rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
      rewriter.getNamedAttr("wave_count",
                            rewriter.getI64IntegerAttr(waveCount)),
      rewriter.getNamedAttr("wave_interval",
                            rewriter.getI64IntegerAttr(waveInterval)),
  });
  if (scaleBinding >= 0)
    state.addAttribute("scale_binding",
                       rewriter.getI64IntegerAttr(scaleBinding));
  rewriter.create(state);
}

VxmOp emitVxmConfigured(mlir::IRRewriter &rewriter, mlir::Location location,
                        mlir::Value value, int64_t cycle, int64_t queue,
                        llvm::StringRef opcode, llvm::StringRef lhsKind,
                        int64_t lhsIndex, float lhsImmediate,
                        llvm::StringRef rhsKind, int64_t rhsIndex,
                        float rhsImmediate, llvm::StringRef castTarget,
                        int64_t outputStream, llvm::StringRef inputHemisphere,
                        llvm::StringRef outputHemisphere, int64_t scaleBinding,
                        int64_t chainDepth, int64_t repeatCount,
                        int64_t repeatInterval) {
  mlir::OperationState state(location, VxmOp::getOperationName());
  state.addOperands({value, value});
  state.addTypes(value.getType());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
      rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
      rewriter.getNamedAttr("chain_depth",
                            rewriter.getI64IntegerAttr(chainDepth)),
      rewriter.getNamedAttr("lhs_kind", rewriter.getStringAttr(lhsKind)),
      rewriter.getNamedAttr("lhs_index", rewriter.getI64IntegerAttr(lhsIndex)),
      rewriter.getNamedAttr("lhs_immediate",
                            rewriter.getF32FloatAttr(lhsImmediate)),
      rewriter.getNamedAttr("rhs_kind", rewriter.getStringAttr(rhsKind)),
      rewriter.getNamedAttr("rhs_index", rewriter.getI64IntegerAttr(rhsIndex)),
      rewriter.getNamedAttr("rhs_immediate",
                            rewriter.getF32FloatAttr(rhsImmediate)),
      rewriter.getNamedAttr("cast_target", rewriter.getStringAttr(castTarget)),
      rewriter.getNamedAttr("output_stream",
                            rewriter.getI64IntegerAttr(outputStream)),
      rewriter.getNamedAttr("repeat_count",
                            rewriter.getI64IntegerAttr(repeatCount)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(repeatInterval)),
      rewriter.getNamedAttr("input_hemisphere",
                            rewriter.getStringAttr(inputHemisphere)),
      rewriter.getNamedAttr("output_hemisphere",
                            rewriter.getStringAttr(outputHemisphere)),
  });
  if (scaleBinding >= 0)
    state.addAttribute("scale_binding",
                       rewriter.getI64IntegerAttr(scaleBinding));
  return llvm::cast<VxmOp>(rewriter.create(state));
}

VxmOp emitVxm(mlir::IRRewriter &rewriter, mlir::Location location,
              mlir::Value value, int64_t cycle, int64_t queue,
              llvm::StringRef opcode, llvm::StringRef lhsKind, int64_t lhsIndex,
              float lhsImmediate, llvm::StringRef rhsKind, int64_t rhsIndex,
              float rhsImmediate, llvm::StringRef castTarget,
              int64_t outputStream, llvm::StringRef inputHemisphere,
              llvm::StringRef outputHemisphere, int64_t scaleBinding) {
  return emitVxmConfigured(
      rewriter, location, value, cycle, queue, opcode, lhsKind, lhsIndex,
      lhsImmediate, rhsKind, rhsIndex, rhsImmediate, castTarget, outputStream,
      inputHemisphere, outputHemisphere, scaleBinding, 8, 1, 1);
}

AttentionProjectionKind projectionKind(int64_t index) {
  return static_cast<AttentionProjectionKind>(index);
}
} // namespace ftlpu::compiler::schedule::attention_detail
