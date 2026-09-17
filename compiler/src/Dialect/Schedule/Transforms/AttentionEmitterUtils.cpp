#include "AttentionEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::attention_detail {

void emitMem(mlir::IRRewriter &rewriter, mlir::Location location, int64_t cycle,
             int64_t queue, llvm::StringRef opcode, int64_t address,
             int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
             int64_t addressStride, llvm::StringRef destination,
             int64_t addressBinding, int64_t bank, int64_t weightPage,
             int64_t logicalBaseRow) {
  emitMem3D(rewriter, location, cycle, queue, opcode, address, packedStream,
            repeatCount, repeatInterval, addressStride, destination,
            addressBinding, 1, 1, 0, 1, 1, 0, bank, weightPage,
            logicalBaseRow);
}

void emitMem3D(mlir::IRRewriter &rewriter, mlir::Location location,
               int64_t cycle, int64_t queue, llvm::StringRef opcode,
               int64_t address, int64_t packedStream, int64_t repeatCount,
               int64_t repeatInterval, int64_t addressStride,
               llvm::StringRef destination, int64_t addressBinding,
               int64_t waveCount, int64_t waveInterval,
               int64_t waveAddressStride, int64_t groupCount,
               int64_t groupInterval, int64_t groupAddressStride,
               int64_t bank, int64_t weightPage, int64_t logicalBaseRow,
               int64_t outerGroupSize, int64_t outerInnerStride,
               int64_t outerGroupStride) {
  const target::LPUTargetModel target;
  const int64_t hemisphere =
      queue / target.memory().slices_per_hemisphere;
  const int64_t slice = queue % target.memory().slices_per_hemisphere;
  (void)destination;
  mlir::OperationState state(location, MemTransferOp::getOperationName());
  state.addAttributes({
      rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
      rewriter.getNamedAttr("hemisphere",
                            rewriter.getI64IntegerAttr(
                                hemisphere)),
      rewriter.getNamedAttr("slice",
                            rewriter.getI64IntegerAttr(
                                slice)),
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
      rewriter.getNamedAttr("group_count",
                            rewriter.getI64IntegerAttr(groupCount)),
      rewriter.getNamedAttr("group_interval",
                            rewriter.getI64IntegerAttr(groupInterval)),
      rewriter.getNamedAttr("group_address_stride",
                            rewriter.getI64IntegerAttr(groupAddressStride)),
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
  if (outerGroupSize > 1) {
    state.addAttribute("outer_group_size",
                       rewriter.getI64IntegerAttr(outerGroupSize));
    state.addAttribute("outer_inner_stride",
                       rewriter.getI64IntegerAttr(outerInnerStride));
    state.addAttribute("outer_group_stride",
                       rewriter.getI64IntegerAttr(outerGroupStride));
  }
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
  emitMem3D(rewriter, location, cycle, queue, opcode, address, packedStream,
            repeatCount, repeatInterval, addressStride, destination,
            addressBinding, waveCount, waveInterval, waveAddressStride,
            1, 1, 0, bank);
}

void emitMxm3D(mlir::IRRewriter &rewriter, mlir::Location location,
               int64_t cycle, int64_t queue, llvm::StringRef opcode,
               int64_t weightBuffer, int64_t weightColumn,
               int64_t activationStream, int64_t outputStream,
               int64_t accumulatorAddress, int64_t accumulatorRowStride,
               llvm::StringRef accumulatorDestination, bool accumulatorClear,
               llvm::StringRef weightLoadMode, int64_t weightInnerColumn,
               llvm::StringRef dataFormat, llvm::StringRef weightInputMode,
               llvm::StringRef accumulatorOutputFormat,
               const MxmDomain3D &domain, int64_t weightStreamBase) {
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
                            rewriter.getI64IntegerAttr(domain.repeat_count)),
      rewriter.getNamedAttr("repeat_interval",
                            rewriter.getI64IntegerAttr(domain.repeat_interval)),
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
      rewriter.getNamedAttr("weight_buffer_mode",
                            rewriter.getStringAttr(domain.weight_buffer_mode)),
      rewriter.getNamedAttr("repeat_weight_column_stride",
                            rewriter.getI64IntegerAttr(
                                domain.repeat_weight_column_stride)),
      rewriter.getNamedAttr("repeat_accumulator_address_stride",
                            rewriter.getI64IntegerAttr(
                                domain.repeat_accumulator_address_stride)),
      rewriter.getNamedAttr("wave_count",
                            rewriter.getI64IntegerAttr(domain.wave_count)),
      rewriter.getNamedAttr("wave_interval",
                            rewriter.getI64IntegerAttr(domain.wave_interval)),
      rewriter.getNamedAttr("wave_weight_column_stride",
                            rewriter.getI64IntegerAttr(
                                domain.wave_weight_column_stride)),
      rewriter.getNamedAttr("wave_accumulator_address_stride",
                            rewriter.getI64IntegerAttr(
                                domain.wave_accumulator_address_stride)),
      rewriter.getNamedAttr("group_count",
                            rewriter.getI64IntegerAttr(domain.group_count)),
      rewriter.getNamedAttr("group_interval",
                            rewriter.getI64IntegerAttr(domain.group_interval)),
      rewriter.getNamedAttr("group_weight_column_stride",
                            rewriter.getI64IntegerAttr(
                                domain.group_weight_column_stride)),
      rewriter.getNamedAttr("group_accumulator_address_stride",
                            rewriter.getI64IntegerAttr(
                                domain.group_accumulator_address_stride)),
  });
  if (!weightInputMode.empty())
    state.addAttribute("weight_input_mode",
                       rewriter.getStringAttr(weightInputMode));
  if (weightStreamBase >= 0)
    state.addAttribute("weight_stream_base",
                       rewriter.getI64IntegerAttr(weightStreamBase));
  if (!accumulatorOutputFormat.empty())
    state.addAttribute("accumulator_output_format",
                       rewriter.getStringAttr(accumulatorOutputFormat));
  if (domain.terminal_dimension >= 0)
    state.addAttribute("terminal_dimension",
                       rewriter.getI64IntegerAttr(domain.terminal_dimension));
  if (!domain.terminal_accumulator_destination.empty())
    state.addAttribute("terminal_accumulator_destination",
                       rewriter.getStringAttr(
                           domain.terminal_accumulator_destination));
  if (domain.terminal_accumulator_clear)
    state.addAttribute("terminal_accumulator_clear",
                       rewriter.getBoolAttr(
                           *domain.terminal_accumulator_clear));
  if (!domain.terminal_accumulator_output_format.empty())
    state.addAttribute("terminal_accumulator_output_format",
                       rewriter.getStringAttr(
                           domain.terminal_accumulator_output_format));
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
                 llvm::StringRef accumulatorOutputFormat, int64_t waveCount,
                 int64_t waveInterval, int64_t waveWeightColumnStride,
                 int64_t groupCount, int64_t groupInterval,
                 int64_t waveAccumulatorAddressStride,
                 int64_t weightStreamBase) {
  MxmDomain3D domain;
  domain.repeat_count = repeatCount;
  domain.repeat_interval = repeatInterval;
  domain.wave_count = waveCount;
  domain.wave_interval = waveInterval;
  domain.wave_weight_column_stride = waveWeightColumnStride;
  domain.wave_accumulator_address_stride = waveAccumulatorAddressStride;
  domain.group_count = groupCount;
  domain.group_interval = groupInterval;
  emitMxm3D(rewriter, location, cycle, queue, opcode, weightBuffer,
            weightColumn, activationStream, outputStream, accumulatorAddress,
            accumulatorRowStride, accumulatorDestination, accumulatorClear,
            weightLoadMode, weightInnerColumn, dataFormat, weightInputMode,
            accumulatorOutputFormat, domain, weightStreamBase);
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
                 llvm::StringRef accumulatorOutputFormat, int64_t waveCount,
                 int64_t waveInterval, int64_t waveWeightColumnStride) {
  emitMxmWave(rewriter, location, cycle, queue, opcode, weightBuffer,
              weightColumn, activationStream, outputStream, repeatCount,
              repeatInterval, accumulatorAddress, accumulatorRowStride,
              accumulatorDestination, accumulatorClear, weightLoadMode,
              weightInnerColumn, dataFormat, weightInputMode,
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
             llvm::StringRef accumulatorOutputFormat,
             int64_t weightStreamBase) {
  emitMxmWave(rewriter, location, cycle, queue, opcode, weightBuffer,
              weightColumn, activationStream, outputStream, repeatCount,
              repeatInterval, accumulatorAddress, accumulatorRowStride,
              accumulatorDestination, accumulatorClear, weightLoadMode,
              weightInnerColumn, dataFormat, weightInputMode,
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
  emitMxmDequant3D(rewriter, location, cycle, unitId, scale, repeatCount,
                   repeatInterval, waveCount, waveInterval, 1, 1,
                   scaleBinding);
}

void emitMxmDequant3D(mlir::IRRewriter &rewriter, mlir::Location location,
                      int64_t cycle, int64_t unitId, float scale,
                      int64_t repeatCount, int64_t repeatInterval,
                      int64_t waveCount, int64_t waveInterval,
                      int64_t groupCount, int64_t groupInterval,
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
      rewriter.getNamedAttr("group_count",
                            rewriter.getI64IntegerAttr(groupCount)),
      rewriter.getNamedAttr("group_interval",
                            rewriter.getI64IntegerAttr(groupInterval)),
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
                        int64_t repeatInterval,
                        llvm::StringRef lhsStreamSource,
                        llvm::StringRef rhsStreamSource,
                        int64_t waveCount, int64_t waveInterval) {
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
      rewriter.getNamedAttr("wave_count",
                            rewriter.getI64IntegerAttr(waveCount)),
      rewriter.getNamedAttr("wave_interval",
                            rewriter.getI64IntegerAttr(waveInterval)),
      rewriter.getNamedAttr("input_hemisphere",
                            rewriter.getStringAttr(inputHemisphere)),
      rewriter.getNamedAttr("output_hemisphere",
                            rewriter.getStringAttr(outputHemisphere)),
  });
  if (scaleBinding >= 0)
    state.addAttribute("scale_binding",
                       rewriter.getI64IntegerAttr(scaleBinding));
  if (!lhsStreamSource.empty())
    state.addAttribute("lhs_stream_source",
                       rewriter.getStringAttr(lhsStreamSource));
  if (!rhsStreamSource.empty())
    state.addAttribute("rhs_stream_source",
                       rewriter.getStringAttr(rhsStreamSource));
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
