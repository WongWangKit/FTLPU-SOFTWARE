#include "AttentionEmitterUtils.hpp"

#include "llvm/ADT/SmallVector.h"

namespace ftlpu::compiler::schedule::attention_detail {


void emitMem(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t address,
    int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t addressStride, llvm::StringRef destination)
{
    const target::LPUTargetModel target;
    mlir::OperationState state(location, MemTransferOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("hemisphere", rewriter.getI64IntegerAttr(
            queue / target.memory().slices_per_hemisphere)),
        rewriter.getNamedAttr("slice", rewriter.getI64IntegerAttr(
            queue % target.memory().slices_per_hemisphere)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("address", rewriter.getI64IntegerAttr(address)),
        rewriter.getNamedAttr("packed_stream", rewriter.getI64IntegerAttr(packedStream)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeatCount)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeatInterval)),
        rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(addressStride)),
        rewriter.getNamedAttr("accumulator_destination", rewriter.getStringAttr(destination)),
    });
    rewriter.create(state);
}

void emitMxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t weightBuffer,
    int64_t weightColumn, int64_t activationStream, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval,
    int64_t accumulatorAddress, int64_t accumulatorRowStride,
    llvm::StringRef accumulatorDestination, bool accumulatorClear)
{
    mlir::OperationState state(location, MxmIssueOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("weight_buffer", rewriter.getI64IntegerAttr(weightBuffer)),
        rewriter.getNamedAttr("weight_column", rewriter.getI64IntegerAttr(weightColumn)),
        rewriter.getNamedAttr("activation_stream_base", rewriter.getI64IntegerAttr(activationStream)),
        rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(outputStream)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeatCount)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeatInterval)),
        rewriter.getNamedAttr("accumulator_address", rewriter.getI64IntegerAttr(accumulatorAddress)),
        rewriter.getNamedAttr("accumulator_row_stride", rewriter.getI64IntegerAttr(accumulatorRowStride)),
        rewriter.getNamedAttr("accumulator_destination", rewriter.getStringAttr(accumulatorDestination)),
        rewriter.getNamedAttr("accumulator_clear", rewriter.getBoolAttr(accumulatorClear)),
    });
    rewriter.create(state);
}

VxmOp emitVxm(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Value value, int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere,
    int64_t scaleBinding)
{
    mlir::OperationState state(location, VxmOp::getOperationName());
    state.addOperands({value, value});
    state.addTypes(value.getType());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("lhs_kind", rewriter.getStringAttr(lhsKind)),
        rewriter.getNamedAttr("lhs_index", rewriter.getI64IntegerAttr(lhsIndex)),
        rewriter.getNamedAttr("lhs_immediate", rewriter.getF32FloatAttr(lhsImmediate)),
        rewriter.getNamedAttr("rhs_kind", rewriter.getStringAttr(rhsKind)),
        rewriter.getNamedAttr("rhs_index", rewriter.getI64IntegerAttr(rhsIndex)),
        rewriter.getNamedAttr("rhs_immediate", rewriter.getF32FloatAttr(rhsImmediate)),
        rewriter.getNamedAttr("cast_target", rewriter.getStringAttr(castTarget)),
        rewriter.getNamedAttr("output_stream", rewriter.getI64IntegerAttr(outputStream)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(1)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
        rewriter.getNamedAttr("input_hemisphere", rewriter.getStringAttr(inputHemisphere)),
        rewriter.getNamedAttr("output_hemisphere", rewriter.getStringAttr(outputHemisphere)),
    });
    if (scaleBinding >= 0)
        state.addAttribute(
            "scale_binding", rewriter.getI64IntegerAttr(scaleBinding));
    return llvm::cast<VxmOp>(rewriter.create(state));
}

AttentionProjectionKind projectionKind(int64_t index)
{
    return static_cast<AttentionProjectionKind>(index);
}
} // namespace ftlpu::compiler::schedule::attention_detail
