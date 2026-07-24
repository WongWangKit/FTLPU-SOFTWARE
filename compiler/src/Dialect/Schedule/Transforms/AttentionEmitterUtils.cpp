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

void emitSxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t hemisphere, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::ArrayRef<int64_t> permuteMap,
    llvm::StringRef weightLayout)
{
    const auto integers = [&](llvm::ArrayRef<int64_t> values) {
        llvm::SmallVector<mlir::Attribute> attributes;
        attributes.reserve(values.size());
        for (int64_t value : values)
            attributes.push_back(rewriter.getI64IntegerAttr(value));
        return rewriter.getArrayAttr(attributes);
    };
    mlir::OperationState state(location, SxmOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("hemisphere", rewriter.getI64IntegerAttr(hemisphere)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("source_streams", integers(sourceStreams)),
        rewriter.getNamedAttr("destination_streams", integers(destinationStreams)),
        rewriter.getNamedAttr("permute_map", integers(permuteMap)),
        rewriter.getNamedAttr("weight_layout", rewriter.getStringAttr(weightLayout)),
    });
    rewriter.create(state);
}

std::array<int64_t, 32> identityMap()
{
    std::array<int64_t, 32> map {};
    for (int64_t lane = 0; lane < static_cast<int64_t>(map.size()); ++lane)
        map[static_cast<std::size_t>(lane)] = lane;
    return map;
}

std::array<int64_t, 32> blockDiagonalMap(int64_t diagonal,
    const target::LPUTargetModel& target)
{
    auto map = identityMap();
    const int64_t rows = target.throughput().tile_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    for (int64_t destination = 0; destination < rows; ++destination) {
        const int64_t source = (diagonal + rows - destination) % rows;
        for (int64_t lane = 0; lane < lanes; ++lane)
            map[static_cast<std::size_t>(destination * lanes + lane)]
                = source * lanes + lane;
    }
    return map;
}

VxmOp emitVxm(mlir::IRRewriter& rewriter, stream::AttentionOp op,
    mlir::Value value, int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere)
{
    mlir::OperationState state(op.getLoc(), VxmOp::getOperationName());
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
    return llvm::cast<VxmOp>(rewriter.create(state));
}

AttentionProjectionKind projectionKind(int64_t index)
{
    return static_cast<AttentionProjectionKind>(index);
}
} // namespace ftlpu::compiler::schedule::attention_detail
