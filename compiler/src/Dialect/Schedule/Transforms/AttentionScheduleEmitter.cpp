#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_graph.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_memory_layout.hpp"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cmath>
#include <vector>

namespace ftlpu::compiler::schedule {
namespace {

void emitMem(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t address,
    int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t addressStride, llvm::StringRef destination = "sram")
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
    int64_t repeatCount, int64_t repeatInterval)
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
    });
    rewriter.create(state);
}

void emitSxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t hemisphere, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::ArrayRef<int64_t> permuteMap,
    llvm::StringRef weightLayout = "vector_columns")
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

} // namespace

AttentionScheduleEmitter::AttentionScheduleEmitter(mlir::IRRewriter& rewriter,
    stream::AttentionOp op, const target::LPUTargetModel& target)
    : rewriter_(rewriter), op_(op), target_(target)
{
}

int64_t AttentionScheduleEmitter::emitProjections()
{
    const AttentionMemoryLayout layout(op_, target_);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t hiddenBlocks = op_.getHidden() / tile;
    const int64_t weightToIw = target_.throughput().vxm_weight_to_iw_latency;
    const int64_t activationLatency = *target_.transport_latency(
        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, layout.activationSlices().front());
    const int64_t projectionHeads[] = {
        op_.getQueryHeads(), op_.getKvHeads(), op_.getKvHeads()};
    const mlir::Value projectionValues[] = {
        op_.getQueryWeight(), op_.getKeyWeight(), op_.getValueWeight()};
    int64_t phaseStart = 0;

    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    const auto emitDequant = [&](int64_t cycle, int64_t hemisphere,
                                 int64_t localMxm, mlir::Value weight) {
        const char* hemi = hemisphere == 0 ? "east" : "west";
        for (int64_t lane = 0; lane < target_.throughput().lanes_per_tile; ++lane) {
            emitVxm(rewriter_, op_, weight, cycle, lane, "multiply",
                "stream_i8", 32 + lane, 0.0f, "immediate", 0, 1.0f,
                "fp32", -1, hemi, hemi);
            emitVxm(rewriter_, op_, weight, cycle + 1, 8 + lane, "cast",
                "alu", lane, 0.0f, "immediate", 0, 0.0f,
                "fp16", localMxm * 16 + lane * 2, hemi, hemi);
        }
    };
    const auto emitRopeOrCast = [&](int64_t cycle, int64_t hemisphere,
                                    bool rope, mlir::Value value) {
        const char* hemi = hemisphere == 0 ? "east" : "west";
        const int64_t aluBase = hemisphere * 8;
        if (!rope) {
            emitVxm(rewriter_, op_, value, cycle, aluBase, "pass", "stream_f32", 32,
                0.0f, "immediate", 0, 0.0f, "fp16", 0, hemi, hemi);
            emitVxm(rewriter_, op_, value, cycle, aluBase + 1, "pass", "stream_f32", 36,
                0.0f, "immediate", 0, 0.0f, "fp16", 2, hemi, hemi);
            return;
        }
        emitVxm(rewriter_, op_, value, cycle, aluBase, "multiply", "stream_f32", 32,
            0.0f, "stream_f16", 40, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle, aluBase + 1, "multiply", "stream_f32", 36,
            0.0f, "stream_f16", 42, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle, aluBase + 3, "multiply", "stream_f32", 36,
            0.0f, "stream_f16", 40, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle, aluBase + 4, "multiply", "stream_f32", 32,
            0.0f, "stream_f16", 42, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle + 1, aluBase + 2, "subtract", "alu", aluBase,
            0.0f, "alu", aluBase + 1, 0.0f, "fp16", 0, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle + 1, aluBase + 5, "add", "alu", aluBase + 3,
            0.0f, "alu", aluBase + 4, 0.0f, "fp16", 2, hemi, hemi);
    };

    const int64_t weightLoadLead =
        (target_.memory().hemispheres - 1) * 8 + 7 + weightToIw + 1;
    int64_t projectionBlock = 0;
    for (int64_t projection = 0; projection < 3; ++projection) {
        const auto kind = projectionKind(projection);
        for (int64_t headBase = 0; headBase < projectionHeads[projection]; headBase += 2) {
            for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks; ++reductionBlock) {
                const int64_t firstCompute = reductionBlock == 0
                    ? phaseStart + readLatency(layout.weightSlices().back())
                        + weightLoadLead
                    : phaseStart;
                const int64_t dequantStart = firstCompute - weightLoadLead;
                const int64_t weightBuffer =
                    projectionBlock % target_.throughput().mxm_weight_buffers;
                for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                    const int64_t head = headBase + hemisphere;
                    if (head >= projectionHeads[projection]) continue;
                    for (int64_t pulse = 0; pulse < 8; ++pulse) {
                        const int64_t localMxm = pulse / 4;
                        const int64_t column = 3 - pulse % 4;
                        const int64_t cycle = dequantStart + hemisphere * 8 + pulse;
                        const int64_t address = layout.weightAddress(kind, head,
                            reductionBlock, localMxm, pulse % 4);
                        for (int64_t stream = 0; stream < 8; ++stream) {
                            const int64_t slice = layout.weightSlices()[stream];
                            emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                                hemisphere * target_.memory().slices_per_hemisphere + slice,
                                "read", address, 32 + stream, 1, 1, 0);
                        }
                        emitDequant(cycle, hemisphere, localMxm,
                            projectionValues[projection]);
                        emitMxm(rewriter_, op_.getLoc(), cycle + weightToIw,
                            hemisphere * 2 + localMxm, "iw", weightBuffer, column,
                            0, 0, 1, 1);
                    }
                }

                const bool finalReduction = reductionBlock + 1 == hiddenBlocks;
                // Final projection tiles feed MXM results directly through
                // RoPE/cast into a 32-slice packed MEM layout. The following
                // tile cannot read its activation planes while that packed
                // writeback owns the same MEM queues.
                const int64_t projectionWritebackDrain =
                    finalReduction ? tile : 0;
                const int64_t computeBlockCycles =
                    target_.mxm_block_issue_interval()
                    + projectionWritebackDrain;
                for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
                    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                        const int64_t head = headBase + hemisphere;
                        if (head >= projectionHeads[projection]) continue;
                        const int64_t computeCycle = firstCompute
                            + tokenBlock * computeBlockCycles;
                        const int64_t inputAddress = layout.activationAddress(
                            reductionBlock, tokenBlock);
                        const int64_t outputAddress = layout.projectionAddress(
                            kind, head, tokenBlock);
                        llvm::SmallVector<int64_t> segmentRows;
                        llvm::SmallVector<int64_t> segmentStreams;
                        const bool prefetchNextWeight = !finalReduction
                            && tokenBlock + 1 == tokenBlocks;
                        if (prefetchNextWeight) {
                            const int64_t nextFirstCompute =
                                firstCompute + tokenBlocks * computeBlockCycles;
                            const int64_t nextDequantStart =
                                nextFirstCompute - weightLoadLead;
                            const int64_t switchRow = nextDequantStart
                                + hemisphere * 8 + weightToIw - computeCycle;
                            if (switchRow > 0) {
                                segmentRows.push_back(switchRow);
                                segmentStreams.push_back(0);
                            }
                            const int64_t switchedRows = std::min<int64_t>(
                                4, tile - switchRow);
                            segmentRows.push_back(switchedRows);
                            segmentStreams.push_back(
                                target_.throughput().mxm_load_streams_per_cycle);
                            if (switchRow + switchedRows < tile) {
                                segmentRows.push_back(
                                    tile - switchRow - switchedRows);
                                segmentStreams.push_back(0);
                            }
                        } else {
                            segmentRows.push_back(tile);
                            segmentStreams.push_back(0);
                        }
                        int64_t rowOffset = 0;
                        for (std::size_t segment = 0;
                             segment < segmentRows.size(); ++segment) {
                            const int64_t rows = segmentRows[segment];
                            const int64_t streamBase = segmentStreams[segment];
                            const int64_t segmentCycle = computeCycle + rowOffset;
                            for (int64_t byte = 0; byte < 4; ++byte) {
                                emitMem(rewriter_, op_.getLoc(),
                                    segmentCycle - activationLatency,
                                    hemisphere * target_.memory().slices_per_hemisphere
                                        + layout.activationSlices()[byte],
                                    "read", inputAddress + rowOffset,
                                    streamBase + byte, rows, 1, 1);
                            }
                            emitMxm(rewriter_, op_.getLoc(), segmentCycle,
                                hemisphere * 2, "compute", weightBuffer, 0,
                                streamBase, 0, rows, 1);
                            emitMxm(rewriter_, op_.getLoc(), segmentCycle,
                                hemisphere * 2 + 1, "compute", weightBuffer, 0,
                                streamBase + 2, 4, rows, 1);
                            rowOffset += rows;
                        }
                        const char* destination = finalReduction ? "stream" : "sram";
                        if (reductionBlock == 0) {
                            for (int64_t byte = 0; byte < 4; ++byte) {
                                emitMem(rewriter_, op_.getLoc(), computeCycle
                                        + target_.throughput().mxm0_accumulator_latency,
                                    hemisphere * target_.memory().slices_per_hemisphere
                                        + target_.memory().accumulator_slice_base + byte,
                                    "write", outputAddress, 32 + byte,
                                    tile, 1, 1, "sram");
                                emitMem(rewriter_, op_.getLoc(), computeCycle
                                        + target_.throughput().mxm1_accumulator_latency,
                                    hemisphere * target_.memory().slices_per_hemisphere
                                        + target_.memory().accumulator_slice_base + 4 + byte,
                                    "write", outputAddress, 36 + byte,
                                    tile, 1, 1, "sram");
                            }
                        } else {
                            emitMem(rewriter_, op_.getLoc(), computeCycle
                                    + target_.throughput().mxm0_accumulator_latency,
                                hemisphere * target_.memory().slices_per_hemisphere
                                    + target_.memory().accumulator_slice_base,
                                "accumulate", outputAddress, 32,
                                tile, 1, 1, destination);
                            emitMem(rewriter_, op_.getLoc(), computeCycle
                                    + target_.throughput().mxm1_accumulator_latency,
                                hemisphere * target_.memory().slices_per_hemisphere
                                    + target_.memory().accumulator_slice_base + 4,
                                "accumulate", outputAddress, 36,
                                tile, 1, 1, destination);
                        }
                        if (!finalReduction) continue;

                        for (int64_t offset = 0; offset < tile; ++offset) {
                            const int64_t token = tokenBlock * tile + offset;
                            const int64_t vxmCycle = computeCycle
                                + target_.throughput().accumulator_to_vxm_latency + offset;
                            if (kind != AttentionProjectionKind::Value) {
                                for (int64_t byte = 0; byte < 4; ++byte) {
                                    const int64_t slice = layout.ropeSlices()[byte];
                                    emitMem(rewriter_, op_.getLoc(), vxmCycle - readLatency(slice),
                                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                                        "read", layout.ropeAddress(token), 40 + byte, 1, 1, 0);
                                }
                            }
                            emitRopeOrCast(vxmCycle, hemisphere,
                                kind != AttentionProjectionKind::Value,
                                projectionValues[projection]);
                            const int64_t writeCycle = vxmCycle
                                + (kind == AttentionProjectionKind::Value ? 1 : 2);
                            if (kind == AttentionProjectionKind::Query) {
                                const int64_t phase = (token % tile) / 8;
                                const int64_t localColumn = token % 8;
                                for (int64_t reduction = 0; reduction < 2; ++reduction) {
                                    const auto& slices = target_.attention_query_iw_slices(reduction);
                                    for (int64_t byte = 0; byte < 2; ++byte) {
                                        const int64_t stream = reduction * 2 + byte;
                                        const int64_t slice = slices[localColumn * 2 + byte];
                                        emitMem(rewriter_, op_.getLoc(), writeCycle + slice / 4,
                                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                                            "write", layout.queryIwAddress(head, tokenBlock, phase),
                                            stream, 1, 1, 0);
                                    }
                                }
                            } else if (kind == AttentionProjectionKind::Key) {
                                for (int64_t byte = 0; byte < 4; ++byte)
                                    emitMem(rewriter_, op_.getLoc(), writeCycle,
                                        hemisphere * target_.memory().slices_per_hemisphere + byte,
                                        "write", outputAddress + offset, byte, 1, 1, 0);
                            } else {
                                const int64_t packedStream = (token % 8) * 2;
                                const int64_t row = (token % tile) / 8;
                                for (int64_t reduction = 0; reduction < 2; ++reduction) {
                                    const auto slices = layout.valuePackSlices(reduction);
                                    for (int64_t byte = 0; byte < 2; ++byte) {
                                        const int64_t slice = slices[packedStream + byte];
                                        emitMem(rewriter_, op_.getLoc(), writeCycle
                                                + slice / target_.streams().mem_slices_per_register_group,
                                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                                            "write", layout.valuePackAddress(head, reduction,
                                                tokenBlock, row), reduction * 2 + byte,
                                            1, 1, 0);
                                    }
                                }
                            }
                        }
                    }
                }
                phaseStart = firstCompute + tokenBlocks * computeBlockCycles;
                ++projectionBlock;
            }
        }
    }
    return phaseStart;
}

void AttentionScheduleEmitter::emitQk(int64_t qkStart,
    int64_t qkWaveCycles, int64_t qkIwToComputeCycles)
{
    const AttentionMemoryLayout layout(op_, target_);
    const AttentionWorkPlanner plan({static_cast<int64_t>(op_.getSeqLen()),
        static_cast<int64_t>(op_.getQueryHeads()), static_cast<int64_t>(op_.getKvHeads()),
        static_cast<int64_t>(op_.getHeadDim())}, target_);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const int64_t issue = target_.mxm_block_issue_interval();

    for (std::size_t waveIndex = 0; waveIndex < plan.qk_waves().size(); ++waveIndex) {
        const int64_t waveStart = qkStart + static_cast<int64_t>(waveIndex) * qkWaveCycles;
        const int64_t firstIwCycle = waveStart + target_.throughput().mxm_earliest_iw_cycle
            + *target_.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, 0);
        const int64_t firstComputeCycle = firstIwCycle + qkIwToComputeCycles;
        for (const auto& work : plan.qk_waves()[waveIndex].slots) {
            if (!work) continue;
            const int64_t mxm = work->hemisphere * target_.throughput().mxms_per_hemisphere
                + work->local_mxm;
            const int64_t activationStream = work->local_mxm * 2;
            const int64_t outputStream = work->local_mxm * 4;
            const int64_t accumulatorSlice = target_.memory().accumulator_slice_base
                + work->local_mxm * target_.memory().accumulator_slices_per_mxm;
            const int64_t accumulatorLatency = work->local_mxm == 0
                ? target_.throughput().mxm0_accumulator_latency
                : target_.throughput().mxm1_accumulator_latency;
            const int64_t keySliceBase = work->local_mxm == 0 ? 0 : 4;
            for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
                const auto& iwSlices = target_.attention_query_iw_slices(reduction);
                const int64_t reductionIwCycle = firstIwCycle
                    + work->local_mxm * 8 + reduction * tile / 8;
                for (int64_t phase = 0; phase < tile / 8; ++phase) {
                    const int64_t iwCycle = reductionIwCycle + phase;
                    const int64_t sourcePhase = tile / 8 - 1 - phase;
                    const int64_t queryAddress = layout.queryIwAddress(
                        work->query_head, work->query_block, sourcePhase);
                    for (int64_t stream = 0; stream < static_cast<int64_t>(iwSlices.size()); ++stream) {
                        const int64_t slice = iwSlices[stream];
                        const int64_t latency = *target_.transport_latency(
                            target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                            target::StreamDirection::East, slice);
                        emitMem(rewriter_, op_.getLoc(), iwCycle - latency,
                            work->hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", queryAddress,
                            work->local_mxm * static_cast<int64_t>(iwSlices.size()) + stream,
                            1, 1, 0);
                    }
                    emitMxm(rewriter_, op_.getLoc(), iwCycle, mxm, "iw",
                        reduction, phase, 0, 0, 1, 1);
                }
            }
            for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
                for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
                    const int64_t computeCycle = firstComputeCycle
                        + (reduction * tokenBlocks + keyBlock) * issue;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = keySliceBase + reduction * 2 + byte;
                        const int64_t latency = *target_.transport_latency(
                            target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
                            target::StreamDirection::East, slice);
                        emitMem(rewriter_, op_.getLoc(), computeCycle - latency,
                            work->hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", layout.keyAddress(work->kv_head, keyBlock),
                            activationStream + byte, tile, 1, 0);
                    }
                    emitMxm(rewriter_, op_.getLoc(), computeCycle, mxm, "compute",
                        reduction, 0, activationStream, outputStream, tile, 1);
                    emitMem(rewriter_, op_.getLoc(), computeCycle + accumulatorLatency,
                        work->hemisphere * target_.memory().slices_per_hemisphere + accumulatorSlice,
                        "accumulate", layout.scoreAddress(work->query_head,
                            work->query_block, keyBlock),
                        32 + outputStream, tile, 1, 0);
                }
            }
        }
    }
}

int64_t AttentionScheduleEmitter::emitSoftmax(int64_t qkEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const AttentionWorkPlanner plan({static_cast<int64_t>(op_.getSeqLen()),
        static_cast<int64_t>(op_.getQueryHeads()), static_cast<int64_t>(op_.getKvHeads()),
        static_cast<int64_t>(op_.getHeadDim())}, target_);
    const float scale = 1.0f / std::sqrt(static_cast<float>(op_.getHeadDim()));
    constexpr float causalMaskValue = -1.0e9f;
    constexpr int64_t outputStream = 8;
    const int64_t softmaxDuration = 3 * op_.getSeqLen() + 20;
    ScheduleGraph graph;
    ResourceScheduler scheduler;
    LPUResourceModel resources(target_);
    std::vector<std::array<std::optional<ScheduleNodeId>, 2>> waveNodes(
        plan.qk_waves().size());
    std::array<std::optional<ScheduleNodeId>, 2> previousNode;
    for (std::size_t waveIndex = 0; waveIndex < plan.qk_waves().size(); ++waveIndex) {
        const AttentionWorkWave& wave = plan.qk_waves()[waveIndex];
        for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
             ++hemisphere) {
            llvm::SmallVector<ResourceWindow, 32> windows;
            bool hasWork = false;
            for (const auto& work : wave.slots) {
                if (!work || work->hemisphere != hemisphere) continue;
                hasWork = true;
                const int64_t lane = work->local_mxm;
                const int64_t aluBase = hemisphere * 6 + lane * 3;
                for (int64_t alu = 0; alu < 3; ++alu)
                    windows.push_back(
                        {resources.vxm_alu(aluBase + alu), 0, softmaxDuration});
                const auto addMemResources = [&](llvm::ArrayRef<int64_t> slices) {
                    for (int64_t slice : slices)
                        windows.push_back(
                            {resources.mem_slice(hemisphere, slice), 0, softmaxDuration});
                };
                addMemResources(layout.scaledScoreSlices(lane));
                addMemResources(layout.expScoreSlices(lane));
                addMemResources(layout.causalMaskSlices(lane));
                addMemResources(layout.probabilitySlices(lane));
            }
            if (!hasWork) continue;
            const auto node = graph.add_node(
                "softmax.wave" + std::to_string(waveIndex) + ".h"
                    + std::to_string(hemisphere),
                qkEnd + 16, softmaxDuration, windows);
            if (previousNode[static_cast<std::size_t>(hemisphere)]
                && mlir::failed(graph.add_dependency(
                    *previousNode[static_cast<std::size_t>(hemisphere)], node))) {
                op_.emitError("failed to construct the softmax schedule graph");
                return qkEnd;
            }
            previousNode[static_cast<std::size_t>(hemisphere)] = node;
            waveNodes[waveIndex][static_cast<std::size_t>(hemisphere)] = node;
        }
    }
    auto scheduled = graph.schedule(scheduler);
    if (mlir::failed(scheduled)) {
        op_.emitError("softmax schedule graph contains a cycle or invalid duration");
        return qkEnd;
    }
    std::array<int64_t, 2> softmaxReady {qkEnd + 16, qkEnd + 16};
    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    const auto vxm = [&](int64_t cycle, int64_t queue, llvm::StringRef opcode,
                         llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
                         llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
                         llvm::StringRef castTarget, int64_t stream, int64_t hemisphere) {
        const char* hemi = hemisphere == 0 ? "east" : "west";
        emitVxm(rewriter_, op_, op_.getInput(), cycle,
            queue, opcode,
            lhsKind, lhsIndex, lhsImmediate, rhsKind, rhsIndex, rhsImmediate,
            castTarget, stream, hemi, hemi);
    };

    for (std::size_t waveIndex = 0; waveIndex < plan.qk_waves().size(); ++waveIndex) {
        const AttentionWorkWave& wave = plan.qk_waves()[waveIndex];
        for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
             ++hemisphere) {
            llvm::SmallVector<const AttentionWorkItem*> workLines;
            for (const auto& work : wave.slots)
                if (work && work->hemisphere == hemisphere) workLines.push_back(&*work);
            if (workLines.empty()) continue;

            const auto node = waveNodes[waveIndex][static_cast<std::size_t>(hemisphere)];
            if (!node) continue;
            const int64_t softmaxCycle = (*scheduled)[*node].cycle;
            const int64_t pass3End = softmaxCycle + 3 * op_.getSeqLen() + 20;
            for (const AttentionWorkItem* work : workLines) {
            const int64_t lane = work->local_mxm;
            const int64_t aluBase = hemisphere * 6 + lane * 3;
            const int64_t inputStream = 32 + lane * 4;
            const int64_t maskStream = 40 + lane * 4;
            const int64_t outputStreamBase = outputStream + lane * 4;
            const int64_t accumulatorBase = target_.memory().accumulator_slice_base
                + lane * target_.memory().accumulator_slices_per_mxm;

            for (int64_t key = 0; key < op_.getSeqLen(); ++key) {
                const int64_t cycle = softmaxCycle + key;
                const int64_t keyBlock =
                    key / target_.throughput().mxm_rows;
                const int64_t localKey =
                    key % target_.throughput().mxm_rows;
                const bool vectorMask = op_.getCausal()
                    && keyBlock == work->query_block && localKey != 0;
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = accumulatorBase + byte;
                    emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "read", layout.scoreTokenAddress(work->query_head,
                            work->query_block, key), inputStream + byte, 1, 1, 0);
                }
                if (vectorMask) {
                    for (int64_t byte = 0; byte < 4; ++byte) {
                        const int64_t slice = layout.causalMaskSlices(lane)[byte];
                        emitMem(rewriter_, op_.getLoc(),
                            cycle + 1 - readLatency(slice),
                            hemisphere * target_.memory().slices_per_hemisphere
                                + slice,
                            "read", layout.causalMaskAddress(localKey),
                            maskStream + byte, 1, 1, 0);
                    }
                }
                vxm(cycle, aluBase, "multiply", "stream_f32", inputStream, 0.0f,
                    "immediate", 0, scale, "fp32",
                    op_.getCausal() ? -1 : outputStreamBase, hemisphere);
                if (op_.getCausal()) {
                    const float immediateMask =
                        keyBlock > work->query_block ? causalMaskValue : 0.0f;
                    vxm(cycle + 1, aluBase + 1, "add", "alu", aluBase, 0.0f,
                        vectorMask ? "stream_f32" : "immediate",
                        vectorMask ? maskStream : 0, immediateMask,
                        "fp32", outputStreamBase, hemisphere);
                    vxm(cycle + 2, aluBase + 2, key == 0 ? "pass" : "max",
                        "alu", key == 0 ? aluBase + 1 : aluBase + 2, 0.0f,
                        key == 0 ? "immediate" : "alu",
                        key == 0 ? 0 : aluBase + 1, 0.0f,
                        "fp32", -1, hemisphere);
                } else {
                    vxm(cycle + 1, aluBase + 2, key == 0 ? "pass" : "max",
                        "alu", key == 0 ? aluBase : aluBase + 2, 0.0f,
                        key == 0 ? "immediate" : "alu",
                        key == 0 ? 0 : aluBase, 0.0f,
                        "fp32", -1, hemisphere);
                }
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = layout.scaledScoreSlices(lane)[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle
                            + (op_.getCausal() ? 2 : 1)
                            + slice / target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "write", layout.scaledScoreAddress(key), outputStreamBase + byte,
                        1, 1, 0);
                }
            }

            const int64_t pass2Start = softmaxCycle + op_.getSeqLen() + 8;
            for (int64_t key = 0; key < op_.getSeqLen(); ++key) {
                const int64_t cycle = pass2Start + key;
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = layout.scaledScoreSlices(lane)[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "read", layout.scaledScoreAddress(key), inputStream + byte, 1, 1, 0);
                }
                vxm(cycle, aluBase, "subtract", "stream_f32", inputStream, 0.0f,
                    "alu", aluBase + 2, 0.0f, "fp32", -1, hemisphere);
                vxm(cycle + 1, aluBase + 1, "exp", "alu", aluBase, 0.0f,
                    "immediate", 0, 0.0f, "fp32", outputStreamBase, hemisphere);
                vxm(cycle + 2, aluBase + 2, key == 0 ? "pass" : "add",
                    "alu", key == 0 ? aluBase + 1 : aluBase + 2, 0.0f,
                    key == 0 ? "immediate" : "alu", key == 0 ? 0 : aluBase + 1, 0.0f,
                    "fp32", -1, hemisphere);
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = layout.expScoreSlices(lane)[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle + 2
                            + slice / target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "write", layout.expScoreAddress(key), outputStreamBase + byte,
                        1, 1, 0);
                }
            }

            const int64_t pass3Start = pass2Start + op_.getSeqLen() + 12;
            for (int64_t key = 0; key < op_.getSeqLen(); ++key) {
                const int64_t cycle = pass3Start + key;
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = layout.expScoreSlices(lane)[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "read", layout.expScoreAddress(key), inputStream + byte, 1, 1, 0);
                }
                vxm(cycle, aluBase, "divide", "stream_f32", inputStream, 0.0f,
                    "alu", aluBase + 2, 0.0f, "fp32", -1, hemisphere);
                vxm(cycle + 1, aluBase + 1, "cast", "alu", aluBase, 0.0f,
                    "immediate", 0, 0.0f, "fp16", outputStreamBase, hemisphere);
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = layout.probabilitySlices(lane)[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle + 2
                            + slice / target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "write", layout.probabilityAddress(work->query_head,
                            work->query_block, key), outputStreamBase + byte,
                        1, 1, 0);
                }
            }
            }
            // Mask and probability use disjoint planes, so a new wave can
            // start as soon as the preceding wave's final write is issued.
            softmaxReady[static_cast<std::size_t>(hemisphere)] =
                std::max(softmaxReady[static_cast<std::size_t>(hemisphere)],
                    pass3End);
        }
    }
    // Probability-pack reads begin with a six-cycle MEM lead. Preserve the
    // final probability-write tail before that next phase turns the planes
    // around.
    return std::max(softmaxReady[0], softmaxReady[1]) + 16;
}

int64_t AttentionScheduleEmitter::emitProbabilityPack(int64_t softmaxEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const AttentionWorkPlanner planner({static_cast<int64_t>(op_.getSeqLen()),
        static_cast<int64_t>(op_.getQueryHeads()), static_cast<int64_t>(op_.getKvHeads()),
        static_cast<int64_t>(op_.getHeadDim())}, target_);
    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    // SoftmaxEnd already includes the final probability-write tail. Two
    // cycles are sufficient to turn the same byte planes around for reading.
    int64_t cycle = softmaxEnd + 2;
    for (const auto& wave : planner.qk_waves()) {
        for (const auto& work : wave.slots) {
            if (!work) continue;
            const char* hemisphere = work->hemisphere == 0 ? "east" : "west";
            for (int64_t key = 0; key < op_.getSeqLen(); ++key, ++cycle) {
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = layout.probabilitySlices(work->local_mxm)[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                        work->hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "read", layout.probabilityAddress(work->query_head,
                            work->query_block, key), 48 + byte, 1, 1, 0);
                    emitVxm(rewriter_, op_, op_.getInput(), cycle, byte, "pass",
                        "stream_i8", 48 + byte, 0.0f,
                        "immediate", 0, 0.0f, "i8", byte,
                        hemisphere, hemisphere);
                }
                const int64_t packedStream = (key % target_.throughput().lanes_per_tile) * 2;
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = layout.probabilityPackSlices()[packedStream + byte];
                    emitMem(rewriter_, op_.getLoc(), cycle + 1
                            + slice / target_.streams().mem_slices_per_register_group,
                        work->hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "write", layout.probabilityPackAddress(work->query_head,
                            work->query_block,
                            key / target_.throughput().lanes_per_tile),
                        byte, 1, 1, 0);
                }
            }
        }
    }
    const int64_t finalPackWriteTail =
        layout.probabilityPackSlices().back()
            / target_.streams().mem_slices_per_register_group
        + 1;
    return cycle + finalPackWriteTail;
}

int64_t AttentionScheduleEmitter::emitProbabilityTranspose(int64_t packEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const AttentionWorkPlanner planner({static_cast<int64_t>(op_.getSeqLen()),
        static_cast<int64_t>(op_.getQueryHeads()), static_cast<int64_t>(op_.getKvHeads()),
        static_cast<int64_t>(op_.getHeadDim())}, target_);
    const int64_t groups = target_.memory().slices_per_hemisphere
        / target_.streams().mem_slices_per_register_group;
    const int64_t memToSxm = target_.throughput().mem_to_sxm_latency;
    const int64_t tokenBlocks = op_.getSeqLen() / target_.throughput().mxm_rows;
    std::array<int64_t, 2> ready {packEnd, packEnd};
    std::array<int64_t, 16> inputStreams {};
    std::array<int64_t, 16> transposeStreams {};
    std::array<int64_t, 16> outputStreams {};
    for (int64_t stream = 0; stream < 16; ++stream) {
        inputStreams[static_cast<std::size_t>(stream)] = stream;
        transposeStreams[static_cast<std::size_t>(stream)] = 16 + stream;
        outputStreams[static_cast<std::size_t>(stream)] = 32 + stream;
    }
    const auto identity = identityMap();

    for (const auto& wave : planner.qk_waves()) {
        for (const auto& work : wave.slots) {
            if (!work) continue;
            const int64_t hemisphere = work->hemisphere;
            for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
                const int64_t start = ready[static_cast<std::size_t>(hemisphere)];
                const int64_t capture = start + memToSxm;
                for (int64_t beat = 0; beat < target_.throughput().tile_rows; ++beat) {
                    for (int64_t stream = 0; stream < 16; ++stream) {
                        const int64_t slice = layout.probabilityPackSlices()[stream];
                        const int64_t latency = memToSxm
                            - slice / target_.streams().mem_slices_per_register_group;
                        emitMem(rewriter_, op_.getLoc(), capture + beat - latency,
                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", layout.probabilityPackAddress(work->query_head,
                                work->query_block,
                                keyBlock * target_.throughput().tile_rows + beat),
                            stream, 1, 1, 0);
                    }
                }
                for (int64_t sxmWave = 0;
                     sxmWave < target_.throughput().tile_rows; ++sxmWave) {
                    const int64_t cycle = capture + sxmWave;
                    emitSxm(rewriter_, op_.getLoc(), cycle, hemisphere, "transpose",
                        inputStreams, transposeStreams, identity);
                    const auto map = blockDiagonalMap(sxmWave, target_);
                    emitSxm(rewriter_, op_.getLoc(), cycle + 1, hemisphere, "permute",
                        transposeStreams, outputStreams, map);
                    for (int64_t stream = 0; stream < 16; ++stream) {
                        const int64_t slice = layout.probabilityDiagonalSlices()[stream];
                        emitMem(rewriter_, op_.getLoc(), cycle + 1 + groups
                                - slice / target_.streams().mem_slices_per_register_group,
                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "write", layout.probabilityDiagonalAddress(work->query_head,
                                work->query_block, keyBlock, sxmWave),
                            32 + stream, 1, 1, 0);
                    }
                }
                ready[static_cast<std::size_t>(hemisphere)] =
                    start + target_.throughput().tile_rows;
            }
        }
    }
    for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
         ++hemisphere) {
        for (int64_t tail = 0; tail < target_.throughput().tile_rows - 1; ++tail) {
            const int64_t cycle = ready[static_cast<std::size_t>(hemisphere)]
                + memToSxm + tail;
            emitSxm(rewriter_, op_.getLoc(), cycle, hemisphere, "transpose",
                inputStreams, transposeStreams, identity);
            const auto map = blockDiagonalMap(tail, target_);
            emitSxm(rewriter_, op_.getLoc(), cycle + 1, hemisphere, "permute",
                transposeStreams, outputStreams, map);
        }
        ready[static_cast<std::size_t>(hemisphere)] +=
            target_.throughput().tile_rows - 1;
    }
    return std::max(ready[0], ready[1]) + memToSxm + groups + 1;
}

int64_t AttentionScheduleEmitter::emitPv(int64_t transposeEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tileRows = target_.throughput().tile_rows;
    const int64_t lanes = target_.throughput().lanes_per_tile;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const int64_t queryHeadsPerKv = op_.getQueryHeads() / op_.getKvHeads();
    const int64_t groups = target_.memory().slices_per_hemisphere
        / target_.streams().mem_slices_per_register_group;
    const int64_t memToSxm = target_.throughput().mem_to_sxm_latency;
    const int64_t memToMxm = target_.throughput().mem_to_mxm_latency;
    int64_t contextWriteTail = 0;
    for (int64_t byte = 0; byte < 8; ++byte) {
        const int64_t slice = layout.contextSlices()[byte];
        contextWriteTail = std::max(contextWriteTail,
            (byte < 4 ? 1 : 2)
                + slice / target_.streams().mem_slices_per_register_group + 1);
    }
    const auto identity = identityMap();
    std::array<int64_t, 16> inputStreams {};
    std::array<int64_t, 16> transposeStreams {};
    for (int64_t stream = 0; stream < 16; ++stream) {
        inputStreams[static_cast<std::size_t>(stream)] = stream;
        transposeStreams[static_cast<std::size_t>(stream)] = 16 + stream;
    }

    // A PV wave owns both MXMs in a hemisphere, so place at most one query
    // head from each hemisphere in a wave. This remains shape-driven for GQA.
    std::vector<std::array<std::optional<int64_t>, 2>> waves;
    for (int64_t head = 0; head < op_.getQueryHeads(); ++head) {
        const int64_t kvHead = head / queryHeadsPerKv;
        const int64_t hemisphere = kvHead % target_.memory().hemispheres;
        bool placed = false;
        for (auto& wave : waves) {
            if (!wave[static_cast<std::size_t>(hemisphere)]) {
                wave[static_cast<std::size_t>(hemisphere)] = head;
                placed = true;
                break;
            }
        }
        if (!placed) {
            std::array<std::optional<int64_t>, 2> wave;
            wave[static_cast<std::size_t>(hemisphere)] = head;
            waves.push_back(wave);
        }
    }

    int64_t phaseStart = transposeEnd;
    for (const auto& wave : waves) {
        for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
            int64_t loadReady = phaseStart;
            for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
                 ++hemisphere) {
                const auto head = wave[static_cast<std::size_t>(hemisphere)];
                if (!head) continue;
                const int64_t kvHead = *head / queryHeadsPerKv;
                int64_t routeStart = phaseStart;
                for (int64_t localMxm = 0; localMxm < headBlocks; ++localMxm) {
                    const int64_t capture = routeStart + memToSxm;
                    const auto slices = layout.valuePackSlices(localMxm);
                    for (int64_t beat = 0; beat < tileRows; ++beat) {
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice = slices[static_cast<std::size_t>(stream)];
                            const int64_t latency = memToSxm
                                - slice / target_.streams().mem_slices_per_register_group;
                            emitMem(rewriter_, op_.getLoc(), capture + beat - latency,
                                hemisphere * target_.memory().slices_per_hemisphere + slice,
                                "read", layout.valuePackAddress(
                                    kvHead, localMxm, keyBlock, beat),
                                stream, 1, 1, 0);
                        }
                    }
                    std::array<int64_t, 16> mxmStreams {};
                    for (int64_t stream = 0; stream < 16; ++stream)
                        mxmStreams[static_cast<std::size_t>(stream)] = localMxm * 16 + stream;
                    for (int64_t waveIndex = 0; waveIndex < tileRows; ++waveIndex) {
                        const int64_t cycle = capture + waveIndex;
                        emitSxm(rewriter_, op_.getLoc(), cycle, hemisphere, "transpose",
                            inputStreams, transposeStreams, identity);
                        const auto map = blockDiagonalMap(waveIndex, target_);
                        emitSxm(rewriter_, op_.getLoc(), cycle + 1, hemisphere, "permute",
                            transposeStreams, mxmStreams, map, "matrix_columns");
                        emitMxm(rewriter_, op_.getLoc(), cycle + 2,
                            hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
                            "iw", 0, waveIndex, 0, 0, 1, 1);
                    }
                    for (int64_t tail = 0; tail < tileRows - 1; ++tail) {
                        const int64_t waveIndex = tileRows + tail;
                        const int64_t cycle = capture + waveIndex;
                        emitSxm(rewriter_, op_.getLoc(), cycle, hemisphere, "transpose",
                            inputStreams, transposeStreams, identity);
                        const auto map = blockDiagonalMap(waveIndex, target_);
                        emitSxm(rewriter_, op_.getLoc(), cycle + 1, hemisphere, "permute",
                            transposeStreams, mxmStreams, map, "matrix_columns");
                    }
                    loadReady = std::max(loadReady, capture + 2 * tileRows + 1);
                    routeStart += 2 * tileRows - 1;
                }
            }
            phaseStart = loadReady + 8;

            for (int64_t queryBlock = 0; queryBlock < tokenBlocks; ++queryBlock) {
                int64_t blockEnd = phaseStart;
                for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const auto head = wave[static_cast<std::size_t>(hemisphere)];
                    if (!head) continue;
                    const int64_t replayStart = phaseStart;
                    const int64_t firstCompute = replayStart + memToMxm;
                    for (int64_t query = 0; query < tile; ++query) {
                        const int64_t row = query % lanes;
                        const int64_t diagonal = query / lanes;
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = layout.probabilityDiagonalSlices()[row * 2 + byte];
                            const int64_t latency = *target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmActivation,
                                target::StreamDirection::East, slice);
                            emitMem(rewriter_, op_.getLoc(), firstCompute + query - latency,
                                hemisphere * target_.memory().slices_per_hemisphere + slice,
                                "read", layout.probabilityDiagonalAddress(
                                    *head, queryBlock, keyBlock, diagonal),
                                byte, 1, 1, 0);
                        }
                    }

                    for (int64_t localMxm = 0; localMxm < headBlocks; ++localMxm) {
                        const int64_t accumulatorSlice = target_.memory().accumulator_slice_base
                            + localMxm * target_.memory().accumulator_slices_per_mxm;
                        const int64_t accumulatorLatency = localMxm == 0
                            ? target_.throughput().mxm0_accumulator_latency
                            : target_.throughput().mxm1_accumulator_latency;
                        const int64_t outputStream = localMxm * 4;
                        if (keyBlock == 0) {
                            for (int64_t byte = 0; byte < 4; ++byte)
                                emitMem(rewriter_, op_.getLoc(), firstCompute + accumulatorLatency,
                                    hemisphere * target_.memory().slices_per_hemisphere
                                        + accumulatorSlice + byte,
                                    "write", layout.contextAddress(*head, queryBlock * tile),
                                    32 + outputStream + byte, tile, 1, 1);
                        } else {
                            emitMem(rewriter_, op_.getLoc(), firstCompute + accumulatorLatency,
                                hemisphere * target_.memory().slices_per_hemisphere
                                    + accumulatorSlice,
                                "accumulate", layout.contextAddress(*head, queryBlock * tile),
                                32 + outputStream, tile, 1, 1, "sram");
                        }
                        emitMxm(rewriter_, op_.getLoc(), firstCompute,
                            hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
                            "compute", 0, 0, 0, outputStream, tile, 1);
                    }

                    if (keyBlock + 1 == tokenBlocks) {
                        const int64_t readStart = firstCompute
                            + target_.throughput().mxm0_accumulator_latency + tile + 13
                            + hemisphere * (tile + 16);
                        const int64_t aluBase = hemisphere == 0 ? 8 : 12;
                        const int64_t contextOutputStreamBase = 8;
                        for (int64_t offset = 0; offset < tile; ++offset) {
                            const int64_t cycle = readStart + offset;
                            for (int64_t localMxm = 0; localMxm < headBlocks; ++localMxm) {
                                for (int64_t byte = 0; byte < 4; ++byte) {
                                    const int64_t slice = target_.memory().accumulator_slice_base
                                        + localMxm * target_.memory().accumulator_slices_per_mxm + byte;
                                    const int64_t latency = *target_.transport_latency(
                                        target::StreamEndpoint::Mem,
                                        target::StreamEndpoint::VxmInput,
                                        target::StreamDirection::West, slice);
                                    emitMem(rewriter_, op_.getLoc(), cycle - latency,
                                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                                        "read", layout.contextAddress(
                                            *head, queryBlock * tile + offset),
                                        32 + localMxm * 4 + byte, 1, 1, 0);
                                }
                            }
                            emitVxm(rewriter_, op_, op_.getInput(), cycle, aluBase,
                                "pass", "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase,
                                hemisphere == 0 ? "east" : "west",
                                hemisphere == 0 ? "east" : "west");
                            emitVxm(rewriter_, op_, op_.getInput(), cycle, aluBase + 1,
                                "pass", "stream_f32", 36, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase + 2,
                                hemisphere == 0 ? "east" : "west",
                                hemisphere == 0 ? "east" : "west");
                            emitVxm(rewriter_, op_, op_.getInput(), cycle + 1, aluBase + 2,
                                "pass", "alu", aluBase, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase + 4,
                                hemisphere == 0 ? "east" : "west",
                                hemisphere == 0 ? "east" : "west");
                            emitVxm(rewriter_, op_, op_.getInput(), cycle + 1, aluBase + 3,
                                "pass", "alu", aluBase + 1, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase + 6,
                                hemisphere == 0 ? "east" : "west",
                                hemisphere == 0 ? "east" : "west");
                            const int64_t mirrorAluBase = hemisphere == 0 ? 0 : 4;
                            const char* inputHemisphere = hemisphere == 0 ? "east" : "west";
                            const char* mirrorHemisphere = hemisphere == 0 ? "west" : "east";
                            emitVxm(rewriter_, op_, op_.getInput(), cycle, mirrorAluBase,
                                "pass", "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase,
                                inputHemisphere, mirrorHemisphere);
                            emitVxm(rewriter_, op_, op_.getInput(), cycle, mirrorAluBase + 1,
                                "pass", "stream_f32", 36, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase + 2,
                                inputHemisphere, mirrorHemisphere);
                            emitVxm(rewriter_, op_, op_.getInput(), cycle + 1, mirrorAluBase + 2,
                                "pass", "alu", mirrorAluBase, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase + 4,
                                inputHemisphere, mirrorHemisphere);
                            emitVxm(rewriter_, op_, op_.getInput(), cycle + 1, mirrorAluBase + 3,
                                "pass", "alu", mirrorAluBase + 1, 0.0f, "immediate", 0, 0.0f,
                                "fp16", contextOutputStreamBase + 6,
                                inputHemisphere, mirrorHemisphere);
                            for (int64_t destinationHemisphere = 0;
                                 destinationHemisphere < target_.memory().hemispheres;
                                 ++destinationHemisphere) {
                                for (int64_t byte = 0; byte < 8; ++byte) {
                                    const int64_t slice = layout.contextSlices()[byte];
                                    emitMem(rewriter_, op_.getLoc(), cycle + (byte < 4 ? 1 : 2)
                                            + slice / target_.streams().mem_slices_per_register_group,
                                        destinationHemisphere
                                                * target_.memory().slices_per_hemisphere + slice,
                                        "write", layout.contextAddress(
                                            *head, queryBlock * tile + offset),
                                    contextOutputStreamBase + byte, 1, 1, 0);
                                }
                            }
                        }
                        blockEnd = std::max(blockEnd,
                            readStart + tile + contextWriteTail);
                    } else {
                        blockEnd = std::max(blockEnd,
                            firstCompute + tile - memToMxm);
                    }
                }
                phaseStart = blockEnd;
            }
        }
    }
    return phaseStart + groups;
}

int64_t AttentionScheduleEmitter::emitOutputProjection(int64_t pvEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t reductionBlocks = op_.getQueryHeads() * op_.getHeadDim() / tile;
    const int64_t outputGroups = op_.getHidden()
        / (tile * target_.memory().hemispheres);
    const int64_t localMxm = 0;
    const int64_t accumulatorSlice = target_.memory().accumulator_slice_base
        + localMxm * target_.memory().accumulator_slices_per_mxm;
    const int64_t accumulatorLatency = target_.throughput().mxm0_accumulator_latency;
    const int64_t weightToIw = target_.throughput().vxm_weight_to_iw_latency;
    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    const int64_t weightLoadLead =
        (target_.memory().hemispheres - 1) * 8 + 3 + weightToIw + 1;

    int64_t phaseStart = pvEnd;
    for (int64_t outputGroup = 0; outputGroup < outputGroups; ++outputGroup) {
        for (int64_t reductionBlock = 0; reductionBlock < reductionBlocks;
             ++reductionBlock) {
            const int64_t firstCompute = reductionBlock == 0
                ? phaseStart + readLatency(layout.outputWeightSlices().back())
                    + weightLoadLead
                : phaseStart;
            const int64_t dequantStart = firstCompute - weightLoadLead;
            const int64_t weightBuffer =
                (outputGroup * reductionBlocks + reductionBlock)
                % target_.throughput().mxm_weight_buffers;
            for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
                 ++hemisphere) {
                const char* hemi = hemisphere == 0 ? "east" : "west";
                for (int64_t pulse = 0; pulse < 4; ++pulse) {
                    const int64_t cycle = dequantStart + hemisphere * 8 + pulse;
                    for (int64_t stream = 0; stream < 8; ++stream) {
                        const int64_t slice = layout.outputWeightSlices()[stream];
                        emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", layout.outputWeightAddress(
                                outputGroup, reductionBlock, pulse),
                            32 + stream, 1, 1, 0);
                    }
                    for (int64_t lane = 0;
                         lane < target_.throughput().lanes_per_tile; ++lane) {
                        emitVxm(rewriter_, op_, op_.getOutputWeight(), cycle, lane,
                            "multiply", "stream_i8", 32 + lane, 0.0f,
                            "immediate", 0, 1.0f, "fp32", -1, hemi, hemi);
                        emitVxm(rewriter_, op_, op_.getOutputWeight(), cycle + 1,
                            8 + lane, "cast", "alu", lane, 0.0f,
                            "immediate", 0, 0.0f, "fp16",
                            localMxm * target_.throughput().mxm_load_streams_per_cycle
                                + lane * 2,
                            hemi, hemi);
                    }
                    emitMxm(rewriter_, op_.getLoc(), cycle + weightToIw,
                        hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
                        "iw", weightBuffer, 3 - pulse, 0, 0, 1, 1);
                }
            }

            const bool firstReduction = reductionBlock == 0;
            const bool finalReduction = reductionBlock + 1 == reductionBlocks;
            const int64_t computeInterval = tile;
            const int64_t queryHead = reductionBlock / (op_.getHeadDim() / tile);
            const int64_t headBlock = reductionBlock % (op_.getHeadDim() / tile);
            for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
                const int64_t computeBase = firstCompute + tokenBlock * computeInterval;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres; ++hemisphere) {
                    const int64_t computeCycle = computeBase;
                    llvm::SmallVector<int64_t> segmentRows;
                    llvm::SmallVector<int64_t> segmentStreams;
                    const bool prefetchNextWeight = !finalReduction
                        && tokenBlock + 1 == tokenBlocks;
                    if (prefetchNextWeight) {
                        const int64_t nextFirstCompute =
                            firstCompute + tokenBlocks * computeInterval;
                        const int64_t nextDequantStart =
                            nextFirstCompute - weightLoadLead;
                        const int64_t switchRow = nextDequantStart
                            + hemisphere * 8 + weightToIw - computeCycle;
                        if (switchRow > 0) {
                            segmentRows.push_back(switchRow);
                            segmentStreams.push_back(0);
                        }
                        const int64_t switchedRows =
                            std::min<int64_t>(4, tile - switchRow);
                        segmentRows.push_back(switchedRows);
                        segmentStreams.push_back(
                            target_.throughput().mxm_load_streams_per_cycle);
                        if (switchRow + switchedRows < tile) {
                            segmentRows.push_back(
                                tile - switchRow - switchedRows);
                            segmentStreams.push_back(0);
                        }
                    } else {
                        segmentRows.push_back(tile);
                        segmentStreams.push_back(0);
                    }
                    int64_t rowOffset = 0;
                    for (std::size_t segment = 0;
                         segment < segmentRows.size(); ++segment) {
                        const int64_t rows = segmentRows[segment];
                        const int64_t streamBase = segmentStreams[segment];
                        const int64_t segmentCycle = computeCycle + rowOffset;
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice =
                                layout.contextSlices()[headBlock * 2 + byte];
                            const int64_t latency = *target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmActivation,
                                target::StreamDirection::East, slice);
                            emitMem(rewriter_, op_.getLoc(),
                                segmentCycle - latency,
                                hemisphere * target_.memory().slices_per_hemisphere
                                    + slice,
                                "read",
                                layout.contextAddress(queryHead,
                                    tokenBlock * tile + rowOffset),
                                streamBase + hemisphere * 2 + byte,
                                rows, 1, 1);
                        }
                        emitMxm(rewriter_, op_.getLoc(), segmentCycle,
                            hemisphere
                                * target_.throughput().mxms_per_hemisphere
                                + localMxm,
                            "compute", weightBuffer, 0,
                            streamBase + hemisphere * 2,
                            8 + hemisphere * 4, rows, 1);
                        rowOffset += rows;
                    }
                    const int64_t accumulatorCycle = computeCycle
                        + accumulatorLatency;
                    if (firstReduction) {
                        for (int64_t byte = 0; byte < 4; ++byte) {
                            emitMem(rewriter_, op_.getLoc(), accumulatorCycle,
                                hemisphere * target_.memory().slices_per_hemisphere
                                    + accumulatorSlice + byte,
                                "write",
                                layout.resultAddress(outputGroup, tokenBlock * tile),
                                40 + hemisphere * 4 + byte, tile, 1, 1);
                        }
                    } else {
                        emitMem(rewriter_, op_.getLoc(), accumulatorCycle,
                            hemisphere * target_.memory().slices_per_hemisphere
                                + accumulatorSlice,
                            "accumulate",
                            layout.resultAddress(outputGroup, tokenBlock * tile),
                            40 + hemisphere * 4, tile, 1, 1,
                            "sram");
                    }
                }

            }
            phaseStart = firstCompute + tokenBlocks * computeInterval
                + (finalReduction
                        ? accumulatorLatency + readLatency(accumulatorSlice)
                        : 0);
            if (finalReduction) {
                const int64_t castStart = phaseStart;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres; ++hemisphere) {
                    const char* inputHemisphere = hemisphere == 0 ? "east" : "west";
                    const int64_t inputStream = 32 + hemisphere * 4;
                    const int64_t outputStream = hemisphere * 2;
                    for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
                        const int64_t vxmCycle = castStart + token;
                        for (int64_t byte = 0; byte < 4; ++byte) {
                            const int64_t slice = accumulatorSlice + byte;
                            const int64_t latency = *target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::VxmInput,
                                target::StreamDirection::West, slice);
                            emitMem(rewriter_, op_.getLoc(), vxmCycle - latency,
                                hemisphere * target_.memory().slices_per_hemisphere + slice,
                                "read", layout.resultAddress(outputGroup, token),
                                inputStream + byte, 1, 1, 0);
                        }
                        emitVxm(rewriter_, op_, op_.getOutputWeight(), vxmCycle,
                            hemisphere, "pass", "stream_f32", inputStream, 0.0f,
                            "immediate", 0, 0.0f, "fp16", outputStream,
                            inputHemisphere, "east");
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = 28 + hemisphere * 2 + byte;
                            const int64_t latency = *target_.transport_latency(
                                target::StreamEndpoint::VxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::East, slice);
                            emitMem(rewriter_, op_.getLoc(), vxmCycle + latency,
                                slice, "write", layout.resultAddress(outputGroup, token),
                                outputStream + byte, 1, 1, 0);
                        }
                    }
                }
                const int64_t finalWriteLatency = *target_.transport_latency(
                    target::StreamEndpoint::VxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::East, 31);
                phaseStart = castStart
                    + op_.getSeqLen() + finalWriteLatency;
            }
        }
    }
    return phaseStart;
}

mlir::FailureOr<AttentionOp> AttentionScheduleEmitter::emit()
{
    const AttentionWorkPlanner attentionPlan({static_cast<int64_t>(op_.getSeqLen()),
        static_cast<int64_t>(op_.getQueryHeads()), static_cast<int64_t>(op_.getKvHeads()),
        static_cast<int64_t>(op_.getHeadDim())}, target_);
    const AttentionProjectionPlanner projectionPlan({static_cast<int64_t>(op_.getSeqLen()),
        static_cast<int64_t>(op_.getHidden()), static_cast<int64_t>(op_.getQueryHeads()),
        static_cast<int64_t>(op_.getKvHeads()), static_cast<int64_t>(op_.getHeadDim())}, target_);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const int64_t issue = target_.mxm_block_issue_interval();

    rewriter_.setInsertionPoint(op_);
    const int64_t projectionEnd = emitProjections();
    const int64_t qkvCycles = projectionEnd - 1;
    if (qkvCycles <= 0) {
        op_.emitError("computed a non-positive QKV duration");
        return mlir::failure();
    }
    const int64_t qkWaveComputeCycles = tokenBlocks * headBlocks * issue;
    const int64_t qkIwToComputeCycles = 24;
    const int64_t qkFirstIwOffset = target_.throughput().mxm_earliest_iw_cycle
        + *target_.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight, target::StreamDirection::East, 0);
    const int64_t qkFirstComputeOffset = qkFirstIwOffset + qkIwToComputeCycles;
    const int64_t qkComputeEnd = qkFirstComputeOffset + qkWaveComputeCycles;
    const int64_t qkWaveInterval = qkComputeEnd - qkFirstIwOffset;
    const int64_t qkWaveDuration = qkComputeEnd
        + std::max(target_.throughput().mxm0_accumulator_latency,
            target_.throughput().mxm1_accumulator_latency);
    const int64_t qkStart = projectionEnd;
    const int64_t qkEnd = qkStart
        + (attentionPlan.qk_waves().size() - 1) * qkWaveInterval
        + qkWaveDuration;
    const int64_t qkCycles = qkEnd - qkStart;
    emitQk(qkStart, qkWaveInterval, qkIwToComputeCycles);
    const int64_t softmaxEnd = emitSoftmax(qkEnd);
    const int64_t softmaxCycles = softmaxEnd - qkEnd;
    const int64_t probabilityPackEnd = emitProbabilityPack(softmaxEnd);
    const int64_t probabilityTransposeEnd = emitProbabilityTranspose(probabilityPackEnd);
    const int64_t pvEnd = emitPv(probabilityTransposeEnd);
    const int64_t pvCycles = pvEnd - softmaxEnd;
    const int64_t outputProjectionEnd = emitOutputProjection(pvEnd);
    const int64_t outputProjectionCycles = outputProjectionEnd - pvEnd;

    llvm::SmallVector<mlir::Attribute> phases;
    llvm::SmallVector<mlir::Attribute> workWaves;
    llvm::SmallVector<mlir::Attribute> projectionWork;
    for (const auto& work : projectionPlan.work()) {
        const char* name = work.projection == AttentionProjection::Query ? "query"
            : work.projection == AttentionProjection::Key ? "key" : "value";
        projectionWork.push_back(rewriter_.getDictionaryAttr({
            rewriter_.getNamedAttr("projection", rewriter_.getStringAttr(name)),
            rewriter_.getNamedAttr("head_group", rewriter_.getI64IntegerAttr(work.head_group)),
            rewriter_.getNamedAttr("reduction_block", rewriter_.getI64IntegerAttr(work.reduction_block)),
            rewriter_.getNamedAttr("token_block", rewriter_.getI64IntegerAttr(work.token_block)),
            rewriter_.getNamedAttr("hemisphere", rewriter_.getI64IntegerAttr(work.hemisphere)),
            rewriter_.getNamedAttr("final_reduction", rewriter_.getBoolAttr(work.final_reduction)),
        }));
    }
    int64_t cycle = 0;
    const auto appendPhase = [&](llvm::StringRef name, int64_t duration) {
        phases.push_back(rewriter_.getDictionaryAttr({
            rewriter_.getNamedAttr("name", rewriter_.getStringAttr(name)),
            rewriter_.getNamedAttr("start", rewriter_.getI64IntegerAttr(cycle)),
            rewriter_.getNamedAttr("end", rewriter_.getI64IntegerAttr(cycle + duration)),
        }));
        cycle += duration;
    };
    appendPhase("qkv", qkvCycles);
    appendPhase("rope", 1);
    appendPhase("qk", qkCycles);
    appendPhase("softmax", softmaxCycles);
    appendPhase("pv", pvCycles);
    appendPhase("o_proj", outputProjectionCycles);
    const int64_t pvStart = probabilityTransposeEnd;
    const auto appendWaves = [&](llvm::StringRef phaseName,
        const std::vector<AttentionWorkWave>& waves, int64_t start,
        int64_t interval, int64_t duration) {
        for (std::size_t index = 0; index < waves.size(); ++index) {
            llvm::SmallVector<mlir::Attribute> slots;
            for (const auto& work : waves[index].slots) {
                if (!work) continue;
                slots.push_back(rewriter_.getDictionaryAttr({
                    rewriter_.getNamedAttr("query_head", rewriter_.getI64IntegerAttr(work->query_head)),
                    rewriter_.getNamedAttr("kv_head", rewriter_.getI64IntegerAttr(work->kv_head)),
                    rewriter_.getNamedAttr("query_block", rewriter_.getI64IntegerAttr(work->query_block)),
                    rewriter_.getNamedAttr("hemisphere", rewriter_.getI64IntegerAttr(work->hemisphere)),
                    rewriter_.getNamedAttr("local_mxm", rewriter_.getI64IntegerAttr(work->local_mxm)),
                }));
            }
            const int64_t waveStart = start + static_cast<int64_t>(index) * interval;
            workWaves.push_back(rewriter_.getDictionaryAttr({
                rewriter_.getNamedAttr("phase", rewriter_.getStringAttr(phaseName)),
                rewriter_.getNamedAttr("index", rewriter_.getI64IntegerAttr(index)),
                rewriter_.getNamedAttr("start", rewriter_.getI64IntegerAttr(waveStart)),
                rewriter_.getNamedAttr("end", rewriter_.getI64IntegerAttr(waveStart + duration)),
                rewriter_.getNamedAttr("slots", rewriter_.getArrayAttr(slots)),
            }));
        }
    };
    appendWaves("qk", attentionPlan.qk_waves(), qkStart,
        qkWaveInterval, qkWaveDuration);
    appendWaves("pv", attentionPlan.pv_waves(), pvStart,
        qkWaveComputeCycles, qkWaveComputeCycles);
    mlir::OperationState state(op_.getLoc(), AttentionOp::getOperationName());
    state.addOperands(op_->getOperands());
    state.addTypes(op_.getResult().getType());
    for (llvm::StringRef name : {"seq_len", "hidden", "query_heads", "kv_heads",
             "head_dim", "rope_theta", "causal", "memory_plan", "routes"})
        state.addAttribute(name, op_->getAttr(name));
    state.addAttribute("phases", rewriter_.getArrayAttr(phases));
    state.addAttribute("work_waves", rewriter_.getArrayAttr(workWaves));
    state.addAttribute("projection_work", rewriter_.getArrayAttr(projectionWork));
    return llvm::cast<AttentionOp>(rewriter_.create(state));
}

} // namespace ftlpu::compiler::schedule
