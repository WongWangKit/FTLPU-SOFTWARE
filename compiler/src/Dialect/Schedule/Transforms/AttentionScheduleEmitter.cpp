#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_memory_layout.hpp"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <vector>

namespace ftlpu::compiler::schedule {
namespace {

void emitMem(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t address,
    int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t addressStride, llvm::StringRef destination = "sram")
{
    mlir::OperationState state(location, MemQueueOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
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
    mlir::OperationState state(location, MxmQueueOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
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
        target::StreamDirection::East, target_.memory().w8a16_activation_slice_base);
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
        if (!rope) {
            emitVxm(rewriter_, op_, value, cycle, 0, "pass", "stream_f32", 32,
                0.0f, "immediate", 0, 0.0f, "fp16", 0, hemi, hemi);
            emitVxm(rewriter_, op_, value, cycle, 1, "pass", "stream_f32", 36,
                0.0f, "immediate", 0, 0.0f, "fp16", 2, hemi, hemi);
            return;
        }
        emitVxm(rewriter_, op_, value, cycle, 0, "multiply", "stream_f32", 32,
            0.0f, "stream_f16", 40, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle, 1, "multiply", "stream_f32", 36,
            0.0f, "stream_f16", 42, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle, 3, "multiply", "stream_f32", 36,
            0.0f, "stream_f16", 40, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle, 4, "multiply", "stream_f32", 32,
            0.0f, "stream_f16", 42, 0.0f, "fp32", -1, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle + 1, 2, "subtract", "alu", 0,
            0.0f, "alu", 1, 0.0f, "fp16", 0, hemi, hemi);
        emitVxm(rewriter_, op_, value, cycle + 1, 5, "add", "alu", 3,
            0.0f, "alu", 4, 0.0f, "fp16", 2, hemi, hemi);
    };

    for (int64_t projection = 0; projection < 3; ++projection) {
        const auto kind = projectionKind(projection);
        for (int64_t headBase = 0; headBase < projectionHeads[projection]; headBase += 2) {
            for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks; ++reductionBlock) {
                const int64_t dequantStart = phaseStart + 10;
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
                            hemisphere * 2 + localMxm, "iw", 0, column,
                            0, 0, 1, 1);
                    }
                }

                const int64_t firstCompute = dequantStart + 32;
                const bool finalReduction = reductionBlock + 1 == hiddenBlocks;
                const int64_t computeBlockCycles = finalReduction
                    ? 2 * tile : target_.mxm_block_issue_interval();
                const int64_t hemisphereSkew = finalReduction ? tile : 0;
                for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
                    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                        const int64_t head = headBase + hemisphere;
                        if (head >= projectionHeads[projection]) continue;
                        const int64_t computeCycle = firstCompute
                            + tokenBlock * computeBlockCycles + hemisphere * hemisphereSkew;
                        const int64_t inputAddress = layout.activationAddress(
                            reductionBlock, tokenBlock);
                        const int64_t outputAddress = layout.projectionAddress(
                            kind, head, tokenBlock);
                        for (int64_t byte = 0; byte < 4; ++byte)
                            emitMem(rewriter_, op_.getLoc(), computeCycle - activationLatency,
                                hemisphere * target_.memory().slices_per_hemisphere
                                    + layout.activationSlices()[byte],
                                "read", inputAddress, byte, tile, 1, 1);
                        const char* destination = finalReduction ? "stream" : "sram";
                        emitMem(rewriter_, op_.getLoc(), computeCycle
                                + target_.throughput().mxm0_accumulator_latency,
                            hemisphere * target_.memory().slices_per_hemisphere
                                + target_.memory().accumulator_slice_base,
                            "accumulate", outputAddress, 32, tile, 1, 1, destination);
                        emitMem(rewriter_, op_.getLoc(), computeCycle
                                + target_.throughput().mxm1_accumulator_latency,
                            hemisphere * target_.memory().slices_per_hemisphere
                                + target_.memory().accumulator_slice_base + 4,
                            "accumulate", outputAddress, 36, tile, 1, 1, destination);
                        emitMxm(rewriter_, op_.getLoc(), computeCycle, hemisphere * 2,
                            "compute", 0, 0, 0, 0, tile, 1);
                        emitMxm(rewriter_, op_.getLoc(), computeCycle, hemisphere * 2 + 1,
                            "compute", 0, 0, 2, 4, tile, 1);
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
                            } else {
                                for (int64_t byte = 0; byte < 4; ++byte)
                                    emitMem(rewriter_, op_.getLoc(), writeCycle,
                                        hemisphere * target_.memory().slices_per_hemisphere + byte,
                                        "write", outputAddress + offset, byte, 1, 1, 0);
                            }
                        }
                    }
                }
                phaseStart = firstCompute + tokenBlocks * computeBlockCycles
                    + (finalReduction ? 16 : 0);
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
    const int64_t hiddenBlocks = op_.getHidden() / tile;
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
    const int64_t qkWaveCycles = target_.throughput().mxm_earliest_iw_cycle
        + *target_.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight, target::StreamDirection::East, 0)
        + qkIwToComputeCycles + qkWaveComputeCycles + 16;
    const int64_t qkCycles = attentionPlan.qk_waves().size() * qkWaveCycles;
    const int64_t softmaxCycles = op_.getQueryHeads() * tokenBlocks
        * (3 * op_.getSeqLen() + 36);
    const int64_t pvCycles = attentionPlan.pv_waves().size()
        * tokenBlocks * headBlocks * issue;
    const int64_t outputProjectionCycles = (op_.getHidden() / tile)
        * hiddenBlocks * tokenBlocks * issue;

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
    const int64_t qkStart = projectionEnd;
    const int64_t pvStart = qkStart + qkCycles + softmaxCycles;
    const auto appendWaves = [&](llvm::StringRef phaseName,
        const std::vector<AttentionWorkWave>& waves, int64_t start, int64_t interval) {
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
                rewriter_.getNamedAttr("end", rewriter_.getI64IntegerAttr(waveStart + interval)),
                rewriter_.getNamedAttr("slots", rewriter_.getArrayAttr(slots)),
            }));
        }
    };
    appendWaves("qk", attentionPlan.qk_waves(), qkStart, qkWaveCycles);
    appendWaves("pv", attentionPlan.pv_waves(), pvStart, qkWaveComputeCycles);
    emitQk(qkStart, qkWaveCycles, qkIwToComputeCycles);

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
