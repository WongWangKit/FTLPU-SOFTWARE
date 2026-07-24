#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;

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
        attention_detail::emitRopeOrCast(
            rewriter_, op_, target_, cycle, hemisphere, rope, value);
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
                        const char* destination =
                            finalReduction ? "stream" : "sram";
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
                                streamBase, 0, rows, 1, outputAddress, 1,
                                destination);
                            emitMxm(rewriter_, op_.getLoc(), segmentCycle,
                                hemisphere * 2 + 1, "compute", weightBuffer, 0,
                                streamBase + 2, 4, rows, 1,
                                outputAddress, 1, destination);
                            rowOffset += rows;
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
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const int64_t issue = target_.mxm_block_issue_interval();

    for (std::size_t waveIndex = 0;
         waveIndex < stage_plan_.qk_waves.size(); ++waveIndex) {
        const int64_t waveStart = qkStart + static_cast<int64_t>(waveIndex) * qkWaveCycles;
        const int64_t firstIwCycle = waveStart + target_.throughput().mxm_earliest_iw_cycle
            + *target_.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, 0);
        const int64_t firstComputeCycle = firstIwCycle + qkIwToComputeCycles;
        for (const auto& work : stage_plan_.qk_waves[waveIndex].slots) {
            if (!work) continue;
            const int64_t mxm = work->hemisphere * target_.throughput().mxms_per_hemisphere
                + work->local_mxm;
            const int64_t activationStream = work->local_mxm * 2;
            const int64_t outputStream = work->local_mxm * 4;
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
                        reduction, 0, activationStream, outputStream, tile, 1,
                        layout.scoreAddress(work->query_head,
                            work->query_block, keyBlock),
                        1, "sram");
                }
            }
        }
    }
}

} // namespace ftlpu::compiler::schedule
