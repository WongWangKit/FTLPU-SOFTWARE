#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_softmax_planner.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;

int64_t AttentionScheduleEmitter::emitSoftmax(
    int64_t qkStart, int64_t qkEnd, bool fusedSoftmax)
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const float scale = 1.0f / std::sqrt(static_cast<float>(op_.getHeadDim()));
    constexpr float causalMaskValue = -1.0e9f;
    constexpr int64_t outputStream = 8;
    constexpr int64_t alusPerWork = 4;
    const int64_t hemisphereCount = target_.memory().hemispheres;
    auto softmaxSchedule = planAttentionSoftmax(
        op_, stage_plan_.qk_waves, qkStart, qkEnd,
        stage_plan_.qk_wave_interval,
        stage_plan_.qk_iw_to_compute_cycles, fusedSoftmax, target_);
    if (mlir::failed(softmaxSchedule)) {
        op_.emitError("failed to plan the softmax schedule");
        return qkEnd;
    }
    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    const auto maxRegisterGroup = [&](llvm::ArrayRef<int64_t> slices) {
        return *std::max_element(slices.begin(), slices.end())
            / target_.streams().mem_slices_per_register_group;
    };
    const auto vxm = [&](int64_t cycle, int64_t queue, llvm::StringRef opcode,
                         llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
                         llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
                         llvm::StringRef castTarget, int64_t stream, int64_t hemisphere) {
        const char* hemi = hemisphere == 0 ? "east" : "west";
        emitVxm(rewriter_, op_.getLoc(), op_.getInput(), cycle,
            queue, opcode,
            lhsKind, lhsIndex, lhsImmediate, rhsKind, rhsIndex, rhsImmediate,
            castTarget, stream, hemi, hemi);
    };

    for (std::size_t waveIndex = 0;
         waveIndex < stage_plan_.qk_waves.size(); ++waveIndex) {
        const AttentionWorkWave& wave = stage_plan_.qk_waves[waveIndex];
        for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
             ++hemisphere) {
            llvm::SmallVector<const AttentionWorkItem*> workLines;
            for (const auto& work : wave.slots)
                if (work && work->hemisphere == hemisphere) workLines.push_back(&*work);
            if (workLines.empty()) continue;

            const auto softmaxCycle =
                softmaxSchedule->wave_cycles[waveIndex]
                    [static_cast<std::size_t>(hemisphere)];
            if (!softmaxCycle) continue;
            for (std::size_t workIndex = 0; workIndex < workLines.size();
                 ++workIndex) {
            const AttentionWorkItem* work = workLines[workIndex];
            const int64_t mxmLane = work->local_mxm;
            const int64_t lane = fusedSoftmax
                ? static_cast<int64_t>(waveIndex % 2) : mxmLane;
            const int64_t aluBase = fusedSoftmax
                ? lane * hemisphereCount * alusPerWork
                    + hemisphere * alusPerWork
                : (hemisphere * target_.throughput().mxms_per_hemisphere
                      + lane)
                    * alusPerWork;
            const int64_t scoreInputStream = 32 + mxmLane * 4;
            const int64_t scratchInputStream = 32 + lane * 4;
            const int64_t maskStream = 40 + lane * 4;
            const int64_t outputStreamBase = outputStream + lane * 4;
            // Keep optional fused banks target-derived and isolated from the
            // physical planes selected for the Tail schedule.
            const auto scaledScoreSlices = fusedSoftmax
                ? layout.fusedScoreSlices(lane)
                : layout.scaledScoreSlices(lane);
            const auto expScoreSlices = fusedSoftmax
                ? layout.fusedScoreSlices(lane)
                : layout.expScoreSlices(lane);
            for (int64_t key = 0; key < op_.getSeqLen(); ++key) {
                const int64_t cycle = *softmaxCycle
                    + static_cast<int64_t>(workIndex)
                        * softmaxSchedule->work_interval
                    + key;
                const int64_t keyBlock =
                    key / target_.throughput().mxm_rows;
                const int64_t localKey =
                    key % target_.throughput().mxm_rows;
                const bool vectorMask = op_.getCausal()
                    && keyBlock == work->query_block && localKey != 0;
                if (!fusedSoftmax) {
                    emitMxm(rewriter_, op_.getLoc(),
                        cycle
                            - target_.throughput()
                                  .accumulator_read_to_vxm_latency,
                        hemisphere
                                * target_.throughput().mxms_per_hemisphere
                            + mxmLane,
                        "accumulator_read", 0, 0, 0, mxmLane * 4, 1, 1,
                        layout.scoreTokenAddress(work->query_head,
                            work->query_block, key),
                        1, "sram", true, "supercell", 0, dataFormat);
                }
                if (vectorMask) {
                    for (int64_t byte = 0; byte < 4; ++byte) {
                        const int64_t slice = fusedSoftmax
                            ? layout.fusedCausalMaskSlices(lane)[byte]
                            : layout.causalMaskSlices(lane)[byte];
                        emitMem(rewriter_, op_.getLoc(),
                            cycle + 1 - readLatency(slice),
                            hemisphere * target_.memory().slices_per_hemisphere
                                + slice,
                            "read", layout.causalMaskAddress(localKey),
                            maskStream + byte, 1, 1, 0);
                    }
                }
                vxm(cycle, aluBase, "multiply", "stream_f32", scoreInputStream, 0.0f,
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
                    const int64_t slice = scaledScoreSlices[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle
                            + (op_.getCausal() ? 2 : 1)
                            + slice / target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "write", layout.scaledScoreAddress(key), outputStreamBase + byte,
                        1, 1, 0);
                }
            }

            const int64_t scaledGroup =
                maxRegisterGroup(scaledScoreSlices);
            const int64_t workStart = *softmaxCycle
                + static_cast<int64_t>(workIndex)
                    * softmaxSchedule->work_interval;
            const int64_t pass2Start = workStart + op_.getSeqLen()
                + (op_.getCausal() ? 4 : 3) + 2 * scaledGroup;
            for (int64_t key = 0; key < op_.getSeqLen(); ++key) {
                const int64_t cycle = pass2Start + key;
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = scaledScoreSlices[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "read", layout.scaledScoreAddress(key),
                        scratchInputStream + byte, 1, 1, 0);
                }
                vxm(cycle, aluBase, "subtract", "stream_f32",
                    scratchInputStream, 0.0f,
                    "alu", aluBase + 2, 0.0f, "fp32", -1, hemisphere);
                vxm(cycle + 1, aluBase + 1, "exp", "alu", aluBase, 0.0f,
                    "immediate", 0, 0.0f, "fp32", outputStreamBase, hemisphere);
                vxm(cycle + 2, aluBase + 3, key == 0 ? "pass" : "add",
                    "alu", key == 0 ? aluBase + 1 : aluBase + 3, 0.0f,
                    key == 0 ? "immediate" : "alu", key == 0 ? 0 : aluBase + 1, 0.0f,
                    "fp32", -1, hemisphere);
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = expScoreSlices[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle + 2
                            + slice / target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "write", layout.expScoreAddress(key), outputStreamBase + byte,
                        1, 1, 0);
                }
            }

            const int64_t expGroup =
                maxRegisterGroup(expScoreSlices);
            const int64_t pass3Start = pass2Start + op_.getSeqLen()
                + 4 + 2 * expGroup;
            for (int64_t key = 0; key < op_.getSeqLen(); ++key) {
                const int64_t cycle = pass3Start + key;
                for (int64_t byte = 0; byte < 4; ++byte) {
                    const int64_t slice = expScoreSlices[byte];
                    emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                        "read", layout.expScoreAddress(key),
                        scratchInputStream + byte, 1, 1, 0);
                }
                vxm(cycle, aluBase, "divide", "stream_f32",
                    scratchInputStream, 0.0f,
                    "alu", aluBase + 3, 0.0f, "fp32", -1, hemisphere);
                const int64_t packedStream =
                    (key % target_.throughput().lanes_per_tile) * 2;
                vxm(cycle + 1, aluBase + 1, "cast", "alu", aluBase, 0.0f,
                    "immediate", 0, 0.0f, dataFormat,
                    packedStream, hemisphere);
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t packSlice =
                        layout.probabilityPackSlices()[packedStream + byte];
                    emitMem(rewriter_, op_.getLoc(), cycle + 2
                            + packSlice
                                / target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere
                            + packSlice,
                        "write", layout.probabilityPackAddress(
                            work->query_head, work->query_block,
                            key / target_.throughput().lanes_per_tile),
                        packedStream + byte,
                        1, 1, 0);
                }
            }
            }
        }
    }
    // Softmax materializes its sole physical result in the packed layout
    // consumed by SXM. No planar probability copy or repack phase is needed.
    return softmaxSchedule->end_cycle;
}

int64_t AttentionScheduleEmitter::emitProbabilityTranspose(int64_t packEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
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
    for (const auto& wave : stage_plan_.qk_waves) {
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
                    emitWavefrontBeat(rewriter_, op_.getLoc(), target_,
                        cycle, hemisphere, sxmWave, inputStreams,
                        transposeStreams, outputStreams);
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
            emitWavefrontTail(rewriter_, op_.getLoc(), target_,
                cycle, hemisphere, tail, transposeStreams, outputStreams);
        }
        ready[static_cast<std::size_t>(hemisphere)] +=
            target_.throughput().tile_rows - 1;
    }
    return std::max(ready[0], ready[1]) + memToSxm + groups + 1;
}

} // namespace ftlpu::compiler::schedule
