#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_softmax_planner.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include "llvm/Support/raw_ostream.h"

namespace ftlpu::compiler::schedule {
using namespace attention_detail;

int64_t AttentionScheduleEmitter::emitSoftmax(
    int64_t qkStart, int64_t qkEnd, bool fusedSoftmax)
{
    (void)qkStart;
    (void)fusedSoftmax;
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const llvm::StringRef streamKind =
        dataFormat == "bf16" ? "stream_bf16" : "stream_f16";
    const float scale = 1.0f / std::sqrt(static_cast<float>(op_.getHeadDim()));
    constexpr float causalMaskValue = -1.0e9f;
    const int64_t sequence = op_.getSeqLen();
    const int64_t tile = target_.throughput().mxm_rows;
    const auto placementBank = [&](llvm::StringRef name) {
        return op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(name)
            .getAs<mlir::IntegerAttr>("bank").getInt();
    };
    const int64_t scoreBank = placementBank("score");
    const int64_t expBank = placementBank("exp");
    const int64_t maskBank = placementBank("causal_mask");
    const int64_t probabilityPackBank = placementBank("probability_pack");
    const auto readLatency = [&](int64_t slice) {
        return *target_.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice);
    };
    const auto writeLatency = [&](int64_t slice) {
        return *target_.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice);
    };
    const auto maxReadLatency = [&](llvm::ArrayRef<int64_t> slices) {
        int64_t latency = 0;
        for (int64_t slice : slices)
            latency = std::max(latency, readLatency(slice));
        return latency;
    };
    const auto maxWriteLatency = [&](llvm::ArrayRef<int64_t> slices) {
        int64_t latency = 0;
        for (int64_t slice : slices)
            latency = std::max(latency, writeLatency(slice));
        return latency;
    };
    const auto emitMirroredRead = [&](llvm::ArrayRef<int64_t> slices,
                                      int64_t address, int64_t stream,
                                      int64_t inputCycle, int64_t count,
                                      int64_t stride, int64_t bank) {
        for (int64_t hemisphere = 0;
             hemisphere < target_.memory().hemispheres; ++hemisphere) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = slices[byte];
                emitMem(rewriter_, op_.getLoc(),
                    inputCycle - readLatency(slice),
                    hemisphere
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", address,
                    32 + hemisphere * 16 + stream + byte,
                    count, 1, stride, "sram", -1, bank);
            }
        }
    };
    const auto emitMirroredWrite = [&](llvm::ArrayRef<int64_t> slices,
                                       int64_t address,
                                       int64_t originalOutputStream,
                                       int64_t outputCycle,
                                       int64_t count, int64_t stride,
                                       int64_t bank) {
        for (int64_t destination = 0;
             destination < target_.memory().hemispheres; ++destination) {
            const int64_t source = 1 - destination;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = slices[byte];
                emitMem(rewriter_, op_.getLoc(),
                    outputCycle + writeLatency(slice),
                    destination
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "write", address,
                    source * 8 + originalOutputStream + byte,
                    count, 1, stride, "sram", -1, bank);
            }
        }
    };
    const auto vxm = [&](int64_t cycle, int64_t queue,
                         llvm::StringRef opcode,
                         llvm::StringRef lhsKind, int64_t lhsIndex,
                         float lhsImmediate,
                         llvm::StringRef rhsKind, int64_t rhsIndex,
                         float rhsImmediate,
                         llvm::StringRef castTarget, int64_t outputStream,
                         int64_t repeatCount, int64_t chainDepth,
                         bool accumulatorReset = false,
                         bool accumulatorWrite = false,
                         bool accumulatorEmit = true,
                         bool localScalarWrite = false) {
        return ffn_detail::create_vxm(rewriter_, op_.getLoc(),
            op_.getInput(), op_.getInput(), op_.getInput().getType(),
            cycle, queue, opcode,
            lhsKind, lhsIndex, lhsImmediate,
            rhsKind, rhsIndex, rhsImmediate,
            castTarget, outputStream, repeatCount, 1,
            "east", "east", -1,
            accumulatorReset, accumulatorWrite, accumulatorEmit,
            localScalarWrite, chainDepth);
    };

    int64_t cursor = qkEnd + 8;
    for (const AttentionWorkWave& wave : stage_plan_.qk_waves) {
        for (const auto& optionalWork : wave.slots) {
            if (!optionalWork) continue;
            const AttentionWorkItem& work = *optionalWork;
            const int64_t hemisphere = work.hemisphere;
            const auto scoreSlices = layout.scaledScoreSlices(work.local_mxm);
            const auto xSlices = layout.expScoreSlices(work.local_mxm);
            const auto maskSlices = layout.causalMaskSlices(work.local_mxm);
            const int64_t scoreAddress = layout.scoreAddress(
                work.query_head, work.query_block, 0);
            const int64_t xAddress = layout.expScoreAddress(0);
            const int64_t maxAddress = scoreAddress;

            // Compact VXM packets execute on the original and mirrored
            // eight-stage halves together. QK owns one score copy, so first
            // mirror the row through the passive cross-hemisphere fabric;
            // every following pass then supplies both fixed input groups.
            int64_t maxScoreReadLatency = 0;
            int64_t maxScoreWriteLatency = 0;
            for (int64_t slice : scoreSlices) {
                maxScoreReadLatency = std::max(
                    maxScoreReadLatency, readLatency(slice));
                maxScoreWriteLatency = std::max(
                    maxScoreWriteLatency, writeLatency(slice));
            }
            const int64_t copyBridge = cursor + maxScoreReadLatency + 1;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = scoreSlices[byte];
                emitMem(rewriter_, op_.getLoc(),
                    copyBridge - readLatency(slice),
                    hemisphere
                            * target_.memory().slices_per_hemisphere
                    + slice,
                    "read", scoreAddress, 32 + byte,
                    sequence, 1, 1, "sram", -1, scoreBank);
                emitMem(rewriter_, op_.getLoc(),
                    copyBridge + writeLatency(slice),
                    (1 - hemisphere)
                            * target_.memory().slices_per_hemisphere
                    + slice,
                    "write", scoreAddress, byte,
                    sequence, 1, 1, "sram", -1, scoreBank);
            }
            const int64_t mirrorWriteEnd = copyBridge
                + maxScoreWriteLatency + sequence - 1;
            const int64_t pass1ReadLead = std::max(
                maxReadLatency(scoreSlices), maxReadLatency(maskSlices));
            // MEM has independent SRAM read/write ports, but each
            // (slice, bank) has one ICU command queue. Account for the read
            // transport lead so pass 1 is not dispatched while the mirrored
            // score write is still repeating on that queue.
            cursor = mirrorWriteEnd + pass1ReadLead;

            // Pass 1: apply the causal mask before scaling. A finite large
            // negative value remains safely below the exponential range after
            // scaling, while keeping both operands in the 16-bit stream ABI.
            const int64_t generateConfig = cursor;
            const int64_t generateInput = generateConfig + 1;
            vxm(generateConfig, 1, "multiply",
                "previous", 0, 0.0f, "immediate", 0, scale,
                dataFormat, 0, sequence, 2);
            for (int64_t key = 0; key < sequence; ++key) {
                const int64_t keyBlock = key / tile;
                const int64_t localKey = key % tile;
                const bool vectorMask = op_.getCausal()
                    && keyBlock == work.query_block && localKey != 0;
                const float immediateMask = op_.getCausal()
                        && keyBlock > work.query_block
                    ? causalMaskValue : 0.0f;
                vxm(generateConfig + key, 0, "add",
                    streamKind, 32, 0.0f,
                    vectorMask ? streamKind : "immediate",
                    vectorMask ? 34 : 0, immediateMask,
                    "fp32", -1, 1, 2);
                if (vectorMask)
                    emitMirroredRead(maskSlices,
                        layout.causalMaskAddress(localKey), 2,
                        generateInput + key, 1, 0, maskBank);
            }
            emitMirroredRead(scoreSlices, scoreAddress, 0,
                generateInput, sequence, 1, scoreBank);
            const int64_t generateOutput = generateInput + 2;
            emitMirroredWrite(xSlices, xAddress, 0,
                generateOutput, sequence, 1, expBank);

            // Pass 2: reduce max in the lane-local FP32 accumulator and emit
            // one compact scalar back to SRAM for the following passes.
            const int64_t pass1WriteEnd = generateOutput
                + maxWriteLatency(xSlices) + sequence - 1;
            const int64_t maxInput = pass1WriteEnd
                + maxReadLatency(xSlices) + 1;
            const int64_t maxConfig = maxInput - 1;
            vxm(maxConfig, 0, "pass", streamKind, 32, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, sequence, 2);
            vxm(maxConfig, 1, "max", "previous", 0, 0.0f,
                "accumulator", 0, 0.0f, "fp32", -1, 1, 2,
                true, true, false);
            if (sequence > 2)
                vxm(maxConfig + 1, 1, "max", "previous", 0, 0.0f,
                    "accumulator", 0, 0.0f, "fp32", -1,
                    sequence - 2, 2, false, true, false);
            vxm(maxConfig + 2, 1, "max", "previous", 0, 0.0f,
                "accumulator", 0, 0.0f, dataFormat, 0, 1, 2,
                false, true, true);
            emitMirroredRead(xSlices, xAddress, 0,
                maxInput, sequence, 1, expBank);
            const int64_t maxOutput = maxInput + sequence;
            emitMirroredWrite(scoreSlices, maxAddress, 0,
                maxOutput, 1, 0, scoreBank);

            // Pass 3: exp(x-max), sum in FP32, then retain reciprocal(sum) in
            // the VXM local scalar register for normalization.
            const int64_t maxScalarWriteEnd = maxOutput
                + maxWriteLatency(scoreSlices);
            const int64_t sumInput = maxScalarWriteEnd
                + maxReadLatency(scoreSlices) + 1;
            const int64_t sumConfig = sumInput - 1;
            vxm(sumConfig, 0, "subtract", streamKind, 32, 0.0f,
                streamKind, 34, 0.0f, "fp32", -1, sequence, 4);
            vxm(sumConfig, 1, "exp", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, sequence, 4);
            vxm(sumConfig, 2, "pass", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, sequence, 4);
            vxm(sumConfig, 3, "add", "previous", 0, 0.0f,
                "accumulator", 0, 0.0f, "fp32", -1, 1, 4,
                true, true, false);
            if (sequence > 2)
                vxm(sumConfig + 1, 3, "add", "previous", 0, 0.0f,
                    "accumulator", 0, 0.0f, "fp32", -1,
                    sequence - 2, 4, false, true, false);
            vxm(sumConfig + 2, 3, "add", "previous", 0, 0.0f,
                "accumulator", 0, 0.0f, "fp32", -1, 1, 4,
                false, true, true);
            vxm(sumConfig + 1, 0, "pass", "feedback", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, 1, 4);
            vxm(sumConfig + 1, 1, "pass", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, 1, 4);
            vxm(sumConfig + 1, 2, "pass", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, 1, 4);
            vxm(sumConfig + 3, 3, "reciprocal",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 1, 4, false, false, true, true);
            emitMirroredRead(xSlices, xAddress, 0,
                sumInput, sequence, 1, expBank);
            emitMirroredRead(scoreSlices, maxAddress, 2,
                sumInput, sequence, 0, scoreBank);

            // Pass 4 recomputes exp and multiplies by the saved reciprocal.
            // The fixed C3 output stream is written directly into each key's
            // physical probability slice pair; no VXM repack pass is needed.
            const int64_t normalizeConfig = sumInput + sequence + 12;
            const int64_t normalizeInput = normalizeConfig + 1;
            vxm(normalizeConfig, 0, "subtract", streamKind, 32, 0.0f,
                streamKind, 34, 0.0f, "fp32", -1, sequence, 4);
            vxm(normalizeConfig, 1, "exp", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, sequence, 4);
            vxm(normalizeConfig, 2, "pass", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1, sequence, 4);
            vxm(normalizeConfig, 3, "multiply", "previous", 0, 0.0f,
                "accumulator", 0, 0.0f, dataFormat, 2,
                sequence, 4);
            emitMirroredRead(xSlices, xAddress, 0,
                normalizeInput, sequence, 1, expBank);
            emitMirroredRead(scoreSlices, maxAddress, 2,
                normalizeInput, sequence, 0, scoreBank);
            constexpr int64_t normalizePipelineLatency = 8;
            for (int64_t key = 0; key < sequence; ++key) {
                const int64_t packedStream =
                    (key % target_.throughput().lanes_per_tile) * 2;
                const auto packSlices = layout.probabilityPackSlices();
                const int64_t address = layout.probabilityPackAddress(
                    work.query_head, work.query_block,
                    key / target_.throughput().lanes_per_tile);
                emitMirroredWrite(
                    llvm::ArrayRef<int64_t>(packSlices).slice(
                        static_cast<std::size_t>(packedStream), 2),
                    address, 2,
                    normalizeInput + normalizePipelineLatency + key,
                    1, 0, probabilityPackBank);
            }
            cursor = normalizeInput + normalizePipelineLatency
                + sequence + 8
                + target_.throughput().tile_rows
                + target_.streams().system_register_columns;
        }
    }
    return cursor;
}

int64_t AttentionScheduleEmitter::emitProbabilityTranspose(int64_t packEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto diagonalPlacement = op_.getMemoryPlan()
        .getAs<mlir::DictionaryAttr>("probability_diagonal");
    const int64_t diagonalBank = diagonalPlacement
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t packBank = op_.getMemoryPlan()
        .getAs<mlir::DictionaryAttr>("probability_pack")
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t groups = target_.memory().slices_per_hemisphere
        / target_.streams().mem_slices_per_register_group;
    const int64_t memToSxm = target_.throughput().mem_to_sxm_latency;
    const int64_t tokenBlocks = op_.getSeqLen() / target_.throughput().mxm_rows;
    std::array<int64_t, 2> ready {packEnd, packEnd};
    std::array<int64_t, 16> inputStreams {};
    std::array<int64_t, 16> transposeStreams {};
    std::array<int64_t, 16> outputStreams {};
    const int64_t sxmInputBase =
        target_.streams().streams_per_direction - 16;
    for (int64_t stream = 0; stream < 16; ++stream) {
        inputStreams[static_cast<std::size_t>(stream)] = sxmInputBase + stream;
        transposeStreams[static_cast<std::size_t>(stream)] = stream;
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
                            sxmInputBase + stream, 1, 1, 0,
                            "sram", -1, packBank);
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
                        const auto latency = target_.transport_latency(
                            target::StreamEndpoint::SxmResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::West, slice);
                        if (!latency) return -1;
                        emitMem(rewriter_, op_.getLoc(), cycle + 1 + *latency,
                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "write", layout.probabilityDiagonalAddress(work->query_head,
                                work->query_block, keyBlock, sxmWave),
                            32 + stream, 1, 1, 0,
                            "sram", -1, diagonalBank);
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
