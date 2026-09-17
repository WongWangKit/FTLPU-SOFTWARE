#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
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
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_,
                    inputCycle - readLatency(slice),
                    hemisphere
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", address,
                    32 + hemisphere * 16 + stream + byte,
                    count, 1, stride, 1, 1, 0, 1, 1, 0, -1, bank);
            }
        }
    };
    const auto emitMirroredWrite = [&](llvm::ArrayRef<int64_t> slices,
                                       int64_t address,
                                       int64_t originalOutputStream,
                                       int64_t outputCycle,
                                       int64_t count, int64_t stride,
                                       int64_t bank,
                                       int64_t repeatInterval = 1) {
        for (int64_t destination = 0;
             destination < target_.memory().hemispheres; ++destination) {
            const int64_t source = 1 - destination;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = slices[byte];
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_,
                    outputCycle + writeLatency(slice),
                    destination
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "write", address,
                    source * 8 + originalOutputStream + byte,
                    count, repeatInterval, stride,
                    1, 1, 0, 1, 1, 0, -1, bank);
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
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_,
                    copyBridge - readLatency(slice),
                    hemisphere
                            * target_.memory().slices_per_hemisphere
                    + slice,
                    "read", scoreAddress, 32 + byte,
                    sequence, 1, 1, 1, 1, 0, 1, 1, 0, -1,
                    scoreBank);
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_,
                    copyBridge + writeLatency(slice),
                    (1 - hemisphere)
                            * target_.memory().slices_per_hemisphere
                    + slice,
                    "write", scoreAddress, byte,
                    sequence, 1, 1, 1, 1, 0, 1, 1, 0, -1,
                    scoreBank);
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
            const auto emitGenerateRun = [&](int64_t firstKey,
                                              int64_t count,
                                              bool vectorMask,
                                              float immediateMask) {
                if (count <= 0) return;
                vxm(generateConfig + firstKey, 0, "add",
                    streamKind, 32, 0.0f,
                    vectorMask ? streamKind : "immediate",
                    vectorMask ? 34 : 0, immediateMask,
                    "fp32", -1, count, 2);
                if (vectorMask) {
                    const int64_t localKey = firstKey % tile;
                    emitMirroredRead(maskSlices,
                        layout.causalMaskAddress(localKey), 2,
                        generateInput + firstKey, count, 1, maskBank);
                }
            };
            if (!op_.getCausal()) {
                emitGenerateRun(0, sequence, false, 0.0f);
            } else {
                const int64_t vectorBegin = work.query_block * tile + 1;
                emitGenerateRun(0, vectorBegin, false, 0.0f);
                emitGenerateRun(vectorBegin, tile - 1, true, 0.0f);
                emitGenerateRun((work.query_block + 1) * tile,
                    sequence - (work.query_block + 1) * tile,
                    false, causalMaskValue);
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
            const int64_t keysPerLane = sequence
                / target_.throughput().lanes_per_tile;
            for (int64_t lane = 0;
                 lane < target_.throughput().lanes_per_tile; ++lane) {
                const int64_t packedStream = lane * 2;
                const auto packSlices = layout.probabilityPackSlices();
                const int64_t address = layout.probabilityPackAddress(
                    work.query_head, work.query_block,
                    0);
                emitMirroredWrite(
                    llvm::ArrayRef<int64_t>(packSlices).slice(
                        static_cast<std::size_t>(packedStream), 2),
                    address, 2,
                    normalizeInput + normalizePipelineLatency + lane,
                    keysPerLane, 1, probabilityPackBank,
                    target_.throughput().lanes_per_tile);
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
    struct TransposeWork {
        int64_t start;
        int64_t packAddress;
        int64_t diagonalAddress;
    };
    std::array<std::vector<TransposeWork>, 2> hemisphereWork;
    std::array<int64_t, 2> plannedReady {packEnd, packEnd};
    const int64_t transposeBeats =
        tokenBlocks * target_.throughput().tile_rows;
    for (const auto& wave : stage_plan_.qk_waves) {
        for (const auto& work : wave.slots) {
            if (!work) continue;
            const auto hemisphere = static_cast<std::size_t>(
                work->hemisphere);
            hemisphereWork[hemisphere].push_back({
                plannedReady[hemisphere],
                layout.probabilityPackAddress(
                    work->query_head, work->query_block, 0),
                layout.probabilityDiagonalAddress(
                    work->query_head, work->query_block, 0, 0)});
            plannedReady[hemisphere] +=
                transposeBeats + target_.throughput().tile_rows;
        }
    }
    std::array<bool, 2> groupedMem {};
    const auto packSlices = layout.probabilityPackSlices();
    const auto diagonalSlices = layout.probabilityDiagonalSlices();
    const bool disjointReadWriteSlices = std::none_of(
        packSlices.begin(), packSlices.end(), [&](int64_t slice) {
            return std::find(diagonalSlices.begin(),
                diagonalSlices.end(), slice) != diagonalSlices.end();
        });
    for (int64_t hemisphere = 0;
         hemisphere < target_.memory().hemispheres; ++hemisphere) {
        const auto& work = hemisphereWork[static_cast<std::size_t>(
            hemisphere)];
        if (work.size() < 2 || !disjointReadWriteSlices) continue;
        const int64_t cycleStride = work[1].start - work[0].start;
        const int64_t packStride =
            work[1].packAddress - work[0].packAddress;
        const int64_t diagonalStride =
            work[1].diagonalAddress - work[0].diagonalAddress;
        const bool affine = cycleStride >= transposeBeats
            && std::adjacent_find(work.begin(), work.end(),
            [&](const TransposeWork& lhs, const TransposeWork& rhs) {
                return rhs.start - lhs.start != cycleStride
                    || rhs.packAddress - lhs.packAddress != packStride
                    || rhs.diagonalAddress - lhs.diagonalAddress
                        != diagonalStride;
            }) == work.end();
        if (!affine) continue;
        groupedMem[static_cast<std::size_t>(hemisphere)] = true;
        const int64_t capture = work[0].start + memToSxm;
        for (int64_t stream = 0; stream < 16; ++stream) {
            const int64_t packSlice =
                layout.probabilityPackSlices()[stream];
            const int64_t readLatency = memToSxm
                - packSlice
                    / target_.streams().mem_slices_per_register_group;
            direct_domain_detail::emitMem3D(
                rewriter_, op_.getLoc(), target_, capture - readLatency,
                hemisphere * target_.memory().slices_per_hemisphere
                    + packSlice,
                "read", work[0].packAddress,
                sxmInputBase + stream, transposeBeats, 1, 1,
                1, 1, 0, static_cast<int64_t>(work.size()),
                cycleStride, packStride, -1, packBank);
            const int64_t diagonalSlice =
                layout.probabilityDiagonalSlices()[stream];
            const auto writeLatency = target_.transport_latency(
                target::StreamEndpoint::SxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::West, diagonalSlice);
            if (!writeLatency) return -1;
            direct_domain_detail::emitMem3D(
                rewriter_, op_.getLoc(), target_,
                capture + 1 + *writeLatency,
                hemisphere * target_.memory().slices_per_hemisphere
                    + diagonalSlice,
                "write", work[0].diagonalAddress,
                32 + stream, transposeBeats, 1, 1,
                1, 1, 0, static_cast<int64_t>(work.size()),
                cycleStride, diagonalStride, -1, diagonalBank);
        }
    }
    for (const auto& wave : stage_plan_.qk_waves) {
        for (const auto& work : wave.slots) {
            if (!work) continue;
            const int64_t hemisphere = work->hemisphere;
            const int64_t start =
                ready[static_cast<std::size_t>(hemisphere)];
            const int64_t capture = start + memToSxm;
            if (!groupedMem[static_cast<std::size_t>(hemisphere)]) {
              for (int64_t stream = 0; stream < 16; ++stream) {
                const int64_t slice = layout.probabilityPackSlices()[stream];
                const int64_t latency = memToSxm
                    - slice
                        / target_.streams().mem_slices_per_register_group;
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_, capture - latency,
                    hemisphere * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", layout.probabilityPackAddress(
                        work->query_head, work->query_block, 0),
                    sxmInputBase + stream, transposeBeats, 1, 1,
                    1, 1, 0, 1, 1, 0, -1, packBank);
              }
            }
            emitSxm(rewriter_, op_.getLoc(), capture, hemisphere,
                "transpose", inputStreams, transposeStreams,
                identityMap(), "vector_columns", -1, -1, -1,
                transposeBeats, 1);
            emitSxm(rewriter_, op_.getLoc(), capture + 1, hemisphere,
                "permute", transposeStreams, outputStreams,
                blockDiagonalMap(0, target_), "vector_columns",
                -1, -1, -1,
                transposeBeats + target_.throughput().tile_rows - 1,
                1, 1, 1,
                target_.throughput().lanes_per_tile);
            if (!groupedMem[static_cast<std::size_t>(hemisphere)]) {
              for (int64_t stream = 0; stream < 16; ++stream) {
                const int64_t slice =
                    layout.probabilityDiagonalSlices()[stream];
                const auto latency = target_.transport_latency(
                    target::StreamEndpoint::SxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::West, slice);
                if (!latency) return -1;
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_,
                    capture + 1 + *latency,
                    hemisphere * target_.memory().slices_per_hemisphere
                        + slice,
                    "write", layout.probabilityDiagonalAddress(
                        work->query_head, work->query_block, 0, 0),
                    32 + stream, transposeBeats, 1, 1,
                    1, 1, 0, 1, 1, 0, -1, diagonalBank);
              }
            }

            // The steady-state and drain beats have the same affine body and
            // map stride.  Keep them in one RUN_2D domain so the physical ICU
            // does not need to switch packets while a transpose wavefront is
            // still live in the SXM tile pipeline.
            ready[static_cast<std::size_t>(hemisphere)] =
                start + transposeBeats
                    + target_.throughput().tile_rows;
        }
    }
    return std::max(ready[0], ready[1]) + memToSxm + groups + 1;
}

} // namespace ftlpu::compiler::schedule
