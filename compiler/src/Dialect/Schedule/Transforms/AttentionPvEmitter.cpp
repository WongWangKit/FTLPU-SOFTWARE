#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"


#include <algorithm>
#include <limits>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;

int64_t AttentionScheduleEmitter::emitPv(int64_t transposeEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
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
    const bool singleMxm =
        target_.throughput().mxms_per_hemisphere == 1;
    const auto resultPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("result");
    const auto resultKind = resultPlacement
        ? resultPlacement.getAs<mlir::StringAttr>("kind")
        : mlir::StringAttr {};
    const bool sourceLocalContext = singleMxm && resultKind
        && resultKind.getValue()
            == "fp16_mxm_block8_distributed_16";
    const int64_t accumulatorHalfStride = 4096;
    const int64_t mxmResultToVxmLatency =
        target_.throughput().accumulator_to_vxm_latency
        - target_.mxm_first_result_latency();
    int64_t finalOutputHemisphereStagger = 0;
    if (!sourceLocalContext) {
        int64_t earliestLocalWrite = std::numeric_limits<int64_t>::max();
        int64_t latestRemoteWrite = 0;
        for (const int64_t slice : layout.contextSlices()) {
            const auto localLatency = target_.transport_latency(
                target::StreamEndpoint::MxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::West, slice);
            if (!localLatency) return -1;
            earliestLocalWrite = std::min(
                earliestLocalWrite, *localLatency);
            const int64_t destinationGroup = slice
                / target_.streams().mem_slices_per_register_group;
            latestRemoteWrite = std::max(latestRemoteWrite,
                mxmResultToVxmLatency + destinationGroup + 1);
        }
        finalOutputHemisphereStagger = latestRemoteWrite + tile
            - earliestLocalWrite;
    }
    int64_t lastContextWriteCycle = transposeEnd - 1;
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

    const auto emitValueLoad = [&](int64_t hemisphere, int64_t head,
                                   int64_t keyBlock, int64_t localMxm,
                                   int64_t routeStart) {
        const int64_t kvHead = head / queryHeadsPerKv;
        const int64_t capture = routeStart + memToSxm;
        const auto slices = layout.valuePackSlices(localMxm);
        for (int64_t beat = 0; beat < tileRows; ++beat) {
            for (int64_t stream = 0; stream < 16; ++stream) {
                const int64_t slice =
                    slices[static_cast<std::size_t>(stream)];
                const int64_t latency = memToSxm
                    - slice
                        / target_.streams().mem_slices_per_register_group;
                emitMem(rewriter_, op_.getLoc(),
                    capture + beat - latency,
                    hemisphere * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", layout.valuePackAddress(
                        kvHead, localMxm, keyBlock, beat),
                    inputStreams[static_cast<std::size_t>(stream)],
                    1, 1, 0);
            }
        }
        std::array<int64_t, 16> mxmStreams {};
        for (int64_t stream = 0; stream < 16; ++stream)
            mxmStreams[static_cast<std::size_t>(stream)] =
                (singleMxm ? 0 : localMxm * 16) + stream;
        for (int64_t wavefront = 0; wavefront < tileRows; ++wavefront) {
            const int64_t cycle = capture + wavefront;
            emitWavefrontBeat(rewriter_, op_.getLoc(), target_, cycle,
                hemisphere, wavefront, inputStreams, transposeStreams,
                mxmStreams, "matrix_columns");
            emitMxm(rewriter_, op_.getLoc(), cycle + 2,
                hemisphere * target_.throughput().mxms_per_hemisphere
                    + (singleMxm ? 0 : localMxm),
                "iw", singleMxm ? localMxm : 0, wavefront,
                0, 0, 1, 1, 0, 1, "stream", true,
                "supercell", 0, dataFormat);
        }
        for (int64_t tail = 0; tail < tileRows - 1; ++tail) {
            const int64_t wavefront = tileRows + tail;
            emitWavefrontTail(rewriter_, op_.getLoc(), target_,
                capture + wavefront, hemisphere, wavefront,
                transposeStreams, mxmStreams, "matrix_columns");
        }
        return capture + 2 * tileRows + 1;
    };

    int64_t phaseStart = transposeEnd;
    for (std::size_t waveIndex = 0; waveIndex < waves.size(); ++waveIndex) {
        const auto& wave = waves[waveIndex];
        for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
            if ((keyBlock == 0 && waveIndex == 0) || !singleMxm) {
                int64_t loadReady = phaseStart;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const auto head = wave[static_cast<std::size_t>(hemisphere)];
                    if (!head) continue;
                    for (int64_t localMxm = 0;
                         localMxm < headBlocks; ++localMxm) {
                        loadReady = std::max(loadReady,
                            emitValueLoad(hemisphere, *head, keyBlock,
                                localMxm,
                                phaseStart
                                    + localMxm * (2 * tileRows - 1)));
                    }
                }
                phaseStart = loadReady + 8;
            }

            const int64_t keyComputeStart = phaseStart;
            const int64_t querySpan =
                tile * (singleMxm ? headBlocks : 1);
            const bool prefetchNextKey = keyBlock + 1 < tokenBlocks;
            const bool prefetchNextWave = !prefetchNextKey
                && waveIndex + 1 < waves.size();
            if (singleMxm && (prefetchNextKey || prefetchNextWave)) {
                const auto& prefetchWave = prefetchNextKey
                    ? wave : waves[waveIndex + 1];
                const int64_t prefetchKeyBlock =
                    prefetchNextKey ? keyBlock + 1 : 0;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const auto head = prefetchWave[
                        static_cast<std::size_t>(hemisphere)];
                    if (!head) continue;
                    for (int64_t localMxm = 0;
                         localMxm < headBlocks; ++localMxm) {
                        const int64_t safeIwCycle = keyComputeStart
                            + (tokenBlocks - 1) * querySpan + memToMxm
                            + (localMxm + 1) * tile;
                        emitValueLoad(hemisphere, *head, prefetchKeyBlock,
                            localMxm,
                            safeIwCycle - 2 - memToSxm);
                    }
                }
            }

            for (int64_t queryBlock = 0; queryBlock < tokenBlocks; ++queryBlock) {
                int64_t blockEnd = phaseStart;
                for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const auto head = wave[static_cast<std::size_t>(hemisphere)];
                    if (!head) continue;
                    const int64_t replayStart = phaseStart;
                    const int64_t firstCompute = replayStart + memToMxm;
                    const bool finalReduction =
                        keyBlock + 1 == tokenBlocks;
                    // Both head results are replicated into both hemispheres.
                    // Stagger the final output waves so they do not contend for
                    // the same context MEM write slices.
                    const int64_t finalComputeStart = firstCompute
                        + (finalReduction && !sourceLocalContext
                                ? hemisphere
                                    * finalOutputHemisphereStagger
                                : 0);
                    for (int64_t localMxm = 0; localMxm < headBlocks; ++localMxm) {
                        const int64_t computeCycle = finalComputeStart
                            + (singleMxm ? localMxm * tile : 0);
                        if (singleMxm || localMxm == 0) {
                            for (int64_t query = 0; query < tile; ++query) {
                                const int64_t row = query % lanes;
                                const int64_t diagonal = query / lanes;
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    const int64_t slice =
                                        layout.probabilityDiagonalSlices()
                                            [row * 2 + byte];
                                    const int64_t latency =
                                        *target_.transport_latency(
                                            target::StreamEndpoint::Mem,
                                            target::StreamEndpoint::MxmActivation,
                                            target::StreamDirection::East,
                                            slice);
                                    emitMem(rewriter_, op_.getLoc(),
                                        computeCycle + query - latency,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "read",
                                        layout.probabilityDiagonalAddress(*head,
                                            queryBlock, keyBlock, diagonal),
                                        16 + byte, 1, 1, 0);
                                }
                            }
                        }
                        const int64_t outputStream =
                            singleMxm ? 0 : localMxm * 2;
                        emitMxm(rewriter_, op_.getLoc(), computeCycle,
                            hemisphere
                                    * target_.throughput().mxms_per_hemisphere
                                + (singleMxm ? 0 : localMxm),
                            "compute", singleMxm ? localMxm : 0,
                            0, 16, outputStream, tile, 1,
                            layout.contextAddress(
                                *head, queryBlock * tile)
                                + (singleMxm
                                        ? localMxm
                                            * accumulatorHalfStride
                                        : 0),
                            1, finalReduction ? "stream" : "sram",
                            finalReduction, "supercell", 0, dataFormat,
                            {}, {}, finalReduction ? "bf16" : "");
                    }

                    if (keyBlock + 1 == tokenBlocks) {
                        for (int64_t half = 0; half < headBlocks; ++half) {
                            const int64_t resultStart = finalComputeStart
                                + (singleMxm ? half * tile : 0)
                                + target_.mxm_first_result_latency();
                            const int64_t resultStream =
                                singleMxm ? 32 : 32 + half * 2;
                            const int64_t firstDestination =
                                sourceLocalContext ? hemisphere : 0;
                            const int64_t destinationEnd = sourceLocalContext
                                ? hemisphere + 1
                                : target_.memory().hemispheres;
                            const int64_t copyCount =
                                sourceLocalContext ? 1 : 2;
                            for (int64_t destinationHemisphere =
                                     firstDestination;
                                 destinationHemisphere < destinationEnd;
                                 ++destinationHemisphere) {
                                for (int64_t copy = 0; copy < copyCount;
                                     ++copy) {
                                    for (int64_t byte = 0; byte < 2; ++byte) {
                                        const int64_t slice =
                                            layout.contextSlices()[
                                                copy * 4 + half * 2 + byte];
                                        const bool local =
                                            destinationHemisphere
                                            == hemisphere;
                                        const int64_t destinationGroup =
                                            slice
                                            / target_.streams()
                                                  .mem_slices_per_register_group;
                                        int64_t writeCycle = 0;
                                        int64_t packedStream = 0;
                                        if (local) {
                                            const auto latency =
                                                target_.transport_latency(
                                                    target::StreamEndpoint::MxmResult,
                                                    target::StreamEndpoint::Mem,
                                                    target::StreamDirection::West,
                                                    slice);
                                            if (!latency) return -1;
                                            writeCycle = resultStart + *latency;
                                            packedStream = resultStream + byte;
                                        } else {
                                            writeCycle = resultStart
                                                + mxmResultToVxmLatency
                                                + destinationGroup + 1;
                                            packedStream = resultStream + byte
                                                - target_.streams()
                                                      .streams_per_direction;
                                        }
                                        // Local writes must preserve the W
                                        // stream for the passive VXM bridge.
                                        // On the remote E path, only the last
                                        // physical copy consumes the stream.
                                        const bool preserveStream =
                                            !sourceLocalContext
                                            && (local
                                                || copy + 1 < copyCount);
                                        emitMem(rewriter_, op_.getLoc(),
                                            writeCycle,
                                            destinationHemisphere
                                                    * target_.memory()
                                                          .slices_per_hemisphere
                                                + slice,
                                            preserveStream
                                                ? "write_tap"
                                                : "write",
                                            layout.contextAddress(*head,
                                                queryBlock * tile),
                                            packedStream, tile, 1, 1);
                                        lastContextWriteCycle = std::max(
                                            lastContextWriteCycle,
                                            writeCycle + tile - 1);
                                    }
                                }
                            }
                        }
                        blockEnd = std::max(blockEnd,
                            sourceLocalContext
                                ? replayStart + querySpan
                                : lastContextWriteCycle + 1);
                    } else {
                        blockEnd = std::max(blockEnd,
                            firstCompute
                                + tile * (singleMxm ? headBlocks : 1)
                                - memToMxm);
                    }
                }
                phaseStart = blockEnd;
            }
        }
    }
    const int64_t end = std::max(
        phaseStart + groups, lastContextWriteCycle + 1);
    return end;
}

} // namespace ftlpu::compiler::schedule
