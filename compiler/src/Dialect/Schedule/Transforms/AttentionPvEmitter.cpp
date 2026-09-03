#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"


#include <algorithm>
#include <cstdlib>
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
    const auto probabilityPlacement = op_.getMemoryPlan()
        .getAs<mlir::DictionaryAttr>("probability_diagonal");
    const int64_t probabilityBank = probabilityPlacement
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const auto placementBank = [&](llvm::StringRef name) {
        const auto placement =
            op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(name);
        const auto bank = placement
            ? placement.getAs<mlir::IntegerAttr>("bank")
            : mlir::IntegerAttr {};
        return bank ? bank.getInt() : 0;
    };
    const int64_t valueBank = placementBank("value");
    const int64_t contextBank = placementBank("context");
    const bool singleMxm =
        target_.throughput().mxms_per_hemisphere == 1;
    const bool sourceLocalContext = false;
    // Accumulator addresses are local to each physical MXM and are not MEM
    // context addresses. A single-MXM target keeps one query-tile window per
    // resident head block; dual-MXM targets get an independent address space
    // per unit.
    const int64_t accumulatorHalfStride = tokenBlocks * tile;
    const auto accumulatorAddress = [&](int64_t queryBlock,
                                        int64_t localMxm) {
        return queryBlock * tile
            + (singleMxm ? localMxm * accumulatorHalfStride : 0);
    };
    const int64_t mxmResultToVxmLatency =
        target_.throughput().accumulator_to_vxm_latency
        - target_.mxm_first_result_latency();
    // E0/E1 carry the previous BF16 context block through the passive VXM
    // bridge. Feed the next probability block on a disjoint pair so PV can
    // keep issuing one MXM row per cycle while that result is replicated.
    const int64_t activationStreamBase = sourceLocalContext ? 0 : 2;
    int64_t finalOutputHemisphereStagger = 0;
    if (!sourceLocalContext && !layout.contextHemispherePaired()) {
        for (const int64_t slice : layout.contextSlices()) {
            const auto localLatency = target_.transport_latency(
                target::StreamEndpoint::MxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::West, slice);
            if (!localLatency) return -1;
            const int64_t destinationGroup = slice
                / target_.streams().mem_slices_per_register_group;
            const int64_t remoteLatency =
                mxmResultToVxmLatency + destinationGroup + 1;
            finalOutputHemisphereStagger = std::max(
                finalOutputHemisphereStagger,
                tile + std::abs(*localLatency - remoteLatency));
        }
    }
    int64_t lastContextWriteCycle = transposeEnd - 1;
    std::array<int64_t, 16> inputStreams {};
    std::array<int64_t, 16> transposeStreams {};
    const int64_t sxmInputBase =
        target_.streams().streams_per_direction - 16;
    for (int64_t stream = 0; stream < 16; ++stream) {
        inputStreams[static_cast<std::size_t>(stream)] = sxmInputBase + stream;
        transposeStreams[static_cast<std::size_t>(stream)] = stream;
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
                    1, 1, 0, "sram", -1, valueBank);
            }
        }
        std::array<int64_t, 16> mxmStreams {};
        const int64_t singleMxmWeightStreamBase =
            target_.streams().streams_per_direction - 16;
        for (int64_t stream = 0; stream < 16; ++stream)
            mxmStreams[static_cast<std::size_t>(stream)] =
                (singleMxm ? singleMxmWeightStreamBase : localMxm * 16)
                + stream;
        for (int64_t wavefront = 0; wavefront < tileRows; ++wavefront) {
            const int64_t cycle = capture + wavefront;
            emitWavefrontBeat(rewriter_, op_.getLoc(), target_, cycle,
                hemisphere, wavefront, inputStreams, transposeStreams,
                mxmStreams, "matrix_columns");
            emitMxm(rewriter_, op_.getLoc(), cycle + 2,
                hemisphere * target_.throughput().mxms_per_hemisphere
                    + (singleMxm ? 0 : localMxm),
                "iw",
                singleMxm
                    ? localMxm
                        % target_.throughput().mxm_weight_buffers
                    : 0,
                wavefront,
                0, 0, 1, 1, 0, 1, "stream", true,
                "supercell", 0, dataFormat, {}, {},
                singleMxm ? singleMxmWeightStreamBase : -1);
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
    if (singleMxm
        && headBlocks > target_.throughput().mxm_weight_buffers) {
        // A head may contain more 32-column blocks than the physical MXM has
        // weight buffers. Page one block at a time, retain each block's ACC
        // rows across key blocks, and emit it before reusing the buffer. A
        // Forward the result through the passive VXM bridge so both
        // hemispheres own a complete planar context.
        const bool pipelineHeadBlocks = tokenBlocks == 1;
        const int64_t valueLoadLead = memToSxm + 2 * tileRows + 1
            + 8 + memToMxm;
        const int64_t firstIwOffset = memToSxm + 2;
        int64_t nextPipelinedCompute = phaseStart + valueLoadLead;
        int64_t pipelinedEnd = phaseStart;
        std::vector<std::vector<int64_t>> weightBufferRelease(
            static_cast<std::size_t>(target_.memory().hemispheres),
            std::vector<int64_t>(
                static_cast<std::size_t>(
                    target_.throughput().mxm_weight_buffers),
                0));
        for (const auto& wave : waves) {
            for (int64_t headBlock = 0;
                 headBlock < headBlocks; ++headBlock) {
                const int64_t weightBuffer = headBlock
                    % target_.throughput().mxm_weight_buffers;
                for (int64_t keyBlock = 0;
                     keyBlock < tokenBlocks; ++keyBlock) {
                    int64_t firstCompute = 0;
                    if (pipelineHeadBlocks) {
                        std::array<int64_t, 2> routeStarts {};
                        firstCompute = nextPipelinedCompute;
                        for (int64_t hemisphere = 0;
                             hemisphere < target_.memory().hemispheres;
                             ++hemisphere) {
                            const auto head =
                                wave[static_cast<std::size_t>(hemisphere)];
                            if (!head) continue;
                            const int64_t hemisphereCompute = firstCompute
                                + hemisphere * finalOutputHemisphereStagger;
                            routeStarts[static_cast<std::size_t>(hemisphere)] =
                                std::max(
                                    hemisphereCompute - valueLoadLead,
                                    weightBufferRelease[
                                        static_cast<std::size_t>(hemisphere)]
                                        [static_cast<std::size_t>(weightBuffer)]
                                        - firstIwOffset);
                        }
                        for (int64_t hemisphere = 0;
                             hemisphere < target_.memory().hemispheres;
                             ++hemisphere) {
                            const auto head =
                                wave[static_cast<std::size_t>(hemisphere)];
                            if (!head) continue;
                            const int64_t ready = emitValueLoad(
                                hemisphere, *head, keyBlock, headBlock,
                                routeStarts[static_cast<std::size_t>(hemisphere)]);
                            firstCompute = std::max(firstCompute,
                                ready + 1
                                    - hemisphere
                                        * finalOutputHemisphereStagger);
                        }
                    } else {
                        int64_t loadReady = phaseStart;
                        for (int64_t hemisphere = 0;
                             hemisphere < target_.memory().hemispheres;
                             ++hemisphere) {
                            const auto head =
                                wave[static_cast<std::size_t>(hemisphere)];
                            if (!head) continue;
                            loadReady = std::max(loadReady,
                                emitValueLoad(hemisphere, *head, keyBlock,
                                    headBlock, phaseStart));
                        }
                        phaseStart = loadReady + 8;
                        firstCompute = phaseStart + memToMxm;
                    }
                    for (int64_t queryBlock = 0;
                         queryBlock < tokenBlocks; ++queryBlock) {
                        const bool finalReduction =
                            keyBlock + 1 == tokenBlocks;
                        int64_t blockEnd = firstCompute + tile;
                        for (int64_t hemisphere = 0;
                             hemisphere
                                 < target_.memory().hemispheres;
                             ++hemisphere) {
                            const auto head =
                                wave[static_cast<std::size_t>(hemisphere)];
                            if (!head) continue;
                            const int64_t hemisphereCompute = firstCompute
                                + (finalReduction && !sourceLocalContext
                                        ? hemisphere
                                            * finalOutputHemisphereStagger
                                        : 0);
                            for (int64_t query = 0; query < tile;
                                 ++query) {
                                const int64_t row = query % lanes;
                                const int64_t diagonal = query / lanes;
                                for (int64_t byte = 0; byte < 2;
                                     ++byte) {
                                    const int64_t slice =
                                        layout.probabilityDiagonalSlices()
                                            [row * 2 + byte];
                                    const auto latency =
                                        target_.transport_latency(
                                            target::StreamEndpoint::Mem,
                                            target::StreamEndpoint::MxmActivation,
                                            target::StreamDirection::East,
                                            slice);
                                    if (!latency) return -1;
                                    emitMem(rewriter_, op_.getLoc(),
                                        hemisphereCompute + query - *latency,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "read",
                                        layout.probabilityDiagonalAddress(
                                            *head, queryBlock, keyBlock,
                                            diagonal),
                                        activationStreamBase + byte,
                                        1, 1, 0,
                                        "sram", -1, probabilityBank);
                                }
                            }
                            emitMxm(rewriter_, op_.getLoc(),
                                hemisphereCompute, hemisphere, "compute",
                                weightBuffer, 0, activationStreamBase, 0,
                                tile, 1,
                                accumulatorAddress(queryBlock, 0),
                                1,
                                finalReduction ? "stream" : "sram",
                                finalReduction, "supercell", 0,
                                dataFormat, {},
                                finalReduction ? "bf16" : "");
                            weightBufferRelease[
                                static_cast<std::size_t>(hemisphere)]
                                [static_cast<std::size_t>(weightBuffer)] =
                                hemisphereCompute
                                + target_.mxm_result_window_cycles(tile);
                            if (!finalReduction) continue;
                            const int64_t resultStart = hemisphereCompute
                                + target_.mxm_first_result_latency();
                            for (int64_t byte = 0; byte < 2;
                                 ++byte) {
                                const int64_t slice =
                                    layout.contextSlice(
                                        *head, headBlock, byte);
                                if (sourceLocalContext) {
                                    const auto latency =
                                        target_.transport_latency(
                                            target::StreamEndpoint::MxmResult,
                                            target::StreamEndpoint::Mem,
                                            target::StreamDirection::West,
                                            slice);
                                    if (!latency) return -1;
                                    const int64_t writeCycle =
                                        resultStart + *latency;
                                    emitMem(rewriter_, op_.getLoc(),
                                        writeCycle,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "write",
                                        layout.contextAddress(
                                            *head, headBlock,
                                            queryBlock * tile),
                                        32 + byte, tile, 1, 1,
                                        "sram", -1, contextBank);
                                    blockEnd = std::max(
                                        blockEnd, writeCycle + tile);
                                    continue;
                                }

                                for (int64_t destinationHemisphere = 0;
                                     destinationHemisphere
                                         < target_.memory().hemispheres;
                                     ++destinationHemisphere) {
                                    const bool local =
                                        destinationHemisphere == hemisphere;
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
                                        packedStream = 32 + byte;
                                    } else {
                                        const int64_t destinationGroup =
                                            slice
                                            / target_.streams()
                                                  .mem_slices_per_register_group;
                                        writeCycle = resultStart
                                            + mxmResultToVxmLatency
                                            + destinationGroup + 1;
                                        packedStream = byte;
                                    }
                                    emitMem(rewriter_, op_.getLoc(),
                                        writeCycle,
                                        destinationHemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        local ? "write_tap" : "write",
                                        layout.contextAddress(
                                            *head, headBlock,
                                            queryBlock * tile),
                                        packedStream, tile, 1, 1,
                                        "sram", -1, contextBank);
                                    blockEnd = std::max(
                                        blockEnd, writeCycle + tile);
                                    lastContextWriteCycle = std::max(
                                        lastContextWriteCycle,
                                        writeCycle + tile - 1);
                                }
                            }
                        }
                        if (pipelineHeadBlocks) {
                            nextPipelinedCompute = firstCompute + tile;
                            pipelinedEnd = std::max(pipelinedEnd, blockEnd + 1);
                        } else {
                            phaseStart = blockEnd + 1;
                        }
                    }
                }
            }
        }
        if (pipelineHeadBlocks)
            phaseStart = std::max(nextPipelinedCompute, pipelinedEnd);
        return phaseStart + groups;
    }

    for (const auto& wave : waves) {
        for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
            {
                const int64_t loadStart = phaseStart + 1;
                int64_t loadReady = loadStart;
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
                                loadStart
                                    + localMxm * (2 * tileRows - 1)));
                    }
                }
                phaseStart = loadReady + 8;
            }

            const int64_t querySpan =
                tile * (singleMxm ? headBlocks : 1);
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
                                        activationStreamBase + byte,
                                        1, 1, 0,
                                        "sram", -1, probabilityBank);
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
                            0, activationStreamBase, outputStream, tile, 1,
                            accumulatorAddress(queryBlock, localMxm),
                            1, finalReduction ? "stream" : "sram",
                            finalReduction, "supercell", 0, dataFormat,
                            {}, finalReduction ? "bf16" : "");
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
                                        const int64_t contextBlock =
                                            copy * 2 + half;
                                        const int64_t slice =
                                            layout.contextSlice(
                                                *head, contextBlock, byte);
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
                                                contextBlock,
                                                queryBlock * tile),
                                            packedStream, tile, 1, 1,
                                            "sram", -1, contextBank);
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
