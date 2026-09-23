#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"


#include <algorithm>
#include <cstdlib>
#include <limits>
#include <tuple>

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
    const int64_t queryBlocks = (op_.getSeqLen() + tile - 1) / tile;
    const int64_t keyBlocks = op_.getKvSeqLen() / tile;
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
    const int64_t accumulatorHalfStride = queryBlocks * tile;
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
    std::array<std::vector<int64_t>, 2> transposeCaptures;
    const auto emitPlannedTransposes = [&] {
        for (int64_t hemisphere = 0;
             hemisphere < target_.memory().hemispheres; ++hemisphere) {
            auto& captures = transposeCaptures[static_cast<std::size_t>(
                hemisphere)];
            std::sort(captures.begin(), captures.end());
            for (std::size_t begin = 0; begin < captures.size();) {
                std::size_t end = begin + 1;
                int64_t interval = 1;
                if (end < captures.size()) {
                    interval = captures[end] - captures[begin];
                    if (interval >= tileRows) {
                        ++end;
                        while (end < captures.size()
                            && captures[end] - captures[end - 1]
                                == interval)
                            ++end;
                    } else {
                        interval = 1;
                    }
                }
                emitSxm(rewriter_, op_.getLoc(), captures[begin],
                    hemisphere, "transpose", inputStreams,
                    transposeStreams, identityMap(), "vector_columns",
                    -1, -1, -1, tileRows, 1,
                    static_cast<int64_t>(end - begin), interval);
                begin = end;
            }
        }
    };

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
                                   int64_t routeStart,
                                   bool emitMemoryRead = true) {
        const int64_t kvHead = head / queryHeadsPerKv;
        const int64_t capture = routeStart + memToSxm;
        const auto slices = layout.valuePackSlices(localMxm);
        if (emitMemoryRead) {
            for (int64_t stream = 0; stream < 16; ++stream) {
                const int64_t slice =
                    slices[static_cast<std::size_t>(stream)];
                const int64_t latency = memToSxm
                    - slice
                        / target_.streams().mem_slices_per_register_group;
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_, capture - latency,
                    hemisphere * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", layout.valuePackAddress(
                        kvHead, localMxm, keyBlock, 0),
                    inputStreams[static_cast<std::size_t>(stream)],
                    tileRows, 1, 1, 1, 1, 0, 1, 1, 0, -1, valueBank);
            }
        }
        std::array<int64_t, 16> mxmStreams {};
        const int64_t singleMxmWeightStreamBase =
            target_.streams().streams_per_direction - 16;
        for (int64_t stream = 0; stream < 16; ++stream)
            mxmStreams[static_cast<std::size_t>(stream)] =
                (singleMxm ? singleMxmWeightStreamBase : localMxm * 16)
                + stream;
        transposeCaptures[static_cast<std::size_t>(hemisphere)]
            .push_back(capture);
        emitSxm(rewriter_, op_.getLoc(), capture + 1, hemisphere, "permute",
            transposeStreams, mxmStreams, blockDiagonalMap(0, target_),
            "matrix_columns", -1, -1, -1, 2 * tileRows - 1,
            1, 1, 1, lanes);
        emitMxmWave(rewriter_, op_.getLoc(), capture + 2,
            hemisphere * target_.throughput().mxms_per_hemisphere
                + (singleMxm ? 0 : localMxm),
            "iw",
            singleMxm
                ? localMxm % target_.throughput().mxm_weight_buffers
                : 0,
            0, 0, 0, 1, 1, 0, 1, "stream", true,
            "supercell", 0, dataFormat, {}, {}, tileRows, 1, 1,
            1, 1, 0,
            singleMxm ? singleMxmWeightStreamBase : -1);
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
        const bool pipelineHeadBlocks = queryBlocks == 1;
        const auto contextSlices = layout.contextSlices();
        const auto disjointFromContext = [&](llvm::ArrayRef<int64_t> slices) {
            return std::none_of(slices.begin(), slices.end(),
                [&](int64_t slice) {
                    return std::find(contextSlices.begin(),
                        contextSlices.end(), slice)
                        != contextSlices.end();
                });
        };
        bool groupPvMemory = pipelineHeadBlocks
            && valueBank != probabilityBank
            && disjointFromContext(layout.probabilityDiagonalSlices());
        for (int64_t headBlock = 0;
             groupPvMemory && headBlock < headBlocks; ++headBlock)
            groupPvMemory &= disjointFromContext(
                layout.valuePackSlices(headBlock));
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
        // The context destinations are disjoint from the PV input queues
        // when groupPvMemory is true. Plan the complete output windows before
        // lowering them, so a physical MEM ICU can execute an entire affine
        // context write as one non-preemptible 3D instruction.
        struct ContextWriteWindow {
            int64_t cycle;
            int64_t queue;
            int64_t address;
            int64_t stream;
            int64_t bank;
            bool tap;
        };
        std::vector<ContextWriteWindow> contextWriteWindows;
        // Keep the work-plan coordinates for every wave.  A value or
        // probability MEM queue may run through several complete PV waves
        // without another command, even though its SRAM address resets (or
        // advances only once) at each wave boundary.
        struct PvReadWavePlan {
            std::array<std::optional<int64_t>, 2> heads;
            std::array<std::vector<int64_t>, 2> probabilityStarts;
            std::array<std::vector<int64_t>, 2> valueRouteStarts;
        };
        std::vector<PvReadWavePlan> readWavePlans;
        for (const auto& wave : waves) {
            // With one query tile, every head block replays the same
            // probability address on its physical MEM queue.  Retain the
            // planned compute starts and lower them as one 3D domain after
            // the wave has been timed.  Keep separate domains when value and
            // probability share a physical MEM queue.
            std::array<std::vector<int64_t>, 2> probabilityStarts;
            std::array<std::vector<int64_t>, 2> valueRouteStarts;
            for (int64_t headBlock = 0;
                 headBlock < headBlocks; ++headBlock) {
                const int64_t weightBuffer = headBlock
                    % target_.throughput().mxm_weight_buffers;
                for (int64_t keyBlock = 0;
                     keyBlock < keyBlocks; ++keyBlock) {
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
                            if (groupPvMemory)
                                valueRouteStarts[static_cast<std::size_t>(
                                    hemisphere)].push_back(
                                        routeStarts[static_cast<std::size_t>(
                                            hemisphere)]);
                            const int64_t ready = emitValueLoad(
                                hemisphere, *head, keyBlock, headBlock,
                                routeStarts[static_cast<std::size_t>(hemisphere)],
                                !groupPvMemory);
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
                          queryBlock < queryBlocks; ++queryBlock) {
                        const bool finalReduction =
                            keyBlock + 1 == keyBlocks;
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
                            if (groupPvMemory) {
                                probabilityStarts[static_cast<std::size_t>(
                                    hemisphere)].push_back(
                                        hemisphereCompute);
                            } else {
                                for (int64_t row = 0; row < lanes; ++row) {
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
                                        direct_domain_detail::emitMem3D(
                                            rewriter_, op_.getLoc(), target_,
                                            hemisphereCompute + row - *latency,
                                            hemisphere
                                                    * target_.memory()
                                                          .slices_per_hemisphere
                                                + slice,
                                            "read",
                                            layout.probabilityDiagonalAddress(
                                                *head, queryBlock, keyBlock,
                                                0),
                                            activationStreamBase + byte,
                                            tileRows, lanes, 1,
                                            1, 1, 0, 1, 1, 0, -1,
                                            probabilityBank);
                                    }
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
                                    const int64_t queue = hemisphere
                                        * target_.memory()
                                              .slices_per_hemisphere
                                        + slice;
                                    const int64_t address =
                                        layout.contextAddress(
                                            *head, headBlock,
                                            queryBlock * tile);
                                    if (groupPvMemory)
                                        contextWriteWindows.push_back({
                                            writeCycle, queue, address,
                                            32 + byte, contextBank, false});
                                    else
                                        direct_domain_detail::emitMem3D(
                                            rewriter_, op_.getLoc(), target_,
                                            writeCycle, queue, "write",
                                            address, 32 + byte, tile, 1, 1,
                                            1, 1, 0, 1, 1, 0, -1,
                                            contextBank);
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
                                    const int64_t queue =
                                        destinationHemisphere
                                            * target_.memory()
                                                  .slices_per_hemisphere
                                        + slice;
                                    const int64_t address =
                                        layout.contextAddress(
                                            *head, headBlock,
                                            queryBlock * tile);
                                    if (groupPvMemory)
                                        contextWriteWindows.push_back({
                                            writeCycle, queue, address,
                                            packedStream, contextBank, local});
                                    else
                                        direct_domain_detail::emitMem3D(
                                            rewriter_, op_.getLoc(), target_,
                                            writeCycle, queue,
                                            local ? "write_tap" : "write",
                                            address, packedStream, tile, 1, 1,
                                            1, 1, 0, 1, 1, 0, -1,
                                            contextBank);
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
            if (groupPvMemory)
                readWavePlans.push_back({wave,
                    std::move(probabilityStarts),
                    std::move(valueRouteStarts)});
        }
        if (groupPvMemory) {
            struct PlannedRead {
                int64_t cycle;
                int64_t queue;
                int64_t address;
                int64_t stream;
                int64_t bank;
                int64_t repeatCount;
                int64_t repeatInterval;
                int64_t addressStride;
                std::size_t wave;
                int64_t headBlock;
            };
            std::vector<PlannedRead> reads;
            for (std::size_t wave = 0; wave < readWavePlans.size();
                 ++wave) {
                const auto& plan = readWavePlans[wave];
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const auto head = plan.heads[
                        static_cast<std::size_t>(hemisphere)];
                    if (!head) continue;
                    const auto& routes = plan.valueRouteStarts[
                        static_cast<std::size_t>(hemisphere)];
                    const auto& starts = plan.probabilityStarts[
                        static_cast<std::size_t>(hemisphere)];
                    const int64_t plannedBlocks = headBlocks * keyBlocks;
                    if (routes.size()
                            != static_cast<std::size_t>(plannedBlocks)
                        || starts.size()
                            != static_cast<std::size_t>(plannedBlocks))
                        return -1;
                    const int64_t kvHead = *head / queryHeadsPerKv;
                    for (int64_t headBlock = 0;
                         headBlock < headBlocks; ++headBlock) {
                      for (int64_t keyBlock = 0;
                           keyBlock < keyBlocks; ++keyBlock) {
                        const int64_t planIndex =
                            headBlock * keyBlocks + keyBlock;
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice =
                                layout.valuePackSlices(headBlock)[stream];
                            const int64_t latency = memToSxm
                                - slice / target_.streams()
                                    .mem_slices_per_register_group;
                            reads.push_back({
                                routes[planIndex] + memToSxm - latency,
                                hemisphere
                                        * target_.memory()
                                              .slices_per_hemisphere
                                    + slice,
                                layout.valuePackAddress(
                                    kvHead, headBlock, keyBlock, 0),
                                inputStreams[static_cast<std::size_t>(stream)],
                                valueBank, tileRows, 1, 1,
                                wave, planIndex});
                        }
                        for (int64_t row = 0; row < lanes; ++row) {
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice =
                                    layout.probabilityDiagonalSlices()
                                        [row * 2 + byte];
                                const auto latency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::Mem,
                                        target::StreamEndpoint::MxmActivation,
                                        target::StreamDirection::East, slice);
                                if (!latency) return -1;
                                reads.push_back({
                                    starts[planIndex] + row - *latency,
                                    hemisphere
                                            * target_.memory()
                                                  .slices_per_hemisphere
                                        + slice,
                                    layout.probabilityDiagonalAddress(
                                        *head, 0, keyBlock, 0),
                                    activationStreamBase + byte,
                                    probabilityBank, tileRows, lanes, 1,
                                    wave, planIndex});
                            }
                        }
                      }
                    }
                }
            }
            std::sort(reads.begin(), reads.end(),
                [](const PlannedRead& lhs, const PlannedRead& rhs) {
                    return std::tie(lhs.queue, lhs.bank, lhs.cycle,
                               lhs.stream)
                        < std::tie(rhs.queue, rhs.bank, rhs.cycle,
                               rhs.stream);
                });
            const auto sameQueueAndShape = [](const PlannedRead& lhs,
                                               const PlannedRead& rhs) {
                return lhs.queue == rhs.queue && lhs.bank == rhs.bank
                    && lhs.stream == rhs.stream
                    && lhs.repeatCount == rhs.repeatCount
                    && lhs.repeatInterval == rhs.repeatInterval
                    && lhs.addressStride == rhs.addressStride;
            };
            for (std::size_t begin = 0; begin < reads.size();) {
                const auto& first = reads[begin];
                std::size_t end = begin + 1;
                int64_t cycleStride = 1;
                int64_t addressStride = 0;
                int64_t outerGroupSize = 1;
                int64_t outerInnerStride = 0;
                int64_t outerGroupStride = 0;
                const int64_t issueSpan =
                    (first.repeatCount - 1) * first.repeatInterval + 1;
                // The head-block index is the blocked outer counter's low
                // bits.  A complete, affine run of at least two waves can
                // therefore reset or advance the address once per wave.
                const int64_t plannedBlocks = headBlocks * keyBlocks;
                if (plannedBlocks > 1
                    && (plannedBlocks & (plannedBlocks - 1)) == 0
                    && first.headBlock == 0
                    && begin + 2 * plannedBlocks <= reads.size()
                    && sameQueueAndShape(first, reads[begin + 1])) {
                    cycleStride = reads[begin + 1].cycle - first.cycle;
                    outerInnerStride =
                        reads[begin + 1].address - first.address;
                    outerGroupStride =
                        reads[begin + plannedBlocks].address - first.address;
                    if (cycleStride >= issueSpan) {
                        std::size_t candidate = begin;
                        while (candidate < reads.size()
                            && candidate - begin < 65536) {
                            const auto offset = candidate - begin;
                            const auto& next = reads[candidate];
                            if (!sameQueueAndShape(first, next)
                                || next.wave != first.wave
                                         + offset / plannedBlocks
                                || next.headBlock
                                    != static_cast<int64_t>(
                                         offset % plannedBlocks)
                                || next.cycle != first.cycle
                                        + static_cast<int64_t>(offset)
                                            * cycleStride
                                || next.address != first.address
                                        + static_cast<int64_t>(
                                             offset % plannedBlocks)
                                            * outerInnerStride
                                        + static_cast<int64_t>(
                                             offset / plannedBlocks)
                                            * outerGroupStride)
                                break;
                            ++candidate;
                        }
                        if (candidate - begin
                                >= static_cast<std::size_t>(
                                     2 * plannedBlocks)) {
                            end = begin + (candidate - begin)
                                    / plannedBlocks * plannedBlocks;
                            outerGroupSize = plannedBlocks;
                        }
                    }
                }
                if (outerGroupSize == 1 && begin + 1 < reads.size()
                    && sameQueueAndShape(first, reads[begin + 1])) {
                    cycleStride = reads[begin + 1].cycle - first.cycle;
                    addressStride =
                        reads[begin + 1].address - first.address;
                    if (cycleStride >= issueSpan) {
                        end = begin + 2;
                        while (end < reads.size()
                            && sameQueueAndShape(first, reads[end])
                            && reads[end].cycle - reads[end - 1].cycle
                                == cycleStride
                            && reads[end].address
                                    - reads[end - 1].address
                                == addressStride)
                            ++end;
                    }
                }
                if (outerGroupSize > 1) {
                    attention_detail::emitMem3D(rewriter_, op_.getLoc(),
                        first.cycle, first.queue, "read", first.address,
                        first.stream, first.repeatCount,
                        first.repeatInterval, first.addressStride,
                        "", -1, 1, 1, 0,
                        static_cast<int64_t>(end - begin), cycleStride, 0,
                        first.bank, -1, -1, outerGroupSize,
                        outerInnerStride, outerGroupStride);
                } else {
                    direct_domain_detail::emitMem3D(rewriter_, op_.getLoc(),
                        target_, first.cycle, first.queue, "read",
                        first.address, first.stream,
                        first.repeatCount, first.repeatInterval,
                        first.addressStride, 1, 1, 0,
                        static_cast<int64_t>(end - begin), cycleStride,
                        addressStride, -1, first.bank);
                }
                begin = end;
            }
            std::sort(contextWriteWindows.begin(), contextWriteWindows.end(),
                [](const auto& lhs, const auto& rhs) {
                    return std::tie(lhs.queue, lhs.cycle, lhs.stream)
                        < std::tie(rhs.queue, rhs.cycle, rhs.stream);
                });
            for (std::size_t begin = 0;
                 begin < contextWriteWindows.size();) {
                const auto& first = contextWriteWindows[begin];
                std::size_t end = begin + 1;
                int64_t cycleStride = 1;
                int64_t addressStride = 0;
                if (end < contextWriteWindows.size()
                    && contextWriteWindows[end].queue == first.queue
                    && contextWriteWindows[end].bank == first.bank
                    && contextWriteWindows[end].stream == first.stream
                    && contextWriteWindows[end].tap == first.tap
                    && contextWriteWindows[end].cycle - first.cycle >= tile) {
                    cycleStride = contextWriteWindows[end].cycle
                        - first.cycle;
                    addressStride = contextWriteWindows[end].address
                        - first.address;
                    while (end < contextWriteWindows.size()) {
                        const auto& previous = contextWriteWindows[end - 1];
                        const auto& next = contextWriteWindows[end];
                        if (next.queue != first.queue
                            || next.bank != first.bank
                            || next.stream != first.stream
                            || next.tap != first.tap
                            || next.cycle - previous.cycle != cycleStride
                            || next.address - previous.address
                                != addressStride)
                            break;
                        ++end;
                    }
                }
                direct_domain_detail::emitMem3D(
                    rewriter_, op_.getLoc(), target_, first.cycle,
                    first.queue, first.tap ? "write_tap" : "write",
                    first.address, first.stream, tile, 1, 1,
                    1, 1, 0, static_cast<int64_t>(end - begin),
                    cycleStride, addressStride, -1, first.bank);
                begin = end;
            }
        }
        if (pipelineHeadBlocks)
            phaseStart = std::max(nextPipelinedCompute, pipelinedEnd);
        emitPlannedTransposes();
        return phaseStart + groups;
    }

    for (const auto& wave : waves) {
        for (int64_t keyBlock = 0; keyBlock < keyBlocks; ++keyBlock) {
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
            for (int64_t queryBlock = 0; queryBlock < queryBlocks; ++queryBlock) {
                int64_t blockEnd = phaseStart;
                for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const auto head = wave[static_cast<std::size_t>(hemisphere)];
                    if (!head) continue;
                    const int64_t replayStart = phaseStart;
                    const int64_t firstCompute = replayStart + memToMxm;
                    const bool finalReduction =
                        keyBlock + 1 == keyBlocks;
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
                            for (int64_t row = 0; row < lanes; ++row) {
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
                                    direct_domain_detail::emitMem3D(
                                        rewriter_, op_.getLoc(), target_,
                                        computeCycle + row - latency,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "read",
                                        layout.probabilityDiagonalAddress(*head,
                                            queryBlock, keyBlock, 0),
                                        activationStreamBase + byte,
                                        tileRows, lanes, 1,
                                        1, 1, 0, 1, 1, 0, -1,
                                        probabilityBank);
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

                    if (keyBlock + 1 == keyBlocks) {
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
                                        direct_domain_detail::emitMem3D(
                                            rewriter_, op_.getLoc(), target_,
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
                                            1, 1, 0, 1, 1, 0, -1,
                                            contextBank);
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
    emitPlannedTransposes();
    return end;
}

} // namespace ftlpu::compiler::schedule
