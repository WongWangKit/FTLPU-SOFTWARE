#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;

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
                        const int64_t outputStream = localMxm * 4;
                        emitMxm(rewriter_, op_.getLoc(), firstCompute,
                            hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
                            "compute", 0, 0, 0, outputStream, tile, 1,
                            layout.contextAddress(
                                *head, queryBlock * tile),
                            1, "sram");
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
                                emitMxm(rewriter_, op_.getLoc(),
                                    cycle
                                        - target_.throughput()
                                              .accumulator_read_to_vxm_latency,
                                    hemisphere
                                            * target_.throughput()
                                                  .mxms_per_hemisphere
                                        + localMxm,
                                    "accumulator_read", 0, 0, 0,
                                    localMxm * 4, 1, 1,
                                    layout.contextAddress(*head,
                                        queryBlock * tile + offset),
                                    1, "sram", true);
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

} // namespace ftlpu::compiler::schedule
