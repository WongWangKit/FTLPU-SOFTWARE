#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_softmax_planner.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
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
    // Softmax spans the resident KV window. Decode keeps one padded query
    // tile, while scores and probabilities cover all past tokens plus the
    // current token in the padded KV window.
    const int64_t sequence = op_.getKvSeqLen();
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
    struct PlannedMemTransfer {
        int64_t cycle;
        int64_t queue;
        llvm::StringRef opcode;
        int64_t address;
        int64_t stream;
        int64_t repeat_count;
        int64_t repeat_interval;
        int64_t address_stride;
        int64_t wave_count;
        int64_t wave_interval;
        int64_t wave_address_stride;
        int64_t bank;
    };
    const auto planMirroredRead = [&](std::vector<PlannedMemTransfer>& plan,
                                      llvm::ArrayRef<int64_t> slices,
                                      int64_t address, int64_t stream,
                                      int64_t inputCycle, int64_t count,
                                      int64_t stride, int64_t bank,
                                      int64_t waveCount = 1,
                                      int64_t waveInterval = 1,
                                      int64_t waveAddressStride = 0) {
        for (int64_t hemisphere = 0;
             hemisphere < target_.memory().hemispheres; ++hemisphere) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = slices[byte];
                plan.push_back({inputCycle - readLatency(slice),
                    hemisphere
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", address,
                    32 + hemisphere * 16 + stream + byte,
                    count, 1, stride,
                    waveCount, waveInterval, waveAddressStride, bank});
            }
        }
    };
    const auto planMirroredWrite = [&](std::vector<PlannedMemTransfer>& plan,
                                       llvm::ArrayRef<int64_t> slices,
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
                plan.push_back({outputCycle + writeLatency(slice),
                    destination
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "write", address,
                    source * 8 + originalOutputStream + byte,
                    count, repeatInterval, stride,
                    1, 1, 0, bank});
            }
        }
    };
    const auto emitPlannedTransfers =
        [&](std::vector<PlannedMemTransfer>& plan) {
        std::sort(plan.begin(), plan.end(),
            [](const PlannedMemTransfer& lhs,
                const PlannedMemTransfer& rhs) {
                return std::tie(lhs.queue, lhs.opcode, lhs.bank, lhs.cycle)
                    < std::tie(rhs.queue, rhs.opcode, rhs.bank, rhs.cycle);
            });
        const auto sameShape = [](const PlannedMemTransfer& lhs,
                                  const PlannedMemTransfer& rhs) {
            return lhs.queue == rhs.queue && lhs.opcode == rhs.opcode
                && lhs.stream == rhs.stream && lhs.bank == rhs.bank
                && lhs.repeat_count == rhs.repeat_count
                && lhs.repeat_interval == rhs.repeat_interval
                && lhs.address_stride == rhs.address_stride
                && lhs.wave_count == rhs.wave_count
                && lhs.wave_interval == rhs.wave_interval
                && lhs.wave_address_stride == rhs.wave_address_stride;
        };
        for (std::size_t begin = 0; begin < plan.size();) {
            const PlannedMemTransfer& base = plan[begin];
            std::size_t end = begin + 1;
            while (end < plan.size() && sameShape(base, plan[end])) ++end;
            const std::size_t count = end - begin;
            const int64_t issueSpan =
                (base.repeat_count - 1) * base.repeat_interval
                + (base.wave_count - 1) * base.wave_interval + 1;
            const int64_t cycleStride = count > 1
                ? plan[begin + 1].cycle - base.cycle : 1;
            bool affineCycles = count > 1 && cycleStride >= issueSpan;
            for (std::size_t index = 0;
                 affineCycles && index < count; ++index)
                affineCycles = plan[begin + index].cycle
                    == base.cycle
                        + static_cast<int64_t>(index) * cycleStride;

            int64_t groupAddressStride = 0;
            bool affineAddress = affineCycles;
            if (count > 1) {
                groupAddressStride =
                    plan[begin + 1].address - base.address;
                for (std::size_t index = 0;
                     affineAddress && index < count; ++index)
                    affineAddress = plan[begin + index].address
                        == base.address
                            + static_cast<int64_t>(index)
                                * groupAddressStride;
            }

            int64_t outerGroupSize = 1;
            int64_t outerInnerStride = 0;
            int64_t outerGroupStride = 0;
            if (affineCycles && !affineAddress) {
                for (std::size_t candidate = 2;
                     candidate < count && candidate <= 65536;
                     candidate *= 2) {
                    const int64_t innerStride =
                        plan[begin + 1].address - base.address;
                    const int64_t outerStride =
                        plan[begin + candidate].address - base.address;
                    bool blocked = true;
                    for (std::size_t index = 0;
                         blocked && index < count; ++index)
                        blocked = plan[begin + index].address
                            == base.address
                                + static_cast<int64_t>(index % candidate)
                                    * innerStride
                                + static_cast<int64_t>(index / candidate)
                                    * outerStride;
                    if (!blocked) continue;
                    outerGroupSize = static_cast<int64_t>(candidate);
                    outerInnerStride = innerStride;
                    outerGroupStride = outerStride;
                    break;
                }
            }

            if (affineAddress || outerGroupSize > 1) {
                attention_detail::emitMem3D(rewriter_, op_.getLoc(),
                    base.cycle, base.queue, base.opcode, base.address,
                    base.stream, base.repeat_count, base.repeat_interval,
                    base.address_stride, "", -1,
                    base.wave_count, base.wave_interval,
                    base.wave_address_stride,
                    static_cast<int64_t>(count), cycleStride,
                    affineAddress ? groupAddressStride : 0,
                    base.bank, -1, -1, outerGroupSize,
                    outerInnerStride, outerGroupStride);
            } else {
                for (std::size_t index = begin; index < end; ++index) {
                    const PlannedMemTransfer& transfer = plan[index];
                    direct_domain_detail::emitMem3D(
                        rewriter_, op_.getLoc(), target_, transfer.cycle,
                        transfer.queue, transfer.opcode, transfer.address,
                        transfer.stream, transfer.repeat_count,
                        transfer.repeat_interval, transfer.address_stride,
                        transfer.wave_count, transfer.wave_interval,
                        transfer.wave_address_stride, 1, 1, 0, -1,
                        transfer.bank);
                }
            }
            begin = end;
        }
    };
    struct ProbabilityPackWrite {
        int64_t cycle;
        int64_t queue;
        int64_t address;
        int64_t stream;
        int64_t count;
        int64_t interval;
    };
    std::vector<ProbabilityPackWrite> pendingPackWrites;
    const auto probabilityPackSlices = layout.probabilityPackSlices();
    const auto canGroupPackSlice = [&](int64_t slice) {
        if (std::count(probabilityPackSlices.begin(),
                probabilityPackSlices.end(), slice) != 1)
            return false;
        for (const auto& wave : stage_plan_.qk_waves) {
            for (const auto& optionalWork : wave.slots) {
                if (!optionalWork) continue;
                const int64_t localMxm = optionalWork->local_mxm;
                const auto overlaps = [&](llvm::ArrayRef<int64_t> other) {
                    return std::find(other.begin(), other.end(), slice)
                        != other.end();
                };
                if ((probabilityPackBank == scoreBank
                        && overlaps(layout.scaledScoreSlices(localMxm)))
                    || (probabilityPackBank == expBank
                        && overlaps(layout.expScoreSlices(localMxm)))
                    || (op_.getCausal() && probabilityPackBank == maskBank
                        && overlaps(layout.causalMaskSlices(localMxm))))
                    return false;
            }
        }
        return true;
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

    struct SoftmaxWorkState {
        const AttentionWorkItem* work = nullptr;
        int64_t hemisphere = 0;
        llvm::SmallVector<int64_t, 4> score_slices;
        llvm::SmallVector<int64_t, 4> x_slices;
        llvm::SmallVector<int64_t, 4> mask_slices;
        int64_t score_address = 0;
        int64_t x_address = 0;
    };
    std::vector<SoftmaxWorkState> workStates;
    const int64_t queryBlocks = (op_.getSeqLen() + tile - 1) / tile;
    for (const AttentionWorkWave& wave : stage_plan_.qk_waves) {
        for (const auto& optionalWork : wave.slots) {
            if (!optionalWork) continue;
            const AttentionWorkItem& work = *optionalWork;
            SoftmaxWorkState state;
            state.work = &work;
            state.hemisphere = work.hemisphere;
            const auto scoreSlices = layout.scaledScoreSlices(work.local_mxm);
            const auto xSlices = layout.expScoreSlices(work.local_mxm);
            const auto maskSlices = layout.causalMaskSlices(work.local_mxm);
            state.score_slices.assign(scoreSlices.begin(), scoreSlices.end());
            state.x_slices.assign(xSlices.begin(), xSlices.end());
            state.mask_slices.assign(maskSlices.begin(), maskSlices.end());
            state.score_address = layout.scoreAddress(
                work.query_head, work.query_block, 0);
            state.x_address = layout.expScoreAddress(
                (work.query_head * queryBlocks + work.query_block)
                    * sequence);
            workStates.push_back(std::move(state));
        }
    }

    int64_t cursor = qkEnd + 8;

    // Phase 0: materialize every cross-hemisphere score mirror first.  Source
    // hemisphere 0 is completed before source hemisphere 1, so a physical MEM
    // queue changes direction at most once during the mirror phase instead of
    // alternating read/write for every attention work item.
    std::vector<PlannedMemTransfer> mirrorTransfers;
    for (int64_t sourceHemisphere = 0;
         sourceHemisphere < target_.memory().hemispheres;
         ++sourceHemisphere) {
        for (const SoftmaxWorkState& state : workStates) {
            if (state.hemisphere != sourceHemisphere) continue;
            const int64_t copyBridge = cursor
                + maxReadLatency(state.score_slices) + 1;
            const int64_t mirrorWriteEnd = copyBridge
                + maxWriteLatency(state.score_slices) + sequence - 1;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = state.score_slices[byte];
                mirrorTransfers.push_back({copyBridge - readLatency(slice),
                    sourceHemisphere
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "read", state.score_address, 32 + byte,
                    sequence, 1, 1, 1, 1, 0, scoreBank});
                mirrorTransfers.push_back({copyBridge + writeLatency(slice),
                    (1 - sourceHemisphere)
                            * target_.memory().slices_per_hemisphere
                        + slice,
                    "write", state.score_address, byte,
                    sequence, 1, 1, 1, 1, 0, scoreBank});
            }
            cursor = mirrorWriteEnd + 1;
        }
    }
    emitPlannedTransfers(mirrorTransfers);

    // Phase 1: generate every scaled/masked x row before any x row is read.
    // The exp allocation already reserves the complete work axis; use its
    // per-work address instead of repeatedly overwriting row zero.
    std::vector<PlannedMemTransfer> generateTransfers;
    for (const SoftmaxWorkState& state : workStates) {
        const AttentionWorkItem& work = *state.work;
        const int64_t passReadLead = std::max(
            maxReadLatency(state.score_slices),
            maxReadLatency(state.mask_slices));
        const int64_t generateInput = cursor + passReadLead + 1;
        const int64_t generateConfig = generateInput - 1;
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
                planMirroredRead(generateTransfers, state.mask_slices,
                    layout.causalMaskAddress(localKey), 2,
                    generateInput + firstKey, count, 1, maskBank);
            }
        };
        if (!op_.getCausal()) {
            emitGenerateRun(0, sequence, false, 0.0f);
        } else {
            const int64_t absoluteQueryBlock =
                op_.getPositionOffset() / tile + work.query_block;
            const int64_t vectorBegin = absoluteQueryBlock * tile + 1;
            emitGenerateRun(0, vectorBegin, false, 0.0f);
            emitGenerateRun(vectorBegin, tile - 1, true, 0.0f);
            emitGenerateRun((absoluteQueryBlock + 1) * tile,
                sequence - (absoluteQueryBlock + 1) * tile,
                false, causalMaskValue);
        }
        planMirroredRead(generateTransfers,
            state.score_slices, state.score_address, 0,
            generateInput, sequence, 1, scoreBank);
        const int64_t generateOutput = generateInput + 2;
        planMirroredWrite(generateTransfers,
            state.x_slices, state.x_address, 0,
            generateOutput, sequence, 1, expBank);
        cursor = generateOutput + maxWriteLatency(state.x_slices)
            + sequence;
    }
    emitPlannedTransfers(generateTransfers);

    // Phase 2: read every x row and write every max scalar.  No x write is
    // left in this phase, so each x queue has one write-to-read boundary for
    // the complete Softmax operator.
    std::vector<PlannedMemTransfer> maxTransfers;
    for (const SoftmaxWorkState& state : workStates) {
        const int64_t maxInput = cursor
            + maxReadLatency(state.x_slices) + 1;
        const int64_t maxConfig = maxInput - 1;
        const int64_t maxOutput = maxInput + sequence;
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
        planMirroredRead(maxTransfers, state.x_slices, state.x_address, 0,
            maxInput, sequence, 1, expBank);
        planMirroredWrite(maxTransfers,
            state.score_slices, state.score_address, 0,
            maxOutput, 1, 0, scoreBank);
        cursor = maxOutput + maxWriteLatency(state.score_slices) + 1;
    }
    emitPlannedTransfers(maxTransfers);

    // Phases 3 and 4 remain adjacent per work item so reciprocal(sum) stays in
    // the FP32 VXM local scalar.  All SRAM operands are reads in both phases;
    // only the disjoint probability-pack queues receive writes.
    std::vector<PlannedMemTransfer> normalizeTransfers;
    for (const SoftmaxWorkState& state : workStates) {
        const AttentionWorkItem& work = *state.work;
        const int64_t sumInput = cursor
            + std::max(maxReadLatency(state.x_slices),
                  maxReadLatency(state.score_slices))
            + 1;
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
        const int64_t normalizeConfig = sumInput + sequence + 12;
        const int64_t normalizeInput = normalizeConfig + 1;

        planMirroredRead(normalizeTransfers,
            state.x_slices, state.x_address, 0,
            sumInput, sequence, 1, expBank, 2,
            normalizeInput - sumInput, 0);
        for (int64_t source = 0;
             source < target_.memory().hemispheres; ++source) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = state.score_slices[byte];
                const int64_t queue = source
                    * target_.memory().slices_per_hemisphere + slice;
                const int64_t stream = 32 + source * 16 + 2 + byte;
                normalizeTransfers.push_back({
                    sumInput - readLatency(slice), queue,
                    "read", state.score_address, stream,
                    sequence, 1, 0, 2,
                    normalizeInput - sumInput, 0, scoreBank});
            }
        }

        vxm(normalizeConfig, 0, "subtract", streamKind, 32, 0.0f,
            streamKind, 34, 0.0f, "fp32", -1, sequence, 4);
        vxm(normalizeConfig, 1, "exp", "previous", 0, 0.0f,
            "immediate", 0, 0.0f, "fp32", -1, sequence, 4);
        vxm(normalizeConfig, 2, "pass", "previous", 0, 0.0f,
            "immediate", 0, 0.0f, "fp32", -1, sequence, 4);
        vxm(normalizeConfig, 3, "multiply", "previous", 0, 0.0f,
            "accumulator", 0, 0.0f, dataFormat, 2,
            sequence, 4);
        constexpr int64_t normalizePipelineLatency = 8;
        const int64_t keysPerLane = sequence
            / target_.throughput().lanes_per_tile;
        for (int64_t lane = 0;
             lane < target_.throughput().lanes_per_tile; ++lane) {
            const int64_t packedStream = lane * 2;
            const int64_t address = layout.probabilityPackAddress(
                work.query_head, work.query_block, 0);
            for (int64_t destination = 0;
                 destination < target_.memory().hemispheres;
                 ++destination) {
                const int64_t source = 1 - destination;
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = probabilityPackSlices[
                        static_cast<std::size_t>(packedStream + byte)];
                    const int64_t cycle = normalizeInput
                        + normalizePipelineLatency + lane
                        + writeLatency(slice);
                    const int64_t queue = destination
                        * target_.memory().slices_per_hemisphere + slice;
                    const int64_t stream = source * 8 + 2 + byte;
                    if (canGroupPackSlice(slice)) {
                        pendingPackWrites.push_back({cycle, queue,
                            address, stream, keysPerLane,
                            target_.throughput().lanes_per_tile});
                    } else {
                        direct_domain_detail::emitMem3D(
                            rewriter_, op_.getLoc(), target_, cycle,
                            queue, "write", address, stream,
                            keysPerLane,
                            target_.throughput().lanes_per_tile, 1,
                            1, 1, 0, 1, 1, 0, -1,
                            probabilityPackBank);
                    }
                }
            }
        }
        cursor = normalizeInput + normalizePipelineLatency
            + sequence + 8
            + target_.throughput().tile_rows
            + target_.streams().system_register_columns;
    }
    emitPlannedTransfers(normalizeTransfers);
    // Probability-pack slices that are not used by score, exp, or mask have
    // one write per work and no intervening command on their physical ICU.
    // Plan the complete work axis before emitting its native 3-D domain.
    std::sort(pendingPackWrites.begin(), pendingPackWrites.end(),
        [](const ProbabilityPackWrite& lhs,
            const ProbabilityPackWrite& rhs) {
            return std::tie(lhs.queue, lhs.cycle)
                < std::tie(rhs.queue, rhs.cycle);
        });
    const auto emitPackWrite = [&](const ProbabilityPackWrite& write) {
        direct_domain_detail::emitMem3D(rewriter_, op_.getLoc(), target_,
            write.cycle, write.queue, "write", write.address,
            write.stream, write.count, write.interval, 1,
            1, 1, 0, 1, 1, 0, -1, probabilityPackBank);
    };
    for (std::size_t begin = 0; begin < pendingPackWrites.size();) {
        std::size_t end = begin + 1;
        while (end < pendingPackWrites.size()
               && pendingPackWrites[end].queue
                   == pendingPackWrites[begin].queue)
            ++end;
        const auto& base = pendingPackWrites[begin];
        const std::size_t count = end - begin;
        const int64_t cycleStride = count > 1
            ? pendingPackWrites[begin + 1].cycle - base.cycle : 0;
        const int64_t innerAddressStride = count > 1
            ? pendingPackWrites[begin + 1].address - base.address : 0;
        const int64_t groupAddressStride = count > 2
            ? pendingPackWrites[begin + 2].address - base.address : 0;
        bool affine = count >= 2 && count % 2 == 0
            && cycleStride > (base.count - 1) * base.interval;
        for (std::size_t index = 0; affine && index < count; ++index) {
            const auto& write = pendingPackWrites[begin + index];
            affine = write.cycle == base.cycle
                    + static_cast<int64_t>(index) * cycleStride
                && write.address == base.address
                    + static_cast<int64_t>(index / 2)
                        * groupAddressStride
                    + static_cast<int64_t>(index % 2)
                        * innerAddressStride
                && write.stream == base.stream
                && write.count == base.count
                && write.interval == base.interval;
        }
        if (affine) {
            attention_detail::emitMem3D(rewriter_, op_.getLoc(),
                base.cycle, base.queue, "write", base.address,
                base.stream, base.count, base.interval, 1, "sram", -1,
                1, 1, 0, static_cast<int64_t>(count), cycleStride, 0,
                probabilityPackBank, -1, -1, 2,
                innerAddressStride, groupAddressStride);
        } else {
            for (std::size_t index = begin; index < end; ++index)
                emitPackWrite(pendingPackWrites[index]);
        }
        begin = end;
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
    const int64_t tokenBlocks =
        op_.getKvSeqLen() / target_.throughput().mxm_rows;
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
    std::array<bool, 2> groupedTranspose {};
    for (int64_t hemisphere = 0;
         hemisphere < target_.memory().hemispheres; ++hemisphere) {
        const auto& work = hemisphereWork[static_cast<std::size_t>(
            hemisphere)];
        if (work.size() < 2) continue;
        const int64_t interval = work[1].start - work[0].start;
        if (interval < transposeBeats
            || std::adjacent_find(work.begin(), work.end(),
                [&](const TransposeWork& lhs, const TransposeWork& rhs) {
                    return rhs.start - lhs.start != interval;
                }) != work.end())
            continue;
        groupedTranspose[static_cast<std::size_t>(hemisphere)] = true;
        emitSxm(rewriter_, op_.getLoc(), work[0].start + memToSxm,
            hemisphere, "transpose", inputStreams, transposeStreams,
            identityMap(), "vector_columns", -1, -1, -1,
            transposeBeats, 1, static_cast<int64_t>(work.size()),
            interval);
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
            if (!groupedTranspose[static_cast<std::size_t>(hemisphere)])
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
