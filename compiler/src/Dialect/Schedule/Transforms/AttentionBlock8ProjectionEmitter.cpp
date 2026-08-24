#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <set>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;
namespace {

int64_t functionArgumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

llvm::SmallVector<int64_t> placementSlices(
    mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    if (!placement) return result;
    const auto values = placement.getAs<mlir::ArrayAttr>("slices");
    if (!values) return result;
    for (mlir::Attribute value : values)
        result.push_back(
            llvm::cast<mlir::IntegerAttr>(value).getInt());
    return result;
}

} // namespace

int64_t AttentionScheduleEmitter::emitBlock8Projections()
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(elementType);
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const auto& throughput = target_.throughput();
    const auto& memory = target_.memory();
    const int64_t tile = throughput.mxm_rows;
    const int64_t blockRows = throughput.mxm_block_rows;
    const int64_t blockIssues = tile / blockRows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t hiddenBlocks = op_.getHidden() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const auto inputPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("input");
    const auto placementBank = [&](llvm::StringRef name) {
        const auto placement = op_.getMemoryPlan()
            .getAs<mlir::DictionaryAttr>(name);
        return placement.getAs<mlir::IntegerAttr>("bank").getInt();
    };
    const int64_t inputBank = placementBank("input");
    const int64_t projectionWeightBanks[] = {
        placementBank("query_weight"), placementBank("key_weight"),
        placementBank("value_weight")};
    const int64_t ropeStagingBank = placementBank("rope_staging");
    const int64_t ropeProductBank = placementBank("rope_product");
    const int64_t queryBank = placementBank("query");
    const int64_t keyBank = placementBank("key");
    const int64_t valueBank = placementBank("value");
    const auto inputSlices = placementSlices(inputPlacement);
    const auto inputKind = inputPlacement
        ? inputPlacement.getAs<mlir::StringAttr>("kind")
        : mlir::StringAttr {};
    if (throughput.mxms_per_hemisphere != 1 || blockRows != 8
        || blockIssues != 4 || inputSlices.size() != 16
        || !inputKind
        || inputKind.getValue() != "fp16_mxm_distributed_16"
        || headBlocks < 2 || headBlocks > 4 || headBlocks % 2 != 0
        || 4 % headBlocks != 0) {
        op_.emitError(
            "Block8 attention requires one MXM per hemisphere and a "
            "16-slice distributed input with a 64- or 128-wide head");
        return -1;
    }

    const int64_t inputBase =
        inputPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t projectionHeads[] = {
        op_.getQueryHeads(), op_.getKvHeads(), op_.getKvHeads()};
    const int64_t queryHeadsPerKv =
        op_.getQueryHeads() / op_.getKvHeads();
    const mlir::Value projectionValues[] = {
        op_.getQueryWeight(), op_.getKeyWeight(), op_.getValueWeight()};
    const auto weightScale = [&](llvm::StringRef name) {
        const auto value =
            op_.query.getConfig().getAs<mlir::FloatAttr>(name);
        return value
            ? static_cast<float>(value.getValueAsDouble())
            : 1.0f;
    };
    const float projectionScales[] = {
        weightScale("query_weight_scale"),
        weightScale("key_weight_scale"),
        weightScale("value_weight_scale"),
    };
    int64_t maxWeightLatency = 0;
    for (int64_t slice : layout.weightSlices()) {
        const auto latency = target_.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        if (!latency) return -1;
        maxWeightLatency = std::max(maxWeightLatency, *latency);
    }
    int64_t maxRopeReadLatency = 0;
    const auto ropePlacement = op_.getMemoryPlan()
        .getAs<mlir::DictionaryAttr>("rope");
    const int64_t ropeBank = ropePlacement
        .getAs<mlir::IntegerAttr>("bank").getInt();
    for (int64_t half = 0; half < 2; ++half) {
        for (int64_t slice : target_.attention_query_iw_slices(half)) {
            const auto latency = target_.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, slice);
            if (!latency) return -1;
            maxRopeReadLatency = std::max(
                maxRopeReadLatency, *latency);
        }
    }
    for (int64_t slice : layout.ropeSlices()) {
        const auto latency = target_.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice);
        if (!latency) return -1;
        maxRopeReadLatency = std::max(maxRopeReadLatency, *latency);
    }

    std::set<int64_t> ropeProductControlCycles;
    std::set<int64_t> ropeCombineControlCycles;
    const auto emitRopeProducts = [&](int64_t cycle,
                                      int64_t inputHemisphere,
                                      int64_t outputHemisphere,
                                      mlir::Value value) {
        if (!ropeProductControlCycles.insert(cycle).second) return;
        const char* input = inputHemisphere == 0 ? "east" : "west";
        const char* output = outputHemisphere == 0 ? "east" : "west";
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 0,
            "multiply", streamKind, 32, 0.0f, streamKind, 34,
            0.0f, "fp32", -1, input, output, -1, 2,
            op_.getSeqLen(), 1);
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 1,
            "pass", "previous", 0, 0.0f, "immediate", 0,
            0.0f, dataFormat, 0, input, output, -1, 2,
            op_.getSeqLen(), 1);
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 2,
            "multiply", streamKind, 36, 0.0f, streamKind, 38,
            0.0f, "fp32", -1, input, output, -1, 2,
            op_.getSeqLen(), 1);
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 3,
            "pass", "previous", 0, 0.0f, "immediate", 0,
            0.0f, dataFormat, 2, input, output, -1, 2,
            op_.getSeqLen(), 1);
    };
    const auto emitRopeCombine = [&](int64_t cycle,
                                     int64_t inputHemisphere,
                                     int64_t outputHemisphere,
                                     mlir::Value value) {
        if (!ropeCombineControlCycles.insert(cycle).second) return;
        const char* input = inputHemisphere == 0 ? "east" : "west";
        const char* output = outputHemisphere == 0 ? "east" : "west";
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 0,
            "subtract", streamKind, 32, 0.0f, streamKind, 34,
            0.0f, "fp32", -1, input, output, -1, 2,
            op_.getSeqLen(), 1);
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 1,
            "pass", "previous", 0, 0.0f, "immediate", 0,
            0.0f, dataFormat, 0, input, output, -1, 2,
            op_.getSeqLen(), 1);
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 2,
            "add", streamKind, 36, 0.0f, streamKind, 38,
            0.0f, "fp32", -1, input, output, -1, 2,
            op_.getSeqLen(), 1);
        emitVxmConfigured(rewriter_, op_.getLoc(), value, cycle, 3,
            "pass", "previous", 0, 0.0f, "immediate", 0,
            0.0f, dataFormat, 2, input, output, -1, 2,
            op_.getSeqLen(), 1);
    };

    int64_t phaseStart = 1;
    int64_t ropeTail = 1;
    for (int64_t projection = 0; projection < 3; ++projection) {
        const auto kind = projectionKind(projection);
        const int64_t projectionOutputBlocks =
            projectionHeads[projection] * headBlocks;
        const int64_t projectionGroups =
            (projectionOutputBlocks + 3) / 4;
        for (int64_t outputGroup = 0;
             outputGroup < projectionGroups; ++outputGroup) {
            const int64_t firstCompute = phaseStart + maxWeightLatency
                + throughput.mxm_local_load_to_compute_latency;
            int64_t projectionEnd = firstCompute;
            const int64_t projectionIssueInterval = 2
                * memory.hemispheres * blockIssues;
            for (int64_t reduction = 0;
                 reduction < hiddenBlocks; ++reduction) {
                const int64_t reductionCompute = firstCompute
                    + reduction * tokenBlocks
                        * projectionIssueInterval;
                for (int64_t half = 0; half < 2; ++half) {
                    const int64_t weightBuffer =
                        half % throughput.mxm_weight_buffers;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres;
                         ++hemisphere) {
                        const int64_t outputBlock = outputGroup * 4
                            + hemisphere * 2 + half;
                        if (outputBlock >= projectionOutputBlocks) continue;
                        // Replace columns in the order already consumed by
                        // the preceding compute wave. This permits IW and
                        // compute to overlap without clobbering a later tile.
                        const int64_t loadSlot =
                            half * memory.hemispheres + hemisphere;
                        const int64_t firstUnitCompute = reductionCompute
                            + loadSlot * blockIssues;
                        const int64_t loadStart = firstUnitCompute
                            - throughput.mxm_local_load_to_compute_latency;
                        const int64_t address = layout.weightAddress(
                            kind, outputBlock, reduction, half,
                            throughput.tile_rows - 1);
                        for (int64_t stream = 0; stream < 8; ++stream) {
                            const int64_t slice =
                                layout.weightSlices()[stream];
                            const auto latency = target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmWeight,
                                target::StreamDirection::East, slice);
                            if (!latency) return -1;
                            emitMemWave(rewriter_, op_.getLoc(),
                                loadStart - *latency,
                                hemisphere * memory.slices_per_hemisphere
                                    + slice,
                                "read", address, stream, 1, 1, 0,
                                "sram",
                                functionArgumentIndex(
                                    projectionValues[projection]),
                                throughput.tile_rows, 1, -1,
                                projectionWeightBanks[projection]);
                        }
                        emitMxmDequantWave(rewriter_, op_.getLoc(),
                            loadStart, hemisphere,
                            projectionScales[projection], 1, 1,
                            throughput.tile_rows, 1,
                            functionArgumentIndex(
                                projectionValues[projection]));
                        emitMxmWave(rewriter_, op_.getLoc(), loadStart,
                            hemisphere, "iw", weightBuffer, 0,
                            0, 0, 1, 1, 0, 1, "sram", true,
                            "supercell", 0, dataFormat,
                            "int8_dequant_bf16", {}, {},
                            throughput.tile_rows, 1, 1);
                    }

                    const bool finalReduction =
                        reduction + 1 == hiddenBlocks;
                    const int64_t activationAddress = inputBase
                        + reduction * blockIssues;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres;
                         ++hemisphere) {
                            const int64_t outputBlock = outputGroup * 4
                                + hemisphere * 2 + half;
                            if (outputBlock >= projectionOutputBlocks)
                                continue;
                            const int64_t head = outputBlock / headBlocks;
                            const int64_t headBlock =
                                outputBlock % headBlocks;
                            const int64_t computeSlot =
                                half * memory.hemispheres + hemisphere;
                            const int64_t computeCycle = reductionCompute
                                + computeSlot * blockIssues;
                            for (int64_t stream = 0; stream < 16;
                                 ++stream) {
                                const int64_t slice = inputSlices[stream];
                                const auto latency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::Mem,
                                        target::StreamEndpoint::MxmActivation,
                                        target::StreamDirection::East,
                                        slice);
                                if (!latency) return -1;
                                emitMemWave(rewriter_, op_.getLoc(),
                                    computeCycle - *latency,
                                    hemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "read", activationAddress,
                                    2 * blockRows + stream,
                                    blockIssues, 1, 1, "sram",
                                    functionArgumentIndex(op_.getInput()),
                                    tokenBlocks, projectionIssueInterval,
                                    hiddenBlocks * blockIssues, inputBank);
                            }
                            const int64_t accumulatorBase =
                                half * tokenBlocks * blockIssues;
                            emitMxmWave(rewriter_, op_.getLoc(), computeCycle,
                                hemisphere, "compute", weightBuffer, 0,
                                2 * blockRows, 0, blockIssues, 1,
                                accumulatorBase, 1,
                                finalReduction ? "stream" : "sram",
                                true, "supercell", 0, dataFormat, {},
                                "block8", {}, tokenBlocks,
                                projectionIssueInterval, 0, 1, 1,
                                blockIssues);
                            if (!finalReduction) continue;

                            // Block8 emits eight BF16 lanes directly on west
                            // streams 0..15. The first result wave appears
                            // after the four-row MXM pipeline is filled, so
                            // MEM can consume it without an identity VXM op.
                            const int64_t resultStart = computeCycle
                                + throughput.tile_rows - 1;
                            const auto destinationSlices =
                                kind == AttentionProjectionKind::Value
                                ? layout.valuePackSlices(headBlock)
                                : layout.ropeStagingSlices();
                            const int64_t destinationAddress =
                                kind == AttentionProjectionKind::Value
                                ? layout.valuePackAddress(
                                      head, headBlock, 0, 0)
                                : layout.ropeStagingAddress(kind, head,
                                      headBlock, 0, 0);
                            for (int64_t stream = 0; stream < 16;
                                 ++stream) {
                                const int64_t slice =
                                    destinationSlices[
                                        kind
                                                == AttentionProjectionKind::Value
                                            ? stream
                                            : (stream + 2 * headBlock) % 16];
                                const auto latency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::MxmResult,
                                        target::StreamEndpoint::Mem,
                                        target::StreamDirection::West,
                                        slice);
                                if (!latency) return -1;
                                const int64_t writeCycle =
                                    resultStart + *latency;
                                emitMemWave(rewriter_, op_.getLoc(),
                                    writeCycle,
                                    hemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "write", destinationAddress,
                                    32 + stream,
                                    blockIssues, 1, 1, "sram", -1,
                                    tokenBlocks, projectionIssueInterval,
                                    blockIssues,
                                    kind == AttentionProjectionKind::Value
                                        ? valueBank : ropeStagingBank);
                                projectionEnd = std::max(projectionEnd,
                                    writeCycle
                                        + (tokenBlocks - 1)
                                            * projectionIssueInterval
                                        + blockIssues);
                            }
                        }
                }
            }
            const int64_t nextProjectionStart = projectionEnd + 1;
            phaseStart = nextProjectionStart;

            if (kind == AttentionProjectionKind::Value) {
                // A 128-wide head is produced as four 32-column blocks, two
                // per hemisphere. PV owns one hemisphere per KV head, so move
                // the remote blocks into that home hemisphere before they are
                // transposed into MXM weights. Unconsumed stream registers
                // cross the passive VXM bridge, so this transfer does not
                // occupy a VXM control queue.
                int64_t copyCycle = phaseStart + maxRopeReadLatency + 1;
                int64_t copyEnd = copyCycle;
                const int64_t firstGroupBlock = outputGroup * 4;
                const int64_t lastGroupBlock = std::min<int64_t>(
                    projectionOutputBlocks, firstGroupBlock + 4);
                for (int64_t outputBlock = firstGroupBlock;
                     outputBlock < lastGroupBlock; ++outputBlock) {
                    const int64_t head = outputBlock / headBlocks;
                    const int64_t headBlock = outputBlock % headBlocks;
                    const int64_t sourceHemisphere =
                        (outputBlock % 4) / 2;
                    const int64_t destinationHemisphere =
                        head % memory.hemispheres;
                    if (sourceHemisphere == destinationHemisphere)
                        continue;
                    const auto slices =
                        layout.valuePackSlices(headBlock);
                    for (int64_t tokenBlock = 0;
                         tokenBlock < tokenBlocks; ++tokenBlock) {
                        for (int64_t beat = 0;
                             beat < blockIssues; ++beat, ++copyCycle) {
                            const int64_t address =
                                layout.valuePackAddress(head, headBlock,
                                    tokenBlock, beat);
                            for (int64_t stream = 0; stream < 16;
                                 ++stream) {
                                const int64_t slice = slices[stream];
                                const auto latency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::Mem,
                                        target::StreamEndpoint::VxmInput,
                                        target::StreamDirection::West,
                                        slice);
                                if (!latency) return -1;
                                emitMem(rewriter_, op_.getLoc(),
                                    copyCycle - *latency,
                                    sourceHemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "read", address, 32 + stream,
                                    1, 1, 0, "sram", -1, valueBank);
                                const auto writeLatency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::VxmResult,
                                        target::StreamEndpoint::Mem,
                                        target::StreamDirection::East,
                                        slice);
                                if (!writeLatency) return -1;
                                emitMem(rewriter_, op_.getLoc(),
                                    copyCycle + *writeLatency,
                                    destinationHemisphere
                                            * memory.slices_per_hemisphere
                                    + slice,
                                    "write", address, stream, 1, 1, 0,
                                    "sram", -1, valueBank);
                                copyEnd = std::max(copyEnd,
                                    copyCycle + *writeLatency + 1);
                            }
                        }
                    }
                }
                phaseStart = std::max(copyCycle + 1, copyEnd);
                continue;
            }

            // One compact VXM packet drives both the C0..C7 chains and their
            // mirrored C8..C15 chains. The two halves have fixed East/West
            // input-group sources, so make the projection staging resident in
            // both hemispheres before issuing RoPE. This is the conservative
            // form of paired-head execution: both halves evaluate the same
            // head and the desired result is available on either side.
            int64_t replicateCycle = phaseStart + tile
                + maxRopeReadLatency + 1;
            int64_t replicateEnd = phaseStart;
            const int64_t firstReplicatedBlock = outputGroup * 4;
            const int64_t lastReplicatedBlock = std::min<int64_t>(
                projectionOutputBlocks, firstReplicatedBlock + 4);
            for (int64_t outputBlock = firstReplicatedBlock;
                 outputBlock < lastReplicatedBlock; ++outputBlock) {
                const int64_t head = outputBlock / headBlocks;
                const int64_t headBlock = outputBlock % headBlocks;
                const int64_t sourceHemisphere =
                    (outputBlock % 4) / 2;
                const int64_t destinationHemisphere =
                    1 - sourceHemisphere;
                const auto slices = layout.ropeStagingSlices();
                for (int64_t tokenBlock = 0;
                     tokenBlock < tokenBlocks; ++tokenBlock) {
                    for (int64_t beat = 0;
                         beat < blockIssues; ++beat, ++replicateCycle) {
                        const int64_t address =
                            layout.ropeStagingAddress(kind, head,
                                headBlock, tokenBlock, beat);
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice = slices[
                                (stream + 2 * headBlock) % 16];
                            const auto readLatency =
                                target_.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::VxmInput,
                                    target::StreamDirection::West,
                                    slice);
                            const auto writeLatency =
                                target_.transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East,
                                    slice);
                            if (!readLatency || !writeLatency) return -1;
                            emitMem(rewriter_, op_.getLoc(),
                                replicateCycle - *readLatency,
                                sourceHemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read", address, 32 + stream,
                                1, 1, 0, "sram", -1,
                                ropeStagingBank);
                            const int64_t writeCycle =
                                replicateCycle + *writeLatency;
                            emitMem(rewriter_, op_.getLoc(), writeCycle,
                                destinationHemisphere
                                        * memory.slices_per_hemisphere
                                + slice,
                                "write", address, stream, 1, 1, 0,
                                "sram", -1, ropeStagingBank);
                            replicateEnd = std::max(
                                replicateEnd, writeCycle + 1);
                        }
                    }
                }
                // The next block may reverse source/destination hemispheres.
                // Wait until all writes from this passive copy have retired
                // before those destination MEM queues become readers.
                replicateCycle = std::max(replicateCycle,
                    replicateEnd + maxRopeReadLatency + 1);
            }
            phaseStart = replicateEnd + tile
                + target_.streams().system_register_columns;
            // The final Block8 result is still travelling west while its MEM
            // write is issued. Delay the first RoPE MEM read so an east-to-
            // west passive link never reuses the same stream-register cell.
            const int64_t ropeStart = std::max(
                phaseStart + maxRopeReadLatency + 1, ropeTail);
            int64_t ropeEnd = ropeStart;
            const int64_t firstGroupBlock = outputGroup * 4;
            const int64_t lastGroupBlock = std::min<int64_t>(
                projectionOutputBlocks, firstGroupBlock + 4);
            const int64_t firstHead = firstGroupBlock / headBlocks;
            const int64_t lastHead =
                (lastGroupBlock - 1) / headBlocks;
            for (int64_t head = firstHead; head <= lastHead; ++head) {
                const int64_t outputHemisphere =
                    kind == AttentionProjectionKind::Query
                    ? (head / queryHeadsPerKv) % memory.hemispheres
                    : head % memory.hemispheres;
                int64_t headEnd = ropeEnd;
                for (int64_t pairBlock = 0;
                     pairBlock < headBlocks / 2; ++pairBlock) {
                    const int64_t lowBlock = pairBlock;
                    const int64_t highBlock =
                        pairBlock + headBlocks / 2;
                    const int64_t lowOutputBlock =
                        head * headBlocks + lowBlock;
                    const int64_t highOutputBlock =
                        head * headBlocks + highBlock;
                    const int64_t inputHemisphere =
                        (highOutputBlock % 4) / 2;
                    const int64_t blocks[] = {lowBlock, highBlock};
                    const int64_t outputBlocks[] = {
                        lowOutputBlock, highOutputBlock};
                    const auto sourceSlices = layout.ropeStagingSlices();
                    const auto productSlices =
                        target_.mxm_distributed_activation_slices();
                    if (productSlices.size() != 16
                        || memory.banks_per_slice < 2) {
                        op_.emitError(
                            "compact RoPE requires 16 product slices and a second SRAM bank");
                        return -1;
                    }
                    const int64_t productBank = ropeProductBank;
                    int64_t maxProductWriteLatency = 0;
                    int64_t maxProductReadLatency = 0;
                    for (int64_t slice : productSlices) {
                        const auto writeLatency = target_.transport_latency(
                            target::StreamEndpoint::VxmResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::East, slice);
                        const auto readLatency = target_.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::VxmInput,
                            target::StreamDirection::West, slice);
                        if (!writeLatency || !readLatency) {
                            op_.emitError("RoPE product scratch slice has no VXM route: ")
                                << slice;
                            return -1;
                        }
                        maxProductWriteLatency = std::max(
                            maxProductWriteLatency, *writeLatency);
                        maxProductReadLatency = std::max(
                            maxProductReadLatency, *readLatency);
                    }
                    const auto emitSource = [&](int64_t token,
                                                int64_t half,
                                                int64_t stream,
                                                int64_t inputCycle) {
                        const int64_t tokenBlock = token / tile;
                        const int64_t rowBlock =
                            (token % tile) / blockRows;
                        const int64_t tokenLane = token % blockRows;
                        const int64_t sourceBlock = blocks[half];
                        const int64_t sourceAddress =
                            layout.ropeStagingAddress(kind, head,
                                sourceBlock, tokenBlock, rowBlock);
                        for (int64_t hemisphere = 0;
                             hemisphere < memory.hemispheres;
                             ++hemisphere) {
                            const int64_t inputStream = stream
                                + hemisphere * 16;
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice = sourceSlices[
                                    (2 * tokenLane + byte
                                        + 2 * sourceBlock) % 16];
                                const auto latency = target_.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::VxmInput,
                                    target::StreamDirection::West, slice);
                                if (!latency) {
                                    op_.emitError("RoPE source slice has no VXM input route: ")
                                        << slice;
                                    return false;
                                }
                                emitMem(rewriter_, op_.getLoc(),
                                    inputCycle - *latency,
                                    hemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "read", sourceAddress,
                                    inputStream + byte, 1, 1, 0,
                                    "sram", -1, ropeStagingBank);
                            }
                        }
                        return true;
                    };
                    const auto emitRopeTable = [&](int64_t token,
                                                   int64_t inputCycle) {
                        for (int64_t hemisphere = 0;
                             hemisphere < memory.hemispheres;
                             ++hemisphere) {
                            for (int64_t byte = 0; byte < 4; ++byte) {
                                const int64_t slice = layout.ropeSlices()[byte];
                                const auto latency = target_.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::VxmInput,
                                    target::StreamDirection::West, slice);
                                if (!latency) {
                                    op_.emitError("RoPE table slice has no VXM input route: ")
                                        << slice;
                                    return false;
                                }
                                const int64_t stream = (byte < 2
                                    ? 34 + byte : 36 + byte)
                                    + hemisphere * 16;
                                emitMem(rewriter_, op_.getLoc(),
                                    inputCycle - *latency,
                                    hemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "read", layout.ropeAddress(token, pairBlock),
                                    stream, 1, 1, 0,
                                    "sram", -1, ropeBank);
                            }
                        }
                        return true;
                    };
                    const auto emitProductWrites = [&](int64_t token,
                                                       int64_t firstProduct,
                                                       int64_t outputCycle) {
                        const int64_t laneOffset = (token % 2) * 8;
                        for (int64_t slot = 0; slot < 2; ++slot) {
                            const int64_t product = firstProduct + slot;
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice = productSlices[
                                    laneOffset + product * 2 + byte];
                                const auto latency = target_.transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East, slice);
                                if (!latency) {
                                    op_.emitError("RoPE product slice has no VXM output route: ")
                                        << slice;
                                    return false;
                                }
                                for (int64_t destination = 0;
                                     destination < memory.hemispheres;
                                     ++destination) {
                                    const int64_t source = 1 - destination;
                                    emitMem(rewriter_, op_.getLoc(),
                                        outputCycle + *latency,
                                        destination
                                                * memory.slices_per_hemisphere
                                            + slice,
                                        "write",
                                        layout.ropeProductAddress(kind, head,
                                            pairBlock, product, token),
                                        source * 8 + slot * 2 + byte,
                                        1, 1, 0, "sram", -1, productBank);
                                }
                            }
                        }
                        return true;
                    };

                    const int64_t productAConfig = headEnd;
                    const int64_t productAInput = productAConfig + 1;
                    emitRopeProducts(productAConfig, inputHemisphere,
                        outputHemisphere, projectionValues[projection]);
                    for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
                        const int64_t inputCycle = productAInput + token;
                        if (!emitSource(token, 0, 32, inputCycle)
                            || !emitSource(token, 1, 36, inputCycle)
                            || !emitRopeTable(token, inputCycle)
                            || !emitProductWrites(token, 0,
                                inputCycle + 2))
                            return -1;
                    }

                    const int64_t productBConfig = productAInput
                        + op_.getSeqLen() + 1 + maxProductWriteLatency
                        + maxRopeReadLatency + 1;
                    const int64_t productBInput = productBConfig + 1;
                    emitRopeProducts(productBConfig, inputHemisphere,
                        outputHemisphere, projectionValues[projection]);
                    for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
                        const int64_t inputCycle = productBInput + token;
                        if (!emitSource(token, 1, 32, inputCycle)
                            || !emitSource(token, 0, 36, inputCycle)
                            || !emitRopeTable(token, inputCycle)
                            || !emitProductWrites(token, 2,
                                inputCycle + 2))
                            return -1;
                    }

                    const int64_t combineConfig = productBInput
                        + op_.getSeqLen() + 1 + maxProductWriteLatency
                        + maxProductReadLatency + 1;
                    const int64_t combineInput = combineConfig + 1;
                    emitRopeCombine(combineConfig, inputHemisphere,
                        outputHemisphere, projectionValues[projection]);
                    for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
                        const int64_t tokenBlock = token / tile;
                        const int64_t rowBlock =
                            (token % tile) / blockRows;
                        const int64_t tokenLane = token % blockRows;
                        const int64_t inputCycle = combineInput + token;
                        const int64_t laneOffset = (token % 2) * 8;
                        for (int64_t product = 0; product < 4; ++product) {
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice = productSlices[
                                    laneOffset + product * 2 + byte];
                                const auto latency = target_.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::VxmInput,
                                    target::StreamDirection::West, slice);
                                if (!latency) return -1;
                                for (int64_t hemisphere = 0;
                                     hemisphere < memory.hemispheres;
                                     ++hemisphere)
                                    emitMem(rewriter_, op_.getLoc(),
                                        inputCycle - *latency,
                                        hemisphere
                                                * memory.slices_per_hemisphere
                                            + slice,
                                        "read", layout.ropeProductAddress(kind,
                                            head, pairBlock, product, token),
                                        32 + hemisphere * 16
                                            + product * 2 + byte,
                                        1, 1, 0, "sram", -1, productBank);
                            }
                        }
                        const int64_t outputCycle = inputCycle + 1;
                        for (int64_t half = 0; half < 2; ++half) {
                            const int64_t reductionBlock = blocks[half];
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice = kind
                                        == AttentionProjectionKind::Query
                                    ? layout.queryIwSlices(reductionBlock)[
                                          2 * tokenLane + byte]
                                    : layout.keySlices(reductionBlock)[byte];
                                const auto latency = target_.transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East, slice);
                                if (!latency) return -1;
                                for (int64_t destination = 0;
                                     destination < memory.hemispheres;
                                     ++destination) {
                                    const int64_t source = 1 - destination;
                                    emitMem(rewriter_, op_.getLoc(),
                                        outputCycle + *latency,
                                        destination
                                                * memory.slices_per_hemisphere
                                            + slice,
                                        "write",
                                        kind == AttentionProjectionKind::Query
                                            ? layout.queryIwAddress(head,
                                                  reductionBlock, tokenBlock,
                                                  rowBlock)
                                            : layout.keyAddress(head,
                                                  reductionBlock, tokenBlock)
                                                + token % tile,
                                        source * 8 + half * 2 + byte,
                                        1, 1, 0, "sram", -1,
                                        kind == AttentionProjectionKind::Query
                                            ? queryBank : keyBank);
                                    headEnd = std::max(headEnd,
                                        outputCycle + *latency + 1);
                                }
                            }
                        }
                    }
                    headEnd = std::max(headEnd,
                        combineInput + op_.getSeqLen() + 1);
                    // headEnd is consumed as the next work item's VXM
                    // configuration cycle. Reserve its backwards-scheduled
                    // MEM read lead as part of this work item's tail.
                    headEnd += maxRopeReadLatency + 1;
                }
                ropeEnd = std::max(ropeEnd, headEnd);
            }
            ropeTail = ropeEnd + 1;
            phaseStart = std::max(nextProjectionStart,
                ropeTail + target_.streams().system_register_columns);
        }
    }

    // Projection producers run at MXM rate while RoPE drains the staging
    // FIFO on VXM. Only the final consumer of this phase waits for both.
    phaseStart = std::max(phaseStart, ropeTail);

    return phaseStart + 16
        + target_.streams().system_register_columns;
}

} // namespace ftlpu::compiler::schedule
