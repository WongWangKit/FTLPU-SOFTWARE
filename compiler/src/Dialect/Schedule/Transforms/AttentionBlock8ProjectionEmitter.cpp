#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

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

    const auto emitRope = [&](int64_t cycle, int64_t inputHemisphere,
                              int64_t outputHemisphere,
                              mlir::Value value) {
        const int64_t alu = inputHemisphere * 8;
        const char* input = inputHemisphere == 0 ? "east" : "west";
        const char* output = outputHemisphere == 0 ? "east" : "west";
        emitVxm(rewriter_, op_.getLoc(), value, cycle, alu,
            "multiply", streamKind, 32, 0.0f, streamKind, 40,
            0.0f, "fp32", -1, input, output);
        emitVxm(rewriter_, op_.getLoc(), value, cycle, alu + 1,
            "multiply", streamKind, 34, 0.0f, streamKind, 42,
            0.0f, "fp32", -1, input, output);
        emitVxm(rewriter_, op_.getLoc(), value, cycle, alu + 3,
            "multiply", streamKind, 34, 0.0f, streamKind, 40,
            0.0f, "fp32", -1, input, output);
        emitVxm(rewriter_, op_.getLoc(), value, cycle, alu + 4,
            "multiply", streamKind, 32, 0.0f, streamKind, 42,
            0.0f, "fp32", -1, input, output);
        emitVxm(rewriter_, op_.getLoc(), value, cycle + 1, alu + 2,
            "subtract", "alu", alu, 0.0f, "alu", alu + 1,
            0.0f, dataFormat, 20, input, output);
        emitVxm(rewriter_, op_.getLoc(), value, cycle + 1, alu + 5,
            "add", "alu", alu + 3, 0.0f, "alu", alu + 4,
            0.0f, dataFormat, 22, input, output);
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
                                throughput.tile_rows, 1, -1);
                        }
                        emitMxmDequantWave(rewriter_, op_.getLoc(),
                            loadStart, hemisphere,
                            projectionScales[projection], 1, 1,
                            throughput.tile_rows, 1);
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
                                    hiddenBlocks * blockIssues);
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
                                    blockIssues);
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
                // transposed into MXM weights. The VXM pass is byte-exact and
                // only provides the bidirectional fabric hop.
                int64_t copyCycle = phaseStart + maxRopeReadLatency + 1;
                const auto hemisphereName = [](int64_t hemisphere) {
                    return hemisphere == 0 ? "east" : "west";
                };
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
                                    1, 1, 0);
                                emitVxm(rewriter_, op_.getLoc(),
                                    projectionValues[projection],
                                    copyCycle, stream, "pass",
                                    "stream_i8", 32 + stream, 0.0f,
                                    "immediate", 0, 0.0f, "i8",
                                    stream,
                                    hemisphereName(sourceHemisphere),
                                    hemisphereName(destinationHemisphere));
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
                                    "write", address, stream, 1, 1, 0);
                            }
                        }
                    }
                }
                phaseStart = copyCycle + 1;
                continue;
            }
            // The final Block8 result is still travelling west while its MEM
            // write is issued. Delay the first RoPE MEM read so an east-to-
            // west passive link never reuses the same stream-register cell.
            const int64_t ropeStart = std::max(
                phaseStart + maxRopeReadLatency + 1, ropeTail);
            int64_t ropeEnd = ropeStart;
            llvm::SmallVector<int64_t, 2> destinationTails(
                memory.hemispheres, ropeStart);
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
                int64_t headEnd = destinationTails[outputHemisphere];
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
                    int64_t ropeCycle = headEnd;
                    for (int64_t token = 0; token < op_.getSeqLen();
                         ++token, ++ropeCycle) {
                        const int64_t tokenBlock = token / tile;
                        const int64_t rowBlock =
                            (token % tile) / blockRows;
                        const int64_t tokenLane = token % blockRows;
                        const int64_t blocks[] = {lowBlock, highBlock};
                        const int64_t outputBlocks[] = {
                            lowOutputBlock, highOutputBlock};
                        for (int64_t half = 0; half < 2; ++half) {
                            const int64_t sourceBlock = blocks[half];
                            const int64_t sourceHemisphere =
                                (outputBlocks[half] % 4) / 2;
                            const auto sourceSlices =
                                layout.ropeStagingSlices();
                            const int64_t sourceAddress =
                                layout.ropeStagingAddress(kind, head,
                                    sourceBlock, tokenBlock, rowBlock);
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice = sourceSlices[
                                    (2 * tokenLane + byte
                                        + 2 * sourceBlock) % 16];
                                const auto latency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::Mem,
                                        target::StreamEndpoint::VxmInput,
                                        target::StreamDirection::West,
                                        slice);
                                if (!latency) return -1;
                                emitMem(rewriter_, op_.getLoc(),
                                    ropeCycle - *latency,
                                    sourceHemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "read", sourceAddress,
                                    32 + half * 2 + byte, 1, 1, 0);
                            }
                        }
                        for (int64_t byte = 0; byte < 4; ++byte) {
                            const int64_t slice = layout.ropeSlices()[byte];
                            const auto latency = target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::VxmInput,
                                target::StreamDirection::West, slice);
                            if (!latency) return -1;
                            emitMem(rewriter_, op_.getLoc(),
                                ropeCycle - *latency,
                                inputHemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read",
                                layout.ropeAddress(token, pairBlock),
                                40 + byte, 1, 1, 0);
                        }
                        emitRope(ropeCycle, inputHemisphere,
                            outputHemisphere,
                            projectionValues[projection]);
                        const int64_t writeBase = ropeCycle + 1;
                        for (int64_t half = 0; half < 2; ++half) {
                            const int64_t reductionBlock = blocks[half];
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice = kind
                                        == AttentionProjectionKind::Query
                                    ? layout.queryIwSlices(
                                          reductionBlock)[
                                          2 * tokenLane + byte]
                                    : layout.keySlices(reductionBlock)[byte];
                                const auto latency =
                                    target_.transport_latency(
                                        target::StreamEndpoint::VxmResult,
                                        target::StreamEndpoint::Mem,
                                        target::StreamDirection::East,
                                        slice);
                                if (!latency) return -1;
                                emitMem(rewriter_, op_.getLoc(),
                                    writeBase + *latency,
                                    outputHemisphere
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
                                    20 + half * 2 + byte, 1, 1, 0);
                                headEnd = std::max(headEnd,
                                    writeBase + *latency + 1);
                            }
                        }
                    }
                    headEnd = std::max(headEnd, ropeCycle);
                }
                destinationTails[outputHemisphere] = headEnd + 1;
                ropeEnd = std::max(ropeEnd, headEnd);
            }
            ropeTail = ropeEnd + 1;
            phaseStart = nextProjectionStart;
        }
    }

    // Projection producers run at MXM rate while RoPE drains the staging
    // FIFO on VXM. Only the final consumer of this phase waits for both.
    phaseStart = std::max(phaseStart, ropeTail);

    return phaseStart + 16;
}

} // namespace ftlpu::compiler::schedule
