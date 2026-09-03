#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include <algorithm>

namespace ftlpu::compiler::schedule::ffn_detail {

namespace {

struct PendingSwishOutput {
    int64_t input_cycle;
    int64_t m_tile;
    int64_t pair;
    int64_t row;
    int64_t source_hemisphere;
};

mlir::LogicalResult emitHiddenMirrorCopies(FfnEmissionContext& context,
    llvm::ArrayRef<PendingSwishOutput> pendingOutputs,
    int64_t lastInputCycle, FfnSwishEmission& result)
{
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    if (pendingOutputs.empty() || !result.hidden || lastInputCycle < 0)
        return mlir::failure();

    // Down projection consumes every hidden block in both hemispheres. Mirror
    // each owner copy through the passive stream path after the Swish stream
    // has drained; this avoids coupling correctness to a duplicated VXM
    // output and also works when temporary and hidden slices share a bank.
    const int64_t hiddenBase = get_base_row(ffn.getHidden0Placement());
    const int64_t hiddenBlocks = ffn.getHidden() / tile;
    const int64_t hiddenBank = ffn.getHidden0Placement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const auto hiddenKind = ffn.getHidden0Placement()
        .getAs<mlir::StringAttr>("kind");
    const bool hiddenDistributed16 = hiddenKind
        && hiddenKind.getValue() == "fp16_mxm_distributed_16";
    int64_t maxOutputLatency = 0;
    int64_t maxReadLatency = 0;
    for (int64_t slice : context.hidden_slices) {
        const auto outputLatency = target.transport_latency(
            target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice);
        const auto readLatency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice);
        if (!outputLatency || !readLatency) {
            ffn.getOperation()->emitError()
                << "vector FFN hidden slice " << slice
                << " is not routable through VXM (output="
                << static_cast<bool>(outputLatency)
                << ", read=" << static_cast<bool>(readLatency) << ")";
            return mlir::failure();
        }
        maxOutputLatency = std::max(maxOutputLatency, *outputLatency);
        maxReadLatency = std::max(maxReadLatency, *readLatency);
    }

    constexpr int64_t kVxmSwishLatency = 17;
    int64_t bridgeInputCycle = lastInputCycle + kVxmSwishLatency
        + maxOutputLatency + maxReadLatency + 2;
    int64_t lastCopyCycle = bridgeInputCycle;
    const mlir::Value bridgeSource = result.hidden;
    mlir::Value lastBridgeWrite = result.hidden;
    for (int64_t sourceHemisphere = 0;
         sourceHemisphere < memory.hemispheres; ++sourceHemisphere) {
        if (sourceHemisphere != 0)
            bridgeInputCycle += maxReadLatency + maxOutputLatency + 1;
        for (const PendingSwishOutput& pending : pendingOutputs) {
            if (pending.source_hemisphere != sourceHemisphere) continue;
            const int64_t token = pending.m_tile * tile + pending.row;
            const int64_t nblock = (pending.pair / 2) * 4
                + pending.source_hemisphere * 2 + pending.pair % 2;
            const int64_t owner = 1 - pending.source_hemisphere;
            const int64_t peer = pending.source_hemisphere;
            for (int64_t byte = 0; byte < 2; ++byte) {
                int64_t slice = context.hidden_slices[
                    2 * (nblock % 2) + byte];
                int64_t address = hiddenBase
                    + (nblock / 2) * context.m() + token;
                if (hiddenDistributed16) {
                    const int64_t tokenWithinBlock = token % tile;
                    const int64_t tokenWave = tokenWithinBlock
                        / throughput.mxm_block_rows;
                    const int64_t tokenLane = tokenWithinBlock
                        % throughput.mxm_block_rows;
                    slice = context.hidden_slices[2 * tokenLane + byte];
                    address = hiddenBase
                        + ((token / tile) * hiddenBlocks + nblock)
                            * throughput.tile_rows
                        + tokenWave;
                }
                const auto readLatency = target.transport_latency(
                    target::StreamEndpoint::Mem,
                    target::StreamEndpoint::VxmInput,
                    target::StreamDirection::West, slice);
                const auto writeLatency = target.transport_latency(
                    target::StreamEndpoint::VxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::East, slice);
                if (!readLatency || !writeLatency) {
                    ffn.getOperation()->emitError()
                        << "vector FFN hidden multicast route is unavailable "
                        << "for slice " << slice << " (read="
                        << static_cast<bool>(readLatency)
                        << ", write=" << static_cast<bool>(writeLatency)
                        << ")";
                    return mlir::failure();
                }
                auto read = context.emitSliceRead(bridgeSource,
                    context.hidden_route,
                    bridgeInputCycle - *readLatency,
                    slice, address, 1, 1, byte, "west", "vxm_bypass",
                    context.hemisphereName(owner));
                auto placement = schedule_placement(context.rewriter,
                    {slice}, address, 1, 1,
                    context.hemisphereName(peer),
                    hiddenDistributed16
                        ? "fp16_mxm_distributed_16"
                        : "fp16_mxm_activation_planar",
                    hiddenBank);
                auto write = context.rewriter.create<MemWriteOp>(
                    ffn.getLoc(), read.getOutput(),
                    bridgeInputCycle + *writeLatency,
                    1, byte, 1, 0,
                    context.rewriter.getStringAttr("east"),
                    ffn.getHidden0Address(), placement, tile);
                lastBridgeWrite = write.getOutput();
                lastCopyCycle = std::max(lastCopyCycle,
                    bridgeInputCycle + *writeLatency);
            }
            ++bridgeInputCycle;
        }
    }
    result.hidden = lastBridgeWrite;
    result.last_cycle = std::max(result.last_cycle, lastCopyCycle);
    return mlir::success();
}

} // namespace

mlir::FailureOr<FfnSwishEmission> emitFfnSwish(
    FfnEmissionContext& context, FfnProjectionEmission emission)
{
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t mTileCount =
        context.projection_timeline.m_tile_count;
    const int64_t pairCount =
        context.projection_timeline.pair_count;
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const auto gateTempSlices = target.ffn_gate_temp_slices();
    const auto upTempSlices = target.ffn_up_temp_slices();
    if (tile <= 0 || mTileCount <= 0 || pairCount <= 0
        || throughput.mxm_block_rows <= 0 || memory.hemispheres <= 0
        || memory.hemispheres > 2
        || memory.sram_depth_rows < mTileCount * tile) {
        ffn.getOperation()->emitError(
            "invalid target geometry for FFN Swish scheduling");
        return mlir::failure();
    }
    const int64_t pairsPerTempGroup =
        memory.sram_depth_rows / (mTileCount * tile);
    const int64_t requiredTempSlices =
        2 * ((pairCount + pairsPerTempGroup - 1) / pairsPerTempGroup);
    if (pairsPerTempGroup <= 0
        || static_cast<int64_t>(gateTempSlices.size()) < requiredTempSlices
        || static_cast<int64_t>(upTempSlices.size()) < requiredTempSlices) {
        ffn.getOperation()->emitError(
            "FFN Swish temporary storage does not cover every projection pair");
        return mlir::failure();
    }

    FfnSwishEmission result;
    result.last_cycle = 0;
    // Queue 7 is the tail of the physical 8-stage VXM chain. Its output
    // register is permanently bound to stream group 3 (byte streams 6/7).
    constexpr int64_t outputStream = 6;
    if (context.strategy == FfnScheduleStrategy::Tail) {
        int64_t inputCycle =
            context.projection_timeline.final_projection_cycle
            + context.projection_timeline.accumulator_queue_release;
        for (const CompletedProjectionTile& completed :
            emission.completed_tiles)
            inputCycle = std::max(
                inputCycle, completed.deferred_ready_cycle);

        llvm::SmallVector<PendingSwishOutput> pendingOutputs;
        mlir::Value gateValue;
        mlir::Value upValue;
        const int64_t firstInputCycle = inputCycle;
        for (int64_t mTile = 0; mTile < mTileCount; ++mTile) {
            for (int64_t pair = 0; pair < pairCount; ++pair) {
                for (int64_t row = 0; row < tile; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres;
                         ++hemisphere) {
                        const auto completed = llvm::find_if(
                            emission.completed_tiles,
                            [&](const CompletedProjectionTile& tile) {
                                return tile.m_tile == mTile
                                    && tile.pair == pair
                                    && tile.hemisphere == hemisphere;
                            });
                        if (completed
                            == emission.completed_tiles.end()) {
                            ffn.getOperation()->emitError(
                                "missing completed FFN projection tile for "
                                "vector Swish input");
                            return mlir::failure();
                        }
                        const int64_t tempGroup = pair / pairsPerTempGroup;
                        const int64_t tempBase =
                            ((pair % pairsPerTempGroup) * mTileCount + mTile)
                            * tile;
                        const int64_t streamBase = hemisphere * 16;
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t gateSlice =
                                gateTempSlices[2 * tempGroup + byte];
                            const int64_t upSlice =
                                upTempSlices[2 * tempGroup + byte];
                            gateValue =
                                context
                                    .emitSliceRead(completed->gate_temp,
                                        context.activation_route,
                                        inputCycle
                                            - context.westLatency(
                                                gateSlice),
                                        gateSlice, tempBase + row, 1, 1,
                                            streamBase + byte, "west",
                                            "vxm_bf16",
                                            context.hemisphereName(
                                                hemisphere))
                                    .getOutput();
                            upValue =
                                context
                                    .emitSliceRead(completed->up_temp,
                                        context.activation_route,
                                        inputCycle
                                            - context.westLatency(
                                                upSlice),
                                        upSlice, tempBase + row, 1, 1,
                                        streamBase + 2 + byte,
                                        "west", "vxm_bf16",
                                        context.hemisphereName(
                                            hemisphere))
                                    .getOutput();
                        }
                        pendingOutputs.push_back({inputCycle, mTile,
                            pair, row, hemisphere});
                    }
                    ++inputCycle;
                }
            }
        }
        const int64_t repeatCount = inputCycle - firstInputCycle;
        if (!gateValue || !upValue || repeatCount <= 0) {
            ffn.getOperation()->emitError(
                "vector FFN projection did not produce Swish temporaries");
            return mlir::failure();
        }
        auto [output, mirroredOutput] = emitFfnSwishAlu(
            context.rewriter, ffn.getLoc(), ffn.getResult().getType(),
            gateValue, upValue, target, context.strategy,
            firstInputCycle - 1, 0, outputStream, repeatCount, 1);
        (void)mirroredOutput;
        for (const PendingSwishOutput& pending : pendingOutputs) {
            result.hidden = emitFfnSwishResultRow(context.rewriter, ffn,
                target, context.hidden_slices, output.getResult(),
                pending.input_cycle, pending.m_tile, pending.pair,
                pending.row, pending.source_hemisphere);
        }

        if (mlir::failed(emitHiddenMirrorCopies(context, pendingOutputs,
                inputCycle - 1, result)))
            return mlir::failure();
        return result;
    }

    FfnSwishScheduleRequest request;
    request.tile_rows = tile;
    const int64_t dequantWindowCycles =
        memory.hemispheres * throughput.mxms_per_hemisphere
            * weightLoadCycles
        + 1;
    if (!context.local_weight_dequant) {
        for (const FfnProjectionBlockSchedule& block :
            context.projection_timeline.blocks) {
            request.dequant_windows.push_back(
                {block.dequant_start,
                    block.dequant_start + dequantWindowCycles});
        }
    }
    for (int64_t hemisphere = 0;
         hemisphere < memory.hemispheres; ++hemisphere) {
        request.temp_mem_windows[hemisphere] =
            std::move(emission.temp_mem_busy_windows[hemisphere]);
    }

    llvm::SmallVector<const CompletedProjectionTile*> deferred;
    for (const CompletedProjectionTile& completed :
        emission.completed_tiles)
        deferred.push_back(&completed);
    llvm::sort(deferred,
        [](const CompletedProjectionTile* lhs,
            const CompletedProjectionTile* rhs) {
            return lhs->compute_cycle < rhs->compute_cycle;
        });
    for (const CompletedProjectionTile* completed : deferred)
        request.tasks.push_back(
            {completed->deferred_ready_cycle, completed->hemisphere});

    auto cycles = planFfnSwishCycles(request, target);
    if (mlir::failed(cycles)) return mlir::failure();
    llvm::SmallVector<PendingSwishOutput> pendingOutputs;
    for (std::size_t index = 0; index < deferred.size(); ++index) {
        const CompletedProjectionTile& completed = *deferred[index];
        const int64_t start = (*cycles)[index];
        const int64_t tempGroup = completed.pair / pairsPerTempGroup;
        llvm::SmallVector<PendingSwishOutput> tileOutputs;
        mlir::Value gateValue;
        mlir::Value upValue;
        for (int64_t row = 0; row < tile; ++row) {
            const int64_t cycle = start + row;
            const int64_t tempBase =
                ((completed.pair % pairsPerTempGroup) * mTileCount
                    + completed.m_tile)
                * tile;
            const int64_t tempStreamBase =
                8 + completed.hemisphere * 8;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t gateSlice =
                    gateTempSlices[2 * tempGroup + byte];
                const int64_t upSlice =
                    upTempSlices[2 * tempGroup + byte];
                gateValue =
                    context
                        .emitSliceRead(completed.gate_temp,
                            context.activation_route,
                            cycle - context.westLatency(gateSlice),
                            gateSlice, tempBase + row, 1, 1,
                            tempStreamBase + byte, "west",
                            "vxm_bf16",
                            context.hemisphereName(
                                completed.hemisphere))
                        .getOutput();
                upValue =
                    context
                        .emitSliceRead(completed.up_temp,
                            context.activation_route,
                            cycle - context.westLatency(upSlice),
                            upSlice, tempBase + row, 1, 1,
                            tempStreamBase + 2 + byte,
                            "west", "vxm_bf16",
                            context.hemisphereName(
                                completed.hemisphere))
                        .getOutput();
            }
            result.last_cycle = std::max(result.last_cycle, cycle);
            tileOutputs.push_back({cycle, completed.m_tile,
                completed.pair, row, completed.hemisphere});
        }
        if (!gateValue || !upValue) {
            ffn.getOperation()->emitError(
                "fused FFN tile did not produce Swish temporaries");
            return mlir::failure();
        }
        auto [output, mirroredOutput] = emitFfnSwishAlu(
            context.rewriter, ffn.getLoc(), ffn.getResult().getType(),
            gateValue, upValue, target, context.strategy,
            start - 1, completed.hemisphere, outputStream, tile, 1);
        (void)mirroredOutput;
        for (const PendingSwishOutput& pending : tileOutputs) {
            result.hidden = emitFfnSwishResultRow(context.rewriter, ffn,
                target, context.hidden_slices, output.getResult(),
                pending.input_cycle, pending.m_tile, pending.pair,
                pending.row, pending.source_hemisphere, true);
            pendingOutputs.push_back(pending);
        }
    }
    if (mlir::failed(emitHiddenMirrorCopies(context, pendingOutputs,
            result.last_cycle, result)))
        return mlir::failure();
    return result;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
