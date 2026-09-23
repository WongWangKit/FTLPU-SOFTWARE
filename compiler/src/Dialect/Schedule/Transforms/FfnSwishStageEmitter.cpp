#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include <algorithm>

namespace ftlpu::compiler::schedule::ffn_detail {

namespace {

mlir::LogicalResult emitHiddenMirrorCopies(FfnEmissionContext& context,
    llvm::ArrayRef<const CompletedProjectionTile*> scheduledTiles,
    int64_t lastInputCycle, FfnSwishEmission& result)
{
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    if (!result.hidden || lastInputCycle < 0
        || (context.strategy != FfnScheduleStrategy::Tail
            && scheduledTiles.empty()))
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
    const bool singleMxmVector =
        throughput.mxms_per_hemisphere == 1;
    const int64_t pairStep = singleMxmVector ? 2 : 1;
    const int64_t pairResidues = std::min<int64_t>(
        pairStep, context.projection_timeline.pair_count);
    const int64_t pairCount = context.projection_timeline.pair_count;
    const int64_t mTileCount = context.projection_timeline.m_tile_count;
    const bool directTailParityDomain =
        context.strategy == FfnScheduleStrategy::Tail
        && hiddenDistributed16 && singleMxmVector
        && mTileCount == 1 && pairResidues == 2
        && pairCount % 2 == 0
        && pairCount <= memory.sram_depth_rows / tile;
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
        // Derive the passive-copy domains directly from the tile shape. Pair
        // parity selects the physical hidden block, while a distributed
        // layout splits only at the physical token-lane boundary.
        const auto emitTileCopy = [&](int64_t mTile, int64_t pair,
                                      FfnLoopDomain3D outerDomain = {},
                                      bool directParityDomain = false) {
            const int64_t tokenBase = mTile * tile;
            const int64_t pairGroup = pair / 2;
            const int64_t pairParity = pair % 2;
            const int64_t nblock = singleMxmVector
                ? pairGroup * 4 + sourceHemisphere * 2 + pairParity
                : pair;
            const int64_t owner = 1 - sourceHemisphere;
            const int64_t peer = sourceHemisphere;
            const auto emitCopy = [&](int64_t slice, int64_t address,
                                      int64_t firstOffset, int64_t byte,
                                      FfnLoopDomain3D domain) {
                const auto readLatency = target.transport_latency(
                    target::StreamEndpoint::Mem,
                    target::StreamEndpoint::VxmInput,
                    target::StreamDirection::West, slice);
                const auto writeLatency = target.transport_latency(
                    target::StreamEndpoint::VxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::East, slice);
                if (!readLatency || !writeLatency) return false;
                const int64_t innerCount = hiddenDistributed16
                    ? 1 : tile;
                auto readPlacement = schedule_placement(context.rewriter,
                    {slice}, address, innerCount, 1,
                    context.hemisphereName(owner), "schedule_slice",
                    hiddenBank);
                mlir::NamedAttrList readAttrs(readPlacement);
                readAttrs.set("binding_placement",
                    ffn.getHidden0Placement());
                auto read = emitFfnMemRead3D(context.rewriter,
                    ffn.getLoc(), bridgeSource,
                    bridgeInputCycle + firstOffset - *readLatency,
                    innerCount, byte, 1,
                    slice
                            / target.streams()
                                  .mem_slices_per_register_group
                        + 1,
                    context.rewriter.getStringAttr("west"),
                    context.rewriter.getStringAttr("vxm_bypass"),
                    context.hidden_route.getAddress(),
                    readAttrs.getDictionary(context.rewriter.getContext()),
                    innerCount * tile, domain);
                auto writePlacement = schedule_placement(context.rewriter,
                    {slice}, address, innerCount, 1,
                    context.hemisphereName(peer),
                    hiddenDistributed16
                        ? "fp16_mxm_distributed_16"
                        : "fp16_mxm_activation_planar",
                    hiddenBank);
                auto write = context.rewriter.create<MemWriteOp>(
                    ffn.getLoc(), read.getOutput(),
                    bridgeInputCycle + firstOffset + *writeLatency,
                    innerCount, byte, 1, 0,
                    context.rewriter.getStringAttr("east"),
                    ffn.getHidden0Address(), writePlacement,
                    innerCount * tile);
                setFfnLoopDomain3D(
                    write.getOperation(), context.rewriter, domain);
                lastBridgeWrite = write.getOutput();
                const int64_t finalOffset = firstOffset
                    + (domain.wave_count - 1) * domain.wave_interval
                    + (domain.group_count - 1) * domain.group_interval
                    + innerCount - 1;
                lastCopyCycle = std::max(lastCopyCycle,
                    bridgeInputCycle + finalOffset + *writeLatency);
                return true;
            };

            if (!hiddenDistributed16) {
                const int64_t token = tokenBase;
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = context.hidden_slices[
                        2 * (nblock % 2) + byte];
                    const int64_t address = hiddenBase
                        + (nblock / 2) * context.m() + token;
                    if (!emitCopy(slice, address, 0, byte, outerDomain))
                        return false;
                }
                bridgeInputCycle += tile * outerDomain.group_count;
                return true;
            }

            const int64_t blockRows = throughput.mxm_block_rows;
            for (int64_t tokenLane = 0; tokenLane < blockRows;
                 ++tokenLane) {
                const int64_t firstOffset =
                    (tokenLane - tokenBase % blockRows + blockRows)
                    % blockRows;
                if (firstOffset >= tile) continue;
                const int64_t occurrenceCount = 1
                    + (tile - 1 - firstOffset) / blockRows;
                const int64_t token = tokenBase + firstOffset;
                const int64_t tokenWave = (token % tile) / blockRows;
                const int64_t address = hiddenBase
                    + ((token / tile) * hiddenBlocks + nblock)
                        * throughput.tile_rows
                    + tokenWave;
                if (directParityDomain) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = context.hidden_slices[
                            2 * tokenLane + byte];
                        const auto readLatency = target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::VxmInput,
                            target::StreamDirection::West, slice);
                        const auto writeLatency = target.transport_latency(
                            target::StreamEndpoint::VxmResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::East, slice);
                        if (!readLatency || !writeLatency)
                            return false;
                        emitFfnMemTransfer3D(context.rewriter, ffn.getLoc(),
                            bridgeInputCycle + firstOffset - *readLatency,
                            owner, slice, "read", address, 32 + byte,
                            occurrenceCount, blockRows, 1, hiddenBank,
                            outerDomain);
                        emitFfnMemTransfer3D(context.rewriter, ffn.getLoc(),
                            bridgeInputCycle + firstOffset + *writeLatency,
                            peer, slice, "write", address, byte,
                            occurrenceCount, blockRows, 1, hiddenBank,
                            outerDomain);
                        lastCopyCycle = std::max(lastCopyCycle,
                            bridgeInputCycle + firstOffset
                                + (occurrenceCount - 1) * blockRows
                                + (outerDomain.wave_count - 1)
                                    * outerDomain.wave_interval
                                + (outerDomain.group_count - 1)
                                    * outerDomain.group_interval
                                + *writeLatency);
                    }
                    lastBridgeWrite = bridgeSource;
                    continue;
                }
                FfnLoopDomain3D domain = outerDomain;
                domain.wave_count = occurrenceCount;
                domain.wave_interval = blockRows;
                domain.wave_address_stride = 1;
                for (int64_t byte = 0; byte < 2; ++byte) {
                    int64_t slice = context.hidden_slices[
                        2 * tokenLane + byte];
                    if (!emitCopy(slice, address, firstOffset, byte, domain))
                        return false;
                }
            }
            bridgeInputCycle += tile * (directParityDomain
                ? pairCount : outerDomain.group_count);
            return true;
        };

        if (context.strategy == FfnScheduleStrategy::Tail) {
            // Tail Swish has drained every projection before mirror copy, so
            // traverse pair parity in contiguous physical-domain runs.  For
            // one source hemisphere, pair p maps to hidden block
            // (p / 2) * 4 + 2 * hemisphere + p % 2.  Fixing parity makes
            // both cycle and address affine without interleaving two coarse
            // commands on the same MEM queue.
            if (directTailParityDomain) {
                FfnLoopDomain3D domain;
                domain.wave_count = pairCount / 2;
                domain.wave_interval = tile;
                domain.wave_address_stride = 4 * throughput.tile_rows;
                domain.group_count = 2;
                domain.group_interval = domain.wave_count * tile;
                domain.group_address_stride = throughput.tile_rows;
                if (!emitTileCopy(0, 0, domain, true))
                    return mlir::failure();
            } else {
                for (int64_t mTile = 0;
                     mTile < mTileCount; ++mTile) {
                    for (int64_t parity = 0;
                         parity < pairResidues; ++parity) {
                        FfnLoopDomain3D domain;
                        domain.group_count = 1
                            + (pairCount - 1 - parity) / pairStep;
                        domain.group_interval = tile;
                        domain.group_address_stride =
                            (singleMxmVector ? 4 : 1)
                            * throughput.tile_rows;
                        if (!emitTileCopy(mTile, parity, domain))
                            return mlir::failure();
                    }
                }
            }
            continue;
        }

        for (const CompletedProjectionTile* completed : scheduledTiles)
            if (completed->hemisphere == sourceHemisphere
                && !emitTileCopy(completed->m_tile, completed->pair))
                return mlir::failure();
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
    const bool singleMxmVector =
        throughput.mxms_per_hemisphere == 1;
    const int64_t pairStep = singleMxmVector ? 2 : 1;
    const int64_t pairResidues =
        std::min<int64_t>(pairStep, pairCount);
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
            "FFN Swish temporary storage does not cover every projection pair: "
            "m_tile_count=") << mTileCount << ", pair_count=" << pairCount
            << ", pairs_per_group=" << pairsPerTempGroup
            << ", required_slices=" << requiredTempSlices
            << ", gate_slices=" << gateTempSlices.size()
            << ", up_slices=" << upTempSlices.size();
        return mlir::failure();
    }
    const auto hiddenKind = ffn.getHidden0Placement()
        .getAs<mlir::StringAttr>("kind");
    const bool combineTailParities =
        context.strategy == FfnScheduleStrategy::Tail
        && singleMxmVector && mTileCount == 1
        && pairResidues == 2 && pairCount % 2 == 0
        && pairCount <= pairsPerTempGroup
        && hiddenKind
        && hiddenKind.getValue() == "fp16_mxm_distributed_16";

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

        mlir::Value gateValue;
        mlir::Value upValue;
        const int64_t firstInputCycle = inputCycle;
        for (int64_t mTile = 0; mTile < mTileCount; ++mTile) {
            // All Tail projection tiles are resident before Swish starts.
            // Execute even pairs followed by odd pairs so each physical
            // hidden-layout residue becomes one chronological affine run.
            for (int64_t parity = 0; parity < pairResidues; ++parity) {
            // Temporary slices change only at a storage-group boundary.
            // Within one group, pairStep advances both the issue cycle
            // and SRAM row by constants, so emit that complete run as
            // one hardware 3-D domain instead of enumerating its pairs.
            for (int64_t pair = parity; pair < pairCount;) {
                    if (combineTailParities && parity != 0) break;
                    const int64_t tempGroup = pair / pairsPerTempGroup;
                    const int64_t groupPairEnd = std::min<int64_t>(
                        pairCount, (tempGroup + 1) * pairsPerTempGroup);
                    const int64_t groupCount =
                        1 + (groupPairEnd - 1 - pair) / pairStep;
                    const int64_t tileInputCycle = inputCycle;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres; ++hemisphere) {
                        const CompletedProjectionTile* leader = nullptr;
                        const int64_t checkedPairs = combineTailParities
                            ? pairCount : groupCount;
                        for (int64_t index = 0;
                             index < checkedPairs; ++index) {
                            const int64_t currentPair = combineTailParities
                                ? (index < groupCount
                                    ? pair + index * pairStep
                                    : pair + 1
                                        + (index - groupCount) * pairStep)
                                : pair + index * pairStep;
                            const auto completed = llvm::find_if(
                                emission.completed_tiles,
                                [&](const CompletedProjectionTile& tile) {
                                    return tile.m_tile == mTile
                                        && tile.pair == currentPair
                                        && tile.hemisphere == hemisphere;
                                });
                            if (completed == emission.completed_tiles.end()) {
                                ffn.getOperation()->emitError(
                                    "missing completed FFN projection tile for "
                                    "vector Swish input");
                                return mlir::failure();
                            }
                            if (index == 0) leader = &*completed;
                        }
                        const int64_t tempBase =
                            ((pair % pairsPerTempGroup) * mTileCount + mTile)
                            * tile;
                        const int64_t streamBase = hemisphere * 16;
                        FfnLoopDomain3D domain;
                        if (combineTailParities) {
                            domain.wave_count = groupCount;
                            domain.wave_interval = tile;
                            domain.wave_address_stride =
                                pairStep * mTileCount * tile;
                            domain.group_count = 2;
                            domain.group_interval = groupCount * tile;
                            domain.group_address_stride =
                                mTileCount * tile;
                        } else {
                            domain.group_count = groupCount;
                            domain.group_interval = tile;
                            domain.group_address_stride =
                                pairStep * mTileCount * tile;
                        }
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t gateSlice =
                                gateTempSlices[2 * tempGroup + byte];
                            const int64_t upSlice =
                                upTempSlices[2 * tempGroup + byte];
                            auto gateRead = context.emitSliceRead(
                                leader->gate_temp,
                                context.activation_route,
                                tileInputCycle
                                    - context.westLatency(gateSlice),
                                gateSlice, tempBase, tile, 1,
                                streamBase + byte, "west", "vxm_bf16",
                                context.hemisphereName(hemisphere));
                            setFfnLoopDomain3D(
                                gateRead.getOperation(), context.rewriter,
                                domain);
                            gateValue = gateRead.getOutput();
                            auto upRead = context.emitSliceRead(
                                leader->up_temp,
                                context.activation_route,
                                tileInputCycle - context.westLatency(upSlice),
                                upSlice, tempBase, tile, 1,
                                streamBase + 2 + byte,
                                "west", "vxm_bf16",
                                context.hemisphereName(hemisphere));
                            setFfnLoopDomain3D(
                                upRead.getOperation(), context.rewriter,
                                domain);
                            upValue = upRead.getOutput();
                        }
                    }
                    inputCycle += (combineTailParities
                        ? pairCount : groupCount) * tile;
                    pair += combineTailParities
                        ? pairCount : groupCount * pairStep;
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
        if (combineTailParities) {
            FfnLoopDomain3D domain;
            domain.wave_count = pairCount / 2;
            domain.wave_interval = tile;
            domain.wave_address_stride = 4 * throughput.tile_rows;
            domain.group_count = 2;
            domain.group_interval = domain.wave_count * tile;
            domain.group_address_stride = throughput.tile_rows;
            for (int64_t hemisphere = 0;
                 hemisphere < memory.hemispheres; ++hemisphere)
                result.hidden = emitFfnSwishResultTile(
                    context.rewriter, ffn, target,
                    context.hidden_slices, output.getResult(),
                    firstInputCycle, 0, 0, hemisphere, tile, false,
                    domain, true);
        } else {
            int64_t pairOrdinal = 0;
            for (int64_t mTile = 0; mTile < mTileCount; ++mTile) {
                for (int64_t parity = 0; parity < pairResidues; ++parity) {
                    FfnLoopDomain3D domain;
                    domain.group_count =
                        1 + (pairCount - 1 - parity) / pairStep;
                    domain.group_interval = tile;
                    domain.group_address_stride =
                        (singleMxmVector ? 4 : 1)
                        * throughput.tile_rows;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres; ++hemisphere)
                        result.hidden = emitFfnSwishResultTile(
                            context.rewriter, ffn, target,
                            context.hidden_slices, output.getResult(),
                            firstInputCycle + pairOrdinal * tile,
                            mTile, parity, hemisphere, tile, false, domain);
                    pairOrdinal += domain.group_count;
                }
            }
        }

        if (mlir::failed(emitHiddenMirrorCopies(context, {},
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
    for (std::size_t index = 0; index < deferred.size(); ++index) {
        const CompletedProjectionTile& completed = *deferred[index];
        const int64_t start = (*cycles)[index];
        const int64_t tempGroup = completed.pair / pairsPerTempGroup;
        mlir::Value gateValue;
        mlir::Value upValue;
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
            gateValue = context.emitSliceRead(completed.gate_temp,
                context.activation_route,
                start - context.westLatency(gateSlice), gateSlice,
                tempBase, tile, 1, tempStreamBase + byte, "west",
                "vxm_bf16",
                context.hemisphereName(completed.hemisphere)).getOutput();
            upValue = context.emitSliceRead(completed.up_temp,
                context.activation_route,
                start - context.westLatency(upSlice), upSlice,
                tempBase, tile, 1, tempStreamBase + 2 + byte,
                "west", "vxm_bf16",
                context.hemisphereName(completed.hemisphere)).getOutput();
        }
        result.last_cycle = std::max(result.last_cycle, start + tile - 1);
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
        result.hidden = emitFfnSwishResultTile(context.rewriter, ffn,
            target, context.hidden_slices, output.getResult(), start,
            completed.m_tile, completed.pair, completed.hemisphere,
            tile, true);
    }
    if (mlir::failed(emitHiddenMirrorCopies(context, deferred,
            result.last_cycle, result)))
        return mlir::failure();
    return result;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
