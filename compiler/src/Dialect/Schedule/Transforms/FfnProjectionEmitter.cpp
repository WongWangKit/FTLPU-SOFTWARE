#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule::ffn_detail {

mlir::FailureOr<FfnProjectionEmission> emitFfnProjection(
    FfnEmissionContext& context)
{
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t m = context.m();
    const int64_t k = context.k();
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const int64_t mTileCount =
        context.projection_timeline.m_tile_count;
    const int64_t projectionSlotInterval =
        context.projection_timeline.projection_slot_interval;
    const int64_t gateAccLatency =
        throughput.mxm0_accumulator_latency;
    const int64_t upAccLatency =
        throughput.mxm1_accumulator_latency;
    const bool singleMxm = throughput.mxms_per_hemisphere == 1;

    FfnProjectionEmission emission;
    rewriter.setInsertionPoint(ffn.getOperation());
    for (const FfnProjectionBlockSchedule& block :
        context.projection_timeline.blocks) {
        const int64_t pair = block.pair;
        const int64_t reduction = block.reduction_block;
        const int64_t dequantStart = block.dequant_start;
        const int64_t weightBuffer = block.weight_buffer;

        for (int64_t hemisphere = 0;
             hemisphere < memory.hemispheres; ++hemisphere) {
            const int64_t projectionCount = singleMxm
                ? 2 : throughput.mxms_per_hemisphere;
            for (int64_t projection = 0;
                 projection < projectionCount; ++projection) {
                const int64_t localMxm = singleMxm ? 0 : projection;
                stream::RouteOp raw =
                    projection == 0 ? context.gate_raw : context.up_raw;
                stream::RouteOp cooked =
                    projection == 0 ? context.gate_route : context.up_route;
                const int64_t start = dequantStart
                    + (singleMxm
                            ? projection * tile
                            : (hemisphere
                                      * throughput.mxms_per_hemisphere
                                  + localMxm)
                                * weightLoadCycles);
                const auto placement = cooked.getPlacement();
                const int64_t bindingBase = placement
                                                .getAs<mlir::IntegerAttr>(
                                                    "base_row")
                                                .getInt();
                int64_t base = bindingBase;
                if (singleMxm) {
                    const int64_t logicalSlots = 2;
                    base += ((pair / logicalSlots) * (k / tile)
                                + reduction)
                            * logicalSlots * weightLoadCycles
                        + (pair % logicalSlots) * weightLoadCycles;
                } else {
                    base += (pair * (k / tile) + reduction)
                        * weightLoadCycles;
                }
                const int64_t logicalBase = base;
                int64_t page = -1;
                int64_t bank = placement
                    .getAs<mlir::IntegerAttr>("bank").getInt();
                llvm::SmallVector<int64_t> selectedWeightSlices =
                    projection == 0 ? context.weight_slices
                                    : context.up_weight_slices;
                if (auto paged = placement.getAs<mlir::BoolAttr>(
                        "paged_weight"); paged && paged.getValue()) {
                    const int64_t wavesPerPage = placement
                        .getAs<mlir::IntegerAttr>("page_granularity")
                        .getInt();
                    const int64_t roleGroupBase = placement
                        .getAs<mlir::IntegerAttr>("page_role_group_base")
                        .getInt();
                    const int64_t itemsPerGroup = placement
                        .getAs<mlir::IntegerAttr>(
                            "page_items_per_slice_group")
                        .getInt();
                    const int64_t bankCount = placement
                        .getAs<mlir::IntegerAttr>("page_bank_count")
                        .getInt();
                    page = pair / wavesPerPage;
                    bank = (bank + page) % bankCount;
                    const int64_t pairInPage = pair % wavesPerPage;
                    const int64_t sliceGroup = roleGroupBase
                        + pairInPage / itemsPerGroup;
                    const int64_t localPair = pairInPage % itemsPerGroup;
                    const auto storage = placement
                        .getAs<mlir::ArrayAttr>("page_storage_slices");
                    const int64_t loadSliceCount =
                        static_cast<int64_t>(selectedWeightSlices.size());
                    selectedWeightSlices.clear();
                    for (int64_t index = 0; index < loadSliceCount; ++index)
                        selectedWeightSlices.push_back(
                            llvm::cast<mlir::IntegerAttr>(
                                storage[sliceGroup * loadSliceCount + index])
                                .getInt());
                    const int64_t logicalSlots = singleMxm ? 2 : 1;
                    base = ((localPair / logicalSlots) * (k / tile)
                              + reduction)
                            * logicalSlots * weightLoadCycles
                        + (localPair % logicalSlots) * weightLoadCycles;
                }
                const auto rawType =
                    llvm::cast<mlir::RankedTensorType>(
                        raw.getInput().getType());
                const auto activationType =
                    llvm::cast<mlir::RankedTensorType>(
                        ffn.getActivation().getType());
                const auto dequantizedType =
                    mlir::RankedTensorType::get(rawType.getShape(),
                        activationType.getElementType());
                emitFfnWeightTile(rewriter, ffn.getLoc(), raw,
                    dequantizedType,
                    selectedWeightSlices,
                    target,
                    projection == 0
                        ? ffn.getGateScale().convertToFloat()
                        : ffn.getUpScale().convertToFloat(),
                    start, base, hemisphere, localMxm,
                    hemisphere * throughput.mxms_per_hemisphere
                        + localMxm,
                    singleMxm ? projection : weightBuffer,
                    context.local_weight_dequant, bank, page,
                    logicalBase);
            }
        }

        for (const FfnProjectionTileSchedule& tileSchedule :
            block.tiles) {
            const int64_t mTile = tileSchedule.m_tile;
            const int64_t computeCycle = tileSchedule.compute_cycle;
            for (int64_t hemisphere = 0;
                 hemisphere < memory.hemispheres; ++hemisphere) {
                const int64_t activationBase =
                    reduction * m + mTile * tile;
                const bool finalReduction = block.final_reduction;
                const int64_t resultStreamBase =
                    finalReduction
                    ? (context.strategy == FfnScheduleStrategy::Fused
                            ? 8 + hemisphere * 8
                            : 0)
                    : 0;

                MxmComputeOp gateCompute;
                MxmComputeOp upCompute;
                int64_t rowOffset = 0;
                for (const FfnStreamSegment& segment :
                    tileSchedule.hemisphere_segments[
                        static_cast<std::size_t>(hemisphere)]) {
                    const int64_t segmentCycle =
                        computeCycle + rowOffset;
                    const auto emitActivation =
                        [&](int64_t consumerCycle) {
                            mlir::Value activationValue;
                            if (context.activation_distributed16) {
                                const int64_t base =
                                    context.activation_route.getPlacement()
                                        .getAs<mlir::IntegerAttr>("base_row")
                                        .getInt();
                                const int64_t reductionBlocks = k / tile;
                                for (int64_t localRow = 0;
                                     localRow < segment.rows; ++localRow) {
                                    const int64_t token =
                                        mTile * tile + rowOffset + localRow;
                                    const int64_t tokenBlock = token / tile;
                                    const int64_t tokenWithinBlock =
                                        token % tile;
                                    const int64_t tokenWave =
                                        tokenWithinBlock / 8;
                                    const int64_t tokenLane =
                                        tokenWithinBlock % 8;
                                    const int64_t row = base
                                        + (tokenBlock * reductionBlocks
                                              + reduction)
                                            * 4
                                        + tokenWave;
                                    for (int64_t byte = 0; byte < 2;
                                         ++byte) {
                                        const int64_t slice =
                                            context.activation_slices[
                                                2 * tokenLane + byte];
                                        activationValue = context
                                            .emitSliceRead(
                                                context.activation_route
                                                    .getInput(),
                                                context.activation_route,
                                                consumerCycle + localRow
                                                    - context.eastMxmLatency(
                                                        slice),
                                                slice, row, 1, 1,
                                                segment.stream_base + byte,
                                                "east", "activation",
                                                context.hemisphereName(
                                                    hemisphere))
                                            .getOutput();
                                    }
                                }
                            } else {
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    activationValue = context.emitSliceRead(
                                        context.activation_route.getInput(),
                                        context.activation_route,
                                        consumerCycle
                                            - context.activation_latency,
                                        context.activation_slices[byte],
                                        activationBase + rowOffset,
                                        segment.rows, 1,
                                        segment.stream_base + byte,
                                        "east", "activation",
                                        context.hemisphereName(hemisphere))
                                        .getOutput();
                                }
                            }
                            return activationValue;
                        };
                    const int64_t gateCycle = segmentCycle;
                    const int64_t upCycle =
                        segmentCycle
                        + (singleMxm ? projectionSlotInterval : 0);
                    mlir::Value gateActivation = emitActivation(gateCycle);
                    mlir::Value upActivation = singleMxm
                        ? emitActivation(upCycle) : gateActivation;
                    gateCompute =
                        rewriter.create<MxmComputeOp>(ffn.getLoc(),
                            gateActivation, ffn.getGateWeight(),
                            context.projection_type, gateCycle,
                            segment.rows,
                            gateCycle
                                + target.mxm_first_result_latency(),
                            target.mxm_result_window_cycles(
                                segment.rows),
                            segment.stream_base, resultStreamBase,
                            singleMxm ? 0 : weightBuffer,
                            hemisphere
                                * throughput.mxms_per_hemisphere,
                            segment.rows, tile, tile);
                    upCompute =
                        rewriter.create<MxmComputeOp>(ffn.getLoc(),
                            upActivation, ffn.getUpWeight(),
                            context.projection_type, upCycle,
                            segment.rows,
                            upCycle
                                + target.mxm_first_result_latency(),
                            target.mxm_result_window_cycles(
                                segment.rows),
                            segment.stream_base,
                            resultStreamBase
                                + throughput.mxm_result_streams,
                            singleMxm ? 1 : weightBuffer,
                            hemisphere
                                    * throughput.mxms_per_hemisphere
                                + (singleMxm ? 0 : 1),
                            segment.rows, tile, tile);
                    rowOffset += segment.rows;
                }

                // Each physical MXM owns its accumulator. A pair is fully
                // reduced and drained before the next pair starts, so all
                // projection pairs can reuse the same token-row window.
                const int64_t accumulatorBase = mTile * tile;

                const auto emitAccumulator =
                    [&](mlir::Value input, int64_t unitId,
                        int64_t accumulatorAddress, int64_t cycle,
                        int64_t streamBase) {
                        mlir::OperationState state(
                            ffn.getLoc(),
                            MxmAccumulateOp::getOperationName());
                        state.addOperands(input);
                        state.addTypes(context.projection_type);
                        state.addAttributes({
                            rewriter.getNamedAttr("cycle",
                                rewriter.getI64IntegerAttr(cycle)),
                            rewriter.getNamedAttr("unit_id",
                                rewriter.getI64IntegerAttr(unitId)),
                            rewriter.getNamedAttr("stream_base",
                                rewriter.getI64IntegerAttr(streamBase)),
                            rewriter.getNamedAttr("stream_count",
                                rewriter.getI64IntegerAttr(
                                    finalReduction ? 2
                                                   : throughput
                                                         .mxm_result_streams)),
                            rewriter.getNamedAttr("accumulator_address",
                                rewriter.getI64IntegerAttr(
                                    accumulatorAddress)),
                            rewriter.getNamedAttr("accumulator_stride",
                                rewriter.getI64IntegerAttr(1)),
                            rewriter.getNamedAttr("destination",
                                rewriter.getStringAttr(
                                    finalReduction ? "stream" : "local")),
                            rewriter.getNamedAttr("repeat_count",
                                rewriter.getI64IntegerAttr(tile)),
                            rewriter.getNamedAttr("repeat_interval",
                                rewriter.getI64IntegerAttr(1)),
                            rewriter.getNamedAttr(
                                "accumulator_output_format",
                                rewriter.getStringAttr(
                                    finalReduction ? "bf16" : "fp32")),
                        });
                        return llvm::cast<MxmAccumulateOp>(
                            rewriter.create(state));
                    };
                auto gateAccumulator = emitAccumulator(
                    gateCompute.getResult(), gateCompute.getUnitId(),
                    accumulatorBase, computeCycle + gateAccLatency,
                    resultStreamBase);
                const int64_t upComputeCycle =
                    computeCycle
                    + (singleMxm ? projectionSlotInterval : 0);
                const int64_t effectiveUpAccLatency =
                    singleMxm ? gateAccLatency : upAccLatency;
                auto upAccumulator = emitAccumulator(
                    upCompute.getResult(), upCompute.getUnitId(),
                    accumulatorBase + (singleMxm ? m : 0),
                    upComputeCycle + effectiveUpAccLatency,
                    resultStreamBase + throughput.mxm_result_streams);

                if (!finalReduction) continue;

                const auto gateTempSlices = target.ffn_gate_temp_slices();
                const auto upTempSlices = target.ffn_up_temp_slices();
                const int64_t pairsPerTempGroup =
                    memory.sram_depth_rows / (mTileCount * tile);
                const int64_t tempGroup = pair / pairsPerTempGroup;
                if (2 * tempGroup + 1
                        >= static_cast<int64_t>(gateTempSlices.size())
                    || 2 * tempGroup + 1
                        >= static_cast<int64_t>(upTempSlices.size()))
                    return mlir::failure();
                mlir::Value gateTemp;
                mlir::Value upTemp;
                int64_t deferredReadyCycle = std::max(
                    computeCycle + gateAccLatency + tile
                        + context.westLatency(
                            gateTempSlices[2 * tempGroup]),
                    upComputeCycle + effectiveUpAccLatency + tile
                        + context.westLatency(
                            upTempSlices[2 * tempGroup]));
                {
                    const int64_t tempBase =
                        ((pair % pairsPerTempGroup) * mTileCount + mTile)
                        * tile;
                    const auto emitTempWrite =
                        [&](MxmAccumulateOp source,
                            llvm::ArrayRef<int64_t> tempSlices,
                            int64_t streamBase, int64_t sourceComputeCycle,
                            mlir::Value& lastWrite) {
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t targetSlice =
                                    tempSlices[byte];
                                const int64_t targetBoundary =
                                    targetSlice
                                        / target.streams()
                                              .mem_slices_per_register_group
                                    + 1;
                                const auto transportLatency =
                                    target.transport_latency(
                                        target::StreamEndpoint::MxmResult,
                                        target::StreamEndpoint::Mem,
                                        target::StreamDirection::West,
                                        targetSlice);
                                if (!transportLatency)
                                    return;
                                const int64_t writeCycle = sourceComputeCycle
                                    + target.mxm_first_result_latency()
                                    + *transportLatency;
                                auto placement = schedule_placement(
                                    rewriter, {targetSlice}, tempBase,
                                    tile, 1,
                                    context.hemisphereName(hemisphere),
                                    "bf16_swiglu_temp_byte",
                                    context.temp_bank);
                                auto write =
                                    rewriter.create<MemWriteOp>(
                                        ffn.getLoc(), source.getOutput(),
                                        writeCycle, tile,
                                        streamBase + byte, 1,
                                        targetBoundary,
                                        rewriter.getStringAttr("west"),
                                        ffn.getHidden1Address(),
                                        placement,
                                        tile
                                            * throughput.lanes_per_tile);
                                lastWrite = write.getOutput();
                                emission
                                    .temp_mem_busy_windows[hemisphere]
                                    .push_back({
                                        writeCycle,
                                        writeCycle + tile});
                                deferredReadyCycle =
                                    std::max(deferredReadyCycle,
                                        writeCycle + tile
                                            + context.westLatency(
                                                targetSlice));
                            }
                        };
                    emitTempWrite(gateAccumulator,
                        llvm::ArrayRef<int64_t>(gateTempSlices)
                            .slice(2 * tempGroup, 2),
                        resultStreamBase, computeCycle,
                        gateTemp);
                    emitTempWrite(upAccumulator,
                        llvm::ArrayRef<int64_t>(upTempSlices)
                            .slice(2 * tempGroup, 2),
                        resultStreamBase
                            + throughput.mxm_result_streams,
                        upComputeCycle,
                        upTemp);
                }
                emission.completed_tiles.push_back({
                    pair,
                    mTile,
                    hemisphere,
                    computeCycle,
                    deferredReadyCycle,
                    gateAccumulator,
                    upAccumulator,
                    gateTemp,
                    upTemp,
                });
            }
        }
    }
    return emission;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
