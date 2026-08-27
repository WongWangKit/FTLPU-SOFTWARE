#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {

mlir::FailureOr<mlir::Value> emitFfnDownProjection(
    FfnEmissionContext& context, const FfnSwishEmission& swish)
{
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t m = context.m();
    const int64_t intermediate = context.hidden();
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const int64_t projectionSlotInterval =
        context.projection_timeline.projection_slot_interval;
    const int64_t gateAccLatency =
        throughput.mxm0_accumulator_latency;
    const int64_t upAccLatency =
        throughput.mxm1_accumulator_latency;
    const bool singleMxm = throughput.mxms_per_hemisphere == 1;
    const int64_t logicalSlotsPerHemisphere =
        singleMxm ? 2 : throughput.mxms_per_hemisphere;
    const int64_t hiddenBaseRow =
        get_base_row(ffn.getHidden0Placement());
    const int64_t resultBank = ffn.getResultPlacement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const auto hiddenKind =
        ffn.getHidden0Placement().getAs<mlir::StringAttr>("kind");
    const bool hiddenDistributed16 = hiddenKind
        && hiddenKind.getValue() == "fp16_mxm_distributed_16";

    auto timeline = planFfnDownProjectionTimeline(
        {context.m(), context.k(), intermediate, context.n()},
        context.projection_timeline, swish.last_cycle,
        context.weight_slices, context.hidden_slices,
        context.result_slices, target);
    if (mlir::failed(timeline)) return mlir::failure();

    for (const FfnDownBlockSchedule& block : timeline->blocks) {
        const int64_t outputWave = block.output_wave;
        const int64_t reduction = block.reduction_block;
        const int64_t activeHemispheres = block.active_hemispheres;
        const int64_t weightBuffer = block.weight_buffer;

        for (int64_t hemisphere = 0;
             hemisphere < activeHemispheres; ++hemisphere) {
            for (int64_t logicalSlot = 0;
                 logicalSlot < logicalSlotsPerHemisphere;
                 ++logicalSlot) {
                const int64_t localMxm = singleMxm ? 0 : logicalSlot;
                const int64_t unit =
                    hemisphere * throughput.mxms_per_hemisphere
                    + localMxm;
                const int64_t start =
                    block.dequant_start
                    + (hemisphere * logicalSlotsPerHemisphere
                          + logicalSlot)
                        * weightLoadCycles;
                const auto placement = context.down_route.getPlacement();
                const int64_t bindingBase = placement
                        .getAs<mlir::IntegerAttr>("base_row")
                        .getInt();
                const int64_t logicalBase = bindingBase
                    + (outputWave * (intermediate / tile) + reduction)
                        * logicalSlotsPerHemisphere
                        * weightLoadCycles
                    + logicalSlot * weightLoadCycles;
                int64_t base = logicalBase;
                int64_t page = -1;
                int64_t bank = placement
                    .getAs<mlir::IntegerAttr>("bank").getInt();
                llvm::SmallVector<int64_t> selectedWeightSlices(
                    context.down_weight_slices.begin(),
                    context.down_weight_slices.end());
                if (auto paged = placement.getAs<mlir::BoolAttr>(
                        "paged_weight"); paged && paged.getValue()) {
                    const int64_t reductionsPerPage = placement
                        .getAs<mlir::IntegerAttr>("page_granularity")
                        .getInt();
                    const int64_t roleGroupBase = placement
                        .getAs<mlir::IntegerAttr>("page_role_group_base")
                        .getInt();
                    const int64_t reductionsPerGroup = placement
                        .getAs<mlir::IntegerAttr>(
                            "page_items_per_slice_group")
                        .getInt();
                    const int64_t bankCount = placement
                        .getAs<mlir::IntegerAttr>("page_bank_count")
                        .getInt();
                    const int64_t pagesPerWave =
                        (intermediate / tile + reductionsPerPage - 1)
                        / reductionsPerPage;
                    page = outputWave * pagesPerWave
                        + reduction / reductionsPerPage;
                    bank = (bank + page) % bankCount;
                    const int64_t reductionInPage =
                        reduction % reductionsPerPage;
                    const int64_t sliceGroup = roleGroupBase
                        + reductionInPage / reductionsPerGroup;
                    const int64_t localReduction =
                        reductionInPage % reductionsPerGroup;
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
                    base = localReduction * logicalSlotsPerHemisphere
                            * weightLoadCycles
                        + logicalSlot * weightLoadCycles;
                }
                const auto rawType =
                    llvm::cast<mlir::RankedTensorType>(
                        context.down_raw.getInput().getType());
                const auto activationType =
                    llvm::cast<mlir::RankedTensorType>(
                        ffn.getActivation().getType());
                const auto dequantizedType =
                    mlir::RankedTensorType::get(rawType.getShape(),
                        activationType.getElementType());
                emitFfnWeightTile(rewriter, ffn.getLoc(),
                    context.down_raw,
                    dequantizedType,
                    selectedWeightSlices, target,
                    ffn.getDownRhsScale().convertToFloat(), start,
                    base, hemisphere, logicalSlot, unit,
                    singleMxm ? logicalSlot : weightBuffer,
                    context.local_weight_dequant, bank, page,
                    logicalBase);
            }
        }

        for (const FfnDownTileSchedule& tileSchedule : block.tiles) {
            const int64_t mTile = tileSchedule.m_tile;
            const int64_t computeCycle = tileSchedule.compute_cycle;
            for (int64_t hemisphere = 0;
                 hemisphere < activeHemispheres; ++hemisphere) {
                MxmComputeOp down0;
                MxmComputeOp down1;
                int64_t rowOffset = 0;
                for (const FfnStreamSegment& segment :
                    tileSchedule.segments) {
                    const int64_t segmentCycle =
                        computeCycle + rowOffset;
                    const auto emitHidden = [&](int64_t consumerCycle) {
                        mlir::Value hiddenValue;
                        if (hiddenDistributed16) {
                            const int64_t hiddenBlocks = intermediate / tile;
                            for (int64_t localRow = 0;
                                 localRow < segment.rows; ++localRow) {
                                const int64_t token =
                                    mTile * tile + rowOffset + localRow;
                                const int64_t tokenWithinBlock = token % tile;
                                const int64_t tokenWave = tokenWithinBlock
                                    / throughput.mxm_block_rows;
                                const int64_t tokenLane = tokenWithinBlock
                                    % throughput.mxm_block_rows;
                                const int64_t address = hiddenBaseRow
                                    + ((token / tile) * hiddenBlocks
                                          + reduction)
                                        * throughput.tile_rows
                                    + tokenWave;
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    const int64_t slice = context.hidden_slices[
                                        2 * tokenLane + byte];
                                    hiddenValue = context.emitSliceRead(
                                        swish.hidden,
                                        context.activation_route,
                                        consumerCycle + localRow
                                            - context.eastMxmLatency(slice),
                                        slice, address, 1, 1,
                                        segment.stream_base + byte, "east",
                                        "activation",
                                        context.hemisphereName(hemisphere))
                                        .getOutput();
                                }
                            }
                            return hiddenValue;
                        }
                        const int64_t hiddenPair = reduction % 2;
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            hiddenValue = context
                                .emitSliceRead(swish.hidden,
                                    context.activation_route,
                                    consumerCycle
                                        - context.eastMxmLatency(
                                            context.hidden_slices[
                                                2 * hiddenPair + byte]),
                                    context.hidden_slices[
                                        2 * hiddenPair + byte],
                                    hiddenBaseRow
                                        + (reduction / 2) * m
                                        + mTile * tile
                                        + rowOffset,
                                    segment.rows, 1,
                                    segment.stream_base + byte, "east",
                                    "activation",
                                    context.hemisphereName(hemisphere))
                                .getOutput();
                        }
                        return hiddenValue;
                    };
                    const int64_t down0Cycle = segmentCycle;
                    const int64_t down1Cycle =
                        segmentCycle
                        + (singleMxm ? projectionSlotInterval : 0);
                    mlir::Value hidden0 = emitHidden(down0Cycle);
                    mlir::Value hidden1 = singleMxm
                        ? emitHidden(down1Cycle) : hidden0;
                    const int64_t unitBase =
                        hemisphere * throughput.mxms_per_hemisphere;
                    down0 = rewriter.create<MxmComputeOp>(ffn.getLoc(),
                        hidden0, ffn.getDownWeight0(),
                        context.projection_type, down0Cycle,
                        segment.rows,
                        down0Cycle
                            + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(segment.rows),
                        segment.stream_base, 0,
                        singleMxm ? 0 : weightBuffer, unitBase,
                        segment.rows, tile, tile);
                    down1 = rewriter.create<MxmComputeOp>(ffn.getLoc(),
                        hidden1, ffn.getDownWeight1(),
                        context.projection_type, down1Cycle,
                        segment.rows,
                        down1Cycle
                            + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(segment.rows),
                        segment.stream_base,
                        throughput.mxm_result_streams,
                        singleMxm ? 1 : weightBuffer,
                        unitBase + (singleMxm ? 0 : 1),
                        segment.rows, tile, tile);
                    rowOffset += segment.rows;
                }

                const int64_t accumulatorBase =
                    context.down_accumulator_base + mTile * tile;
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
                                    block.final_reduction ? 2
                                                          : throughput
                                                                .mxm_result_streams)),
                            rewriter.getNamedAttr("accumulator_address",
                                rewriter.getI64IntegerAttr(
                                    accumulatorAddress)),
                            rewriter.getNamedAttr("accumulator_stride",
                                rewriter.getI64IntegerAttr(1)),
                            rewriter.getNamedAttr("destination",
                                rewriter.getStringAttr(
                                    block.final_reduction ? "stream"
                                                          : "local")),
                            rewriter.getNamedAttr("repeat_count",
                                rewriter.getI64IntegerAttr(tile)),
                            rewriter.getNamedAttr("repeat_interval",
                                rewriter.getI64IntegerAttr(1)),
                            rewriter.getNamedAttr(
                                "accumulator_output_format",
                                rewriter.getStringAttr(
                                    block.final_reduction ? "bf16"
                                                          : "fp32")),
                        });
                        return llvm::cast<MxmAccumulateOp>(
                            rewriter.create(state));
                    };
                auto acc0 = emitAccumulator(down0.getResult(),
                    down0.getUnitId(), accumulatorBase,
                    computeCycle + gateAccLatency, 0);
                const int64_t down1ComputeCycle =
                    computeCycle
                    + (singleMxm ? projectionSlotInterval : 0);
                auto acc1 = emitAccumulator(down1.getResult(),
                    down1.getUnitId(),
                    accumulatorBase + (singleMxm ? m : 0),
                    down1ComputeCycle
                        + (singleMxm ? gateAccLatency : upAccLatency),
                    throughput.mxm_result_streams);
                if (!block.final_reduction) continue;

                for (int64_t row = 0; row < tile; ++row) {
                    for (int64_t byte = 0;
                         byte < throughput.mxm_result_streams; ++byte) {
                        const int64_t streamBase = byte < 2
                            ? 0 : throughput.mxm_result_streams;
                        const int64_t slice = context.result_slices[byte];
                        const auto latency = target.transport_latency(
                            target::StreamEndpoint::MxmResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::West, slice);
                        if (!latency) return mlir::failure();
                        auto placement = schedule_placement(rewriter,
                            {slice},
                            outputWave * m + mTile * tile + row, 1, 1,
                            context.hemisphereName(hemisphere),
                            "fp16_pair_planar", resultBank);
                        mlir::NamedAttrList attributes(placement);
                        llvm::SmallVector<mlir::Attribute> allSlices;
                        for (int64_t slice : context.result_slices)
                            allSlices.push_back(
                                rewriter.getI64IntegerAttr(slice));
                        attributes.set("binding_slices",
                            rewriter.getArrayAttr(allSlices));
                        attributes.set("binding_instruction_count",
                            rewriter.getI64IntegerAttr(
                                ffn.getM() * timeline->wave_count));
                        attributes.set("binding_placement",
                            schedule_placement(rewriter,
                                context.result_slices, 0,
                                ffn.getM() * timeline->wave_count, 1,
                                "both", "fp16_pair_planar", resultBank));
                        auto write = rewriter.create<MemWriteOp>(
                            ffn.getLoc(),
                            byte < 2 ? acc0.getOutput()
                                     : acc1.getOutput(),
                            (byte < 2 ? computeCycle
                                      : down1ComputeCycle)
                                + target.mxm_first_result_latency()
                                + row + *latency,
                            1, streamBase + byte % 2, 1, 0,
                            rewriter.getStringAttr("west"),
                            ffn.getResultAddress(),
                            attributes.getDictionary(
                                rewriter.getContext()),
                            tile);
                        (void)write;
                    }
                }
            }
        }
    }
    auto resultType = llvm::cast<mlir::RankedTensorType>(
        ffn.getResult().getType());
    mlir::OperationState bindingState(
        ffn.getLoc(), BindingOp::getOperationName());
    bindingState.addTypes(resultType);
    bindingState.addAttributes({
        context.rewriter.getNamedAttr("index",
            context.rewriter.getI64IntegerAttr(0)),
        context.rewriter.getNamedAttr("access",
            context.rewriter.getStringAttr("output")),
        context.rewriter.getNamedAttr("role",
            context.rewriter.getStringAttr("result")),
        context.rewriter.getNamedAttr("bytes",
            context.rewriter.getI64IntegerAttr(
                resultType.getNumElements() * 2)),
        context.rewriter.getNamedAttr(
            "placement", ffn.getResultPlacement()),
    });
    auto output = llvm::cast<BindingOp>(
        context.rewriter.create(bindingState));
    return output.getValue();
}

} // namespace ftlpu::compiler::schedule::ffn_detail
