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
    const int64_t intermediate = context.hidden();
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const int64_t mTileCount =
        context.projection_timeline.m_tile_count;
    const int64_t gateAccLatency =
        throughput.mxm0_accumulator_latency;
    const int64_t upAccLatency =
        throughput.mxm1_accumulator_latency;

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
            for (int64_t localMxm = 0;
                 localMxm < throughput.mxms_per_hemisphere; ++localMxm) {
                stream::RouteOp raw =
                    localMxm == 0 ? context.gate_raw : context.up_raw;
                stream::RouteOp cooked =
                    localMxm == 0 ? context.gate_route : context.up_route;
                const int64_t start = dequantStart
                    + (hemisphere * throughput.mxms_per_hemisphere
                          + localMxm)
                        * weightLoadCycles;
                const int64_t base =
                    cooked.getPlacement()
                        .getAs<mlir::IntegerAttr>("base_row")
                        .getInt()
                    + (pair * (k / tile) + reduction)
                        * weightLoadCycles;
                emitFfnWeightTile(rewriter, ffn.getLoc(), raw,
                    cooked.getInput().getType(), context.weight_slices,
                    target,
                    localMxm == 0
                        ? ffn.getGateScale().convertToFloat()
                        : ffn.getUpScale().convertToFloat(),
                    start, base, hemisphere, localMxm,
                    hemisphere * throughput.mxms_per_hemisphere
                        + localMxm,
                    weightBuffer);
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
                    context.strategy == FfnScheduleStrategy::Fused
                        && finalReduction
                    ? 8 + hemisphere * 8
                    : 0;

                MxmComputeOp gateCompute;
                MxmComputeOp upCompute;
                int64_t rowOffset = 0;
                for (const FfnStreamSegment& segment :
                    tileSchedule.hemisphere_segments[
                        static_cast<std::size_t>(hemisphere)]) {
                    mlir::Value activationValue;
                    const int64_t segmentCycle =
                        computeCycle + rowOffset;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        activationValue =
                            context
                                .emitSliceRead(
                                    context.activation_route.getInput(),
                                    context.activation_route,
                                    segmentCycle
                                        - context.activation_latency,
                                    context.activation_slices[byte],
                                    activationBase + rowOffset,
                                    segment.rows, 1,
                                    segment.stream_base + byte, "east",
                                    "activation",
                                    context.hemisphereName(hemisphere))
                                .getOutput();
                    }
                    gateCompute =
                        rewriter.create<MxmComputeOp>(ffn.getLoc(),
                            activationValue, ffn.getGateWeight(),
                            context.projection_type, segmentCycle,
                            segment.rows,
                            segmentCycle
                                + target.mxm_first_result_latency(),
                            target.mxm_result_window_cycles(
                                segment.rows),
                            segment.stream_base, resultStreamBase,
                            weightBuffer,
                            hemisphere
                                * throughput.mxms_per_hemisphere,
                            segment.rows, tile, tile);
                    upCompute =
                        rewriter.create<MxmComputeOp>(ffn.getLoc(),
                            activationValue, ffn.getUpWeight(),
                            context.projection_type, segmentCycle,
                            segment.rows,
                            segmentCycle
                                + target.mxm_first_result_latency(),
                            target.mxm_result_window_cycles(
                                segment.rows),
                            segment.stream_base,
                            resultStreamBase
                                + throughput.mxm_result_streams,
                            weightBuffer,
                            hemisphere
                                    * throughput.mxms_per_hemisphere
                                + 1,
                            segment.rows, tile, tile);
                    rowOffset += segment.rows;
                }

                const int64_t outputBlock =
                    pair * memory.hemispheres + hemisphere;
                const int64_t accumulatorBase =
                    mTile * tile * (intermediate / tile) + outputBlock;
                auto gatePlacement = schedule_placement(rewriter,
                    context.gate_acc_slices, accumulatorBase, tile,
                    intermediate / tile,
                    context.hemisphereName(hemisphere),
                    "fp32_accumulator");
                auto upPlacement = schedule_placement(rewriter,
                    context.up_acc_slices, accumulatorBase, tile,
                    intermediate / tile,
                    context.hemisphereName(hemisphere),
                    "fp32_accumulator");

                const auto emitAccumulator =
                    [&](mlir::Value input, mlir::DictionaryAttr address,
                        mlir::DictionaryAttr placement, int64_t cycle,
                        int64_t streamBase) {
                        mlir::OperationState state(
                            ffn.getLoc(),
                            MemAccumulateOp::getOperationName());
                        state.addOperands(input);
                        state.addTypes(context.projection_type);
                        state.addAttributes({
                            rewriter.getNamedAttr("cycle",
                                rewriter.getI64IntegerAttr(cycle)),
                            rewriter.getNamedAttr("stream_base",
                                rewriter.getI64IntegerAttr(streamBase)),
                            rewriter.getNamedAttr("stream_count",
                                rewriter.getI64IntegerAttr(
                                    throughput.mxm_result_streams)),
                            rewriter.getNamedAttr("address", address),
                            rewriter.getNamedAttr(
                                "placement", placement),
                            rewriter.getNamedAttr("hemisphere",
                                rewriter.getStringAttr(
                                    context.hemisphereName(
                                        hemisphere))),
                            rewriter.getNamedAttr("destination",
                                rewriter.getStringAttr(
                                    context.strategy
                                            == FfnScheduleStrategy::Fused
                                            && finalReduction
                                        ? "stream"
                                        : "sram")),
                            rewriter.getNamedAttr("repeat_count",
                                rewriter.getI64IntegerAttr(tile)),
                            rewriter.getNamedAttr("repeat_interval",
                                rewriter.getI64IntegerAttr(1)),
                            rewriter.getNamedAttr("address_stride",
                                rewriter.getI64IntegerAttr(
                                    intermediate / tile)),
                        });
                        return llvm::cast<MemAccumulateOp>(
                            rewriter.create(state));
                    };
                auto gateAccumulator = emitAccumulator(
                    gateCompute.getResult(), ffn.getHidden0AddressAttr(),
                    gatePlacement, computeCycle + gateAccLatency,
                    resultStreamBase);
                auto upAccumulator = emitAccumulator(
                    upCompute.getResult(), ffn.getHidden1AddressAttr(),
                    upPlacement, computeCycle + upAccLatency,
                    resultStreamBase + throughput.mxm_result_streams);

                if (!finalReduction) continue;

                mlir::Value gateTemp;
                mlir::Value upTemp;
                int64_t deferredReadyCycle = computeCycle
                    + std::max(
                        gateAccLatency + tile
                            + context.westLatency(
                                context.gate_acc_slices.front()),
                        upAccLatency + tile
                            + context.westLatency(
                                context.up_acc_slices.front()));
                if (context.strategy == FfnScheduleStrategy::Fused) {
                    const int64_t tempBase =
                        (pair * mTileCount + mTile) * tile;
                    const auto emitTempWrite =
                        [&](MemAccumulateOp source,
                            llvm::ArrayRef<int64_t> tempSlices,
                            llvm::ArrayRef<int64_t> accSlices,
                            int64_t accCycle, int64_t streamBase,
                            mlir::Value& lastWrite) {
                            for (int64_t byte = 0;
                                 byte < throughput.mxm_result_streams;
                                 ++byte) {
                                const int64_t targetSlice =
                                    tempSlices[byte];
                                const int64_t sourceBoundary =
                                    accSlices.front()
                                    / target.streams()
                                          .mem_slices_per_register_group;
                                const int64_t targetBoundary =
                                    targetSlice
                                        / target.streams()
                                              .mem_slices_per_register_group
                                    + 1;
                                const int64_t writeCycle = accCycle
                                    + sourceBoundary - targetBoundary + 1;
                                auto placement = schedule_placement(
                                    rewriter, {targetSlice}, tempBase,
                                    tile, 1,
                                    context.hemisphereName(hemisphere),
                                    "fp32_swiglu_temp_byte");
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
                                        writeCycle
                                            + context.westLatency(
                                                targetSlice),
                                        writeCycle + tile
                                            + context.westLatency(
                                                targetSlice)});
                                deferredReadyCycle =
                                    std::max(deferredReadyCycle,
                                        writeCycle + tile
                                            + context.westLatency(
                                                targetSlice));
                            }
                        };
                    emitTempWrite(gateAccumulator,
                        memory.w8a16_fused_gate_temp_slices,
                        context.gate_acc_slices,
                        computeCycle + gateAccLatency, resultStreamBase,
                        gateTemp);
                    emitTempWrite(upAccumulator,
                        memory.w8a16_fused_up_temp_slices,
                        context.up_acc_slices,
                        computeCycle + upAccLatency,
                        resultStreamBase
                            + throughput.mxm_result_streams,
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
