#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {

mlir::FailureOr<mlir::Value> emitFfnDownProjection(
    FfnEmissionContext& context, const FfnSwishEmission& swish)
{
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t m = context.m();
    const int64_t intermediate = context.hidden();
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const int64_t gateAccLatency =
        throughput.mxm0_accumulator_latency;
    const int64_t upAccLatency =
        throughput.mxm1_accumulator_latency;

    auto timeline = planFfnDownProjectionTimeline(
        {context.m(), context.k(), intermediate, context.n()},
        context.projection_timeline, swish.last_cycle,
        context.weight_slices, context.hidden_slices,
        context.result_slices, target);
    if (mlir::failed(timeline)) return mlir::failure();

    mlir::Value finalValue;
    for (const FfnDownBlockSchedule& block : timeline->blocks) {
        const int64_t outputWave = block.output_wave;
        const int64_t reduction = block.reduction_block;
        const int64_t activeHemispheres = block.active_hemispheres;
        const int64_t weightBuffer = block.weight_buffer;

        for (int64_t hemisphere = 0;
             hemisphere < activeHemispheres; ++hemisphere) {
            for (int64_t localMxm = 0;
                 localMxm < throughput.mxms_per_hemisphere;
                 ++localMxm) {
                const int64_t unit =
                    hemisphere * throughput.mxms_per_hemisphere
                    + localMxm;
                const int64_t start =
                    block.dequant_start + unit * weightLoadCycles;
                const int64_t base =
                    context.down_route.getPlacement()
                        .getAs<mlir::IntegerAttr>("base_row")
                        .getInt()
                    + (outputWave * (intermediate / tile) + reduction)
                        * throughput.mxms_per_hemisphere
                        * weightLoadCycles
                    + localMxm * weightLoadCycles;
                emitFfnWeightTile(rewriter, ffn.getLoc(),
                    context.down_raw,
                    context.down_route.getInput().getType(),
                    context.weight_slices, target,
                    ffn.getDownRhsScale().convertToFloat(), start,
                    base, hemisphere, localMxm, unit, weightBuffer);
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
                    mlir::Value hiddenValue;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        hiddenValue =
                            context
                                .emitSliceRead(swish.hidden,
                                    context.activation_route,
                                    segmentCycle
                                        - context.eastMxmLatency(
                                            context.hidden_slices[byte]),
                                    context.hidden_slices[byte],
                                    reduction * m + mTile * tile
                                        + rowOffset,
                                    segment.rows, 1,
                                    segment.stream_base + byte, "east",
                                    "activation",
                                    context.hemisphereName(hemisphere))
                                .getOutput();
                    }
                    const int64_t unitBase =
                        hemisphere * throughput.mxms_per_hemisphere;
                    down0 = rewriter.create<MxmComputeOp>(ffn.getLoc(),
                        hiddenValue, ffn.getDownWeight0(),
                        context.projection_type, segmentCycle,
                        segment.rows,
                        segmentCycle
                            + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(segment.rows),
                        segment.stream_base, 0, weightBuffer, unitBase,
                        segment.rows, tile, tile);
                    down1 = rewriter.create<MxmComputeOp>(ffn.getLoc(),
                        hiddenValue, ffn.getDownWeight1(),
                        context.projection_type, segmentCycle,
                        segment.rows,
                        segmentCycle
                            + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(segment.rows),
                        segment.stream_base,
                        throughput.mxm_result_streams, weightBuffer,
                        unitBase + 1, segment.rows, tile, tile);
                    rowOffset += segment.rows;
                }

                const int64_t accumulatorBase =
                    context.down_accumulator_base + mTile * tile;
                auto acc0Placement = schedule_placement(rewriter,
                    context.gate_acc_slices, accumulatorBase, tile, 1,
                    context.hemisphereName(hemisphere),
                    "fp32_accumulator");
                auto acc1Placement = schedule_placement(rewriter,
                    context.up_acc_slices, accumulatorBase, tile, 1,
                    context.hemisphereName(hemisphere),
                    "fp32_accumulator");
                const auto emitAccumulator =
                    [&](mlir::Value input,
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
                            rewriter.getNamedAttr(
                                "address", ffn.getResultAddressAttr()),
                            rewriter.getNamedAttr(
                                "placement", placement),
                            rewriter.getNamedAttr("hemisphere",
                                rewriter.getStringAttr(
                                    context.hemisphereName(
                                        hemisphere))),
                            rewriter.getNamedAttr("destination",
                                rewriter.getStringAttr(
                                    block.final_reduction ? "stream"
                                                          : "sram")),
                            rewriter.getNamedAttr("repeat_count",
                                rewriter.getI64IntegerAttr(tile)),
                            rewriter.getNamedAttr("repeat_interval",
                                rewriter.getI64IntegerAttr(1)),
                            rewriter.getNamedAttr("address_stride",
                                rewriter.getI64IntegerAttr(1)),
                        });
                        return llvm::cast<MemAccumulateOp>(
                            rewriter.create(state));
                    };
                auto acc0 = emitAccumulator(down0.getResult(),
                    acc0Placement, computeCycle + gateAccLatency, 0);
                auto acc1 = emitAccumulator(down1.getResult(),
                    acc1Placement, computeCycle + upAccLatency,
                    throughput.mxm_result_streams);
                if (!block.final_reduction) continue;

                const int64_t resultStreamBase =
                    timeline->output_stream_base;
                const int64_t outputAluBase = hemisphere
                    * timeline->vxm_queues_per_hemisphere;
                for (int64_t row = 0; row < tile; ++row) {
                    const int64_t vxmCycle = computeCycle
                        + throughput.accumulator_to_vxm_latency + row;
                    auto cast0 = create_vxm(rewriter, ffn.getLoc(),
                        acc0.getOutput(), acc1.getOutput(),
                        ffn.getResult().getType(), vxmCycle,
                        outputAluBase, "pass", "stream_f32",
                        timeline->first_accumulator_stream, 0,
                        "immediate", 0, 0, "fp16", resultStreamBase, 1,
                        1, context.hemisphereName(hemisphere),
                        context.hemisphereName(hemisphere));
                    auto cast1 = create_vxm(rewriter, ffn.getLoc(),
                        acc0.getOutput(), acc1.getOutput(),
                        ffn.getResult().getType(), vxmCycle,
                        outputAluBase + 1, "pass", "stream_f32",
                        timeline->second_accumulator_stream, 0,
                        "immediate", 0, 0, "fp16",
                        resultStreamBase + 2, 1, 1,
                        context.hemisphereName(hemisphere),
                        context.hemisphereName(hemisphere));
                    for (int64_t byte = 0;
                         byte < throughput.mxm_result_streams; ++byte) {
                        auto placement = schedule_placement(rewriter,
                            {context.result_slices[byte]},
                            outputWave * m + mTile * tile + row, 1, 1,
                            context.hemisphereName(hemisphere),
                            "fp16_pair_planar");
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
                                "both", "fp16_pair_planar"));
                        auto write = rewriter.create<MemWriteOp>(
                            ffn.getLoc(),
                            byte < 2 ? cast0.getResult()
                                     : cast1.getResult(),
                            vxmCycle + 1
                                + context.result_slices[byte]
                                    / target.streams()
                                          .mem_slices_per_register_group,
                            1, resultStreamBase + byte, 1, 0,
                            rewriter.getStringAttr("east"),
                            ffn.getResultAddress(),
                            attributes.getDictionary(
                                rewriter.getContext()),
                            tile);
                        finalValue = write.getOutput();
                    }
                }
            }
        }
    }
    return finalValue;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
