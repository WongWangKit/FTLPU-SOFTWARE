#include "FfnEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {

MxmLoadOp emitFfnWeightTile(mlir::IRRewriter& rewriter,
    mlir::Location location, stream::RouteOp rawRoute,
    mlir::Type dequantizedType, llvm::ArrayRef<int64_t> weightSlices,
    const target::LPUTargetModel& target, float scale, int64_t startCycle,
    int64_t baseRow, int64_t hemisphere, int64_t localMxm,
    int64_t unit, int64_t weightBuffer)
{
    const auto& throughput = target.throughput();
    const int64_t duration =
        throughput.mxm_rows / throughput.lanes_per_tile;
    const int64_t encodedStreamBase =
        target.streams().streams_per_direction;
    const auto hemi = hemisphere_name(hemisphere);
    mlir::Value readValue;

    for (int64_t stream = 0; stream < rawRoute.getStreamCount(); ++stream) {
        const int64_t slice = weightSlices[stream];
        const int64_t latency =
            slice / target.streams().mem_slices_per_register_group + 2;
        auto placement = schedule_placement(rewriter, {slice}, baseRow,
            duration, 1, hemi, "schedule_slice");
        mlir::NamedAttrList attributes(placement);
        attributes.set("binding_placement", rawRoute.getPlacement());
        auto read = rewriter.create<MemReadOp>(location, rawRoute.getInput(),
            startCycle - latency, duration, stream, 1,
            slice / target.streams().mem_slices_per_register_group + 1,
            rewriter.getStringAttr("west"),
            rewriter.getStringAttr("weight_i8"), rawRoute.getAddress(),
            attributes.getDictionary(rewriter.getContext()),
            duration * throughput.mxm_rows);
        readValue = read.getOutput();
    }

    mlir::Value value = readValue;
    for (int64_t stream = 0; stream < rawRoute.getStreamCount(); ++stream) {
        value = create_vxm(rewriter, location, readValue, readValue,
            dequantizedType, startCycle, stream, "multiply",
            "stream_i8", encodedStreamBase + stream, 0.0f,
            "immediate", 0, scale, "fp32", -1, duration, 1,
            hemi, hemi).getResult();
        value = create_vxm(rewriter, location, value, readValue,
            dequantizedType, startCycle + 1, 8 + stream, "cast",
            "alu", stream, 0.0f, "immediate", 0, 0.0f, "fp16",
            localMxm * throughput.mxm_load_streams_per_cycle + stream * 2,
            duration, 1, hemi, hemi).getResult();
    }

    return rewriter.create<MxmLoadOp>(location, value,
        startCycle + throughput.vxm_weight_to_iw_latency,
        duration, 0, throughput.mxm_load_streams_per_cycle,
        unit, weightBuffer);
}

} // namespace ftlpu::compiler::schedule::ffn_detail
