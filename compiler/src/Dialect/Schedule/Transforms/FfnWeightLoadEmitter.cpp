#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {
namespace {

int64_t functionArgumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

} // namespace

MxmLoadOp emitFfnWeightTile(mlir::IRRewriter& rewriter,
    mlir::Location location, stream::RouteOp rawRoute,
    mlir::Type dequantizedType, llvm::ArrayRef<int64_t> weightSlices,
    const target::LPUTargetModel& target, float scale, int64_t startCycle,
    int64_t baseRow, int64_t hemisphere, int64_t localMxm,
    int64_t unit, int64_t weightBuffer, bool localDequant)
{
    const auto& throughput = target.throughput();
    const int64_t duration =
        throughput.mxm_rows / throughput.lanes_per_tile;
    const int64_t encodedStreamBase =
        target.streams().streams_per_direction;
    const auto hemi = hemisphere_name(hemisphere);
    const auto dataFormat = lpu_16bit_data_format(
        llvm::cast<mlir::RankedTensorType>(
            dequantizedType).getElementType());
    mlir::Value readValue;

    if (localDequant) {
        const int64_t streamBase = localMxm
            * throughput.mxm_int8_load_streams_per_cycle;
        for (int64_t stream = 0;
             stream < rawRoute.getStreamCount(); ++stream) {
            const int64_t slice = weightSlices[stream];
            const int64_t latency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, slice)
                                        .value_or(
                                            slice
                                                    / target.streams()
                                                          .mem_slices_per_register_group
                                                + 2);
            auto placement = schedule_placement(rewriter, {slice}, baseRow,
                duration, 1, hemi, "schedule_slice");
            mlir::NamedAttrList attributes(placement);
            attributes.set("binding_placement", rawRoute.getPlacement());
            auto read = rewriter.create<MemReadOp>(location,
                rawRoute.getInput(), startCycle - latency, duration,
                streamBase + stream, 1,
                slice / target.streams().mem_slices_per_register_group + 1,
                rewriter.getStringAttr("east"),
                rewriter.getStringAttr("weight_i8"), rawRoute.getAddress(),
                attributes.getDictionary(rewriter.getContext()),
                duration * throughput.lanes_per_tile);
            readValue = read.getOutput();
        }

        mlir::OperationState dequantState(
            location, MxmDequantOp::getOperationName());
        dequantState.addAttributes({
            rewriter.getNamedAttr(
                "cycle", rewriter.getI64IntegerAttr(startCycle)),
            rewriter.getNamedAttr(
                "unit_id", rewriter.getI64IntegerAttr(unit)),
            rewriter.getNamedAttr(
                "scale", rewriter.getF32FloatAttr(scale)),
            rewriter.getNamedAttr(
                "repeat_count", rewriter.getI64IntegerAttr(duration)),
            rewriter.getNamedAttr(
                "repeat_interval", rewriter.getI64IntegerAttr(1)),
        });
        rewriter.create(dequantState);

        auto load = rewriter.create<MxmLoadOp>(location, readValue,
            startCycle, duration, streamBase,
            throughput.mxm_int8_load_streams_per_cycle, unit,
            weightBuffer);
        load->setAttr("data_format", rewriter.getStringAttr(dataFormat));
        load->setAttr(
            "weight_load_mode", rewriter.getStringAttr("supercell"));
        load->setAttr("weight_input_mode",
            rewriter.getStringAttr("int8_dequant_bf16"));
        return load;
    }

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
            hemi, hemi,
            functionArgumentIndex(rawRoute.getInput())).getResult();
        value = create_vxm(rewriter, location, value, readValue,
            dequantizedType, startCycle + 1, 8 + stream, "cast",
            "alu", stream, 0.0f, "immediate", 0, 0.0f, dataFormat,
            localMxm * throughput.mxm_load_streams_per_cycle + stream * 2,
            duration, 1, hemi, hemi).getResult();
    }

    return rewriter.create<MxmLoadOp>(location, value,
        startCycle + throughput.vxm_weight_to_iw_latency,
        duration, 0, throughput.mxm_load_streams_per_cycle,
        unit, weightBuffer);
}

} // namespace ftlpu::compiler::schedule::ffn_detail
