#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {

mlir::LogicalResult lower_swiglu(
    tensor::SwigluOp op, LoweringContext& context)
{
    const target::LPUTargetModel& target = context.target;
    stream::StreamAllocator& allocator = context.allocator;
    mlir::IRRewriter& rewriter = context.rewriter;
    int64_t& stage = context.stage;

    const auto input_slice = get_mem_slice(op.getInputAddress());
    const auto gate_slice = get_mem_slice(op.getGateWeightAddress());
    const auto up_slice = get_mem_slice(op.getUpWeightAddress());
    const auto result_slice = get_mem_slice(op.getResultAddress());
    if (mlir::failed(input_slice) || mlir::failed(gate_slice)
        || mlir::failed(up_slice) || mlir::failed(result_slice)) {
        op.emitError("requires valid MEM addresses before stream lowering");
        return mlir::failure();
    }
    const auto gate_binding = allocator.allocate(target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
        *gate_slice, stage, stage + 2);
    const auto up_binding = allocator.allocate(target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
        *up_slice, stage, stage + 2);
    const auto activation_binding = allocator.allocate(target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmActivation, target::StreamDirection::East,
        *input_slice, stage + 2, stage + 4);
    const auto output_binding = allocator.allocate(target::StreamEndpoint::VxmResult,
        target::StreamEndpoint::Mem, target::StreamDirection::East,
        *result_slice, stage + 4, stage + 6);
    const auto gate_latency = target.transport_latency(target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight, target::StreamDirection::East, *gate_slice);
    const auto up_latency = target.transport_latency(target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight, target::StreamDirection::East, *up_slice);
    const auto activation_latency = target.transport_latency(target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmActivation, target::StreamDirection::East, *input_slice);
    const auto output_latency = target.transport_latency(target::StreamEndpoint::VxmResult,
        target::StreamEndpoint::Mem, target::StreamDirection::East, *result_slice);
    if (mlir::failed(gate_binding) || mlir::failed(up_binding)
        || mlir::failed(activation_binding) || mlir::failed(output_binding)
        || !gate_latency || !up_latency || !activation_latency || !output_latency) {
        op.emitError("cannot allocate the dual-MXM/VXM stream topology");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op);
    auto route = [&](mlir::Value value, const stream::StreamBinding& binding,
                     target::StreamEndpoint destination, int64_t unit,
                     mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
                     int64_t bytes, int64_t latency) {
        return rewriter.create<stream::RouteOp>(op.getLoc(), value,
            binding.stream_base, binding.stream_count, binding.register_id,
            rewriter.getStringAttr("east"), rewriter.getStringAttr("MEM"),
            rewriter.getStringAttr(target::LPUTargetModel::endpoint_name(destination)),
            -1, unit, address, placement, bytes, latency);
    };
    auto gate_route = route(op.getGateWeight(), *gate_binding,
        target::StreamEndpoint::MxmWeight, 0, op.getGateWeightAddress(),
        op.getGateWeightPlacement(), op.getGateWeightBytes(), *gate_latency);
    auto up_route = route(op.getUpWeight(), *up_binding,
        target::StreamEndpoint::MxmWeight, 1, op.getUpWeightAddress(),
        op.getUpWeightPlacement(), op.getUpWeightBytes(), *up_latency);
    auto activation_route = route(op.getInput(), *activation_binding,
        target::StreamEndpoint::MxmActivation, 0, op.getInputAddress(),
        op.getInputPlacement(), op.getInputBytes(), *activation_latency);
    mlir::OperationState state(op.getLoc(), stream::SwigluOp::getOperationName());
    state.addOperands({activation_route.getOutput(), gate_route.getOutput(), up_route.getOutput()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        rewriter.getNamedAttr("m", op.getMAttr()), rewriter.getNamedAttr("n", op.getNAttr()),
        rewriter.getNamedAttr("k", op.getKAttr()),
        rewriter.getNamedAttr("gate_unit_id", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("up_unit_id", rewriter.getI64IntegerAttr(1)),
        rewriter.getNamedAttr("gate_weight_buffer", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("up_weight_buffer", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("gate_output_stream_base", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("up_output_stream_base", rewriter.getI64IntegerAttr(4)),
        rewriter.getNamedAttr("vxm_output_stream", rewriter.getI64IntegerAttr(31)),
        rewriter.getNamedAttr("output_register_id", rewriter.getI64IntegerAttr(output_binding->register_id)),
        rewriter.getNamedAttr("output_transport_latency", rewriter.getI64IntegerAttr(*output_latency)),
        rewriter.getNamedAttr("gate_scale", op.getGateScaleAttr()),
        rewriter.getNamedAttr("up_scale", op.getUpScaleAttr()),
        rewriter.getNamedAttr("output_scale", op.getOutputScaleAttr()),
        rewriter.getNamedAttr("output_zero_point", op.getOutputZeroPointAttr()),
        rewriter.getNamedAttr("result_address", op.getResultAddressAttr()),
        rewriter.getNamedAttr("result_placement", op.getResultPlacementAttr()),
        rewriter.getNamedAttr("result_bytes", op.getResultBytesAttr()),
    });
    auto lowered = llvm::cast<stream::SwigluOp>(rewriter.create(state));
    rewriter.replaceOp(op, lowered.getResult());
    stage += 6;
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
