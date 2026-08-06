#include "TensorToStreamLowering.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include <optional>

namespace ftlpu::compiler::stream_lowering {

mlir::LogicalResult lower_ffn(
    FfnTaskGraph& op, LoweringContext& context)
{
    const target::LPUTargetModel& target = context.target;
    stream::StreamAllocator& allocator = context.allocator;
    mlir::IRRewriter& rewriter = context.rewriter;
    int64_t& stage = context.stage;

    const auto element_type = [](mlir::Value value) {
        return llvm::cast<mlir::RankedTensorType>(value.getType())
            .getElementType();
    };
    const bool w8a16 = is_lpu_16bit_float(
                           element_type(op.getInput()))
        && element_type(op.getGateWeight()).isInteger(8)
        && element_type(op.getUpWeight()).isInteger(8)
        && element_type(op.getDownWeight()).isInteger(8)
        && is_lpu_16bit_float(element_type(op.getResult()))
        && target.supports_w8a16_ffn_shape(
            op.getM(), op.getK(), op.getHidden(), op.getN());
    auto executionPolicy =
        target::mxm_execution_policy_from_operation(
            op.down.getOperation());
    if (mlir::failed(executionPolicy)) {
        op.emitError("invalid MXM execution policy");
        return mlir::failure();
    }
    bool block8Ffn = false;
    if (w8a16) {
        auto strategy = target::plan_mxm_execution_strategy(
            {static_cast<int64_t>(op.getM()),
                static_cast<int64_t>(op.getN()),
                static_cast<int64_t>(op.getHidden()),
                element_type(op.getInput()).isBF16(), true, true, true},
            target, *executionPolicy);
        if (mlir::failed(strategy)) {
            op.emitError("cannot select an MXM execution strategy for FFN");
            return mlir::failure();
        }
        block8Ffn = strategy->uses_block8();
    }
    const auto input_slice = get_mem_slice(op.getInputAddress());
    const auto gate_slice = get_mem_slice(op.getGateWeightAddress());
    const auto up_slice = get_mem_slice(op.getUpWeightAddress());
    const auto down_slice = get_mem_slice(op.getDownWeightAddress());
    const auto hidden0_slice = get_mem_slice(op.getHidden0Address());
    const auto hidden1_slice = get_mem_slice(op.getHidden1Address());
    const auto result_slice = get_mem_slice(op.getResultAddress());
    if (mlir::failed(input_slice) || mlir::failed(gate_slice)
        || mlir::failed(up_slice) || mlir::failed(down_slice)
        || mlir::failed(hidden0_slice) || mlir::failed(hidden1_slice)
        || mlir::failed(result_slice)) {
        op.emitError("requires valid complete-FFN MEM addresses");
        return mlir::failure();
    }
    auto allocate = [&](target::StreamEndpoint destination, int64_t slice,
                        int64_t begin, int64_t end,
                        std::optional<int64_t> streamCount = std::nullopt) {
        return allocator.allocate(target::StreamEndpoint::Mem, destination,
            target::StreamDirection::East, slice, begin, end,
            streamCount);
    };
    const auto projectionWeightEndpoint =
        w8a16 && !block8Ffn ? target::StreamEndpoint::VxmInput
              : target::StreamEndpoint::MxmWeight;
    const auto downWeightEndpoint = block8Ffn
        ? target::StreamEndpoint::MxmWeight
        : projectionWeightEndpoint;
    const auto gate_binding = allocate(projectionWeightEndpoint,
        *gate_slice, stage, stage + 2);
    const auto up_binding = allocate(projectionWeightEndpoint,
        *up_slice, stage, stage + 2);
    const auto activation_binding = allocate(target::StreamEndpoint::MxmActivation,
        *input_slice, stage + (w8a16 ? 4 : 2), stage + (w8a16 ? 8 : 6),
        block8Ffn ? std::optional<int64_t>(16) : std::nullopt);
    const auto down0_binding = allocate(downWeightEndpoint,
        *down_slice, stage + 7, stage + 9,
        block8Ffn ? std::optional<int64_t>(16) : std::nullopt);
    auto down1_binding = down0_binding;
    if (!block8Ffn)
        down1_binding = allocate(downWeightEndpoint,
            *down_slice, stage + 7, stage + 9);
    const auto hidden0_binding = allocate(
        target::StreamEndpoint::MxmActivation,
        *hidden0_slice, stage + 11, stage + 13,
        block8Ffn ? std::optional<int64_t>(16) : std::nullopt);
    auto hidden1_binding = hidden0_binding;
    if (!block8Ffn)
        hidden1_binding = allocate(
            target::StreamEndpoint::MxmActivation,
            *hidden1_slice, stage + 11, stage + 13);
    if (mlir::failed(gate_binding) || mlir::failed(up_binding)
        || mlir::failed(activation_binding) || mlir::failed(down0_binding)
        || mlir::failed(down1_binding)
        || mlir::failed(hidden0_binding)
        || mlir::failed(hidden1_binding)) {
        auto diagnostic = op.emitError("cannot allocate complete-FFN stream ranges:");
        if (mlir::failed(gate_binding)) diagnostic << " gate";
        if (mlir::failed(up_binding)) diagnostic << " up";
        if (mlir::failed(activation_binding)) diagnostic << " activation";
        if (mlir::failed(down0_binding)) diagnostic << " down0";
        if (mlir::failed(down1_binding)) diagnostic << " down1";
        if (mlir::failed(hidden0_binding)) diagnostic << " hidden0";
        if (mlir::failed(hidden1_binding)) diagnostic << " hidden1";
        return mlir::failure();
    }
    auto latency = [&](target::StreamEndpoint endpoint, int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem, endpoint,
            target::StreamDirection::East, slice);
    };
    const auto gate_latency =
        latency(projectionWeightEndpoint, *gate_slice);
    const auto up_latency =
        latency(projectionWeightEndpoint, *up_slice);
    const auto input_latency = latency(target::StreamEndpoint::MxmActivation, *input_slice);
    const auto down_latency = latency(downWeightEndpoint, *down_slice);
    const auto hidden0_latency = latency(
        target::StreamEndpoint::MxmActivation, *hidden0_slice);
    const auto hidden1_latency = latency(
        target::StreamEndpoint::MxmActivation, *hidden1_slice);
    if (!gate_latency || !up_latency || !input_latency || !down_latency
        || !hidden0_latency || !hidden1_latency) {
        op.emitError("complete-FFN route is unsupported by the target");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op.down);
    auto route = [&](mlir::Value value, const stream::StreamBinding& binding,
                     target::StreamEndpoint destination, int64_t unit,
                     mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
                     int64_t bytes, int64_t route_latency) {
        return rewriter.create<stream::RouteOp>(op.getLoc(), value,
            binding.stream_base, binding.stream_count, binding.register_id,
            rewriter.getStringAttr("east"), rewriter.getStringAttr("MEM"),
            rewriter.getStringAttr(target::LPUTargetModel::endpoint_name(destination)),
            -1, unit, address, placement, bytes, route_latency);
    };
    auto gate_route = route(op.getGateWeight(), *gate_binding,
        projectionWeightEndpoint, 0, op.getGateWeightAddress(),
        op.getGateWeightPlacement(), op.getGateWeightBytes(), *gate_latency);
    auto up_route = route(op.getUpWeight(), *up_binding,
        projectionWeightEndpoint, block8Ffn || !w8a16 ? 1 : 0,
        op.getUpWeightAddress(),
        op.getUpWeightPlacement(), op.getUpWeightBytes(), *up_latency);
    auto activation_route = route(op.getInput(), *activation_binding,
        target::StreamEndpoint::MxmActivation, 0, op.getInputAddress(),
        op.getInputPlacement(), op.getInputBytes(), *input_latency);
    auto down0_route = route(op.getDownWeight(), *down0_binding,
        downWeightEndpoint, 0, op.getDownWeightAddress(),
        op.getDownWeightPlacement(), op.getDownWeightBytes(), *down_latency);
    auto down1_route = route(op.getDownWeight(), *down1_binding,
        downWeightEndpoint, block8Ffn || !w8a16 ? 1 : 0,
        op.getDownWeightAddress(),
        op.getDownWeightPlacement(), op.getDownWeightBytes(), *down_latency);
    if (w8a16 && !block8Ffn) {
        auto connect_dequant = [&](stream::RouteOp raw, int64_t slice,
                                   int64_t unit, int64_t begin, float scale) {
            auto input_type = llvm::cast<mlir::RankedTensorType>(raw.getOutput().getType());
            auto float16_type = mlir::RankedTensorType::get(
                input_type.getShape(), element_type(op.getInput()));
            mlir::OperationState dequant_state(op.getLoc(),
                stream::DequantizeOp::getOperationName());
            dequant_state.addOperands(raw.getOutput());
            dequant_state.addTypes(float16_type);
            dequant_state.addAttributes({
                rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
                rewriter.getNamedAttr("input_stream_base", rewriter.getI64IntegerAttr(raw.getStreamBase())),
                rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("input_hemisphere", rewriter.getStringAttr("east")),
                rewriter.getNamedAttr("output_hemisphere", rewriter.getStringAttr("east")),
            });
            auto dequant = llvm::cast<stream::DequantizeOp>(rewriter.create(dequant_state));
            auto binding = allocator.allocate(target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
                slice, begin, begin + 2);
            if (mlir::failed(binding)) return stream::RouteOp{};
            const auto latency = target.transport_latency(target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, slice);
            return rewriter.create<stream::RouteOp>(op.getLoc(), dequant.getResult(),
                binding->stream_base, binding->stream_count, binding->register_id,
                rewriter.getStringAttr("east"), rewriter.getStringAttr("VXM.result"),
                rewriter.getStringAttr("MXM.weight"), 0, unit, raw.getAddress(),
                raw.getPlacement(), raw.getBytes(), *latency);
        };
        gate_route = connect_dequant(gate_route, *gate_slice, 0, stage + 2,
            op.getGateScale().convertToFloat());
        up_route = connect_dequant(up_route, *up_slice, 1, stage + 2,
            op.getUpScale().convertToFloat());
        down0_route = connect_dequant(down0_route,
            *down_slice, 0, stage + 9,
            op.getDownRhsScale().convertToFloat());
        down1_route = connect_dequant(down1_route,
            *down_slice, 1, stage + 9,
            op.getDownRhsScale().convertToFloat());
        if (!gate_route || !up_route || !down0_route || !down1_route) {
            op.emitError("cannot allocate W8A16 VXM-to-MXM streams");
            return mlir::failure();
        }
    }
    const auto empty_allocations = rewriter.getArrayAttr({});
    const auto i64_array = [&](std::initializer_list<int64_t> values) {
        llvm::SmallVector<mlir::Attribute> attributes;
        for (int64_t value : values)
            attributes.push_back(rewriter.getI64IntegerAttr(value));
        return rewriter.getArrayAttr(attributes);
    };
    const auto allocation = [&](mlir::DictionaryAttr address,
                                mlir::DictionaryAttr placement,
                                int64_t bytes) {
        return rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("address", address),
            rewriter.getNamedAttr("placement", placement),
            rewriter.getNamedAttr(
                "bytes", rewriter.getI64IntegerAttr(bytes)),
        });
    };
    const auto hidden_allocations = rewriter.getArrayAttr({
        allocation(op.getHidden0Address(), op.getHidden0Placement(),
            op.getHiddenPassBytes()),
        allocation(op.getHidden1Address(), op.getHidden1Placement(),
            op.getHiddenPassBytes()),
    });
    const auto result_allocations = rewriter.getArrayAttr({
        allocation(op.getResultAddress(), op.getResultPlacement(),
            op.getResultBytes()),
    });
    const auto projection_type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(op.getM()),
            static_cast<int64_t>(op.getHidden())},
        rewriter.getF32Type());
    const auto hidden_type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(op.getM()),
            static_cast<int64_t>(op.getHidden())},
        w8a16 ? element_type(op.getInput())
              : mlir::Type(rewriter.getI8Type()));
    const auto down_partial_type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(op.getM()),
            static_cast<int64_t>(op.getN())},
        rewriter.getF32Type());
    const int64_t result_stream_count =
        target.throughput().mxm_result_streams;
    const int64_t second_result_stream = result_stream_count;

    auto create_matmul_task =
        [&](mlir::ValueRange lhs, mlir::ValueRange rhs,
            mlir::Type result_type, int64_t m, int64_t n, int64_t k,
            mlir::ArrayAttr unit_ids, mlir::ArrayAttr buffers,
            mlir::ArrayAttr stream_bases,
            mlir::ArrayAttr stream_counts,
            mlir::ArrayAttr result_plan,
            mlir::DictionaryAttr config) {
            mlir::OperationState task_state(
                op.getLoc(), stream::MatmulTaskOp::getOperationName());
            task_state.addOperands(lhs);
            task_state.addOperands(rhs);
            task_state.addTypes(result_type);
            task_state.addAttributes({
                rewriter.getNamedAttr("m",
                    rewriter.getI64IntegerAttr(m)),
                rewriter.getNamedAttr("n",
                    rewriter.getI64IntegerAttr(n)),
                rewriter.getNamedAttr("k",
                    rewriter.getI64IntegerAttr(k)),
                rewriter.getNamedAttr("unit_ids", unit_ids),
                rewriter.getNamedAttr("weight_buffers", buffers),
                rewriter.getNamedAttr(
                    "result_stream_bases", stream_bases),
                rewriter.getNamedAttr(
                    "result_stream_counts", stream_counts),
                rewriter.getNamedAttr(
                    "result_allocations", result_plan),
                rewriter.getNamedAttr("config", config),
                rewriter.getNamedAttr("operandSegmentSizes",
                    rewriter.getDenseI32ArrayAttr(
                        {static_cast<int32_t>(lhs.size()),
                            static_cast<int32_t>(rhs.size())})),
            });
            return llvm::cast<stream::MatmulTaskOp>(
                rewriter.create(task_state));
        };

    auto gate_task = create_matmul_task(
        mlir::ValueRange{activation_route.getOutput()},
        mlir::ValueRange{gate_route.getOutput()}, projection_type,
        op.getM(), op.getHidden(), op.getK(), i64_array({0}),
        i64_array({0}), i64_array({0}),
        i64_array({result_stream_count}),
        empty_allocations, rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "rhs_scale", op.getGateScaleAttr()),
        }));
    auto up_task = create_matmul_task(
        mlir::ValueRange{activation_route.getOutput()},
        mlir::ValueRange{up_route.getOutput()}, projection_type,
        op.getM(), op.getHidden(), op.getK(), i64_array({1}),
        i64_array({0}), i64_array({second_result_stream}),
        i64_array({result_stream_count}),
        empty_allocations, rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "rhs_scale", op.getUpScaleAttr()),
        }));

    mlir::OperationState swish_state(
        op.getLoc(), stream::SwishTaskOp::getOperationName());
    swish_state.addOperands(gate_task.getResult());
    swish_state.addTypes(projection_type);
    swish_state.addAttributes({
        rewriter.getNamedAttr(
            "input_stream_base", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr(
            "output_stream_base", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr(
            "config", rewriter.getDictionaryAttr({})),
    });
    auto swish =
        llvm::cast<stream::SwishTaskOp>(rewriter.create(swish_state));

    mlir::OperationState multiply_state(
        op.getLoc(), stream::ElementwiseTaskOp::getOperationName());
    multiply_state.addOperands(
        {swish.getResult(), up_task.getResult()});
    multiply_state.addTypes(hidden_type);
    multiply_state.addAttributes({
        rewriter.getNamedAttr(
            "kind", rewriter.getStringAttr("multiply")),
        rewriter.getNamedAttr(
            "input_stream_bases",
            i64_array({0, second_result_stream})),
        rewriter.getNamedAttr(
            "output_stream_base", rewriter.getI64IntegerAttr(
                w8a16 ? 0
                      : target.streams().streams_per_direction - 1)),
        rewriter.getNamedAttr(
            "lhs_allocations", empty_allocations),
        rewriter.getNamedAttr(
            "rhs_allocations", empty_allocations),
        rewriter.getNamedAttr(
            "result_allocations", hidden_allocations),
        rewriter.getNamedAttr("config",
            rewriter.getDictionaryAttr({
                rewriter.getNamedAttr(
                    "output_scale", op.getHiddenScaleAttr()),
                rewriter.getNamedAttr("output_zero_point",
                    op.getHiddenZeroPointAttr()),
            })),
    });
    auto multiply = llvm::cast<stream::ElementwiseTaskOp>(
        rewriter.create(multiply_state));

    auto hidden0_route = route(multiply.getResult(), *hidden0_binding,
        target::StreamEndpoint::MxmActivation, 0,
        op.getHidden0Address(), op.getHidden0Placement(),
        op.getHiddenPassBytes(), *hidden0_latency);
    auto hidden1_route = route(multiply.getResult(), *hidden1_binding,
        target::StreamEndpoint::MxmActivation, 1,
        op.getHidden1Address(), op.getHidden1Placement(),
        op.getHiddenPassBytes(), *hidden1_latency);
    auto down0_task = create_matmul_task(
        mlir::ValueRange{hidden0_route.getOutput()},
        mlir::ValueRange{down0_route.getOutput()}, down_partial_type,
        op.getM(), op.getN(), op.getHidden(), i64_array({0}),
        i64_array({0}), i64_array({0}),
        i64_array({result_stream_count}),
        empty_allocations, rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "lhs_scale", op.getDownLhsScaleAttr()),
            rewriter.getNamedAttr(
                "rhs_scale", op.getDownRhsScaleAttr()),
        }));
    auto down1_task = create_matmul_task(
        mlir::ValueRange{hidden1_route.getOutput()},
        mlir::ValueRange{down1_route.getOutput()}, down_partial_type,
        op.getM(), op.getN(), op.getHidden(), i64_array({1}),
        i64_array({0}), i64_array({second_result_stream}),
        i64_array({result_stream_count}),
        empty_allocations, rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "lhs_scale", op.getDownLhsScaleAttr()),
            rewriter.getNamedAttr(
                "rhs_scale", op.getDownRhsScaleAttr()),
        }));

    mlir::OperationState add_state(
        op.getLoc(), stream::ElementwiseTaskOp::getOperationName());
    add_state.addOperands(
        {down0_task.getResult(), down1_task.getResult()});
    add_state.addTypes(op.getResult().getType());
    add_state.addAttributes({
        rewriter.getNamedAttr(
            "kind", rewriter.getStringAttr("add_quant")),
        rewriter.getNamedAttr(
            "input_stream_bases",
            i64_array({0, second_result_stream})),
        rewriter.getNamedAttr(
            "output_stream_base", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr(
            "lhs_allocations", empty_allocations),
        rewriter.getNamedAttr(
            "rhs_allocations", empty_allocations),
        rewriter.getNamedAttr(
            "result_allocations", result_allocations),
        rewriter.getNamedAttr("config",
            rewriter.getDictionaryAttr({
                rewriter.getNamedAttr(
                    "output_scale", op.getOutputScaleAttr()),
                rewriter.getNamedAttr("output_zero_point",
                    op.getOutputZeroPointAttr()),
            })),
    });
    auto add = llvm::cast<stream::ElementwiseTaskOp>(
        rewriter.create(add_state));
    rewriter.replaceOp(op.down, add.getResult());
    rewriter.eraseOp(op.multiply);
    rewriter.eraseOp(op.swish);
    rewriter.eraseOp(op.up);
    rewriter.eraseOp(op.gate);
    stage += 14;
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
