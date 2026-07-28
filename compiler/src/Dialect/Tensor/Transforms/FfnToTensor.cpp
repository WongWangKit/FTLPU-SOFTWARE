#include "KernelToTensorLowering.hpp"

#include "llvm/ADT/STLExtras.h"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target, EastMemoryAllocator& allocator,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter)
{
    kernel::MatmulOp op = graph.output;
    const int64_t m = graph.gate.getM();
    const int64_t k = graph.gate.getK();
    const int64_t hidden = graph.gate.getN();
    const int64_t n = graph.output.getN();
    const mlir::Value input_value = graph.gate.getLhs();
    const mlir::Value gate_weight = graph.gate.getRhs();
    const mlir::Value up_weight = graph.up.getRhs();
    const mlir::Value down_weight = graph.output.getRhs();
    const bool w8a16 = is_w8a16_ffn(graph, target);
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    llvm::SmallVector<int64_t> projection_weight_slices;
    for (int64_t index = 0;
         index < memory.w8a16_weight_slice_count; ++index)
        projection_weight_slices.push_back(
            index * memory.w8a16_weight_slice_stride);
    const llvm::SmallVector<int64_t> down_weight_slices =
        projection_weight_slices;
    llvm::SmallVector<int64_t> activation_slices;
    llvm::SmallVector<int64_t> hidden_slices;
    llvm::SmallVector<int64_t> result_slices;
    for (int64_t index = 0; index < throughput.mxm_activation_streams; ++index) {
        activation_slices.push_back(memory.w8a16_activation_slice_base + index);
        hidden_slices.push_back(memory.w8a16_hidden_slice_base
            + index + (index == 3 ? 5 : 0));
        result_slices.push_back(memory.w8a16_result_slice_base + index);
    }
    const int64_t hidden_pass_bytes = w8a16
        ? m * (hidden / target.memory().hemispheres) * 2
        : m * 320;
    const int64_t gate_rows = w8a16
        ? k * hidden
            / (memory.hemispheres * memory.w8a16_weight_slice_count
                * throughput.tile_rows * throughput.lanes_per_tile)
        : 0;
    const int64_t down_rows = w8a16
        ? hidden * n
            / (memory.w8a16_weight_slice_count
                * throughput.tile_rows * throughput.lanes_per_tile)
        : 0;
    const bool largeSramProfile =
        memory.banks_per_slice * memory.words_per_bank >= 17000;
    const int64_t ffnWeightBase = largeSramProfile ? 10000 : 0;
    auto input = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Activation,
            activation_slices, 0, m * k / throughput.mxm_rows,
            m * k * 2))
        : allocate_value(input_value, PlacementKind::Activation);
    auto gate = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
            projection_weight_slices, ffnWeightBase,
            gate_rows, k * hidden))
        : allocate_value(gate_weight, PlacementKind::Weight);
    auto up = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
            projection_weight_slices, ffnWeightBase + gate_rows,
            gate_rows, k * hidden))
        : allocate_value(up_weight, PlacementKind::Weight);
    auto down = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
            down_weight_slices, ffnWeightBase + 2 * gate_rows,
            down_rows, hidden * n))
        : allocate_value(down_weight, PlacementKind::Weight);
    auto hidden0 = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::VxmResult,
            hidden_slices, 0,
            m * (hidden / memory.hemispheres)
                / throughput.mxm_rows,
            hidden_pass_bytes))
        : allocator.allocate(PlacementKind::VxmResult, hidden_pass_bytes);
    auto hidden1 = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::VxmResult1,
            hidden_slices, 0,
            m * (hidden / memory.hemispheres)
                / throughput.mxm_rows,
            hidden_pass_bytes))
        : allocator.allocate(PlacementKind::VxmResult1, hidden_pass_bytes);
    const auto result_bytes =
        get_static_tensor_bytes(graph.output.getResult().getType());
    const auto result = w8a16 && mlir::succeeded(result_bytes)
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::FinalResult,
            result_slices, 0,
            m * n
                / (throughput.mxm_rows * throughput.mxms_per_hemisphere),
            *result_bytes))
        : mlir::succeeded(result_bytes)
        ? allocator.allocate(PlacementKind::FinalResult, *result_bytes)
        : mlir::FailureOr<Allocation>(mlir::failure());
    if (mlir::failed(input) || mlir::failed(gate) || mlir::failed(up)
        || mlir::failed(down) || mlir::failed(hidden0) || mlir::failed(hidden1)
        || mlir::failed(result)) {
        op.emitError("cannot allocate complete FFN storage");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op);
    const auto input_placement = w8a16
        ? make_profile_placement(rewriter, *input, "fp16_mxm_activation_planar", "both")
        : make_placement_attr(rewriter, *input);
    const auto gate_placement = w8a16
        ? make_profile_placement(rewriter, *gate, "w8a16_mxm_weight_striped", "both")
        : make_placement_attr(rewriter, *gate);
    const auto up_placement = w8a16
        ? make_profile_placement(rewriter, *up, "w8a16_mxm_weight_striped", "both")
        : make_placement_attr(rewriter, *up);
    const auto down_placement = w8a16
        ? make_profile_placement(rewriter, *down,
            "w8a16_mxm_weight_wave_striped", "both")
        : make_placement_attr(rewriter, *down);
    const auto hidden0_placement = w8a16
        ? make_profile_placement(rewriter, *hidden0,
            "fp16_mxm_activation_planar", "west")
        : make_placement_attr(rewriter, *hidden0);
    const auto hidden1_placement = w8a16
        ? make_profile_placement(rewriter, *hidden1,
            "fp16_mxm_activation_planar", "east")
        : make_placement_attr(rewriter, *hidden1);
    const auto result_placement = w8a16
        ? make_profile_placement(rewriter, *result, "fp16_pair_planar", "both")
        : make_placement_attr(rewriter, *result);

    auto inheritedInputAllocations =
        get_value_task_allocations(input_value);
    const auto input_allocations =
        mlir::succeeded(inheritedInputAllocations)
        ? *inheritedInputAllocations
        : make_task_allocations(rewriter,
            {make_task_allocation(rewriter, *input, input_placement)});
    const auto gate_allocations = make_task_allocations(rewriter,
        {make_task_allocation(rewriter, *gate, gate_placement)});
    const auto up_allocations = make_task_allocations(rewriter,
        {make_task_allocation(rewriter, *up, up_placement)});
    const auto down_allocations = make_task_allocations(rewriter,
        {make_task_allocation(rewriter, *down, down_placement)});
    const auto hidden_allocations = make_task_allocations(rewriter, {
        make_task_allocation(rewriter, *hidden0, hidden0_placement),
        make_task_allocation(rewriter, *hidden1, hidden1_placement),
    });
    const auto result_allocations = make_task_allocations(rewriter,
        {make_task_allocation(rewriter, *result, result_placement)});
    const auto transient = rewriter.getArrayAttr({});
    const auto empty_config = rewriter.getDictionaryAttr({});
    const auto unit_scale = rewriter.getF32FloatAttr(1.0f);
    const auto gate_scale = graph.gate.getRhsScaleAttr();
    const auto up_scale = graph.up.getRhsScaleAttr();
    const auto down_scale = graph.output.getRhsScaleAttr();
    const auto zero_point = rewriter.getI64IntegerAttr(0);
    const mlir::Type projection_element_type = w8a16
        ? mlir::Type(rewriter.getF32Type())
        : mlir::Type(rewriter.getI32Type());
    const mlir::Type hidden_element_type = w8a16
        ? mlir::Type(rewriter.getF16Type())
        : mlir::Type(rewriter.getI8Type());
    const auto projection_type = mlir::RankedTensorType::get(
        {m, hidden},
        projection_element_type);
    const auto hidden_type = mlir::RankedTensorType::get(
        {m, hidden},
        hidden_element_type);

    auto create_matmul_task = [&](mlir::Value lhs, mlir::Value rhs,
                                  mlir::Type result_type, int64_t m,
                                  int64_t n, int64_t k,
                                  mlir::ArrayAttr lhs_plan,
                                  mlir::ArrayAttr rhs_plan,
                                  mlir::ArrayAttr result_plan,
                                  mlir::DictionaryAttr config) {
        mlir::OperationState task_state(op.getLoc(),
            tensor::MatmulTaskOp::getOperationName());
        task_state.addOperands({lhs, rhs});
        task_state.addTypes(result_type);
        task_state.addAttributes({
            rewriter.getNamedAttr("m", rewriter.getI64IntegerAttr(m)),
            rewriter.getNamedAttr("n", rewriter.getI64IntegerAttr(n)),
            rewriter.getNamedAttr("k", rewriter.getI64IntegerAttr(k)),
            rewriter.getNamedAttr("lhs_allocations", lhs_plan),
            rewriter.getNamedAttr("rhs_allocations", rhs_plan),
            rewriter.getNamedAttr("result_allocations", result_plan),
            rewriter.getNamedAttr("config", config),
        });
        return llvm::cast<tensor::MatmulTaskOp>(rewriter.create(task_state));
    };

    auto gate_task = create_matmul_task(input_value, gate_weight,
        projection_type, m, hidden, k,
        input_allocations, gate_allocations, transient,
        rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("rhs_scale", gate_scale),
        }));
    auto up_task = create_matmul_task(input_value, up_weight,
        projection_type, m, hidden, k,
        input_allocations, up_allocations, transient,
        rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("rhs_scale", up_scale),
        }));

    mlir::OperationState swish_state(op.getLoc(),
        tensor::SwishTaskOp::getOperationName());
    swish_state.addOperands(gate_task.getResult());
    swish_state.addTypes(projection_type);
    swish_state.addAttributes({
        rewriter.getNamedAttr("result_allocations", transient),
        rewriter.getNamedAttr("config", empty_config),
    });
    auto swish = llvm::cast<tensor::SwishTaskOp>(rewriter.create(swish_state));

    mlir::OperationState multiply_state(op.getLoc(),
        tensor::ElementwiseTaskOp::getOperationName());
    multiply_state.addOperands({swish.getResult(), up_task.getResult()});
    multiply_state.addTypes(hidden_type);
    multiply_state.addAttributes({
        rewriter.getNamedAttr("kind", rewriter.getStringAttr("multiply")),
        rewriter.getNamedAttr("lhs_allocations", transient),
        rewriter.getNamedAttr("rhs_allocations", transient),
        rewriter.getNamedAttr("result_allocations", hidden_allocations),
        rewriter.getNamedAttr("config", rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("output_scale", unit_scale),
            rewriter.getNamedAttr("output_zero_point", zero_point),
        })),
    });
    auto multiply =
        llvm::cast<tensor::ElementwiseTaskOp>(rewriter.create(multiply_state));

    auto down_task = create_matmul_task(multiply.getResult(),
        down_weight, graph.output.getResult().getType(), m, n,
        hidden, hidden_allocations, down_allocations,
        result_allocations, rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("lhs_scale", unit_scale),
            rewriter.getNamedAttr("rhs_scale", down_scale),
            rewriter.getNamedAttr("output_scale", unit_scale),
            rewriter.getNamedAttr("output_zero_point", zero_point),
        }));
    mlir::Operation* output_operation = graph.output.getOperation();
    rewriter.replaceOp(graph.output, down_task.getResult());
    for (mlir::Operation* operation : llvm::reverse(graph.operations)) {
        if (operation != output_operation && operation->use_empty())
            rewriter.eraseOp(operation);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
