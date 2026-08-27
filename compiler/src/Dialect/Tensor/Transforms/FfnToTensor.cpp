#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Dialect/Tensor/Analysis/ffn_weight_tile_plan.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include "llvm/ADT/STLExtras.h"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target, EastMemoryAllocator& allocator,
    AllocateValueFn allocate_value, int64_t weight_bank,
    mlir::IRRewriter& rewriter)
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
    auto executionPolicy =
        target::mxm_execution_policy_from_operation(op);
    if (mlir::failed(executionPolicy)) {
        op.emitError("invalid MXM execution policy");
        return mlir::failure();
    }
    if (w8a16) {
        auto strategy = target::plan_mxm_execution_strategy(
            {m, n, hidden,
                llvm::cast<mlir::RankedTensorType>(
                    input_value.getType())
                    .getElementType()
                    .isBF16(),
                true, true, true},
            target, *executionPolicy);
        if (mlir::failed(strategy)) {
            op.emitError(
                "cannot select an MXM execution strategy for FFN");
            return mlir::failure();
        }
    }
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const auto gate_weight_slices = target.ffn_projection_weight_slices(
        target::FfnProjectionKind::Gate);
    const auto up_weight_slices = target.ffn_projection_weight_slices(
        target::FfnProjectionKind::Up);
    const auto down_weight_slices =
        target.ffn_down_projection_weight_slices();
    llvm::SmallVector<int64_t> activation_slices;
    llvm::SmallVector<int64_t> hidden_slices = target.ffn_hidden_slices();
    llvm::SmallVector<int64_t> result_slices;
    for (int64_t index = 0;
         index < throughput.mxm_activation_streams; ++index) {
        activation_slices.push_back(
            memory.w8a16_activation_slice_base + index);
        result_slices.push_back(
            memory.w8a16_result_slice_base + index);
    }
    const int64_t hidden_pass_bytes = w8a16
        ? m * hidden * 2
        : m * 320;
    const bool singleMxmVector = w8a16
        && throughput.mxms_per_hemisphere == 1;
    const int64_t gate_rows = w8a16
        ? k * hidden
            / ((singleMxmVector ? memory.hemispheres : 1)
                * memory.w8a16_weight_slice_count
                * throughput.tile_rows * throughput.lanes_per_tile)
        : 0;
    const int64_t down_wave_columns = memory.hemispheres
        * (singleMxmVector
                ? 2 : throughput.mxms_per_hemisphere)
        * throughput.tile_rows * throughput.lanes_per_tile;
    const int64_t down_rows = w8a16
        ? ((n + down_wave_columns - 1) / down_wave_columns)
            * (hidden / throughput.mxm_rows)
            * memory.w8a16_weight_slice_count
        : 0;
    const bool requiresWeightPaging = singleMxmVector
        && std::max(gate_rows, down_rows) > memory.sram_depth_rows;
    const bool pagedWeights = weight_bank >= 0 || requiresWeightPaging;
    const int64_t initialWeightBank = std::max<int64_t>(0, weight_bank);
    std::optional<tensor::FfnWeightTilePlan> weightTilePlan;
    if (requiresWeightPaging) {
        auto planned = tensor::planFfnWeightTiles(
            {m, k, hidden, n}, target, initialWeightBank);
        if (mlir::failed(planned)) {
            op.emitError("cannot tile FFN weights into the configured SRAM banks");
            return mlir::failure();
        }
        weightTilePlan = std::move(*planned);
        if (weightTilePlan->minimum_hidden_slices
            > static_cast<int64_t>(hidden_slices.size()))
            hidden_slices = target.mxm_distributed_activation_slices();
    }
    const bool tiledWeights = weightTilePlan.has_value();
    const bool hiddenDistributed16 = tiledWeights
        && hidden_slices.size() == 16;
    const auto inheritedInputPlacement = get_value_placement(input_value);
    const auto inheritedInputBank = mlir::succeeded(inheritedInputPlacement)
        ? inheritedInputPlacement->getAs<mlir::IntegerAttr>("bank")
        : mlir::IntegerAttr{};
    const int64_t inputBank = inheritedInputBank
        ? inheritedInputBank.getInt() : 0;
    const int64_t workingBank = target.uses_dedicated_slice_roles()
        && memory.banks_per_slice > 1
        ? (inputBank + 1) % memory.banks_per_slice
        : pagedWeights
            ? (initialWeightBank + 1) % memory.banks_per_slice : 0;
    const int64_t hiddenBank = tiledWeights ? inputBank : workingBank;
    const int64_t resultBank = target.uses_dedicated_slice_roles()
        ? inputBank : workingBank;
    const bool largeSramProfile =
        memory.banks_per_slice * memory.words_per_bank >= 17000;
    const int64_t ffnWeightBase = pagedWeights
        ? 0 : largeSramProfile ? 10000 : 0;
    const int64_t distributedElementsPerRow =
        throughput.mxm_rows * throughput.mxm_block_rows;
    const int64_t distributedHiddenRows =
        m * hidden / distributedElementsPerRow;
    auto input = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Activation,
            activation_slices, 0, m * k / throughput.mxm_rows,
            m * k * 2, "fp16_mxm_activation_planar",
            "both", inputBank))
        : allocate_value(input_value, PlacementKind::Activation);
    auto gate = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
            gate_weight_slices, ffnWeightBase,
            tiledWeights ? memory.sram_depth_rows : gate_rows,
            k * hidden, {}, "east",
            initialWeightBank))
        : allocate_value(gate_weight, PlacementKind::Weight);
    auto up = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
            up_weight_slices, ffnWeightBase,
            tiledWeights ? memory.sram_depth_rows : gate_rows,
            k * hidden, {}, "east",
            initialWeightBank))
        : allocate_value(up_weight, PlacementKind::Weight);
    auto down = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
            down_weight_slices, ffnWeightBase,
            tiledWeights ? memory.sram_depth_rows : down_rows,
            hidden * n, {}, "east",
            initialWeightBank))
        : allocate_value(down_weight, PlacementKind::Weight);
    auto hidden0 = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::VxmResult,
            hidden_slices, memory.w8a16_hidden_base_row,
            hiddenDistributed16
                ? distributedHiddenRows
                : m * hidden / throughput.mxm_rows,
            hidden_pass_bytes,
            hiddenDistributed16 ? "fp16_mxm_distributed_16" : "", "both",
            hiddenBank))
        : allocator.allocate(PlacementKind::VxmResult, hidden_pass_bytes);
    auto hidden1 = w8a16
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::VxmResult1,
            hidden_slices, memory.w8a16_hidden_base_row,
            hiddenDistributed16
                ? distributedHiddenRows
                : m * hidden / throughput.mxm_rows,
            hidden_pass_bytes,
            hiddenDistributed16 ? "fp16_mxm_distributed_16" : "", "both",
            hiddenBank))
        : allocator.allocate(PlacementKind::VxmResult1, hidden_pass_bytes);
    const auto result_bytes =
        get_static_tensor_bytes(graph.output.getResult().getType());
    const auto result = w8a16 && mlir::succeeded(result_bytes)
        ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::FinalResult,
            result_slices, 0,
            m * n
                / (throughput.mxm_rows
                    * throughput.mxms_per_hemisphere),
            *result_bytes, {},
            "both", target.uses_dedicated_slice_roles()
                ? resultBank
                : pagedWeights ? workingBank : 0))
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
        ? make_profile_placement(rewriter, *input,
            "fp16_mxm_activation_planar",
            "both")
        : make_placement_attr(rewriter, *input);
    const int64_t projectionPageCount = tiledWeights
        ? (weightTilePlan->projection_wave_count
              + weightTilePlan->projection_waves_per_page - 1)
            / weightTilePlan->projection_waves_per_page
        : 0;
    const int64_t downPageCount = tiledWeights
        ? static_cast<int64_t>(weightTilePlan->pages.size())
            - projectionPageCount
        : 0;
    llvm::SmallVector<mlir::Attribute> weightStorageSliceAttrs;
    if (tiledWeights)
        for (int64_t slice : target.weight_storage_slices())
            weightStorageSliceAttrs.push_back(
                rewriter.getI64IntegerAttr(slice));
    const auto withWeightPaging = [&](mlir::DictionaryAttr placement,
                                      int64_t pageCount,
                                      int64_t pageGranularity,
                                      int64_t pageRoleGroupBase,
                                      int64_t pageRoleGroupCount,
                                      int64_t itemsPerSliceGroup) {
        if (!tiledWeights) return placement;
        mlir::NamedAttrList attrs(placement);
        attrs.set("paged_weight", rewriter.getBoolAttr(true));
        attrs.set("page_count", rewriter.getI64IntegerAttr(pageCount));
        attrs.set("page_rows",
            rewriter.getI64IntegerAttr(memory.sram_depth_rows));
        attrs.set("page_granularity",
            rewriter.getI64IntegerAttr(pageGranularity));
        attrs.set("page_role_group_base",
            rewriter.getI64IntegerAttr(pageRoleGroupBase));
        attrs.set("page_role_group_count",
            rewriter.getI64IntegerAttr(pageRoleGroupCount));
        attrs.set("page_items_per_slice_group",
            rewriter.getI64IntegerAttr(itemsPerSliceGroup));
        attrs.set("page_storage_slices",
            rewriter.getArrayAttr(weightStorageSliceAttrs));
        attrs.set("page_bank_count",
            rewriter.getI64IntegerAttr(memory.banks_per_slice));
        return attrs.getDictionary(rewriter.getContext());
    };
    const auto gate_placement = withWeightPaging(w8a16
        ? make_profile_placement(rewriter, *gate,
            singleMxmVector
                ? "w8a16_mxm_weight_wave_striped"
                : "w8a16_mxm_weight_replicated",
            "both")
        : make_placement_attr(rewriter, *gate), projectionPageCount,
        tiledWeights ? weightTilePlan->projection_waves_per_page : 0,
        0,
        tiledWeights ? weightTilePlan->projection_slice_groups_per_role : 0,
        tiledWeights ? weightTilePlan->projection_waves_per_slice_group : 0);
    const auto up_placement = withWeightPaging(w8a16
        ? make_profile_placement(rewriter, *up,
            singleMxmVector
                ? "w8a16_mxm_weight_wave_striped"
                : "w8a16_mxm_weight_replicated",
            "both")
        : make_placement_attr(rewriter, *up), projectionPageCount,
        tiledWeights ? weightTilePlan->projection_waves_per_page : 0,
        tiledWeights ? weightTilePlan->projection_slice_groups_per_role : 0,
        tiledWeights ? weightTilePlan->projection_slice_groups_per_role : 0,
        tiledWeights ? weightTilePlan->projection_waves_per_slice_group : 0);
    const auto down_placement = withWeightPaging(w8a16
        ? make_profile_placement(rewriter, *down,
            "w8a16_mxm_weight_wave_striped",
            "both")
        : make_placement_attr(rewriter, *down), downPageCount,
        tiledWeights ? weightTilePlan->down_reduction_blocks_per_page : 0,
        0, tiledWeights ? weightTilePlan->slice_group_count : 0,
        tiledWeights
            ? weightTilePlan->down_reduction_blocks_per_slice_group : 0);
    const auto hidden0_placement = w8a16
        ? make_profile_placement(rewriter, *hidden0,
            hiddenDistributed16 ? "fp16_mxm_distributed_16"
                       : "fp16_mxm_activation_planar",
            hiddenDistributed16 ? "both" : "west")
        : make_placement_attr(rewriter, *hidden0);
    const auto hidden1_placement = w8a16
        ? make_profile_placement(rewriter, *hidden1,
            hiddenDistributed16 ? "fp16_mxm_distributed_16"
                       : "fp16_mxm_activation_planar",
            hiddenDistributed16 ? "both" : "east")
        : make_placement_attr(rewriter, *hidden1);
    const auto result_placement = w8a16
        ? make_profile_placement(rewriter, *result,
            "fp16_pair_planar",
            "both")
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
    const auto projectionConfig = [&](mlir::FloatAttr scale) {
        mlir::NamedAttrList attrs;
        attrs.set("rhs_scale", scale);
        if (tiledWeights) {
            attrs.set("weight_page_count",
                rewriter.getI64IntegerAttr(projectionPageCount));
            attrs.set("weight_page_rows",
                rewriter.getI64IntegerAttr(memory.sram_depth_rows));
            attrs.set("output_waves_per_page", rewriter.getI64IntegerAttr(
                weightTilePlan->projection_waves_per_page));
        }
        return attrs.getDictionary(rewriter.getContext());
    };
    const auto downConfig = [&] {
        mlir::NamedAttrList attrs;
        attrs.set("lhs_scale", unit_scale);
        attrs.set("rhs_scale", down_scale);
        attrs.set("output_scale", unit_scale);
        attrs.set("output_zero_point", zero_point);
        if (tiledWeights) {
            attrs.set("weight_page_count",
                rewriter.getI64IntegerAttr(downPageCount));
            attrs.set("weight_page_rows",
                rewriter.getI64IntegerAttr(memory.sram_depth_rows));
            attrs.set("reduction_blocks_per_page",
                rewriter.getI64IntegerAttr(
                    weightTilePlan->down_reduction_blocks_per_page));
        }
        return attrs.getDictionary(rewriter.getContext());
    };
    const mlir::Type projection_element_type = w8a16
        ? mlir::Type(rewriter.getF32Type())
        : mlir::Type(rewriter.getI32Type());
    const mlir::Type hidden_element_type = w8a16
        ? llvm::cast<mlir::RankedTensorType>(
              input_value.getType()).getElementType()
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
        projectionConfig(gate_scale));
    auto up_task = create_matmul_task(input_value, up_weight,
        projection_type, m, hidden, k,
        input_allocations, up_allocations, transient,
        projectionConfig(up_scale));

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
        result_allocations, downConfig());
    mlir::Operation* output_operation = graph.output.getOperation();
    rewriter.replaceOp(graph.output, down_task.getResult());
    for (mlir::Operation* operation : llvm::reverse(graph.operations)) {
        if (operation != output_operation && operation->use_empty())
            rewriter.eraseOp(operation);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
