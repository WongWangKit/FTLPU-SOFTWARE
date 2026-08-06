#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Dialect/Tensor/Analysis/physical_memory_allocator.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_attention(kernel::AttentionGraph& graph,
    const target::LPUTargetModel& target, mlir::IRRewriter& rewriter)
{
    kernel::MatmulOp op = graph.output;
    const int64_t seq_len = graph.query.getM();
    const int64_t hidden = graph.query.getK();
    const int64_t query_heads = graph.query_rope.getHeads();
    const int64_t kv_heads = graph.key_rope.getHeads();
    const int64_t head_dim = graph.query_rope.getHeadDim();
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t block_rows = target.throughput().mxm_block_rows;
    const int64_t blocks = seq_len / tile;
    const int64_t query_width = query_heads * head_dim;
    const int64_t kv_width = kv_heads * head_dim;
    auto execution_policy =
        target::mxm_execution_policy_from_operation(op);
    if (mlir::failed(execution_policy)) {
        op.emitError("invalid MXM execution policy");
        return mlir::failure();
    }
    const auto input_type = llvm::cast<mlir::RankedTensorType>(
        graph.query.getLhs().getType());
    const auto query_weight_type = llvm::cast<mlir::RankedTensorType>(
        graph.query.getRhs().getType());
    auto projection_strategy = target::plan_mxm_execution_strategy(
        {seq_len, query_width, hidden,
            is_lpu_16bit_float(input_type.getElementType()),
            query_weight_type.getElementType().isInteger(8),
            true, true},
        target, *execution_policy);
    if (mlir::failed(projection_strategy)) {
        op.emitError(
            "cannot select an MXM execution strategy for attention");
        return mlir::failure();
    }
    const bool block8_attention = projection_strategy->uses_block8();
    const auto attention_weight_rows = [&](int64_t columns) {
        const int64_t head_groups = (columns + 2 * head_dim - 1)
            / (2 * head_dim);
        return head_groups * (hidden / tile)
            * target.throughput().lanes_per_tile;
    };
    const auto weight_slices = target.attention_weight_slices();
    const auto output_weight_slices =
        target.attention_output_weight_slices();
    const auto planar_activation_slices =
        target.attention_activation_slices();
    const auto activation_slices = block8_attention
        ? target.mxm_distributed_activation_slices()
        : planar_activation_slices;
    if (planar_activation_slices.size() < 2
        || (block8_attention && activation_slices.size() != 16)) {
        op.emitError(
            "target does not provide the attention activation layout");
        return mlir::failure();
    }
    const auto output_slices =
        target.attention_projection_output_slices();
    const auto key_slices = block8_attention
        ? target.attention_qk_key_slices()
        : output_slices;
    const int64_t q_weight_rows = attention_weight_rows(query_width);
    const int64_t k_weight_rows = attention_weight_rows(kv_width);
    const int64_t v_weight_rows = attention_weight_rows(kv_width);
    const int64_t o_weight_rows = hidden * query_width
        / (target.memory().hemispheres * target.memory().w8a16_weight_slice_count * tile);
    const int64_t query_rows = query_heads * blocks * target.throughput().tile_rows;
    const int64_t rope_staging_rows =
        2 * target.memory().slices_per_hemisphere * 2 * blocks
        * (tile / block_rows);
    const int64_t score_rows = query_heads * blocks * seq_len;
    const int64_t context_rows = query_heads * seq_len;
    const int64_t output_activation_rows =
        query_width / tile * blocks * (tile / block_rows);
    const int64_t distributed_input_rows =
        seq_len * hidden / (tile * block_rows);
    const int64_t output_activation_base = distributed_input_rows;
    const int64_t distributed_result_base = std::max(
        target.memory().w8a16_hidden_base_row,
        distributed_input_rows + output_activation_rows);
    const int64_t input_staging_rows = seq_len * hidden / tile;
    const int64_t input_staging_base =
        target.memory().banks_per_slice
            * target.memory().words_per_bank
        - input_staging_rows;
    llvm::SmallVector<int64_t, 36> scratch_candidates;
    for (int64_t slice = 0;
         slice < target.memory().accumulator_slice_base; ++slice)
        scratch_candidates.push_back(slice);
    tensor::PhysicalMemoryAllocator physical_allocator(target);
    const llvm::SmallVector<int64_t, 16> input_staging_slices(
        planar_activation_slices.begin(),
        planar_activation_slices.begin() + 2);
    if (mlir::failed(physical_allocator.reserve({"input_staging",
            input_staging_slices, input_staging_base,
            input_staging_rows, 0, 2, true}))) {
        op.emitError("failed to reserve the attention input staging buffer");
        return mlir::failure();
    }
    const int64_t output_weight_base =
        q_weight_rows + k_weight_rows + v_weight_rows;
    if (mlir::failed(physical_allocator.reserve({"output_weight",
            llvm::SmallVector<int64_t, 16>(
                output_weight_slices.begin(), output_weight_slices.end()),
            output_weight_base, o_weight_rows,
            0, 6, false}))) {
        op.emitError("failed to reserve the live O-projection weights");
        return mlir::failure();
    }
    const auto probability_pack_slices = target.attention_query_iw_slices(1);
    const auto staging_slice_values =
        target.attention_rope_staging_slices();
    const llvm::SmallVector<int64_t, 16> rope_staging_slices(
        staging_slice_values.begin(), staging_slice_values.end());
    if (mlir::failed(physical_allocator.reserve({"rope_staging",
            rope_staging_slices, 0, rope_staging_rows, 0, 2}))) {
        op.emitError("failed to reserve the attention RoPE staging FIFO");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"probability_pack",
            llvm::SmallVector<int64_t, 16>(
            probability_pack_slices.begin(), probability_pack_slices.end()),
            target.attention_probability_pack_base_row(),
            query_heads * blocks
                * (seq_len
                    / target.throughput().lanes_per_tile),
            3, 5}))) {
        op.emitError("failed to reserve the attention probability-pack layout");
        return mlir::failure();
    }
    if (target.memory().slices_per_hemisphere < 16) {
        op.emitError("target cannot reserve fused attention scratch banks");
        return mlir::failure();
    }
    const int64_t fused_slice_base =
        target.memory().slices_per_hemisphere - 16;
    const llvm::SmallVector<int64_t, 16> fused_score0 {
        fused_slice_base, fused_slice_base + 1,
        fused_slice_base + 2, fused_slice_base + 3};
    const llvm::SmallVector<int64_t, 16> fused_score1 {
        fused_slice_base + 4, fused_slice_base + 5,
        fused_slice_base + 6, fused_slice_base + 7};
    const llvm::SmallVector<int64_t, 16> fused_mask0 {
        fused_slice_base + 8, fused_slice_base + 9,
        fused_slice_base + 10, fused_slice_base + 11};
    const llvm::SmallVector<int64_t, 16> fused_mask1 {
        fused_slice_base + 12, fused_slice_base + 13,
        fused_slice_base + 14, fused_slice_base + 15};
    for (const auto& reservation : {
             tensor::PhysicalAllocation {"fused_score",
                 fused_score0, target.attention_score_base_row(),
                 score_rows, 2, 3},
             tensor::PhysicalAllocation {"fused_score_bank1",
                 fused_score1, target.attention_score_base_row(),
                 score_rows, 2, 3},
             tensor::PhysicalAllocation {"fused_causal_mask",
                 fused_mask0, target.attention_mask_base_row(),
                 tile - 1, 2, 3},
             tensor::PhysicalAllocation {"fused_causal_mask_bank1",
                 fused_mask1, target.attention_mask_base_row(),
                 tile - 1, 2, 3}}) {
        if (mlir::failed(physical_allocator.reserve(reservation))) {
            op.emitError("failed to reserve fused attention scratch");
            return mlir::failure();
        }
    }
    const auto allocate_scratch = [&](llvm::StringRef name,
                                      int64_t slice_count,
                                      int64_t base_row,
                                      int64_t rows,
                                      int64_t live_start,
                                      int64_t live_end)
        -> mlir::FailureOr<tensor::PhysicalAllocation> {
        return physical_allocator.allocate({name.str(), slice_count,
            base_row, rows, live_start, live_end, scratch_candidates});
    };
    auto score0 = allocate_scratch(
        "score_mxm0", 4, target.attention_score_base_row(),
        score_rows, 2, 3);
    auto score1 = allocate_scratch(
        "score_mxm1", 4, target.attention_score_base_row(),
        score_rows, 2, 3);
    auto exp0 = allocate_scratch(
        "exp_mxm0", 4, target.attention_score_base_row(),
        score_rows, 2, 3);
    auto exp1 = allocate_scratch(
        "exp_mxm1", 4, target.attention_score_base_row(),
        score_rows, 2, 3);
    auto mask0 = allocate_scratch(
        "causal_mask_mxm0", 4, target.attention_mask_base_row(),
        tile - 1, 2, 3);
    auto mask1 = allocate_scratch(
        "causal_mask_mxm1", 4, target.attention_mask_base_row(),
        tile - 1, 2, 3);
    if (mlir::failed(score0) || mlir::failed(score1)
        || mlir::failed(exp0) || mlir::failed(exp1)
        || mlir::failed(mask0) || mlir::failed(mask1)) {
        op.emitError("attention scratch memory allocation failed");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op);
    auto inheritedInputPlacement =
        get_value_placement(graph.query.getLhs());
    const auto inputPlacement = mlir::succeeded(inheritedInputPlacement)
        ? *inheritedInputPlacement
        : make_attention_placement(rewriter,
            block8_attention ? "fp16_mxm_distributed_16"
                             : "fp16_mxm_activation_planar",
            activation_slices, 0,
            block8_attention
                ? seq_len * hidden / (tile * block_rows)
                : seq_len * hidden / tile,
            "both");
    const auto plan = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("input", inputPlacement),
        rewriter.getNamedAttr("input_staging",
            make_attention_placement(rewriter,
                "fp16_pair_planar", input_staging_slices,
                input_staging_base, input_staging_rows, "both")),
        rewriter.getNamedAttr("query_weight", make_attention_placement(rewriter,
            "w8a16_attention_weight_striped", weight_slices, 0, q_weight_rows, "both")),
        rewriter.getNamedAttr("key_weight", make_attention_placement(rewriter,
            "w8a16_attention_weight_striped", weight_slices, q_weight_rows, k_weight_rows, "both")),
        rewriter.getNamedAttr("value_weight", make_attention_placement(rewriter,
            "w8a16_attention_weight_striped", weight_slices, q_weight_rows + k_weight_rows, v_weight_rows, "both")),
        rewriter.getNamedAttr("output_weight", make_attention_placement(rewriter,
            "w8a16_mxm_weight_striped", output_weight_slices, output_weight_base,
            o_weight_rows, "both")),
        rewriter.getNamedAttr("query", make_attention_placement(rewriter,
            "fp16_query_iw", output_slices,
            target.attention_query_iw_base_row(), query_rows, "both")),
        rewriter.getNamedAttr("key", make_attention_placement(rewriter,
            "fp16_head_planar", key_slices, 0,
            kv_heads * seq_len, "both")),
        rewriter.getNamedAttr("value", make_attention_placement(rewriter,
            "fp16_value_x16", target.attention_value_slices(),
            target.attention_value_base_row(),
            kv_heads * (head_dim / tile) * blocks
                * target.throughput().tile_rows, "both")),
        rewriter.getNamedAttr("score", make_attention_placement(rewriter,
            "fp16_score_block", score0->slices,
            target.attention_score_base_row(), score_rows, "both")),
        rewriter.getNamedAttr("score_mxm1", make_attention_placement(rewriter,
            "fp16_score_block", score1->slices,
            target.attention_score_base_row(), score_rows, "both")),
        rewriter.getNamedAttr("exp", make_attention_placement(rewriter,
            "fp16_score_block", exp0->slices,
            target.attention_score_base_row(), score_rows, "both")),
        rewriter.getNamedAttr("exp_mxm1", make_attention_placement(rewriter,
            "fp16_score_block", exp1->slices,
            target.attention_score_base_row(), score_rows, "both")),
        rewriter.getNamedAttr("causal_mask", make_attention_placement(rewriter,
            "fp32_causal_mask_tile", mask0->slices,
            target.attention_mask_base_row(), tile - 1, "both")),
        rewriter.getNamedAttr("causal_mask_mxm1", make_attention_placement(rewriter,
            "fp32_causal_mask_tile", mask1->slices,
            target.attention_mask_base_row(), tile - 1, "both")),
        rewriter.getNamedAttr("fused_score", make_attention_placement(rewriter,
            "fp32_score_block", fused_score0,
            target.attention_score_base_row(), score_rows, "both")),
        rewriter.getNamedAttr("fused_score_bank1", make_attention_placement(rewriter,
            "fp32_score_block", fused_score1,
            target.attention_score_base_row(), score_rows, "both")),
        rewriter.getNamedAttr("fused_causal_mask", make_attention_placement(rewriter,
            "fp32_causal_mask_tile", fused_mask0,
            target.attention_mask_base_row(), tile - 1, "both")),
        rewriter.getNamedAttr("fused_causal_mask_bank1", make_attention_placement(rewriter,
            "fp32_causal_mask_tile", fused_mask1,
            target.attention_mask_base_row(), tile - 1, "both")),
        rewriter.getNamedAttr("probability_pack", make_attention_placement(rewriter,
            "fp16_probability_x16", target.attention_query_iw_slices(1),
            target.attention_probability_pack_base_row(),
            query_heads * blocks
                * (seq_len / target.throughput().lanes_per_tile), "both")),
        rewriter.getNamedAttr("probability_diagonal", make_attention_placement(rewriter,
            "fp16_probability_diagonal", target.attention_query_iw_slices(0),
            target.attention_probability_diagonal_base_row(),
            query_heads * blocks * blocks
                * target.throughput().tile_rows, "both")),
        rewriter.getNamedAttr("rope", make_attention_placement(rewriter,
            "fp16_rope_table", target.attention_rope_slices(),
            target.attention_probability_diagonal_base_row(),
            seq_len, "both")),
        rewriter.getNamedAttr("rope_staging",
            make_attention_placement(rewriter,
                "fp16_rope_fifo_x16", rope_staging_slices,
                0, rope_staging_rows, "both")),
        rewriter.getNamedAttr("context", make_attention_placement(rewriter,
            "fp16_head_planar", target.attention_context_slices(),
            target.attention_context_base_row(), context_rows, "both")),
        rewriter.getNamedAttr("output_activation",
            block8_attention
                ? make_attention_placement(rewriter,
                      "fp16_mxm_distributed_16",
                      target.attention_output_activation_slices(),
                      output_activation_base,
                      output_activation_rows, "both")
                : make_attention_placement(rewriter,
                      "fp16_pair_planar", input_staging_slices,
                      input_staging_base, input_staging_rows,
                      "both")),
        rewriter.getNamedAttr("result", make_attention_placement(rewriter,
            block8_attention
                ? "fp16_mxm_block8_distributed_16"
                : "fp16_pair_planar",
            block8_attention
                ? target.mxm_distributed_activation_slices()
                : target.attention_result_slices(),
            block8_attention
                ? distributed_result_base
                : 0,
            block8_attention
                ? seq_len * hidden / (tile * block_rows)
                : seq_len * hidden / (tile * 2),
            block8_attention ? "both" : "east")),
    });
    const auto config = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr(
            "seq_len", rewriter.getI64IntegerAttr(seq_len)),
        rewriter.getNamedAttr(
            "hidden", rewriter.getI64IntegerAttr(hidden)),
        rewriter.getNamedAttr(
            "query_heads", graph.query_rope.getHeadsAttr()),
        rewriter.getNamedAttr(
            "kv_heads", graph.key_rope.getHeadsAttr()),
        rewriter.getNamedAttr(
            "head_dim", graph.query_rope.getHeadDimAttr()),
        rewriter.getNamedAttr(
            "rope_theta", graph.query_rope.getThetaAttr()),
        rewriter.getNamedAttr(
            "causal", graph.softmax.getCausalAttr()),
        rewriter.getNamedAttr(
            "query_weight_scale", graph.query.getRhsScaleAttr()),
        rewriter.getNamedAttr(
            "key_weight_scale", graph.key.getRhsScaleAttr()),
        rewriter.getNamedAttr(
            "value_weight_scale", graph.value.getRhsScaleAttr()),
        rewriter.getNamedAttr(
            "output_weight_scale", graph.output.getRhsScaleAttr()),
    });
    const auto subplan =
        [&](std::initializer_list<llvm::StringRef> names) {
            llvm::SmallVector<mlir::NamedAttribute> entries;
            for (llvm::StringRef name : names)
                entries.push_back(rewriter.getNamedAttr(
                    name, plan.get(name)));
            return rewriter.getDictionaryAttr(entries);
        };
    const auto emptyPlan = rewriter.getDictionaryAttr({});
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(
            graph.query.getLhs().getType())
            .getElementType();
    const auto matrixType = [&](int64_t rows, int64_t columns) {
        return mlir::RankedTensorType::get(
            {rows, columns}, elementType);
    };
    const auto scoreType = mlir::RankedTensorType::get(
        {query_heads, seq_len, seq_len},
        elementType);
    const auto createProjection =
        [&](mlir::Value input, mlir::Value weight,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::DictionaryAttr memoryPlan) {
            mlir::OperationState state(
                op.getLoc(),
                tensor::ProjectionTaskOp::getOperationName());
            state.addOperands({input, weight});
            state.addTypes(resultType);
            state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr(kind)),
                rewriter.getNamedAttr("config", config),
                rewriter.getNamedAttr("memory_plan", memoryPlan),
            });
            return llvm::cast<tensor::ProjectionTaskOp>(
                rewriter.create(state));
        };
    const auto createUnary =
        [&](llvm::StringRef operationName, mlir::Value input,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::DictionaryAttr memoryPlan) {
            mlir::OperationState state(op.getLoc(), operationName);
            state.addOperands(input);
            state.addTypes(resultType);
            state.addAttribute("config", config);
            state.addAttribute("memory_plan", memoryPlan);
            if (!kind.empty())
                state.addAttribute(
                    "kind", rewriter.getStringAttr(kind));
            return rewriter.create(state)->getResult(0);
        };
    const auto createBatchMatmul =
        [&](mlir::Value lhs, mlir::Value rhs,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::DictionaryAttr memoryPlan) {
            mlir::OperationState state(
                op.getLoc(),
                tensor::BatchMatmulTaskOp::getOperationName());
            state.addOperands({lhs, rhs});
            state.addTypes(resultType);
            state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr(kind)),
                rewriter.getNamedAttr("config", config),
                rewriter.getNamedAttr("memory_plan", memoryPlan),
            });
            return llvm::cast<tensor::BatchMatmulTaskOp>(
                rewriter.create(state));
        };

    auto query = createProjection(
        graph.query.getLhs(), graph.query.getRhs(),
        "query", matrixType(seq_len, query_width),
        subplan({"input", "input_staging",
            "query_weight", "query", "rope_staging"}));
    auto key = createProjection(
        graph.key.getLhs(), graph.key.getRhs(),
        "key", matrixType(seq_len, kv_width),
        subplan({"key_weight", "key"}));
    auto value = createProjection(
        graph.value.getLhs(), graph.value.getRhs(),
        "value", matrixType(seq_len, kv_width),
        subplan({"value_weight", "value"}));
    const mlir::Value rotatedQuery = createUnary(
        tensor::RopeTaskOp::getOperationName(), query.getResult(),
        "query", query.getResult().getType(), subplan({"rope"}));
    const mlir::Value rotatedKey = createUnary(
        tensor::RopeTaskOp::getOperationName(), key.getResult(),
        "key", key.getResult().getType(), emptyPlan);
    auto qk = createBatchMatmul(
        rotatedQuery, rotatedKey, "qk", scoreType,
        subplan({"score", "score_mxm1"}));
    const mlir::Value probability = createUnary(
        tensor::SoftmaxTaskOp::getOperationName(), qk.getResult(),
        "", scoreType,
        subplan({"exp", "exp_mxm1", "causal_mask",
            "causal_mask_mxm1", "probability_pack",
            "probability_diagonal", "fused_score",
            "fused_score_bank1", "fused_causal_mask",
            "fused_causal_mask_bank1"}));
    const mlir::Value transposedProbability = createUnary(
        tensor::TransposeTaskOp::getOperationName(), probability,
        "probability", scoreType, emptyPlan);
    const mlir::Value transposedValue = createUnary(
        tensor::TransposeTaskOp::getOperationName(),
        value.getResult(), "value", value.getResult().getType(),
        emptyPlan);
    auto pv = createBatchMatmul(
        transposedProbability, transposedValue, "pv",
        matrixType(seq_len, query_width),
        subplan({"context"}));
    auto output = createProjection(pv.getResult(),
        graph.output.getRhs(), "output",
        graph.output.getResult().getType(),
        subplan({"output_activation", "output_weight", "result"}));
    mlir::Operation* output_operation = graph.output.getOperation();
    rewriter.replaceOp(graph.output, output.getResult());
    for (mlir::Operation* operation :
         llvm::reverse(graph.operations)) {
        if (operation != output_operation && operation->use_empty())
            rewriter.eraseOp(operation);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
