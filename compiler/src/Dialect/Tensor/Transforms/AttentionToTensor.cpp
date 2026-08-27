#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Dialect/Tensor/Analysis/attention_weight_tile_plan.hpp"
#include "ftlpu/compiler/Dialect/Tensor/Analysis/physical_memory_allocator.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <optional>

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_attention(kernel::AttentionGraph& graph,
    const target::LPUTargetModel& target, int64_t weight_bank,
    mlir::IRRewriter& rewriter)
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
    const auto attention_weight_rows = [&](int64_t columns) {
        // W8A16AttentionWeightStriped stores 128 output columns per
        // physical wave: two 32-column groups in each hemisphere.
        const int64_t columns_per_wave = 4 * tile;
        const int64_t waves = (columns + columns_per_wave - 1)
            / columns_per_wave;
        return waves * (hidden / tile)
            * target.throughput().lanes_per_tile;
    };
    const bool paged_weights = weight_bank >= 0;
    const int64_t scratch_bank = paged_weights
        ? (weight_bank + 1) % target.memory().banks_per_slice : 0;
    const int64_t secondary_scratch_bank = paged_weights
        ? weight_bank : scratch_bank;
    const auto weight_slices = paged_weights
        ? target.page_resident_attention_weight_slices()
        : target.attention_weight_slices();
    const auto output_weight_slices = paged_weights
        ? weight_slices : target.attention_output_weight_slices();
    const auto planar_activation_slices =
        target.attention_activation_slices();
    const auto activation_slices = planar_activation_slices;
    if (planar_activation_slices.size() < 2) {
        op.emitError(
            "target does not provide the attention activation layout");
        return mlir::failure();
    }
    const auto output_slices =
        target.attention_projection_output_slices();
    const auto key_slices = target.attention_qk_key_slices();
    const int64_t q_weight_rows = attention_weight_rows(query_width);
    const int64_t k_weight_rows = attention_weight_rows(kv_width);
    const int64_t v_weight_rows = attention_weight_rows(kv_width);
    const int64_t o_weight_rows = hidden * query_width
        / (target.memory().hemispheres * target.memory().w8a16_weight_slice_count * tile);
    const bool requires_weight_tiling = paged_weights
        && target.throughput().mxms_per_hemisphere == 1
        && std::max({q_weight_rows, k_weight_rows, v_weight_rows,
               o_weight_rows}) > target.memory().sram_depth_rows;
    std::optional<tensor::AttentionWeightTilePlan> weight_tile_plan;
    if (requires_weight_tiling) {
        auto planned = tensor::planAttentionWeightTiles(hidden,
            query_heads, kv_heads, head_dim,
            std::max<int64_t>(0, weight_bank), target);
        if (mlir::failed(planned)) {
            op.emitError(
                "cannot tile attention weights into the configured SRAM banks");
            return mlir::failure();
        }
        weight_tile_plan = std::move(*planned);
    }
    const bool tiled_weights = weight_tile_plan.has_value();
    const int64_t head_blocks = head_dim / tile;
    const int64_t logical_head_banks = (head_blocks + 1) / 2;
    const int64_t query_rows = query_heads * logical_head_banks * blocks
        * target.throughput().tile_rows;
    const int64_t rope_frequency_blocks = head_dim / (2 * tile);
    const int64_t rope_rows = seq_len * rope_frequency_blocks;
    // Projection and RoPE consume one 128-column physical wave at a time.
    // Only its four 32-column blocks are live in the staging FIFO; later
    // heads reuse the same rows.
    const int64_t rope_staging_rows = 4 * seq_len;
    const int64_t rope_product_rows =
        (query_heads + kv_heads) * (head_blocks / 2) * 4
        * ((seq_len + 1) / 2);
    const int64_t score_rows = query_heads * blocks * seq_len;
    const int64_t probability_pack_rows = query_heads * blocks
        * (seq_len / target.throughput().lanes_per_tile);
    const int64_t context_rows = query_heads * seq_len;
    const int64_t output_activation_rows =
        query_width / tile * blocks * (tile / block_rows);
    const int64_t distributed_input_rows =
        seq_len * hidden / (tile * block_rows);
    const int64_t output_activation_base = distributed_input_rows;
    const int64_t input_staging_rows = seq_len * hidden / tile;
    const int64_t input_staging_base =
        target.memory().words_per_bank - input_staging_rows;
    // The function input remains live until the attention residual add. Keep
    // Q/V and score scratch above its distributed16 rows. The RoPE table and
    // causal mask are initialized before execution, so the mask must also sit
    // below the late input-staging window instead of relying on stage lifetime
    // reuse.
    const int64_t score_base = target.uses_dedicated_slice_roles()
        ? distributed_input_rows : target.attention_score_base_row();
    const int64_t query_base = score_base + score_rows;
    const int64_t value_base = query_base + query_rows;
    const int64_t causal_mask_base = target.uses_dedicated_slice_roles()
        ? rope_rows : target.attention_mask_base_row();
    if (value_base
            + kv_heads * (head_dim / tile) * blocks
                * target.throughput().tile_rows
            > target.memory().words_per_bank
        || causal_mask_base + tile > input_staging_base) {
        op.emitError(
            "attention activation scratch does not fit around persistent "
            "input and initialized constants");
        return mlir::failure();
    }
    llvm::SmallVector<int64_t, 36> scratch_candidates;
    scratch_candidates = target.activation_storage_slices();
    const llvm::SmallVector<int64_t, 2> scratch_banks {
        scratch_bank, secondary_scratch_bank};
    tensor::PhysicalMemoryAllocator physical_allocator(target);
    const int64_t input_staging_bank = target.uses_dedicated_slice_roles()
        ? secondary_scratch_bank : scratch_bank;
    const auto input_staging_values = target.uses_dedicated_slice_roles()
        ? target.activation_storage_slices() : planar_activation_slices;
    const llvm::SmallVector<int64_t, 16> input_staging_slices(
        target.uses_dedicated_slice_roles()
            ? input_staging_values.end() - 2
            : input_staging_values.begin(),
        target.uses_dedicated_slice_roles()
            ? input_staging_values.end()
            : input_staging_values.begin() + 2);
    if (mlir::failed(physical_allocator.reserve({"input_staging",
            input_staging_slices, input_staging_base,
            input_staging_rows, 0, 2, true, input_staging_bank}))) {
        op.emitError("failed to reserve the attention input staging buffer");
        return mlir::failure();
    }
    // Keep the long-lived O-projection weights out of the fixed probability
    // staging window. Larger models can make O weights cross that window even
    // though the compact SmolLM2 shape does not.
    const int64_t output_weight_base = tiled_weights ? 0 : paged_weights
        ? q_weight_rows + k_weight_rows + v_weight_rows
        : std::max({q_weight_rows + k_weight_rows + v_weight_rows,
              score_base + score_rows,
              target.attention_probability_pack_base_row()
                  + probability_pack_rows,
              causal_mask_base + tile});
    if (!paged_weights
        && mlir::failed(physical_allocator.reserve({"output_weight",
            llvm::SmallVector<int64_t, 16>(
                output_weight_slices.begin(), output_weight_slices.end()),
            output_weight_base, o_weight_rows,
            0, 6, false, std::max<int64_t>(0, weight_bank)}))) {
        op.emitError("failed to reserve the live O-projection weights");
        return mlir::failure();
    }
    llvm::SmallVector<int64_t, 16> probability_pack_slices;
    if (tiled_weights && target.uses_dedicated_slice_roles()) {
        const auto storage = target.weight_storage_slices();
        probability_pack_slices.assign(
            storage.end() - 16, storage.end());
    } else {
        const auto values = target.attention_query_iw_slices(1);
        probability_pack_slices.assign(values.begin(), values.end());
    }
    const auto staging_slice_values =
        target.attention_rope_staging_slices();
    const llvm::SmallVector<int64_t, 16> rope_staging_slices(
        staging_slice_values.begin(), staging_slice_values.end());
    const int64_t rope_staging_bank = target.uses_dedicated_slice_roles()
        ? secondary_scratch_bank : scratch_bank;
    llvm::SmallVector<int64_t, 16> rope_product_slices;
    if (tiled_weights && target.uses_dedicated_slice_roles()) {
        const auto storage = target.weight_storage_slices();
        rope_product_slices.assign(
            storage.begin(), storage.begin() + 16);
    } else {
        const auto values = target.mxm_distributed_activation_slices();
        rope_product_slices.assign(values.begin(), values.end());
    }
    llvm::SmallVector<int64_t, 4> rope_table_slices;
    int64_t rope_table_bank = scratch_bank;
    const auto rope_slices = target.attention_rope_slices();
    rope_table_slices.assign(rope_slices.begin(), rope_slices.end());
    if (mlir::failed(physical_allocator.reserve({"rope_staging",
            rope_staging_slices, 0, rope_staging_rows, 0, 2, true,
            rope_staging_bank}))) {
        op.emitError("failed to reserve the attention RoPE staging FIFO");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"rope_product",
            rope_product_slices, 0, rope_product_rows, 0, 2, true,
            scratch_bank}))) {
        op.emitError("failed to reserve the attention RoPE product FIFO");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"probability_pack",
            llvm::SmallVector<int64_t, 16>(
            probability_pack_slices.begin(), probability_pack_slices.end()),
            target.attention_probability_pack_base_row(),
            probability_pack_rows,
            3, 4, true, scratch_bank}))) {
        op.emitError("failed to reserve the attention probability-pack layout");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"probability_diagonal",
            rope_product_slices,
            target.attention_probability_diagonal_base_row(),
            query_heads * blocks * blocks
                * target.throughput().tile_rows,
            4, 5, true, scratch_bank}))) {
        op.emitError(
            "failed to reserve the attention probability-diagonal ports");
        return mlir::failure();
    }
    const auto output_activation_slice_values =
        target.attention_output_activation_slices(paged_weights);
    const llvm::SmallVector<int64_t, 16> output_activation_slices(
        output_activation_slice_values.begin(),
        output_activation_slice_values.end());
    if (mlir::failed(physical_allocator.reserve({"output_activation",
            output_activation_slices, output_activation_base,
            output_activation_rows, 5, 6, true, scratch_bank}))) {
        op.emitError(
            "failed to reserve the attention output-activation ports");
        return mlir::failure();
    }
    if (target.memory().slices_per_hemisphere < 16) {
        op.emitError("target cannot reserve fused attention scratch banks");
        return mlir::failure();
    }
    const auto activation_storage = target.activation_storage_slices();
    const int64_t fused_slice_base = target.uses_dedicated_slice_roles()
        ? activation_storage.front()
        : target.memory().slices_per_hemisphere - 16;
    const int64_t fused_scratch_bank =
        target.uses_dedicated_slice_roles()
        ? secondary_scratch_bank : scratch_bank;
    const llvm::SmallVector<int64_t, 16> fused_score0 {
        fused_slice_base, fused_slice_base + 1,
        fused_slice_base + 2, fused_slice_base + 3};
    const llvm::SmallVector<int64_t, 16> fused_score1 {
        fused_slice_base + 4, fused_slice_base + 5,
        fused_slice_base + 6, fused_slice_base + 7};
    const llvm::SmallVector<int64_t, 16> fused_mask0 {
        fused_slice_base + 8, fused_slice_base + 9};
    const llvm::SmallVector<int64_t, 16> fused_mask1 {
        fused_slice_base + 10, fused_slice_base + 11};
    for (const auto& reservation : {
             tensor::PhysicalAllocation {"fused_score",
                  fused_score0, target.attention_score_base_row(),
                  score_rows, 2, 3, true, fused_scratch_bank},
             tensor::PhysicalAllocation {"fused_score_bank1",
                  fused_score1, target.attention_score_base_row(),
                  score_rows, 2, 3, true, fused_scratch_bank},
             tensor::PhysicalAllocation {"fused_causal_mask",
                  fused_mask0, target.attention_mask_base_row(),
                  tile - 1, 2, 3, true, fused_scratch_bank},
             tensor::PhysicalAllocation {"fused_causal_mask_bank1",
                  fused_mask1, target.attention_mask_base_row(),
                  tile - 1, 2, 3, true, fused_scratch_bank}}) {
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
            base_row, rows, live_start, live_end, scratch_candidates,
            true, scratch_banks});
    };
    auto score0 = allocate_scratch(
        "score_mxm0", 4, score_base,
        score_rows, 2, 3);
    auto score1 = allocate_scratch(
        "score_mxm1", 4, score_base,
        score_rows, 2, 3);
    auto exp0 = allocate_scratch(
        "exp_mxm0", 4, score_base,
        score_rows, 2, 3);
    auto exp1 = allocate_scratch(
        "exp_mxm1", 4, score_base,
        score_rows, 2, 3);
    auto mask0 = allocate_scratch(
        "causal_mask_mxm0", 2, causal_mask_base,
        tile - 1, 2, 3);
    auto mask1 = allocate_scratch(
        "causal_mask_mxm1", 2, causal_mask_base,
        tile - 1, 2, 3);
    auto context = allocate_scratch(
        "context",
        static_cast<int64_t>(target.attention_context_slices().size()),
        target.attention_context_base_row(), context_rows, 4, 6);
    if (mlir::failed(score0) || mlir::failed(score1)
        || mlir::failed(exp0) || mlir::failed(exp1)
        || mlir::failed(mask0) || mlir::failed(mask1)
        || mlir::failed(context)) {
        op.emitError("attention scratch memory allocation failed");
        return mlir::failure();
    }
    const llvm::SmallVector<int64_t, 1> result_banks {
        secondary_scratch_bank};
    auto vector_result = physical_allocator.allocate({"result",
        target.throughput().mxm_result_streams,
        0, seq_len * hidden / (tile * 2), 5, 7,
        scratch_candidates, true, result_banks});
    if (mlir::failed(vector_result)) {
        op.emitError(
            "attention result cannot be placed without overlapping "
            "live context or residual storage");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Attribute, 32> weight_storage_slice_attrs;
    if (tiled_weights)
        for (int64_t slice : weight_tile_plan->storage_slices)
            weight_storage_slice_attrs.push_back(
                rewriter.getI64IntegerAttr(slice));
    const auto with_weight_paging = [&](mlir::DictionaryAttr placement,
                                        tensor::AttentionWeightTileKind kind,
                                        int64_t phase) {
        if (!tiled_weights) return placement;
        const auto& tile_plan = weight_tile_plan->get(kind);
        mlir::NamedAttrList attrs(placement);
        attrs.set("paged_weight", rewriter.getBoolAttr(true));
        attrs.set("page_count", rewriter.getI64IntegerAttr(1));
        attrs.set("page_rows",
            rewriter.getI64IntegerAttr(weight_tile_plan->bank_rows));
        attrs.set("page_granularity",
            rewriter.getI64IntegerAttr(tile_plan.item_count));
        attrs.set("page_role_group_base",
            rewriter.getI64IntegerAttr(tile_plan.slice_group_begin));
        attrs.set("page_role_group_count",
            rewriter.getI64IntegerAttr(tile_plan.slice_group_count));
        attrs.set("page_items_per_slice_group",
            rewriter.getI64IntegerAttr(tile_plan.items_per_slice_group));
        attrs.set("page_storage_slices",
            rewriter.getArrayAttr(weight_storage_slice_attrs));
        attrs.set("page_bank_count", rewriter.getI64IntegerAttr(
            target.memory().banks_per_slice));
        attrs.set("page_phase", rewriter.getI64IntegerAttr(phase));
        attrs.set("page_transfer_cycles",
            rewriter.getI64IntegerAttr(tile_plan.transfer_cycles));
        return attrs.getDictionary(rewriter.getContext());
    };
    const auto weight_placement = [&](llvm::StringRef layout,
                                      tensor::AttentionWeightTileKind kind,
                                      int64_t fallbackBase,
                                      int64_t fallbackRows,
                                      llvm::ArrayRef<int64_t> slices,
                                      int64_t phase) {
        if (!tiled_weights)
            return make_attention_placement(rewriter, layout, slices,
                fallbackBase, fallbackRows, "both",
                std::max<int64_t>(0, weight_bank));
        const auto& tile_plan = weight_tile_plan->get(kind);
        return with_weight_paging(make_attention_placement(rewriter,
            layout, slices, tile_plan.base_row, tile_plan.rows, "both",
            tile_plan.bank), kind, phase);
    };
    auto inheritedInputPlacement =
        get_value_placement(graph.query.getLhs());
    const auto inputPlacement = mlir::succeeded(inheritedInputPlacement)
        ? *inheritedInputPlacement
        : make_attention_placement(rewriter,
            "fp16_mxm_activation_planar",
            activation_slices, 0, seq_len * hidden / tile,
            "both");
    const auto plan = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("input", inputPlacement),
        rewriter.getNamedAttr("input_staging",
            make_attention_placement(rewriter,
                "fp16_pair_planar", input_staging_slices,
                input_staging_base, input_staging_rows, "both",
                input_staging_bank)),
        rewriter.getNamedAttr("query_weight", weight_placement(
            "w8a16_attention_weight_striped",
            tensor::AttentionWeightTileKind::Query, 0, q_weight_rows,
            weight_slices, 0)),
        rewriter.getNamedAttr("key_weight", weight_placement(
            "w8a16_attention_weight_striped",
            tensor::AttentionWeightTileKind::Key, q_weight_rows,
            k_weight_rows, weight_slices, 0)),
        rewriter.getNamedAttr("value_weight", weight_placement(
            "w8a16_attention_weight_striped",
            tensor::AttentionWeightTileKind::Value,
            q_weight_rows + k_weight_rows, v_weight_rows,
            weight_slices, 0)),
        rewriter.getNamedAttr("output_weight", weight_placement(
            "w8a16_mxm_weight_striped",
            tensor::AttentionWeightTileKind::Output, output_weight_base,
            o_weight_rows, output_weight_slices, 1)),
        rewriter.getNamedAttr("query", make_attention_placement(rewriter,
            "fp16_query_iw", target.attention_query_iw_slices(0),
            query_base, query_rows, "both",
            scratch_bank)),
        rewriter.getNamedAttr("key", make_attention_placement(rewriter,
            "fp16_head_planar", key_slices, 0,
            kv_heads * logical_head_banks * seq_len, "both",
            target.uses_dedicated_slice_roles()
                ? secondary_scratch_bank : scratch_bank)),
        rewriter.getNamedAttr("value", make_attention_placement(rewriter,
            "fp16_value_x16", target.attention_value_slices(),
            value_base,
            kv_heads * (head_dim / tile) * blocks
                * target.throughput().tile_rows, "both", scratch_bank)),
        rewriter.getNamedAttr("score", make_attention_placement(rewriter,
            "fp16_score_block", score0->slices,
            score_base, score_rows, "both",
            score0->bank)),
        rewriter.getNamedAttr("score_mxm1", make_attention_placement(rewriter,
            "fp16_score_block", score1->slices,
            score_base, score_rows, "both",
            score1->bank)),
        rewriter.getNamedAttr("exp", make_attention_placement(rewriter,
            "fp16_score_block", exp0->slices,
            score_base, score_rows, "both",
            exp0->bank)),
        rewriter.getNamedAttr("exp_mxm1", make_attention_placement(rewriter,
            "fp16_score_block", exp1->slices,
            score_base, score_rows, "both",
            exp1->bank)),
        rewriter.getNamedAttr("causal_mask", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", mask0->slices,
            causal_mask_base, tile - 1, "both",
            mask0->bank)),
        rewriter.getNamedAttr("causal_mask_mxm1", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", mask1->slices,
            causal_mask_base, tile - 1, "both",
            mask1->bank)),
        rewriter.getNamedAttr("fused_score", make_attention_placement(rewriter,
            "fp32_score_block", fused_score0,
            target.attention_score_base_row(), score_rows, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("fused_score_bank1", make_attention_placement(rewriter,
            "fp32_score_block", fused_score1,
            target.attention_score_base_row(), score_rows, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("fused_causal_mask", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", fused_mask0,
            target.attention_mask_base_row(), tile - 1, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("fused_causal_mask_bank1", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", fused_mask1,
            target.attention_mask_base_row(), tile - 1, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("probability_pack", make_attention_placement(rewriter,
            "fp16_probability_x16", probability_pack_slices,
            target.attention_probability_pack_base_row(),
            query_heads * blocks
                * (seq_len / target.throughput().lanes_per_tile), "both",
            scratch_bank)),
        rewriter.getNamedAttr("probability_diagonal", make_attention_placement(rewriter,
            "fp16_probability_diagonal",
            tiled_weights && target.uses_dedicated_slice_roles()
                ? llvm::ArrayRef<int64_t>(rope_product_slices)
                : llvm::ArrayRef<int64_t>(
                      target.attention_query_iw_slices(0)),
            target.attention_probability_diagonal_base_row(),
            query_heads * blocks * blocks
                * target.throughput().tile_rows, "both",
            scratch_bank)),
        rewriter.getNamedAttr("rope", make_attention_placement(rewriter,
            "fp16_rope_table", rope_table_slices,
            target.attention_probability_diagonal_base_row(),
            rope_rows, "both", rope_table_bank)),
        rewriter.getNamedAttr("rope_staging",
            make_attention_placement(rewriter,
                "fp16_rope_fifo_x16", rope_staging_slices,
                0, rope_staging_rows, "both", rope_staging_bank)),
        rewriter.getNamedAttr("rope_product",
            make_attention_placement(rewriter,
                "fp16_rope_product_x16", rope_product_slices,
                0, rope_product_rows, "both",
                scratch_bank)),
        rewriter.getNamedAttr("context", make_attention_placement(rewriter,
            "fp16_head_planar", context->slices,
            target.attention_context_base_row(), context_rows, "both",
            context->bank)),
        rewriter.getNamedAttr("output_activation",
            make_attention_placement(rewriter,
                "fp16_pair_planar", input_staging_slices,
                input_staging_base, input_staging_rows,
                "both", scratch_bank)),
        rewriter.getNamedAttr("result", make_attention_placement(rewriter,
            "fp16_pair_planar",
            vector_result->slices, vector_result->base_row,
            seq_len * hidden / (tile * 2), "east",
            vector_result->bank)),
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
            "query_weight", "query", "rope_staging", "rope_product"}));
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
