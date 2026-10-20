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
    const llvm::DenseMap<mlir::Value, FeedbackGammaPlacement>& gamma_placements,
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
    // A one-token decode is one physical query block even though its public
    // tensor has no padded rows. Schedule uses decode_current_len to mask the
    // remaining lanes in that block.
    const int64_t blocks = (seq_len + tile - 1) / tile;
    int64_t decode_past_len = 0;
    int64_t decode_current_len = 0;
    int64_t kv_cache_capacity = 0;
    if (const auto module = op->getParentOfType<mlir::ModuleOp>()) {
        if (const auto capacity = module->getAttrOfType<mlir::IntegerAttr>(
                "ftlpu.kv_cache_capacity"))
            kv_cache_capacity = capacity.getInt();
        if (const auto past = module->getAttrOfType<mlir::IntegerAttr>(
                "ftlpu.decode_past_len"))
            decode_past_len = past.getInt();
        if (const auto current = module->getAttrOfType<mlir::IntegerAttr>(
                "ftlpu.decode_current_len"))
            decode_current_len = current.getInt();
    }
    const bool decode = decode_current_len != 0;
    if (decode && (decode_current_len > seq_len
            || decode_past_len % tile != 0)) {
        op.emitError(
            "decode requires current_len no greater than its padded input "
            "tile and a tile-aligned past_len");
        return mlir::failure();
    }
    const int64_t kv_seq_len = decode
        ? ((decode_past_len + decode_current_len + tile - 1) / tile) * tile
        : seq_len;
    const int64_t kv_cache_page_tokens = tile;
    const int64_t kv_cache_resident_tokens =
        ((kv_seq_len + kv_cache_page_tokens - 1) / kv_cache_page_tokens)
        * kv_cache_page_tokens;
    if (kv_cache_capacity != 0
        && kv_cache_capacity < kv_cache_resident_tokens) {
        op.emitError(
            "KV cache capacity must be zero or no smaller than the "
            "tile-aligned resident prefill window");
        return mlir::failure();
    }
    const bool kv_cache_enabled = kv_cache_capacity != 0;
    const int64_t kv_storage_tokens = kv_cache_enabled
        ? kv_cache_resident_tokens : seq_len;
    // Projection and attention routing use the global KV-head index in each
    // participating hemisphere. Preserve that stride in persistent storage;
    // K is replicated and V is copied to its owning hemisphere by Schedule.
    const int64_t kv_storage_heads = kv_heads;
    const int64_t query_width = query_heads * head_dim;
    const int64_t kv_width = kv_heads * head_dim;
    const mlir::Value query_bias =
        graph.query_bias ? graph.query_bias.getBias() : mlir::Value {};
    const mlir::Value key_bias =
        graph.key_bias ? graph.key_bias.getBias() : mlir::Value {};
    const mlir::Value value_bias =
        graph.value_bias ? graph.value_bias.getBias() : mlir::Value {};
    const mlir::Value query_norm_weight =
        graph.query_norm ? graph.query_norm.getWeight() : mlir::Value {};
    const mlir::Value key_norm_weight =
        graph.key_norm ? graph.key_norm.getWeight() : mlir::Value {};
    const bool has_qk_norm = query_norm_weight && key_norm_weight;
    const bool has_qk_bias = query_bias || key_bias;
    const bool has_attention_bias = query_bias || key_bias || value_bias;
    bool projection_rope_overlap = false;
    if (const auto module = op->getParentOfType<mlir::ModuleOp>())
        if (const auto overlap = module->getAttrOfType<mlir::BoolAttr>(
                "ftlpu.projection_rope_overlap"))
            projection_rope_overlap = overlap.getValue();
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
    // Q/K/V/O still use the proven vector layout while their decode-specific
    // bias/RoPE/KV postprocess is being moved to the Native4 accumulator
    // drain.  The FFN path below already exercises the real Native4 ISA.
    const bool native4 = false;
    const auto attention_weight_rows = [&](int64_t columns) {
        if (native4)
            return (columns / 64) * (hidden / 32);
        // W8A16AttentionWeightStriped stores 128 output columns per
        // physical wave: two 32-column groups in each hemisphere.
        const int64_t columns_per_wave = 4 * tile;
        const int64_t waves = (columns + columns_per_wave - 1)
            / columns_per_wave;
        return waves * (hidden / tile)
            * 2 * (tile / target.throughput().lanes_per_tile);
    };
    const bool paged_weights = weight_bank >= 0;
    const int64_t scratch_bank = paged_weights
        ? (weight_bank + 1) % target.memory().banks_per_slice : 0;
    const int64_t secondary_scratch_bank = paged_weights
        ? weight_bank : scratch_bank;
    llvm::SmallVector<int64_t, 32> weight_slices;
    if (native4) {
        for (int64_t slice = 20; slice < 52; ++slice)
            weight_slices.push_back(slice);
    } else {
        const auto selected = paged_weights
            ? target.page_resident_attention_weight_slices()
            : target.attention_weight_slices();
        weight_slices.assign(selected.begin(), selected.end());
    }
    llvm::SmallVector<int64_t, 32> output_weight_slices;
    if (native4 || paged_weights)
        output_weight_slices = weight_slices;
    else {
        const auto selected = target.attention_output_weight_slices();
        output_weight_slices.assign(selected.begin(), selected.end());
    }
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
    const int64_t q_weight_rows = attention_weight_rows(query_width);
    const int64_t k_weight_rows = attention_weight_rows(kv_width);
    const int64_t v_weight_rows = attention_weight_rows(kv_width);
    const int64_t o_weight_rows = native4
        ? (hidden / 64) * (query_width / 32)
        : hidden * query_width
            / (target.memory().hemispheres
                * target.memory().w8a16_weight_slice_count * tile);
    const auto module = op->getParentOfType<mlir::ModuleOp>();
    const bool graph_weight_pages = module && module->hasAttr("ftlpu.runtime_weight_pages");
    const bool requires_weight_tiling = paged_weights
        && (graph_weight_pages || std::max({q_weight_rows, k_weight_rows, v_weight_rows,
                o_weight_rows}) > target.memory().sram_depth_rows
            || q_weight_rows + k_weight_rows + v_weight_rows
                    + o_weight_rows
                > target.memory().sram_depth_rows);
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
    const bool serial_attention_weight_pages = tiled_weights
        && weight_tile_plan->transfer_phase_count >= 4;
    const bool compact_rope_products =
        target.uses_dedicated_slice_roles()
        && target.memory().banks_per_slice > 1
        && target.activation_storage_slices().size() >= 20;
    const int64_t head_blocks = head_dim / tile;
    // The physical SR fabric does not replay an operand into both
    // directions. Match the scheduler's explicit SRAM RoPE path.
    const bool direct_rope_streaming = false;
    const int64_t logical_head_banks = (head_blocks + 1) / 2;
    const int64_t key_rows =
        kv_storage_heads * logical_head_banks * kv_storage_tokens;
    const int64_t value_rows = kv_storage_heads * head_blocks
        * (kv_storage_tokens / tile) * target.throughput().tile_rows;
    const int64_t query_rows = query_heads * logical_head_banks * blocks
        * target.throughput().tile_rows;
    const int64_t rope_frequency_blocks = head_dim / (2 * tile);
    const int64_t rope_table_tokens = decode_past_len + seq_len;
    const int64_t rope_rows = rope_table_tokens * rope_frequency_blocks;
    // The overlapped schedule only retains one four-block projection wave.
    // Serial projection finishes every Q head before RoPE consumes any of
    // them, so each output block needs a distinct staging address.
    const int64_t rope_staging_rows =
        (projection_rope_overlap ? 4
                                 : std::max(query_heads, kv_heads) * head_blocks)
        * seq_len;
    const int64_t qk_norm_matrix_rows =
        seq_len * head_dim
        / (tile * target.throughput().lanes_per_tile);
    // Direct projection-to-RoPE streams never materialize the four explicit
    // product planes.  Keep a one-row placeholder in the memory plan so the
    // common placement schema remains valid without reserving an all-head
    // scratch tensor that hardware never accesses.
    const int64_t rope_product_rows = direct_rope_streaming
        ? 1
        : (query_heads + kv_heads) * (head_blocks / 2)
            * (compact_rope_products ? 2 : 4)
            * (compact_rope_products ? seq_len : (seq_len + 1) / 2);
    const int64_t score_rows = query_heads * blocks * kv_seq_len;
    const int64_t probability_pack_rows = query_heads * blocks
        * (kv_seq_len / target.throughput().lanes_per_tile);
    const int64_t kv_blocks = kv_seq_len / tile;
    const int64_t probability_diagonal_rows = query_heads * blocks
        * kv_blocks * target.throughput().tile_rows;
    const int64_t context_rows = query_heads
        * (compact_rope_products ? head_blocks : 1) * seq_len;
    const int64_t output_activation_rows =
        query_width / tile * blocks * (tile / block_rows);
    // fp16_mxm_distributed_16 allocates addresses by a full 32-token block:
    // each hidden block owns one address per tile row even when decode has a
    // single logical token.  Scratch and initialized constants must begin
    // after that padded physical span, otherwise decode RoPE initialization
    // overwrites the still-live residual input.
    const int64_t distributed_input_rows =
        ((seq_len + tile - 1) / tile) * (hidden / tile)
        * target.throughput().tile_rows;
    const int64_t output_activation_base = distributed_input_rows;
    // Serial matrix prefill can read token lanes directly from distributed
    // SRAM. Do not materialize a second full input in one planar MEM pair.
    auto inheritedInputPlacement = get_value_placement(graph.query.getLhs());
    const bool distributedInputAvailable = mlir::failed(inheritedInputPlacement)
        || inheritedInputPlacement->getAs<mlir::StringAttr>("kind").getValue()
            == "fp16_mxm_distributed_16";
    const bool direct_projection_input = !decode && !projection_rope_overlap
        && distributedInputAvailable && target.uses_dedicated_slice_roles()
        && target.throughput().mxms_per_hemisphere == 1;
    const int64_t input_staging_rows = direct_projection_input
        ? 0 : seq_len * hidden / tile;
    const int64_t input_staging_base =
        target.memory().words_per_bank - input_staging_rows;
    const int64_t output_staging_rows = seq_len * query_width / tile;
    const int64_t output_staging_base =
        target.memory().words_per_bank - output_staging_rows;
    // Eight head blocks put the rotary halves four output blocks apart;
    // four bias plane pairs keep those simultaneous reads on distinct ICUs.
    const bool wide_projection_bias = tile == 16 && (decode || head_blocks == 8);
    const auto projection_bias_rows = [&](int64_t width) {
        const int64_t output_blocks = (width + tile - 1) / tile;
        const int64_t blocks_per_group = wide_projection_bias ? 8 : 4;
        return 2 * ((output_blocks + blocks_per_group - 1)
            / blocks_per_group);
    };
    const int64_t query_bias_rows = query_bias
        ? projection_bias_rows(query_width) : 0;
    const int64_t key_bias_rows = key_bias
        ? projection_bias_rows(kv_width) : 0;
    const int64_t value_bias_rows = value_bias
        ? projection_bias_rows(kv_width) : 0;
    const int64_t projection_bias_total_rows =
        query_bias_rows + key_bias_rows + value_bias_rows;
    // The legacy attention constants assume an 8192-row bank.  Keep both
    // decode and prefill scratch inside a smaller dedicated bank; the
    // physical allocator still checks concurrent slice/bank live ranges.
    const bool compact_decode_sram = decode
        && target.uses_dedicated_slice_roles()
        && target.memory().words_per_bank <= 1024;
    const bool compact_prefill_sram = !decode
        && target.uses_dedicated_slice_roles()
        && target.memory().words_per_bank <= 2048;
    const int64_t context_base = compact_decode_sram
        ? target.memory().words_per_bank / 2
        : compact_prefill_sram
        ? std::max<int64_t>(64,
            target.memory().words_per_bank - context_rows)
        : target.attention_context_base_row();
    const int64_t probability_pack_base = compact_decode_sram
            || compact_prefill_sram
        ? target.memory().words_per_bank / 4
        : target.attention_probability_pack_base_row();
    const int64_t probability_diagonal_base = compact_decode_sram
            || compact_prefill_sram
        ? target.memory().words_per_bank / 2
        : target.attention_probability_diagonal_base_row();
    const int64_t projection_bias_base =
        context_base - projection_bias_total_rows;
    const int64_t query_bias_base = projection_bias_base;
    const int64_t key_bias_base = query_bias_base + query_bias_rows;
    const int64_t value_bias_base = key_bias_base + key_bias_rows;
    const int64_t qk_norm_gamma_base =
        input_staging_base - 2 * head_dim;
    const auto query_gamma = gamma_placements.find(query_norm_weight);
    const auto key_gamma = gamma_placements.find(key_norm_weight);
    const int64_t query_norm_gamma_base = query_gamma != gamma_placements.end()
        ? query_gamma->second.base_row : qk_norm_gamma_base;
    const int64_t key_norm_gamma_base =
        key_gamma != gamma_placements.end() ? key_gamma->second.base_row
                                           : qk_norm_gamma_base + head_dim;
    const int64_t query_norm_bank = query_gamma != gamma_placements.end()
        ? query_gamma->second.bank : scratch_bank;
    const int64_t key_norm_bank = key_gamma != gamma_placements.end()
        ? key_gamma->second.bank : scratch_bank;
    // Attention intermediates live in the activation partition. Keep their
    // low row above the persistent residual input so later weight pages may
    // remain resident in the dedicated weight slices throughout RoPE and
    // softmax.
    // Compact products use the four parameter/temporary slices, outside
    // the persistent distributed16 input. Their low rows can start at zero;
    // leaving an input-sized hole here needlessly displaces resident gamma.
    const int64_t rope_product_base = compact_rope_products ? 0 : distributed_input_rows;
    // Decode RoPE is compact and only needs two rows per resident token.
    // Its long-lived primary copy shares activation slices with the transient
    // score plane, so place it immediately after that plane. The persistent
    // Key cache continues to grow down from the high end of SRAM.
    // Decode's mirrored constants use the low activation slices and are
    // initialized before RMSNorm. Preserve both the live input plane and
    // its adjacent RMSNorm transpose scratch until the constants are read.
    const int64_t rope_mirror_base = decode
        ? std::max(rope_product_base + rope_product_rows,
            2 * distributed_input_rows)
        : rope_product_base + rope_product_rows;
    const int64_t rope_staging_mirror_base = rope_mirror_base + rope_rows;
    // The function input remains live until the attention residual add. Keep
    // Q/V and score scratch above its distributed16 rows. The RoPE table and
    // causal mask are initialized before execution, so the mask must also sit
    // below the late input-staging window instead of relying on stage lifetime
    // reuse.
    const int64_t score_base = target.uses_dedicated_slice_roles()
        ? (compact_decode_sram ? 0
            : std::max(distributed_input_rows, key_rows))
        : target.attention_score_base_row();
    // Compact Query IW occupies distributed activation slices; scores occupy
    // the separate planar score slices. Their row ranges need not be stacked.
    // Dual-MXM QK needs two score planes concurrently with both Query
    // halves. Those allocations share activation slices, so preserve their
    // disjoint row ranges even on a compact prefill target.
    const int64_t query_base = compact_decode_sram
            || (compact_prefill_sram
                && target.throughput().mxms_per_hemisphere == 1)
        ? distributed_input_rows : score_base + score_rows;
    // Key low-half products share the distributed input/Query slice group.
    // Place them above both live tensors so another head batch can replay
    // the same normalized input without an external reload.
    const int64_t key_product_base = compact_prefill_sram
        ? query_base + query_rows : rope_product_base;
    const int64_t key_product_rows = compact_rope_products
        ? kv_heads * (head_blocks / 2) * 2 * seq_len : rope_product_rows;
    const int64_t rope_table_base = decode
        ? query_base
        : compact_prefill_sram
        ? rope_product_base + rope_product_rows
        : probability_diagonal_base;
    // Serial Q/K projection retains every raw head until RoPE starts. Its
    // larger staging allocation must stay clear of the Query IW rows, which
    // remain live for QK and can occupy either scratch bank by rotary half.
    const int64_t rope_staging_base = projection_rope_overlap
        ? distributed_input_rows : query_base + query_rows;
    // Value is independent of QK and may be projected before Key so the
    // following layer's weight refill can overlap the rest of attention.
    // Keep it beyond the transient RoPE product plane for that longer live
    // range; half-open row intervals still allow exact adjacency.
    const int64_t transient_value_base = std::max(
        query_base + query_rows, rope_product_base + rope_product_rows);
    // A persistent cache is paged out only after the executable finishes, so
    // its SRAM window must not be reused by later decoder stages. Keep it at
    // the high end of its slice group; ordinary attention scratch grows from
    // low rows and can retain its existing lifetime reuse.
    const int64_t value_base = kv_cache_enabled
        ? target.memory().words_per_bank - value_rows
        : transient_value_base;
    const int64_t qk_norm_scratch_base = std::max(
        kv_cache_enabled ? transient_value_base : value_base + value_rows,
        rope_mirror_base + rope_rows);
    const int64_t qk_norm_input_base = qk_norm_scratch_base;
    const int64_t qk_norm_output_base = qk_norm_scratch_base;
    const int64_t causal_mask_base = target.uses_dedicated_slice_roles()
        ? rope_rows : target.attention_mask_base_row();
    if (value_base < 0 || value_base + value_rows
            > target.memory().words_per_bank
        || (has_qk_norm
            && (qk_norm_input_base + qk_norm_matrix_rows + 1
                    > input_staging_base
                || qk_norm_output_base + qk_norm_matrix_rows
                    > input_staging_base))
        || rope_staging_base + rope_staging_rows
            > target.memory().words_per_bank
        || rope_product_base + rope_product_rows
            > target.memory().words_per_bank
        || (compact_rope_products
            && rope_mirror_base + rope_rows > target.memory().words_per_bank)
        || causal_mask_base + tile > input_staging_base) {
        op.emitError(
            "attention activation scratch does not fit around persistent "
            "input and initialized constants")
            << ": bank_rows=" << target.memory().words_per_bank
            << ", input_staging_rows=" << input_staging_rows
            << ", rope_product_rows=" << rope_product_rows
            << ", context_rows=" << context_rows
            << ", score_end=" << score_base + score_rows
            << ", query_end=" << query_base + query_rows;
        return mlir::failure();
    }
    llvm::SmallVector<int64_t, 36> scratch_candidates;
    scratch_candidates = target.activation_storage_slices();
    const llvm::SmallVector<int64_t, 2> scratch_banks {
        scratch_bank, secondary_scratch_bank};
    tensor::PhysicalMemoryAllocator physical_allocator(target);
    // These compact gamma images are function inputs and remain resident
    // until the post-attention RMSNorm materializes its local read layout.
    for (const auto& entry : gamma_placements) {
        const auto& gamma = entry.second;
        if (gamma.segment_rows <= 0) continue;
        if (mlir::failed(physical_allocator.reserve({"resident_post_norm_gamma",
                {gamma.slice_base, gamma.slice_base + 1,
                    gamma.slice_base + 2, gamma.slice_base + 3},
                gamma.base_row, gamma.segment_rows, 0, 7, false, gamma.bank}))) {
            op.emitError("post-attention gamma conflicts with attention storage");
            return mlir::failure();
        }
    }
    const int64_t value_bank = compact_rope_products
        ? secondary_scratch_bank : scratch_bank;
    const bool split_value_banks = tiled_weights
        && target.throughput().mxms_per_hemisphere == 2;
    if (kv_cache_enabled) {
        llvm::SmallVector<int64_t, 16> value_cache_slices;
        for (int64_t slice : target.attention_value_slices())
            if (!llvm::is_contained(value_cache_slices, slice))
                value_cache_slices.push_back(slice);
        const bool valueReservationFailed =
            mlir::failed(physical_allocator.reserve({"value_cache",
                value_cache_slices, value_base, value_rows, 0, 7, false,
                value_bank}))
            || (split_value_banks
                && mlir::failed(physical_allocator.reserve({"value_cache_second_bank",
                    value_cache_slices, value_base, value_rows, 0, 7,
                    false, scratch_bank})));
        if (valueReservationFailed) {
            op.emitError(
                "persistent Value cache overlaps attention scratch memory");
            return mlir::failure();
        }
    }
    const auto weightStorage = target.weight_storage_slices();
    const bool can_use_disjoint_q_rope_slices =
        !decode && query_bias && tiled_weights && compact_rope_products &&
        target.uses_dedicated_slice_roles() &&
        target.memory().banks_per_slice == 2 && !has_qk_norm &&
        seq_len > 0 && seq_len <= tile && hidden == 1536 &&
        query_heads == 12 &&
        kv_heads == 2 && head_dim == 128 && weight_bank == 1 &&
        target.name() == "ftlpu-lpu32" && weightStorage.size() >= 30 &&
        weightStorage[0] == 20 && weightStorage[8] == 28 &&
        weightStorage[12] == 32 && weightStorage[22] == 42 &&
        weightStorage[29] == 49 &&
        weight_tile_plan->get(tensor::AttentionWeightTileKind::Output).bank ==
            scratch_bank &&
        weight_tile_plan->get(tensor::AttentionWeightTileKind::Output)
                .slice_group_begin == 0 &&
        weight_tile_plan->get(tensor::AttentionWeightTileKind::Output)
                .slice_group_count == 1;
    const bool use_disjoint_q_rope_slices =
        projection_rope_overlap && can_use_disjoint_q_rope_slices;
    const bool split_serial_q_rope_staging =
        !projection_rope_overlap && can_use_disjoint_q_rope_slices &&
        head_blocks == 4;
    llvm::SmallVector<int64_t, 16> projection_bias_slices;
    if (has_attention_bias) {
        const auto storage = target.activation_storage_slices();
        if (storage.size() < 20 || projection_bias_base < 0) {
            op.emitError(
                "attention projection bias requires four activation slices "
                "and reserved rows below the context buffer");
            return mlir::failure();
        }
        // Q's overlapping mirror or serial alternate staging uses bank-0
        // activation slices. Keep projection-bias reads on free bank-0 weight
        // slices beyond the pages runtime may preload before Q. The output
        // page owns slices 20..27; FFN Down uses a subset of 31..41.
        if (use_disjoint_q_rope_slices || split_serial_q_rope_staging) {
            projection_bias_slices.assign(
                weightStorage.begin() + 22, weightStorage.begin() + 26);
        } else {
            const int64_t slice_begin = wide_projection_bias ? 8 : 12;
            projection_bias_slices.assign(
                storage.begin() + slice_begin,
                storage.begin() + slice_begin
                    + (wide_projection_bias ? 8 : 4));
        }
    }
    llvm::SmallVector<int64_t, 16> qk_norm_input_slices;
    llvm::SmallVector<int64_t, 16> qk_norm_output_slices;
    llvm::SmallVector<int64_t, 2> qk_norm_weight_slices;
    if (has_qk_norm) {
        const auto storage = target.activation_storage_slices();
        const auto norm_scratch =
            target.mxm_distributed_activation_slices();
        if (storage.size() < 2 || qk_norm_gamma_base < 0
            || norm_scratch.size() < 16) {
            op.emitError(
                "Q/K head RMSNorm requires 16 activation scratch slices "
                "and two activation gamma slices");
            return mlir::failure();
        }
        qk_norm_input_slices.assign(
            norm_scratch.begin(), norm_scratch.begin() + 16);
        qk_norm_output_slices.assign(
            norm_scratch.begin(), norm_scratch.begin() + 16);
        qk_norm_weight_slices.assign(
            storage.end() - 2, storage.end());
    }
    const int64_t projection_bias_bank = scratch_bank;
    const int64_t input_staging_bank = scratch_bank;
    const auto input_staging_values = target.uses_dedicated_slice_roles()
        ? target.activation_storage_slices() : planar_activation_slices;
    llvm::SmallVector<int64_t, 16> input_staging_slices;
    if (tiled_weights && target.throughput().mxms_per_hemisphere == 2
        && weightStorage.size() >= 2) {
        // Keep the last staged input read off the Q result MEM ICUs.
        input_staging_slices.assign(
            weightStorage.end() - 2, weightStorage.end());
    } else {
        const auto input_staging_begin = direct_rope_streaming
            ? input_staging_values.end() - 2
            : compact_rope_products
                ? input_staging_values.begin() + 8
                : input_staging_values.begin();
        input_staging_slices.assign(
            input_staging_begin, input_staging_begin + 2);
    }
    if (!direct_projection_input && mlir::failed(physical_allocator.reserve({"input_staging",
            input_staging_slices, input_staging_base,
            input_staging_rows, 0, 2, true, input_staging_bank}))) {
        op.emitError("failed to reserve the attention input staging buffer");
        return mlir::failure();
    }
    llvm::SmallVector<int64_t, 16> input_staging_pong_slices;
    if (projection_rope_overlap && tiled_weights
        && target.throughput().mxms_per_hemisphere == 1
            && compact_rope_products && !direct_rope_streaming
            && input_staging_values.size() >= 10) {
        // Keep both copies inside the activation partition.  The primary
        // pair (8/9) is preferred for its shorter MXM route; the 0/1 copy is
        // used only for issue cycles where the previous group's RoPE owns a
        // primary MEM ICU.  Equal row coordinates make the switch affine.
        input_staging_pong_slices.assign(
            input_staging_values.begin(), input_staging_values.begin() + 2);
        if (mlir::failed(physical_allocator.reserve({"input_staging_pong",
                input_staging_pong_slices, input_staging_base,
                input_staging_rows, 0, 2, true, input_staging_bank}))) {
            op.emitError(
                "failed to reserve the attention input staging pong buffer");
            return mlir::failure();
        }
    }
    // Keep the long-lived O-projection weights out of the fixed probability
    // staging window. Larger models can make O weights cross that window even
    // though the compact SmolLM2 shape does not.
    const int64_t output_weight_base = tiled_weights ? 0 : paged_weights
        ? q_weight_rows + k_weight_rows + v_weight_rows
        : std::max({q_weight_rows + k_weight_rows + v_weight_rows,
              score_base + score_rows,
              probability_pack_base
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
    const auto staging_slice_values =
        target.attention_rope_staging_slices();
    const llvm::SmallVector<int64_t, 16> rope_staging_slices(
        staging_slice_values.begin(), staging_slice_values.end());
    llvm::SmallVector<int64_t, 16> alternate_rope_staging_slices(
        rope_staging_slices.begin(), rope_staging_slices.end());
    if (split_serial_q_rope_staging) {
        // The long Q activation READ_3D owns staging slices 8/9 until the
        // projection ends. Keep the alternate raw output off those ICUs.
        alternate_rope_staging_slices[8] = weightStorage[8];
        alternate_rope_staging_slices[9] = weightStorage[9];
    }
    // Explicit RoPE products alternate banks. Keep the table on the opposite
    // bank from the first product pair so an II=1 phase can read the table and
    // write products without asking one SRAM bank for both operations.
    const int64_t rope_staging_bank =
        target.uses_dedicated_slice_roles()
            ? secondary_scratch_bank : scratch_bank;
    // A serial Q projection can retain each rotary half on a different
    // staging bank. Query IW already uses opposite banks for its two halves;
    // place each raw half opposite its own final Query half. The other half
    // can still use the same ICU, so this is not a global read/write split.
    const int64_t alternate_rope_staging_base =
        rope_staging_mirror_base;
    const int64_t rope_product_bank =
        compact_rope_products && direct_rope_streaming
            ? secondary_scratch_bank : scratch_bank;
    llvm::SmallVector<int64_t, 16> rope_product_slices;
    llvm::SmallVector<int64_t, 4> rope_product_key_slices;
    if (compact_rope_products) {
        const auto storage = target.activation_storage_slices();
        rope_product_slices.assign(
            storage.begin() + 16, storage.begin() + 20);
        rope_product_key_slices.assign(
            storage.begin() + 4, storage.begin() + 8);
    } else {
        const auto values = target.mxm_distributed_activation_slices();
        rope_product_slices.assign(values.begin(), values.end());
        rope_product_key_slices.assign(
            values.begin(), values.begin() + 4);
    }
    llvm::SmallVector<int64_t, 16> probability_diagonal_slices;
    if (tiled_weights && target.uses_dedicated_slice_roles()) {
        const auto storage = target.activation_storage_slices();
        if (storage.size() < 16) {
            op.emitError(
                "target needs 16 activation slices for attention staging");
            return mlir::failure();
        }
        probability_diagonal_slices.assign(
            storage.begin(), storage.begin() + 16);
    } else {
        const auto values = target.attention_query_iw_slices(0);
        probability_diagonal_slices.assign(values.begin(), values.end());
    }
    const int64_t probability_diagonal_bank = scratch_bank;
    llvm::SmallVector<int64_t, 4> rope_table_slices;
    llvm::SmallVector<int64_t, 4> rope_mirror_slices;
    llvm::SmallVector<int64_t, 16> rope_staging_mirror_slices;
    int64_t rope_table_bank = compact_rope_products
        ? secondary_scratch_bank : scratch_bank;
    int64_t rope_mirror_bank = compact_rope_products
        ? scratch_bank : rope_table_bank;
    const auto rope_slices = target.attention_rope_slices();
    rope_table_slices.assign(rope_slices.begin(), rope_slices.end());
    if (split_serial_q_rope_staging) {
        // Decode Q postprocessing writes the activation-region slices while
        // consuming RoPE constants. Keep both constant copies on otherwise
        // idle scratch-bank weight-region ICUs so neither non-preemptible MEM
        // program has to interleave with a Q write.
        rope_table_slices.assign(
            weightStorage.begin() + 26, weightStorage.begin() + 30);
        rope_table_bank = scratch_bank;
    }
    if (compact_rope_products) {
        if (split_serial_q_rope_staging) {
            rope_mirror_slices.assign(
                weightStorage.begin() + 18, weightStorage.begin() + 22);
            rope_mirror_bank = scratch_bank;
        } else if (use_disjoint_q_rope_slices)
            rope_mirror_slices.assign(
                weightStorage.begin() + 26, weightStorage.begin() + 30);
        if (use_disjoint_q_rope_slices) {
            // Keep the four busy staging mirrors away from both
            // pre-execution weight pages and the Q bias streams.
            rope_staging_mirror_slices.assign(
                rope_staging_slices.begin(), rope_staging_slices.end());
            llvm::SmallVector<int64_t, 4> busyActivationSlices;
            busyActivationSlices.append(input_staging_slices.begin(),
                                        input_staging_slices.end());
            busyActivationSlices.append(input_staging_pong_slices.begin(),
                                        input_staging_pong_slices.end());
            std::sort(busyActivationSlices.begin(),
                      busyActivationSlices.end());
            busyActivationSlices.erase(
                std::unique(busyActivationSlices.begin(),
                            busyActivationSlices.end()),
                busyActivationSlices.end());
            if (busyActivationSlices.size() != 4 ||
                rope_staging_mirror_base + rope_staging_rows >
                    target.memory().words_per_bank) {
                op.emitError("cannot place the Q RoPE staging mirror");
                return mlir::failure();
            }
            for (int64_t index = 0; index < 4; ++index) {
                auto it = std::find(rope_staging_slices.begin(),
                                    rope_staging_slices.end(),
                                    busyActivationSlices[index]);
                if (it == rope_staging_slices.end()) {
                    op.emitError("Q staging slice is absent from RoPE staging");
                    return mlir::failure();
                }
                rope_staging_mirror_slices[std::distance(
                    rope_staging_slices.begin(), it)] =
                    weightStorage[8 + index + (index == 3 ? 1 : 0)];
            }
        } else if (!split_serial_q_rope_staging) {
            const auto storage = target.activation_storage_slices();
            rope_mirror_slices.assign(storage.begin(), storage.begin() + 4);
        }
    } else {
        rope_mirror_slices = rope_table_slices;
    }
    if (mlir::failed(physical_allocator.reserve({"rope_table",
            llvm::SmallVector<int64_t, 16>(
                rope_table_slices.begin(), rope_table_slices.end()),
            rope_table_base, rope_rows, 0, 7, false, rope_table_bank}))) {
        op.emitError("failed to reserve the attention RoPE table");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"rope_staging",
            rope_staging_slices, rope_staging_base,
            rope_staging_rows, 0, 2, true,
            rope_staging_bank}))) {
        op.emitError("failed to reserve the attention RoPE staging FIFO");
        return mlir::failure();
    }
    if (split_serial_q_rope_staging
        && mlir::failed(physical_allocator.reserve({
            "rope_staging_alternate", alternate_rope_staging_slices,
            alternate_rope_staging_base, rope_staging_rows, 0, 2, false,
            scratch_bank}))) {
        op.emitError("failed to reserve the alternate Q RoPE staging bank");
        return mlir::failure();
    }
    if (use_disjoint_q_rope_slices &&
        mlir::failed(physical_allocator.reserve({"rope_staging_mirror",
            rope_staging_mirror_slices, rope_staging_mirror_base,
            rope_staging_rows, 0, 2, false, scratch_bank}))) {
        op.emitError("failed to reserve the Q RoPE staging mirror");
        return mlir::failure();
    }
    if (has_qk_norm
        && (mlir::failed(physical_allocator.reserve({"qk_norm_input",
                qk_norm_input_slices, qk_norm_input_base,
                qk_norm_matrix_rows + 1, 0, 2, false, scratch_bank}))
            || mlir::failed(physical_allocator.reserve({"qk_norm_output",
                qk_norm_output_slices, qk_norm_output_base,
                qk_norm_matrix_rows, 0, 2, false,
                secondary_scratch_bank})))) {
        op.emitError("failed to reserve Q/K head RMSNorm scratch");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"rope_product",
            rope_product_slices, rope_product_base, rope_product_rows,
            0, 2, !compact_rope_products,
            rope_product_bank}))) {
        op.emitError("failed to reserve the attention RoPE product FIFO");
        return mlir::failure();
    }
    int64_t next_bias_base = projection_bias_base;
    const auto reserve_bias = [&](llvm::StringRef name, int64_t rows) {
        if (rows == 0) return mlir::success();
        const int64_t base = next_bias_base;
        next_bias_base += rows;
        return physical_allocator.reserve({name.str(),
            projection_bias_slices, base, rows, 0, 2, false,
            projection_bias_bank});
    };
    if (mlir::failed(reserve_bias("query_bias", query_bias_rows))
        || mlir::failed(reserve_bias("key_bias", key_bias_rows))
        || mlir::failed(reserve_bias("value_bias", value_bias_rows))) {
        op.emitError(
            "failed to reserve the attention projection-bias constants");
        return mlir::failure();
    }
    auto query_norm_weight_slices = qk_norm_weight_slices;
    auto key_norm_weight_slices = qk_norm_weight_slices;
    if (query_gamma != gamma_placements.end())
        query_norm_weight_slices = {query_gamma->second.slice_base,
                                   query_gamma->second.slice_base + 1};
    if (key_gamma != gamma_placements.end())
        key_norm_weight_slices = {key_gamma->second.slice_base,
                                 key_gamma->second.slice_base + 1};
    if (has_qk_norm
        && (mlir::failed(physical_allocator.reserve({"query_norm_weight",
                llvm::SmallVector<int64_t, 16>(query_norm_weight_slices.begin(),
                    query_norm_weight_slices.end()), query_norm_gamma_base, head_dim,
                0, 3, false, query_norm_bank}))
            || mlir::failed(physical_allocator.reserve({"key_norm_weight",
                llvm::SmallVector<int64_t, 16>(key_norm_weight_slices.begin(),
                    key_norm_weight_slices.end()), key_norm_gamma_base, head_dim,
                0, 3, false, key_norm_bank})))) {
        op.emitError("failed to reserve Q/K head RMSNorm gamma constants");
        return mlir::failure();
    }
    if (compact_rope_products
        && mlir::failed(physical_allocator.reserve({"rope_product_alternate",
            rope_product_slices, rope_product_base, rope_product_rows,
            0, 2, false,
            (rope_product_bank + 1) % target.memory().banks_per_slice}))) {
        op.emitError(
            "failed to reserve the interleaved RoPE product FIFO");
        return mlir::failure();
    }
    if (compact_rope_products
        && mlir::failed(physical_allocator.reserve({"rope_product_key",
            llvm::SmallVector<int64_t, 16>(
                rope_product_key_slices.begin(),
                rope_product_key_slices.end()),
            key_product_base, key_product_rows,
            0, 2, false, scratch_bank}))) {
        op.emitError("failed to reserve the key RoPE product FIFO");
        return mlir::failure();
    }
    if (compact_rope_products
        && mlir::failed(physical_allocator.reserve({"rope_mirror",
            llvm::SmallVector<int64_t, 16>(
                rope_mirror_slices.begin(), rope_mirror_slices.end()),
            rope_mirror_base, rope_rows,
            0, 2, false, rope_mirror_bank}))) {
        op.emitError("failed to reserve the mirrored attention RoPE table");
        return mlir::failure();
    }
    if (mlir::failed(physical_allocator.reserve({"probability_diagonal",
            probability_diagonal_slices,
            probability_diagonal_base,
            query_heads * blocks * blocks
                * target.throughput().tile_rows,
            4, 5, true, probability_diagonal_bank}))) {
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
    // Resident attention followed by runtime-paged FFN uses the weight plane
    // in both SRAM banks.  Keep the complete tail-Softmax working set in the
    // activation plane: score/mask inputs use bank A and the probability pack
    // uses bank B.  This leaves every weight-plane MEM ICU available for the
    // long DDR-to-C2C Gate prefetch.
    const bool isolate_probability_from_runtime_weight_plane =
        paged_weights
        && target.uses_dedicated_slice_roles()
        && target.memory().banks_per_slice > 1
        && activation_storage.size() >= 16;
    const int64_t fused_slice_base = target.uses_dedicated_slice_roles()
        ? activation_storage.front()
        : target.memory().slices_per_hemisphere - 16;
    const int64_t fused_scratch_bank =
        target.uses_dedicated_slice_roles()
        ? secondary_scratch_bank
        : scratch_bank;
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
    // These placements belong to the alternative fused-attention schedule.
    // The resident-attention/runtime-paged-FFN layout selects the tail
    // schedule, whose score/mask inputs and probability output intentionally
    // occupy opposite activation banks. Do not make the unused fused variant
    // consume those same allocator lifetimes.
    if (!isolate_probability_from_runtime_weight_plane)
        for (const auto& reservation : {
             tensor::PhysicalAllocation {"fused_score",
                  fused_score0, score_base,
                  score_rows, 2, 3, true, fused_scratch_bank},
             tensor::PhysicalAllocation {"fused_score_bank1",
                  fused_score1, score_base,
                  score_rows, 2, 3, true, fused_scratch_bank},
             tensor::PhysicalAllocation {"fused_causal_mask",
                  fused_mask0, causal_mask_base,
                  tile - 1, 2, 3, true, fused_scratch_bank},
             tensor::PhysicalAllocation {"fused_causal_mask_bank1",
                  fused_mask1, causal_mask_base,
                  tile - 1, 2, 3, true, fused_scratch_bank}}) {
        if (mlir::failed(physical_allocator.reserve(reservation))) {
            op.emitError("failed to reserve fused attention scratch");
            return mlir::failure();
        }
        }
    const int64_t key_bank = target.uses_dedicated_slice_roles()
        ? secondary_scratch_bank : scratch_bank;
    const llvm::SmallVector<int64_t, 1> key_candidate_banks {key_bank};
    llvm::SmallVector<int64_t, 32> qk_excluded_slices;
    const int64_t query_segment_banks[2] {
        scratch_bank,
        target.uses_dedicated_slice_roles()
                && target.memory().banks_per_slice >= 2
            ? (scratch_bank + 1) % target.memory().banks_per_slice
            : scratch_bank,
    };
    for (int64_t group = 0; group < 2; ++group) {
        const auto querySlices = target.attention_query_iw_slices(group);
        // Query is retained from projection completion through QK. Its
        // production within stage 0..2 is ordered against transient staging
        // by the projection scheduler; persistent constants span both phases.
        if (mlir::failed(physical_allocator.reserve({
                "query_iw_half_" + std::to_string(group),
                llvm::SmallVector<int64_t, 16>(querySlices.begin(), querySlices.end()),
                query_base, query_rows, 2, 3, false, query_segment_banks[group]}))) {
            op.emitError("Query IW storage overlaps a live attention allocation");
            return mlir::failure();
        }
    }
    for (int64_t group = 0; group < 2; ++group) {
        if (query_segment_banks[group] != key_bank) continue;
        for (int64_t slice : target.attention_query_iw_slices(group))
            if (!llvm::is_contained(qk_excluded_slices, slice))
                qk_excluded_slices.push_back(slice);
    }
    // Key and Query are consumed concurrently by QK. Allocate Key from the
    // activation pool while excluding every Query slice on the same bank.
    // The exact cycle-level MEM ICU check is performed by the QK planner.
    const auto allocate_key = [&](int64_t base_row) {
        return physical_allocator.allocate({"key",
            static_cast<int64_t>(target.throughput().mxm_result_streams),
            base_row, key_rows,
            kv_cache_enabled ? 0 : 2, kv_cache_enabled ? 7 : 3,
            scratch_candidates, false, key_candidate_banks,
            qk_excluded_slices});
    };
    const int64_t preferred_key_base = kv_cache_enabled
        ? target.memory().words_per_bank - key_rows : 0;
    auto key_allocation = allocate_key(preferred_key_base);
    if (kv_cache_enabled && mlir::failed(key_allocation)) {
        for (int64_t base_row = preferred_key_base - 1;
             base_row >= 0; --base_row) {
            key_allocation = allocate_key(base_row);
            if (mlir::succeeded(key_allocation)) break;
        }
    }
    if (mlir::failed(key_allocation)) {
        op.emitError(
            "cannot place QK Key storage outside concurrent Query MEM ICU "
            "slices");
        return mlir::failure();
    }
    const auto allocate_scratch_in_banks = [&](llvm::StringRef name,
                                               int64_t slice_count,
                                               int64_t base_row,
                                               int64_t rows,
                                               int64_t live_start,
                                               int64_t live_end,
                                               llvm::ArrayRef<int64_t> banks)
        -> mlir::FailureOr<tensor::PhysicalAllocation> {
        return physical_allocator.allocate({name.str(), slice_count,
            base_row, rows, live_start, live_end, scratch_candidates,
            true, banks});
    };
    const auto allocate_scratch = [&](llvm::StringRef name,
                                      int64_t slice_count,
                                      int64_t base_row,
                                      int64_t rows,
                                      int64_t live_start,
                                      int64_t live_end)
        -> mlir::FailureOr<tensor::PhysicalAllocation> {
        return allocate_scratch_in_banks(name, slice_count, base_row, rows,
            live_start, live_end, scratch_banks);
    };
    // The final QK partial streams scores while the next wave reloads Query
    // IW. Query's first rotary half is on scratch_bank, so keep tail-softmax
    // scores on the other bank and let both MEM ICU queues run concurrently.
    const llvm::SmallVector<int64_t, 1> qk_score_banks {
        isolate_probability_from_runtime_weight_plane
            ? scratch_bank : secondary_scratch_bank};
    llvm::SmallVector<int64_t, 4> isolated_score_slices;
    if (isolate_probability_from_runtime_weight_plane)
        isolated_score_slices.assign(activation_storage.end() - 4, activation_storage.end());
    auto score0 = isolate_probability_from_runtime_weight_plane
        ? physical_allocator.allocate({"score_mxm0", 4, score_base,
            score_rows, 2, 3, isolated_score_slices, true, qk_score_banks})
        : allocate_scratch_in_banks("score_mxm0", 4, score_base,
            score_rows, 2, 3, qk_score_banks);
    auto score1 = target.throughput().mxms_per_hemisphere == 1 ? score0 : allocate_scratch_in_banks(
        "score_mxm1", 4, score_base,
        score_rows, 2, 3, qk_score_banks);
    auto exp0 = allocate_scratch(
        "exp_mxm0", 4, score_base,
        score_rows, 2, 3);
    auto exp1 = target.throughput().mxms_per_hemisphere == 1 ? exp0 : allocate_scratch(
        "exp_mxm1", 4, score_base,
        score_rows, 2, 3);
    // Mask initializers survive until softmax. A following paged FFN can
    // prefetch into every weight slice before that point, so paged attention
    // keeps masks in the activation region beyond its Q/K scratch instead.
    // Low activation rows still belong to the input/residual tensors.
    const bool mask_in_activation = paged_weights || tiled_weights;
    const llvm::SmallVector<int64_t, 1> mask_banks {scratch_bank};
    const auto allocate_mask = [&](llvm::StringRef name, int64_t offset)
        -> mlir::FailureOr<tensor::PhysicalAllocation> {
        if (offset < 0 || offset + 2
                > static_cast<int64_t>(rope_table_slices.size()))
            return mlir::failure();
        const llvm::SmallVector<int64_t, 2> candidates {
            mask_in_activation ? activation_storage[12+offset]
                : isolate_probability_from_runtime_weight_plane
                ? weightStorage[weightStorage.size() - 4 + offset]
                : rope_table_slices[static_cast<std::size_t>(offset)],
            mask_in_activation ? activation_storage[13+offset]
                : isolate_probability_from_runtime_weight_plane
                ? weightStorage[weightStorage.size() - 3 + offset]
                : rope_table_slices[static_cast<std::size_t>(offset + 1)]};
        return physical_allocator.allocate({name.str(), 2,
            mask_in_activation ? qk_norm_scratch_base+qk_norm_matrix_rows+1 : causal_mask_base, tile - 1, 0, 3, candidates, true,
            mask_banks});
    };
    auto mask0 = allocate_mask("causal_mask_mxm0", 0);
    auto mask1 = allocate_mask("causal_mask_mxm1", 2);
    const auto require_allocation = [&](bool allocated,
                                        llvm::StringRef name) {
        if (allocated) return mlir::success();
        op.emitError("attention scratch memory allocation failed for ")
            << name;
        return mlir::failure();
    };
    if (mlir::failed(require_allocation(
        mlir::succeeded(score0), "score_mxm0"))
        || mlir::failed(require_allocation(
            mlir::succeeded(score1), "score_mxm1"))
        || mlir::failed(require_allocation(
            mlir::succeeded(exp0), "exp_mxm0"))
        || mlir::failed(require_allocation(
            mlir::succeeded(exp1), "exp_mxm1"))
        || mlir::failed(require_allocation(
            mlir::succeeded(mask0), "causal_mask_mxm0"))
        || mlir::failed(require_allocation(
            mlir::succeeded(mask1), "causal_mask_mxm1")))
        return mlir::failure();

    // Softmax broadcasts the score scalar and streams the exponentials while
    // it writes the packed probability vectors.  These accesses overlap in
    // time, so the probability-pack window must not reuse either input's MEM
    // slices on the same bank. For the bank-ping-pong Qwen layout, Q/K/V have
    // drained by this phase: reuse their bank-A groups and leave bank B's
    // complete weight plane available to the incoming Gate page.
    llvm::SmallVector<int64_t, 64> probability_pack_candidates;
    const bool reuse_attention_weight_plane_for_probability =
        paged_weights && tiled_weights && !isolate_probability_from_runtime_weight_plane
        && weight_bank == secondary_scratch_bank
        && weightStorage.size() >= 16;
    // A resident attention layer can still be followed by runtime-paged FFN
    // weights. Keep probability scratch in the activation slice partition so
    // a long DDR transfer never owns the same MEM ICUs as Softmax. The second
    // SRAM bank supplies the disjoint pack/diagonal ping-pong without using
    // any of the page-resident weight slices.
    if (reuse_attention_weight_plane_for_probability) {
        probability_pack_candidates.append(
            weightStorage.begin(), weightStorage.begin() + 16);
    } else {
        probability_pack_candidates.append(
            activation_storage.begin(), activation_storage.end());
    }
    if (!reuse_attention_weight_plane_for_probability
        && !isolate_probability_from_runtime_weight_plane && paged_weights
        && target.uses_dedicated_slice_roles()) {
        const auto weight_storage = target.weight_storage_slices();
        probability_pack_candidates.append(
            weight_storage.begin(), weight_storage.end());
    }
    llvm::SmallVector<int64_t, 32> probability_pack_excluded;
    const int64_t requested_probability_pack_bank =
        reuse_attention_weight_plane_for_probability
            || isolate_probability_from_runtime_weight_plane
        ? secondary_scratch_bank : scratch_bank;
    const auto exclude_softmax_input = [&](const auto& allocation) {
        if (allocation->bank != requested_probability_pack_bank) return;
        for (int64_t slice : allocation->slices)
            if (!llvm::is_contained(probability_pack_excluded, slice))
                probability_pack_excluded.push_back(slice);
    };
    exclude_softmax_input(score0);
    exclude_softmax_input(score1);
    exclude_softmax_input(exp0);
    exclude_softmax_input(exp1);
    exclude_softmax_input(mask0);
    exclude_softmax_input(mask1);
    // The pack-to-diagonal transpose pipelines reads and writes across its
    // nominal phase boundary.  Keep those slice sets disjoint as well.
    if (probability_diagonal_bank == requested_probability_pack_bank)
        for (int64_t slice : probability_diagonal_slices)
            if (!llvm::is_contained(probability_pack_excluded, slice))
                probability_pack_excluded.push_back(slice);
    // Value survives Softmax until PV consumes it. Its transient placement
    // is not reserved by the allocator, so explicitly keep an overlapping
    // probability window off the same physical bank/slice/row cells.
    if (value_bank == requested_probability_pack_bank
        && probability_pack_base < value_base + value_rows
        && value_base < probability_pack_base + probability_pack_rows) {
        for (int64_t slice : target.attention_value_slices())
            if (!llvm::is_contained(probability_pack_excluded, slice))
                probability_pack_excluded.push_back(slice);
    }
    const llvm::SmallVector<int64_t, 1> probability_pack_banks {
        requested_probability_pack_bank};
    auto probability_pack = physical_allocator.allocate({"probability_pack",
        16, probability_pack_base,
        probability_pack_rows, 3, 4, probability_pack_candidates,
        true, probability_pack_banks, probability_pack_excluded});
    if (mlir::failed(probability_pack)) {
        op.emitError(
            "cannot place the attention probability pack outside concurrent "
            "Softmax score/exp MEM slices");
        return mlir::failure();
    }
    probability_pack_slices.assign(
        probability_pack->slices.begin(), probability_pack->slices.end());
    const int64_t probability_pack_bank = probability_pack->bank;

    auto context = [&]() -> mlir::FailureOr<tensor::PhysicalAllocation> {
        if (!compact_rope_products)
            return allocate_scratch(
                "context",
                static_cast<int64_t>(
                    target.attention_context_slices().size()),
                context_base, context_rows, 4, 6);
        const auto storage = target.activation_storage_slices();
        const llvm::SmallVector<int64_t, 4> context_candidates {
            storage[16], storage[17], storage[18], storage[19]};
        // The long-lived RoPE table and the alternate product FIFO fill the
        // other bank on these four slices. The primary product retires before
        // context starts, so its bank can be reused without an ICU conflict.
        const llvm::SmallVector<int64_t, 1> context_banks {
            compact_prefill_sram ? scratch_bank : secondary_scratch_bank};
        return physical_allocator.allocate({"context", 4,
            context_base, context_rows, 4, 6,
            context_candidates, true, context_banks});
    }();
    if (mlir::failed(require_allocation(
            mlir::succeeded(context), "context")))
        return mlir::failure();
    const bool split_context_banks = split_value_banks
        && compact_rope_products && head_blocks == 2;
    if (split_context_banks
        && mlir::failed(physical_allocator.reserve({"context_second_bank",
            context->slices, context_base, context_rows / 2,
            4, 6, true, secondary_scratch_bank}))) {
        op.emitError("cannot place the second attention context half");
        return mlir::failure();
    }
    mlir::FailureOr<tensor::PhysicalAllocation> gate_input = mlir::failure();
    mlir::FailureOr<tensor::PhysicalAllocation> gated_context = mlir::failure();
    if (graph.output_gate) {
        const llvm::SmallVector<int64_t, 2> gate_banks {0, 1};
        const int64_t gate_base = std::max({
            qk_norm_scratch_base + qk_norm_matrix_rows + 1,
            rope_staging_base + rope_staging_rows, rope_mirror_base + rope_rows});
        if (!llvm::isa<mlir::BlockArgument>(graph.output_gate.getGate())) {
            auto inherited = get_value_placement(graph.output_gate.getGate());
            if (mlir::failed(inherited)) return mlir::failure();
            tensor::PhysicalAllocation preserved;
            preserved.name = "output_gate";
            for (auto slice : inherited->getAs<mlir::ArrayAttr>("slices"))
                preserved.slices.push_back(llvm::cast<mlir::IntegerAttr>(slice).getInt());
            preserved.base_row = inherited->getAs<mlir::IntegerAttr>("base_row").getInt();
            preserved.rows = inherited->getAs<mlir::IntegerAttr>("instruction_count").getInt();
            preserved.bank = inherited->getAs<mlir::IntegerAttr>("bank").getInt();
            preserved.live_start = 0; preserved.live_end = 7;
            preserved.reserve_slice_port = false;
            if (mlir::failed(physical_allocator.reserve(preserved))) {
                op.emitError("attention scratch overlaps the live gate projection");
                return mlir::failure();
            }
            gate_input = preserved;
        } else {
            gate_input = physical_allocator.allocate({"output_gate", 2, gate_base,
                seq_len * query_width / tile, 0, 7, scratch_candidates, false, gate_banks});
        }
        gated_context = physical_allocator.allocate({"gated_context",
            static_cast<int64_t>(context->slices.size()), 0, context_rows,
            4, 7, scratch_candidates, true, gate_banks,
            mlir::succeeded(gate_input) ? llvm::ArrayRef<int64_t>(gate_input->slices)
                                       : llvm::ArrayRef<int64_t>{}});
        if (mlir::failed(gate_input) || mlir::failed(gated_context)) {
            op.emitError("cannot allocate independent attention output gate buffers");
            return mlir::failure();
        }
    }
    const llvm::SmallVector<int64_t, 1> result_banks {
        secondary_scratch_bank};
    // Use both hemispheres when the canonical east-only result exceeds a
    // bank. Four consecutive feature blocks map to E pair0/pair1, W pair0/pair1.
    const bool partition_result = seq_len * hidden / (tile * 2)
        > target.memory().words_per_bank;
    const int64_t result_rows = seq_len * ((hidden + tile * (partition_result ? 4 : 2) - 1)
        / (tile * (partition_result ? 4 : 2)));
    auto vector_result = physical_allocator.allocate({"result",
        target.throughput().mxm_result_streams,
        0, result_rows, 5, 7,
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
        const auto pages = weight_tile_plan->getPages(kind);
        attrs.set("page_count", rewriter.getI64IntegerAttr(pages.size()));
        attrs.set("page_rows",
            rewriter.getI64IntegerAttr(weight_tile_plan->bank_rows));
        attrs.set("page_granularity",
            rewriter.getI64IntegerAttr(pages.front().item_count));
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
        if (pages.size() > 1) {
            llvm::SmallVector<int64_t> pageBanks, pageGroupBases,
                pageGroupCounts, pageBaseRows, pageRowCounts;
            for (const auto& page : pages) {
                pageBanks.push_back(page.bank);
                pageGroupBases.push_back(page.slice_group_begin);
                pageGroupCounts.push_back(page.slice_group_count);
                // Placement base_row is already stored in the binding. The
                // per-page base is relative to that row.
                pageBaseRows.push_back(page.base_row - tile_plan.base_row);
                pageRowCounts.push_back(page.rows);
            }
            attrs.set("page_banks", rewriter.getI64ArrayAttr(pageBanks));
            attrs.set("page_slice_group_bases",
                rewriter.getI64ArrayAttr(pageGroupBases));
            attrs.set("page_slice_group_counts",
                rewriter.getI64ArrayAttr(pageGroupCounts));
            attrs.set("page_base_rows", rewriter.getI64ArrayAttr(pageBaseRows));
            attrs.set("page_row_counts", rewriter.getI64ArrayAttr(pageRowCounts));
        }
        if (weight_tile_plan->bank_level_ping_pong) {
            attrs.set("page_bank_ping_pong", rewriter.getBoolAttr(true));
            // Only Q is required before execution. Subsequent projections
            // enter through the runtime C2C queues under the current compute.
            attrs.set("runtime_prefetch", rewriter.getBoolAttr(
                kind != tensor::AttentionWeightTileKind::Query));
        }
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
    auto inputPlacement = mlir::succeeded(inheritedInputPlacement)
        ? *inheritedInputPlacement
        : make_attention_placement(rewriter,
            "fp16_mxm_activation_planar",
            activation_slices, 0, seq_len * hidden / tile,
            "both");
    if ((graph_weight_pages || direct_projection_input)
        && mlir::failed(inheritedInputPlacement)) {
        // Keep the complete activation tensor out of query-IW storage.
        // Q normalization/RoPE precedes K/V projection in the serial path;
        // its query-IW writes may reuse the short-lived planar staging pair.
        // The shared RMSNorm distributed layout remains live throughout QKV.
        const auto distributed = target.mxm_distributed_activation_slices();
        inputPlacement = make_attention_placement(rewriter,
            "fp16_mxm_distributed_16", distributed, 0,
            blocks * (hidden / tile) * target.throughput().tile_rows,
            "both", scratch_bank);
    }
    const int64_t blocks_per_rotary_half =
        std::max<int64_t>(1, head_blocks / 2);
    const auto make_segment = [&](int64_t reduction_begin,
                                  int64_t reduction_end, int64_t bank,
                                  llvm::ArrayRef<int64_t> slices,
                                  int64_t base_row) {
        return rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("reduction_begin",
                rewriter.getI64IntegerAttr(reduction_begin)),
            rewriter.getNamedAttr("reduction_end",
                rewriter.getI64IntegerAttr(reduction_end)),
            rewriter.getNamedAttr(
                "bank", rewriter.getI64IntegerAttr(bank)),
            rewriter.getNamedAttr(
                "base_row", rewriter.getI64IntegerAttr(base_row)),
            rewriter.getNamedAttr("slices",
                rewriter.getI64ArrayAttr(slices)),
        });
    };
    const auto query_segments = rewriter.getArrayAttr({
        make_segment(0, blocks_per_rotary_half, query_segment_banks[0],
            target.attention_query_iw_slices(0), query_base),
        make_segment(blocks_per_rotary_half, head_blocks,
            query_segment_banks[1], target.attention_query_iw_slices(1),
            query_base),
    });
    const auto key_segments = rewriter.getArrayAttr({
        make_segment(0, blocks_per_rotary_half, key_allocation->bank,
            llvm::ArrayRef<int64_t>(key_allocation->slices).take_front(2),
            key_allocation->base_row),
        make_segment(blocks_per_rotary_half, head_blocks,
            key_allocation->bank,
            llvm::ArrayRef<int64_t>(key_allocation->slices).take_back(2),
            key_allocation->base_row),
    });
    const auto with_segments = [&](mlir::DictionaryAttr placement,
                                   mlir::ArrayAttr segments) {
        mlir::NamedAttrList attributes(placement);
        attributes.set("segments", segments);
        return attributes.getDictionary(rewriter.getContext());
    };
    const auto query_placement = with_segments(
        make_attention_placement(rewriter, "fp16_query_iw",
            target.attention_query_iw_slices(0), query_base, query_rows,
            "both", scratch_bank),
        query_segments);
    const auto key_placement = with_segments(
        make_attention_placement(rewriter, "fp16_head_planar",
            key_allocation->slices, key_allocation->base_row,
            key_rows, "both",
            key_allocation->bank),
        key_segments);
    const auto valueSlices = target.attention_value_slices();
    auto valuePlacement = make_attention_placement(rewriter,
        "fp16_value_x16", valueSlices, value_base, value_rows,
        "both", value_bank);
    if (split_value_banks)
        valuePlacement = with_segments(valuePlacement, rewriter.getArrayAttr({
            make_segment(0, 1, value_bank,
                llvm::ArrayRef<int64_t>(valueSlices).take_front(16), value_base),
            make_segment(1, 2, scratch_bank,
                llvm::ArrayRef<int64_t>(valueSlices).take_back(16), value_base),
        }));
    auto ropeProductPlacement = make_attention_placement(rewriter,
        compact_rope_products ? "fp16_rope_product_banked_x4"
                              : "fp16_rope_product_x16",
        rope_product_slices, rope_product_base, rope_product_rows, "both",
        rope_product_bank);
    if (compact_rope_products) {
        mlir::NamedAttrList attrs(ropeProductPlacement);
        attrs.set("bank_interleaved", rewriter.getBoolAttr(true));
        ropeProductPlacement = attrs.getDictionary(rewriter.getContext());
    }
    auto contextPlacement = make_attention_placement(rewriter,
        compact_rope_products ? "fp16_head_block_packed"
                              : "fp16_head_planar",
        context->slices,
        context_base, context_rows, "both",
        context->bank);
    if (compact_rope_products) {
        mlir::NamedAttrList attrs(contextPlacement);
        attrs.set("head_block_packed", rewriter.getBoolAttr(true));
        attrs.set("hemisphere_paired", rewriter.getBoolAttr(true));
        if (split_context_banks)
            attrs.set("segments", rewriter.getArrayAttr({
                make_segment(0, 1, context->bank, context->slices,
                    context_base),
                make_segment(1, 2, secondary_scratch_bank, context->slices,
                    context_base),
            }));
        contextPlacement = attrs.getDictionary(rewriter.getContext());
    }
    auto plan = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("input", inputPlacement),
        rewriter.getNamedAttr("input_staging",
            direct_projection_input ? inputPlacement : make_attention_placement(rewriter,
                "fp16_pair_planar", input_staging_slices,
                input_staging_base, input_staging_rows, "both",
                input_staging_bank)),
        rewriter.getNamedAttr("query_weight", weight_placement(
            native4 ? "w8a16_native4_weight"
                    : "w8a16_attention_weight_striped",
            tensor::AttentionWeightTileKind::Query, 0, q_weight_rows,
            weight_slices, 0)),
        rewriter.getNamedAttr("key_weight", weight_placement(
            native4 ? "w8a16_native4_weight"
                    : "w8a16_attention_weight_striped",
            tensor::AttentionWeightTileKind::Key, q_weight_rows,
            k_weight_rows, weight_slices,
            serial_attention_weight_pages ? 2 : 0)),
        rewriter.getNamedAttr("value_weight", weight_placement(
            native4 ? "w8a16_native4_weight"
                    : "w8a16_attention_weight_striped",
            tensor::AttentionWeightTileKind::Value,
            q_weight_rows + k_weight_rows, v_weight_rows,
            weight_slices, serial_attention_weight_pages ? 1 : 0)),
        rewriter.getNamedAttr("output_weight", weight_placement(
            native4 ? "w8a16_native4_weight"
                    : "w8a16_mxm_weight_striped",
            tensor::AttentionWeightTileKind::Output, output_weight_base,
            o_weight_rows, output_weight_slices,
            serial_attention_weight_pages ? 3 : 1)),
        rewriter.getNamedAttr("query", query_placement),
        rewriter.getNamedAttr("key", key_placement),
        rewriter.getNamedAttr("value", valuePlacement),
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
            mask0->base_row, tile - 1, "both",
            mask0->bank)),
        rewriter.getNamedAttr("causal_mask_mxm1", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", mask1->slices,
            mask1->base_row, tile - 1, "both",
            mask1->bank)),
        rewriter.getNamedAttr("fused_score", make_attention_placement(rewriter,
            "fp32_score_block", fused_score0,
            score_base, score_rows, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("fused_score_bank1", make_attention_placement(rewriter,
            "fp32_score_block", fused_score1,
            score_base, score_rows, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("fused_causal_mask", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", fused_mask0,
            causal_mask_base, tile - 1, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("fused_causal_mask_bank1", make_attention_placement(rewriter,
            "fp16_causal_mask_tile", fused_mask1,
            causal_mask_base, tile - 1, "both",
            fused_scratch_bank)),
        rewriter.getNamedAttr("probability_pack", make_attention_placement(rewriter,
            "fp16_probability_x16", probability_pack_slices,
            probability_pack_base,
            probability_pack_rows, "both",
            probability_pack_bank)),
        rewriter.getNamedAttr("probability_diagonal", make_attention_placement(rewriter,
            "fp16_probability_diagonal",
            probability_diagonal_slices,
            probability_diagonal_base,
            probability_diagonal_rows, "both",
            probability_diagonal_bank)),
        rewriter.getNamedAttr("rope", make_attention_placement(rewriter,
            decode ? "fp16_rope_table_decode_compact" : "fp16_rope_table",
            rope_table_slices,
            rope_table_base,
            rope_rows, "both", rope_table_bank)),
        rewriter.getNamedAttr("rope_mirror", make_attention_placement(rewriter,
            decode ? "fp16_rope_table_decode_compact" : "fp16_rope_table",
            rope_mirror_slices,
            compact_rope_products ? rope_mirror_base
                                  : probability_diagonal_base,
            rope_rows, "both", rope_mirror_bank)),
        rewriter.getNamedAttr("rope_staging",
            make_attention_placement(rewriter,
                "fp16_rope_fifo_x16", rope_staging_slices,
                rope_staging_base, rope_staging_rows, "both",
                rope_staging_bank)),
        rewriter.getNamedAttr("rope_product", ropeProductPlacement),
        rewriter.getNamedAttr("rope_product_key",
            make_attention_placement(rewriter,
                "fp16_rope_product_key_x4", rope_product_key_slices,
                key_product_base, key_product_rows, "both",
                scratch_bank)),
        rewriter.getNamedAttr("context", contextPlacement),
        rewriter.getNamedAttr("output_activation",
            make_attention_placement(rewriter,
                "fp16_pair_planar", input_staging_slices,
                output_staging_base, output_staging_rows,
                "both", scratch_bank)),
        rewriter.getNamedAttr("result", make_attention_placement(rewriter,
            "fp16_pair_planar",
            vector_result->slices, vector_result->base_row,
            result_rows, partition_result ? "both" : "east",
            vector_result->bank)),
    });
    if (!input_staging_pong_slices.empty()) {
        mlir::NamedAttrList plan_attributes(plan);
        plan_attributes.set("input_staging_pong",
            make_attention_placement(rewriter,
                "fp16_pair_planar", input_staging_pong_slices,
                input_staging_base, input_staging_rows, "both",
                input_staging_bank));
        plan = plan_attributes.getDictionary(rewriter.getContext());
    }
    if (split_serial_q_rope_staging) {
        mlir::NamedAttrList plan_attributes(plan);
        plan_attributes.set("rope_staging_alternate",
            make_attention_placement(rewriter,
                "fp16_rope_fifo_x16", alternate_rope_staging_slices,
                alternate_rope_staging_base, rope_staging_rows, "both",
                scratch_bank));
        plan = plan_attributes.getDictionary(rewriter.getContext());
    }
    if (use_disjoint_q_rope_slices) {
        mlir::NamedAttrList plan_attributes(plan);
        plan_attributes.set("rope_staging_mirror",
            make_attention_placement(rewriter,
                "fp16_rope_fifo_mirror_x16",
                rope_staging_mirror_slices,
                rope_staging_mirror_base, rope_staging_rows, "both",
                scratch_bank));
        plan = plan_attributes.getDictionary(rewriter.getContext());
    }
    if (has_qk_norm) {
        mlir::NamedAttrList plan_attributes(plan);
        const auto gammaPlacement = [&](llvm::ArrayRef<int64_t> slices,
                                       int64_t base, int64_t bank) {
            mlir::NamedAttrList placement(make_attention_placement(rewriter,
                "fp16_vxm_gamma_broadcast", slices, base, head_dim, "both", bank));
            // Gamma is a persistent activation-region constant, not a matrix
            // weight whose bank may be reassigned after physical planning.
            placement.set("bank_locked", rewriter.getBoolAttr(true));
            return placement.getDictionary(rewriter.getContext());
        };
        plan_attributes.set("query_norm_weight", gammaPlacement(
            query_norm_weight_slices, query_norm_gamma_base, query_norm_bank));
        plan_attributes.set("key_norm_weight", gammaPlacement(
            key_norm_weight_slices, key_norm_gamma_base, key_norm_bank));
        plan_attributes.set("query_norm_input",
            make_attention_placement(rewriter,
                "fp16_vxm_distributed_16", qk_norm_input_slices,
                qk_norm_input_base, qk_norm_matrix_rows + 1, "both",
                scratch_bank));
        plan_attributes.set("query_norm_output",
            make_attention_placement(rewriter,
                "fp16_vxm_distributed_16", qk_norm_output_slices,
                qk_norm_output_base, qk_norm_matrix_rows, "both",
                secondary_scratch_bank));
        plan_attributes.set("key_norm_input",
            make_attention_placement(rewriter,
                "fp16_vxm_distributed_16", qk_norm_input_slices,
                qk_norm_input_base, qk_norm_matrix_rows + 1, "both",
                scratch_bank));
        plan_attributes.set("key_norm_output",
            make_attention_placement(rewriter,
                "fp16_vxm_distributed_16", qk_norm_output_slices,
                qk_norm_output_base, qk_norm_matrix_rows, "both",
                secondary_scratch_bank));
        plan = plan_attributes.getDictionary(rewriter.getContext());
    }
    if (has_attention_bias) {
        mlir::NamedAttrList plan_attributes(plan);
        const auto add_bias_placement = [&](llvm::StringRef name,
                                            mlir::Value bias,
                                            int64_t base,
                                            int64_t rows) {
            if (!bias) return;
            plan_attributes.set(name, make_attention_placement(rewriter,
                "fp16_projection_bias_x4", projection_bias_slices,
                base, rows, "both", projection_bias_bank));
        };
        add_bias_placement(
            "query_bias", query_bias, query_bias_base, query_bias_rows);
        add_bias_placement(
            "key_bias", key_bias, key_bias_base, key_bias_rows);
        add_bias_placement(
            "value_bias", value_bias, value_bias_base, value_bias_rows);
        plan = plan_attributes.getDictionary(rewriter.getContext());
    }
    if (graph.output_gate) {
        mlir::NamedAttrList entries(plan);
        auto gatePlacement = make_attention_placement(rewriter,
            "fp16_pair_planar", gate_input->slices, gate_input->base_row,
            seq_len * query_width / tile, "both", gate_input->bank);
        if (!llvm::isa<mlir::BlockArgument>(graph.output_gate.getGate())) {
            auto inheritedGate = get_value_placement(graph.output_gate.getGate());
            if (mlir::failed(inheritedGate)) {
                op.emitError("attention gate producer has no physical placement");
                return mlir::failure();
            }
            gatePlacement = *inheritedGate;
        }
        entries.set("output_gate", gatePlacement);
        mlir::NamedAttrList gated(contextPlacement);
        gated.set("slices", rewriter.getI64ArrayAttr(gated_context->slices));
        gated.set("base_row", rewriter.getI64IntegerAttr(gated_context->base_row));
        gated.set("bank", rewriter.getI64IntegerAttr(gated_context->bank));
        entries.set("gated_context", gated.getDictionary(rewriter.getContext()));
        plan = entries.getDictionary(rewriter.getContext());
    }
    auto config = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr(
            "seq_len", rewriter.getI64IntegerAttr(seq_len)),
        rewriter.getNamedAttr(
            "kv_seq_len", rewriter.getI64IntegerAttr(kv_seq_len)),
        rewriter.getNamedAttr(
            "position_offset", rewriter.getI64IntegerAttr(decode_past_len)),
        rewriter.getNamedAttr(
            "current_len", rewriter.getI64IntegerAttr(
                decode ? decode_current_len : seq_len)),
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
        rewriter.getNamedAttr("rotary_dim", graph.query_rope->getAttr("rotary_dim")
            ? graph.query_rope->getAttr("rotary_dim") : rewriter.getI64IntegerAttr(graph.query_rope.getHeadDim())),
        rewriter.getNamedAttr("rope_attention_factor", graph.query_rope->getAttr("attention_factor")
            ? graph.query_rope->getAttr("attention_factor") : rewriter.getF32FloatAttr(1.0f)),
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
    if (auto frequencies = graph.query_rope->getAttr("inverse_frequencies")) {
        mlir::NamedAttrList attributes(config);
        attributes.set("rope_inverse_frequencies", frequencies);
        config = attributes.getDictionary(rewriter.getContext());
    }
    if (has_qk_norm) {
        mlir::NamedAttrList config_attributes(config);
        config_attributes.set("qk_norm_epsilon",
            graph.query_norm.getEpsilonAttr());
        config = config_attributes.getDictionary(rewriter.getContext());
    }
    if (kv_cache_enabled) {
        mlir::NamedAttrList config_attributes(config);
        config_attributes.set("kv_cache_capacity",
            rewriter.getI64IntegerAttr(kv_cache_capacity));
        config_attributes.set("kv_cache_page_tokens",
            rewriter.getI64IntegerAttr(kv_cache_page_tokens));
        config_attributes.set("kv_cache_resident_tokens",
            rewriter.getI64IntegerAttr(kv_cache_resident_tokens));
        config = config_attributes.getDictionary(rewriter.getContext());
    }
    const auto subplan =
        [&](std::initializer_list<llvm::StringRef> names) {
            llvm::SmallVector<mlir::NamedAttribute> entries;
            for (llvm::StringRef name : names)
                if (mlir::Attribute placement = plan.get(name))
                    entries.push_back(
                        rewriter.getNamedAttr(name, placement));
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
        {query_heads, seq_len, kv_seq_len},
        elementType);
    const auto createProjection =
        [&](mlir::Value input, mlir::Value weight, mlir::Value bias,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::DictionaryAttr memoryPlan) {
            mlir::OperationState state(
                op.getLoc(),
                tensor::ProjectionTaskOp::getOperationName());
            state.addOperands({input, weight});
            if (bias) state.addOperands(bias);
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
    const auto createRope =
        [&](mlir::Value input, mlir::Value normWeight,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::DictionaryAttr memoryPlan) {
            mlir::OperationState state(
                op.getLoc(), tensor::RopeTaskOp::getOperationName());
            state.addOperands(input);
            if (normWeight) state.addOperands(normWeight);
            state.addTypes(resultType);
            state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr(kind)),
                rewriter.getNamedAttr("config", config),
                rewriter.getNamedAttr("memory_plan", memoryPlan),
            });
            return llvm::cast<tensor::RopeTaskOp>(
                rewriter.create(state)).getResult();
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
        graph.query.getLhs(), graph.query.getRhs(), query_bias,
        "query", matrixType(seq_len, query_width),
        subplan({"input", "input_staging", "input_staging_pong",
            "query_weight", "query", "rope_staging",
            "rope_staging_alternate", "rope_product",
            "rope_mirror", "rope_staging_mirror", "query_bias"}));
    auto key = createProjection(
        graph.key.getLhs(), graph.key.getRhs(), key_bias,
        "key", matrixType(seq_len, kv_width),
        subplan({"key_weight", "key", "rope_product_key", "key_bias"}));
    auto value = createProjection(
        graph.value.getLhs(), graph.value.getRhs(), value_bias,
        "value", matrixType(seq_len, kv_width),
        subplan({"value_weight", "value", "value_bias"}));
    const mlir::Value rotatedQuery = createRope(
        query.getResult(), query_norm_weight, "query",
        query.getResult().getType(),
        subplan({"rope", "query_norm_weight",
            "query_norm_input", "query_norm_output"}));
    const mlir::Value rotatedKey = createRope(
        key.getResult(), key_norm_weight, "key",
        key.getResult().getType(),
        subplan({"key_norm_weight",
            "key_norm_input", "key_norm_output"}));
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
    mlir::Value gatedContext = pv.getResult();
    if (graph.output_gate) {
        mlir::OperationState state(op.getLoc(), tensor::SigmoidMultiplyTaskOp::getOperationName());
        state.addOperands({gatedContext, graph.output_gate.getGate()});
        state.addTypes(gatedContext.getType());
        state.addAttribute("config", config);
        state.addAttribute("memory_plan", subplan({"output_gate", "gated_context"}));
        gatedContext = rewriter.create(state)->getResult(0);
    }
    auto output = createProjection(gatedContext,
        graph.output.getRhs(), {}, "output",
        graph.output.getResult().getType(),
        subplan({"output_activation", "output_weight", "result"}));
    mlir::Operation* output_operation = graph.output.getOperation();
    rewriter.replaceOp(graph.output, output.getResult());
    // Optional Q/K RMSNorm inserts a second producer chain between the
    // projection and the heads reshape.  Remove dead graph nodes to a fixed
    // point because a single reverse walk cannot cover both chains with the
    // legacy operation ordering.
    bool erasedOperation = true;
    while (erasedOperation) {
        erasedOperation = false;
        for (mlir::Operation*& operation :
             llvm::reverse(graph.operations)) {
            if (operation && operation != output_operation
                && operation->use_empty()) {
                rewriter.eraseOp(operation);
                operation = nullptr;
                erasedOperation = true;
            }
        }
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
