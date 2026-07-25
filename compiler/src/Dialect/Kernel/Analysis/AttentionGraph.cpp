#include "ftlpu/compiler/Dialect/Kernel/Analysis/attention_graph.hpp"

namespace ftlpu::compiler::kernel {

std::optional<AttentionGraph> match_attention_graph(MatmulOp output)
{
    auto context_reshape = output.getLhs().getDefiningOp<ReshapeOp>();
    if (!context_reshape) return std::nullopt;
    auto context_transpose =
        context_reshape.getInput().getDefiningOp<TransposeOp>();
    auto pv = context_transpose
        ? context_transpose.getInput().getDefiningOp<BatchMatmulOp>()
        : BatchMatmulOp {};
    if (!pv || pv.getRole() != "pv" || pv.getTransposeRhs())
        return std::nullopt;

    auto softmax = pv.getLhs().getDefiningOp<SoftmaxOp>();
    auto value_transpose = pv.getRhs().getDefiningOp<TransposeOp>();
    auto value_broadcast = value_transpose
        ? value_transpose.getInput().getDefiningOp<GqaBroadcastOp>()
        : GqaBroadcastOp {};
    auto value_reshape = value_broadcast
        ? value_broadcast.getInput().getDefiningOp<ReshapeOp>()
        : ReshapeOp {};
    auto value = value_reshape
        ? value_reshape.getInput().getDefiningOp<MatmulOp>()
        : MatmulOp {};
    auto qk = softmax
        ? softmax.getInput().getDefiningOp<BatchMatmulOp>()
        : BatchMatmulOp {};
    if (!softmax || !value || !qk || qk.getRole() != "qk"
        || !qk.getTransposeRhs())
        return std::nullopt;

    auto query_transpose = qk.getLhs().getDefiningOp<TransposeOp>();
    auto key_transpose = qk.getRhs().getDefiningOp<TransposeOp>();
    auto query_rope = query_transpose
        ? query_transpose.getInput().getDefiningOp<RopeOp>()
        : RopeOp {};
    auto key_broadcast = key_transpose
        ? key_transpose.getInput().getDefiningOp<GqaBroadcastOp>()
        : GqaBroadcastOp {};
    auto key_rope = key_broadcast
        ? key_broadcast.getInput().getDefiningOp<RopeOp>()
        : RopeOp {};
    auto query_reshape = query_rope
        ? query_rope.getInput().getDefiningOp<ReshapeOp>()
        : ReshapeOp {};
    auto key_reshape = key_rope
        ? key_rope.getInput().getDefiningOp<ReshapeOp>()
        : ReshapeOp {};
    auto query = query_reshape
        ? query_reshape.getInput().getDefiningOp<MatmulOp>()
        : MatmulOp {};
    auto key = key_reshape
        ? key_reshape.getInput().getDefiningOp<MatmulOp>()
        : MatmulOp {};
    if (!query || !key || query.getLhs() != key.getLhs()
        || query.getLhs() != value.getLhs()
        || query_rope.getHeadDim() != key_rope.getHeadDim()
        || query_rope.getTheta() != key_rope.getTheta()
        || value_broadcast.getQueryHeads() != query_rope.getHeads()
        || value_broadcast.getKvHeads() != key_rope.getHeads()
        || key_broadcast.getQueryHeads() != query_rope.getHeads()
        || key_broadcast.getKvHeads() != key_rope.getHeads())
        return std::nullopt;

    return AttentionGraph {
        output,
        query,
        key,
        value,
        context_reshape,
        query_reshape,
        key_reshape,
        value_reshape,
        context_transpose,
        query_transpose,
        key_transpose,
        value_transpose,
        query_rope,
        key_rope,
        key_broadcast,
        value_broadcast,
        qk,
        softmax,
        pv,
        {query.getOperation(), key.getOperation(), value.getOperation(),
            query_reshape.getOperation(), key_reshape.getOperation(),
            value_reshape.getOperation(), query_rope.getOperation(),
            key_rope.getOperation(), key_broadcast.getOperation(),
            value_broadcast.getOperation(), query_transpose.getOperation(),
            key_transpose.getOperation(), value_transpose.getOperation(),
            qk.getOperation(), softmax.getOperation(), pv.getOperation(),
            context_transpose.getOperation(), context_reshape.getOperation(),
            output.getOperation()}
    };
}

} // namespace ftlpu::compiler::kernel
