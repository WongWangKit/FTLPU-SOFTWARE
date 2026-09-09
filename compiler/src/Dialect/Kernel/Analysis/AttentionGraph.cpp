#include "ftlpu/compiler/Dialect/Kernel/Analysis/attention_graph.hpp"

namespace ftlpu::compiler::kernel {

namespace {

struct ProjectionRoot {
    MatmulOp matmul;
    BiasAddOp bias;
    RmsNormOp norm;
    ReshapeOp norm_input_reshape;
};

ProjectionRoot match_projection_root(mlir::Value value)
{
    if (auto bias = value.getDefiningOp<BiasAddOp>())
        return {bias.getInput().getDefiningOp<MatmulOp>(), bias, {}, {}};
    return {value.getDefiningOp<MatmulOp>(), {}, {}, {}};
}

ProjectionRoot match_normalized_projection_root(ReshapeOp heads_reshape)
{
    if (!heads_reshape) return {};
    auto norm = heads_reshape.getInput().getDefiningOp<RmsNormOp>();
    if (!norm) return match_projection_root(heads_reshape.getInput());
    auto norm_input_reshape =
        norm.getInput().getDefiningOp<ReshapeOp>();
    if (!norm_input_reshape) return {};
    ProjectionRoot root =
        match_projection_root(norm_input_reshape.getInput());
    root.norm = norm;
    root.norm_input_reshape = norm_input_reshape;
    return root;
}

} // namespace

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
    auto value_root = value_reshape
        ? match_projection_root(value_reshape.getInput())
        : ProjectionRoot {};
    auto value = value_root.matmul;
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
    auto query_root = query_reshape
        ? match_normalized_projection_root(query_reshape)
        : ProjectionRoot {};
    auto key_root = key_reshape
        ? match_normalized_projection_root(key_reshape)
        : ProjectionRoot {};
    auto query = query_root.matmul;
    auto key = key_root.matmul;
    if (!query || !key || static_cast<bool>(query_root.norm)
            != static_cast<bool>(key_root.norm)
        || query.getLhs() != key.getLhs()
        || query.getLhs() != value.getLhs()
        || query_rope.getHeadDim() != key_rope.getHeadDim()
        || query_rope.getTheta() != key_rope.getTheta()
        || value_broadcast.getQueryHeads() != query_rope.getHeads()
        || value_broadcast.getKvHeads() != key_rope.getHeads()
        || key_broadcast.getQueryHeads() != query_rope.getHeads()
        || key_broadcast.getKvHeads() != key_rope.getHeads())
        return std::nullopt;
    if (query_root.norm
        && (query_root.norm.getEpsilon() != key_root.norm.getEpsilon()
            || query_root.norm.getAxis() != -1
            || key_root.norm.getAxis() != -1))
        return std::nullopt;

    AttentionGraph graph {
        output,
        query,
        key,
        value,
        query_root.bias,
        key_root.bias,
        value_root.bias,
        query_root.norm,
        key_root.norm,
        context_reshape,
        query_reshape,
        key_reshape,
        value_reshape,
        query_root.norm_input_reshape,
        key_root.norm_input_reshape,
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
    std::size_t bias_index = 3;
    if (query_root.bias)
        graph.operations.insert(graph.operations.begin() + bias_index++,
            query_root.bias.getOperation());
    if (key_root.bias)
        graph.operations.insert(graph.operations.begin() + bias_index++,
            key_root.bias.getOperation());
    if (value_root.bias)
        graph.operations.insert(graph.operations.begin() + bias_index++,
            value_root.bias.getOperation());
    if (query_root.norm) {
        graph.operations.push_back(
            query_root.norm_input_reshape.getOperation());
        graph.operations.push_back(query_root.norm.getOperation());
    }
    if (key_root.norm) {
        graph.operations.push_back(
            key_root.norm_input_reshape.getOperation());
        graph.operations.push_back(key_root.norm.getOperation());
    }
    return graph;
}

} // namespace ftlpu::compiler::kernel
