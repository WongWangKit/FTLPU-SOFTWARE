#pragma once

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::analysis {

template <typename ProjectionOp, typename RopeOp, typename BatchMatmulOp,
    typename SoftmaxOp, typename TransposeOp>
struct AttentionTaskGraphView {
    using ProjectionTask = ProjectionOp;
    using RopeTask = RopeOp;
    using BatchMatmulTask = BatchMatmulOp;
    using SoftmaxTask = SoftmaxOp;
    using TransposeTask = TransposeOp;

    ProjectionOp query;
    ProjectionOp key;
    ProjectionOp value;
    ProjectionOp output;
    RopeOp query_rope;
    RopeOp key_rope;
    BatchMatmulOp qk;
    SoftmaxOp softmax;
    TransposeOp probability_transpose;
    TransposeOp value_transpose;
    BatchMatmulOp pv;

    mlir::DictionaryAttr config() const
    {
        auto op = output;
        return op.getConfig();
    }
    mlir::Location getLoc() const
    {
        auto op = output;
        return op.getLoc();
    }
    int64_t getSeqLen() const { return integer_config("seq_len"); }
    int64_t getHidden() const { return integer_config("hidden"); }
    int64_t getQueryHeads() const { return integer_config("query_heads"); }
    int64_t getKvHeads() const { return integer_config("kv_heads"); }
    int64_t getHeadDim() const { return integer_config("head_dim"); }
    bool getCausal() const
    {
        return config().template getAs<mlir::BoolAttr>("causal").getValue();
    }
    mlir::Value getInput() const
    {
        auto op = query;
        return op.getInput();
    }
    mlir::Value getQueryWeight() const
    {
        auto op = query;
        return op.getWeight();
    }
    mlir::Value getKeyWeight() const
    {
        auto op = key;
        return op.getWeight();
    }
    mlir::Value getValueWeight() const
    {
        auto op = value;
        return op.getWeight();
    }
    mlir::Value getOutputWeight() const
    {
        auto op = output;
        return op.getWeight();
    }
    mlir::Value getResult() const
    {
        auto op = output;
        return op.getResult();
    }
    mlir::InFlightDiagnostic emitError(llvm::Twine message) const
    {
        auto op = output;
        return op.emitError(message);
    }

private:
    int64_t integer_config(llvm::StringRef name) const
    {
        return config().template getAs<mlir::IntegerAttr>(name).getInt();
    }
};

template <typename Op>
Op defining_attention_task(mlir::Value value, llvm::StringRef kind)
{
    Op op = value.getDefiningOp<Op>();
    if (!op) return {};
    if constexpr (requires { op.getKind(); })
        if (op.getKind() != kind) return {};
    return op;
}

template <typename Graph>
bool has_common_attention_config(const Graph& graph)
{
    const mlir::DictionaryAttr config = graph.config();
    auto query = graph.query;
    auto key = graph.key;
    auto value = graph.value;
    auto query_rope = graph.query_rope;
    auto key_rope = graph.key_rope;
    auto qk = graph.qk;
    auto softmax = graph.softmax;
    auto probability_transpose = graph.probability_transpose;
    auto value_transpose = graph.value_transpose;
    auto pv = graph.pv;
    return query.getConfig() == config
        && key.getConfig() == config
        && value.getConfig() == config
        && query_rope.getConfig() == config
        && key_rope.getConfig() == config
        && qk.getConfig() == config
        && softmax.getConfig() == config
        && probability_transpose.getConfig() == config
        && value_transpose.getConfig() == config
        && pv.getConfig() == config;
}

template <typename Graph, typename Validator>
mlir::FailureOr<llvm::SmallVector<Graph, 2>>
collect_attention_task_graphs(
    mlir::func::FuncOp function, Validator&& validate)
{
    using ProjectionOp = typename Graph::ProjectionTask;
    using RopeOp = typename Graph::RopeTask;
    using BatchMatmulOp = typename Graph::BatchMatmulTask;
    using SoftmaxOp = typename Graph::SoftmaxTask;
    using TransposeOp = typename Graph::TransposeTask;

    llvm::SmallVector<Graph, 2> graphs;
    llvm::SmallVector<ProjectionOp> outputs;
    function.walk([&](ProjectionOp op) {
        if (op.getKind() == "output") outputs.push_back(op);
    });

    for (ProjectionOp output : outputs) {
        Graph graph;
        graph.output = output;
        graph.pv = defining_attention_task<BatchMatmulOp>(
            output.getInput(), "pv");
        if (graph.pv) {
            graph.probability_transpose =
                defining_attention_task<TransposeOp>(
                    graph.pv.getLhs(), "probability");
            graph.value_transpose =
                defining_attention_task<TransposeOp>(
                    graph.pv.getRhs(), "value");
        }
        if (graph.probability_transpose)
            graph.softmax = graph.probability_transpose.getInput()
                                .template getDefiningOp<SoftmaxOp>();
        if (graph.softmax)
            graph.qk = defining_attention_task<BatchMatmulOp>(
                graph.softmax.getInput(), "qk");
        if (graph.qk) {
            graph.query_rope = defining_attention_task<RopeOp>(
                graph.qk.getLhs(), "query");
            graph.key_rope = defining_attention_task<RopeOp>(
                graph.qk.getRhs(), "key");
        }
        if (graph.query_rope)
            graph.query = defining_attention_task<ProjectionOp>(
                graph.query_rope.getInput(), "query");
        if (graph.key_rope)
            graph.key = defining_attention_task<ProjectionOp>(
                graph.key_rope.getInput(), "key");
        if (graph.value_transpose)
            graph.value = defining_attention_task<ProjectionOp>(
                graph.value_transpose.getInput(), "value");

        if (!graph.query || !graph.key || !graph.value || !graph.pv
            || !graph.probability_transpose || !graph.value_transpose
            || !graph.softmax || !graph.qk || !graph.query_rope
            || !graph.key_rope) {
            output.emitError(
                "does not terminate a complete primitive attention task graph");
            return mlir::failure();
        }
        if (graph.query.getInput() != graph.key.getInput()
            || graph.query.getInput() != graph.value.getInput()
            || !has_common_attention_config(graph)) {
            output.emitError(
                "primitive attention tasks disagree on input or configuration");
            return mlir::failure();
        }
        if (mlir::failed(validate(graph))) return mlir::failure();
        graphs.push_back(graph);
    }
    return graphs;
}

} // namespace ftlpu::compiler::analysis
