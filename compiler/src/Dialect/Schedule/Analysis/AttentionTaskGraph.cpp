#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_task_graph.hpp"

#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::schedule {
namespace {

int64_t integerConfig(
    const AttentionTaskGraph& graph, llvm::StringRef name)
{
    return graph.config().getAs<mlir::IntegerAttr>(name).getInt();
}

template <typename Op>
Op definingTask(mlir::Value value, llvm::StringRef kind)
{
    Op op = value.getDefiningOp<Op>();
    if (!op) return {};
    if constexpr (requires { op.getKind(); })
        if (op.getKind() != kind) return {};
    return op;
}

bool hasCommonConfig(const AttentionTaskGraph& graph)
{
    const mlir::DictionaryAttr config = graph.config();
    auto query = graph.query;
    auto key = graph.key;
    auto value = graph.value;
    auto queryRope = graph.query_rope;
    auto keyRope = graph.key_rope;
    auto qk = graph.qk;
    auto softmax = graph.softmax;
    auto probabilityTranspose = graph.probability_transpose;
    auto valueTranspose = graph.value_transpose;
    auto pv = graph.pv;
    return query.getConfig() == config && key.getConfig() == config
        && value.getConfig() == config && queryRope.getConfig() == config
        && keyRope.getConfig() == config && qk.getConfig() == config
        && softmax.getConfig() == config
        && probabilityTranspose.getConfig() == config
        && valueTranspose.getConfig() == config && pv.getConfig() == config;
}

} // namespace

int64_t AttentionTaskGraph::getSeqLen() const
{
    return integerConfig(*this, "seq_len");
}

int64_t AttentionTaskGraph::getHidden() const
{
    return integerConfig(*this, "hidden");
}

int64_t AttentionTaskGraph::getQueryHeads() const
{
    return integerConfig(*this, "query_heads");
}

int64_t AttentionTaskGraph::getKvHeads() const
{
    return integerConfig(*this, "kv_heads");
}

int64_t AttentionTaskGraph::getHeadDim() const
{
    return integerConfig(*this, "head_dim");
}

bool AttentionTaskGraph::getCausal() const
{
    return config().getAs<mlir::BoolAttr>("causal").getValue();
}

mlir::ArrayAttr AttentionTaskGraph::getRoutes() const
{
    llvm::SmallVector<mlir::Attribute> routes;
    const auto append = [&](auto task) {
        for (mlir::Attribute route : task.getRoutes())
            routes.push_back(route);
    };
    append(query);
    append(key);
    append(value);
    append(query_rope);
    append(key_rope);
    append(qk);
    append(softmax);
    append(probability_transpose);
    append(value_transpose);
    append(pv);
    append(output);
    auto operation = output;
    return mlir::ArrayAttr::get(operation.getContext(), routes);
}

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function)
{
    llvm::SmallVector<AttentionTaskGraph, 2> graphs;
    llvm::SmallVector<stream::ProjectionTaskOp> outputs;
    function.walk([&](stream::ProjectionTaskOp op) {
        if (op.getKind() == "output") outputs.push_back(op);
    });

    for (stream::ProjectionTaskOp output : outputs) {
        AttentionTaskGraph graph;
        graph.output = output;
        graph.pv = definingTask<stream::BatchMatmulTaskOp>(
            output.getInput(), "pv");
        if (graph.pv) {
            graph.probability_transpose =
                definingTask<stream::TransposeTaskOp>(
                    graph.pv.getLhs(), "probability");
            graph.value_transpose =
                definingTask<stream::TransposeTaskOp>(
                    graph.pv.getRhs(), "value");
        }
        if (graph.probability_transpose)
            graph.softmax =
                graph.probability_transpose.getInput()
                    .getDefiningOp<stream::SoftmaxTaskOp>();
        if (graph.softmax)
            graph.qk = definingTask<stream::BatchMatmulTaskOp>(
                graph.softmax.getInput(), "qk");
        if (graph.qk) {
            graph.query_rope = definingTask<stream::RopeTaskOp>(
                graph.qk.getLhs(), "query");
            graph.key_rope = definingTask<stream::RopeTaskOp>(
                graph.qk.getRhs(), "key");
        }
        if (graph.query_rope)
            graph.query = definingTask<stream::ProjectionTaskOp>(
                graph.query_rope.getInput(), "query");
        if (graph.key_rope)
            graph.key = definingTask<stream::ProjectionTaskOp>(
                graph.key_rope.getInput(), "key");
        if (graph.value_transpose)
            graph.value = definingTask<stream::ProjectionTaskOp>(
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
            || !hasCommonConfig(graph)) {
            output.emitError(
                "primitive attention tasks disagree on input or configuration");
            return mlir::failure();
        }
        graphs.push_back(graph);
    }
    return graphs;
}

} // namespace ftlpu::compiler::schedule
