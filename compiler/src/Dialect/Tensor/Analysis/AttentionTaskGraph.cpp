#include "ftlpu/compiler/Dialect/Tensor/Analysis/attention_task_graph.hpp"

#include "llvm/ADT/StringSet.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::tensor {
namespace {

template <typename Op>
Op definingTask(mlir::Value value, llvm::StringRef kind)
{
    Op op = value.getDefiningOp<Op>();
    if (!op || op.getKind() != kind) return {};
    return op;
}

bool hasCommonConfig(const AttentionTaskGraph& graph)
{
    auto output = graph.output;
    const mlir::DictionaryAttr config = output.getConfig();
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

bool hasCompleteDisjointMemoryPlan(const AttentionTaskGraph& graph)
{
    llvm::StringSet<> names;
    bool disjoint = true;
    const auto append = [&](auto task) {
        for (mlir::NamedAttribute entry : task.getMemoryPlan())
            disjoint &= names.insert(entry.getName().strref()).second;
    };
    append(graph.query);
    append(graph.key);
    append(graph.value);
    append(graph.query_rope);
    append(graph.key_rope);
    append(graph.qk);
    append(graph.softmax);
    append(graph.probability_transpose);
    append(graph.value_transpose);
    append(graph.pv);
    append(graph.output);
    if (!disjoint) return false;
    for (llvm::StringRef required :
        {"input", "query_weight", "key_weight", "value_weight",
            "output_weight", "query", "key", "value", "score",
            "score_mxm1", "exp", "exp_mxm1", "causal_mask",
            "causal_mask_mxm1", "probability", "probability_mxm1",
            "probability_pack", "probability_diagonal", "rope", "context",
            "result"})
        if (!names.contains(required)) return false;
    return true;
}

int64_t integerConfig(
    const AttentionTaskGraph& graph, llvm::StringRef name)
{
    return graph.config().getAs<mlir::IntegerAttr>(name).getInt();
}

} // namespace

mlir::DictionaryAttr AttentionTaskGraph::config() const
{
    auto operation = output;
    return operation.getConfig();
}

mlir::DictionaryAttr AttentionTaskGraph::getMemoryPlan() const
{
    llvm::SmallVector<mlir::NamedAttribute> entries;
    const auto append = [&](auto task) {
        for (mlir::NamedAttribute entry : task.getMemoryPlan())
            entries.push_back(entry);
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
    return mlir::DictionaryAttr::get(operation.getContext(), entries);
}

mlir::Location AttentionTaskGraph::getLoc() const
{
    auto operation = output;
    return operation.getLoc();
}

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

mlir::Value AttentionTaskGraph::getInput() const
{
    auto operation = query;
    return operation.getInput();
}

mlir::Value AttentionTaskGraph::getQueryWeight() const
{
    auto operation = query;
    return operation.getWeight();
}

mlir::Value AttentionTaskGraph::getKeyWeight() const
{
    auto operation = key;
    return operation.getWeight();
}

mlir::Value AttentionTaskGraph::getValueWeight() const
{
    auto operation = value;
    return operation.getWeight();
}

mlir::Value AttentionTaskGraph::getOutputWeight() const
{
    auto operation = output;
    return operation.getWeight();
}

mlir::Value AttentionTaskGraph::getResult() const
{
    auto operation = output;
    return operation.getResult();
}

mlir::InFlightDiagnostic
AttentionTaskGraph::emitError(llvm::Twine message) const
{
    auto operation = output;
    return operation.emitError(message);
}

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function)
{
    llvm::SmallVector<AttentionTaskGraph, 2> graphs;
    llvm::SmallVector<ProjectionTaskOp> outputs;
    function.walk([&](ProjectionTaskOp operation) {
        if (operation.getKind() == "output") outputs.push_back(operation);
    });

    for (ProjectionTaskOp output : outputs) {
        AttentionTaskGraph graph;
        graph.output = output;
        graph.pv = definingTask<BatchMatmulTaskOp>(output.getInput(), "pv");
        if (graph.pv) {
            graph.probability_transpose =
                definingTask<TransposeTaskOp>(
                    graph.pv.getLhs(), "probability");
            graph.value_transpose =
                definingTask<TransposeTaskOp>(graph.pv.getRhs(), "value");
        }
        if (graph.probability_transpose)
            graph.softmax = graph.probability_transpose.getInput()
                                .getDefiningOp<SoftmaxTaskOp>();
        if (graph.softmax)
            graph.qk =
                definingTask<BatchMatmulTaskOp>(graph.softmax.getInput(), "qk");
        if (graph.qk) {
            graph.query_rope =
                definingTask<RopeTaskOp>(graph.qk.getLhs(), "query");
            graph.key_rope =
                definingTask<RopeTaskOp>(graph.qk.getRhs(), "key");
        }
        if (graph.query_rope)
            graph.query = definingTask<ProjectionTaskOp>(
                graph.query_rope.getInput(), "query");
        if (graph.key_rope)
            graph.key = definingTask<ProjectionTaskOp>(
                graph.key_rope.getInput(), "key");
        if (graph.value_transpose)
            graph.value = definingTask<ProjectionTaskOp>(
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
        if (!hasCompleteDisjointMemoryPlan(graph)) {
            output.emitError(
                "requires complete, uniquely owned physical memory sub-plans");
            return mlir::failure();
        }
        graphs.push_back(graph);
    }
    return graphs;
}

} // namespace ftlpu::compiler::tensor
