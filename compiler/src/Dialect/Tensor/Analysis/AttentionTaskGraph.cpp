#include "ftlpu/compiler/Dialect/Tensor/Analysis/attention_task_graph.hpp"

#include "llvm/ADT/StringSet.h"

namespace ftlpu::compiler::tensor {
namespace {

bool has_complete_disjoint_memory_plan(const AttentionTaskGraph& graph)
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

} // namespace

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
    return mlir::DictionaryAttr::get(
        operation.getOperation()->getContext(), entries);
}

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function)
{
    return analysis::collect_attention_task_graphs<AttentionTaskGraph>(
        function, [](const AttentionTaskGraph& graph) {
            if (has_complete_disjoint_memory_plan(graph))
                return mlir::success();
            graph.emitError(
                "requires complete, uniquely owned physical memory sub-plans");
            return mlir::failure();
        });
}

} // namespace ftlpu::compiler::tensor
