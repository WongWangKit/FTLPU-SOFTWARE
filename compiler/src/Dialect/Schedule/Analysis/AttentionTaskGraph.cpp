#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_task_graph.hpp"

namespace ftlpu::compiler::schedule {

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
    return mlir::ArrayAttr::get(
        operation.getOperation()->getContext(), routes);
}

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function)
{
    return analysis::collect_attention_task_graphs<AttentionTaskGraph>(
        function, [](const AttentionTaskGraph&) {
            return mlir::success();
        });
}

} // namespace ftlpu::compiler::schedule
