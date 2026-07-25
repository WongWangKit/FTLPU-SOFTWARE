#pragma once

#include "ftlpu/compiler/Analysis/attention_task_graph_view.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::tensor {

// A validated view over one primitive Tensor IR grouped-query attention DAG.
struct AttentionTaskGraph
    : analysis::AttentionTaskGraphView<ProjectionTaskOp, RopeTaskOp,
          BatchMatmulTaskOp, SoftmaxTaskOp, TransposeTaskOp> {
    mlir::DictionaryAttr getMemoryPlan() const;
};

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function);

} // namespace ftlpu::compiler::tensor
