#pragma once

#include "ftlpu/compiler/Analysis/attention_task_graph_view.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::schedule {

// A validated view over the primitive Stream IR tasks that implement one
// grouped-query attention operation.
struct AttentionTaskGraph
    : analysis::AttentionTaskGraphView<stream::ProjectionTaskOp,
          stream::RopeTaskOp, stream::BatchMatmulTaskOp,
          stream::SoftmaxTaskOp, stream::TransposeTaskOp> {
    mlir::DictionaryAttr getMemoryPlan() const {
        auto op = output;
        return *op.getMemoryPlan();
    }
    mlir::ArrayAttr getRoutes() const;
};

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function);

} // namespace ftlpu::compiler::schedule
