#pragma once

#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::tensor {

// A validated view over one primitive Tensor IR grouped-query attention DAG.
struct AttentionTaskGraph {
    ProjectionTaskOp query;
    ProjectionTaskOp key;
    ProjectionTaskOp value;
    ProjectionTaskOp output;
    RopeTaskOp query_rope;
    RopeTaskOp key_rope;
    BatchMatmulTaskOp qk;
    SoftmaxTaskOp softmax;
    TransposeTaskOp probability_transpose;
    TransposeTaskOp value_transpose;
    BatchMatmulTaskOp pv;

    mlir::DictionaryAttr config() const;
    mlir::DictionaryAttr getMemoryPlan() const;
    mlir::Location getLoc() const;
    int64_t getSeqLen() const;
    int64_t getHidden() const;
    int64_t getQueryHeads() const;
    int64_t getKvHeads() const;
    int64_t getHeadDim() const;
    bool getCausal() const;
    mlir::Value getInput() const;
    mlir::Value getQueryWeight() const;
    mlir::Value getKeyWeight() const;
    mlir::Value getValueWeight() const;
    mlir::Value getOutputWeight() const;
    mlir::Value getResult() const;
    mlir::InFlightDiagnostic emitError(llvm::Twine message) const;
};

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function);

} // namespace ftlpu::compiler::tensor
