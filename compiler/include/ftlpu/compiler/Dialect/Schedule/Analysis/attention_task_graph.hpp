#pragma once

#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace ftlpu::compiler::schedule {

// A validated view over the primitive Stream IR tasks that implement one
// grouped-query attention operation.
struct AttentionTaskGraph {
    stream::ProjectionTaskOp query;
    stream::ProjectionTaskOp key;
    stream::ProjectionTaskOp value;
    stream::ProjectionTaskOp output;
    stream::RopeTaskOp query_rope;
    stream::RopeTaskOp key_rope;
    stream::BatchMatmulTaskOp qk;
    stream::SoftmaxTaskOp softmax;
    stream::TransposeTaskOp probability_transpose;
    stream::TransposeTaskOp value_transpose;
    stream::BatchMatmulTaskOp pv;

    mlir::DictionaryAttr config() const {
        auto op = output;
        return op.getConfig();
    }
    mlir::DictionaryAttr getMemoryPlan() const {
        auto op = output;
        return *op.getMemoryPlan();
    }
    mlir::ArrayAttr getRoutes() const;
    mlir::Location getLoc() const {
        auto op = output;
        return op.getLoc();
    }

    int64_t getSeqLen() const;
    int64_t getHidden() const;
    int64_t getQueryHeads() const;
    int64_t getKvHeads() const;
    int64_t getHeadDim() const;
    bool getCausal() const;

    mlir::Value getInput() const {
        auto op = query;
        return op.getInput();
    }
    mlir::Value getQueryWeight() const {
        auto op = query;
        return op.getWeight();
    }
    mlir::Value getKeyWeight() const {
        auto op = key;
        return op.getWeight();
    }
    mlir::Value getValueWeight() const {
        auto op = value;
        return op.getWeight();
    }
    mlir::Value getOutputWeight() const {
        auto op = output;
        return op.getWeight();
    }
    mlir::Value getResult() const {
        auto op = output;
        return op.getResult();
    }

    mlir::InFlightDiagnostic emitError(llvm::Twine message) const {
        auto op = output;
        return op.emitError(message);
    }
};

mlir::FailureOr<llvm::SmallVector<AttentionTaskGraph, 2>>
collectAttentionTaskGraphs(mlir::func::FuncOp function);

} // namespace ftlpu::compiler::schedule
