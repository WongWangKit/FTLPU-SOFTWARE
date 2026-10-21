#pragma once

#include "ftlpu/compiler/Dialect/Kernel/Analysis/attention_graph.hpp"
#include "ftlpu/compiler/Dialect/Kernel/Analysis/ffn_graph.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace ftlpu::compiler::kernel {

// A stateful mixer is a different graph pattern from softmax attention. Its
// boundaries remain ordinary SSA values, so the enclosing decoder owns their
// lifetimes (including retained state and any external snapshots).
struct GatedDeltaNetGraph {
    GatedDeltaRuleOp recurrence;
    llvm::SmallVector<mlir::Operation*> operations;
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Value> outputs;
};

struct DecoderGraph {
    llvm::SmallVector<AttentionGraph, 2> attentions;
    llvm::SmallVector<FfnGraph, 2> ffns;
    llvm::SmallVector<GatedDeltaNetGraph, 2> delta_nets;
};

// Shared by the native whole-function path and connected tensor-view graphs.
// No model names, physical layouts, weights, or scheduling policy are used.
DecoderGraph match_decoder_graph(mlir::func::FuncOp function);

} // namespace ftlpu::compiler::kernel
