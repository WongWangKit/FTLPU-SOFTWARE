#pragma once

#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ftlpu::compiler::kernel {

struct AttentionGraph {
    MatmulOp output;
    MatmulOp query;
    MatmulOp key;
    MatmulOp value;
    BiasAddOp query_bias;
    BiasAddOp key_bias;
    BiasAddOp value_bias;
    RmsNormOp query_norm;
    RmsNormOp key_norm;
    ReshapeOp context_reshape;
    ReshapeOp query_reshape;
    ReshapeOp key_reshape;
    ReshapeOp value_reshape;
    ReshapeOp query_norm_input_reshape;
    ReshapeOp key_norm_input_reshape;
    TransposeOp context_transpose;
    TransposeOp query_transpose;
    TransposeOp key_transpose;
    TransposeOp value_transpose;
    RopeOp query_rope;
    RopeOp key_rope;
    GqaBroadcastOp key_broadcast;
    GqaBroadcastOp value_broadcast;
    BatchMatmulOp qk;
    SoftmaxOp softmax;
    BatchMatmulOp pv;
    llvm::SmallVector<mlir::Operation*, 32> operations;
};

std::optional<AttentionGraph> match_attention_graph(MatmulOp output);

} // namespace ftlpu::compiler::kernel
