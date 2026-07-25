#pragma once

#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ftlpu::compiler::kernel {

struct FfnGraph {
    MatmulOp output;
    MatmulOp gate;
    MatmulOp up;
    SwishOp swish;
    ElementwiseOp multiply;
    llvm::SmallVector<mlir::Operation*, 5> operations;
};

std::optional<FfnGraph> match_ffn_graph(MatmulOp output);

} // namespace ftlpu::compiler::kernel
