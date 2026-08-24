#pragma once

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/PatternMatch.h"

#include <array>

namespace ftlpu::compiler::schedule::sxm_detail {

void emitSxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t hemisphere, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::ArrayRef<int64_t> permuteMap,
    llvm::StringRef weightLayout = "vector_columns",
    int64_t outputRow = -1, int64_t inputRow = -1,
    int64_t outputTile = -1);

std::array<int64_t, 32> identityMap();

std::array<int64_t, 32> blockDiagonalMap(int64_t diagonal,
    const target::LPUTargetModel& target);

// Emits one valid full-width capture followed by its registered Permute.
void emitWavefrontBeat(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle,
    int64_t hemisphere, int64_t diagonal,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> transposeStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::StringRef weightLayout = "vector_columns");

// Captures one block and drains it completely before the transpose bank is
// reused. Use this for materialized MEM destinations that are not consumed as
// one continuous functional-unit wavefront.
void emitBufferedWavefrontBeat(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    int64_t cycle, int64_t hemisphere, int64_t diagonal,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> transposeStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::StringRef weightLayout = "vector_columns");

// Drains a northbound wave already resident in the transpose buffer.
void emitWavefrontTail(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle,
    int64_t hemisphere, int64_t diagonal,
    llvm::ArrayRef<int64_t> transposeStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::StringRef weightLayout = "vector_columns");

} // namespace ftlpu::compiler::schedule::sxm_detail
