#pragma once

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

namespace ftlpu::compiler::schedule::sxm_detail {

void emitSxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t hemisphere, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::ArrayRef<int64_t> permuteMap,
    llvm::StringRef weightLayout = "vector_columns",
    int64_t outputRow = -1, int64_t inputRow = -1,
    int64_t outputTile = -1, int64_t repeatCount = 1,
    int64_t repeatInterval = 1, int64_t waveCount = 1,
    int64_t waveInterval = 1, int64_t permuteMapStride = 0);

llvm::SmallVector<int64_t, 64> identityMap(int64_t width = 32);
llvm::SmallVector<int64_t, 64> identityMap(
    const target::LPUTargetModel& target);

llvm::SmallVector<int64_t, 64> blockDiagonalMap(int64_t diagonal,
    const target::LPUTargetModel& target);

// Emits one valid full-width capture followed by its registered Permute.
void emitWavefrontBeat(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle,
    int64_t hemisphere, int64_t diagonal,
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
