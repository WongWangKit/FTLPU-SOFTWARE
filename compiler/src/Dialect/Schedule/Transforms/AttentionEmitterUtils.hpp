#pragma once

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/PatternMatch.h"

#include <array>

namespace ftlpu::compiler::schedule::attention_detail {

void emitMem(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t address,
    int64_t packedStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t addressStride, llvm::StringRef destination = "sram");

void emitMxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t weightBuffer,
    int64_t weightColumn, int64_t activationStream, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval);

void emitSxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t hemisphere, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::ArrayRef<int64_t> permuteMap,
    llvm::StringRef weightLayout = "vector_columns");

std::array<int64_t, 32> identityMap();

std::array<int64_t, 32> blockDiagonalMap(int64_t diagonal,
    const target::LPUTargetModel& target);

VxmOp emitVxm(mlir::IRRewriter& rewriter, stream::AttentionOp op,
    mlir::Value value, int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere);

AttentionProjectionKind projectionKind(int64_t index);

void emitRopeOrCast(mlir::IRRewriter& rewriter, stream::AttentionOp op,
    const target::LPUTargetModel& target, int64_t cycle, int64_t hemisphere,
    bool rope, mlir::Value value);

} // namespace ftlpu::compiler::schedule::attention_detail
