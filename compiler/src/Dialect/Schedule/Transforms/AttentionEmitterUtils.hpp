#pragma once

#include "SxmWavefrontEmitterUtils.hpp"
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
    int64_t addressStride, llvm::StringRef destination = "sram",
    int64_t addressBinding = -1, int64_t bank = -1);

void emitMemWave(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    int64_t address, int64_t packedStream, int64_t repeatCount,
    int64_t repeatInterval, int64_t addressStride,
    llvm::StringRef destination, int64_t addressBinding,
    int64_t waveCount, int64_t waveInterval,
    int64_t waveAddressStride);

void emitMemWave(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    int64_t address, int64_t packedStream, int64_t repeatCount,
    int64_t repeatInterval, int64_t addressStride,
    llvm::StringRef destination, int64_t addressBinding,
    int64_t waveCount, int64_t waveInterval,
    int64_t waveAddressStride, int64_t bank);

void emitMxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t weightBuffer,
    int64_t weightColumn, int64_t activationStream, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval,
    int64_t accumulatorAddress = 0, int64_t accumulatorRowStride = 1,
    llvm::StringRef accumulatorDestination = "stream",
    bool accumulatorClear = true,
    llvm::StringRef weightLoadMode = "supercell",
    int64_t weightInnerColumn = 0,
    llvm::StringRef dataFormat = "fp16",
    llvm::StringRef weightInputMode = {},
    llvm::StringRef computeMode = {},
    llvm::StringRef accumulatorOutputFormat = {});

void emitMxmWave(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    int64_t weightBuffer, int64_t weightColumn,
    int64_t activationStream, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval,
    int64_t accumulatorAddress, int64_t accumulatorRowStride,
    llvm::StringRef accumulatorDestination, bool accumulatorClear,
    llvm::StringRef weightLoadMode, int64_t weightInnerColumn,
    llvm::StringRef dataFormat, llvm::StringRef weightInputMode,
    llvm::StringRef computeMode, llvm::StringRef accumulatorOutputFormat,
    int64_t waveCount, int64_t waveInterval,
    int64_t waveWeightColumnStride, int64_t groupCount = 1,
    int64_t groupInterval = 1,
    int64_t waveAccumulatorAddressStride = 0);

void emitMxmDequant(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t unitId,
    float scale, int64_t repeatCount = 1,
    int64_t repeatInterval = 1);

void emitMxmDequantWave(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t unitId,
    float scale, int64_t repeatCount, int64_t repeatInterval,
    int64_t waveCount, int64_t waveInterval);

using sxm_detail::blockDiagonalMap;
using sxm_detail::emitSxm;
using sxm_detail::emitWavefrontBeat;
using sxm_detail::emitWavefrontTail;
using sxm_detail::identityMap;

VxmOp emitVxm(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Value value, int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhsKind, int64_t lhsIndex, float lhsImmediate,
    llvm::StringRef rhsKind, int64_t rhsIndex, float rhsImmediate,
    llvm::StringRef castTarget, int64_t outputStream,
    llvm::StringRef inputHemisphere, llvm::StringRef outputHemisphere,
    int64_t scaleBinding = -1);

AttentionProjectionKind projectionKind(int64_t index);

void emitRopeOrCast(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle, int64_t hemisphere,
    bool rope, mlir::Value value, mlir::Type elementType);

} // namespace ftlpu::compiler::schedule::attention_detail
