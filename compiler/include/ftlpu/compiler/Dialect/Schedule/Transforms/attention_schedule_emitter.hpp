#pragma once

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace ftlpu::compiler::schedule {

class AttentionScheduleEmitter {
public:
    AttentionScheduleEmitter(mlir::IRRewriter& rewriter,
        stream::AttentionOp op, const target::LPUTargetModel& target);

    mlir::FailureOr<schedule::AttentionOp> emit();

private:
    int64_t emitProjections();
    void emitQk(int64_t qkStart, int64_t qkWaveCycles,
        int64_t qkIwToComputeCycles);
    int64_t emitSoftmax(int64_t qkEnd);
    int64_t emitProbabilityPack(int64_t softmaxEnd);
    int64_t emitProbabilityTranspose(int64_t packEnd);
    int64_t emitPv(int64_t transposeEnd);
    int64_t emitOutputProjection(int64_t pvEnd);

    mlir::IRRewriter& rewriter_;
    stream::AttentionOp op_;
    const target::LPUTargetModel& target_;
};

} // namespace ftlpu::compiler::schedule
