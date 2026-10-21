#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule::attention_detail {

void emitRopeOrCast(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle, int64_t hemisphere,
    bool rope, mlir::Value value, mlir::Type elementType,
    int64_t repeatCount)
{
    const char* hemisphereName = hemisphere == 0 ? "east" : "west";
    const char* returnDirection = hemisphere == 0 ? "west" : "east";
    // East streams enter C0 and advance toward increasing column numbers.
    // West streams enter the other end of the same physical fabric and use
    // columns mirrored around the configured VXM width.
    const auto column = [&](int64_t stage) {
        return hemisphere == 0
            ? stage
            : target.throughput().vxm_alus - 1 - stage;
    };
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(elementType);
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const auto emit = [&](int64_t operationCycle, int64_t queue,
                          llvm::StringRef opcode, llvm::StringRef lhsKind,
                          int64_t lhsIndex, float lhsImmediate,
                          llvm::StringRef rhsKind, int64_t rhsIndex,
                          float rhsImmediate, llvm::StringRef castTarget,
                          int64_t outputStream) {
        const bool returning = !rope || queue == column(6) || queue == column(7);
        auto instruction = emitVxmConfigured(rewriter, location, value, operationCycle, queue,
            opcode, lhsKind, lhsIndex, lhsImmediate, rhsKind, rhsIndex,
            rhsImmediate, castTarget, outputStream, hemisphereName,
            returning ? returnDirection : hemisphereName, -1, 8, repeatCount, 1);
        if (rope && (queue == column(0) || queue == column(1))) {
            instruction->setAttr("lhs_stream_tap", rewriter.getBoolAttr(true));
            instruction->setAttr("rhs_stream_tap", rewriter.getBoolAttr(true));
        }
    };
    if (!rope) {
        emit(cycle, column(0), "pass",
            "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 0);
        emit(cycle, column(1), "pass",
            "stream_f32", 36, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 2);
        return;
    }
    emit(cycle, column(0), "multiply",
        "stream_f32", 32, 0.0f, streamKind, 40, 0.0f,
        "fp32", 16);
    emit(cycle, column(1), "multiply",
        "stream_f32", 36, 0.0f, streamKind, 42, 0.0f,
        "fp32", 20);
    emit(cycle, column(3), "multiply",
        "stream_f32", 36, 0.0f, streamKind, 40, 0.0f,
        "fp32", 24);
    emit(cycle, column(4), "multiply",
        "stream_f32", 32, 0.0f, streamKind, 42, 0.0f,
        "fp32", 28);
    emit(cycle + 1, column(2), "subtract",
        "stream_f32", 16, 0.0f, "stream_f32", 20, 0.0f,
        dataFormat, 12);
    emit(cycle + 1, column(5), "add",
        "stream_f32", 24, 0.0f, "stream_f32", 28, 0.0f,
        dataFormat, 14);
    // The raw E0/E2 or W0/W2 inputs continue toward the second product.
    // Move the combined values into the architectural output SRs only after
    // the last raw-input reader, so both may traverse the same columns.
    emit(cycle + 2, column(6), "pass",
        streamKind, 12, 0.0f, "immediate", 0, 0.0f,
        dataFormat, 0);
    emit(cycle + 2, column(7), "pass",
        streamKind, 14, 0.0f, "immediate", 0, 0.0f,
        dataFormat, 2);
}

} // namespace ftlpu::compiler::schedule::attention_detail
