#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule::attention_detail {

void emitRopeOrCast(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle, int64_t hemisphere,
    bool rope, mlir::Value value, mlir::Type elementType)
{
    const char* hemisphereName = hemisphere == 0 ? "east" : "west";
    // East streams enter C0 and advance toward increasing column numbers.
    // West streams enter the other end of the same physical fabric and must
    // therefore use the mirrored columns C15, C14, ... .
    const auto column = [&](int64_t stage) {
        return hemisphere == 0
            ? stage
            : target.throughput().vxm_alus - 1 - stage;
    };
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(elementType);
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    if (!rope) {
        emitVxm(rewriter, location, value, cycle, column(0), "pass",
            "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 0, hemisphereName, hemisphereName);
        emitVxm(rewriter, location, value, cycle, column(1), "pass",
            "stream_f32", 36, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 2, hemisphereName, hemisphereName);
        return;
    }
    emitVxm(rewriter, location, value, cycle, column(0), "multiply",
        "stream_f32", 32, 0.0f, streamKind, 40, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle, column(1), "multiply",
        "stream_f32", 36, 0.0f, streamKind, 42, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle, column(3), "multiply",
        "stream_f32", 36, 0.0f, streamKind, 40, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle, column(4), "multiply",
        "stream_f32", 32, 0.0f, streamKind, 42, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle + 1, column(2), "subtract",
        "alu", column(0), 0.0f, "alu", column(1), 0.0f,
        dataFormat, 0, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle + 1, column(5), "add",
        "alu", column(3), 0.0f, "alu", column(4), 0.0f,
        dataFormat, 2, hemisphereName, hemisphereName);
}

} // namespace ftlpu::compiler::schedule::attention_detail
