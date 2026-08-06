#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule::attention_detail {

void emitRopeOrCast(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle, int64_t hemisphere,
    bool rope, mlir::Value value, mlir::Type elementType)
{
    const char* hemisphereName = hemisphere == 0 ? "east" : "west";
    const int64_t aluBase = hemisphere * 8;
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(elementType);
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    if (!rope) {
        emitVxm(rewriter, location, value, cycle, aluBase, "pass",
            "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 0, hemisphereName, hemisphereName);
        emitVxm(rewriter, location, value, cycle, aluBase + 1, "pass",
            "stream_f32", 36, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 2, hemisphereName, hemisphereName);
        return;
    }
    emitVxm(rewriter, location, value, cycle, aluBase, "multiply",
        "stream_f32", 32, 0.0f, streamKind, 40, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle, aluBase + 1, "multiply",
        "stream_f32", 36, 0.0f, streamKind, 42, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle, aluBase + 3, "multiply",
        "stream_f32", 36, 0.0f, streamKind, 40, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle, aluBase + 4, "multiply",
        "stream_f32", 32, 0.0f, streamKind, 42, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle + 1, aluBase + 2, "subtract",
        "alu", aluBase, 0.0f, "alu", aluBase + 1, 0.0f,
        dataFormat, 0, hemisphereName, hemisphereName);
    emitVxm(rewriter, location, value, cycle + 1, aluBase + 5, "add",
        "alu", aluBase + 3, 0.0f, "alu", aluBase + 4, 0.0f,
        dataFormat, 2, hemisphereName, hemisphereName);
}

} // namespace ftlpu::compiler::schedule::attention_detail
