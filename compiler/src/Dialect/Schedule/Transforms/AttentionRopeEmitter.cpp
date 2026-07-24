#include "AttentionEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::attention_detail {

void emitRopeOrCast(mlir::IRRewriter& rewriter, stream::AttentionOp op,
    const target::LPUTargetModel& target, int64_t cycle, int64_t hemisphere,
    bool rope, mlir::Value value)
{
    const char* hemisphereName = hemisphere == 0 ? "east" : "west";
    const int64_t aluBase = hemisphere * 8;
    if (!rope) {
        emitVxm(rewriter, op, value, cycle, aluBase, "pass",
            "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
            "fp16", 0, hemisphereName, hemisphereName);
        emitVxm(rewriter, op, value, cycle, aluBase + 1, "pass",
            "stream_f32", 36, 0.0f, "immediate", 0, 0.0f,
            "fp16", 2, hemisphereName, hemisphereName);
        return;
    }
    emitVxm(rewriter, op, value, cycle, aluBase, "multiply",
        "stream_f32", 32, 0.0f, "stream_f16", 40, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, op, value, cycle, aluBase + 1, "multiply",
        "stream_f32", 36, 0.0f, "stream_f16", 42, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, op, value, cycle, aluBase + 3, "multiply",
        "stream_f32", 36, 0.0f, "stream_f16", 40, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, op, value, cycle, aluBase + 4, "multiply",
        "stream_f32", 32, 0.0f, "stream_f16", 42, 0.0f,
        "fp32", -1, hemisphereName, hemisphereName);
    emitVxm(rewriter, op, value, cycle + 1, aluBase + 2, "subtract",
        "alu", aluBase, 0.0f, "alu", aluBase + 1, 0.0f,
        "fp16", 0, hemisphereName, hemisphereName);
    emitVxm(rewriter, op, value, cycle + 1, aluBase + 5, "add",
        "alu", aluBase + 3, 0.0f, "alu", aluBase + 4, 0.0f,
        "fp16", 2, hemisphereName, hemisphereName);
}

} // namespace ftlpu::compiler::schedule::attention_detail
