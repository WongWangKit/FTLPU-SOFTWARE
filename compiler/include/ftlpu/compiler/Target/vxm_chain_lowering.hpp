#pragma once

#include "ftlpu/vxm/compact_instruction.hpp"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ftlpu::compiler::target {

inline VxmLaneOperand lowerVxmChainOperand(llvm::StringRef kind,
    int64_t index, float immediate, int64_t queue, bool rhs,
    llvm::StringRef source)
{
    if (queue < 0 || queue >= 8)
        throw std::runtime_error("chain VXM has 8 shared ICU queues (C0..C7); explicit mirrored queues require re-lowering");
    if (kind == "previous") return VxmLaneOperand::Previous();
    if (kind == "original") return VxmLaneOperand::Original();
    if (kind == "auxiliary") return VxmLaneOperand::Aux();
    if (kind == "accumulator") return VxmLaneOperand::Acc();
    if (kind == "feedback") return VxmLaneOperand::Feedback();
    if (kind == "immediate") return VxmLaneOperand::Imm(immediate);
    if (kind == "alu" && index == queue - 1) return VxmLaneOperand::Previous();
    if (kind == "stream_f16" || kind == "stream_bf16") {
        const auto expected = 2 * VxmLane::fixed_input_group_for_stage(queue, rhs);
        if (index < 0 || index >= 64 || index % 32 != expected)
            throw std::runtime_error("chain VXM fixed input port requires stream "
                + std::to_string(expected) + "; requested " + std::to_string(index)
                + ". Re-layout MEM inputs instead of discarding the stream index");
        if (!source.empty() && source != "local" && source != "east")
            throw std::runtime_error("chain VXM absolute input hemisphere requires a configured group mux; current runtime uses mirrored local inputs");
        return kind == "stream_f16" ? VxmLaneOperand::StreamFloat16()
                                    : VxmLaneOperand::StreamBFloat16();
    }
    throw std::runtime_error("operand " + kind.str()
        + " requires chain legalization; chain VXM cannot consume arbitrary SR/FP32 stream operands");
}

template <typename Op>
inline void validateVxmChainControls(Op op)
{
    if (op.getLhsStreamTap().value_or(false) || op.getRhsStreamTap().value_or(false)
        || op.getAccumulatorResetFirst().value_or(false)
        || op.getAccumulatorEmitLast().value_or(false))
        throw std::runtime_error("explicit-SR tap/terminal controls are not encoded by chain VXM; split or re-lower the operation");
}

} // namespace ftlpu::compiler::target
