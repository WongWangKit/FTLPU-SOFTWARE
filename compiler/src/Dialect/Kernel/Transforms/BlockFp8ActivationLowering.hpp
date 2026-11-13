#pragma once

#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "mlir/IR/PatternMatch.h"

#include <cmath>
#include <utility>

namespace ftlpu::compiler::kernel {

// Both operators share a group scale. The QDQ result carries its dequantized
// BF16 values; byte-valued E4M3FN/E8M0 results require separate lowering.
inline void expandBlockFp8Arithmetic(mlir::Operation* op, mlir::Value x,
                                    bool qdq, mlir::IRRewriter& rewriter,
                                    bool groupsInColumns = false)
{
    rewriter.setInsertionPoint(op);
    const auto inputType = llvm::cast<mlir::RankedTensorType>(x.getType());
    const auto rowType = mlir::RankedTensorType::get(
        groupsInColumns ? llvm::SmallVector<int64_t>{1,inputType.getDimSize(1)}
                        : llvm::SmallVector<int64_t>{inputType.getDimSize(0),1},
        rewriter.getF32Type());
    const auto alu = [&](llvm::StringRef kind, mlir::Value lhs, mlir::Value rhs,
                         float immediate, mlir::Type type, bool reduce = false) {
        mlir::OperationState state(op->getLoc(), VectorAluOp::getOperationName());
        state.addOperands({lhs, rhs ? rhs : lhs});
        state.addTypes(type);
        state.addAttribute("kind", rewriter.getStringAttr(kind));
        state.addAttribute("use_immediate", rewriter.getBoolAttr(!rhs));
        state.addAttribute("immediate", rewriter.getF32FloatAttr(immediate));
        state.addAttribute("reduce", rewriter.getBoolAttr(reduce));
        state.addAttribute("ftlpu.vector_require_fp32", rewriter.getUnitAttr());
        if (reduce) {
            state.addAttribute("reduction_kind", rewriter.getStringAttr("max"));
            if (groupsInColumns) {
                state.addAttribute("reduction_axis", rewriter.getI64IntegerAttr(0));
                state.addAttribute("reduction_seed_first", rewriter.getBoolAttr(true));
            }
        }
        if (const auto region = op->getAttr("ftlpu.logical_region"))
            state.addAttribute("ftlpu.logical_region", region);
        return rewriter.create(state)->getResult(0);
    };
    const auto scalar = [&](llvm::StringRef kind, mlir::Value lhs, float immediate) {
        return alu(kind, lhs, {}, immediate, lhs.getType());
    };
    const auto binary = [&](llvm::StringRef kind, mlir::Value lhs, mlir::Value rhs) {
        return alu(kind, lhs, rhs, 0.f, lhs.getType());
    };
    // Each boundary separates consecutive BF16 magnitudes. A full BF16 ULP
    // past it already saturates the mask, so both outputs are powers of two.
    const auto factors = [&](mlir::Value value, float boundary, float inverseUlp,
                             int stride, bool needInverse) {
        const auto distance = scalar("subtract", value, boundary);
        const auto positive = scalar("max", scalar("multiply", distance, inverseUlp), 0.f);
        const auto mask = scalar("max", scalar("negate", positive, 0.f), -1.f);
        const auto forward = scalar("add",
            scalar("multiply", mask, 1.f - std::ldexp(1.f, stride)), 1.f);
        const auto inverse = needInverse
            ? scalar("max", scalar("add", mask, 1.f), std::ldexp(1.f, -stride))
            : mlir::Value{};
        return std::make_pair(forward, inverse);
    };
    const auto product = [&](mlir::Value factor, mlir::Value prior, float initial) {
        return prior ? binary("multiply", factor, prior) : scalar("multiply", factor, initial);
    };

    const auto negated = alu("negate", x, {}, 0.f, x.getType());
    const auto absolute = alu("max", x, negated, 0.f, x.getType());
    auto maximum = alu(groupsInColumns?"multiply":"max", absolute, {},
                       groupsInColumns?1.f:0.f, rowType, true);

    // The 1e-4 floor gives a minimum scale of 2^-22. Finite BF16 values
    // need at most 2^120. At each candidate exponent, a strict comparison
    // distinguishes equality from the next possible BF16 magnitude (one ULP
    // = threshold/224). The clamp makes the decision exactly 0 or -1.
    mlir::Value scale, inverseScale;
    for (int stride : {64, 64, 32, 16, 8, 4, 2, 1}) {
        const float threshold = std::ldexp(448.f, -22 + stride - 1);
        const auto [factor, inverse] = factors(maximum, threshold, 224.f / threshold,
                                               stride, qdq || stride > 1);
        scale = product(factor, scale, 0x1p-22f);
        if (qdq) inverseScale = product(inverse, inverseScale, 0x1p22f);
        if (stride > 1) maximum = binary("multiply", maximum, inverse);
    }
    if (!qdq) {
        rewriter.replaceOp(op, scale);
        return;
    }

    const auto fullType = mlir::RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
    const auto normalized = alu("multiply", x, inverseScale, 0.f, fullType);
    auto magnitude = binary("max", normalized, scalar("negate", normalized, 0.f));
    mlir::Value step, inverseStep;
    // Subnormals have spacing 2^-9. Above 2^-6 each following exponent
    // doubles the spacing. The predecessor of an exact power of two has
    // relative distance 2^-8 in BF16, including after power-of-two scaling.
    for (int stride : {8, 4, 2, 1}) {
        const float threshold = std::ldexp(1.f, -6 + stride);
        const auto [factor, inverse] = factors(magnitude, threshold * (1.f - 0x1p-8f),
                                               256.f / threshold, stride, true);
        step = product(factor, step, 0x1p-9f);
        inverseStep = product(inverse, inverseStep, 0x1p9f);
        if (stride > 1) magnitude = binary("multiply", magnitude, inverse);
    }
    // An FP32 ULP is exactly one at 1.5*2^23. The add rounds the signed
    // significand to the nearest even integer. Restore the input zero sign
    // before multiplying back by the spacing and the group scale.
    const auto mantissa = binary("multiply", normalized, inverseStep);
    constexpr float unitUlp = 0x1.8p23f;
    const auto integer = scalar("subtract", scalar("add", mantissa, unitUlp), unitUlp);
    const auto signedZero = alu("multiply", x, {}, 0.f, fullType);
    const auto positive = binary("max", signedZero, integer);
    const auto negative = binary("max", scalar("negate", signedZero, 0.f),
                                 scalar("negate", integer, 0.f));
    const auto rounded = binary("multiply", binary("subtract", positive, negative), step);
    rewriter.replaceOp(op, alu("multiply", rounded, scale, 0.f, inputType));
}

} // namespace ftlpu::compiler::kernel
