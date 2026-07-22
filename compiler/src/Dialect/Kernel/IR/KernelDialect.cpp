#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include <cmath>

using namespace mlir;

#include "ftlpu/compiler/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ftlpu/compiler/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace ftlpu::compiler::kernel {

void KernelDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "ftlpu/compiler/Dialect/Kernel/IR/KernelOps.cpp.inc"
        >();
}

LogicalResult MatmulOp::verify()
{
    const auto lhs_type = getLhs().getType();
    const auto rhs_type = getRhs().getType();
    const auto result_type = getResult().getType();
    if (lhs_type.getRank() != 2 || rhs_type.getRank() != 2 || result_type.getRank() != 2)
        return emitOpError("requires rank-2 tensors");
    if (!lhs_type.hasStaticShape() || !rhs_type.hasStaticShape() || !result_type.hasStaticShape())
        return emitOpError("requires static tensor shapes");
    if (lhs_type.getDimSize(0) != getM() || lhs_type.getDimSize(1) != getK()
        || rhs_type.getDimSize(0) != getK() || rhs_type.getDimSize(1) != getN()
        || result_type.getDimSize(0) != getM() || result_type.getDimSize(1) != getN())
        return emitOpError("tensor shapes do not match m, n, and k");
    if (getUnit() != "MXM") return emitOpError("unit must be MXM");
    return success();
}

LogicalResult SwigluOp::verify()
{
    const auto input = getInput().getType();
    const auto gate = getGateWeight().getType();
    const auto up = getUpWeight().getType();
    const auto result = getResult().getType();
    if (input.getRank() != 2 || gate.getRank() != 2 || up.getRank() != 2
        || result.getRank() != 2 || !input.hasStaticShape() || !gate.hasStaticShape()
        || !up.hasStaticShape() || !result.hasStaticShape())
        return emitOpError("requires static rank-2 tensors");
    if (input.getDimSize(0) != getM() || input.getDimSize(1) != getK()
        || gate.getDimSize(0) != getK() || gate.getDimSize(1) != getN()
        || up.getShape() != gate.getShape() || result.getDimSize(0) != getM()
        || result.getDimSize(1) != getN())
        return emitOpError("tensor shapes do not match m, n, and k");
    if (!input.getElementType().isInteger(8) || !gate.getElementType().isInteger(8)
        || !up.getElementType().isInteger(8) || !result.getElementType().isInteger(8))
        return emitOpError("requires int8 inputs, weights, and output");
    if (!std::isfinite(getGateScale().convertToFloat())
        || !std::isfinite(getUpScale().convertToFloat())
        || !std::isfinite(getOutputScale().convertToFloat())
        || getOutputScale().convertToFloat() <= 0.0f)
        return emitOpError("requires finite scales and a positive output scale");
    return success();
}

LogicalResult FfnOp::verify()
{
    const auto input = getInput().getType();
    const auto gate = getGateWeight().getType();
    const auto up = getUpWeight().getType();
    const auto down = getDownWeight().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !gate.hasStaticShape() || !up.hasStaticShape()
        || !down.hasStaticShape() || !result.hasStaticShape()
        || input.getRank() != 2 || gate.getRank() != 2 || up.getRank() != 2
        || down.getRank() != 2 || result.getRank() != 2)
        return emitOpError("requires static rank-2 tensors");
    if (input.getDimSize(0) != static_cast<int64_t>(getM())
        || input.getDimSize(1) != static_cast<int64_t>(getK())
        || gate.getDimSize(0) != static_cast<int64_t>(getK())
        || gate.getDimSize(1) != static_cast<int64_t>(getHidden())
        || up.getShape() != gate.getShape()
        || down.getDimSize(0) != static_cast<int64_t>(getHidden())
        || down.getDimSize(1) != static_cast<int64_t>(getN())
        || result.getDimSize(0) != static_cast<int64_t>(getM())
        || result.getDimSize(1) != static_cast<int64_t>(getN()))
        return emitOpError("tensor shapes do not match m, k, hidden, and n");
    const bool legacy_w8a8 = getK() == 320 && getHidden() == 640 && getN() == 320
        && input.getElementType().isInteger(8) && gate.getElementType().isInteger(8)
        && up.getElementType().isInteger(8) && down.getElementType().isInteger(8)
        && result.getElementType().isInteger(8);
    const target::LPUTargetModel target;
    const bool w8a16 = input.getElementType().isF16() && gate.getElementType().isInteger(8)
        && up.getElementType().isInteger(8) && down.getElementType().isInteger(8)
        && result.getElementType().isF16()
        && target.supports_w8a16_ffn_shape(getM(), getK(), getHidden(), getN());
    if (!legacy_w8a8 && !w8a16)
        return emitOpError("requires the legacy W8A8 profile or a tile-aligned W8A16 FFN");
    for (float scale : {getGateScale().convertToFloat(), getUpScale().convertToFloat(),
             getHiddenScale().convertToFloat(), getDownLhsScale().convertToFloat(),
             getDownRhsScale().convertToFloat(), getOutputScale().convertToFloat()})
        if (!std::isfinite(scale) || scale <= 0.0f)
            return emitOpError("requires finite positive quantization scales");
    return success();
}

LogicalResult AttentionOp::verify()
{
    const auto input = getInput().getType();
    const auto query = getQueryWeight().getType();
    const auto key = getKeyWeight().getType();
    const auto value = getValueWeight().getType();
    const auto output = getOutputWeight().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !query.hasStaticShape() || !key.hasStaticShape()
        || !value.hasStaticShape() || !output.hasStaticShape() || !result.hasStaticShape()
        || input.getRank() != 2 || query.getRank() != 2 || key.getRank() != 2
        || value.getRank() != 2 || output.getRank() != 2 || result.getRank() != 2)
        return emitOpError("requires static rank-2 tensors");
    const int64_t query_width = getQueryHeads() * getHeadDim();
    const int64_t kv_width = getKvHeads() * getHeadDim();
    if (getSeqLen() <= 0 || getHidden() <= 0 || getQueryHeads() <= 0 || getKvHeads() <= 0
        || getHeadDim() <= 0 || getQueryHeads() % getKvHeads() != 0)
        return emitOpError("requires positive dimensions and an integral GQA group size");
    if (input.getDimSize(0) != getSeqLen() || input.getDimSize(1) != getHidden()
        || query.getDimSize(0) != getHidden() || query.getDimSize(1) != query_width
        || key.getDimSize(0) != getHidden() || key.getDimSize(1) != kv_width
        || value.getShape() != key.getShape()
        || output.getDimSize(0) != query_width || output.getDimSize(1) != getHidden()
        || result.getDimSize(0) != getSeqLen() || result.getDimSize(1) != getHidden())
        return emitOpError("tensor shapes do not match the attention configuration");
    if (getHeadDim() % 32 != 0 || getSeqLen() % 32 != 0 || getHidden() % 32 != 0)
        return emitOpError("currently requires 32-element MXM tile alignment");
    if (!std::isfinite(getRopeTheta().convertToFloat()) || getRopeTheta().convertToFloat() <= 1.0f)
        return emitOpError("requires a finite RoPE theta greater than one");
    return success();
}

} // namespace ftlpu::compiler::kernel
