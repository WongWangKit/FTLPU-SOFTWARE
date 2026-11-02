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
    if (!std::isfinite(getRhsScale().convertToDouble())
        || getRhsScale().convertToDouble() <= 0.0)
        return emitOpError("requires a finite positive rhs_scale");
    return success();
}

LogicalResult BiasAddOp::verify()
{
    const auto input = getInput().getType();
    const auto bias = getBias().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !bias.hasStaticShape()
        || input.getRank() != 2 || bias.getRank() != 1 || result != input)
        return emitOpError(
            "requires a static rank-2 input/result and rank-1 bias");
    if (bias.getDimSize(0) != input.getDimSize(1)
        || bias.getElementType() != input.getElementType())
        return emitOpError(
            "requires a bias matching the final input dimension");
    return success();
}

LogicalResult ReshapeOp::verify()
{
    const auto input = getInput().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !result.hasStaticShape())
        return emitOpError("requires static tensor shapes");
    if (input.getNumElements() != result.getNumElements()
        || input.getElementType() != result.getElementType())
        return emitOpError("must preserve element count and element type");
    return success();
}

LogicalResult IndexedViewOp::verify()
{
    const auto output = getResult().getType();
    if (!output.hasStaticShape() || getSources().size() != output.getNumElements()
        || getIndices().size() != getSources().size())
        return emitOpError("requires one logical source/index per output element");
    for (auto [source, index] : llvm::zip(getSources(), getIndices())) {
        if (source < 0 || source >= getInputs().size())
            return emitOpError("source operand index is out of bounds");
        auto input = llvm::cast<RankedTensorType>(getInputs()[source].getType());
        const bool highPlanes = (*this)->hasAttr("truncate_bf16")
            && input.getElementType().isF32() && output.getElementType().isBF16();
        if (!input.hasStaticShape() || (!highPlanes && input.getElementType() != output.getElementType())
            || index < 0 || index >= input.getNumElements())
            return emitOpError("view element is out of bounds or has a different type");
    }
    return success();
}

LogicalResult TransposeOp::verify()
{
    const auto input = getInput().getType();
    const auto result = getResult().getType();
    const auto permutation = getPermutation();
    if (!input.hasStaticShape() || !result.hasStaticShape()
        || input.getRank() != result.getRank()
        || static_cast<int64_t>(permutation.size()) != input.getRank())
        return emitOpError("requires static equal-rank tensors and a full permutation");
    llvm::SmallVector<bool> seen(permutation.size(), false);
    for (std::size_t index = 0; index < permutation.size(); ++index) {
        const int64_t source = permutation[index];
        if (source < 0 || source >= input.getRank() || seen[source]
            || result.getDimSize(index) != input.getDimSize(source))
            return emitOpError("has an invalid permutation or result shape");
        seen[source] = true;
    }
    return success();
}

LogicalResult RopeOp::verify()
{
    if (const auto dim = (*this)->getAttrOfType<mlir::IntegerAttr>("rotary_dim"))
        if (dim.getInt() <= 0 || dim.getInt() > getHeadDim() || dim.getInt() % 2)
            return emitOpError("rotary_dim must be positive, even and <= head_dim");
    if (const auto factor = (*this)->getAttrOfType<mlir::FloatAttr>("attention_factor"))
        if (!std::isfinite(factor.getValueAsDouble()) || factor.getValueAsDouble() <= 0)
            return emitOpError("attention_factor must be finite and positive");
    const auto input = getInput().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || input.getRank() != 3
        || result != input || input.getDimSize(1) != getHeads()
        || input.getDimSize(2) != getHeadDim() || getHeadDim() % 2 != 0)
        return emitOpError("requires [sequence, heads, even head_dim] input and matching result");
    if (!std::isfinite(getTheta().convertToFloat())
        || getTheta().convertToFloat() <= 1.0f)
        return emitOpError("requires a finite theta greater than one");
    return success();
}

LogicalResult GqaBroadcastOp::verify()
{
    const auto input = getInput().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !result.hasStaticShape()
        || input.getRank() != 3 || result.getRank() != 3
        || getQueryHeads() <= 0 || getKvHeads() <= 0
        || getQueryHeads() % getKvHeads() != 0
        || input.getDimSize(1) != getKvHeads()
        || result.getDimSize(1) != getQueryHeads()
        || input.getDimSize(0) != result.getDimSize(0)
        || input.getDimSize(2) != result.getDimSize(2)
        || input.getElementType() != result.getElementType())
        return emitOpError("requires a valid grouped-query head expansion");
    return success();
}

LogicalResult BatchMatmulOp::verify()
{
    const auto lhs = getLhs().getType();
    const auto rhs = getRhs().getType();
    const auto result = getResult().getType();
    if (!lhs.hasStaticShape() || !rhs.hasStaticShape() || !result.hasStaticShape()
        || lhs.getRank() != 3 || rhs.getRank() != 3 || result.getRank() != 3)
        return emitOpError("requires static rank-3 tensors");
    const int64_t rhs_k = getTransposeRhs() ? rhs.getDimSize(2) : rhs.getDimSize(1);
    const int64_t rhs_n = getTransposeRhs() ? rhs.getDimSize(1) : rhs.getDimSize(2);
    if (lhs.getDimSize(0) != rhs.getDimSize(0)
        || lhs.getDimSize(0) != result.getDimSize(0)
        || lhs.getDimSize(2) != rhs_k
        || lhs.getDimSize(1) != result.getDimSize(1)
        || rhs_n != result.getDimSize(2))
        return emitOpError("tensor shapes do not satisfy batched matmul");
    if (getRole() != "qk" && getRole() != "pv")
        return emitOpError("role must be qk or pv");
    return success();
}

LogicalResult SoftmaxOp::verify()
{
    const auto input = getInput().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || result != input)
        return emitOpError("requires matching static input and result tensors");
    const int64_t axis = getAxisAttr().getInt();
    const int64_t rank = input.getRank();
    if (axis < -rank || axis >= rank)
        return emitOpError("axis is outside the tensor rank");
    if (!std::isfinite(getScale().convertToFloat())
        || getScale().convertToFloat() <= 0.0f)
        return emitOpError("requires a finite positive scale");
    return success();
}

LogicalResult L2NormOp::verify()
{
    const auto input = getInput().getType();
    const int64_t axis = getAxisAttr().getInt();
    if (!input.hasStaticShape() || input.getRank() != 2
        || input.getDimSize(0) <= 0 || input.getDimSize(1) <= 0
        || (!input.getElementType().isBF16() && !input.getElementType().isF16())
        || getResult().getType() != input || (axis != 1 && axis != -1))
        return emitOpError("requires a static rank-2 16-bit floating input/result and final axis");
    if (!std::isfinite(getEpsilon().convertToFloat()) || getEpsilon().convertToFloat() <= 0
        || !std::isfinite(getScale().convertToFloat()))
        return emitOpError("requires positive finite epsilon and finite scale");
    return success();
}

LogicalResult RmsNormOp::verify()
{
    const auto input = getInput().getType();
    const auto weight = getWeight().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !weight.hasStaticShape()
        || input.getRank() != 2 || weight.getRank() != 1 || result != input)
        return emitOpError(
            "requires a static rank-2 input/result and rank-1 scale");
    const int64_t axis = getAxisAttr().getInt();
    const int64_t normalizedAxis = axis < 0 ? axis + input.getRank() : axis;
    if (normalizedAxis != input.getRank() - 1
        || weight.getDimSize(0) != input.getDimSize(normalizedAxis)
        || weight.getElementType() != input.getElementType())
        return emitOpError(
            "requires a scale matching the final input dimension");
    if (!std::isfinite(getEpsilon().convertToFloat())
        || getEpsilon().convertToFloat() <= 0.0f)
        return emitOpError("requires a finite positive epsilon");
    return success();
}

LogicalResult SwishOp::verify()
{
    if (getInput().getType() != getResult().getType())
        return emitOpError("requires matching input and result types");
    return success();
}

LogicalResult ScaledMatvecOp::verify()
{
    const auto m=llvm::cast<RankedTensorType>(getMatrix().getType());
    const auto d=llvm::cast<RankedTensorType>(getScale().getType());
    const auto k=llvm::cast<RankedTensorType>(getVector().getType());
    const auto p=llvm::cast<RankedTensorType>(getProduct().getType());
    const auto offset=static_cast<int64_t>(getGroupOffset());
    if(m.getRank()!=2 || d.getRank()!=2 || k.getRank()!=2 || p.getRank()!=2)
        return emitOpError("requires rank-two matrix, scale, vector and product");
    if(!m.hasStaticShape() || !d.hasStaticShape() || !k.hasStaticShape() || !p.hasStaticShape()
        || k.getDimSize(0)<=0 || m.getDimSize(0)<=0 || m.getDimSize(1)<=0
        || m.getDimSize(0)%k.getDimSize(0)
        || d.getDimSize(0)!=1 || offset<0 || offset>d.getDimSize(1)
        || d.getDimSize(1)-offset<k.getDimSize(0)
        || m.getDimSize(1)!=k.getDimSize(1) || getScaled().getType()!=m
        || p.getDimSize(0)!=k.getDimSize(0)
        || p.getDimSize(1)!=m.getDimSize(0)/k.getDimSize(0))
        return emitOpError("inconsistent grouped scaled matvec shapes");
    if(getEpilogue().size()!=0 && getEpilogue().size()!=2 && getEpilogue().size()!=3)
        return emitOpError("epilogue requires value, coefficient and optional readout vector operands");
    if(getEpilogue().empty() ? !getCorrection().empty() :
        getEpilogue().size()==3 ? getCorrection().size()!=3 :
        (getCorrection().size()!=1 && getCorrection().size()!=2))
        return emitOpError("epilogue requires correction, optional updated matrix, and readout when its vector is provided");
    if(!getEpilogue().empty()) {
        auto v=llvm::cast<RankedTensorType>(getEpilogue()[0].getType());
        auto b=llvm::cast<RankedTensorType>(getEpilogue()[1].getType());
        if(v.getShape()!=p.getShape() || !v.getElementType().isBF16()
            || b!=d || getCorrection()[0].getType()!=p)
            return emitOpError("epilogue needs BF16 value shaped like product, coefficient shaped like scale, and FP32 correction");
        if(getCorrection().size()>=2 && getCorrection()[1].getType()!=m)
            return emitOpError("rank-one updated matrix must match matrix type");
        if(getEpilogue().size()==3 && (getEpilogue()[2].getType()!=k || getCorrection()[2].getType()!=p))
            return emitOpError("readout vector must match vector type and readout result must match product type");
    }
    return success();
}

LogicalResult VectorAluOp::verify()
{
    const auto lhs = getLhs().getType();
    const auto rhs = getRhs().getType();
    const auto result = getResult().getType();
    const auto floating = [](Type t) {
        return t.isBF16() || t.isF16() || t.isF32();
    };
    if (getKind()=="weighted_gram") {
        const bool transposed=rhs.getElementType().isBF16();
        const int64_t heads=rhs.getRank()==2 ? (transposed?rhs.getDimSize(0)/32:rhs.getDimSize(1)) : 0;
        // Output rows group even heads before odd heads. This matches the
        // native bilateral stream drain and avoids an output interleave copy.
        if (!lhs.hasStaticShape() || !rhs.hasStaticShape() || !result.hasStaticShape() ||
            lhs.getRank()!=2 || rhs.getRank()!=2 || result.getRank()!=2 ||
            lhs.getDimSize(0)!=32 || heads<=0 || heads>32 || heads%2 ||
            (transposed ? (rhs.getDimSize(0)!=32*heads || rhs.getDimSize(1)!=32) : rhs.getDimSize(0)!=1024) ||
            lhs.getDimSize(1)<=0 || lhs.getDimSize(1)%(32*heads) ||
            !lhs.getElementType().isBF16() || !(transposed || rhs.getElementType().isF32()) ||
            !result.getElementType().isF32() || getUseImmediate() || getReduce() ||
            result.getDimSize(0)!=32*heads || result.getDimSize(1)!=32)
            return emitOpError("weighted_gram requires BF16 K[32,H*D], FP32 weights[1024,H] or native BF16 weights[32*H,32] in [i,h,j] order, FP32 result[H*32,32], even H<=32 and D multiple of 32");
        return success();
    }
    if (getKind() == "causal_exp_difference") {
        const auto transpose=(*this)->getAttrOfType<BoolAttr>("transpose_result");
        const bool transposed=transpose && transpose.getValue();
        const auto scaling=(*this)->getAttrOfType<BoolAttr>("row_scale");
        const bool rowScale=scaling && scaling.getValue();
        // Optional RHS scales each output row; strict_lower excludes its diagonal.
        // A generic pairwise map: y[i*T+j,h] = exp(x[i,h]-x[j,h])
        // below the diagonal, one on the diagonal, and zero above it.
        // transpose_result fuses the lane/time transpose and upper-plane
        // BF16 truncation: y[i*H+h,j], with T=32 for wide SXM capture.
        if (!lhs.hasStaticShape() || !result.hasStaticShape() || lhs.getRank()!=2 ||
            result.getRank()!=2 || lhs.getDimSize(0)<=0 || lhs.getDimSize(0)>32 ||
            lhs.getDimSize(1)<=0 || lhs.getDimSize(1)>32 || rhs!=lhs ||
            !(lhs.getElementType().isF32() || lhs.getElementType().isBF16()) ||
            !(transposed?result.getElementType().isBF16():result.getElementType().isF32()) || getUseImmediate()==rowScale || getReduce() ||
            (transposed && lhs.getDimSize(0)!=32) ||
            result.getDimSize(0)!=lhs.getDimSize(0)*(transposed?lhs.getDimSize(1):lhs.getDimSize(0)) ||
            result.getDimSize(1)!=(transposed?lhs.getDimSize(0):lhs.getDimSize(1)))
            return emitOpError("causal_exp_difference requires [T,H] BF16/FP32 input (same-shaped RHS when row_scale=true) and [T*T,H] FP32 output; transpose_result requires T=32 and BF16 [T*H,T] output, 1<=T,H<=32");
        return success();
    }
    if (!lhs.hasStaticShape() || !rhs.hasStaticShape() || !result.hasStaticShape()
        || lhs.getRank() != 2 || lhs.getDimSize(0) <= 0 || lhs.getDimSize(1) <= 0
        || rhs.getRank() != 2 || result.getRank() != 2
        || (rhs.getDimSize(0) != 1 && rhs.getDimSize(0) != lhs.getDimSize(0))
        || (rhs.getDimSize(1) != 1 && rhs.getDimSize(1) != lhs.getDimSize(1))
        || !floating(lhs.getElementType()) || !floating(rhs.getElementType())
        || !floating(result.getElementType())
        || result.getDimSize(0) != lhs.getDimSize(0)
        || result.getDimSize(1) != (getReduce() ? 1 : lhs.getDimSize(1)))
        return emitOpError("requires static rank-2 floating operands, broadcast-compatible RHS and a map or row-reduction result");
    if (getKind() != "add" && getKind() != "subtract"
        && getKind() != "multiply" && getKind() != "max"
        && getKind() != "negate" && getKind() != "cast"
        && getKind() != "exp" && getKind() != "reciprocal"
        && getKind() != "rsqrt" && getKind() != "log" && getKind() != "sigmoid"
        && getKind() != "softplus" && getKind() != "cumsum_rows")
        return emitOpError("unsupported generic vector ALU operation");
    if (getKind() == "cumsum_rows" &&
        (!getUseImmediate() || getReduce() || !result.getElementType().isF32()
         || lhs.getDimSize(1)>32 || !(lhs.getElementType().isF32() || lhs.getElementType().isBF16())))
        return emitOpError("cumsum_rows requires unary BF16/FP32 input, FP32 output, and at most 32 columns; inclusive scan along rows");
    if ((getKind() == "sigmoid" || getKind() == "softplus") &&
        (!getUseImmediate() || getReduce() || !result.getElementType().isF32()))
        return emitOpError("semantic sigmoid/softplus require unary non-reducing FP32 output");
    if (getReduce() && getKind() != "multiply")
        return emitOpError("row reduction currently sums products");
    return success();
}

LogicalResult DepthwiseConv1DOp::verify()
{
    const auto input = getInput().getType();
    const auto weight = getWeight().getType();
    const auto result = getResult().getType();
    if (!input.hasStaticShape() || !weight.hasStaticShape()
        || input.getRank() != 2 || weight.getRank() != 2
        || result != input)
        return emitOpError(
            "requires static [tokens, channels] input/result and "
            "[channels, kernel] weight");
    if (getKernelSize() <= 0
        || weight.getDimSize(0) != input.getDimSize(1)
        || weight.getDimSize(1) != getKernelSize()
        || weight.getElementType() != input.getElementType())
        return emitOpError("weight shape or type does not match the input");
    if (!getCausal())
        return emitOpError("currently requires causal=true");
    if (getActivation() != "none" && getActivation() != "silu")
        return emitOpError("activation must be none or silu");
    return success();
}

LogicalResult GatedDeltaRuleOp::verify()
{
    const auto query = getQuery().getType();
    const auto key = getKey().getType();
    const auto value = getValue().getType();
    const auto decay = getDecay().getType();
    const auto beta = getBeta().getType();
    const auto state = getInitialState().getType();
    const auto output = getOutput().getType();
    const auto finalState = getFinalState().getType();
    if (!query.hasStaticShape() || !key.hasStaticShape()
        || !value.hasStaticShape() || !decay.hasStaticShape()
        || !beta.hasStaticShape() || !state.hasStaticShape()
        || !output.hasStaticShape() || !finalState.hasStaticShape()
        || query.getRank() != 2 || key.getRank() != 2
        || value.getRank() != 2 || decay.getRank() != 2
        || beta.getRank() != 2 || state.getRank() != 2
        || output.getRank() != 2 || finalState.getRank() != 2)
        return emitOpError("requires static rank-2 tensors");
    if (getHeads() <= 0 || getKeyDim() <= 0 || getValueDim() <= 0)
        return emitOpError("heads and head dimensions must be positive");
    const int64_t tokens = query.getDimSize(0);
    if (key.getShape() != query.getShape()
        || query.getDimSize(1) != getHeads() * getKeyDim()
        || value.getDimSize(0) != tokens
        || value.getDimSize(1) != getHeads() * getValueDim()
        || output.getShape() != value.getShape()
        || decay.getDimSize(0) != tokens
        || decay.getDimSize(1) != getHeads()
        || beta.getShape() != decay.getShape()
        || state.getDimSize(0) != getHeads() * getKeyDim()
        || state.getDimSize(1) != getValueDim()
        || finalState != state)
        return emitOpError("tensor shapes do not match the recurrent domain");
    if (query.getElementType() != key.getElementType()
        || query.getElementType() != value.getElementType()
        || query.getElementType() != output.getElementType()
        || (!query.getElementType().isF16()
            && !query.getElementType().isBF16()))
        return emitOpError("Q/K/V/output must use one 16-bit float type");
    if ((!state.getElementType().isF32() && !state.getElementType().isBF16())
        || decay.getElementType()!=state.getElementType()
        || beta.getElementType()!=state.getElementType())
        return emitOpError("decay, beta, and recurrent state must share FP32 or BF16 storage");
    return success();
}

LogicalResult ElementwiseOp::verify()
{
    if (getLhs().getType() != getRhs().getType()
        || getLhs().getType() != getResult().getType())
        return emitOpError("requires matching operand and result types");
    if (getKind() != "multiply" && getKind() != "add")
        return emitOpError("kind must be multiply or add");
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

} // namespace ftlpu::compiler::kernel
