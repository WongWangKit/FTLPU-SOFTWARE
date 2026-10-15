#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"

#include <cmath>
#include <optional>
#include <algorithm>
#include <limits>
#include <functional>

namespace ftlpu::compiler {
namespace {

bool named(mlir::Operation* op, llvm::StringRef name)
{
    return op && op->getName().getStringRef() == name;
}

void collect_backward_slice(mlir::Value value,
    llvm::SmallPtrSetImpl<mlir::Operation*>& operations)
{
    mlir::Operation* operation = value.getDefiningOp();
    if (!operation || !operations.insert(operation).second) return;
    for (mlir::Value operand : operation->getOperands())
        collect_backward_slice(operand, operations);
}

mlir::Operation* find_backward(mlir::Value value, llvm::StringRef name,
    llvm::SmallPtrSetImpl<mlir::Operation*>& visited)
{
    mlir::Operation* operation = value.getDefiningOp();
    if (!operation || !visited.insert(operation).second) return nullptr;
    if (named(operation, name)) return operation;
    for (mlir::Value operand : operation->getOperands())
        if (mlir::Operation* found = find_backward(operand, name, visited))
            return found;
    return nullptr;
}

mlir::Operation* find_backward(mlir::Value value, llvm::StringRef name)
{
    llvm::SmallPtrSet<mlir::Operation*, 32> visited;
    return find_backward(value, name, visited);
}

bool backward_slice_contains(mlir::Value value, llvm::StringRef name)
{
    return find_backward(value, name) != nullptr;
}

struct AttentionMatch {
    mlir::Operation* output_dot;
    mlir::Value input;
    mlir::Value query_weight;
    mlir::Value key_weight;
    mlir::Value value_weight;
    mlir::Value output_weight;
    mlir::Value query_bias;
    mlir::Value key_bias;
    mlir::Value value_bias;
    float query_scale;
    float key_scale;
    float value_scale;
    float output_scale;
    int64_t seq_len;
    int64_t hidden;
    int64_t query_heads;
    int64_t kv_heads;
    int64_t head_dim;
    float rope_theta;
    bool causal;
    llvm::SmallPtrSet<mlir::Operation*, 32> operations;
};

std::optional<float> scalar_constant_value(mlir::Operation* operation)
{
    if (!named(operation, "stablehlo.constant")) return std::nullopt;
    const auto value =
        operation->getAttrOfType<mlir::DenseFPElementsAttr>("value");
    if (!value || !value.isSplat() || value.getNumElements() != 1)
        return std::nullopt;
    return value.getSplatValue<llvm::APFloat>().convertToFloat();
}

std::optional<float> match_rope_theta(mlir::Operation* qk_dot)
{
    mlir::Operation* power =
        find_backward(qk_dot->getOperand(0), "stablehlo.power");
    if (!power || power->getNumOperands() != 2) return std::nullopt;
    mlir::Operation* constant =
        find_backward(power->getOperand(0), "stablehlo.constant");
    const auto theta = scalar_constant_value(constant);
    if (theta && *theta > 1.0f) return theta;
    return std::nullopt;
}

mlir::Operation* unary_input(mlir::Operation* op, llvm::StringRef name)
{
    return named(op, name) && op->getNumOperands() == 1 && op->getNumResults() == 1
        ? op->getOperand(0).getDefiningOp() : nullptr;
}

struct SwigluMatch {
    mlir::Operation* final_convert;
    mlir::Operation* swiglu_mul;
    mlir::Operation* gated_mul;
    mlir::Operation* logistic;
    mlir::Operation* gate_convert;
    mlir::Operation* up_convert;
    mlir::Operation* gate_dot;
    mlir::Operation* up_dot;
};

struct FfnMatch {
    mlir::Operation* final_convert;
    mlir::Operation* down_dot;
    SwigluMatch swiglu;
};

struct W8A16FfnMatch {
    mlir::Operation* final_convert;
    mlir::Operation* down_dot;
    mlir::Operation* swiglu_mul;
    mlir::Operation* gated_mul;
    mlir::Operation* logistic;
    mlir::Operation* gate_dot;
    mlir::Operation* up_dot;
    mlir::Value input;
    mlir::Value gate_weight;
    mlir::Value up_weight;
    mlir::Value down_weight;
    float gate_scale;
    float up_scale;
    float down_scale;
};

struct RmsNormMatch {
    mlir::Operation* final_convert;
    mlir::Value input;
    mlir::Value weight;
    float epsilon;
    llvm::SmallPtrSet<mlir::Operation*, 16> operations;
};

mlir::Value converted_source(mlir::Value value)
{
    auto* op = value.getDefiningOp();
    return named(op, "stablehlo.convert") && op->getNumOperands() == 1
        ? op->getOperand(0) : mlir::Value{};
}

mlir::Operation* defining(mlir::Value value, llvm::StringRef name)
{
    mlir::Operation* operation = value.getDefiningOp();
    return named(operation, name) ? operation : nullptr;
}

mlir::Value strip_broadcast(mlir::Value value)
{
    mlir::Operation* operation = value.getDefiningOp();
    return named(operation, "stablehlo.broadcast_in_dim")
            && operation->getNumOperands() == 1
        ? operation->getOperand(0)
        : value;
}

std::optional<float> scalar_constant(mlir::Value value)
{
    value = strip_broadcast(value);
    mlir::Operation* operation = value.getDefiningOp();
    if (named(operation, "stablehlo.convert")
        && operation->getNumOperands() == 1)
        operation = operation->getOperand(0).getDefiningOp();
    if (!named(operation, "stablehlo.constant")) return std::nullopt;
    const auto elements =
        operation->getAttrOfType<mlir::DenseFPElementsAttr>("value");
    if (!elements || !elements.isSplat()) return std::nullopt;
    return elements.getSplatValue<llvm::APFloat>().convertToFloat();
}

struct ScaledWeight {
    mlir::Value storage;
    float scale;
};

std::optional<ScaledWeight> match_scaled_weight(mlir::Value value)
{
    if (mlir::Value storage = converted_source(value))
        return ScaledWeight {storage, 1.0f};
    auto* multiply = defining(value, "stablehlo.multiply");
    if (!multiply || multiply->getNumOperands() != 2) return std::nullopt;
    for (unsigned order = 0; order < 2; ++order) {
        mlir::Value storage =
            converted_source(multiply->getOperand(order));
        const auto scale =
            scalar_constant(multiply->getOperand(1 - order));
        if (storage && scale && std::isfinite(*scale) && *scale > 0.0f)
            return ScaledWeight {storage, *scale};
    }
    return std::nullopt;
}

mlir::Value match_projection_bias(mlir::Operation* dot)
{
    if (!dot || dot->getNumResults() != 1) return {};
    const auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(
        dot->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2) return {};
    for (mlir::Operation* user : dot->getResult(0).getUsers()) {
        if (!named(user, "stablehlo.add") || user->getNumOperands() != 2
            || user->getNumResults() != 1)
            continue;
        for (unsigned order = 0; order < 2; ++order) {
            if (user->getOperand(order) != dot->getResult(0)) continue;
            auto broadcast = user->getOperand(1 - order)
                .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
            if (!broadcast || broadcast.getBroadcastDimensions()
                    != llvm::ArrayRef<int64_t>({1})) continue;
            mlir::Value bias = broadcast.getOperand();
            const auto biasType = llvm::dyn_cast<mlir::RankedTensorType>(
                bias.getType());
            if (biasType && biasType.getRank() == 1
                && biasType.hasStaticShape()
                && biasType.getDimSize(0) == resultType.getDimSize(1)
                && biasType.getElementType() == resultType.getElementType())
                return bias;
        }
    }
    return {};
}

std::optional<RmsNormMatch> match_rms_norm(
    mlir::Operation* final_convert)
{
    if (!named(final_convert, "stablehlo.convert")
        || final_convert->getNumOperands() != 1
        || final_convert->getNumResults() != 1)
        return std::nullopt;
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(
        final_convert->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2) return std::nullopt;

    auto* outputMultiply =
        defining(final_convert->getOperand(0), "stablehlo.multiply");
    if (!outputMultiply || outputMultiply->getNumOperands() != 2)
        return std::nullopt;

    for (unsigned scaleOrder = 0; scaleOrder < 2; ++scaleOrder) {
        mlir::Value scaledValue = outputMultiply->getOperand(scaleOrder);
        mlir::Value weightValue =
            strip_broadcast(outputMultiply->getOperand(1 - scaleOrder));
        mlir::Value weight = converted_source(weightValue);
        if (!weight) weight = weightValue;
        auto weightType =
            llvm::dyn_cast<mlir::RankedTensorType>(weight.getType());
        auto* scaled = defining(scaledValue, "stablehlo.multiply");
        if (!scaled || !weightType || weightType.getRank() != 1)
            continue;

        for (unsigned inputOrder = 0; inputOrder < 2; ++inputOrder) {
            mlir::Value inputValue = scaled->getOperand(inputOrder);
            mlir::Value input = converted_source(inputValue);
            if (!input) continue;
            mlir::Value invBroadcast =
                scaled->getOperand(1 - inputOrder);
            mlir::Value invValue = strip_broadcast(invBroadcast);
            auto* rsqrt = defining(invValue, "stablehlo.rsqrt");
            if (!rsqrt || rsqrt->getNumOperands() != 1) continue;
            auto* epsilonAdd =
                defining(rsqrt->getOperand(0), "stablehlo.add");
            if (!epsilonAdd || epsilonAdd->getNumOperands() != 2)
                continue;

            mlir::Operation* mean = nullptr;
            std::optional<float> epsilon;
            for (unsigned meanOrder = 0; meanOrder < 2; ++meanOrder) {
                mean = defining(
                    epsilonAdd->getOperand(meanOrder), "stablehlo.divide");
                epsilon = scalar_constant(
                    epsilonAdd->getOperand(1 - meanOrder));
                if (mean && epsilon) break;
            }
            if (!mean || !epsilon || *epsilon <= 0.0f
                || mean->getNumOperands() != 2)
                continue;

            mlir::Operation* reduce = nullptr;
            for (mlir::Value operand : mean->getOperands()) {
                mlir::Value source = strip_broadcast(operand);
                reduce = defining(source, "stablehlo.reduce");
                if (reduce) break;
            }
            if (!reduce || reduce->getNumOperands() < 1) continue;
            auto* square =
                defining(reduce->getOperand(0), "stablehlo.multiply");
            if (!square || square->getNumOperands() != 2
                || square->getOperand(0) != inputValue
                || square->getOperand(1) != inputValue)
                continue;

            RmsNormMatch match {
                final_convert, input, weight, *epsilon, {}};
            const auto remember = [&](mlir::Operation* operation) {
                if (operation) match.operations.insert(operation);
            };
            remember(final_convert);
            remember(outputMultiply);
            remember(scaled);
            remember(inputValue.getDefiningOp());
            remember(outputMultiply->getOperand(1 - scaleOrder)
                         .getDefiningOp());
            remember(weightValue.getDefiningOp());
            remember(invBroadcast.getDefiningOp());
            remember(invValue.getDefiningOp());
            remember(rsqrt);
            remember(epsilonAdd);
            remember(mean);
            remember(reduce);
            remember(square);
            for (mlir::Value operand : reduce->getOperands())
                remember(operand.getDefiningOp());
            for (mlir::Value operand : epsilonAdd->getOperands())
                remember(operand.getDefiningOp());
            for (mlir::Value operand : mean->getOperands())
                remember(operand.getDefiningOp());
            return match;
        }
    }
    return std::nullopt;
}

std::optional<AttentionMatch> match_standard_attention(mlir::Operation* output_dot)
{
    if (!named(output_dot, "stablehlo.dot_general")
        || output_dot->getNumOperands() != 2 || output_dot->getNumResults() != 1)
        return std::nullopt;

    const auto result_type =
        llvm::dyn_cast<mlir::RankedTensorType>(output_dot->getResult(0).getType());
    const auto output_weight_type =
        llvm::dyn_cast<mlir::RankedTensorType>(output_dot->getOperand(1).getType());
    if (!result_type || !output_weight_type || result_type.getRank() != 2
        || output_weight_type.getRank() != 2)
        return std::nullopt;

    mlir::Operation* pv_dot =
        find_backward(output_dot->getOperand(0), "stablehlo.dot_general");
    if (!pv_dot || pv_dot->getNumOperands() != 2) return std::nullopt;
    mlir::Operation* qk_dot =
        find_backward(pv_dot->getOperand(0), "stablehlo.dot_general");
    mlir::Operation* value_dot =
        find_backward(pv_dot->getOperand(1), "stablehlo.dot_general");
    if (!qk_dot || !value_dot || qk_dot->getNumOperands() != 2)
        return std::nullopt;
    mlir::Operation* query_dot =
        find_backward(qk_dot->getOperand(0), "stablehlo.dot_general");
    mlir::Operation* key_dot =
        find_backward(qk_dot->getOperand(1), "stablehlo.dot_general");
    if (!query_dot || !key_dot || query_dot->getNumOperands() != 2
        || key_dot->getNumOperands() != 2 || value_dot->getNumOperands() != 2)
        return std::nullopt;

    mlir::Value input = query_dot->getOperand(0);
    if (key_dot->getOperand(0) != input || value_dot->getOperand(0) != input)
        return std::nullopt;
    const auto query_weight = match_scaled_weight(query_dot->getOperand(1));
    const auto key_weight = match_scaled_weight(key_dot->getOperand(1));
    const auto value_weight = match_scaled_weight(value_dot->getOperand(1));
    const auto output_weight = match_scaled_weight(output_dot->getOperand(1));
    if (!query_weight || !key_weight || !value_weight || !output_weight)
        return std::nullopt;

    const auto input_type = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    const auto qk_lhs_type =
        llvm::dyn_cast<mlir::RankedTensorType>(qk_dot->getOperand(0).getType());
    const auto key_weight_storage =
        llvm::dyn_cast<mlir::RankedTensorType>(key_weight->storage.getType());
    if (!input_type || !qk_lhs_type || !key_weight_storage
        || input_type.getRank() != 2 || qk_lhs_type.getRank() != 3
        || key_weight_storage.getRank() != 2)
        return std::nullopt;

    const int64_t query_heads = qk_lhs_type.getDimSize(0);
    const int64_t head_dim = qk_lhs_type.getDimSize(2);
    if (query_heads <= 0 || head_dim <= 0
        || key_weight_storage.getDimSize(1) % head_dim != 0)
        return std::nullopt;

    const bool has_softmax =
        backward_slice_contains(pv_dot->getOperand(0), "stablehlo.exponential")
        && backward_slice_contains(pv_dot->getOperand(0), "stablehlo.reduce")
        && backward_slice_contains(pv_dot->getOperand(0), "stablehlo.divide");
    if (!has_softmax) return std::nullopt;
    const bool causal =
        backward_slice_contains(pv_dot->getOperand(0), "stablehlo.select")
        && backward_slice_contains(pv_dot->getOperand(0), "stablehlo.compare")
        && backward_slice_contains(pv_dot->getOperand(0), "stablehlo.iota");

    AttentionMatch match {
        output_dot,
        input,
        query_weight->storage,
        key_weight->storage,
        value_weight->storage,
        output_weight->storage,
        match_projection_bias(query_dot),
        match_projection_bias(key_dot),
        match_projection_bias(value_dot),
        query_weight->scale,
        key_weight->scale,
        value_weight->scale,
        output_weight->scale,
        input_type.getDimSize(0),
        input_type.getDimSize(1),
        query_heads,
        key_weight_storage.getDimSize(1) / head_dim,
        head_dim,
        100000.0f,
        causal,
        {},
    };
    collect_backward_slice(output_dot->getResult(0), match.operations);
    if (const auto theta = match_rope_theta(qk_dot))
        match.rope_theta = *theta;
    return match;
}

std::optional<W8A16FfnMatch> match_w8a16_ffn(mlir::Operation* final_convert)
{
    if (!named(final_convert, "stablehlo.convert") || final_convert->getNumOperands() != 1)
        return std::nullopt;
    auto result_type = llvm::dyn_cast<mlir::RankedTensorType>(final_convert->getResult(0).getType());
    if (!result_type
        || !is_lpu_16bit_float(result_type.getElementType()))
        return std::nullopt;
    auto* down_dot = final_convert->getOperand(0).getDefiningOp();
    if (!named(down_dot, "stablehlo.dot_general")) return std::nullopt;
    auto* swiglu_mul = down_dot->getOperand(0).getDefiningOp();
    if (!named(swiglu_mul, "stablehlo.multiply")) return std::nullopt;

    for (unsigned order = 0; order < 2; ++order) {
        auto* gated_mul = swiglu_mul->getOperand(order).getDefiningOp();
        auto* up_dot = swiglu_mul->getOperand(1 - order).getDefiningOp();
        if (!named(gated_mul, "stablehlo.multiply") || !named(up_dot, "stablehlo.dot_general"))
            continue;
        for (unsigned gate_order = 0; gate_order < 2; ++gate_order) {
            auto* gate_dot = gated_mul->getOperand(gate_order).getDefiningOp();
            auto* logistic = gated_mul->getOperand(1 - gate_order).getDefiningOp();
            if (!named(gate_dot, "stablehlo.dot_general") || !named(logistic, "stablehlo.logistic")
                || logistic->getOperand(0) != gate_dot->getResult(0))
                continue;
            mlir::Value input = converted_source(gate_dot->getOperand(0));
            mlir::Value up_input = converted_source(up_dot->getOperand(0));
            const auto gate_weight =
                match_scaled_weight(gate_dot->getOperand(1));
            const auto up_weight =
                match_scaled_weight(up_dot->getOperand(1));
            const auto down_weight =
                match_scaled_weight(down_dot->getOperand(1));
            if (!input || input != up_input || !gate_weight || !up_weight
                || !down_weight)
                continue;
            return W8A16FfnMatch {final_convert, down_dot, swiglu_mul, gated_mul,
                logistic, gate_dot, up_dot, input, gate_weight->storage,
                up_weight->storage, down_weight->storage, gate_weight->scale,
                up_weight->scale, down_weight->scale};
        }
    }
    return std::nullopt;
}

std::optional<SwigluMatch> match_swiglu(mlir::Operation* final_convert)
{
    if (!named(final_convert, "stablehlo.convert") || final_convert->getNumOperands() != 1)
        return std::nullopt;
    auto result_type = llvm::dyn_cast<mlir::RankedTensorType>(final_convert->getResult(0).getType());
    if (!result_type || !result_type.getElementType().isInteger(8)) return std::nullopt;
    auto* swiglu_mul = final_convert->getOperand(0).getDefiningOp();
    if (!named(swiglu_mul, "stablehlo.multiply") || swiglu_mul->getNumOperands() != 2)
        return std::nullopt;
    for (unsigned order = 0; order < 2; ++order) {
        auto* gated_mul = swiglu_mul->getOperand(order).getDefiningOp();
        auto* up_convert = swiglu_mul->getOperand(1 - order).getDefiningOp();
        if (!named(gated_mul, "stablehlo.multiply")
            || !named(up_convert, "stablehlo.convert")) continue;
        for (unsigned gate_order = 0; gate_order < 2; ++gate_order) {
            auto* gate_convert = gated_mul->getOperand(gate_order).getDefiningOp();
            auto* logistic = gated_mul->getOperand(1 - gate_order).getDefiningOp();
            if (!named(gate_convert, "stablehlo.convert")
                || !named(logistic, "stablehlo.logistic")
                || logistic->getOperand(0) != gate_convert->getResult(0)) continue;
            auto* gate_dot = gate_convert->getOperand(0).getDefiningOp();
            auto* up_dot = up_convert->getOperand(0).getDefiningOp();
            if (!named(gate_dot, "stablehlo.dot_general")
                || !named(up_dot, "stablehlo.dot_general")
                || gate_dot->getOperand(0) != up_dot->getOperand(0)) continue;
            if (!swiglu_mul->getResult(0).hasOneUse()
                || !gated_mul->getResult(0).hasOneUse()
                || !logistic->getResult(0).hasOneUse()
                || !up_convert->getResult(0).hasOneUse()
                || !gate_dot->getResult(0).hasOneUse()
                || !up_dot->getResult(0).hasOneUse()
                || !gate_convert->getResult(0).hasNUses(2))
                continue;
            return SwigluMatch {final_convert, swiglu_mul, gated_mul, logistic,
                gate_convert, up_convert, gate_dot, up_dot};
        }
    }
    return std::nullopt;
}

std::optional<FfnMatch> match_ffn(mlir::Operation* final_convert)
{
    if (!named(final_convert, "stablehlo.convert") || final_convert->getNumOperands() != 1)
        return std::nullopt;
    auto* down_dot = final_convert->getOperand(0).getDefiningOp();
    if (!named(down_dot, "stablehlo.dot_general") || down_dot->getNumOperands() != 2
        || !down_dot->getResult(0).hasOneUse())
        return std::nullopt;
    auto* hidden_convert = down_dot->getOperand(0).getDefiningOp();
    auto swiglu = match_swiglu(hidden_convert);
    if (!swiglu) return std::nullopt;
    return FfnMatch {final_convert, down_dot, *swiglu};
}

// XLA keeps singleton leading dimensions around HF activation functions. A
// reshape is element preserving; unlike a transpose it can be traversed here
// without guessing a head, token, or contracting axis.
mlir::Value without_reshapes(mlir::Value value)
{
    while (auto* reshape = defining(value, "stablehlo.reshape"))
        value = reshape->getOperand(0);
    return value;
}

bool default_dot_contract(mlir::Operation* operation, int64_t lhsAxis,
    int64_t rhsAxis, llvm::ArrayRef<int64_t> batch = {})
{
    auto dot = llvm::dyn_cast_or_null<mlir::stablehlo::DotGeneralOp>(operation);
    if (!dot || dot.getAlgorithmAttr()) return false;
    if (auto precision = dot.getPrecisionConfigAttr())
        for (mlir::Attribute attribute : precision)
            if (llvm::cast<mlir::stablehlo::PrecisionAttr>(attribute).getValue()
                != mlir::stablehlo::Precision::DEFAULT)
                return false;
    auto dimensions = dot.getDotDimensionNumbers();
    return dimensions.getLhsBatchingDimensions() == batch
        && dimensions.getRhsBatchingDimensions() == batch
        && dimensions.getLhsContractingDimensions()
            == llvm::ArrayRef<int64_t>({lhsAxis})
        && dimensions.getRhsContractingDimensions()
            == llvm::ArrayRef<int64_t>({rhsAxis});
}

bool sum_reduce(mlir::Operation* operation, int64_t axis)
{
    auto reduce = llvm::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(operation);
    if (!reduce || reduce->getNumOperands() != 2
        || reduce.getDimensions() != llvm::ArrayRef<int64_t>({axis})
        || scalar_constant(reduce->getOperand(1)) != 0.0f)
        return false;
    auto& block = reduce.getBody().front();
    if (block.getOperations().size() != 2) return false;
    auto* add = &block.front();
    auto* terminator = block.getTerminator();
    return named(add, "stablehlo.add")
        && add->getOperand(0) == block.getArgument(0)
        && add->getOperand(1) == block.getArgument(1)
        && terminator->getOperand(0) == add->getResult(0);
}

bool hf_backend_profile(mlir::func::FuncOp function)
{
    auto profile = function->getAttrOfType<mlir::StringAttr>(
        "ftlpu.backend_numeric_profile");
    return profile && profile.getValue() == "legacy_w8a16_bf16";
}

std::optional<ScaledWeight> match_hf_scaled_weight(mlir::Value value)
{
    while (true) {
        auto match = match_scaled_weight(value);
        if (match) {
            auto type = llvm::dyn_cast<mlir::RankedTensorType>(match->storage.getType());
            if (type && type.getElementType().isInteger(8)) return match;
        }
        auto source = converted_source(value);
        if (!source) return std::nullopt;
        value = source;
    }
}

mlir::Value hf_profile_source(mlir::Value value)
{
    while (true) {
        value = without_reshapes(value);
        auto source = converted_source(value);
        if (!source) return value;
        auto from = llvm::dyn_cast<mlir::RankedTensorType>(source.getType());
        auto to = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
        if (!from || !to
            || (!from.getElementType().isF32() && !from.getElementType().isBF16())
            || (!to.getElementType().isF32() && !to.getElementType().isBF16()))
            return value;
        value = source;
    }
}

// Preserve the HF graph's operand identities and prove the last-axis mean.
// The backend's BF16/LUT rounding is an explicit numerical profile, not an
// equivalence promised by this structural fusion.
std::optional<RmsNormMatch> match_hf_rms_norm(mlir::Operation* root)
{
    if (!root || root->getNumResults() != 1) return std::nullopt;
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(
        root->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2
        || !is_lpu_16bit_float(resultType.getElementType()))
        return std::nullopt;
    mlir::Value output = without_reshapes(root->getResult(0));
    if (auto source = converted_source(output)) output = source;
    auto* outputMultiply = defining(output, "stablehlo.multiply");
    if (!outputMultiply) return std::nullopt;
    for (unsigned order = 0; order < 2; ++order) {
        mlir::Value weightBroadcast = outputMultiply->getOperand(order);
        auto broadcast = weightBroadcast.getDefiningOp<
            mlir::stablehlo::BroadcastInDimOp>();
        auto activationType = llvm::cast<mlir::RankedTensorType>(output.getType());
        const int64_t lastAxis = activationType.getRank() - 1;
        if (activationType.getRank() != 2
            && !(activationType.getRank() == 3 && activationType.getDimSize(0) == 1))
            continue;
        if (!broadcast || broadcast.getBroadcastDimensions()
                != llvm::ArrayRef<int64_t>({lastAxis}))
            continue;
        mlir::Value weight = broadcast.getOperand();
        if (auto source = converted_source(weight)) weight = source;
        auto weightType = llvm::dyn_cast<mlir::RankedTensorType>(weight.getType());
        if (!weightType || weightType.getRank() != 1
            || weightType.getElementType() != resultType.getElementType()
            || weightType.getDimSize(0) != resultType.getDimSize(1))
            continue;
        mlir::Value normalized = hf_profile_source(outputMultiply->getOperand(1 - order));
        auto* normalizedMultiply = defining(normalized, "stablehlo.multiply");
        if (!normalizedMultiply) continue;
        for (unsigned inputOrder = 0; inputOrder < 2; ++inputOrder) {
            mlir::Value inputFloat = normalizedMultiply->getOperand(inputOrder);
            mlir::Value input = converted_source(inputFloat);
            if (!input) continue;
            input = without_reshapes(input);
            auto inputType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
            if (!inputType || inputType.getElementType() != resultType.getElementType()
                || (inputType != resultType && inputType.getShape()
                    != llvm::ArrayRef<int64_t>({1, resultType.getDimSize(0),
                        resultType.getDimSize(1)}))) continue;
            auto invBroadcast = normalizedMultiply->getOperand(1 - inputOrder)
                .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
            const llvm::SmallVector<int64_t> invDimensions = lastAxis == 1
                ? llvm::SmallVector<int64_t> {0}
                : llvm::SmallVector<int64_t> {0, 1};
            if (!invBroadcast || invBroadcast.getBroadcastDimensions()
                    != llvm::ArrayRef<int64_t>(invDimensions))
                continue;
            auto* rsqrt = defining(without_reshapes(invBroadcast.getOperand()),
                "stablehlo.rsqrt");
            if (!rsqrt) continue;
            auto* add = defining(without_reshapes(rsqrt->getOperand(0)),
                "stablehlo.add");
            if (!add) continue;
            for (unsigned meanOrder = 0; meanOrder < 2; ++meanOrder) {
                const auto epsilon = scalar_constant(add->getOperand(1 - meanOrder));
                if (!epsilon || !std::isfinite(*epsilon) || *epsilon <= 0.0f)
                    continue;
                auto* mean = defining(without_reshapes(add->getOperand(meanOrder)),
                    "stablehlo.multiply");
                bool divide = false;
                if (!mean) {
                    mean = defining(without_reshapes(add->getOperand(meanOrder)),
                        "stablehlo.divide");
                    divide = true;
                }
                if (!mean) continue;
                for (unsigned reduceOrder = 0; reduceOrder < (divide ? 1u : 2u);
                     ++reduceOrder) {
                    auto* reduce = defining(without_reshapes(mean->getOperand(reduceOrder)),
                        "stablehlo.reduce");
                    const auto factor = scalar_constant(mean->getOperand(1 - reduceOrder));
                    const float expected = divide
                        ? static_cast<float>(resultType.getDimSize(1))
                        : 1.0f / static_cast<float>(resultType.getDimSize(1));
                    if (!reduce || !factor || *factor != expected
                        || !sum_reduce(reduce, lastAxis)) continue;
                    auto* square = reduce->getOperand(0).getDefiningOp();
                    bool isSquare = named(square, "stablehlo.multiply")
                        && square->getOperand(0) == inputFloat
                        && square->getOperand(1) == inputFloat;
                    isSquare |= named(square, "stablehlo.power")
                        && square->getOperand(0) == inputFloat
                        && scalar_constant(square->getOperand(1)) == 2.0f;
                    if (!isSquare) continue;
                    RmsNormMatch match {root, input, weight, *epsilon, {}};
                    // Do not include producers of input/weight: they are
                    // boundaries and remain available to other computations.
                    llvm::SmallVector<mlir::Value> work {output};
                    while (!work.empty()) {
                        auto value = work.pop_back_val();
                        if (value == input || value == weight) continue;
                        auto* operation = value.getDefiningOp();
                        if (!operation || !match.operations.insert(operation).second)
                            continue;
                        for (auto operand : operation->getOperands()) work.push_back(operand);
                    }
                    match.operations.insert(root);
                    return match;
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<W8A16FfnMatch> match_hf_ffn(mlir::Operation* root)
{
    if (!root || root->getNumResults() != 1) return std::nullopt;
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2
        || !is_lpu_16bit_float(resultType.getElementType())) return std::nullopt;
    auto* down = root;
    if (named(root, "stablehlo.convert"))
        down = without_reshapes(root->getOperand(0)).getDefiningOp();
    if (!default_dot_contract(down, 1, 0)) return std::nullopt;
    auto* multiply = defining(hf_profile_source(down->getOperand(0)),
        "stablehlo.multiply");
    if (!multiply) return std::nullopt;
    for (unsigned order = 0; order < 2; ++order) {
        auto* gated = defining(without_reshapes(multiply->getOperand(order)),
            "stablehlo.multiply");
        auto* up = hf_profile_source(multiply->getOperand(1 - order)).getDefiningOp();
        if (!gated || !default_dot_contract(up, 1, 0)) continue;
        for (unsigned gateOrder = 0; gateOrder < 2; ++gateOrder) {
            auto gateValue = gated->getOperand(gateOrder);
            auto* logistic = defining(gated->getOperand(1 - gateOrder), "stablehlo.logistic");
            auto* gate = hf_profile_source(gateValue).getDefiningOp();
            if (!logistic || logistic->getOperand(0) != gateValue
                || !default_dot_contract(gate, 1, 0)
                || gate->getOperand(0) != up->getOperand(0)) continue;
            auto input = gate->getOperand(0);
            if (auto source = converted_source(input)) input = source;
            auto inputType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
            auto gw = match_hf_scaled_weight(gate->getOperand(1));
            auto uw = match_hf_scaled_weight(up->getOperand(1));
            auto dw = match_hf_scaled_weight(down->getOperand(1));
            if (!inputType || inputType.getRank() != 2
                || !is_lpu_16bit_float(inputType.getElementType())
                || !gw || !uw || !dw) continue;
            const auto int8Weight = [](mlir::Value value) {
                auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
                return type && type.getRank() == 2 && type.getElementType().isInteger(8);
            };
            if (!int8Weight(gw->storage) || !int8Weight(uw->storage)
                || !int8Weight(dw->storage)) continue;
            return W8A16FfnMatch {root, down, multiply, gated, logistic, gate,
                up, input, gw->storage, uw->storage, dw->storage,
                gw->scale, uw->scale, dw->scale};
        }
    }
    return std::nullopt;
}

std::optional<llvm::SmallVector<float>> constant_floats(mlir::Value value)
{
    value = without_reshapes(value);
    if (auto source = converted_source(value)) value = without_reshapes(source);
    auto* constant = defining(value, "stablehlo.constant");
    if (!constant) return std::nullopt;
    auto dense = constant->getAttrOfType<mlir::DenseFPElementsAttr>("value");
    if (!dense) return std::nullopt;
    llvm::SmallVector<float> values;
    for (const llvm::APFloat& element : dense.getValues<llvm::APFloat>())
        values.push_back(element.convertToFloat());
    return values;
}

mlir::Value strip_single_convert(mlir::Value value)
{
    if (auto source = converted_source(value)) return source;
    return value;
}

// Verify the rotary tables originate in the observed checkpoint frequency
// vector and the fixed 0..S-1 positions. theta is source metadata, never a
// value guessed from head shape or an unrelated power in a backward slice.
bool fixed_rope_angles(mlir::Value expanded, int64_t seq, int64_t dim,
    float theta)
{
    expanded = strip_single_convert(expanded);
    auto concat = expanded.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!concat || concat.getDimension() != 2 || concat->getNumOperands() != 2
        || concat->getOperand(0) != concat->getOperand(1)) return false;
    auto transpose = concat->getOperand(0).getDefiningOp<
        mlir::stablehlo::TransposeOp>();
    if (!transpose || transpose.getPermutation()
            != llvm::ArrayRef<int64_t>({0, 2, 1})) return false;
    auto* dot = without_reshapes(transpose.getOperand()).getDefiningOp();
    if (!default_dot_contract(dot, 1, 0)) return false;
    auto inverse = constant_floats(dot->getOperand(0));
    auto positions = constant_floats(dot->getOperand(1));
    if (!inverse || !positions || inverse->size() != static_cast<size_t>(dim / 2)
        || positions->size() != static_cast<size_t>(seq)) return false;
    for (int64_t i = 0; i < seq; ++i)
        if ((*positions)[i] != static_cast<float>(i)) return false;
    for (int64_t i = 0; i < dim / 2; ++i) {
        const float expected = std::pow(theta, -2.0f * static_cast<float>(i) / dim);
        if (!std::isfinite((*inverse)[i])
            || std::abs((*inverse)[i] - expected) > std::abs(expected) * 3.0e-7f)
            return false;
    }
    return true;
}

struct HfRope {
    mlir::Operation* projection;
    mlir::Value tableAngles;
    int64_t heads;
};

mlir::Operation* hf_projection(mlir::Value value)
{
    auto* root = without_reshapes(value).getDefiningOp();
    if (default_dot_contract(root, 1, 0))
        return match_projection_bias(root) ? nullptr : root;
    if (!named(root, "stablehlo.add")) return nullptr;
    for (unsigned order = 0; order < 2; ++order) {
        auto* dot = root->getOperand(order).getDefiningOp();
        if (!default_dot_contract(dot, 1, 0)
            || !dot->getResult(0).hasOneUse()) continue;
        auto broadcast = root->getOperand(1 - order)
            .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
        if (!broadcast || broadcast.getBroadcastDimensions()
                != llvm::ArrayRef<int64_t>({1})) continue;
        auto biasType = llvm::dyn_cast<mlir::RankedTensorType>(broadcast.getOperand().getType());
        auto resultType = llvm::cast<mlir::RankedTensorType>(dot->getResult(0).getType());
        if (biasType && biasType.getRank() == 1
            && biasType.getDimSize(0) == resultType.getDimSize(1)
            && biasType.getElementType() == resultType.getElementType()) return dot;
    }
    return nullptr;
}

bool exact_half_slice(mlir::Operation* operation, mlir::Value input,
    bool high)
{
    auto slice = llvm::dyn_cast_or_null<mlir::stablehlo::SliceOp>(operation);
    auto type = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!slice || !type || type.getRank() != 4 || slice.getOperand() != input)
        return false;
    llvm::SmallVector<int64_t> start(4, 0), limit(type.getShape());
    start[3] = high ? type.getDimSize(3) / 2 : 0;
    limit[3] = high ? type.getDimSize(3) : type.getDimSize(3) / 2;
    return slice.getStartIndices() == llvm::ArrayRef<int64_t>(start)
        && slice.getLimitIndices() == llvm::ArrayRef<int64_t>(limit)
        && slice.getStrides() == llvm::ArrayRef<int64_t>({1, 1, 1, 1});
}

std::optional<HfRope> match_hf_rope(mlir::Value value, int64_t seq,
    int64_t dim, float theta)
{
    value = without_reshapes(value);
    auto* add = defining(value, "stablehlo.add");
    if (!add) return std::nullopt;
    for (unsigned order = 0; order < 2; ++order) {
        auto* plainMul = defining(hf_profile_source(add->getOperand(order)), "stablehlo.multiply");
        auto* rotateMul = defining(hf_profile_source(add->getOperand(1 - order)), "stablehlo.multiply");
        if (!plainMul || !rotateMul) continue;
        for (unsigned plainOrder = 0; plainOrder < 2; ++plainOrder) {
            auto input = hf_profile_source(plainMul->getOperand(plainOrder));
            auto type = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
            if (!type || type.getShape().size() != 4 || type.getDimSize(0) != 1
                || type.getDimSize(2) != seq || type.getDimSize(3) != dim) continue;
            auto cosineBroadcast = plainMul->getOperand(1 - plainOrder)
                .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
            if (!cosineBroadcast || cosineBroadcast.getBroadcastDimensions()
                    != llvm::ArrayRef<int64_t>({0, 2, 3})) continue;
            auto* cosine = defining(hf_profile_source(cosineBroadcast.getOperand()),
                "stablehlo.cosine");
            if (!cosine || !fixed_rope_angles(cosine->getOperand(0), seq, dim, theta)) continue;
            for (unsigned rotateOrder = 0; rotateOrder < 2; ++rotateOrder) {
                auto rotate = hf_profile_source(rotateMul->getOperand(rotateOrder))
                    .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
                auto sineBroadcast = rotateMul->getOperand(1 - rotateOrder)
                    .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
                if (!rotate || rotate.getDimension() != 3
                    || rotate->getNumOperands() != 2 || !sineBroadcast
                    || sineBroadcast.getBroadcastDimensions()
                        != llvm::ArrayRef<int64_t>({0, 2, 3})) continue;
                auto* negative = defining(rotate->getOperand(0), "stablehlo.negate");
                auto* sine = defining(hf_profile_source(sineBroadcast.getOperand()),
                    "stablehlo.sine");
                if (!negative || !sine || sine->getOperand(0) != cosine->getOperand(0)
                    || !exact_half_slice(negative->getOperand(0).getDefiningOp(), input, true)
                    || !exact_half_slice(rotate->getOperand(1).getDefiningOp(), input, false)) continue;
                auto layout = input.getDefiningOp<mlir::stablehlo::TransposeOp>();
                if (!layout || layout.getPermutation()
                        != llvm::ArrayRef<int64_t>({0, 2, 1, 3})) continue;
                auto* projection = hf_projection(layout.getOperand());
                if (!projection) continue;
                return HfRope {projection, cosine->getOperand(0), type.getDimSize(1)};
            }
        }
    }
    return std::nullopt;
}

// Return the pre-repeat [B, KVheads, S, D] value only for the actual HF
// repeat_interleave layout, not any broadcast somewhere in the ancestry.
mlir::Value hf_gqa_source(mlir::Value value, int64_t heads, int64_t seq,
    int64_t dim)
{
    auto identity = without_reshapes(value);
    auto identityType = llvm::dyn_cast<mlir::RankedTensorType>(identity.getType());
    if (identityType && identityType.getShape()
            == llvm::ArrayRef<int64_t>({1, heads, seq, dim}))
        return identity;
    auto* reshape = defining(value, "stablehlo.reshape");
    if (!reshape) return {};
    auto broadcast = reshape->getOperand(0).getDefiningOp<
        mlir::stablehlo::BroadcastInDimOp>();
    if (!broadcast || broadcast.getBroadcastDimensions()
            != llvm::ArrayRef<int64_t>({0, 1, 3, 4})) return {};
    auto input = broadcast.getOperand();
    auto type = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto repeated = llvm::dyn_cast<mlir::RankedTensorType>(broadcast.getResult().getType());
    if (!type || !repeated || type.getRank() != 4 || repeated.getRank() != 5
        || type.getDimSize(0) != 1 || type.getDimSize(2) != seq
        || type.getDimSize(3) != dim || type.getDimSize(1) <= 0
        || heads % type.getDimSize(1) != 0
        || repeated.getDimSize(2) != heads / type.getDimSize(1)) return {};
    return input;
}

bool exact_reduction(mlir::Operation* op, llvm::StringRef kind, int64_t axis,
    float initial)
{
    auto reduce = llvm::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op);
    if (!reduce || reduce->getNumOperands() != 2
        || reduce.getDimensions() != llvm::ArrayRef<int64_t>({axis})
        || scalar_constant(reduce->getOperand(1)) != initial) return false;
    auto& block = reduce.getBody().front();
    if (block.getOperations().size() != 2) return false;
    auto* computation = &block.front();
    return named(computation, kind)
        && computation->getOperand(0) == block.getArgument(0)
        && computation->getOperand(1) == block.getArgument(1)
        && block.getTerminator()->getOperand(0) == computation->getResult(0);
}

struct HfSoftmax {
    mlir::Operation* qk;
    float scale;
};

std::optional<HfSoftmax> match_hf_softmax(mlir::Value value, int64_t seq,
    int64_t dim)
{
    value = strip_single_convert(without_reshapes(value));
    auto* divide = defining(value, "stablehlo.divide");
    if (!divide) return std::nullopt;
    auto sumBroadcast = divide->getOperand(1).getDefiningOp<
        mlir::stablehlo::BroadcastInDimOp>();
    if (!sumBroadcast || sumBroadcast.getBroadcastDimensions()
            != llvm::ArrayRef<int64_t>({0, 1, 2})) return std::nullopt;
    auto* exponential = defining(divide->getOperand(0), "stablehlo.exponential");
    auto* sum = defining(strip_broadcast(divide->getOperand(1)), "stablehlo.reduce");
    if (!exponential || !sum || !exact_reduction(sum, "stablehlo.add", 3, 0.0f)
        || sum->getOperand(0) != exponential->getResult(0)) return std::nullopt;
    auto* subtract = defining(exponential->getOperand(0), "stablehlo.subtract");
    if (!subtract) return std::nullopt;
    auto maxBroadcast = subtract->getOperand(1).getDefiningOp<
        mlir::stablehlo::BroadcastInDimOp>();
    if (!maxBroadcast || maxBroadcast.getBroadcastDimensions()
            != llvm::ArrayRef<int64_t>({0, 1, 2})) return std::nullopt;
    auto* maximum = defining(strip_broadcast(subtract->getOperand(1)), "stablehlo.reduce");
    if (!maximum || !exact_reduction(maximum, "stablehlo.maximum", 3,
            -std::numeric_limits<float>::infinity())
        || maximum->getOperand(0) != subtract->getOperand(0)) return std::nullopt;
    auto* masked = defining(strip_single_convert(subtract->getOperand(0)), "stablehlo.add");
    if (!masked) return std::nullopt;
    for (unsigned order = 0; order < 2; ++order) {
        auto maskValue = masked->getOperand(order);
        if (auto broadcast = maskValue.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
            auto sourceType = llvm::cast<mlir::RankedTensorType>(broadcast.getOperand().getType());
            auto dimensions = broadcast.getBroadcastDimensions();
            if (sourceType.getRank() < 2
                || sourceType.getDimSize(sourceType.getRank() - 2) != seq
                || sourceType.getDimSize(sourceType.getRank() - 1) != seq
                || dimensions[dimensions.size() - 2] != 2
                || dimensions.back() != 3) continue;
            maskValue = broadcast.getOperand();
        } else {
            auto maskType = llvm::cast<mlir::RankedTensorType>(maskValue.getType());
            if (maskType.getShape() != llvm::ArrayRef<int64_t>({1, 1, seq, seq}))
                continue;
        }
        auto mask = constant_floats(maskValue);
        if (!mask || mask->size() != static_cast<size_t>(seq * seq)) continue;
        bool pureCausal = true;
        for (int64_t row = 0; row < seq; ++row)
            for (int64_t col = 0; col < seq; ++col) {
                const float element = (*mask)[row * seq + col];
                pureCausal &= col <= row ? element == 0.0f
                    : element == -std::numeric_limits<float>::infinity()
                        || element == std::numeric_limits<float>::lowest()
                        || element == -3.3895313892515355e38f;
            }
        if (!pureCausal) continue;
        auto* scaled = defining(hf_profile_source(masked->getOperand(1 - order)), "stablehlo.multiply");
        if (!scaled) continue;
        for (unsigned scaleOrder = 0; scaleOrder < 2; ++scaleOrder) {
            auto scale = scalar_constant(scaled->getOperand(scaleOrder));
            auto* qk = hf_profile_source(scaled->getOperand(1 - scaleOrder)).getDefiningOp();
            if (!scale || *scale != 1.0f / std::sqrt(static_cast<float>(dim))
                || !default_dot_contract(qk, 2, 1, {0})) continue;
            return HfSoftmax {qk, *scale};
        }
    }
    return std::nullopt;
}

std::optional<AttentionMatch> match_hf_attention(mlir::Operation* root,
    mlir::func::FuncOp function)
{
    if (!hf_backend_profile(function) || !default_dot_contract(root, 1, 0))
        return std::nullopt;
    auto theta = function->getAttrOfType<mlir::FloatAttr>("ftlpu.rope_theta");
    if (!theta || !std::isfinite(theta.getValueAsDouble())
        || theta.getValueAsDouble() <= 1.0) return std::nullopt;
    auto contextTranspose = without_reshapes(root->getOperand(0))
        .getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!contextTranspose || contextTranspose.getPermutation()
            != llvm::ArrayRef<int64_t>({0, 2, 1, 3})) return std::nullopt;
    auto* pv = without_reshapes(contextTranspose.getOperand()).getDefiningOp();
    if (!default_dot_contract(pv, 2, 1, {0})) return std::nullopt;
    auto pvType = llvm::cast<mlir::RankedTensorType>(pv->getResult(0).getType());
    if (pvType.getRank() != 3 || !pvType.hasStaticShape()) return std::nullopt;
    const int64_t heads = pvType.getDimSize(0), seq = pvType.getDimSize(1), dim = pvType.getDimSize(2);
    const auto pvLeft = llvm::cast<mlir::RankedTensorType>(pv->getOperand(0).getType());
    const auto pvRight = llvm::cast<mlir::RankedTensorType>(pv->getOperand(1).getType());
    if (heads <= 0 || seq <= 0 || dim <= 0 || dim % 2 != 0
        || pvLeft.getShape() != llvm::ArrayRef<int64_t>({heads, seq, seq})
        || pvRight.getShape() != llvm::ArrayRef<int64_t>({heads, seq, dim})) return std::nullopt;
    auto probability = match_hf_softmax(pv->getOperand(0), seq, dim);
    if (!probability) return std::nullopt;
    const auto qkLeft = llvm::cast<mlir::RankedTensorType>(probability->qk->getOperand(0).getType());
    const auto qkRight = llvm::cast<mlir::RankedTensorType>(probability->qk->getOperand(1).getType());
    const auto qkResult = llvm::cast<mlir::RankedTensorType>(probability->qk->getResult(0).getType());
    if (qkLeft.getShape() != llvm::ArrayRef<int64_t>({heads, seq, dim})
        || qkRight.getShape() != llvm::ArrayRef<int64_t>({heads, dim, seq})
        || qkResult.getShape() != llvm::ArrayRef<int64_t>({heads, seq, seq}))
        return std::nullopt;
    auto query = match_hf_rope(probability->qk->getOperand(0), seq, dim,
        theta.getValueAsDouble());
    auto keyTranspose = without_reshapes(probability->qk->getOperand(1))
        .getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!query || !keyTranspose || keyTranspose.getPermutation()
            != llvm::ArrayRef<int64_t>({0, 1, 3, 2})) return std::nullopt;
    auto keySource = hf_gqa_source(keyTranspose.getOperand(), heads, seq, dim);
    auto key = keySource ? match_hf_rope(keySource, seq, dim,
        theta.getValueAsDouble()) : std::nullopt;
    auto valueRepeat = pv->getOperand(1).getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!valueRepeat) return std::nullopt;
    auto valueSource = hf_gqa_source(valueRepeat.getResult(), heads, seq, dim);
    auto valueTranspose = valueSource
        ? valueSource.getDefiningOp<mlir::stablehlo::TransposeOp>()
        : mlir::stablehlo::TransposeOp {};
    if (!key || query->tableAngles != key->tableAngles || !valueTranspose
        || valueTranspose.getPermutation() != llvm::ArrayRef<int64_t>({0, 2, 1, 3})) return std::nullopt;
    auto valueHeadType = llvm::cast<mlir::RankedTensorType>(valueTranspose.getResult().getType());
    if (valueHeadType.getShape()
        != llvm::ArrayRef<int64_t>({1, key->heads, seq, dim})) return std::nullopt;
    auto* value = hf_projection(valueTranspose.getOperand());
    if (!value) return std::nullopt;
    auto input = query->projection->getOperand(0);
    if (key->projection->getOperand(0) != input || value->getOperand(0) != input) return std::nullopt;
    auto qw = match_hf_scaled_weight(query->projection->getOperand(1));
    auto kw = match_hf_scaled_weight(key->projection->getOperand(1));
    auto vw = match_hf_scaled_weight(value->getOperand(1));
    auto ow = match_hf_scaled_weight(root->getOperand(1));
    if (!qw || !kw || !vw || !ow) return std::nullopt;
    auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
    if (!is_lpu_16bit_float(inputType.getElementType()) || inputType.getRank() != 2
        || inputType.getDimSize(0) != seq || query->heads != heads) return std::nullopt;
    for (auto weight : {qw->storage, kw->storage, vw->storage, ow->storage}) {
        auto type = llvm::dyn_cast<mlir::RankedTensorType>(weight.getType());
        if (!type || type.getRank() != 2 || !type.getElementType().isInteger(8)) return std::nullopt;
    }
    AttentionMatch match {root, input, qw->storage, kw->storage, vw->storage,
        ow->storage, match_projection_bias(query->projection),
        match_projection_bias(key->projection), match_projection_bias(value),
        qw->scale, kw->scale, vw->scale, ow->scale, seq, inputType.getDimSize(1),
        heads, key->heads, dim, static_cast<float>(theta.getValueAsDouble()), true, {}};
    collect_backward_slice(root->getResult(0), match.operations);
    return match;
}

class LowerStableHloToKernelPass final
    : public mlir::PassWrapper<LowerStableHloToKernelPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStableHloToKernelPass)

    llvm::StringRef getArgument() const final { return "ftlpu-stablehlo-to-kernel"; }
    llvm::StringRef getDescription() const final
    {
        return "Lowers matrix and rank-3 batched StableHLO dots to Kernel IR";
    }

    void runOnOperation() final
    {
        llvm::SmallVector<mlir::Operation*> attention_roots;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.dot_general"))
                attention_roots.push_back(operation);
        });
        mlir::IRRewriter rewriter(&getContext());
        llvm::DenseMap<mlir::Value, mlir::Value> activationViews;
        std::function<mlir::Value(mlir::Value)> activation2d;
        activation2d = [&](mlir::Value value) -> mlir::Value {
            if (auto found = activationViews.find(value);
                found != activationViews.end()) return found->second;
            auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
            if (!type || !type.hasStaticShape()) return {};
            if (type.getRank() == 2) return value;
            if (type.getRank() != 3 || type.getDimSize(0) != 1) return {};
            if (auto* reshape = defining(value, "stablehlo.reshape")) {
                auto source = activation2d(reshape->getOperand(0));
                if (source && source.getType() == mlir::RankedTensorType::get(
                        type.getShape().drop_front(), type.getElementType()))
                    return activationViews[value] = source;
                return {};
            }
            auto* operation = value.getDefiningOp();
            if ((!named(operation, "stablehlo.add")
                    && !named(operation, "stablehlo.multiply"))
                || operation->getOperand(0).getType() != type
                || operation->getOperand(1).getType() != type) return {};
            auto left = activation2d(operation->getOperand(0));
            auto right = activation2d(operation->getOperand(1));
            if (!left || !right || left.getType() != right.getType()) return {};
            rewriter.setInsertionPoint(operation);
            mlir::OperationState state(operation->getLoc(),
                kernel::ElementwiseOp::getOperationName());
            state.addOperands({left, right});
            state.addTypes(left.getType());
            state.addAttribute("kind", rewriter.getStringAttr(
                named(operation, "stablehlo.add") ? "add" : "multiply"));
            return activationViews[value] = rewriter.create(state)->getResult(0);
        };
        llvm::SmallPtrSet<mlir::Operation*, 32> erased_attention_operations;
        for (mlir::Operation* root : llvm::reverse(attention_roots)) {
            if (erased_attention_operations.contains(root)) continue;
            if (!root->getBlock()) continue;
            const auto match = hf_backend_profile(getOperation())
                ? match_hf_attention(root, getOperation())
                : match_standard_attention(root);
            if (!match) continue;
            erased_attention_operations.insert(
                match->operations.begin(), match->operations.end());
            rewriter.setInsertionPoint(root);
            const auto result_type =
                llvm::cast<mlir::RankedTensorType>(root->getResult(0).getType());
            const mlir::Type element_type = result_type.getElementType();
            const int64_t query_width = match->query_heads * match->head_dim;
            const int64_t kv_width = match->kv_heads * match->head_dim;
            const auto tensor_type = [&](llvm::ArrayRef<int64_t> shape) {
                return mlir::RankedTensorType::get(shape, element_type);
            };
            const auto create_reshape = [&](mlir::Value input, mlir::Type type) {
                mlir::OperationState state(root->getLoc(),
                    kernel::ReshapeOp::getOperationName());
                state.addOperands(input);
                state.addTypes(type);
                return llvm::cast<kernel::ReshapeOp>(rewriter.create(state));
            };
            const auto create_transpose = [&](mlir::Value input, mlir::Type type) {
                mlir::OperationState state(root->getLoc(),
                    kernel::TransposeOp::getOperationName());
                state.addOperands(input);
                state.addTypes(type);
                state.addAttribute("permutation",
                    rewriter.getDenseI64ArrayAttr({1, 0, 2}));
                return llvm::cast<kernel::TransposeOp>(rewriter.create(state));
            };
            const auto create_rope = [&](mlir::Value input, int64_t heads) {
                mlir::OperationState state(root->getLoc(),
                    kernel::RopeOp::getOperationName());
                state.addOperands(input);
                state.addTypes(input.getType());
                state.addAttribute("heads", rewriter.getI64IntegerAttr(heads));
                state.addAttribute("head_dim",
                    rewriter.getI64IntegerAttr(match->head_dim));
                state.addAttribute("theta",
                    rewriter.getF32FloatAttr(match->rope_theta));
                return llvm::cast<kernel::RopeOp>(rewriter.create(state));
            };
            const auto create_bias_add = [&](mlir::Value input,
                                              mlir::Value bias) -> mlir::Value {
                if (!bias) return input;
                mlir::OperationState state(root->getLoc(),
                    kernel::BiasAddOp::getOperationName());
                state.addOperands({input, bias});
                state.addTypes(input.getType());
                return rewriter.create(state)->getResult(0);
            };
            const auto create_gqa = [&](mlir::Value input) {
                mlir::OperationState state(root->getLoc(),
                    kernel::GqaBroadcastOp::getOperationName());
                state.addOperands(input);
                state.addTypes(tensor_type(
                    {match->seq_len, match->query_heads, match->head_dim}));
                state.addAttribute("query_heads",
                    rewriter.getI64IntegerAttr(match->query_heads));
                state.addAttribute("kv_heads",
                    rewriter.getI64IntegerAttr(match->kv_heads));
                return llvm::cast<kernel::GqaBroadcastOp>(rewriter.create(state));
            };
            const auto create_batch_matmul = [&](mlir::Value lhs, mlir::Value rhs,
                                                  mlir::Type type,
                                                  bool transpose_rhs,
                                                  llvm::StringRef role) {
                mlir::OperationState state(root->getLoc(),
                    kernel::BatchMatmulOp::getOperationName());
                state.addOperands({lhs, rhs});
                state.addTypes(type);
                state.addAttribute("transpose_rhs",
                    rewriter.getBoolAttr(transpose_rhs));
                state.addAttribute("role", rewriter.getStringAttr(role));
                return llvm::cast<kernel::BatchMatmulOp>(rewriter.create(state));
            };

            auto query_2d = rewriter.create<kernel::MatmulOp>(root->getLoc(),
                match->input, match->query_weight,
                tensor_type({match->seq_len, query_width}),
                match->seq_len, query_width, match->hidden);
            query_2d.setRhsScaleAttr(
                rewriter.getF32FloatAttr(match->query_scale));
            auto key_2d = rewriter.create<kernel::MatmulOp>(root->getLoc(),
                match->input, match->key_weight,
                tensor_type({match->seq_len, kv_width}),
                match->seq_len, kv_width, match->hidden);
            key_2d.setRhsScaleAttr(
                rewriter.getF32FloatAttr(match->key_scale));
            auto value_2d = rewriter.create<kernel::MatmulOp>(root->getLoc(),
                match->input, match->value_weight,
                tensor_type({match->seq_len, kv_width}),
                match->seq_len, kv_width, match->hidden);
            value_2d.setRhsScaleAttr(
                rewriter.getF32FloatAttr(match->value_scale));

            const mlir::Value biased_query = create_bias_add(
                query_2d.getResult(), match->query_bias);
            const mlir::Value biased_key = create_bias_add(
                key_2d.getResult(), match->key_bias);
            const mlir::Value biased_value = create_bias_add(
                value_2d.getResult(), match->value_bias);
            auto query_heads = create_reshape(biased_query,
                tensor_type({match->seq_len, match->query_heads, match->head_dim}));
            auto key_heads = create_reshape(biased_key,
                tensor_type({match->seq_len, match->kv_heads, match->head_dim}));
            auto value_heads = create_reshape(biased_value,
                tensor_type({match->seq_len, match->kv_heads, match->head_dim}));
            auto query_rope = create_rope(query_heads.getResult(), match->query_heads);
            auto key_rope = create_rope(key_heads.getResult(), match->kv_heads);
            auto key_gqa = create_gqa(key_rope.getResult());
            auto value_gqa = create_gqa(value_heads.getResult());
            const auto bhsd_type =
                tensor_type({match->query_heads, match->seq_len, match->head_dim});
            auto query_bhsd = create_transpose(query_rope.getResult(), bhsd_type);
            auto key_bhsd = create_transpose(key_gqa.getResult(), bhsd_type);
            auto value_bhsd = create_transpose(value_gqa.getResult(), bhsd_type);
            const auto score_type =
                tensor_type({match->query_heads, match->seq_len, match->seq_len});
            auto scores = create_batch_matmul(query_bhsd.getResult(),
                key_bhsd.getResult(), score_type, true, "qk");

            mlir::OperationState softmax_state(root->getLoc(),
                kernel::SoftmaxOp::getOperationName());
            softmax_state.addOperands(scores.getResult());
            softmax_state.addTypes(score_type);
            softmax_state.addAttribute("axis", rewriter.getI64IntegerAttr(-1));
            softmax_state.addAttribute("scale", rewriter.getF32FloatAttr(
                1.0f / std::sqrt(static_cast<float>(match->head_dim))));
            softmax_state.addAttribute("causal", rewriter.getBoolAttr(match->causal));
            auto probability = llvm::cast<kernel::SoftmaxOp>(
                rewriter.create(softmax_state));
            auto context_bhsd = create_batch_matmul(probability.getResult(),
                value_bhsd.getResult(), bhsd_type, false, "pv");
            auto context_shd = create_transpose(context_bhsd.getResult(),
                tensor_type({match->seq_len, match->query_heads, match->head_dim}));
            auto context = create_reshape(context_shd.getResult(),
                tensor_type({match->seq_len, query_width}));
            auto output = rewriter.create<kernel::MatmulOp>(root->getLoc(),
                context.getResult(), match->output_weight, result_type,
                match->seq_len, match->hidden, query_width);
            output.setRhsScaleAttr(
                rewriter.getF32FloatAttr(match->output_scale));
            rewriter.replaceOp(root, output.getResult());

            llvm::SmallVector<mlir::Operation*> block_operations;
            for (mlir::Operation& operation : getOperation().getBody().front())
                block_operations.push_back(&operation);
            for (mlir::Operation* operation : llvm::reverse(block_operations)) {
                if (match->operations.contains(operation) && operation->use_empty())
                    rewriter.eraseOp(operation);
            }
        }

        if (hf_backend_profile(getOperation())) {
            llvm::SmallVector<mlir::Operation*> normRoots;
            for (mlir::Operation& operation : getOperation().getBody().front())
                if (named(&operation, "stablehlo.reshape")
                    || named(&operation, "stablehlo.convert")
                    || named(&operation, "stablehlo.multiply"))
                    normRoots.push_back(&operation);
            llvm::SmallPtrSet<mlir::Operation*, 32> erasedNorms;
            for (auto* root : llvm::reverse(normRoots)) {
                if (erasedNorms.contains(root)) continue;
                auto match = match_hf_rms_norm(root);
                if (!match) continue;
                auto input2d = activation2d(match->input);
                if (!input2d) continue;
                rewriter.setInsertionPoint(root);
                mlir::OperationState state(root->getLoc(),
                    kernel::RmsNormOp::getOperationName());
                state.addOperands({input2d, match->weight});
                state.addTypes(root->getResult(0).getType());
                state.addAttribute("axis", rewriter.getI64IntegerAttr(-1));
                state.addAttribute("epsilon", rewriter.getF32FloatAttr(match->epsilon));
                auto value = rewriter.create(state)->getResult(0);
                erasedNorms.insert(root);
                rewriter.replaceOp(root, value);
                bool progress = true;
                while (progress) {
                    progress = false;
                    llvm::SmallVector<mlir::Operation*> operations;
                    for (mlir::Operation& operation : getOperation().getBody().front())
                        operations.push_back(&operation);
                    for (auto* operation : llvm::reverse(operations))
                        if (match->operations.contains(operation) && operation->use_empty()) {
                            erasedNorms.insert(operation);
                            rewriter.eraseOp(operation);
                            progress = true;
                        }
                }
            }
        }

        llvm::SmallVector<mlir::Operation*> converts;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.convert")) converts.push_back(operation);
        });
        for (mlir::Operation* convert : llvm::reverse(converts)) {
            if (!convert->getBlock()) continue;
            auto match = hf_backend_profile(getOperation())
                ? std::nullopt : match_rms_norm(convert);
            if (!match) continue;
            rewriter.setInsertionPoint(convert);
            mlir::OperationState state(
                convert->getLoc(), kernel::RmsNormOp::getOperationName());
            state.addOperands({match->input, match->weight});
            state.addTypes(convert->getResult(0).getType());
            state.addAttribute("axis", rewriter.getI64IntegerAttr(-1));
            state.addAttribute(
                "epsilon", rewriter.getF32FloatAttr(match->epsilon));
            auto lowered =
                llvm::cast<kernel::RmsNormOp>(rewriter.create(state));
            rewriter.replaceOp(convert, lowered.getResult());
            llvm::SmallVector<mlir::Operation*> blockOperations;
            for (mlir::Operation& operation :
                getOperation().getBody().front())
                blockOperations.push_back(&operation);
            bool erased = true;
            while (erased) {
                erased = false;
                for (mlir::Operation*& operation :
                    llvm::reverse(blockOperations)) {
                    if (operation && match->operations.contains(operation)
                        && operation->getBlock()
                        && operation->use_empty()) {
                        rewriter.eraseOp(operation);
                        operation = nullptr;
                        erased = true;
                    }
                }
            }
        }
        converts.clear();
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.convert"))
                converts.push_back(operation);
            else if (hf_backend_profile(getOperation())
                && named(operation, "stablehlo.dot_general"))
                converts.push_back(operation);
        });
        const auto create_ffn_graph = [&](mlir::Location location,
                                          mlir::Value input,
                                          mlir::Value gate_weight,
                                          mlir::Value up_weight,
                                          mlir::Value down_weight,
                                          mlir::Type hidden_type,
                                          mlir::Type result_type,
                                          int64_t m, int64_t k,
                                          int64_t hidden, int64_t n,
                                          float gate_scale,
                                          float up_scale,
                                          float down_scale) {
            auto gate = rewriter.create<kernel::MatmulOp>(location,
                input, gate_weight, hidden_type, m, hidden, k);
            gate.setRhsScaleAttr(
                rewriter.getF32FloatAttr(gate_scale));
            auto up = rewriter.create<kernel::MatmulOp>(location,
                input, up_weight, hidden_type, m, hidden, k);
            up.setRhsScaleAttr(
                rewriter.getF32FloatAttr(up_scale));
            mlir::OperationState swish_state(location,
                kernel::SwishOp::getOperationName());
            swish_state.addOperands(gate.getResult());
            swish_state.addTypes(hidden_type);
            auto swish = llvm::cast<kernel::SwishOp>(
                rewriter.create(swish_state));
            mlir::OperationState multiply_state(location,
                kernel::ElementwiseOp::getOperationName());
            multiply_state.addOperands({swish.getResult(), up.getResult()});
            multiply_state.addTypes(hidden_type);
            multiply_state.addAttribute("kind",
                rewriter.getStringAttr("multiply"));
            auto gated = llvm::cast<kernel::ElementwiseOp>(
                rewriter.create(multiply_state));
            auto down = rewriter.create<kernel::MatmulOp>(location,
                gated.getResult(), down_weight, result_type, m, n, hidden);
            down.setRhsScaleAttr(
                rewriter.getF32FloatAttr(down_scale));
            return down;
        };
        llvm::SmallPtrSet<mlir::Operation*, 32> erasedFfn;
        for (mlir::Operation* convert : llvm::reverse(converts)) {
            if (erasedFfn.contains(convert)) continue;
            if (!convert->getBlock()) continue;
            auto ffnMatch = hf_backend_profile(getOperation())
                ? match_hf_ffn(convert) : match_w8a16_ffn(convert);
            if (auto match = ffnMatch) {
                auto input_type = llvm::cast<mlir::RankedTensorType>(match->input.getType());
                auto gate_type = llvm::cast<mlir::RankedTensorType>(match->gate_weight.getType());
                auto down_type = llvm::cast<mlir::RankedTensorType>(match->down_weight.getType());
                auto result_type = llvm::cast<mlir::RankedTensorType>(convert->getResult(0).getType());
                rewriter.setInsertionPoint(convert);
                auto lowered = create_ffn_graph(convert->getLoc(),
                    match->input, match->gate_weight, match->up_weight,
                    match->down_weight, match->gate_dot->getResult(0).getType(),
                    result_type, input_type.getDimSize(0), input_type.getDimSize(1),
                    gate_type.getDimSize(1), down_type.getDimSize(1),
                    match->gate_scale, match->up_scale, match->down_scale);
                erasedFfn.insert(convert);
                rewriter.replaceOp(convert, lowered.getResult());
                for (mlir::Operation* op : {match->down_dot, match->swiglu_mul,
                         match->gated_mul, match->logistic, match->gate_dot, match->up_dot})
                    if (op != convert && op->use_empty()) {
                        erasedFfn.insert(op);
                        rewriter.eraseOp(op);
                    }
                llvm::SmallVector<mlir::Operation*> dead_converts;
                getOperation().walk([&](mlir::Operation* op) {
                    if (named(op, "stablehlo.convert") && op->use_empty()) dead_converts.push_back(op);
                });
                for (mlir::Operation* op : llvm::reverse(dead_converts)) {
                    erasedFfn.insert(op);
                    rewriter.eraseOp(op);
                }
                continue;
            }
            if (auto match = match_ffn(convert)) {
                auto input_type = llvm::cast<mlir::RankedTensorType>(
                    match->swiglu.gate_dot->getOperand(0).getType());
                auto gate_type = llvm::cast<mlir::RankedTensorType>(
                    match->swiglu.gate_dot->getOperand(1).getType());
                auto down_type = llvm::cast<mlir::RankedTensorType>(
                    match->down_dot->getOperand(1).getType());
                auto result_type = llvm::cast<mlir::RankedTensorType>(
                    match->final_convert->getResult(0).getType());
                rewriter.setInsertionPoint(match->final_convert);
                auto lowered = create_ffn_graph(match->final_convert->getLoc(),
                    match->swiglu.gate_dot->getOperand(0),
                    match->swiglu.gate_dot->getOperand(1),
                    match->swiglu.up_dot->getOperand(1),
                    match->down_dot->getOperand(1),
                    match->swiglu.gate_dot->getResult(0).getType(), result_type,
                    input_type.getDimSize(0), input_type.getDimSize(1),
                    gate_type.getDimSize(1), down_type.getDimSize(1),
                    1.0f, 1.0f, 1.0f);
                rewriter.replaceOp(match->final_convert, lowered.getResult());
                rewriter.eraseOp(match->down_dot);
                for (mlir::Operation* op : {match->swiglu.final_convert,
                         match->swiglu.swiglu_mul, match->swiglu.gated_mul,
                         match->swiglu.logistic, match->swiglu.gate_convert,
                         match->swiglu.up_convert, match->swiglu.gate_dot,
                         match->swiglu.up_dot})
                    rewriter.eraseOp(op);
                continue;
            }
            auto match = match_swiglu(convert);
            if (!match) continue;
            auto input_type = llvm::dyn_cast<mlir::RankedTensorType>(
                match->gate_dot->getOperand(0).getType());
            auto weight_type = llvm::dyn_cast<mlir::RankedTensorType>(
                match->gate_dot->getOperand(1).getType());
            auto result_type = llvm::dyn_cast<mlir::RankedTensorType>(
                match->final_convert->getResult(0).getType());
            if (!input_type || !weight_type || !result_type) continue;
            rewriter.setInsertionPoint(match->final_convert);
            auto lowered = rewriter.create<kernel::SwigluOp>(match->final_convert->getLoc(),
                match->gate_dot->getOperand(0), match->gate_dot->getOperand(1),
                match->up_dot->getOperand(1), result_type,
                input_type.getDimSize(0), weight_type.getDimSize(1), input_type.getDimSize(1),
                1.0f, 1.0f, 1.0f, 0);
            rewriter.replaceOp(match->final_convert, lowered.getResult());
            for (mlir::Operation* op : {match->swiglu_mul, match->gated_mul,
                     match->logistic, match->gate_convert, match->up_convert,
                     match->gate_dot, match->up_dot})
                rewriter.eraseOp(op);
        }

        // Remove only a singleton leading dimension around pointwise residuals.
        // The add/multiply arithmetic and its evaluation order are unchanged;
        // this exposes the same rank-2 activation boundary as the old planner.
        if (hf_backend_profile(getOperation())) {
            llvm::SmallVector<mlir::Operation*> views;
            for (mlir::Operation& operation : getOperation().getBody().front())
                if (named(&operation, "stablehlo.reshape")) views.push_back(&operation);
            for (auto* view : views) {
                auto result = llvm::dyn_cast<mlir::RankedTensorType>(view->getResult(0).getType());
                if (!result || result.getRank() != 2) continue;
                auto value = activation2d(view->getOperand(0));
                if (value && value.getType() == result) rewriter.replaceOp(view, value);
            }
        }

        // Quantized StableHLO represents runtime weights as convert followed
        // by a scalar multiply. Composite attention/FFN lowering replaces the
        // dots with scaled Kernel IR, so remove the now-dead scale subgraphs
        // before generic elementwise lowering sees them.
        bool erased_dead_stablehlo = true;
        while (erased_dead_stablehlo) {
            erased_dead_stablehlo = false;
            llvm::SmallVector<mlir::Operation*> operations;
            for (mlir::Operation& operation :
                 getOperation().getBody().front())
                operations.push_back(&operation);
            for (mlir::Operation* operation : llvm::reverse(operations)) {
                if (operation->getName().getDialectNamespace()
                        != "stablehlo"
                    || !operation->use_empty()
                    || operation->hasTrait<
                        mlir::OpTrait::IsTerminator>())
                    continue;
                rewriter.eraseOp(operation);
                erased_dead_stablehlo = true;
            }
        }

        llvm::SmallVector<mlir::Operation*> dot_generals;
        getOperation().walk([&](mlir::Operation* operation) {
            if (operation->getName().getStringRef() == "stablehlo.dot_general") {
                dot_generals.push_back(operation);
            }
        });

        for (mlir::Operation* dot_general : dot_generals) {
            if (dot_general->getNumOperands() != 2 || dot_general->getNumResults() != 1) {
                dot_general->emitError("expected two operands and one result");
                signalPassFailure();
                return;
            }
            const auto lhs = llvm::dyn_cast<mlir::RankedTensorType>(dot_general->getOperand(0).getType());
            const auto rhs = llvm::dyn_cast<mlir::RankedTensorType>(dot_general->getOperand(1).getType());
            const auto result = llvm::dyn_cast<mlir::RankedTensorType>(dot_general->getResult(0).getType());
            auto dot = llvm::cast<mlir::stablehlo::DotGeneralOp>(dot_general);
            const auto dimensions = dot.getDotDimensionNumbers();
            const auto lhs_batch = dimensions.getLhsBatchingDimensions();
            const auto rhs_batch = dimensions.getRhsBatchingDimensions();
            const auto lhs_contract = dimensions.getLhsContractingDimensions();
            const auto rhs_contract = dimensions.getRhsContractingDimensions();
            if (lhs && rhs && result && lhs.getRank() == 3
                && rhs.getRank() == 3 && result.getRank() == 3) {
                // Kernel matmul does not encode StableHLO algorithm/precision
                // overrides, so do not silently discard their numeric contract.
                bool default_precision = !dot.getAlgorithmAttr();
                if (const auto precision = dot.getPrecisionConfigAttr()) {
                    for (mlir::Attribute attribute : precision)
                        default_precision &= llvm::cast<mlir::stablehlo::PrecisionAttr>(
                            attribute).getValue() == mlir::stablehlo::Precision::DEFAULT;
                }
                if (!default_precision) {
                    dot_general->emitError(
                        "rank-3 matmul supports only DEFAULT precision and no explicit algorithm");
                    signalPassFailure();
                    return;
                }
                if (!lhs.hasStaticShape() || !rhs.hasStaticShape()
                    || !result.hasStaticShape()
                    || lhs_batch != llvm::ArrayRef<int64_t>({0})
                    || rhs_batch != llvm::ArrayRef<int64_t>({0})
                    || lhs_contract != llvm::ArrayRef<int64_t>({2})
                    || rhs_contract.size() != 1
                    || (rhs_contract[0] != 1 && rhs_contract[0] != 2)) {
                    dot_general->emitError(
                        "rank-3 matmul requires static shapes, batch dimension 0, "
                        "and contracting dimensions [2] x [1 or 2]");
                    signalPassFailure();
                    return;
                }
                const bool transpose_rhs = rhs_contract[0] == 2;
                const int64_t m = lhs.getDimSize(1);
                const int64_t k = lhs.getDimSize(2);
                const int64_t n = rhs.getDimSize(transpose_rhs ? 1 : 2);
                if (lhs.getDimSize(0) != rhs.getDimSize(0)
                    || result.getShape() != llvm::ArrayRef<int64_t>(
                        {lhs.getDimSize(0), m, n})
                    || k != rhs.getDimSize(rhs_contract[0])) {
                    dot_general->emitError("rank-3 tensor shapes do not satisfy batched matmul");
                    signalPassFailure();
                    return;
                }
                rewriter.setInsertionPoint(dot_general);
                if (lhs.getDimSize(0) != 1) {
                    mlir::OperationState state(dot_general->getLoc(),
                        kernel::BatchMatmulOp::getOperationName());
                    state.addOperands(dot_general->getOperands());
                    state.addTypes(result);
                    state.addAttribute("transpose_rhs",
                        rewriter.getBoolAttr(transpose_rhs));
                    // A generic dot is not an identified QK/PV attention graph.
                    state.addAttribute("role", rewriter.getStringAttr("generic"));
                    rewriter.replaceOp(dot_general, rewriter.create(state)->getResult(0));
                } else {
                    // Removing a singleton batch preserves every element and
                    // lets the existing rank-2 matmul handle e.g. RoPE angles.
                    const auto reshape = [&](mlir::Value input,
                                             mlir::RankedTensorType type) {
                        mlir::OperationState state(dot_general->getLoc(),
                            kernel::ReshapeOp::getOperationName());
                        state.addOperands(input);
                        state.addTypes(type);
                        return rewriter.create(state)->getResult(0);
                    };
                    const auto lhs_2d = mlir::RankedTensorType::get(
                        {m, k}, lhs.getElementType());
                    auto lhs_value = reshape(dot_general->getOperand(0), lhs_2d);
                    auto rhs_value = reshape(dot_general->getOperand(1),
                        mlir::RankedTensorType::get(
                            rhs.getShape().drop_front(), rhs.getElementType()));
                    if (transpose_rhs) {
                        mlir::OperationState state(dot_general->getLoc(),
                            kernel::TransposeOp::getOperationName());
                        state.addOperands(rhs_value);
                        state.addTypes(mlir::RankedTensorType::get(
                            {k, n}, rhs.getElementType()));
                        state.addAttribute("permutation",
                            rewriter.getDenseI64ArrayAttr({1, 0}));
                        rhs_value = rewriter.create(state)->getResult(0);
                    }
                    auto matmul = rewriter.create<kernel::MatmulOp>(
                        dot_general->getLoc(), lhs_value, rhs_value,
                        mlir::RankedTensorType::get({m, n}, result.getElementType()),
                        m, n, k);
                    rewriter.replaceOp(dot_general, reshape(matmul.getResult(), result));
                }
                continue;
            }
            if (!lhs || !rhs || !result || lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2
                || lhs.getShape()[1] != rhs.getShape()[0] || lhs.getShape()[0] != result.getShape()[0]
                || rhs.getShape()[1] != result.getShape()[1]) {
                dot_general->emitError("only well-formed rank-2 matrix multiplication is supported");
                signalPassFailure();
                return;
            }
            if (!lhs_batch.empty() || !rhs_batch.empty()
                || lhs_contract != llvm::ArrayRef<int64_t>({1})
                || rhs_contract != llvm::ArrayRef<int64_t>({0})) {
                dot_general->emitError(
                    "rank-2 matmul requires no batching and contracting dimensions [1] x [0]");
                signalPassFailure();
                return;
            }
            if (!default_dot_contract(dot_general, 1, 0)) {
                dot_general->emitError(
                    "rank-2 matmul supports only DEFAULT precision and no explicit algorithm");
                signalPassFailure();
                return;
            }

            mlir::Value loweredRhs = dot_general->getOperand(1);
            float rhsScale = 1.0f;
            if (const auto scaledWeight =
                    match_scaled_weight(dot_general->getOperand(1))) {
                const auto storageType =
                    llvm::dyn_cast<mlir::RankedTensorType>(
                        scaledWeight->storage.getType());
                if (storageType
                    && storageType.getElementType().isInteger(8)
                    && is_lpu_16bit_float(lhs.getElementType())
                    && is_lpu_16bit_float(result.getElementType())) {
                    loweredRhs = scaledWeight->storage;
                    rhsScale = scaledWeight->scale;
                }
            }
            rewriter.setInsertionPoint(dot_general);
            const auto lowered = rewriter.create<kernel::MatmulOp>(dot_general->getLoc(),
                dot_general->getOperand(0), loweredRhs, result,
                lhs.getShape()[0], rhs.getShape()[1], lhs.getShape()[1]);
            lowered->setAttr(
                "rhs_scale", rewriter.getF32FloatAttr(rhsScale));
            rewriter.replaceOp(dot_general, lowered->getResult(0));
        }

        bool erasedDeadStablehlo = true;
        while (erasedDeadStablehlo) {
            erasedDeadStablehlo = false;
            llvm::SmallVector<mlir::Operation*> operations;
            for (mlir::Operation& operation :
                 getOperation().getBody().front())
                operations.push_back(&operation);
            for (mlir::Operation* operation :
                 llvm::reverse(operations)) {
                if (operation->getName().getDialectNamespace()
                        != "stablehlo"
                    || !operation->use_empty()
                    || operation->hasTrait<
                        mlir::OpTrait::IsTerminator>())
                    continue;
                rewriter.eraseOp(operation);
                erasedDeadStablehlo = true;
            }
        }

        llvm::SmallVector<mlir::Operation*> elementwiseOperations;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.add")
                || named(operation, "stablehlo.multiply"))
                elementwiseOperations.push_back(operation);
        });
        for (mlir::Operation* operation :
            llvm::reverse(elementwiseOperations)) {
            if (!operation->getBlock() || operation->getNumOperands() != 2
                || operation->getNumResults() != 1)
                continue;
            const auto lhs = llvm::dyn_cast<mlir::RankedTensorType>(
                operation->getOperand(0).getType());
            const auto rhs = llvm::dyn_cast<mlir::RankedTensorType>(
                operation->getOperand(1).getType());
            const auto result = llvm::dyn_cast<mlir::RankedTensorType>(
                operation->getResult(0).getType());
            if (!lhs || lhs != rhs || lhs != result || lhs.getRank() != 2)
                continue;
            rewriter.setInsertionPoint(operation);
            mlir::OperationState state(operation->getLoc(),
                kernel::ElementwiseOp::getOperationName());
            state.addOperands(operation->getOperands());
            state.addTypes(result);
            state.addAttribute("kind", rewriter.getStringAttr(
                named(operation, "stablehlo.add") ? "add" : "multiply"));
            auto lowered =
                llvm::cast<kernel::ElementwiseOp>(rewriter.create(state));
            rewriter.replaceOp(operation, lowered.getResult());
        }

        llvm::SmallVector<mlir::Operation*> deadConstants;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.constant")
                && operation->use_empty())
                deadConstants.push_back(operation);
        });
        for (mlir::Operation* operation : llvm::reverse(deadConstants))
            rewriter.eraseOp(operation);
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_stablehlo_to_kernel_pass()
{
    return std::make_unique<LowerStableHloToKernelPass>();
}

void register_ftlpu_passes()
{
    static mlir::PassRegistration<LowerStableHloToKernelPass> registration;
    (void)registration;
}

} // namespace ftlpu::compiler
