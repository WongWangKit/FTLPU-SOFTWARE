#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <cmath>
#include <optional>

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
    int64_t seq_len;
    int64_t hidden;
    int64_t query_heads;
    int64_t kv_heads;
    int64_t head_dim;
    float rope_theta;
    bool causal;
    llvm::SmallPtrSet<mlir::Operation*, 32> operations;
};

std::optional<float> find_large_scalar_constant(
    const llvm::SmallPtrSetImpl<mlir::Operation*>& operations)
{
    for (mlir::Operation* operation : operations) {
        if (!named(operation, "stablehlo.constant")) continue;
        const auto value = operation->getAttrOfType<mlir::DenseFPElementsAttr>("value");
        if (!value || !value.isSplat() || value.getNumElements() != 1) continue;
        const float scalar = value.getSplatValue<llvm::APFloat>().convertToFloat();
        if (scalar > 1000.0f) return scalar;
    }
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
};

mlir::Value converted_source(mlir::Value value)
{
    auto* op = value.getDefiningOp();
    return named(op, "stablehlo.convert") && op->getNumOperands() == 1
        ? op->getOperand(0) : mlir::Value{};
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
    mlir::Value query_weight = converted_source(query_dot->getOperand(1));
    mlir::Value key_weight = converted_source(key_dot->getOperand(1));
    mlir::Value value_weight = converted_source(value_dot->getOperand(1));
    mlir::Value output_weight = converted_source(output_dot->getOperand(1));
    if (!query_weight || !key_weight || !value_weight || !output_weight)
        return std::nullopt;

    const auto input_type = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    const auto qk_lhs_type =
        llvm::dyn_cast<mlir::RankedTensorType>(qk_dot->getOperand(0).getType());
    const auto key_weight_storage =
        llvm::dyn_cast<mlir::RankedTensorType>(key_weight.getType());
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
        query_weight,
        key_weight,
        value_weight,
        output_weight,
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
    if (const auto theta = find_large_scalar_constant(match.operations))
        match.rope_theta = *theta;
    return match;
}

std::optional<W8A16FfnMatch> match_w8a16_ffn(mlir::Operation* final_convert)
{
    if (!named(final_convert, "stablehlo.convert") || final_convert->getNumOperands() != 1)
        return std::nullopt;
    auto result_type = llvm::dyn_cast<mlir::RankedTensorType>(final_convert->getResult(0).getType());
    if (!result_type || !result_type.getElementType().isF16()) return std::nullopt;
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
            mlir::Value gate_weight = converted_source(gate_dot->getOperand(1));
            mlir::Value up_weight = converted_source(up_dot->getOperand(1));
            mlir::Value down_weight = converted_source(down_dot->getOperand(1));
            if (!input || input != up_input || !gate_weight || !up_weight || !down_weight)
                continue;
            return W8A16FfnMatch {final_convert, down_dot, swiglu_mul, gated_mul,
                logistic, gate_dot, up_dot, input, gate_weight, up_weight, down_weight};
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

class LowerStableHloToKernelPass final
    : public mlir::PassWrapper<LowerStableHloToKernelPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStableHloToKernelPass)

    llvm::StringRef getArgument() const final { return "ftlpu-stablehlo-to-kernel"; }
    llvm::StringRef getDescription() const final
    {
        return "Lowers rank-2 stablehlo.dot_general operations to ftlpu.kernel.matmul";
    }

    void runOnOperation() final
    {
        llvm::SmallVector<mlir::Operation*> attention_roots;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.dot_general"))
                attention_roots.push_back(operation);
        });
        mlir::IRRewriter rewriter(&getContext());
        llvm::SmallPtrSet<mlir::Operation*, 32> erased_attention_operations;
        for (mlir::Operation* root : llvm::reverse(attention_roots)) {
            if (erased_attention_operations.contains(root)) continue;
            if (!root->getBlock()) continue;
            const auto match = match_standard_attention(root);
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
            auto key_2d = rewriter.create<kernel::MatmulOp>(root->getLoc(),
                match->input, match->key_weight,
                tensor_type({match->seq_len, kv_width}),
                match->seq_len, kv_width, match->hidden);
            auto value_2d = rewriter.create<kernel::MatmulOp>(root->getLoc(),
                match->input, match->value_weight,
                tensor_type({match->seq_len, kv_width}),
                match->seq_len, kv_width, match->hidden);

            auto query_heads = create_reshape(query_2d.getResult(),
                tensor_type({match->seq_len, match->query_heads, match->head_dim}));
            auto key_heads = create_reshape(key_2d.getResult(),
                tensor_type({match->seq_len, match->kv_heads, match->head_dim}));
            auto value_heads = create_reshape(value_2d.getResult(),
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
            rewriter.replaceOp(root, output.getResult());

            llvm::SmallVector<mlir::Operation*> block_operations;
            for (mlir::Operation& operation : getOperation().getBody().front())
                block_operations.push_back(&operation);
            for (mlir::Operation* operation : llvm::reverse(block_operations)) {
                if (match->operations.contains(operation) && operation->use_empty())
                    rewriter.eraseOp(operation);
            }
        }

        llvm::SmallVector<mlir::Operation*> converts;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.convert")) converts.push_back(operation);
        });
        const auto create_ffn_graph = [&](mlir::Location location,
                                          mlir::Value input,
                                          mlir::Value gate_weight,
                                          mlir::Value up_weight,
                                          mlir::Value down_weight,
                                          mlir::Type hidden_type,
                                          mlir::Type result_type,
                                          int64_t m, int64_t k,
                                          int64_t hidden, int64_t n) {
            auto gate = rewriter.create<kernel::MatmulOp>(location,
                input, gate_weight, hidden_type, m, hidden, k);
            auto up = rewriter.create<kernel::MatmulOp>(location,
                input, up_weight, hidden_type, m, hidden, k);
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
            return rewriter.create<kernel::MatmulOp>(location,
                gated.getResult(), down_weight, result_type, m, n, hidden);
        };
        for (mlir::Operation* convert : llvm::reverse(converts)) {
            if (!convert->getBlock()) continue;
            if (auto match = match_w8a16_ffn(convert)) {
                auto input_type = llvm::cast<mlir::RankedTensorType>(match->input.getType());
                auto gate_type = llvm::cast<mlir::RankedTensorType>(match->gate_weight.getType());
                auto down_type = llvm::cast<mlir::RankedTensorType>(match->down_weight.getType());
                auto result_type = llvm::cast<mlir::RankedTensorType>(convert->getResult(0).getType());
                rewriter.setInsertionPoint(convert);
                auto lowered = create_ffn_graph(convert->getLoc(),
                    match->input, match->gate_weight, match->up_weight,
                    match->down_weight, match->gate_dot->getResult(0).getType(),
                    result_type, input_type.getDimSize(0), input_type.getDimSize(1),
                    gate_type.getDimSize(1), down_type.getDimSize(1));
                rewriter.replaceOp(convert, lowered.getResult());
                for (mlir::Operation* op : {match->down_dot, match->swiglu_mul,
                         match->gated_mul, match->logistic, match->gate_dot, match->up_dot})
                    rewriter.eraseOp(op);
                llvm::SmallVector<mlir::Operation*> dead_converts;
                getOperation().walk([&](mlir::Operation* op) {
                    if (named(op, "stablehlo.convert") && op->use_empty()) dead_converts.push_back(op);
                });
                for (mlir::Operation* op : llvm::reverse(dead_converts)) rewriter.eraseOp(op);
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
                    gate_type.getDimSize(1), down_type.getDimSize(1));
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
            if (!lhs || !rhs || !result || lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2
                || lhs.getShape()[1] != rhs.getShape()[0] || lhs.getShape()[0] != result.getShape()[0]
                || rhs.getShape()[1] != result.getShape()[1]) {
                dot_general->emitError("only well-formed rank-2 matrix multiplication is supported");
                signalPassFailure();
                return;
            }

            rewriter.setInsertionPoint(dot_general);
            const auto lowered = rewriter.create<kernel::MatmulOp>(dot_general->getLoc(),
                dot_general->getOperand(0), dot_general->getOperand(1), result,
                lhs.getShape()[0], rhs.getShape()[1], lhs.getShape()[1]);
            rewriter.replaceOp(dot_general, lowered->getResult(0));
        }
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
