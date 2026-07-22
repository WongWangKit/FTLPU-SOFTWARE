#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

bool named(mlir::Operation* op, llvm::StringRef name)
{
    return op && op->getName().getStringRef() == name;
}

bool attention_custom_call(mlir::Operation* op)
{
    if (!named(op, "stablehlo.custom_call") || op->getNumOperands() != 5
        || op->getNumResults() != 1)
        return false;
    const auto target = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    return target && target.getValue() == "ftlpu.attention";
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
        llvm::SmallVector<mlir::Operation*> attention_calls;
        getOperation().walk([&](mlir::Operation* operation) {
            if (attention_custom_call(operation)) attention_calls.push_back(operation);
        });
        mlir::IRRewriter rewriter(&getContext());
        for (mlir::Operation* call : attention_calls) {
            const auto input = llvm::dyn_cast<mlir::RankedTensorType>(call->getOperand(0).getType());
            const auto query = llvm::dyn_cast<mlir::RankedTensorType>(call->getOperand(1).getType());
            const auto key = llvm::dyn_cast<mlir::RankedTensorType>(call->getOperand(2).getType());
            const auto result = llvm::dyn_cast<mlir::RankedTensorType>(call->getResult(0).getType());
            if (!input || !query || !key || !result || input.getRank() != 2 || query.getRank() != 2
                || key.getRank() != 2 || result.getRank() != 2) {
                call->emitError("ftlpu.attention custom call requires rank-2 tensors");
                signalPassFailure();
                return;
            }
            const auto query_heads = call->getAttrOfType<mlir::IntegerAttr>("query_heads");
            const auto kv_heads = call->getAttrOfType<mlir::IntegerAttr>("kv_heads");
            const auto head_dim = call->getAttrOfType<mlir::IntegerAttr>("head_dim");
            const auto rope_theta = call->getAttrOfType<mlir::FloatAttr>("rope_theta");
            const auto causal = call->getAttrOfType<mlir::BoolAttr>("causal");
            if (!query_heads || !kv_heads || !head_dim || !rope_theta || !causal) {
                call->emitError("ftlpu.attention requires query_heads, kv_heads, head_dim, rope_theta, and causal attributes");
                signalPassFailure();
                return;
            }
            rewriter.setInsertionPoint(call);
            mlir::OperationState state(call->getLoc(), kernel::AttentionOp::getOperationName());
            state.addOperands(call->getOperands());
            state.addTypes(result);
            state.addAttributes({
                rewriter.getNamedAttr("seq_len", rewriter.getI64IntegerAttr(input.getDimSize(0))),
                rewriter.getNamedAttr("hidden", rewriter.getI64IntegerAttr(input.getDimSize(1))),
                rewriter.getNamedAttr("query_heads", query_heads),
                rewriter.getNamedAttr("kv_heads", kv_heads),
                rewriter.getNamedAttr("head_dim", head_dim),
                rewriter.getNamedAttr("rope_theta", rope_theta),
                rewriter.getNamedAttr("causal", causal),
            });
            auto lowered = llvm::cast<kernel::AttentionOp>(rewriter.create(state));
            rewriter.replaceOp(call, lowered.getResult());
        }

        llvm::SmallVector<mlir::Operation*> converts;
        getOperation().walk([&](mlir::Operation* operation) {
            if (named(operation, "stablehlo.convert")) converts.push_back(operation);
        });
        for (mlir::Operation* convert : llvm::reverse(converts)) {
            if (!convert->getBlock()) continue;
            if (auto match = match_w8a16_ffn(convert)) {
                auto input_type = llvm::cast<mlir::RankedTensorType>(match->input.getType());
                auto gate_type = llvm::cast<mlir::RankedTensorType>(match->gate_weight.getType());
                auto down_type = llvm::cast<mlir::RankedTensorType>(match->down_weight.getType());
                auto result_type = llvm::cast<mlir::RankedTensorType>(convert->getResult(0).getType());
                rewriter.setInsertionPoint(convert);
                mlir::OperationState state(convert->getLoc(), kernel::FfnOp::getOperationName());
                state.addOperands({match->input, match->gate_weight, match->up_weight, match->down_weight});
                state.addTypes(result_type);
                state.addAttributes({
                    rewriter.getNamedAttr("m", rewriter.getI64IntegerAttr(input_type.getDimSize(0))),
                    rewriter.getNamedAttr("k", rewriter.getI64IntegerAttr(input_type.getDimSize(1))),
                    rewriter.getNamedAttr("hidden", rewriter.getI64IntegerAttr(gate_type.getDimSize(1))),
                    rewriter.getNamedAttr("n", rewriter.getI64IntegerAttr(down_type.getDimSize(1))),
                    rewriter.getNamedAttr("gate_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("up_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("hidden_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("hidden_zero_point", rewriter.getI64IntegerAttr(0)),
                    rewriter.getNamedAttr("down_lhs_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("down_rhs_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("output_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("output_zero_point", rewriter.getI64IntegerAttr(0)),
                });
                auto lowered = llvm::cast<kernel::FfnOp>(rewriter.create(state));
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
                mlir::OperationState state(match->final_convert->getLoc(),
                    kernel::FfnOp::getOperationName());
                state.addOperands({match->swiglu.gate_dot->getOperand(0),
                    match->swiglu.gate_dot->getOperand(1),
                    match->swiglu.up_dot->getOperand(1),
                    match->down_dot->getOperand(1)});
                state.addTypes(result_type);
                state.addAttributes({
                    rewriter.getNamedAttr("m", rewriter.getI64IntegerAttr(input_type.getDimSize(0))),
                    rewriter.getNamedAttr("k", rewriter.getI64IntegerAttr(input_type.getDimSize(1))),
                    rewriter.getNamedAttr("hidden", rewriter.getI64IntegerAttr(gate_type.getDimSize(1))),
                    rewriter.getNamedAttr("n", rewriter.getI64IntegerAttr(down_type.getDimSize(1))),
                    rewriter.getNamedAttr("gate_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("up_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("hidden_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("hidden_zero_point", rewriter.getI64IntegerAttr(0)),
                    rewriter.getNamedAttr("down_lhs_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("down_rhs_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("output_scale", rewriter.getF32FloatAttr(1.0f)),
                    rewriter.getNamedAttr("output_zero_point", rewriter.getI64IntegerAttr(0)),
                });
                auto lowered = llvm::cast<kernel::FfnOp>(rewriter.create(state));
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
