#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

struct AttentionGraph {
    kernel::MatmulOp output;
    kernel::MatmulOp query;
    kernel::MatmulOp key;
    kernel::MatmulOp value;
    kernel::RopeOp query_rope;
    kernel::RopeOp key_rope;
    kernel::SoftmaxOp softmax;
    llvm::SmallVector<mlir::Operation*, 20> operations;
};

std::optional<AttentionGraph> match_attention(kernel::MatmulOp output)
{
    auto context = output.getLhs().getDefiningOp<kernel::ReshapeOp>();
    if (!context) return std::nullopt;
    auto context_transpose =
        context.getInput().getDefiningOp<kernel::TransposeOp>();
    auto pv = context_transpose
        ? context_transpose.getInput().getDefiningOp<kernel::BatchMatmulOp>()
        : kernel::BatchMatmulOp{};
    if (!pv || pv.getRole() != "pv" || pv.getTransposeRhs())
        return std::nullopt;
    auto softmax = pv.getLhs().getDefiningOp<kernel::SoftmaxOp>();
    auto value_transpose =
        pv.getRhs().getDefiningOp<kernel::TransposeOp>();
    auto value_gqa = value_transpose
        ? value_transpose.getInput().getDefiningOp<kernel::GqaBroadcastOp>()
        : kernel::GqaBroadcastOp{};
    auto value_reshape = value_gqa
        ? value_gqa.getInput().getDefiningOp<kernel::ReshapeOp>()
        : kernel::ReshapeOp{};
    auto value = value_reshape
        ? value_reshape.getInput().getDefiningOp<kernel::MatmulOp>()
        : kernel::MatmulOp{};
    auto qk = softmax
        ? softmax.getInput().getDefiningOp<kernel::BatchMatmulOp>()
        : kernel::BatchMatmulOp{};
    if (!softmax || !value || !qk || qk.getRole() != "qk"
        || !qk.getTransposeRhs())
        return std::nullopt;

    auto query_transpose =
        qk.getLhs().getDefiningOp<kernel::TransposeOp>();
    auto key_transpose =
        qk.getRhs().getDefiningOp<kernel::TransposeOp>();
    auto query_rope = query_transpose
        ? query_transpose.getInput().getDefiningOp<kernel::RopeOp>()
        : kernel::RopeOp{};
    auto key_gqa = key_transpose
        ? key_transpose.getInput().getDefiningOp<kernel::GqaBroadcastOp>()
        : kernel::GqaBroadcastOp{};
    auto key_rope = key_gqa
        ? key_gqa.getInput().getDefiningOp<kernel::RopeOp>()
        : kernel::RopeOp{};
    auto query_reshape = query_rope
        ? query_rope.getInput().getDefiningOp<kernel::ReshapeOp>()
        : kernel::ReshapeOp{};
    auto key_reshape = key_rope
        ? key_rope.getInput().getDefiningOp<kernel::ReshapeOp>()
        : kernel::ReshapeOp{};
    auto query = query_reshape
        ? query_reshape.getInput().getDefiningOp<kernel::MatmulOp>()
        : kernel::MatmulOp{};
    auto key = key_reshape
        ? key_reshape.getInput().getDefiningOp<kernel::MatmulOp>()
        : kernel::MatmulOp{};
    if (!query || !key || query.getLhs() != key.getLhs()
        || query.getLhs() != value.getLhs()
        || query_rope.getHeadDim() != key_rope.getHeadDim()
        || query_rope.getTheta() != key_rope.getTheta()
        || value_gqa.getQueryHeads() != query_rope.getHeads()
        || value_gqa.getKvHeads() != key_rope.getHeads())
        return std::nullopt;

    AttentionGraph graph {
        output, query, key, value, query_rope, key_rope, softmax,
        {query.getOperation(), key.getOperation(), value.getOperation(),
         query_reshape.getOperation(), key_reshape.getOperation(),
         value_reshape.getOperation(), query_rope.getOperation(),
         key_rope.getOperation(), key_gqa.getOperation(),
         value_gqa.getOperation(), query_transpose.getOperation(),
         key_transpose.getOperation(), value_transpose.getOperation(),
         qk.getOperation(), softmax.getOperation(), pv.getOperation(),
         context_transpose.getOperation(), context.getOperation(),
         output.getOperation()}
    };
    return graph;
}

struct FfnGraph {
    kernel::MatmulOp output;
    kernel::MatmulOp gate;
    kernel::MatmulOp up;
    kernel::SwishOp swish;
    kernel::ElementwiseOp multiply;
    llvm::SmallVector<mlir::Operation*, 5> operations;
};

std::optional<FfnGraph> match_ffn(kernel::MatmulOp output)
{
    auto multiply = output.getLhs().getDefiningOp<kernel::ElementwiseOp>();
    if (!multiply || multiply.getKind() != "multiply") return std::nullopt;
    kernel::SwishOp swish =
        multiply.getLhs().getDefiningOp<kernel::SwishOp>();
    kernel::MatmulOp up =
        multiply.getRhs().getDefiningOp<kernel::MatmulOp>();
    if (!swish || !up) {
        swish = multiply.getRhs().getDefiningOp<kernel::SwishOp>();
        up = multiply.getLhs().getDefiningOp<kernel::MatmulOp>();
    }
    auto gate = swish
        ? swish.getInput().getDefiningOp<kernel::MatmulOp>()
        : kernel::MatmulOp{};
    if (!gate || !up || gate.getLhs() != up.getLhs()
        || gate.getM() != up.getM() || gate.getN() != up.getN()
        || gate.getK() != up.getK())
        return std::nullopt;
    return FfnGraph {output, gate, up, swish, multiply,
        {gate.getOperation(), up.getOperation(), swish.getOperation(),
         multiply.getOperation(), output.getOperation()}};
}

class ComposeKernelPlansPass final
    : public mlir::PassWrapper<ComposeKernelPlansPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComposeKernelPlansPass)

    llvm::StringRef getArgument() const final
    {
        return "ftlpu-compose-kernel-plans";
    }

    llvm::StringRef getDescription() const final
    {
        return "Temporarily composes primitive Kernel IR for the legacy Tensor backend";
    }

    void runOnOperation() final
    {
        llvm::SmallVector<kernel::MatmulOp> roots;
        getOperation().walk(
            [&](kernel::MatmulOp op) { roots.push_back(op); });
        mlir::IRRewriter rewriter(&getContext());
        llvm::SmallPtrSet<mlir::Operation*, 32> erased;

        for (kernel::MatmulOp root : llvm::reverse(roots)) {
            mlir::Operation* root_operation = root.getOperation();
            if (erased.contains(root_operation)) continue;
            if (auto attention = match_attention(root)) {
                erased.insert(attention->operations.begin(),
                    attention->operations.end());
                rewriter.setInsertionPoint(root);
                mlir::OperationState state(
                    root.getLoc(), kernel::AttentionOp::getOperationName());
                state.addOperands({attention->query.getLhs(),
                    attention->query.getRhs(), attention->key.getRhs(),
                    attention->value.getRhs(), root.getRhs()});
                state.addTypes(root.getResult().getType());
                state.addAttributes({
                    rewriter.getNamedAttr("seq_len",
                        rewriter.getI64IntegerAttr(attention->query.getM())),
                    rewriter.getNamedAttr("hidden",
                        rewriter.getI64IntegerAttr(attention->query.getK())),
                    rewriter.getNamedAttr("query_heads",
                        attention->query_rope.getHeadsAttr()),
                    rewriter.getNamedAttr("kv_heads",
                        attention->key_rope.getHeadsAttr()),
                    rewriter.getNamedAttr("head_dim",
                        attention->query_rope.getHeadDimAttr()),
                    rewriter.getNamedAttr("rope_theta",
                        attention->query_rope.getThetaAttr()),
                    rewriter.getNamedAttr("causal",
                        attention->softmax.getCausalAttr()),
                });
                auto plan =
                    llvm::cast<kernel::AttentionOp>(rewriter.create(state));
                rewriter.replaceOp(root, plan.getResult());
                for (mlir::Operation* operation :
                     llvm::reverse(attention->operations)) {
                    if (operation != root_operation
                        && operation->use_empty())
                        rewriter.eraseOp(operation);
                }
                continue;
            }

            auto ffn = match_ffn(root);
            if (!ffn) continue;
            erased.insert(ffn->operations.begin(), ffn->operations.end());
            rewriter.setInsertionPoint(root);
            mlir::OperationState state(
                root.getLoc(), kernel::FfnOp::getOperationName());
            state.addOperands({ffn->gate.getLhs(), ffn->gate.getRhs(),
                ffn->up.getRhs(), root.getRhs()});
            state.addTypes(root.getResult().getType());
            state.addAttributes({
                rewriter.getNamedAttr("m",
                    rewriter.getI64IntegerAttr(ffn->gate.getM())),
                rewriter.getNamedAttr("k",
                    rewriter.getI64IntegerAttr(ffn->gate.getK())),
                rewriter.getNamedAttr("hidden",
                    rewriter.getI64IntegerAttr(ffn->gate.getN())),
                rewriter.getNamedAttr("n",
                    rewriter.getI64IntegerAttr(root.getN())),
                rewriter.getNamedAttr("gate_scale",
                    rewriter.getF32FloatAttr(1.0f)),
                rewriter.getNamedAttr("up_scale",
                    rewriter.getF32FloatAttr(1.0f)),
                rewriter.getNamedAttr("hidden_scale",
                    rewriter.getF32FloatAttr(1.0f)),
                rewriter.getNamedAttr("hidden_zero_point",
                    rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("down_lhs_scale",
                    rewriter.getF32FloatAttr(1.0f)),
                rewriter.getNamedAttr("down_rhs_scale",
                    rewriter.getF32FloatAttr(1.0f)),
                rewriter.getNamedAttr("output_scale",
                    rewriter.getF32FloatAttr(1.0f)),
                rewriter.getNamedAttr("output_zero_point",
                    rewriter.getI64IntegerAttr(0)),
            });
            auto plan = llvm::cast<kernel::FfnOp>(rewriter.create(state));
            rewriter.replaceOp(root, plan.getResult());
            for (mlir::Operation* operation : llvm::reverse(ffn->operations)) {
                if (operation != root_operation && operation->use_empty())
                    rewriter.eraseOp(operation);
            }
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_compose_kernel_plans_pass()
{
    return std::make_unique<ComposeKernelPlansPass>();
}

} // namespace ftlpu::compiler
