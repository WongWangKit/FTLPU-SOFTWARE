#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

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
        return "Composes recognized FFN graphs into target planning operations";
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
