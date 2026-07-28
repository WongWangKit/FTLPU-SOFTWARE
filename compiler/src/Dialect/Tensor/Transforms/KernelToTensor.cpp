#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {
using namespace tensor_lowering;

class LowerKernelToTensorPass final
    : public mlir::PassWrapper<LowerKernelToTensorPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKernelToTensorPass)

    LowerKernelToTensorPass() = default;
    explicit LowerKernelToTensorPass(RmsNormLoweringStrategy strategy)
        : rmsnorm_strategy_(strategy)
    {
    }

    llvm::StringRef getArgument() const final { return "ftlpu-kernel-to-tensor"; }
    llvm::StringRef getDescription() const final
    {
        return "Assigns physical LPU MEM storage to Kernel IR tensors";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError("MEM lifetime allocation currently requires a single-block function");
            signalPassFailure();
            return;
        }

        EastMemoryAllocator allocator;
        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        llvm::SmallVector<kernel::MatmulOp> matmuls;
        llvm::SmallVector<kernel::SwigluOp> swiglus;
        llvm::SmallVector<kernel::RmsNormOp> rmsNorms;
        llvm::SmallVector<kernel::ElementwiseOp> elementwiseOps;
        llvm::SmallVector<kernel::FfnGraph, 2> ffns;
        llvm::SmallDenseSet<mlir::Operation*, 16> ffn_operations;
        llvm::SmallDenseSet<mlir::Operation*, 16> fixed_ffn_operations;
        llvm::SmallVector<kernel::AttentionGraph, 2> attentions;
        llvm::SmallDenseSet<mlir::Operation*, 32> attention_operations;
        llvm::SmallVector<mlir::Operation*, 64> operation_order;
        for (mlir::Operation& operation : function.getBody().front()) {
            operation_order.push_back(&operation);
            if (auto matmul = llvm::dyn_cast<kernel::MatmulOp>(&operation)) matmuls.push_back(matmul);
            if (auto swiglu = llvm::dyn_cast<kernel::SwigluOp>(&operation)) swiglus.push_back(swiglu);
            if (auto rmsNorm =
                    llvm::dyn_cast<kernel::RmsNormOp>(&operation))
                rmsNorms.push_back(rmsNorm);
            if (auto elementwise =
                    llvm::dyn_cast<kernel::ElementwiseOp>(&operation))
                elementwiseOps.push_back(elementwise);
        }
        for (kernel::MatmulOp root : llvm::reverse(matmuls)) {
            auto graph = kernel::match_attention_graph(root);
            if (!graph) continue;
            attention_operations.insert(
                graph->operations.begin(), graph->operations.end());
            attentions.push_back(std::move(*graph));
        }
        llvm::erase_if(matmuls, [&](kernel::MatmulOp op) {
            return attention_operations.contains(op.getOperation());
        });
        for (kernel::MatmulOp root : llvm::reverse(matmuls)) {
            auto graph = kernel::match_ffn_graph(root);
            if (!graph) continue;
            ffn_operations.insert(
                graph->operations.begin(), graph->operations.end());
            if (is_w8a16_ffn(*graph, target))
                fixed_ffn_operations.insert(
                    graph->operations.begin(), graph->operations.end());
            ffns.push_back(std::move(*graph));
        }
        llvm::erase_if(matmuls, [&](kernel::MatmulOp op) {
            return ffn_operations.contains(op.getOperation());
        });
        llvm::erase_if(elementwiseOps, [&](kernel::ElementwiseOp op) {
            return ffn_operations.contains(op.getOperation())
                || attention_operations.contains(op.getOperation());
        });

        FunctionMemoryPlanner planner(function, allocator);
        auto allocate_value = [&](mlir::Value value, PlacementKind kind) {
            return planner.allocate(value, kind);
        };

        auto argument_kind = [](mlir::BlockArgument argument) {
            for (mlir::OpOperand& use : argument.getUses()) {
                if (auto matmul = llvm::dyn_cast<kernel::MatmulOp>(use.getOwner())) {
                    if (use.getOperandNumber() == 1) return PlacementKind::Weight;
                }
                if (auto swiglu = llvm::dyn_cast<kernel::SwigluOp>(use.getOwner())) {
                    if (use.getOperandNumber() == 1 || use.getOperandNumber() == 2)
                        return PlacementKind::Weight;
                }
            }
            return PlacementKind::Activation;
        };

        // Function arguments are model inputs and coexist in MEM at entry.
        int64_t rmsWeightBase = 0;
        for (mlir::BlockArgument argument : function.getArguments()) {
            if (argument.use_empty()) continue;
            const auto type =
                llvm::dyn_cast<mlir::RankedTensorType>(argument.getType());
            if (type && type.getRank() == 2
                && type.getElementType().isF16()
                && type.getDimSize(1) % target.throughput().mxm_rows == 0) {
                bool feedsFeedbackRmsNorm = false;
                if (rmsnorm_strategy_
                    == RmsNormLoweringStrategy::VxmFeedback) {
                    for (mlir::OpOperand& use : argument.getUses()) {
                        auto rmsNorm = llvm::dyn_cast<kernel::RmsNormOp>(
                            use.getOwner());
                        feedsFeedbackRmsNorm |= rmsNorm
                            && use.getOperandNumber() == 0;
                    }
                }
                if (feedsFeedbackRmsNorm) {
                    const int64_t bytes = type.getNumElements() * 2;
                    const auto allocation = fixed_allocation(
                        PlacementKind::Activation,
                        target.mxm_distributed_activation_slices(),
                        4096, type.getNumElements() / 128, bytes,
                        "fp16_mxm_distributed_16", "both");
                    if (mlir::failed(planner.bind(argument, allocation))) {
                        function.emitError(
                            "conflicting distributed input placement");
                        signalPassFailure();
                        return;
                    }
                    continue;
                }
                continue;
            }
            bool rmsWeight = false;
            for (mlir::OpOperand& use : argument.getUses()) {
                auto rmsNorm =
                    llvm::dyn_cast<kernel::RmsNormOp>(use.getOwner());
                rmsWeight |= rmsNorm && use.getOperandNumber() == 1;
            }
            if (rmsWeight) {
                const auto type =
                    llvm::cast<mlir::RankedTensorType>(
                        argument.getType());
                const int64_t instructions =
                    type.getNumElements()
                    / target.throughput().mxm_rows;
                const auto allocation = fixed_allocation(
                    PlacementKind::Activation, {20, 21},
                    rmsWeightBase, instructions,
                    type.getNumElements() * 2,
                    "fp16_pair_planar", "both");
                rmsWeightBase +=
                    std::max<int64_t>(1,
                        (instructions - 1) * 16 + 1);
                if (mlir::failed(planner.bind(argument, allocation))) {
                    function.emitError(
                        "conflicting RMSNorm weight placement");
                    signalPassFailure();
                    return;
                }
                continue;
            }
            bool fixed_w8a16_operand = false;
            for (mlir::OpOperand& use : argument.getUses()) {
                fixed_w8a16_operand |=
                    fixed_ffn_operations.contains(use.getOwner());
                fixed_w8a16_operand |=
                    attention_operations.contains(use.getOwner());
            }
            if (fixed_w8a16_operand) continue;
            if (mlir::failed(allocate_value(argument, argument_kind(argument)))) {
                function.emitError("cannot allocate function inputs in the east MEM hemisphere");
                signalPassFailure();
                return;
            }
        }

        llvm::DenseMap<mlir::Operation*, size_t> attention_roots;
        llvm::DenseMap<mlir::Operation*, size_t> ffn_roots;
        for (size_t index = 0; index < attentions.size(); ++index)
            attention_roots[attentions[index].output.getOperation()] = index;
        for (size_t index = 0; index < ffns.size(); ++index)
            ffn_roots[ffns[index].output.getOperation()] = index;

        mlir::IRRewriter rewriter(&getContext());
        for (mlir::Operation* operation : operation_order) {
            planner.release_before(operation);
            if (auto found = attention_roots.find(operation);
                found != attention_roots.end()) {
                if (mlir::failed(lower_attention(
                        attentions[found->second], target, rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (auto found = ffn_roots.find(operation);
                found != ffn_roots.end()) {
                if (mlir::failed(lower_ffn(ffns[found->second], target,
                        allocator, allocate_value, rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (attention_operations.contains(operation)
                || ffn_operations.contains(operation))
                continue;
            if (auto op = llvm::dyn_cast<kernel::RmsNormOp>(operation)) {
                if (mlir::failed(lower_rms_norm(
                        op, target, rmsnorm_strategy_, allocate_value,
                        rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (auto op = llvm::dyn_cast<kernel::ElementwiseOp>(operation)) {
                if (mlir::failed(lower_elementwise(
                        op, target, allocator, allocate_value,
                        rmsnorm_strategy_, rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (auto op = llvm::dyn_cast<kernel::SwigluOp>(operation)) {
                if (mlir::failed(lower_swiglu(
                        op, allocator, allocate_value, rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (auto op = llvm::dyn_cast<kernel::MatmulOp>(operation)) {
                if (mlir::failed(lower_matmul(
                        op, planner, allocate_value, rewriter))) {
                    signalPassFailure();
                    return;
                }
            }
        }
    }

private:
    RmsNormLoweringStrategy rmsnorm_strategy_ =
        RmsNormLoweringStrategy::VxmSquareMxmReduce;
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_kernel_to_tensor_pass(
    RmsNormLoweringStrategy rmsnorm_strategy)
{
    return std::make_unique<LowerKernelToTensorPass>(rmsnorm_strategy);
}

} // namespace ftlpu::compiler
