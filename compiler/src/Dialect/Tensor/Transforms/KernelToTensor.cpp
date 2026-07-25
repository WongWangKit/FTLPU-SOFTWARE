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
        llvm::DenseMap<mlir::Value, Allocation> allocations;
        llvm::DenseMap<mlir::Value, int64_t> last_uses;
        llvm::DenseMap<mlir::Operation*, int64_t> ordinals;
        llvm::SmallVector<kernel::MatmulOp> matmuls;
        llvm::SmallVector<kernel::SwigluOp> swiglus;
        llvm::SmallVector<kernel::FfnGraph, 2> ffns;
        llvm::SmallDenseSet<mlir::Operation*, 16> ffn_operations;
        llvm::SmallDenseSet<mlir::Operation*, 16> fixed_ffn_operations;
        llvm::SmallVector<kernel::AttentionGraph, 2> attentions;
        llvm::SmallDenseSet<mlir::Operation*, 32> attention_operations;
        int64_t ordinal = 0;
        for (mlir::Operation& operation : function.getBody().front()) {
            ordinals[&operation] = ordinal;
            for (mlir::Value operand : operation.getOperands()) last_uses[operand] = ordinal;
            if (auto matmul = llvm::dyn_cast<kernel::MatmulOp>(&operation)) matmuls.push_back(matmul);
            if (auto swiglu = llvm::dyn_cast<kernel::SwigluOp>(&operation)) swiglus.push_back(swiglu);
            ++ordinal;
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

        auto allocate_value = [&](mlir::Value value, PlacementKind kind) -> mlir::FailureOr<Allocation> {
            if (const auto found = allocations.find(value); found != allocations.end())
                return found->second.kind == kind ? mlir::FailureOr<Allocation>(found->second)
                                                  : mlir::FailureOr<Allocation>(mlir::failure());
            const auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
            const auto bytes = get_static_tensor_bytes(type);
            if (mlir::failed(bytes)) return mlir::failure();
            const auto allocation = allocator.allocate(kind, *bytes);
            if (mlir::failed(allocation)) return mlir::failure();
            allocations.try_emplace(value, *allocation);
            return *allocation;
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
        for (mlir::BlockArgument argument : function.getArguments()) {
            if (!last_uses.contains(argument)) continue;
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

        mlir::IRRewriter rewriter(&getContext());
        for (kernel::AttentionGraph& graph : attentions) {
            if (mlir::failed(lower_attention(graph, target, rewriter))) {
                signalPassFailure();
                return;
            }
        }
        for (kernel::FfnGraph& graph : ffns) {
            if (mlir::failed(lower_ffn(graph, target, allocator,
                    allocate_value, rewriter))) {
                signalPassFailure();
                return;
            }
        }
        for (kernel::SwigluOp op : swiglus) {
            if (mlir::failed(lower_swiglu(op, allocator,
                    allocate_value, rewriter))) {
                signalPassFailure();
                return;
            }
        }
        for (kernel::MatmulOp op : matmuls) {
            if (mlir::failed(lower_matmul(op, allocator, allocations,
                    last_uses, ordinals, allocate_value, rewriter))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_kernel_to_tensor_pass()
{
    return std::make_unique<LowerKernelToTensorPass>();
}

} // namespace ftlpu::compiler
