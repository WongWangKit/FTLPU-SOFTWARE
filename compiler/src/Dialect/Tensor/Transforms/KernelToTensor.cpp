#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
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

struct FeedbackGammaPlacement {
    int64_t base_row = 0;
    int64_t slice_base = -1;
    int64_t bank = 0;
};

class LowerKernelToTensorPass final
    : public mlir::PassWrapper<LowerKernelToTensorPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKernelToTensorPass)

    LowerKernelToTensorPass() = default;
    LowerKernelToTensorPass(
        RmsNormLoweringStrategy strategy, int64_t weightBank)
        : rmsnorm_strategy_(strategy)
        , weight_bank_(weightBank)
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

        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        EastMemoryAllocator allocator(target);
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
        llvm::DenseMap<mlir::Value, FeedbackGammaPlacement>
            feedbackRmsWeightPlacements;
        if (rmsnorm_strategy_
                == RmsNormLoweringStrategy::VxmFeedback
            && target.uses_dedicated_slice_roles()) {
            const auto distributedSlices =
                target.mxm_distributed_activation_slices();
            llvm::SmallVector<int64_t> constantSlices;
            for (int64_t slice : target.activation_storage_slices())
                if (!llvm::is_contained(distributedSlices, slice))
                    constantSlices.push_back(slice);
            if (constantSlices.size() < 2) {
                function.emitError(
                    "feedback RMSNorm requires an activation constant slice pair");
                signalPassFailure();
                return;
            }
            const std::size_t pairCount = constantSlices.size() / 2;
            llvm::SmallVector<int64_t> nextRows(
                pairCount, target.memory().words_per_bank);
            // Attention keeps the direct-stream RoPE table in the weight
            // bank. Put eagerly uploaded gamma constants in the opposite
            // activation bank so host initialization cannot overwrite either
            // constant before execution begins.
            const int64_t constantBank = target.memory().banks_per_slice > 1
                ? (std::max<int64_t>(0, weight_bank_) + 1)
                    % target.memory().banks_per_slice
                : 0;
            for (kernel::RmsNormOp rmsNorm : llvm::reverse(rmsNorms)) {
                if (feedbackRmsWeightPlacements.contains(rmsNorm.getWeight()))
                    continue;
                const int64_t rows =
                    rmsNorm.getWeight().getType().getNumElements();
                std::optional<std::size_t> selected;
                for (std::size_t pair = 0; pair < pairCount; ++pair) {
                    if (nextRows[pair] >= rows) {
                        selected = pair;
                        break;
                    }
                }
                if (!selected.has_value()) {
                    function.emitError(
                        "RMSNorm gamma constants exceed the activation constant slice pairs");
                    signalPassFailure();
                    return;
                }
                const auto pair = *selected;
                nextRows[pair] -= rows;
                feedbackRmsWeightPlacements[rmsNorm.getWeight()] = {
                    nextRows[pair], constantSlices[2 * pair],
                    constantBank};
            }
        } else if (weight_bank_ >= 0
            && rmsnorm_strategy_
                == RmsNormLoweringStrategy::VxmFeedback) {
            int64_t totalRows = 0;
            for (kernel::RmsNormOp rmsNorm : rmsNorms)
                totalRows += rmsNorm.getWeight().getType().getNumElements();
            if (totalRows > target.memory().words_per_bank) {
                function.emitError(
                    "paged RMSNorm weights exceed one SRAM bank");
                signalPassFailure();
                return;
            }
            rmsWeightBase = target.memory().words_per_bank - totalRows;
        }
        for (mlir::BlockArgument argument : function.getArguments()) {
            if (argument.use_empty()) continue;
            const auto type =
                llvm::dyn_cast<mlir::RankedTensorType>(argument.getType());
            if (type && type.getRank() == 2
                && is_lpu_16bit_float(type.getElementType())
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
                    const int64_t inputBank =
                        target.uses_dedicated_slice_roles()
                            && weight_bank_ >= 0
                        ? (weight_bank_ + 1)
                            % target.memory().banks_per_slice
                        : 0;
                    const auto allocation = fixed_allocation(
                        PlacementKind::Activation,
                        target.mxm_distributed_activation_slices(),
                        target.uses_dedicated_slice_roles() ? 0 : 4096,
                        type.getNumElements() / 128, bytes,
                        "fp16_mxm_distributed_16", "both", inputBank);
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
                const bool distributed =
                    rmsnorm_strategy_
                    == RmsNormLoweringStrategy::VxmFeedback;
                // Feedback RMSNorm chooses weight slices from the actual
                // input placement when the op is lowered. Binding every
                // RMSNorm weight to one target-wide slice set here can
                // collide with the second norm's transpose scratch.
                const int64_t instructions = distributed
                    ? type.getNumElements()
                    : type.getNumElements()
                        / target.throughput().mxm_rows;
                const int64_t baseRow = distributed && weight_bank_ < 0
                    ? 7168 + rmsWeightBase : rmsWeightBase;
                if (distributed) {
                    if (!target.uses_dedicated_slice_roles()) {
                        feedbackRmsWeightPlacements[argument] = {
                            baseRow, -1};
                        rmsWeightBase += instructions;
                    } else if (!feedbackRmsWeightPlacements.contains(argument)) {
                        function.emitError(
                            "feedback RMSNorm gamma was not assigned to the activation constant area");
                        signalPassFailure();
                        return;
                    }
                    continue;
                }
                const auto allocation = fixed_allocation(
                    PlacementKind::Activation,
                    distributed
                        ? target.mxm_distributed_activation_slices()
                        : llvm::ArrayRef<int64_t>({20, 21}),
                    baseRow, instructions, type.getNumElements() * 2,
                    distributed ? "fp16_vxm_row_parallel_8"
                                : "fp16_pair_planar",
                    "both", std::max<int64_t>(0, weight_bank_));
                rmsWeightBase += distributed
                    ? instructions
                    : std::max<int64_t>(1,
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
                        attentions[found->second], target,
                        weight_bank_, rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (auto found = ffn_roots.find(operation);
                found != ffn_roots.end()) {
                if (mlir::failed(lower_ffn(ffns[found->second], target,
                        allocator, allocate_value, weight_bank_, rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (attention_operations.contains(operation)
                || ffn_operations.contains(operation))
                continue;
            if (auto op = llvm::dyn_cast<kernel::RmsNormOp>(operation)) {
                int64_t feedbackWeightBaseRow = 0;
                int64_t feedbackWeightSliceBase = -1;
                int64_t feedbackWeightBank =
                    std::max<int64_t>(0, weight_bank_);
                if (rmsnorm_strategy_
                    == RmsNormLoweringStrategy::VxmFeedback) {
                    auto found =
                        feedbackRmsWeightPlacements.find(op.getWeight());
                    if (found == feedbackRmsWeightPlacements.end()) {
                        op.emitError(
                            "feedback RMSNorm weight is not a "
                            "function argument");
                        signalPassFailure();
                        return;
                    }
                    feedbackWeightBaseRow = found->second.base_row;
                    feedbackWeightSliceBase = found->second.slice_base;
                    feedbackWeightBank = found->second.bank;
                }
                if (mlir::failed(lower_rms_norm(
                        op, target, rmsnorm_strategy_,
                        feedbackWeightBaseRow, feedbackWeightSliceBase,
                        feedbackWeightBank, weight_bank_, planner,
                        rewriter))) {
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (auto op = llvm::dyn_cast<kernel::ElementwiseOp>(operation)) {
                if (mlir::failed(lower_elementwise(
                        op, target, allocator, allocate_value,
                        rmsnorm_strategy_, weight_bank_, rewriter))) {
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
                        op, target, planner, allocate_value, rewriter))) {
                    signalPassFailure();
                    return;
                }
            }
        }
    }

private:
    RmsNormLoweringStrategy rmsnorm_strategy_ =
        RmsNormLoweringStrategy::VxmSquareMxmReduce;
    int64_t weight_bank_ = -1;
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_kernel_to_tensor_pass(
    RmsNormLoweringStrategy rmsnorm_strategy, std::int64_t weight_bank)
{
    return std::make_unique<LowerKernelToTensorPass>(
        rmsnorm_strategy, weight_bank);
}

} // namespace ftlpu::compiler
