#include "ftlpu/compiler/Dialect/Kernel/Analysis/decoder_graph.hpp"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

namespace ftlpu::compiler::kernel {
DecoderGraph match_decoder_graph(mlir::func::FuncOp function) {
    DecoderGraph result;
    llvm::SmallPtrSet<mlir::Operation*, 32> claimed;
    llvm::SmallVector<MatmulOp> matmuls;
    function.walk([&](MatmulOp op) { matmuls.push_back(op); });
    for (auto root : llvm::reverse(matmuls)) {
        if (claimed.contains(root)) continue;
        if (auto graph = match_attention_graph(root)) {
            claimed.insert(graph->operations.begin(), graph->operations.end());
            result.attentions.push_back(std::move(*graph));
        }
    }
    for (auto root : llvm::reverse(matmuls)) {
        if (claimed.contains(root)) continue;
        if (auto graph = match_ffn_graph(root)) {
            // An externally used intermediate must stay visible to the planner.
            bool closed = true;
            llvm::SmallPtrSet<mlir::Operation*, 8> members;
            members.insert(graph->operations.begin(), graph->operations.end());
            for (auto *op : graph->operations)
                if (op != root.getOperation())
                    for (auto *user : op->getUsers()) closed &= members.contains(user);
            if (!closed) continue;
            claimed.insert(graph->operations.begin(), graph->operations.end());
            result.ffns.push_back(std::move(*graph));
        }
    }
    function.walk([&](GatedDeltaRuleOp recurrence) {
        GatedDeltaNetGraph graph;
        graph.recurrence = recurrence;
        llvm::SmallPtrSet<mlir::Operation*, 32> members;
        // Follow the mathematical preprocessing back to projections. RMSNorm
        // and residual/FFN stay outside this mixer and use the common lowerers.
        auto visit = [&](auto&& self, mlir::Value value) -> void {
            auto *op = value.getDefiningOp();
            if (!op || claimed.contains(op) || llvm::isa<RmsNormOp>(op)) return;
            if (llvm::isa<GatedDeltaRuleOp>(op) && op != recurrence.getOperation()) return;
            if (!members.insert(op).second) return;
            if (llvm::isa<MatmulOp>(op)) return;
            for (auto input : op->getOperands()) self(self, input);
        };
        members.insert(recurrence);
        for (auto value : recurrence->getOperands()) visit(visit, value);
        // Include post-scan normalization/gating and the output projection,
        // but stop before the surrounding residual and dense FFN.
        auto forward = [&](auto&& self, mlir::Value value) -> void {
            for (auto *user : value.getUsers()) {
                if (claimed.contains(user) || !llvm::isa<VectorAluOp,
                        IndexedViewOp, ReshapeOp, MatmulOp>(user)) continue;
                const bool fresh = !members.contains(user);
                for (auto result : user->getResults()) visit(visit,result);
                if (fresh && !llvm::isa<MatmulOp>(user))
                    for (auto result : user->getResults()) self(self,result);
            }
        };
        forward(forward,recurrence.getOutput());
        for (auto &op : function.front()) {
            if (!members.contains(&op)) continue;
            graph.operations.push_back(&op);
            for (auto operand : op.getOperands())
                if (!members.contains(operand.getDefiningOp()) &&
                    !llvm::is_contained(graph.inputs, operand)) graph.inputs.push_back(operand);
            for (auto value : op.getResults())
                if (llvm::any_of(value.getUsers(), [&](auto *user) { return !members.contains(user); }))
                    graph.outputs.push_back(value);
        }
        result.delta_nets.push_back(std::move(graph));
    });
    return result;
}
} // namespace ftlpu::compiler::kernel
