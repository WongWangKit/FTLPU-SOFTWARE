#include "KernelToTensorLowering.hpp"

#include "llvm/ADT/DenseSet.h"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_matmul(kernel::MatmulOp op,
    EastMemoryAllocator& allocator,
    llvm::DenseMap<mlir::Value, Allocation>& allocations,
    llvm::DenseMap<mlir::Value, int64_t>& last_uses,
    const llvm::DenseMap<mlir::Operation*, int64_t>& ordinals,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter)
{
    const int64_t current_ordinal = ordinals.lookup(op.getOperation());
    llvm::SmallVector<mlir::Value> expired;
    for (const auto& [value, allocation] : allocations) {
        const auto use = last_uses.find(value);
        if (use == last_uses.end() || use->second < current_ordinal) expired.push_back(value);
    }
    for (mlir::Value value : expired) {
        allocator.release(allocations.lookup(value));
        allocations.erase(value);
    }

    const auto lhs = allocate_value(op.getLhs(), PlacementKind::Activation);
    const auto rhs = allocate_value(op.getRhs(), PlacementKind::Weight);
    const auto result_bytes = get_static_tensor_bytes(op.getResult().getType());
    const auto result = mlir::succeeded(result_bytes) ? allocator.allocate(PlacementKind::Result, *result_bytes)
                                                      : mlir::FailureOr<Allocation>(mlir::failure());
    if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(result)) {
        op.emitError("cannot allocate static tensor storage in the east MEM hemisphere");
        return mlir::failure();
    }

    const mlir::Value old_result = op.getResult();
    rewriter.setInsertionPoint(op);
    auto lowered = rewriter.create<tensor::MatmulOp>(op.getLoc(), op.getLhs(), op.getRhs(),
        op.getResult().getType(), op.getM(), op.getN(), op.getK(), op.getUnitAttr(),
        make_address_attr(rewriter, *lhs), make_placement_attr(rewriter, *lhs),
        make_address_attr(rewriter, *rhs), make_placement_attr(rewriter, *rhs),
        make_address_attr(rewriter, *result), make_placement_attr(rewriter, *result),
        lhs->bytes, rhs->bytes, result->bytes);
    allocations.try_emplace(lowered.getResult(), *result);
    if (const auto use = last_uses.find(old_result); use != last_uses.end())
        last_uses[lowered.getResult()] = use->second;
    rewriter.replaceOp(op, lowered.getResult());

    llvm::SmallDenseSet<mlir::Value> candidates;
    candidates.insert(lowered.getLhs());
    candidates.insert(lowered.getRhs());
    candidates.insert(lowered.getResult());
    for (mlir::Value value : candidates) {
        const auto use = last_uses.find(value);
        if (use != last_uses.end() && use->second > current_ordinal) continue;
        const auto allocation = allocations.find(value);
        if (allocation == allocations.end()) continue;
        allocator.release(allocation->second);
        allocations.erase(allocation);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
