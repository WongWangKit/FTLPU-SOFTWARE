#include "KernelToTensorLowering.hpp"

namespace ftlpu::compiler::tensor_lowering {
namespace {

mlir::FailureOr<mlir::ArrayAttr> planValue(mlir::Value value,
    PlacementKind kind, llvm::ArrayRef<int64_t> slices,
    llvm::StringRef layout, llvm::StringRef hemisphere,
    int64_t instructionCount, AllocateValueFn allocateValue,
    mlir::IRRewriter& rewriter)
{
    if (auto existing = get_value_task_allocations(value);
        mlir::succeeded(existing))
        return *existing;
    static_cast<void>(allocateValue);
    const auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
    auto allocation = fixed_allocation(kind, slices, 0, instructionCount,
        type.getNumElements() * (type.getElementType().isF32() ? 4 : 2),
        layout, hemisphere);
    return make_task_allocations(rewriter,
        {make_task_allocation(rewriter, allocation,
            make_placement_attr(rewriter, allocation))});
}

int64_t planarRows(mlir::RankedTensorType type)
{
    return type.getDimSize(0) * ((type.getDimSize(1) + 31) / 32);
}

} // namespace

mlir::LogicalResult lower_gated_delta_rule(kernel::GatedDeltaRuleOp op,
    EastMemoryAllocator& allocator, AllocateValueFn allocateValue,
    mlir::IRRewriter& rewriter)
{
    const auto queryType = op.getQuery().getType();
    const auto valueType = op.getValue().getType();
    const auto scalarType = op.getDecay().getType();
    const auto stateType = op.getInitialState().getType();
    auto query = planValue(op.getQuery(), PlacementKind::Activation,
        {0, 1}, "fp16_scalar_broadcast_planar", "east",
        queryType.getNumElements(),
        allocateValue, rewriter);
    auto key = planValue(op.getKey(), PlacementKind::Activation,
        {2, 3}, "fp16_scalar_broadcast_planar", "east",
        queryType.getNumElements(),
        allocateValue, rewriter);
    auto value = planValue(op.getValue(), PlacementKind::Activation,
        {4, 5}, "fp16_pair_planar", "east", planarRows(valueType),
        allocateValue, rewriter);
    auto decay = planValue(op.getDecay(), PlacementKind::Activation,
        {6, 7, 8, 9}, "fp32_scalar_broadcast_planar", "east",
        scalarType.getNumElements(), allocateValue, rewriter);
    auto beta = planValue(op.getBeta(), PlacementKind::Activation,
        {10, 11, 12, 13}, "fp32_scalar_broadcast_planar", "east",
        scalarType.getNumElements(), allocateValue, rewriter);
    auto state = planValue(op.getInitialState(), PlacementKind::Activation,
        {16, 17, 18, 19}, "fp32_quad_planar", "east",
        planarRows(stateType), allocateValue, rewriter);
    auto output = planValue(op.getOutput(), PlacementKind::FinalResult,
        {42, 43}, "fp16_pair_planar", "east", planarRows(valueType),
        allocateValue, rewriter);
    auto finalState = planValue(op.getFinalState(),
        PlacementKind::FinalResult, {44, 45, 46, 47},
        "fp32_quad_planar", "east", planarRows(stateType),
        allocateValue, rewriter);
    if (mlir::failed(query) || mlir::failed(key) || mlir::failed(value)
        || mlir::failed(decay) || mlir::failed(beta)
        || mlir::failed(state) || mlir::failed(output)
        || mlir::failed(finalState))
        return op.emitError("cannot allocate gated delta-rule buffers");

    const auto scratchPlan = [&](llvm::ArrayRef<int64_t> slices,
                                 llvm::StringRef layout,
                                 llvm::StringRef hemisphere,
                                 int64_t bytes, int64_t rows,
                                 PlacementKind kind)
        -> mlir::FailureOr<mlir::DictionaryAttr> {
        // These buffers occupy disjoint physical slices. Their actual row
        // bounds are checked by Schedule verification, rather than by the
        // legacy byte-based allocator shared with unrelated layouts.
        const auto allocation = fixed_allocation(kind, slices, 0, rows,
            bytes, layout, hemisphere);
        return make_task_allocation(rewriter, allocation,
            make_placement_attr(rewriter, allocation));
    };
    const int64_t stateBytes = stateType.getNumElements() * 4;
    const int64_t vectorBytes = op.getHeads() * op.getValueDim() * 4;
    auto pingPong = scratchPlan({20, 21, 22, 23},
        "fp32_quad_planar", "east", stateBytes,
        planarRows(stateType), PlacementKind::Result);
    auto productState = scratchPlan({24, 25, 26, 27},
        "fp32_quad_planar", "east", stateBytes,
        planarRows(stateType), PlacementKind::Result);
    auto remembered = scratchPlan({28, 29, 30, 31},
        "fp32_quad_planar", "east", vectorBytes,
        op.getHeads() * ((op.getValueDim() + 31) / 32),
        PlacementKind::VxmResult);
    auto delta = scratchPlan({34, 35, 36, 37},
        "fp32_quad_planar", "east", vectorBytes,
        op.getHeads() * ((op.getValueDim() + 31) / 32),
        PlacementKind::VxmResult1);
    if (mlir::failed(pingPong) || mlir::failed(productState)
        || mlir::failed(remembered)
        || mlir::failed(delta))
        return op.emitError("cannot allocate gated delta-rule scratch");

    rewriter.setInsertionPoint(op);
    mlir::OperationState task(
        op.getLoc(), tensor::GatedDeltaRuleTaskOp::getOperationName());
    task.addOperands({op.getQuery(), op.getKey(), op.getValue(),
        op.getDecay(), op.getBeta(), op.getInitialState()});
    task.addTypes({op.getOutput().getType(),
        op.getFinalState().getType()});
    task.addAttributes({
        rewriter.getNamedAttr("heads", op.getHeadsAttr()),
        rewriter.getNamedAttr("key_dim", op.getKeyDimAttr()),
        rewriter.getNamedAttr("value_dim", op.getValueDimAttr()),
        rewriter.getNamedAttr("query_allocations", *query),
        rewriter.getNamedAttr("key_allocations", *key),
        rewriter.getNamedAttr("value_allocations", *value),
        rewriter.getNamedAttr("decay_allocations", *decay),
        rewriter.getNamedAttr("beta_allocations", *beta),
        rewriter.getNamedAttr("state_allocations", *state),
        rewriter.getNamedAttr("scratch_allocations",
            rewriter.getArrayAttr(
                {*pingPong, *productState, *remembered, *delta})),
        rewriter.getNamedAttr("output_allocations", *output),
        rewriter.getNamedAttr("final_state_allocations", *finalState),
        rewriter.getNamedAttr("config", rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("lowering",
                rewriter.getStringAttr("direct_recurrent_scan")),
            rewriter.getNamedAttr("state_precision",
                rewriter.getStringAttr("fp32")),
        })),
    });
    auto lowered = llvm::cast<tensor::GatedDeltaRuleTaskOp>(
        rewriter.create(task));
    rewriter.replaceOp(op,
        {lowered.getOutput(), lowered.getFinalState()});
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
