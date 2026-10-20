#include "KernelToTensorLowering.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_l2_norm(kernel::L2NormOp op,
    const target::LPUTargetModel& target, FunctionMemoryPlanner& planner,
    mlir::IRRewriter& rewriter)
{
    const auto type = op.getInput().getType();
    const int64_t rows = type.getDimSize(0), width = type.getDimSize(1);
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t physicalRows = rows;
    if (!is_lpu_16bit_float(type.getElementType()) || physicalRows % tile
        || width % tile || tile != 32 || lanes != 8)
        return op.emitError("L2Norm feedback requires tile-aligned 16-bit tensors and 32x8 geometry");
    const auto slices = target.mxm_distributed_activation_slices();
    if (slices.size() != 16 || target.memory().banks_per_slice < 2)
        return op.emitError("L2Norm feedback requires 16 activation slices and two banks");
    const int64_t dataRows = physicalRows * width / (tile * lanes);
    // The square-sum scalar lives after the transposed feedback input.
    const int64_t feedbackRows = dataRows + physicalRows / tile;
    if (3 * dataRows > target.memory().words_per_bank
        || feedbackRows > target.memory().words_per_bank)
        return op.emitError("L2Norm distributed input and reduction scalars exceed a bank");
    const auto input = fixed_allocation(PlacementKind::Activation, slices,
        0, dataRows, rows * width * 2,
        "fp16_mxm_distributed_16", "both", 0);
    const auto feedbackInput = fixed_allocation(PlacementKind::VxmResult, slices,
        0, feedbackRows, (rows * width + rows) * 2,
        "fp16_vxm_distributed_16", "both", 1);
    const auto feedbackOutput = fixed_allocation(PlacementKind::VxmResult1, slices,
        dataRows, dataRows, rows * width * 2,
        "fp16_vxm_distributed_16", "both", 0);
    const auto result = fixed_allocation(PlacementKind::FinalResult, slices,
        2 * dataRows, dataRows, rows * width * 2,
        "fp16_mxm_distributed_16", "both", 1);
    if (mlir::failed(planner.bind(op.getInput(), input))
        || mlir::failed(planner.bind(op.getResult(), result)))
        return op.emitError("L2Norm physical layout conflicts with the memory plan");
    const auto allocations = [&](const Allocation& a) {
        return make_task_allocations(rewriter,
            {make_task_allocation(rewriter, a, make_placement_attr(rewriter, a))});
    };
    rewriter.setInsertionPoint(op);
    mlir::OperationState state(op.getLoc(), tensor::L2NormTaskOp::getOperationName());
    state.addOperands(op.getInput());
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        rewriter.getNamedAttr("axis", op.getAxisAttr()),
        rewriter.getNamedAttr("epsilon", op.getEpsilonAttr()),
        rewriter.getNamedAttr("scale", op.getScaleAttr()),
        rewriter.getNamedAttr("input_allocations", allocations(input)),
        rewriter.getNamedAttr("scratch_allocations", make_task_allocations(rewriter, {
            make_task_allocation(rewriter, feedbackInput, make_placement_attr(rewriter, feedbackInput)),
            make_task_allocation(rewriter, feedbackOutput, make_placement_attr(rewriter, feedbackOutput))})),
        rewriter.getNamedAttr("result_allocations", allocations(result)),
    });
    auto lowered = llvm::cast<tensor::L2NormTaskOp>(rewriter.create(state));
    planner.replace_value(op.getResult(), lowered.getResult());
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}
} // namespace ftlpu::compiler::tensor_lowering
