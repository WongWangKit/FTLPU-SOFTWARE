#include "KernelToTensorLowering.hpp"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_depthwise_conv1d(kernel::DepthwiseConv1DOp op,
    EastMemoryAllocator& allocator, AllocateValueFn allocateValue,
    mlir::IRRewriter& rewriter)
{
    const auto customPlan = [&](mlir::Value value, PlacementKind kind,
                                llvm::ArrayRef<int64_t> slices,
                                llvm::StringRef layout,
                                int64_t instructionCount,
                                llvm::StringRef hemisphere)
        -> mlir::FailureOr<mlir::ArrayAttr> {
        if (auto existing = get_value_task_allocations(value);
            mlir::succeeded(existing))
            return *existing;
        auto allocation = allocateValue(value, kind);
        if (mlir::failed(allocation)) return mlir::failure();
        allocation->slices.assign(slices.begin(), slices.end());
        allocation->layout = layout.str();
        allocation->hemisphere = hemisphere.str();
        allocation->instruction_count = instructionCount;
        allocation->address_stride = 1;
        return make_task_allocations(rewriter,
            {make_task_allocation(rewriter, *allocation,
                make_placement_attr(rewriter, *allocation))});
    };
    const auto resultType = llvm::cast<mlir::RankedTensorType>(
        op.getResult().getType());
    const int64_t tokens = resultType.getDimSize(0);
    const int64_t channels = resultType.getDimSize(1);
    const int64_t featureBlocks = (channels + 31) / 32;
    auto input = customPlan(op.getInput(), PlacementKind::Activation,
        {32, 33}, "fp16_pair_planar", tokens * featureBlocks, "east");
    auto weight = customPlan(op.getWeight(), PlacementKind::Weight,
        {0, 1, 2, 3, 4, 5, 6, 7}, "fp16_depthwise_kernel4",
        op.getKernelSize() * featureBlocks, "west");
    const auto function = op->getParentOfType<mlir::func::FuncOp>();
    const auto distributed = function->getAttrOfType<mlir::BoolAttr>(
        "ftlpu.depthwise_conv_distributed_output");
    const bool distributedOutput = distributed && distributed.getValue();
    llvm::SmallVector<int64_t> resultSlices;
    if (distributedOutput) {
        for (int64_t slice = 8; slice < 24; ++slice)
            resultSlices.push_back(slice);
    } else {
        resultSlices = {48, 49};
    }
    auto result = customPlan(op.getResult(), PlacementKind::FinalResult,
        resultSlices,
        distributedOutput ? "fp16_mxm_distributed_16" : "fp16_pair_planar",
        distributedOutput ? tokens * featureBlocks / 8
                          : tokens * featureBlocks,
        "west");
    auto bytes = get_static_tensor_bytes(op.getResult().getType());
    if (mlir::failed(input) || mlir::failed(weight)
        || mlir::failed(result) || mlir::failed(bytes))
        return op.emitError("cannot allocate depthwise Conv1D buffers");

    const int64_t paddedBytes =
        (resultType.getDimSize(0) + op.getKernelSize() - 1)
        * resultType.getDimSize(1) * 2;
    llvm::SmallVector<mlir::Attribute, 4> scratch;
    for (int64_t index = 0; index < 3; ++index) {
        auto allocation = allocator.allocate(
            index == 0 ? PlacementKind::Activation
                       : index == 1 ? PlacementKind::VxmResult
                       : index == 2 ? PlacementKind::VxmResult1
                                    : PlacementKind::Result,
            index == 0 ? paddedBytes : *bytes);
        if (mlir::failed(allocation))
            return op.emitError("cannot allocate depthwise Conv1D scratch");
        if (index == 0) {
            // Four independent read ports feed the four tap multipliers.
            // Replicate the padded input via non-consuming MEM write taps.
            allocation->slices = {34, 35, 36, 37, 38, 39, 40, 41};
            allocation->layout = "bf16_causal_padded";
            allocation->bytes = 4 * paddedBytes;
            allocation->instruction_count =
                (tokens + op.getKernelSize() - 1) * featureBlocks;
            allocation->address_stride = 1;
            allocation->hemisphere = "west";
        } else {
            const int64_t first = 42 + 2 * (index - 1);
            allocation->slices = {first, first + 1};
            allocation->layout = "fp16_pair_planar";
            allocation->instruction_count = tokens * featureBlocks;
            allocation->address_stride = 1;
            // The fixed chain drains across hemispheres. Expose that scratch
            // home to the shared planner instead of changing it in emission.
            allocation->hemisphere = index == 1 && resultType.getElementType().isBF16()
                ? "east" : "west";
        }
        scratch.push_back(make_task_allocation(rewriter, *allocation,
            make_placement_attr(rewriter, *allocation)));
    }

    rewriter.setInsertionPoint(op);
    mlir::OperationState state(
        op.getLoc(), tensor::DepthwiseConv1DTaskOp::getOperationName());
    state.addOperands({op.getInput(), op.getWeight()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        rewriter.getNamedAttr("kernel_size", op.getKernelSizeAttr()),
        rewriter.getNamedAttr("causal", op.getCausalAttr()),
        rewriter.getNamedAttr("activation", op.getActivationAttr()),
        rewriter.getNamedAttr("input_allocations", *input),
        rewriter.getNamedAttr("weight_allocations", *weight),
        rewriter.getNamedAttr(
            "scratch_allocations", rewriter.getArrayAttr(scratch)),
        rewriter.getNamedAttr("result_allocations", *result),
        rewriter.getNamedAttr("config", rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("lowering",
                rewriter.getStringAttr("direct_closed_form")),
        })),
    });
    auto lowered = llvm::cast<tensor::DepthwiseConv1DTaskOp>(
        rewriter.create(state));
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
