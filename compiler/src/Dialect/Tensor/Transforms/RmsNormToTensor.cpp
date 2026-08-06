#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_rms_norm(kernel::RmsNormOp op,
    const target::LPUTargetModel& target,
    RmsNormLoweringStrategy strategy, int64_t feedbackWeightBaseRow,
    FunctionMemoryPlanner& planner,
    mlir::IRRewriter& rewriter)
{
    const auto inputType = op.getInput().getType();
    const auto weightType = op.getWeight().getType();
    const int64_t rows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const int64_t tile = target.throughput().mxm_rows;
    if (!is_lpu_16bit_float(inputType.getElementType())
        || weightType.getElementType()
            != inputType.getElementType()
        || op.getResult().getType().getElementType()
            != inputType.getElementType()
        || rows % tile != 0 || hidden % tile != 0) {
        op.emitError(
            "current RMSNorm strategy requires tile-aligned matching "
            "16-bit float tensors");
        return mlir::failure();
    }

    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    llvm::SmallVector<int64_t> inputSlices;
    llvm::SmallVector<int64_t> squareSlices;
    llvm::SmallVector<int64_t> resultSlices;
    const bool preservesInput = llvm::any_of(
        op.getInput().getUses(), [&](mlir::OpOperand& use) {
            return use.getOwner() != op.getOperation();
        });
    for (int64_t index = 0;
         index < throughput.mxm_activation_streams; ++index) {
        inputSlices.push_back(
            memory.w8a16_activation_slice_base + index);
        squareSlices.push_back(
            memory.accumulator_slice_base + index);
        resultSlices.push_back(preservesInput
                ? memory.slices_per_hemisphere
                    - throughput.mxm_activation_streams + index
                : memory.w8a16_activation_slice_base + index);
    }
    const int64_t matrixRows = rows * hidden / tile;
    const int64_t matrixBytes = rows * hidden * 2;
    const int64_t factorBytes = rows * tile * 2;
    auto existingInput = planner.lookup(op.getInput());
    const auto distributedInputSlices =
        target.mxm_distributed_activation_slices();
    if (distributedInputSlices.size() != 16) {
        op.emitError(
            "target does not provide 16 non-weight MXM activation slices");
        return mlir::failure();
    }
    const bool existingInputIsDistributed =
        mlir::succeeded(existingInput)
        && existingInput->layout == "fp16_mxm_distributed_16";
    const bool existingInputIsMxmPlanar =
        mlir::succeeded(existingInput)
        && existingInput->layout == "fp16_mxm_activation_planar"
        && existingInput->slices.size()
            >= static_cast<std::size_t>(
                throughput.mxm_activation_streams);
    const Allocation input =
        strategy == RmsNormLoweringStrategy::VxmFeedback
        ? existingInputIsDistributed
            ? *existingInput
            : fixed_allocation(PlacementKind::Activation,
                  distributedInputSlices, 4096,
                  rows * hidden / 256, matrixBytes,
                  "fp16_mxm_distributed_16", "both")
        : existingInputIsMxmPlanar
            ? *existingInput
            : fixed_allocation(PlacementKind::Activation,
                  inputSlices, 0, matrixRows, matrixBytes,
                  "fp16_mxm_activation_planar", "both");
    auto producerInput =
        get_value_task_allocations(op.getInput());
    llvm::SmallVector<int64_t, 16> effectiveInputSlices(
        input.slices.begin(), input.slices.end());
    if (mlir::succeeded(producerInput) && !producerInput->empty()) {
        const auto placement =
            llvm::cast<mlir::DictionaryAttr>((*producerInput)[0])
                .getAs<mlir::DictionaryAttr>("placement");
        effectiveInputSlices.clear();
        for (mlir::Attribute slice :
            placement.getAs<mlir::ArrayAttr>("slices"))
            effectiveInputSlices.push_back(
                llvm::cast<mlir::IntegerAttr>(slice).getInt());
    }
    const auto plannedWeight = planner.lookup(op.getWeight());
    const auto distributedWeightBindingSlices =
        effectiveInputSlices;
    const Allocation weight =
        strategy == RmsNormLoweringStrategy::VxmFeedback
        ? fixed_allocation(PlacementKind::Activation,
              distributedWeightBindingSlices,
              feedbackWeightBaseRow,
              hidden / throughput.lanes_per_tile, hidden * 2,
              "fp16_vxm_distributed_16", "both")
        : mlir::succeeded(plannedWeight)
            && plannedWeight->layout == "fp16_pair_planar"
        ? *plannedWeight
        : fixed_allocation(PlacementKind::Activation,
              {20, 21}, 0, hidden / tile, hidden * 2,
              "fp16_pair_planar", "both");
    llvm::SmallVector<Allocation> scratch;
    if (strategy == RmsNormLoweringStrategy::VxmFeedback) {
        llvm::SmallVector<int64_t, 16> distributedSlices;
        for (int64_t slice = 0;
             slice < memory.slices_per_hemisphere
             && distributedSlices.size() < 16;
             ++slice) {
            if (std::find(effectiveInputSlices.begin(),
                    effectiveInputSlices.end(), slice)
                == effectiveInputSlices.end())
                distributedSlices.push_back(slice);
        }
        if (distributedSlices.size() != 16) {
            op.emitError(
                "target does not provide 16 VXM scratch slices disjoint "
                "from MXM distributed activation");
            return mlir::failure();
        }
        const auto distributedWeightSlices = effectiveInputSlices;
        const auto canonicalResultSlices =
            target.mxm_distributed_activation_slices();
        llvm::SmallVector<int64_t, 16> normalizedSlices;
        for (int64_t slice = 0;
             slice < memory.slices_per_hemisphere
             && normalizedSlices.size() < 16;
             ++slice) {
            if (std::find(canonicalResultSlices.begin(),
                    canonicalResultSlices.end(), slice)
                == canonicalResultSlices.end())
                normalizedSlices.push_back(slice);
        }
        if (normalizedSlices.size() != 16) {
            op.emitError(
                "target does not provide 16 normalized scratch slices "
                "disjoint from canonical MXM output");
            return mlir::failure();
        }
        const int64_t packedRows = rows * hidden / 256;
        const int64_t packedWeightRows = hidden / 8;
        scratch.push_back(fixed_allocation(PlacementKind::VxmResult,
            distributedSlices, 4608, packedRows, matrixBytes,
            "fp16_vxm_distributed_16", "both"));
        scratch.push_back(fixed_allocation(PlacementKind::VxmResult1,
            distributedWeightSlices, weight.base_row, packedWeightRows,
            hidden * tile * 2,
            "fp16_vxm_distributed_16", "both"));
        scratch.push_back(fixed_allocation(PlacementKind::VxmResult1,
            normalizedSlices, 5632, packedRows, matrixBytes,
            "fp16_vxm_distributed_16", "both"));
    } else {
        scratch.push_back(fixed_allocation(PlacementKind::VxmResult,
            squareSlices, 0, matrixRows, matrixBytes,
            "fp16_mxm_activation_planar", "both"));
        scratch.push_back(fixed_allocation(PlacementKind::VxmResult1,
            {10, 11}, 0, rows, factorBytes,
            "fp16_pair_planar", "both"));
    }
    const auto distributedResultSlices =
        target.mxm_distributed_activation_slices();
    const Allocation result =
        strategy == RmsNormLoweringStrategy::VxmFeedback
        ? fixed_allocation(PlacementKind::FinalResult,
              distributedResultSlices, 5632,
              rows * hidden / 256, matrixBytes,
              "fp16_mxm_distributed_16", "both")
        : fixed_allocation(PlacementKind::FinalResult,
              resultSlices, 0, matrixRows, matrixBytes,
              "fp16_mxm_activation_planar", "both");
    if (mlir::failed(planner.bind(op.getInput(), input))
        || mlir::failed(planner.bind(op.getWeight(), weight))
        || mlir::failed(planner.bind(op.getResult(), result))) {
        op.emitError("RMSNorm physical layout conflicts with the global memory plan");
        return mlir::failure();
    }

    const auto inputPlacement = make_profile_placement(rewriter, input,
        input.layout, input.hemisphere);
    const auto weightPlacement = make_profile_placement(rewriter, weight,
        weight.layout, "both");
    const auto resultPlacement = make_profile_placement(rewriter, result,
        result.layout, "both");
    const auto inputAllocations = mlir::succeeded(producerInput)
        ? *producerInput
        : make_task_allocations(rewriter,
            {make_task_allocation(rewriter, input, inputPlacement)});
    const auto weightAllocations = make_task_allocations(rewriter,
        {make_task_allocation(rewriter, weight, weightPlacement)});
    llvm::SmallVector<mlir::Attribute> scratchAttrs;
    for (const Allocation& allocation : scratch) {
        const auto placement = make_profile_placement(rewriter, allocation,
            allocation.layout, allocation.hemisphere);
        scratchAttrs.push_back(
            make_task_allocation(rewriter, allocation, placement));
    }
    const auto scratchAllocations = rewriter.getArrayAttr(scratchAttrs);
    const auto resultAllocations = make_task_allocations(rewriter,
        {make_task_allocation(rewriter, result, resultPlacement)});

    rewriter.setInsertionPoint(op);
    mlir::OperationState state(
        op.getLoc(), tensor::RmsNormTaskOp::getOperationName());
    state.addOperands({op.getInput(), op.getWeight()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        rewriter.getNamedAttr("axis", op.getAxisAttr()),
        rewriter.getNamedAttr("epsilon", op.getEpsilonAttr()),
        rewriter.getNamedAttr("input_allocations", inputAllocations),
        rewriter.getNamedAttr("weight_allocations", weightAllocations),
        rewriter.getNamedAttr("scratch_allocations", scratchAllocations),
        rewriter.getNamedAttr("result_allocations", resultAllocations),
        rewriter.getNamedAttr("config", rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("strategy",
                rewriter.getStringAttr(
                    strategy == RmsNormLoweringStrategy::VxmFeedback
                        ? "vxm_feedback"
                        : "vxm_square_mxm_reduce")),
            rewriter.getNamedAttr("reduction_width",
                rewriter.getI64IntegerAttr(tile)),
        })),
    });
    const mlir::Value oldResult = op.getResult();
    auto lowered =
        llvm::cast<tensor::RmsNormTaskOp>(rewriter.create(state));
    planner.replace_value(oldResult, lowered.getResult());
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}

mlir::LogicalResult lower_elementwise(kernel::ElementwiseOp op,
    const target::LPUTargetModel& target,
    EastMemoryAllocator& allocator, AllocateValueFn allocateValue,
    RmsNormLoweringStrategy rmsnormStrategy,
    mlir::IRRewriter& rewriter)
{
    const auto allocationPlan =
        [&](mlir::Value value) -> mlir::FailureOr<mlir::ArrayAttr> {
        if (auto existing = get_value_task_allocations(value);
            mlir::succeeded(existing))
            return *existing;
        auto allocation =
            allocateValue(value, PlacementKind::Activation);
        if (mlir::failed(allocation)) return mlir::failure();
        return make_task_allocations(rewriter,
            {make_task_allocation(rewriter, *allocation,
                make_placement_attr(rewriter, *allocation))});
    };
    auto lhsPlan = allocationPlan(op.getLhs());
    auto rhsPlan = allocationPlan(op.getRhs());
    if (mlir::failed(lhsPlan) || mlir::failed(rhsPlan)) {
        op.emitError("cannot resolve elementwise operand layouts");
        return mlir::failure();
    }
    mlir::ArrayAttr resultPlan;
    const auto resultType =
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    if (op.getKind() == "add"
        && is_lpu_16bit_float(resultType.getElementType())
        && resultType.getRank() == 2
        && resultType.getDimSize(1) % 64 == 0) {
        const int64_t rows = resultType.getDimSize(0);
        const int64_t columns = resultType.getDimSize(1);
        bool feedsFeedbackRmsNorm = false;
        if (rmsnormStrategy == RmsNormLoweringStrategy::VxmFeedback) {
            for (mlir::OpOperand& use : op.getResult().getUses()) {
                auto rmsNorm =
                    llvm::dyn_cast<kernel::RmsNormOp>(use.getOwner());
                feedsFeedbackRmsNorm |= rmsNorm
                    && use.getOperandNumber() == 0;
            }
        }
        llvm::SmallVector<int64_t, 32> occupiedSlices;
        const auto collectSlices =
            [&](mlir::ArrayAttr plan) {
                for (mlir::Attribute allocation : plan) {
                    const auto placement =
                        llvm::cast<mlir::DictionaryAttr>(allocation)
                            .getAs<mlir::DictionaryAttr>("placement");
                    for (mlir::Attribute slice :
                        placement.getAs<mlir::ArrayAttr>("slices"))
                        occupiedSlices.push_back(
                            llvm::cast<mlir::IntegerAttr>(slice)
                                .getInt());
                }
            };
        collectSlices(*lhsPlan);
        collectSlices(*rhsPlan);
        for (int64_t index = 0;
             index < target.throughput().mxm_result_streams; ++index)
            occupiedSlices.push_back(
                target.memory().w8a16_result_slice_base + index);
        for (int64_t slice : target.attention_result_slices())
            occupiedSlices.push_back(slice);
        llvm::SmallVector<int64_t, 16> distributedSlices;
        for (int64_t slice = 0;
             slice < target.memory().slices_per_hemisphere
             && distributedSlices.size() < 16;
             ++slice) {
            if (std::find(occupiedSlices.begin(),
                    occupiedSlices.end(), slice)
                == occupiedSlices.end())
                distributedSlices.push_back(slice);
        }
        if (feedsFeedbackRmsNorm && distributedSlices.size() != 16)
            return op.emitError(
                "target does not provide 16 ping-pong residual slices");
        const bool feedsRmsNorm = llvm::any_of(
            op.getResult().getUses(), [](mlir::OpOperand& use) {
                return llvm::isa<kernel::RmsNormOp>(use.getOwner())
                    && use.getOperandNumber() == 0;
            });
        const bool isFunctionResult = llvm::any_of(
            op.getResult().getUses(), [](mlir::OpOperand& use) {
                return llvm::isa<mlir::func::ReturnOp>(use.getOwner());
            });
        const bool usesPersistentActivationAbi =
            rmsnormStrategy == RmsNormLoweringStrategy::VxmFeedback
            && isFunctionResult;
        const Allocation result = usesPersistentActivationAbi
            ? fixed_allocation(PlacementKind::FinalResult,
                target.mxm_distributed_activation_slices(),
                4096, rows * columns / 128,
                rows * columns * 2,
                "fp16_mxm_distributed_16", "both")
            : feedsFeedbackRmsNorm
            ? fixed_allocation(PlacementKind::FinalResult,
                distributedSlices, 4096, rows * columns / 256,
                rows * columns * 2,
                "fp16_mxm_distributed_16", "both")
            : feedsRmsNorm
            ? fixed_allocation(PlacementKind::FinalResult,
                {16, 17, 18, 19}, 0, rows * columns / 32,
                rows * columns * 2,
                "fp16_mxm_activation_planar", "both")
            : fixed_allocation(PlacementKind::FinalResult,
                {32, 33, 34, 35}, 0, rows * columns / 64,
                rows * columns * 2, "fp16_pair_planar", "both");
        resultPlan = make_task_allocations(rewriter,
            {make_task_allocation(rewriter, result,
                make_placement_attr(rewriter, result))});
    } else {
        auto resultBytes =
            get_static_tensor_bytes(op.getResult().getType());
        auto result = mlir::succeeded(resultBytes)
            ? allocator.allocate(PlacementKind::FinalResult, *resultBytes)
            : mlir::FailureOr<Allocation>(mlir::failure());
        if (mlir::failed(result)) {
            op.emitError("cannot allocate elementwise result");
            return mlir::failure();
        }
        resultPlan = make_task_allocations(rewriter,
            {make_task_allocation(rewriter, *result,
                make_placement_attr(rewriter, *result))});
    }
    rewriter.setInsertionPoint(op);
    mlir::OperationState state(
        op.getLoc(), tensor::ElementwiseTaskOp::getOperationName());
    state.addOperands({op.getLhs(), op.getRhs()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        rewriter.getNamedAttr("kind", op.getKindAttr()),
        rewriter.getNamedAttr("lhs_allocations", *lhsPlan),
        rewriter.getNamedAttr("rhs_allocations", *rhsPlan),
        rewriter.getNamedAttr("result_allocations", resultPlan),
        rewriter.getNamedAttr(
            "config", rewriter.getDictionaryAttr({})),
    });
    auto lowered =
        llvm::cast<tensor::ElementwiseTaskOp>(rewriter.create(state));
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
