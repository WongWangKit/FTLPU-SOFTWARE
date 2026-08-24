#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

namespace ftlpu::compiler::tensor_lowering {

mlir::LogicalResult lower_matmul(kernel::MatmulOp op,
    const target::LPUTargetModel& target,
    FunctionMemoryPlanner& planner,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter)
{
    const auto lhsType = op.getLhs().getType();
    const auto rhsType = op.getRhs().getType();
    const auto resultType = op.getResult().getType();
    const int64_t tile = target.throughput().mxm_rows;
    const bool w8a16 = is_lpu_16bit_float(lhsType.getElementType())
        && rhsType.getElementType().isInteger(8)
        && is_lpu_16bit_float(resultType.getElementType());
    if (w8a16) {
        if (op.getM() % tile || op.getN() % (2 * tile)
            || op.getK() % tile) {
            op.emitError(
                "W8A16 linear projection requires M/K aligned to the "
                "MXM tile and N aligned to two MXM tiles");
            return mlir::failure();
        }
        auto executionPolicy =
            target::mxm_execution_policy_from_operation(op.getOperation());
        if (mlir::failed(executionPolicy)) {
            op.emitError("invalid MXM execution policy");
            return mlir::failure();
        }
        auto strategy = target::plan_mxm_execution_strategy(
            {static_cast<int64_t>(op.getM()),
                static_cast<int64_t>(op.getN()),
                static_cast<int64_t>(op.getK()),
                lhsType.getElementType().isBF16(),
                rhsType.getElementType().isInteger(8),
                is_lpu_16bit_float(resultType.getElementType()), true},
            target, *executionPolicy);
        if (mlir::failed(strategy)) {
            op.emitError("cannot select an MXM execution strategy");
            return mlir::failure();
        }
        const auto activationSlices = strategy->uses_block8()
            ? target.mxm_distributed_activation_slices()
            : target.attention_activation_slices();
        const auto weightSlices =
            target.attention_output_weight_slices();
        const auto resultSlices = strategy->uses_block8()
            ? target.mxm_distributed_activation_slices()
            : target.attention_result_slices();
        const std::size_t expectedActivationSlices =
            strategy->uses_block8() ? 16 : 2;
        const std::size_t expectedResultSlices =
            strategy->uses_block8() ? 16 : 4;
        if (activationSlices.size() < expectedActivationSlices
            || weightSlices.size() != 8
            || resultSlices.size() != expectedResultSlices) {
            op.emitError(
                "target does not provide the W8A16 linear projection "
                "memory lanes");
            return mlir::failure();
        }
        const int64_t lhsBytes = lhsType.getNumElements() * 2;
        const int64_t rhsBytes = rhsType.getNumElements();
        const int64_t resultBytes = resultType.getNumElements() * 2;
        const int64_t distributedElementsPerRow =
            target.throughput().mxm_rows
            * target.throughput().mxm_block_rows;
        const auto input = strategy->uses_block8()
            ? fixed_allocation(PlacementKind::Activation,
                  activationSlices, 0,
                  op.getM() * op.getK()
                      / distributedElementsPerRow,
                  lhsBytes, "fp16_mxm_distributed_16", "both")
            : fixed_allocation(PlacementKind::Activation,
                  llvm::ArrayRef<int64_t>(activationSlices).take_front(2),
                  0, op.getM() * op.getK() / tile, lhsBytes,
                  "fp16_pair_planar", "both");
        const auto weight = fixed_allocation(PlacementKind::Weight,
            weightSlices, 0,
            (op.getN() / (2 * tile)) * (op.getK() / tile) * 4,
            rhsBytes, "w8a16_mxm_weight_striped", "both");
        const auto result = strategy->uses_block8()
            ? fixed_allocation(PlacementKind::FinalResult,
                  resultSlices,
                  target.memory().matmul_result_base_row,
                  op.getM() * op.getN()
                      / distributedElementsPerRow,
                  resultBytes, "fp16_mxm_block8_distributed_16",
                  "both")
            : fixed_allocation(PlacementKind::FinalResult,
                  resultSlices, 0,
                  op.getM() * op.getN() / (2 * tile),
                  resultBytes, "fp16_pair_planar", "east");
        const auto memoryPlan = rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "input", make_placement_attr(rewriter, input)),
            rewriter.getNamedAttr(
                "weight", make_placement_attr(rewriter, weight)),
            rewriter.getNamedAttr(
                "result", make_placement_attr(rewriter, result)),
        });
        const auto config = rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "m", rewriter.getI64IntegerAttr(op.getM())),
            rewriter.getNamedAttr(
                "n", rewriter.getI64IntegerAttr(op.getN())),
            rewriter.getNamedAttr(
                "k", rewriter.getI64IntegerAttr(op.getK())),
            rewriter.getNamedAttr(
                "rhs_scale", op.getRhsScaleAttr()),
        });
        const mlir::Value oldResult = op.getResult();
        rewriter.setInsertionPoint(op);
        mlir::OperationState state(
            op.getLoc(), tensor::ProjectionTaskOp::getOperationName());
        state.addOperands({op.getLhs(), op.getRhs()});
        state.addTypes(resultType);
        state.addAttributes({
            rewriter.getNamedAttr(
                "kind", rewriter.getStringAttr("linear")),
            rewriter.getNamedAttr("config", config),
            rewriter.getNamedAttr("memory_plan", memoryPlan),
        });
        auto lowered = llvm::cast<tensor::ProjectionTaskOp>(
            rewriter.create(state));
        planner.replace_value(oldResult, lowered.getResult());
        rewriter.replaceOp(op, lowered.getResult());
        return mlir::success();
    }

    const auto lhs = allocate_value(op.getLhs(), PlacementKind::Activation);
    const auto rhs = allocate_value(op.getRhs(), PlacementKind::Weight);
    const auto result =
        planner.allocate(op.getResult(), PlacementKind::Result);
    if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(result)) {
        op.emitError("cannot allocate static tensor storage in the east MEM hemisphere");
        return mlir::failure();
    }

    const mlir::Value old_result = op.getResult();
    rewriter.setInsertionPoint(op);
    auto lowered = rewriter.create<tensor::MatmulOp>(op.getLoc(), op.getLhs(), op.getRhs(),
        op.getResult().getType(), op.getM(), op.getN(), op.getK(), op.getUnitAttr(),
        op.getRhsScaleAttr(),
        make_address_attr(rewriter, *lhs), make_placement_attr(rewriter, *lhs),
        make_address_attr(rewriter, *rhs), make_placement_attr(rewriter, *rhs),
        make_address_attr(rewriter, *result), make_placement_attr(rewriter, *result),
        lhs->bytes, rhs->bytes, result->bytes);
    planner.replace_value(old_result, lowered.getResult());
    rewriter.replaceOp(op, lowered.getResult());
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
