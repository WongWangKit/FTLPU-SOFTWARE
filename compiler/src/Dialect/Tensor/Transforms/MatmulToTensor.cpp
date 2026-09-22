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
        const bool decode = op.getM() == 1;
        if ((!decode && op.getM() % tile) || op.getN() % (2 * tile)
            || op.getK() % tile) {
            op.emitError(
                "W8A16 linear projection requires either M=1 decode or "
                "tile-aligned M/K, with N aligned to two MXM tiles");
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
        const auto activationSlices = target.attention_activation_slices();
        auto weightSlices = decode
            ? target.weight_storage_slices()
            : target.attention_output_weight_slices();
        if (decode && weightSlices.size() > 32)
            weightSlices.resize(32);
        const auto resultSlices = target.attention_result_slices();
        const std::size_t expectedActivationSlices = 2;
        if (activationSlices.size() < expectedActivationSlices
            || weightSlices.size() != (decode ? 32u : 8u)
            || (decode ? resultSlices.size() < 2
                       : resultSlices.size() != 4)) {
            op.emitError(
                "target does not provide the W8A16 linear projection "
                "memory lanes");
            return mlir::failure();
        }
        const int64_t lhsBytes = lhsType.getNumElements() * 2;
        const int64_t rhsBytes = rhsType.getNumElements();
        const int64_t resultBytes = resultType.getNumElements() * 2;
        const auto input = fixed_allocation(PlacementKind::Activation,
            llvm::ArrayRef<int64_t>(activationSlices).take_front(2),
            0, op.getM() * op.getK() / tile, lhsBytes,
            "fp16_pair_planar", "both");
        const auto weight = fixed_allocation(PlacementKind::Weight,
            weightSlices, 0,
            decode
                ? (op.getN() / (2 * tile)) * (op.getK() / tile)
                : (op.getN() / (2 * tile)) * (op.getK() / tile) * 4,
            rhsBytes,
            decode ? "w8a16_mxm_decode_streamed"
                   : "w8a16_mxm_weight_striped",
            "both");
        const auto result = fixed_allocation(PlacementKind::FinalResult,
            decode
                ? llvm::ArrayRef<int64_t>(resultSlices).take_front(2)
                : llvm::ArrayRef<int64_t>(resultSlices), 0,
            op.getM() * op.getN() / (2 * tile),
            resultBytes,
            decode ? "fp16_mxm_decode_pair_planar" : "fp16_pair_planar",
            decode ? "both" : "east");
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
