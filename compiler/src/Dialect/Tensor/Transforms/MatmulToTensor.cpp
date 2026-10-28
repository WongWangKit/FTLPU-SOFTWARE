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
        if ((op.getM() > tile && op.getM() % tile) || op.getN() % (2 * tile)
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
        const auto activationSlices = target.attention_activation_slices();
        const auto weightSlices =
            target.attention_output_weight_slices();
        const auto resultSlices = target.attention_result_slices();
        const std::size_t expectedActivationSlices = 2;
        const std::size_t expectedResultSlices = 4;
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
        auto input = fixed_allocation(PlacementKind::Activation,
            llvm::ArrayRef<int64_t>(activationSlices).take_front(2),
            0, op.getM() * op.getK() / tile, lhsBytes,
            "fp16_pair_planar", "both");
        auto weight = fixed_allocation(PlacementKind::Weight,
            weightSlices, 0,
            (op.getN() / (2 * tile)) * (op.getK() / tile) * 4,
            rhsBytes, "w8a16_mxm_weight_striped", "both");
        auto result = fixed_allocation(PlacementKind::FinalResult,
            llvm::ArrayRef<int64_t>(resultSlices).take_front(2), 0,
            op.getM() * op.getN() / (2 * tile),
            resultBytes, "fp16_hemisphere_pair_planar", "both",
            target.memory().banks_per_slice > 1 ? 1 : 0);
        const bool replicated = llvm::any_of(op.getResult().getUsers(),
            [](mlir::Operation* user) { return llvm::isa<kernel::SigmoidMultiplyOp>(user); });
        mlir::DictionaryAttr sourceInput;
        if (replicated) {
            auto inherited = get_value_placement(op.getLhs());
            if (mlir::failed(inherited)) {
                op.emitError("gated projection requires a planned activation input");
                return mlir::failure();
            }
            sourceInput = *inherited;
            const auto storage = target.weight_storage_slices();
            if (storage.size() < 8) {
                op.emitError("gated projection requires eight weight storage slices");
                return mlir::failure();
            }
            input.slices = {storage[storage.size()-8], storage[storage.size()-7]};
            input.bank = target.memory().banks_per_slice > 1 ? 1 : 0;
            result = fixed_allocation(PlacementKind::Result,
                resultSlices,
                target.memory().words_per_bank/2, op.getM()*op.getN()/(2*tile),
                resultBytes, "fp16_pair_planar", "both", input.bank);
        }
        auto memoryPlan = rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "input", make_placement_attr(rewriter, input)),
            rewriter.getNamedAttr(
                "weight", make_placement_attr(rewriter, weight)),
            rewriter.getNamedAttr(
                "result", make_placement_attr(rewriter, result)),
        });
        auto config = rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(
                "m", rewriter.getI64IntegerAttr(op.getM())),
            rewriter.getNamedAttr(
                "n", rewriter.getI64IntegerAttr(op.getN())),
            rewriter.getNamedAttr(
                "k", rewriter.getI64IntegerAttr(op.getK())),
            rewriter.getNamedAttr(
                "rhs_scale", op.getRhsScaleAttr()),
        });
        if (replicated) {
            mlir::NamedAttrList plan(memoryPlan);
            plan.set("source_input", sourceInput);
            mlir::NamedAttrList paged(make_placement_attr(rewriter, weight));
            paged.set("paged_weight", rewriter.getBoolAttr(true));
            paged.set("page_count", rewriter.getI64IntegerAttr(1));
            paged.set("page_rows", rewriter.getI64IntegerAttr(weight.instruction_count));
            paged.set("page_granularity", rewriter.getI64IntegerAttr(op.getN()/(2*tile)));
            paged.set("page_items_per_slice_group", rewriter.getI64IntegerAttr(op.getN()/(2*tile)));
            paged.set("page_role_group_base", rewriter.getI64IntegerAttr(0));
            paged.set("page_role_group_count", rewriter.getI64IntegerAttr(1));
            paged.set("page_bank_count", rewriter.getI64IntegerAttr(2));
            paged.set("page_storage_slices", rewriter.getI64ArrayAttr(weight.slices));
            plan.set("weight", paged.getDictionary(rewriter.getContext()));
            for (auto key : {"input", "result"}) {
                mlir::NamedAttrList locked(llvm::cast<mlir::DictionaryAttr>(plan.get(key)));
                locked.set("bank_locked", rewriter.getBoolAttr(true));
                plan.set(key, locked.getDictionary(rewriter.getContext()));
            }
            memoryPlan = plan.getDictionary(rewriter.getContext());
            mlir::NamedAttrList flags(config);
            flags.set("replicated_result", rewriter.getBoolAttr(true));
            config = flags.getDictionary(rewriter.getContext());
        }
        const mlir::Value oldResult = op.getResult();
        if (replicated && mlir::failed(planner.bind(oldResult, result))) {
            op.emitError("gated projection result conflicts with a live dense tensor");
            return mlir::failure();
        }
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
