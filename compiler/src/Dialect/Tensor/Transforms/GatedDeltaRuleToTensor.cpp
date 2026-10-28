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
    const bool native=op->hasAttr("native_chunkwise_bf16");
    const auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto bf16Spill = module
        ? module->getAttrOfType<mlir::BoolAttr>("ftlpu.vector_bf16_spill")
        : mlir::BoolAttr{};
    const bool fixedDecode = queryType.getDimSize(0) == 1
        && bf16Spill && bf16Spill.getValue();
    auto query = planValue(op.getQuery(), PlacementKind::Activation,
        {0, 1}, native ? "fp16_pair_planar" : "fp16_scalar_broadcast_planar", "east",
        native ? planarRows(queryType) : queryType.getNumElements(),
        allocateValue, rewriter);
    auto key = planValue(op.getKey(), PlacementKind::Activation,
        {2, 3}, native ? "fp16_pair_planar" : "fp16_scalar_broadcast_planar", "east",
        native ? planarRows(queryType) : queryType.getNumElements(),
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
        "fp32_quad_planar", native ? "west" : "east", planarRows(stateType),
        allocateValue, rewriter);
    if (mlir::failed(query) || mlir::failed(key) || mlir::failed(value)
        || mlir::failed(decay) || mlir::failed(beta)
        || mlir::failed(state) || mlir::failed(output)
        || mlir::failed(finalState))
        return op.emitError("cannot allocate gated delta-rule buffers");

    if (fixedDecode) {
        for (auto* allocations : {&query, &key, &value, &decay, &beta, &state, &output, &finalState}) {
            llvm::SmallVector<mlir::Attribute> mirrored;
            for (auto item : **allocations) {
                mlir::NamedAttrList entry(llvm::cast<mlir::DictionaryAttr>(item));
                mlir::NamedAttrList p(llvm::cast<mlir::DictionaryAttr>(entry.get("placement")));
                p.set("hemisphere", rewriter.getStringAttr("both"));
                if (allocations == &state)
                    if (const auto memoryBank = module->getAttrOfType<mlir::IntegerAttr>("ftlpu.gdn_state_bank"))
                        p.set("bank", memoryBank);
                const auto partition = module->getAttrOfType<mlir::BoolAttr>("ftlpu.gdn_partition_heads");
                if (partition && partition.getValue() && op.getHeads() % 2 == 0) {
                    if (allocations == &state || allocations == &finalState) {
                        p.set("kind", rewriter.getStringAttr("token_partition_planar"));
                        p.set("instruction_count", rewriter.getI64IntegerAttr(planarRows(stateType) / 2));
                    }
                    if (allocations == &output) {
                        p.set("kind", rewriter.getStringAttr("column_partition_planar"));
                        p.set("instruction_count", rewriter.getI64IntegerAttr(planarRows(valueType) / 2));
                    }
                }
                entry.set("placement", p.getDictionary(rewriter.getContext()));
                mirrored.push_back(entry.getDictionary(rewriter.getContext()));
            }
            *allocations = rewriter.getArrayAttr(mirrored);
        }
    }

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
        "fp32_quad_planar", fixedDecode ? "both" : "east", stateBytes,
        planarRows(stateType), PlacementKind::Result);
    auto productState = scratchPlan({24, 25, 26, 27},
        "fp32_quad_planar", fixedDecode ? "both" : "east", stateBytes,
        planarRows(stateType), PlacementKind::Result);
    auto remembered = scratchPlan({28, 29, 30, 31},
        "fp32_quad_planar", fixedDecode ? "both" : "east", vectorBytes,
        op.getHeads() * ((op.getValueDim() + 31) / 32),
        PlacementKind::VxmResult);
    auto delta = scratchPlan({34, 35, 36, 37},
        "fp32_quad_planar", fixedDecode ? "both" : "east", vectorBytes,
        op.getHeads() * ((op.getValueDim() + 31) / 32),
        PlacementKind::VxmResult1);
    if (mlir::failed(pingPong) || mlir::failed(productState)
        || mlir::failed(remembered)
        || mlir::failed(delta))
        return op.emitError("cannot allocate gated delta-rule scratch");

    llvm::SmallVector<mlir::Attribute> scratch{*pingPong,*productState,*remembered,*delta};
    if(native){
        const auto relocate=[&](mlir::FailureOr<mlir::ArrayAttr>& values){
            llvm::SmallVector<mlir::Attribute> result;
            for(auto value:*values){
                mlir::NamedAttrList entry(llvm::cast<mlir::DictionaryAttr>(value));
                mlir::NamedAttrList place(llvm::cast<mlir::DictionaryAttr>(entry.get("placement")));
                place.set("bank",rewriter.getI64IntegerAttr(1));
                // Graph edges have one owner; only the internal compute arena
                // is mirrored. The native ingress copies to both hemispheres.
                entry.set("placement",place.getDictionary(rewriter.getContext()));
                result.push_back(entry.getDictionary(rewriter.getContext()));
            }
            values=rewriter.getArrayAttr(result);
        };
        for(auto* values:{&query,&key,&value,&decay,&beta,&state,&output,&finalState})relocate(*values);
        scratch.clear();
        // The native node owns bank 0 while running. Live graph edges are
        // allocated in bank 1 and copied at this explicit operator boundary.
        for(int64_t pair=0;pair<52;pair+=2){
            auto p=scratchPlan({pair,pair+1},"fp16_pair_planar","both",4096*64,4096,PlacementKind::Result);
            if(mlir::failed(p))return op.emitError("cannot reserve native GDN arena");
            scratch.push_back(*p);
        }
    }

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
            rewriter.getArrayAttr(scratch)),
        rewriter.getNamedAttr("output_allocations", *output),
        rewriter.getNamedAttr("final_state_allocations", *finalState),
        rewriter.getNamedAttr("config", rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("lowering",
                rewriter.getStringAttr(native ? "native_chunkwise_bf16" : "direct_recurrent_scan")),
            rewriter.getNamedAttr("state_precision",
                rewriter.getStringAttr(native ? "bf16" : "fp32")),
            rewriter.getNamedAttr("ingress_conversion",
                rewriter.getStringAttr(native ? "fp32_upper16_truncate" : "none")),
        })),
    });
    auto lowered = llvm::cast<tensor::GatedDeltaRuleTaskOp>(
        rewriter.create(task));
    rewriter.replaceOp(op,
        {lowered.getOutput(), lowered.getFinalState()});
    return mlir::success();
}

} // namespace ftlpu::compiler::tensor_lowering
