#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_softmax_planner.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace ftlpu::compiler::schedule {
namespace {

int64_t elementTypeBytes(mlir::Type type)
{
    if (type.isInteger(8)) return 1;
    if (is_lpu_16bit_float(type)) return 2;
    if (type.isF32()) return 4;
    return 0;
}

BindingOp createBinding(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::ValueRange source, int64_t index, llvm::StringRef access,
    llvm::StringRef role, mlir::RankedTensorType type,
    mlir::DictionaryAttr placement, llvm::StringRef name = {},
    llvm::StringRef initializer = {},
    mlir::DictionaryAttr initializerConfig = {})
{
    mlir::OperationState state(location, BindingOp::getOperationName());
    state.addOperands(source);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr(
            "index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr(
            "access", rewriter.getStringAttr(access)),
        rewriter.getNamedAttr("role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * elementTypeBytes(type.getElementType()))),
        rewriter.getNamedAttr("placement", placement),
    });
    if (!name.empty())
        state.addAttribute("name", rewriter.getStringAttr(name));
    if (!initializer.empty())
        state.addAttribute(
            "initializer", rewriter.getStringAttr(initializer));
    if (initializerConfig)
        state.addAttribute("initializer_config", initializerConfig);
    return llvm::cast<BindingOp>(rewriter.create(state));
}

void createTimeline(mlir::IRRewriter& rewriter, mlir::Location location,
    llvm::StringRef name, int64_t start, int64_t end)
{
    mlir::OperationState state(location, TimelineOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)),
        rewriter.getNamedAttr("start", rewriter.getI64IntegerAttr(start)),
        rewriter.getNamedAttr("end", rewriter.getI64IntegerAttr(end)),
    });
    rewriter.create(state);
}

int64_t annotateStateTransfers(mlir::func::FuncOp function,
    BindingOp binding, mlir::IRRewriter& rewriter)
{
    const auto placement = binding.getPlacement();
    const int64_t base =
        placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t count =
        placement.getAs<mlir::IntegerAttr>("instruction_count").getInt();
    const int64_t stride =
        placement.getAs<mlir::IntegerAttr>("address_stride").getInt();
    const int64_t final = base + (count - 1) * stride;
    const int64_t begin = std::min(base, final);
    const int64_t end = std::max(base, final) + 1;
    const int64_t bank = placement.getAs<mlir::IntegerAttr>("bank")
        ? placement.getAs<mlir::IntegerAttr>("bank").getInt() : 0;
    const llvm::StringRef hemisphere =
        placement.getAs<mlir::StringAttr>("hemisphere").getValue();
    llvm::SmallDenseSet<int64_t, 32> slices;
    for (mlir::Attribute value : placement.getAs<mlir::ArrayAttr>("slices"))
        slices.insert(llvm::cast<mlir::IntegerAttr>(value).getInt());

    int64_t annotated = 0;
    function.walk([&](MemTransferOp transfer) {
        if (transfer.getAddressBinding()
            || transfer.getBank().value_or(0) != bank
            || !slices.contains(transfer.getSlice()))
            return;
        if ((hemisphere == "east" && transfer.getHemisphere() != 0)
            || (hemisphere == "west" && transfer.getHemisphere() != 1))
            return;
        const int64_t waveCount = transfer.getWaveCount().value_or(1);
        const int64_t waveStride =
            transfer.getWaveAddressStride().value_or(0);
        const int64_t corners[] = {
            transfer.getAddress(),
            transfer.getAddress()
                + (transfer.getRepeatCount() - 1)
                    * transfer.getAddressStride(),
            transfer.getAddress() + (waveCount - 1) * waveStride,
            transfer.getAddress() + (waveCount - 1) * waveStride
                + (transfer.getRepeatCount() - 1)
                    * transfer.getAddressStride(),
        };
        if (llvm::any_of(corners, [&](int64_t address) {
                return address < begin || address >= end;
            }))
            return;
        transfer->setAttr("address_binding",
            rewriter.getI64IntegerAttr(binding.getIndex()));
        transfer->setAttr("address_binding_access",
            rewriter.getStringAttr("internal"));
        ++annotated;
    });
    return annotated;
}

} // namespace

AttentionScheduleEmitter::AttentionScheduleEmitter(mlir::IRRewriter& rewriter,
    AttentionTaskGraph op, const target::LPUTargetModel& target,
    AttentionStagePlan stagePlan, AttentionScheduleStrategy strategy)
    : rewriter_(rewriter)
    , op_(op)
    , target_(target)
    , stage_plan_(std::move(stagePlan))
    , strategy_(strategy)
{
}

mlir::FailureOr<mlir::Value>
AttentionScheduleEmitter::emit(int64_t outputIndex)
{
    const AttentionStagePlan& stagePlan = stage_plan_;
    if (mlir::failed(stagePlan.tasks.validate())) {
        op_.emitError("failed to construct the attention task DAG");
        return mlir::failure();
    }
    const int64_t tile = target_.throughput().mxm_rows;

    rewriter_.setInsertionPoint(op_.output);
    const auto memoryPlan = op_.getMemoryPlan();
    const mlir::Value inputs[] = {op_.getInput(), op_.getQueryWeight(),
        op_.getKeyWeight(), op_.getValueWeight(), op_.getOutputWeight()};
    const char* placements[] = {"input", "query_weight", "key_weight",
        "value_weight", "output_weight"};
    for (std::size_t index = 0; index < std::size(inputs); ++index) {
        const auto argument =
            llvm::dyn_cast<mlir::BlockArgument>(inputs[index]);
        if (!argument) {
            if (index == 0) continue;
            op_.emitError("attention runtime input is not a block argument");
            return mlir::failure();
        }
        createBinding(rewriter_, op_.getLoc(), inputs[index],
            argument.getArgNumber(), "input",
            index == 0 ? "activation" : "weight",
            llvm::cast<mlir::RankedTensorType>(argument.getType()),
            memoryPlan.getAs<mlir::DictionaryAttr>(placements[index]));
    }
    const mlir::Value biases[] = {
        op_.getQueryBias(), op_.getKeyBias(), op_.getValueBias()};
    const char* biasPlacements[] = {
        "query_bias", "key_bias", "value_bias"};
    for (std::size_t index = 0; index < std::size(biases); ++index) {
        if (!biases[index]) continue;
        const auto argument =
            llvm::dyn_cast<mlir::BlockArgument>(biases[index]);
        const auto placement = memoryPlan.getAs<mlir::DictionaryAttr>(
            biasPlacements[index]);
        if (!argument || !placement) {
            op_.emitError(
                "attention bias is missing a runtime argument or placement");
            return mlir::failure();
        }
        createBinding(rewriter_, op_.getLoc(), biases[index],
            argument.getArgNumber(), "input", "bias",
            llvm::cast<mlir::RankedTensorType>(argument.getType()),
            placement, biasPlacements[index]);
    }
    const mlir::Value normWeights[] = {
        op_.getQueryNormWeight(), op_.getKeyNormWeight()};
    const char* normPlacements[] = {
        "query_norm_weight", "key_norm_weight"};
    for (std::size_t index = 0; index < std::size(normWeights); ++index) {
        if (!normWeights[index]) continue;
        const auto argument =
            llvm::dyn_cast<mlir::BlockArgument>(normWeights[index]);
        const auto placement = memoryPlan.getAs<mlir::DictionaryAttr>(
            normPlacements[index]);
        if (!argument || !placement) {
            op_.emitError(
                "Q/K head RMSNorm weight is missing a runtime argument or placement");
            return mlir::failure();
        }
        createBinding(rewriter_, op_.getLoc(), normWeights[index],
            argument.getArgNumber(), "input", "weight",
            llvm::cast<mlir::RankedTensorType>(argument.getType()),
            placement, normPlacements[index]);
    }
    if (op_.getCausal()) {
        const auto maskType = mlir::RankedTensorType::get(
            {tile - 1, tile},
            llvm::cast<mlir::RankedTensorType>(
                op_.getInput().getType()).getElementType());
        const char* maskPlacements[] = {"causal_mask",
            "causal_mask_mxm1", "fused_causal_mask",
            "fused_causal_mask_bank1"};
        const int64_t maskCount = strategy_ == AttentionScheduleStrategy::Fused
            ? 4 : 2;
        for (int64_t maskIndex = 0; maskIndex < maskCount; ++maskIndex) {
            const char* placement = maskPlacements[maskIndex];
            createBinding(rewriter_, op_.getLoc(), {}, maskIndex,
                "internal", "constant", maskType,
                memoryPlan.getAs<mlir::DictionaryAttr>(placement),
                "causal_mask." + std::to_string(maskIndex),
                "causal_mask", rewriter_.getDictionaryAttr({}));
        }
    }
    const auto ropeType = mlir::RankedTensorType::get(
        {op_.getSeqLen(), op_.getHeadDim() / 2, 2},
        llvm::cast<mlir::RankedTensorType>(
            op_.getInput().getType()).getElementType());
    const int64_t constantBindingBase =
        strategy_ == AttentionScheduleStrategy::Fused ? 4 : 2;
    createBinding(rewriter_, op_.getLoc(), {}, constantBindingBase, "internal",
        "constant", ropeType,
        memoryPlan.getAs<mlir::DictionaryAttr>("rope"),
        "rope.cos_sin", "rope_table",
        rewriter_.getDictionaryAttr({
            rewriter_.getNamedAttr(
                "theta", op_.config().getAs<mlir::FloatAttr>("rope_theta")),
            rewriter_.getNamedAttr("head_dim",
                rewriter_.getI64IntegerAttr(op_.getHeadDim())),
        }));
    const auto ropeProduct =
        memoryPlan.getAs<mlir::DictionaryAttr>("rope_product");
    const auto bankInterleaved =
        ropeProduct.getAs<mlir::BoolAttr>("bank_interleaved");
    const bool hasRopeMirror = bankInterleaved && bankInterleaved.getValue();
    if (hasRopeMirror) {
        createBinding(rewriter_, op_.getLoc(), {}, constantBindingBase + 1,
            "internal", "constant", ropeType,
            memoryPlan.getAs<mlir::DictionaryAttr>("rope_mirror"),
            "rope.cos_sin.mirror", "rope_table",
            rewriter_.getDictionaryAttr({
                rewriter_.getNamedAttr("theta",
                    op_.config().getAs<mlir::FloatAttr>("rope_theta")),
                rewriter_.getNamedAttr("head_dim",
                    rewriter_.getI64IntegerAttr(op_.getHeadDim())),
            }));
    }
    const auto probabilityType = mlir::RankedTensorType::get(
        {op_.getQueryHeads(), op_.getSeqLen(), op_.getSeqLen()},
        llvm::cast<mlir::RankedTensorType>(
            op_.getInput().getType()).getElementType());
    const int64_t workspaceBindingBase =
        constantBindingBase + 1 + (hasRopeMirror ? 1 : 0);
    auto probabilityPackBinding = createBinding(
        rewriter_, op_.getLoc(), {}, workspaceBindingBase,
        "internal", "workspace", probabilityType,
        memoryPlan.getAs<mlir::DictionaryAttr>("probability_pack"),
        "attention.probability_pack");
    auto probabilityDiagonalBinding = createBinding(
        rewriter_, op_.getLoc(), {}, workspaceBindingBase + 1,
        "internal", "workspace", probabilityType,
        memoryPlan.getAs<mlir::DictionaryAttr>("probability_diagonal"),
        "attention.probability_diagonal");
    const auto elementType = llvm::cast<mlir::RankedTensorType>(
        op_.getInput().getType()).getElementType();
    const auto kvCapacity =
        op_.config().getAs<mlir::IntegerAttr>("kv_cache_capacity");
    const auto kvResident =
        op_.config().getAs<mlir::IntegerAttr>(
            "kv_cache_resident_tokens");
    BindingOp keyBinding;
    BindingOp valueBinding;
    if (kvCapacity) {
        const int64_t residentTokens =
            kvResident ? kvResident.getInt() : kvCapacity.getInt();
        const auto stateType = mlir::RankedTensorType::get(
            {residentTokens, op_.getKvHeads(), op_.getHeadDim()},
            elementType);
        key_state_binding_index_ = kAttentionKeyStateBindingIndex;
        value_state_binding_index_ = kAttentionValueStateBindingIndex;
        keyBinding = createBinding(rewriter_, op_.getLoc(), {},
            key_state_binding_index_, "internal", "state.kv.key",
            stateType, memoryPlan.getAs<mlir::DictionaryAttr>("key"),
            "attention.key_cache");
        valueBinding = createBinding(rewriter_, op_.getLoc(), {},
            value_state_binding_index_, "internal", "state.kv.value",
            stateType, memoryPlan.getAs<mlir::DictionaryAttr>("value"),
            "attention.value_cache");
    } else {
        const auto valueType = mlir::RankedTensorType::get(
            {op_.getSeqLen(), op_.getKvHeads() * op_.getHeadDim()},
            elementType);
        valueBinding = createBinding(
            rewriter_, op_.getLoc(), {}, workspaceBindingBase + 2,
            "internal", "workspace", valueType,
            memoryPlan.getAs<mlir::DictionaryAttr>("value"),
            "attention.value");
    }
    const auto contextType = mlir::RankedTensorType::get(
        {op_.getSeqLen(), op_.getQueryHeads() * op_.getHeadDim()},
        llvm::cast<mlir::RankedTensorType>(
            op_.getInput().getType()).getElementType());
    auto contextBinding = createBinding(
        rewriter_, op_.getLoc(), {}, workspaceBindingBase + 3,
        "internal", "workspace", contextType,
        memoryPlan.getAs<mlir::DictionaryAttr>("context"),
        "attention.context");
    const int64_t projectionEnd = emitProjections();
    const int64_t qkvCycles = projectionEnd - 1;
    if (qkvCycles <= 0) {
        op_.emitError("computed a non-positive QKV duration");
        return mlir::failure();
    }
    const int64_t qkStart = projectionEnd;
    const int64_t qkEnd = stagePlan.qkEnd(qkStart);
    // The compact 8-queue CModel path currently uses the correctness-first
    // tail schedule. Fused planning remains an optional optimization, but it
    // must not suppress the BF16 score materialization consumed below.
    const bool fusedSoftmax = false;
    if (mlir::failed(emitQk(qkStart, stagePlan.qk_wave_interval,
            stagePlan.qk_iw_to_compute_cycles, fusedSoftmax)))
        return mlir::failure();
    const int64_t softmaxEnd = emitSoftmax(qkStart, qkEnd, fusedSoftmax);
    const int64_t softmaxStart = fusedSoftmax
        ? qkStart + target_.throughput().mxm_earliest_iw_cycle
            + *target_.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, 0)
            + stagePlan.qk_iw_to_compute_cycles
            + (op_.getHeadDim() / tile - 1)
                * (op_.getSeqLen() / tile)
                * target_.mxm_block_issue_interval()
            + target_.mxm_first_result_latency()
        : qkEnd;
    const int64_t probabilityTransposeStart =
        std::max(softmaxEnd, qkEnd);
    const int64_t probabilityTransposeEnd =
        emitProbabilityTranspose(probabilityTransposeStart);
    probabilityPackBinding->setAttr("ready_cycle",
        rewriter_.getI64IntegerAttr(probabilityTransposeStart));
    probabilityDiagonalBinding->setAttr("ready_cycle",
        rewriter_.getI64IntegerAttr(probabilityTransposeEnd));
    valueBinding->setAttr("ready_cycle",
        rewriter_.getI64IntegerAttr(projectionEnd));
    if (keyBinding)
        keyBinding->setAttr("ready_cycle",
            rewriter_.getI64IntegerAttr(projectionEnd));
    const int64_t pvEnd = emitPv(probabilityTransposeEnd);
    contextBinding->setAttr("ready_cycle",
        rewriter_.getI64IntegerAttr(pvEnd));
    const int64_t outputProjectionEnd =
        emitOutputProjection(pvEnd, projectionEnd);
    if (keyBinding) {
        const auto function = op_.output->getParentOfType<mlir::func::FuncOp>();
        if (annotateStateTransfers(function, keyBinding, rewriter_) == 0
            || annotateStateTransfers(function, valueBinding, rewriter_) == 0) {
            op_.emitError(
                "persistent KV binding has no relocatable MEM transfers");
            return mlir::failure();
        }
    }

    createTimeline(rewriter_, op_.getLoc(), "qkv", 0, qkvCycles);
    createTimeline(rewriter_, op_.getLoc(), "rope", qkvCycles, qkStart);
    createTimeline(rewriter_, op_.getLoc(), "qk", qkStart, qkEnd);
    createTimeline(rewriter_, op_.getLoc(),
        fusedSoftmax ? "softmax_fused" : "softmax",
        softmaxStart, softmaxEnd);
    createTimeline(rewriter_, op_.getLoc(), "probability_transpose",
        probabilityTransposeStart, probabilityTransposeEnd);
    createTimeline(rewriter_, op_.getLoc(), "pv",
        probabilityTransposeEnd, pvEnd);
    createTimeline(rewriter_, op_.getLoc(), "o_proj",
        pvEnd, outputProjectionEnd);
    auto outputBinding = createBinding(rewriter_, op_.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op_.getResult().getType()),
        memoryPlan.getAs<mlir::DictionaryAttr>("result"),
        "attention.result");
    return outputBinding.getValue();
}

mlir::LogicalResult lowerAttentionSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target,
    AttentionScheduleStrategy strategy)
{
    auto graphs = collectAttentionTaskGraphs(function);
    if (mlir::failed(graphs)) return mlir::failure();
    int64_t outputIndex = 0;
    for (AttentionTaskGraph operation : *graphs) {
        AttentionStagePlan stagePlan = planAttentionStages(
            {static_cast<int64_t>(operation.getSeqLen()),
                static_cast<int64_t>(operation.getHidden()),
                static_cast<int64_t>(operation.getQueryHeads()),
                static_cast<int64_t>(operation.getKvHeads()),
                static_cast<int64_t>(operation.getHeadDim())},
            target);
        if (mlir::failed(stagePlan.tasks.validate())) {
            operation.emitError("failed to construct the attention task DAG");
            return mlir::failure();
        }
        AttentionScheduleEmitter emitter(
            rewriter, operation, target, std::move(stagePlan), strategy);
        auto lowered = emitter.emit(outputIndex++);
        if (mlir::failed(lowered)) return mlir::failure();
        rewriter.replaceOp(operation.output, *lowered);
        rewriter.eraseOp(operation.pv);
        rewriter.eraseOp(operation.probability_transpose);
        rewriter.eraseOp(operation.value_transpose);
        rewriter.eraseOp(operation.softmax);
        rewriter.eraseOp(operation.qk);
        rewriter.eraseOp(operation.query_rope);
        rewriter.eraseOp(operation.key_rope);
        rewriter.eraseOp(operation.query);
        rewriter.eraseOp(operation.key);
        rewriter.eraseOp(operation.value);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
