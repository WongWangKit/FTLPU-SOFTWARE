#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace ftlpu::compiler::schedule {
namespace {

int64_t elementTypeBytes(mlir::Type type)
{
    if (type.isInteger(8)) return 1;
    if (type.isF16()) return 2;
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

} // namespace

AttentionScheduleEmitter::AttentionScheduleEmitter(mlir::IRRewriter& rewriter,
    AttentionTaskGraph op, const target::LPUTargetModel& target,
    AttentionStagePlan stagePlan)
    : rewriter_(rewriter)
    , op_(op)
    , target_(target)
    , stage_plan_(std::move(stagePlan))
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
    if (op_.getCausal()) {
        const auto maskType = mlir::RankedTensorType::get(
            {tile - 1, tile}, rewriter_.getF32Type());
        int64_t maskIndex = 0;
        for (const char* placement :
            {"causal_mask", "causal_mask_mxm1"}) {
            createBinding(rewriter_, op_.getLoc(), {}, maskIndex,
                "internal", "constant", maskType,
                memoryPlan.getAs<mlir::DictionaryAttr>(placement),
                "causal_mask." + std::to_string(maskIndex),
                "causal_mask", rewriter_.getDictionaryAttr({}));
            ++maskIndex;
        }
    }
    const auto ropeType = mlir::RankedTensorType::get(
        {op_.getSeqLen(), op_.getHeadDim() / 2, 2},
        rewriter_.getF16Type());
    createBinding(rewriter_, op_.getLoc(), {}, 2, "internal",
        "constant", ropeType,
        memoryPlan.getAs<mlir::DictionaryAttr>("rope"),
        "rope.cos_sin", "rope_table",
        rewriter_.getDictionaryAttr({
            rewriter_.getNamedAttr(
                "theta", op_.config().getAs<mlir::FloatAttr>("rope_theta")),
            rewriter_.getNamedAttr("head_dim",
                rewriter_.getI64IntegerAttr(op_.getHeadDim())),
        }));

    const int64_t projectionEnd = emitProjections();
    const int64_t qkvCycles = projectionEnd - 1;
    if (qkvCycles <= 0) {
        op_.emitError("computed a non-positive QKV duration");
        return mlir::failure();
    }
    const int64_t qkStart = projectionEnd;
    const int64_t qkEnd = stagePlan.qkEnd(qkStart);
    const int64_t qkCycles = qkEnd - qkStart;
    emitQk(qkStart, stagePlan.qk_wave_interval,
        stagePlan.qk_iw_to_compute_cycles);
    const int64_t softmaxEnd = emitSoftmax(qkEnd);
    const int64_t softmaxCycles = softmaxEnd - qkEnd;
    const int64_t probabilityPackEnd = emitProbabilityPack(softmaxEnd);
    const int64_t probabilityTransposeEnd = emitProbabilityTranspose(probabilityPackEnd);
    const int64_t pvEnd = emitPv(probabilityTransposeEnd);
    const int64_t pvCycles = pvEnd - softmaxEnd;
    const int64_t outputProjectionEnd = emitOutputProjection(pvEnd);
    const int64_t outputProjectionCycles = outputProjectionEnd - pvEnd;

    int64_t cycle = 0;
    const auto appendPhase = [&](llvm::StringRef name, int64_t duration) {
        createTimeline(
            rewriter_, op_.getLoc(), name, cycle, cycle + duration);
        cycle += duration;
    };
    appendPhase("qkv", qkvCycles);
    appendPhase("rope", 1);
    appendPhase("qk", qkCycles);
    appendPhase("softmax", softmaxCycles);
    appendPhase("pv", pvCycles);
    appendPhase("o_proj", outputProjectionCycles);
    auto outputBinding = createBinding(rewriter_, op_.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op_.getResult().getType()),
        memoryPlan.getAs<mlir::DictionaryAttr>("result"));
    return outputBinding.getValue();
}

mlir::LogicalResult lowerAttentionSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
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
            rewriter, operation, target, std::move(stagePlan));
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
