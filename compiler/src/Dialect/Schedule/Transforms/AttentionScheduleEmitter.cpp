#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_stage_plan.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace ftlpu::compiler::schedule {

AttentionScheduleEmitter::AttentionScheduleEmitter(mlir::IRRewriter& rewriter,
    stream::AttentionOp op, const target::LPUTargetModel& target,
    AttentionStagePlan stagePlan)
    : rewriter_(rewriter)
    , op_(op)
    , target_(target)
    , stage_plan_(std::move(stagePlan))
{
}

mlir::FailureOr<AttentionOp> AttentionScheduleEmitter::emit()
{
    const AttentionStagePlan& stagePlan = stage_plan_;
    if (mlir::failed(stagePlan.tasks.validate())) {
        op_.emitError("failed to construct the attention task DAG");
        return mlir::failure();
    }
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const int64_t issue = target_.mxm_block_issue_interval();

    rewriter_.setInsertionPoint(op_);
    const int64_t projectionEnd = emitProjections();
    const int64_t qkvCycles = projectionEnd - 1;
    if (qkvCycles <= 0) {
        op_.emitError("computed a non-positive QKV duration");
        return mlir::failure();
    }
    const int64_t qkWaveComputeCycles = tokenBlocks * headBlocks * issue;
    const int64_t qkIwToComputeCycles = 24;
    const int64_t qkFirstIwOffset = target_.throughput().mxm_earliest_iw_cycle
        + *target_.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight, target::StreamDirection::East, 0);
    const int64_t qkFirstComputeOffset = qkFirstIwOffset + qkIwToComputeCycles;
    const int64_t qkComputeEnd = qkFirstComputeOffset + qkWaveComputeCycles;
    const int64_t qkWaveInterval = qkComputeEnd - qkFirstIwOffset;
    const int64_t qkWaveDuration = qkComputeEnd
        + std::max(target_.throughput().mxm0_accumulator_latency,
            target_.throughput().mxm1_accumulator_latency);
    const int64_t qkStart = projectionEnd;
    const int64_t qkEnd = qkStart
        + (stagePlan.qk_waves.size() - 1) * qkWaveInterval
        + qkWaveDuration;
    const int64_t qkCycles = qkEnd - qkStart;
    emitQk(qkStart, qkWaveInterval, qkIwToComputeCycles);
    const int64_t softmaxEnd = emitSoftmax(qkEnd);
    const int64_t softmaxCycles = softmaxEnd - qkEnd;
    const int64_t probabilityPackEnd = emitProbabilityPack(softmaxEnd);
    const int64_t probabilityTransposeEnd = emitProbabilityTranspose(probabilityPackEnd);
    const int64_t pvEnd = emitPv(probabilityTransposeEnd);
    const int64_t pvCycles = pvEnd - softmaxEnd;
    const int64_t outputProjectionEnd = emitOutputProjection(pvEnd);
    const int64_t outputProjectionCycles = outputProjectionEnd - pvEnd;

    llvm::SmallVector<mlir::Attribute> phases;
    llvm::SmallVector<mlir::Attribute> workWaves;
    llvm::SmallVector<mlir::Attribute> projectionWork;
    for (const auto& work : stagePlan.projection_work) {
        const char* name = work.projection == AttentionProjection::Query ? "query"
            : work.projection == AttentionProjection::Key ? "key" : "value";
        projectionWork.push_back(rewriter_.getDictionaryAttr({
            rewriter_.getNamedAttr("projection", rewriter_.getStringAttr(name)),
            rewriter_.getNamedAttr("head_group", rewriter_.getI64IntegerAttr(work.head_group)),
            rewriter_.getNamedAttr("reduction_block", rewriter_.getI64IntegerAttr(work.reduction_block)),
            rewriter_.getNamedAttr("token_block", rewriter_.getI64IntegerAttr(work.token_block)),
            rewriter_.getNamedAttr("hemisphere", rewriter_.getI64IntegerAttr(work.hemisphere)),
            rewriter_.getNamedAttr("final_reduction", rewriter_.getBoolAttr(work.final_reduction)),
        }));
    }
    int64_t cycle = 0;
    const auto appendPhase = [&](llvm::StringRef name, int64_t duration) {
        phases.push_back(rewriter_.getDictionaryAttr({
            rewriter_.getNamedAttr("name", rewriter_.getStringAttr(name)),
            rewriter_.getNamedAttr("start", rewriter_.getI64IntegerAttr(cycle)),
            rewriter_.getNamedAttr("end", rewriter_.getI64IntegerAttr(cycle + duration)),
        }));
        cycle += duration;
    };
    appendPhase("qkv", qkvCycles);
    appendPhase("rope", 1);
    appendPhase("qk", qkCycles);
    appendPhase("softmax", softmaxCycles);
    appendPhase("pv", pvCycles);
    appendPhase("o_proj", outputProjectionCycles);
    const int64_t pvStart = probabilityTransposeEnd;
    const auto appendWaves = [&](llvm::StringRef phaseName,
        const std::vector<AttentionWorkWave>& waves, int64_t start,
        int64_t interval, int64_t duration) {
        for (std::size_t index = 0; index < waves.size(); ++index) {
            llvm::SmallVector<mlir::Attribute> slots;
            for (const auto& work : waves[index].slots) {
                if (!work) continue;
                slots.push_back(rewriter_.getDictionaryAttr({
                    rewriter_.getNamedAttr("query_head", rewriter_.getI64IntegerAttr(work->query_head)),
                    rewriter_.getNamedAttr("kv_head", rewriter_.getI64IntegerAttr(work->kv_head)),
                    rewriter_.getNamedAttr("query_block", rewriter_.getI64IntegerAttr(work->query_block)),
                    rewriter_.getNamedAttr("hemisphere", rewriter_.getI64IntegerAttr(work->hemisphere)),
                    rewriter_.getNamedAttr("local_mxm", rewriter_.getI64IntegerAttr(work->local_mxm)),
                }));
            }
            const int64_t waveStart = start + static_cast<int64_t>(index) * interval;
            workWaves.push_back(rewriter_.getDictionaryAttr({
                rewriter_.getNamedAttr("phase", rewriter_.getStringAttr(phaseName)),
                rewriter_.getNamedAttr("index", rewriter_.getI64IntegerAttr(index)),
                rewriter_.getNamedAttr("start", rewriter_.getI64IntegerAttr(waveStart)),
                rewriter_.getNamedAttr("end", rewriter_.getI64IntegerAttr(waveStart + duration)),
                rewriter_.getNamedAttr("slots", rewriter_.getArrayAttr(slots)),
            }));
        }
    };
    appendWaves("qk", stagePlan.qk_waves, qkStart,
        qkWaveInterval, qkWaveDuration);
    appendWaves("pv", stagePlan.pv_waves, pvStart,
        qkWaveComputeCycles, qkWaveComputeCycles);
    mlir::OperationState state(op_.getLoc(), AttentionOp::getOperationName());
    state.addOperands(op_->getOperands());
    state.addTypes(op_.getResult().getType());
    for (llvm::StringRef name : {"seq_len", "hidden", "query_heads", "kv_heads",
             "head_dim", "rope_theta", "causal", "memory_plan", "routes"})
        state.addAttribute(name, op_->getAttr(name));
    state.addAttribute("phases", rewriter_.getArrayAttr(phases));
    state.addAttribute("work_waves", rewriter_.getArrayAttr(workWaves));
    state.addAttribute("projection_work", rewriter_.getArrayAttr(projectionWork));
    return llvm::cast<AttentionOp>(rewriter_.create(state));
}

mlir::LogicalResult lowerAttentionSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::AttentionOp> operations;
    for (mlir::Operation& operation : function.getBody().front()) {
        if (operation.getName().getStringRef()
            == stream::AttentionOp::getOperationName())
            operations.emplace_back(&operation);
    }
    for (stream::AttentionOp operation : operations) {
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
        auto lowered = emitter.emit();
        if (mlir::failed(lowered)) return mlir::failure();
        rewriter.replaceOp(operation, lowered->getResult());
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
