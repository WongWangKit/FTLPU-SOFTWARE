#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_memory_layout.hpp"

#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::schedule {

AttentionMemoryLayout::AttentionMemoryLayout(stream::AttentionOp op,
    const target::LPUTargetModel& target)
    : target_(target), seqLen_(op.getSeqLen()), hidden_(op.getHidden()),
      kvHeads_(op.getKvHeads())
{
    const auto plan = op.getMemoryPlan();
    const char* names[] = {"query_weight", "key_weight", "value_weight"};
    for (std::size_t i = 0; i < weightBases_.size(); ++i) {
        const auto placement = plan.getAs<mlir::DictionaryAttr>(names[i]);
        weightBases_[i] = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    }
    if (const auto rope = plan.getAs<mlir::DictionaryAttr>("rope"))
        ropeBase_ = rope.getAs<mlir::IntegerAttr>("base_row").getInt();
}

int64_t AttentionMemoryLayout::weightBase(AttentionProjectionKind projection) const
{
    return weightBases_[static_cast<std::size_t>(projection)];
}

int64_t AttentionMemoryLayout::weightAddress(AttentionProjectionKind projection,
    int64_t head, int64_t reductionBlock, int64_t localMxm, int64_t column) const
{
    const int64_t hiddenBlocks = hidden_ / target_.throughput().mxm_rows;
    return weightBase(projection)
        + ((head / 2) * hiddenBlocks + reductionBlock) * 8
        + localMxm * 4 + column;
}

int64_t AttentionMemoryLayout::activationAddress(
    int64_t reductionBlock, int64_t tokenBlock) const
{
    return reductionBlock * seqLen_
        + tokenBlock * target_.throughput().mxm_rows;
}

int64_t AttentionMemoryLayout::projectionAddress(AttentionProjectionKind projection,
    int64_t head, int64_t tokenBlock) const
{
    const int64_t valueBase = kvHeads_ * seqLen_;
    return (projection == AttentionProjectionKind::Value ? valueBase : 0)
        + head * seqLen_ + tokenBlock * target_.throughput().mxm_rows;
}

int64_t AttentionMemoryLayout::queryIwAddress(
    int64_t head, int64_t tokenBlock, int64_t phase) const
{
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = seqLen_ / tile;
    return target_.attention_query_iw_base_row()
        + (head * tokenBlocks + tokenBlock) * (tile / 8) + phase;
}

int64_t AttentionMemoryLayout::keyAddress(int64_t kvHead, int64_t keyBlock) const
{
    return kvHead * seqLen_ + keyBlock * target_.throughput().mxm_rows;
}

int64_t AttentionMemoryLayout::scoreAddress(int64_t queryHead,
    int64_t queryBlock, int64_t keyBlock) const
{
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = seqLen_ / tile;
    return target_.attention_score_base_row()
        + (queryHead * tokenBlocks + queryBlock) * seqLen_
        + keyBlock * tile;
}

int64_t AttentionMemoryLayout::ropeAddress(int64_t token) const
{
    return ropeBase_ + token;
}

} // namespace ftlpu::compiler::schedule
