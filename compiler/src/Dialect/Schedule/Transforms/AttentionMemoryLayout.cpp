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
    if (const auto input = plan.getAs<mlir::DictionaryAttr>("input")) {
        const auto slices = input.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < activationSlices_.size(); ++i)
            activationSlices_[i] =
                llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
    if (const auto weight = plan.getAs<mlir::DictionaryAttr>("output_weight")) {
        outputWeightBase_ = weight.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto slices = weight.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < outputWeightSlices_.size(); ++i)
            outputWeightSlices_[i] = llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
    if (const auto rope = plan.getAs<mlir::DictionaryAttr>("rope"))
        ropeBase_ = rope.getAs<mlir::IntegerAttr>("base_row").getInt();
    const auto readPlacement = [&](llvm::StringRef name, auto& slices, int64_t& base) {
        const auto placement = plan.getAs<mlir::DictionaryAttr>(name);
        base = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto values = placement.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < slices.size(); ++i)
            slices[i] = llvm::cast<mlir::IntegerAttr>(values[i]).getInt();
    };
    readPlacement("score", scaledScoreSlices_[0], scaledScoreBase_);
    int64_t ignoredBase = 0;
    readPlacement("score_mxm1", scaledScoreSlices_[1], ignoredBase);
    readPlacement("exp", expScoreSlices_[0], expScoreBase_);
    readPlacement("exp_mxm1", expScoreSlices_[1], ignoredBase);
    readPlacement("causal_mask", causalMaskSlices_[0], causalMaskBase_);
    readPlacement("causal_mask_mxm1", causalMaskSlices_[1], ignoredBase);
    readPlacement("probability", probabilitySlices_[0], probabilityBase_);
    readPlacement("probability_mxm1", probabilitySlices_[1], ignoredBase);
    readPlacement("probability_pack", probabilityPackSlices_, probabilityPackBase_);
    readPlacement("probability_diagonal", probabilityDiagonalSlices_, probabilityDiagonalBase_);
    if (const auto value = plan.getAs<mlir::DictionaryAttr>("value"))
        valuePackBase_ = value.getAs<mlir::IntegerAttr>("base_row").getInt();
    readPlacement("context", contextSlices_, contextBase_);
    if (const auto result = plan.getAs<mlir::DictionaryAttr>("result"))
        resultBase_ = result.getAs<mlir::IntegerAttr>("base_row").getInt();
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
    return scoreTokenAddress(queryHead, queryBlock,
        keyBlock * target_.throughput().mxm_rows);
}

int64_t AttentionMemoryLayout::scoreTokenAddress(int64_t queryHead,
    int64_t queryBlock, int64_t key) const
{
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = seqLen_ / tile;
    return target_.attention_score_base_row()
        + (queryHead * tokenBlocks + queryBlock) * seqLen_
        + key;
}

int64_t AttentionMemoryLayout::probabilityAddress(int64_t queryHead,
    int64_t queryBlock, int64_t key) const
{
    const int64_t tokenBlocks = seqLen_ / target_.throughput().mxm_rows;
    return probabilityBase_
        + (queryHead * tokenBlocks + queryBlock) * seqLen_ + key;
}

int64_t AttentionMemoryLayout::probabilityPackAddress(int64_t queryHead,
    int64_t queryBlock, int64_t keyBlock) const
{
    const int64_t tokenBlocks = seqLen_ / target_.throughput().mxm_rows;
    return probabilityPackBase_
        + (queryHead * tokenBlocks + queryBlock)
            * (seqLen_ / target_.throughput().lanes_per_tile)
        + keyBlock;
}

int64_t AttentionMemoryLayout::probabilityDiagonalAddress(int64_t queryHead,
    int64_t queryBlock, int64_t keyBlock, int64_t diagonal) const
{
    const int64_t tokenBlocks = seqLen_ / target_.throughput().mxm_rows;
    return probabilityDiagonalBase_
        + ((queryHead * tokenBlocks + queryBlock) * tokenBlocks + keyBlock)
            * target_.throughput().tile_rows
        + diagonal;
}

int64_t AttentionMemoryLayout::valuePackAddress(int64_t head,
    int64_t reductionBlock, int64_t tokenBlock, int64_t row) const
{
    const int64_t tileRows = target_.throughput().tile_rows;
    const int64_t tokenBlocks = seqLen_ / target_.throughput().mxm_rows;
    return valuePackBase_
        + ((head * 2 + reductionBlock) * tokenBlocks + tokenBlock) * tileRows + row;
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::valuePackSlices(
    int64_t reductionBlock) const
{
    return valuePackSlices_.at(static_cast<std::size_t>(reductionBlock));
}

int64_t AttentionMemoryLayout::contextAddress(int64_t queryHead, int64_t token) const
{
    return contextBase_ + queryHead * seqLen_ + token;
}

int64_t AttentionMemoryLayout::outputWeightAddress(int64_t outputGroup,
    int64_t reductionBlock, int64_t column) const
{
    const int64_t reductionBlocks = hidden_ / target_.throughput().mxm_rows;
    return outputWeightBase_
        + (outputGroup * reductionBlocks + reductionBlock) * 4 + column;
}

int64_t AttentionMemoryLayout::resultAddress(
    int64_t outputGroup, int64_t token) const
{
    return resultBase_ + outputGroup * seqLen_ + token;
}

int64_t AttentionMemoryLayout::ropeAddress(int64_t token) const
{
    return ropeBase_ + token;
}

} // namespace ftlpu::compiler::schedule
