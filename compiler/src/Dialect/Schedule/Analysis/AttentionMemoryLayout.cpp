#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"

#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::schedule {

AttentionMemoryLayout::AttentionMemoryLayout(const AttentionTaskGraph& op,
    const target::LPUTargetModel& target)
    : target_(target), seqLen_(op.getSeqLen()), hidden_(op.getHidden()),
      kvHeads_(op.getKvHeads()),
      headBlocks_(op.getHeadDim() / target.throughput().mxm_rows)
{
    const auto plan = op.getMemoryPlan();
    const char* names[] = {"query_weight", "key_weight", "value_weight"};
    for (std::size_t i = 0; i < weightBases_.size(); ++i) {
        const auto placement = plan.getAs<mlir::DictionaryAttr>(names[i]);
        weightBases_[i] = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
        if (i == 0) {
            const auto slices = placement.getAs<mlir::ArrayAttr>("slices");
            for (std::size_t lane = 0; lane < weightSlices_.size(); ++lane)
                weightSlices_[lane] =
                    llvm::cast<mlir::IntegerAttr>(slices[lane]).getInt();
        }
    }
    if (const auto input = plan.getAs<mlir::DictionaryAttr>("input")) {
        const auto slices = input.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < activationSlices_.size(); ++i)
            activationSlices_[i] =
                llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
    if (const auto key = plan.getAs<mlir::DictionaryAttr>("key")) {
        const auto slices = key.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < keySlices_.size(); ++i)
            keySlices_[i] =
                llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
    if (const auto weight = plan.getAs<mlir::DictionaryAttr>("output_weight")) {
        outputWeightBase_ = weight.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto slices = weight.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < outputWeightSlices_.size(); ++i)
            outputWeightSlices_[i] = llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
    if (const auto rope = plan.getAs<mlir::DictionaryAttr>("rope")) {
        ropeBase_ = rope.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto slices = rope.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < ropeSlices_.size(); ++i)
            ropeSlices_[i] =
                llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
    if (const auto staging =
            plan.getAs<mlir::DictionaryAttr>("rope_staging")) {
        ropeStagingBase_ =
            staging.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto slices = staging.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < ropeStagingSlices_.size(); ++i)
            ropeStagingSlices_[i] =
                llvm::cast<mlir::IntegerAttr>(slices[i]).getInt();
    }
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
    readPlacement("probability_pack", probabilityPackSlices_, probabilityPackBase_);
    readPlacement("probability_diagonal", probabilityDiagonalSlices_, probabilityDiagonalBase_);
    if (const auto value = plan.getAs<mlir::DictionaryAttr>("value")) {
        valuePackBase_ = value.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto slices = value.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t block = 0; block < valuePackSlices_.size(); ++block)
            for (std::size_t stream = 0;
                 stream < valuePackSlices_[block].size(); ++stream)
                valuePackSlices_[block][stream] =
                    llvm::cast<mlir::IntegerAttr>(
                        slices[block * valuePackSlices_[block].size()
                            + stream]).getInt();
    }
    readPlacement("context", contextSlices_, contextBase_);
    if (const auto result = plan.getAs<mlir::DictionaryAttr>("result"))
        resultBase_ = result.getAs<mlir::IntegerAttr>("base_row").getInt();
}

int64_t AttentionMemoryLayout::weightBase(AttentionProjectionKind projection) const
{
    return weightBases_[static_cast<std::size_t>(projection)];
}

int64_t AttentionMemoryLayout::weightAddress(AttentionProjectionKind projection,
    int64_t outputBlock, int64_t reductionBlock, int64_t localMxm,
    int64_t column) const
{
    const int64_t hiddenBlocks = hidden_ / target_.throughput().mxm_rows;
    return weightBase(projection)
        + ((outputBlock / 4) * hiddenBlocks + reductionBlock) * 8
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
    int64_t head, int64_t reductionBlock, int64_t tokenBlock,
    int64_t phase) const
{
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = seqLen_ / tile;
    const int64_t blocksPerRotaryHalf = std::max<int64_t>(1, headBlocks_ / 2);
    return target_.attention_query_iw_base_row()
        + ((head * blocksPerRotaryHalf
               + reductionBlock % blocksPerRotaryHalf)
              * tokenBlocks
              + tokenBlock)
            * (tile / 8)
        + phase;
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::queryIwSlices(
    int64_t reductionBlock) const
{
    const int64_t blocksPerRotaryHalf = std::max<int64_t>(1, headBlocks_ / 2);
    return target_.attention_query_iw_slices(
        reductionBlock / blocksPerRotaryHalf);
}

int64_t AttentionMemoryLayout::keyAddress(int64_t kvHead,
    int64_t reductionBlock, int64_t keyBlock) const
{
    const int64_t blocksPerRotaryHalf = std::max<int64_t>(1, headBlocks_ / 2);
    return (kvHead * blocksPerRotaryHalf
               + reductionBlock % blocksPerRotaryHalf)
            * seqLen_
        + keyBlock * target_.throughput().mxm_rows;
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::keySlices(
    int64_t reductionBlock) const
{
    const int64_t blocksPerRotaryHalf = std::max<int64_t>(1, headBlocks_ / 2);
    const int64_t bank = reductionBlock / blocksPerRotaryHalf;
    return llvm::ArrayRef<int64_t>(keySlices_).slice(bank * 2, 2);
}

int64_t AttentionMemoryLayout::scoreAccumulatorAddress(int64_t queryHead,
    int64_t queryBlock, int64_t keyBlock) const
{
    return scoreAccumulatorTokenAddress(queryHead, queryBlock,
        keyBlock * target_.throughput().mxm_rows);
}

int64_t AttentionMemoryLayout::scoreAccumulatorTokenAddress(int64_t queryHead,
    int64_t queryBlock, int64_t key) const
{
    (void)queryHead;
    (void)queryBlock;
    // QK waves sharing a physical MXM do not overlap. Keep only one
    // query-tile window in its finite accumulator SRAM and drain every final
    // partial to MEM before the next wave reuses the same rows.
    return key;
}

int64_t AttentionMemoryLayout::scoreAddress(int64_t queryHead,
    int64_t queryBlock, int64_t key) const
{
    const int64_t tokenBlocks =
        seqLen_ / target_.throughput().mxm_rows;
    return scaledScoreBase_
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
        + ((head * headBlocks_ + reductionBlock) * tokenBlocks
              + tokenBlock)
            * tileRows
        + row;
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::valuePackSlices(
    int64_t reductionBlock) const
{
    return valuePackSlices_.at(
        static_cast<std::size_t>(reductionBlock % valuePackSlices_.size()));
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
    return ropeAddress(token, 0);
}

int64_t AttentionMemoryLayout::ropeAddress(
    int64_t token, int64_t frequencyBlock) const
{
    return ropeBase_ + frequencyBlock * seqLen_ + token;
}

int64_t AttentionMemoryLayout::ropeStagingAddress(
    AttentionProjectionKind projection, int64_t head, int64_t half,
    int64_t tokenBlock, int64_t row) const
{
    const int64_t tokenBlocks =
        seqLen_ / target_.throughput().mxm_rows;
    const int64_t rowsPerBlock = target_.throughput().mxm_rows
        / target_.throughput().mxm_block_rows;
    const int64_t projectionIndex =
        projection == AttentionProjectionKind::Query ? 0 : 1;
    return ropeStagingBase_
        + ((((projectionIndex * target_.memory().slices_per_hemisphere
                  + head)
                 * headBlocks_
                + half)
               * tokenBlocks
              + tokenBlock)
                 * rowsPerBlock)
        + row;
}

int64_t AttentionMemoryLayout::ropeProductAddress(
    AttentionProjectionKind projection, int64_t head, int64_t pairBlock,
    int64_t product, int64_t token) const
{
    const int64_t tokenBlocks =
        seqLen_ / target_.throughput().mxm_rows;
    const int64_t rowsPerBlock = target_.throughput().mxm_rows
        / target_.throughput().mxm_block_rows;
    const int64_t stagingRows =
        2 * target_.memory().slices_per_hemisphere * headBlocks_
        * tokenBlocks * rowsPerBlock;
    const int64_t queryHeads = hidden_
        / (headBlocks_ * target_.throughput().mxm_rows);
    const int64_t globalHead = projection == AttentionProjectionKind::Query
        ? head : queryHeads + head;
    const int64_t productRowsPerVector = (seqLen_ + 1) / 2;
    return ropeStagingBase_ + stagingRows
        + ((globalHead * (headBlocks_ / 2) + pairBlock) * 4 + product)
            * productRowsPerVector
        + token / 2;
}

} // namespace ftlpu::compiler::schedule
