#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"

#include "mlir/IR/BuiltinAttributes.h"

namespace ftlpu::compiler::schedule {

AttentionMemoryLayout::AttentionMemoryLayout(const AttentionTaskGraph& op,
    const target::LPUTargetModel& target)
    : target_(target), seqLen_(op.getSeqLen()), hidden_(op.getHidden()),
      queryHeads_(op.getQueryHeads()), kvHeads_(op.getKvHeads()),
      headBlocks_(op.getHeadDim() / target.throughput().mxm_rows)
{
    const auto plan = op.getMemoryPlan();
    for (std::size_t group = 0; group < queryIwSlices_.size(); ++group)
        queryIwSlices_[group] = target_.attention_query_iw_slices(
            static_cast<int64_t>(group));
    const auto readPaging = [&](mlir::DictionaryAttr placement,
                                WeightPagingLayout& paging) {
        const auto paged = placement.getAs<mlir::BoolAttr>("paged_weight");
        if (!paged || !paged.getValue()) return;
        paging.enabled = true;
        paging.page = 0;
        paging.bank = placement.getAs<mlir::IntegerAttr>("bank").getInt();
        paging.groupBase = placement
            .getAs<mlir::IntegerAttr>("page_role_group_base").getInt();
        paging.groupCount = placement
            .getAs<mlir::IntegerAttr>("page_role_group_count").getInt();
        paging.itemsPerGroup = placement
            .getAs<mlir::IntegerAttr>(
                "page_items_per_slice_group").getInt();
        for (mlir::Attribute slice :
             placement.getAs<mlir::ArrayAttr>("page_storage_slices"))
            paging.storageSlices.push_back(
                llvm::cast<mlir::IntegerAttr>(slice).getInt());
    };
    const char* names[] = {"query_weight", "key_weight", "value_weight"};
    for (std::size_t i = 0; i < weightBases_.size(); ++i) {
        const auto placement = plan.getAs<mlir::DictionaryAttr>(names[i]);
        weightBases_[i] = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
        readPaging(placement, weightPaging_[i]);
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
        const int64_t bank = key.getAs<mlir::IntegerAttr>("bank").getInt();
        keyBanks_.fill(bank);
    }
    if (const auto weight = plan.getAs<mlir::DictionaryAttr>("output_weight")) {
        outputWeightBase_ = weight.getAs<mlir::IntegerAttr>("base_row").getInt();
        readPaging(weight, outputWeightPaging_);
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
    if (const auto product =
            plan.getAs<mlir::DictionaryAttr>("rope_product")) {
        ropeProductBase_ =
            product.getAs<mlir::IntegerAttr>("base_row").getInt();
        const auto slices = product.getAs<mlir::ArrayAttr>("slices");
        for (std::size_t i = 0; i < ropeProductSlices_.size(); ++i)
            ropeProductSlices_[i] =
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
    if (const auto query = plan.getAs<mlir::DictionaryAttr>("query")) {
        queryIwBase_ = query.getAs<mlir::IntegerAttr>("base_row").getInt();
        const int64_t baseBank =
            query.getAs<mlir::IntegerAttr>("bank").getInt();
        queryIwBanks_.fill(baseBank);
        if (target_.uses_dedicated_slice_roles()
            && target_.memory().banks_per_slice >= 2)
            queryIwBanks_[1] = (baseBank + 1)
                % target_.memory().banks_per_slice;
    }
    const auto readSegments = [&](llvm::StringRef name, bool query) {
        const auto placement = plan.getAs<mlir::DictionaryAttr>(name);
        const auto segments = placement
            ? placement.getAs<mlir::ArrayAttr>("segments") : mlir::ArrayAttr{};
        if (!segments) return;
        const int64_t blocksPerHalf = std::max<int64_t>(1, headBlocks_ / 2);
        for (mlir::Attribute value : segments) {
            const auto segment = llvm::dyn_cast<mlir::DictionaryAttr>(value);
            const auto begin = segment
                ? segment.getAs<mlir::IntegerAttr>("reduction_begin")
                : mlir::IntegerAttr{};
            const auto bank = segment
                ? segment.getAs<mlir::IntegerAttr>("bank")
                : mlir::IntegerAttr{};
            const auto slices = segment
                ? segment.getAs<mlir::ArrayAttr>("slices")
                : mlir::ArrayAttr{};
            if (!begin || !bank || !slices) continue;
            const std::size_t group = static_cast<std::size_t>(
                std::min<int64_t>(1, begin.getInt() / blocksPerHalf));
            if (query) {
                if (slices.size() != queryIwSlices_[group].size()) continue;
                queryIwBanks_[group] = bank.getInt();
                for (std::size_t index = 0; index < slices.size(); ++index)
                    queryIwSlices_[group][index] =
                        llvm::cast<mlir::IntegerAttr>(slices[index]).getInt();
            } else {
                if (slices.size() != 2) continue;
                keyBanks_[group] = bank.getInt();
                for (std::size_t index = 0; index < 2; ++index)
                    keySlices_[group * 2 + index] =
                        llvm::cast<mlir::IntegerAttr>(slices[index]).getInt();
            }
        }
    };
    readSegments("query", true);
    readSegments("key", false);
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
    const auto& paging = weightPaging_[static_cast<std::size_t>(projection)];
    const int64_t item = outputBlock / 4;
    const int64_t localItem = paging.enabled
        ? item % paging.itemsPerGroup : item;
    return weightBase(projection)
        + (localItem * hiddenBlocks + reductionBlock) * 8
        + localMxm * 4 + column;
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::pagingSlices(
    const WeightPagingLayout& paging, int64_t item,
    llvm::ArrayRef<int64_t> fallback) const
{
    if (!paging.enabled) return fallback;
    const int64_t group = paging.groupBase
        + item / paging.itemsPerGroup;
    const int64_t groupSize = static_cast<int64_t>(fallback.size());
    if (group < paging.groupBase
        || group >= paging.groupBase + paging.groupCount
        || (group + 1) * groupSize
            > static_cast<int64_t>(paging.storageSlices.size()))
        return {};
    return llvm::ArrayRef<int64_t>(paging.storageSlices)
        .slice(group * groupSize, groupSize);
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::weightSlices(
    AttentionProjectionKind projection, int64_t outputBlock) const
{
    return pagingSlices(
        weightPaging_[static_cast<std::size_t>(projection)],
        outputBlock / 4, weightSlices_);
}

int64_t AttentionMemoryLayout::weightPage(
    AttentionProjectionKind projection) const
{
    return weightPaging_[static_cast<std::size_t>(projection)].page;
}

int64_t AttentionMemoryLayout::weightBank(
    AttentionProjectionKind projection) const
{
    return weightPaging_[static_cast<std::size_t>(projection)].bank;
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
    return queryIwBase_
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
    const std::size_t group = static_cast<std::size_t>(
        std::min<int64_t>(1, reductionBlock / blocksPerRotaryHalf));
    return queryIwSlices_[group];
}

int64_t AttentionMemoryLayout::queryIwBank(int64_t reductionBlock) const
{
    const int64_t blocksPerRotaryHalf = std::max<int64_t>(1, headBlocks_ / 2);
    const std::size_t group = static_cast<std::size_t>(
        std::min<int64_t>(1, reductionBlock / blocksPerRotaryHalf));
    return queryIwBanks_[group];
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

int64_t AttentionMemoryLayout::keyBank(int64_t reductionBlock) const
{
    const int64_t blocksPerRotaryHalf = std::max<int64_t>(1, headBlocks_ / 2);
    const std::size_t group = static_cast<std::size_t>(
        std::min<int64_t>(1, reductionBlock / blocksPerRotaryHalf));
    return keyBanks_[group];
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
    const int64_t localGroup = outputWeightPaging_.enabled
        ? outputGroup % outputWeightPaging_.itemsPerGroup : outputGroup;
    return outputWeightBase_
        + (localGroup * reductionBlocks + reductionBlock) * 4 + column;
}

llvm::ArrayRef<int64_t> AttentionMemoryLayout::outputWeightSlices(
    int64_t outputGroup) const
{
    return pagingSlices(outputWeightPaging_, outputGroup,
        outputWeightSlices_);
}

int64_t AttentionMemoryLayout::outputWeightPage() const
{
    return outputWeightPaging_.page;
}

int64_t AttentionMemoryLayout::outputWeightBank() const
{
    return outputWeightPaging_.bank;
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
    return ropeStagingBase_
        + ((head * headBlocks_ + half) % 4) * seqLen_
        + tokenBlock * target_.throughput().mxm_rows + row;
}

int64_t AttentionMemoryLayout::ropeProductAddress(
    AttentionProjectionKind projection, int64_t head, int64_t pairBlock,
    int64_t product, int64_t token) const
{
    const int64_t globalHead = projection == AttentionProjectionKind::Query
        ? head : queryHeads_ + head;
    const int64_t productRowsPerVector = (seqLen_ + 1) / 2;
    return ropeProductBase_
        + ((globalHead * (headBlocks_ / 2) + pairBlock) * 4 + product)
            * productRowsPerVector
        + token / 2;
}

} // namespace ftlpu::compiler::schedule
