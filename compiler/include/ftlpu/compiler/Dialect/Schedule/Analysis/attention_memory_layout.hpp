#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_task_graph.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <cstdint>

namespace ftlpu::compiler::schedule {

enum class AttentionProjectionKind : std::uint8_t {
    Query,
    Key,
    Value,
};

class AttentionMemoryLayout {
public:
    AttentionMemoryLayout(const AttentionTaskGraph& graph,
        const target::LPUTargetModel& target);

    int64_t weightAddress(AttentionProjectionKind projection,
        int64_t outputBlock,
        int64_t reductionBlock, int64_t localMxm, int64_t column) const;
    int64_t activationAddress(int64_t reductionBlock, int64_t tokenBlock) const;
    int64_t projectionAddress(AttentionProjectionKind projection, int64_t head,
        int64_t tokenBlock) const;
    int64_t queryIwAddress(int64_t head, int64_t reductionBlock,
        int64_t tokenBlock, int64_t phase) const;
    llvm::ArrayRef<int64_t> queryIwSlices(int64_t reductionBlock) const;
    int64_t queryIwBank(int64_t reductionBlock) const;
    int64_t keyAddress(int64_t kvHead, int64_t reductionBlock,
        int64_t keyBlock) const;
    llvm::ArrayRef<int64_t> keySlices(int64_t reductionBlock) const;
    int64_t keyBank(int64_t reductionBlock) const;
    int64_t scoreAccumulatorAddress(int64_t queryHead, int64_t queryBlock,
        int64_t keyBlock) const;
    int64_t scoreAccumulatorTokenAddress(int64_t queryHead, int64_t queryBlock,
        int64_t key) const;
    int64_t scoreAddress(int64_t queryHead, int64_t queryBlock,
        int64_t key) const;
    int64_t expScoreAddress(int64_t key) const { return expScoreBase_ + key; }
    int64_t causalMaskAddress(int64_t localKey) const {
        return causalMaskBase_ + localKey - 1;
    }
    int64_t probabilityPackAddress(int64_t queryHead, int64_t queryBlock,
        int64_t keyBlock) const;
    int64_t probabilityDiagonalAddress(int64_t queryHead, int64_t queryBlock,
        int64_t keyBlock, int64_t diagonal) const;
    int64_t valuePackAddress(int64_t head, int64_t reductionBlock,
        int64_t tokenBlock, int64_t row) const;
    int64_t contextAddress(int64_t queryHead, int64_t token) const;
    int64_t outputWeightAddress(int64_t outputGroup,
        int64_t reductionBlock, int64_t column) const;
    int64_t resultAddress(int64_t outputGroup, int64_t token) const;
    int64_t ropeAddress(int64_t token) const;
    int64_t ropeAddress(int64_t token, int64_t frequencyBlock) const;
    int64_t ropeStagingAddress(AttentionProjectionKind projection,
        int64_t head, int64_t half, int64_t tokenBlock,
        int64_t row) const;
    int64_t ropeProductAddress(AttentionProjectionKind projection,
        int64_t head, int64_t pairBlock, int64_t product,
        int64_t token) const;

    llvm::ArrayRef<int64_t> weightSlices() const { return weightSlices_; }
    llvm::ArrayRef<int64_t> outputWeightSlices() const { return outputWeightSlices_; }
    llvm::ArrayRef<int64_t> weightSlices(
        AttentionProjectionKind projection, int64_t outputBlock) const;
    llvm::ArrayRef<int64_t> outputWeightSlices(int64_t outputGroup) const;
    int64_t weightPage(AttentionProjectionKind projection) const;
    int64_t weightBank(AttentionProjectionKind projection) const;
    int64_t outputWeightPage() const;
    int64_t outputWeightBank() const;
    llvm::ArrayRef<int64_t> activationSlices() const { return activationSlices_; }
    llvm::ArrayRef<int64_t> ropeSlices() const { return ropeSlices_; }
    llvm::ArrayRef<int64_t> ropeStagingSlices() const {
        return ropeStagingSlices_;
    }
    llvm::ArrayRef<int64_t> ropeProductSlices() const {
        return ropeProductSlices_;
    }
    llvm::ArrayRef<int64_t> scaledScoreSlices(int64_t localMxm) const {
        return scaledScoreSlices_.at(static_cast<std::size_t>(localMxm));
    }
    llvm::ArrayRef<int64_t> expScoreSlices(int64_t localMxm) const {
        return expScoreSlices_.at(static_cast<std::size_t>(localMxm));
    }
    llvm::ArrayRef<int64_t> causalMaskSlices(int64_t localMxm) const {
        return causalMaskSlices_.at(static_cast<std::size_t>(localMxm));
    }
    // Optional overlays are computed from target topology and never perturb
    // the allocator-selected Tail planes.
    std::array<int64_t, 4> fusedScoreSlices(int64_t bank) const {
        const int64_t begin = target_.memory().slices_per_hemisphere
            - 16 + bank * 4;
        return {begin, begin + 1, begin + 2, begin + 3};
    }
    std::array<int64_t, 2> fusedCausalMaskSlices(int64_t bank) const {
        const int64_t begin = target_.memory().slices_per_hemisphere
            - 4 + bank * 2;
        return {begin, begin + 1};
    }
    llvm::ArrayRef<int64_t> probabilityPackSlices() const { return probabilityPackSlices_; }
    llvm::ArrayRef<int64_t> probabilityDiagonalSlices() const { return probabilityDiagonalSlices_; }
    llvm::ArrayRef<int64_t> valuePackSlices(int64_t reductionBlock) const;
    llvm::ArrayRef<int64_t> contextSlices() const { return contextSlices_; }

private:
    struct WeightPagingLayout {
        bool enabled = false;
        int64_t page = -1;
        int64_t bank = 0;
        int64_t groupBase = 0;
        int64_t groupCount = 0;
        int64_t itemsPerGroup = 0;
        llvm::SmallVector<int64_t, 32> storageSlices;
    };

    int64_t weightBase(AttentionProjectionKind projection) const;
    llvm::ArrayRef<int64_t> pagingSlices(
        const WeightPagingLayout& paging, int64_t item,
        llvm::ArrayRef<int64_t> fallback) const;

    const target::LPUTargetModel& target_;
    int64_t seqLen_ = 0;
    int64_t hidden_ = 0;
    int64_t queryHeads_ = 0;
    int64_t kvHeads_ = 0;
    int64_t headBlocks_ = 0;
    std::array<int64_t, 3> weightBases_ {};
    std::array<WeightPagingLayout, 3> weightPaging_ {};
    int64_t outputWeightBase_ = 0;
    WeightPagingLayout outputWeightPaging_ {};
    std::array<int64_t, 8> weightSlices_ {0, 4, 8, 12, 16, 20, 24, 28};
    std::array<int64_t, 8> outputWeightSlices_ {0, 4, 8, 12, 2, 18, 24, 28};
    std::array<int64_t, 4> activationSlices_ {32, 33, 34, 35};
    std::array<int64_t, 4> keySlices_ {0, 1, 2, 3};
    std::array<std::array<int64_t, 16>, 2> queryIwSlices_ {};
    std::array<int64_t, 2> queryIwBanks_ {0, 0};
    std::array<int64_t, 2> keyBanks_ {0, 0};
    std::array<int64_t, 4> ropeSlices_ {};
    std::array<int64_t, 16> ropeStagingSlices_ {};
    std::array<int64_t, 16> ropeProductSlices_ {};
    // Each local MXM owns an independent softmax scratch plane. This permits
    // the two work items in a hemisphere wave to use VXM concurrently.
    std::array<std::array<int64_t, 4>, 2> scaledScoreSlices_ {{{{8, 9, 10, 11}}, {{0, 1, 2, 3}}}};
    std::array<std::array<int64_t, 4>, 2> expScoreSlices_ {{{{12, 13, 14, 15}}, {{4, 5, 6, 7}}}};
    std::array<std::array<int64_t, 2>, 2> causalMaskSlices_ {{{{24, 25}}, {{20, 21}}}};
    std::array<int64_t, 16> probabilityPackSlices_ {
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35};
    std::array<int64_t, 16> probabilityDiagonalSlices_ {
        0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33};
    std::array<std::array<int64_t, 16>, 2> valuePackSlices_ {{
        {{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
        {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
    }};
    std::array<int64_t, 8> contextSlices_ {44, 45, 46, 47, 48, 49, 50, 51};
    int64_t ropeBase_ = 7000;
    int64_t ropeStagingBase_ = 0;
    int64_t ropeProductBase_ = 0;
    int64_t queryIwBase_ = 7600;
    int64_t scaledScoreBase_ = 0;
    int64_t expScoreBase_ = 0;
    int64_t causalMaskBase_ = 8128;
    int64_t probabilityPackBase_ = 6000;
    int64_t probabilityDiagonalBase_ = 7000;
    int64_t valuePackBase_ = 7800;
    int64_t contextBase_ = 2000;
    int64_t resultBase_ = 0;
};

} // namespace ftlpu::compiler::schedule
