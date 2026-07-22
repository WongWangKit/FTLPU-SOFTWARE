#pragma once

#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
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
    AttentionMemoryLayout(stream::AttentionOp op,
        const target::LPUTargetModel& target);

    int64_t weightAddress(AttentionProjectionKind projection, int64_t head,
        int64_t reductionBlock, int64_t localMxm, int64_t column) const;
    int64_t activationAddress(int64_t reductionBlock, int64_t tokenBlock) const;
    int64_t projectionAddress(AttentionProjectionKind projection, int64_t head,
        int64_t tokenBlock) const;
    int64_t queryIwAddress(int64_t head, int64_t tokenBlock, int64_t phase) const;
    int64_t keyAddress(int64_t kvHead, int64_t keyBlock) const;
    int64_t scoreAddress(int64_t queryHead, int64_t queryBlock,
        int64_t keyBlock) const;
    int64_t scoreTokenAddress(int64_t queryHead, int64_t queryBlock,
        int64_t key) const;
    int64_t scaledScoreAddress(int64_t key) const { return scaledScoreBase_ + key; }
    int64_t expScoreAddress(int64_t key) const { return expScoreBase_ + key; }
    int64_t probabilityAddress(int64_t queryHead, int64_t queryBlock,
        int64_t key) const;
    int64_t ropeAddress(int64_t token) const;

    llvm::ArrayRef<int64_t> weightSlices() const { return weightSlices_; }
    llvm::ArrayRef<int64_t> activationSlices() const { return activationSlices_; }
    llvm::ArrayRef<int64_t> ropeSlices() const { return ropeSlices_; }
    llvm::ArrayRef<int64_t> scaledScoreSlices() const { return scaledScoreSlices_; }
    llvm::ArrayRef<int64_t> expScoreSlices() const { return expScoreSlices_; }
    llvm::ArrayRef<int64_t> probabilitySlices() const { return probabilitySlices_; }

private:
    int64_t weightBase(AttentionProjectionKind projection) const;

    const target::LPUTargetModel& target_;
    int64_t seqLen_ = 0;
    int64_t hidden_ = 0;
    int64_t kvHeads_ = 0;
    std::array<int64_t, 3> weightBases_ {};
    std::array<int64_t, 8> weightSlices_ {0, 4, 8, 12, 16, 20, 24, 28};
    std::array<int64_t, 4> activationSlices_ {32, 33, 34, 35};
    std::array<int64_t, 4> ropeSlices_ {4, 5, 6, 7};
    std::array<int64_t, 4> scaledScoreSlices_ {8, 9, 10, 11};
    std::array<int64_t, 4> expScoreSlices_ {12, 13, 14, 15};
    std::array<int64_t, 2> probabilitySlices_ {16, 17};
    int64_t ropeBase_ = 7000;
    int64_t scaledScoreBase_ = 0;
    int64_t expScoreBase_ = 0;
    int64_t probabilityBase_ = 0;
};

} // namespace ftlpu::compiler::schedule
