#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstdint>

namespace ftlpu::compiler::tensor {

enum class AttentionWeightTileKind : std::uint8_t {
    Query,
    Key,
    Value,
    Output,
};

struct AttentionWeightTilePlacement {
    AttentionWeightTileKind kind;
    int64_t bank;
    int64_t page_index;
    int64_t base_row;
    int64_t rows;
    int64_t slice_group_begin;
    int64_t slice_group_count;
    int64_t items_per_slice_group;
    int64_t item_count;
    int64_t rows_per_item;
    int64_t transfer_cycles;
};

struct AttentionWeightTilePlan {
    int64_t bank_rows;
    int64_t load_slice_count;
    int64_t storage_slice_count;
    int64_t slice_group_count;
    int64_t transfer_phase_count;
    llvm::SmallVector<int64_t, 32> storage_slices;
    std::array<AttentionWeightTilePlacement, 4> placements;

    const AttentionWeightTilePlacement& get(
        AttentionWeightTileKind kind) const {
        return placements[static_cast<std::size_t>(kind)];
    }
};

mlir::FailureOr<AttentionWeightTilePlan> planAttentionWeightTiles(
    int64_t hidden, int64_t queryHeads, int64_t kvHeads,
    int64_t headDim, int64_t initialBank,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::tensor
