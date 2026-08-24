#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace ftlpu::compiler::tensor {

struct FfnWeightShape {
    int64_t m;
    int64_t k;
    int64_t hidden;
    int64_t n;
};

enum class FfnWeightTileKind : std::uint8_t { Gate, Up, Down };

struct FfnWeightTileSpan {
    FfnWeightTileKind kind;
    int64_t page_base_row;
    int64_t slice_group_begin;
    int64_t slice_group_count;
    int64_t items_per_slice_group;
    int64_t output_wave_begin;
    int64_t output_wave_count;
    int64_t reduction_block_begin;
    int64_t reduction_block_count;
    int64_t output_blocks_per_hemisphere;
    int64_t rows_per_slice;
};

struct FfnWeightTilePage {
    int64_t index;
    int64_t bank;
    int64_t base_row;
    int64_t rows_per_slice;
    int64_t transfer_vectors;
    int64_t transfer_cycles;
    llvm::SmallVector<FfnWeightTileSpan, 2> spans;
};

struct FfnWeightTilePlan {
    int64_t bank_rows;
    int64_t bank_bytes;
    int64_t weight_load_slice_count;
    int64_t weight_storage_slice_count;
    int64_t slice_group_count;
    int64_t projection_wave_count;
    int64_t projection_waves_per_page;
    int64_t projection_slice_groups_per_role;
    int64_t projection_waves_per_slice_group;
    int64_t down_wave_count;
    int64_t down_reduction_blocks_per_page;
    int64_t down_reduction_blocks_per_slice_group;
    int64_t minimum_hidden_slices;
    llvm::SmallVector<FfnWeightTilePage, 32> pages;
};

mlir::FailureOr<FfnWeightTilePlan> planFfnWeightTiles(
    FfnWeightShape shape, const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::tensor
