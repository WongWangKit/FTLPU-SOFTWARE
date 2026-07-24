#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <vector>

namespace ftlpu::compiler::schedule {

struct FfnStreamSegment {
    int64_t rows;
    int64_t stream_base;
};

struct FfnProjectionTileSchedule {
    int64_t m_tile;
    int64_t compute_cycle;
    bool prefetch_next_weight;
    std::vector<std::vector<FfnStreamSegment>> hemisphere_segments;
};

struct FfnProjectionBlockSchedule {
    int64_t index;
    int64_t pair;
    int64_t reduction_block;
    int64_t weight_compute_cycle;
    int64_t dequant_start;
    int64_t weight_buffer;
    bool final_reduction;
    std::vector<FfnProjectionTileSchedule> tiles;
};

struct FfnProjectionTimeline {
    int64_t weight_load_cycles;
    int64_t pipelined_block_interval;
    int64_t weight_block_interval;
    int64_t initial_compute_cycle;
    int64_t final_projection_cycle;
    int64_t accumulator_queue_release;
    int64_t pair_count;
    int64_t m_tile_count;
    std::vector<FfnProjectionBlockSchedule> blocks;
};

struct FfnDownTileSchedule {
    int64_t m_tile;
    int64_t compute_cycle;
    bool prefetch_next_weight;
    std::vector<FfnStreamSegment> segments;
};

struct FfnDownBlockSchedule {
    int64_t index;
    int64_t output_wave;
    int64_t reduction_block;
    int64_t active_hemispheres;
    int64_t weight_compute_cycle;
    int64_t dequant_start;
    int64_t weight_buffer;
    bool final_reduction;
    std::vector<FfnDownTileSchedule> tiles;
};

struct FfnDownProjectionTimeline {
    int64_t phase_start;
    int64_t pair_transition_interval;
    int64_t reduction_block_count;
    int64_t columns_per_wave;
    int64_t wave_count;
    int64_t output_stream_base;
    int64_t first_accumulator_stream;
    int64_t second_accumulator_stream;
    int64_t vxm_queues_per_hemisphere;
    std::vector<FfnDownBlockSchedule> blocks;
};

mlir::FailureOr<FfnProjectionTimeline> planFfnProjectionTimeline(
    FfnScheduleShape shape, llvm::ArrayRef<int64_t> weightSlices,
    const target::LPUTargetModel& target);

mlir::FailureOr<FfnDownProjectionTimeline> planFfnDownProjectionTimeline(
    FfnScheduleShape shape, const FfnProjectionTimeline& projection,
    int64_t lastSwishCycle, llvm::ArrayRef<int64_t> weightSlices,
    llvm::ArrayRef<int64_t> hiddenSlices,
    llvm::ArrayRef<int64_t> resultSlices,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
