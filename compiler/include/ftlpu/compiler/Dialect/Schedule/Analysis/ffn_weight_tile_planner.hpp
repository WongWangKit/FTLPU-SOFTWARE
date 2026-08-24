#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_builders.hpp"
#include "ftlpu/compiler/Dialect/Tensor/Analysis/ffn_weight_tile_plan.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace ftlpu::compiler::schedule {

using tensor::FfnWeightTileKind;
using tensor::FfnWeightTilePage;
using tensor::FfnWeightTilePlan;
using tensor::FfnWeightTileSpan;

struct FfnWeightTileTaskIds {
    ScheduleTaskId prefetch;
    ScheduleTaskId compute;
};

struct FfnWeightTileTaskPlan {
    SchedulePlan tasks;
    llvm::SmallVector<FfnWeightTileTaskIds, 32> page_tasks;
};

FfnWeightTileTaskPlan buildFfnWeightTileTaskPlan(
    const FfnWeightTilePlan& tiles, FfnScheduleShape shape,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
