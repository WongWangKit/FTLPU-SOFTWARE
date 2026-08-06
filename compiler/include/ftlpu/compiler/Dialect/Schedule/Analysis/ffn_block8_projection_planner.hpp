#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace ftlpu::compiler::schedule {

struct FfnBlock8ReductionSchedule {
    int64_t reduction;
    int64_t weight_buffer;
    int64_t load_cycle;
    llvm::SmallVector<int64_t> compute_cycles;
    llvm::SmallVector<int64_t> activation_stream_bases;
};

struct FfnBlock8ProjectionSchedule {
    llvm::SmallVector<FfnBlock8ReductionSchedule> reductions;
    int64_t end_cycle;
};

mlir::FailureOr<FfnBlock8ProjectionSchedule>
planFfnBlock8ProjectionSchedule(int64_t reductionBlocks,
    int64_t tokenBlocks, int64_t startCycle,
    llvm::ArrayRef<int64_t> gateWeightSlices,
    llvm::ArrayRef<int64_t> upWeightSlices,
    llvm::ArrayRef<int64_t> activationSlices,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
