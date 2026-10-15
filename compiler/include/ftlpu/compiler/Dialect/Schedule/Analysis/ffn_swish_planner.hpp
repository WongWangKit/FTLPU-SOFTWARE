#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ftlpu::compiler::schedule {

struct FfnCycleWindow {
    int64_t start;
    int64_t end;
};

struct FfnSwishTaskRequest {
    int64_t earliest_cycle;
    int64_t hemisphere;
};

struct FfnMemResourceWindow {
    int64_t hemisphere;
    int64_t slice;
    int64_t bank;
    int64_t start;
    int64_t end;
};

struct FfnSwishScheduleRequest {
    std::vector<FfnSwishTaskRequest> tasks;
    std::vector<FfnCycleWindow> dequant_windows;
    std::array<std::vector<FfnCycleWindow>, 2> temp_mem_windows;
    std::vector<FfnMemResourceWindow> mem_windows;
    std::vector<int64_t> hidden_slices;
    int64_t hidden_bank{0};
    int64_t temp_bank{0};
    int64_t tile_rows;
};

inline int64_t ffn_swish_result_latency(const target::LPUTargetModel& target)
{
    return 18 + (target.throughput().vxm_alus - 8);
}

mlir::FailureOr<std::vector<int64_t>> planFfnSwishCycles(
    const FfnSwishScheduleRequest& request,
    const target::LPUTargetModel& target);

} // namespace ftlpu::compiler::schedule
