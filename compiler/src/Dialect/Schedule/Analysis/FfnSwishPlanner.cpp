#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_swish_planner.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

namespace ftlpu::compiler::schedule {

mlir::FailureOr<std::vector<int64_t>> planFfnSwishCycles(
    const FfnSwishScheduleRequest& request,
    const target::LPUTargetModel& target)
{
    if (request.tile_rows <= 0
        || target.memory().hemispheres != 2)
        return mlir::failure();

    ResourceScheduler resources;
    LPUResourceModel resourceModel(target);
    llvm::SmallVector<ResourceWindow, 16> allVxmAlus;
    for (int64_t alu = 0; alu < 16; ++alu)
        allVxmAlus.push_back({resourceModel.vxm_alu(alu), 0, 1});

    for (FfnCycleWindow dequant : request.dequant_windows) {
        if (dequant.end <= dequant.start) return mlir::failure();
        llvm::SmallVector<ResourceWindow, 16> windows;
        for (const auto& resource : allVxmAlus)
            windows.push_back(
                {resource.resource, 0, dequant.end - dequant.start});
        resources.reserve_at(dequant.start, windows);
    }

    for (int64_t hemisphere = 0;
         hemisphere < target.memory().hemispheres; ++hemisphere) {
        llvm::SmallVector<int64_t, 8> tempSlices;
        tempSlices.append(
            target.memory().w8a16_fused_gate_temp_slices.begin(),
            target.memory().w8a16_fused_gate_temp_slices.end());
        tempSlices.append(
            target.memory().w8a16_fused_up_temp_slices.begin(),
            target.memory().w8a16_fused_up_temp_slices.end());
        for (FfnCycleWindow busy :
             request.temp_mem_windows[static_cast<std::size_t>(hemisphere)]) {
            if (busy.end <= busy.start) return mlir::failure();
            llvm::SmallVector<ResourceWindow, 8> windows;
            for (int64_t slice : tempSlices)
                windows.push_back({resourceModel.mem_slice(hemisphere, slice),
                    0, busy.end - busy.start});
            resources.reserve_at(busy.start, windows);
        }
    }

    SchedulePlan plan;
    std::vector<ScheduleTaskId> taskIds;
    taskIds.reserve(request.tasks.size());
    for (std::size_t index = 0; index < request.tasks.size(); ++index) {
        const auto& task = request.tasks[index];
        if (task.earliest_cycle < 0 || task.hemisphere < 0
            || task.hemisphere >= target.memory().hemispheres)
            return mlir::failure();
        llvm::SmallVector<ResourceWindow, 24> windows;
        for (const auto& resource : allVxmAlus)
            windows.push_back(
                {resource.resource, 0, request.tile_rows + 5});
        for (int64_t slice :
             target.memory().w8a16_fused_gate_temp_slices)
            windows.push_back({resourceModel.mem_slice(task.hemisphere, slice),
                0, request.tile_rows});
        for (int64_t slice :
             target.memory().w8a16_fused_up_temp_slices)
            windows.push_back({resourceModel.mem_slice(task.hemisphere, slice),
                0, request.tile_rows});
        taskIds.push_back(plan.addTask(
            llvm::formatv("ffn.swish.{0}", index).str(),
            ScheduleTaskKind::VxmCompute, ScheduleStage::FfnSwish,
            task.earliest_cycle, request.tile_rows, windows));
    }

    auto assignment = plan.schedule(resources);
    if (mlir::failed(assignment)) return mlir::failure();
    std::vector<int64_t> cycles;
    cycles.reserve(taskIds.size());
    for (ScheduleTaskId id : taskIds)
        cycles.push_back((*assignment)[id].cycle);
    return cycles;
}

} // namespace ftlpu::compiler::schedule
