#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_swish_planner.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>

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
    for (int64_t alu = 0; alu < target.throughput().vxm_alus; ++alu)
        allVxmAlus.push_back({resourceModel.vxm_alu(alu), 0, 1});

    int64_t earliestTaskAnchor = 0;
    int64_t minimumTaskOffset = 0;
    if (!request.tasks.empty()) {
        earliestTaskAnchor = request.tasks.front().earliest_cycle;
        for (const FfnSwishTaskRequest& task : request.tasks) {
            if (task.earliest_cycle < 0 || task.hemisphere < 0
                || task.hemisphere >= target.memory().hemispheres)
                return mlir::failure();
            earliestTaskAnchor =
                std::min(earliestTaskAnchor, task.earliest_cycle);
        }
        for (int64_t slice : target.ffn_gate_temp_slices()) {
            const auto latency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, slice);
            if (!latency) return mlir::failure();
            minimumTaskOffset = std::min(minimumTaskOffset, -*latency);
        }
        for (int64_t slice : target.ffn_up_temp_slices()) {
            const auto latency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, slice);
            if (!latency) return mlir::failure();
            minimumTaskOffset = std::min(minimumTaskOffset, -*latency);
        }
    }
    // Anchors only move forward. An imported interval ending before the first
    // possible task window can never participate in a conflict, and retaining
    // the complete projection history makes long-sequence scheduling needlessly
    // expensive.
    const int64_t earliestRelevantCycle =
        earliestTaskAnchor + minimumTaskOffset;

    for (FfnCycleWindow dequant : request.dequant_windows) {
        if (dequant.end <= dequant.start) return mlir::failure();
        if (!request.tasks.empty() && dequant.end <= earliestRelevantCycle)
            continue;
        llvm::SmallVector<ResourceWindow, 16> windows;
        for (const auto& resource : allVxmAlus)
            windows.push_back(
                {resource.resource, 0, dequant.end - dequant.start});
        resources.block_at(dequant.start, windows);
    }

    for (int64_t hemisphere = 0;
         hemisphere < target.memory().hemispheres; ++hemisphere) {
        llvm::SmallVector<int64_t, 8> tempSlices;
        const auto gateTempSlices = target.ffn_gate_temp_slices();
        const auto upTempSlices = target.ffn_up_temp_slices();
        tempSlices.append(gateTempSlices.begin(), gateTempSlices.end());
        tempSlices.append(upTempSlices.begin(), upTempSlices.end());
        for (FfnCycleWindow busy :
             request.temp_mem_windows[static_cast<std::size_t>(hemisphere)]) {
            if (busy.end <= busy.start) return mlir::failure();
            if (!request.tasks.empty() && busy.end <= earliestRelevantCycle)
                continue;
            llvm::SmallVector<ResourceWindow, 8> windows;
            for (int64_t slice : tempSlices) {
                windows.push_back({resourceModel.mem_icu(
                                       hemisphere, slice, request.temp_bank),
                    0, busy.end - busy.start});
            }
            resources.block_at(busy.start, windows);
        }
    }

    for (const FfnMemResourceWindow& busy : request.mem_windows) {
        if (busy.hemisphere < 0
            || busy.hemisphere >= target.memory().hemispheres
            || busy.end <= busy.start)
            return mlir::failure();
        if (!request.tasks.empty() && busy.end <= earliestRelevantCycle)
            continue;
        const ResourceWindow window{resourceModel.mem_icu(
                                        busy.hemisphere, busy.slice,
                                        busy.bank),
            0, busy.end - busy.start};
        resources.block_at(busy.start, window);
    }

    SchedulePlan plan;
    std::vector<ScheduleTaskId> taskIds;
    taskIds.reserve(request.tasks.size());
    for (std::size_t index = 0; index < request.tasks.size(); ++index) {
        const auto& task = request.tasks[index];
        llvm::SmallVector<ResourceWindow, 24> windows;
        for (const auto& resource : allVxmAlus)
            windows.push_back(
                {resource.resource, 0, request.tile_rows + 5});
        for (int64_t slice : target.ffn_gate_temp_slices()) {
            const auto latency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, slice);
            if (!latency) return mlir::failure();
            windows.push_back({resourceModel.mem_icu(
                                   task.hemisphere, slice, request.temp_bank),
                -*latency, request.tile_rows});
        }
        for (int64_t slice : target.ffn_up_temp_slices()) {
            const auto latency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, slice);
            if (!latency) return mlir::failure();
            windows.push_back({resourceModel.mem_icu(
                                   task.hemisphere, slice, request.temp_bank),
                -*latency, request.tile_rows});
        }
        const int64_t swishLatency = ffn_swish_result_latency(target);
        for (int64_t hemisphere = 0;
             hemisphere < target.memory().hemispheres; ++hemisphere) {
            for (int64_t slice : request.hidden_slices) {
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::VxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::East, slice);
                if (!latency) return mlir::failure();
                const int64_t offset = swishLatency + *latency;
                windows.push_back({resourceModel.mem_icu(
                                       hemisphere, slice,
                                       request.hidden_bank),
                    offset, request.tile_rows});
            }
        }
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
