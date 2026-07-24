#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_softmax_planner.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <string>

namespace ftlpu::compiler::schedule {

mlir::FailureOr<AttentionSoftmaxSchedule> planAttentionSoftmax(
    stream::AttentionOp op, const std::vector<AttentionWorkWave>& waves,
    int64_t qkEnd, const target::LPUTargetModel& target)
{
    const AttentionMemoryLayout layout(op, target);
    const int64_t duration = 3 * op.getSeqLen() + 20;
    const int64_t hemisphereCount = target.memory().hemispheres;
    if (hemisphereCount != 2) return mlir::failure();

    SchedulePlan tasks;
    LPUResourceModel resources(target);
    std::vector<std::array<std::optional<ScheduleTaskId>, 2>> taskIds(
        waves.size());
    std::array<std::optional<ScheduleTaskId>, 2> previous;

    for (std::size_t waveIndex = 0; waveIndex < waves.size(); ++waveIndex) {
        for (int64_t hemisphere = 0; hemisphere < hemisphereCount;
             ++hemisphere) {
            llvm::SmallVector<ResourceWindow, 32> windows;
            bool hasWork = false;
            for (const auto& work : waves[waveIndex].slots) {
                if (!work || work->hemisphere != hemisphere) continue;
                hasWork = true;
                const int64_t aluBase = hemisphere * 6 + work->local_mxm * 3;
                for (int64_t alu = 0; alu < 3; ++alu)
                    windows.push_back(
                        {resources.vxm_alu(aluBase + alu), 0, duration});
                const auto reserveSlices = [&](llvm::ArrayRef<int64_t> slices) {
                    for (int64_t slice : slices)
                        windows.push_back(
                            {resources.mem_slice(hemisphere, slice), 0, duration});
                };
                reserveSlices(layout.scaledScoreSlices(work->local_mxm));
                reserveSlices(layout.expScoreSlices(work->local_mxm));
                reserveSlices(layout.causalMaskSlices(work->local_mxm));
                reserveSlices(layout.probabilitySlices(work->local_mxm));
            }
            if (!hasWork) continue;

            const auto id = tasks.addTask(
                "attention.softmax.wave" + std::to_string(waveIndex) + ".h"
                    + std::to_string(hemisphere),
                ScheduleTaskKind::VxmCompute, ScheduleStage::Softmax,
                qkEnd + 16, duration, windows);
            if (previous[static_cast<std::size_t>(hemisphere)]
                && mlir::failed(tasks.addDependency(
                    *previous[static_cast<std::size_t>(hemisphere)], id)))
                return mlir::failure();
            previous[static_cast<std::size_t>(hemisphere)] = id;
            taskIds[waveIndex][static_cast<std::size_t>(hemisphere)] = id;
        }
    }

    ResourceScheduler scheduler;
    auto assignment = tasks.schedule(scheduler);
    if (mlir::failed(assignment)) return mlir::failure();

    AttentionSoftmaxSchedule result;
    result.wave_cycles.resize(waves.size());
    result.end_cycle = qkEnd + 16;
    for (std::size_t waveIndex = 0; waveIndex < waves.size(); ++waveIndex) {
        for (int64_t hemisphere = 0; hemisphere < hemisphereCount;
             ++hemisphere) {
            const auto id =
                taskIds[waveIndex][static_cast<std::size_t>(hemisphere)];
            if (!id) continue;
            const ScheduledTask& scheduled = (*assignment)[*id];
            result.wave_cycles[waveIndex][static_cast<std::size_t>(hemisphere)] =
                scheduled.cycle;
            result.end_cycle = std::max(result.end_cycle, scheduled.end_cycle);
        }
    }
    result.end_cycle += 16;
    return result;
}

} // namespace ftlpu::compiler::schedule
