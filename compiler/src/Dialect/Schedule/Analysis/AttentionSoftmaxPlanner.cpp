#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_softmax_planner.hpp"

#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_plan.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <string>

namespace ftlpu::compiler::schedule {

mlir::FailureOr<AttentionSoftmaxSchedule> planAttentionSoftmax(
    const AttentionTaskGraph& op,
    const std::vector<AttentionWorkWave>& waves,
    int64_t qkStart, int64_t qkEnd, int64_t qkWaveInterval,
    int64_t qkIwToComputeCycles, bool fused,
    const target::LPUTargetModel& target)
{
    const AttentionMemoryLayout layout(op, target);
    const auto maxRegisterGroup = [&](llvm::ArrayRef<int64_t> slices) {
        return *std::max_element(slices.begin(), slices.end())
            / target.streams().mem_slices_per_register_group;
    };
    int64_t maxScaledGroup = 0;
    int64_t maxExpGroup = 0;
    int64_t maxProbabilityGroup = 0;
    const int64_t scratchPlanes = fused
        ? 2 : target.throughput().mxms_per_hemisphere;
    for (int64_t lane = 0; lane < scratchPlanes; ++lane) {
        // Fused scratch banks are derived from the target instead of stored
        // in the common layout object, so the Tail object layout is stable.
        maxScaledGroup = std::max(
            maxScaledGroup, maxRegisterGroup(fused
                ? layout.fusedScoreSlices(lane)
                : layout.scaledScoreSlices(lane)));
        maxExpGroup = std::max(
            maxExpGroup, maxRegisterGroup(fused
                ? layout.fusedScoreSlices(lane)
                : layout.expScoreSlices(lane)));
    }
    maxProbabilityGroup = std::max(maxProbabilityGroup,
        maxRegisterGroup(layout.probabilityPackSlices()));
    const int64_t pass2Offset = op.getSeqLen()
        + (op.getCausal() ? 4 : 3) + 2 * maxScaledGroup;
    const int64_t pass3Offset = pass2Offset + op.getSeqLen()
        + 4 + 2 * maxExpGroup;
    const int64_t duration = pass3Offset + op.getSeqLen()
        + 2 + maxProbabilityGroup;
    const int64_t hemisphereCount = target.memory().hemispheres;
    if (hemisphereCount != 2) return mlir::failure();
    // The fused schedule itself is legal, but the current global ICU repeat
    // compressor can merge an earlier QKV command with a later fused VXM
    // command and introduce a passive-SR collision. Keep the experimental
    // planner/emitter available, but force the public Fused request through
    // the established Tail fallback until phase-local encoding is available.
    constexpr bool phaseLocalIcuRepeatEncoding = false;
    if (fused && !phaseLocalIcuRepeatEncoding) return mlir::failure();
    constexpr int64_t alusPerWork = 4;
    const int64_t softmaxBanks = fused ? 2 : 1;
    if (hemisphereCount * softmaxBanks * alusPerWork
            > target.throughput().vxm_alus
        || (fused
            && (target.throughput().mxms_per_hemisphere != 1
                || !target.supports_mxm_weight_activation_overlap()
                || target.throughput().mxm_weight_buffers < 2)))
        return mlir::failure();

    AttentionSoftmaxSchedule result;
    result.wave_cycles.resize(waves.size());
    result.work_interval = duration;
    result.end_cycle = fused ? qkStart : qkEnd + 16;
    if (fused) {
        const int64_t tile = target.throughput().mxm_rows;
        const int64_t tokenBlocks = op.getSeqLen() / tile;
        const int64_t headBlocks = op.getHeadDim() / tile;
        const int64_t firstIwOffset =
            target.throughput().mxm_earliest_iw_cycle
            + *target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, 0);
        const int64_t firstResultOffset = firstIwOffset
            + qkIwToComputeCycles
            + (headBlocks - 1) * tokenBlocks
                * target.mxm_block_issue_interval()
            + target.mxm_first_result_latency();
        std::array<std::array<int64_t, 2>, 2> bankReady {};
        for (std::size_t waveIndex = 0; waveIndex < waves.size(); ++waveIndex) {
            const int64_t bank = static_cast<int64_t>(waveIndex % 2);
            for (int64_t hemisphere = 0; hemisphere < hemisphereCount;
                 ++hemisphere) {
                int64_t workCount = 0;
                for (const auto& work : waves[waveIndex].slots)
                    workCount += work && work->hemisphere == hemisphere;
                if (workCount == 0) continue;
                if (workCount != 1) return mlir::failure();
                const int64_t cycle = qkStart
                    + static_cast<int64_t>(waveIndex) * qkWaveInterval
                    + firstResultOffset;
                if (cycle < bankReady[static_cast<std::size_t>(hemisphere)]
                                     [static_cast<std::size_t>(bank)])
                    return mlir::failure();
                result.wave_cycles[waveIndex]
                    [static_cast<std::size_t>(hemisphere)] = cycle;
                bankReady[static_cast<std::size_t>(hemisphere)]
                         [static_cast<std::size_t>(bank)] = cycle + duration;
                result.end_cycle = std::max(result.end_cycle, cycle + duration);
            }
        }
        result.end_cycle += 16;
        return result;
    }

    SchedulePlan tasks;
    LPUResourceModel resources(target);
    std::vector<std::array<std::optional<ScheduleTaskId>, 2>> taskIds(
        waves.size());
    std::array<std::optional<ScheduleTaskId>, 2> previous;

    for (std::size_t waveIndex = 0; waveIndex < waves.size(); ++waveIndex) {
        for (int64_t hemisphere = 0; hemisphere < hemisphereCount;
             ++hemisphere) {
            llvm::SmallVector<ResourceWindow, 32> windows;
            int64_t workCount = 0;
            for (const auto& work : waves[waveIndex].slots) {
                if (!work || work->hemisphere != hemisphere) continue;
                const int64_t workOffset = workCount++ * duration;
                const int64_t aluBase =
                    (hemisphere * target.throughput().mxms_per_hemisphere
                        + work->local_mxm)
                    * alusPerWork;
                for (int64_t alu = 0; alu < alusPerWork; ++alu)
                    windows.push_back(
                        {resources.vxm_alu(aluBase + alu), workOffset, duration});
                const auto reserveSlices = [&](llvm::ArrayRef<int64_t> slices) {
                    for (int64_t slice : slices) {
                        windows.push_back(
                            {resources.mem_read_port(hemisphere, slice), workOffset, duration});
                        windows.push_back(
                            {resources.mem_write_port(hemisphere, slice), workOffset, duration});
                    }
                };
                reserveSlices(layout.scaledScoreSlices(work->local_mxm));
                reserveSlices(layout.expScoreSlices(work->local_mxm));
                reserveSlices(layout.causalMaskSlices(work->local_mxm));
            }
            if (workCount == 0) continue;

            const auto id = tasks.addTask(
                "attention.softmax.wave" + std::to_string(waveIndex) + ".h"
                    + std::to_string(hemisphere),
                ScheduleTaskKind::VxmCompute, ScheduleStage::Softmax,
                qkEnd + 16, workCount * duration, windows);
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
