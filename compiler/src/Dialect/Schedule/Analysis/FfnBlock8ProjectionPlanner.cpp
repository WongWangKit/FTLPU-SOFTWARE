#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_block8_projection_planner.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <utility>

namespace ftlpu::compiler::schedule {

mlir::FailureOr<FfnBlock8ProjectionSchedule>
planFfnBlock8ProjectionSchedule(int64_t reductionBlocks,
    int64_t tokenBlocks, int64_t startCycle,
    llvm::ArrayRef<int64_t> gateWeightSlices,
    llvm::ArrayRef<int64_t> upWeightSlices,
    llvm::ArrayRef<int64_t> activationSlices,
    const target::LPUTargetModel& target)
{
    const auto& throughput = target.throughput();
    const int64_t blockIssues = throughput.mxm_rows
        / throughput.mxm_block_rows;
    const int64_t serializedWeightLoadCycles = throughput.tile_rows
        * throughput.mxms_per_hemisphere;
    if (reductionBlocks <= 0 || tokenBlocks <= 0 || startCycle < 0
        || blockIssues <= 0 || serializedWeightLoadCycles <= 0
        || throughput.mxm_weight_buffers <= 0
        || throughput.mxm_block_group_interval < blockIssues)
        return mlir::failure();

    const auto areDisjoint = [](llvm::ArrayRef<int64_t> lhs,
                                llvm::ArrayRef<int64_t> rhs) {
        llvm::SmallDenseSet<int64_t, 32> values(lhs.begin(), lhs.end());
        return llvm::none_of(rhs,
            [&](int64_t value) { return values.contains(value); });
    };
    const bool independentWeights =
        areDisjoint(gateWeightSlices, upWeightSlices);
    const bool independentActivation =
        throughput.mxm_weight_activation_overlap_enabled != 0
        && areDisjoint(gateWeightSlices, activationSlices)
        && areDisjoint(upWeightSlices, activationSlices);
    const int64_t activationStreamCount =
        2 * throughput.mxm_block_rows;
    const bool wavefrontOverlap = independentWeights
        && independentActivation
        && throughput.mxm_weight_buffers >= 2
        && throughput.mxm_local_load_to_compute_latency <= blockIssues
        && throughput.mxm_block_group_interval >= 2 * blockIssues
        && target.streams().streams_per_direction
            >= 2 * activationStreamCount;

    if (wavefrontOverlap) {
        const int64_t loadCycles = throughput.tile_rows;
        const int64_t loadLead =
            throughput.mxm_local_load_to_compute_latency;
        FfnBlock8ProjectionSchedule result;
        int64_t pairFirstCompute = startCycle + loadLead;
        for (int64_t base = 0; base < reductionBlocks; base += 2) {
            const int64_t pairSize = std::min<int64_t>(2,
                reductionBlocks - base);
            for (int64_t slot = 0; slot < pairSize; ++slot) {
                const int64_t reduction = base + slot;
                const int64_t firstCompute =
                    pairFirstCompute + slot * blockIssues;
                FfnBlock8ReductionSchedule reductionSchedule {
                    reduction,
                    reduction % throughput.mxm_weight_buffers,
                    firstCompute - loadLead,
                    {},
                    {},
                };
                for (int64_t token = 0; token < tokenBlocks; ++token) {
                    reductionSchedule.compute_cycles.push_back(
                        firstCompute
                        + token * throughput.mxm_block_group_interval);
                    reductionSchedule.activation_stream_bases.push_back(0);
                }
                result.reductions.push_back(
                    std::move(reductionSchedule));
            }
            pairFirstCompute += tokenBlocks
                * throughput.mxm_block_group_interval;
        }

        for (auto& reduction : result.reductions) {
            for (std::size_t index = 0;
                 index < reduction.compute_cycles.size(); ++index) {
                const int64_t computeStart =
                    reduction.compute_cycles[index];
                const int64_t computeEnd = computeStart + blockIssues;
                const bool overlapsLoad = llvm::any_of(result.reductions,
                    [&](const FfnBlock8ReductionSchedule& candidate) {
                        if (candidate.reduction == reduction.reduction)
                            return false;
                        const int64_t loadStart = candidate.load_cycle;
                        const int64_t loadEnd = loadStart + loadCycles;
                        return computeStart < loadEnd
                            && computeEnd > loadStart;
                    });
                if (overlapsLoad)
                    reduction.activation_stream_bases[index] =
                        activationStreamCount;
            }
        }
        result.end_cycle = startCycle;
        for (const auto& reduction : result.reductions) {
            result.end_cycle = std::max(result.end_cycle,
                reduction.load_cycle + loadCycles);
            if (!reduction.compute_cycles.empty())
                result.end_cycle = std::max(result.end_cycle,
                    reduction.compute_cycles.back() + blockIssues);
        }
        return result;
    }

    const int64_t weightLoadCycles = serializedWeightLoadCycles;
    const int64_t weightLoadLead = weightLoadCycles
        + throughput.mxm_local_load_to_compute_latency
        - throughput.tile_rows;
    const int64_t loadInterval = independentWeights
        ? throughput.tile_rows : weightLoadCycles;
    const int64_t interleaveWidth = std::max<int64_t>(1,
        std::min(throughput.mxm_weight_buffers,
            throughput.mxm_block_group_interval / blockIssues));
    const int64_t computeSpan =
        (tokenBlocks - 1) * throughput.mxm_block_group_interval
        + blockIssues;

    FfnBlock8ProjectionSchedule result;
    int64_t nextLoadCycle = startCycle;
    int64_t scheduleEnd = startCycle;
    llvm::SmallVector<int64_t> bufferFreeCycles(
        throughput.mxm_weight_buffers, startCycle);
    for (int64_t base = 0; base < reductionBlocks;
         base += interleaveWidth) {
        const int64_t groupSize =
            std::min(interleaveWidth, reductionBlocks - base);
        llvm::SmallVector<int64_t> loadCycles;
        int64_t allLoadsEnd = nextLoadCycle;
        for (int64_t slot = 0; slot < groupSize; ++slot) {
            const int64_t reduction = base + slot;
            const int64_t buffer =
                reduction % throughput.mxm_weight_buffers;
            const int64_t loadCycle = std::max(
                nextLoadCycle, bufferFreeCycles[buffer]);
            loadCycles.push_back(loadCycle);
            allLoadsEnd = std::max(
                allLoadsEnd, loadCycle + weightLoadCycles);
            nextLoadCycle = loadCycle + loadInterval;
        }
        int64_t groupEnd = allLoadsEnd;
        for (int64_t slot = 0; slot < groupSize; ++slot) {
            const int64_t reduction = base + slot;
            const int64_t loadCycle = loadCycles[slot];
            const int64_t firstCompute = std::max(
                loadCycle + weightLoadLead,
                allLoadsEnd + slot * blockIssues);
            FfnBlock8ReductionSchedule reductionSchedule {
                reduction,
                reduction % throughput.mxm_weight_buffers,
                loadCycle,
                {},
                {},
            };
            for (int64_t token = 0; token < tokenBlocks; ++token) {
                reductionSchedule.compute_cycles.push_back(
                    firstCompute
                    + token * throughput.mxm_block_group_interval);
                reductionSchedule.activation_stream_bases.push_back(0);
            }
            groupEnd = std::max(groupEnd, firstCompute + computeSpan);
            bufferFreeCycles[reductionSchedule.weight_buffer] =
                firstCompute + computeSpan;
            result.reductions.push_back(std::move(reductionSchedule));
        }
        scheduleEnd = std::max(scheduleEnd, groupEnd);
        if (!independentActivation) nextLoadCycle = groupEnd;
    }
    result.end_cycle = scheduleEnd;
    return result;
}

} // namespace ftlpu::compiler::schedule
