#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"


#include <algorithm>
#include <set>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;
namespace {

int64_t functionArgumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

llvm::SmallVector<int64_t> placementSlices(
    mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    if (!placement) return result;
    const auto values = placement.getAs<mlir::ArrayAttr>("slices");
    if (!values) return result;
    for (mlir::Attribute value : values)
        result.push_back(
            llvm::cast<mlir::IntegerAttr>(value).getInt());
    return result;
}

} // namespace

int64_t AttentionScheduleEmitter::emitBlock8OutputProjection(
    int64_t pvEnd)
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const auto& throughput = target_.throughput();
    const auto& memory = target_.memory();
    const int64_t tile = throughput.mxm_rows;
    const int64_t blockRows = throughput.mxm_block_rows;
    const int64_t blockIssues = tile / blockRows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t reductionBlocks =
        op_.getQueryHeads() * op_.getHeadDim() / tile;
    const int64_t hiddenBlocks = op_.getHidden() / tile;
    const int64_t outputGroups =
        op_.getHidden() / (tile * memory.hemispheres);
    const auto activationPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(
            "output_activation");
    const auto resultPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("result");
    const auto activationSlices =
        placementSlices(activationPlacement);
    const auto resultSlices = placementSlices(resultPlacement);
    if (throughput.mxms_per_hemisphere != 1 || blockRows != 8
        || blockIssues != 4 || activationSlices.size() != 16
        || resultSlices.size() != 16) {
        op_.emitError(
            "Block8 O projection requires 16-slice activation and "
            "result layouts");
        return -1;
    }
    const int64_t activationBase = activationPlacement
        .getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t resultBase = resultPlacement
        .getAs<mlir::IntegerAttr>("base_row").getInt();
    const auto outputWeightScaleAttr = op_.output.getConfig()
        .getAs<mlir::FloatAttr>("output_weight_scale");
    const float outputWeightScale = outputWeightScaleAttr
        ? static_cast<float>(
              outputWeightScaleAttr.getValueAsDouble())
        : 1.0f;

    // PV is head-planar. A westbound context stream taps low local MEM slices,
    // crosses the passive VXM bridge, and is consumed by matching remote MEM
    // slices. This creates distributed16 input without occupying a VXM ALU.
    int64_t stagingCycle = pvEnd + 1;
    int64_t stagingEnd = stagingCycle;
    std::set<std::pair<int64_t, int64_t>> occupiedWritePorts;
    for (int64_t reduction = 0;
         reduction < reductionBlocks; ++reduction) {
        const int64_t queryHead =
            reduction / (op_.getHeadDim() / tile);
        const int64_t headBlock =
            reduction % (op_.getHeadDim() / tile);
        for (int64_t tokenBlock = 0;
             tokenBlock < tokenBlocks; ++tokenBlock) {
            for (int64_t rowBlock = 0;
                 rowBlock < blockIssues; ++rowBlock) {
                const int64_t destinationAddress = activationBase
                    + (reduction * tokenBlocks + tokenBlock)
                        * blockIssues
                    + rowBlock;
                for (int64_t tokenLane = 0;
                     tokenLane < blockRows;
                     ++tokenLane, ++stagingCycle) {
                    const int64_t token = tokenBlock * tile
                        + rowBlock * blockRows + tokenLane;
                    const int64_t queryHeadsPerKv =
                        op_.getQueryHeads() / op_.getKvHeads();
                    const int64_t sourceHemisphere =
                        (queryHead / queryHeadsPerKv)
                        % memory.hemispheres;
                    // PV writes its direct BF16 accumulator result on W0/W1.
                    // Use a separate westward bank so O-projection staging
                    // can overlap the tail of the final PV result wave.
                    const int64_t contextStreamBase =
                        34 + headBlock * 2;
                    for (;;) {
                        bool conflict = false;
                        for (int64_t hemisphere = 0;
                             hemisphere < memory.hemispheres;
                             ++hemisphere) {
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t destinationSlice =
                                    activationSlices[tokenLane * 2 + byte];
                                const int64_t destinationGroup =
                                    destinationSlice
                                    / target_.streams()
                                          .mem_slices_per_register_group;
                                const bool local =
                                    hemisphere == sourceHemisphere;
                                const int64_t writeCycle = local
                                    ? stagingCycle - destinationGroup - 1
                                    : stagingCycle + destinationGroup + 1;
                                const int64_t queue = hemisphere
                                        * memory.slices_per_hemisphere
                                    + destinationSlice;
                                conflict |= occupiedWritePorts.count(
                                    {queue, writeCycle}) != 0;
                            }
                        }
                        if (!conflict) break;
                        ++stagingCycle;
                    }
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t sourceSlice =
                            layout.contextSlices()[headBlock * 2 + byte];
                        const auto readLatency = target_.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::VxmInput,
                            target::StreamDirection::West, sourceSlice);
                        if (!readLatency) return -1;
                        emitMem(rewriter_, op_.getLoc(),
                            stagingCycle - *readLatency,
                            sourceHemisphere * memory.slices_per_hemisphere
                                + sourceSlice,
                            "read", layout.contextAddress(queryHead, token),
                            contextStreamBase + byte, 1, 1, 0);
                    }
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres;
                         ++hemisphere) {
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t destinationSlice =
                                activationSlices[
                                    tokenLane * 2 + byte];
                            const int64_t sourceSlice =
                                layout.contextSlices()[headBlock * 2 + byte];
                            const int64_t sourceGroup = sourceSlice
                                / target_.streams().mem_slices_per_register_group;
                            const int64_t destinationGroup = destinationSlice
                                / target_.streams().mem_slices_per_register_group;
                            if (destinationGroup >= sourceGroup) {
                                op_.emitError(
                                    "passive O-projection staging requires "
                                    "activation slices west of context slices");
                                return -1;
                            }
                            const bool local = hemisphere == sourceHemisphere;
                            const int64_t writeCycle = local
                                ? stagingCycle - destinationGroup - 1
                                : stagingCycle + destinationGroup + 1;
                            const int64_t packedStream = local
                                ? contextStreamBase + byte
                                : contextStreamBase + byte
                                    - target_.streams().streams_per_direction;
                            emitMem(rewriter_, op_.getLoc(),
                                writeCycle,
                                hemisphere * memory.slices_per_hemisphere
                                        + destinationSlice,
                                local ? "write_tap" : "write",
                                destinationAddress, packedStream, 1, 1, 0);
                            occupiedWritePorts.insert({
                                hemisphere * memory.slices_per_hemisphere
                                    + destinationSlice,
                                writeCycle});
                            stagingEnd = std::max(stagingEnd,
                                writeCycle + 1);
                        }
                    }
                }
            }
        }
    }

    int64_t maxWeightLatency = 0;
    for (int64_t slice : layout.outputWeightSlices()) {
        const auto latency = target_.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        if (!latency) return -1;
        maxWeightLatency = std::max(maxWeightLatency, *latency);
    }
    int64_t phaseStart = stagingEnd + 1;
    int64_t phaseEnd = phaseStart;
    for (int64_t groupBase = 0; groupBase < outputGroups;
         groupBase += throughput.mxm_weight_buffers) {
        const int64_t groupCount = std::min<int64_t>(
            throughput.mxm_weight_buffers, outputGroups - groupBase);
        const bool duplicateSingleton = groupCount == 1
            && throughput.mxm_weight_buffers >= 2
            && 2 * blockIssues
                >= throughput.mxm_block_group_interval;
        const int64_t tokenIssueInterval = duplicateSingleton
            ? blockIssues
            : std::max<int64_t>(groupCount * blockIssues,
                  throughput.mxm_block_group_interval);
        const int64_t firstCompute = phaseStart + maxWeightLatency
            + throughput.mxm_local_load_to_compute_latency;
        int64_t pairEnd = firstCompute;
        int64_t lastComputeEnd = firstCompute;
        for (int64_t reduction = 0;
             reduction < reductionBlocks; ++reduction) {
            const bool finalReduction =
                reduction + 1 == reductionBlocks;
            const int64_t reductionCompute = firstCompute
                + reduction * tokenBlocks
                    * tokenIssueInterval;
            for (int64_t groupSlot = 0; groupSlot < groupCount;
                 ++groupSlot) {
                const int64_t outputGroup = groupBase + groupSlot;
                const int64_t loadReplicaCount =
                    duplicateSingleton ? 2 : 1;
                for (int64_t loadReplica = 0;
                     loadReplica < loadReplicaCount; ++loadReplica) {
                    const int64_t weightBuffer = duplicateSingleton
                        ? loadReplica
                        : groupSlot;
                    const int64_t loadSlot = duplicateSingleton
                        ? loadReplica
                        : groupSlot;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres; ++hemisphere) {
                        // Replace weights behind the compute wavefront as it
                        // advances across MXM columns.
                        for (int64_t pulse = 0;
                             pulse < throughput.tile_rows; ++pulse) {
                            const int64_t firstUnitCompute = reductionCompute
                                + loadSlot * blockIssues;
                            const int64_t cycle = firstUnitCompute
                                - throughput.mxm_local_load_to_compute_latency
                                + pulse;
                            for (int64_t stream = 0; stream < 8; ++stream) {
                                const int64_t slice =
                                    layout.outputWeightSlices()[stream];
                                const auto latency = target_.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::MxmWeight,
                                    target::StreamDirection::East, slice);
                                if (!latency) return -1;
                                emitMem(rewriter_, op_.getLoc(),
                                    cycle - *latency,
                                    hemisphere
                                            * memory.slices_per_hemisphere
                                        + slice,
                                    "read", layout.outputWeightAddress(
                                        outputGroup, reduction,
                                        throughput.tile_rows - 1 - pulse),
                                    stream, 1, 1, 0, "sram",
                                    functionArgumentIndex(
                                        op_.getOutputWeight()));
                            }
                            emitMxmDequant(rewriter_, op_.getLoc(), cycle,
                                hemisphere, outputWeightScale);
                            emitMxm(rewriter_, op_.getLoc(), cycle,
                                hemisphere, "iw", weightBuffer,
                                pulse, 0, 0, 1, 1, 0, 1, "sram",
                                true, "supercell", 0, dataFormat,
                                "int8_dequant_bf16");
                        }
                    }
                }

                for (int64_t tokenBlock = 0;
                     tokenBlock < tokenBlocks; ++tokenBlock) {
                    const int64_t weightBuffer = duplicateSingleton
                        ? tokenBlock % 2
                        : groupSlot;
                    const int64_t activationAddress = activationBase
                        + (reduction * tokenBlocks + tokenBlock)
                            * blockIssues;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres; ++hemisphere) {
                        const int64_t computeSlot = groupSlot;
                        const int64_t computeCycle = reductionCompute
                            + tokenBlock * tokenIssueInterval
                            + computeSlot * blockIssues;
                        for (int64_t stream = 0; stream < 16;
                             ++stream) {
                            const int64_t slice =
                                activationSlices[stream];
                            const auto latency = target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmActivation,
                                target::StreamDirection::East, slice);
                            if (!latency) return -1;
                            emitMem(rewriter_, op_.getLoc(),
                                computeCycle - *latency,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read", activationAddress,
                                2 * blockRows + stream,
                                blockIssues, 1, 1);
                        }
                        const int64_t accumulatorBase =
                            (groupSlot * tokenBlocks + tokenBlock)
                            * blockIssues;
                        emitMxm(rewriter_, op_.getLoc(), computeCycle,
                            hemisphere, "compute", weightBuffer, 0,
                            2 * blockRows, 0, blockIssues, 1,
                            accumulatorBase, 1,
                            finalReduction ? "stream" : "sram", true,
                            "supercell", 0, dataFormat, {}, "block8");
                        lastComputeEnd = std::max(lastComputeEnd,
                            computeCycle + blockIssues);
                        if (!finalReduction) continue;

                        const int64_t resultCycle = computeCycle
                            + throughput.accumulator_to_vxm_latency;
                        const int64_t resultAddress = resultBase
                            + (tokenBlock * hiddenBlocks
                                  + outputGroup * memory.hemispheres
                                  + hemisphere)
                                * blockIssues;
                        for (int64_t stream = 0; stream < 16;
                             ++stream) {
                            const int64_t slice = resultSlices[stream];
                            const int64_t group = slice
                                / target_.streams()
                                      .mem_slices_per_register_group;
                            const int64_t writeCycle =
                                resultCycle - group - 1;
                            emitMem(rewriter_, op_.getLoc(), writeCycle,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", resultAddress,
                                target_.streams().streams_per_direction
                                    + stream,
                                blockIssues, 1, 1);
                            pairEnd = std::max(pairEnd,
                                writeCycle + blockIssues);
                        }
                    }
                }
            }
        }
        phaseEnd = std::max(phaseEnd, pairEnd + 1);
        phaseStart = lastComputeEnd - maxWeightLatency
            - throughput.mxm_local_load_to_compute_latency;
    }
    return phaseEnd;
}

} // namespace ftlpu::compiler::schedule
