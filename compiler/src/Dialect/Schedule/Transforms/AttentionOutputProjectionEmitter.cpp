#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/paged_weight_residency.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;
namespace {

constexpr int64_t kC2cTransportGuardCycles = 64;

int64_t functionArgumentIndex(mlir::Value value) {
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
    return argument.getArgNumber();
  return -1;
}

} // namespace

int64_t AttentionScheduleEmitter::emitOutputProjection(
    int64_t pvEnd, int64_t qkvEnd) {
  const AttentionMemoryLayout layout(op_, target_);
  const auto contextPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("context");
  const int64_t contextBank =
      contextPlacement.getAs<mlir::IntegerAttr>("bank").getInt();
  const auto outputResultPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("result");
  const auto resultSliceAttrs =
      outputResultPlacement.getAs<mlir::ArrayAttr>("slices");
  std::array<int64_t, 4> resultSlices{};
  for (std::size_t i = 0; i < resultSlices.size(); ++i)
    resultSlices[i] =
        llvm::cast<mlir::IntegerAttr>(resultSliceAttrs[i]).getInt();
  const int64_t resultBank =
      outputResultPlacement.getAs<mlir::IntegerAttr>("bank").getInt();
  const auto elementType =
      llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
          .getElementType();
  const llvm::StringRef dataFormat = lpu_16bit_data_format(elementType);
  const int64_t tile = target_.throughput().mxm_rows;
  const int64_t tokenBlocks = (op_.getSeqLen() + tile - 1) / tile;
  const int64_t projectionRows = std::min<int64_t>(tile, op_.getSeqLen());
  const int64_t headBlocks = op_.getHeadDim() / tile;
  const int64_t reductionBlocks = op_.getQueryHeads() * op_.getHeadDim() / tile;
  const int64_t outputGroups =
      op_.getHidden() / (tile * target_.memory().hemispheres);
  int64_t projectionIssueInterval =
      op_.getSeqLen() == 1
          ? std::max(target_.throughput().mxm0_accumulator_latency,
                     target_.throughput().mxm1_accumulator_latency)
          : tile;
  const int64_t localMxm = 0;
  const int64_t weightToIw = target_.throughput().vxm_weight_to_iw_latency;
  const bool localDequant = target_.supports_mxm_local_dequant();
  const int64_t loadToIw = localDequant ? 0 : weightToIw;
  const int64_t weightStreamBase =
      localDequant ? target_.streams().streams_per_direction -
                         target_.throughput().mxm_int8_load_streams_per_cycle
                   : target_.streams().streams_per_direction;
  const auto readLatency = [&](int64_t slice) {
    return slice / target_.streams().mem_slices_per_register_group + 2;
  };
  const auto weightReadLatency = [&](int64_t slice) {
    return target_
        .transport_latency(target::StreamEndpoint::Mem,
                           target::StreamEndpoint::MxmWeight,
                           target::StreamDirection::East, slice)
        .value_or(readLatency(slice));
  };
  const int64_t weightLoadLead =
      (target_.memory().hemispheres - 1) * 8 + 3 + loadToIw + 1;
  if (op_.getSeqLen() == 1) {
    // With two alternating weight buffers, the next use of a buffer cannot
    // begin loading until the previous reduction has released it.  Fold that
    // residency constraint into a uniform reduction cadence.  Besides being
    // the fastest legal cadence, keeping it affine lets MEM/MXM ICUs express
    // the whole projection as one closed-form domain instead of interleaving
    // many long-lived descriptors.
    const int64_t weightBuffers = target_.throughput().mxm_weight_buffers;
    const int64_t bufferResidency =
        weightLoadLead + target_.mxm_result_window_cycles(projectionRows);
    projectionIssueInterval =
        std::max(projectionIssueInterval,
                 (bufferResidency + weightBuffers - 1) / weightBuffers);
  }
  const int64_t computeInterval = projectionIssueInterval;
  int64_t contextReadLatency = 0;
  for (int64_t slice : layout.contextSlices())
    contextReadLatency =
        std::max(contextReadLatency,
                 target_
                     .transport_latency(target::StreamEndpoint::Mem,
                                        target::StreamEndpoint::MxmActivation,
                                        target::StreamDirection::East, slice)
                     .value_or(0));
  // MXM accumulator SRAM is a separate fixed-depth resource from MEM.
  const int64_t accumulatorCapacity = 32 * target_.throughput().mxm_rows;
  const int64_t outputAccumulatorBase = accumulatorCapacity - op_.getSeqLen();
  const auto outputWeightScaleAttr =
      op_.output.getConfig().getAs<mlir::FloatAttr>("output_weight_scale");
  const float outputWeightScale =
      outputWeightScaleAttr
          ? static_cast<float>(outputWeightScaleAttr.getValueAsDouble())
          : 1.0f;
  const auto accumulatorAddress = [&](int64_t token) {
    return outputAccumulatorBase + token;
  };

  int64_t phaseStart = pvEnd;
  const auto outputWeightPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("output_weight");
  if (outputWeightPlacement) {
    if (const auto transferCycles =
            outputWeightPlacement.getAs<mlir::IntegerAttr>(
                "page_transfer_cycles")) {
      const auto memoryPlan = op_.getMemoryPlan();
      const char *priorWeights[] = {
          "query_weight", "key_weight", "value_weight"};
      const bool reusesPriorResidency = std::ranges::any_of(
          priorWeights, [&](const char *name) {
            return pagedWeightResidencyOverlaps(
                memoryPlan.getAs<mlir::DictionaryAttr>(name),
                outputWeightPlacement);
          });
      if (reusesPriorResidency)
        phaseStart = std::max(
            phaseStart,
            qkvEnd + transferCycles.getInt() + kC2cTransportGuardCycles);
    }
  }
  const auto plannedOutputGroupEnd = [&](int64_t finalFirstCompute) {
    const int64_t resultCycle =
        finalFirstCompute + target_.mxm_first_result_latency();
    int64_t finalWriteEnd = finalFirstCompute;
    for (int64_t hemisphere = 0;
         hemisphere < target_.memory().hemispheres; ++hemisphere) {
      for (int64_t byte = 0; byte < 2; ++byte) {
        const int64_t slice = resultSlices[hemisphere * 2 + byte];
        const int64_t latency = *target_.transport_latency(
            target::StreamEndpoint::MxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::West, slice);
        finalWriteEnd = std::max(
            finalWriteEnd, resultCycle + latency + op_.getSeqLen());
        if (hemisphere != 0) {
          const int64_t registerGroup =
              slice / target_.streams().mem_slices_per_register_group;
          finalWriteEnd = std::max(
              finalWriteEnd,
              resultCycle + target_.streams().system_register_columns +
                  registerGroup + 1 + op_.getSeqLen());
        }
      }
    }
    return std::max(finalFirstCompute + tokenBlocks * computeInterval,
                    finalWriteEnd);
  };
  struct OutputGroupPlan {
    int64_t start;
    int64_t end;
    std::vector<int64_t> firstComputeCycles;
    std::vector<int64_t> dequantStartCycles;
  };
  std::vector<OutputGroupPlan> groupPlans;
  groupPlans.reserve(static_cast<std::size_t>(outputGroups));
  int64_t plannedPhaseStart = phaseStart;
  for (int64_t outputGroup = 0; outputGroup < outputGroups; ++outputGroup) {
    const auto selectedWeightSlices = layout.outputWeightSlices(outputGroup);
    int64_t maxWeightReadLatency = 0;
    for (int64_t slice : selectedWeightSlices)
      maxWeightReadLatency =
          std::max(maxWeightReadLatency, weightReadLatency(slice));
    const int64_t initialReadyLead =
        std::max(maxWeightReadLatency + weightLoadLead, contextReadLatency);
    OutputGroupPlan plan;
    plan.start = plannedPhaseStart;
    plan.firstComputeCycles.resize(static_cast<std::size_t>(reductionBlocks));
    plan.dequantStartCycles.resize(static_cast<std::size_t>(reductionBlocks));
    int64_t plannedNextCompute = plannedPhaseStart + initialReadyLead;
    std::vector<int64_t> plannedWeightBufferRelease(
        static_cast<std::size_t>(target_.throughput().mxm_weight_buffers),
        std::numeric_limits<int64_t>::min());
    for (int64_t reductionBlock = 0; reductionBlock < reductionBlocks;
         ++reductionBlock) {
      const int64_t weightBuffer =
          (outputGroup * reductionBlocks + reductionBlock) %
          target_.throughput().mxm_weight_buffers;
      int64_t firstCompute = plannedNextCompute;
      int64_t dequantStart = std::max(
          firstCompute - weightLoadLead,
          plannedWeightBufferRelease[static_cast<std::size_t>(weightBuffer)]);
      firstCompute = std::max(firstCompute, dequantStart + weightLoadLead);
      plan.firstComputeCycles[static_cast<std::size_t>(reductionBlock)] =
          firstCompute;
      plan.dequantStartCycles[static_cast<std::size_t>(reductionBlock)] =
          dequantStart;
      const int64_t lastCompute =
          firstCompute + (tokenBlocks - 1) * computeInterval;
      plannedWeightBufferRelease[static_cast<std::size_t>(weightBuffer)] =
          lastCompute + target_.mxm_result_window_cycles(projectionRows);
      plannedNextCompute = firstCompute + tokenBlocks * computeInterval;
    }
    plan.end = plannedOutputGroupEnd(plan.firstComputeCycles.back());
    plannedPhaseStart = plan.end;
    groupPlans.push_back(std::move(plan));
  }

  // Prove the output-group recurrence from the operator schedule before
  // emitting any ICU instruction.  An affine recurrence can occupy the third
  // MEM counter when the per-group domain uses only row and reduction axes.
  int64_t outerGroupInterval = 1;
  bool affineOutputGroups = outputGroups > 1;
  if (affineOutputGroups) {
    outerGroupInterval = groupPlans[1].start - groupPlans[0].start;
    affineOutputGroups = outerGroupInterval > 0;
    for (int64_t group = 1; affineOutputGroups && group < outputGroups;
         ++group) {
      const auto &base = groupPlans[0].firstComputeCycles;
      const auto &current = groupPlans[static_cast<std::size_t>(group)];
      if (current.start != groupPlans[0].start +
                               group * outerGroupInterval)
        affineOutputGroups = false;
      for (int64_t reduction = 0;
           affineOutputGroups && reduction < reductionBlocks; ++reduction)
        if (current.firstComputeCycles[static_cast<std::size_t>(reduction)] !=
            base[static_cast<std::size_t>(reduction)] +
                group * outerGroupInterval)
          affineOutputGroups = false;
    }
  }
  const auto slicesIntersect = [](llvm::ArrayRef<int64_t> lhs,
                                  llvm::ArrayRef<int64_t> rhs) {
    return llvm::any_of(lhs, [&](int64_t slice) {
      return llvm::is_contained(rhs, slice);
    });
  };
  bool contextConflictsWithWeight = false;
  bool resultConflictsWithWeight = false;
  for (int64_t group = 0; group < outputGroups; ++group) {
    const auto slices = layout.outputWeightSlices(group);
    contextConflictsWithWeight |=
        contextBank == layout.outputWeightBank() &&
        slicesIntersect(slices, layout.contextSlices());
    resultConflictsWithWeight |=
        resultBank == layout.outputWeightBank() &&
        llvm::any_of(resultSlices, [&](int64_t slice) {
          return llvm::is_contained(slices, slice);
        });
  }
  const bool contextConflictsWithResult =
      contextBank == resultBank &&
      llvm::any_of(resultSlices, [&](int64_t slice) {
        return llvm::is_contained(layout.contextSlices(), slice);
      });
  const bool mergeContextOutputGroups =
      affineOutputGroups && !contextConflictsWithWeight &&
      !contextConflictsWithResult;
  bool mergeResultOutputGroups =
      affineOutputGroups && !resultConflictsWithWeight &&
      !contextConflictsWithResult && outerGroupInterval >= op_.getSeqLen();
  const int64_t resultGroupAddressStride =
      outputGroups > 1 ? layout.resultAddress(1, 0) -
                             layout.resultAddress(0, 0)
                       : 0;
  for (int64_t group = 1; mergeResultOutputGroups && group < outputGroups;
       ++group)
    if (layout.resultAddress(group, 0) !=
        layout.resultAddress(0, 0) + group * resultGroupAddressStride)
      mergeResultOutputGroups = false;

  // The O MXM has three independent loop coordinates: row/weight byte,
  // reduction, and output group.  Reindex the per-group reduction domain so
  // the hardware can keep one LOAD/DEQUANT/COMPUTE descriptor per MXM queue.
  // In particular, the terminal result belongs to the last reduction of
  // every output group, not to the last output group of the entire operator.
  int64_t mxmReductionInterval = 0;
  bool mergeMxmOutputGroups = localDequant && tokenBlocks == 1 &&
      target_.throughput().mxm_weight_buffers == 2 &&
      outputGroups > 1 && reductionBlocks > 1 &&
      reductionBlocks % 2 == 0 && affineOutputGroups;
  if (mergeMxmOutputGroups) {
    const auto &base = groupPlans[0];
    mxmReductionInterval = base.dequantStartCycles[1] -
                           base.dequantStartCycles[0];
    mergeMxmOutputGroups =
        mxmReductionInterval >= projectionIssueInterval &&
        base.firstComputeCycles[1] - base.firstComputeCycles[0] ==
            mxmReductionInterval &&
        outerGroupInterval >=
            (reductionBlocks - 1) * mxmReductionInterval +
                projectionIssueInterval;
    const auto firstWeightSlices = layout.outputWeightSlices(0);
    for (int64_t group = 0;
         mergeMxmOutputGroups && group < outputGroups; ++group) {
      const auto &plan = groupPlans[static_cast<std::size_t>(group)];
      mergeMxmOutputGroups &=
          std::ranges::equal(layout.outputWeightSlices(group),
                             firstWeightSlices);
      for (int64_t reduction = 0;
           mergeMxmOutputGroups && reduction < reductionBlocks; ++reduction) {
        const int64_t expected = reduction * mxmReductionInterval +
            group * outerGroupInterval;
        mergeMxmOutputGroups &=
            plan.dequantStartCycles[static_cast<std::size_t>(reduction)] ==
                base.dequantStartCycles[0] + expected &&
            plan.firstComputeCycles[static_cast<std::size_t>(reduction)] ==
                base.firstComputeCycles[0] + expected;
      }
    }
  }
  if (mergeMxmOutputGroups) {
    const auto &base = groupPlans[0];
    for (int64_t hemisphere = 0;
         hemisphere < target_.memory().hemispheres; ++hemisphere) {
      const int64_t loadCycle = base.dequantStartCycles[0] + hemisphere * 8;
      emitMxmDequant3D(
          rewriter_, op_.getLoc(), loadCycle, hemisphere,
          outputWeightScale, 4, 1, reductionBlocks,
          mxmReductionInterval, outputGroups, outerGroupInterval,
          functionArgumentIndex(op_.getOutputWeight()));

      MxmDomain3D loadDomain;
      loadDomain.repeat_count = 4;
      loadDomain.repeat_interval = 1;
      loadDomain.repeat_weight_column_stride = -1;
      loadDomain.wave_count = reductionBlocks;
      loadDomain.wave_interval = mxmReductionInterval;
      loadDomain.group_count = outputGroups;
      loadDomain.group_interval = outerGroupInterval;
      loadDomain.weight_buffer_mode = "toggle_dim1";
      emitMxm3D(
          rewriter_, op_.getLoc(), loadCycle + loadToIw,
          hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
          "iw", 0, 3, 0, 0, 0, 1, "stream", true,
          "supercell", 0, dataFormat, "int8_dequant_bf16",
          llvm::StringRef{}, loadDomain, weightStreamBase);

      MxmDomain3D computeDomain;
      computeDomain.repeat_count = projectionRows;
      computeDomain.repeat_interval = 1;
      computeDomain.wave_count = reductionBlocks;
      computeDomain.wave_interval = mxmReductionInterval;
      computeDomain.group_count = outputGroups;
      computeDomain.group_interval = outerGroupInterval;
      computeDomain.weight_buffer_mode = "toggle_dim1";
      computeDomain.terminal_dimension = 1;
      computeDomain.terminal_accumulator_destination = "stream";
      computeDomain.terminal_accumulator_clear = true;
      computeDomain.terminal_accumulator_output_format = dataFormat;
      emitMxm3D(
          rewriter_, op_.getLoc(), base.firstComputeCycles[0],
          hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
          "compute", 0, 0, hemisphere * 2, 0,
          accumulatorAddress(0), 1, "sram", false,
          "supercell", 0, dataFormat, llvm::StringRef{}, "fp32",
          computeDomain);
    }
  }

  int64_t nextOutputWeightDomain = 0;
  for (int64_t outputGroup = 0; outputGroup < outputGroups; ++outputGroup) {
    const auto &plan = groupPlans[static_cast<std::size_t>(outputGroup)];
    phaseStart = plan.start;
    const auto selectedWeightSlices = layout.outputWeightSlices(outputGroup);
    const auto &firstComputeCycles = plan.firstComputeCycles;
    const auto &dequantStartCycles = plan.dequantStartCycles;
    const int64_t outputGroupInterval = plan.end - plan.start;

    // A long-lived MEM descriptor owns one physical (hemisphere,slice,bank)
    // ICU until its last generated request.  Do not carry a weight-read domain
    // across an output-group boundary when context reads or result writes use
    // that same ICU between two weight waves.
    const bool localContextConflictsWithWeight =
        contextBank == layout.outputWeightBank() &&
        slicesIntersect(selectedWeightSlices, layout.contextSlices());
    const bool localResultConflictsWithWeight =
        resultBank == layout.outputWeightBank() &&
        llvm::any_of(resultSlices, [&](int64_t slice) {
          return llvm::is_contained(selectedWeightSlices, slice);
        });
    const int64_t weightDomainSpan =
        dequantStartCycles.back() - dequantStartCycles.front() + 4;
    const bool mayMergeOutputGroups =
        !localContextConflictsWithWeight && !localResultConflictsWithWeight &&
        outputGroupInterval >= weightDomainSpan;
    const auto sameWeightSlices = [](llvm::ArrayRef<int64_t> lhs,
                                     llvm::ArrayRef<int64_t> rhs) {
      return lhs.size() == rhs.size() && std::ranges::equal(lhs, rhs);
    };
    int64_t weightOutputGroupCount = 1;
    if (mayMergeOutputGroups && outputGroup == nextOutputWeightDomain) {
      const int64_t baseAddress =
          layout.outputWeightAddress(outputGroup, 0, 0);
      const int64_t groupAddressStride =
          outputGroup + 1 < outputGroups
              ? layout.outputWeightAddress(outputGroup + 1, 0, 0) -
                    baseAddress
              : 0;
      weightOutputGroupCount = 1;
      while (outputGroup + weightOutputGroupCount < outputGroups) {
        const int64_t candidateGroup =
            outputGroup + weightOutputGroupCount;
        if (!sameWeightSlices(layout.outputWeightSlices(candidateGroup),
                              selectedWeightSlices) ||
            layout.outputWeightAddress(candidateGroup, 0, 0) !=
                baseAddress +
                    weightOutputGroupCount * groupAddressStride)
          break;
        ++weightOutputGroupCount;
      }
      nextOutputWeightDomain = outputGroup + weightOutputGroupCount;
    } else if (mayMergeOutputGroups) {
      // The first group in this placement run already emitted the complete
      // outer domain.
      weightOutputGroupCount = 0;
    } else {
      nextOutputWeightDomain = outputGroup + 1;
    }

    // Context activation reads are affine in three logical coordinates:
    // row, reduction within one physical-slice run, and recurring slice run.
    // Flatten the contiguous token blocks into the row coordinate so that the
    // other two MEM counters can cover head blocks/query heads directly.  The
    // domain is emitted here from the operator layout; no scalar MEM events
    // are formed and subsequently compressed.
    const auto emitContextReadDomain =
        [&](int64_t firstReduction, int64_t waveReductionStride,
            int64_t waveCount, int64_t groupReductionStride,
            int64_t groupCount, int64_t hemisphere, int64_t byte) {
          if (firstReduction < 0 || waveReductionStride <= 0 ||
              waveCount <= 0 || groupReductionStride <= 0 ||
              groupCount <= 0)
            return false;
          const auto reductionAt = [&](int64_t wave, int64_t group) {
            return firstReduction + wave * waveReductionStride +
                   group * groupReductionStride;
          };
          const int64_t lastReduction =
              reductionAt(waveCount - 1, groupCount - 1);
          if (lastReduction >= reductionBlocks)
            return false;

          const int64_t queryHead = firstReduction / headBlocks;
          const int64_t headBlock = firstReduction % headBlocks;
          const int64_t slice =
              layout.contextSlice(queryHead, headBlock, byte);
          // A long-lived READ_3D context cannot cross an output-weight read on
          // the same physical MEM ICU.  Keep that topology as an explicit
          // piecewise-domain boundary.
          if (std::ranges::find(selectedWeightSlices, slice) !=
              selectedWeightSlices.end())
            return false;
          const int64_t address =
              layout.contextAddress(queryHead, headBlock, 0);
          const int64_t latency = *target_.transport_latency(
              target::StreamEndpoint::Mem,
              target::StreamEndpoint::MxmActivation,
              target::StreamDirection::East, slice);
          const int64_t cycle =
              firstComputeCycles[static_cast<std::size_t>(firstReduction)] -
              latency;
          const auto addressAt = [&](int64_t reduction) {
            return layout.contextAddress(reduction / headBlocks,
                                         reduction % headBlocks, 0);
          };
          const int64_t waveInterval =
              waveCount > 1
                  ? firstComputeCycles[static_cast<std::size_t>(
                        reductionAt(1, 0))] -
                        firstComputeCycles[static_cast<std::size_t>(
                            firstReduction)]
                  : 1;
          const int64_t waveAddressStride =
              waveCount > 1 ? addressAt(reductionAt(1, 0)) - address : 0;
          const int64_t groupInterval =
              groupCount > 1
                  ? firstComputeCycles[static_cast<std::size_t>(
                        reductionAt(0, 1))] -
                        firstComputeCycles[static_cast<std::size_t>(
                            firstReduction)]
                  : 1;
          const int64_t groupAddressStride =
              groupCount > 1 ? addressAt(reductionAt(0, 1)) - address : 0;

          // Prove that the requested rectangle is affine before materializing
          // its single hardware descriptor.  This inspects the operator's
          // closed-form schedule plan, not a list of generated instructions.
          for (int64_t group = 0; group < groupCount; ++group) {
            for (int64_t wave = 0; wave < waveCount; ++wave) {
              const int64_t reduction = reductionAt(wave, group);
              const int64_t candidateHead = reduction / headBlocks;
              const int64_t candidateBlock = reduction % headBlocks;
              if (layout.contextSlice(candidateHead, candidateBlock, byte) !=
                      slice ||
                  firstComputeCycles[static_cast<std::size_t>(reduction)] !=
                      firstComputeCycles[static_cast<std::size_t>(
                          firstReduction)] +
                          wave * waveInterval + group * groupInterval ||
                  layout.contextAddress(candidateHead, candidateBlock, 0) !=
                      address + wave * waveAddressStride +
                          group * groupAddressStride)
                return false;
            }
          }
          const int64_t repeatSpan = op_.getSeqLen() - 1;
          const int64_t waveSpan =
              repeatSpan + (waveCount - 1) * waveInterval;
          if ((waveCount > 1 && waveInterval <= repeatSpan) ||
              (groupCount > 1 && groupInterval <= waveSpan))
            return false;

          const bool useOutputGroupCounter =
              mergeContextOutputGroups && groupCount == 1 &&
              outerGroupInterval > waveSpan;
          if (useOutputGroupCounter && outputGroup != 0)
            return true;

          emitMem3D(
              rewriter_, op_.getLoc(), cycle,
              hemisphere * target_.memory().slices_per_hemisphere + slice,
              "read", address, hemisphere * 2 + byte, op_.getSeqLen(), 1, 1,
              "sram", -1, waveCount, waveInterval, waveAddressStride,
              useOutputGroupCounter ? outputGroups : groupCount,
              useOutputGroupCounter ? outerGroupInterval : groupInterval,
              useOutputGroupCounter ? 0 : groupAddressStride, contextBank);
          return true;
        };
    const auto emitContextReadPiece =
        [&](int64_t firstReduction, int64_t waveReductionStride,
            int64_t waveCount, int64_t groupReductionStride,
            int64_t groupCount, int64_t hemisphere, int64_t byte) {
          if (emitContextReadDomain(firstReduction, waveReductionStride,
                                    waveCount, groupReductionStride,
                                    groupCount, hemisphere, byte))
            return;
          // A non-affine outer recurrence is still emitted directly as a
          // finite set of closed-form slice-run domains.  Scalar reduction
          // domains remain only for a true physical/body conflict.
          for (int64_t group = 0; group < groupCount; ++group) {
            const int64_t groupBase =
                firstReduction + group * groupReductionStride;
            if (emitContextReadDomain(groupBase, waveReductionStride,
                                      waveCount, 1, 1, hemisphere, byte))
              continue;
            for (int64_t wave = 0; wave < waveCount; ++wave) {
              const int64_t reduction =
                  groupBase + wave * waveReductionStride;
              const int64_t queryHead = reduction / headBlocks;
              const int64_t headBlock = reduction % headBlocks;
              const int64_t slice =
                  layout.contextSlice(queryHead, headBlock, byte);
              const int64_t latency = *target_.transport_latency(
                  target::StreamEndpoint::Mem,
                  target::StreamEndpoint::MxmActivation,
                  target::StreamDirection::East, slice);
              emitMem3D(
                  rewriter_, op_.getLoc(),
                  firstComputeCycles[static_cast<std::size_t>(reduction)] -
                      latency,
                  hemisphere * target_.memory().slices_per_hemisphere + slice,
                  "read", layout.contextAddress(queryHead, headBlock, 0),
                  hemisphere * 2 + byte, projectionRows, 1, 1, "sram", -1,
                  tokenBlocks, computeInterval, tile, 1, 1, 0, contextBank);
            }
          }
        };

    for (int64_t hemisphere = 0;
         hemisphere < target_.memory().hemispheres; ++hemisphere) {
      for (int64_t byte = 0; byte < 2; ++byte) {
        if (!layout.contextHeadBlockPacked()) {
          for (int64_t headBlock = 0; headBlock < headBlocks; ++headBlock)
            emitContextReadPiece(headBlock, headBlocks, op_.getQueryHeads(), 1,
                                 1, hemisphere, byte);
          continue;
        }
        if (!layout.contextHemispherePaired()) {
          emitContextReadPiece(0, 1, headBlocks, headBlocks,
                               op_.getQueryHeads(), hemisphere, byte);
          continue;
        }
        const int64_t queryHeadsPerKv =
            op_.getQueryHeads() / op_.getKvHeads();
        const int64_t sourceGroups =
            queryHeadsPerKv > 0 ? op_.getQueryHeads() / queryHeadsPerKv : 0;
        const int64_t reductionsPerSourceGroup =
            queryHeadsPerKv * headBlocks;
        if (queryHeadsPerKv <= 0 ||
            queryHeadsPerKv * op_.getKvHeads() != op_.getQueryHeads()) {
          emitContextReadPiece(0, 1, reductionBlocks, 1, 1, hemisphere,
                               byte);
          continue;
        }
        for (int64_t sourceHemisphere = 0;
             sourceHemisphere < target_.memory().hemispheres &&
             sourceHemisphere < sourceGroups;
             ++sourceHemisphere) {
          const int64_t recurringGroups =
              1 + (sourceGroups - 1 - sourceHemisphere) /
                      target_.memory().hemispheres;
          emitContextReadPiece(
              sourceHemisphere * reductionsPerSourceGroup, 1,
              reductionsPerSourceGroup,
              target_.memory().hemispheres * reductionsPerSourceGroup,
              recurringGroups, hemisphere, byte);
        }
      }
    }

    int64_t nextReductionDomain = 0;
    for (int64_t reductionBlock = 0; reductionBlock < reductionBlocks;
         ++reductionBlock) {
      const int64_t weightBuffer =
          (outputGroup * reductionBlocks + reductionBlock) %
          target_.throughput().mxm_weight_buffers;
      const int64_t firstCompute =
          firstComputeCycles[static_cast<std::size_t>(reductionBlock)];
      const int64_t dequantStart =
          dequantStartCycles[static_cast<std::size_t>(reductionBlock)];
      int64_t reductionDomainCount = 0;
      int64_t reductionDomainInterval = reductionBlocks > 1
                                            ? dequantStartCycles[1] -
                                                  dequantStartCycles[0]
                                            : 1;
      if (target_.throughput().mxm_weight_buffers > 2) {
        reductionDomainCount = 1;
        nextReductionDomain = reductionBlock + 1;
      } else if (reductionBlock == nextReductionDomain) {
        reductionDomainCount = 1;
        if (reductionBlock + 1 < reductionBlocks) {
          reductionDomainInterval =
              dequantStartCycles[static_cast<std::size_t>(reductionBlock + 1)] -
              dequantStart;
          while (reductionBlock + reductionDomainCount < reductionBlocks &&
                 dequantStartCycles[static_cast<std::size_t>(
                     reductionBlock + reductionDomainCount)] ==
                     dequantStart +
                         reductionDomainCount * reductionDomainInterval)
            ++reductionDomainCount;
        }
        nextReductionDomain = reductionBlock + reductionDomainCount;
      }
      for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
           ++hemisphere) {
        const char *hemi = hemisphere == 0 ? "east" : "west";
        const int64_t loadCycle = dequantStart + hemisphere * 8;
        if (reductionDomainCount != 0 && weightOutputGroupCount != 0) {
          for (int64_t stream = 0; stream < 8; ++stream) {
            const int64_t slice = selectedWeightSlices[stream];
            emitMem3D(
                rewriter_, op_.getLoc(), loadCycle - weightReadLatency(slice),
                hemisphere * target_.memory().slices_per_hemisphere + slice,
                "read",
                layout.outputWeightAddress(outputGroup, reductionBlock, 0),
                weightStreamBase + stream, 4, 1, 1, "sram",
                functionArgumentIndex(op_.getOutputWeight()),
                reductionDomainCount, reductionDomainInterval, 4,
                weightOutputGroupCount, outputGroupInterval,
                reductionBlocks * 4,
                layout.outputWeightBank(), layout.outputWeightPage());
          }
        }
        if (reductionDomainCount != 0 && localDequant &&
            !mergeMxmOutputGroups)
          emitMxmDequant3D(
              rewriter_, op_.getLoc(), loadCycle, hemisphere,
              outputWeightScale, 4, 1, reductionDomainCount,
              reductionDomainInterval, 1, 1,
              functionArgumentIndex(op_.getOutputWeight()));
        if (!localDequant) {
          for (int64_t pulse = 0; pulse < 4; ++pulse) {
            const int64_t cycle = loadCycle + pulse;
            for (int64_t lane = 0; lane < target_.throughput().lanes_per_tile;
                 ++lane) {
              emitVxm(rewriter_, op_.getLoc(), op_.getOutputWeight(), cycle,
                      lane, "multiply", "stream_i8", 32 + lane, 0.0f,
                      "immediate", 0, outputWeightScale, "fp32", -1, hemi, hemi,
                      functionArgumentIndex(op_.getOutputWeight()));
              emitVxm(rewriter_, op_.getLoc(), op_.getOutputWeight(), cycle + 1,
                      lane, "cast", "alu", lane, 0.0f, "immediate", 0, 0.0f,
                      dataFormat,
                      localMxm *
                              target_.throughput().mxm_load_streams_per_cycle +
                          lane * 2,
                      hemi, hemi);
            }
          }
        }
        if (!mergeMxmOutputGroups &&
            (reductionDomainCount != 0 ||
             target_.throughput().mxm_weight_buffers > 2)) {
          const int64_t loadGroupCount =
              target_.throughput().mxm_weight_buffers > 2
                  ? 1
                  : reductionDomainCount;
          MxmDomain3D loadDomain;
          loadDomain.wave_count = 4;
          loadDomain.wave_interval = 1;
          loadDomain.wave_weight_column_stride = -1;
          loadDomain.group_count = loadGroupCount;
          loadDomain.group_interval = reductionDomainInterval;
          if (loadGroupCount > 1 &&
              target_.throughput().mxm_weight_buffers == 2)
            loadDomain.weight_buffer_mode = "toggle_dim2";
          emitMxm3D(
              rewriter_, op_.getLoc(), loadCycle + loadToIw,
              hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
              "iw", weightBuffer, 3, 0, 0, 0, 1, "stream", true,
              "supercell", 0, dataFormat,
              localDequant ? "int8_dequant_bf16" : llvm::StringRef{},
              llvm::StringRef{}, loadDomain,
              localDequant ? weightStreamBase : -1);
        }
      }

      const bool finalReduction = reductionBlock + 1 == reductionBlocks;
      int64_t finalWriteEnd = firstCompute;
      for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
           ++hemisphere) {
        if (!mergeMxmOutputGroups &&
            (reductionDomainCount != 0 ||
             target_.throughput().mxm_weight_buffers > 2)) {
          const int64_t computeGroupCount =
              target_.throughput().mxm_weight_buffers > 2
                  ? 1
                  : reductionDomainCount;
          const bool includesFinal =
              reductionBlock + computeGroupCount == reductionBlocks;
          const bool finalOnly = includesFinal && computeGroupCount == 1;
          MxmDomain3D computeDomain;
          computeDomain.repeat_count = projectionRows;
          computeDomain.repeat_accumulator_address_stride = 0;
          computeDomain.wave_count = tokenBlocks;
          computeDomain.wave_interval = computeInterval;
          computeDomain.wave_accumulator_address_stride = tile;
          computeDomain.group_count = computeGroupCount;
          computeDomain.group_interval = reductionDomainInterval;
          if (computeGroupCount > 1 &&
              target_.throughput().mxm_weight_buffers == 2)
            computeDomain.weight_buffer_mode = "toggle_dim2";
          if (includesFinal && computeGroupCount > 1) {
            computeDomain.terminal_dimension = 2;
            computeDomain.terminal_accumulator_destination = "stream";
            computeDomain.terminal_accumulator_clear = true;
            computeDomain.terminal_accumulator_output_format = dataFormat;
          }
          emitMxm3D(
              rewriter_, op_.getLoc(), firstCompute,
              hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
              "compute", weightBuffer, 0, hemisphere * 2, 0,
              accumulatorAddress(0), 1, finalOnly ? "stream" : "sram",
              finalOnly, "supercell", 0, dataFormat, llvm::StringRef{},
              finalOnly ? dataFormat : "fp32", computeDomain);
        }

        if (!finalReduction)
          continue;

        const int64_t resultCycle =
            firstCompute + target_.mxm_first_result_latency();
        for (int64_t byte = 0; byte < 2; ++byte) {
          const int64_t slice = resultSlices[hemisphere * 2 + byte];
          const int64_t latency = *target_.transport_latency(
              target::StreamEndpoint::MxmResult,
              target::StreamEndpoint::Mem,
              target::StreamDirection::West, slice);
          const int64_t packedStream =
              target_.streams().streams_per_direction + byte;
          const int64_t localWriteCycle = resultCycle + latency;
          const bool remoteResult = hemisphere != 0;
          if (!mergeResultOutputGroups || outputGroup == 0)
            emitMem3D(
                rewriter_, op_.getLoc(), localWriteCycle,
                hemisphere * target_.memory().slices_per_hemisphere + slice,
                remoteResult ? "write_tap" : "write",
                layout.resultAddress(outputGroup, 0), packedStream,
                op_.getSeqLen(), 1, 1, "sram", -1, 1, 1, 0,
                mergeResultOutputGroups ? outputGroups : 1,
                mergeResultOutputGroups ? outerGroupInterval : 1,
                mergeResultOutputGroups ? resultGroupAddressStride : 0,
                resultBank);
          finalWriteEnd =
              std::max(finalWriteEnd, localWriteCycle + op_.getSeqLen());
          if (remoteResult) {
            const int64_t group =
                slice / target_.streams().mem_slices_per_register_group;
            const int64_t remoteWriteCycle =
                resultCycle + target_.streams().system_register_columns +
                group + 1;
            if (!mergeResultOutputGroups || outputGroup == 0)
              emitMem3D(
                  rewriter_, op_.getLoc(), remoteWriteCycle, slice, "write",
                  layout.resultAddress(outputGroup, 0), byte,
                  op_.getSeqLen(), 1, 1, "sram", -1, 1, 1, 0,
                  mergeResultOutputGroups ? outputGroups : 1,
                  mergeResultOutputGroups ? outerGroupInterval : 1,
                  mergeResultOutputGroups ? resultGroupAddressStride : 0,
                  resultBank);
            finalWriteEnd =
                std::max(finalWriteEnd, remoteWriteCycle + op_.getSeqLen());
          }
        }
      }
      if (finalReduction)
        phaseStart = std::max(firstCompute + tokenBlocks * computeInterval,
                              finalWriteEnd);
    }
  }
  return phaseStart;
}

} // namespace ftlpu::compiler::schedule
