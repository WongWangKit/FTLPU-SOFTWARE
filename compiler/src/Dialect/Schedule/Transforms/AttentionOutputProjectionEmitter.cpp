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
  const int64_t tokenBlocks = op_.getSeqLen() / tile;
  const int64_t reductionBlocks = op_.getQueryHeads() * op_.getHeadDim() / tile;
  const int64_t outputGroups =
      op_.getHidden() / (tile * target_.memory().hemispheres);
  const int64_t localMxm = 0;
  const int64_t accumulatorLatency =
      target_.throughput().mxm0_accumulator_latency;
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
  for (int64_t outputGroup = 0; outputGroup < outputGroups; ++outputGroup) {
    const auto selectedWeightSlices = layout.outputWeightSlices(outputGroup);
    int64_t maxWeightReadLatency = 0;
    for (int64_t slice : selectedWeightSlices)
      maxWeightReadLatency =
          std::max(maxWeightReadLatency, weightReadLatency(slice));
    const int64_t initialReadyLead =
        std::max(maxWeightReadLatency + weightLoadLead, contextReadLatency);
    int64_t nextReductionCompute = phaseStart + initialReadyLead;
    std::vector<int64_t> weightBufferRelease(
        static_cast<std::size_t>(target_.throughput().mxm_weight_buffers),
        std::numeric_limits<int64_t>::min());
    for (int64_t reductionBlock = 0; reductionBlock < reductionBlocks;
         ++reductionBlock) {
      const int64_t weightBuffer =
          (outputGroup * reductionBlocks + reductionBlock) %
          target_.throughput().mxm_weight_buffers;
      int64_t firstCompute = nextReductionCompute;
      int64_t dequantStart = firstCompute - weightLoadLead;
      dequantStart = std::max(dequantStart,
          weightBufferRelease[static_cast<std::size_t>(weightBuffer)]);
      firstCompute = std::max(
          firstCompute, dequantStart + weightLoadLead);
      for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
           ++hemisphere) {
        const char *hemi = hemisphere == 0 ? "east" : "west";
        for (int64_t pulse = 0; pulse < 4; ++pulse) {
          const int64_t cycle = dequantStart + hemisphere * 8 + pulse;
          for (int64_t stream = 0; stream < 8; ++stream) {
            const int64_t slice = selectedWeightSlices[stream];
            emitMem(
                rewriter_, op_.getLoc(), cycle - weightReadLatency(slice),
                hemisphere * target_.memory().slices_per_hemisphere + slice,
                "read",
                layout.outputWeightAddress(outputGroup, reductionBlock, pulse),
                weightStreamBase + stream, 1, 1, 0, "sram",
                functionArgumentIndex(op_.getOutputWeight()),
                layout.outputWeightBank(), layout.outputWeightPage());
          }
          if (localDequant) {
            emitMxmDequant(rewriter_, op_.getLoc(), cycle, hemisphere,
                           outputWeightScale, 1, 1,
                           functionArgumentIndex(op_.getOutputWeight()));
          } else {
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
          emitMxm(rewriter_, op_.getLoc(), cycle + loadToIw,
                  hemisphere * target_.throughput().mxms_per_hemisphere +
                      localMxm,
                  "iw", weightBuffer, 3 - pulse, 0, 0, 1, 1, 0, 1, "stream",
                  true, "supercell", 0, dataFormat,
                  localDequant ? "int8_dequant_bf16" : "", {},
                  localDequant ? weightStreamBase : -1);
        }
      }

      const bool finalReduction = reductionBlock + 1 == reductionBlocks;
      const int64_t computeInterval = tile;
      const int64_t queryHead = reductionBlock / (op_.getHeadDim() / tile);
      const int64_t headBlock = reductionBlock % (op_.getHeadDim() / tile);
      for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
        const int64_t computeBase = firstCompute + tokenBlock * computeInterval;
        for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
             ++hemisphere) {
          const int64_t computeCycle = computeBase;
          for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice =
                layout.contextSlice(queryHead, headBlock, byte);
            const int64_t latency = *target_.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation,
                target::StreamDirection::East, slice);
            emitMem(rewriter_, op_.getLoc(), computeCycle - latency,
                    hemisphere * target_.memory().slices_per_hemisphere + slice,
                    "read", layout.contextAddress(
                        queryHead, headBlock, tokenBlock * tile),
                    hemisphere * 2 + byte, tile, 1, 1, "sram", -1,
                    contextBank);
          }
          emitMxm(rewriter_, op_.getLoc(), computeCycle,
                  hemisphere * target_.throughput().mxms_per_hemisphere +
                      localMxm,
                  "compute", weightBuffer, 0, hemisphere * 2, 0, tile, 1,
                  accumulatorAddress(tokenBlock * tile), 1, "sram", true,
                  "supercell", 0, dataFormat);
        }
      }
      const int64_t lastCompute =
          firstCompute + (tokenBlocks - 1) * computeInterval;
      weightBufferRelease[static_cast<std::size_t>(weightBuffer)] =
          lastCompute + target_.mxm_result_window_cycles(tile);
      nextReductionCompute = firstCompute + tokenBlocks * computeInterval;
      if (finalReduction) {
        phaseStart = nextReductionCompute + accumulatorLatency;
        const int64_t writeStart = phaseStart;
        int64_t finalWriteEnd = writeStart;
        for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
             ++hemisphere) {
          const int64_t mxmOutputStream = 0;
          for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
            const int64_t writeCycle = writeStart + token;
            emitMxm(rewriter_, op_.getLoc(),
                    writeCycle,
                    hemisphere * target_.throughput().mxms_per_hemisphere +
                        localMxm,
                    "accumulator_read", 0, 0, 0, mxmOutputStream, 1, 1,
                    accumulatorAddress(token), 1, "stream", true, "supercell",
                    0, dataFormat, {}, dataFormat);
            for (int64_t byte = 0; byte < 2; ++byte) {
              const int64_t slice = resultSlices[hemisphere * 2 + byte];
              const int64_t latency = *target_.transport_latency(
                  target::StreamEndpoint::MxmResult,
                  target::StreamEndpoint::Mem,
                  target::StreamDirection::West, slice);
              const int64_t packedStream =
                  target_.streams().streams_per_direction + mxmOutputStream +
                  byte;
              const int64_t localWriteCycle = writeCycle + latency;
              const bool remoteResult = hemisphere != 0;
              emitMem(rewriter_, op_.getLoc(), localWriteCycle,
                      hemisphere * target_.memory().slices_per_hemisphere +
                          slice,
                      remoteResult ? "write_tap" : "write",
                      layout.resultAddress(outputGroup, token), packedStream,
                      1, 1, 0, "sram", -1, resultBank);
              finalWriteEnd = std::max(finalWriteEnd, localWriteCycle + 1);
              if (remoteResult) {
                const int64_t group =
                    slice / target_.streams().mem_slices_per_register_group;
                const int64_t remoteWriteCycle =
                    writeCycle + target_.streams().system_register_columns +
                    group + 1;
                emitMem(rewriter_, op_.getLoc(), remoteWriteCycle, slice,
                        "write", layout.resultAddress(outputGroup, token),
                        mxmOutputStream + byte, 1, 1, 0, "sram", -1,
                        resultBank);
                finalWriteEnd =
                    std::max(finalWriteEnd, remoteWriteCycle + 1);
              }
            }
          }
        }
        phaseStart = finalWriteEnd;
      }
    }
  }
  return phaseStart;
}

} // namespace ftlpu::compiler::schedule
