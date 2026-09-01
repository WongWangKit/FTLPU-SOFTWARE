#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/paged_weight_residency.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

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

int64_t AttentionScheduleEmitter::emitProjections() {
  const AttentionMemoryLayout layout(op_, target_);
  const auto elementType =
      llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
          .getElementType();
  const llvm::StringRef streamKind = lpu_16bit_stream_kind(elementType);
  const llvm::StringRef dataFormat = lpu_16bit_data_format(elementType);
  const int64_t tile = target_.throughput().mxm_rows;
  const int64_t tokenBlocks = op_.getSeqLen() / tile;
  const int64_t hiddenBlocks = op_.getHidden() / tile;
  const int64_t projectionHeadBlocks = op_.getHeadDim() / tile;
  const int64_t weightToIw = target_.throughput().vxm_weight_to_iw_latency;
  const bool localDequant = target_.supports_mxm_local_dequant();
  const int64_t loadToIw = localDequant ? 0 : weightToIw;
  const int64_t weightStreamBase =
      localDequant ? target_.streams().streams_per_direction -
                         target_.throughput().mxm_int8_load_streams_per_cycle
                   : target_.streams().streams_per_direction;
  const int64_t overlapActivationStreamBase =
      target_.throughput().mxm_activation_streams;
  const auto inputPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("input");
  const auto inputKind = inputPlacement.getAs<mlir::StringAttr>("kind");
  const bool inputDistributed16 =
      inputKind && inputKind.getValue() == "fp16_mxm_distributed_16";
  const int64_t inputBase =
      inputPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
  const auto inputStagingPlacement =
      inputDistributed16
          ? op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("input_staging")
          : mlir::DictionaryAttr{};
  llvm::SmallVector<int64_t, 16> inputSlices;
  for (mlir::Attribute value : inputPlacement.getAs<mlir::ArrayAttr>("slices"))
    inputSlices.push_back(llvm::cast<mlir::IntegerAttr>(value).getInt());
  llvm::SmallVector<int64_t, 2> inputStagingSlices;
  if (inputStagingPlacement)
    for (mlir::Attribute value :
         inputStagingPlacement.getAs<mlir::ArrayAttr>("slices"))
      inputStagingSlices.push_back(
          llvm::cast<mlir::IntegerAttr>(value).getInt());
  const auto projectionActivationSlices =
      inputDistributed16
          ? inputStagingSlices
          : llvm::SmallVector<int64_t>(layout.activationSlices().begin(),
                                       layout.activationSlices().end());
  const int64_t activationLatency = *target_.transport_latency(
      target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
      target::StreamDirection::East, projectionActivationSlices.front());
  const int64_t projectionHeads[] = {op_.getQueryHeads(), op_.getKvHeads(),
                                     op_.getKvHeads()};
  const mlir::Value projectionValues[] = {
      op_.getQueryWeight(), op_.getKeyWeight(), op_.getValueWeight()};
  const auto weightScale = [&](llvm::StringRef name) {
    const auto value = op_.query.getConfig().getAs<mlir::FloatAttr>(name);
    return value ? static_cast<float>(value.getValueAsDouble()) : 1.0f;
  };
  const float projectionScales[] = {
      weightScale("query_weight_scale"),
      weightScale("key_weight_scale"),
      weightScale("value_weight_scale"),
  };
  int64_t phaseStart = 0;

  const auto readLatency = [&](int64_t slice) {
    return slice / target_.streams().mem_slices_per_register_group + 2;
  };
  const auto vxmInputReadLatency = [&](int64_t slice) {
    return target_
        .transport_latency(target::StreamEndpoint::Mem,
                           target::StreamEndpoint::VxmInput,
                           target::StreamDirection::West, slice)
        .value_or(readLatency(slice));
  };
  const auto weightReadLatency = [&](int64_t slice) {
    return target_
        .transport_latency(target::StreamEndpoint::Mem,
                           target::StreamEndpoint::MxmWeight,
                           target::StreamDirection::East, slice)
        .value_or(readLatency(slice));
  };
  const auto emitDequant = [&](int64_t cycle, int64_t hemisphere,
                               int64_t localMxm, mlir::Value weight,
                               float scale) {
    const char *hemi = hemisphere == 0 ? "east" : "west";
    for (int64_t lane = 0; lane < target_.throughput().lanes_per_tile; ++lane) {
      emitVxm(rewriter_, op_.getLoc(), weight, cycle, lane, "multiply",
              "stream_i8", 32 + lane, 0.0f, "immediate", 0, scale, "fp32", -1,
              hemi, hemi, functionArgumentIndex(weight));
      emitVxm(rewriter_, op_.getLoc(), weight, cycle + 1, lane, "cast", "alu",
              lane, 0.0f, "immediate", 0, 0.0f, dataFormat,
              localMxm * 16 + lane * 2, hemi, hemi);
    }
  };
  const auto emitRopeOrCast = [&](int64_t cycle, int64_t hemisphere, bool rope,
                                  mlir::Value value) {
    attention_detail::emitRopeOrCast(rewriter_, op_.getLoc(), target_, cycle,
                                     hemisphere, rope, value, elementType);
  };
  const auto placementBank = [&](llvm::StringRef name) {
    return op_.getMemoryPlan()
        .getAs<mlir::DictionaryAttr>(name)
        .getAs<mlir::IntegerAttr>("bank")
        .getInt();
  };
  const int64_t inputBank = placementBank("input");
  const int64_t inputStagingBank = inputDistributed16
                                       ? placementBank("input_staging")
                                       : inputBank;
  const int64_t projectionActivationBank =
      inputDistributed16 ? inputStagingBank : inputBank;
  const int64_t stagingBank = placementBank("rope_staging");
  const int64_t ropeBank = placementBank("rope");
  const auto ropeMirrorPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("rope_mirror");
  const int64_t ropeMirrorBank = ropeMirrorPlacement
                                     ? ropeMirrorPlacement
                                           .getAs<mlir::IntegerAttr>("bank")
                                           .getInt()
                                     : ropeBank;
  const int64_t keyBank = placementBank("key");
  const int64_t valueBank = placementBank("value");
  const int64_t productBank = placementBank("rope_product");
  const auto keyProductPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("rope_product_key");
  const int64_t keyProductBank = keyProductPlacement
                                     ? keyProductPlacement
                                           .getAs<mlir::IntegerAttr>("bank")
                                           .getInt()
                                     : productBank;
  const auto slicesOverlap = [](llvm::ArrayRef<int64_t> lhs,
                                 llvm::ArrayRef<int64_t> rhs) {
    return llvm::any_of(lhs, [&](int64_t slice) {
      return llvm::is_contained(rhs, slice);
    });
  };
  const auto productConflictsWithInput =
      [&](int64_t bank, llvm::ArrayRef<int64_t> slices) {
        return inputStagingBank == bank &&
               slicesOverlap(inputStagingSlices, slices);
      };
  const int64_t alternateProductBank =
      (productBank + 1) % target_.memory().banks_per_slice;
  const bool directRopeCapable =
      target_.uses_dedicated_slice_roles() &&
      target_.memory().banks_per_slice > 1 &&
      target_.activation_storage_slices().size() >= 20 &&
      target_.throughput().vxm_cross_hemisphere_streams_enabled != 0 &&
      target_.throughput().vxm_fma_enabled != 0 &&
      target_.throughput().vxm_alus >= 4 &&
      target_.memory().hemispheres == 2 && projectionHeadBlocks == 4;
  const bool projectionRopeOverlap =
      !directRopeCapable && inputDistributed16 &&
      overlapActivationStreamBase + 2 <=
          target_.streams().streams_per_direction &&
      !productConflictsWithInput(productBank,
                                 layout.ropeProductSlices()) &&
      !productConflictsWithInput(alternateProductBank,
                                 layout.ropeProductSlices()) &&
      !productConflictsWithInput(keyProductBank,
                                 layout.ropeProductKeySlices());

  if (inputDistributed16) {
    const auto &stagingSlices = inputStagingSlices;
    if (inputSlices.size() != 16 || stagingSlices.size() < 2) {
      op_.emitError("distributed attention input requires 16 source slices "
                    "and one FP16 staging pair");
      return -1;
    }
    int64_t stagingCycle = 16;
    int64_t lastWriteCycle = stagingCycle;
    const int64_t stagingBase =
        inputStagingPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks;
         ++reductionBlock) {
      for (int64_t token = 0; token < op_.getSeqLen();
           ++token, ++stagingCycle) {
        const int64_t tokenBlock = token / tile;
        const int64_t tokenWithinBlock = token % tile;
        const int64_t tokenWave = tokenWithinBlock / 8;
        const int64_t tokenLane = tokenWithinBlock % 8;
        const int64_t sourceAddress =
            inputBase + (tokenBlock * hiddenBlocks + reductionBlock) * 4 +
            tokenWave;
        const int64_t stagingAddress =
            stagingBase + reductionBlock * op_.getSeqLen() + token;
        for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
             ++hemisphere) {
          for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t sourceSlice = inputSlices[2 * tokenLane + byte];
            const int64_t sourceLatency = *target_.transport_latency(
                target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, sourceSlice);
            emitMem(rewriter_, op_.getLoc(), stagingCycle - sourceLatency,
                    hemisphere * target_.memory().slices_per_hemisphere +
                        sourceSlice,
                    "read", sourceAddress, 32 + hemisphere * 16 + byte, 1, 1,
                    0, "sram", -1, inputBank);
          }
          if (hemisphere == 0) {
            // VXM configuration reaches tile 0 one cycle before the ALU
            // consumes its first bundle.  Announce the stream group before
            // the MEM-edge transfer phase so the fabric captures, rather
            // than passively bridges, the arriving activation bytes.
            const int64_t configCycle = stagingCycle - 1;
            emitVxmConfigured(rewriter_, op_.getLoc(), op_.getInput(),
                              configCycle, 0, "pass", streamKind, 32, 0.0f,
                              "immediate", 0, 0.0f, "fp32", -1, "east", "east",
                              -1, 2, 1, 1);
            emitVxmConfigured(rewriter_, op_.getLoc(), op_.getInput(),
                              configCycle, 1, "pass", "previous", 0, 0.0f,
                              "immediate", 0, 0.0f, dataFormat, 0, "east",
                              "east", -1, 2, 1, 1);
          }
          for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t destinationSlice = stagingSlices[byte];
            // Logical VXM C0 is mirrored to physical C8. Their fixed output
            // blocks are 0 (routed West) and 4 (routed East), respectively.
            const int64_t outputStream =
                (hemisphere == 0 ? int64_t{8} : int64_t{0}) + byte;
            const int64_t destinationLatency = *target_.transport_latency(
                target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
                target::StreamDirection::East, destinationSlice);
            const int64_t writeCycle = stagingCycle + 1 + destinationLatency;
            emitMem(rewriter_, op_.getLoc(), writeCycle,
                    hemisphere * target_.memory().slices_per_hemisphere +
                        destinationSlice,
                    "write", stagingAddress, outputStream, 1, 1, 0,
                    "sram", -1, inputStagingBank);
            lastWriteCycle = std::max(lastWriteCycle, writeCycle);
          }
        }
      }
    }
    // The first MXM activation read is scheduled before its compute
    // issue. Keep that backwards read window beyond the final staging
    // write, not merely the first weight-load cycle.
    phaseStart = std::max(phaseStart, lastWriteCycle + activationLatency + 1);
  }

  if (target_.throughput().mxms_per_hemisphere == 1) {
    const int64_t accumulatorHalfStride =
        target_.throughput().mxm_accumulator_blocks
        * target_.throughput().mxm_rows / 2;
    const int64_t conservativeComputeSpacing = tile;
    int64_t projectionBlock = 0;
    int64_t postprocessReady = phaseStart;
    int64_t projectionOverlapDeadline = -1;
    struct ResidentWeight {
      mlir::DictionaryAttr placement;
      int64_t releaseCycle;
    };
    llvm::SmallVector<ResidentWeight, 3> residentWeights;

    // Value is consumed only by PV, while Key gates the immediately following
    // QK stage. Scheduling Q -> V -> K shortens Key's live range and releases
    // Value's weight residency early enough to hide a conflicting next-stage
    // page refill without adding an MXM bubble.
    const int64_t projectionOrder[] = {0, 2, 1};
    for (int64_t projectionPosition = 0; projectionPosition < 3;
         ++projectionPosition) {
      const int64_t projection = projectionOrder[projectionPosition];
      const auto kind = projectionKind(projection);
      const char *weightPlacementNames[] = {
          "query_weight", "key_weight", "value_weight"};
      const auto weightPlacement = op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(
          weightPlacementNames[projection]);
      if (weightPlacement) {
        if (const auto transferCycles =
                weightPlacement.getAs<mlir::IntegerAttr>(
                    "page_transfer_cycles")) {
          int64_t maxWeightReadLatency = 0;
          for (mlir::Attribute slice : weightPlacement.getAs<mlir::ArrayAttr>(
                   "page_storage_slices"))
            maxWeightReadLatency = std::max(
                maxWeightReadLatency,
                weightReadLatency(
                    llvm::cast<mlir::IntegerAttr>(slice).getInt()));
          for (const ResidentWeight &resident : residentWeights) {
            if (!pagedWeightResidencyOverlaps(
                    resident.placement, weightPlacement))
              continue;
            phaseStart = std::max(
                phaseStart,
                resident.releaseCycle + transferCycles.getInt() +
                    kC2cTransportGuardCycles + maxWeightReadLatency);
          }
        }
      }
      int64_t currentProjectionWeightRelease = -1;
      const int64_t projectionOutputBlocks =
          projectionHeads[projection] * projectionHeadBlocks;
      const int64_t projectionGroups = (projectionOutputBlocks + 3) / 4;
      for (int64_t outputGroup = 0; outputGroup < projectionGroups;
           ++outputGroup) {
        int64_t rawWriteEnd = phaseStart;
        for (int64_t half = 0; half < 2; ++half) {
          for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks;
               ++reductionBlock) {
            const bool finalReduction = reductionBlock + 1 == hiddenBlocks;
            bool overlapsRopeProducts = false;
            if (projectionRopeOverlap && phaseStart < postprocessReady) {
              const int64_t computeEnd =
                  phaseStart + 4 + loadToIw + tokenBlocks * tile;
              if (projectionOverlapDeadline < 0 ||
                  computeEnd > projectionOverlapDeadline)
                phaseStart = postprocessReady;
              else
                overlapsRopeProducts = true;
            }
            if (projectionRopeOverlap && finalReduction) {
              phaseStart = std::max(phaseStart, postprocessReady);
              overlapsRopeProducts = false;
            }
            const int64_t activationStreamBase =
                overlapsRopeProducts ? overlapActivationStreamBase : 0;
            const int64_t weightBuffer =
                projectionBlock % target_.throughput().mxm_weight_buffers;
            const int64_t dequantStart = phaseStart;
            for (int64_t hemisphere = 0;
                 hemisphere < target_.memory().hemispheres; ++hemisphere) {
              const int64_t outputBlock =
                  outputGroup * 4 + hemisphere * 2 + half;
              if (outputBlock >= projectionOutputBlocks)
                continue;
              const auto selectedWeightSlices =
                  layout.weightSlices(kind, outputBlock);
              for (int64_t pulse = 0; pulse < 4; ++pulse) {
                const int64_t cycle = dequantStart + pulse;
                const int64_t address = layout.weightAddress(
                    kind, outputBlock, reductionBlock, half, pulse);
                for (int64_t stream = 0; stream < 8; ++stream) {
                  const int64_t slice = selectedWeightSlices[stream];
                  const int64_t readCycle = cycle - weightReadLatency(slice);
                  emitMem(rewriter_, op_.getLoc(), readCycle,
                          hemisphere * target_.memory().slices_per_hemisphere +
                              slice,
                          "read", address, weightStreamBase + stream, 1, 1, 0,
                          "sram",
                          functionArgumentIndex(projectionValues[projection]),
                          layout.weightBank(kind), layout.weightPage(kind));
                  currentProjectionWeightRelease =
                      std::max(currentProjectionWeightRelease, readCycle + 1);
                }
                if (localDequant)
                  emitMxmDequant(
                      rewriter_, op_.getLoc(), cycle, hemisphere,
                      projectionScales[projection], 1, 1,
                      functionArgumentIndex(projectionValues[projection]));
                else
                  emitDequant(cycle, hemisphere, 0,
                              projectionValues[projection],
                              projectionScales[projection]);
                emitMxm(rewriter_, op_.getLoc(), cycle + loadToIw, hemisphere,
                        "iw", weightBuffer, 3 - pulse, 0, 0, 1, 1, 0, 1,
                        "stream", true, "supercell", 0, dataFormat,
                        localDequant ? "int8_dequant_bf16" : "", {},
                        localDequant ? weightStreamBase : -1);
              }
            }

            const int64_t firstCompute = dequantStart + 4 + loadToIw;
            const int64_t computeSpacing =
                finalReduction
                    ? tile + target_.throughput().accumulator_to_vxm_latency +
                          activationLatency + 16
                    : conservativeComputeSpacing;
            for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks;
                 ++tokenBlock) {
              const int64_t computeCycle =
                  firstCompute + tokenBlock * computeSpacing;
              for (int64_t hemisphere = 0;
                   hemisphere < target_.memory().hemispheres; ++hemisphere) {
                const int64_t outputBlock =
                    outputGroup * 4 + hemisphere * 2 + half;
                if (outputBlock >= projectionOutputBlocks)
                  continue;
                const int64_t head = outputBlock / projectionHeadBlocks;
                const int64_t headBlock = outputBlock % projectionHeadBlocks;
                const int64_t inputAddress =
                    layout.activationAddress(reductionBlock, tokenBlock) +
                    (inputDistributed16
                         ? inputStagingPlacement
                               .getAs<mlir::IntegerAttr>("base_row")
                               .getInt()
                         : 0);
                for (int64_t byte = 0; byte < (inputDistributed16 ? 2 : 4);
                     ++byte) {
                  const int64_t slice = projectionActivationSlices[byte];
                  emitMem(rewriter_, op_.getLoc(),
                          computeCycle - activationLatency,
                          hemisphere * target_.memory().slices_per_hemisphere +
                              slice,
                          "read", inputAddress, activationStreamBase + byte,
                          tile, 1, 1,
                          "sram", -1, projectionActivationBank);
                }
                // Projection waves are fully drained before another head or
                // output group reuses this physical MXM. Keep accumulator
                // addresses in one token window instead of aliasing the much
                // larger MEM result address space.
                const int64_t accumulatorAddress =
                    tokenBlock * tile + half * accumulatorHalfStride;
                const int64_t resultStreamBase =
                    directRopeCapable && finalReduction &&
                            kind != AttentionProjectionKind::Value &&
                            hemisphere == 1
                        ? target_.streams().streams_per_direction / 2
                        : 0;
                emitMxm(rewriter_, op_.getLoc(), computeCycle, hemisphere,
                        "compute", weightBuffer, 0, activationStreamBase,
                        resultStreamBase,
                        tile, 1,
                        accumulatorAddress, 1,
                        finalReduction ? "stream" : "sram", finalReduction,
                        "supercell", 0, dataFormat, {},
                        finalReduction ? dataFormat : "fp32");
                if (!finalReduction)
                  continue;

                for (int64_t offset = 0; offset < tile; ++offset) {
                  const int64_t token = tokenBlock * tile + offset;
                  const int64_t resultCycle =
                      computeCycle + target_.mxm_first_result_latency() +
                      offset;
                  if (kind != AttentionProjectionKind::Value) {
                    if (directRopeCapable) {
                      if (hemisphere != 0)
                        continue;

                      const int64_t vxmInputCycle =
                          computeCycle +
                          target_.throughput().accumulator_to_vxm_latency +
                          offset;
                      if (offset == 0) {
                        const int64_t configCycle = vxmInputCycle - 1;
                        emitVxmConfigured(
                            rewriter_, op_.getLoc(),
                            projectionValues[projection], configCycle, 0,
                            "multiply", streamKind, 32, 0.0f, streamKind, 40,
                            0.0f, "fp32", -1, "east", "east", -1, 2, tile,
                            1, "east", "east");
                        emitVxmConfigured(
                            rewriter_, op_.getLoc(),
                            projectionValues[projection], configCycle, 1,
                            "fms", streamKind, 32, 0.0f, streamKind, 42,
                            0.0f, dataFormat, 0, "east", "east", -1, 2, tile,
                            1, "west", "east");
                        emitVxmConfigured(
                            rewriter_, op_.getLoc(),
                            projectionValues[projection], configCycle, 2,
                            "multiply", streamKind, 32, 0.0f, streamKind, 40,
                            0.0f, "fp32", -1, "east", "east", -1, 2, tile,
                            1, "west", "east");
                        emitVxmConfigured(
                            rewriter_, op_.getLoc(),
                            projectionValues[projection], configCycle, 3,
                            "fma", streamKind, 32, 0.0f, streamKind, 42,
                            0.0f, dataFormat, 2, "east", "east", -1, 2, tile,
                            1, "east", "east");
                      }

                      const bool useMirrorTable =
                          kind == AttentionProjectionKind::Key;
                      const auto ropeSlices = useMirrorTable
                                                  ? layout.ropeMirrorSlices()
                                                  : layout.ropeSlices();
                      const int64_t directRopeBank = useMirrorTable
                                                         ? layout.ropeMirrorBank()
                                                         : layout.ropeBank();
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t cosineSlice =
                            ropeSlices[byte];
                        emitMem(rewriter_, op_.getLoc(),
                                vxmInputCycle -
                                    vxmInputReadLatency(cosineSlice),
                                cosineSlice, "read",
                                useMirrorTable
                                    ? layout.ropeMirrorAddress(token, half)
                                    : layout.ropeAddress(token, half),
                                40 + byte, 1, 1, 0, "sram", -1,
                                directRopeBank);
                        const int64_t sineSlice =
                            ropeSlices[2 + byte];
                        emitMem(rewriter_, op_.getLoc(),
                                vxmInputCycle -
                                    vxmInputReadLatency(sineSlice),
                                sineSlice, "read",
                                useMirrorTable
                                    ? layout.ropeMirrorAddress(token, half)
                                    : layout.ropeAddress(token, half),
                                42 + byte, 1, 1, 0, "sram", -1,
                                directRopeBank);
                      }

                      const int64_t tokenLane =
                          token % target_.throughput().mxm_block_rows;
                      const int64_t rowBlock =
                          (token % tile) /
                          target_.throughput().mxm_block_rows;
                      const int64_t blocks[] = {half, half + 2};
                      const int64_t outputCycle = vxmInputCycle + 3;
                      for (int64_t outputHalf = 0; outputHalf < 2;
                           ++outputHalf) {
                        const int64_t reduction = blocks[outputHalf];
                        for (int64_t byte = 0; byte < 2; ++byte) {
                          const int64_t slice =
                              kind == AttentionProjectionKind::Query
                                  ? layout.queryIwSlices(reduction)
                                        [2 * tokenLane + byte]
                                  : layout.keySlices(reduction)[byte];
                          const int64_t latency =
                              target_
                                  .transport_latency(
                                      target::StreamEndpoint::VxmResult,
                                      target::StreamEndpoint::Mem,
                                      target::StreamDirection::East, slice)
                                  .value_or(readLatency(slice));
                          for (int64_t destination = 0;
                               destination < target_.memory().hemispheres;
                               ++destination) {
                            const int64_t source = 1 - destination;
                            const int64_t bank =
                                kind == AttentionProjectionKind::Query
                                    ? layout.queryIwBank(reduction)
                                    : layout.keyBank(reduction);
                            emitMem(
                                rewriter_, op_.getLoc(),
                                outputCycle + latency,
                                destination *
                                        target_.memory()
                                            .slices_per_hemisphere +
                                    slice,
                                "write",
                                kind == AttentionProjectionKind::Query
                                    ? layout.queryIwAddress(
                                          head, reduction, tokenBlock,
                                          rowBlock)
                                    : layout.keyAddress(head, reduction,
                                                        tokenBlock) +
                                          token % tile,
                                source * 8 + outputHalf * 2 + byte, 1, 1, 0,
                                "sram", -1, bank);
                            rawWriteEnd = std::max(
                                rawWriteEnd, outputCycle + latency + 1);
                          }
                        }
                      }
                      continue;
                    }
                    const int64_t packedStream = (token % 8) * 2;
                    const int64_t row = (token % tile) / 8;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                      const int64_t slice =
                          layout.ropeStagingSlices()[(packedStream + byte +
                                                      2 * headBlock) %
                                                     16];
                      const int64_t latency = *target_.transport_latency(
                          target::StreamEndpoint::MxmResult,
                          target::StreamEndpoint::Mem,
                          target::StreamDirection::West, slice);
                      const int64_t cycle = resultCycle + latency;
                      emitMem(rewriter_, op_.getLoc(), cycle,
                              hemisphere *
                                      target_.memory().slices_per_hemisphere +
                                  slice,
                              "write",
                              layout.ropeStagingAddress(kind, head, headBlock,
                                                        tokenBlock, row),
                              32 + byte, 1, 1, 0, "sram", -1,
                              stagingBank);
                      rawWriteEnd = std::max(rawWriteEnd, cycle + 1);
                    }
                  } else {
                    const int64_t packedStream = (token % 8) * 2;
                    const int64_t row = (token % tile) / 8;
                    const auto slices = layout.valuePackSlices(headBlock);
                    for (int64_t byte = 0; byte < 2; ++byte) {
                      const int64_t slice = slices[packedStream + byte];
                      const int64_t latency = *target_.transport_latency(
                          target::StreamEndpoint::MxmResult,
                          target::StreamEndpoint::Mem,
                          target::StreamDirection::West, slice);
                      const int64_t cycle = resultCycle + latency;
                      emitMem(rewriter_, op_.getLoc(), cycle,
                              hemisphere *
                                      target_.memory().slices_per_hemisphere +
                                  slice,
                              "write",
                              layout.valuePackAddress(head, headBlock,
                                                      tokenBlock, row),
                              32 + byte, 1, 1, 0, "sram", -1,
                              placementBank("value"));
                      rawWriteEnd = std::max(rawWriteEnd, cycle + 1);
                    }
                  }
                }
              }
            }
            // Double-buffer the next weight tile under the current 32-row
            // compute window. The next compute starts exactly when the last
            // issued row retires; result transport is tracked independently
            // by rawWriteEnd and may overlap a different accumulator window.
            const int64_t lastCompute = firstCompute
                + (tokenBlocks - 1) * computeSpacing;
            phaseStart = lastCompute + tile - (4 + loadToIw);
            ++projectionBlock;
          }
        }

        const auto &memory = target_.memory();
        const int64_t blockRows = target_.throughput().mxm_block_rows;
        const int64_t blockIssues = tile / blockRows;
        int64_t maxStagingReadLatency = 0;
        for (int64_t slice : layout.ropeStagingSlices())
          maxStagingReadLatency =
              std::max(maxStagingReadLatency, readLatency(slice));
        const int64_t firstOutputBlock = outputGroup * 4;
        const int64_t lastOutputBlock =
            std::min<int64_t>(projectionOutputBlocks, firstOutputBlock + 4);

        if (kind == AttentionProjectionKind::Value) {
          int64_t copyCycle =
              std::max({phaseStart, postprocessReady,
                        rawWriteEnd + maxStagingReadLatency + 1});
          int64_t copyEnd = copyCycle;
          for (int64_t outputBlock = firstOutputBlock;
               outputBlock < lastOutputBlock; ++outputBlock) {
            const int64_t head = outputBlock / projectionHeadBlocks;
            const int64_t headBlock = outputBlock % projectionHeadBlocks;
            const int64_t sourceHemisphere = (outputBlock % 4) / 2;
            const int64_t destinationHemisphere = head % memory.hemispheres;
            if (sourceHemisphere == destinationHemisphere)
              continue;
            const auto slices = layout.valuePackSlices(headBlock);
            for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks;
                 ++tokenBlock) {
              for (int64_t beat = 0; beat < blockIssues; ++beat, ++copyCycle) {
                const int64_t address =
                    layout.valuePackAddress(head, headBlock, tokenBlock, beat);
                for (int64_t stream = 0; stream < 16; ++stream) {
                  const int64_t slice = slices[stream];
                  emitMem(
                      rewriter_, op_.getLoc(), copyCycle - readLatency(slice),
                      sourceHemisphere * memory.slices_per_hemisphere + slice,
                      "read", address, 32 + stream, 1, 1, 0, "sram", -1,
                      placementBank("value"));
                  const int64_t latency =
                      target_
                          .transport_latency(
                              target::StreamEndpoint::VxmBridgeResult,
                              target::StreamEndpoint::Mem,
                              target::StreamDirection::East, slice)
                          .value_or(readLatency(slice));
                  emitMem(rewriter_, op_.getLoc(), copyCycle + latency,
                          destinationHemisphere * memory.slices_per_hemisphere +
                              slice,
                          "write", address, stream, 1, 1, 0, "sram", -1,
                          placementBank("value"));
                  copyEnd = std::max(copyEnd, copyCycle + latency + 1);
                }
              }
            }
            copyCycle =
                std::max(copyCycle, copyEnd + maxStagingReadLatency + 1);
          }
          phaseStart = std::max(copyCycle + 1, copyEnd);
          postprocessReady = phaseStart;
          projectionOverlapDeadline = -1;
          continue;
        }

        if (directRopeCapable) {
          phaseStart = std::max(phaseStart, rawWriteEnd);
          postprocessReady = phaseStart;
          projectionOverlapDeadline = -1;
          continue;
        }

        int64_t replicateCycle =
            std::max({phaseStart, postprocessReady,
                      rawWriteEnd + maxStagingReadLatency + 1});
        int64_t replicateEnd = replicateCycle;
        for (int64_t outputBlock = firstOutputBlock;
             outputBlock < lastOutputBlock; ++outputBlock) {
          const int64_t head = outputBlock / projectionHeadBlocks;
          const int64_t headBlock = outputBlock % projectionHeadBlocks;
          const int64_t sourceHemisphere = (outputBlock % 4) / 2;
          const int64_t destinationHemisphere = 1 - sourceHemisphere;
          for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
            for (int64_t beat = 0; beat < blockIssues;
                 ++beat, ++replicateCycle) {
              const int64_t address = layout.ropeStagingAddress(
                  kind, head, headBlock, tokenBlock, beat);
              for (int64_t stream = 0; stream < 16; ++stream) {
                const int64_t slice =
                    layout.ropeStagingSlices()[(stream + 2 * headBlock) % 16];
                emitMem(rewriter_, op_.getLoc(),
                        replicateCycle - readLatency(slice),
                        sourceHemisphere * memory.slices_per_hemisphere + slice,
                        "read", address, 32 + stream, 1, 1, 0, "sram", -1,
                        stagingBank);
                const int64_t latency =
                    target_
                        .transport_latency(
                            target::StreamEndpoint::VxmBridgeResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::East, slice)
                        .value_or(readLatency(slice));
                emitMem(rewriter_, op_.getLoc(), replicateCycle + latency,
                        destinationHemisphere * memory.slices_per_hemisphere +
                            slice,
                        "write", address, stream, 1, 1, 0, "sram", -1,
                        stagingBank);
                replicateEnd =
                    std::max(replicateEnd, replicateCycle + latency + 1);
              }
            }
          }
          replicateCycle = std::max(replicateCycle,
                                    replicateEnd + maxStagingReadLatency + 1);
        }

        const auto emitRopeProducts = [&](int64_t cycle,
                                          int64_t inputHemisphere,
                                          int64_t outputHemisphere) {
          const char *input = inputHemisphere == 0 ? "east" : "west";
          const char *output = outputHemisphere == 0 ? "east" : "west";
          emitVxmConfigured(rewriter_, op_.getLoc(),
                            projectionValues[projection], cycle, 0, "multiply",
                            streamKind, 32, 0.0f, streamKind, 34, 0.0f, "fp32",
                            -1, input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(
              rewriter_, op_.getLoc(), projectionValues[projection], cycle, 1,
              "pass", "previous", 0, 0.0f, "immediate", 0, 0.0f, dataFormat, 0,
              input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(rewriter_, op_.getLoc(),
                            projectionValues[projection], cycle, 2, "multiply",
                            streamKind, 36, 0.0f, streamKind, 38, 0.0f, "fp32",
                            -1, input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(
              rewriter_, op_.getLoc(), projectionValues[projection], cycle, 3,
              "pass", "previous", 0, 0.0f, "immediate", 0, 0.0f, dataFormat, 2,
              input, output, -1, 2, op_.getSeqLen(), 1);
        };
        const auto emitRopeCombine = [&](int64_t cycle, int64_t inputHemisphere,
                                         int64_t outputHemisphere) {
          const char *input = inputHemisphere == 0 ? "east" : "west";
          const char *output = outputHemisphere == 0 ? "east" : "west";
          emitVxmConfigured(rewriter_, op_.getLoc(),
                            projectionValues[projection], cycle, 0, "subtract",
                            streamKind, 32, 0.0f, streamKind, 34, 0.0f, "fp32",
                            -1, input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(
              rewriter_, op_.getLoc(), projectionValues[projection], cycle, 1,
              "pass", "previous", 0, 0.0f, "immediate", 0, 0.0f, dataFormat, 0,
              input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(rewriter_, op_.getLoc(),
                            projectionValues[projection], cycle, 2, "add",
                            streamKind, 36, 0.0f, streamKind, 38, 0.0f, "fp32",
                            -1, input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(
              rewriter_, op_.getLoc(), projectionValues[projection], cycle, 3,
              "pass", "previous", 0, 0.0f, "immediate", 0, 0.0f, dataFormat, 2,
              input, output, -1, 2, op_.getSeqLen(), 1);
        };

        int64_t ropeEnd =
            std::max(replicateCycle, replicateEnd + maxStagingReadLatency + 1);
        // MXM activation reads are issued before compute. Keep that backward
        // transport window beyond the full-width cross-hemisphere replication
        // before switching to the disjoint overlap stream group.
        const int64_t activationReadLead =
            std::max<int64_t>(0, activationLatency - (4 + loadToIw));
        const int64_t nextProjectionStart = ropeEnd + activationReadLead;
        const int64_t firstHead = firstOutputBlock / projectionHeadBlocks;
        const int64_t lastHead = (lastOutputBlock - 1) / projectionHeadBlocks;
        const int64_t queryHeadsPerKv = op_.getQueryHeads() / op_.getKvHeads();
        int64_t firstCombineConfig = -1;
        const auto baseProductSlices = layout.ropeProductSlices();
        int64_t maxProductWriteLatency = 0;
        int64_t maxProductReadLatency = 0;
        for (int64_t slice : baseProductSlices) {
          maxProductWriteLatency = std::max(
              maxProductWriteLatency,
              target_
                  .transport_latency(target::StreamEndpoint::VxmResult,
                                     target::StreamEndpoint::Mem,
                                     target::StreamDirection::East, slice)
                  .value_or(readLatency(slice)));
          maxProductReadLatency =
              std::max(maxProductReadLatency, vxmInputReadLatency(slice));
        }
        for (int64_t slice : layout.ropeProductKeySlices()) {
          maxProductWriteLatency = std::max(
              maxProductWriteLatency,
              target_
                  .transport_latency(target::StreamEndpoint::VxmResult,
                                     target::StreamEndpoint::Mem,
                                     target::StreamDirection::East, slice)
                  .value_or(readLatency(slice)));
          maxProductReadLatency =
              std::max(maxProductReadLatency, vxmInputReadLatency(slice));
        }
        for (int64_t head = firstHead; head <= lastHead; ++head) {
          int64_t headEnd = ropeEnd;
          for (int64_t pairBlock = 0; pairBlock < projectionHeadBlocks / 2;
               ++pairBlock) {
            const int64_t blocks[] = {pairBlock,
                                      pairBlock + projectionHeadBlocks / 2};
            const int64_t highOutputBlock =
                head * projectionHeadBlocks + blocks[1];
            const int64_t inputHemisphere = (highOutputBlock % 4) / 2;
            const int64_t outputHemisphere =
                kind == AttentionProjectionKind::Query
                    ? (head / queryHeadsPerKv) % memory.hemispheres
                    : head % memory.hemispheres;
            const LPUResourceModel resourceModel(target_);
            const auto appendMemResources = [&](auto &windows, int64_t cycle,
                                                int64_t hemisphere,
                                                int64_t slice, int64_t bank,
                                                bool write) {
              windows.push_back(
                  {resourceModel.mem_icu(hemisphere, slice, bank), cycle, 1});
              windows.push_back(
                  {write ? resourceModel.mem_write_port(hemisphere, slice, bank)
                         : resourceModel.mem_read_port(hemisphere, slice, bank),
                   cycle, 1});
            };
            const auto ropeResourcesAreLegal =
                [&](llvm::ArrayRef<int64_t> candidateSlices) {
                  llvm::SmallVector<ResourceWindow, 0> windows;
                  const bool compactProductLayout =
                      candidateSlices.size() < 16;
                  const auto candidateProductSlice =
                      [&](int64_t product, int64_t token, int64_t byte) {
                        if (compactProductLayout)
                          return layout.ropeProductSlice(
                              kind, product, token, byte);
                        return candidateSlices[(token % 2) * 8 +
                                               product * 2 + byte];
                      };
                  const auto candidateProductBank = [&](int64_t product) {
                    return compactProductLayout
                               ? layout.ropeProductBank(
                                     kind, productBank, product)
                               : productBank;
                  };
                  const int64_t productAInputOffset = 1;
                  const int64_t productBConfigOffset =
                      productAInputOffset + op_.getSeqLen() + 1 +
                      maxProductWriteLatency + maxStagingReadLatency + 1;
                  const int64_t productBInputOffset =
                      productBConfigOffset + 1;
                  const int64_t combineConfigOffset =
                      productBInputOffset + op_.getSeqLen() + 1 +
                      maxProductWriteLatency + maxProductReadLatency + 1;
                  const int64_t combineInputOffset = combineConfigOffset + 1;

                  const auto appendProductPhase =
                      [&](int64_t inputOffset, int64_t firstProduct,
                          bool mirroredRopeTable) {
                        for (int64_t token = 0; token < op_.getSeqLen();
                             ++token) {
                          const int64_t tokenLane = token % blockRows;
                          const int64_t inputCycle = inputOffset + token;
                          for (int64_t half = 0; half < 2; ++half) {
                            const int64_t sourceBlock = blocks[half];
                            for (int64_t hemisphere = 0;
                                 hemisphere < memory.hemispheres;
                                 ++hemisphere) {
                              for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t slice =
                                    layout.ropeStagingSlices()
                                        [(2 * tokenLane + byte +
                                             2 * sourceBlock) %
                                            16];
                                appendMemResources(
                                    windows,
                                    inputCycle - vxmInputReadLatency(slice),
                                    hemisphere, slice, stagingBank, false);
                              }
                            }
                          }
                          for (int64_t hemisphere = 0;
                               hemisphere < memory.hemispheres; ++hemisphere) {
                            for (int64_t byte = 0; byte < 4; ++byte) {
                              const int64_t slice = layout.ropeSlices()[byte];
                              appendMemResources(
                                  windows,
                                  inputCycle - vxmInputReadLatency(slice),
                                  hemisphere, slice,
                                  mirroredRopeTable ? ropeMirrorBank : ropeBank,
                                  false);
                            }
                          }
                          for (int64_t slot = 0; slot < 2; ++slot) {
                            const int64_t product = firstProduct + slot;
                            for (int64_t byte = 0; byte < 2; ++byte) {
                              const int64_t slice = candidateProductSlice(
                                  product, token, byte);
                              const int64_t latency =
                                  target_
                                      .transport_latency(
                                          target::StreamEndpoint::VxmResult,
                                          target::StreamEndpoint::Mem,
                                          target::StreamDirection::East,
                                          slice)
                                      .value_or(readLatency(slice));
                              for (int64_t destination = 0;
                                   destination < memory.hemispheres;
                                   ++destination)
                                appendMemResources(
                                    windows, inputCycle + 2 + latency,
                                    destination, slice,
                                    candidateProductBank(product), true);
                            }
                          }
                        }
                      };
                  appendProductPhase(productAInputOffset, 0, false);
                  appendProductPhase(productBInputOffset, 2, true);

                  for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
                    const int64_t tokenLane = token % blockRows;
                    const int64_t inputCycle = combineInputOffset + token;
                    for (int64_t product = 0; product < 4; ++product) {
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = candidateProductSlice(
                            product, token, byte);
                        for (int64_t hemisphere = 0;
                             hemisphere < memory.hemispheres; ++hemisphere)
                          appendMemResources(
                              windows,
                              inputCycle - vxmInputReadLatency(slice),
                              hemisphere, slice,
                              candidateProductBank(product), false);
                      }
                    }
                    const int64_t outputCycle = inputCycle + 1;
                    for (int64_t half = 0; half < 2; ++half) {
                      const int64_t reductionBlock = blocks[half];
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice =
                            kind == AttentionProjectionKind::Query
                                ? layout.queryIwSlices(reductionBlock)
                                      [2 * tokenLane + byte]
                                : layout.keySlices(reductionBlock)[byte];
                        const int64_t latency =
                            target_
                                .transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East, slice)
                                .value_or(readLatency(slice));
                        const int64_t bank =
                            kind == AttentionProjectionKind::Query
                                ? layout.queryIwBank(reductionBlock)
                                : layout.keyBank(reductionBlock);
                        for (int64_t destination = 0;
                             destination < memory.hemispheres; ++destination)
                          appendMemResources(windows, outputCycle + latency,
                                             destination, slice, bank, true);
                      }
                    }
                  }
                  ResourceScheduler resources;
                  return resources.try_reserve_at(0, windows);
                };

            llvm::SmallVector<int64_t, 16> productSlices(
                baseProductSlices.begin(), baseProductSlices.end());
            if (!ropeResourcesAreLegal(productSlices)) {
              if (productSlices.size() < 16) {
                op_.emitError(
                    "compact RoPE product placement conflicts with a MEM "
                    "ICU/read-write resource");
                return -1;
              }
              llvm::SmallVector<int64_t, 16> interleaved;
              interleaved.reserve(productSlices.size());
              // Product storage uses two slices per BF16 pair. Group pairs
              // by token-lane parity so an II=1 product read cannot meet the
              // same Query-IW slice write from the opposite token parity.
              for (int64_t parity = 0; parity < 2; ++parity)
                for (std::size_t index = 0; index < productSlices.size();
                     ++index)
                  if (static_cast<int64_t>(index / 2) % 2 == parity)
                    interleaved.push_back(productSlices[index]);
              if (interleaved.empty() ||
                  !ropeResourcesAreLegal(interleaved)) {
                op_.emitError(
                    "cannot find a conflict-free MEM ICU schedule for RoPE "
                    "Product A/B/Combine");
                return -1;
              }
              productSlices = std::move(interleaved);
            }
            const auto emitSource = [&](int64_t token, int64_t half,
                                        int64_t stream, int64_t inputCycle) {
              const int64_t tokenBlock = token / tile;
              const int64_t rowBlock = (token % tile) / blockRows;
              const int64_t tokenLane = token % blockRows;
              const int64_t sourceBlock = blocks[half];
              const int64_t address = layout.ropeStagingAddress(
                  kind, head, sourceBlock, tokenBlock, rowBlock);
              for (int64_t hemisphere = 0; hemisphere < memory.hemispheres;
                   ++hemisphere) {
                for (int64_t byte = 0; byte < 2; ++byte) {
                  const int64_t slice =
                      layout.ropeStagingSlices()[(2 * tokenLane + byte +
                                                  2 * sourceBlock) %
                                                 16];
                  emitMem(rewriter_, op_.getLoc(),
                          inputCycle - vxmInputReadLatency(slice),
                          hemisphere * memory.slices_per_hemisphere + slice,
                          "read", address, stream + hemisphere * 16 + byte, 1,
                          1, 0, "sram", -1, stagingBank);
                }
              }
            };
            const auto emitRopeTable = [&](int64_t token, int64_t inputCycle,
                                           bool mirror) {
              for (int64_t hemisphere = 0; hemisphere < memory.hemispheres;
                   ++hemisphere) {
                for (int64_t byte = 0; byte < 4; ++byte) {
                  const int64_t slice = layout.ropeSlices()[byte];
                  const int64_t stream =
                      (byte < 2 ? 34 + byte : 36 + byte) + hemisphere * 16;
                  emitMem(rewriter_, op_.getLoc(),
                          inputCycle - vxmInputReadLatency(slice),
                          hemisphere * memory.slices_per_hemisphere + slice,
                          "read",
                          mirror
                              ? layout.ropeMirrorAddress(token, pairBlock)
                              : layout.ropeAddress(token, pairBlock),
                          stream, 1, 1, 0, "sram", -1,
                          mirror ? ropeMirrorBank : ropeBank);
                }
              }
            };
            const auto emitProductWrites = [&](int64_t token,
                                               int64_t firstProduct,
                                               int64_t outputCycle) {
              for (int64_t slot = 0; slot < 2; ++slot) {
                const int64_t product = firstProduct + slot;
                for (int64_t byte = 0; byte < 2; ++byte) {
                  const int64_t slice =
                      layout.ropeProductSlice(kind, product, token, byte);
                  const int64_t latency =
                      target_
                          .transport_latency(target::StreamEndpoint::VxmResult,
                                             target::StreamEndpoint::Mem,
                                             target::StreamDirection::East,
                                             slice)
                          .value_or(readLatency(slice));
                  for (int64_t destination = 0;
                       destination < memory.hemispheres; ++destination) {
                    const int64_t source = 1 - destination;
                    emitMem(rewriter_, op_.getLoc(), outputCycle + latency,
                            destination * memory.slices_per_hemisphere + slice,
                            "write",
                            layout.ropeProductAddress(kind, head, pairBlock,
                                                      product, token),
                            source * 8 + slot * 2 + byte, 1, 1, 0, "sram", -1,
                            layout.ropeProductBank(kind, productBank,
                                                   product));
                  }
                }
              }
            };

            const int64_t productAConfig = headEnd;
            const int64_t productAInput = productAConfig + 1;
            emitRopeProducts(productAConfig, inputHemisphere, outputHemisphere);
            for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
              const int64_t inputCycle = productAInput + token;
              emitSource(token, 0, 32, inputCycle);
              emitSource(token, 1, 36, inputCycle);
              emitRopeTable(token, inputCycle, false);
              emitProductWrites(token, 0, inputCycle + 2);
            }

            const int64_t productBConfig = productAInput + op_.getSeqLen() + 1 +
                                           maxProductWriteLatency +
                                           maxStagingReadLatency + 1;
            const int64_t productBInput = productBConfig + 1;
            emitRopeProducts(productBConfig, inputHemisphere, outputHemisphere);
            for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
              const int64_t inputCycle = productBInput + token;
              emitSource(token, 1, 32, inputCycle);
              emitSource(token, 0, 36, inputCycle);
              emitRopeTable(token, inputCycle, true);
              emitProductWrites(token, 2, inputCycle + 2);
            }

            const int64_t combineConfig = productBInput + op_.getSeqLen() + 1 +
                                           maxProductWriteLatency +
                                           maxProductReadLatency + 1;
            if (firstCombineConfig < 0)
              firstCombineConfig = combineConfig;
            const int64_t combineInput = combineConfig + 1;
            emitRopeCombine(combineConfig, inputHemisphere, outputHemisphere);
            for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
              const int64_t tokenBlock = token / tile;
              const int64_t rowBlock = (token % tile) / blockRows;
              const int64_t tokenLane = token % blockRows;
              const int64_t inputCycle = combineInput + token;
              for (int64_t product = 0; product < 4; ++product) {
                for (int64_t byte = 0; byte < 2; ++byte) {
                  const int64_t slice =
                      layout.ropeProductSlice(kind, product, token, byte);
                  for (int64_t hemisphere = 0; hemisphere < memory.hemispheres;
                       ++hemisphere)
                    emitMem(rewriter_, op_.getLoc(),
                            inputCycle - vxmInputReadLatency(slice),
                            hemisphere * memory.slices_per_hemisphere + slice,
                            "read",
                            layout.ropeProductAddress(kind, head, pairBlock,
                                                      product, token),
                            32 + hemisphere * 16 + product * 2 + byte, 1, 1, 0,
                            "sram", -1,
                            layout.ropeProductBank(kind, productBank,
                                                   product));
                }
              }
              const int64_t outputCycle = inputCycle + 1;
              for (int64_t half = 0; half < 2; ++half) {
                const int64_t reductionBlock = blocks[half];
                for (int64_t byte = 0; byte < 2; ++byte) {
                  const int64_t slice =
                      kind == AttentionProjectionKind::Query
                          ? layout.queryIwSlices(
                                reductionBlock)[2 * tokenLane + byte]
                          : layout.keySlices(reductionBlock)[byte];
                  const int64_t latency =
                      target_
                          .transport_latency(target::StreamEndpoint::VxmResult,
                                             target::StreamEndpoint::Mem,
                                             target::StreamDirection::East,
                                             slice)
                          .value_or(readLatency(slice));
                  for (int64_t destination = 0;
                       destination < memory.hemispheres; ++destination) {
                    const int64_t source = 1 - destination;
                    const int64_t baseBank =
                        kind == AttentionProjectionKind::Query
                            ? layout.queryIwBank(reductionBlock)
                            : layout.keyBank(reductionBlock);
                    emitMem(rewriter_, op_.getLoc(), outputCycle + latency,
                            destination * memory.slices_per_hemisphere + slice,
                            "write",
                            kind == AttentionProjectionKind::Query
                                ? layout.queryIwAddress(head, reductionBlock,
                                                        tokenBlock, rowBlock)
                                : layout.keyAddress(head, reductionBlock,
                                                    tokenBlock) +
                                      token % tile,
                            source * 8 + half * 2 + byte, 1, 1, 0, "sram", -1,
                            baseBank);
                    headEnd = std::max(headEnd, outputCycle + latency + 1);
                  }
                }
              }
            }
            headEnd = std::max(headEnd, combineInput + op_.getSeqLen() + 1) +
                      maxStagingReadLatency + 1;
          }
          ropeEnd = std::max(ropeEnd, headEnd);
        }
        postprocessReady =
            ropeEnd + target_.streams().system_register_columns;
        if (projectionRopeOverlap) {
          phaseStart = std::max(phaseStart, nextProjectionStart);
          projectionOverlapDeadline = firstCombineConfig;
        } else {
          phaseStart = std::max(phaseStart, postprocessReady);
          projectionOverlapDeadline = -1;
        }
      }
      residentWeights.push_back(
          {weightPlacement, currentProjectionWeightRelease});
    }
    return std::max(phaseStart, postprocessReady) + 16;
  }

  const int64_t weightLoadLead =
      (target_.memory().hemispheres - 1) * 8 + 7 + weightToIw + 1;
  int64_t projectionBlock = 0;
  for (int64_t projection = 0; projection < 3; ++projection) {
    const auto kind = projectionKind(projection);
    for (int64_t headBase = 0; headBase < projectionHeads[projection];
         headBase += 2) {
      for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks;
           ++reductionBlock) {
        const int64_t firstCompute =
            reductionBlock == 0
                ? phaseStart + readLatency(layout.weightSlices().back()) +
                      weightLoadLead
                : phaseStart;
        const int64_t dequantStart = firstCompute - weightLoadLead;
        const int64_t weightBuffer =
            projectionBlock % target_.throughput().mxm_weight_buffers;
        for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
          const int64_t head = headBase + hemisphere;
          if (head >= projectionHeads[projection])
            continue;
          for (int64_t pulse = 0; pulse < 8; ++pulse) {
            const int64_t localMxm = pulse / 4;
            const int64_t column = 3 - pulse % 4;
            const int64_t cycle = dequantStart + hemisphere * 8 + pulse;
            const int64_t address = layout.weightAddress(
                kind, head * projectionHeadBlocks + localMxm, reductionBlock,
                localMxm, pulse % 4);
            for (int64_t stream = 0; stream < 8; ++stream) {
              const int64_t slice = layout.weightSlices()[stream];
              emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                      hemisphere * target_.memory().slices_per_hemisphere +
                          slice,
                      "read", address, 32 + stream, 1, 1, 0, "sram",
                      functionArgumentIndex(projectionValues[projection]),
                      layout.weightBank(kind));
            }
            emitDequant(cycle, hemisphere, localMxm,
                        projectionValues[projection],
                        projectionScales[projection]);
            emitMxm(rewriter_, op_.getLoc(), cycle + weightToIw,
                    hemisphere * 2 + localMxm, "iw", weightBuffer, column, 0, 0,
                    1, 1, 0, 1, "stream", true, "supercell", 0, dataFormat);
          }
        }

        const bool finalReduction = reductionBlock + 1 == hiddenBlocks;
        int64_t finalWriteEnd = firstCompute;
        // A final-reduction tile reaches MEM after ACC -> VXM and
        // projection-specific transport. The next token tile may
        // reuse those MEM slice ports for its activation read.
        int64_t writebackDelay = 0;
        if (kind == AttentionProjectionKind::Query) {
          for (int64_t reduction = 0; reduction < 2; ++reduction) {
            for (int64_t slice : target_.attention_query_iw_slices(reduction))
              writebackDelay = std::max(writebackDelay, 2 + slice / 4);
          }
        } else if (kind == AttentionProjectionKind::Key) {
          writebackDelay = 2;
        } else {
          for (int64_t reduction = 0; reduction < 2; ++reduction) {
            for (int64_t slice : layout.valuePackSlices(reduction))
              writebackDelay = std::max(
                  writebackDelay,
                  1 + slice / target_.streams().mem_slices_per_register_group);
          }
        }
        const int64_t computeBlockCycles =
            finalReduction
                ? std::max(target_.mxm_block_issue_interval() + tile,
                           target_.throughput().accumulator_to_vxm_latency +
                               tile + writebackDelay + activationLatency)
                : target_.mxm_block_issue_interval();
        for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
          for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            const int64_t head = headBase + hemisphere;
            if (head >= projectionHeads[projection])
              continue;
            const int64_t computeCycle =
                firstCompute + tokenBlock * computeBlockCycles;
            const int64_t inputAddress =
                layout.activationAddress(reductionBlock, tokenBlock) +
                (inputDistributed16 ? inputStagingPlacement
                                          .getAs<mlir::IntegerAttr>("base_row")
                                          .getInt()
                                    : 0);
            const int64_t outputAddress =
                layout.projectionAddress(kind, head, tokenBlock);
            llvm::SmallVector<int64_t> segmentRows;
            llvm::SmallVector<int64_t> segmentStreams;
            const bool prefetchNextWeight =
                !finalReduction && tokenBlock + 1 == tokenBlocks;
            if (prefetchNextWeight) {
              const int64_t nextFirstCompute =
                  firstCompute + tokenBlocks * computeBlockCycles;
              const int64_t nextDequantStart =
                  nextFirstCompute - weightLoadLead;
              const int64_t switchRow =
                  nextDequantStart + hemisphere * 8 + weightToIw - computeCycle;
              if (switchRow > 0) {
                segmentRows.push_back(switchRow);
                segmentStreams.push_back(0);
              }
              const int64_t switchedRows = std::min<int64_t>(
                  target_.throughput().tile_rows, tile - switchRow);
              segmentRows.push_back(switchedRows);
              segmentStreams.push_back(
                  target_.throughput().mxm_load_streams_per_cycle);
              if (switchRow + switchedRows < tile) {
                segmentRows.push_back(tile - switchRow - switchedRows);
                segmentStreams.push_back(0);
              }
            } else {
              segmentRows.push_back(tile);
              segmentStreams.push_back(0);
            }
            const char *destination = finalReduction ? "stream" : "sram";
            int64_t rowOffset = 0;
            for (std::size_t segment = 0; segment < segmentRows.size();
                 ++segment) {
              const int64_t rows = segmentRows[segment];
              const int64_t streamBase = segmentStreams[segment];
              const int64_t segmentCycle = computeCycle + rowOffset;
              if (inputDistributed16) {
                for (int64_t byte = 0; byte < 2; ++byte) {
                  emitMem(rewriter_, op_.getLoc(),
                          segmentCycle - activationLatency,
                          hemisphere * target_.memory().slices_per_hemisphere +
                              projectionActivationSlices[byte],
                          "read", inputAddress + rowOffset, streamBase + byte,
                          rows, 1, 1, "sram", -1,
                          projectionActivationBank);
                }
              } else {
                for (int64_t byte = 0; byte < 4; ++byte) {
                  emitMem(rewriter_, op_.getLoc(),
                          segmentCycle - activationLatency,
                          hemisphere * target_.memory().slices_per_hemisphere +
                              layout.activationSlices()[byte],
                          "read", inputAddress + rowOffset, streamBase + byte,
                          rows, 1, 1, "sram", -1,
                          projectionActivationBank);
                }
              }
              emitMxm(rewriter_, op_.getLoc(), segmentCycle, hemisphere * 2,
                      "compute", weightBuffer, 0, streamBase, 0, rows, 1,
                      outputAddress, 1, destination, true, "supercell", 0,
                      dataFormat);
              emitMxm(rewriter_, op_.getLoc(), segmentCycle, hemisphere * 2 + 1,
                      "compute", weightBuffer, 0,
                      streamBase + (inputDistributed16 ? 0 : 2), 4, rows, 1,
                      outputAddress, 1, destination, true, "supercell", 0,
                      dataFormat);
              rowOffset += rows;
            }
            if (!finalReduction)
              continue;

            for (int64_t offset = 0; offset < tile; ++offset) {
              const int64_t token = tokenBlock * tile + offset;
              const int64_t vxmCycle =
                  computeCycle +
                  target_.throughput().accumulator_to_vxm_latency + offset;
              if (kind != AttentionProjectionKind::Value) {
                for (int64_t byte = 0; byte < 4; ++byte) {
                  const int64_t slice = layout.ropeSlices()[byte];
                  emitMem(
                      rewriter_, op_.getLoc(), vxmCycle - readLatency(slice),
                      hemisphere * target_.memory().slices_per_hemisphere +
                          slice,
                      "read", layout.ropeAddress(token), 40 + byte, 1, 1, 0,
                      "sram", -1, ropeBank);
                }
              }
              emitRopeOrCast(vxmCycle, hemisphere,
                             kind != AttentionProjectionKind::Value,
                             projectionValues[projection]);
              const int64_t writeCycle =
                  vxmCycle + (kind == AttentionProjectionKind::Value ? 1 : 2);
              if (kind == AttentionProjectionKind::Query) {
                const int64_t phase = (token % tile) / 8;
                const int64_t localColumn = token % 8;
                for (int64_t reduction = 0; reduction < 2; ++reduction) {
                  const auto &slices =
                      target_.attention_query_iw_slices(reduction);
                  for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t stream = reduction * 2 + byte;
                    const int64_t slice = slices[localColumn * 2 + byte];
                    emitMem(rewriter_, op_.getLoc(), writeCycle + slice / 4,
                            hemisphere *
                                    target_.memory().slices_per_hemisphere +
                                slice,
                            "write",
                            layout.queryIwAddress(head, reduction, tokenBlock,
                                                  phase),
                            stream, 1, 1, 0, "sram", -1,
                            layout.queryIwBank(reduction));
                    finalWriteEnd =
                        std::max(finalWriteEnd, writeCycle + slice / 4 + 1);
                  }
                }
              } else if (kind == AttentionProjectionKind::Key) {
                for (int64_t byte = 0; byte < 4; ++byte) {
                  emitMem(rewriter_, op_.getLoc(), writeCycle,
                          hemisphere * target_.memory().slices_per_hemisphere +
                              byte,
                          "write", outputAddress + offset, byte, 1, 1, 0,
                          "sram", -1, keyBank);
                  finalWriteEnd = std::max(finalWriteEnd, writeCycle + 1);
                }
              } else {
                const int64_t packedStream = (token % 8) * 2;
                const int64_t row = (token % tile) / 8;
                for (int64_t reduction = 0; reduction < 2; ++reduction) {
                  const auto slices = layout.valuePackSlices(reduction);
                  for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = slices[packedStream + byte];
                    emitMem(
                        rewriter_, op_.getLoc(),
                        writeCycle +
                            slice /
                                target_.streams().mem_slices_per_register_group,
                        hemisphere * target_.memory().slices_per_hemisphere +
                            slice,
                        "write",
                        layout.valuePackAddress(head, reduction, tokenBlock,
                                                row),
                        reduction * 2 + byte, 1, 1, 0, "sram", -1,
                        valueBank);
                    finalWriteEnd = std::max(
                        finalWriteEnd,
                        writeCycle +
                            slice / target_.streams()
                                        .mem_slices_per_register_group +
                            1);
                  }
                }
              }
            }
          }
        }
        phaseStart = firstCompute + tokenBlocks * computeBlockCycles;
        if (finalReduction)
          phaseStart = std::max(phaseStart, finalWriteEnd);
        ++projectionBlock;
      }
    }
  }

  const auto hemisphereName = [](int64_t hemisphere) {
    return hemisphere == 0 ? "east" : "west";
  };
  const int64_t groups = target_.streams().mem_slices_per_register_group;
  const int64_t headBlocks = op_.getHeadDim() / tile;

  // Projection alternates logical heads across hemispheres. GQA QK work is
  // placed beside the shared KV head, so copy only Q heads whose projection
  // home differs from that KV home.
  int64_t copyCycle = phaseStart + 16;
  const int64_t queryHeadsPerKv = op_.getQueryHeads() / op_.getKvHeads();
  for (int64_t queryHead = 0; queryHead < op_.getQueryHeads(); ++queryHead) {
    const int64_t kvHead = queryHead / queryHeadsPerKv;
    const int64_t source = queryHead % target_.memory().hemispheres;
    const int64_t destination = kvHead % target_.memory().hemispheres;
    if (source == destination)
      continue;
    for (int64_t queryBlock = 0; queryBlock < tokenBlocks; ++queryBlock) {
      for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
        const auto &slices = layout.queryIwSlices(reduction);
        for (int64_t phase = 0; phase < target_.throughput().tile_rows;
             ++phase, ++copyCycle) {
          for (int64_t stream = 0; stream < static_cast<int64_t>(slices.size());
               ++stream) {
            const int64_t slice = slices[stream];
            emitMem(
                rewriter_, op_.getLoc(), copyCycle - readLatency(slice),
                source * target_.memory().slices_per_hemisphere + slice, "read",
                layout.queryIwAddress(queryHead, reduction, queryBlock, phase),
                32 + stream, 1, 1, 0, "sram", -1,
                layout.queryIwBank(reduction));
            emitVxm(rewriter_, op_.getLoc(), op_.getInput(), copyCycle, stream,
                    "pass", "stream_i8", 32 + stream, 0.0f, "immediate", 0,
                    0.0f, "i8", stream, hemisphereName(source),
                    hemisphereName(destination));
            emitMem(
                rewriter_, op_.getLoc(), copyCycle + 1 + slice / groups,
                destination * target_.memory().slices_per_hemisphere + slice,
                "write",
                layout.queryIwAddress(queryHead, reduction, queryBlock, phase),
                stream, 1, 1, 0, "sram", -1,
                layout.queryIwBank(reduction));
          }
        }
      }
    }
    copyCycle += 20;
  }

  // Each hemisphere has two MXMs. Duplicate K's two 16-bit pairs onto slices
  // 4..7 so MXM0 and MXM1 can consume the same KV head concurrently.
  int64_t keyCopyCycle = copyCycle + 16;
  for (int64_t keyHead = 0; keyHead < op_.getKvHeads(); ++keyHead) {
    const int64_t hemisphere = keyHead % target_.memory().hemispheres;
    const char *hemi = hemisphereName(hemisphere);
    for (int64_t token = 0; token < op_.getSeqLen(); ++token, ++keyCopyCycle) {
      for (int64_t byte = 0; byte < 4; ++byte) {
        emitMem(rewriter_, op_.getLoc(), keyCopyCycle - readLatency(byte),
                hemisphere * target_.memory().slices_per_hemisphere + byte,
                "read",
                layout.keyAddress(keyHead, 0, token / tile) + token % tile,
                32 + byte, 1, 1, 0, "sram", -1, keyBank);
      }
      emitVxm(rewriter_, op_.getLoc(), op_.getKeyWeight(), keyCopyCycle, 0,
              "pass", streamKind, 32, 0.0f, "immediate", 0, 0.0f, dataFormat, 0,
              hemi, hemi);
      emitVxm(rewriter_, op_.getLoc(), op_.getKeyWeight(), keyCopyCycle, 1,
              "pass", streamKind, 34, 0.0f, "immediate", 0, 0.0f, dataFormat, 2,
              hemi, hemi);
      for (int64_t byte = 0; byte < 4; ++byte) {
        const int64_t reduction = byte < 2 ? 0 : headBlocks / 2;
        const int64_t slice = layout.keySlices(reduction)[byte % 2];
        emitMem(rewriter_, op_.getLoc(), keyCopyCycle + 1 + slice / groups,
                hemisphere * target_.memory().slices_per_hemisphere + slice,
                "write",
                layout.keyAddress(keyHead, 0, token / tile) + token % tile,
                byte, 1, 1, 0, "sram", -1, layout.keyBank(reduction));
      }
    }
    keyCopyCycle += 12;
  }
  return keyCopyCycle + 16;
}

mlir::LogicalResult AttentionScheduleEmitter::emitQk(
    int64_t qkStart, int64_t qkWaveCycles,
    int64_t qkIwToComputeCycles, bool fusedSoftmax) {
  const AttentionMemoryLayout layout(op_, target_);
  const auto placementBank = [&](llvm::StringRef name) {
    const auto placement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(name);
    const auto bank = placement ? placement.getAs<mlir::IntegerAttr>("bank")
                                : mlir::IntegerAttr{};
    return bank ? bank.getInt() : 0;
  };
  const auto elementType =
      llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
          .getElementType();
  const llvm::StringRef dataFormat = lpu_16bit_data_format(elementType);
  const int64_t tile = target_.throughput().mxm_rows;
  const int64_t tokenBlocks = op_.getSeqLen() / tile;
  const int64_t headBlocks = op_.getHeadDim() / tile;
  const int64_t issue = target_.mxm_block_issue_interval();
  const bool wavefront = target_.supports_mxm_weight_activation_overlap() &&
                         target_.throughput().mxm_weight_buffers >= 2 &&
                         target_.throughput().mxms_per_hemisphere == 1;
  const auto reductionComputeCycle = [&](int64_t firstComputeCycle,
                                         int64_t reduction) {
    return firstComputeCycle + reduction * tokenBlocks * issue;
  };
  const auto reductionIwCycle = [&](int64_t firstIwCycle,
                                    int64_t firstComputeCycle,
                                    int64_t localMxm, int64_t reduction) {
    const int64_t compute = reductionComputeCycle(firstComputeCycle, reduction);
    return wavefront
               ? compute - qkIwToComputeCycles -
                     localMxm * target_.throughput().tile_rows
               : firstIwCycle + localMxm * 8 + reduction * tile / 8;
  };

  // Plan the complete QK input bundle before emitting any IR. Query-IW and
  // Key activation traffic use different MXM paths, but still share one MEM
  // ICU for each (hemisphere, slice, bank).
  ResourceScheduler qkResources;
  const LPUResourceModel resourceModel(target_);
  for (std::size_t waveIndex = 0; waveIndex < stage_plan_.qk_waves.size();
       ++waveIndex) {
    llvm::SmallVector<ResourceWindow, 256> windows;
    const int64_t waveStart =
        qkStart + static_cast<int64_t>(waveIndex) * qkWaveCycles;
    const int64_t firstIwCycle =
        waveStart + target_.throughput().mxm_earliest_iw_cycle +
        *target_.transport_latency(target::StreamEndpoint::Mem,
                                   target::StreamEndpoint::MxmWeight,
                                   target::StreamDirection::East, 0);
    const int64_t firstComputeCycle = firstIwCycle + qkIwToComputeCycles;
    const auto appendRead = [&](int64_t cycle, int64_t duration,
                                int64_t hemisphere, int64_t slice,
                                int64_t bank) {
      windows.push_back(
          {resourceModel.mem_icu(hemisphere, slice, bank), cycle, duration});
      windows.push_back({resourceModel.mem_read_port(
                             hemisphere, slice, bank),
                         cycle, duration});
    };
    for (const auto &work : stage_plan_.qk_waves[waveIndex].slots) {
      if (!work)
        continue;
      for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
        const int64_t iw = reductionIwCycle(
            firstIwCycle, firstComputeCycle, work->local_mxm, reduction);
        for (int64_t phase = 0; phase < tile / 8; ++phase) {
          for (int64_t slice : layout.queryIwSlices(reduction)) {
            const int64_t latency = *target_.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, slice);
            appendRead(iw + phase - latency, 1, work->hemisphere, slice,
                       layout.queryIwBank(reduction));
          }
        }
        for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
          const int64_t compute =
              reductionComputeCycle(firstComputeCycle, reduction) +
              keyBlock * issue;
          for (int64_t slice : layout.keySlices(reduction)) {
            const int64_t latency = *target_.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation,
                target::StreamDirection::East, slice);
            appendRead(compute - latency, tile, work->hemisphere, slice,
                       layout.keyBank(reduction));
          }
        }
      }
    }
    if (!qkResources.try_reserve_at(0, windows)) {
      op_.emitError(
          "QK input placement has an overlapping MEM ICU/read-port bundle");
      return mlir::failure();
    }
  }

  for (std::size_t waveIndex = 0; waveIndex < stage_plan_.qk_waves.size();
       ++waveIndex) {
    const int64_t waveStart =
        qkStart + static_cast<int64_t>(waveIndex) * qkWaveCycles;
    const int64_t firstIwCycle =
        waveStart + target_.throughput().mxm_earliest_iw_cycle +
        *target_.transport_latency(target::StreamEndpoint::Mem,
                                   target::StreamEndpoint::MxmWeight,
                                   target::StreamDirection::East, 0);
    const int64_t firstComputeCycle = firstIwCycle + qkIwToComputeCycles;
    for (const auto &work : stage_plan_.qk_waves[waveIndex].slots) {
      if (!work)
        continue;
      const int64_t mxm =
          work->hemisphere * target_.throughput().mxms_per_hemisphere +
          work->local_mxm;
      // Direct16 Q weights occupy E0..E15. Keep K activations on the
      // upper east streams so the next Q block can be loaded while the
      // current block computes.
      const int64_t activationStream = 16 + work->local_mxm * 2;
      const int64_t outputStream = work->local_mxm * 4;
      for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
        const auto iwSlices = layout.queryIwSlices(reduction);
        const int64_t reductionIw = reductionIwCycle(
            firstIwCycle, firstComputeCycle, work->local_mxm, reduction);
        for (int64_t phase = 0; phase < tile / 8; ++phase) {
          const int64_t iwCycle = reductionIw + phase;
          const int64_t sourcePhase = tile / 8 - 1 - phase;
          const int64_t queryAddress = layout.queryIwAddress(
              work->query_head, reduction, work->query_block, sourcePhase);
          for (int64_t stream = 0;
               stream < static_cast<int64_t>(iwSlices.size()); ++stream) {
            const int64_t slice = iwSlices[stream];
            const int64_t latency = *target_.transport_latency(
                target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                target::StreamDirection::East, slice);
            emitMem(rewriter_, op_.getLoc(), iwCycle - latency,
                    work->hemisphere * target_.memory().slices_per_hemisphere +
                        slice,
                    "read", queryAddress,
                    work->local_mxm * static_cast<int64_t>(iwSlices.size()) +
                        stream,
                    1, 1, 0, "sram", -1,
                    layout.queryIwBank(reduction));
          }
          emitMxm(rewriter_, op_.getLoc(), iwCycle, mxm, "iw",
                  reduction % target_.throughput().mxm_weight_buffers,
                  tile / 8 - 1 - phase, 0, 0, 1, 1, 0, 1, "stream", true,
                  "supercell", 0, dataFormat);
        }
      }
      for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
        const bool finalReduction = reduction + 1 == headBlocks;
        for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
          const int64_t computeCycle =
              reductionComputeCycle(firstComputeCycle, reduction) +
              keyBlock * issue;
          for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice =
                target_.throughput().mxms_per_hemisphere == 1
                    ? layout.keySlices(reduction)[byte]
                    : work->local_mxm * 4 + (reduction % 2) * 2 + byte;
            const int64_t latency = *target_.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation,
                target::StreamDirection::East, slice);
            emitMem(rewriter_, op_.getLoc(), computeCycle - latency,
                    work->hemisphere * target_.memory().slices_per_hemisphere +
                        slice,
                    "read",
                    layout.keyAddress(work->kv_head, reduction, keyBlock),
                    activationStream + byte, tile, 1, 1, "sram", -1,
                    layout.keyBank(reduction));
          }
          emitMxm(rewriter_, op_.getLoc(), computeCycle, mxm, "compute",
                  reduction % target_.throughput().mxm_weight_buffers, 0,
                  activationStream, outputStream, tile, 1,
                  layout.scoreAccumulatorAddress(work->query_head,
                                                 work->query_block, keyBlock),
                  1, finalReduction ? "stream" : "sram", finalReduction,
                  "supercell", 0, dataFormat, {},
                  finalReduction ? dataFormat : llvm::StringRef{});
          if (!fusedSoftmax && finalReduction) {
            // Tail softmax consumes compact 16-bit scores. The
            // final partial converts in the accumulator and clears
            // the row while streaming, avoiding a separate FP32
            // accumulator-read/cast pass.
            for (int64_t byte = 0; byte < 2; ++byte) {
              const int64_t slice =
                  layout.scaledScoreSlices(work->local_mxm)[byte];
              const auto latency = target_.transport_latency(
                  target::StreamEndpoint::MxmResult,
                  target::StreamEndpoint::Mem, target::StreamDirection::West,
                  slice);
              if (!latency)
                continue;
              emitMem(
                  rewriter_, op_.getLoc(),
                  computeCycle + target_.mxm_first_result_latency() + *latency,
                  work->hemisphere * target_.memory().slices_per_hemisphere +
                      slice,
                  "write",
                  layout.scoreAddress(work->query_head, work->query_block,
                                      keyBlock * tile),
                  32 + outputStream + byte, tile, 1, 1, "sram", -1,
                  placementBank(work->local_mxm == 0 ? "score" : "score_mxm1"));
            }
          }
        }
      }
    }
  }
  return mlir::success();
}

} // namespace ftlpu::compiler::schedule
