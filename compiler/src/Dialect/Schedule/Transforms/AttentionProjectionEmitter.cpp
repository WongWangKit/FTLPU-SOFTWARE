#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/paged_weight_residency.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
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

llvm::SmallVector<int64_t> placementSlices(
    mlir::DictionaryAttr placement) {
  llvm::SmallVector<int64_t> result;
  for (mlir::Attribute value : placement.getAs<mlir::ArrayAttr>("slices"))
    result.push_back(llvm::cast<mlir::IntegerAttr>(value).getInt());
  return result;
}

int64_t placementBase(mlir::DictionaryAttr placement) {
  return placement.getAs<mlir::IntegerAttr>("base_row").getInt();
}

int64_t placementBankValue(mlir::DictionaryAttr placement) {
  return placement.getAs<mlir::IntegerAttr>("bank").getInt();
}

// Convert one projected head between the RoPE FIFO's MXM-oriented layout and
// the packed VXM layout used by the feedback RMSNorm implementation.
int64_t emitRopeNormTranspose(
    mlir::IRRewriter &rewriter, mlir::Location location,
    const target::LPUTargetModel &target,
    const AttentionMemoryLayout &layout, AttentionProjectionKind kind,
    int64_t head, int64_t seqLen, int64_t headDim,
    int64_t rawBank, mlir::DictionaryAttr packedPlacement,
    bool rawToPacked, int64_t start) {
  const auto rawSlices = layout.ropeStagingSlices();
  const auto packedSlices = placementSlices(packedPlacement);
  const int64_t width = 2 * target.throughput().lanes_per_tile;
  const int64_t tile = target.throughput().mxm_rows;
  const int64_t tileRows = target.throughput().tile_rows;
  const int64_t headBlocks = headDim / tile;
  const int64_t inputBeats = (seqLen / tile) * headBlocks * tileRows;
  if (rawSlices.size() != static_cast<std::size_t>(width) ||
      packedSlices.size() != static_cast<std::size_t>(width))
    return start;

  llvm::SmallVector<int64_t> sourceStreams(
      static_cast<std::size_t>(width));
  llvm::SmallVector<int64_t> transposeStreams(
      static_cast<std::size_t>(width));
  llvm::SmallVector<int64_t> outputStreams(
      static_cast<std::size_t>(width));
  const int64_t sourceBase = target.streams().streams_per_direction - width;
  const int64_t outputBase = target.streams().streams_per_direction;
  for (int64_t stream = 0; stream < width; ++stream) {
    sourceStreams[static_cast<std::size_t>(stream)] = sourceBase + stream;
    transposeStreams[static_cast<std::size_t>(stream)] = stream;
    outputStreams[static_cast<std::size_t>(stream)] = outputBase + stream;
  }
  const auto readLatency = [&](int64_t slice) {
    return target
        .transport_latency(target::StreamEndpoint::Mem,
                           target::StreamEndpoint::SxmInput,
                           target::StreamDirection::East, slice)
        .value();
  };
  const auto writeLatency = [&](int64_t slice) {
    return target
        .transport_latency(target::StreamEndpoint::SxmResult,
                           target::StreamEndpoint::Mem,
                           target::StreamDirection::West, slice)
        .value();
  };
  int64_t maxReadLatency = 0;
  int64_t maxWriteLatency = 0;
  for (int64_t slice : rawToPacked ? rawSlices
                                   : llvm::ArrayRef<int64_t>(packedSlices))
    maxReadLatency = std::max(maxReadLatency, readLatency(slice));
  for (int64_t slice : rawToPacked ? llvm::ArrayRef<int64_t>(packedSlices)
                                   : rawSlices)
    maxWriteLatency = std::max(maxWriteLatency, writeLatency(slice));

  const int64_t packedBank = placementBankValue(packedPlacement);
  const int64_t captureStart = start + maxReadLatency;
  const int64_t tokenBlocks = seqLen / tile;
  for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
    for (int64_t headBlock = 0; headBlock < headBlocks; ++headBlock) {
      const int64_t firstWave =
          tokenBlock * headBlocks * tileRows + headBlock * tileRows;
      const int64_t rawAddress = layout.ropeStagingAddress(
          kind, head, headBlock, tokenBlock, 0);
      const int64_t packedAddress =
          placementBase(packedPlacement) + firstWave;
      for (int64_t hemisphere = 0;
           hemisphere < target.memory().hemispheres; ++hemisphere) {
        for (int64_t stream = 0; stream < width; ++stream) {
          const int64_t rawSlice = rawSlices[(stream + 2 * headBlock) % width];
          const int64_t inputSlice =
              rawToPacked ? rawSlice : packedSlices[stream];
          emitMem3D(
              rewriter, location,
              captureStart + firstWave - readLatency(inputSlice),
              hemisphere * target.memory().slices_per_hemisphere + inputSlice,
              "read", rawToPacked ? rawAddress : packedAddress,
              sourceBase + stream, tileRows, 1, 1, "sram", -1, 1, 1, 0, 1,
              1, 0, rawToPacked ? rawBank : packedBank);

          const int64_t outputSlice =
              rawToPacked ? packedSlices[stream] : rawSlice;
          emitMem3D(
              rewriter, location,
              captureStart + firstWave + 1 + writeLatency(outputSlice),
              hemisphere * target.memory().slices_per_hemisphere + outputSlice,
              "write", rawToPacked ? packedAddress : rawAddress,
              outputBase + stream, tileRows, 1, 1, "sram", -1, 1, 1, 0, 1,
              1, 0, rawToPacked ? packedBank : rawBank);
        }
      }
    }
  }
  // This wavefront is an affine SXM domain.  Transpose has an invariant body;
  // Permute advances the diagonal by one lane group on every beat.  Emit the
  // native RUN_2D induction fields directly instead of first materializing one
  // schedule op per beat.  Steady-state and drain beats have the same affine
  // body, so keep them in one hardware context.
  for (int64_t hemisphere = 0;
       hemisphere < target.memory().hemispheres; ++hemisphere) {
    emitSxm(rewriter, location, captureStart, hemisphere, "transpose",
            sourceStreams, transposeStreams, identityMap(), "vector_columns",
            -1, -1, -1, inputBeats, 1);
    emitSxm(rewriter, location, captureStart + 1, hemisphere, "permute",
            transposeStreams, outputStreams, blockDiagonalMap(0, target),
            "vector_columns", -1, -1, -1, inputBeats + tileRows - 1,
            1, 1, 1,
            target.throughput().lanes_per_tile);
  }
  return captureStart + inputBeats +
         std::max(tileRows, maxWriteLatency + 1);
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
  if (!projection_rope_overlap_enabled_ &&
      target_.throughput().mxms_per_hemisphere == 1) {
    const auto staging =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("rope_staging");
    const auto rows = staging
                          ? staging.getAs<mlir::IntegerAttr>("instruction_count")
                          : mlir::IntegerAttr{};
    const int64_t requiredRows =
        std::max(op_.getQueryHeads(), op_.getKvHeads()) *
        projectionHeadBlocks * op_.getSeqLen();
    if (!rows || rows.getInt() < requiredRows) {
      op_.emitError("serial Q/K projection requires a full-head RoPE "
                    "staging allocation; rerun StableHLO-to-Stream with "
                    "--projection-rope-overlap off");
      return -1;
    }
  }
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
  const auto inputStagingPongPlacement =
      inputDistributed16
          ? op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(
                "input_staging_pong")
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
  llvm::SmallVector<int64_t, 2> inputStagingPongSlices;
  if (inputStagingPongPlacement)
    for (mlir::Attribute value :
         inputStagingPongPlacement.getAs<mlir::ArrayAttr>("slices"))
      inputStagingPongSlices.push_back(
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
  const mlir::Value projectionBiases[] = {
      op_.getQueryBias(), op_.getKeyBias(), op_.getValueBias()};
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
  const int64_t inputStagingPongBank = inputStagingPongPlacement
                                          ? placementBank("input_staging_pong")
                                          : inputStagingBank;
  const int64_t projectionActivationBank =
      inputDistributed16 ? inputStagingBank : inputBank;
  const int64_t stagingBank = placementBank("rope_staging");
  const auto ropeStagingMirrorPlacement =
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(
          "rope_staging_mirror");
  const auto ropeStagingMirrorSlices =
      ropeStagingMirrorPlacement
          ? placementSlices(ropeStagingMirrorPlacement)
          : llvm::SmallVector<int64_t>{};
  const int64_t ropeStagingBase = placementBase(
      op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("rope_staging"));
  const int64_t ropeStagingMirrorBase = ropeStagingMirrorPlacement
      ? placementBase(ropeStagingMirrorPlacement)
      : ropeStagingBase;
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
  const auto stagingPairRoutable = [&](llvm::ArrayRef<int64_t> slices) {
    return slices.size() >= 2 && llvm::all_of(slices, [&](int64_t slice) {
        return target_.transport_latency(
                   target::StreamEndpoint::Mem,
                   target::StreamEndpoint::MxmActivation,
                   target::StreamDirection::East, slice)
                   .has_value() &&
               target_.transport_latency(
                   target::StreamEndpoint::VxmResult,
                   target::StreamEndpoint::Mem,
                   target::StreamDirection::East, slice)
                   .has_value();
      });
  };
  const bool inputStagingRoutable =
      stagingPairRoutable(inputStagingSlices) &&
      stagingPairRoutable(inputStagingPongSlices);
  const bool hasQkBias = projectionBiases[0] || projectionBiases[1];
  const bool directRopeCapable =
      projection_rope_overlap_enabled_ && target_.uses_dedicated_slice_roles() &&
      target_.memory().banks_per_slice > 1 &&
      target_.activation_storage_slices().size() >= 20 &&
      target_.throughput().vxm_cross_hemisphere_streams_enabled != 0 &&
      target_.throughput().vxm_fma_enabled != 0 &&
      target_.throughput().vxm_alus >= 4 && !hasQkBias &&
      !op_.hasQkNorm() &&
      target_.memory().hemispheres == 2 && projectionHeadBlocks == 4;
  const bool projectionRopeOverlap =
      projection_rope_overlap_enabled_ && !directRopeCapable && inputDistributed16 &&
      inputStagingRoutable &&
      tokenBlocks == 1 &&
      overlapActivationStreamBase + 2 <=
          target_.streams().streams_per_direction &&
      !productConflictsWithInput(productBank,
                                 layout.ropeProductSlices()) &&
      !productConflictsWithInput(alternateProductBank,
                                 layout.ropeProductSlices()) &&
      !productConflictsWithInput(keyProductBank,
                                 layout.ropeProductKeySlices());
  const bool secondaryActivationOverlap =
      projectionRopeOverlap &&
      target_.streams().streams_per_direction >= 18;

  if (inputDistributed16) {
    const auto &stagingSlices = inputStagingSlices;
    if (inputSlices.size() != 16 || stagingSlices.size() < 2) {
      op_.emitError("distributed attention input requires 16 source slices "
                    "and one FP16 staging pair");
      return -1;
    }
    const int64_t stagingCycle = 16;
    const int64_t stagingBase =
        inputStagingPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t tokenWaves = tile / 8;
    for (int64_t tokenLane = 0; tokenLane < 8; ++tokenLane) {
      for (int64_t hemisphere = 0;
           hemisphere < target_.memory().hemispheres; ++hemisphere) {
        for (int64_t byte = 0; byte < 2; ++byte) {
          const int64_t sourceSlice = inputSlices[2 * tokenLane + byte];
          const int64_t sourceLatency = *target_.transport_latency(
              target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
              target::StreamDirection::West, sourceSlice);
          emitMem3D(
              rewriter_, op_.getLoc(),
              stagingCycle + tokenLane - sourceLatency,
              hemisphere * target_.memory().slices_per_hemisphere +
                  sourceSlice,
              "read", inputBase, 32 + hemisphere * 16 + byte,
              tokenWaves, 8, 1, "sram", -1, tokenBlocks, tile,
              hiddenBlocks * 4, hiddenBlocks, op_.getSeqLen(), 4, inputBank);
        }
      }
    }

    // The VXM pass and staging writes advance over the flattened
    // reduction-by-token domain with unit cycle and address strides.
    const int64_t stagingItems = hiddenBlocks * op_.getSeqLen();
    emitVxmConfigured(rewriter_, op_.getLoc(), op_.getInput(),
                      stagingCycle - 1, 0, "pass", streamKind, 32, 0.0f,
                      "immediate", 0, 0.0f, "fp32", -1, "east", "east", -1,
                      2, stagingItems, 1);
    emitVxmConfigured(rewriter_, op_.getLoc(), op_.getInput(),
                      stagingCycle - 1, 1, "pass", "previous", 0, 0.0f,
                      "immediate", 0, 0.0f, dataFormat, 0, "east", "east", -1,
                      2, stagingItems, 1);

    int64_t lastWriteCycle = stagingCycle;
    for (int64_t hemisphere = 0;
         hemisphere < target_.memory().hemispheres; ++hemisphere) {
      for (int64_t byte = 0; byte < 2; ++byte) {
        const int64_t outputStream =
            (hemisphere == 0 ? int64_t{8} : int64_t{0}) + byte;
        const auto emitStagingCopy = [&](int64_t destinationSlice,
                                         int64_t destinationBank,
                                         int64_t destinationBase,
                                         bool preserveStream) {
          const int64_t destinationLatency = *target_.transport_latency(
              target::StreamEndpoint::VxmResult,
              target::StreamEndpoint::Mem,
              target::StreamDirection::East, destinationSlice);
          const int64_t writeCycle = stagingCycle + 1 + destinationLatency;
          emitMem3D(
              rewriter_, op_.getLoc(), writeCycle,
              hemisphere * target_.memory().slices_per_hemisphere +
                  destinationSlice,
              preserveStream ? "write_tap" : "write", destinationBase,
              outputStream, stagingItems, 1, 1,
              "sram", -1, 1, 1, 0, 1, 1, 0, destinationBank);
          lastWriteCycle =
              std::max(lastWriteCycle, writeCycle + stagingItems - 1);
        };
        // Slice 0/1 is upstream of slice 8/9 on this route.  Capture the pong
        // copy with WRITE_TAP so the same VXM result continues to the primary
        // staging pair.
        if (inputStagingPongPlacement &&
            inputStagingPongSlices.size() >= 2)
          emitStagingCopy(
              inputStagingPongSlices[byte], inputStagingPongBank,
              placementBase(inputStagingPongPlacement), true);
        emitStagingCopy(
            stagingSlices[byte], inputStagingBank, stagingBase, false);
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
    struct MemIcuInterval {
      int64_t hemisphere;
      int64_t slice;
      int64_t bank;
      int64_t begin;
      int64_t end;
    };
    llvm::SmallVector<MemIcuInterval, 64> previousRopeMemUses;
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
      const mlir::Value projectionBias = projectionBiases[projection];
      const char *biasPlacementNames[] = {
          "query_bias", "key_bias", "value_bias"};
      const auto biasPlacement = projectionBias
          ? op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(
                biasPlacementNames[projection])
          : mlir::DictionaryAttr {};
      llvm::SmallVector<int64_t, 4> biasSlices;
      int64_t biasBase = 0;
      int64_t biasBank = 0;
      if (projectionBias) {
        const auto biasKind = biasPlacement
            ? biasPlacement.getAs<mlir::StringAttr>("kind")
            : mlir::StringAttr {};
        if (!biasKind
            || biasKind.getValue() != "fp16_projection_bias_x4") {
          op_.emitError(
              "attention projection bias has no x4 physical placement");
          return -1;
        }
        for (mlir::Attribute slice :
             biasPlacement.getAs<mlir::ArrayAttr>("slices"))
          biasSlices.push_back(
              llvm::cast<mlir::IntegerAttr>(slice).getInt());
        if (biasSlices.size() != 4) {
          op_.emitError(
              "attention projection bias requires four MEM slices");
          return -1;
        }
        biasBase =
            biasPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
        biasBank = biasPlacement.getAs<mlir::IntegerAttr>("bank").getInt();
      }
      const auto biasAddress = [&](int64_t outputBlock) {
        return biasBase + (outputBlock / 4) * 2 + outputBlock % 2;
      };
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
      const bool projectionCanOverlap =
          projectionRopeOverlap &&
          (kind == AttentionProjectionKind::Query ||
           secondaryActivationOverlap);
      // With uninterrupted Q/K/V reductions, each output group occupies
      // exactly two 48-reduction halves.  Their weight queues have no
      // intervening work: flatten all halves into the third READ_3D counter
      // and use blocked-outer addressing for [group][half] physical layout.
      bool mergeProjectionWeightReads =
          (projectionCanOverlap || !projection_rope_overlap_enabled_) &&
          tokenBlocks == 1 &&
          target_.throughput().mxm_weight_buffers == 2 &&
          projectionOutputBlocks == projectionGroups * 4 &&
          projectionGroups > 1;
      if (mergeProjectionWeightReads) {
        for (int64_t hemisphere = 0;
             hemisphere < target_.memory().hemispheres; ++hemisphere) {
          const auto firstSlices = layout.weightSlices(kind, hemisphere * 2);
          for (int64_t stream = 0; stream < 8; ++stream) {
            const int64_t baseAddress = layout.weightAddress(
                kind, hemisphere * 2, 0, 0, 0);
            const int64_t halfStride =
                layout.weightAddress(kind, hemisphere * 2 + 1, 0, 1, 0) -
                baseAddress;
            const int64_t groupStride =
                layout.weightAddress(kind, hemisphere * 2 + 4, 0, 0, 0) -
                baseAddress;
            for (int64_t group = 0; group < projectionGroups; ++group) {
              for (int64_t half = 0; half < 2; ++half) {
                const int64_t outputBlock =
                    group * 4 + hemisphere * 2 + half;
                const auto slices = layout.weightSlices(kind, outputBlock);
                if (slices[stream] != firstSlices[stream] ||
                    layout.weightAddress(kind, outputBlock, 0, half, 0) !=
                        baseAddress + group * groupStride +
                            half * halfStride)
                  mergeProjectionWeightReads = false;
              }
            }
          }
        }
      }
      if (mergeProjectionWeightReads) {
        for (int64_t hemisphere = 0;
             hemisphere < target_.memory().hemispheres; ++hemisphere) {
          const auto slices = layout.weightSlices(kind, hemisphere * 2);
          const int64_t baseAddress =
              layout.weightAddress(kind, hemisphere * 2, 0, 0, 0);
          const int64_t halfStride =
              layout.weightAddress(kind, hemisphere * 2 + 1, 0, 1, 0) -
              baseAddress;
          const int64_t groupStride =
              layout.weightAddress(kind, hemisphere * 2 + 4, 0, 0, 0) -
              baseAddress;
          for (int64_t stream = 0; stream < 8; ++stream) {
            const int64_t slice = slices[stream];
            emitMem3D(
                rewriter_, op_.getLoc(),
                phaseStart - weightReadLatency(slice),
                hemisphere * target_.memory().slices_per_hemisphere + slice,
                "read", baseAddress, weightStreamBase + stream, 4, 1, 1,
                "sram", functionArgumentIndex(projectionValues[projection]),
                hiddenBlocks, op_.getSeqLen(), 8, projectionGroups * 2,
                hiddenBlocks * op_.getSeqLen(), 0,
                layout.weightBank(kind), layout.weightPage(kind), -1,
                2, halfStride, groupStride);
          }
        }
      }
      // In the serial policy all projection halves run before postprocessing.
      // Their activation reads revisit the same staging rows every
      // hiddenBlocks * seqLen cycles, so emit the whole [token][reduction]
      // [output half] domain directly as one READ_3D per physical MEM ICU.
      const bool mergeProjectionActivationReads =
          !projection_rope_overlap_enabled_ && mergeProjectionWeightReads &&
          inputDistributed16;
      // A serial projection has one uninterrupted reduction stream per MXM.
      // Keep the output-half axis in the hardware loop instead of creating a
      // new LOAD/DEQUANT/COMPUTE descriptor for every half.
      const bool mergeProjectionMxm =
          mergeProjectionActivationReads && localDequant;
      if (mergeProjectionActivationReads) {
        for (int64_t hemisphere = 0;
             hemisphere < target_.memory().hemispheres; ++hemisphere) {
          for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = projectionActivationSlices[byte];
            const int64_t readToActivation =
                *target_.transport_latency(
                    target::StreamEndpoint::Mem,
                    target::StreamEndpoint::MxmActivation,
                    target::StreamDirection::East, slice);
            emitMem3D(
                rewriter_, op_.getLoc(),
                phaseStart + 4 + loadToIw - readToActivation,
                hemisphere * target_.memory().slices_per_hemisphere + slice,
                "read", placementBase(inputStagingPlacement), byte, tile, 1,
                1, "sram", -1, hiddenBlocks, op_.getSeqLen(),
                op_.getSeqLen(), projectionGroups * 2,
                hiddenBlocks * op_.getSeqLen(), 0, projectionActivationBank);
          }
        }
      }
      // The serial policy first materializes every projection output group.
      // Q/K RoPE and V's cross-hemisphere packing run only after the complete
      // projection, leaving each activation and weight MEM queue uninterrupted.
      const bool deferPostprocess = !projection_rope_overlap_enabled_;
      llvm::SmallVector<int64_t> deferredRawWriteEnds(
          static_cast<std::size_t>(deferPostprocess ? projectionGroups : 0));
      const int64_t scheduleGroups =
          deferPostprocess ? 2 * projectionGroups : projectionGroups;
      for (int64_t scheduleGroup = 0; scheduleGroup < scheduleGroups;
           ++scheduleGroup) {
        if (deferPostprocess && scheduleGroup == projectionGroups) {
          // Postprocessing reads must follow the last raw output write, not
          // just the final MXM issue.
          int64_t maxReadLead = 0;
          for (int64_t slice : layout.ropeStagingSlices())
            maxReadLead = std::max(maxReadLead, readLatency(slice));
          phaseStart = std::max(
              phaseStart,
              *std::max_element(deferredRawWriteEnds.begin(),
                                deferredRawWriteEnds.end()) +
                  maxReadLead + 1);
        }
        const bool emitProjectionPass =
            !deferPostprocess || scheduleGroup < projectionGroups;
        const int64_t outputGroup = scheduleGroup % projectionGroups;
        // The Q staging write and its first cross-hemisphere copy read visit
        // the same four SRAM rows with different issue rates. Keep their
        // common address domain until both times are known, then emit one
        // hardware WRITE_READ_2D descriptor for each physical bank queue.
        const bool qWriteRead2D = projectionCanOverlap &&
            kind == AttentionProjectionKind::Query && !deferPostprocess &&
            !op_.hasQkNorm() && tokenBlocks == 1 &&
            target_.memory().banks_per_slice > 1 &&
            ropeStagingMirrorSlices.size() ==
                layout.ropeStagingSlices().size() &&
            placementBankValue(ropeStagingMirrorPlacement) ==
                (stagingBank + 1) % target_.memory().banks_per_slice &&
            projectionHeadBlocks == 4 &&
            outputGroup * 4 + 3 < projectionOutputBlocks;
        // The remote copy cannot use the raw bank while its single ICU
        // WRITE_READ context spans the two MXM halves and copy reads.
        const int64_t replicatedStagingBank = qWriteRead2D
            ? (stagingBank + 1) % target_.memory().banks_per_slice
            : stagingBank;
        // Q activation uses eastbound SR streams 0..7, and Q weights use
        // 24..31. Keep the mirror in 8..23 while Q groups overlap. At the
        // final Q/V boundary, V activation takes 16/17 and no next Q group
        // needs 0..7, so switch only that group's mirror to 0..15.
        const int64_t replicateStreamBase =
            qWriteRead2D && outputGroup + 1 < projectionGroups ? 8 : 0;
        const auto mirroredStagingSlice = [&](int64_t sourceSlice) {
          if (!qWriteRead2D)
            return sourceSlice;
          const auto stagingSlices = layout.ropeStagingSlices();
          const auto it = std::find(stagingSlices.begin(),
                                    stagingSlices.end(), sourceSlice);
          return ropeStagingMirrorSlices[
              std::distance(stagingSlices.begin(), it)];
        };
        const auto mirroredStagingAddress = [&](int64_t address) {
          return qWriteRead2D
              ? address + ropeStagingMirrorBase - ropeStagingBase
              : address;
        };
        const auto stagingBankFor = [&](int64_t sourceBlock,
                                        int64_t hemisphere) {
          return qWriteRead2D && hemisphere != sourceBlock / 2
                     ? replicatedStagingBank
                     : stagingBank;
        };
        struct PendingQWriteRead {
          int64_t writeCycle[2] = {-1, -1};
          int64_t writeAddress[2] = {-1, -1};
          int64_t writeStream[2] = {-1, -1};
          int64_t readCycle[2] = {-1, -1};
          int64_t readAddress[2] = {-1, -1};
          int64_t readStream[2] = {-1, -1};
        };
        std::array<std::array<PendingQWriteRead, 16>, 2> pendingQWriteRead{};
        llvm::SmallVector<MemIcuInterval, 64> currentRopeMemUses;
        const auto isActivationStagingQueue =
            [&](int64_t slice, int64_t bank) {
              return (bank == inputStagingBank &&
                      llvm::is_contained(inputStagingSlices, slice)) ||
                     (bank == inputStagingPongBank &&
                      llvm::is_contained(inputStagingPongSlices, slice));
            };
        const auto recordRopeMemDomain =
            [&](int64_t queue, int64_t bank, int64_t cycle,
                int64_t repeatCount, int64_t repeatInterval,
                int64_t waveCount, int64_t waveInterval,
                int64_t groupCount, int64_t groupInterval) {
              const int64_t hemisphere =
                  queue / target_.memory().slices_per_hemisphere;
              const int64_t slice =
                  queue % target_.memory().slices_per_hemisphere;
              if (!isActivationStagingQueue(slice, bank))
                return;
              // One MEM ICU executes one coarse instruction at a time.  Its
              // PC remains occupied across all loop-axis gaps, so reserve the
              // complete descriptor span rather than only its FU issue beats.
              const int64_t lastIssue =
                  cycle + (groupCount - 1) * groupInterval +
                  (waveCount - 1) * waveInterval +
                  (repeatCount - 1) * repeatInterval;
              currentRopeMemUses.push_back(
                  {hemisphere, slice, bank, cycle, lastIssue + 1});
            };
        struct PendingWeightReadDomain {
          bool valid = false;
          int64_t cycle = 0;
          int64_t queue = 0;
          int64_t address = 0;
          int64_t packedStream = 0;
          int64_t addressBinding = -1;
          int64_t bank = -1;
          int64_t weightPage = -1;
        };
        std::array<PendingWeightReadDomain, 16> pendingWeightReads{};
        const auto emitWeightReadDomain =
            [&](const PendingWeightReadDomain &domain, int64_t groupCount,
                int64_t groupInterval, int64_t groupAddressStride) {
              emitMem3D(
                  rewriter_, op_.getLoc(), domain.cycle, domain.queue, "read",
                  domain.address, domain.packedStream, 4, 1, 1, "sram",
                  domain.addressBinding, hiddenBlocks, op_.getSeqLen(), 8,
                  groupCount, groupInterval, groupAddressStride, domain.bank,
                  domain.weightPage);
            };
        const auto flushPendingWeightRead =
            [&](PendingWeightReadDomain &domain) {
              if (!domain.valid)
                return;
              emitWeightReadDomain(domain, 1, 1, 0);
              domain.valid = false;
            };
        int64_t rawWriteEnd = emitProjectionPass
                                  ? phaseStart
                                  : deferredRawWriteEnds[outputGroup];
        if (emitProjectionPass) {
        const bool rawQkHalfDomain =
            !directRopeCapable &&
            kind != AttentionProjectionKind::Value &&
            projectionHeadBlocks == 4 &&
            outputGroup * 4 + 3 < projectionOutputBlocks;
        int64_t firstRawWriteCycle[2][16];
        int64_t firstRawWriteAddress[2][16];
        std::fill(&firstRawWriteCycle[0][0],
                  &firstRawWriteCycle[0][0] + 32, -1);
        std::fill(&firstRawWriteAddress[0][0],
                  &firstRawWriteAddress[0][0] + 32, -1);
        for (int64_t half = 0; half < 2; ++half) {
          int64_t nextWeightDomain = 0;
          int64_t nextActivationDomain = 0;
          int64_t nextComputeDomain = 0;
          for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks;
               ++reductionBlock) {
            const bool finalReduction = reductionBlock + 1 == hiddenBlocks;
            // Once a half starts underneath the preceding output group's RoPE
            // pipeline, keep the whole reduction domain on the disjoint MXM
            // activation stream pair.  MEM reads select between the two
            // activation-region staging copies below; MXM issue remains one
            // continuous reduction domain.
            const bool overlapsRopeProducts =
                projectionCanOverlap && phaseStart < postprocessReady;
            const int64_t activationStreamBase =
                overlapsRopeProducts
                    ? (secondaryActivationOverlap &&
                               kind != AttentionProjectionKind::Query
                           ? 16
                           : overlapActivationStreamBase)
                    : 0;
            const int64_t weightBuffer =
                projectionBlock % target_.throughput().mxm_weight_buffers;
            const int64_t dequantStart = phaseStart;
            int64_t weightDomainCount = 0;
            if (target_.throughput().mxm_weight_buffers > 2) {
              weightDomainCount = 1;
              nextWeightDomain = reductionBlock + 1;
            } else if (reductionBlock == nextWeightDomain) {
              weightDomainCount = 1;
              while (reductionBlock + weightDomainCount < hiddenBlocks)
                ++weightDomainCount;
              nextWeightDomain = reductionBlock + weightDomainCount;
            }
            int64_t activationDomainCount = 0;
            if (reductionBlock == nextActivationDomain) {
              activationDomainCount = nextWeightDomain - reductionBlock;
              if (tokenBlocks > 1 && reductionBlock + 1 < hiddenBlocks &&
                  reductionBlock + activationDomainCount == hiddenBlocks)
                --activationDomainCount;
              activationDomainCount =
                  std::max<int64_t>(1, activationDomainCount);
              nextActivationDomain =
                  reductionBlock + activationDomainCount;
            }
            int64_t computeDomainCount = 0;
            if (reductionBlock == nextComputeDomain) {
              computeDomainCount = nextWeightDomain - reductionBlock;
              if (target_.throughput().mxm_weight_buffers > 2)
                computeDomainCount = 1;
              if (tokenBlocks > 1 && reductionBlock + 1 < hiddenBlocks &&
                  reductionBlock + computeDomainCount == hiddenBlocks)
                --computeDomainCount;
              computeDomainCount =
                  std::max<int64_t>(1, computeDomainCount);
              nextComputeDomain = reductionBlock + computeDomainCount;
            }
            for (int64_t hemisphere = 0;
                 hemisphere < target_.memory().hemispheres; ++hemisphere) {
              const int64_t outputBlock =
                  outputGroup * 4 + hemisphere * 2 + half;
              if (outputBlock >= projectionOutputBlocks)
                continue;
              const auto selectedWeightSlices =
                  layout.weightSlices(kind, outputBlock);
              if (weightDomainCount != 0) {
                for (int64_t stream = 0; stream < 8; ++stream) {
                  const int64_t slice = selectedWeightSlices[stream];
                  const int64_t readCycle =
                      dequantStart - weightReadLatency(slice);
                  const PendingWeightReadDomain current{
                      true,
                      readCycle,
                      hemisphere * target_.memory().slices_per_hemisphere +
                          slice,
                      layout.weightAddress(kind, outputBlock, reductionBlock,
                                           half, 0),
                      weightStreamBase + stream,
                      functionArgumentIndex(projectionValues[projection]),
                      layout.weightBank(kind),
                      layout.weightPage(kind)};
                  if (!mergeProjectionWeightReads) {
                    auto &pending = pendingWeightReads[static_cast<std::size_t>(
                        hemisphere * 8 + stream)];
                    const bool completeReductionDomain =
                        reductionBlock == 0 &&
                        weightDomainCount == hiddenBlocks;
                    if (half == 0 && completeReductionDomain) {
                      // Keep the first complete half as an operator-domain
                      // coordinate.  If the second half has the same physical
                      // queue/stream/page/bank placement, both halves become
                      // the group dimension of one READ_3D instruction.
                      pending = current;
                    } else {
                      const int64_t groupInterval =
                          readCycle - pending.cycle;
                      const int64_t groupAddressStride =
                          current.address - pending.address;
                      const int64_t innerDomainLastOffset =
                          (hiddenBlocks - 1) * op_.getSeqLen() + 3;
                      const bool pairCompleteHalves =
                          half == 1 && completeReductionDomain && pending.valid &&
                          current.queue == pending.queue &&
                          current.packedStream == pending.packedStream &&
                          current.addressBinding == pending.addressBinding &&
                          current.bank == pending.bank &&
                          current.weightPage == pending.weightPage &&
                          groupAddressStride == 4 &&
                          groupInterval > innerDomainLastOffset;
                      if (pairCompleteHalves) {
                        emitWeightReadDomain(pending, 2, groupInterval,
                                             groupAddressStride);
                        pending.valid = false;
                      } else {
                        flushPendingWeightRead(pending);
                        emitMem3D(
                            rewriter_, op_.getLoc(), current.cycle,
                            current.queue, "read", current.address,
                            current.packedStream, 4, 1, 1, "sram",
                            current.addressBinding, weightDomainCount,
                            op_.getSeqLen(), 8, 1, 1, 0, current.bank,
                            current.weightPage);
                      }
                    }
                  }
                  currentProjectionWeightRelease = std::max(
                      currentProjectionWeightRelease,
                      readCycle + (weightDomainCount - 1) * op_.getSeqLen() +
                          4);
                }
                if (localDequant &&
                    (!mergeProjectionMxm ||
                     (outputGroup == 0 && half == 0)))
                  emitMxmDequant3D(
                      rewriter_, op_.getLoc(), dequantStart, hemisphere,
                      projectionScales[projection], 4, 1, weightDomainCount,
                      op_.getSeqLen(),
                      mergeProjectionMxm ? projectionGroups * 2 : 1,
                      mergeProjectionMxm
                          ? hiddenBlocks * op_.getSeqLen()
                          : 1,
                      functionArgumentIndex(projectionValues[projection]));
              }
              if (!localDequant) {
                for (int64_t pulse = 0; pulse < 4; ++pulse)
                  emitDequant(dequantStart + pulse, hemisphere, 0,
                              projectionValues[projection],
                              projectionScales[projection]);
              }
              if ((weightDomainCount != 0 ||
                   target_.throughput().mxm_weight_buffers > 2) &&
                  (!mergeProjectionMxm ||
                   (outputGroup == 0 && half == 0))) {
                const int64_t loadGroupCount =
                    target_.throughput().mxm_weight_buffers > 2
                        ? 1
                        : weightDomainCount;
                MxmDomain3D loadDomain;
                loadDomain.wave_count = 4;
                loadDomain.wave_interval = 1;
                loadDomain.wave_weight_column_stride = -1;
                loadDomain.group_count = loadGroupCount;
                loadDomain.group_interval = op_.getSeqLen();
                if (loadGroupCount > 1 &&
                    target_.throughput().mxm_weight_buffers == 2)
                  loadDomain.weight_buffer_mode = "toggle_dim2";
                if (mergeProjectionMxm) {
                  loadDomain.repeat_count = 4;
                  loadDomain.repeat_interval = 1;
                  loadDomain.repeat_weight_column_stride = -1;
                  loadDomain.wave_count = hiddenBlocks;
                  loadDomain.wave_interval = op_.getSeqLen();
                  loadDomain.wave_weight_column_stride = 0;
                  loadDomain.group_count = projectionGroups * 2;
                  loadDomain.group_interval =
                      hiddenBlocks * op_.getSeqLen();
                  loadDomain.weight_buffer_mode = "toggle_dim1";
                }
                emitMxm3D(
                    rewriter_, op_.getLoc(), dequantStart + loadToIw,
                    hemisphere, "iw", weightBuffer, 3, 0, 0, 0, 1, "stream",
                    true, "supercell", 0, dataFormat,
                    localDequant ? "int8_dequant_bf16" : llvm::StringRef{},
                    llvm::StringRef{}, loadDomain,
                    localDequant ? weightStreamBase : -1);
              }
            }

            const int64_t firstCompute = dequantStart + 4 + loadToIw;
            struct ActivationReadSegment {
              int64_t begin;
              int64_t count;
              bool pong;
            };
            llvm::SmallVector<ActivationReadSegment, 8>
                activationReadSegments;
            if (activationDomainCount != 0 && overlapsRopeProducts) {
              const auto pairIsFree =
                  [&](llvm::ArrayRef<int64_t> slices, int64_t bank,
                      int64_t consumeCycle) {
                    for (int64_t candidateHemisphere = 0;
                         candidateHemisphere < target_.memory().hemispheres;
                         ++candidateHemisphere) {
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = slices[byte];
                        const int64_t issueCycle =
                            consumeCycle -
                            *target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmActivation,
                                target::StreamDirection::East, slice);
                        if (llvm::any_of(
                                previousRopeMemUses,
                                [&](const MemIcuInterval &use) {
                                  return use.hemisphere == candidateHemisphere &&
                                         use.slice == slice &&
                                         use.bank == bank &&
                                         issueCycle >= use.begin &&
                                         issueCycle < use.end;
                                }))
                          return false;
                      }
                    }
                    return true;
                  };
              const int64_t flattenedRows =
                  activationDomainCount * op_.getSeqLen();
              for (int64_t flat = 0; flat < flattenedRows; ++flat) {
                const int64_t consumeCycle = firstCompute + flat;
                const bool primaryFree = pairIsFree(
                    inputStagingSlices, inputStagingBank, consumeCycle);
                const bool pongFree = pairIsFree(
                    inputStagingPongSlices, inputStagingPongBank,
                    consumeCycle);
                if (!primaryFree && !pongFree) {
                  op_.emitError(
                      "RoPE overlap has no conflict-free activation staging "
                      "copy for a continuous MXM reduction");
                  return -1;
                }
                const bool usePong = !primaryFree;
                if (activationReadSegments.empty() ||
                    activationReadSegments.back().pong != usePong) {
                  activationReadSegments.push_back({flat, 1, usePong});
                } else {
                  ++activationReadSegments.back().count;
                }
              }
            }
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
                if (tokenBlock == 0 && activationDomainCount != 0 &&
                    !mergeProjectionActivationReads) {
                  if (overlapsRopeProducts) {
                    for (const ActivationReadSegment &segment :
                         activationReadSegments) {
                      const auto &slices = segment.pong
                                               ? inputStagingPongSlices
                                               : inputStagingSlices;
                      const int64_t bank = segment.pong
                                               ? inputStagingPongBank
                                               : inputStagingBank;
                      const int64_t base = segment.pong
                                               ? placementBase(
                                                     inputStagingPongPlacement)
                                               : placementBase(
                                                     inputStagingPlacement);
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = slices[byte];
                        const int64_t readToActivation =
                            *target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmActivation,
                                target::StreamDirection::East, slice);
                        emitMem3D(
                            rewriter_, op_.getLoc(),
                            firstCompute + segment.begin - readToActivation,
                            hemisphere *
                                    target_.memory().slices_per_hemisphere +
                                slice,
                            "read",
                            base + layout.activationAddress(reductionBlock, 0) +
                                segment.begin,
                            activationStreamBase + byte, segment.count, 1, 1,
                            "sram", -1, 1, 1, 0, 1, 1, 0, bank);
                      }
                    }
                  } else {
                    const int64_t inputAddress =
                        layout.activationAddress(reductionBlock, 0) +
                        (inputDistributed16
                             ? placementBase(inputStagingPlacement)
                             : 0);
                    for (int64_t byte = 0;
                         byte < (inputDistributed16 ? 2 : 4); ++byte) {
                      const int64_t slice = projectionActivationSlices[byte];
                      const int64_t readToActivation =
                          *target_.transport_latency(
                              target::StreamEndpoint::Mem,
                              target::StreamEndpoint::MxmActivation,
                              target::StreamDirection::East, slice);
                      emitMem3D(
                          rewriter_, op_.getLoc(),
                          firstCompute - readToActivation,
                          hemisphere *
                                  target_.memory().slices_per_hemisphere +
                              slice,
                          "read", inputAddress, activationStreamBase + byte,
                          tile, 1, 1, "sram", -1, tokenBlocks, computeSpacing,
                          tile, activationDomainCount, op_.getSeqLen(),
                          op_.getSeqLen(), projectionActivationBank);
                    }
                  }
                }
                const bool computeDomainIncludesFinal =
                    computeDomainCount != 0 &&
                    reductionBlock + computeDomainCount == hiddenBlocks;
                const int64_t resultStreamBase =
                    computeDomainIncludesFinal && hemisphere == 1 &&
                            ((directRopeCapable &&
                              kind != AttentionProjectionKind::Value) ||
                             (kind == AttentionProjectionKind::Value &&
                              projectionBias))
                        ? target_.streams().streams_per_direction / 2
                        : 0;
                if (tokenBlock == 0 && computeDomainCount != 0 &&
                    (!mergeProjectionMxm ||
                     (outputGroup == 0 && half == 0))) {
                  const bool terminalDomain =
                      computeDomainIncludesFinal && computeDomainCount > 1;
                  MxmDomain3D computeDomain;
                  computeDomain.repeat_count = tile;
                  computeDomain.repeat_accumulator_address_stride = 0;
                  computeDomain.wave_count = tokenBlocks;
                  computeDomain.wave_interval = computeSpacing;
                  computeDomain.wave_accumulator_address_stride = tile;
                  computeDomain.group_count = computeDomainCount;
                  computeDomain.group_interval = op_.getSeqLen();
                  if (computeDomainCount > 1 &&
                      target_.throughput().mxm_weight_buffers == 2)
                    computeDomain.weight_buffer_mode = "toggle_dim2";
                  if (terminalDomain) {
                    computeDomain.terminal_dimension = 2;
                    computeDomain.terminal_accumulator_destination = "stream";
                    computeDomain.terminal_accumulator_clear = true;
                    computeDomain.terminal_accumulator_output_format =
                        dataFormat;
                  }
                  if (mergeProjectionMxm) {
                    computeDomain.repeat_count = tile;
                    computeDomain.repeat_interval = 1;
                    computeDomain.wave_count = hiddenBlocks;
                    computeDomain.wave_interval = op_.getSeqLen();
                    computeDomain.wave_accumulator_address_stride = 0;
                    computeDomain.group_count = projectionGroups * 2;
                    computeDomain.group_interval =
                        hiddenBlocks * op_.getSeqLen();
                    computeDomain.weight_buffer_mode = "toggle_dim1";
                    computeDomain.terminal_dimension = 1;
                    computeDomain.terminal_accumulator_destination = "stream";
                    computeDomain.terminal_accumulator_clear = true;
                    computeDomain.terminal_accumulator_output_format =
                        dataFormat;
                  }
                  const bool finalOnly =
                      computeDomainIncludesFinal && computeDomainCount == 1;
                  emitMxm3D(
                      rewriter_, op_.getLoc(), firstCompute, hemisphere,
                      "compute", weightBuffer, 0, activationStreamBase,
                      resultStreamBase, half * accumulatorHalfStride, 1,
                      finalOnly ? "stream" : "sram", finalOnly, "supercell",
                      0, dataFormat, llvm::StringRef{},
                      finalOnly ? dataFormat : "fp32", computeDomain);
                }
                if (!finalReduction)
                  continue;

                // On the bias-free Qwen2.5 path the final MXM reduction feeds
                // RoPE directly.  The table reads and the Q/K writes are
                // affine in the token coordinates, so form their MEM domains
                // here instead of emitting one schedule op per token and
                // relying on a later grouping pass.
                if (directRopeCapable &&
                    kind != AttentionProjectionKind::Value) {
                  if (hemisphere != 0)
                    continue;

                  const int64_t firstVxmInput =
                      computeCycle +
                      target_.throughput().accumulator_to_vxm_latency;
                  const int64_t configCycle = firstVxmInput - 1;
                  emitVxmConfigured(
                      rewriter_, op_.getLoc(), projectionValues[projection],
                      configCycle, 0, "multiply", streamKind, 32, 0.0f,
                      streamKind, 40, 0.0f, "fp32", -1, "east", "east", -1,
                      2, tile, 1, "east", "east");
                  emitVxmConfigured(
                      rewriter_, op_.getLoc(), projectionValues[projection],
                      configCycle, 1, "fms", streamKind, 32, 0.0f,
                      streamKind, 42, 0.0f, dataFormat, 0, "east", "east", -1,
                      2, tile, 1, "west", "east");
                  emitVxmConfigured(
                      rewriter_, op_.getLoc(), projectionValues[projection],
                      configCycle, 2, "multiply", streamKind, 32, 0.0f,
                      streamKind, 40, 0.0f, "fp32", -1, "east", "east", -1,
                      2, tile, 1, "west", "east");
                  emitVxmConfigured(
                      rewriter_, op_.getLoc(), projectionValues[projection],
                      configCycle, 3, "fma", streamKind, 32, 0.0f,
                      streamKind, 42, 0.0f, dataFormat, 2, "east", "east", -1,
                      2, tile, 1, "east", "east");

                  const bool useMirrorTable =
                      kind == AttentionProjectionKind::Key;
                  const auto ropeSlices = useMirrorTable
                                              ? layout.ropeMirrorSlices()
                                              : layout.ropeSlices();
                  const int64_t directRopeBank =
                      useMirrorTable ? layout.ropeMirrorBank()
                                     : layout.ropeBank();
                  for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t cosineSlice = ropeSlices[byte];
                    emitMem3D(
                        rewriter_, op_.getLoc(),
                        firstVxmInput - vxmInputReadLatency(cosineSlice),
                        cosineSlice, "read",
                        useMirrorTable
                            ? layout.ropeMirrorAddress(tokenBlock * tile, half)
                            : layout.ropeAddress(tokenBlock * tile, half),
                        40 + byte, tile, 1, 1, "sram", -1, 1, 1, 0, 1, 1, 0,
                        directRopeBank);
                    const int64_t sineSlice = ropeSlices[2 + byte];
                    emitMem3D(
                        rewriter_, op_.getLoc(),
                        firstVxmInput - vxmInputReadLatency(sineSlice),
                        sineSlice, "read",
                        useMirrorTable
                            ? layout.ropeMirrorAddress(tokenBlock * tile, half)
                            : layout.ropeAddress(tokenBlock * tile, half),
                        42 + byte, tile, 1, 1, "sram", -1, 1, 1, 0, 1, 1, 0,
                        directRopeBank);
                  }

                  const int64_t blocks[] = {half, half + 2};
                  for (int64_t outputHalf = 0; outputHalf < 2;
                       ++outputHalf) {
                    const int64_t reduction = blocks[outputHalf];
                    for (int64_t byte = 0; byte < 2; ++byte) {
                      if (kind == AttentionProjectionKind::Query) {
                        for (int64_t tokenLane = 0;
                             tokenLane < target_.throughput().mxm_block_rows;
                             ++tokenLane) {
                          const int64_t slice =
                              layout.queryIwSlices(reduction)
                                  [2 * tokenLane + byte];
                          const int64_t latency =
                              target_
                                  .transport_latency(
                                      target::StreamEndpoint::VxmResult,
                                      target::StreamEndpoint::Mem,
                                      target::StreamDirection::East, slice)
                                  .value_or(readLatency(slice));
                          const int64_t cycle =
                              firstVxmInput + 3 + tokenLane + latency;
                          for (int64_t destination = 0;
                               destination < target_.memory().hemispheres;
                               ++destination) {
                            const int64_t source = 1 - destination;
                            emitMem3D(
                                rewriter_, op_.getLoc(), cycle,
                                destination *
                                        target_.memory().slices_per_hemisphere +
                                    slice,
                                "write",
                                layout.queryIwAddress(
                                    head, reduction, tokenBlock, 0),
                                source * 8 + outputHalf * 2 + byte,
                                target_.throughput().tile_rows,
                                target_.throughput().mxm_block_rows, 1,
                                "sram", -1, 1, 1, 0, 1, 1, 0,
                                layout.queryIwBank(reduction));
                          }
                          rawWriteEnd = std::max(
                              rawWriteEnd,
                              cycle +
                                  (target_.throughput().tile_rows - 1) *
                                      target_.throughput().mxm_block_rows +
                                  1);
                        }
                      } else {
                        const int64_t slice =
                            layout.keySlices(reduction)[byte];
                        const int64_t latency =
                            target_
                                .transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East, slice)
                                .value_or(readLatency(slice));
                        const int64_t cycle = firstVxmInput + 3 + latency;
                        for (int64_t destination = 0;
                             destination < target_.memory().hemispheres;
                             ++destination) {
                          const int64_t source = 1 - destination;
                          emitMem3D(
                              rewriter_, op_.getLoc(), cycle,
                              destination *
                                      target_.memory().slices_per_hemisphere +
                                  slice,
                              "write",
                              layout.keyAddress(
                                  head, reduction, tokenBlock),
                              source * 8 + outputHalf * 2 + byte,
                              tile, 1, 1, "sram", -1, 1, 1, 0, 1, 1, 0,
                              layout.keyBank(reduction));
                        }
                        rawWriteEnd =
                            std::max(rawWriteEnd, cycle + tile);
                      }
                    }
                  }
                  continue;
                }

                for (int64_t offset = 0; offset < tile; ++offset) {
                  const int64_t token = tokenBlock * tile + offset;
                  const int64_t resultCycle =
                      computeCycle + target_.mxm_first_result_latency() +
                      offset;
                  if (kind != AttentionProjectionKind::Value) {
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
                      if (tokenBlock == 0 && row == 0) {
                        const int64_t cycle = resultCycle + latency;
                        const int64_t address =
                            layout.ropeStagingAddress(
                                kind, head, headBlock, 0, 0);
                        if (qWriteRead2D) {
                          auto &pending = pendingQWriteRead[hemisphere][slice];
                          pending.writeCycle[half] = cycle;
                          pending.writeAddress[half] = address;
                          pending.writeStream[half] = 32 + byte;
                        } else if (rawQkHalfDomain && half == 0) {
                          firstRawWriteCycle[hemisphere][slice] = cycle;
                          firstRawWriteAddress[hemisphere][slice] = address;
                        } else {
                          const bool pairHalves =
                              rawQkHalfDomain &&
                              firstRawWriteCycle[hemisphere][slice] >= 0;
                          const int64_t domainCycle = pairHalves
                              ? firstRawWriteCycle[hemisphere][slice]
                              : cycle;
                          const int64_t domainAddress = pairHalves
                              ? firstRawWriteAddress[hemisphere][slice]
                              : address;
                          const bool mergeWholeProjection =
                              mergeProjectionMxm && pairHalves;
                          if (!mergeWholeProjection || outputGroup == 0) {
                            // The two output halves form dimension 1; all
                            // heads form dimension 2. Each physical queue
                            // keeps its own half timing and address offset.
                            emitMem3D(
                                rewriter_, op_.getLoc(), domainCycle,
                                hemisphere *
                                        target_.memory().slices_per_hemisphere +
                                    slice,
                                "write", domainAddress, 32 + byte,
                                target_.throughput().tile_rows, 8, 1,
                                "sram", -1,
                                mergeWholeProjection ? 2 : tokenBlocks,
                                mergeWholeProjection ? cycle - domainCycle
                                                     : computeSpacing,
                                mergeWholeProjection ? address - domainAddress
                                                     : tile,
                                mergeWholeProjection
                                    ? projectionGroups
                                    : (pairHalves ? 2 : 1),
                                mergeWholeProjection
                                    ? 2 * hiddenBlocks * op_.getSeqLen()
                                    : (pairHalves ? cycle - domainCycle : 1),
                                mergeWholeProjection
                                    ? layout.ropeStagingAddress(
                                          kind, head + 1, headBlock - 1, 0, 0) -
                                          domainAddress
                                    : (pairHalves ? address - domainAddress
                                                  : 0),
                                stagingBank);
                          }
                        }
                        rawWriteEnd = std::max(
                            rawWriteEnd,
                            cycle + (target_.throughput().tile_rows - 1) * 8 +
                                (tokenBlocks - 1) * computeSpacing + 1);
                      }
                    }
                  } else {
                    const int64_t packedStream = (token % 8) * 2;
                    const int64_t row = (token % tile) / 8;
                    const auto slices = layout.valuePackSlices(headBlock);
                    if (projectionBias) {
                      const int64_t vxmInputCycle =
                          computeCycle +
                          target_.throughput().accumulator_to_vxm_latency +
                          offset;
                      if (offset == 0 && hemisphere == 0) {
                        for (int64_t source = 0;
                             source < target_.memory().hemispheres;
                             ++source) {
                          const int64_t sourceBlock =
                              outputGroup * 4 + source * 2 + half;
                          if (sourceBlock >= projectionOutputBlocks)
                            continue;
                          const char *sourceName =
                              source == 0 ? "east" : "west";
                          const int64_t alu = 4 + source * 2;
                          emitVxmConfigured(
                              rewriter_, op_.getLoc(), projectionBias,
                              vxmInputCycle - 1, alu, "add", streamKind, 32,
                              0.0f, streamKind, 40, 0.0f, "fp32", -1,
                              sourceName, sourceName, -1, 2, tile, 1,
                              sourceName, sourceName);
                          emitVxmConfigured(
                              rewriter_, op_.getLoc(), projectionBias,
                              vxmInputCycle - 1, alu + 1, "pass", "previous",
                              0, 0.0f, "immediate", 0, 0.0f, dataFormat,
                              alu, sourceName, sourceName, -1, 2, tile, 1);
                        }
                      }
                      const int64_t pair = (outputBlock / 2) % 2;
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = biasSlices[2 * pair + byte];
                        if (offset == 0 && tokenBlock == 0)
                          emitMem3D(
                              rewriter_, op_.getLoc(),
                              vxmInputCycle - vxmInputReadLatency(slice),
                              hemisphere *
                                      target_.memory().slices_per_hemisphere +
                                  slice,
                              "read", biasAddress(outputBlock),
                              40 + hemisphere * 16 + byte, tile, 1, 0, "sram",
                              functionArgumentIndex(projectionBias),
                              tokenBlocks, computeSpacing, 0, 1, 1, 0,
                              biasBank);
                      }
                      const int64_t outputCycle = vxmInputCycle + 1;
                      const int64_t outputStream =
                          (1 - hemisphere) * 8 + 4 + hemisphere * 2;
                      for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = slices[packedStream + byte];
                        const int64_t latency =
                            target_
                                .transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East, slice)
                                .value_or(readLatency(slice));
                        if (tokenBlock == 0 && row == 0) {
                          const int64_t cycle = outputCycle + latency;
                          if (!mergeProjectionMxm ||
                              (outputGroup == 0 && half == 0))
                            emitMem3D(
                                rewriter_, op_.getLoc(), cycle,
                                hemisphere *
                                        target_.memory()
                                            .slices_per_hemisphere +
                                    slice,
                                "write",
                                layout.valuePackAddress(head, headBlock, 0, 0),
                                outputStream + byte,
                                target_.throughput().tile_rows, 8, 1,
                                "sram", -1,
                                mergeProjectionMxm ? 1 : tokenBlocks,
                                mergeProjectionMxm ? 1 : computeSpacing,
                                mergeProjectionMxm ? 0 : tile,
                                mergeProjectionMxm ? projectionGroups * 2 : 1,
                                mergeProjectionMxm
                                    ? hiddenBlocks * op_.getSeqLen()
                                    : 1,
                                0, placementBank("value"), -1, -1,
                                mergeProjectionMxm ? 2 : 1,
                                mergeProjectionMxm
                                    ? layout.valuePackAddress(
                                          head, headBlock + 1, 0, 0) -
                                          layout.valuePackAddress(
                                              head, headBlock, 0, 0)
                                    : 0,
                                mergeProjectionMxm
                                    ? layout.valuePackAddress(
                                          head + 1, headBlock, 0, 0) -
                                          layout.valuePackAddress(
                                              head, headBlock, 0, 0)
                                    : 0);
                          rawWriteEnd = std::max(
                              rawWriteEnd,
                              cycle +
                                  (target_.throughput().tile_rows - 1) * 8 +
                                  (tokenBlocks - 1) * computeSpacing + 1);
                        }
                      }
                      continue;
                    }
                    for (int64_t byte = 0; byte < 2; ++byte) {
                      const int64_t slice = slices[packedStream + byte];
                      const int64_t latency = *target_.transport_latency(
                          target::StreamEndpoint::MxmResult,
                          target::StreamEndpoint::Mem,
                          target::StreamDirection::West, slice);
                      if (tokenBlock == 0 && row == 0) {
                        const int64_t cycle = resultCycle + latency;
                        if (!mergeProjectionMxm ||
                            (outputGroup == 0 && half == 0))
                          emitMem3D(
                              rewriter_, op_.getLoc(), cycle,
                              hemisphere *
                                      target_.memory().slices_per_hemisphere +
                                  slice,
                              "write",
                              layout.valuePackAddress(head, headBlock, 0, 0),
                              32 + byte, target_.throughput().tile_rows, 8, 1,
                              "sram", -1,
                              mergeProjectionMxm ? 1 : tokenBlocks,
                              mergeProjectionMxm ? 1 : computeSpacing,
                              mergeProjectionMxm ? 0 : tile,
                              mergeProjectionMxm ? projectionGroups * 2 : 1,
                              mergeProjectionMxm
                                  ? hiddenBlocks * op_.getSeqLen()
                                  : 1,
                              0, placementBank("value"), -1, -1,
                              mergeProjectionMxm ? 2 : 1,
                              mergeProjectionMxm
                                  ? layout.valuePackAddress(
                                        head, headBlock + 1, 0, 0) -
                                        layout.valuePackAddress(
                                            head, headBlock, 0, 0)
                                  : 0,
                              mergeProjectionMxm
                                  ? layout.valuePackAddress(
                                        head + 1, headBlock, 0, 0) -
                                        layout.valuePackAddress(
                                            head, headBlock, 0, 0)
                                  : 0);
                        rawWriteEnd = std::max(
                            rawWriteEnd,
                            cycle + (target_.throughput().tile_rows - 1) * 8 +
                                (tokenBlocks - 1) * computeSpacing + 1);
                      }
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
        for (PendingWeightReadDomain &domain : pendingWeightReads)
          flushPendingWeightRead(domain);
        }

        if (deferPostprocess && emitProjectionPass) {
          deferredRawWriteEnds[outputGroup] = rawWriteEnd;
          continue;
        }

        const auto &memory = target_.memory();
        const int64_t blockRows = target_.throughput().mxm_block_rows;
        const int64_t tileRows = target_.throughput().tile_rows;
        const int64_t blockIssues = tile / blockRows;
        const auto ropeTokenCycleOffset = [&](int64_t token) {
          const int64_t tokenBlock = token / tile;
          const int64_t tokenWithinBlock = token % tile;
          const int64_t tokenLane = tokenWithinBlock % blockRows;
          const int64_t row = tokenWithinBlock / blockRows;
          return tokenBlock * tile +
                 tokenLane * target_.throughput().tile_rows + row;
        };
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
            for (int64_t stream = 0; stream < 16; ++stream) {
              const int64_t slice = slices[stream];
              const int64_t address =
                  layout.valuePackAddress(head, headBlock, 0, 0);
              emitMem3D(
                  rewriter_, op_.getLoc(), copyCycle - readLatency(slice),
                  sourceHemisphere * memory.slices_per_hemisphere + slice,
                  "read", address, 32 + stream, blockIssues, 1, 1, "sram", -1,
                  tokenBlocks, blockIssues, blockIssues, 1, 1, 0,
                  placementBank("value"));
              const int64_t latency =
                  target_
                      .transport_latency(
                          target::StreamEndpoint::VxmBridgeResult,
                          target::StreamEndpoint::Mem,
                          target::StreamDirection::East, slice)
                      .value_or(readLatency(slice));
              emitMem3D(
                  rewriter_, op_.getLoc(), copyCycle + latency,
                  destinationHemisphere * memory.slices_per_hemisphere + slice,
                  "write", address, stream, blockIssues, 1, 1, "sram", -1,
                  tokenBlocks, blockIssues, blockIssues, 1, 1, 0,
                  placementBank("value"));
              copyEnd = std::max(
                  copyEnd,
                  copyCycle + latency + tokenBlocks * blockIssues);
            }
            copyCycle += tokenBlocks * blockIssues;
            copyCycle =
                std::max(copyCycle, copyEnd + maxStagingReadLatency + 1);
          }
          // The next group's MXM can run while the completed Value group is
          // copied, provided its activation stream pair is disjoint from the
          // preceding group's VXM result stream and the copy's SR links.
          postprocessReady =
              std::max({postprocessReady, copyCycle + 1, copyEnd});
          if (!projectionCanOverlap)
            phaseStart = std::max(phaseStart, postprocessReady);
          previousRopeMemUses.clear();
          continue;
        }

        if (directRopeCapable) {
          phaseStart = std::max(phaseStart, rawWriteEnd);
          postprocessReady = phaseStart;
          previousRopeMemUses.clear();
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
          for (int64_t stream = 0; stream < 16; ++stream) {
            const int64_t sourceChannel =
                (stream + 2 * headBlock) % 16;
            const int64_t slice =
                layout.ropeStagingSlices()[sourceChannel];
            const int64_t replicateStream = replicateStreamBase +
                (qWriteRead2D ? sourceChannel : stream);
            const int64_t address =
                layout.ropeStagingAddress(kind, head, headBlock, 0, 0);
            const int64_t copyReadCycle =
                replicateCycle - readLatency(slice);
            if (qWriteRead2D) {
              auto &pending = pendingQWriteRead[sourceHemisphere][slice];
              const int64_t half = outputBlock % 2;
              pending.readCycle[half] = copyReadCycle;
              pending.readAddress[half] = address;
              pending.readStream[half] = 32 + replicateStream;
              if (half == 1) {
                const int64_t writeOuter =
                    pending.writeCycle[1] - pending.writeCycle[0];
                const int64_t readOuter =
                    pending.readCycle[1] - pending.readCycle[0];
                const int64_t readStart =
                    pending.readCycle[0] - pending.writeCycle[0];
                const int64_t addressOuter =
                    pending.writeAddress[1] - pending.writeAddress[0];
                const int64_t streamOuter =
                    pending.readStream[1] - pending.readStream[0];
                const bool encodable = pending.writeCycle[0] >= 0 &&
                    pending.readCycle[0] >= 0 &&
                    pending.writeStream[0] == pending.writeStream[1] &&
                    pending.writeAddress[0] == pending.readAddress[0] &&
                    pending.writeAddress[1] == pending.readAddress[1] &&
                    writeOuter > 3 * 8 && readOuter > 3 &&
                    readStart > 0 && readStart < (int64_t{1} << 24) &&
                    writeOuter < (int64_t{1} << 24) &&
                    readOuter < (int64_t{1} << 24) &&
                    addressOuter >= -(int64_t{1} << 19) &&
                    addressOuter < (int64_t{1} << 19) &&
                    streamOuter >= -32 && streamOuter <= 31;
                if (encodable) {
                  mlir::OperationState state(
                      op_.getLoc(), MemWriteRead2DOp::getOperationName());
                  const auto attr = [&](llvm::StringRef name, int64_t value) {
                    state.addAttribute(name,
                                       rewriter_.getI64IntegerAttr(value));
                  };
                  attr("cycle", pending.writeCycle[0]);
                  attr("hemisphere", sourceHemisphere);
                  attr("slice", slice);
                  attr("bank", stagingBank);
                  attr("address", pending.writeAddress[0]);
                  attr("count0", target_.throughput().tile_rows);
                  attr("count1", 2);
                  attr("write_cycle_stride0",
                       target_.throughput().mxm_block_rows);
                  attr("write_cycle_stride1", writeOuter);
                  attr("read_cycle_stride0", 1);
                  attr("read_cycle_stride1", readOuter);
                  attr("read_start_offset", readStart);
                  attr("address_stride0", 1);
                  attr("address_stride1", addressOuter);
                  attr("write_stream", pending.writeStream[0]);
                  attr("read_stream_base", pending.readStream[0]);
                  attr("read_stream_outer_stride", streamOuter);
                  rewriter_.create(state);
                } else {
                  // Layouts with a different row or stream permutation stay
                  // on the ordinary direct MEM3D lowering path.
                  for (int64_t part = 0; part < 2; ++part) {
                    emitMem3D(
                        rewriter_, op_.getLoc(), pending.writeCycle[part],
                        sourceHemisphere * memory.slices_per_hemisphere +
                            slice,
                        "write", pending.writeAddress[part],
                        pending.writeStream[part],
                        target_.throughput().tile_rows,
                        target_.throughput().mxm_block_rows, 1, "sram", -1,
                        1, 1, 0, 1, 1, 0, stagingBank);
                    emitMem3D(
                        rewriter_, op_.getLoc(), pending.readCycle[part],
                        sourceHemisphere * memory.slices_per_hemisphere +
                            slice,
                        "read", pending.readAddress[part],
                        pending.readStream[part], blockIssues, 1, 1,
                        "sram", -1, tokenBlocks, blockIssues, tile, 1, 1,
                        0, stagingBank);
                  }
                }
              }
            } else {
              emitMem3D(
                  rewriter_, op_.getLoc(), copyReadCycle,
                  sourceHemisphere * memory.slices_per_hemisphere + slice,
                  "read", address, 32 + replicateStream,
                  blockIssues, 1, 1,
                  "sram", -1, tokenBlocks, blockIssues, tile, 1, 1, 0,
                  stagingBank);
            }
            const int64_t latency =
                target_
                    .transport_latency(
                        target::StreamEndpoint::VxmBridgeResult,
                        target::StreamEndpoint::Mem,
                        target::StreamDirection::East,
                        mirroredStagingSlice(slice))
                    .value_or(readLatency(mirroredStagingSlice(slice)));
            emitMem3D(
                rewriter_, op_.getLoc(), replicateCycle + latency,
                destinationHemisphere * memory.slices_per_hemisphere +
                    mirroredStagingSlice(slice),
                "write", mirroredStagingAddress(address),
                replicateStream,
                blockIssues, 1, 1, "sram", -1,
                tokenBlocks, blockIssues, tile, 1, 1, 0,
                replicatedStagingBank);
            replicateEnd = std::max(
                replicateEnd,
                replicateCycle + latency + tokenBlocks * blockIssues);
          }
          replicateCycle += tokenBlocks * blockIssues;
          replicateCycle = std::max(replicateCycle,
                                    replicateEnd + maxStagingReadLatency + 1);
        }

        if (op_.hasQkNorm()) {
          if (firstOutputBlock % projectionHeadBlocks != 0 ||
              lastOutputBlock - firstOutputBlock != projectionHeadBlocks) {
            op_.emitError("Q/K head RMSNorm requires one complete head per "
                          "projection output group");
            return -1;
          }
          const int64_t head = firstOutputBlock / projectionHeadBlocks;
          const char *normInputName =
              kind == AttentionProjectionKind::Query ? "query_norm_input"
                                                     : "key_norm_input";
          const char *normOutputName =
              kind == AttentionProjectionKind::Query ? "query_norm_output"
                                                     : "key_norm_output";
          const auto normInputPlacement =
              op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(normInputName);
          const auto normOutputPlacement =
              op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(normOutputName);
          const char *normWeightName =
              kind == AttentionProjectionKind::Query ? "query_norm_weight"
                                                     : "key_norm_weight";
          const auto normWeightPlacement =
              op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(normWeightName);
          const mlir::Value normWeight =
              kind == AttentionProjectionKind::Query
                  ? op_.getQueryNormWeight()
                  : op_.getKeyNormWeight();
          const auto epsilonAttr = op_.query_rope.getConfig()
              .getAs<mlir::FloatAttr>("qk_norm_epsilon");
          if (!normInputPlacement || !normOutputPlacement ||
              !normWeightPlacement || !epsilonAttr) {
            op_.emitError("Q/K head RMSNorm is missing a placement or epsilon");
            return -1;
          }
          const int64_t normStart = std::max(
              replicateCycle, replicateEnd + maxStagingReadLatency + 1);
          const int64_t transposeEnd = emitRopeNormTranspose(
              rewriter_, op_.getLoc(), target_, layout, kind, head,
              op_.getSeqLen(), op_.getHeadDim(), stagingBank,
              normInputPlacement, true, normStart);
          const auto headType = mlir::RankedTensorType::get(
              {op_.getSeqLen(), op_.getHeadDim()}, elementType);
          const int64_t feedbackEnd = emitVxmFeedbackRmsNorm(
              rewriter_, op_.getLoc(), projectionValues[projection],
              normWeight, headType, epsilonAttr.getValueAsDouble(), target_,
              normInputPlacement, normWeightPlacement, normOutputPlacement,
              transposeEnd);
          const int64_t restoreEnd = emitRopeNormTranspose(
              rewriter_, op_.getLoc(), target_, layout, kind, head,
              op_.getSeqLen(), op_.getHeadDim(), stagingBank,
              normOutputPlacement, false, feedbackEnd);
          replicateCycle = restoreEnd;
          replicateEnd = restoreEnd;
        }

        const int64_t productPipelineLatency = projectionBias ? 3 : 2;
        const auto emitRopeProducts = [&](int64_t cycle,
                                          int64_t inputHemisphere,
                                          int64_t outputHemisphere,
                                          bool swapBiasHalves) {
          const char *input = inputHemisphere == 0 ? "east" : "west";
          const char *output = outputHemisphere == 0 ? "east" : "west";
          emitVxmConfigured(rewriter_, op_.getLoc(),
                            projectionValues[projection], cycle, 0, "multiply",
                            streamKind, 32, 0.0f, streamKind, 34, 0.0f, "fp32",
                            -1, input, output, -1, 2, op_.getSeqLen(), 1);
          if (projectionBias)
            emitVxmConfigured(
                rewriter_, op_.getLoc(), projectionBias, cycle, 1, "fma",
                streamKind, swapBiasHalves ? 42 : 40, 0.0f, streamKind,
                34, 0.0f, dataFormat, 0, input, output, -1, 2,
                op_.getSeqLen(), 1, input, input);
          else
            emitVxmConfigured(
                rewriter_, op_.getLoc(), projectionValues[projection], cycle,
                1, "pass", "previous", 0, 0.0f, "immediate", 0, 0.0f,
                dataFormat, 0, input, output, -1, 2, op_.getSeqLen(), 1);
          emitVxmConfigured(rewriter_, op_.getLoc(),
                            projectionValues[projection], cycle, 2, "multiply",
                            streamKind, 36, 0.0f, streamKind, 38, 0.0f, "fp32",
                            -1, input, output, -1, 2, op_.getSeqLen(), 1);
          if (projectionBias)
            emitVxmConfigured(
                rewriter_, op_.getLoc(), projectionBias, cycle, 3, "fma",
                streamKind, swapBiasHalves ? 40 : 42, 0.0f, streamKind,
                38, 0.0f, dataFormat, 2, input, output, -1, 2,
                op_.getSeqLen(), 1, input, input);
          else
            emitVxmConfigured(
                rewriter_, op_.getLoc(), projectionValues[projection], cycle,
                3, "pass", "previous", 0, 0.0f, "immediate", 0, 0.0f,
                dataFormat, 2, input, output, -1, 2, op_.getSeqLen(), 1);
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
        const int64_t firstHead = firstOutputBlock / projectionHeadBlocks;
        const int64_t lastHead = (lastOutputBlock - 1) / projectionHeadBlocks;
        const int64_t queryHeadsPerKv = op_.getQueryHeads() / op_.getKvHeads();
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
          // A MEM ICU has one iMEM, FIFO, and PC for both reads and writes.
          // Keeping a descriptor alive across the gap between rotary pair
          // blocks would require another MEM command to execute in that gap
          // (for example, the mirror-table read spans a Query-IW/product
          // write).  Emit each pair block as its own closed-form 3D domain so
          // the single hardware queue can retire one descriptor before the
          // next descriptor starts.
          const bool pairBlockDomain = false;
          int64_t firstPairProductOutput[2] = {-1, -1};
          int64_t firstPairBiasInput = -1;
          int64_t firstPairMirrorTableInput = -1;
          int64_t firstPairCombineInput = -1;
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
                      productAInputOffset + op_.getSeqLen() +
                      (productPipelineLatency - 1) +
                      maxProductWriteLatency + maxStagingReadLatency + 1;
                  const int64_t productBInputOffset =
                      productBConfigOffset + 1;
                  const int64_t combineConfigOffset =
                      productBInputOffset + op_.getSeqLen() +
                      (productPipelineLatency - 1) +
                      maxProductWriteLatency + maxProductReadLatency + 1;
                  const int64_t combineInputOffset = combineConfigOffset + 1;

                  const auto appendProductPhase =
                      [&](int64_t inputOffset, int64_t firstProduct,
                          bool mirroredRopeTable) {
                        for (int64_t token = 0; token < op_.getSeqLen();
                             ++token) {
                          const int64_t tokenLane = token % blockRows;
                          const int64_t inputCycle =
                              inputOffset + ropeTokenCycleOffset(token);
                          for (int64_t half = 0; half < 2; ++half) {
                            const int64_t sourceBlock = blocks[half];
                            for (int64_t hemisphere = 0;
                                 hemisphere < memory.hemispheres;
                                 ++hemisphere) {
                              for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t stagingSlice =
                                    layout.ropeStagingSlices()
                                        [(2 * tokenLane + byte +
                                             2 * sourceBlock) %
                                            16];
                                const int64_t slice =
                                    qWriteRead2D &&
                                            hemisphere != sourceBlock / 2
                                        ? mirroredStagingSlice(stagingSlice)
                                        : stagingSlice;
                                appendMemResources(
                                    windows,
                                    inputCycle - vxmInputReadLatency(slice),
                                    hemisphere, slice,
                                    stagingBankFor(sourceBlock, hemisphere),
                                    false);
                              }
                            }
                          }
                          for (int64_t hemisphere = 0;
                               hemisphere < memory.hemispheres; ++hemisphere) {
                            for (int64_t byte = 0; byte < 4; ++byte) {
                              const int64_t slice = mirroredRopeTable
                                  ? layout.ropeMirrorSlices()[byte]
                                  : layout.ropeSlices()[byte];
                              appendMemResources(
                                  windows,
                                  inputCycle - vxmInputReadLatency(slice),
                                  hemisphere, slice,
                                  mirroredRopeTable ? ropeMirrorBank : ropeBank,
                                  false);
                            }
                          }
                          if (projectionBias) {
                            for (int64_t half = 0; half < 2; ++half) {
                              const int64_t outputBlock =
                                  head * projectionHeadBlocks + blocks[half];
                              const int64_t pair = (outputBlock / 2) % 2;
                              for (int64_t hemisphere = 0;
                                   hemisphere < memory.hemispheres;
                                   ++hemisphere) {
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                  const int64_t slice =
                                      biasSlices[2 * pair + byte];
                                  appendMemResources(
                                      windows,
                                      inputCycle - vxmInputReadLatency(slice),
                                      hemisphere, slice, biasBank, false);
                                }
                              }
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
                                    windows,
                                    inputCycle + productPipelineLatency +
                                        latency,
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
                    const int64_t inputCycle =
                        combineInputOffset + ropeTokenCycleOffset(token);
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
            const bool compactProductLayout = productSlices.size() < 16;
            const auto productSlice = [&](int64_t product, int64_t token,
                                          int64_t byte) {
              if (compactProductLayout)
                return layout.ropeProductSlice(kind, product, token, byte);
              return productSlices[(token % 2) * 8 + product * 2 + byte];
            };
            const auto productStorageBank = [&](int64_t product) {
              return compactProductLayout
                         ? layout.ropeProductBank(kind, productBank, product)
                         : productBank;
            };
            const auto emitSourceDomain = [&](int64_t half, int64_t stream,
                                              int64_t inputCycle) {
              const int64_t sourceBlock = blocks[half];
              for (int64_t tokenLane = 0; tokenLane < blockRows;
                   ++tokenLane) {
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                  for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t stagingSlice =
                        layout.ropeStagingSlices()[(2 * tokenLane + byte +
                                                    2 * sourceBlock) %
                                                   16];
                    const int64_t slice =
                        qWriteRead2D && hemisphere != sourceBlock / 2
                            ? mirroredStagingSlice(stagingSlice)
                            : stagingSlice;
                    emitMem3D(
                        rewriter_, op_.getLoc(),
                        inputCycle +
                            tokenLane * target_.throughput().tile_rows -
                            vxmInputReadLatency(slice),
                        hemisphere * memory.slices_per_hemisphere + slice,
                        "read",
                        qWriteRead2D && hemisphere != sourceBlock / 2
                            ? mirroredStagingAddress(
                                  layout.ropeStagingAddress(
                                      kind, head, sourceBlock, 0, 0))
                            : layout.ropeStagingAddress(
                                  kind, head, sourceBlock, 0, 0),
                        stream + hemisphere * 16 + byte,
                        target_.throughput().tile_rows, 1, 1, "sram", -1,
                        tokenBlocks, tile, tile, 1, 1, 0,
                        stagingBankFor(sourceBlock, hemisphere));
                  }
                }
              }
            };
            const auto emitRopeTableDomain = [&](int64_t inputCycle,
                                                 bool mirror) {
              if (pairBlockDomain && mirror && pairBlock == 0) {
                firstPairMirrorTableInput = inputCycle;
                return;
              }
              const bool pairDomain =
                  pairBlockDomain && mirror && pairBlock != 0;
              const int64_t domainPairBlock = pairDomain ? 0 : pairBlock;
              const int64_t domainInputCycle =
                  pairDomain ? firstPairMirrorTableInput : inputCycle;
              for (int64_t hemisphere = 0; hemisphere < memory.hemispheres;
                   ++hemisphere) {
                for (int64_t byte = 0; byte < 4; ++byte) {
                  const int64_t slice = mirror
                      ? layout.ropeMirrorSlices()[byte]
                      : layout.ropeSlices()[byte];
                  const int64_t stream =
                      (byte < 2 ? 34 + byte : 36 + byte) + hemisphere * 16;
                  const int64_t bank = mirror ? ropeMirrorBank : ropeBank;
                  const int64_t queue =
                      hemisphere * memory.slices_per_hemisphere + slice;
                  const int64_t domainCycle =
                      domainInputCycle - vxmInputReadLatency(slice);
                  emitMem3D(
                      rewriter_, op_.getLoc(), domainCycle, queue,
                      "read",
                      mirror ? layout.ropeMirrorAddress(0, domainPairBlock)
                             : layout.ropeAddress(0, domainPairBlock),
                      stream, target_.throughput().tile_rows, 1, blockRows,
                      "sram", -1, blockRows,
                      target_.throughput().tile_rows, 1,
                      pairDomain ? 2 : tokenBlocks,
                      pairDomain
                          ? inputCycle - firstPairMirrorTableInput
                          : tile,
                      pairDomain ? op_.getSeqLen() : tile, bank);
                  recordRopeMemDomain(
                      queue, bank, domainCycle,
                      target_.throughput().tile_rows, 1,
                      blockRows, target_.throughput().tile_rows,
                      pairDomain ? 2 : tokenBlocks,
                      pairDomain ? inputCycle - firstPairMirrorTableInput
                                 : tile);
                }
              }
            };
            const auto emitProjectionBiasDomain = [&](int64_t inputCycle,
                                                       int64_t phaseInterval) {
              if (!projectionBias)
                return;
              if (pairBlockDomain && pairBlock == 0) {
                firstPairBiasInput = inputCycle;
                return;
              }
              const bool pairDomain = pairBlockDomain && pairBlock != 0;
              const int64_t domainInputCycle =
                  pairDomain ? firstPairBiasInput : inputCycle;
              for (int64_t half = 0; half < 2; ++half) {
                const int64_t domainReductionBlock = pairDomain
                    ? half * (projectionHeadBlocks / 2)
                    : blocks[half];
                const int64_t outputBlock =
                    head * projectionHeadBlocks + domainReductionBlock;
                const int64_t pair = (outputBlock / 2) % 2;
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                  for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = biasSlices[2 * pair + byte];
                    emitMem3D(
                        rewriter_, op_.getLoc(),
                        domainInputCycle - vxmInputReadLatency(slice),
                        hemisphere * memory.slices_per_hemisphere + slice,
                        "read", biasAddress(outputBlock),
                        40 + hemisphere * 16 + half * 2 + byte,
                        op_.getSeqLen(), 1, 0, "sram",
                        functionArgumentIndex(projectionBias), 2,
                        phaseInterval, 0, pairDomain ? 2 : 1,
                        pairDomain ? inputCycle - firstPairBiasInput : 1,
                        0, biasBank);
                  }
                }
              }
            };
            const auto emitProductWriteDomain = [&](int64_t firstProduct,
                                                    int64_t outputCycle) {
              const int64_t productPhase = firstProduct / 2;
              if (pairBlockDomain && pairBlock == 0) {
                firstPairProductOutput[productPhase] = outputCycle;
                return;
              }
              const int64_t domainPairBlock = pairBlockDomain ? 0 : pairBlock;
              const int64_t domainOutputCycle =
                  pairBlockDomain
                      ? firstPairProductOutput[productPhase]
                      : outputCycle;
              for (int64_t slot = 0; slot < 2; ++slot) {
                const int64_t product = firstProduct + slot;
                for (int64_t byte = 0; byte < 2; ++byte) {
                  const int64_t slice0 = productSlice(product, 0, byte);
                  const int64_t slice1 = op_.getSeqLen() > 1
                                             ? productSlice(product, 1, byte)
                                             : slice0;
                  const bool paritySplit = slice0 != slice1;
                  const int64_t parityCount = paritySplit ? 2 : 1;
                  for (int64_t parity = 0; parity < parityCount; ++parity) {
                    const int64_t slice = productSlice(product, parity, byte);
                    const int64_t latency =
                        target_
                            .transport_latency(
                                target::StreamEndpoint::VxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::East, slice)
                            .value_or(readLatency(slice));
                    const int64_t baseAddress = layout.ropeProductAddress(
                        kind, head, domainPairBlock, product, parity);
                    const int64_t laneStride = paritySplit ? 2 : 1;
                    const int64_t laneCount =
                        paritySplit ? (blockRows - parity + 1) / 2 : blockRows;
                    const int64_t rowAddressStride =
                        layout.ropeProductAddress(
                            kind, head, domainPairBlock, product,
                            parity + blockRows) -
                        baseAddress;
                    const int64_t laneAddressStride =
                        layout.ropeProductAddress(
                            kind, head, domainPairBlock, product,
                            parity + laneStride) -
                        baseAddress;
                    const int64_t groupAddressStride = pairBlockDomain
                        ? layout.ropeProductAddress(
                              kind, head, 1, product, parity) -
                              baseAddress
                        : layout.ropeProductAddress(
                              kind, head, pairBlock, product,
                              parity + tile) -
                              baseAddress;
                    for (int64_t destination = 0;
                         destination < memory.hemispheres; ++destination) {
                      const int64_t source = 1 - destination;
                      emitMem3D(
                          rewriter_, op_.getLoc(),
                          domainOutputCycle +
                              parity * target_.throughput().tile_rows + latency,
                          destination * memory.slices_per_hemisphere + slice,
                          "write", baseAddress,
                          source * 8 + slot * 2 + byte,
                          target_.throughput().tile_rows, 1, rowAddressStride,
                          "sram", -1, laneCount,
                          laneStride * target_.throughput().tile_rows,
                          laneAddressStride,
                          pairBlockDomain ? 2 : tokenBlocks,
                          pairBlockDomain
                              ? outputCycle - firstPairProductOutput[productPhase]
                              : tile,
                          groupAddressStride,
                          productStorageBank(product));
                    }
                  }
                }
              }
            };

            const int64_t productAConfig = headEnd;
            const int64_t productAInput = productAConfig + 1;
            const int64_t productBConfig =
                productAInput + op_.getSeqLen() +
                (productPipelineLatency - 1) +
                maxProductWriteLatency + maxStagingReadLatency + 1;
            const int64_t productBInput = productBConfig + 1;
            emitRopeProducts(productAConfig, inputHemisphere, outputHemisphere,
                             false);
            emitRopeTableDomain(productAInput, false);
            // Product A and Product B consume the same two bias vectors on
            // the same physical MEM read queues.  No other read instruction
            // targets those bank-0 slice queues between the two phases, so
            // make the phase axis native MEM3D work instead of emitting two
            // independent descriptors.
            emitProjectionBiasDomain(productAInput,
                                     productBInput - productAInput);
            emitProductWriteDomain(
                0, productAInput + productPipelineLatency);
            emitSourceDomain(0, 32, productAInput);
            emitSourceDomain(1, 36, productAInput);

            emitRopeProducts(productBConfig, inputHemisphere, outputHemisphere,
                             true);
            emitRopeTableDomain(productBInput, true);
            emitProductWriteDomain(
                2, productBInput + productPipelineLatency);
            emitSourceDomain(1, 32, productBInput);
            emitSourceDomain(0, 36, productBInput);

            const int64_t combineConfig =
                productBInput + op_.getSeqLen() +
                (productPipelineLatency - 1) +
                                           maxProductWriteLatency +
                                           maxProductReadLatency + 1;
            const int64_t combineInput = combineConfig + 1;
            if (pairBlockDomain && pairBlock == 0)
              firstPairCombineInput = combineInput;
            emitRopeCombine(combineConfig, inputHemisphere, outputHemisphere);
            for (int64_t product = 0; product < 4; ++product) {
              for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice0 = productSlice(product, 0, byte);
                const int64_t slice1 = op_.getSeqLen() > 1
                                           ? productSlice(product, 1, byte)
                                           : slice0;
                const bool paritySplit = slice0 != slice1;
                const int64_t parityCount = paritySplit ? 2 : 1;
                for (int64_t parity = 0; parity < parityCount; ++parity) {
                  const int64_t firstToken = paritySplit ? parity : 0;
                  const int64_t slice =
                      productSlice(product, firstToken, byte);
                  // Query's low products and all Key products use scratch
                  // read queues that are untouched between the two rotary
                  // pair-block combines.  Make pairBlock the outer MEM3D
                  // counter.  Query's high products share their queues with
                  // the next normal-table read and therefore stay split.
                  const bool pairCombineDomain =
                      pairBlockDomain &&
                      (kind == AttentionProjectionKind::Key || product < 2);
                  if (pairCombineDomain && pairBlock == 0)
                    continue;
                  const int64_t domainPairBlock =
                      pairCombineDomain ? 0 : pairBlock;
                  const int64_t domainCombineInput = pairCombineDomain
                      ? firstPairCombineInput
                      : combineInput;
                  const int64_t address = layout.ropeProductAddress(
                      kind, head, domainPairBlock, product, firstToken);
                  const int64_t laneStride = paritySplit ? 2 : 1;
                  const int64_t rowAddressStride =
                      layout.ropeProductAddress(
                          kind, head, domainPairBlock, product,
                          firstToken + blockRows) -
                      address;
                  const int64_t laneAddressStride =
                      layout.ropeProductAddress(
                          kind, head, domainPairBlock, product,
                          firstToken + laneStride) -
                      address;
                  const int64_t groupAddressStride = pairCombineDomain
                      ? layout.ropeProductAddress(
                            kind, head, 1, product, firstToken) -
                            address
                      : layout.ropeProductAddress(
                            kind, head, pairBlock, product,
                            firstToken + tile) -
                            address;
                  const int64_t laneCount =
                      paritySplit ? (blockRows - parity + 1) / 2 : blockRows;
                  for (int64_t hemisphere = 0;
                       hemisphere < memory.hemispheres; ++hemisphere)
                    emitMem3D(
                        rewriter_, op_.getLoc(),
                        domainCombineInput + firstToken * tileRows -
                            vxmInputReadLatency(slice),
                        hemisphere * memory.slices_per_hemisphere + slice,
                        "read", address,
                        32 + hemisphere * 16 + product * 2 + byte, tileRows,
                        1, rowAddressStride, "sram", -1, laneCount,
                        laneStride * tileRows, laneAddressStride,
                        pairCombineDomain ? 2 : tokenBlocks,
                        pairCombineDomain
                            ? combineInput - firstPairCombineInput
                            : tile,
                        groupAddressStride,
                        productStorageBank(product));
                }
              }
            }

            for (int64_t half = 0; half < 2; ++half) {
              const int64_t reductionBlock = blocks[half];
              const int64_t domainReductionBlock =
                  pairBlockDomain
                      ? half * (projectionHeadBlocks / 2)
                      : reductionBlock;
              for (int64_t byte = 0; byte < 2; ++byte) {
                if (kind == AttentionProjectionKind::Query) {
                  for (int64_t tokenLane = 0; tokenLane < blockRows;
                       ++tokenLane) {
                    const int64_t slice = layout.queryIwSlices(
                        domainReductionBlock)[2 * tokenLane + byte];
                    const int64_t latency =
                        target_
                            .transport_latency(
                                target::StreamEndpoint::VxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::East, slice)
                            .value_or(readLatency(slice));
                    const int64_t currentCycle =
                        combineInput + tokenLane * tileRows + 1 + latency;
                    if (!pairBlockDomain || pairBlock != 0) {
                      const int64_t domainCycle =
                          pairBlockDomain
                              ? firstPairCombineInput +
                                    tokenLane * tileRows + 1 + latency
                              : currentCycle;
                      const int64_t baseAddress = layout.queryIwAddress(
                          head, domainReductionBlock, 0, 0);
                      const int64_t pairAddressStride = pairBlockDomain
                          ? layout.queryIwAddress(
                                head, domainReductionBlock + 1, 0, 0) -
                                baseAddress
                          : 0;
                      for (int64_t destination = 0;
                           destination < memory.hemispheres; ++destination) {
                        const int64_t source = 1 - destination;
                        const int64_t queue =
                            destination * memory.slices_per_hemisphere + slice;
                        const int64_t bank =
                            layout.queryIwBank(domainReductionBlock);
                        const int64_t groupCount =
                            pairBlockDomain ? 2 : 1;
                        const int64_t groupInterval = pairBlockDomain
                            ? combineInput - firstPairCombineInput
                            : 1;
                        // Each pair block remains an independent closed
                        // descriptor.  Combining them into one sparse outer
                        // group would hold this MEM ICU across the 136-cycle
                        // gap and prevent activation ping/pong from using it.
                        emitMem3D(
                            rewriter_, op_.getLoc(), domainCycle, queue,
                            "write", baseAddress,
                            source * 8 + half * 2 + byte,
                            target_.throughput().tile_rows, 1, 1, "sram", -1,
                            tokenBlocks, tile,
                            target_.throughput().tile_rows,
                            groupCount, groupInterval, pairAddressStride,
                            bank);
                        recordRopeMemDomain(
                            queue, bank, domainCycle,
                            target_.throughput().tile_rows, 1,
                            tokenBlocks, tile,
                            groupCount, groupInterval);
                      }
                    }
                    headEnd = std::max(
                        headEnd,
                        currentCycle +
                            (target_.throughput().tile_rows - 1) +
                            (tokenBlocks - 1) * tile + 1);
                  }
                } else {
                  const int64_t slice =
                      layout.keySlices(domainReductionBlock)[byte];
                  const int64_t latency =
                      target_
                          .transport_latency(
                              target::StreamEndpoint::VxmResult,
                              target::StreamEndpoint::Mem,
                              target::StreamDirection::East, slice)
                          .value_or(readLatency(slice));
                  const int64_t currentCycle = combineInput + 1 + latency;
                  if (!pairBlockDomain || pairBlock != 0) {
                    const int64_t domainCycle = pairBlockDomain
                        ? firstPairCombineInput + 1 + latency
                        : currentCycle;
                    const int64_t baseAddress = layout.keyAddress(
                        head, domainReductionBlock, 0);
                    const int64_t groupAddressStride = pairBlockDomain
                        ? layout.keyAddress(
                              head, domainReductionBlock + 1, 0) -
                              baseAddress
                        : tile;
                    for (int64_t destination = 0;
                         destination < memory.hemispheres; ++destination) {
                      const int64_t source = 1 - destination;
                      emitMem3D(
                          rewriter_, op_.getLoc(), domainCycle,
                          destination * memory.slices_per_hemisphere + slice,
                          "write", baseAddress,
                          source * 8 + half * 2 + byte, tileRows, 1, blockRows,
                          "sram", -1, blockRows, tileRows, 1,
                          pairBlockDomain ? 2 : tokenBlocks,
                          pairBlockDomain
                              ? combineInput - firstPairCombineInput
                              : tile,
                          groupAddressStride,
                          layout.keyBank(domainReductionBlock));
                    }
                  }
                  headEnd = std::max(headEnd,
                                     currentCycle + op_.getSeqLen());
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
        previousRopeMemUses = std::move(currentRopeMemUses);
        if (!projectionCanOverlap) {
          phaseStart = std::max(phaseStart, postprocessReady);
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
      const int64_t phases = tile / 8;
      const int64_t firstSourcePhase = phases - 1;
      const int64_t firstReductionIw = reductionIwCycle(
          firstIwCycle, firstComputeCycle, work->local_mxm, 0);
      const int64_t iwReductionInterval =
          headBlocks > 1
              ? reductionIwCycle(firstIwCycle, firstComputeCycle,
                                 work->local_mxm, 1) -
                    firstReductionIw
              : 1;
      const int64_t blocksPerRotaryHalf =
          std::max<int64_t>(1, headBlocks / 2);
      for (int64_t reductionBase = 0; reductionBase < headBlocks;
           reductionBase += blocksPerRotaryHalf) {
        const int64_t reductionCount = std::min<int64_t>(
            blocksPerRotaryHalf, headBlocks - reductionBase);
        const auto iwSlices = layout.queryIwSlices(reductionBase);
        const int64_t reductionIw = firstReductionIw +
                                    reductionBase * iwReductionInterval;
        const int64_t queryAddress = layout.queryIwAddress(
            work->query_head, reductionBase, work->query_block,
            firstSourcePhase);
        const int64_t reductionAddressStride =
            reductionCount > 1
                ? layout.queryIwAddress(work->query_head, reductionBase + 1,
                                        work->query_block, firstSourcePhase) -
                      queryAddress
                : 0;
        for (int64_t stream = 0;
             stream < static_cast<int64_t>(iwSlices.size()); ++stream) {
          const int64_t slice = iwSlices[stream];
          const int64_t latency = *target_.transport_latency(
              target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
              target::StreamDirection::East, slice);
          emitMem3D(
              rewriter_, op_.getLoc(), reductionIw - latency,
              work->hemisphere * target_.memory().slices_per_hemisphere + slice,
              "read", queryAddress,
              work->local_mxm * static_cast<int64_t>(iwSlices.size()) + stream,
              phases, 1, -1, "sram", -1, reductionCount,
              iwReductionInterval, reductionAddressStride, 1, 1, 0,
              layout.queryIwBank(reductionBase));
        }
      }

      if (target_.throughput().mxm_weight_buffers <= 2) {
        MxmDomain3D iwDomain;
        iwDomain.wave_count = phases;
        iwDomain.wave_interval = 1;
        iwDomain.wave_weight_column_stride = -1;
        iwDomain.group_count = headBlocks;
        iwDomain.group_interval = iwReductionInterval;
        if (headBlocks > 1 && target_.throughput().mxm_weight_buffers == 2)
          iwDomain.weight_buffer_mode = "toggle_dim2";
        emitMxm3D(
            rewriter_, op_.getLoc(), firstReductionIw, mxm, "iw", 0,
            firstSourcePhase, 0, 0, 0, 1, "stream", true, "supercell", 0,
            dataFormat, llvm::StringRef{}, llvm::StringRef{}, iwDomain);
      } else {
        for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
          MxmDomain3D iwDomain;
          iwDomain.wave_count = phases;
          iwDomain.wave_interval = 1;
          iwDomain.wave_weight_column_stride = -1;
          emitMxm3D(
              rewriter_, op_.getLoc(),
              firstReductionIw + reduction * iwReductionInterval, mxm, "iw",
              reduction % target_.throughput().mxm_weight_buffers,
              firstSourcePhase, 0, 0, 0, 1, "stream", true, "supercell", 0,
              dataFormat, llvm::StringRef{}, llvm::StringRef{}, iwDomain);
        }
      }
      const int64_t computeReductionInterval = tokenBlocks * issue;
      const int64_t keyReductionGroup =
          target_.throughput().mxms_per_hemisphere == 1
              ? blocksPerRotaryHalf
              : 1;
      for (int64_t reduction = 0; reduction < headBlocks;
           reduction += keyReductionGroup) {
        const int64_t reductionCount =
            std::min<int64_t>(keyReductionGroup, headBlocks - reduction);
        const int64_t computeCycle = firstComputeCycle +
                                     reduction * computeReductionInterval;
        for (int64_t byte = 0; byte < 2; ++byte) {
          const int64_t slice =
              target_.throughput().mxms_per_hemisphere == 1
                  ? layout.keySlices(reduction)[byte]
                  : work->local_mxm * 4 + (reduction % 2) * 2 + byte;
          const int64_t latency = *target_.transport_latency(
              target::StreamEndpoint::Mem,
              target::StreamEndpoint::MxmActivation,
              target::StreamDirection::East, slice);
          emitMem3D(
              rewriter_, op_.getLoc(), computeCycle - latency,
              work->hemisphere * target_.memory().slices_per_hemisphere + slice,
              "read", layout.keyAddress(work->kv_head, reduction, 0),
              activationStream + byte, tile, 1, 1, "sram", -1, tokenBlocks,
              issue, tile, reductionCount, computeReductionInterval,
              reductionCount > 1
                  ? layout.keyAddress(work->kv_head, reduction + 1, 0) -
                        layout.keyAddress(work->kv_head, reduction, 0)
                  : 0,
              layout.keyBank(reduction));
        }
      }

      if (target_.throughput().mxm_weight_buffers <= 2) {
        MxmDomain3D computeDomain;
        computeDomain.repeat_count = tile;
        computeDomain.repeat_accumulator_address_stride = 0;
        computeDomain.wave_count = tokenBlocks;
        computeDomain.wave_interval = issue;
        computeDomain.wave_accumulator_address_stride = tile;
        computeDomain.group_count = headBlocks;
        computeDomain.group_interval = computeReductionInterval;
        if (headBlocks > 1 && target_.throughput().mxm_weight_buffers == 2)
          computeDomain.weight_buffer_mode = "toggle_dim2";
        if (headBlocks > 1) {
          computeDomain.terminal_dimension = 2;
          computeDomain.terminal_accumulator_destination = "stream";
          computeDomain.terminal_accumulator_clear = true;
          computeDomain.terminal_accumulator_output_format = dataFormat;
        }
        emitMxm3D(
            rewriter_, op_.getLoc(), firstComputeCycle, mxm, "compute", 0, 0,
            activationStream, outputStream,
            layout.scoreAccumulatorAddress(work->query_head,
                                           work->query_block, 0),
            1, headBlocks == 1 ? "stream" : "sram", headBlocks == 1,
            "supercell", 0, dataFormat, llvm::StringRef{},
            headBlocks == 1 ? dataFormat : "fp32", computeDomain);
      } else {
        for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
          const bool finalReduction = reduction + 1 == headBlocks;
          MxmDomain3D computeDomain;
          computeDomain.repeat_count = tile;
          computeDomain.repeat_accumulator_address_stride = 0;
          computeDomain.wave_count = tokenBlocks;
          computeDomain.wave_interval = issue;
          computeDomain.wave_accumulator_address_stride = tile;
          emitMxm3D(
              rewriter_, op_.getLoc(),
              firstComputeCycle + reduction * computeReductionInterval, mxm,
              "compute", reduction % target_.throughput().mxm_weight_buffers,
              0, activationStream, outputStream,
              layout.scoreAccumulatorAddress(work->query_head,
                                             work->query_block, 0),
              1, finalReduction ? "stream" : "sram", finalReduction,
              "supercell", 0, dataFormat, llvm::StringRef{},
              finalReduction ? dataFormat : "fp32", computeDomain);
        }
      }

      if (!fusedSoftmax) {
        const int64_t finalComputeCycle =
            firstComputeCycle + (headBlocks - 1) * computeReductionInterval;
        // Tail softmax consumes compact 16-bit scores. The final partial
        // converts in the accumulator and clears the row while streaming.
        for (int64_t byte = 0; byte < 2; ++byte) {
          const int64_t slice =
              layout.scaledScoreSlices(work->local_mxm)[byte];
          const auto latency = target_.transport_latency(
              target::StreamEndpoint::MxmResult,
              target::StreamEndpoint::Mem, target::StreamDirection::West,
              slice);
          if (!latency)
            continue;
          emitMem3D(
              rewriter_, op_.getLoc(),
              finalComputeCycle + target_.mxm_first_result_latency() + *latency,
              work->hemisphere * target_.memory().slices_per_hemisphere + slice,
              "write",
              layout.scoreAddress(work->query_head, work->query_block, 0),
              32 + outputStream + byte, tile, 1, 1, "sram", -1, tokenBlocks,
              issue, tile, 1, 1, 0,
              placementBank(work->local_mxm == 0 ? "score" : "score_mxm1"));
        }
      }
    }
  }
  return mlir::success();
}

} // namespace ftlpu::compiler::schedule
