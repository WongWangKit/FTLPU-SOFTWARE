#include "FfnStageEmitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_block8_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/stream_fabric_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace ftlpu::compiler::schedule::ffn_detail {
namespace {

int64_t functionArgumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    if (auto binding = value.getDefiningOp<BindingOp>())
        return binding.getIndex();
    return -1;
}

BindingOp createInputBinding(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::Value value, int64_t index,
    llvm::StringRef role, mlir::DictionaryAttr placement)
{
    const auto type = llvm::cast<mlir::RankedTensorType>(value.getType());
    const int64_t bytesPerElement =
        type.getElementType().isInteger(8) ? 1 : 2;
    mlir::OperationState state(location, BindingOp::getOperationName());
    state.addOperands(value);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("input")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * bytesPerElement)),
        rewriter.getNamedAttr("placement", placement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

std::pair<VxmOp, VxmOp> emitScratchSwish(
    FfnEmissionContext& context, int64_t cycle, int64_t repeatCount = 1,
    int64_t repeatInterval = 1)
{
    using attention_detail::emitVxm;
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    constexpr llvm::StringLiteral east = "east";
    constexpr llvm::StringLiteral west = "west";
    const auto dataFormat = lpu_16bit_data_format(
        llvm::cast<mlir::RankedTensorType>(
            ffn.getActivation().getType()).getElementType());
    mlir::Value value = ffn.getActivation();
    const auto repeated = [&](VxmOp op) {
        op.setRepeatCountAttr(rewriter.getI64IntegerAttr(repeatCount));
        op.setRepeatIntervalAttr(
            rewriter.getI64IntegerAttr(repeatInterval));
        return op;
    };
    const int64_t configCycle = cycle - 1;

    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 0,
        "negate", lpu_16bit_stream_kind(
            llvm::cast<mlir::RankedTensorType>(
                ffn.getActivation().getType()).getElementType()),
        0, 0.0f, lpu_16bit_stream_kind(
            llvm::cast<mlir::RankedTensorType>(
                ffn.getActivation().getType()).getElementType()),
        2, 0.0f, "fp32", -1, east, east)).getResult();
    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 1,
        "exp", "previous", 0, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, east, east)).getResult();
    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 2,
        "add", "previous", 0, 0.0f, "immediate", 0, 1.0f,
        "fp32", -1, east, east)).getResult();
    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 3,
        "reciprocal", "previous", 0, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, east, east)).getResult();
    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 4,
        "multiply", "previous", 0, 0.0f, "original", 0, 0.0f,
        "fp32", -1, east, east)).getResult();
    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 5,
        "multiply", "previous", 0, 0.0f, "auxiliary", 0, 0.0f,
        "fp32", -1, east, east)).getResult();
    value = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 6,
        "bypass", "previous", 0, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, east, east)).getResult();
    auto tail = repeated(emitVxm(rewriter, ffn.getLoc(), value, configCycle, 7,
        "bypass", "previous", 0, 0.0f, "immediate", 0, 0.0f,
        dataFormat, 6, east, west));
    return {tail, tail};
}

constexpr int64_t kVxmSwishLatency = 17;

} // namespace

mlir::FailureOr<FfnSwishEmission> emitFfnBlock8ProjectionAndSwish(
    FfnEmissionContext& context)
{
    using attention_detail::emitMem;
    using attention_detail::emitMemWave;
    using attention_detail::emitMxm;
    using attention_detail::emitMxmWave;
    using attention_detail::emitMxmDequant;
    using attention_detail::emitMxmDequantWave;
    using attention_detail::emitVxm;

    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t blockRows = throughput.mxm_block_rows;
    const int64_t blockIssues = tile / blockRows;
    const int64_t tokenBlocks = context.m() / tile;
    const int64_t reductionBlocks = context.k() / tile;
    const int64_t pairCount =
        context.hidden() / (memory.hemispheres * tile);
    const int64_t hiddenBlocks = context.hidden() / tile;
    if (blockRows != 8 || blockIssues != 4
        || context.activation_slices.size() != 16
        || context.hidden_slices.size() != 16
        || context.weight_slices.size() != 8
        || context.up_weight_slices.size() != 8
        || (throughput.mxms_per_hemisphere != 1
            && throughput.mxms_per_hemisphere != 2)
        || throughput.mxm_weight_buffers < 2)
        return mlir::failure();

    // Projection scratch and post-Swish hidden storage have disjoint row
    // ranges and lifetimes. Reuse the Tensor IR placement instead of guessing
    // spare slices: a live weight slice cannot be treated as deferred storage.
    const auto gateScratchSlices = context.hidden_slices;
    const int64_t gateScratchBank = 0;
    const int64_t upScratchBank = memory.banks_per_slice > 1 ? 1 : 0;
    if (upScratchBank == gateScratchBank) {
        ffn.add.emitError(
            "Block8 FFN projection scratch requires two SRAM banks");
        return mlir::failure();
    }
    const auto upScratchSlices = context.hidden_slices;

    const mlir::Value activation = context.activation_route.getInput();
    const mlir::Value gateWeight = context.gate_raw.getInput();
    const mlir::Value upWeight = context.up_raw.getInput();
    const int64_t activationBinding = functionArgumentIndex(activation);
    const int64_t gateBinding = functionArgumentIndex(gateWeight);
    const int64_t upBinding = functionArgumentIndex(upWeight);
    if (gateBinding < 0 || upBinding < 0) {
        ffn.add.emitError(
            "Block8 FFN projection weights must be function arguments or bindings");
        return mlir::failure();
    }

    rewriter.setInsertionPoint(ffn.getOperation());
    if (activationBinding >= 0)
        createInputBinding(rewriter, ffn.getLoc(), activation,
            activationBinding, "activation",
            context.activation_route.getPlacement());
    createInputBinding(rewriter, ffn.getLoc(), gateWeight,
        gateBinding, "weight", context.gate_raw.getPlacement());
    createInputBinding(rewriter, ffn.getLoc(), upWeight,
        upBinding, "weight", context.up_raw.getPlacement());

    const auto inputType = llvm::cast<mlir::RankedTensorType>(
        activation.getType());
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(inputType.getElementType());
    const int64_t activationBase =
        get_base_row(context.activation_route.getPlacement());
    const int64_t gateWeightBase =
        get_base_row(context.gate_raw.getPlacement());
    const int64_t upWeightBase =
        get_base_row(context.up_raw.getPlacement());
    const int64_t hiddenBase = get_base_row(ffn.getHidden0Placement());
    const int64_t hiddenRows =
        context.m() * context.hidden() / (tile * blockRows);
    const auto placementEnd = [](mlir::DictionaryAttr placement) {
        return get_base_row(placement)
            + placement.getAs<mlir::IntegerAttr>(
                  "instruction_count").getInt();
    };
    const auto placementBank = [](mlir::DictionaryAttr placement) {
        const auto bank = placement.getAs<mlir::IntegerAttr>("bank");
        return bank ? bank.getInt() : int64_t {0};
    };
    const auto scratchLiveEnd = [&](mlir::DictionaryAttr placement) {
        const auto slices = get_slices(placement);
        const auto overlaps = [&](llvm::ArrayRef<int64_t> scratchSlices) {
            return llvm::any_of(slices, [&](int64_t slice) {
                return llvm::is_contained(scratchSlices, slice);
            });
        };
        const int64_t bank = placementBank(placement);
        const bool conflicts =
            (bank == gateScratchBank && overlaps(gateScratchSlices))
            || (bank == upScratchBank && overlaps(upScratchSlices));
        return conflicts ? placementEnd(placement) : int64_t {0};
    };
    const int64_t scratchBase = std::max({hiddenBase + hiddenRows,
        scratchLiveEnd(context.activation_route.getPlacement()),
        scratchLiveEnd(context.gate_raw.getPlacement()),
        scratchLiveEnd(context.up_raw.getPlacement()),
        scratchLiveEnd(context.down_raw.getPlacement())});
    const int64_t scratchRowsPerPair = tokenBlocks * blockIssues;
    const int64_t requiredScratchEnd =
        scratchBase + pairCount * scratchRowsPerPair;
    if (requiredScratchEnd > memory.sram_depth_rows) {
        ffn.add.emitError(
            "Block8 FFN projection scratch requires row ")
            << requiredScratchEnd << " but target provides "
            << memory.sram_depth_rows << " rows per bank";
        return mlir::failure();
    }
    int64_t maxWeightLatency = 0;
    llvm::SmallVector<int64_t> allWeightSlices(
        context.weight_slices.begin(), context.weight_slices.end());
    allWeightSlices.append(context.up_weight_slices.begin(),
        context.up_weight_slices.end());
    for (int64_t slice : allWeightSlices) {
        const auto latency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        if (!latency) return mlir::failure();
        maxWeightLatency = std::max(maxWeightLatency, *latency);
    }
    const auto areDisjoint = [](llvm::ArrayRef<int64_t> lhs,
                                llvm::ArrayRef<int64_t> rhs) {
        return llvm::none_of(lhs, [&](int64_t value) {
            return llvm::is_contained(rhs, value);
        });
    };
    const bool wavefrontWeightLoads =
        throughput.mxm_weight_activation_overlap_enabled != 0
        && areDisjoint(context.weight_slices,
            context.up_weight_slices)
        && areDisjoint(context.weight_slices,
            context.activation_slices)
        && areDisjoint(context.up_weight_slices,
            context.activation_slices)
        && target.streams().streams_per_direction >= 32;
    const bool fused = context.strategy == FfnScheduleStrategy::Fused
        && context.fused_output.has_value();
    int64_t phaseStart = maxWeightLatency + 1;
    mlir::Value lastHidden = activation;
    int64_t lastSwishCycle = 0;
    int64_t projectionDrainEnd = phaseStart;
    llvm::SmallVector<int64_t> pairDrainEnds;
    pairDrainEnds.reserve(pairCount);
    const auto memResource = [](llvm::StringRef access,
                                int64_t hemi, int64_t slice,
                                int64_t bank) {
        (void)access;
        return "ffn.block8.mem." + std::to_string(hemi)
            + "." + std::to_string(slice)
            + ".bank" + std::to_string(bank);
    };
    const int64_t activationBank =
        placementBank(context.activation_route.getPlacement());
    const int64_t hiddenBank = placementBank(ffn.getHidden0Placement());
    ResourceScheduler memScheduler;
    // Pair-level compaction must repeat the complete reduction window. A
    // per-command outer repeat changes the accumulator order from
    // pair-major to reduction-major and corrupts sparse projections.
    const bool compactPairWaves = false;
    const int64_t pairGroupCount = compactPairWaves ? pairCount : 1;
    int64_t pairGroupInterval = reductionBlocks * tokenBlocks
        * throughput.mxm_block_group_interval;
    const int64_t emittedPairCount = compactPairWaves ? 1 : pairCount;
    llvm::SmallVector<mlir::Operation*> compactWaveOps;
    const auto rememberCompactWave = [&]() {
        if (compactPairWaves)
            compactWaveOps.push_back(ffn.getOperation()->getPrevNode());
    };
    for (int64_t pair = 0; pair < emittedPairCount; ++pair) {
        int64_t pairDrainEnd = phaseStart;
        if (throughput.mxms_per_hemisphere == 1) {
            const int64_t rowBlocks = context.m() / blockRows;
            const auto emitWeight = [&](int64_t projection,
                                        int64_t reduction,
                                        int64_t loadStart,
                                        int64_t weightBuffer)
                -> mlir::LogicalResult {
                const int64_t weightBase = projection == 0
                    ? gateWeightBase : upWeightBase;
                const auto& weightSlices = projection == 0
                    ? context.weight_slices : context.up_weight_slices;
                const int64_t binding = projection == 0
                    ? gateBinding : upBinding;
                const int64_t weightBank = placementBank(
                    projection == 0 ? context.gate_raw.getPlacement()
                                    : context.up_raw.getPlacement());
                const float scale = projection == 0
                    ? ffn.getGateScale().convertToFloat()
                    : ffn.getUpScale().convertToFloat();
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    const int64_t unit = hemisphere;
                    const int64_t firstAddress = weightBase
                        + (pair * reductionBlocks + reduction)
                            * throughput.tile_rows
                        + throughput.tile_rows - 1;
                    for (int64_t stream = 0; stream < 8; ++stream) {
                        const int64_t slice = weightSlices[stream];
                        const auto latency = target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmWeight,
                            target::StreamDirection::East, slice);
                        if (!latency) return mlir::failure();
                        emitMemWave(rewriter, ffn.getLoc(),
                            loadStart - *latency,
                            hemisphere * memory.slices_per_hemisphere
                                + slice,
                            "read", firstAddress, stream,
                            throughput.tile_rows, 1, -1,
                            "sram", binding, pairGroupCount,
                            pairGroupInterval,
                            reductionBlocks * throughput.tile_rows,
                            weightBank);
                        rememberCompactWave();
                        memScheduler.reserve_at(loadStart - *latency,
                            {{memResource("read", hemisphere, slice,
                                  weightBank),
                                0, throughput.tile_rows}});
                    }
                    emitMxmDequantWave(rewriter, ffn.getLoc(),
                        loadStart, unit, scale,
                        throughput.tile_rows, 1,
                        pairGroupCount, pairGroupInterval, binding);
                    rememberCompactWave();
                    emitMxmWave(rewriter, ffn.getLoc(), loadStart, unit,
                        "iw", weightBuffer, 0,
                        0, 0, 1, 1, 0, 1, "sram", true,
                        "supercell", 0, dataFormat,
                        "int8_dequant_bf16", {}, {},
                        throughput.tile_rows, 1, 1,
                        pairGroupCount, pairGroupInterval);
                    rememberCompactWave();
                }
                return mlir::success();
            };

            const auto emitCompute = [&](int64_t projection,
                                         int64_t reduction,
                                         int64_t firstCompute,
                                         int64_t weightBuffer)
                -> mlir::LogicalResult {
                const bool finalReduction =
                    reduction + 1 == reductionBlocks;
                for (int64_t tokenBlock = 0;
                     tokenBlock < tokenBlocks; ++tokenBlock) {
                    const int64_t computeCycle = firstCompute
                        + tokenBlock
                            * throughput.mxm_block_group_interval;
                    const bool overlapsWeightLoad = projection == 0
                        || (tokenBlock + 1 == tokenBlocks
                            && (reduction + 1 < reductionBlocks
                                || pair + 1 < pairCount));
                    const int64_t activationStreamBase =
                        context.strategy == FfnScheduleStrategy::Fused
                            || overlapsWeightLoad
                        ? 2 * blockRows : 0;
                    const int64_t activationAddress = activationBase
                        + (tokenBlock * reductionBlocks + reduction)
                            * blockIssues;
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres; ++hemisphere) {
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice =
                                context.activation_slices[stream];
                            const auto latency = target.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmActivation,
                                target::StreamDirection::East, slice);
                            if (!latency) return mlir::failure();
                            emitMemWave(rewriter, ffn.getLoc(),
                                computeCycle - *latency,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read", activationAddress,
                                activationStreamBase + stream,
                                blockIssues, 1, 1,
                                "sram", activationBinding,
                                pairGroupCount, pairGroupInterval, 0);
                            rememberCompactWave();
                            memScheduler.reserve_at(computeCycle - *latency,
                                {{memResource("read", hemisphere, slice,
                                      activationBank),
                                    0, blockIssues}});
                        }
                        const int64_t accumulatorBase =
                            projection * rowBlocks
                            + tokenBlock * blockIssues;
                        emitMxmWave(rewriter, ffn.getLoc(), computeCycle,
                            hemisphere, "compute", weightBuffer, 0,
                            activationStreamBase, 0,
                            blockIssues, 1, accumulatorBase, 1,
                            finalReduction ? "stream" : "sram",
                            finalReduction,
                            "supercell", 0, dataFormat, {}, "block8",
                            finalReduction ? dataFormat
                                           : llvm::StringRef {},
                            1, 1, 0,
                            pairGroupCount, pairGroupInterval);
                        rememberCompactWave();
                        if (!finalReduction) continue;

                        const auto& slices = projection == 0
                            ? gateScratchSlices : upScratchSlices;
                        const int64_t scratchBank = projection == 0
                            ? gateScratchBank : upScratchBank;
                        const int64_t scratchAddress = scratchBase
                            + pair * scratchRowsPerPair
                            + tokenBlock * blockIssues;
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice = slices[stream];
                            const auto latency = target.transport_latency(
                                target::StreamEndpoint::MxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::West, slice);
                            if (!latency) return mlir::failure();
                            const int64_t writeCycle = computeCycle
                                + throughput.accumulator_to_vxm_latency
                                + *latency;
                            emitMemWave(rewriter, ffn.getLoc(), writeCycle,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", scratchAddress,
                                target.streams().streams_per_direction
                                    + stream,
                                blockIssues, 1, 1, "sram", -1,
                                pairGroupCount, pairGroupInterval,
                                scratchRowsPerPair, scratchBank);
                            rememberCompactWave();
                            memScheduler.reserve_at(writeCycle,
                                {{memResource("write", hemisphere, slice,
                                      scratchBank),
                                    0, blockIssues}});
                            pairDrainEnd = std::max(pairDrainEnd,
                                writeCycle + blockIssues);
                        }
                    }
                }
                return mlir::success();
            };

            const int64_t firstGateCompute = phaseStart
                + throughput.mxm_local_load_to_compute_latency;
            for (int64_t reduction = 0;
                 reduction < reductionBlocks; ++reduction) {
                const int64_t gateCompute = firstGateCompute
                    + reduction * tokenBlocks
                        * throughput.mxm_block_group_interval;
                const int64_t upCompute = gateCompute + blockIssues;
                if (mlir::failed(emitWeight(
                        0, reduction,
                        gateCompute
                            - throughput.mxm_local_load_to_compute_latency,
                        0))
                    || mlir::failed(emitWeight(
                        1, reduction,
                        upCompute
                            - throughput.mxm_local_load_to_compute_latency,
                        1))
                    || mlir::failed(emitCompute(
                        0, reduction, gateCompute, 0))
                    || mlir::failed(emitCompute(
                        1, reduction, upCompute, 1)))
                    return mlir::failure();
            }
            if (compactPairWaves) {
                const int64_t nominalInterval = pairGroupInterval;
                pairGroupInterval =
                    memScheduler.minimum_non_overlapping_shift(
                        nominalInterval);
                if (pairGroupInterval != nominalInterval) {
                    for (mlir::Operation* operation : compactWaveOps) {
                        if (operation->hasAttr("group_interval"))
                            operation->setAttr("group_interval",
                                rewriter.getI64IntegerAttr(
                                    pairGroupInterval));
                        else if (operation->hasAttr("wave_interval"))
                            operation->setAttr("wave_interval",
                                rewriter.getI64IntegerAttr(
                                    pairGroupInterval));
                    }
                }
            }
            const int64_t pairComputeEnd = firstGateCompute
                + reductionBlocks * tokenBlocks
                    * throughput.mxm_block_group_interval;
            phaseStart = std::max(
                pairComputeEnd
                    - throughput.mxm_local_load_to_compute_latency
                    + (pairGroupCount - 1) * pairGroupInterval,
                pairDrainEnd + maxWeightLatency);
            pairDrainEnd +=
                (pairGroupCount - 1) * pairGroupInterval;
            projectionDrainEnd = std::max(
                projectionDrainEnd, pairDrainEnd);
            pairDrainEnds.push_back(pairDrainEnd);
            continue;
        }

        const auto emitReductionLoad = [&](int64_t reduction,
                                           int64_t loadStart) {
            const int64_t weightBuffer = reduction
                % throughput.mxm_weight_buffers;
            for (int64_t hemisphere = 0;
                 hemisphere < memory.hemispheres; ++hemisphere) {
                for (int64_t localMxm = 0;
                     localMxm < throughput.mxms_per_hemisphere;
                     ++localMxm) {
                    const int64_t unit = hemisphere
                            * throughput.mxms_per_hemisphere
                        + localMxm;
                    const int64_t localStart = loadStart
                        + (wavefrontWeightLoads
                                ? 0
                                : localMxm * throughput.tile_rows);
                    const int64_t weightBase = localMxm == 0
                        ? gateWeightBase : upWeightBase;
                    const auto& weightSlices = localMxm == 0
                        ? context.weight_slices
                        : context.up_weight_slices;
                    const int64_t binding = localMxm == 0
                        ? gateBinding : upBinding;
                    const int64_t weightBank = placementBank(
                        localMxm == 0 ? context.gate_raw.getPlacement()
                                      : context.up_raw.getPlacement());
                    const float scale = localMxm == 0
                        ? ffn.getGateScale().convertToFloat()
                        : ffn.getUpScale().convertToFloat();
                    for (int64_t pulse = 0;
                         pulse < throughput.tile_rows; ++pulse) {
                        const int64_t cycle = localStart + pulse;
                        const int64_t physicalColumn = wavefrontWeightLoads
                            ? pulse
                            : throughput.tile_rows - 1 - pulse;
                        const int64_t addressPulse = wavefrontWeightLoads
                            ? throughput.tile_rows - 1 - pulse
                            : pulse;
                        const int64_t address = weightBase
                            + (pair * reductionBlocks + reduction)
                                * throughput.tile_rows
                            + addressPulse;
                        for (int64_t stream = 0; stream < 8; ++stream) {
                            const int64_t slice = weightSlices[stream];
                            const auto latency = target.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmWeight,
                                target::StreamDirection::East, slice);
                            if (!latency) return mlir::failure();
                            emitMem(rewriter, ffn.getLoc(),
                                cycle - *latency,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read", address,
                                localMxm
                                        * throughput
                                              .mxm_int8_load_streams_per_cycle
                                    + stream,
                                1, 1, 0, "sram", binding, weightBank);
                            memScheduler.reserve_at(cycle - *latency,
                                {{memResource("read", hemisphere, slice,
                                      weightBank),
                                    0, 1}});
                        }
                        emitMxmDequant(rewriter, ffn.getLoc(),
                            cycle, unit, scale, 1, 1, binding);
                        emitMxm(rewriter, ffn.getLoc(), cycle, unit,
                            "iw", weightBuffer,
                            physicalColumn,
                            0, 0, 1, 1, 0, 1, "sram", true,
                            "supercell", 0, dataFormat,
                            "int8_dequant_bf16");
                    }
                }
            }
            return mlir::success();
        };

        const auto emitReductionCompute = [&](
            const FfnBlock8ReductionSchedule& schedule)
            -> mlir::LogicalResult {
            const int64_t reduction = schedule.reduction;
            const bool finalReduction =
                reduction + 1 == reductionBlocks;
            for (int64_t tokenBlock = 0;
                 tokenBlock < tokenBlocks; ++tokenBlock) {
                const int64_t computeCycle =
                    schedule.compute_cycles[tokenBlock];
                const bool preloadsNextOutputPair = finalReduction
                    && tokenBlock + 1 == tokenBlocks
                    && pair + 1 < pairCount;
                const int64_t activationStreamBase =
                    preloadsNextOutputPair
                    ? 2 * blockRows
                    : schedule.activation_stream_bases[tokenBlock];
                const int64_t address = activationBase
                    + (tokenBlock * reductionBlocks + reduction)
                        * blockIssues;
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    for (int64_t stream = 0; stream < 16; ++stream) {
                        const int64_t slice =
                            context.activation_slices[stream];
                        const auto latency = target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmActivation,
                            target::StreamDirection::East, slice);
                        if (!latency) return mlir::failure();
                        emitMem(rewriter, ffn.getLoc(),
                            computeCycle - *latency,
                            hemisphere * memory.slices_per_hemisphere
                                + slice,
                            "read", address,
                            activationStreamBase + stream,
                            blockIssues, 1, 1,
                            "sram", activationBinding, activationBank);
                        memScheduler.reserve_at(computeCycle - *latency,
                            {{memResource("read", hemisphere, slice,
                                  activationBank),
                                0, blockIssues}});
                    }
                    for (int64_t localMxm = 0;
                         localMxm < throughput.mxms_per_hemisphere;
                         ++localMxm) {
                        const int64_t outputStreamBase =
                            localMxm * blockRows * 2;
                        emitMxm(rewriter, ffn.getLoc(), computeCycle,
                            hemisphere
                                    * throughput.mxms_per_hemisphere
                                + localMxm,
                            "compute", schedule.weight_buffer, 0,
                            activationStreamBase, outputStreamBase,
                            blockIssues, 1,
                            tokenBlock * blockIssues,
                            1, finalReduction ? "stream" : "sram",
                            finalReduction, "supercell", 0,
                            dataFormat, {}, "block8",
                            finalReduction ? dataFormat
                                           : llvm::StringRef {});
                        if (!finalReduction) continue;

                        const auto& slices = localMxm == 0
                            ? gateScratchSlices : upScratchSlices;
                        const int64_t scratchBank = localMxm == 0
                            ? gateScratchBank : upScratchBank;
                        const int64_t scratchAddress = scratchBase
                            + pair * scratchRowsPerPair
                            + tokenBlock * blockIssues;
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice = slices[stream];
                            const auto latency = target.transport_latency(
                                target::StreamEndpoint::MxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::West, slice);
                            if (!latency) return mlir::failure();
                            // The final partial leaves through the BF16
                            // accumulator stream path, whose latency differs
                            // from a normal MXM result.
                            const int64_t writeCycle =
                                computeCycle
                                + throughput.accumulator_to_vxm_latency
                                + *latency;
                            emitMem(rewriter, ffn.getLoc(), writeCycle,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", scratchAddress,
                                target.streams().streams_per_direction
                                    + outputStreamBase + stream,
                                blockIssues, 1, 1,
                                "sram", -1, scratchBank);
                            memScheduler.reserve_at(writeCycle,
                                {{memResource("write", hemisphere, slice,
                                      scratchBank),
                                    0, blockIssues}});
                            pairDrainEnd = std::max(pairDrainEnd,
                                writeCycle + blockIssues);
                        }
                    }
                }
            }
            return mlir::success();
        };

        auto projectionSchedule = planFfnBlock8ProjectionSchedule(
            reductionBlocks, tokenBlocks, phaseStart,
            context.weight_slices, context.up_weight_slices,
            context.activation_slices, target);
        if (mlir::failed(projectionSchedule)) return mlir::failure();
        for (const auto& reduction : projectionSchedule->reductions) {
            if (mlir::failed(emitReductionLoad(
                    reduction.reduction, reduction.load_cycle))
                || mlir::failed(emitReductionCompute(reduction)))
                return mlir::failure();
        }
        // Direct ACC output clears at the MXM boundary. Scratch writes may
        // continue westward while the next output pair loads its first
        // weights; waiting for MEM completion created a false 30-cycle gap.
        phaseStart = projectionSchedule->end_cycle
            - throughput.mxm_local_load_to_compute_latency;
        projectionDrainEnd = std::max(
            projectionDrainEnd, pairDrainEnd);
        pairDrainEnds.push_back(pairDrainEnd);
    }

    int64_t maxScratchReadLatency = 0;
    for (int64_t slice : gateScratchSlices)
        maxScratchReadLatency = std::max(
            maxScratchReadLatency, context.westLatency(slice));
    for (int64_t slice : upScratchSlices)
        maxScratchReadLatency = std::max(
            maxScratchReadLatency, context.westLatency(slice));
    int64_t swishCycle = (fused ? pairDrainEnds.front()
                                : projectionDrainEnd)
        + maxScratchReadLatency + 1;
    // A token-block boundary may need one issue slot per hemisphere while
    // the MEM read/write ports switch to the next distributed address wave.
    // Include those slots in the outer pair repeat so repeated VXM programs
    // cannot overlap the tail of the representative pair.
    mlir::Value localSwish = activation;
    mlir::Value peerSwish = activation;
    for (int64_t pair = 0; pair < pairCount; ++pair) {
        if (fused)
            swishCycle = std::max(swishCycle,
                pairDrainEnds[pair] + maxScratchReadLatency + 1);
        for (int64_t tokenBlock = 0;
             tokenBlock < tokenBlocks; ++tokenBlock) {
            llvm::SmallVector<ResourceWindow> blockMemWindows;
            for (int64_t row = 0; row < tile; ++row) {
                const int64_t tokenLane = row % blockRows;
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t gateSlice =
                            gateScratchSlices[2 * tokenLane + byte];
                        const int64_t upSlice =
                            upScratchSlices[2 * tokenLane + byte];
                        blockMemWindows.push_back({
                            memResource("read", hemisphere, gateSlice,
                                gateScratchBank),
                            row - context.westLatency(gateSlice), 1});
                        blockMemWindows.push_back({
                            memResource("read", hemisphere, upSlice,
                                upScratchBank),
                            row - context.westLatency(upSlice), 1});
                        const int64_t destination = 1 - hemisphere;
                        const int64_t outputSlice = context.hidden_slices[
                            2 * tokenLane + byte];
                        const int64_t outputTransport =
                            *target.transport_latency(
                                target::StreamEndpoint::VxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::East,
                                outputSlice);
                        blockMemWindows.push_back({
                            memResource("write", destination, outputSlice,
                                hiddenBank),
                            row + kVxmSwishLatency + outputTransport,
                            1});
                    }
                }
            }
            const int64_t blockSwishCycle =
                memScheduler.reserve(swishCycle, blockMemWindows);
            auto outputs = emitScratchSwish(
                context, blockSwishCycle, tile, 1);
            localSwish = outputs.first.getResult();
            peerSwish = outputs.second.getResult();
            for (int64_t row = 0; row < tile; ++row) {
                const int64_t rowSwishCycle = blockSwishCycle + row;
                const int64_t rowBlock = row / blockRows;
                const int64_t tokenLane = row % blockRows;
                const int64_t scratchAddress = scratchBase
                    + pair * scratchRowsPerPair
                    + tokenBlock * blockIssues + rowBlock;
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    const int64_t inputStreamBase = hemisphere * 16;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t gateSlice =
                            gateScratchSlices[2 * tokenLane + byte];
                        const int64_t upSlice =
                            upScratchSlices[2 * tokenLane + byte];
                        emitMem(rewriter, ffn.getLoc(),
                            rowSwishCycle
                                - context.westLatency(gateSlice),
                            hemisphere * memory.slices_per_hemisphere
                                + gateSlice,
                            "read", scratchAddress,
                            target.streams().streams_per_direction
                                + inputStreamBase + byte,
                            1, 1, 0, "sram", -1,
                            gateScratchBank);
                        emitMem(rewriter, ffn.getLoc(),
                            rowSwishCycle
                                - context.westLatency(upSlice),
                            hemisphere * memory.slices_per_hemisphere
                                + upSlice,
                            "read", scratchAddress,
                            target.streams().streams_per_direction
                                + inputStreamBase + 2 + byte,
                            1, 1, 0, "sram", -1,
                            upScratchBank);
                    }
                }
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    const int64_t outputBlock =
                        pair * memory.hemispheres + hemisphere;
                    const int64_t tokenWave = row / blockRows;
                    const int64_t outputAddress = hiddenBase
                        + (tokenBlock * hiddenBlocks + outputBlock)
                            * throughput.tile_rows
                        + tokenWave;
                    const int64_t destination = 1 - hemisphere;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice =
                                context.hidden_slices[
                                    2 * tokenLane + byte];
                            const int64_t physicalOutputStream =
                                hemisphere == 0 ? 6 : 14;
                            const int64_t outputTransport =
                                *target.transport_latency(
                                    target::StreamEndpoint::VxmResult,
                                    target::StreamEndpoint::Mem,
                                    target::StreamDirection::East, slice);
                            emitMem(rewriter, ffn.getLoc(),
                                rowSwishCycle + kVxmSwishLatency
                                    + outputTransport,
                                destination
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", outputAddress,
                                physicalOutputStream + byte,
                                1, 1, 0, "sram", -1, hiddenBank);
                            lastSwishCycle = std::max(lastSwishCycle,
                                rowSwishCycle + kVxmSwishLatency
                                    + outputTransport);
                    }
                    lastHidden = hemisphere == 0
                        ? localSwish : peerSwish;
                }
            }
            swishCycle = blockSwishCycle + tile;
        }
    }

    // A Down reduction is consumed by both hemisphere-local MXMs. The VXM
    // output router has one destination per output block, so first retain one
    // owner copy above, then multicast through SRAM: owner MEM emits west,
    // the passive VXM bridge forwards it to the peer east fabric, and peer
    // MEM captures it. This avoids a VXM ALU pass and keeps one canonical
    // result until the value actually needs two consumers.
    StreamFabricScheduler copyFabric(
        target.streams().system_register_columns,
        target.streams().streams_per_direction);
    int64_t copyIssueCycle = lastSwishCycle + 1;
    int64_t lastCopyCycle = lastSwishCycle;
    for (int64_t pair = 0; pair < pairCount; ++pair) {
        for (int64_t tokenBlock = 0;
             tokenBlock < tokenBlocks; ++tokenBlock) {
            for (int64_t row = 0; row < tile; ++row) {
                int64_t nextCopyIssueCycle = copyIssueCycle + 1;
                const int64_t tokenLane = row % blockRows;
                const int64_t tokenWave = row / blockRows;
                for (int64_t sourceHemisphere = 0;
                     sourceHemisphere < memory.hemispheres;
                     ++sourceHemisphere) {
                    const int64_t outputBlock =
                        pair * memory.hemispheres + sourceHemisphere;
                    const int64_t address = hiddenBase
                        + (tokenBlock * hiddenBlocks + outputBlock)
                            * throughput.tile_rows
                        + tokenWave;
                    const int64_t owner = 1 - sourceHemisphere;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice =
                            context.hidden_slices[2 * tokenLane + byte];
                        const int64_t group = slice
                            / target.streams()
                                  .mem_slices_per_register_group;
                        const int64_t westSource = group;
                        const int64_t westEdge = 0;
                        StreamRouteWindow westRoute {
                            target::StreamDirection::West,
                            westSource,
                            westEdge,
                            byte,
                            1,
                            1,
                            1,
                            copyFabric.allocate_token_range(1),
                            StreamConsumerMode::Tap,
                        };
                        const int64_t readCycle =
                            copyFabric.reserve(copyIssueCycle, westRoute);
                        nextCopyIssueCycle = std::max(
                            nextCopyIssueCycle, readCycle + 1);
                        const auto westLatency = target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::VxmInput,
                            target::StreamDirection::West, slice);
                        const auto eastLatency = target.transport_latency(
                            target::StreamEndpoint::VxmResult,
                            target::StreamEndpoint::Mem,
                            target::StreamDirection::East, slice);
                        if (!westLatency || !eastLatency)
                            return mlir::failure();
                        const int64_t bridgeCycle =
                            readCycle + *westLatency;
                        StreamRouteWindow eastRoute {
                            target::StreamDirection::East,
                            0,
                            group,
                            byte,
                            1,
                            1,
                            1,
                            westRoute.token_id,
                            StreamConsumerMode::Consume,
                        };
                        const int64_t eastStart =
                            copyFabric.reserve(bridgeCycle, eastRoute);
                        emitMem(rewriter, ffn.getLoc(), readCycle,
                            owner * memory.slices_per_hemisphere + slice,
                            "read", address,
                            target.streams().streams_per_direction + byte,
                            1, 1, 0, "sram", -1, hiddenBank);
                        const int64_t writeCycle = eastStart + *eastLatency;
                        emitMem(rewriter, ffn.getLoc(), writeCycle,
                            sourceHemisphere
                                    * memory.slices_per_hemisphere
                                + slice,
                            "write", address, byte, 1, 1, 0,
                            "sram", -1, hiddenBank);
                        lastCopyCycle =
                            std::max(lastCopyCycle, writeCycle);
                    }
                }
                copyIssueCycle = nextCopyIssueCycle;
            }
        }
    }

    return FfnSwishEmission {lastHidden, lastCopyCycle};
}

} // namespace ftlpu::compiler::schedule::ffn_detail
