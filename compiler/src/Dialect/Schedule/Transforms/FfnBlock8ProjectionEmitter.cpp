#include "FfnStageEmitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_block8_projection_planner.hpp"
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

llvm::SmallVector<int64_t> chooseScratchSlices(
    const target::LPUTargetModel& target,
    llvm::ArrayRef<int64_t> excluded,
    llvm::ArrayRef<int64_t> deferred,
    std::size_t count)
{
    llvm::SmallVector<int64_t> result;
    const auto append = [&](bool includeDeferred) {
        for (int64_t slice = 0;
             slice < target.memory().slices_per_hemisphere
             && result.size() < count; ++slice) {
            if (llvm::is_contained(excluded, slice)
                || llvm::is_contained(result, slice)
                || (llvm::is_contained(deferred, slice)
                    != includeDeferred))
                continue;
            result.push_back(slice);
        }
    };
    append(false);
    append(true);
    return result;
}

std::pair<VxmOp, VxmOp> emitScratchSwish(
    FfnEmissionContext& context, int64_t cycle, int64_t hemisphere,
    int64_t outputStreamBase)
{
    using attention_detail::emitVxm;
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto hemi = context.hemisphereName(hemisphere);
    const auto dataFormat = lpu_16bit_data_format(
        llvm::cast<mlir::RankedTensorType>(
            ffn.getActivation().getType()).getElementType());
    const int64_t west = target.streams().streams_per_direction;
    mlir::Value value = ffn.getActivation();

    value = emitVxm(rewriter, ffn.getLoc(), value, cycle, 0,
        "negate", lpu_16bit_stream_kind(
            llvm::cast<mlir::RankedTensorType>(
                ffn.getActivation().getType()).getElementType()),
        west, 0.0f, "immediate", 0, 0.0f, "fp32", -1,
        hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle, 1,
        "multiply", lpu_16bit_stream_kind(
            llvm::cast<mlir::RankedTensorType>(
                ffn.getActivation().getType()).getElementType()),
        west, 0.0f, lpu_16bit_stream_kind(
            llvm::cast<mlir::RankedTensorType>(
                ffn.getActivation().getType()).getElementType()),
        west + 2, 0.0f, "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 1, 2,
        "exp", "alu", 0, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 1, 5,
        "pass", "alu", 1, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 2, 3,
        "add", "alu", 2, 0.0f, "immediate", 0, 1.0f,
        "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 2, 6,
        "pass", "alu", 5, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 3, 4,
        "divide", "immediate", 0, 1.0f, "alu", 3, 0.0f,
        "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 3, 7,
        "pass", "alu", 6, 0.0f, "immediate", 0, 0.0f,
        "fp32", -1, hemi, hemi).getResult();
    value = emitVxm(rewriter, ffn.getLoc(), value, cycle + 4, 8,
        "multiply", "alu", 7, 0.0f, "alu", 4, 0.0f,
        "fp32", -1, hemi, hemi).getResult();
    auto local = emitVxm(rewriter, ffn.getLoc(), value, cycle + 5, 9,
        "cast", "alu", 8, 0.0f, "immediate", 0, 0.0f,
        dataFormat, outputStreamBase, hemi, hemi);
    auto peer = emitVxm(rewriter, ffn.getLoc(), value, cycle + 5, 10,
        "cast", "alu", 8, 0.0f, "immediate", 0, 0.0f,
        dataFormat, outputStreamBase, hemi,
        context.hemisphereName(1 - hemisphere));
    return {local, peer};
}

} // namespace

mlir::FailureOr<FfnSwishEmission> emitFfnBlock8ProjectionAndSwish(
    FfnEmissionContext& context)
{
    using attention_detail::emitMem;
    using attention_detail::emitMxm;
    using attention_detail::emitMxmDequant;
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

    llvm::SmallVector<int64_t> scratchReserved(
        context.activation_slices.begin(), context.activation_slices.end());
    scratchReserved.append(
        context.hidden_slices.begin(), context.hidden_slices.end());
    llvm::SmallVector<int64_t> scratchDeferred(
        context.weight_slices.begin(), context.weight_slices.end());
    scratchDeferred.append(context.up_weight_slices.begin(),
        context.up_weight_slices.end());
    const auto gateScratchSlices = chooseScratchSlices(
        target, scratchReserved, scratchDeferred,
        context.hidden_slices.size());
    const auto upScratchSlices = context.hidden_slices;
    if (gateScratchSlices.size() != context.hidden_slices.size()
        || upScratchSlices.size() != context.hidden_slices.size())
        return mlir::failure();

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
    const int64_t scratchBase = std::max({hiddenBase + hiddenRows,
        placementEnd(context.gate_raw.getPlacement()),
        placementEnd(context.up_raw.getPlacement()),
        placementEnd(context.down_raw.getPlacement())});
    const int64_t scratchRowsPerPair = tokenBlocks * blockIssues;
    if (scratchBase + pairCount * scratchRowsPerPair
        > memory.sram_depth_rows)
        return mlir::failure();
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
    const int64_t swishOutputStreamBase = fused
        ? context.fused_output->stream_base
        : 0;

    int64_t phaseStart = maxWeightLatency + 1;
    mlir::Value lastHidden = activation;
    int64_t lastSwishCycle = 0;
    int64_t projectionDrainEnd = phaseStart;
    llvm::SmallVector<int64_t> pairDrainEnds;
    pairDrainEnds.reserve(pairCount);
    const auto memResource = [](llvm::StringRef access,
                                int64_t hemi, int64_t slice) {
        (void)access;
        return "ffn.block8.mem." + std::to_string(hemi)
            + "." + std::to_string(slice);
    };
    ResourceScheduler memScheduler;
    for (int64_t pair = 0; pair < pairCount; ++pair) {
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
                const float scale = projection == 0
                    ? ffn.getGateScale().convertToFloat()
                    : ffn.getUpScale().convertToFloat();
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    const int64_t unit = hemisphere;
                    for (int64_t pulse = 0;
                         pulse < throughput.tile_rows; ++pulse) {
                        const int64_t cycle = loadStart + pulse;
                        const int64_t address = weightBase
                            + (pair * reductionBlocks + reduction)
                                * throughput.tile_rows
                            + throughput.tile_rows - 1 - pulse;
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
                                "read", address, stream,
                                1, 1, 0, "sram", binding);
                            memScheduler.reserve_at(cycle - *latency,
                                {{memResource("read", hemisphere, slice),
                                    0, 1}});
                        }
                        emitMxmDequant(rewriter, ffn.getLoc(),
                            cycle, unit, scale);
                        emitMxm(rewriter, ffn.getLoc(), cycle, unit,
                            "iw", weightBuffer, pulse,
                            0, 0, 1, 1, 0, 1, "sram", true,
                            "supercell", 0, dataFormat,
                            "int8_dequant_bf16");
                    }
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
                            emitMem(rewriter, ffn.getLoc(),
                                computeCycle - *latency,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read", activationAddress,
                                activationStreamBase + stream,
                                blockIssues, 1, 1,
                                "sram", activationBinding);
                            memScheduler.reserve_at(computeCycle - *latency,
                                {{memResource("read", hemisphere, slice),
                                    0, blockIssues}});
                        }
                        const int64_t accumulatorBase =
                            projection * rowBlocks
                            + tokenBlock * blockIssues;
                        emitMxm(rewriter, ffn.getLoc(), computeCycle,
                            hemisphere, "compute", weightBuffer, 0,
                            activationStreamBase, 0,
                            blockIssues, 1, accumulatorBase, 1,
                            finalReduction ? "stream" : "sram", true,
                            "supercell", 0, dataFormat, {}, "block8");
                        if (!finalReduction) continue;

                        const auto& slices = projection == 0
                            ? gateScratchSlices : upScratchSlices;
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
                                + target.mxm_first_result_latency()
                                + *latency;
                            emitMem(rewriter, ffn.getLoc(), writeCycle,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", scratchAddress,
                                target.streams().streams_per_direction
                                    + stream,
                                blockIssues, 1, 1);
                            memScheduler.reserve_at(writeCycle,
                                {{memResource("write", hemisphere, slice),
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
            const int64_t pairComputeEnd = firstGateCompute
                + reductionBlocks * tokenBlocks
                    * throughput.mxm_block_group_interval;
            phaseStart = pairComputeEnd
                - throughput.mxm_local_load_to_compute_latency;
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
                                1, 1, 0, "sram", binding);
                            memScheduler.reserve_at(cycle - *latency,
                                {{memResource("read", hemisphere, slice),
                                    0, 1}});
                        }
                        emitMxmDequant(rewriter, ffn.getLoc(),
                            cycle, unit, scale);
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
                            "sram", activationBinding);
                        memScheduler.reserve_at(computeCycle - *latency,
                            {{memResource("read", hemisphere, slice),
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
                            true, "supercell", 0,
                            dataFormat, {}, "block8");
                        if (!finalReduction) continue;

                        const auto& slices = localMxm == 0
                            ? gateScratchSlices : upScratchSlices;
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
                            // A Block8 compute reaches tile 3 after three
                            // cycles. Its four 8x8 result blocks then leave
                            // the MXM boundary one per cycle, matching the
                            // MEM instruction's north-to-south tile walk.
                            const int64_t writeCycle =
                                computeCycle
                                + target.mxm_first_result_latency()
                                + *latency;
                            emitMem(rewriter, ffn.getLoc(), writeCycle,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", scratchAddress,
                                target.streams().streams_per_direction
                                    + outputStreamBase + stream,
                                blockIssues, 1, 1);
                            memScheduler.reserve_at(writeCycle,
                                {{memResource("write", hemisphere, slice),
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
    for (int64_t pair = 0; pair < pairCount; ++pair) {
        if (fused)
            swishCycle = std::max(swishCycle,
                pairDrainEnds[pair] + maxScratchReadLatency + 1);
        for (int64_t tokenBlock = 0;
             tokenBlock < tokenBlocks; ++tokenBlock) {
            for (int64_t row = 0; row < tile; ++row) {
                const int64_t rowBlock = row / blockRows;
                const int64_t tokenLane = row % blockRows;
                const int64_t scratchAddress = scratchBase
                    + pair * scratchRowsPerPair
                    + tokenBlock * blockIssues + rowBlock;
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    llvm::SmallVector<ResourceWindow> memWindows;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t gateSlice =
                            gateScratchSlices[2 * tokenLane + byte];
                        const int64_t upSlice =
                            upScratchSlices[2 * tokenLane + byte];
                        memWindows.push_back({
                            memResource("read", hemisphere, gateSlice),
                            -context.westLatency(gateSlice), 1});
                        memWindows.push_back({
                            memResource("read", hemisphere, upSlice),
                            -context.westLatency(upSlice), 1});
                        for (int64_t destination = 0;
                             destination < memory.hemispheres;
                             ++destination) {
                            const int64_t outputSlice =
                                context.hidden_slices[
                                    2 * tokenLane + byte];
                            memWindows.push_back({
                                memResource("write", destination,
                                    outputSlice),
                                6 + outputSlice
                                    / target.streams()
                                          .mem_slices_per_register_group,
                                1});
                        }
                    }
                    swishCycle = memScheduler.reserve(
                        swishCycle, memWindows);
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t gateSlice =
                            gateScratchSlices[2 * tokenLane + byte];
                        const int64_t upSlice =
                            upScratchSlices[2 * tokenLane + byte];
                        emitMem(rewriter, ffn.getLoc(),
                            swishCycle - context.westLatency(gateSlice),
                            hemisphere * memory.slices_per_hemisphere
                                + gateSlice,
                            "read", scratchAddress,
                            target.streams().streams_per_direction + byte,
                            1, 1, 0);
                        emitMem(rewriter, ffn.getLoc(),
                            swishCycle - context.westLatency(upSlice),
                            hemisphere * memory.slices_per_hemisphere
                                + upSlice,
                            "read", scratchAddress,
                            target.streams().streams_per_direction
                                + 2 + byte,
                            1, 1, 0);
                    }
                    auto [local, peer] =
                        emitScratchSwish(context, swishCycle, hemisphere,
                            swishOutputStreamBase);
                    const int64_t outputBlock =
                        pair * memory.hemispheres + hemisphere;
                    const int64_t tokenWave = row / blockRows;
                    const int64_t outputAddress = hiddenBase
                        + (tokenBlock * hiddenBlocks + outputBlock)
                            * throughput.tile_rows
                        + tokenWave;
                    for (int64_t destination = 0;
                         destination < memory.hemispheres;
                         ++destination) {
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice =
                                context.hidden_slices[
                                    2 * tokenLane + byte];
                            emitMem(rewriter, ffn.getLoc(),
                                swishCycle + 6
                                    + slice
                                        / target.streams()
                                              .mem_slices_per_register_group,
                                destination
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", outputAddress,
                                swishOutputStreamBase + byte,
                                1, 1, 0);
                        }
                        lastHidden = destination == hemisphere
                            ? local.getResult() : peer.getResult();
                    }
                    lastSwishCycle = swishCycle;
                    ++swishCycle;
                }
            }
        }
    }

    return FfnSwishEmission {lastHidden, lastSwishCycle};
}

} // namespace ftlpu::compiler::schedule::ffn_detail
