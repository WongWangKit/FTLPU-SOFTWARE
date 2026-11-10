#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ProjectionGemmSchedule.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;
namespace {

int64_t elementTypeBytes(mlir::Type type)
{
    if (type.isInteger(8) || llvm::isa<mlir::Float8E4M3FNType>(type)) return 1;
    if (is_lpu_16bit_float(type)) return 2;
    if (type.isF32()) return 4;
    return 0;
}

int64_t argumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

llvm::SmallVector<int64_t> slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    const auto values =
        placement.getAs<mlir::ArrayAttr>("slices");
    if (!values) return result;
    for (mlir::Attribute value : values)
        result.push_back(
            llvm::cast<mlir::IntegerAttr>(value).getInt());
    return result;
}

BindingOp createBinding(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::ValueRange source, int64_t index,
    llvm::StringRef access, llvm::StringRef role,
    mlir::RankedTensorType type, mlir::DictionaryAttr placement)
{
    mlir::OperationState state(
        location, BindingOp::getOperationName());
    state.addOperands(source);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr(
            "index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr(
            "access", rewriter.getStringAttr(access)),
        rewriter.getNamedAttr(
            "role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("bytes",
            rewriter.getI64IntegerAttr(type.getNumElements()
                * elementTypeBytes(type.getElementType()))),
        rewriter.getNamedAttr("placement", placement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

mlir::FailureOr<mlir::Value> emitLinearProjection(
    mlir::IRRewriter& rewriter, stream::ProjectionTaskOp op,
    const target::LPUTargetModel& target, int64_t outputIndex)
{
    const auto memoryPlan = *op.getMemoryPlan();
    const auto replicatedAttr = op.getConfig().getAs<mlir::BoolAttr>("replicated_result");
    const bool replicated = replicatedAttr && replicatedAttr.getValue();
    const auto overlapAttr =
        op.getConfig().getAs<mlir::BoolAttr>("projection_solve_overlap");
    const bool solveOverlap = overlapAttr && overlapAttr.getValue();
    const auto convOverlapAttr =
        op.getConfig().getAs<mlir::BoolAttr>("projection_conv_overlap");
    const bool convOverlap = convOverlapAttr && convOverlapAttr.getValue();
    const auto vectorOverlapAttr =
        op.getConfig().getAs<mlir::BoolAttr>("projection_vector_overlap");
    const bool vectorOverlap = convOverlap ||
        (vectorOverlapAttr && vectorOverlapAttr.getValue());
    const auto inputPlacement =
        memoryPlan.getAs<mlir::DictionaryAttr>("input");
    const auto weightPlacement =
        memoryPlan.getAs<mlir::DictionaryAttr>("weight");
    const auto resultPlacement =
        memoryPlan.getAs<mlir::DictionaryAttr>("result");
    const auto mAttr = op.getConfig().getAs<mlir::IntegerAttr>("m");
    const auto nAttr = op.getConfig().getAs<mlir::IntegerAttr>("n");
    const auto kAttr = op.getConfig().getAs<mlir::IntegerAttr>("k");
    const auto scaleAttr =
        op.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
    if (!inputPlacement || !weightPlacement || !resultPlacement
        || !mAttr || !nAttr || !kAttr || !scaleAttr) {
        op.emitError("linear projection has an incomplete physical plan");
        return mlir::failure();
    }
    const auto inputSlices = slices(inputPlacement);
    const auto weightSlices = slices(weightPlacement);
    const auto resultSlices = slices(resultPlacement);
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const auto weightType =
        llvm::cast<mlir::RankedTensorType>(op.getWeight().getType());
    const auto resultType =
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    const int64_t m = mAttr.getInt();
    const int64_t n = nAttr.getInt();
    const int64_t k = kAttr.getInt();
    const bool fp32Partial = llvm::isa<mlir::Float8E4M3FNType>(weightType.getElementType())
        && resultType.getElementType().isF32();
    auto executionPolicy =
        target::mxm_execution_policy_from_operation(op.getOperation());
    if (mlir::failed(executionPolicy)) {
        op.emitError("invalid MXM execution policy");
        return mlir::failure();
    }
    auto execution = target::plan_mxm_execution_strategy(
        {m, n, k, inputType.getElementType().isBF16(),
            (weightType.getElementType().isInteger(8) || llvm::isa<mlir::Float8E4M3FNType>(weightType.getElementType())),
            is_lpu_16bit_float(resultType.getElementType()) || fp32Partial, true},
        target, *executionPolicy);
    if (mlir::failed(execution)) {
        op.emitError("cannot select an MXM execution strategy");
        return mlir::failure();
    }
    const auto weightInputMode = llvm::isa<mlir::Float8E4M3FNType>(weightType.getElementType())
        ? llvm::StringRef("fp8_dequant_bf16") : execution->weight_input_mode();
    const bool localDequant = execution->uses_local_dequant();
    const std::size_t expectedActivationSlices = 2;
    const std::size_t expectedResultSlices = replicated || fp32Partial ? 4 : 2;
    if (inputSlices.size() != expectedActivationSlices
        || weightSlices.size() != 8
        || resultSlices.size() != expectedResultSlices) {
        op.emitError(
            "linear projection physical layout does not match the "
            "selected MXM execution strategy");
        return mlir::failure();
    }

    const int64_t tile = target.throughput().mxm_rows;
    const bool tokenPartition = resultPlacement.getAs<mlir::StringAttr>("kind").getValue()
        == "token_partition_planar";
    if (tokenPartition && (!localDequant || m != tile || n != 2 * tile || replicated)) {
        op.emitError("token-partitioned projection requires one locally dequantized output tile");
        return mlir::failure();
    }
    if (replicated && (!localDequant || m != tile)) {
        op.emitError("replicated gated projection requires one token tile with local dequantization");
        return mlir::failure();
    }
    if ((m > tile && m % tile) || n % (2 * tile) || k % tile
        || (m < tile && !localDequant)
        || m > target.throughput().mxm_accumulator_blocks * tile) {
        op.emitError("linear projection dimensions are not MXM aligned");
        return mlir::failure();
    }
    const int64_t inputBase =
        inputPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t weightBase =
        weightPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t resultBase =
        resultPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t resultBank = resultPlacement.getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t inputBinding = argumentIndex(op.getInput());
    const int64_t weightBinding = argumentIndex(op.getWeight());
    // FP8 tile scales are explicit constants on the projection.
    const int64_t scaleBinding = llvm::isa<mlir::Float8E4M3FNType>(
        weightType.getElementType()) ? -1 : weightBinding;
    if (weightBinding < 0) {
        op.emitError(
            "linear projection weight must be a function argument");
        return mlir::failure();
    }

    rewriter.setInsertionPoint(op);
    if (inputBinding >= 0) createBinding(rewriter, op.getLoc(), op.getInput(), inputBinding,
        "input", "activation",
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType()),
        inputPlacement);
    createBinding(rewriter, op.getLoc(), op.getWeight(), weightBinding,
        "input", "weight",
        llvm::cast<mlir::RankedTensorType>(op.getWeight().getType()),
        weightPlacement);

    const llvm::StringRef dataFormat = lpu_16bit_data_format(
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType())
            .getElementType());
    const auto readLatency = [&](int64_t slice) {
        return slice
                / target.streams().mem_slices_per_register_group
            + 2;
    };
    const auto weightReadLatency = [&](int64_t slice) {
        if (!localDequant) return readLatency(slice);
        const auto latency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        return latency.value_or(-1);
    };
    int64_t maxWeightLatency = 0;
    for (int64_t slice : weightSlices) {
        if (weightReadLatency(slice) < 0)
            return mlir::failure();
        maxWeightLatency =
            std::max(maxWeightLatency, weightReadLatency(slice));
    }
    const int64_t weightToIw =
        target.throughput().vxm_weight_to_iw_latency;
    const int64_t vectorWeightLoadLead =
        (target.memory().hemispheres - 1)
            * target.throughput().lanes_per_tile
        + 3 + weightToIw + 1;
    const int64_t tokenBlocks = (m + tile - 1) / tile;
    if (solveOverlap && (!localDequant || tokenBlocks != 1
        || replicated || tokenPartition)) {
        op.emitError("projection/solve overlap requires a single token tile "
            "with local dequantization and hemisphere-local results");
        return mlir::failure();
    }
    const int64_t reductionBlocks = k / tile;
    const int64_t outputGroups = n / (2 * tile);
    const bool continuousMultiTile = localDequant && tile == 32 && tokenBlocks > 1
        && !replicated && !tokenPartition && target.supports_mxm_weight_activation_overlap()
        && target.throughput().mxm_weight_buffers == 2 && reductionBlocks % 2 == 0;
    if (vectorOverlap && (solveOverlap || !localDequant || replicated || tokenPartition
        || m % tile || (tokenBlocks > 1 && !continuousMultiTile)
        || target.streams().streams_per_direction != 32
        || target.throughput().mxm_prefill_weight_streams != 1)) {
        op.emitError("projection/vector overlap requires hemisphere-local OneVector projection tiles");
        return mlir::failure();
    }
    // Vector kernels use fixed VXM operand/result ports. A concurrent MXM uses
    // the independent E10/11 activation and W10/11 result pairs.
    const int64_t resultStream = vectorOverlap ? 10 : (solveOverlap ? 8 : 0);
    const auto activationStreamFor = [&](int64_t hemisphere) {
        return (vectorOverlap || solveOverlap) ? resultStream : hemisphere * 2;
    };
    const int64_t weightStreamBase = target.streams().streams_per_direction
        - ((vectorOverlap || solveOverlap) ? 1 : target.throughput().mxm_int8_load_streams_per_cycle);
    const auto pagedAttr = weightPlacement.getAs<mlir::BoolAttr>("paged_weight");
    const int64_t weightPage = pagedAttr && pagedAttr.getValue() ? 0 : -1;
    int64_t stagingEnd = 0;
    if (const auto source = memoryPlan.getAs<mlir::DictionaryAttr>("source_input")) {
        stagingEnd = emitDistributedActivationStaging(rewriter, op.getLoc(), target,
            source, inputPlacement, m, k);
        if (stagingEnd < 0) {
            op.emitError("cannot stage distributed projection input");
            return mlir::failure();
        }
    }
    const int64_t localMxm = 0;
    const int64_t accumulatorAddress =
        target.throughput().mxm_accumulator_blocks
            * target.throughput().mxm_rows
        - m;
    const int64_t accumulatorLatency =
        target.throughput().mxm0_accumulator_latency;
    const float scale =
        static_cast<float>(scaleAttr.getValueAsDouble());
    const auto westScale=op.getConfig().getAs<mlir::FloatAttr>("rhs_scale_west");
    const auto scaleFor=[&](int64_t hemisphere) {
        return hemisphere && westScale?static_cast<float>(westScale.getValueAsDouble()):scale;
    };

    if (localDequant && tokenBlocks == 1) {
        // Emit the projection's loop nest directly. Final-K compute exports
        // BF16 and clears its accumulator, so no accumulator-read command
        // has to interrupt the COMPUTE_3D instruction between output groups.
        // MEM transport is a startup lead, not a per-K bubble. With two
        // weight buffers the next K tile loads while the current tile computes.
        // Keep the short-row/decode policy unchanged until validated separately.
        const bool columnLoad = target.throughput().mxm_prefill_weight_streams == 1;
        const int64_t loadLead = columnLoad ? 32 : target.throughput().mxm_local_load_to_compute_latency;
        const int64_t period = m == tile && reductionBlocks % 2 == 0
            && target.throughput().mxm_weight_buffers == 2
            ? std::max({m, accumulatorLatency, int64_t{4}, (loadLead + m + 1) / 2})
            : std::max<int64_t>(64, maxWeightLatency + loadLead + m);
        // toggle_dim1 restarts at buffer 0 in each group. Odd K counts retain
        // the conservative cadence to finish consuming that buffer first.
        const int64_t groupPeriod = reductionBlocks * period;
        const int64_t weightCycle = stagingEnd + maxWeightLatency;
        const int64_t computeCycle = weightCycle + loadLead;
        // The concurrent FP32 forward solve owns West 0..7 and returns
        // results on East 2..5. Keep projection traffic on separate existing
        // SR pairs without changing domains, buffer rotation, or cadence.
        for (int64_t hemisphere = 0; hemisphere < target.memory().hemispheres; ++hemisphere) {
            const int64_t unit = hemisphere * target.throughput().mxms_per_hemisphere;
            const int64_t activationStream = activationStreamFor(hemisphere);
            for (int64_t stream = 0; stream < 8; ++stream) {
                const int64_t slice = weightSlices[stream];
                emitMem3D(rewriter, op.getLoc(), weightCycle - weightReadLatency(slice) + (columnLoad ? stream : 0),
                    hemisphere * target.memory().slices_per_hemisphere + slice,
                    "read", weightBase + (columnLoad ? 3 : 0), weightStreamBase + (columnLoad ? 0 : stream), 4, columnLoad ? 8 : 1, columnLoad ? -1 : 1, "sram", weightBinding,
                    reductionBlocks, period, 4, outputGroups, groupPeriod, reductionBlocks * 4,
                    weightPlacement.getAs<mlir::IntegerAttr>("bank").getInt(), weightPage);
            }
            emitMxmDequant3D(rewriter, op.getLoc(), weightCycle, unit, scaleFor(hemisphere),
                columnLoad ? 32 : 4, 1, reductionBlocks, period, outputGroups, groupPeriod, scaleBinding);
            const projection_detail::ProjectionGemmSchedule gemm{
                m, reductionBlocks, outputGroups, period, groupPeriod};
            const projection_detail::ProjectionGemmSchedule::Binding gemmBinding{
                unit, 0, activationStream, resultStream, accumulatorAddress, dataFormat};
            gemm.emitLoad(rewriter, op.getLoc(), weightCycle, gemmBinding,
                columnLoad ? 32 : 4, columnLoad, weightInputMode, weightStreamBase, "sram");
            for (int64_t byte = 0; byte < 2; ++byte) {
                const auto slice = inputSlices[byte];
                const auto latency = target.transport_latency(target::StreamEndpoint::Mem,
                    target::StreamEndpoint::MxmActivation, target::StreamDirection::East, slice);
                if (!latency) return mlir::failure();
                emitMem3D(rewriter, op.getLoc(), computeCycle - *latency,
                    hemisphere * target.memory().slices_per_hemisphere + slice,
                    "read", inputBase, activationStream + byte, m, 1, 1,
                    "sram", inputBinding, reductionBlocks, period, m,
                    outputGroups, groupPeriod, 0,
                    inputPlacement.getAs<mlir::IntegerAttr>("bank").getInt());
            }
            gemm.emitCompute(rewriter, op.getLoc(), computeCycle,
                gemmBinding, false, {}, fp32Partial ? "fp32" : "");
            if (tokenPartition) {
                if (hemisphere != 0) continue; // only the first feature word is live
                const int64_t half = (m + 1) / 2;
                const int64_t resultCycle = computeCycle + (reductionBlocks - 1) * period
                    + target.mxm_first_result_latency();
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const auto slice = resultSlices[byte];
                    const auto local = target.transport_latency(target::StreamEndpoint::MxmResult,
                        target::StreamEndpoint::Mem, target::StreamDirection::West, slice);
                    const auto bridge = target.transport_latency(target::StreamEndpoint::VxmBridgeResult,
                        target::StreamEndpoint::Mem, target::StreamDirection::East, slice);
                    if (!local || !bridge) return mlir::failure();
                    // Let the first half flow passively to the opposite MEM;
                    // consume the second half at the local result destination.
                    emitMem3D(rewriter, op.getLoc(), resultCycle + half + *local,
                        slice, "write", resultBase, target.streams().streams_per_direction + byte,
                        m - half, 1, 1, "sram", -1, 1, 1, 0, 1, 1, 0, resultBank);
                    const int64_t delay = target.throughput().accumulator_to_vxm_latency
                        - target.mxm_first_result_latency() + *bridge;
                    emitMem3D(rewriter, op.getLoc(), resultCycle + delay,
                        target.memory().slices_per_hemisphere + slice,
                        "write", resultBase, byte, half, 1, 1, "sram", -1,
                        1, 1, 0, 1, 1, 0, resultBank);
                }
                continue;
            }
            for (int64_t byte = 0; byte < (fp32Partial ? 4 : 2); ++byte) {
                const auto slice = resultSlices[(replicated ? 2*hemisphere : 0)+byte];
                const auto latency = target.transport_latency(target::StreamEndpoint::MxmResult,
                    target::StreamEndpoint::Mem, target::StreamDirection::West, slice);
                if (!latency) return mlir::failure();
                emitMem3D(rewriter, op.getLoc(), computeCycle + (reductionBlocks - 1) * period
                        + target.mxm_first_result_latency() + *latency,
                    hemisphere * target.memory().slices_per_hemisphere + slice,
                    replicated ? "write_tap" : "write", resultBase,
                    target.streams().streams_per_direction + resultStream + byte,
                    m, 1, 1, "sram", -1, outputGroups, groupPeriod, m,
                    1, 1, 0, resultBank);
                if (replicated) {
                    const auto bridge = target.transport_latency(target::StreamEndpoint::VxmBridgeResult,
                        target::StreamEndpoint::Mem,target::StreamDirection::East,slice);
                    if (!bridge) return mlir::failure();
                    const auto delay = target.throughput().accumulator_to_vxm_latency
                        - target.mxm_first_result_latency() + *bridge;
                    emitMem3D(rewriter, op.getLoc(), computeCycle+(reductionBlocks-1)*period
                        +target.mxm_first_result_latency()+delay,
                        (1-hemisphere)*target.memory().slices_per_hemisphere+slice,
                        "write",resultBase,byte,m,1,1,"sram",-1,
                        outputGroups,groupPeriod,m,1,1,0,resultBank);
                }
            }
        }
        mlir::OperationState timeline(op.getLoc(), TimelineOp::getOperationName());
        timeline.addAttributes({rewriter.getNamedAttr("name", rewriter.getStringAttr("linear_projection")),
            rewriter.getNamedAttr("start", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("end", rewriter.getI64IntegerAttr(computeCycle + outputGroups * groupPeriod + 64))});
        rewriter.create(timeline);
        return createBinding(rewriter, op.getLoc(), {}, outputIndex, "output", "result", resultType, resultPlacement).getValue();
    }

    if (continuousMultiTile) {
        // A weight tile is reused by every token tile before advancing K.
        // Load K+1 into the other existing weight buffer while K computes;
        // MEM transport and IW are startup leads, not per-K barriers.
        // The compute descriptor needs [row, token tile, K], leaving no
        // dimension for N. Emit one complete descriptor per output group,
        // in queue order, rather than interleaving unfinished instructions.
        // Odd K retains the fallback: the cross-N load domain resets its
        // toggle_dim1 buffer at each output group.
        const bool columnLoad =
            target.throughput().mxm_prefill_weight_streams == 1;
        const int64_t loadLead = columnLoad ? 32
            : target.throughput().mxm_local_load_to_compute_latency;
        const int64_t period = std::max(
            {m, accumulatorLatency, int64_t{4}, (loadLead + m + 1) / 2});
        const int64_t groupPeriod = reductionBlocks * period;
        const int64_t weightCycle = stagingEnd + maxWeightLatency;
        const int64_t computeCycle = weightCycle + loadLead;
        const projection_detail::ProjectionGemmSchedule loadSchedule{
            m, reductionBlocks, outputGroups, period, groupPeriod};
        for (int64_t hemisphere = 0;
             hemisphere < target.memory().hemispheres; ++hemisphere) {
            const int64_t unit =
                hemisphere * target.throughput().mxms_per_hemisphere;
            const int64_t activationStream = activationStreamFor(hemisphere);
            for (int64_t stream = 0; stream < 8; ++stream) {
                const int64_t slice = weightSlices[stream];
                emitMem3D(rewriter, op.getLoc(),
                    weightCycle - weightReadLatency(slice)
                        + (columnLoad ? stream : 0),
                    hemisphere * target.memory().slices_per_hemisphere + slice,
                    "read", weightBase + (columnLoad ? 3 : 0),
                    weightStreamBase + (columnLoad ? 0 : stream),
                    4, columnLoad ? 8 : 1, columnLoad ? -1 : 1,
                    "sram", weightBinding,
                    reductionBlocks, period, 4,
                    outputGroups, groupPeriod, reductionBlocks * 4,
                    weightPlacement.getAs<mlir::IntegerAttr>("bank").getInt(),
                    weightPage);
            }
            emitMxmDequant3D(rewriter, op.getLoc(), weightCycle, unit,
                scaleFor(hemisphere), columnLoad ? 32 : 4, 1,
                reductionBlocks, period, outputGroups, groupPeriod,
                scaleBinding);
            const projection_detail::ProjectionGemmSchedule::Binding binding{
                unit, 0, activationStream, resultStream, accumulatorAddress, dataFormat};
            loadSchedule.emitLoad(rewriter, op.getLoc(), weightCycle, binding,
                columnLoad ? 32 : 4, columnLoad,
                weightInputMode, weightStreamBase, "sram");
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = inputSlices[byte];
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::Mem,
                    target::StreamEndpoint::MxmActivation,
                    target::StreamDirection::East, slice);
                if (!latency) return mlir::failure();
                emitMem3D(rewriter, op.getLoc(), computeCycle - *latency,
                    hemisphere * target.memory().slices_per_hemisphere + slice,
                    "read", inputBase, activationStream + byte,
                    m, 1, 1, "sram", inputBinding,
                    reductionBlocks, period, m,
                    outputGroups, groupPeriod, 0,
                    inputPlacement.getAs<mlir::IntegerAttr>("bank").getInt());
            }
            MxmDomain3D computeDomain;
            computeDomain.repeat_count = tile;
            computeDomain.wave_count = tokenBlocks;
            computeDomain.wave_interval = tile;
            // MXM's row index wraps at 32. The next token tile must select
            // a different ACC block instead of reusing the first 32 rows.
            computeDomain.wave_accumulator_address_stride = tile;
            computeDomain.group_count = reductionBlocks;
            computeDomain.group_interval = period;
            computeDomain.weight_buffer_mode = "toggle_dim2";
            computeDomain.terminal_dimension = 2;
            computeDomain.terminal_accumulator_destination = "stream";
            computeDomain.terminal_accumulator_clear = true;
            computeDomain.terminal_accumulator_output_format = dataFormat;
            for (int64_t outputGroup = 0;
                 outputGroup < outputGroups; ++outputGroup) {
                emitMxm3D(rewriter, op.getLoc(),
                    computeCycle + outputGroup * groupPeriod, unit,
                    "compute", 0, 0, activationStream, resultStream,
                    accumulatorAddress, 1, "sram", false, "block", 0,
                    dataFormat, {}, {}, computeDomain);
            }
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = resultSlices[byte];
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::MxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::West, slice);
                if (!latency) return mlir::failure();
                emitMem3D(rewriter, op.getLoc(),
                    computeCycle + (reductionBlocks - 1) * period
                        + target.mxm_first_result_latency() + *latency,
                    hemisphere * target.memory().slices_per_hemisphere + slice,
                    "write", resultBase,
                    target.streams().streams_per_direction + resultStream + byte,
                    m, 1, 1, "sram", -1,
                    outputGroups, groupPeriod, m, 1, 1, 0, resultBank);
            }
        }
        mlir::OperationState timeline(op.getLoc(), TimelineOp::getOperationName());
        timeline.addAttributes({
            rewriter.getNamedAttr("name", rewriter.getStringAttr("linear_projection")),
            rewriter.getNamedAttr("start", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("end", rewriter.getI64IntegerAttr(
                computeCycle + outputGroups * groupPeriod + 64))});
        rewriter.create(timeline);
        return createBinding(rewriter, op.getLoc(), {}, outputIndex,
            "output", "result", resultType, resultPlacement).getValue();
    }

    const bool columnLoad = localDequant
        && target.throughput().mxm_prefill_weight_streams == 1;
    const int64_t columnWeightStream = target.streams().streams_per_direction
        - target.throughput().mxm_int8_load_streams_per_cycle;
    int64_t phaseStart = stagingEnd;
    for (int64_t outputGroup = 0;
         outputGroup < outputGroups; ++outputGroup) {
        for (int64_t reduction = 0;
             reduction < reductionBlocks; ++reduction) {
            const int64_t dequantStart =
                phaseStart + maxWeightLatency;
            const int64_t firstCompute = localDequant
                ? dequantStart + (columnLoad ? 32 : target.throughput().mxm_local_load_to_compute_latency)
                : dequantStart + vectorWeightLoadLead;
            const int64_t weightBuffer =
                (outputGroup * reductionBlocks + reduction)
                % target.throughput().mxm_weight_buffers;
            for (int64_t hemisphere = 0;
                 hemisphere < target.memory().hemispheres;
                 ++hemisphere) {
                const char* hemi =
                    hemisphere == 0 ? "east" : "west";
                const int64_t cycle = dequantStart + (localDequant ? 0
                    : hemisphere * target.throughput().lanes_per_tile);
                const int64_t address = weightBase
                    + (outputGroup * reductionBlocks + reduction) * 4;
                for (int64_t stream = 0; stream < 8; ++stream) {
                    const int64_t slice = weightSlices[stream];
                    emitMem3D(rewriter, op.getLoc(), cycle - weightReadLatency(slice) + (columnLoad ? stream : 0),
                        hemisphere * target.memory().slices_per_hemisphere + slice,
                        "read", address + (columnLoad ? 3 : 0),
                        columnLoad ? columnWeightStream : localDequant ? localMxm * target.throughput().mxm_int8_load_streams_per_cycle + stream
                            : target.streams().streams_per_direction + stream,
                        4, columnLoad ? 8 : 1, columnLoad ? -1 : 1, "sram", weightBinding, 1, 1, 0, 1, 1, 0,
                        weightPlacement.getAs<mlir::IntegerAttr>("bank").getInt(), weightPage);
                }
                const int64_t unit = hemisphere * target.throughput().mxms_per_hemisphere + localMxm;
                if (localDequant) {
                    emitMxmDequant(rewriter, op.getLoc(), cycle, unit, scaleFor(hemisphere), columnLoad ? 32 : 4, 1, scaleBinding);
                } else {
                    for (int64_t lane = 0; lane < target.throughput().lanes_per_tile; ++lane) {
                        const int64_t multiplyColumn = hemisphere == 0 ? lane
                            : target.throughput().vxm_alus - 1 - lane;
                        const int64_t castColumn = hemisphere == 0 ? 8 + lane
                            : target.throughput().vxm_alus - 1 - (8 + lane);
                        emitVxmConfigured(rewriter, op.getLoc(), op.getWeight(), cycle,
                            multiplyColumn, "multiply", "stream_i8",
                            target.streams().streams_per_direction + lane, 0.0f,
                            "immediate", 0, scale, "fp32", -1, hemi, hemi,
                            weightBinding, 8, 4, 1);
                        emitVxmConfigured(rewriter, op.getLoc(), op.getWeight(), cycle + 1,
                            castColumn, "cast", "alu", multiplyColumn, 0.0f,
                            "immediate", 0, 0.0f, dataFormat,
                            localMxm * target.throughput().mxm_load_streams_per_cycle + lane * 2,
                            hemi, hemi, -1, 8, 4, 1);
                    }
                }
                MxmDomain3D loadDomain;
                loadDomain.repeat_count = columnLoad ? 32 : 4;
                loadDomain.repeat_weight_column_stride = columnLoad ? 1 : -1;
                emitMxm3D(rewriter, op.getLoc(), cycle + (localDequant ? 0 : weightToIw),
                    unit, "iw", weightBuffer, columnLoad ? 0 : 3, 0, 0, 0, 1,
                    localDequant ? "sram" : "stream", true, columnLoad ? "vector" : "block", 0,
                    dataFormat, weightInputMode, {}, loadDomain,
                    columnLoad ? columnWeightStream : -1);
            }

            const bool finalReduction =
                reduction + 1 == reductionBlocks;
            for (int64_t tokenBlock = 0;
                 tokenBlock < tokenBlocks; ++tokenBlock) {
                const int64_t computeCycle =
                    firstCompute + tokenBlock * tile;
                for (int64_t hemisphere = 0;
                     hemisphere < target.memory().hemispheres;
                     ++hemisphere) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = inputSlices[byte];
                        const auto latency = target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmActivation,
                            target::StreamDirection::East, slice);
                        if (!latency) return mlir::failure();
                        emitMem(rewriter, op.getLoc(),
                            computeCycle - *latency,
                            hemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + slice,
                            "read",
                            inputBase + reduction * m
                                + tokenBlock * tile,
                            hemisphere * 2 + byte, tile, 1, 1,
                            "sram", inputBinding,
                            inputPlacement.getAs<mlir::IntegerAttr>("bank").getInt());
                    }
                    emitMxm(rewriter, op.getLoc(), computeCycle,
                        hemisphere
                                * target.throughput()
                                      .mxms_per_hemisphere
                            + localMxm,
                        "compute", weightBuffer, 0,
                        hemisphere * 2, 0, tile, 1,
                        accumulatorAddress + tokenBlock * tile,
                        1, "sram", false, "block", 0,
                        dataFormat);
                }
            }
            phaseStart = firstCompute + tokenBlocks * tile;
            if (!finalReduction) continue;

            const int64_t castStart = phaseStart
                + accumulatorLatency
                + target.throughput().accumulator_read_to_vxm_latency;
            for (int64_t hemisphere = 0;
                 hemisphere < target.memory().hemispheres;
                 ++hemisphere) {
                const int64_t outputStream = 0;
                const int64_t readCycle = phaseStart + accumulatorLatency;
                MxmDomain3D readDomain;
                readDomain.repeat_count = m;
                readDomain.repeat_accumulator_address_stride = 1;
                emitMxm3D(rewriter, op.getLoc(), readCycle,
                    hemisphere * target.throughput().mxms_per_hemisphere + localMxm,
                    "accumulator_read", 0, 0, 0, outputStream, accumulatorAddress, 1,
                    "stream", true, "block", 0, dataFormat, {}, dataFormat, readDomain);
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t slice = resultSlices[byte];
                    const auto latency = target.transport_latency(target::StreamEndpoint::MxmResult,
                        target::StreamEndpoint::Mem, target::StreamDirection::West, slice);
                    if (!latency) return mlir::failure();
                    emitMem3D(rewriter, op.getLoc(), readCycle + *latency,
                        hemisphere * target.memory().slices_per_hemisphere + slice, "write",
                        resultBase + outputGroup * m,
                        target.streams().streams_per_direction + outputStream + byte,
                        m, 1, 1, "sram", -1, 1, 1, 0, 1, 1, 0, resultBank);
                }
            }

            int64_t finalWriteLatency = 0;
            for (int64_t slice : resultSlices) {
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::MxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::West, slice);
                if (!latency) return mlir::failure();
                finalWriteLatency =
                    std::max(finalWriteLatency, *latency);
            }
            phaseStart = castStart + m + finalWriteLatency;
        }
    }

    mlir::OperationState timeline(
        op.getLoc(), TimelineOp::getOperationName());
    timeline.addAttributes({
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("linear_projection")),
        rewriter.getNamedAttr(
            "start", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr(
            "end", rewriter.getI64IntegerAttr(phaseStart)),
    });
    rewriter.create(timeline);
    auto output = createBinding(rewriter, op.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType()),
        resultPlacement);
    return output.getValue();
}

} // namespace

mlir::LogicalResult lowerLinearProjectionSchedules(
    mlir::IRRewriter& rewriter, mlir::func::FuncOp function,
    const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::ProjectionTaskOp> projections;
    function.walk([&](stream::ProjectionTaskOp op) {
        if (op.getKind() == "linear") projections.push_back(op);
    });
    int64_t outputIndex = 0;
    for (stream::ProjectionTaskOp projection : projections) {
        auto result = emitLinearProjection(
            rewriter, projection, target, outputIndex++);
        if (mlir::failed(result)) return mlir::failure();
        rewriter.replaceOp(projection, *result);
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
