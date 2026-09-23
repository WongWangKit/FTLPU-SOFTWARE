#include <algorithm>

#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"

#include "FfnStageEmitter.hpp"
#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

namespace {

using namespace ftlpu::compiler;

mlir::DictionaryAttr withNative4ReadShape(mlir::IRRewriter& rewriter,
    mlir::DictionaryAttr bindingPlacement, llvm::ArrayRef<int64_t> slices,
    int64_t baseRow, int64_t rowCount, int64_t bank,
    llvm::StringRef hemisphere, bool weight, int64_t addressStride = 1)
{
    llvm::SmallVector<mlir::Attribute> sliceAttrs;
    for (int64_t slice : slices)
        sliceAttrs.push_back(rewriter.getI64IntegerAttr(slice));
    mlir::NamedAttrList attrs;
    attrs.set("kind", rewriter.getStringAttr(
        weight ? "w8a16_native4_weight" : "fp16_native4_activation"));
    attrs.set("hemisphere", rewriter.getStringAttr(hemisphere));
    attrs.set("slices", rewriter.getArrayAttr(sliceAttrs));
    attrs.set("base_row", rewriter.getI64IntegerAttr(baseRow));
    attrs.set("instruction_count", rewriter.getI64IntegerAttr(rowCount));
    attrs.set("address_stride", rewriter.getI64IntegerAttr(addressStride));
    attrs.set("bank", rewriter.getI64IntegerAttr(bank));
    attrs.set("binding_placement", bindingPlacement);
    if (weight)
        attrs.set("weight_page", rewriter.getI64IntegerAttr(0));
    return attrs.getDictionary(rewriter.getContext());
}

schedule::MxmIssueOp emitNative4Mxm(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t unit,
    llvm::StringRef opcode, int64_t buffer, int64_t accumulator,
    int64_t outputStream, int64_t repeatCount, int64_t repeatInterval,
    int64_t repeatAccumulatorStride, int64_t groupCount,
    int64_t groupInterval)
{
    mlir::OperationState state(location,
        schedule::MxmIssueOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(unit)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("weight_buffer", rewriter.getI64IntegerAttr(buffer)),
        rewriter.getNamedAttr("weight_column", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("activation_stream_base", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(outputStream)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeatCount)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeatInterval)),
        rewriter.getNamedAttr("repeat_accumulator_address_stride",
            rewriter.getI64IntegerAttr(repeatAccumulatorStride)),
        rewriter.getNamedAttr("accumulator_address", rewriter.getI64IntegerAttr(accumulator)),
        rewriter.getNamedAttr("accumulator_row_stride", rewriter.getI64IntegerAttr(1)),
        rewriter.getNamedAttr("accumulator_destination", rewriter.getStringAttr("sram")),
        rewriter.getNamedAttr("accumulator_clear", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("data_format", rewriter.getStringAttr("bf16")),
        rewriter.getNamedAttr("decode_layout", rewriter.getStringAttr("native4")),
        rewriter.getNamedAttr("group_count", rewriter.getI64IntegerAttr(groupCount)),
        rewriter.getNamedAttr("group_interval", rewriter.getI64IntegerAttr(groupInterval)),
    });
    return llvm::cast<schedule::MxmIssueOp>(rewriter.create(state));
}

schedule::MxmDequantOp emitNative4Scale(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t cycle, int64_t unit,
    float scale, int64_t scaleBinding, int64_t repeatCount,
    int64_t repeatInterval, int64_t groupCount, int64_t groupInterval)
{
    mlir::OperationState state(location,
        schedule::MxmDequantOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(unit)),
        rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeatCount)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeatInterval)),
        rewriter.getNamedAttr("group_count", rewriter.getI64IntegerAttr(groupCount)),
        rewriter.getNamedAttr("group_interval", rewriter.getI64IntegerAttr(groupInterval)),
    });
    if (scaleBinding >= 0)
        state.addAttribute("scale_binding", rewriter.getI64IntegerAttr(scaleBinding));
    return llvm::cast<schedule::MxmDequantOp>(rewriter.create(state));
}

std::optional<int64_t> native4BindingIndex(mlir::Value value)
{
    for (int64_t depth = 0; depth < 16; ++depth) {
        if (const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
            return argument.getArgNumber();
        mlir::Operation* definition = value.getDefiningOp();
        if (!definition || definition->getNumOperands() != 1)
            return std::nullopt;
        value = definition->getOperand(0);
    }
    return std::nullopt;
}

struct Native4ProjectionResult {
    int64_t endCycle{0};
    mlir::Value value;
};

int64_t native4WeightTransferCycles(
    const target::LPUTargetModel& target, int64_t k, int64_t n)
{
    const int64_t bytes = k * n;
    const int64_t lanes = target.streams().c2c_streams_per_direction;
    const int64_t bytesPerLane =
        target.streams().c2c_bytes_per_stream_per_cycle;
    const int64_t sideBytes = (bytes + 1) / 2;
    const int64_t c2cCycles =
        (sideBytes + lanes * bytesPerLane - 1)
        / (lanes * bytesPerLane);
    const auto& external = target.external_memory();
    const int64_t queueDrain =
        (external.ddr_request_queue_depth + lanes - 1) / lanes;
    const int64_t transportGuard = queueDrain
        + target.streams().mem_boundary_register_columns
        + target.throughput().tile_rows + lanes;
    return std::max(target.external_read_transfer_cycles(bytes), c2cCycles)
        + transportGuard;
}

mlir::FailureOr<Native4ProjectionResult> emitNative4Projection(
    mlir::IRRewriter& rewriter, schedule::PrimitiveFfnSchedulePlan& plan,
    const target::LPUTargetModel& target, stream::RouteOp activationRoute,
    stream::RouteOp weightRoute, mlir::Value activationValue,
    int64_t startCycle, int64_t k, int64_t n, float scale,
    llvm::ArrayRef<int64_t> outputSlices, int64_t outputBank,
    int64_t outputBaseRow, bool activationDistributed16)
{
    const int64_t reductions = k / 32;
    const int64_t localWaves = n / 64;
    if (reductions <= 0 || localWaves <= 0 || k % 64 != 0 || n % 64 != 0
        || outputSlices.size() < 4)
        return mlir::failure();
    const int64_t waveStages = target.throughput().tile_rows
        + target.throughput().mxm_block_rows / 2 + 2;
    const int64_t reductionInterval = localWaves + waveStages + 4;
    auto weightSlices = schedule::ffn_detail::get_slices(weightRoute.getPlacement());
    if (weightSlices.size() != 32) return mlir::failure();
    int64_t weightReadLatency = 0;
    for (int64_t slice : weightSlices) {
        const auto latency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        if (!latency) return mlir::failure();
        weightReadLatency = std::max(weightReadLatency, *latency);
    }
    const int64_t weightReadOffset = 32;
    const int64_t computeOffset = weightReadOffset + weightReadLatency;
    const int64_t weightBank = weightRoute.getPlacement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t parityCount = reductions / 2;
    const int64_t parityPhaseSpan = parityCount * reductionInterval;
    constexpr int64_t native4ColumnStreams = 8;
    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        // A Native4 wave traverses the 4x4 array diagonally.  Stream columns
        // 0..3 therefore have to enter on four consecutive cycles.  Reading
        // all 32 streams on every wave cycle overwrites columns that the
        // preceding wave has not consumed yet.  Keep four closed-form 3D
        // domains (one per physical stream column), each spanning all output
        // waves and reductions.
        for (int64_t column = 0;
             column < target.throughput().tile_rows; ++column) {
            const auto columnSlices = llvm::ArrayRef<int64_t>(weightSlices)
                .slice(column * native4ColumnStreams,
                    native4ColumnStreams);
            int64_t columnLatency = 0;
            for (int64_t slice : columnSlices) {
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::Mem,
                    target::StreamEndpoint::MxmWeight,
                    target::StreamDirection::East, slice);
                if (!latency) return mlir::failure();
                columnLatency = std::max(columnLatency, *latency);
            }
            auto weightPlacement = withNative4ReadShape(
                rewriter, weightRoute.getPlacement(), columnSlices, 0,
                localWaves, weightBank,
                schedule::ffn_detail::hemisphere_name(hemisphere), true,
                reductions);
            auto weightRead = rewriter.create<schedule::MemReadOp>(
                plan.getLoc(), weightRoute.getInput(),
                startCycle + computeOffset + column - columnLatency,
                localWaves, column * native4ColumnStreams,
                native4ColumnStreams, 0, rewriter.getStringAttr("east"),
                rewriter.getStringAttr("weight"), weightRoute.getAddress(),
                weightPlacement,
                reductions * localWaves * native4ColumnStreams);
            // Each hemisphere has its own physical MEM and MXM ICUs.  Emit
            // the same four domains independently on each side.
            weightRead->setAttr(
                "wave_count", rewriter.getI64IntegerAttr(parityCount));
            weightRead->setAttr("wave_interval",
                rewriter.getI64IntegerAttr(reductionInterval));
            weightRead->setAttr("wave_address_stride",
                rewriter.getI64IntegerAttr(2));
            weightRead->setAttr(
                "group_count", rewriter.getI64IntegerAttr(2));
            weightRead->setAttr("group_interval",
                rewriter.getI64IntegerAttr(parityPhaseSpan));
            weightRead->setAttr("group_address_stride",
                rewriter.getI64IntegerAttr(1));
        }
    }

    const int64_t activationBank = activationRoute.getPlacement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t activationBase = activationRoute.getPlacement()
        .getAs<mlir::IntegerAttr>("base_row").getInt();
    auto activationSlices = schedule::ffn_detail::get_slices(activationRoute.getPlacement());
    if (activationDistributed16) {
        if (activationSlices.size() < 2) return mlir::failure();
        activationSlices.resize(2);
    } else if (activationSlices.size() < 4) {
        return mlir::failure();
    }
    int64_t activationReadLatency = 0;
    for (int64_t slice : activationSlices) {
        const auto latency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmActivation,
            target::StreamDirection::East, slice);
        if (!latency) return mlir::failure();
        activationReadLatency = std::max(activationReadLatency, *latency);
    }
    const int64_t activationLoadOffset =
        computeOffset - target.throughput().tile_rows;
    const int64_t activationReadOffset =
        activationLoadOffset - activationReadLatency;
    if (activationReadOffset < 0) return mlir::failure();
    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        const int64_t activationPairCount =
            activationDistributed16 ? 1 : 2;
        const int64_t wavesPerPair = parityCount / activationPairCount;
        if (wavesPerPair <= 0
            || wavesPerPair * activationPairCount != parityCount)
            return mlir::failure();
        for (int64_t pair = 0; pair < activationPairCount; ++pair) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                schedule::ffn_detail::FfnLoopDomain3D domain;
                const int64_t reductionAddressStride =
                    activationDistributed16 ? 4 : 1;
                domain.wave_count = wavesPerPair;
                domain.wave_interval = reductionInterval;
                domain.wave_address_stride =
                    2 * reductionAddressStride;
                domain.group_count = 2;
                domain.group_interval = parityPhaseSpan;
                domain.group_address_stride = reductionAddressStride;
                const int64_t slice = activationDistributed16
                    ? activationSlices[byte]
                    : activationSlices[2 * pair + byte];
                schedule::ffn_detail::emitFfnMemTransfer3D(rewriter,
                    plan.getLoc(), startCycle + activationReadOffset
                        + pair * wavesPerPair * reductionInterval,
                    hemisphere, slice, "read", activationBase, byte,
                    1, 1, 1, activationBank, domain);
            }
        }
        for (int64_t parity = 0; parity < 2; ++parity) {
            const int64_t phaseStart =
                startCycle + parity * parityPhaseSpan;
            emitNative4Scale(rewriter, plan.getLoc(),
                phaseStart + computeOffset, hemisphere,
                scale, native4BindingIndex(weightRoute.getInput()).value_or(-1),
                localWaves, 1, parityCount, reductionInterval);
            // DecodeLoadActivation and DecodeStreamCompute both execute on the
            // physical MXM compute ICU.  Since that ICU has one loop context,
            // materialize the alternating pair per reduction instead of
            // interleaving two long coarse domains.
            for (int64_t reduction = 0; reduction < parityCount;
                 ++reduction) {
                const int64_t loadCycle = phaseStart
                    + reduction * reductionInterval
                    + activationLoadOffset;
                emitNative4Mxm(rewriter, plan.getLoc(), loadCycle, hemisphere,
                    "decode_load_activation", parity, 0, 0,
                    1, 1, 0, 1, 1);
                const int64_t computeCycle =
                    loadCycle + target.throughput().tile_rows;
                emitNative4Mxm(rewriter, plan.getLoc(), computeCycle,
                    hemisphere, "decode_stream_compute", parity, 0, 0,
                    localWaves, 1, 1, 1, 1);
            }
        }
    }

    const int64_t drainCycle = startCycle
        + (reductions - 1) * reductionInterval
        + computeOffset + localWaves + waveStages;
    mlir::Value result = activationValue;
    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        mlir::OperationState drainState(plan.getLoc(),
            schedule::MxmIssueOp::getOperationName());
        drainState.addAttributes({
            rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(drainCycle)),
            rewriter.getNamedAttr("unit_id", rewriter.getI64IntegerAttr(hemisphere)),
            rewriter.getNamedAttr("opcode", rewriter.getStringAttr("accumulator_read")),
            rewriter.getNamedAttr("weight_buffer", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("weight_column", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("activation_stream_base", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(localWaves)),
            rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
            rewriter.getNamedAttr("repeat_accumulator_address_stride", rewriter.getI64IntegerAttr(1)),
            rewriter.getNamedAttr("accumulator_address", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("accumulator_row_stride", rewriter.getI64IntegerAttr(1)),
            rewriter.getNamedAttr("accumulator_destination", rewriter.getStringAttr("stream")),
            rewriter.getNamedAttr("accumulator_clear", rewriter.getBoolAttr(true)),
            rewriter.getNamedAttr("data_format", rewriter.getStringAttr("bf16")),
            rewriter.getNamedAttr("accumulator_output_format", rewriter.getStringAttr("bf16")),
        });
        rewriter.create(drainState);
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = outputSlices[2 * hemisphere + byte];
            const auto latency = target.transport_latency(
                target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem,
                target::StreamDirection::West, slice);
            if (!latency) return mlir::failure();
            auto placement = schedule::ffn_detail::schedule_placement(
                rewriter, {slice}, outputBaseRow, localWaves, 1,
                schedule::ffn_detail::hemisphere_name(hemisphere),
                "fp16_native4_stage", outputBank);
            auto write = rewriter.create<schedule::MemWriteOp>(plan.getLoc(),
                activationValue, drainCycle + *latency, localWaves,
                byte, 1, 0, rewriter.getStringAttr("west"),
                plan.getHidden0Address(), placement, localWaves * 32);
            result = write.getOutput();
        }
    }
    return Native4ProjectionResult {drainCycle + localWaves + 20, result};
}

mlir::FailureOr<mlir::Value> lowerNative4Ffn(
    mlir::IRRewriter& rewriter, schedule::PrimitiveFfnSchedulePlan& plan,
    const target::LPUTargetModel& target)
{
    const auto activationKind = plan.activation_route.getPlacement()
        .getAs<mlir::StringAttr>("kind");
    const bool distributed = activationKind
        && activationKind.getValue() == "fp16_mxm_distributed_16";
    const std::array<int64_t, 4> gateSlices {0, 1, 2, 3};
    const std::array<int64_t, 4> upSlices {4, 5, 6, 7};
    auto gate = emitNative4Projection(rewriter, plan, target,
        plan.activation_route, plan.gate_route,
        plan.activation_route.getInput(), 16, plan.getK(), plan.getHidden(),
        plan.getGateScale().convertToFloat(), gateSlices, 0, 0, distributed);
    if (mlir::failed(gate)) return mlir::failure();
    const int64_t upTransferCycles = native4WeightTransferCycles(
        target, plan.getK(), plan.getHidden());
    auto up = emitNative4Projection(rewriter, plan, target,
        plan.activation_route, plan.up_route,
        plan.activation_route.getInput(), gate->endCycle + upTransferCycles,
        plan.getK(), plan.getHidden(), plan.getUpScale().convertToFloat(),
        upSlices, 0, 0, distributed);
    if (mlir::failed(up)) return mlir::failure();

    const int64_t localWaves = plan.getHidden() / 64;
    const int64_t hiddenBank = plan.getHidden0Placement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t hiddenBase = schedule::ffn_detail::get_base_row(plan.getHidden0Placement());
    auto hiddenSlices = schedule::ffn_detail::get_slices(plan.getHidden0Placement());
    if (hiddenSlices.size() < 4) return mlir::failure();
    const int64_t swishCycle = up->endCycle + 32;
    mlir::Value hidden = up->value;
    std::array<mlir::Value, 2> swishOwnerValues;
    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        // Both hemispheres feed the same physical eight-stage VXM chain.
        // Keep each hemisphere as one contiguous coarse command and place the
        // second domain immediately after the first one has drained.
        const int64_t hemisphereSwishCycle =
            swishCycle + hemisphere * localWaves;
        int64_t swishInputLatency = 0;
        for (int64_t byte = 0; byte < 2; ++byte) {
            for (int64_t slice : {
                     gateSlices[2 * hemisphere + byte],
                     upSlices[2 * hemisphere + byte]}) {
                const auto latency = target.transport_latency(
                    target::StreamEndpoint::Mem,
                    target::StreamEndpoint::VxmInput,
                    target::StreamDirection::West, slice);
                if (!latency) return mlir::failure();
                swishInputLatency = std::max(swishInputLatency, *latency);
            }
        }
        const int64_t swishInputCycle =
            hemisphereSwishCycle + swishInputLatency;
        for (int64_t byte = 0; byte < 2; ++byte) {
            schedule::ffn_detail::FfnLoopDomain3D domain;
            domain.wave_count = localWaves;
            domain.wave_interval = 1;
            domain.wave_address_stride = 1;
            const int64_t gateSlice =
                gateSlices[2 * hemisphere + byte];
            const int64_t upSlice =
                upSlices[2 * hemisphere + byte];
            const auto gateLatency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, gateSlice);
            const auto upLatency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, upSlice);
            if (!gateLatency || !upLatency) return mlir::failure();
            schedule::ffn_detail::emitFfnMemTransfer3D(rewriter,
                plan.getLoc(), swishInputCycle - *gateLatency, hemisphere,
                gateSlice, "read", 0,
                32 + hemisphere * 16 + byte, 1, 1, 1, 0, domain);
            schedule::ffn_detail::emitFfnMemTransfer3D(rewriter,
                plan.getLoc(), swishInputCycle - *upLatency, hemisphere,
                upSlice, "read", 0,
                34 + hemisphere * 16 + byte, 1, 1, 1, 0, domain);
        }
        auto outputs = schedule::ffn_detail::emitFfnSwishAlu(
            rewriter, plan.getLoc(), plan.hidden0_route.getInput().getType(),
            gate->value, up->value, target, FfnScheduleStrategy::Tail,
            swishInputCycle - 1, hemisphere, 6,
            localWaves, 1);
        const int64_t owner = 1 - hemisphere;
        const int64_t swishOutputStream =
            owner == 0 ? 14 : 6;
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = hiddenSlices[2 * hemisphere + byte];
            const auto latency = target.transport_latency(
                target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::East, slice);
            if (!latency) return mlir::failure();
            auto placement = schedule::ffn_detail::schedule_placement(
                rewriter, {slice}, hiddenBase, localWaves, 1,
                schedule::ffn_detail::hemisphere_name(owner),
                "fp16_mxm_activation_planar", hiddenBank);
            auto write = rewriter.create<schedule::MemWriteOp>(plan.getLoc(),
                outputs.first.getResult(),
                swishInputCycle + 17 + *latency,
                localWaves, swishOutputStream + byte, 1, 0,
                rewriter.getStringAttr("east"),
                plan.getHidden0Address(), placement, localWaves * 32);
            swishOwnerValues[hemisphere] = write.getOutput();
            hidden = write.getOutput();
        }
    }

    // The down projection needs the full hidden vector on both hemispheres.
    // Mirror each Swish half through the passive MEM stream path after the
    // shared VXM chain has drained.  This keeps queue 7 on its fixed 6/7
    // output group and avoids pretending that the VXM can broadcast east and
    // west in the same cycle.
    const int64_t mirrorBaseCycle = swishCycle + 2 * localWaves + 64;
    const int64_t mirrorStride = localWaves + 32;
    for (int64_t sourceHemisphere = 0; sourceHemisphere < 2;
         ++sourceHemisphere) {
        const int64_t owner = 1 - sourceHemisphere;
        const int64_t mirrorInputCycle =
            mirrorBaseCycle + sourceHemisphere * mirrorStride;
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = hiddenSlices[2 * sourceHemisphere + byte];
            const auto readLatency = target.transport_latency(
                target::StreamEndpoint::Mem,
                target::StreamEndpoint::VxmInput,
                target::StreamDirection::West, slice);
            const auto writeLatency = target.transport_latency(
                target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::East, slice);
            if (!readLatency || !writeLatency) return mlir::failure();
            auto readPlacement = schedule::ffn_detail::schedule_placement(
                rewriter, {slice}, hiddenBase, localWaves, 1,
                schedule::ffn_detail::hemisphere_name(owner),
                "schedule_slice", hiddenBank);
            mlir::NamedAttrList readAttrs(readPlacement);
            readAttrs.set("binding_placement", plan.getHidden0Placement());
            auto read = rewriter.create<schedule::MemReadOp>(plan.getLoc(),
                swishOwnerValues[sourceHemisphere],
                mirrorInputCycle - *readLatency, localWaves, byte, 1,
                slice / target.streams().mem_slices_per_register_group + 1,
                rewriter.getStringAttr("west"),
                rewriter.getStringAttr("vxm_bypass"),
                plan.getHidden0Address(),
                readAttrs.getDictionary(rewriter.getContext()),
                localWaves * 32);
            auto writePlacement = schedule::ffn_detail::schedule_placement(
                rewriter, {slice}, hiddenBase, localWaves, 1,
                schedule::ffn_detail::hemisphere_name(sourceHemisphere),
                "fp16_mxm_activation_planar", hiddenBank);
            auto write = rewriter.create<schedule::MemWriteOp>(plan.getLoc(),
                read.getOutput(), mirrorInputCycle + *writeLatency,
                localWaves, byte, 1, 0, rewriter.getStringAttr("east"),
                plan.getHidden0Address(), writePlacement, localWaves * 32);
            hidden = write.getOutput();
        }
    }
    const int64_t downTransferCycles = native4WeightTransferCycles(
        target, plan.getHidden(), plan.getN());
    const int64_t downStartCycle = std::max(
        mirrorBaseCycle + 2 * mirrorStride + 32,
        up->endCycle + downTransferCycles);
    auto down = emitNative4Projection(rewriter, plan, target,
        plan.hidden0_route, plan.down0_route, hidden,
        downStartCycle,
        plan.getHidden(), plan.getN(),
        plan.getDownRhsScale().convertToFloat(),
        schedule::ffn_detail::get_slices(plan.getResultPlacement()),
        plan.getResultPlacement().getAs<mlir::IntegerAttr>("bank").getInt(),
        schedule::ffn_detail::get_base_row(plan.getResultPlacement()), false);
    if (mlir::failed(down)) return mlir::failure();
    return down->value;
}

void shiftProjectionTimeline(
    ftlpu::compiler::schedule::FfnProjectionTimeline& timeline,
    int64_t offset)
{
    timeline.initial_compute_cycle += offset;
    timeline.final_projection_cycle += offset;
    for (auto& block : timeline.blocks) {
        block.weight_compute_cycle += offset;
        block.dequant_start += offset;
        for (auto& tile : block.tiles)
            tile.compute_cycle += offset;
    }
}

void emitAccumulatorClearPrelude(
    ftlpu::compiler::schedule::ffn_detail::FfnEmissionContext& context,
    int64_t rows)
{
    auto& rewriter = context.rewriter;
    rewriter.setInsertionPoint(context.ffn.getOperation());
    const auto& throughput = context.target.throughput();
    const int64_t unitCount =
        context.target.memory().hemispheres
        * throughput.mxms_per_hemisphere;
    const auto inputType = llvm::cast<mlir::RankedTensorType>(
        context.ffn.getActivation().getType());
    const llvm::StringRef dataFormat =
        ftlpu::compiler::lpu_16bit_data_format(inputType.getElementType());
    for (int64_t unit = 0; unit < unitCount; ++unit) {
        const int64_t outputStream =
            (unit % throughput.mxms_per_hemisphere)
            * throughput.mxm_result_streams;
        mlir::OperationState state(
            context.ffn.getLoc(),
            ftlpu::compiler::schedule::MxmIssueOp::getOperationName());
        state.addAttributes({
            rewriter.getNamedAttr("cycle",
                rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("unit_id",
                rewriter.getI64IntegerAttr(unit)),
            rewriter.getNamedAttr("opcode",
                rewriter.getStringAttr("accumulator_read")),
            rewriter.getNamedAttr("weight_buffer",
                rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("weight_column",
                rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("activation_stream_base",
                rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("output_stream_base",
                rewriter.getI64IntegerAttr(outputStream)),
            rewriter.getNamedAttr("repeat_count",
                rewriter.getI64IntegerAttr(rows)),
            rewriter.getNamedAttr("repeat_interval",
                rewriter.getI64IntegerAttr(1)),
            rewriter.getNamedAttr("repeat_accumulator_address_stride",
                rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("accumulator_address",
                rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("accumulator_row_stride",
                rewriter.getI64IntegerAttr(1)),
            rewriter.getNamedAttr("accumulator_destination",
                rewriter.getStringAttr("sram")),
            rewriter.getNamedAttr("accumulator_clear",
                rewriter.getBoolAttr(true)),
            rewriter.getNamedAttr("data_format",
                rewriter.getStringAttr(dataFormat)),
        });
        rewriter.create(state);
    }
}

} // namespace

namespace ftlpu::compiler {

mlir::FailureOr<mlir::Value> schedule::lowerFfnSchedule(
    mlir::IRRewriter& rewriter,
    schedule::PrimitiveFfnSchedulePlan& plan,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target)
{
    auto executionPolicy = target::mxm_execution_policy_from_operation(
        plan.getOperation());
    if (mlir::succeeded(executionPolicy)
        && *executionPolicy == target::MxmExecutionPolicy::Native4)
        return lowerNative4Ffn(rewriter, plan, target);
    auto context = schedule::ffn_detail::createFfnEmissionContext(
        rewriter, plan, strategy, target);
    if (mlir::failed(context)) {
        plan.getOperation()->emitError(
            "failed to create the FFN emission context");
        return mlir::failure();
    }
    const int64_t accumulatorRows =
        (*context)->down_accumulator_base + (*context)->m();
    emitAccumulatorClearPrelude(**context, accumulatorRows);
    const int64_t clearCycles = accumulatorRows
        + target.throughput().accumulator_read_to_vxm_latency + 1;
    shiftProjectionTimeline(
        (*context)->projection_timeline, clearCycles);

    auto projection =
        schedule::ffn_detail::emitFfnProjection(**context);
    if (mlir::failed(projection)) {
        plan.getOperation()->emitError(
            "failed to emit the FFN projection schedule");
        return mlir::failure();
    }
    auto swish = schedule::ffn_detail::emitFfnSwish(
        **context, std::move(*projection));
    if (mlir::failed(swish)) {
        plan.getOperation()->emitError(
            "failed to emit the FFN Swish schedule");
        return mlir::failure();
    }
    auto result = schedule::ffn_detail::emitFfnDownProjection(
        **context, *swish);
    if (mlir::failed(result))
        plan.getOperation()->emitError(
            "failed to emit the FFN down-projection schedule");
    return result;
}

} // namespace ftlpu::compiler
