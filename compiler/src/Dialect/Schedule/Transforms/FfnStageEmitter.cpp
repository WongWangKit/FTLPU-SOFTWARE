#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>

namespace ftlpu::compiler::schedule::ffn_detail {

int64_t FfnEmissionContext::westLatency(int64_t slice) const
{
    const auto latency = target.transport_latency(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::VxmInput,
        target::StreamDirection::West, slice);
    assert(latency && "validated FFN slice must have a MEM-to-VXM route");
    return *latency;
}

int64_t FfnEmissionContext::eastMxmLatency(int64_t slice) const
{
    const auto latency = target.transport_latency(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight,
        target::StreamDirection::East, slice);
    assert(latency && "validated FFN slice must have a MEM-to-MXM route");
    return *latency;
}

llvm::StringRef FfnEmissionContext::hemisphereName(
    int64_t hemisphere) const
{
    return hemisphere == 0 ? "east" : "west";
}

schedule::MemReadOp FfnEmissionContext::emitSliceRead(
    mlir::Value value, stream::RouteOp route, int64_t cycle,
    int64_t slice, int64_t base, int64_t count, int64_t stride,
    int64_t stream, llvm::StringRef direction, llvm::StringRef role,
    llvm::StringRef hemisphere)
{
    int64_t bank = route.getPlacement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    if (auto write = value.getDefiningOp<schedule::MemWriteOp>())
        bank = write.getPlacement()
            .getAs<mlir::IntegerAttr>("bank").getInt();
    auto placement = schedule_placement(rewriter, {slice}, base, count,
        stride, hemisphere, "schedule_slice", bank);
    mlir::NamedAttrList placementAttrs(placement);
    placementAttrs.set("binding_placement", route.getPlacement());
    return rewriter.create<schedule::MemReadOp>(ffn.getLoc(), value,
        cycle, count, stream, 1,
        slice / target.streams().mem_slices_per_register_group + 1,
        rewriter.getStringAttr(direction), rewriter.getStringAttr(role),
        route.getAddress(),
        placementAttrs.getDictionary(rewriter.getContext()),
        count * tile());
}

mlir::FailureOr<std::unique_ptr<FfnEmissionContext>>
createFfnEmissionContext(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& ffn, FfnScheduleStrategy strategy,
    const target::LPUTargetModel& target)
{
    if (mlir::failed(ffn.task_plan.tasks.validate())
        || !target.supports_w8a16_ffn_shape(
            ffn.getM(), ffn.getK(), ffn.getHidden(), ffn.getN())) {
        ffn.getOperation()->emitError(
            "FFN task DAG or target shape is invalid");
        return mlir::failure();
    }

    auto activationRoute =
        ffn.getActivation().getDefiningOp<stream::RouteOp>();
    auto gateRoute = ffn.getGateWeight().getDefiningOp<stream::RouteOp>();
    auto upRoute = ffn.getUpWeight().getDefiningOp<stream::RouteOp>();
    auto downRoute =
        ffn.getDownWeight0().getDefiningOp<stream::RouteOp>();
    auto hiddenRoute = ffn.hidden0_route;
    if (!activationRoute || !gateRoute || !upRoute || !downRoute
        || !hiddenRoute) {
        ffn.getOperation()->emitError(
            "FFN schedule is missing a required stream route");
        return mlir::failure();
    }

    const auto rawRoute = [](stream::RouteOp route) {
        auto dequant =
            route.getInput().getDefiningOp<stream::DequantizeOp>();
        return dequant
            ? dequant.getInput().getDefiningOp<stream::RouteOp>()
            : route;
    };
    auto gateRaw = rawRoute(gateRoute);
    auto upRaw = rawRoute(upRoute);
    auto downRaw = rawRoute(downRoute);
    if (!gateRaw || !upRaw || !downRaw) {
        ffn.getOperation()->emitError(
            "FFN schedule cannot resolve raw weight routes");
        return mlir::failure();
    }

    auto weightSlices = get_slices(gateRaw.getPlacement());
    auto upWeightSlices = get_slices(upRaw.getPlacement());
    auto downWeightSlices = get_slices(downRaw.getPlacement());
    auto activationSlices = get_slices(activationRoute.getPlacement());
    const auto activationKind =
        activationRoute.getPlacement().getAs<mlir::StringAttr>("kind");
    const bool activationDistributed16 = activationKind
        && activationKind.getValue() == "fp16_mxm_distributed_16";
    auto hiddenSlices = get_slices(ffn.getHidden0Placement());
    const auto hiddenKind =
        ffn.getHidden0Placement().getAs<mlir::StringAttr>("kind");
    const bool hiddenDistributed16 = hiddenKind
        && hiddenKind.getValue() == "fp16_mxm_distributed_16";
    const auto pagedWeightsAttr = gateRaw.getPlacement()
        .getAs<mlir::BoolAttr>("paged_weight");
    const bool pagedWeights = pagedWeightsAttr && pagedWeightsAttr.getValue();
    const auto& memory = target.memory();
    const int64_t hiddenBank = ffn.getHidden0Placement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const int64_t tempBank = pagedWeights && memory.banks_per_slice > 1
        ? (hiddenBank + 1) % memory.banks_per_slice : hiddenBank;
    auto resultSlices = get_slices(ffn.getResultPlacement());
    auto executionPolicy =
        target::mxm_execution_policy_from_operation(
            ffn.getOperation());
    if (mlir::failed(executionPolicy)) {
        ffn.getOperation()->emitError("invalid FFN MXM execution policy");
        return mlir::failure();
    }
    const auto activationType = llvm::cast<mlir::RankedTensorType>(
        ffn.getActivation().getType());
    auto execution = target::plan_mxm_execution_strategy(
        {static_cast<int64_t>(ffn.getM()),
            static_cast<int64_t>(ffn.getN()),
            static_cast<int64_t>(ffn.getHidden()),
            activationType.getElementType().isBF16(), true, true, true},
        target, *executionPolicy);
    if (mlir::failed(execution)) {
        ffn.getOperation()->emitError(
            "cannot plan the FFN MXM execution strategy");
        return mlir::failure();
    }
    const bool block8Ffn = execution->uses_block8();
    const auto& throughput = target.throughput();
    // The legacy path does not model passive stream-fabric transport
    // lifetimes precisely enough to prove a fused route. Keep its requested
    // fused mode on the validated tail baseline.
    if (!block8Ffn && strategy == FfnScheduleStrategy::Fused)
        strategy = FfnScheduleStrategy::Tail;
    if (weightSlices.size()
            != static_cast<std::size_t>(
                memory.w8a16_weight_slice_count)
        || upWeightSlices.size()
            != static_cast<std::size_t>(
                memory.w8a16_weight_slice_count)
        || (block8Ffn && !activationDistributed16)
        || (!activationDistributed16
            && activationSlices.size()
                != static_cast<std::size_t>(
                    throughput.mxm_activation_streams))
        || (activationDistributed16
            && activationSlices.size() != 16)
        || downWeightSlices.size()
            != static_cast<std::size_t>(
                block8Ffn
                    ? throughput.mxms_per_hemisphere
                        * memory.w8a16_weight_slice_count
                    : memory.w8a16_weight_slice_count)
        || hiddenSlices.size()
            != static_cast<std::size_t>(
                (block8Ffn || hiddenDistributed16) ? 16
                           : throughput.mxm_activation_streams)
        || resultSlices.size()
            != static_cast<std::size_t>(
                block8Ffn ? 16
                           : throughput.mxm_result_streams))
    {
        ffn.getOperation()->emitError(
            "FFN physical slices do not match the selected MXM strategy");
        return mlir::failure();
    }

    auto activationLatency = target.transport_latency(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, activationSlices.front());
    auto projectionTimeline = planFfnProjectionTimeline(
        {static_cast<int64_t>(ffn.getM()),
            static_cast<int64_t>(ffn.getK()),
            static_cast<int64_t>(ffn.getHidden()),
            static_cast<int64_t>(ffn.getN())},
        weightSlices, target, execution->uses_local_dequant(),
        !block8Ffn && throughput.mxms_per_hemisphere != 1);
    if (!activationLatency || mlir::failed(projectionTimeline)) {
        ffn.getOperation()->emitError(
            "cannot plan FFN activation transport or projection timeline");
        return mlir::failure();
    }

    const int64_t tile = throughput.mxm_rows;
    // Projection final partials are drained to MEM and clear their rows.
    // Down projection starts afterwards and can reuse the same token window.
    const int64_t downAccumulatorBase = 0;
    auto projectionType = mlir::RankedTensorType::get(
        {tile, tile}, rewriter.getF32Type());

    std::optional<stream::StreamBinding> fusedOutput;
    if (block8Ffn && strategy == FfnScheduleStrategy::Fused) {
        stream::StreamAllocator allocator(target);
        const int64_t projectionStreamBase = 0;
        const int64_t projectionStreamCount =
            throughput.mxms_per_hemisphere == 1
            ? throughput.mxm_int8_load_streams_per_cycle
            : 24;
        if (mlir::succeeded(allocator.reserve(
                target::StreamDirection::East, projectionStreamBase,
                projectionStreamCount, 0, 1))) {
            auto output = allocator.allocate(
                target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem,
                target::StreamDirection::East,
                hiddenSlices.front(), 0, 1, 2);
            if (mlir::succeeded(output)) fusedOutput = *output;
        }
        if (!fusedOutput) strategy = FfnScheduleStrategy::Tail;
    }

    return std::make_unique<FfnEmissionContext>(FfnEmissionContext {
        rewriter,
        ffn,
        strategy,
        target,
        activationRoute,
        gateRoute,
        upRoute,
        downRoute,
        hiddenRoute,
        gateRaw,
        upRaw,
        downRaw,
        std::move(weightSlices),
        std::move(upWeightSlices),
        std::move(downWeightSlices),
        std::move(activationSlices),
        std::move(hiddenSlices),
        std::move(resultSlices),
        std::move(*projectionTimeline),
        projectionType,
        *activationLatency,
        downAccumulatorBase,
        activationDistributed16,
        execution->uses_local_dequant(),
        block8Ffn,
        block8Ffn,
        tempBank,
        fusedOutput,
    });
}

} // namespace ftlpu::compiler::schedule::ffn_detail
