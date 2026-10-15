#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>

namespace ftlpu::compiler::schedule::ffn_detail {
namespace {

FfnProjectionOrder chooseProjectionOrder(const target::LPUTargetModel& target)
{
    if (target.throughput().mxms_per_hemisphere != 1)
        return FfnProjectionOrder::Interleaved;
    // One physical MXM executes the complete Gate domain, then the complete
    // Up domain. Weight refill availability must not reverse this order;
    // page-ready events absorb ingress delays and protect SRAM reuse.
    return FfnProjectionOrder::GateThenUp;
}

mlir::DictionaryAttr orderDownPagesByProjectionRelease(
    mlir::OpBuilder& builder, mlir::DictionaryAttr down,
    mlir::DictionaryAttr gate, mlir::DictionaryAttr up,
    FfnProjectionOrder order)
{
    const auto pageGroups = down.getAs<mlir::ArrayAttr>(
        "page_slice_group_bases");
    const auto pageGroupCounts = down.getAs<mlir::ArrayAttr>(
        "page_slice_group_counts");
    if (!pageGroups || !pageGroupCounts
        || pageGroups.size() != pageGroupCounts.size()
        || order == FfnProjectionOrder::Interleaved)
        return down;
    const auto integer = [](mlir::DictionaryAttr placement,
                             llvm::StringRef name) -> std::optional<int64_t> {
        if (const auto value = placement.getAs<mlir::IntegerAttr>(name))
            return value.getInt();
        return std::nullopt;
    };
    const auto gateBase = integer(gate, "page_role_group_base");
    const auto gateCount = integer(gate, "page_role_group_count");
    const auto upBase = integer(up, "page_role_group_base");
    const auto upCount = integer(up, "page_role_group_count");
    if (!gateBase || !gateCount || !upBase || !upCount
        || *gateCount <= 0 || *upCount <= 0)
        return down;

    llvm::SmallVector<int64_t> canonical;
    llvm::SmallVector<int64_t> releaseOrder;
    const auto appendRange = [](llvm::SmallVectorImpl<int64_t>& values,
                                 int64_t base, int64_t count) {
        for (int64_t index = 0; index < count; ++index)
            values.push_back(base + index);
    };
    appendRange(canonical, *gateBase, *gateCount);
    appendRange(canonical, *upBase, *upCount);
    if (order == FfnProjectionOrder::UpThenGate) {
        appendRange(releaseOrder, *upBase, *upCount);
        appendRange(releaseOrder, *gateBase, *gateCount);
    } else {
        appendRange(releaseOrder, *gateBase, *gateCount);
        appendRange(releaseOrder, *upBase, *upCount);
    }
    if (canonical.size() != releaseOrder.size()) return down;

    llvm::SmallVector<mlir::Attribute> reordered;
    reordered.reserve(pageGroups.size());
    for (std::size_t page = 0; page < pageGroups.size(); ++page) {
        const int64_t group =
            llvm::cast<mlir::IntegerAttr>(pageGroups[page]).getInt();
        const int64_t groupCount =
            llvm::cast<mlir::IntegerAttr>(pageGroupCounts[page]).getInt();
        const auto position = llvm::find(canonical, group);
        const bool remappable = groupCount == 1
            && position != canonical.end();
        const int64_t mapped = remappable
            ? releaseOrder[std::distance(canonical.begin(), position)]
            : group;
        reordered.push_back(builder.getI64IntegerAttr(mapped));
    }
    mlir::NamedAttrList attrs(down);
    attrs.set("page_slice_group_bases", builder.getArrayAttr(reordered));
    return attrs.getDictionary(builder.getContext());
}

} // namespace

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
    return emitFfnMemRead3D(rewriter, ffn.getLoc(), value,
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
    const auto gateTempSlices = target.ffn_gate_temp_slices();
    const auto upTempSlices = target.ffn_up_temp_slices();
    const bool tempOverlapsHiddenSlices = llvm::any_of(hiddenSlices,
        [&](int64_t slice) {
            return llvm::is_contained(gateTempSlices, slice)
                || llvm::is_contained(upTempSlices, slice);
        });
    const int64_t tempBank = memory.banks_per_slice > 1
            && (pagedWeights || tempOverlapsHiddenSlices)
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
            "cannot plan the FFN MXM execution strategy")
            << ": m=" << ffn.getM()
            << ", n=" << ffn.getN()
            << ", k=" << ffn.getHidden()
            << ", mxm_rows=" << target.throughput().mxm_rows
            << ", mxm_columns=" << target.throughput().mxm_columns;
        return mlir::failure();
    }
    const auto& throughput = target.throughput();
    const bool supportsConcurrentTempAndHiddenWrites =
        tempBank != hiddenBank || !tempOverlapsHiddenSlices;
    const int64_t activationBank = activationRoute.getPlacement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const bool activationOverlapsHiddenSlices = llvm::any_of(hiddenSlices,
        [&](int64_t slice) {
            return llvm::is_contained(activationSlices, slice);
        });
    const bool supportsConcurrentActivationAndHiddenWrites =
        activationBank != hiddenBank || !activationOverlapsHiddenSlices;
    // Fused Swish relies on the current vector path's local MXM dequant and
    // one logical Gate/Up slot per physical MXM. Its hidden writes must also
    // be physically independent from projection temporary writes. Keep Tail
    // as the portable fallback when either condition cannot be proven.
    const bool supportsFusedSwish = execution->uses_local_dequant()
        && throughput.mxms_per_hemisphere == 1
        && memory.hemispheres == 2
        && throughput.mxm_result_streams == 4
        && target.streams().streams_per_direction >= 24
        && supportsConcurrentTempAndHiddenWrites
        && supportsConcurrentActivationAndHiddenWrites;
    if (strategy == FfnScheduleStrategy::Fused && !supportsFusedSwish)
        strategy = FfnScheduleStrategy::Tail;
    if (weightSlices.size()
            != static_cast<std::size_t>(
                memory.w8a16_weight_slice_count)
        || upWeightSlices.size()
            != static_cast<std::size_t>(
                memory.w8a16_weight_slice_count)
        || (!activationDistributed16
            && activationSlices.size()
                != static_cast<std::size_t>(
                    throughput.mxm_activation_streams))
        || (activationDistributed16
            && activationSlices.size() != 16)
        || downWeightSlices.size()
            != static_cast<std::size_t>(
                memory.w8a16_weight_slice_count)
        || hiddenSlices.size()
            != static_cast<std::size_t>(
                hiddenDistributed16 ? 16
                           : throughput.mxm_activation_streams)
        || resultSlices.size()
            != static_cast<std::size_t>(
                throughput.mxm_result_streams))
    {
        ffn.getOperation()->emitError(
            "FFN physical slices do not match the selected MXM strategy");
        return mlir::failure();
    }

    auto activationLatency = target.transport_latency(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, activationSlices.front());
    const FfnProjectionOrder projectionOrder = chooseProjectionOrder(target);
    auto projectionTimeline = planFfnProjectionTimeline(
        {static_cast<int64_t>(ffn.getM()),
            static_cast<int64_t>(ffn.getK()),
            static_cast<int64_t>(ffn.getHidden()),
            static_cast<int64_t>(ffn.getN())},
        weightSlices, target, execution->uses_local_dequant(),
        throughput.mxms_per_hemisphere != 1, projectionOrder);
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
    const auto downWeightPlacement = orderDownPagesByProjectionRelease(
        rewriter, downRaw.getPlacement(), gateRaw.getPlacement(),
        upRaw.getPlacement(), projectionOrder);

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
        downWeightPlacement,
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
        tempBank,
    });
}

} // namespace ftlpu::compiler::schedule::ffn_detail
