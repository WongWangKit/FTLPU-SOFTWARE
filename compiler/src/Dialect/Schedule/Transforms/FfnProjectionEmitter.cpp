#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

namespace ftlpu::compiler::schedule::ffn_detail {

mlir::FailureOr<FfnProjectionEmission> emitFfnProjection(
    FfnEmissionContext& context)
{
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t m = context.m();
    const int64_t k = context.k();
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const int64_t mTileCount =
        context.projection_timeline.m_tile_count;
    const int64_t projectionSlotInterval =
        context.projection_timeline.projection_slot_interval;
    const int64_t gateAccLatency =
        throughput.mxm0_accumulator_latency;
    const int64_t upAccLatency =
        throughput.mxm1_accumulator_latency;
    const bool singleMxm = throughput.mxms_per_hemisphere == 1;
    const FfnProjectionOrder projectionOrder =
        context.projection_timeline.projection_order;
    const bool serializedProjections = singleMxm
        && projectionOrder != FfnProjectionOrder::Interleaved;
    const int64_t reductionBlockCount = k / tile;
    const auto activationType = llvm::cast<mlir::RankedTensorType>(
        ffn.getActivation().getType());
    const llvm::StringRef dataFormat =
        activationType.getElementType().isBF16() ? "bf16" : "fp16";
    const int64_t pairCount = context.projection_timeline.pair_count;
    const int64_t totalProjectionBlocks = pairCount * reductionBlockCount;

    // Local-dequant projection planning has exactly one non-affine timing
    // boundary: the first reuse of the finite MXM weight-buffer set.  The
    // planner delays that load by the result-window hazard, after which all
    // remaining block starts have the steady projection-block interval.
    // Record that known hardware boundary directly instead of scanning a
    // list of per-reduction operations and rediscovering affine runs.
    int64_t dequantSplitBlock = -1;
    const int64_t firstBufferReuse =
        std::min(totalProjectionBlocks, throughput.mxm_weight_buffers);
    if (context.local_weight_dequant && firstBufferReuse > 0
        && firstBufferReuse < totalProjectionBlocks) {
        const auto& before = context.projection_timeline.blocks[
            firstBufferReuse - 1];
        const auto& after = context.projection_timeline.blocks[
            firstBufferReuse];
        if (after.dequant_start - before.dequant_start
            != context.projection_timeline.projection_block_interval)
            dequantSplitBlock = firstBufferReuse;
    }
    int64_t weightStartupDelay = 0;
    if (dequantSplitBlock > 0
        && dequantSplitBlock < reductionBlockCount) {
        const int64_t blockInterval =
            context.projection_timeline.projection_block_interval;
        const int64_t delayedStart =
            context.projection_timeline.blocks[dequantSplitBlock]
                .dequant_start;
        const int64_t nominalStart =
            context.projection_timeline.blocks.front().dequant_start
                + dequantSplitBlock * blockInterval;
        bool regularPrelude = true;
        for (int64_t index = 1; index < dequantSplitBlock; ++index)
            regularPrelude &=
                context.projection_timeline.blocks[index].dequant_start
                    - context.projection_timeline.blocks[index - 1]
                          .dequant_start
                == blockInterval;
        if (regularPrelude
            && delayedStart - nominalStart
                == target.mxm_first_result_latency()) {
            // Loading the short prelude at the steady-state offset keeps
            // every later FU issue unchanged.  Its first pair can then join
            // the following pairs in one blocked-address MEM 3D domain.
            weightStartupDelay = delayedStart - nominalStart;
            dequantSplitBlock = -1;
        }
    }

    struct ReductionDomain {
        bool leader{false};
        int64_t count{0};
        int64_t interval{1};
    };
    const auto weightReductionDomain =
        [&](int64_t pair, int64_t reduction) {
            const int64_t flatBegin = pair * reductionBlockCount;
            const int64_t flatEnd = flatBegin + reductionBlockCount;
            int64_t first = 0;
            int64_t end = reductionBlockCount;
            if (dequantSplitBlock > flatBegin
                && dequantSplitBlock < flatEnd) {
                const int64_t split = dequantSplitBlock - flatBegin;
                if (reduction < split)
                    end = split;
                else
                    first = split;
            }
            if (reduction != first) return ReductionDomain {};
            const int64_t count = end - first;
            const int64_t index = flatBegin + first;
            const int64_t interval = count > 1
                ? context.projection_timeline.blocks[index + 1]
                        .dequant_start
                    - context.projection_timeline.blocks[index]
                          .dequant_start
                : 1;
            return ReductionDomain {true, count, interval};
        };

    // Pair is the outer weight-address dimension.  Page and slice-group
    // boundaries change the physical bank/slice mapping and therefore end a
    // MEM command body.  Within one such resident region, the hardware
    // blocked-outer address generator represents the alternating two-slot
    // one-MXM layout without splitting the pair domain.
    const auto makeWeightPairDomains = [&](mlir::DictionaryAttr placement) {
        std::vector<int64_t> result(
            static_cast<std::size_t>(pairCount), 0);
        const bool paged = [&] {
            const auto value = placement.getAs<mlir::BoolAttr>(
                "paged_weight");
            return value && value.getValue();
        }();
        const int64_t pairsPerPage = paged
            ? placement.getAs<mlir::IntegerAttr>("page_granularity")
                  .getInt()
            : pairCount;
        const int64_t pairsPerSliceGroup = paged
            ? placement
                  .getAs<mlir::IntegerAttr>(
                      "page_items_per_slice_group")
                  .getInt()
            : pairCount;
        for (int64_t pair = 0; pair < pairCount;) {
            const int64_t flatBegin = pair * reductionBlockCount;
            const int64_t flatEnd = flatBegin + reductionBlockCount;
            const bool splitInsidePair = dequantSplitBlock > flatBegin
                && dequantSplitBlock < flatEnd;
            int64_t count = 1;
            if (!splitInsidePair) {
                const int64_t pairInPage = paged
                    ? pair % pairsPerPage : pair;
                const int64_t pageRemaining = paged
                    ? pairsPerPage - pairInPage : pairCount - pair;
                const int64_t sliceGroupRemaining = paged
                    ? pairsPerSliceGroup
                        - pairInPage % pairsPerSliceGroup
                    : pairCount - pair;
                count = std::min({pairCount - pair, pageRemaining,
                    sliceGroupRemaining});
                // A split exactly at a pair boundary changes the temporal
                // stride and cannot be crossed by one descriptor.
                if (dequantSplitBlock > flatBegin
                    && dequantSplitBlock < (pair + count)
                        * reductionBlockCount)
                    count = std::max<int64_t>(1,
                        dequantSplitBlock / reductionBlockCount - pair);
            }
            result[static_cast<std::size_t>(pair)] = count;
            pair += count;
        }
        return result;
    };
    const std::array<std::vector<int64_t>, 2> weightPairDomains = {
        makeWeightPairDomains(context.gate_route.getPlacement()),
        makeWeightPairDomains(context.up_route.getPlacement()),
    };
    // MXM control does not depend on the physical MEM slice group.  It only
    // needs an instruction boundary when page residency changes.  Splitting
    // it at every MEM slice-group boundary would restart the MXM domain
    // counters partway through a resident page.
    const auto makeWeightPageDomains = [&](mlir::DictionaryAttr placement) {
        std::vector<int64_t> result(
            static_cast<std::size_t>(pairCount), 0);
        const auto paged = placement.getAs<mlir::BoolAttr>("paged_weight");
        const bool isPaged = paged && paged.getValue();
        const int64_t pairsPerPage = isPaged
            ? placement.getAs<mlir::IntegerAttr>("page_granularity").getInt()
            : pairCount;
        for (int64_t pair = 0; pair < pairCount;) {
            const int64_t flatBegin = pair * reductionBlockCount;
            int64_t count = std::min(pairCount - pair,
                isPaged ? pairsPerPage - pair % pairsPerPage
                        : pairCount - pair);
            if (dequantSplitBlock > flatBegin
                && dequantSplitBlock
                    < (pair + count) * reductionBlockCount)
                count = std::max<int64_t>(1,
                    dequantSplitBlock / reductionBlockCount - pair);
            result[static_cast<std::size_t>(pair)] = count;
            pair += count;
        }
        return result;
    };
    const std::array<std::vector<int64_t>, 2> weightPageDomains = {
        makeWeightPageDomains(context.gate_route.getPlacement()),
        makeWeightPageDomains(context.up_route.getPlacement()),
    };
    const auto isPagedPlacement = [](mlir::DictionaryAttr placement) {
        const auto paged = placement.getAs<mlir::BoolAttr>("paged_weight");
        return paged && paged.getValue();
    };
    const bool splitSerializedActivationDomains = singleMxm
        && serializedProjections
        && (isPagedPlacement(context.gate_route.getPlacement())
            || isPagedPlacement(context.up_route.getPlacement()));

    // The active Qwen path uses local dequant, whose projection planner emits
    // one stable stream segment per hemisphere and tile.  Its compute launch
    // domain is therefore closed form even when a weight-load hazard splits
    // the corresponding MEM domain.
    const bool closedProjectionDomains = context.local_weight_dequant
        && reductionBlockCount > 0 && pairCount > 0;
    // Weight SRAM descriptors stop at page/slice-group boundaries.  MXM
    // dequant and load must stop at the same boundaries: a hardware ICU
    // cannot suspend an already decoded 3-D domain while the next dynamic
    // weight page arrives.  Each resident affine region therefore owns one
    // MEM domain and one matching MXM control domain.
    int64_t mxmPairInterval = 1;
    bool closedWeightMxmControl = closedProjectionDomains
        && pairCount > 1 && dequantSplitBlock < 0
        && (serializedProjections || !singleMxm)
        && pairCount * reductionBlockCount <= 65536;
    if (closedWeightMxmControl) {
        const int64_t first =
            context.projection_timeline.blocks.front().dequant_start
                + weightStartupDelay;
        mxmPairInterval =
            context.projection_timeline.blocks[reductionBlockCount]
                .dequant_start - first;
        closedWeightMxmControl = mxmPairInterval
            == reductionBlockCount
                * context.projection_timeline.projection_block_interval;
        for (int64_t pair = 1;
             closedWeightMxmControl && pair < pairCount; ++pair) {
            closedWeightMxmControl =
                context.projection_timeline.blocks[
                    pair * reductionBlockCount].dequant_start
                == first + pair * mxmPairInterval;
        }
    }
    const auto computeProjectionOffset = [&](int64_t projection) {
        if (!singleMxm) return int64_t{0};
        if (!serializedProjections)
            return projection * projectionSlotInterval;
        const bool first =
            (projectionOrder == FfnProjectionOrder::GateThenUp)
            == (projection == 0);
        return first ? int64_t{0}
                     : context.projection_timeline
                           .second_projection_offset;
    };

    FfnProjectionEmission emission;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, mlir::Value>
        tempDomainOutputs;
    rewriter.setInsertionPoint(ffn.getOperation());
    for (const FfnProjectionBlockSchedule& block :
        context.projection_timeline.blocks) {
        const int64_t pair = block.pair;
        const int64_t reduction = block.reduction_block;
        const int64_t dequantStart = block.dequant_start;
        const int64_t weightBuffer = block.weight_buffer;

        for (int64_t hemisphere = 0;
             hemisphere < memory.hemispheres; ++hemisphere) {
            const int64_t projectionCount = singleMxm
                ? 2 : throughput.mxms_per_hemisphere;
            for (int64_t projection = 0;
                 projection < projectionCount; ++projection) {
                const int64_t localMxm = singleMxm ? 0 : projection;
                stream::RouteOp raw =
                    projection == 0 ? context.gate_raw : context.up_raw;
                stream::RouteOp cooked =
                    projection == 0 ? context.gate_route : context.up_route;
                const int64_t start = dequantStart
                    + (singleMxm
                            ? (serializedProjections
                                  ? computeProjectionOffset(projection)
                                  : projection * tile)
                            : (hemisphere
                                      * throughput.mxms_per_hemisphere
                                  + localMxm)
                                * weightLoadCycles);
                const bool shiftedFirstWeight =
                    weightStartupDelay > 0 && pair == 0
                    && reduction == 0;
                const int64_t weightStart = start
                    + (shiftedFirstWeight ? weightStartupDelay : 0);
                const auto placement = cooked.getPlacement();
                const int64_t bindingBase = placement
                                                .getAs<mlir::IntegerAttr>(
                                                    "base_row")
                                                .getInt();
                int64_t base = bindingBase;
                if (singleMxm) {
                    const int64_t logicalSlots = 2;
                    base += ((pair / logicalSlots) * (k / tile)
                                + reduction)
                            * logicalSlots * weightLoadCycles
                        + (pair % logicalSlots) * weightLoadCycles;
                } else {
                    base += (pair * (k / tile) + reduction)
                        * weightLoadCycles;
                }
                const int64_t logicalBase = base;
                int64_t page = -1;
                int64_t bank = placement
                    .getAs<mlir::IntegerAttr>("bank").getInt();
                llvm::SmallVector<int64_t> selectedWeightSlices =
                    projection == 0 ? context.weight_slices
                                    : context.up_weight_slices;
                if (auto paged = placement.getAs<mlir::BoolAttr>(
                        "paged_weight"); paged && paged.getValue()) {
                    const int64_t wavesPerPage = placement
                        .getAs<mlir::IntegerAttr>("page_granularity")
                        .getInt();
                    const int64_t itemsPerGroup = placement
                        .getAs<mlir::IntegerAttr>(
                            "page_items_per_slice_group")
                        .getInt();
                    if (wavesPerPage <= 0 || itemsPerGroup <= 0) {
                        ffn.getOperation()->emitError(
                            "paged FFN projection has invalid page geometry");
                        return mlir::failure();
                    }
                    page = pair / wavesPerPage;
                    auto pagePlacement = resolve_page_placement(
                        placement, page);
                    if (mlir::failed(pagePlacement)) {
                        ffn.getOperation()->emitError(
                            "paged FFN projection has invalid physical page placement")
                            << ": projection=" << projection
                            << ", pair=" << pair
                            << ", pair_count=" << pairCount
                            << ", page=" << page
                            << ", waves_per_page=" << wavesPerPage;
                        return mlir::failure();
                    }
                    bank = pagePlacement->bank;
                    const int64_t pairInPage = pair % wavesPerPage;
                    const int64_t sliceGroup =
                        pagePlacement->slice_group_base
                        + pairInPage / itemsPerGroup;
                    const int64_t localPair = pairInPage % itemsPerGroup;
                    const auto storage = placement
                        .getAs<mlir::ArrayAttr>("page_storage_slices");
                    const int64_t loadSliceCount =
                        static_cast<int64_t>(selectedWeightSlices.size());
                    const int64_t storageOffset =
                        sliceGroup * loadSliceCount;
                    if (sliceGroup
                            >= pagePlacement->slice_group_base
                                + pagePlacement->slice_group_count
                        || storageOffset < 0
                        || storageOffset + loadSliceCount
                            > static_cast<int64_t>(storage.size())) {
                        ffn.getOperation()->emitError(
                            "paged FFN projection slice group is outside "
                            "page_storage_slices")
                            << ": projection=" << projection
                            << ", pair=" << pair
                            << ", page=" << page
                            << ", slice_group=" << sliceGroup
                            << ", load_slices=" << loadSliceCount
                            << ", storage_slices=" << storage.size();
                        return mlir::failure();
                    }
                    selectedWeightSlices.clear();
                    for (int64_t index = 0; index < loadSliceCount; ++index)
                        selectedWeightSlices.push_back(
                            llvm::cast<mlir::IntegerAttr>(
                                storage[storageOffset + index])
                                .getInt());
                    const int64_t logicalSlots = singleMxm ? 2 : 1;
                    base = bindingBase + pagePlacement->base_row
                        + ((localPair / logicalSlots) * (k / tile)
                              + reduction)
                            * logicalSlots * weightLoadCycles
                        + (localPair % logicalSlots) * weightLoadCycles;
                }
                const auto rawType =
                    llvm::cast<mlir::RankedTensorType>(
                        raw.getInput().getType());
                const auto activationType =
                    llvm::cast<mlir::RankedTensorType>(
                        ffn.getActivation().getType());
                const auto dequantizedType =
                    mlir::RankedTensorType::get(rawType.getShape(),
                        activationType.getElementType());
                const ReductionDomain reductionDomain =
                    weightReductionDomain(pair, reduction);
                if (!reductionDomain.leader) continue;
                const int64_t weightPairCount =
                    reductionDomain.count == reductionBlockCount
                    ? weightPairDomains[static_cast<std::size_t>(projection)]
                          [static_cast<std::size_t>(pair)]
                    : 1;
                if (weightPairCount == 0) continue;
                FfnLoopDomain3D weightDomain;
                weightDomain.wave_count = reductionDomain.count;
                weightDomain.wave_interval = reductionDomain.interval;
                weightDomain.wave_address_stride =
                    (singleMxm ? 2 : 1) * weightLoadCycles;
                weightDomain.group_count = weightPairCount;
                if (weightPairCount > 1) {
                    const auto& nextPairBlock =
                        context.projection_timeline.blocks[
                            (pair + 1) * reductionBlockCount + reduction];
                    weightDomain.group_interval =
                        nextPairBlock.dequant_start - dequantStart
                            - (shiftedFirstWeight
                                  ? weightStartupDelay : 0);
                    if (singleMxm) {
                        const int64_t addressPair = page >= 0
                            ? pair
                                % placement
                                      .getAs<mlir::IntegerAttr>(
                                          "page_granularity")
                                      .getInt()
                            : pair;
                        // Logical pair rows use [slot0, slot1] inside each
                        // reduction-sized physical group.  The first domain
                        // may start on either slot, so encode its local
                        // parity in the within-group stride while retaining
                        // the regular two-pair carry stride.
                        weightDomain.blocked_outer_address = true;
                        weightDomain.outer_group_size = 2;
                        weightDomain.outer_inner_stride =
                            (addressPair & 1)
                            ? (2 * reductionBlockCount - 1)
                                * weightLoadCycles
                            : weightLoadCycles;
                        weightDomain.outer_group_stride =
                            2 * reductionBlockCount * weightLoadCycles;
                    } else {
                        weightDomain.group_address_stride =
                            reductionBlockCount * weightLoadCycles;
                    }
                }
                std::optional<FfnLoopDomain3D> mxmControlDomain;
                const int64_t mxmControlPairCount =
                    weightPageDomains[static_cast<std::size_t>(projection)]
                        [static_cast<std::size_t>(pair)];
                if (closedWeightMxmControl && reduction == 0
                    && mxmControlPairCount > 0) {
                    mxmControlDomain = weightDomain;
                    mxmControlDomain->group_count = mxmControlPairCount;
                    mxmControlDomain->group_interval = mxmPairInterval;
                }
                emitFfnWeightTile(rewriter, ffn.getLoc(), raw,
                    dequantizedType,
                    selectedWeightSlices,
                    target,
                    projection == 0
                        ? ffn.getGateScale().convertToFloat()
                        : ffn.getUpScale().convertToFloat(),
                    weightStart, base, hemisphere, localMxm,
                    hemisphere * throughput.mxms_per_hemisphere
                        + localMxm,
                    singleMxm && !serializedProjections
                        ? projection : weightBuffer,
                    context.local_weight_dequant, bank, page,
                    logicalBase, {}, weightDomain,
                    reductionDomain.count * weightPairCount > 1
                        ? "toggle_dim2" : "fixed",
                    mxmControlDomain,
                    !closedWeightMxmControl
                        || (reduction == 0 && mxmControlPairCount > 0));
            }
        }

        for (const FfnProjectionTileSchedule& tileSchedule :
            block.tiles) {
            const int64_t mTile = tileSchedule.m_tile;
            const int64_t computeCycle = tileSchedule.compute_cycle;
            for (int64_t hemisphere = 0;
                 hemisphere < memory.hemispheres; ++hemisphere) {
                const int64_t activationBase =
                    reduction * m + mTile * tile;
                const bool finalReduction = block.final_reduction;
                const int64_t resultStreamBase =
                    finalReduction
                    ? (context.strategy == FfnScheduleStrategy::Fused
                            ? 8 + hemisphere * 8
                            : 0)
                    : 0;
                const int64_t terminalStreamBase =
                    context.strategy == FfnScheduleStrategy::Fused
                    ? 8 + hemisphere * 8 : 0;
                const bool computeCanCrossPairs =
                    closedProjectionDomains
                    && (reductionBlockCount % 2 == 0
                        || reductionBlockCount == 1);
                const std::array<int64_t, 2> computeDomainPairCounts = {
                    computeCanCrossPairs
                        ? weightPageDomains[0][static_cast<std::size_t>(pair)]
                        : 1,
                    computeCanCrossPairs
                        ? weightPageDomains[1][static_cast<std::size_t>(pair)]
                        : 1,
                };
                const std::array<bool, 2> closedComputeLeaders = {
                    closedProjectionDomains && reduction == 0
                        && (!computeCanCrossPairs
                            || computeDomainPairCounts[0] > 0),
                    closedProjectionDomains && reduction == 0
                        && (!computeCanCrossPairs
                            || computeDomainPairCounts[1] > 0),
                };

                mlir::Value gateAccumulatorValue;
                mlir::Value upAccumulatorValue;
                int64_t rowOffset = 0;
                for (const FfnStreamSegment& segment :
                    tileSchedule.hemisphere_segments[
                        static_cast<std::size_t>(hemisphere)]) {
                    const int64_t segmentCycle =
                        computeCycle + rowOffset;
                    const int64_t gateCycle = segmentCycle
                        + computeProjectionOffset(0);
                    const int64_t upCycle = segmentCycle
                        + computeProjectionOffset(1);
                    const int64_t reductionInterval =
                        reductionBlockCount > 1
                        ? context.projection_timeline.blocks[
                              pair * reductionBlockCount + 1]
                                  .tiles[mTile]
                                  .compute_cycle
                            - context.projection_timeline.blocks[
                                  pair * reductionBlockCount]
                                      .tiles[mTile]
                                      .compute_cycle
                        : 1;
                    const int64_t pairInterval = pairCount > 1
                        ? context.projection_timeline.blocks[
                              reductionBlockCount]
                                  .tiles[mTile]
                                  .compute_cycle
                            - context.projection_timeline.blocks.front()
                                  .tiles[mTile]
                                  .compute_cycle
                        : reductionBlockCount
                            * context.projection_timeline
                                  .projection_block_interval;
                    const auto emitActivation =
                        [&](int64_t consumerCycle,
                            FfnLoopDomain3D projectionDomain) {
                            mlir::Value activationValue =
                                context.activation_route.getInput();
                            if (context.activation_distributed16) {
                                const int64_t base =
                                    context.activation_route.getPlacement()
                                        .getAs<mlir::IntegerAttr>("base_row")
                                        .getInt();
                                const int64_t reductionBlocks = k / tile;
                                const int64_t blockRows =
                                    throughput.mxm_block_rows;
                                const int64_t segmentTokenBase =
                                    mTile * tile + rowOffset;
                                for (int64_t tokenLane = 0;
                                     tokenLane < blockRows; ++tokenLane) {
                                    const int64_t firstOffset =
                                        (tokenLane
                                            - segmentTokenBase % blockRows
                                            + blockRows)
                                        % blockRows;
                                    if (firstOffset >= segment.rows) continue;
                                    const int64_t occurrenceCount = 1
                                        + (segment.rows - 1 - firstOffset)
                                            / blockRows;
                                    const int64_t token =
                                        segmentTokenBase + firstOffset;
                                    const int64_t tokenBlock = token / tile;
                                    const int64_t tokenWithinBlock = token % tile;
                                    const int64_t tokenWave =
                                        tokenWithinBlock / blockRows;
                                    const int64_t row = base
                                        + (tokenBlock * reductionBlocks
                                              + reduction)
                                            * throughput.tile_rows
                                        + tokenWave;
                                    for (int64_t byte = 0; byte < 2;
                                         ++byte) {
                                        const int64_t slice =
                                            context.activation_slices[
                                                2 * tokenLane + byte];
                                        FfnLoopDomain3D domain =
                                            projectionDomain;
                                        emitFfnMemTransfer3D(rewriter,
                                            ffn.getLoc(),
                                            consumerCycle + firstOffset
                                                - context.eastMxmLatency(slice),
                                            hemisphere, slice, "read", row,
                                            segment.stream_base + byte,
                                            occurrenceCount, blockRows, 1,
                                            context.activation_route
                                                .getPlacement()
                                                .getAs<mlir::IntegerAttr>(
                                                    "bank")
                                                .getInt(),
                                            domain);
                                    }
                                }
                            } else {
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    const int64_t slice =
                                        context.activation_slices[byte];
                                    emitFfnMemTransfer3D(rewriter,
                                        ffn.getLoc(),
                                        consumerCycle
                                            - context.activation_latency,
                                        hemisphere, slice, "read",
                                        activationBase + rowOffset,
                                        segment.stream_base + byte,
                                        segment.rows, 1, 1,
                                        context.activation_route
                                            .getPlacement()
                                            .getAs<mlir::IntegerAttr>("bank")
                                            .getInt(),
                                        projectionDomain);
                                }
                            }
                            return activationValue;
                        };
                    mlir::Value gateActivation =
                        context.activation_route.getInput();
                    mlir::Value upActivation = gateActivation;
                    if (closedProjectionDomains) {
                        if (splitSerializedActivationDomains
                            && reduction == 0) {
                            // The MEM activation producer must stop at every
                            // dynamic weight-page boundary together with the
                            // MXM consumer. A hardware ICU cannot suspend one
                            // decoded READ_3D while COMPUTE_3D waits on the
                            // next page, so lower one matching affine domain
                            // per projection page directly.
                            const auto makeActivationDomain =
                                [&](int64_t domainPairCount) {
                                    FfnLoopDomain3D domain;
                                    domain.wave_count = reductionBlockCount;
                                    domain.wave_interval = reductionInterval;
                                    domain.wave_address_stride =
                                        context.activation_distributed16
                                        ? throughput.tile_rows : m;
                                    domain.group_count = domainPairCount;
                                    domain.group_interval = pairInterval;
                                    return domain;
                                };
                            if (closedComputeLeaders[0])
                                gateActivation = emitActivation(gateCycle,
                                    makeActivationDomain(
                                        computeDomainPairCounts[0]));
                            if (closedComputeLeaders[1])
                                upActivation = emitActivation(upCycle,
                                    makeActivationDomain(
                                        computeDomainPairCounts[1]));
                        } else if (pair == 0 && reduction == 0) {
                            FfnLoopDomain3D activationDomain;
                            activationDomain.wave_count =
                                reductionBlockCount;
                            activationDomain.wave_interval =
                                reductionInterval;
                            activationDomain.wave_address_stride =
                                context.activation_distributed16
                                ? throughput.tile_rows : m;
                            activationDomain.group_count = pairCount;
                            activationDomain.group_interval = pairInterval;
                            const int64_t projectionSpan =
                                std::abs(upCycle - gateCycle);
                            const bool contiguousSerializedProjections =
                                serializedProjections
                                && projectionSpan
                                    == pairCount * pairInterval;
                            if (contiguousSerializedProjections) {
                                activationDomain.group_count = 2 * pairCount;
                                gateActivation = emitActivation(
                                    std::min(gateCycle, upCycle),
                                    activationDomain);
                                upActivation = gateActivation;
                            } else {
                                gateActivation = emitActivation(
                                    gateCycle, activationDomain);
                                if (singleMxm)
                                    upActivation = emitActivation(
                                        upCycle, activationDomain);
                            }
                        }
                    } else {
                        const int64_t activationReductionCount =
                            mTileCount == 1 && reduction == 0
                            ? reductionBlockCount
                            : mTileCount == 1 ? 0 : 1;
                        if (activationReductionCount > 0) {
                            FfnLoopDomain3D activationDomain;
                            const bool groupedProjections = singleMxm
                                && !serializedProjections
                                && upCycle > gateCycle;
                            if (groupedProjections) {
                                activationDomain.wave_count = 2;
                                activationDomain.wave_interval =
                                    upCycle - gateCycle;
                            }
                            activationDomain.group_count =
                                activationReductionCount;
                            activationDomain.group_interval =
                                reductionInterval;
                            // In the legacy layout reduction is the outer
                            // counter, so its address increment belongs to
                            // dimension 2 rather than the direct domain's
                            // reduction dimension 1.
                            activationDomain.group_address_stride =
                                context.activation_distributed16
                                ? throughput.tile_rows : m;
                            activationDomain.wave_address_stride = 0;
                            gateActivation = emitActivation(
                                gateCycle, activationDomain);
                            if (singleMxm && !groupedProjections)
                                upActivation = emitActivation(
                                    upCycle, activationDomain);
                        }
                    }
                    const auto makeComputeDomain =
                        [&](int64_t domainPairCount) {
                            FfnMxmDomain3D domain;
                            domain.repeat_count = segment.rows;
                            domain.repeat_accumulator_address_stride = 0;
                            domain.wave_count = reductionBlockCount;
                            domain.wave_interval = reductionInterval;
                            domain.group_count = domainPairCount;
                            domain.group_interval = pairInterval;
                            if (throughput.mxm_weight_buffers == 2) {
                                if (reductionBlockCount > 1)
                                    domain.weight_buffer_mode = "toggle_dim1";
                                else if (domainPairCount > 1)
                                    domain.weight_buffer_mode = "toggle_dim2";
                            }
                            if (reductionBlockCount > 1) {
                                // Reduction is dimension 1. Terminal mode
                                // fires at its last coordinate for every
                                // outer pair, draining and clearing that pair
                                // before reuse.
                                domain.terminal_dimension = 1;
                                domain.terminal_accumulator_destination =
                                    "stream";
                                domain.terminal_accumulator_clear = true;
                                domain.terminal_accumulator_output_format =
                                    "bf16";
                            }
                            return domain;
                        };
                    const bool finalOnly = reductionBlockCount == 1;
                    if (closedComputeLeaders[0]) {
                        const FfnMxmDomain3D domain =
                            makeComputeDomain(computeDomainPairCounts[0]);
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), gateCycle,
                            hemisphere * throughput.mxms_per_hemisphere,
                            "compute", weightBuffer, 0,
                            segment.stream_base, terminalStreamBase,
                            mTile * tile + rowOffset, 1,
                            finalOnly ? "stream" : "sram", true,
                            dataFormat, finalOnly ? "bf16" : "fp32",
                            domain);
                    }
                    if (closedComputeLeaders[1]) {
                        const FfnMxmDomain3D domain =
                            makeComputeDomain(computeDomainPairCounts[1]);
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), upCycle,
                            hemisphere * throughput.mxms_per_hemisphere
                                + (singleMxm ? 0 : 1),
                            "compute", weightBuffer, 0,
                            segment.stream_base,
                            terminalStreamBase
                                + throughput.mxm_result_streams,
                            mTile * tile + rowOffset
                                + (singleMxm ? m : 0), 1,
                            finalOnly ? "stream" : "sram", true,
                            dataFormat, finalOnly ? "bf16" : "fp32",
                            domain);
                    } else if (!closedProjectionDomains) {
                        FfnMxmDomain3D domain;
                        domain.repeat_count = segment.rows;
                        domain.repeat_accumulator_address_stride = 0;
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), gateCycle,
                            hemisphere * throughput.mxms_per_hemisphere,
                            "compute",
                            singleMxm && !serializedProjections
                                ? 0 : weightBuffer,
                            0, segment.stream_base, resultStreamBase,
                            mTile * tile + rowOffset, 1,
                            finalReduction ? "stream" : "sram", true,
                            dataFormat,
                            finalReduction ? "bf16" : "fp32", domain);
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), upCycle,
                            hemisphere * throughput.mxms_per_hemisphere
                                + (singleMxm ? 0 : 1),
                            "compute",
                            singleMxm && !serializedProjections
                                ? 1 : weightBuffer,
                            0, segment.stream_base,
                            resultStreamBase
                                + throughput.mxm_result_streams,
                            mTile * tile + rowOffset
                                + (singleMxm ? m : 0), 1,
                            finalReduction ? "stream" : "sram", true,
                            dataFormat,
                            finalReduction ? "bf16" : "fp32", domain);
                    }
                    (void)gateActivation;
                    (void)upActivation;
                    rowOffset += segment.rows;
                }

                // Each physical MXM owns its accumulator. A pair is fully
                // reduced and drained before the next pair starts, so all
                // projection pairs can reuse the same token-row window.
                const int64_t gateComputeCycle = computeCycle
                    + computeProjectionOffset(0);
                const int64_t upComputeCycle = computeCycle
                    + computeProjectionOffset(1);

                gateAccumulatorValue = context.activation_route.getInput();
                upAccumulatorValue = context.activation_route.getInput();
                const int64_t effectiveUpAccLatency =
                    singleMxm ? gateAccLatency : upAccLatency;

                if (!finalReduction) continue;

                const auto gateTempSlices = target.ffn_gate_temp_slices();
                const auto upTempSlices = target.ffn_up_temp_slices();
                const int64_t pairsPerTempGroup =
                    memory.sram_depth_rows / (mTileCount * tile);
                const int64_t tempGroup = pair / pairsPerTempGroup;
                if (2 * tempGroup + 1
                        >= static_cast<int64_t>(gateTempSlices.size())
                    || 2 * tempGroup + 1
                        >= static_cast<int64_t>(upTempSlices.size()))
                    return mlir::failure();
                mlir::Value gateTemp;
                mlir::Value upTemp;
                int64_t deferredReadyCycle = std::max(
                    gateComputeCycle + gateAccLatency + tile
                        + context.westLatency(
                            gateTempSlices[2 * tempGroup]),
                    upComputeCycle + effectiveUpAccLatency + tile
                        + context.westLatency(
                            upTempSlices[2 * tempGroup]));
                {
                    const int64_t tempBase =
                        ((pair % pairsPerTempGroup) * mTileCount + mTile)
                        * tile;
                        const auto emitTempWrite =
                            [&](mlir::Value source,
                            llvm::ArrayRef<int64_t> tempSlices,
                            int64_t streamBase, int64_t sourceComputeCycle,
                            int64_t projection,
                            mlir::Value& lastWrite) {
                            for (int64_t byte = 0; byte < 2; ++byte) {
                                const int64_t targetSlice =
                                    tempSlices[byte];
                                const int64_t targetBoundary =
                                    targetSlice
                                        / target.streams()
                                              .mem_slices_per_register_group
                                    + 1;
                                const auto transportLatency =
                                    target.transport_latency(
                                        target::StreamEndpoint::MxmResult,
                                        target::StreamEndpoint::Mem,
                                        target::StreamDirection::West,
                                        targetSlice);
                                if (!transportLatency)
                                    return;
                                const int64_t writeCycle = sourceComputeCycle
                                    + target.mxm_first_result_latency()
                                    + *transportLatency;
                                const auto& pageDomains =
                                    weightPageDomains[
                                        static_cast<std::size_t>(projection)];
                                int64_t pagePairBase = pair;
                                while (pagePairBase > 0
                                    && pageDomains[static_cast<std::size_t>(
                                        pagePairBase)] == 0)
                                    --pagePairBase;
                                const int64_t pagePairCount =
                                    pageDomains[static_cast<std::size_t>(
                                        pagePairBase)];
                                const int64_t tempGroupPairBase =
                                    tempGroup * pairsPerTempGroup;
                                const int64_t domainPairBase = std::max(
                                    pagePairBase, tempGroupPairBase);
                                const int64_t domainPairEnd = std::min({
                                    pagePairBase + pagePairCount,
                                    tempGroupPairBase + pairsPerTempGroup,
                                    context.projection_timeline.pair_count});
                                const int64_t pairsInDomain =
                                    domainPairEnd - domainPairBase;
                                const auto key = std::tuple {hemisphere,
                                    projection, domainPairBase, byte};
                                const bool domainLeader =
                                    pair == domainPairBase && mTile == 0;
                                if (domainLeader) {
                                    FfnLoopDomain3D domain;
                                    domain.wave_count = mTileCount;
                                    domain.wave_interval = context
                                        .projection_timeline
                                        .pipelined_block_interval;
                                    domain.wave_address_stride = tile;
                                    domain.group_count = pairsInDomain;
                                    domain.group_address_stride =
                                        mTileCount * tile;
                                    if (pairsInDomain > 1) {
                                        const int64_t reductionBlocks =
                                            k / tile;
                                        const auto& nextFinal = context
                                            .projection_timeline.blocks[
                                                reductionBlocks
                                                    * (domainPairBase + 2)
                                                - 1];
                                        domain.group_interval =
                                            nextFinal.tiles.front()
                                                .compute_cycle
                                            - computeCycle;
                                    }
                                    auto placement = schedule_placement(
                                        rewriter, {targetSlice}, tempBase,
                                        tile, 1,
                                        context.hemisphereName(hemisphere),
                                        "bf16_swiglu_temp_byte",
                                        context.temp_bank);
                                    auto write = rewriter.create<MemWriteOp>(
                                        ffn.getLoc(), source,
                                        writeCycle, tile,
                                        streamBase + byte, 1,
                                        targetBoundary,
                                        rewriter.getStringAttr("west"),
                                        ffn.getHidden1Address(), placement,
                                        tile
                                            * throughput.lanes_per_tile);
                                    setFfnLoopDomain3D(
                                        write.getOperation(), rewriter,
                                        domain);
                                    tempDomainOutputs[key] =
                                        write.getOutput();
                                }
                                const auto output =
                                    tempDomainOutputs.find(key);
                                if (output == tempDomainOutputs.end())
                                    return;
                                lastWrite = output->second;
                                emission
                                    .temp_mem_busy_windows[hemisphere]
                                    .push_back({
                                        writeCycle,
                                        writeCycle + tile});
                                deferredReadyCycle =
                                    std::max(deferredReadyCycle,
                                        writeCycle + tile
                                            + context.westLatency(
                                                targetSlice));
                            }
                        };
                    emitTempWrite(gateAccumulatorValue,
                        llvm::ArrayRef<int64_t>(gateTempSlices)
                            .slice(2 * tempGroup, 2),
                        resultStreamBase, gateComputeCycle, 0,
                        gateTemp);
                    emitTempWrite(upAccumulatorValue,
                        llvm::ArrayRef<int64_t>(upTempSlices)
                            .slice(2 * tempGroup, 2),
                        resultStreamBase
                            + throughput.mxm_result_streams,
                        upComputeCycle, 1,
                        upTemp);
                }
                emission.completed_tiles.push_back({
                    pair,
                    mTile,
                    hemisphere,
                    std::max(gateComputeCycle, upComputeCycle),
                    deferredReadyCycle,
                    gateTemp,
                    upTemp,
                });
            }
        }
    }
    return emission;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
