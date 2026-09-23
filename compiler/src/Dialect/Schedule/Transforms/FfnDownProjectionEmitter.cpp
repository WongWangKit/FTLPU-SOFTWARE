#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>
#include <map>
#include <tuple>

namespace ftlpu::compiler::schedule::ffn_detail {

mlir::FailureOr<mlir::Value> emitFfnDownProjection(
    FfnEmissionContext& context, const FfnSwishEmission& swish)
{
    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t m = context.m();
    const int64_t intermediate = context.hidden();
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;
    const int64_t projectionSlotInterval =
        context.projection_timeline.projection_slot_interval;
    const bool singleMxm = throughput.mxms_per_hemisphere == 1;
    const auto activationType = llvm::cast<mlir::RankedTensorType>(
        ffn.getActivation().getType());
    const llvm::StringRef dataFormat =
        activationType.getElementType().isBF16() ? "bf16" : "fp16";
    const int64_t logicalSlotsPerHemisphere =
        singleMxm ? 2 : throughput.mxms_per_hemisphere;
    const int64_t hiddenBaseRow =
        get_base_row(ffn.getHidden0Placement());
    const int64_t resultBank = ffn.getResultPlacement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const auto hiddenKind =
        ffn.getHidden0Placement().getAs<mlir::StringAttr>("kind");
    const bool hiddenDistributed16 = hiddenKind
        && hiddenKind.getValue() == "fp16_mxm_distributed_16";

    const auto downPlacement = context.down_weight_placement;
    int64_t reductionsPerWeightPage = 0;
    if (const auto paged = downPlacement.getAs<mlir::BoolAttr>(
            "paged_weight"); paged && paged.getValue()) {
        if (const auto granularity = downPlacement.getAs<mlir::IntegerAttr>(
                "page_granularity"))
            reductionsPerWeightPage = granularity.getInt();
    }
    auto timeline = planFfnDownProjectionTimeline(
        {context.m(), context.k(), intermediate, context.n()},
        context.projection_timeline, swish.last_cycle,
        context.down_weight_slices, context.hidden_slices,
        context.result_slices, target, reductionsPerWeightPage);
    if (mlir::failed(timeline)) return mlir::failure();
    int64_t downEnd = timeline->phase_start;
    struct PendingResultWrite {
        mlir::Value source;
        mlir::DictionaryAttr placement;
        int64_t cycle;
        int64_t outputWave;
        int64_t mTile;
        int64_t hemisphere;
        int64_t slice;
        int64_t stream;
        int64_t address;
    };
    std::map<std::tuple<int64_t, int64_t, int64_t>,
        llvm::SmallVector<PendingResultWrite>> pendingResultWrites;
    const int64_t reductionBlockCount =
        timeline->reduction_block_count;
    const auto sameWeightStorageDomain = [&](int64_t outputWave,
                                              int64_t lhs,
                                              int64_t rhs) {
        if (reductionsPerWeightPage <= 0) return true;
        const int64_t reductionsPerGroup = downPlacement
            .getAs<mlir::IntegerAttr>("page_items_per_slice_group")
            .getInt();
        const int64_t pagesPerWave =
            (reductionBlockCount + reductionsPerWeightPage - 1)
            / reductionsPerWeightPage;
        const auto domainKey = [&](int64_t reduction) {
            const int64_t page = outputWave * pagesPerWave
                + reduction / reductionsPerWeightPage;
            const int64_t group =
                (reduction % reductionsPerWeightPage)
                / reductionsPerGroup;
            return std::pair {page, group};
        };
        return domainKey(lhs) == domainKey(rhs);
    };
    const auto reductionRun = [&](int64_t outputWave, int64_t reduction) {
        std::pair<int64_t, int64_t> result {0, 0};
        const int64_t blockBase = outputWave * reductionBlockCount;
        int64_t cursor = 0;
        while (cursor < reductionBlockCount) {
            int64_t count = 1;
            int64_t interval = 1;
            if (cursor + 1 < reductionBlockCount
                && sameWeightStorageDomain(
                    outputWave, cursor, cursor + 1)) {
                interval = timeline->blocks[blockBase + cursor + 1]
                    .dequant_start
                    - timeline->blocks[blockBase + cursor].dequant_start;
                while (cursor + count < reductionBlockCount
                    && sameWeightStorageDomain(outputWave, cursor,
                        cursor + count)) {
                    const auto& previous = timeline->blocks[
                        blockBase + cursor + count - 1];
                    const auto& current = timeline->blocks[
                        blockBase + cursor + count];
                    if (current.dequant_start - previous.dequant_start
                        != interval)
                        break;
                    ++count;
                }
            }
            if (cursor == reduction) return std::pair {count, interval};
            if (reduction > cursor && reduction < cursor + count)
                return result;
            cursor += count;
        }
        return result;
    };
    const int64_t mTileCount =
        context.projection_timeline.m_tile_count;
    const auto storageRunBounds = [&](int64_t outputWave,
                                       int64_t reduction) {
        int64_t begin = reduction;
        while (begin > 0
            && sameWeightStorageDomain(
                outputWave, begin - 1, begin))
            --begin;
        int64_t end = reduction + 1;
        while (end < reductionBlockCount
            && sameWeightStorageDomain(
                outputWave, end - 1, end))
            ++end;
        return std::pair {begin, end - begin};
    };
    bool directSingleMxm = singleMxm
        && context.local_weight_dequant && mTileCount == 1
        && throughput.mxm_weight_buffers == 2
        && timeline->reduction_interval == 2 * projectionSlotInterval
        && projectionSlotInterval >= tile;
    if (directSingleMxm) {
        for (int64_t outputWave = 0;
             outputWave < timeline->wave_count && directSingleMxm;
             ++outputWave) {
            const int64_t blockBase = outputWave * reductionBlockCount;
            for (int64_t begin = 0; begin < reductionBlockCount;) {
                const auto [runBegin, runCount] =
                    storageRunBounds(outputWave, begin);
                if (runBegin != begin || runCount <= 0) {
                    directSingleMxm = false;
                    break;
                }
                const auto& signature = timeline->blocks[
                    blockBase + begin];
                if (signature.tiles.size() != 1
                    || signature.tiles.front().segments.size() != 1) {
                    directSingleMxm = false;
                    break;
                }
                const auto& signatureSegment =
                    signature.tiles.front().segments.front();
                for (int64_t offset = 0; offset < runCount; ++offset) {
                    const auto& current = timeline->blocks[
                        blockBase + begin + offset];
                    if (current.active_hemispheres
                            != signature.active_hemispheres
                        || current.tiles.size() != 1
                        || current.tiles.front().segments.size() != 1
                        || current.tiles.front().compute_cycle
                            != signature.tiles.front().compute_cycle
                                + offset * timeline->reduction_interval) {
                        directSingleMxm = false;
                        break;
                    }
                    const auto& segment =
                        current.tiles.front().segments.front();
                    if (segment.rows != signatureSegment.rows
                        || segment.stream_base
                            != signatureSegment.stream_base) {
                        directSingleMxm = false;
                        break;
                    }
                }
                begin += runCount;
            }
        }
    }
    const auto directSlotCycle = [&](int64_t outputWave,
                                      int64_t reduction,
                                      int64_t logicalSlot) {
        const auto [runBegin, runCount] =
            storageRunBounds(outputWave, reduction);
        const int64_t blockBase = outputWave * reductionBlockCount;
        const int64_t runCycle = timeline->blocks[
            blockBase + runBegin].tiles.front().compute_cycle;
        return runCycle
            + (logicalSlot * runCount + reduction - runBegin)
                * projectionSlotInterval;
    };
    const auto hiddenReductionRun = [&](int64_t outputWave,
                                         int64_t reduction,
                                         int64_t mTile,
                                         int64_t step) {
        if (directSingleMxm) {
            const auto [runBegin, runCount] =
                storageRunBounds(outputWave, reduction);
            const int64_t offset = reduction - runBegin;
            if (offset < 0 || offset >= step)
                return std::pair<int64_t, int64_t> {0, 0};
            return std::pair<int64_t, int64_t> {
                1 + (runCount - 1 - offset) / step,
                step * projectionSlotInterval};
        }
        if (mTileCount != 1)
            return std::pair<int64_t, int64_t> {1, 1};
        std::pair<int64_t, int64_t> result {0, 0};
        const int64_t blockBase = outputWave * reductionBlockCount;
        int64_t cursor = reduction % step;
        while (cursor < reductionBlockCount) {
            int64_t count = 1;
            int64_t interval = 1;
            if (cursor + step < reductionBlockCount
                && sameWeightStorageDomain(
                    outputWave, cursor, cursor + step)) {
                interval = timeline->blocks[blockBase + cursor + step]
                    .tiles[mTile]
                    .compute_cycle
                    - timeline->blocks[blockBase + cursor]
                          .tiles[mTile]
                          .compute_cycle;
                while (cursor + count * step < reductionBlockCount) {
                    const int64_t previousReduction =
                        cursor + (count - 1) * step;
                    const int64_t currentReduction = cursor + count * step;
                    // The activation stream is consumed by the MXM domain
                    // for the same dynamic weight page.  Keep its MEM 3-D
                    // loop inside that page so the linker can place one
                    // page-ready SYNC before every participating ICU.  A
                    // hardware ICU cannot stop halfway through an already
                    // decoded 3-D loop while the next page arrives.
                    if (!sameWeightStorageDomain(outputWave,
                            previousReduction, currentReduction))
                        break;
                    const auto& previous = timeline->blocks[
                        blockBase + previousReduction];
                    const auto& current = timeline->blocks[
                        blockBase + currentReduction];
                    if (current.tiles[mTile].compute_cycle
                            - previous.tiles[mTile].compute_cycle
                        != interval)
                        break;
                    ++count;
                }
            }
            if (cursor == reduction) return std::pair {count, interval};
            if (reduction > cursor
                && (reduction - cursor) % step == 0
                && reduction < cursor + count * step)
                return result;
            cursor += count * step;
        }
        return result;
    };
    const auto hasClosedComputeDomain = [&](int64_t outputWave,
                                             int64_t hemisphere) {
        if (singleMxm || !context.local_weight_dequant || mTileCount != 1
            || reductionBlockCount <= 0
            || (reductionBlockCount > 1
                && throughput.mxm_weight_buffers != 2))
            return false;
        const auto [weightCount, weightInterval] =
            reductionRun(outputWave, 0);
        (void)weightInterval;
        if (weightCount != reductionBlockCount) return false;
        const int64_t blockBase = outputWave * reductionBlockCount;
        const auto& first = timeline->blocks[blockBase];
        if (hemisphere >= first.active_hemispheres
            || first.tiles.size() != 1
            || first.tiles.front().segments.size() != 1)
            return false;
        const auto& signature = first.tiles.front().segments.front();
        const int64_t interval = reductionBlockCount > 1
            ? timeline->blocks[blockBase + 1].tiles.front().compute_cycle
                - first.tiles.front().compute_cycle
            : 1;
        if (reductionBlockCount > 1 && interval < signature.rows)
            return false;
        for (int64_t reduction = 0; reduction < reductionBlockCount;
             ++reduction) {
            if (reduction != 0
                && !sameWeightStorageDomain(
                    outputWave, reduction - 1, reduction))
                return false;
            const auto& current = timeline->blocks[blockBase + reduction];
            if (hemisphere >= current.active_hemispheres
                || current.tiles.size() != 1
                || current.tiles.front().segments.size() != 1)
                return false;
            const auto& segment = current.tiles.front().segments.front();
            if (segment.rows != signature.rows
                || segment.stream_base != signature.stream_base
                || current.tiles.front().compute_cycle
                    != first.tiles.front().compute_cycle
                        + reduction * interval)
                return false;
            if (throughput.mxm_weight_buffers == 2
                && current.weight_buffer
                    != (first.weight_buffer ^ (reduction & 1)))
                return false;
        }
        return true;
    };

    for (const FfnDownBlockSchedule& block : timeline->blocks) {
        const int64_t outputWave = block.output_wave;
        const int64_t reduction = block.reduction_block;
        const int64_t activeHemispheres = block.active_hemispheres;
        const int64_t weightBuffer = block.weight_buffer;

        const auto [directRunBegin, directRunCount] =
            storageRunBounds(outputWave, reduction);
        const auto [legacyWeightRunCount, legacyWeightRunInterval] =
            reductionRun(outputWave, reduction);
        for (int64_t hemisphere = 0;
             hemisphere < activeHemispheres; ++hemisphere) {
            const int64_t emittedLogicalSlots = directSingleMxm
                ? 1
                : (singleMxm ? 1 : logicalSlotsPerHemisphere);
            for (int64_t logicalSlot = 0;
                 logicalSlot < emittedLogicalSlots;
                 ++logicalSlot) {
                int64_t weightRunCount = legacyWeightRunCount;
                int64_t weightRunInterval = legacyWeightRunInterval;
                if (directSingleMxm) {
                    // Delay the entire storage run by the buffer-retirement
                    // latency.  The former prelude used the undelayed first
                    // load and split each page into two MEM 3D domains.  Its
                    // later loads already used this offset, so moving only
                    // that prelude gives one affine domain without changing
                    // the steady-state load or compute cycles.
                    weightRunCount = reduction == directRunBegin
                        ? directRunCount : 0;
                    weightRunInterval = projectionSlotInterval;
                }
                if (weightRunCount == 0) continue;
                const int64_t localMxm = singleMxm ? 0 : logicalSlot;
                const int64_t unit =
                    hemisphere * throughput.mxms_per_hemisphere
                    + localMxm;
                const int64_t start = directSingleMxm
                    ? directSlotCycle(
                          outputWave, reduction, logicalSlot) - tile
                        + target.mxm_first_result_latency()
                    : singleMxm
                    ? block.dequant_start + logicalSlot * tile
                    : block.dequant_start
                        + (hemisphere * logicalSlotsPerHemisphere
                              + logicalSlot)
                            * weightLoadCycles;
                const auto placement = context.down_weight_placement;
                const int64_t bindingBase = placement
                        .getAs<mlir::IntegerAttr>("base_row")
                        .getInt();
                const int64_t logicalBase = bindingBase
                    + (outputWave * (intermediate / tile) + reduction)
                        * logicalSlotsPerHemisphere
                        * weightLoadCycles
                    + logicalSlot * weightLoadCycles;
                int64_t base = logicalBase;
                int64_t page = -1;
                int64_t bank = placement
                    .getAs<mlir::IntegerAttr>("bank").getInt();
                llvm::SmallVector<int64_t> selectedWeightSlices(
                    context.down_weight_slices.begin(),
                    context.down_weight_slices.end());
                if (auto paged = placement.getAs<mlir::BoolAttr>(
                        "paged_weight"); paged && paged.getValue()) {
                    const int64_t reductionsPerPage = placement
                        .getAs<mlir::IntegerAttr>("page_granularity")
                        .getInt();
                    const int64_t reductionsPerGroup = placement
                        .getAs<mlir::IntegerAttr>(
                            "page_items_per_slice_group")
                        .getInt();
                    if (reductionsPerPage <= 0
                        || reductionsPerGroup <= 0) {
                        ffn.getOperation()->emitError(
                            "paged Down projection has invalid page geometry");
                        return mlir::failure();
                    }
                    const int64_t pagesPerWave =
                        (intermediate / tile + reductionsPerPage - 1)
                        / reductionsPerPage;
                    page = outputWave * pagesPerWave
                        + reduction / reductionsPerPage;
                    auto pagePlacement = resolve_page_placement(
                        placement, page);
                    if (mlir::failed(pagePlacement)) {
                        ffn.getOperation()->emitError(
                            "paged Down projection has invalid physical page placement")
                            << ": page=" << page;
                        return mlir::failure();
                    }
                    bank = pagePlacement->bank;
                    const int64_t reductionInPage =
                        reduction % reductionsPerPage;
                    const int64_t sliceGroup =
                        pagePlacement->slice_group_base
                        + reductionInPage / reductionsPerGroup;
                    const int64_t localReduction =
                        reductionInPage % reductionsPerGroup;
                    const auto storage = placement
                        .getAs<mlir::ArrayAttr>("page_storage_slices");
                    const int64_t loadSliceCount =
                        static_cast<int64_t>(selectedWeightSlices.size());
                    if (sliceGroup
                            >= pagePlacement->slice_group_base
                                + pagePlacement->slice_group_count) {
                        ffn.getOperation()->emitError(
                            "paged Down reduction exceeds its physical slice groups")
                            << ": page=" << page
                            << ", reduction=" << reduction;
                        return mlir::failure();
                    }
                    selectedWeightSlices.clear();
                    for (int64_t index = 0; index < loadSliceCount; ++index)
                        selectedWeightSlices.push_back(
                            llvm::cast<mlir::IntegerAttr>(
                                storage[sliceGroup * loadSliceCount + index])
                                .getInt());
                    base = bindingBase + pagePlacement->base_row
                        + localReduction * logicalSlotsPerHemisphere
                            * weightLoadCycles
                        + logicalSlot * weightLoadCycles;
                }
                const auto rawType =
                    llvm::cast<mlir::RankedTensorType>(
                        context.down_raw.getInput().getType());
                const auto activationType =
                    llvm::cast<mlir::RankedTensorType>(
                        ffn.getActivation().getType());
                const auto dequantizedType =
                    mlir::RankedTensorType::get(rawType.getShape(),
                        activationType.getElementType());
                FfnLoopDomain3D weightDomain;
                if (directSingleMxm) {
                    // The two logical slots execute consecutively on the
                    // same physical MXM.  Reuse its INT8 stream window and
                    // express slot selection as the outer MEM dimension.
                    weightDomain.wave_count = weightRunCount;
                    weightDomain.wave_interval = weightRunInterval;
                    weightDomain.wave_address_stride =
                        logicalSlotsPerHemisphere * weightLoadCycles;
                    weightDomain.group_count = logicalSlotsPerHemisphere;
                    weightDomain.group_interval =
                        weightRunCount * weightRunInterval;
                    weightDomain.group_address_stride = weightLoadCycles;
                } else if (singleMxm) {
                    weightDomain.wave_count =
                        logicalSlotsPerHemisphere;
                    weightDomain.wave_interval = tile;
                    weightDomain.wave_address_stride = weightLoadCycles;
                    weightDomain.group_count = weightRunCount;
                    weightDomain.group_interval = weightRunInterval;
                    weightDomain.group_address_stride =
                        logicalSlotsPerHemisphere * weightLoadCycles;
                } else {
                    weightDomain.wave_count = weightRunCount;
                    weightDomain.wave_interval = weightRunInterval;
                    weightDomain.wave_address_stride =
                        logicalSlotsPerHemisphere * weightLoadCycles;
                }
                // Down projection keeps its per-window MXM control domains.
                emitFfnWeightTile(rewriter, ffn.getLoc(),
                    context.down_raw,
                    dequantizedType,
                    selectedWeightSlices, target,
                    ffn.getDownRhsScale().convertToFloat(), start,
                    base, hemisphere, logicalSlot, unit,
                    directSingleMxm
                        ? ((logicalSlot * directRunCount) % 2)
                        : (singleMxm ? logicalSlot : weightBuffer),
                    context.local_weight_dequant, bank, page,
                    logicalBase, context.down_weight_placement,
                    weightDomain,
                    directSingleMxm
                        ? (weightRunCount * logicalSlotsPerHemisphere > 1
                                ? "toggle_dim2" : "fixed")
                    : weightRunCount > 1 || singleMxm
                        ? "toggle_dim2" : "fixed");
            }
        }

        for (const FfnDownTileSchedule& tileSchedule : block.tiles) {
            const int64_t mTile = tileSchedule.m_tile;
            const int64_t computeCycle = tileSchedule.compute_cycle;
            const int64_t down0ComputeCycle = directSingleMxm
                ? directSlotCycle(outputWave, reduction, 0)
                : computeCycle;
            const int64_t down1ComputeCycle = directSingleMxm
                ? directSlotCycle(outputWave, reduction, 1)
                : computeCycle
                    + (singleMxm ? projectionSlotInterval : 0);
            downEnd = std::max(downEnd,
                down1ComputeCycle
                    + target.mxm_first_result_latency() + tile);
            for (int64_t hemisphere = 0;
                 hemisphere < activeHemispheres; ++hemisphere) {
                const bool closedComputeDomain =
                    hasClosedComputeDomain(outputWave, hemisphere);
                const bool closedComputeLeader =
                    closedComputeDomain && reduction == 0;
                int64_t rowOffset = 0;
                for (const FfnStreamSegment& segment :
                    tileSchedule.segments) {
                    const int64_t segmentCycle =
                        down0ComputeCycle + rowOffset;
                    const int64_t down0Cycle = segmentCycle;
                    const int64_t down1Cycle =
                        down1ComputeCycle + rowOffset;
                    const auto emitHidden = [&](int64_t consumerCycle,
                                                bool includeSecondSlot) {
                        mlir::Value hiddenValue = swish.hidden;
                        const int64_t reductionStep =
                            hiddenDistributed16 ? 1 : 2;
                        const auto [hiddenRunCount, hiddenRunInterval] =
                            hiddenReductionRun(outputWave, reduction,
                                mTile, reductionStep);
                        if (hiddenRunCount == 0) return hiddenValue;
                        FfnLoopDomain3D slotDomain;
                        if (directSingleMxm) {
                            // RUN_3D iterates dimension 1 before dimension 2.
                            // Put the dense reduction run in dimension 1 and
                            // the two serialized logical output slots in the
                            // outer dimension so the ICU never has to jump
                            // backwards in time.
                            slotDomain.wave_count = hiddenRunCount;
                            slotDomain.wave_interval = hiddenRunInterval;
                            slotDomain.wave_address_stride =
                                hiddenDistributed16
                                ? reductionStep * throughput.tile_rows : m;
                            if (includeSecondSlot) {
                                slotDomain.group_count = 2;
                                slotDomain.group_interval =
                                    down1Cycle - down0Cycle;
                            }
                        } else {
                            if (includeSecondSlot) {
                                slotDomain.wave_count = 2;
                                slotDomain.wave_interval =
                                    down1Cycle - down0Cycle;
                            }
                            slotDomain.group_count = hiddenRunCount;
                            slotDomain.group_interval = hiddenRunInterval;
                            slotDomain.group_address_stride =
                                hiddenDistributed16
                                ? reductionStep * throughput.tile_rows : m;
                        }
                        if (hiddenDistributed16) {
                            const int64_t hiddenBlocks = intermediate / tile;
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
                                const int64_t tokenWithinBlock = token % tile;
                                const int64_t tokenWave = tokenWithinBlock
                                    / blockRows;
                                const int64_t address = hiddenBaseRow
                                    + ((token / tile) * hiddenBlocks
                                          + reduction)
                                        * throughput.tile_rows
                                    + tokenWave;
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    const int64_t slice = context.hidden_slices[
                                        2 * tokenLane + byte];
                                    FfnLoopDomain3D domain = slotDomain;
                                    emitFfnMemTransfer3D(rewriter,
                                        ffn.getLoc(),
                                        consumerCycle + firstOffset
                                            - context.eastMxmLatency(slice),
                                        hemisphere, slice, "read", address,
                                        segment.stream_base + byte,
                                        occurrenceCount, blockRows, 1,
                                        ffn.getHidden0Placement()
                                            .getAs<mlir::IntegerAttr>("bank")
                                            .getInt(),
                                        domain);
                                }
                            }
                            return hiddenValue;
                        }
                        const int64_t hiddenPair = reduction % 2;
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = context.hidden_slices[
                                2 * hiddenPair + byte];
                            emitFfnMemTransfer3D(rewriter,
                                ffn.getLoc(),
                                consumerCycle
                                    - context.eastMxmLatency(slice),
                                hemisphere, slice, "read",
                                hiddenBaseRow + (reduction / 2) * m
                                    + mTile * tile + rowOffset,
                                segment.stream_base + byte,
                                segment.rows, 1, 1,
                                ffn.getHidden0Placement()
                                    .getAs<mlir::IntegerAttr>("bank")
                                    .getInt(),
                                slotDomain);
                        }
                        return hiddenValue;
                    };
                    mlir::Value hidden0 = emitHidden(
                        down0Cycle, singleMxm);
                    mlir::Value hidden1 = hidden0;
                    const int64_t unitBase =
                        hemisphere * throughput.mxms_per_hemisphere;
                    if (directSingleMxm
                        && reduction == directRunBegin) {
                        const bool finalRun =
                            directRunBegin + directRunCount
                            == reductionBlockCount;
                        FfnMxmDomain3D domain;
                        domain.repeat_count = segment.rows;
                        domain.repeat_accumulator_address_stride = 0;
                        domain.group_count = directRunCount;
                        domain.group_interval = projectionSlotInterval;
                        if (directRunCount > 1)
                            domain.weight_buffer_mode = "toggle_dim2";
                        if (finalRun && directRunCount > 1) {
                            domain.terminal_dimension = 2;
                            domain.terminal_accumulator_destination =
                                "stream";
                            domain.terminal_accumulator_clear = true;
                            domain.terminal_accumulator_output_format =
                                "bf16";
                        }
                        const bool finalOnly =
                            finalRun && directRunCount == 1;
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(),
                            down0Cycle, unitBase, "compute", 0, 0,
                            segment.stream_base, 0,
                            context.down_accumulator_base + mTile * tile,
                            1, finalOnly ? "stream" : "sram", true,
                            dataFormat, finalOnly ? "bf16" : "fp32",
                            domain);
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(),
                            down1Cycle, unitBase, "compute",
                            (directRunCount & 1), 0,
                            segment.stream_base,
                            throughput.mxm_result_streams,
                            context.down_accumulator_base + mTile * tile + m,
                            1, finalOnly ? "stream" : "sram", true,
                            dataFormat, finalOnly ? "bf16" : "fp32",
                            domain);
                    } else if (!directSingleMxm && closedComputeLeader) {
                        const int64_t blockBase =
                            outputWave * reductionBlockCount;
                        const int64_t groupInterval =
                            reductionBlockCount > 1
                            ? timeline->blocks[blockBase + 1]
                                  .tiles.front().compute_cycle
                                - computeCycle
                            : 1;
                        FfnMxmDomain3D domain;
                        domain.repeat_count = segment.rows;
                        domain.repeat_accumulator_address_stride = 0;
                        domain.group_count = reductionBlockCount;
                        domain.group_interval = groupInterval;
                        if (reductionBlockCount > 1
                            && throughput.mxm_weight_buffers == 2)
                            domain.weight_buffer_mode = "toggle_dim2";
                        if (reductionBlockCount > 1) {
                            domain.terminal_dimension = 2;
                            domain.terminal_accumulator_destination = "stream";
                            domain.terminal_accumulator_clear = true;
                            domain.terminal_accumulator_output_format =
                                "bf16";
                        }
                        const bool finalOnly = reductionBlockCount == 1;
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), down0Cycle,
                            unitBase, "compute", weightBuffer, 0,
                            segment.stream_base, 0,
                            context.down_accumulator_base + mTile * tile,
                            1, finalOnly ? "stream" : "sram", true,
                            dataFormat, finalOnly ? "bf16" : "fp32",
                            domain);
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), down1Cycle,
                            unitBase + 1, "compute", weightBuffer, 0,
                            segment.stream_base,
                            throughput.mxm_result_streams,
                            context.down_accumulator_base + mTile * tile,
                            1, finalOnly ? "stream" : "sram", true,
                            dataFormat, finalOnly ? "bf16" : "fp32",
                            domain);
                    } else if (!directSingleMxm
                        && !closedComputeDomain) {
                        FfnMxmDomain3D domain;
                        domain.repeat_count = segment.rows;
                        domain.repeat_accumulator_address_stride = 0;
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), down0Cycle,
                            unitBase, "compute",
                            singleMxm ? 0 : weightBuffer, 0,
                            segment.stream_base, 0,
                            context.down_accumulator_base + mTile * tile
                                + rowOffset,
                            1, block.final_reduction ? "stream" : "sram",
                            true, dataFormat,
                            block.final_reduction ? "bf16" : "fp32",
                            domain);
                        emitFfnMxmIssue3D(rewriter, ffn.getLoc(), down1Cycle,
                            unitBase + (singleMxm ? 0 : 1), "compute",
                            singleMxm ? 1 : weightBuffer, 0,
                            segment.stream_base,
                            throughput.mxm_result_streams,
                            context.down_accumulator_base + mTile * tile
                                + rowOffset + (singleMxm ? m : 0),
                            1, block.final_reduction ? "stream" : "sram",
                            true, dataFormat,
                            block.final_reduction ? "bf16" : "fp32",
                            domain);
                    }
                    (void)hidden0;
                    (void)hidden1;
                    rowOffset += segment.rows;
                }

                if (!block.final_reduction) continue;

                for (int64_t byte = 0;
                     byte < throughput.mxm_result_streams; ++byte) {
                    const int64_t streamBase = byte < 2
                        ? 0 : throughput.mxm_result_streams;
                    const int64_t slice = context.result_slices[byte];
                    const auto latency = target.transport_latency(
                        target::StreamEndpoint::MxmResult,
                        target::StreamEndpoint::Mem,
                        target::StreamDirection::West, slice);
                    if (!latency) return mlir::failure();
                    auto placement = schedule_placement(rewriter,
                        {slice}, outputWave * m + mTile * tile,
                        tile, 1, context.hemisphereName(hemisphere),
                        "fp16_pair_planar", resultBank);
                    mlir::NamedAttrList attributes(placement);
                    llvm::SmallVector<mlir::Attribute> allSlices;
                    for (int64_t resultSlice : context.result_slices)
                        allSlices.push_back(
                            rewriter.getI64IntegerAttr(resultSlice));
                    attributes.set("binding_slices",
                        rewriter.getArrayAttr(allSlices));
                    attributes.set("binding_instruction_count",
                        rewriter.getI64IntegerAttr(
                            ffn.getM() * timeline->wave_count));
                    attributes.set("binding_placement",
                        schedule_placement(rewriter,
                            context.result_slices, 0,
                            ffn.getM() * timeline->wave_count, 1,
                            "both", "fp16_pair_planar", resultBank));
                    const int64_t writeCycle =
                        (byte < 2 ? down0ComputeCycle : down1ComputeCycle)
                        + target.mxm_first_result_latency() + *latency;
                    pendingResultWrites[{hemisphere, slice, resultBank}]
                        .push_back(PendingResultWrite {
                            swish.hidden,
                            attributes.getDictionary(rewriter.getContext()),
                            writeCycle, outputWave, mTile, hemisphere,
                            slice, streamBase + byte % 2,
                            outputWave * m + mTile * tile});
                }
            }
        }
    }
    // Plan the whole Down output before emitting MEM instructions.  A result
    // queue that is independent of Down weights/hidden reads and has one
    // equally spaced tile per output wave needs only one non-preemptible ICU
    // WRITE_3D.  Keep the original per-tile path for other layouts/timelines.
    const auto resultSliceIsIndependent = [&](int64_t slice) {
        const auto contains = [&](llvm::ArrayRef<int64_t> slices) {
            return std::find(slices.begin(), slices.end(), slice)
                != slices.end();
        };
        if (contains(context.hidden_slices)
            || contains(context.down_weight_slices))
            return false;
        if (auto storage = downPlacement.getAs<mlir::ArrayAttr>(
                "page_storage_slices")) {
            for (mlir::Attribute entry : storage)
                if (llvm::cast<mlir::IntegerAttr>(entry).getInt() == slice)
                    return false;
        }
        return true;
    };
    for (auto& [resource, writes] : pendingResultWrites) {
        std::sort(writes.begin(), writes.end(),
            [](const PendingResultWrite& lhs,
               const PendingResultWrite& rhs) {
                return lhs.cycle < rhs.cycle;
            });
        const PendingResultWrite& first = writes.front();
        const int64_t waveCount = timeline->wave_count;
        const int64_t cycleStride = writes.size() > 1
            ? writes[1].cycle - first.cycle : 1;
        const int64_t addressStride = writes.size() > 1
            ? writes[1].address - first.address : 0;
        // Every paged Down output wave is produced after a distinct weight
        // residency barrier.  One WRITE_3D cannot remain decoded across that
        // runtime-dependent pause, so keep each wave as its own hardware ICU
        // instruction.  Resident weights retain the larger closed domain.
        bool closedDomain = reductionsPerWeightPage <= 0
            && mTileCount == 1 && waveCount > 1
            && static_cast<int64_t>(writes.size()) == waveCount
            && cycleStride >= tile && addressStride == m
            && resultSliceIsIndependent(first.slice);
        for (int64_t wave = 0;
             wave < static_cast<int64_t>(writes.size()) && closedDomain;
             ++wave) {
            const auto& write = writes[wave];
            closedDomain = write.outputWave == wave && write.mTile == 0
                && write.source == first.source
                && write.hemisphere == first.hemisphere
                && write.slice == first.slice
                && write.stream == first.stream
                && write.cycle == first.cycle + wave * cycleStride
                && write.address == first.address + wave * addressStride;
        }
        const auto emitWrite = [&](const PendingResultWrite& write) {
            return rewriter.create<MemWriteOp>(ffn.getLoc(), write.source,
                write.cycle, tile, write.stream, 1, 0,
                rewriter.getStringAttr("west"), ffn.getResultAddress(),
                write.placement, tile * tile);
        };
        if (closedDomain) {
            auto write = emitWrite(first);
            FfnLoopDomain3D domain;
            domain.wave_count = waveCount;
            domain.wave_interval = cycleStride;
            domain.wave_address_stride = addressStride;
            setFfnLoopDomain3D(write.getOperation(), rewriter, domain);
        } else {
            for (const auto& write : writes)
                emitWrite(write);
        }
    }
    mlir::OperationState timelineState(
        ffn.getLoc(), TimelineOp::getOperationName());
    timelineState.addAttributes({
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("ffn.down.vector")),
        rewriter.getNamedAttr(
            "start", rewriter.getI64IntegerAttr(timeline->phase_start)),
        rewriter.getNamedAttr(
            "end", rewriter.getI64IntegerAttr(downEnd)),
    });
    rewriter.create(timelineState);
    auto resultType = llvm::cast<mlir::RankedTensorType>(
        ffn.getResult().getType());
    mlir::OperationState bindingState(
        ffn.getLoc(), BindingOp::getOperationName());
    bindingState.addTypes(resultType);
    bindingState.addAttributes({
        context.rewriter.getNamedAttr("index",
            context.rewriter.getI64IntegerAttr(0)),
        context.rewriter.getNamedAttr("access",
            context.rewriter.getStringAttr("output")),
        context.rewriter.getNamedAttr("role",
            context.rewriter.getStringAttr("result")),
        context.rewriter.getNamedAttr("name",
            context.rewriter.getStringAttr("ffn.result")),
        context.rewriter.getNamedAttr("bytes",
            context.rewriter.getI64IntegerAttr(
                resultType.getNumElements() * 2)),
        context.rewriter.getNamedAttr(
            "placement", ffn.getResultPlacement()),
    });
    auto output = llvm::cast<BindingOp>(
        context.rewriter.create(bindingState));
    return output.getValue();
}

} // namespace ftlpu::compiler::schedule::ffn_detail
