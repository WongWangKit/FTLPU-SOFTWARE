#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_up_3d_lowering.hpp"

#include <bit>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ftlpu::compiler::schedule {
namespace {

mlir::LogicalResult fail(std::string* error, std::string message)
{
    if (error) *error = std::move(message);
    return mlir::failure();
}

bool positive(std::int64_t value)
{
    return value > 0;
}

std::size_t asSize(std::int64_t value, const char* field)
{
    if (value < 0)
        throw std::invalid_argument(std::string(field)
            + " must be non-negative");
    return static_cast<std::size_t>(value);
}

std::size_t checkedSizeProduct(
    std::size_t lhs, std::size_t rhs, const char* field)
{
    if (lhs != 0
        && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::overflow_error(std::string(field) + " overflows size_t");
    return lhs * rhs;
}

std::int64_t checkedInt64(std::size_t value, const char* field)
{
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error(std::string(field) + " overflows int64");
    return static_cast<std::int64_t>(value);
}

std::size_t addPairOffset(std::int64_t base, std::int64_t firstPair,
    std::int64_t pairStride)
{
    const auto unsignedBase = asSize(base, "start cycle");
    const auto pair = asSize(firstPair, "first pair");
    const auto stride = asSize(pairStride, "pair cycle stride");
    if (pair != 0
        && stride > std::numeric_limits<std::size_t>::max() / pair)
        throw std::overflow_error("3-D region start cycle overflows");
    const auto offset = pair * stride;
    if (unsignedBase > std::numeric_limits<std::size_t>::max() - offset)
        throw std::overflow_error("3-D region start cycle overflows");
    return unsignedBase + offset;
}

std::size_t routeLatency(
    const std::array<std::int64_t, hw::kMemSliceColumns>& latencies,
    std::int64_t slice, const char* field)
{
    const auto index = asSize(slice, "MEM slice");
    if (index >= latencies.size())
        throw std::out_of_range("MEM route-latency slice is outside its hemisphere");
    if (!positive(latencies[index]))
        throw std::invalid_argument(std::string(field) + " must be positive");
    return asSize(latencies[index], field);
}

std::size_t issueBefore(std::size_t consumerCycle,
    std::size_t latency, const char* field)
{
    if (consumerCycle < latency)
        throw std::invalid_argument(std::string(field)
            + " precedes its MEM transport latency");
    return consumerCycle - latency;
}

std::size_t issueAfter(std::size_t producerCycle,
    std::size_t latency, const char* field)
{
    if (producerCycle > std::numeric_limits<std::size_t>::max() - latency)
        throw std::overflow_error(std::string(field) + " overflows");
    return producerCycle + latency;
}

std::size_t memQueue(const FfnUp3DPlacement& placement,
    std::size_t hemisphere, std::int64_t slice, std::int64_t bank)
{
    const auto slices = asSize(placement.mem_slices_per_hemisphere,
        "MEM slices per hemisphere");
    const auto banks = asSize(placement.mem_banks_per_slice,
        "MEM banks per slice");
    const auto physicalSlice = asSize(slice, "MEM slice");
    const auto physicalBank = asSize(bank, "MEM bank");
    if (physicalSlice >= slices)
        throw std::out_of_range("MEM slice is outside its hemisphere");
    if (physicalBank >= banks)
        throw std::out_of_range("MEM bank is outside its slice");
    return (hemisphere * slices + physicalSlice) * banks + physicalBank;
}

template <std::size_t N>
void validateUniqueSlices(const std::array<std::int64_t, N>& slices,
    std::int64_t sliceCount, const char* field)
{
    std::unordered_set<std::int64_t> unique;
    for (const auto slice : slices) {
        if (slice < 0 || slice >= sliceCount)
            throw std::out_of_range(std::string(field)
                + " is outside its hemisphere");
        if (!unique.insert(slice).second)
            throw std::invalid_argument(std::string(field)
                + " entries must be unique");
    }
}

} // namespace

mlir::FailureOr<FfnUp3DLoweringResult> lowerFfnUpToFu3D(
    FfnUp3DShape shape, const FfnUp3DPlacement& placement,
    const FfnUp3DTimeline& timeline,
    const FfnUp3DRouteLatencies& routeLatencies,
    std::uint16_t dequantScaleBf16, std::string* error)
{
    if (error) error->clear();
    if (shape.m != static_cast<std::int64_t>(hw::kMxmRows)
        || !positive(shape.k) || !positive(shape.n)
        || shape.k % static_cast<std::int64_t>(hw::kMxmRows) != 0)
        return fail(error,
            "direct FFN Up 3-D lowering currently requires one full MXM row tile and tile-aligned K");
    if (placement.hemispheres.size() != hw::kHemispheres
        || placement.mem_slices_per_hemisphere
            != static_cast<std::int64_t>(hw::kMemSliceColumns)
        || placement.mem_banks_per_slice
            != static_cast<std::int64_t>(hw::kMemBanksPerSlice)
        || placement.mxms_per_hemisphere <= 0
        || placement.mxms_per_hemisphere
            > static_cast<std::int64_t>(hw::kMxmsPerHemisphere))
        return fail(error,
            "FFN Up 3-D placement does not match physical FU geometry");

    const auto hemisphereCount = static_cast<std::int64_t>(
        placement.hemispheres.size());
    const auto tile = static_cast<std::int64_t>(hw::kMxmRows);
    if (shape.n % (hemisphereCount * tile) != 0)
        return fail(error,
            "FFN Up output size is not divisible by hemisphere tile width");
    const auto pairCount = shape.n / (hemisphereCount * tile);
    const auto reductionCount = shape.k / tile;
    const auto pulseCount = static_cast<std::int64_t>(
        hw::kMxmRows / hw::kLanesPerTile);
    if (pulseCount <= 0
        || !positive(timeline.reduction_cycle_stride)
        || !positive(timeline.pair_cycle_stride)
        || timeline.mxm_load_start_cycle
            != timeline.mxm_dequant_start_cycle
        || placement.weight_outer_group_size <= 0
        || !std::has_single_bit(static_cast<std::uint64_t>(
            placement.weight_outer_group_size)))
        return fail(error,
            "FFN Up 3-D loop strides must be positive, load/dequant must be paired, and the weight address group must be a power of two");

    try {
        if (placement.weight_stream_base < 0
            || placement.weight_stream_base
                    + static_cast<std::int64_t>(
                        hw::kMxmInt8WeightStreamsPerCycle)
                > static_cast<std::int64_t>(hw::kEastStreams)
            || placement.activation_stream_base < 0
            || placement.activation_stream_base
                    + static_cast<std::int64_t>(sizeof(std::uint16_t))
                > static_cast<std::int64_t>(hw::kEastStreams))
            throw std::out_of_range(
                "FFN Up input stream range exceeds the east stream set");
        if (placement.activation_bank < 0
            || placement.activation_bank
                >= placement.mem_banks_per_slice
            || placement.result_bank < 0
            || placement.result_bank >= placement.mem_banks_per_slice)
            throw std::out_of_range(
                "FFN Up activation/result bank is outside a MEM slice");

        std::unordered_set<std::int64_t> mxmQueues;
        for (std::size_t h = 0; h < placement.hemispheres.size(); ++h) {
            const auto& hemisphere = placement.hemispheres[h];
            if (hemisphere.mxm_queue < 0
                || hemisphere.mxm_queue
                    >= static_cast<std::int64_t>(hw::kHemispheres)
                        * placement.mxms_per_hemisphere
                || hemisphere.mxm_queue
                        / placement.mxms_per_hemisphere
                    != static_cast<std::int64_t>(h)
                || !mxmQueues.insert(hemisphere.mxm_queue).second)
                throw std::invalid_argument(
                    "FFN Up hemispheres must name valid executable-local MXM queues");
            if (hemisphere.result_stream_base < 0
                || hemisphere.result_stream_base
                        + static_cast<std::int64_t>(sizeof(std::uint16_t))
                    > static_cast<std::int64_t>(hw::kWestStreams))
                throw std::out_of_range(
                    "FFN Up result stream range exceeds the west stream set");
            validateUniqueSlices(hemisphere.activation_slices,
                placement.mem_slices_per_hemisphere,
                "FFN Up activation slice");
            validateUniqueSlices(hemisphere.result_slices,
                placement.mem_slices_per_hemisphere,
                "FFN Up result slice");

            std::int64_t nextPair = 0;
            for (const auto& region : hemisphere.weight_regions) {
                if (region.first_pair != nextPair || region.pair_count <= 0)
                    throw std::invalid_argument(
                        "FFN Up weight regions must cover pairs contiguously");
                if (region.pair_count > pairCount - nextPair)
                    throw std::out_of_range(
                        "FFN Up weight region exceeds the output pair domain");
                if (region.first_slice < 0
                    || region.first_slice
                            + static_cast<std::int64_t>(
                                hw::kMxmInt8WeightStreamsPerCycle)
                        > placement.mem_slices_per_hemisphere)
                    throw std::out_of_range(
                        "FFN Up weight slice group exceeds its hemisphere");
                if (region.bank < 0
                    || region.bank >= placement.mem_banks_per_slice)
                    throw std::out_of_range(
                        "FFN Up weight bank is outside a MEM slice");
                asSize(region.base_address, "weight base address");
                nextPair += region.pair_count;
            }
            if (nextPair != pairCount)
                throw std::invalid_argument(
                    "FFN Up weight regions do not cover the output pair domain");
        }

        FfnUp3DLoweringResult result;
        const auto loopReductionCount = asSize(reductionCount,
            "reduction count");
        const auto loopPairCount = asSize(pairCount, "pair count");
        const auto reductionStride = asSize(
            timeline.reduction_cycle_stride, "reduction cycle stride");
        const auto pairStride = asSize(
            timeline.pair_cycle_stride, "pair cycle stride");
        const auto groupSize = asSize(placement.weight_outer_group_size,
            "weight outer group size");
        const auto pulseRows = asSize(pulseCount, "weight pulse count");
        const auto weightMiddleStrideSize = checkedSizeProduct(
            pulseRows, groupSize, "weight middle address stride");
        const auto weightMiddleStride = checkedInt64(
            weightMiddleStrideSize, "weight middle address stride");
        const auto weightOuterGroupStrideSize = checkedSizeProduct(
            loopReductionCount, weightMiddleStrideSize,
            "weight outer-group address stride");
        const auto weightOuterGroupStride = checkedInt64(
            weightOuterGroupStrideSize,
            "weight outer-group address stride");

        // Weight MEM reads: one descriptor per physical source slice and
        // resident region. No loop point is ever materialized here.
        for (std::size_t h = 0; h < placement.hemispheres.size(); ++h) {
            const auto& hemisphere = placement.hemispheres[h];
            for (const auto& region : hemisphere.weight_regions) {
                const auto address = MemIcuAddress3D::BlockedOuter(
                    asSize(region.base_address, "weight base address"),
                    1,
                    weightMiddleStride,
                    groupSize,
                    static_cast<std::int64_t>(pulseRows),
                    weightOuterGroupStride);
                for (std::size_t lane = 0;
                     lane < hw::kMxmInt8WeightStreamsPerCycle; ++lane) {
                    const auto slice = region.first_slice
                        + static_cast<std::int64_t>(lane);
                    const auto consumerStart = addPairOffset(
                        timeline.mxm_load_start_cycle,
                        region.first_pair, timeline.pair_cycle_stride);
                    const auto cycle = issueBefore(consumerStart,
                        routeLatency(
                            routeLatencies.mem_to_mxm_weight_cycles,
                            slice, "MEM-to-MXM weight latency"),
                        "MXM weight consumer anchor");
                    const IcuLoop3D loop {
                        0,
                        {pulseRows, loopReductionCount,
                            asSize(region.pair_count, "weight pair count")},
                        {1, reductionStride, pairStride},
                    };
                    auto instruction = MemIcuInstruction::Read3D(loop,
                        address, StreamId::East(
                            asSize(placement.weight_stream_base,
                                "weight stream base") + lane));
                    detail::validate_mem_icu_instruction(instruction);
                    result.mem_commands.push_back({
                        memQueue(placement, h, slice, region.bank),
                        cycle,
                        instruction});
                }
            }
        }

        // Distributed activation layout: each slice owns one token lane and
        // one BF16 byte. Four row waves cover the 32-token prefill tile.
        for (std::size_t h = 0; h < placement.hemispheres.size(); ++h) {
            const auto& hemisphere = placement.hemispheres[h];
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto address = MemIcuAddress3D::Affine(
                    asSize(placement.activation_base_address,
                        "activation base address"),
                    {1, static_cast<std::int64_t>(pulseRows), 0});
                for (std::size_t byte = 0; byte < sizeof(std::uint16_t);
                     ++byte) {
                    const auto slice = hemisphere.activation_slices[
                        2 * lane + byte];
                    auto consumerStart = asSize(
                        timeline.mxm_compute_start_cycle,
                        "MXM compute start cycle");
                    if (consumerStart
                        > std::numeric_limits<std::size_t>::max() - lane)
                        throw std::overflow_error(
                            "activation consumer anchor overflows");
                    consumerStart += lane;
                    const auto cycle = issueBefore(consumerStart,
                        routeLatency(
                            routeLatencies.mem_to_mxm_activation_cycles,
                            slice, "MEM-to-MXM activation latency"),
                        "MXM activation consumer anchor");
                    const IcuLoop3D loop {
                        0,
                        {pulseRows, loopReductionCount, loopPairCount},
                        {hw::kLanesPerTile, reductionStride, pairStride},
                    };
                    auto instruction = MemIcuInstruction::Read3D(loop,
                        address, StreamId::East(
                            asSize(placement.activation_stream_base,
                                "activation stream base") + byte));
                    detail::validate_mem_icu_instruction(instruction);
                    result.mem_commands.push_back({
                        memQueue(placement, h, slice,
                            placement.activation_bank),
                        cycle,
                        instruction});
                }
            }
        }

        // Final BF16 result drains into two byte slices in each hemisphere.
        for (std::size_t h = 0; h < placement.hemispheres.size(); ++h) {
            const auto& hemisphere = placement.hemispheres[h];
            const auto address = MemIcuAddress3D::Affine(
                asSize(placement.result_base_address,
                    "result base address"),
                {1, shape.m, 0});
            for (std::size_t byte = 0; byte < sizeof(std::uint16_t);
                 ++byte) {
                const auto slice = hemisphere.result_slices[byte];
                const auto cycle = issueAfter(
                    asSize(timeline.mxm_result_start_cycle,
                        "MXM result start cycle"),
                    routeLatency(
                        routeLatencies.mxm_result_to_mem_cycles,
                        slice, "MXM-to-MEM result latency"),
                    "MEM result consumer cycle");
                const IcuLoop3D loop {
                    0,
                    {asSize(shape.m, "M"), loopPairCount, 1},
                    {1, pairStride, 1},
                };
                auto instruction = MemIcuInstruction::Write3D(loop,
                    address, StreamId::West(
                        asSize(hemisphere.result_stream_base,
                            "result stream base") + byte));
                detail::validate_mem_icu_instruction(instruction);
                result.mem_commands.push_back({
                    memQueue(placement, h, slice,
                        placement.result_bank),
                    cycle,
                    instruction});
            }
        }

        const IcuLoop3D loadLoop {
            0,
            {pulseRows, loopReductionCount, loopPairCount},
            {1, reductionStride, pairStride},
        };
        const IcuLoop3D dequantLoop {
            0,
            loadLoop.counts, loadLoop.cycle_strides,
        };
        const IcuLoop3D computeLoop {
            0,
            {asSize(shape.m, "M"), loopReductionCount, loopPairCount},
            {1, reductionStride, pairStride},
        };
        const MxmComputeIcuMode regularMode {
            MxmAccumulatorDestination::Sram,
            false,
            MxmAccumulatorOutputFormat::Float32,
        };
        const MxmComputeIcuMode terminalMode {
            MxmAccumulatorDestination::Stream,
            true,
            MxmAccumulatorOutputFormat::BFloat16,
        };
        for (const auto& hemisphere : placement.hemispheres) {
            const auto queue = asSize(hemisphere.mxm_queue,
                "executable-local MXM queue");
            auto load = MxmLoadIcuInstruction::Load3D(loadLoop, 0,
                MxmIcuBufferMode::ToggleDimension1, 0, {1, 0, 0},
                asSize(placement.weight_stream_base,
                    "weight stream base"),
                MxmWeightInputMode::Int8DequantBf16);
            auto dequant = MxmDequantIcuInstruction::Dequant3D(
                dequantLoop,
                MxmDequantInstruction::ScaleBits(dequantScaleBf16));
            auto compute = MxmComputeIcuInstruction::Compute3D(
                computeLoop, 0, MxmIcuBufferMode::ToggleDimension1,
                asSize(placement.activation_stream_base,
                    "activation stream base"),
                asSize(hemisphere.result_stream_base,
                    "result stream base"),
                asSize(placement.accumulator_address_base,
                    "accumulator address base"),
                {0, 0, 0}, 1, MxmDataFormat::BFloat16,
                regularMode, 1, terminalMode);
            detail::validate_mxm_load_icu_instruction(load);
            detail::validate_mxm_dequant_icu_instruction(dequant);
            detail::validate_mxm_compute_icu_instruction(compute);
            result.mxm_load_commands.push_back({queue,
                asSize(timeline.mxm_load_start_cycle,
                    "MXM load start cycle"),
                load});
            result.mxm_dequant_commands.push_back({queue,
                asSize(timeline.mxm_dequant_start_cycle,
                    "MXM dequant start cycle"),
                dequant});
            result.mxm_compute_commands.push_back({queue,
                asSize(timeline.mxm_compute_start_cycle,
                    "MXM compute start cycle"),
                compute});
        }
        return result;
    } catch (const std::exception& exception) {
        return fail(error, exception.what());
    }
}

} // namespace ftlpu::compiler::schedule
