#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Command/Transforms/fu_3d_command_materializer.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_up_3d_lowering.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"
#include "ftlpu/core/bf16.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace ftlpu::compiler {
namespace {

std::optional<std::int64_t> integer(
    mlir::DictionaryAttr dictionary, llvm::StringRef name)
{
    if (!dictionary) return std::nullopt;
    if (auto attribute = dictionary.getAs<mlir::IntegerAttr>(name))
        return attribute.getInt();
    return std::nullopt;
}

std::optional<llvm::SmallVector<std::int64_t>> integerArray(
    mlir::DictionaryAttr dictionary, llvm::StringRef name)
{
    if (!dictionary) return std::nullopt;
    const auto array = dictionary.getAs<mlir::ArrayAttr>(name);
    if (!array) return std::nullopt;
    llvm::SmallVector<std::int64_t> values;
    values.reserve(array.size());
    for (mlir::Attribute attribute : array) {
        const auto value = llvm::dyn_cast<mlir::IntegerAttr>(attribute);
        if (!value) return std::nullopt;
        values.push_back(value.getInt());
    }
    return values;
}

std::optional<llvm::SmallVector<std::int64_t>> integerArray(
    mlir::ArrayAttr array)
{
    if (!array) return std::nullopt;
    llvm::SmallVector<std::int64_t> values;
    values.reserve(array.size());
    for (mlir::Attribute attribute : array) {
        const auto value = llvm::dyn_cast<mlir::IntegerAttr>(attribute);
        if (!value) return std::nullopt;
        values.push_back(value.getInt());
    }
    return values;
}

std::optional<std::int64_t> tensorBytes(mlir::RankedTensorType type)
{
    if (!type.hasStaticShape()) return std::nullopt;
    const auto width = type.getElementType().getIntOrFloatBitWidth();
    if (width == 0 || width % 8 != 0) return std::nullopt;
    const auto elements = type.getNumElements();
    if (elements < 0
        || elements > std::numeric_limits<std::int64_t>::max()
                / static_cast<std::int64_t>(width / 8))
        return std::nullopt;
    return elements * static_cast<std::int64_t>(width / 8);
}

bool stringEquals(mlir::DictionaryAttr dictionary,
    llvm::StringRef name, llvm::StringRef expected)
{
    const auto value = dictionary
        ? dictionary.getAs<mlir::StringAttr>(name) : mlir::StringAttr {};
    return value && value.getValue() == expected;
}

bool addressMatches(mlir::DictionaryAttr address, std::int64_t bank,
    std::int64_t slice, std::int64_t word)
{
    return integer(address, "bank") == bank
        && integer(address, "slice") == slice
        && integer(address, "word") == word
        && integer(address, "byte") == 0
        && integer(address, "device") == 0
        && stringEquals(address, "hemisphere", "east");
}

mlir::LogicalResult buildLoweringInput(stream::MatmulTaskOp op,
    const target::LPUTargetModel& target,
    schedule::FfnUp3DShape& shape,
    schedule::FfnUp3DPlacement& placement,
    schedule::FfnUp3DTimeline& timeline,
    schedule::FfnUp3DRouteLatencies& routeLatencies,
    std::uint16_t& scaleBits)
{
    const auto fail = [&](llvm::Twine message) {
        op.emitError(message);
        return mlir::failure();
    };
    const auto config = op.getConfig();
    const auto role = config.getAs<mlir::StringAttr>("projection_kind");
    if (!role || role.getValue() != "up")
        return fail("direct FU 3-D mode requires config.projection_kind = \"up\"");
    const auto direct =
        config.getAs<mlir::DictionaryAttr>("direct_3d");
    if (!direct)
        return fail("direct FU 3-D Up requires config.direct_3d");
    if (op.getLhs().size() != 1 || op.getRhs().size() != 1)
        return fail("direct FU 3-D Up requires one activation and one weight route");
    auto activationRoute =
        op.getLhs().front().getDefiningOp<stream::RouteOp>();
    auto weightRoute =
        op.getRhs().front().getDefiningOp<stream::RouteOp>();
    if (!activationRoute || !weightRoute
        || activationRoute.getSource() != "MEM"
        || activationRoute.getDestination() != "MXM.activation"
        || weightRoute.getSource() != "MEM"
        || weightRoute.getDestination() != "MXM.weight"
        || activationRoute.getDirection() != "east"
        || weightRoute.getDirection() != "east"
        || activationRoute.getSourceUnitId() != -1
        || weightRoute.getSourceUnitId() != -1)
        return fail("direct FU 3-D Up requires canonical MEM-to-MXM routes");

    const auto activationType = llvm::dyn_cast<mlir::RankedTensorType>(
        activationRoute.getInput().getType());
    const auto weightType = llvm::dyn_cast<mlir::RankedTensorType>(
        weightRoute.getInput().getType());
    const auto resultType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!activationType || !weightType || !resultType
        || !activationType.getElementType().isBF16()
        || !weightType.getElementType().isInteger(8)
        || !resultType.getElementType().isBF16())
        return fail("direct FU 3-D Up supports BF16 x INT8 -> BF16 tensors");

    shape = {static_cast<std::int64_t>(op.getM()),
        static_cast<std::int64_t>(op.getK()),
        static_cast<std::int64_t>(op.getN())};
    if (activationType.getRank() != 2 || weightType.getRank() != 2
        || resultType.getRank() != 2
        || activationType.getDimSize(0) != shape.m
        || activationType.getDimSize(1) != shape.k
        || weightType.getDimSize(0) != shape.k
        || weightType.getDimSize(1) != shape.n
        || resultType.getDimSize(0) != shape.m
        || resultType.getDimSize(1) != shape.n)
        return fail("direct FU 3-D Up tensor shapes must exactly match M, K, and N");
    if (target.memory().hemispheres
            != static_cast<std::int64_t>(hw::kHemispheres)
        || target.memory().slices_per_hemisphere
            != static_cast<std::int64_t>(hw::kMemSliceColumns)
        || target.memory().banks_per_slice
            != static_cast<std::int64_t>(hw::kMemBanksPerSlice)
        || target.throughput().mxms_per_hemisphere <= 0
        || target.throughput().mxms_per_hemisphere
            > static_cast<std::int64_t>(hw::kMxmsPerHemisphere))
        return fail("direct FU 3-D Up target does not match the hardware ICU geometry");

    const auto unitIds = integerArray(op.getUnitIds());
    const auto weightBuffers = integerArray(op.getWeightBuffers());
    const auto taskResultBases = integerArray(op.getResultStreamBases());
    const auto taskResultCounts = integerArray(op.getResultStreamCounts());
    if (!unitIds || unitIds->size() != 1
        || unitIds->front() < 0
        || unitIds->front() >= target.throughput().mxms_per_hemisphere
        || !weightBuffers || *weightBuffers != llvm::SmallVector<std::int64_t> {0}
        || !taskResultBases || taskResultBases->size() != 1
        || !taskResultCounts
        || *taskResultCounts != llvm::SmallVector<std::int64_t> {
               target.throughput().mxm_activation_streams})
        return fail("direct FU 3-D Up requires one local MXM, buffer zero, and one native-width result stream");
    if (activationRoute.getDestinationUnitId() != unitIds->front()
        || weightRoute.getDestinationUnitId() != unitIds->front()
        || activationRoute.getStreamCount()
            != target.throughput().mxm_activation_streams
        || weightRoute.getStreamCount()
            != target.throughput().mxm_int8_load_streams_per_cycle)
        return fail("direct FU 3-D Up route unit or stream width does not match the selected MXM");
    const auto activationAddressSlice =
        integer(activationRoute.getAddress(), "slice");
    const auto weightAddressSlice = integer(weightRoute.getAddress(), "slice");
    if (!activationAddressSlice || !weightAddressSlice)
        return fail("direct FU 3-D Up routes require concrete MEM slices");
    const auto activationRegister = target.stream_register_id(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, *activationAddressSlice);
    const auto activationLatency = target.transport_latency(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, *activationAddressSlice);
    const auto weightRegister = target.stream_register_id(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight,
        target::StreamDirection::East, *weightAddressSlice);
    const auto weightLatency = target.transport_latency(
        target::StreamEndpoint::Mem,
        target::StreamEndpoint::MxmWeight,
        target::StreamDirection::East, *weightAddressSlice);
    if (!activationRegister || !activationLatency || !weightRegister
        || !weightLatency
        || activationRoute.getRegisterId() != *activationRegister
        || activationRoute.getTransportLatency() != *activationLatency
        || weightRoute.getRegisterId() != *weightRegister
        || weightRoute.getTransportLatency() != *weightLatency)
        return fail("direct FU 3-D Up route register/latency does not match the target topology");
    if (tensorBytes(activationType) != activationRoute.getBytes()
        || tensorBytes(weightType) != weightRoute.getBytes())
        return fail("direct FU 3-D Up route byte counts must match their tensors");

    const auto activationSlices =
        integerArray(activationRoute.getPlacement(), "slices");
    if (!activationSlices
        || activationSlices->size()
            != 2 * hw::kLanesPerTile)
        return fail("activation route must provide sixteen distributed BF16 byte slices");
    const auto activationBank =
        integer(activationRoute.getPlacement(), "bank");
    const auto activationBase =
        integer(activationRoute.getPlacement(), "base_row");
    if (!activationBank || !activationBase
        || !stringEquals(activationRoute.getPlacement(), "kind",
            "fp16_mxm_distributed_16")
        || !stringEquals(activationRoute.getPlacement(), "hemisphere", "both")
        || integer(activationRoute.getPlacement(), "address_stride") != 1
        || !addressMatches(activationRoute.getAddress(), *activationBank,
            activationSlices->front(), *activationBase))
        return fail("activation route placement/address must name the distributed BF16 SRAM region");

    if (op.getResultAllocations().size() != 1)
        return fail("direct FU 3-D Up requires one physical result allocation");
    const auto resultAllocation = llvm::dyn_cast<mlir::DictionaryAttr>(
        op.getResultAllocations()[0]);
    const auto resultPlacement = resultAllocation
        ? resultAllocation.getAs<mlir::DictionaryAttr>("placement")
        : mlir::DictionaryAttr {};
    const auto resultSlices = integerArray(resultPlacement, "slices");
    const auto resultBank = integer(resultPlacement, "bank");
    const auto resultBase = integer(resultPlacement, "base_row");
    const auto resultBytes = integer(resultAllocation, "bytes");
    const auto resultAddress = resultAllocation
        ? resultAllocation.getAs<mlir::DictionaryAttr>("address")
        : mlir::DictionaryAttr {};
    if (!resultAllocation || !resultSlices
        || resultSlices->size() != sizeof(std::uint16_t)
        || !resultBank || !resultBase || !resultBytes
        || tensorBytes(resultType) != resultBytes
        || !stringEquals(resultPlacement, "kind", "fp16_pair_planar")
        || !stringEquals(resultPlacement, "hemisphere", "both")
        || integer(resultPlacement, "address_stride") != 1
        || !addressMatches(resultAddress, *resultBank,
            resultSlices->front(), *resultBase))
        return fail("result allocation must exactly describe the two-slice, dual-hemisphere BF16 SRAM result");

    const auto resultStreamBases =
        integerArray(direct, "result_stream_bases");
    const auto accumulatorBase = integer(direct, "accumulator_address_base");
    const auto outerGroup = integer(direct, "weight_outer_group_size");
    if (!resultStreamBases
        || resultStreamBases->size() != hw::kHemispheres
        || !accumulatorBase || !outerGroup)
        return fail("direct_3d requires per-hemisphere result streams, one MXM unit, accumulator base, and weight group size");
    if (taskResultBases->front() != resultStreamBases->front())
        return fail("matmul result stream must match the east direct_3d result stream");

    const auto regionAttrs =
        direct.getAs<mlir::ArrayAttr>("weight_regions");
    if (!regionAttrs || regionAttrs.empty())
        return fail("direct_3d requires closed-form weight_regions");
    llvm::SmallVector<schedule::FfnUpWeightRegion3DPlacement> regions;
    regions.reserve(regionAttrs.size());
    for (mlir::Attribute attribute : regionAttrs) {
        const auto dictionary = llvm::dyn_cast<mlir::DictionaryAttr>(attribute);
        const auto firstPair = integer(dictionary, "first_pair");
        const auto pairCount = integer(dictionary, "pair_count");
        const auto bank = integer(dictionary, "bank");
        const auto firstSlice = integer(dictionary, "first_slice");
        const auto baseAddress = integer(dictionary, "base_address");
        if (!dictionary || !firstPair || !pairCount || !bank
            || !firstSlice || !baseAddress)
            return fail("each direct_3d weight region requires first_pair, pair_count, bank, first_slice, and base_address");
        regions.push_back({*firstPair, *pairCount, *bank,
            *firstSlice, *baseAddress});
    }
    const auto weightSlices =
        integerArray(weightRoute.getPlacement(), "slices");
    std::vector<std::int64_t> regionSlices;
    for (const auto& region : regions)
        for (std::size_t lane = 0;
             lane < hw::kMxmInt8WeightStreamsPerCycle; ++lane)
            regionSlices.push_back(region.first_slice
                + static_cast<std::int64_t>(lane));
    std::ranges::sort(regionSlices);
    regionSlices.erase(std::unique(regionSlices.begin(), regionSlices.end()),
        regionSlices.end());
    auto declaredWeightSlices = weightSlices
        ? std::vector<std::int64_t>(weightSlices->begin(), weightSlices->end())
        : std::vector<std::int64_t> {};
    std::ranges::sort(declaredWeightSlices);
    const auto weightBank = integer(weightRoute.getPlacement(), "bank");
    const auto weightBase = integer(weightRoute.getPlacement(), "base_row");
    if (!weightSlices || declaredWeightSlices != regionSlices
        || !weightBank || !weightBase
        || !stringEquals(weightRoute.getPlacement(), "kind",
            "w8a16_direct_3d")
        || !stringEquals(weightRoute.getPlacement(), "hemisphere", "both")
        || integer(weightRoute.getPlacement(), "address_stride") != 1
        || *weightBank != regions.front().bank
        || *weightBase != regions.front().base_address
        || !addressMatches(weightRoute.getAddress(), regions.front().bank,
            regions.front().first_slice, regions.front().base_address))
        return fail("weight route placement/address must exactly cover the direct_3d weight regions");

    placement.mem_slices_per_hemisphere =
        target.memory().slices_per_hemisphere;
    placement.mem_banks_per_slice = target.memory().banks_per_slice;
    placement.mxms_per_hemisphere =
        target.throughput().mxms_per_hemisphere;
    placement.weight_stream_base = weightRoute.getStreamBase();
    placement.activation_stream_base = activationRoute.getStreamBase();
    placement.activation_bank = *activationBank;
    placement.activation_base_address = *activationBase;
    placement.result_bank = *resultBank;
    placement.result_base_address = *resultBase;
    placement.accumulator_address_base = *accumulatorBase;
    placement.weight_outer_group_size = *outerGroup;
    placement.hemispheres.resize(hw::kHemispheres);
    for (std::size_t hemisphere = 0;
         hemisphere < hw::kHemispheres; ++hemisphere) {
        auto& physical = placement.hemispheres[hemisphere];
        physical.weight_regions.assign(regions.begin(), regions.end());
        std::copy(activationSlices->begin(), activationSlices->end(),
            physical.activation_slices.begin());
        std::copy(resultSlices->begin(), resultSlices->end(),
            physical.result_slices.begin());
        // Binary queue ids are dense in the executable's logical topology.
        // The runtime maps them onto the CModel's physical MXM stride.
        physical.mxm_queue = static_cast<std::int64_t>(hemisphere)
                * target.throughput().mxms_per_hemisphere
            + unitIds->front();
        physical.result_stream_base = (*resultStreamBases)[hemisphere];
    }

    const auto timelineAttr =
        direct.getAs<mlir::DictionaryAttr>("timeline");
    const auto loadStart = integer(timelineAttr, "mxm_load_start_cycle");
    const auto dequantStart =
        integer(timelineAttr, "mxm_dequant_start_cycle");
    const auto computeStart =
        integer(timelineAttr, "mxm_compute_start_cycle");
    const auto resultStart =
        integer(timelineAttr, "mxm_result_start_cycle");
    const auto reductionStride =
        integer(timelineAttr, "reduction_cycle_stride");
    const auto pairStride =
        integer(timelineAttr, "pair_cycle_stride");
    if (!timelineAttr || !loadStart || !dequantStart || !computeStart
        || !resultStart || !reductionStride || !pairStride)
        return fail("direct_3d.timeline is incomplete");
    timeline = {*loadStart, *dequantStart, *computeStart, *resultStart,
        *reductionStride, *pairStride};

    for (std::size_t slice = 0;
         slice < hw::kMemSliceColumns; ++slice) {
        const auto weightLatency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East,
            static_cast<std::int64_t>(slice));
        const auto activationLatency = target.transport_latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmActivation,
            target::StreamDirection::East,
            static_cast<std::int64_t>(slice));
        const auto resultLatency = target.transport_latency(
            target::StreamEndpoint::MxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::West,
            static_cast<std::int64_t>(slice));
        if (!weightLatency || !activationLatency || !resultLatency)
            return fail("target is missing a required direct FU 3-D route");
        routeLatencies.mem_to_mxm_weight_cycles[slice] = *weightLatency;
        routeLatencies.mem_to_mxm_activation_cycles[slice] =
            *activationLatency;
        routeLatencies.mxm_result_to_mem_cycles[slice] = *resultLatency;
    }

    const auto scale = config.getAs<mlir::FloatAttr>("rhs_scale");
    if (!scale || !std::isfinite(scale.getValueAsDouble())
        || scale.getValueAsDouble() <= 0.0)
        return fail("direct FU 3-D Up requires a positive rhs_scale");
    scaleBits = Bf16::from_float(
        static_cast<float>(scale.getValueAsDouble())).bits();
    return mlir::success();
}

class LowerStandaloneFfnUpTo3DCommandPass final
    : public mlir::PassWrapper<LowerStandaloneFfnUpTo3DCommandPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        LowerStandaloneFfnUpTo3DCommandPass)

    llvm::StringRef getArgument() const final
    {
        return "ftlpu-standalone-ffn-up-to-3d-command";
    }

    llvm::StringRef getDescription() const final
    {
        return "Lower one explicitly marked standalone FFN Up projection directly to FU 3-D Command IR";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()
            || function.getFunctionType().getNumInputs() != 2
            || function.getFunctionType().getNumResults() != 1) {
            function.emitError(
                "direct FU 3-D Up lowering requires one defined block, two inputs, and one result");
            signalPassFailure();
            return;
        }

        mlir::Block& block = function.getBody().front();
        llvm::SmallVector<stream::MatmulTaskOp> matmuls;
        llvm::SmallVector<stream::RouteOp> routes;
        llvm::SmallVector<mlir::func::ReturnOp> returns;
        mlir::Operation* unsupported = nullptr;
        for (mlir::Operation& operation : block.getOperations()) {
            if (auto route = llvm::dyn_cast<stream::RouteOp>(&operation)) {
                routes.push_back(route);
                continue;
            }
            if (auto matmul =
                    llvm::dyn_cast<stream::MatmulTaskOp>(&operation)) {
                matmuls.push_back(matmul);
                continue;
            }
            if (auto returnOp =
                    llvm::dyn_cast<mlir::func::ReturnOp>(&operation)) {
                returns.push_back(returnOp);
                continue;
            }
            if (!unsupported) unsupported = &operation;
        }
        if (unsupported) {
            unsupported->emitError(
                "standalone direct FU 3-D Up permits only two routes, one matmul task, and one return");
            signalPassFailure();
            return;
        }
        if (routes.size() != 2 || matmuls.size() != 1
            || returns.size() != 1) {
            function.emitError(
                "standalone direct FU 3-D Up requires exactly two routes, one matmul task, and one return");
            signalPassFailure();
            return;
        }
        stream::MatmulTaskOp op = matmuls.front();
        const auto role = op.getConfig()
            .getAs<mlir::StringAttr>("projection_kind");
        if (!role || role.getValue() != "up") {
            op.emitError(
                "direct FU 3-D mode requires config.projection_kind = \"up\"");
            signalPassFailure();
            return;
        }
        if (op.getLhs().size() != 1 || op.getRhs().size() != 1) {
            op.emitError(
                "standalone direct FU 3-D Up requires one activation route and one weight route");
            signalPassFailure();
            return;
        }
        auto activationRoute =
            op.getLhs().front().getDefiningOp<stream::RouteOp>();
        auto weightRoute =
            op.getRhs().front().getDefiningOp<stream::RouteOp>();
        if (!activationRoute || !weightRoute
            || activationRoute == weightRoute
            || !((routes[0] == activationRoute && routes[1] == weightRoute)
                || (routes[1] == activationRoute
                    && routes[0] == weightRoute))) {
            op.emitError(
                "standalone direct FU 3-D Up requires exactly its two operand routes");
            signalPassFailure();
            return;
        }
        mlir::func::ReturnOp returnOp = returns.front();
        if (returnOp.getNumOperands() != 1
            || returnOp.getOperand(0) != op.getResult()
            || !op.getResult().hasOneUse()) {
            returnOp.emitError(
                "standalone direct FU 3-D Up must return only the matmul result");
            signalPassFailure();
            return;
        }
        const auto activationArgument = llvm::dyn_cast<mlir::BlockArgument>(
            activationRoute.getInput());
        const auto weightArgument = llvm::dyn_cast<mlir::BlockArgument>(
            weightRoute.getInput());
        if (!activationArgument || !weightArgument
            || activationArgument.getOwner() != &block
            || weightArgument.getOwner() != &block
            || activationArgument.getArgNumber() != 0
            || weightArgument.getArgNumber() != 1) {
            op.emitError(
                "standalone direct FU 3-D Up routes must read input 0 activation and input 1 weight");
            signalPassFailure();
            return;
        }

        auto target = target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target)) {
            signalPassFailure();
            return;
        }
        schedule::FfnUp3DShape shape {};
        schedule::FfnUp3DPlacement placement;
        schedule::FfnUp3DTimeline timeline;
        schedule::FfnUp3DRouteLatencies routeLatencies;
        std::uint16_t scaleBits = 0;
        if (mlir::failed(buildLoweringInput(op, *target, shape,
                placement, timeline, routeLatencies, scaleBits))) {
            signalPassFailure();
            return;
        }

        std::string error;
        auto lowered = schedule::lowerFfnUpToFu3D(shape, placement,
            timeline, routeLatencies, scaleBits, &error);
        if (mlir::failed(lowered)) {
            op.emitError("direct FU 3-D Up lowering failed: ") << error;
            signalPassFailure();
            return;
        }

        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPoint(op);
        if (mlir::failed(command::materializeFfnUp3DCommands(
                builder, op.getLoc(), *lowered, &error))) {
            op.emitError("cannot materialize direct FU 3-D commands: ")
                << error;
            signalPassFailure();
            return;
        }

        returnOp->setOperands(mlir::ValueRange {});
        op.erase();
        activationRoute.erase();
        weightRoute.erase();
        block.eraseArguments(0, block.getNumArguments());
        function.setType(mlir::FunctionType::get(
            &getContext(), mlir::TypeRange {}, mlir::TypeRange {}));
    }
};

} // namespace

std::unique_ptr<mlir::Pass>
create_lower_standalone_ffn_up_to_3d_command_pass()
{
    return std::make_unique<LowerStandaloneFfnUpTo3DCommandPass>();
}

} // namespace ftlpu::compiler
