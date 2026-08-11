#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>

using namespace mlir;

#include "ftlpu/compiler/Dialect/Stream/IR/StreamOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ftlpu/compiler/Dialect/Stream/IR/StreamOps.cpp.inc"

namespace ftlpu::compiler::stream {
namespace {

std::optional<target::StreamEndpoint> parse_endpoint(StringRef name)
{
    if (name == "MEM") return target::StreamEndpoint::Mem;
    if (name == "MXM.activation") return target::StreamEndpoint::MxmActivation;
    if (name == "MXM.weight") return target::StreamEndpoint::MxmWeight;
    if (name == "MXM.result") return target::StreamEndpoint::MxmResult;
    if (name == "VXM.input") return target::StreamEndpoint::VxmInput;
    if (name == "VXM.result") return target::StreamEndpoint::VxmResult;
    return std::nullopt;
}

std::optional<target::StreamDirection> parse_direction(StringRef name)
{
    if (name == "east") return target::StreamDirection::East;
    if (name == "west") return target::StreamDirection::West;
    return std::nullopt;
}

LogicalResult verify_attention_config(
    Operation* operation, DictionaryAttr config)
{
    const auto seqLen = config.getAs<IntegerAttr>("seq_len");
    const auto hidden = config.getAs<IntegerAttr>("hidden");
    const auto queryHeads = config.getAs<IntegerAttr>("query_heads");
    const auto kvHeads = config.getAs<IntegerAttr>("kv_heads");
    const auto headDim = config.getAs<IntegerAttr>("head_dim");
    const auto ropeTheta = config.getAs<FloatAttr>("rope_theta");
    const auto causal = config.getAs<BoolAttr>("causal");
    if (!seqLen || !hidden || !queryHeads || !kvHeads || !headDim
        || !ropeTheta || !causal || seqLen.getInt() <= 0
        || hidden.getInt() <= 0 || queryHeads.getInt() <= 0
        || kvHeads.getInt() <= 0 || headDim.getInt() <= 0
        || queryHeads.getInt() % kvHeads.getInt() != 0)
        return operation->emitOpError(
            "requires a complete grouped-query attention config");
    return success();
}

LogicalResult verify_attention_routes(
    Operation* operation, ArrayAttr routes)
{
    if (routes.empty())
        return operation->emitOpError("requires allocated stream routes");
    auto targetModel = target::LPUTargetModel::from_operation(operation);
    if (failed(targetModel)) return failure();
    const int64_t streams = targetModel->streams().streams_per_direction;
    for (Attribute attribute : routes) {
        const auto route = llvm::dyn_cast<DictionaryAttr>(attribute);
        if (!route)
            return operation->emitOpError("routes must be dictionaries");
        for (StringRef field : {"phase", "role", "source", "destination",
                 "direction", "stream_base", "stream_count", "register_id",
                 "producer_stage", "consumer_stage", "transport_latency"})
            if (!route.get(field))
                return operation->emitOpError()
                    << "route is missing '" << field << "'";
        const auto base = route.getAs<IntegerAttr>("stream_base");
        const auto count = route.getAs<IntegerAttr>("stream_count");
        const auto begin = route.getAs<IntegerAttr>("producer_stage");
        const auto end = route.getAs<IntegerAttr>("consumer_stage");
        if (!base || !count || !begin || !end || base.getInt() < 0
            || count.getInt() <= 0 || base.getInt() + count.getInt() > streams
            || end.getInt() <= begin.getInt())
            return operation->emitOpError(
                "route has an invalid stream range or lifetime window");
    }
    return success();
}

LogicalResult verify_optional_attention_routes(
    Operation* operation, ArrayAttr routes)
{
    if (routes.empty()) return success();
    return verify_attention_routes(operation, routes);
}

} // namespace

ParseResult RouteOp::parse(OpAsmParser& parser, OperationState& result)
{
    OpAsmParser::UnresolvedOperand operand;
    const SMLoc operand_location = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperand(operand) || parser.parseOptionalAttrDict(result.attributes)
        || parser.parseColonType(type) || parser.resolveOperand(operand, type, result.operands))
        return failure();
    result.addTypes(type);
    return success();
}

void RouteOp::print(OpAsmPrinter& printer)
{
    printer << " " << getInput() << " {";
    printer.increaseIndent();
    const auto print_attribute = [&](StringRef name, Attribute attribute, bool comma) {
        printer.printNewline();
        printer << name << " = ";
        printer.printAttribute(attribute);
        if (comma) printer << ',';
    };
    print_attribute("stream_base", getStreamBaseAttr(), true);
    print_attribute("stream_count", getStreamCountAttr(), true);
    print_attribute("register_id", getRegisterIdAttr(), true);
    print_attribute("direction", getDirectionAttr(), true);
    print_attribute("source", getSourceAttr(), true);
    print_attribute("destination", getDestinationAttr(), true);
    print_attribute("source_unit_id", getSourceUnitIdAttr(), true);
    print_attribute("destination_unit_id", getDestinationUnitIdAttr(), true);
    print_attribute("address", getAddressAttr(), true);
    print_attribute("placement", getPlacementAttr(), true);
    print_attribute("bytes", getBytesAttr(), true);
    print_attribute("transport_latency", getTransportLatencyAttr(), false);
    printer.decreaseIndent();
    printer.printNewline();
    printer << "} : " << getInput().getType();
}

LogicalResult RouteOp::verify()
{
    if (getInput().getType() != getOutput().getType())
        return emitOpError("input and output tensor types must match");
    const int64_t bytes = getBytesAttr().getInt();
    const int64_t latency = getTransportLatencyAttr().getInt();
    if (bytes <= 0 || latency <= 0)
        return emitOpError("requires positive size and latency");

    const auto source = parse_endpoint(getSource());
    const auto destination = parse_endpoint(getDestination());
    const auto direction = parse_direction(getDirection());
    const auto slice = getAddress().getAs<IntegerAttr>("slice");
    if (!source || !destination || !direction || !slice)
        return emitOpError("has an invalid endpoint, direction, or MEM address");

    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const target::LPUTargetModel& target = *target_model;
    const auto valid_unit = [&](target::StreamEndpoint endpoint, int64_t unit_id) {
        if (endpoint == target::StreamEndpoint::Mem) return unit_id == -1;
        if (endpoint == target::StreamEndpoint::VxmResult
            || endpoint == target::StreamEndpoint::VxmInput) return unit_id == 0;
        return target.is_valid_mxm_unit(unit_id);
    };
    if (!valid_unit(*source, getSourceUnitIdAttr().getInt())
        || !valid_unit(*destination, getDestinationUnitIdAttr().getInt()))
        return emitOpError("endpoint unit id does not match the LPU target");
    const auto expected_register = target.stream_register_id(
        *source, *destination, *direction, slice.getInt());
    const auto expected_latency = target.transport_latency(
        *source, *destination, *direction, slice.getInt());
    const auto expected_stream_count = target.route_stream_count(
        *source, *destination, *direction);
    if (!expected_register || !expected_latency || !expected_stream_count)
        return emitOpError("route is not supported by the LPU topology");
    const int64_t stream_base = getStreamBaseAttr().getInt();
    const int64_t stream_count = getStreamCountAttr().getInt();
    const auto placementKind =
        getPlacement().getAs<StringAttr>("kind");
    const bool distributedBlock8Activation =
        *source == target::StreamEndpoint::Mem
        && *destination == target::StreamEndpoint::MxmActivation
        && placementKind
        && placementKind.getValue() == "fp16_mxm_distributed_16"
        && stream_count
            == 2 * target.throughput().mxm_block_rows;
    const bool block8Weight =
        *source == target::StreamEndpoint::Mem
        && *destination == target::StreamEndpoint::MxmWeight
        && placementKind
        && placementKind.getValue()
            == "w8a16_block8_weight_wave_striped"
        && stream_count
            == target.throughput().mxm_int8_load_streams_per_cycle;
    const bool localInt8Weight =
        *source == target::StreamEndpoint::Mem
        && *destination == target::StreamEndpoint::MxmWeight
        && getInput().getType().getElementType().isInteger(8)
        && target.supports_mxm_local_dequant()
        && stream_count
            == target.throughput().mxm_int8_load_streams_per_cycle;
    const bool validStreamCount =
        stream_count == *expected_stream_count
        || distributedBlock8Activation || block8Weight
        || localInt8Weight;
    if (stream_base < 0 || !validStreamCount
        || stream_base + stream_count > target.streams().streams_per_direction
        || getRegisterId() != static_cast<uint64_t>(*expected_register)
        || latency != *expected_latency)
        return emitOpError("stream binding does not match the LPU target model")
            << " (base=" << stream_base << ", count=" << stream_count
            << ", expected_count=" << *expected_stream_count
            << ", register=" << getRegisterId() << ", expected_register="
            << *expected_register << ")";
    return success();
}

LogicalResult DequantizeOp::verify()
{
    auto input = getInput().getType();
    auto result = getResult().getType();
    if (!input.getElementType().isInteger(8)
        || !is_lpu_16bit_float(result.getElementType())
        || input.getShape() != result.getShape())
        return emitOpError(
            "requires shape-preserving i8 to 16-bit float conversion");
    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const auto& streams = target_model->streams();
    const auto& throughput = target_model->throughput();
    if (getInputStreamBase() < 0
        || getInputStreamBase() + throughput.lanes_per_tile
            > streams.streams_per_direction
        || getOutputStreamBase() < 0
        || getOutputStreamBase() + throughput.mxm_load_streams_per_cycle
            > streams.streams_per_direction)
        return emitOpError(
            "dequant stream ranges do not fit the configured target");
    if ((getInputHemisphere() != "east" && getInputHemisphere() != "west")
        || (getOutputHemisphere() != "east" && getOutputHemisphere() != "west"))
        return emitOpError("hemisphere must be east or west");
    return success();
}

LogicalResult MatmulOp::verify()
{
    const auto lhs = getLhs().getType();
    const auto rhs = getRhs().getType();
    const auto result = getResult().getType();
    if (lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2)
        return emitOpError("requires rank-2 tensors");
    if (lhs.getDimSize(0) != getM() || lhs.getDimSize(1) != getK()
        || rhs.getDimSize(0) != getK() || rhs.getDimSize(1) != getN()
        || result.getDimSize(0) != getM() || result.getDimSize(1) != getN())
        return emitOpError("tensor shapes do not match m, n, and k");
    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const target::LPUTargetModel& target = *target_model;
    if (!target.is_valid_mxm_unit(getUnitId()))
        return emitOpError("unit_id must select MXM0 or MXM1");
    if (!target.is_valid_weight_buffer(getWeightBuffer()))
        return emitOpError("weight_buffer must select buffer 0 or 1");
    auto lhs_route = getLhs().getDefiningOp<RouteOp>();
    auto rhs_route = getRhs().getDefiningOp<RouteOp>();
    if (!lhs_route || !rhs_route
        || lhs_route.getDestinationUnitId() != getUnitId()
        || rhs_route.getDestinationUnitId() != getUnitId())
        return emitOpError("operand routes must terminate at the selected MXM unit");
    return success();
}

static LogicalResult verify_task_allocations(
    Operation* op, ArrayAttr allocations)
{
    for (auto [index, attribute] : llvm::enumerate(allocations)) {
        const auto allocation = llvm::dyn_cast<DictionaryAttr>(attribute);
        if (!allocation || !allocation.getAs<DictionaryAttr>("address")
            || !allocation.getAs<DictionaryAttr>("placement")
            || !allocation.getAs<IntegerAttr>("bytes"))
            return op->emitOpError()
                << "result_allocations[" << index
                << "] requires address, placement, and bytes";
    }
    return success();
}

LogicalResult MatmulTaskOp::verify()
{
    const size_t parallelism = std::max(getLhs().size(), getRhs().size());
    if (parallelism == 0 || getLhs().empty() || getRhs().empty()
        || (getLhs().size() != 1 && getLhs().size() != parallelism)
        || (getRhs().size() != 1 && getRhs().size() != parallelism)
        || getUnitIds().size() != parallelism
        || getWeightBuffers().size() != parallelism
        || getResultStreamBases().size() != parallelism
        || getResultStreamCounts().size() != parallelism)
        return emitOpError("parallel operand and stream arrays have inconsistent sizes");

    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const target::LPUTargetModel& target = *target_model;
    for (size_t index = 0; index < parallelism; ++index) {
        const auto unit = llvm::dyn_cast<IntegerAttr>(getUnitIds()[index]);
        const auto buffer =
            llvm::dyn_cast<IntegerAttr>(getWeightBuffers()[index]);
        const auto stream =
            llvm::dyn_cast<IntegerAttr>(getResultStreamBases()[index]);
        const auto count =
            llvm::dyn_cast<IntegerAttr>(getResultStreamCounts()[index]);
        if (!unit || !buffer || !stream || !count
            || !target.is_valid_mxm_unit(unit.getInt())
            || !target.is_valid_weight_buffer(buffer.getInt())
            || stream.getInt() < 0 || count.getInt() <= 0
            || stream.getInt() + count.getInt()
                > target.streams().streams_per_direction)
            return emitOpError("contains an invalid MXM or result-stream binding");

        const auto lhs = getLhs()[std::min(index, getLhs().size() - 1)];
        const auto rhs = getRhs()[std::min(index, getRhs().size() - 1)];
        const auto lhs_type = llvm::cast<RankedTensorType>(lhs.getType());
        const auto rhs_type = llvm::cast<RankedTensorType>(rhs.getType());
        if (lhs_type.getRank() != 2 || rhs_type.getRank() != 2
            || lhs_type.getDimSize(0) != getM()
            || lhs_type.getDimSize(1) != getK()
            || rhs_type.getDimSize(0) != getK()
            || rhs_type.getDimSize(1) != getN())
            return emitOpError("operand shapes do not match m, n, and k");
        auto lhs_route = lhs.getDefiningOp<RouteOp>();
        auto rhs_route = rhs.getDefiningOp<RouteOp>();
        if (!lhs_route || !rhs_route
            || lhs_route.getDestination() != "MXM.activation"
            || rhs_route.getDestination() != "MXM.weight"
            || rhs_route.getDestinationUnitId() != unit.getInt())
            return emitOpError("operands require canonical routes to the selected MXM");
    }
    const auto result_type = getResult().getType();
    if (result_type.getRank() != 2
        || result_type.getDimSize(0) != getM()
        || result_type.getDimSize(1) != getN())
        return emitOpError("result shape does not match m and n");
    return verify_task_allocations(getOperation(), getResultAllocations());
}

LogicalResult SwishTaskOp::verify()
{
    if (getInput().getType().getShape() != getResult().getType().getShape())
        return emitOpError("input and result shapes must match");
    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const int64_t streams =
        target_model->streams().streams_per_direction;
    if (getInputStreamBase() < 0 || getInputStreamBase() >= streams
        || getOutputStreamBase() < 0 || getOutputStreamBase() >= streams)
        return emitOpError("contains an invalid direction-local stream");
    return success();
}

LogicalResult ElementwiseTaskOp::verify()
{
    if (getKind() != "multiply" && getKind() != "add"
        && getKind() != "add_quant")
        return emitOpError("supports multiply, add, and add_quant");
    if (getLhs().getType().getShape() != getRhs().getType().getShape()
        || getLhs().getType().getShape() != getResult().getType().getShape())
        return emitOpError("operand and result shapes must match");
    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const int64_t streams =
        target_model->streams().streams_per_direction;
    if (getInputStreamBases().size() != 2
        || getOutputStreamBase() < 0 || getOutputStreamBase() >= streams)
        return emitOpError("requires two input streams and one valid output stream");
    if (!getLhsAllocations().empty()
        && failed(verify_task_allocations(
            getOperation(), getLhsAllocations())))
        return failure();
    if (!getRhsAllocations().empty()
        && failed(verify_task_allocations(
            getOperation(), getRhsAllocations())))
        return failure();
    return verify_task_allocations(getOperation(), getResultAllocations());
}

LogicalResult RmsNormTaskOp::verify()
{
    const auto input = getInput().getType();
    const auto weight = getWeight().getType();
    const auto result = getResult().getType();
    const int64_t rawAxis = getAxisAttr().getInt();
    const int64_t axis =
        rawAxis < 0 ? rawAxis + input.getRank() : rawAxis;
    if (!input.hasStaticShape() || !weight.hasStaticShape()
        || input.getRank() != 2 || weight.getRank() != 1
        || result != input || axis != input.getRank() - 1
        || weight.getDimSize(0) != input.getDimSize(axis)
        || weight.getElementType() != input.getElementType())
        return emitOpError("has incompatible RMSNorm tensor types");
    if (!std::isfinite(getEpsilon().convertToFloat())
        || getEpsilon().convertToFloat() <= 0.0f)
        return emitOpError("requires a finite positive epsilon");
    const auto strategy = getConfig().getAs<mlir::StringAttr>("strategy");
    if (!strategy
        || (strategy.getValue() != "vxm_square_mxm_reduce"
            && strategy.getValue() != "vxm_feedback"))
        return emitOpError("requires a supported RMSNorm strategy");
    const std::size_t expectedScratch =
        strategy.getValue() == "vxm_feedback" ? 3 : 2;
    if (getScratchAllocations().size() != expectedScratch)
        return emitOpError("scratch allocation count does not match strategy");
    if (getRoutes().size() < 2)
        return emitOpError("requires input and scale stream routes");
    if (failed(verify_task_allocations(
            getOperation(), getInputAllocations()))
        || failed(verify_task_allocations(
            getOperation(), getWeightAllocations()))
        || failed(verify_task_allocations(
            getOperation(), getScratchAllocations()))
        || failed(verify_task_allocations(
            getOperation(), getResultAllocations())))
        return failure();
    return success();
}

LogicalResult SwigluOp::verify()
{
    auto target_model = target::LPUTargetModel::from_operation(getOperation());
    if (failed(target_model)) return failure();
    const target::LPUTargetModel& target = *target_model;
    if (!target.is_valid_mxm_unit(getGateUnitId())
        || !target.is_valid_mxm_unit(getUpUnitId())
        || getGateUnitId() == getUpUnitId()
        || !target.is_valid_weight_buffer(getGateWeightBuffer())
        || !target.is_valid_weight_buffer(getUpWeightBuffer()))
        return emitOpError("requires distinct valid MXM units and valid weight buffers");
    const int64_t result_streams =
        target.throughput().mxm_result_streams;
    const int64_t streams = target.streams().streams_per_direction;
    if (getGateOutputStreamBase() < 0
        || getGateOutputStreamBase() + result_streams > streams
        || getUpOutputStreamBase() < 0
        || getUpOutputStreamBase() + result_streams > streams
        || getVxmOutputStream() < 0 || getVxmOutputStream() >= streams)
        return emitOpError("contains an invalid direction-local stream range");
    auto activation = getActivation().getDefiningOp<RouteOp>();
    auto gate = getGateWeight().getDefiningOp<RouteOp>();
    auto up = getUpWeight().getDefiningOp<RouteOp>();
    if (!activation || !gate || !up || activation.getDestination() != "MXM.activation"
        || gate.getDestination() != "MXM.weight" || up.getDestination() != "MXM.weight"
        || gate.getDestinationUnitId() != getGateUnitId()
        || up.getDestinationUnitId() != getUpUnitId())
        return emitOpError("requires canonical activation and dual weight routes");
    return success();
}

LogicalResult ProjectionTaskOp::verify()
{
    if (getKind() == "linear") {
        const auto m = getConfig().getAs<IntegerAttr>("m");
        const auto n = getConfig().getAs<IntegerAttr>("n");
        const auto k = getConfig().getAs<IntegerAttr>("k");
        const auto rhsScale =
            getConfig().getAs<FloatAttr>("rhs_scale");
        const auto input = getInput().getType();
        const auto weight = getWeight().getType();
        const auto result = getResult().getType();
        if (!m || !n || !k || !rhsScale || !getMemoryPlan()
            || m.getInt() <= 0 || n.getInt() <= 0 || k.getInt() <= 0
            || !std::isfinite(rhsScale.getValueAsDouble())
            || rhsScale.getValueAsDouble() <= 0.0
            || input.getRank() != 2 || weight.getRank() != 2
            || result.getRank() != 2
            || input.getShape()
                != llvm::ArrayRef<int64_t>({m.getInt(), k.getInt()})
            || weight.getShape()
                != llvm::ArrayRef<int64_t>({k.getInt(), n.getInt()})
            || result.getShape()
                != llvm::ArrayRef<int64_t>({m.getInt(), n.getInt()})
            || !is_lpu_16bit_float(input.getElementType())
            || !weight.getElementType().isInteger(8)
            || result.getElementType() != input.getElementType())
            return emitOpError(
                "requires a valid rank-2 W8A16 linear projection config");
        return success();
    }
    if (getKind() != "query" && getKind() != "key"
        && getKind() != "value" && getKind() != "output")
        return emitOpError("kind must be query, key, value, or output");
    if (failed(verify_attention_config(getOperation(), getConfig())))
        return failure();
    if (getKind() == "output") {
        const auto memoryPlan = getMemoryPlan();
        if (!memoryPlan || memoryPlan->empty())
            return emitOpError("requires a physical attention memory plan");
    }
    return verify_attention_routes(getOperation(), getRoutes());
}

LogicalResult RopeTaskOp::verify()
{
    if (getKind() != "query" && getKind() != "key")
        return emitOpError("kind must be query or key");
    if (getInput().getType() != getResult().getType())
        return emitOpError("must preserve its tensor type");
    if (failed(verify_attention_config(getOperation(), getConfig())))
        return failure();
    return verify_optional_attention_routes(getOperation(), getRoutes());
}

LogicalResult BatchMatmulTaskOp::verify()
{
    if (getKind() != "qk" && getKind() != "pv")
        return emitOpError("kind must be qk or pv");
    if (failed(verify_attention_config(getOperation(), getConfig())))
        return failure();
    return verify_optional_attention_routes(getOperation(), getRoutes());
}

LogicalResult SoftmaxTaskOp::verify()
{
    if (getInput().getType() != getResult().getType())
        return emitOpError("must preserve its tensor type");
    if (failed(verify_attention_config(getOperation(), getConfig())))
        return failure();
    return verify_optional_attention_routes(getOperation(), getRoutes());
}

LogicalResult TransposeTaskOp::verify()
{
    if (getKind() != "probability" && getKind() != "value")
        return emitOpError("kind must be probability or value");
    if (failed(verify_attention_config(getOperation(), getConfig())))
        return failure();
    return verify_optional_attention_routes(getOperation(), getRoutes());
}

void StreamDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "ftlpu/compiler/Dialect/Stream/IR/StreamOps.cpp.inc"
        >();
}

} // namespace ftlpu::compiler::stream
