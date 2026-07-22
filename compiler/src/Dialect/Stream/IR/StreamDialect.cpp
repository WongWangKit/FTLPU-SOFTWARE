#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

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

    const target::LPUTargetModel target;
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
    if (stream_base < 0 || stream_count != *expected_stream_count
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
    if (!input.getElementType().isInteger(8) || !result.getElementType().isF16()
        || input.getShape() != result.getShape())
        return emitOpError("requires shape-preserving i8 to f16 conversion");
    if (getInputStreamBase() < 0 || getInputStreamBase() + 8 > 32
        || getOutputStreamBase() < 0 || getOutputStreamBase() + 16 > 32)
        return emitOpError("requires an 8-stream input and 16-stream output range");
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
    const target::LPUTargetModel target;
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

LogicalResult SwigluOp::verify()
{
    const target::LPUTargetModel target;
    if (!target.is_valid_mxm_unit(getGateUnitId())
        || !target.is_valid_mxm_unit(getUpUnitId())
        || getGateUnitId() == getUpUnitId()
        || !target.is_valid_weight_buffer(getGateWeightBuffer())
        || !target.is_valid_weight_buffer(getUpWeightBuffer()))
        return emitOpError("requires distinct valid MXM units and valid weight buffers");
    if (getGateOutputStreamBase() < 0 || getGateOutputStreamBase() + 4 > 32
        || getUpOutputStreamBase() < 0 || getUpOutputStreamBase() + 4 > 32
        || getVxmOutputStream() < 0 || getVxmOutputStream() >= 32)
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

LogicalResult FfnOp::verify()
{
    const bool legacy = getK() == 320 && getHidden() == 640 && getN() == 320;
    const target::LPUTargetModel target;
    const bool w8a16 = getActivation().getType().getElementType().isF16()
        && getGateWeight().getType().getElementType().isF16()
        && getUpWeight().getType().getElementType().isF16()
        && getDownWeight0().getType().getElementType().isF16()
        && getResult().getType().getElementType().isF16()
        && target.supports_w8a16_ffn_shape(getM(), getK(), getHidden(), getN());
    if (!legacy && !w8a16)
        return emitOpError("requires a supported legacy or tile-aligned W8A16 FFN");
    auto activation = getActivation().getDefiningOp<RouteOp>();
    auto gate = getGateWeight().getDefiningOp<RouteOp>();
    auto up = getUpWeight().getDefiningOp<RouteOp>();
    auto down0 = getDownWeight0().getDefiningOp<RouteOp>();
    auto down1 = getDownWeight1().getDefiningOp<RouteOp>();
    if (!activation || !gate || !up || !down0 || !down1
        || activation.getDestination() != "MXM.activation"
        || gate.getDestinationUnitId() != 0
        || up.getDestinationUnitId() != 1
        || down0.getDestinationUnitId() != 0
        || down1.getDestinationUnitId() != 1
        || gate.getDestination() != "MXM.weight"
        || up.getDestination() != "MXM.weight"
        || down0.getDestination() != "MXM.weight"
        || down1.getDestination() != "MXM.weight"
        || (w8a16 && (gate.getSource() != "VXM.result" || up.getSource() != "VXM.result"
            || down0.getSource() != "VXM.result" || down1.getSource() != "VXM.result")))
        return emitOpError("requires canonical activation and three-stage dual-MXM routes");
    if (getGateOutputStreamBase() != 0 || getUpOutputStreamBase() != 4
        || (legacy && getVxmOutputStream() != 31)
        || (w8a16 && getVxmOutputStream() != 0))
        return emitOpError("contains non-canonical FFN stream bases for the selected target profile");
    return success();
}

LogicalResult AttentionOp::verify()
{
    if (getQueryHeads() <= 0 || getKvHeads() <= 0 || getHeadDim() <= 0
        || getQueryHeads() % getKvHeads() != 0 || getRoutes().empty())
        return emitOpError("requires grouped-query dimensions and at least one stream route");
    for (Attribute attribute : getRoutes()) {
        const auto route = llvm::dyn_cast<DictionaryAttr>(attribute);
        if (!route) return emitOpError("routes must be dictionaries");
        for (StringRef field : {"phase", "role", "source", "destination", "direction",
                 "stream_base", "stream_count", "register_id", "producer_stage",
                 "consumer_stage", "transport_latency"})
            if (!route.get(field)) return emitOpError() << "route is missing '" << field << "'";
        const auto base = route.getAs<IntegerAttr>("stream_base");
        const auto count = route.getAs<IntegerAttr>("stream_count");
        const auto begin = route.getAs<IntegerAttr>("producer_stage");
        const auto end = route.getAs<IntegerAttr>("consumer_stage");
        if (!base || !count || !begin || !end || base.getInt() < 0 || count.getInt() <= 0
            || base.getInt() + count.getInt() > 32 || end.getInt() <= begin.getInt())
            return emitOpError("route has an invalid stream range or lifetime window");
    }
    return success();
}

void StreamDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "ftlpu/compiler/Dialect/Stream/IR/StreamOps.cpp.inc"
        >();
}

} // namespace ftlpu::compiler::stream
