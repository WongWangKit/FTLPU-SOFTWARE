// Keep generated property accessors rebuilt after Command ODS changes.
#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include <cmath>

using namespace mlir;

#include "ftlpu/compiler/Dialect/Command/IR/CommandOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ftlpu/compiler/Dialect/Command/IR/CommandOps.cpp.inc"

namespace ftlpu::compiler::command {

LogicalResult BindingOp::verify()
{
    if (getIndex() < 0 || getBytes() <= 0)
        return emitOpError("requires a non-negative index and positive byte size");
    if (getAccess() != "input" && getAccess() != "output"
        && getAccess() != "internal")
        return emitOpError("access must be input, output, or internal");
    if (getRole() != "activation" && getRole() != "weight"
        && getRole() != "result" && getRole() != "constant")
        return emitOpError("role must be activation, weight, result, or constant");
    if (getElementType() != "i8" && getElementType() != "i32"
        && getElementType() != "f16" && getElementType() != "f32")
        return emitOpError("element_type must be i8, i32, f16, or f32");
    if (getShape().empty()) return emitOpError("requires a ranked shape");
    for (Attribute dimension : getShape()) {
        auto integer = llvm::dyn_cast<IntegerAttr>(dimension);
        if (!integer || integer.getInt() <= 0)
            return emitOpError("shape dimensions must be positive integers");
    }
    if (!getPlacement().getAs<StringAttr>("kind")
        || !getPlacement().getAs<ArrayAttr>("slices")
        || !getPlacement().getAs<IntegerAttr>("base_row")
        || !getPlacement().getAs<IntegerAttr>("instruction_count")
        || !getPlacement().getAs<IntegerAttr>("address_stride"))
        return emitOpError("placement is missing physical layout fields");
    return success();
}

LogicalResult MemOp::verify()
{
    const target::LPUTargetModel target;
    if (getCycle() < 0 || getQueue() < 0
        || getQueue() >= target.memory().hemispheres * target.memory().slices_per_hemisphere
        || getAddress() < 0 || getAddress() >= 8192
        || getPackedStream() < 0 || getPackedStream() >= target.streams().encoded_streams
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0)
        return emitOpError("contains an invalid ICU MEM queue command field");
    if (getOpcode() != "read" && getOpcode() != "write" && getOpcode() != "accumulate")
        return emitOpError("opcode must be read, write, or accumulate");
    if (getAccumulatorDestination() != "sram" && getAccumulatorDestination() != "stream")
        return emitOpError("accumulator_destination must be sram or stream");
    const int64_t final_address = getAddress()
        + (getRepeatCount() - 1) * getAddressStride();
    if (final_address < 0 || final_address >= 8192)
        return emitOpError("repeated address range is outside SRAM");
    return success();
}

LogicalResult MxmOp::verify()
{
    const target::LPUTargetModel target;
    if (getCycle() < 0 || !target.is_valid_mxm_unit(getQueue())
        || !target.is_valid_weight_buffer(getWeightBuffer())
        || getWeightColumn() < 0 || getWeightColumn() >= target.throughput().tile_rows
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0)
        return emitOpError("contains an invalid ICU MXM queue command field");
    if (getOpcode() != "iw" && getOpcode() != "compute")
        return emitOpError("opcode must be iw or compute");
    if (getActivationStreamBase() < 0
        || getActivationStreamBase() >= target.streams().encoded_streams
        || getOutputStreamBase() < 0
        || getOutputStreamBase() + target.throughput().mxm_result_streams - 1
            >= target.streams().encoded_streams)
        return emitOpError("contains an invalid MXM stream selector");
    return success();
}

LogicalResult VxmOp::verify()
{
    const target::LPUTargetModel target;
    const int64_t cycle = getCycleAttr().getInt();
    const int64_t queue = getQueueAttr().getInt();
    const int64_t lhs_index = getLhsIndexAttr().getInt();
    const int64_t rhs_index = getRhsIndexAttr().getInt();
    const int64_t output_stream = getOutputStreamAttr().getInt();
    const int64_t repeat_count = getRepeatCountAttr().getInt();
    const int64_t repeat_interval = getRepeatIntervalAttr().getInt();
    if (cycle < 0 || !target.is_valid_vxm_alu(queue)
        || repeat_count <= 0 || repeat_interval <= 0)
        return emitOpError("contains an invalid ICU VXM queue command field");

    const auto opcode = getOpcode();
    if (opcode != "pass" && opcode != "add" && opcode != "subtract"
        && opcode != "multiply" && opcode != "divide" && opcode != "negate"
        && opcode != "abs" && opcode != "min" && opcode != "max"
        && opcode != "clamp" && opcode != "square" && opcode != "sqrt"
        && opcode != "exp" && opcode != "log" && opcode != "relu"
        && opcode != "cast")
        return emitOpError("contains an unsupported VXM opcode");

    const auto verify_operand = [&](StringRef kind, int64_t index) {
        if (kind == "immediate") return index == 0;
        if (kind == "alu") return target.is_valid_vxm_alu(index);
        if (kind == "stream_i32" || kind == "stream_f32")
            return index >= 0 && index + 3 < target.streams().encoded_streams;
        if (kind == "stream_i8") return index >= 0 && index < target.streams().encoded_streams;
        if (kind == "stream_f16") return index >= 0 && index + 1 < target.streams().encoded_streams;
        return false;
    };
    if (!verify_operand(getLhsKind(), lhs_index)
        || !verify_operand(getRhsKind(), rhs_index))
        return emitOpError("contains an invalid VXM operand kind or index");
    if (!std::isfinite(getLhsImmediateAttr().getValueAsDouble())
        || !std::isfinite(getRhsImmediateAttr().getValueAsDouble()))
        return emitOpError("VXM immediate operands must be finite");
    if (getCastTarget() != "fp32" && getCastTarget() != "fp16"
        && getCastTarget() != "i8")
        return emitOpError("cast_target must be fp32, fp16, or i8");
    if (output_stream < -1 || output_stream >= target.streams().encoded_streams)
        return emitOpError("output_stream must be -1 or a packed stream selector");
    if ((getInputHemisphere() != "east" && getInputHemisphere() != "west")
        || (getOutputHemisphere() != "east" && getOutputHemisphere() != "west"))
        return emitOpError("hemisphere must be east or west");
    return success();
}

LogicalResult SxmOp::verify()
{
    const target::LPUTargetModel target;
    if (getCycle() < 0 || getHemisphere() < 0
        || getHemisphere() >= target.memory().hemispheres)
        return emitOpError("contains an invalid ICU SXM queue selector");
    if (getOpcode() != "transpose" && getOpcode() != "permute")
        return emitOpError("opcode must be transpose or permute");
    if (getSourceStreams().empty() || getDestinationStreams().empty()
        || getPermuteMap().size() != 32)
        return emitOpError("requires non-empty stream lists and a 32-lane map");
    const auto valid_stream = [&](Attribute attribute) {
        const auto value = llvm::dyn_cast<IntegerAttr>(attribute);
        return value && value.getInt() >= 0
            && value.getInt() < target.streams().encoded_streams;
    };
    for (Attribute stream : getSourceStreams())
        if (!valid_stream(stream)) return emitOpError("source stream is outside the encoded range");
    for (Attribute stream : getDestinationStreams())
        if (!valid_stream(stream)) return emitOpError("destination stream is outside the encoded range");
    for (Attribute lane : getPermuteMap()) {
        const auto value = llvm::dyn_cast<IntegerAttr>(lane);
        if (!value || value.getInt() < -1 || value.getInt() >= 32)
            return emitOpError("permute map lanes must be -1 or [0, 31]");
    }
    if (getWeightLayout() != "vector_columns" && getWeightLayout() != "matrix_columns")
        return emitOpError("weight_layout must be vector_columns or matrix_columns");
    return success();
}

void CommandDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "ftlpu/compiler/Dialect/Command/IR/CommandOps.cpp.inc"
        >();
}

} // namespace ftlpu::compiler::command
