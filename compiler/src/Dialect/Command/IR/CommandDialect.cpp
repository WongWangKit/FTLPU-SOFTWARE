// Keep generated wave/repeat accessors rebuilt after Command ODS changes.
#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include <cmath>

using namespace mlir;

// Keep generated binding metadata accessors synchronized with CommandOps.td.
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
        && getRole() != "result" && getRole() != "constant"
        && getRole() != "workspace")
        return emitOpError(
            "role must be activation, weight, result, constant, or workspace");
    if (getElementType() != "i8" && getElementType() != "i32"
        && getElementType() != "f16" && getElementType() != "bf16"
        && getElementType() != "f32")
        return emitOpError(
            "element_type must be i8, i32, f16, bf16, or f32");
    if (getShape().empty()) return emitOpError("requires a ranked shape");
    if (getInitializer() != "none" && getInitializer() != "zero"
        && getInitializer() != "causal_mask"
        && getInitializer() != "rope_table")
        return emitOpError(
            "initializer must be none, zero, causal_mask, or rope_table");
    if (getAccess() != "internal" && getInitializer() != "none")
        return emitOpError(
            "only internal bindings may have an initializer");
    if (getInitializer() == "rope_table"
        && (!getInitializerConfig().getAs<FloatAttr>("theta")
            || !getInitializerConfig().getAs<IntegerAttr>("head_dim")))
        return emitOpError(
            "rope_table initializer requires theta and head_dim");
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
    if (auto bank = getPlacement().getAs<IntegerAttr>("bank")) {
        auto targetModel = target::LPUTargetModel::from_operation(*this);
        if (failed(targetModel)) return failure();
        if (bank.getInt() < 0
            || bank.getInt() >= targetModel->memory().banks_per_slice)
            return emitOpError("placement bank is outside the target");
    }
    return success();
}

LogicalResult TimelineOp::verify()
{
    if (getName().empty())
        return emitOpError("requires a non-empty name");
    if (getStart() < 0 || getEnd() < getStart())
        return emitOpError("requires 0 <= start <= end");
    return success();
}

LogicalResult WeightPageOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    if (getBindingIndex() < 0 || getPageIndex() < 0 || getBank() < 0
        || getBank() >= targetModel->memory().banks_per_slice
        || getReadyCycle() < 0 || getReleaseCycle() <= getReadyCycle())
        return emitOpError(
            "requires a valid binding/page/bank and non-empty residency interval");
    return success();
}

LogicalResult MemOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    const auto& target = *targetModel;
    const int64_t memoryRows = target.memory().sram_depth_rows;
    const int64_t waveCount = getWaveCount().value_or(1);
    const int64_t waveInterval = getWaveInterval().value_or(1);
    const int64_t waveAddressStride =
        getWaveAddressStride().value_or(0);
    if (getCycle() < 0 || getQueue() < 0
        || getQueue() >= target.memory().hemispheres
                * target.memory().slices_per_hemisphere
                * target.memory().banks_per_slice
        || getAddress() < 0 || getAddress() >= memoryRows
        || getPackedStream() < 0 || getPackedStream() >= target.streams().encoded_streams
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0
        || waveCount <= 0 || waveInterval <= 0)
        return emitOpError("contains an invalid ICU MEM queue command field");
    if (getOpcode() != "read" && getOpcode() != "write"
        && getOpcode() != "write_tap")
        return emitOpError("opcode must be read, write, or write_tap");
    const int64_t corners[] = {
        getAddress(),
        getAddress() + (getRepeatCount() - 1) * getAddressStride(),
        getAddress() + (waveCount - 1) * waveAddressStride,
        getAddress() + (waveCount - 1) * waveAddressStride
            + (getRepeatCount() - 1) * getAddressStride(),
    };
    if (llvm::any_of(corners,
            [&](int64_t address) {
                return address < 0 || address >= memoryRows;
            }))
        return emitOpError("wave/repeat address range is outside SRAM");
    return success();
}

LogicalResult MemBundleOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    const auto& target = *targetModel;
    const int64_t queueCount = target.memory().hemispheres
        * target.memory().slices_per_hemisphere
        * target.memory().banks_per_slice;
    const int64_t waveCount = getWaveCount().value_or(1);
    const int64_t waveInterval = getWaveInterval().value_or(1);
    const int64_t waveStride = getWaveAddressStride().value_or(0);
    if (getCycles().empty()
        || getCycles().size() != getQueues().size()
        || getCycles().size() != getAddresses().size()
        || getCycles().size() != getPackedStreams().size()
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0
        || waveCount <= 0 || waveInterval <= 0)
        return emitOpError(
            "requires equal non-empty lane arrays and positive repeat fields");
    if (getOpcode() != "read" && getOpcode() != "write"
        && getOpcode() != "write_tap")
        return emitOpError("opcode must be read, write, or write_tap");
    for (auto [cycleAttr, queueAttr, addressAttr, streamAttr] :
        llvm::zip(getCycles(), getQueues(), getAddresses(),
            getPackedStreams())) {
        const int64_t cycle =
            llvm::cast<IntegerAttr>(cycleAttr).getInt();
        const int64_t queue =
            llvm::cast<IntegerAttr>(queueAttr).getInt();
        const int64_t address =
            llvm::cast<IntegerAttr>(addressAttr).getInt();
        const int64_t stream =
            llvm::cast<IntegerAttr>(streamAttr).getInt();
        const int64_t corners[] = {
            address,
            address + (getRepeatCount() - 1) * getAddressStride(),
            address + (waveCount - 1) * waveStride,
            address + (waveCount - 1) * waveStride
                + (getRepeatCount() - 1) * getAddressStride(),
        };
        if (cycle < 0 || queue < 0 || queue >= queueCount
            || stream < 0 || stream >= target.streams().encoded_streams
            || llvm::any_of(corners, [&](int64_t corner) {
                   return corner < 0
                       || corner >= target.memory().sram_depth_rows;
               }))
            return emitOpError("contains an invalid bundled MEM lane");
    }
    return success();
}

LogicalResult MxmOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    const auto& target = *targetModel;
    if (getCycle() < 0 || !target.is_valid_mxm_unit(getQueue())
        || !target.is_valid_weight_buffer(getWeightBuffer())
        || getWeightColumn() < 0 || getWeightColumn() >= target.throughput().tile_rows
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0)
        return emitOpError("contains an invalid ICU MXM queue command field");
    if (getOpcode() != "iw" && getOpcode() != "compute"
        && getOpcode() != "accumulator_read")
        return emitOpError("opcode must be iw, compute, or accumulator_read");
    const int64_t waveCount = getWaveCount().value_or(1);
    const int64_t waveInterval = getWaveInterval().value_or(1);
    const int64_t waveColumnStride =
        getWaveWeightColumnStride().value_or(0);
    const int64_t waveAccumulatorStride =
        getWaveAccumulatorAddressStride().value_or(0);
    const int64_t groupCount = getGroupCount().value_or(1);
    const int64_t groupInterval = getGroupInterval().value_or(1);
    const int64_t finalWeightColumn = getWeightColumn()
        + (waveCount - 1) * waveColumnStride;
    if (waveCount <= 0 || waveInterval <= 0
        || groupCount <= 0 || groupInterval <= 0
        || finalWeightColumn < 0
        || finalWeightColumn >= target.throughput().tile_rows)
        return emitOpError("contains an invalid ICU MXM command wave");
    if (waveColumnStride != 0 && waveAccumulatorStride != 0)
        return emitOpError(
            "an ICU MXM command wave may induct only one hardware field");
    if (getOpcode() == "iw" && waveAccumulatorStride != 0)
        return emitOpError(
            "an iw wave cannot induct the MXM accumulator address");
    if (getOpcode() != "iw" && waveColumnStride != 0)
        return emitOpError(
            "only an iw wave may induct the MXM weight column");
    const llvm::StringRef dataFormat =
        getDataFormat().value_or("fp16");
    if (dataFormat != "fp16" && dataFormat != "bf16")
        return emitOpError("data_format must be fp16 or bf16");
    const llvm::StringRef accumulatorOutputFormat =
        getAccumulatorOutputFormat().value_or("fp32");
    if (accumulatorOutputFormat != "fp32"
        && accumulatorOutputFormat != "bf16")
        return emitOpError(
            "accumulator_output_format must be fp32 or bf16");
    const llvm::StringRef loadMode =
        getWeightLoadMode().value_or("supercell");
    const int64_t innerColumn = getWeightInnerColumn().value_or(0);
    if (loadMode != "supercell" && loadMode != "column")
        return emitOpError(
            "weight_load_mode must be supercell or column");
    if (getOpcode() != "iw" && loadMode != "supercell")
        return emitOpError(
            "only iw may use column weight loading");
    if ((loadMode == "supercell" && innerColumn != 0)
        || (loadMode == "column"
            && (innerColumn < 0
                || innerColumn >= target.throughput().lanes_per_tile)))
        return emitOpError("contains an invalid MXM inner weight column");
    if (getActivationStreamBase() < 0
        || getActivationStreamBase() >= target.streams().encoded_streams
        || getOutputStreamBase() < 0
        || getOutputStreamBase() + target.throughput().mxm_result_streams - 1
            >= target.streams().encoded_streams)
        return emitOpError("contains an invalid MXM stream selector");
    const int64_t accumulatorRows =
        target.throughput().mxm_accumulator_blocks
        * target.throughput().mxm_rows;
    const int64_t finalAccumulatorAddress = getAccumulatorAddress()
        + (waveCount - 1) * waveAccumulatorStride;
    if (getAccumulatorAddress() < 0
        || getAccumulatorAddress() >= accumulatorRows
        || finalAccumulatorAddress < 0
        || finalAccumulatorAddress >= accumulatorRows
        || getAccumulatorRowStride() <= 0)
        return emitOpError(
            "contains an invalid MXM accumulator address or stride: address=")
            << getAccumulatorAddress()
            << ", stride=" << getAccumulatorRowStride();
    if (getAccumulatorDestination() != "sram"
        && getAccumulatorDestination() != "stream")
        return emitOpError("accumulator_destination must be sram or stream");
    const llvm::StringRef inputMode =
        getWeightInputMode().value_or("direct16");
    if (inputMode != "direct16"
        && inputMode != "int8_dequant_bf16")
        return emitOpError(
            "weight_input_mode must be direct16 or int8_dequant_bf16");
    const int64_t weightStreams = loadMode == "column"
        ? inputMode == "int8_dequant_bf16" ? 1 : 2
        : inputMode == "int8_dequant_bf16"
        ? target.throughput().mxm_int8_load_streams_per_cycle
        : target.throughput().mxm_load_streams_per_cycle;
    const int64_t weightStreamBase = getWeightStreamBase().value_or(0);
    if (weightStreamBase < 0
        || weightStreamBase + weightStreams
            > target.streams().streams_per_direction)
        return emitOpError("contains an invalid MXM weight stream range");
    return success();
}

LogicalResult LoopOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    const auto& target = *targetModel;
    const int64_t cycle = getCycleAttr().getInt();
    const int64_t queue = getQueueAttr().getInt();
    const int64_t windowSize = getWindowSizeAttr().getInt();
    const int64_t count = getCountAttr().getInt();
    const int64_t interval = getIntervalAttr().getInt();
    const int64_t addressStride = getAddressStrideAttr().getInt();
    if (cycle < 0 || queue < 0
        || windowSize <= 0 || windowSize > 63
        || count <= 0 || count > 255
        || interval < windowSize || interval > 255
        || addressStride < -128 || addressStride > 127)
        return emitOpError("contains an invalid ICU Loop field");
    const auto kind = getQueueKind();
    const bool validQueue =
        (kind == "mem"
            && queue < target.memory().hemispheres
                    * target.memory().slices_per_hemisphere)
        || ((kind == "mxm_load" || kind == "mxm_compute"
                || kind == "mxm_dequant")
            && target.is_valid_mxm_unit(queue))
        || (kind == "vxm" && target.is_valid_vxm_alu(queue))
        || ((kind == "sxm_transpose" || kind == "sxm_permute")
            && queue < target.memory().hemispheres);
    if (!validQueue)
        return emitOpError("contains an invalid ICU Loop queue");
    if (kind != "mem" && addressStride != 0)
        return emitOpError(
            "only a MEM ICU Loop may use address_stride");
    return success();
}

LogicalResult MxmDequantOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    if (getCycle() < 0
        || !targetModel->is_valid_mxm_unit(getQueue())
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0
        || getWaveCount().value_or(1) <= 0
        || getWaveInterval().value_or(1) <= 0
        || (getScaleBinding() && *getScaleBinding() < 0)
        || !std::isfinite(getScaleAttr().getValueAsDouble()))
        return emitOpError(
            "contains an invalid MXM dequant queue command field");
    return success();
}

LogicalResult VxmOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    const auto& target = *targetModel;
    const int64_t cycle = getCycleAttr().getInt();
    const int64_t queue = getQueueAttr().getInt();
    const int64_t lhs_index = getLhsIndexAttr().getInt();
    const int64_t rhs_index = getRhsIndexAttr().getInt();
    const int64_t output_stream = getOutputStreamAttr().getInt();
    const int64_t repeat_count = getRepeatCountAttr().getInt();
    const int64_t repeat_interval = getRepeatIntervalAttr().getInt();
    const int64_t chainDepth = getChainDepth().value_or(8);
    if (cycle < 0 || !target.is_valid_vxm_alu(queue) || queue >= 8
        || (chainDepth != 2 && chainDepth != 4 && chainDepth != 8)
        || repeat_count <= 0 || repeat_interval <= 0)
        return emitOpError("contains an invalid ICU VXM queue command field: cycle=")
            << cycle << ", queue=" << queue
            << ", chain_depth=" << chainDepth
            << ", repeat_count=" << repeat_count
            << ", repeat_interval=" << repeat_interval;
    if (getAccumulatorReset().value_or(false)
        && !getAccumulatorWrite().value_or(false))
        return emitOpError(
            "accumulator_reset requires accumulator_write");
    if (!getAccumulatorWrite().value_or(false)
        && !getAccumulatorEmit().value_or(true))
        return emitOpError(
            "accumulator_emit=false requires accumulator_write");
    if (getLocalScalarWrite().value_or(false)
        && getAccumulatorWrite().value_or(false))
        return emitOpError(
            "local_scalar_write cannot be combined with accumulator_write");

    const auto opcode = getOpcode();
    if (opcode != "pass" && opcode != "bypass" && opcode != "cast"
        && opcode != "add" && opcode != "subtract"
        && opcode != "multiply" && opcode != "negate"
        && opcode != "max" && opcode != "exp"
        && opcode != "reciprocal" && opcode != "rsqrt")
        return emitOpError("contains an unsupported VXM opcode");

    const auto verify_operand = [&](StringRef kind, int64_t index) {
        if (kind == "immediate") return index == 0;
        if (kind == "previous" || kind == "original"
            || kind == "auxiliary" || kind == "accumulator"
            || kind == "feedback")
            return index == 0;
        if (kind == "alu") return index == queue - 1;
        if (kind == "stream_i8" || kind == "stream_f16"
            || kind == "stream_bf16" || kind == "stream_f32")
            return index >= 0
                && index + 1 < target.streams().encoded_streams;
        return false;
    };
    if (!verify_operand(getLhsKind(), lhs_index)
        || !verify_operand(getRhsKind(), rhs_index))
        return emitOpError("contains an invalid VXM operand kind or index");
    const bool chainHead = queue % chainDepth == 0;
    const auto isStreamOperand = [](StringRef kind) {
        return kind == "stream_i8" || kind == "stream_f16"
            || kind == "stream_bf16" || kind == "stream_f32";
    };
    if (chainHead) {
        const bool validLhs = isStreamOperand(getLhsKind())
            || getLhsKind() == "immediate" || getLhsKind() == "feedback";
        const bool validRhs = isStreamOperand(getRhsKind())
            || getRhsKind() == "immediate";
        if (!validLhs || !validRhs)
            return emitOpError(
                "VXM chain head operands must use stream/immediate, "
                "except lhs may use feedback");
    } else {
        const bool validRhs = getRhsKind() == "original"
            || getRhsKind() == "auxiliary" || getRhsKind() == "immediate"
            || (getRhsKind() == "accumulator"
                && (queue % 4 == 1 || queue % 4 == 3));
        const bool validLhs = getLhsKind() == "previous"
            || (getLhsKind() == "alu" && lhs_index == queue - 1);
        if (!validLhs || !validRhs)
            return emitOpError(
                "VXM internal stage operands are outside the fixed local mux");
    }
    if (!std::isfinite(getLhsImmediateAttr().getValueAsDouble())
        || !std::isfinite(getRhsImmediateAttr().getValueAsDouble()))
        return emitOpError("VXM immediate operands must be finite");
    if (getCastTarget() != "fp32" && getCastTarget() != "fp16"
        && getCastTarget() != "bf16" && getCastTarget() != "i8")
        return emitOpError(
            "cast_target must be fp32, fp16, bf16, or i8");
    if (output_stream < -1 || output_stream >= target.streams().encoded_streams)
        return emitOpError("output_stream must be -1 or a packed stream selector");
    if (output_stream >= 0 && queue % chainDepth != chainDepth - 1)
        return emitOpError("output_stream requires a configured VXM chain tail: queue=")
            << queue << ", chain_depth=" << chainDepth
            << ", output_stream=" << output_stream
            << ", cycle=" << cycle;
    if (output_stream >= 0 && output_stream != (queue / 2) * 2)
        return emitOpError("output_stream does not match the fixed VXM output block: queue=")
            << queue << ", output_stream=" << output_stream
            << ", expected=" << (queue / 2) * 2;
    if ((getInputHemisphere() != "east" && getInputHemisphere() != "west")
        || (getOutputHemisphere() != "east" && getOutputHemisphere() != "west"))
        return emitOpError("hemisphere must be east or west");
    return success();
}

LogicalResult SxmOp::verify()
{
    auto targetModel = target::LPUTargetModel::from_operation(*this);
    if (failed(targetModel)) return failure();
    const auto& target = *targetModel;
    if (getCycle() < 0 || getHemisphere() < 0
        || getHemisphere() >= target.memory().hemispheres)
        return emitOpError("contains an invalid ICU SXM queue selector");
    if (getRepeatCount().value_or(1) <= 0
        || getRepeatInterval().value_or(1) <= 0)
        return emitOpError("repeat count and interval must be positive");
    if (getOpcode() != "transpose" && getOpcode() != "permute")
        return emitOpError("opcode must be transpose or permute");
    if (getOutputRow()
        && (*getOutputRow() < 0
            || *getOutputRow() >= target.throughput().lanes_per_tile))
        return emitOpError("output_row must be in the physical lane range");
      if (getInputRow()
          && (*getInputRow() < 0
              || *getInputRow() >= target.throughput().lanes_per_tile))
          return emitOpError("input_row must be in the physical lane range");
      if (getOutputTile()
          && (*getOutputTile() < 0
              || *getOutputTile() >= target.throughput().tile_rows))
          return emitOpError("output_tile must be in the physical tile range");
    const int64_t physicalWidth =
        2 * target.throughput().lanes_per_tile;
    const int64_t sourceWidth = getInputRow() ? 2 : physicalWidth;
    if (getSourceStreams().size() != sourceWidth
        || getDestinationStreams().size() != physicalWidth
        || getPermuteMap().size() != 32)
        return emitOpError()
            << "requires " << sourceWidth << " source and "
            << physicalWidth
            << " destination byte streams and a 32-lane map";
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
