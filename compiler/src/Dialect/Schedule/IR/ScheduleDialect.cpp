#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#include "ftlpu/compiler/Dialect/Schedule/IR/ScheduleOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ftlpu/compiler/Dialect/Schedule/IR/ScheduleOps.cpp.inc"

namespace ftlpu::compiler::schedule {
namespace {

LogicalResult verify_timing(Operation* op, int64_t cycle, int64_t duration)
{
    if (cycle < 0 || duration <= 0)
        return op->emitOpError("requires a non-negative cycle and positive duration");
    return success();
}

LogicalResult verify_queue_issue(Operation* op, int64_t cycle, int64_t queue,
    int64_t repeat_count, int64_t repeat_interval)
{
    if (cycle < 0 || queue < 0 || repeat_count <= 0 || repeat_interval <= 0)
        return op->emitOpError("requires non-negative cycle/queue and positive repeat fields");
    return success();
}

LogicalResult verify_stream_range(Operation* op, int64_t base, int64_t count)
{
    const target::LPUTargetModel target;
    if (base < 0 || count <= 0 || base + count > target.streams().streams_per_direction)
        return op->emitOpError("stream range is outside the target direction");
    return success();
}

} // namespace

LogicalResult MemReadOp::verify()
{
    if (failed(verify_timing(*this, getCycle(), getDuration()))
        || failed(verify_stream_range(*this, getStreamBase(), getStreamCount())))
        return failure();
    auto placement = getPlacement();
    auto kind = placement.getAs<StringAttr>("kind");
    if (kind && kind.getValue() == "schedule_slice") {
        if (getDirection() != "east" && getDirection() != "west")
            return emitOpError("direction must be east or west");
        auto hemisphere = placement.getAs<StringAttr>("hemisphere");
        if (!hemisphere || (hemisphere.getValue() != "east"
                            && hemisphere.getValue() != "west"))
            return emitOpError("scheduled MEM read requires a physical hemisphere");
        if (getRole() != "activation" && getRole() != "weight_i8"
            && getRole() != "vxm_fp16" && getRole() != "vxm_fp32")
            return emitOpError("contains an unsupported scheduled MEM role");
        auto slices = placement.getAs<ArrayAttr>("slices");
        if (!slices || slices.size() != 1 || getStreamCount() != 1)
            return emitOpError("scheduled MEM read must select one slice and one stream");
        return success();
    }
    if (getDirection() != "east") return emitOpError("must use east streams");
    target::StreamEndpoint endpoint;
    if (getRole() == "weight") endpoint = target::StreamEndpoint::MxmWeight;
    else if (getRole() == "activation") endpoint = target::StreamEndpoint::MxmActivation;
    else return emitOpError("role must be weight or activation");
    const target::LPUTargetModel target;
    const auto expected_count = target.route_stream_count(
        target::StreamEndpoint::Mem, endpoint, target::StreamDirection::East);
    auto instruction_count = placement.getAs<IntegerAttr>("instruction_count");
    if (!expected_count || !instruction_count || getStreamCount() != *expected_count
        || getDuration() != instruction_count.getInt())
        return emitOpError("stream count or duration does not match the target model");
    return success();
}

LogicalResult MxmLoadOp::verify()
{
    const target::LPUTargetModel target;
    if (failed(verify_timing(*this, getCycle(), getDuration()))
        || failed(verify_stream_range(*this, getStreamBase(), getStreamCount())))
        return failure();
    if (getStreamCount() != target.throughput().mxm_load_streams_per_cycle)
        return emitOpError("must consume 16 weight streams");
    if (!target.is_valid_mxm_unit(getUnitId()))
        return emitOpError("unit_id must select MXM0 or MXM1");
    if (!target.is_valid_weight_buffer(getWeightBuffer()))
        return emitOpError("weight_buffer must select buffer 0 or 1");
    return success();
}

LogicalResult MxmComputeOp::verify()
{
    const target::LPUTargetModel target;
    if (failed(verify_timing(*this, getCycle(), getDuration()))) return failure();
    if (getM() <= 0 || getN() <= 0 || getK() <= 0)
        return emitOpError("matrix dimensions must be positive");
    if (!target.is_valid_mxm_unit(getUnitId())
        || !target.is_valid_weight_buffer(getWeightBuffer()))
        return emitOpError("contains an invalid MXM unit or weight buffer");
    if (getDuration() != target.mxm_compute_issue_cycles(getM())
        || getResultCycle() != getCycle() + target.mxm_first_result_latency()
        || getResultDuration() != target.mxm_result_window_cycles(getM()))
        return emitOpError("compute or result timing does not match the MXM pipeline");
    if (failed(verify_stream_range(*this, getActivationStreamBase(), 1))
        || failed(verify_stream_range(*this, getOutputStreamBase(), 4)))
        return failure();
    return success();
}

LogicalResult VxmOp::verify()
{
    const target::LPUTargetModel target;
    if (getCycleAttr().getInt() < 0 || !target.is_valid_vxm_alu(getQueue())
        || getRepeatCount() <= 0 || getRepeatInterval() <= 0)
        return emitOpError("contains invalid cycle, ALU queue, or repeat metadata");
    const auto valid_kind = [](StringRef kind) {
        return kind == "alu" || kind == "stream_i32" || kind == "stream_f32"
            || kind == "stream_i8" || kind == "stream_f16"
            || kind == "immediate";
    };
    if (!valid_kind(getLhsKind()) || !valid_kind(getRhsKind()))
        return emitOpError("contains an invalid operand kind");
    const auto valid_operand = [&](StringRef kind, int64_t index) {
        if (kind == "immediate") return index == 0;
        if (kind == "alu") return target.is_valid_vxm_alu(index);
        return index >= 0 && index + 3 < target.streams().encoded_streams;
    };
    if (!valid_operand(getLhsKind(), getLhsIndex())
        || !valid_operand(getRhsKind(), getRhsIndex()))
        return emitOpError("contains an invalid operand index");
    const int64_t output = getOutputStreamAttr().getInt();
    if (output < -1 || output >= target.streams().encoded_streams)
        return emitOpError("contains an invalid output stream");
    if ((getInputHemisphere() != "east" && getInputHemisphere() != "west")
        || (getOutputHemisphere() != "east" && getOutputHemisphere() != "west"))
        return emitOpError("hemisphere must be east or west");
    return success();
}

LogicalResult MemAccumulateOp::verify()
{
    if (getRepeatCount() <= 0 || getRepeatInterval() <= 0
        || getAddressStride() > 4095)
        return emitOpError("contains invalid accumulation timing or stride: cycle=")
            << getCycle() << ", repeat_count=" << getRepeatCount()
            << ", repeat_interval=" << getRepeatInterval()
            << ", address_stride=" << getAddressStride();
    if (getHemisphere() != "east" && getHemisphere() != "west")
        return emitOpError("hemisphere must be east or west");
    if (getDestination() != "sram" && getDestination() != "stream")
        return emitOpError("destination must be sram or stream");
    if (failed(verify_stream_range(*this, getStreamBase(), getStreamCount())))
        return failure();
    return success();
}

LogicalResult MemWriteOp::verify()
{
    const target::LPUTargetModel target;
    if (failed(verify_timing(*this, getCycle(), getDuration()))
        || failed(verify_stream_range(*this, getStreamBase(), getStreamCount())))
        return failure();
    auto placement = getPlacement();
    auto kind = placement.getAs<StringAttr>("kind");
    auto hemisphere = placement.getAs<StringAttr>("hemisphere");
    if (!hemisphere) hemisphere = getAddress().getAs<StringAttr>("hemisphere");
    auto slices = placement.getAs<ArrayAttr>("slices");
    auto instructionCount = placement.getAs<IntegerAttr>("instruction_count");
    const bool exactPhysicalWrite = hemisphere && slices && instructionCount
        && (hemisphere.getValue() == "east" || hemisphere.getValue() == "west")
        && instructionCount.getInt() == getDuration()
        && static_cast<int64_t>(slices.size()) == getStreamCount();
    if ((kind && kind.getValue() == "schedule_slice") || exactPhysicalWrite) {
        if (getDirection() != "east" && getDirection() != "west")
            return emitOpError("direction must be east or west");
        if (!exactPhysicalWrite)
            return emitOpError("scheduled MEM write does not match its physical slices");
        return success();
    }
    const bool vxm = static_cast<bool>(getInput().getDefiningOp<VxmOp>());
    auto instruction_count = placement.getAs<IntegerAttr>("instruction_count");
    const auto expected_direction = vxm ? "east" : "west";
    const int64_t expected_streams = vxm ? 1 : target.throughput().mxm_result_streams;
    if (getDirection() != expected_direction || getStreamCount() != expected_streams
        || !instruction_count || getDuration() != instruction_count.getInt())
        return emitOpError("stream count or duration does not match the result route");
    return success();
}

LogicalResult AttentionOp::verify()
{
    int64_t previous_end = 0;
    for (Attribute attribute : getPhases()) {
        const auto phase = llvm::dyn_cast<DictionaryAttr>(attribute);
        const auto name = phase ? phase.getAs<StringAttr>("name") : StringAttr {};
        const auto start = phase ? phase.getAs<IntegerAttr>("start") : IntegerAttr {};
        const auto end = phase ? phase.getAs<IntegerAttr>("end") : IntegerAttr {};
        if (!name || !start || !end || start.getInt() < previous_end || end.getInt() <= start.getInt())
            return emitOpError() << "contains non-monotonic attention phase timing at '"
                                 << (name ? name.getValue() : "<missing>") << "'";
        previous_end = end.getInt();
    }
    if (getPhases().size() != 6)
        return emitOpError("requires qkv, rope, qk, softmax, pv, and o_proj phases");
    if (getWorkWaves().empty()) return emitOpError("requires scheduled QK/PV work waves");
    for (Attribute attribute : getWorkWaves()) {
        const auto wave = llvm::dyn_cast<DictionaryAttr>(attribute);
        const auto phase = wave ? wave.getAs<StringAttr>("phase") : StringAttr {};
        const auto start = wave ? wave.getAs<IntegerAttr>("start") : IntegerAttr {};
        const auto end = wave ? wave.getAs<IntegerAttr>("end") : IntegerAttr {};
        const auto slots = wave ? wave.getAs<ArrayAttr>("slots") : ArrayAttr {};
        if (!phase || (phase.getValue() != "qk" && phase.getValue() != "pv")
            || !start || !end || end.getInt() <= start.getInt() || !slots || slots.empty())
            return emitOpError("contains an invalid physical attention work wave");
    }
    return success();
}

LogicalResult MemTransferOp::verify()
{
    const target::LPUTargetModel target;
    if (failed(verify_queue_issue(*this, getCycle(), getSlice(), getRepeatCount(), getRepeatInterval()))
        || getHemisphere() < 0 || getHemisphere() >= target.memory().hemispheres
        || getSlice() >= target.memory().slices_per_hemisphere
        || getAddress() < 0 || getAddress() >= target.memory().words_per_bank * target.memory().banks_per_slice
        || getPackedStream() < 0 || getPackedStream() >= target.streams().encoded_streams)
        return emitOpError("contains an invalid MEM transfer: cycle=")
            << getCycle() << ", hemisphere=" << getHemisphere()
            << ", slice=" << getSlice()
            << ", address=" << getAddress()
            << ", packed_stream=" << getPackedStream()
            << ", repeat_count=" << getRepeatCount()
            << ", repeat_interval=" << getRepeatInterval();
    if (getOpcode() != "read" && getOpcode() != "write" && getOpcode() != "accumulate")
        return emitOpError("opcode must be read, write, or accumulate");
    return success();
}

LogicalResult MxmIssueOp::verify()
{
    const target::LPUTargetModel target;
    if (failed(verify_queue_issue(*this, getCycle(), getUnitId(), getRepeatCount(), getRepeatInterval()))
        || !target.is_valid_mxm_unit(getUnitId()) || !target.is_valid_weight_buffer(getWeightBuffer())
        || getWeightColumn() < 0 || getWeightColumn() >= target.throughput().tile_rows)
        return emitOpError("contains an invalid MXM issue: cycle=")
            << getCycle() << ", unit_id=" << getUnitId()
            << ", weight_buffer=" << getWeightBuffer()
            << ", weight_column=" << getWeightColumn()
            << ", activation_stream_base=" << getActivationStreamBase()
            << ", output_stream_base=" << getOutputStreamBase();
    if (getOpcode() != "iw" && getOpcode() != "compute")
        return emitOpError("opcode must be iw or compute");
    return success();
}

LogicalResult SxmOp::verify()
{
    const target::LPUTargetModel target;
    if (getCycle() < 0 || getHemisphere() < 0
        || getHemisphere() >= target.memory().hemispheres)
        return emitOpError("contains an invalid SXM queue selector");
    if (getOpcode() != "transpose" && getOpcode() != "permute")
        return emitOpError("opcode must be transpose or permute");
    if (getSourceStreams().empty() || getDestinationStreams().empty()
        || getPermuteMap().size() != 32)
        return emitOpError("requires stream lists and a 32-lane map");
    return success();
}

void ScheduleDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "ftlpu/compiler/Dialect/Schedule/IR/ScheduleOps.cpp.inc"
        >();
}

} // namespace ftlpu::compiler::schedule
