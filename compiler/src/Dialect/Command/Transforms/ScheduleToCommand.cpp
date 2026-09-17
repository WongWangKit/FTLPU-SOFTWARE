#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Command/Transforms/fu_3d_command_materializer.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace ftlpu::compiler {
namespace {

constexpr int64_t kMaxRepeatCount = 1023;
constexpr int64_t kMaxRepeatInterval = 255;
constexpr int64_t kMinRepeatStride = -2048;
constexpr int64_t kMaxRepeatStride = 2047;

llvm::SmallVector<int64_t> placement_slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute attribute : placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(attribute).getInt());
    return result;
}

int64_t placement_integer(mlir::DictionaryAttr placement, llvm::StringRef name)
{
    return placement.getAs<mlir::IntegerAttr>(name).getInt();
}

int64_t placement_integer_or(mlir::DictionaryAttr placement,
    llvm::StringRef name, int64_t fallback)
{
    if (auto value = placement.getAs<mlir::IntegerAttr>(name))
        return value.getInt();
    return fallback;
}

int64_t mem_queue(const target::LPUTargetModel& target,
    int64_t hemisphere, int64_t slice, int64_t bank)
{
    return hemisphere * target.memory().slices_per_hemisphere
            * target.memory().banks_per_slice
        + slice * target.memory().banks_per_slice + bank;
}

mlir::StringAttr placement_hemisphere(mlir::DictionaryAttr placement,
    mlir::DictionaryAttr address)
{
    if (auto hemisphere = placement.getAs<mlir::StringAttr>("hemisphere"))
        return hemisphere;
    return address.getAs<mlir::StringAttr>("hemisphere");
}

llvm::StringRef element_type_name(mlir::Type type)
{
    auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
    if (integer && integer.getWidth() == 8) return "i8";
    if (integer && integer.getWidth() == 32) return "i32";
    if (type.isF16()) return "f16";
    if (type.isBF16()) return "bf16";
    if (type.isF32()) return "f32";
    return "unsupported";
}

int64_t element_type_bytes(mlir::Type type)
{
    if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
        return (integer.getWidth() + 7) / 8;
    if (is_lpu_16bit_float(type)) return 2;
    if (type.isF32()) return 4;
    return 0;
}

std::optional<int64_t> source_binding_index(mlir::Value value)
{
    llvm::SmallVector<mlir::Value> pending {value};
    llvm::SmallDenseSet<mlir::Value, 16> visited;
    std::optional<int64_t> result;
    while (!pending.empty()) {
        mlir::Value current = pending.pop_back_val();
        if (!visited.insert(current).second) continue;
        if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(current)) {
            const int64_t index = argument.getArgNumber();
            if (result && *result != index) return std::nullopt;
            result = index;
            continue;
        }
        mlir::Operation* defining = current.getDefiningOp();
        if (!defining) continue;
        if (auto binding = llvm::dyn_cast<schedule::BindingOp>(defining)) {
            if (binding.getAccess() != "input") continue;
            const int64_t index = binding.getIndex();
            if (result && *result != index) return std::nullopt;
            result = index;
            continue;
        }
        for (mlir::Value operand : defining->getOperands())
            pending.push_back(operand);
    }
    return result;
}

void create_binding(mlir::OpBuilder& builder, mlir::Location location,
    int64_t index, llvm::StringRef access, llvm::StringRef role,
    llvm::StringRef name, int64_t readyCycle, mlir::RankedTensorType type,
    int64_t bytes, mlir::DictionaryAttr placement,
    llvm::StringRef initializer = "none",
    mlir::DictionaryAttr initializerConfig = {})
{
    llvm::SmallVector<mlir::Attribute> shape;
    for (int64_t dimension : type.getShape())
        shape.push_back(builder.getI64IntegerAttr(dimension));
    mlir::OperationState state(location, command::BindingOp::getOperationName());
    state.addAttributes({
        builder.getNamedAttr("index", builder.getI64IntegerAttr(index)),
        builder.getNamedAttr("access", builder.getStringAttr(access)),
        builder.getNamedAttr("role", builder.getStringAttr(role)),
        builder.getNamedAttr("name", builder.getStringAttr(name)),
        builder.getNamedAttr(
            "ready_cycle", builder.getI64IntegerAttr(readyCycle)),
        builder.getNamedAttr(
            "initializer", builder.getStringAttr(initializer)),
        builder.getNamedAttr("initializer_config",
            initializerConfig
                ? initializerConfig
                : builder.getDictionaryAttr({})),
        builder.getNamedAttr("shape", builder.getArrayAttr(shape)),
        builder.getNamedAttr("element_type", builder.getStringAttr(element_type_name(type.getElementType()))),
        builder.getNamedAttr("bytes", builder.getI64IntegerAttr(bytes)),
        builder.getNamedAttr("placement", placement),
    });
    builder.create(state);
}

VxmLaneOperation parse_schedule_vxm_operation(llvm::StringRef value)
{
    if (value == "pass" || value == "bypass" || value == "cast")
        return VxmAluOpcode::Bypass;
    if (value == "add") return VxmAluOpcode::Add;
    if (value == "subtract") return VxmAluOpcode::Subtract;
    if (value == "multiply") return VxmAluOpcode::Multiply;
    if (value == "fma") return VxmAluOpcode::FusedMultiplyAdd;
    if (value == "fms") return VxmAluOpcode::FusedMultiplySubtract;
    if (value == "negate") return VxmAluOpcode::Negate;
    if (value == "max") return VxmAluOpcode::Max;
    if (value == "exp") return VxmSpecialAluOpcode::Exp;
    if (value == "reciprocal" || value == "divide")
        return VxmSpecialAluOpcode::Reciprocal;
    if (value == "rsqrt") return VxmSpecialAluOpcode::Rsqrt;
    throw std::runtime_error(
        "Schedule IR VXM operation is not implemented by the current CModel");
}

std::size_t as_size(int64_t value, llvm::StringRef field)
{
    if (value < 0)
        throw std::runtime_error((field + " must be non-negative").str());
    return static_cast<std::size_t>(value);
}

IcuLoop3D loop_3d(int64_t cycle, int64_t innerCount,
    int64_t innerInterval, int64_t middleCount = 1,
    int64_t middleInterval = 1, int64_t outerCount = 1,
    int64_t outerInterval = 1)
{
    return IcuLoop3D {
        as_size(cycle, "start cycle"),
        {as_size(innerCount, "inner count"),
            as_size(middleCount, "middle count"),
            as_size(outerCount, "outer count")},
        {as_size(innerInterval, "inner interval"),
            as_size(middleInterval, "middle interval"),
            as_size(outerInterval, "outer interval")},
    };
}

MemIcuAddress3D mem_address_3d(mlir::Operation* operation,
    std::size_t baseAddress, int64_t innerStride, int64_t middleStride,
    int64_t affineOuterStride)
{
    const auto outerGroupSize =
        operation->getAttrOfType<mlir::IntegerAttr>("outer_group_size");
    if (!outerGroupSize)
        return MemIcuAddress3D::Affine(baseAddress,
            {innerStride, middleStride, affineOuterStride});
    const auto outerInnerStride =
        operation->getAttrOfType<mlir::IntegerAttr>("outer_inner_stride");
    const auto outerGroupStride =
        operation->getAttrOfType<mlir::IntegerAttr>("outer_group_stride");
    if (!outerInnerStride || !outerGroupStride)
        throw std::runtime_error(
            "blocked MEM address fields must be specified together");
    return MemIcuAddress3D::BlockedOuter(baseAddress, innerStride,
        middleStride,
        as_size(outerGroupSize.getInt(), "MEM outer group size"),
        outerInnerStride.getInt(), outerGroupStride.getInt());
}

MxmDataFormat mxm_data_format(llvm::StringRef value)
{
    if (value == "bf16") return MxmDataFormat::BFloat16;
    if (value == "fp16") return MxmDataFormat::Float16;
    throw std::runtime_error("MXM data format must be fp16 or bf16");
}

MxmAccumulatorOutputFormat mxm_accumulator_format(llvm::StringRef value)
{
    if (value == "bf16") return MxmAccumulatorOutputFormat::BFloat16;
    if (value == "fp32") return MxmAccumulatorOutputFormat::Float32;
    throw std::runtime_error("MXM accumulator format must be fp32 or bf16");
}

MxmAccumulatorDestination mxm_destination(llvm::StringRef value)
{
    return value == "stream"
        ? MxmAccumulatorDestination::Stream
        : MxmAccumulatorDestination::Sram;
}

MxmWeightInputMode mxm_weight_input_mode(llvm::StringRef value)
{
    return value == "int8_dequant_bf16"
        ? MxmWeightInputMode::Int8DequantBf16
        : MxmWeightInputMode::Direct16;
}

MxmIcuBufferMode mxm_buffer_mode(llvm::StringRef value)
{
    if (value.empty() || value == "fixed") return MxmIcuBufferMode::Fixed;
    if (value == "toggle_dim0") return MxmIcuBufferMode::ToggleDimension0;
    if (value == "toggle_dim1") return MxmIcuBufferMode::ToggleDimension1;
    if (value == "toggle_dim2") return MxmIcuBufferMode::ToggleDimension2;
    throw std::runtime_error(
        "MXM buffer mode must be fixed or toggle_dim0/1/2");
}

void require_materialized(mlir::LogicalResult result,
    llvm::StringRef unit, const std::string& error)
{
    if (mlir::failed(result))
        throw std::runtime_error(("invalid direct " + unit +
            " lowering: " + llvm::StringRef(error)).str());
}

VxmStreamSource parse_schedule_vxm_stream_source(llvm::StringRef value)
{
    if (value.empty() || value == "local") return VxmStreamSource::Local;
    if (value == "east") return VxmStreamSource::East;
    if (value == "west") return VxmStreamSource::West;
    throw std::runtime_error(
        "Schedule IR VXM stream source must be local, east, or west");
}

VxmLaneOperand parse_schedule_vxm_operand(llvm::StringRef kind,
    int64_t index, float immediate, int64_t queue,
    llvm::StringRef streamSource)
{
    if (kind == "previous") return VxmLaneOperand::Previous();
    if (kind == "original") return VxmLaneOperand::Original();
    if (kind == "auxiliary") return VxmLaneOperand::Aux();
    if (kind == "accumulator") return VxmLaneOperand::Acc();
    if (kind == "feedback") return VxmLaneOperand::Feedback();
    if (kind == "alu") {
        if (index == queue - 1) return VxmLaneOperand::Previous();
        throw std::runtime_error(
            "arbitrary VXM alu(N) references require chain legalization");
    }
    const auto streamGroup = [&]() -> std::int32_t {
        if (index < 0 || index >= 64 || index % 2 != 0)
            throw std::runtime_error(
                "VXM 16-bit stream operand requires an even packed stream index");
        return static_cast<std::int32_t>(((index % 32) / 2) % 8);
    };
    if (kind == "stream_f16")
        return VxmLaneOperand::StreamFloat16(1.0f, streamGroup(),
            parse_schedule_vxm_stream_source(streamSource));
    if (kind == "stream_bf16")
        return VxmLaneOperand::StreamBFloat16(1.0f, streamGroup(),
            parse_schedule_vxm_stream_source(streamSource));
    if (kind == "immediate") return VxmLaneOperand::Imm(immediate);
    throw std::runtime_error(
        "legacy integer/FP32 VXM stream operands require BF16 legalization");
}

VxmCastTarget parse_schedule_vxm_cast_target(llvm::StringRef value)
{
    if (value == "fp32") return VxmCastTarget::Float32;
    if (value == "fp16") return VxmCastTarget::Float16;
    if (value == "bf16") return VxmCastTarget::BFloat16;
    if (value == "i8") return VxmCastTarget::Int8;
    throw std::runtime_error("unsupported Schedule IR VXM cast target");
}

void create_vxm_command(mlir::OpBuilder& builder, schedule::VxmOp op)
{
    const int64_t effectiveRepeatCount = op.getRepeatCount();
    const int64_t effectiveRepeatInterval = op.getRepeatInterval();
        const int64_t queue = op.getQueue();
        try {
        auto instruction = VxmLaneAluInstruction {};
        instruction.operation = parse_schedule_vxm_operation(op.getOpcode());
        const auto lhsSource =
            op->getAttrOfType<mlir::StringAttr>("lhs_stream_source");
        const auto rhsSource =
            op->getAttrOfType<mlir::StringAttr>("rhs_stream_source");
        instruction.lhs = parse_schedule_vxm_operand(op.getLhsKind(),
            op.getLhsIndex(),
            static_cast<float>(op.getLhsImmediateAttr().getValueAsDouble()),
            queue, lhsSource ? lhsSource.getValue() : llvm::StringRef {});
        instruction.rhs = parse_schedule_vxm_operand(op.getRhsKind(),
            op.getRhsIndex(),
            static_cast<float>(op.getRhsImmediateAttr().getValueAsDouble()),
            queue, rhsSource ? rhsSource.getValue() : llvm::StringRef {});
        instruction.output_type =
            parse_schedule_vxm_cast_target(op.getCastTarget());
        instruction.precision = VxmAluPrecision::Float32;
        const bool contiguousRun = effectiveRepeatInterval == 1;
        instruction.repeat_count = static_cast<std::size_t>(
            contiguousRun ? effectiveRepeatCount : 1);
        instruction.accumulator_reset =
            op.getAccumulatorReset().value_or(false);
        instruction.accumulator_write =
            op.getAccumulatorWrite().value_or(false);
        instruction.accumulator_emit =
            op.getAccumulatorEmit().value_or(true);
        instruction.local_scalar_write =
            op.getLocalScalarWrite().value_or(false);
        const int64_t outputStream = op.getOutputStreamAttr().getInt();
        if (outputStream >= 0)
            instruction.output_stream =
                static_cast<std::size_t>(outputStream);
        const auto depth = static_cast<VxmChainDepth>(
            op->getAttrOfType<mlir::IntegerAttr>("chain_depth")
                ? op->getAttrOfType<mlir::IntegerAttr>("chain_depth").getInt()
                : 8);
        const auto run = VxmIcuRun2DInstruction::Run2D(
            static_cast<std::size_t>(op.getCycle()),
            {static_cast<std::size_t>(
                 contiguousRun ? 1 : effectiveRepeatCount),
                static_cast<std::size_t>(op.getWaveCount().value_or(1))},
            {static_cast<std::size_t>(
                 contiguousRun ? 1 : effectiveRepeatInterval),
                static_cast<std::size_t>(op.getWaveInterval().value_or(1))},
            isa::encode_vxm_instruction(queue, depth, instruction));
        std::string error;
        if (mlir::failed(command::materializeVxmRun2DCommand(builder,
                op.getLoc(), static_cast<std::size_t>(queue), run, &error,
                op.getScaleBinding().value_or(-1))))
            throw std::runtime_error(error);
        } catch (const std::exception& exception) {
            throw std::runtime_error("invalid Schedule IR VXM RUN_2D at cycle "
                + std::to_string(op.getCycle()) + ", queue "
                + std::to_string(queue) + ", chain_depth "
                + std::to_string(op.getChainDepth().value_or(8)) + ": "
                + exception.what());
        }
}

void create_sxm_command(mlir::OpBuilder& builder, schedule::SxmOp op)
{
    SxmInstruction instruction {};
    instruction.opcode = op.getOpcode() == "transpose"
        ? SxmOpcode::Transpose : SxmOpcode::Permute;
    if (op.getOutputRow())
        instruction.output_row =
            static_cast<std::size_t>(*op.getOutputRow());
    if (op.getInputRow())
        instruction.input_row =
            static_cast<std::size_t>(*op.getInputRow());
    if (op.getOutputTile())
        instruction.output_tile =
            static_cast<std::size_t>(*op.getOutputTile());
    for (mlir::Attribute stream : op.getSourceStreams())
        instruction.src_streams.push_back(SxmStreamId {
            static_cast<std::size_t>(
                llvm::cast<mlir::IntegerAttr>(stream).getInt())});
    for (mlir::Attribute stream : op.getDestinationStreams())
        instruction.dst_streams.push_back(SxmStreamId {
            static_cast<std::size_t>(
                llvm::cast<mlir::IntegerAttr>(stream).getInt())});
    for (std::size_t lane = 0;
         lane < instruction.permute_map.size(); ++lane) {
        const auto value = llvm::cast<mlir::IntegerAttr>(
            op.getPermuteMap()[lane]).getInt();
        instruction.permute_map[lane] = value < 0
            ? SxmInstruction::kZeroFill
            : static_cast<std::size_t>(value);
    }
    const auto run = SxmIcuRun2DInstruction::Run2D(
        static_cast<std::size_t>(op.getCycle()),
        {static_cast<std::size_t>(op.getRepeatCount().value_or(1)),
            static_cast<std::size_t>(op.getWaveCount().value_or(1))},
        {static_cast<std::size_t>(op.getRepeatInterval().value_or(1)),
            static_cast<std::size_t>(op.getWaveInterval().value_or(1))},
        std::move(instruction),
        static_cast<std::size_t>(op->getAttrOfType<mlir::IntegerAttr>(
            "permute_map_stride")
                ? op->getAttrOfType<mlir::IntegerAttr>(
                    "permute_map_stride").getInt()
                : 0));
    std::string error;
    if (mlir::failed(command::materializeSxmRun2DCommand(builder,
            op.getLoc(), op.getOpcode() == "transpose",
            static_cast<std::size_t>(op.getHemisphere()), run, &error)))
        throw std::runtime_error(error);
}

void create_mem_transfer_command(mlir::OpBuilder& builder,
    schedule::MemTransferOp op, const target::LPUTargetModel& target)
{
    const auto loop = loop_3d(op.getCycle(), op.getRepeatCount(),
        op.getRepeatInterval(), op.getWaveCount().value_or(1),
        op.getWaveInterval().value_or(1), op.getGroupCount().value_or(1),
        op.getGroupInterval().value_or(1));
    const auto address = mem_address_3d(op.getOperation(),
        as_size(op.getAddress(), "MEM address"),
        static_cast<int64_t>(op.getAddressStride()),
        static_cast<int64_t>(op.getWaveAddressStride().value_or(0)),
        static_cast<int64_t>(op.getGroupAddressStride().value_or(0)));
    const auto stream = StreamId::from_packed(
        as_size(op.getPackedStream(), "MEM stream"));
    const auto instruction = op.getOpcode() == "read"
        ? MemIcuInstruction::Read3D(loop, address, stream)
        : op.getOpcode() == "write_tap"
        ? MemIcuInstruction::WriteTap3D(loop, address, stream)
        : MemIcuInstruction::Write3D(loop, address, stream);
    std::string error;
    require_materialized(command::materializeMem3DCommand(builder,
        op.getLoc(), as_size(mem_queue(target, op.getHemisphere(),
            op.getSlice(), op.getBank().value_or(0)), "MEM queue"),
        instruction, op.getAddressBinding().value_or(-1),
        op.getAddressBindingAccess().value_or(""), &error),
        "MEM cycle " + std::to_string(op.getCycle()) + " stream " +
            std::to_string(op.getPackedStream()), error);
}

void create_mem_write_read_2d_command(mlir::OpBuilder& builder,
    schedule::MemWriteRead2DOp op, const target::LPUTargetModel& target)
{
    MemIcuWriteRead2DInstruction instruction{};
    instruction.counts = {
        as_size(op.getCount0(), "MEM write/read inner count"),
        as_size(op.getCount1(), "MEM write/read outer count")};
    instruction.write_cycle_strides = {
        as_size(op.getWriteCycleStride0(), "MEM write stride 0"),
        as_size(op.getWriteCycleStride1(), "MEM write stride 1")};
    instruction.read_cycle_strides = {
        as_size(op.getReadCycleStride0(), "MEM read stride 0"),
        as_size(op.getReadCycleStride1(), "MEM read stride 1")};
    instruction.read_start_offset =
        as_size(op.getReadStartOffset(), "MEM read start offset");
    instruction.base_address = as_size(op.getAddress(), "MEM address");
    instruction.address_strides = {
        static_cast<int64_t>(op.getAddressStride0()),
        static_cast<int64_t>(op.getAddressStride1())};
    instruction.write_stream = as_size(op.getWriteStream(), "MEM write stream");
    instruction.read_stream_base =
        as_size(op.getReadStreamBase(), "MEM read stream base");
    instruction.read_stream_outer_stride =
        static_cast<int64_t>(op.getReadStreamOuterStride());
    const auto queue = mem_queue(target, op.getHemisphere(),
        op.getSlice(), op.getBank());
    std::string error;
    require_materialized(command::materializeMemWriteRead2DCommand(builder,
        op.getLoc(), as_size(op.getCycle(), "MEM write/read cycle"),
        as_size(queue, "MEM write/read queue"), instruction, &error),
        "MEM write/read", error);
}

void create_mxm_issue_command(mlir::OpBuilder& builder, schedule::MxmIssueOp op)
{
    const auto loop = loop_3d(op.getCycle(), op.getRepeatCount(),
        op.getRepeatInterval(), op.getWaveCount().value_or(1),
        op.getWaveInterval().value_or(1), op.getGroupCount().value_or(1),
        op.getGroupInterval().value_or(1));
    const auto queue = as_size(op.getUnitId(), "MXM queue");
    std::string error;
    if (op.getOpcode() == "iw") {
        if (op.getWeightLoadMode().value_or("supercell") == "column"
            || op.getWeightInnerColumn().value_or(0) != 0)
            throw std::runtime_error(
                "direct MXM LOAD_3D does not encode legacy IWColumn mode");
        auto instruction = MxmLoadIcuInstruction::Load3D(loop,
            as_size(op.getWeightBuffer(), "weight buffer"),
            mxm_buffer_mode(op.getWeightBufferMode().value_or("fixed")),
            as_size(op.getWeightColumn(), "weight column"),
            {static_cast<int64_t>(
                 op.getRepeatWeightColumnStride().value_or(0)),
                static_cast<int64_t>(
                    op.getWaveWeightColumnStride().value_or(0)),
                static_cast<int64_t>(
                    op.getGroupWeightColumnStride().value_or(0))},
            as_size(op.getWeightStreamBase().value_or(0),
                "weight stream base"),
            mxm_weight_input_mode(
                op.getWeightInputMode().value_or("direct16")));
        require_materialized(command::materializeMxmLoad3DCommand(builder,
            op.getLoc(), queue, instruction, &error), "MXM load", error);
        return;
    }
    const auto destination = mxm_destination(op.getAccumulatorDestination());
    const auto outputFormat = mxm_accumulator_format(
        op.getAccumulatorOutputFormat().value_or("fp32"));
    if (op.getOpcode() == "accumulator_read") {
        auto instruction = MxmComputeIcuInstruction::AccumulatorRead3D(loop,
            as_size(op.getOutputStreamBase(), "MXM output stream"),
            as_size(op.getAccumulatorAddress(), "accumulator address"),
            {static_cast<int64_t>(
                 op.getRepeatAccumulatorAddressStride().value_or(0)),
                static_cast<int64_t>(
                    op.getWaveAccumulatorAddressStride().value_or(0)),
                static_cast<int64_t>(
                    op.getGroupAccumulatorAddressStride().value_or(0))},
            op.getAccumulatorClear(), outputFormat, destination);
        require_materialized(command::materializeMxmCompute3DCommand(builder,
            op.getLoc(), queue, instruction, &error), "MXM accumulator read",
            error);
        return;
    }
    if (op.getOpcode() != "compute")
        throw std::runtime_error("unsupported direct MXM opcode");
    const MxmComputeIcuMode mode {destination, op.getAccumulatorClear(),
        outputFormat};
    const MxmComputeIcuMode terminalMode {
        mxm_destination(op.getTerminalAccumulatorDestination().value_or(
            op.getAccumulatorDestination())),
        op.getTerminalAccumulatorClear().value_or(op.getAccumulatorClear()),
        mxm_accumulator_format(
            op.getTerminalAccumulatorOutputFormat().value_or(
                op.getAccumulatorOutputFormat().value_or("fp32")))};
    auto instruction = MxmComputeIcuInstruction::Compute3D(loop,
        as_size(op.getWeightBuffer(), "weight buffer"),
        mxm_buffer_mode(op.getWeightBufferMode().value_or("fixed")),
        as_size(op.getActivationStreamBase(), "activation stream"),
        as_size(op.getOutputStreamBase(), "output stream"),
        as_size(op.getAccumulatorAddress(), "accumulator address"),
        {static_cast<int64_t>(
             op.getRepeatAccumulatorAddressStride().value_or(0)),
            static_cast<int64_t>(
                op.getWaveAccumulatorAddressStride().value_or(0)),
            static_cast<int64_t>(
                op.getGroupAccumulatorAddressStride().value_or(0))},
        as_size(op.getAccumulatorRowStride(), "accumulator row stride"),
        mxm_data_format(op.getDataFormat().value_or("fp16")), mode,
        static_cast<std::size_t>(op.getTerminalDimension().value_or(
            MxmComputeIcuInstruction::kNoTerminalDimension)), terminalMode);
    require_materialized(command::materializeMxmCompute3DCommand(builder,
        op.getLoc(), queue, instruction, &error), "MXM compute", error);
}

void create_mxm_dequant_command(
    mlir::OpBuilder& builder, schedule::MxmDequantOp op)
{
    auto instruction = MxmDequantIcuInstruction::Dequant3D(
        loop_3d(op.getCycle(), op.getRepeatCount(), op.getRepeatInterval(),
            op.getWaveCount().value_or(1),
            op.getWaveInterval().value_or(1),
            op.getGroupCount().value_or(1),
            op.getGroupInterval().value_or(1)),
        MxmDequantInstruction::Scale(
            static_cast<float>(op.getScaleAttr().getValueAsDouble())));
    std::string error;
    require_materialized(command::materializeMxmDequant3DCommand(builder,
        op.getLoc(), as_size(op.getUnitId(), "MXM dequant queue"),
        instruction, op.getScaleBinding().value_or(-1), &error),
        "MXM dequant", error);
}

class ScheduleToCommandPass final
    : public mlir::PassWrapper<ScheduleToCommandPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleToCommandPass)

    llvm::StringRef getArgument() const final { return "ftlpu-schedule-to-command"; }
    llvm::StringRef getDescription() const final
    {
        return "Lowers cycle-accurate Schedule IR to ICU queue Command IR";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        bool has_commands = false;
        function.walk([&](command::MemOp) { has_commands = true; });
        function.walk([&](command::MemBundleOp) {
            has_commands = true;
        });
        function.walk([&](command::MxmOp) { has_commands = true; });
        function.walk([&](command::MxmDequantOp) {
            has_commands = true;
        });
        function.walk([&](command::Mem3DOp) { has_commands = true; });
        function.walk([&](command::MemWriteRead2DOp) { has_commands = true; });
        function.walk([&](command::MxmLoad3DOp) {
            has_commands = true;
        });
        function.walk([&](command::MxmDequant3DOp) {
            has_commands = true;
        });
        function.walk([&](command::MxmCompute3DOp) {
            has_commands = true;
        });
        function.walk([&](command::VxmOp) { has_commands = true; });
        function.walk([&](command::SxmOp) { has_commands = true; });
        function.walk([&](command::VxmRun2DOp) { has_commands = true; });
        function.walk([&](command::SxmRun2DOp) { has_commands = true; });
        if (has_commands) {
            function.emitError("Command IR has already been generated");
            signalPassFailure();
            return;
        }

        llvm::SmallVector<schedule::MemReadOp> reads;
        llvm::SmallVector<schedule::MxmLoadOp> loads;
        llvm::SmallVector<schedule::MxmComputeOp> computes;
        llvm::SmallVector<schedule::MxmAccumulatorReadOp> accumulator_reads;
        llvm::SmallVector<schedule::VxmOp> vxms;
        llvm::SmallVector<schedule::SxmOp> sxms;
        llvm::SmallVector<schedule::MemTransferOp> mem_transfers;
        llvm::SmallVector<schedule::MemWriteRead2DOp> mem_write_reads;
        llvm::SmallVector<schedule::MxmIssueOp> mxm_issues;
        llvm::SmallVector<schedule::MxmDequantOp> mxm_dequants;
        llvm::SmallVector<schedule::BindingOp> bindings;
        llvm::SmallVector<schedule::TimelineOp> timelines;
        llvm::SmallVector<schedule::MemWriteOp> writes;
        llvm::SmallVector<schedule::MxmAccumulateOp> accumulates;
        function.walk([&](schedule::MemReadOp op) {
            reads.push_back(op);
        });
        function.walk([&](schedule::MxmLoadOp op) {
            loads.push_back(op);
        });
        function.walk([&](schedule::MxmComputeOp op) {
            computes.push_back(op);
        });
        function.walk([&](schedule::MxmAccumulatorReadOp op) {
            accumulator_reads.push_back(op);
        });
        function.walk([&](schedule::VxmOp op) {
            vxms.push_back(op);
        });
        function.walk([&](schedule::SxmOp op) {
            sxms.push_back(op);
        });
        function.walk([&](schedule::MemTransferOp op) {
            mem_transfers.push_back(op);
        });
        function.walk([&](schedule::MemWriteRead2DOp op) {
            mem_write_reads.push_back(op);
        });
        function.walk([&](schedule::MxmIssueOp op) {
            mxm_issues.push_back(op);
        });
        function.walk([&](schedule::MxmDequantOp op) {
            mxm_dequants.push_back(op);
        });
        function.walk([&](schedule::BindingOp op) {
            bindings.push_back(op);
        });
        function.walk([&](schedule::TimelineOp op) {
            timelines.push_back(op);
        });
        function.walk([&](schedule::MemWriteOp op) {
            writes.push_back(op);
        });
        function.walk([&](schedule::MxmAccumulateOp op) {
            accumulates.push_back(op);
        });
        if (reads.empty() && loads.empty() && computes.empty()
            && accumulator_reads.empty() && vxms.empty() && sxms.empty()
            && mem_transfers.empty() && mxm_issues.empty()
            && mem_write_reads.empty()
            && mxm_dequants.empty()
            && bindings.empty() && timelines.empty() && writes.empty()
            && accumulates.empty()) {
            function.emitError("requires Schedule IR operations");
            signalPassFailure();
            return;
        }

        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToStart(&function.getBody().front());
        llvm::SmallDenseSet<unsigned> bound_inputs;
        int64_t outputReadyCycle = 0;
        for (schedule::MemWriteOp write : writes)
            outputReadyCycle = std::max(
                outputReadyCycle,
                static_cast<int64_t>(
                    write.getCycle()
                    + (write.getGroupCount().value_or(1) - 1)
                        * write.getGroupInterval().value_or(1)
                    + (write.getWaveCount().value_or(1) - 1)
                        * write.getWaveInterval().value_or(1)
                    + write.getDuration()));
        for (schedule::BindingOp binding : bindings) {
            if (binding.getAccess() == "input"
                && !bound_inputs.insert(binding.getIndex()).second)
                continue;
            auto type = llvm::cast<mlir::RankedTensorType>(
                binding.getValue().getType());
            create_binding(builder, binding.getLoc(), binding.getIndex(),
                binding.getAccess(), binding.getRole(),
                binding.getName().value_or(binding.getRole()),
                binding.getReadyCycle().value_or(
                    binding.getAccess() == "output"
                        ? outputReadyCycle : 0),
                type,
                binding.getBytes(), binding.getPlacement(),
                binding.getInitializer().value_or("none"),
                binding.getInitializerConfig().value_or(
                    builder.getDictionaryAttr({})));
        }
        for (schedule::TimelineOp timeline : timelines) {
            mlir::OperationState state(
                timeline.getLoc(), command::TimelineOp::getOperationName());
            state.addAttributes({
                builder.getNamedAttr("name", timeline.getNameAttr()),
                builder.getNamedAttr("start", timeline.getStartAttr()),
                builder.getNamedAttr("end", timeline.getEndAttr()),
            });
            builder.create(state);
        }
        struct WeightPageInterval {
            int64_t binding;
            int64_t page;
            int64_t bank;
            int64_t ready{std::numeric_limits<int64_t>::max()};
            int64_t release{0};
            mlir::Location location;
        };
        std::map<std::tuple<int64_t, int64_t, int64_t>,
            WeightPageInterval> weightPages;
        for (schedule::MemReadOp read : reads) {
            const auto page = read.getPlacement()
                .getAs<mlir::IntegerAttr>("weight_page");
            if (!page) continue;
            const auto binding = source_binding_index(read.getInput());
            if (!binding) {
                read.emitError(
                    "paged weight read must resolve to one input binding");
                signalPassFailure();
                return;
            }
            const int64_t bank = placement_integer_or(
                read.getPlacement(), "bank", 0);
            auto [position, inserted] = weightPages.try_emplace(
                std::tuple {*binding, page.getInt(), bank},
                WeightPageInterval {*binding, page.getInt(), bank,
                    std::numeric_limits<int64_t>::max(), 0,
                    read.getLoc()});
            auto& interval = position->second;
            interval.ready = std::min<int64_t>(
                interval.ready, read.getCycle());
            interval.release = std::max<int64_t>(interval.release,
                read.getCycle()
                    + (read.getGroupCount().value_or(1) - 1)
                        * read.getGroupInterval().value_or(1)
                    + (read.getWaveCount().value_or(1) - 1)
                        * read.getWaveInterval().value_or(1)
                    + read.getDuration());
        }
        for (schedule::MemTransferOp transfer : mem_transfers) {
            const auto page = transfer.getWeightPage();
            if (!page) continue;
            const auto binding = transfer.getAddressBinding();
            if (!binding) {
                transfer.emitError(
                    "paged weight transfer must resolve to one input binding");
                signalPassFailure();
                return;
            }
            const int64_t bindingIndex = static_cast<int64_t>(*binding);
            const int64_t pageIndex = static_cast<int64_t>(*page);
            const int64_t bank = transfer.getBank().value_or(0);
            auto [position, inserted] = weightPages.try_emplace(
                std::tuple {bindingIndex, pageIndex, bank},
                WeightPageInterval {bindingIndex, pageIndex, bank,
                    std::numeric_limits<int64_t>::max(), 0,
                    transfer.getLoc()});
            auto& interval = position->second;
            const int64_t endCycle = transfer.getCycle()
                + (transfer.getGroupCount().value_or(1) - 1)
                    * transfer.getGroupInterval().value_or(1)
                + (transfer.getWaveCount().value_or(1) - 1)
                    * transfer.getWaveInterval().value_or(1)
                + (transfer.getRepeatCount() - 1)
                    * transfer.getRepeatInterval()
                + 1;
            interval.ready = std::min<int64_t>(
                interval.ready, transfer.getCycle());
            interval.release = std::max<int64_t>(
                interval.release, endCycle);
        }
        builder.setInsertionPointToStart(&function.getBody().front());
        for (const auto& [key, interval] : weightPages) {
            (void)key;
            mlir::OperationState state(interval.location,
                command::WeightPageOp::getOperationName());
            state.addAttributes({
                builder.getNamedAttr("binding_index",
                    builder.getI64IntegerAttr(interval.binding)),
                builder.getNamedAttr("page_index",
                    builder.getI64IntegerAttr(interval.page)),
                builder.getNamedAttr("bank",
                    builder.getI64IntegerAttr(interval.bank)),
                builder.getNamedAttr("ready_cycle",
                    builder.getI64IntegerAttr(interval.ready)),
                builder.getNamedAttr("release_cycle",
                    builder.getI64IntegerAttr(interval.release)),
            });
            builder.create(state);
        }
        for (schedule::MemReadOp read : reads) {
            auto argument =
                llvm::dyn_cast<mlir::BlockArgument>(read.getInput());
            if (!argument
                || !bound_inputs.insert(
                    argument.getArgNumber()).second)
                continue;
            auto type = llvm::cast<mlir::RankedTensorType>(
                argument.getType());
            const int64_t bindingBytes = type.getNumElements()
                * element_type_bytes(type.getElementType());
            mlir::DictionaryAttr bindingPlacement =
                read.getPlacement().getAs<mlir::DictionaryAttr>(
                    "binding_placement");
            if (!bindingPlacement)
                bindingPlacement = read.getPlacement();
            create_binding(builder, read.getLoc(),
                argument.getArgNumber(), "input",
                argument.getArgNumber() == 0
                    ? "activation"
                    : "weight",
                "arg" + std::to_string(argument.getArgNumber()), 0,
                type, bindingBytes, bindingPlacement);
        }
        llvm::SmallDenseSet<mlir::Value> returnedValues;
        function.walk([&](mlir::func::ReturnOp op) {
            for (mlir::Value value : op.getOperands())
                returnedValues.insert(value);
        });
        int64_t outputIndex = 0;
        for (schedule::MemWriteOp write : writes) {
            if (!returnedValues.contains(write.getOutput()))
                continue;
            auto type = llvm::cast<mlir::RankedTensorType>(
                write.getOutput().getType());
            mlir::DictionaryAttr explicitBinding =
                write.getPlacement().getAs<mlir::DictionaryAttr>(
                    "binding_placement");
            mlir::NamedAttrList placement(explicitBinding
                    ? explicitBinding
                    : write.getPlacement());
            if (auto slices =
                    write.getPlacement().getAs<mlir::ArrayAttr>(
                        "binding_slices"))
                placement.set("slices", slices);
            if (auto count =
                    write.getPlacement().getAs<mlir::IntegerAttr>(
                        "binding_instruction_count"))
                placement.set("instruction_count", count);
            placement.set(
                "base_row", builder.getI64IntegerAttr(0));
            create_binding(builder, write.getLoc(), outputIndex++,
                "output", "result", "result",
                write.getCycle()
                    + (write.getGroupCount().value_or(1) - 1)
                        * write.getGroupInterval().value_or(1)
                    + (write.getWaveCount().value_or(1) - 1)
                        * write.getWaveInterval().value_or(1)
                    + write.getDuration(), type,
                type.getNumElements()
                    * element_type_bytes(type.getElementType()),
                placement.getDictionary(&getContext()));
        }
        llvm::SmallVector<int64_t> loadRepeatCounts;
        loadRepeatCounts.reserve(loads.size());
        for (schedule::MxmLoadOp load : loads) {
            int64_t repeatCount = load.getDuration();
            if (auto read =
                    load.getInput()
                        .getDefiningOp<schedule::MemReadOp>())
                repeatCount = placement_integer(
                    read.getPlacement(), "instruction_count");
            loadRepeatCounts.push_back(repeatCount);
        }
        for (schedule::MemTransferOp transfer : mem_transfers) {
            builder.setInsertionPointAfter(transfer);
            create_mem_transfer_command(builder, transfer, target);
            transfer.erase();
        }
        for (schedule::MemWriteRead2DOp transfer : mem_write_reads) {
            builder.setInsertionPointAfter(transfer);
            create_mem_write_read_2d_command(builder, transfer, target);
            transfer.erase();
        }
        for (schedule::MxmIssueOp mxm : mxm_issues) {
            builder.setInsertionPointAfter(mxm);
            create_mxm_issue_command(builder, mxm);
            mxm.erase();
        }
        for (schedule::MxmDequantOp dequant : mxm_dequants) {
            builder.setInsertionPointAfter(dequant);
            create_mxm_dequant_command(builder, dequant);
            dequant.erase();
        }
        for (schedule::VxmOp vxm : vxms) {
            builder.setInsertionPointAfter(vxm);
            create_vxm_command(builder, vxm);
        }
        for (schedule::SxmOp sxm : sxms) {
            builder.setInsertionPointAfter(sxm);
            create_sxm_command(builder, sxm);
        }
        for (schedule::MemReadOp read : reads) {
            const auto slices = placement_slices(read.getPlacement());
            const int64_t base_row = placement_integer(read.getPlacement(), "base_row");
            const int64_t count = placement_integer(read.getPlacement(), "instruction_count");
            const int64_t stride = placement_integer(read.getPlacement(), "address_stride");
            const int64_t bank = placement_integer_or(
                read.getPlacement(), "bank", 0);
            auto hemisphere = placement_hemisphere(read.getPlacement(), read.getAddress());
            if (!hemisphere || slices.empty()
                || (slices.size() != 1
                    && static_cast<int64_t>(slices.size()) != read.getStreamCount())) {
                read.emitError("MEM read placement does not match its producer streams");
                signalPassFailure();
                return;
            }
            const bool west_hemisphere = hemisphere.getValue() == "west";
            const bool west_stream = read.getDirection() == "west";
            const target::StreamDirection direction = west_stream
                ? target::StreamDirection::West
                : target::StreamDirection::East;
            const bool weightRead = read.getRole().starts_with("weight");
            const target::StreamEndpoint destination =
                weightRead
                ? target::StreamEndpoint::MxmWeight
                : read.getRole() == "activation"
                ? target::StreamEndpoint::MxmActivation
                : target::StreamEndpoint::VxmInput;
            int64_t max_latency = 0;
            for (int64_t slice : slices) {
                auto latency = target.transport_latency(
                    target::StreamEndpoint::Mem, destination,
                    direction, slice);
                if (!latency) {
                    read.emitError(
                        "target does not support the scheduled MEM route");
                    signalPassFailure();
                    return;
                }
                max_latency = std::max(max_latency, *latency);
            }
            const int64_t command_base = stride < 0
                ? base_row - (count - 1) * stride : base_row;
            const int64_t addressBinding =
                source_binding_index(read.getInput()).value_or(-1);
            if (weightRead && addressBinding < 0) {
                read.emitError(
                    "weight MEM read must resolve to one input binding");
                signalPassFailure();
                return;
            }
            builder.setInsertionPointAfter(read);
            for (size_t index = 0; index < slices.size(); ++index) {
                const int64_t latency = *target.transport_latency(
                    target::StreamEndpoint::Mem, destination,
                    direction, slices[index]);
                const int64_t cycle = static_cast<int64_t>(read.getCycle())
                    + max_latency - latency;
                const int64_t queue = mem_queue(target,
                    west_hemisphere ? 1 : 0, slices[index], bank);
                const int64_t packedStream = (west_stream ? 32 : 0)
                    + static_cast<int64_t>(read.getStreamBase())
                    + static_cast<int64_t>(index);
                const auto loop = loop_3d(cycle, count, 1,
                    read.getWaveCount().value_or(1),
                    read.getWaveInterval().value_or(1),
                    read.getGroupCount().value_or(1),
                    read.getGroupInterval().value_or(1));
                const auto instruction = MemIcuInstruction::Read3D(loop,
                    mem_address_3d(read.getOperation(),
                        as_size(command_base, "MEM read address"), stride,
                        static_cast<int64_t>(
                            read.getWaveAddressStride().value_or(0)),
                        static_cast<int64_t>(
                            read.getGroupAddressStride().value_or(0))),
                    StreamId::from_packed(as_size(packedStream,
                        "MEM read stream")));
                std::string error;
                require_materialized(command::materializeMem3DCommand(
                    builder, read.getLoc(), as_size(queue, "MEM read queue"),
                    instruction, addressBinding,
                    addressBinding >= 0 ? "input" : "", &error),
                    "MEM read", error);
            }
        }

        for (std::size_t loadIndex = 0;
             loadIndex < loads.size(); ++loadIndex) {
            schedule::MxmLoadOp load = loads[loadIndex];
            builder.setInsertionPointAfter(load);
            const int64_t repeat_count =
                loadRepeatCounts[loadIndex];
            const int64_t columnsPerWave =
                target.throughput().tile_rows;
            const int64_t fullWaves = repeat_count / columnsPerWave;
            const int64_t tailColumns = repeat_count % columnsPerWave;
            const auto materializeLoadDomain = [&](int64_t startOffset,
                                                   int64_t innerCount,
                                                   int64_t waveCount) {
                // The compiler emits one rectangular full-wave domain and,
                // when needed, one tail domain instead of enumerating weight
                // columns. West-to-east pulses reach the column controls in
                // reverse physical order within each wave.
                const auto loop = loop_3d(load.getCycle() + startOffset,
                    innerCount, 1, waveCount, columnsPerWave,
                    load.getGroupCount().value_or(1),
                    load.getGroupInterval().value_or(1));
                auto instruction = MxmLoadIcuInstruction::Load3D(loop,
                    as_size(load.getWeightBuffer(), "weight buffer"),
                    mxm_buffer_mode(
                        load.getWeightBufferMode().value_or("fixed")),
                    as_size(columnsPerWave - 1, "weight column"),
                    {-1, 0, 0},
                    as_size(load.getStreamBase(), "weight stream base"),
                    mxm_weight_input_mode(
                        load.getWeightInputMode().value_or("direct16")));
                std::string error;
                require_materialized(
                    command::materializeMxmLoad3DCommand(builder,
                        load.getLoc(), as_size(load.getUnitId(), "MXM queue"),
                        instruction, &error), "MXM load", error);
            };
            if (fullWaves > 0)
                materializeLoadDomain(0, columnsPerWave, fullWaves);
            if (tailColumns > 0)
                materializeLoadDomain(
                    fullWaves * columnsPerWave, tailColumns, 1);
        }
        for (schedule::MxmComputeOp compute : computes) {
            schedule::MxmAccumulateOp accumulator;
            for (mlir::Operation* user : compute.getResult().getUsers()) {
                if (auto candidate = llvm::dyn_cast<schedule::MxmAccumulateOp>(user)) {
                    if (accumulator) {
                        compute.emitError("must have exactly one accumulator consumer");
                        signalPassFailure();
                        return;
                    }
                    accumulator = candidate;
                }
            }
            if (!accumulator) {
                for (mlir::Operation* operation = compute->getNextNode();
                     operation; operation = operation->getNextNode()) {
                    auto candidate =
                        llvm::dyn_cast<schedule::MxmAccumulateOp>(
                            operation);
                    if (!candidate
                        || candidate.getStreamBase()
                            != compute.getOutputStreamBase()
                        || candidate.getUnitId() != compute.getUnitId())
                        continue;
                    accumulator = candidate;
                    break;
                }
            }
            if (!accumulator) {
                compute.emitError("requires an accumulator consumer");
                signalPassFailure();
                return;
            }
            builder.setInsertionPointAfter(compute);
            const auto destination = accumulator.getDestination() == "local"
                ? MxmAccumulatorDestination::Sram
                : MxmAccumulatorDestination::Stream;
            const auto outputFormat = mxm_accumulator_format(
                accumulator.getAccumulatorOutputFormat().value_or("fp32"));
            const MxmComputeIcuMode mode {destination, true, outputFormat};
            const MxmComputeIcuMode terminalMode {
                mxm_destination(compute
                    .getTerminalAccumulatorDestination()
                    .value_or(accumulator.getDestination() == "local"
                            ? "sram" : "stream")),
                compute.getTerminalAccumulatorClear().value_or(true),
                mxm_accumulator_format(compute
                    .getTerminalAccumulatorOutputFormat()
                    .value_or(accumulator
                        .getAccumulatorOutputFormat().value_or("fp32")))};
            const int64_t groupCount = compute.getGroupCount().value_or(
                accumulator.getGroupCount().value_or(1));
            const int64_t groupInterval = compute.getGroupInterval().value_or(
                accumulator.getGroupInterval().value_or(1));
            const int64_t groupAccumulatorStride =
                compute.getGroupAccumulatorAddressStride().value_or(
                    accumulator
                        .getGroupAccumulatorAddressStride().value_or(0));
            auto instruction = MxmComputeIcuInstruction::Compute3D(
                loop_3d(compute.getCycle(), compute.getDuration(), 1,
                    compute.getWaveCount().value_or(1),
                    compute.getWaveInterval().value_or(1), groupCount,
                    groupInterval),
                as_size(compute.getWeightBuffer(), "weight buffer"),
                mxm_buffer_mode(
                    compute.getWeightBufferMode().value_or("fixed")),
                as_size(compute.getActivationStreamBase(),
                    "activation stream"),
                as_size(compute.getOutputStreamBase(), "output stream"),
                as_size(accumulator.getAccumulatorAddress(),
                    "accumulator address"),
                {0, static_cast<int64_t>(compute
                        .getWaveAccumulatorAddressStride().value_or(0)),
                    groupAccumulatorStride},
                as_size(accumulator.getAccumulatorStride(),
                    "accumulator row stride"),
                mxm_data_format(compute.getDataFormat().value_or("fp16")),
                mode, static_cast<std::size_t>(
                    compute.getTerminalDimension().value_or(
                        MxmComputeIcuInstruction::kNoTerminalDimension)),
                terminalMode);
            std::string error;
            require_materialized(command::materializeMxmCompute3DCommand(
                builder, compute.getLoc(),
                as_size(compute.getUnitId(), "MXM queue"), instruction,
                &error), "MXM compute", error);
        }
        for (schedule::MxmAccumulatorReadOp read : accumulator_reads) {
            builder.setInsertionPointAfter(read);
            auto instruction = MxmComputeIcuInstruction::AccumulatorRead3D(
                loop_3d(read.getCycle(), read.getRepeatCount().value_or(1),
                    read.getRepeatInterval().value_or(1),
                    read.getWaveCount().value_or(1),
                    read.getWaveInterval().value_or(1),
                    read.getGroupCount().value_or(1),
                    read.getGroupInterval().value_or(1)),
                as_size(read.getOutputStreamBase(), "output stream"),
                as_size(read.getAccumulatorAddress(), "accumulator address"),
                {static_cast<int64_t>(
                     read.getRepeatAccumulatorAddressStride().value_or(0)),
                    static_cast<int64_t>(
                        read.getWaveAccumulatorAddressStride().value_or(0)),
                    static_cast<int64_t>(
                        read.getGroupAccumulatorAddressStride().value_or(0))},
                read.getClear(),
                read.getDataFormat().value_or("fp32") == "bf16"
                    ? MxmAccumulatorOutputFormat::BFloat16
                    : MxmAccumulatorOutputFormat::Float32,
                MxmAccumulatorDestination::Stream);
            std::string error;
            require_materialized(command::materializeMxmCompute3DCommand(
                builder, read.getLoc(),
                as_size(read.getUnitId(), "MXM queue"), instruction,
                &error), "MXM accumulator read", error);
        }
        for (schedule::MemWriteOp write : writes) {
            const auto slices = placement_slices(write.getPlacement());
            const int64_t base_row = placement_integer(write.getPlacement(), "base_row");
            const int64_t count = placement_integer(write.getPlacement(), "instruction_count");
            const int64_t stride = placement_integer(write.getPlacement(), "address_stride");
            const int64_t bank = placement_integer_or(
                write.getPlacement(), "bank", 0);
            auto hemisphere = placement_hemisphere(write.getPlacement(), write.getAddress());
            if (!hemisphere
                || static_cast<int64_t>(slices.size()) != write.getStreamCount()) {
                write.emitError("result write placement does not match its producer streams");
                signalPassFailure();
                return;
            }
            const bool west_hemisphere = hemisphere.getValue() == "west";
            const bool west_stream = write.getDirection() == "west";
            builder.setInsertionPointAfter(write);
            for (size_t index = 0; index < slices.size(); ++index) {
                const int64_t queue = mem_queue(target,
                    west_hemisphere ? 1 : 0, slices[index], bank);
                const int64_t packedStream = (west_stream ? 32 : 0)
                    + write.getStreamBase()
                    + static_cast<int64_t>(index);
                const auto instruction = MemIcuInstruction::Write3D(
                    loop_3d(write.getCycle(), count, 1,
                        write.getWaveCount().value_or(1),
                        write.getWaveInterval().value_or(1),
                        write.getGroupCount().value_or(1),
                        write.getGroupInterval().value_or(1)),
                    mem_address_3d(write.getOperation(),
                        as_size(base_row, "MEM write address"), stride,
                        static_cast<int64_t>(
                            write.getWaveAddressStride().value_or(0)),
                        static_cast<int64_t>(
                            write.getGroupAddressStride().value_or(0))),
                    StreamId::from_packed(as_size(packedStream,
                        "MEM write stream")));
                std::string error;
                require_materialized(command::materializeMem3DCommand(
                    builder, write.getLoc(), as_size(queue, "MEM write queue"),
                    instruction, -1, {}, &error), "MEM write", error);
            }
        }

        function.walk([](mlir::func::ReturnOp op) {
            op->setOperands(mlir::ValueRange {});
        });
        function.setType(mlir::FunctionType::get(
            &getContext(), function.getArgumentTypes(), mlir::TypeRange {}));
        // Preserve block order while collecting the complete lowered SSA
        // graph. Reverse deletion then removes Schedule consumers before the
        // Stream route aliases they consume, and aliases before bindings.
        llvm::SmallVector<mlir::Operation*> lowered_ops;
        function.walk([&](mlir::Operation* op) {
            if (llvm::isa<schedule::MemReadOp, schedule::MxmLoadOp,
                    schedule::MxmComputeOp,
                    schedule::MxmAccumulatorReadOp, schedule::VxmOp,
                    schedule::SxmOp,
                    schedule::MemTransferOp, schedule::MxmIssueOp,
                    schedule::MxmDequantOp,
                    schedule::MxmAccumulateOp, schedule::MemWriteOp,
                    schedule::BindingOp, schedule::TimelineOp,
                    stream::RouteOp, stream::DequantizeOp>(op))
                lowered_ops.push_back(op);
        });
        for (auto it = lowered_ops.rbegin(); it != lowered_ops.rend(); ++it) {
            if (!(*it)->use_empty()) {
                (*it)->emitError(
                    "lowered operation still has live users after Command lowering");
                signalPassFailure();
                return;
            }
            (*it)->erase();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_schedule_to_command_pass()
{
    return std::make_unique<ScheduleToCommandPass>();
}

} // namespace ftlpu::compiler
