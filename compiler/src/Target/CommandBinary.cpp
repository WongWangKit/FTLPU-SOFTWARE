// Keep this translation unit rebuilt when BinaryProgram or binding ABI evolves.
#include "ftlpu/compiler/Target/command_binary.hpp"

#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "ftlpu/core/instruction_codec.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace ftlpu::compiler::target {
namespace {

// QueueCommand has a variable SXM payload; keep this translation unit rebuilt
// with the runtime queue ABI rather than relying on a transitive header edge.

using software::runtime::BinaryBinding;
using software::runtime::BinaryAddressRelocation;
using software::runtime::BinaryMemoryFloor;
using software::runtime::BinaryTimeline;
using software::runtime::BindingAccess;
using software::runtime::BindingElementType;
using software::runtime::BindingLayout;
using software::runtime::BinaryScaleRelocation;
using software::runtime::InstructionKind;
using software::runtime::QueueCommand;
using software::runtime::QueueKind;
using software::runtime::QueueProgram;
using software::runtime::VxmImmediateOperand;

struct CommandSequence {
    int64_t cycle{0};
    int64_t repeat_count{0};
    int64_t repeat_interval{0};
    int64_t address_stride{0};
    QueueCommand instruction;
    int64_t scale_binding{-1};
    int64_t address_binding{-1};
    int64_t write_address_binding{-1};
    bool is_loop{false};
    int64_t loop_window_size{0};
    int64_t outer_count{1};
    int64_t outer_interval{1};
    int64_t outer_stride{0};
    IcuInductionTarget induction_target{IcuInductionTarget::None};
};

int64_t command_cycle(mlir::Operation* op)
{
    return llvm::cast<mlir::IntegerAttr>(op->getAttrDictionary().get("cycle")).getInt();
}

int64_t command_integer(mlir::Operation* op, llvm::StringRef name)
{
    return llvm::cast<mlir::IntegerAttr>(op->getAttrDictionary().get(name)).getInt();
}

using QueueKey = std::pair<QueueKind, int64_t>;
using QueueMap = std::map<QueueKey, std::vector<CommandSequence>>;

QueueKind parse_loop_queue_kind(llvm::StringRef kind)
{
    if (kind == "mem") return QueueKind::Mem;
    if (kind == "mxm_load") return QueueKind::MxmLoad;
    if (kind == "mxm_compute") return QueueKind::MxmCompute;
    if (kind == "mxm_dequant") return QueueKind::MxmDequant;
    if (kind == "vxm") return QueueKind::Vxm;
    if (kind == "sxm_transpose") return QueueKind::SxmTranspose;
    if (kind == "sxm_permute") return QueueKind::SxmPermute;
    throw std::runtime_error("unsupported Command IR Loop queue kind");
}

void collect_loop(command::LoopOp op, QueueMap& queues)
{
    queues[{parse_loop_queue_kind(op.getQueueKind()), op.getQueue()}]
        .push_back(CommandSequence {
            static_cast<int64_t>(op.getCycle()),
            static_cast<int64_t>(op.getCount()),
            static_cast<int64_t>(op.getInterval()),
            static_cast<int64_t>(op.getAddressStride()),
            {}, -1, -1, -1, true,
            static_cast<int64_t>(op.getWindowSize()),
        });
}

QueueCommand mem_instruction_command(isa::EncodedMemInstruction encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    };
}

QueueCommand mxm_instruction_command(isa::EncodedMxmInstruction encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    };
}

QueueCommand mxm_dequant_instruction_command(
    isa::EncodedMxmDequantInstruction encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction),
        InstructionKind::MxmDequant,
        1,
        {static_cast<std::uint32_t>(encoded), 0, 0, 0},
    };
}

QueueCommand vxm_instruction_command(const isa::EncodedVxmInstruction& encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Vxm,
        3,
        {
            static_cast<std::uint32_t>(encoded.control),
            static_cast<std::uint32_t>(encoded.control >> 32),
            encoded.immediate_bits,
            0,
        },
    };
}

QueueCommand sxm_instruction_command(const SxmInstruction& instruction)
{
    QueueCommand command {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Sxm, 4, {},
    };
    command.words[0] = static_cast<std::uint32_t>(instruction.opcode);
    command.words[1] = static_cast<std::uint32_t>(instruction.shift_source);
    command.words[2] = static_cast<std::uint32_t>(instruction.shift_distance);
    command.words[3] =
        (instruction.output_row == SxmInstruction::kAllOutputRows ? 0xffu
            : static_cast<std::uint32_t>(instruction.output_row))
        | ((instruction.input_row == SxmInstruction::kAllInputRows ? 0xffu
            : static_cast<std::uint32_t>(instruction.input_row)) << 8)
        | ((instruction.output_tile == SxmInstruction::kAllOutputTiles ? 0xffu
            : static_cast<std::uint32_t>(instruction.output_tile)) << 16);
    command.extension_words.push_back(static_cast<std::uint32_t>(instruction.src_streams.size()));
    command.extension_words.push_back(static_cast<std::uint32_t>(instruction.dst_streams.size()));
    for (const auto stream : instruction.src_streams)
        command.extension_words.push_back(static_cast<std::uint32_t>(stream.stream));
    for (const auto stream : instruction.dst_streams)
        command.extension_words.push_back(static_cast<std::uint32_t>(stream.stream));
    for (const auto lane : instruction.permute_map)
        command.extension_words.push_back(lane == SxmInstruction::kZeroFill
            ? UINT32_MAX : static_cast<std::uint32_t>(lane));
    return command;
}

QueueCommand control_command(isa::EncodedIcuCommand command)
{
    return QueueCommand {command, InstructionKind::None, 0, {}};
}

QueueCommand repeat_2d_command(const IcuRepeat2D& repeat)
{
    const auto encoded = isa::encode_icu_repeat_2d(repeat);
    return QueueCommand {
        encoded.words[0], InstructionKind::None, 3,
        {encoded.words[0], encoded.words[1], encoded.words[2], 0},
    };
}

std::vector<BinaryMemoryFloor> static_memory_floors(
    mlir::ModuleOp module, int64_t slices_per_hemisphere,
    int64_t banks_per_slice, int64_t rows_per_bank)
{
    std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> floors;
    module.walk([&](command::MemOp op) {
        if (op->hasAttr("address_binding")) return;
        const auto integer = [&](llvm::StringRef name, int64_t fallback) {
            if (const auto attr =
                    op->getAttrOfType<mlir::IntegerAttr>(name))
                return attr.getInt();
            return fallback;
        };
        const int64_t base = op.getAddress();
        const int64_t repeat_count = op.getRepeatCount();
        const int64_t repeat_stride = op.getAddressStride();
        const int64_t wave_count = integer("wave_count", 1);
        const int64_t wave_stride =
            integer("wave_address_stride", 0);
        const int64_t queue = op.getQueue();
        const int64_t hemisphere =
            queue / (slices_per_hemisphere * banks_per_slice);
        const int64_t localQueue =
            queue % (slices_per_hemisphere * banks_per_slice);
        const int64_t slice = localQueue / banks_per_slice;
        const int64_t bank = localQueue % banks_per_slice;
        if (queue < 0 || hemisphere >= 2)
            throw std::runtime_error(
                "Command IR MEM queue is outside the target");
        for (int64_t repeat : {int64_t {0}, repeat_count - 1})
            for (int64_t wave : {int64_t {0}, wave_count - 1}) {
                const int64_t address = base
                    + repeat * repeat_stride
                    + wave * wave_stride;
                if (address < 0 || address >= rows_per_bank)
                    throw std::runtime_error(
                        "Command IR MEM scratch address is outside the target");
                auto& floor = floors[{hemisphere, slice, bank}];
                floor = std::max(floor, address + 1);
            }
    });
    std::vector<BinaryMemoryFloor> result;
    result.reserve(floors.size());
    for (const auto& entry : floors) {
        const auto [hemisphere, slice, bank] = entry.first;
        const int64_t floor = entry.second;
        result.push_back(BinaryMemoryFloor {
            static_cast<std::uint16_t>(hemisphere),
            static_cast<std::uint16_t>(slice),
            static_cast<std::uint32_t>(floor),
            static_cast<std::uint16_t>(bank),
        });
    }
    return result;
}

BindingLayout parse_layout(llvm::StringRef value)
{
    if (value == "vector") return BindingLayout::Vector;
    if (value == "mxm_weight_striped") return BindingLayout::MxmWeightStriped;
    if (value == "int32_byte_planar") return BindingLayout::Int32BytePlanar;
    if (value == "fp16_byte_planar") return BindingLayout::Fp16BytePlanar;
    if (value == "fp16_mxm_activation_planar") return BindingLayout::Fp16MxmActivationPlanar;
    if (value == "w8a16_mxm_weight_striped") return BindingLayout::W8A16MxmWeightStriped;
    if (value == "w8a16_mxm_weight_replicated")
        return BindingLayout::W8A16MxmWeightReplicated;
    if (value == "w8a16_mxm_weight_wave_striped")
        return BindingLayout::W8A16MxmWeightWaveStriped;
    if (value == "w8a16_block8_weight_wave_striped")
        return BindingLayout::W8A16Block8WeightWaveStriped;
    if (value == "w8a16_attention_weight_striped")
        return BindingLayout::W8A16AttentionWeightStriped;
    if (value == "fp16_pair_planar") return BindingLayout::Fp16PairPlanar;
    if (value == "fp32_causal_mask_tile")
        return BindingLayout::Fp32CausalMaskTile;
    if (value == "fp16_sxm_distributed_16")
        return BindingLayout::Fp16SxmDistributed16;
    if (value == "fp16_vxm_distributed_16")
        return BindingLayout::Fp16VxmDistributed16;
    if (value == "fp16_vxm_row_parallel_8")
        return BindingLayout::Fp16VxmRowParallel8;
    if (value == "fp16_mxm_distributed_16")
        return BindingLayout::Fp16MxmDistributed16;
    if (value == "fp16_mxm_block8_distributed_16")
        return BindingLayout::Fp16MxmBlock8Distributed16;
    if (value == "fp16_rope_table")
        return BindingLayout::Fp16RopeTable;
    throw std::runtime_error("unsupported Command IR binding layout");
}

BinaryBinding translate_binding(command::BindingOp op)
{
    BinaryBinding binding;
    binding.index = static_cast<std::uint32_t>(op.getIndex());
    binding.role = op.getRole().str();
    binding.name =
        op->getAttrOfType<mlir::StringAttr>("name").getValue().str();
    binding.ready_cycle = static_cast<std::uint64_t>(
        op->getAttrOfType<mlir::IntegerAttr>("ready_cycle").getInt());
    binding.access = op.getAccess() == "input" ? BindingAccess::Input
        : op.getAccess() == "output" ? BindingAccess::Output
        : BindingAccess::Internal;
    binding.element_type = op.getElementType() == "i8" ? BindingElementType::I8
        : op.getElementType() == "f16" ? BindingElementType::F16
        : op.getElementType() == "bf16" ? BindingElementType::BF16
        : op.getElementType() == "f32" ? BindingElementType::F32
        : BindingElementType::I32;
    binding.byte_size = static_cast<std::uint64_t>(op.getBytes());
    binding.layout = parse_layout(op.getPlacement().getAs<mlir::StringAttr>("kind").getValue());
    auto hemisphere = op.getPlacement().getAs<mlir::StringAttr>("hemisphere");
    binding.hemisphere_mask = !hemisphere || hemisphere.getValue() == "east" ? 1
        : hemisphere.getValue() == "west" ? 2 : 3;
    if (auto bank = op.getPlacement().getAs<mlir::IntegerAttr>("bank"))
        binding.bank = static_cast<std::uint16_t>(bank.getInt());
    binding.base_row = op.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt();
    binding.instruction_count = op.getPlacement().getAs<mlir::IntegerAttr>("instruction_count").getInt();
    binding.address_stride = op.getPlacement().getAs<mlir::IntegerAttr>("address_stride").getInt();
    const llvm::StringRef initializer = op.getInitializer();
    binding.initializer = initializer == "zero"
        ? software::runtime::BindingInitializer::Zero
        : initializer == "causal_mask"
        ? software::runtime::BindingInitializer::CausalMask
        : initializer == "rope_table"
        ? software::runtime::BindingInitializer::RopeTable
        : software::runtime::BindingInitializer::None;
    if (binding.initializer
        == software::runtime::BindingInitializer::RopeTable) {
        const auto config = op.getInitializerConfig();
        binding.rope_theta = static_cast<float>(
            config.getAs<mlir::FloatAttr>("theta").getValueAsDouble());
        binding.rope_head_dim = static_cast<std::uint32_t>(
            config.getAs<mlir::IntegerAttr>("head_dim").getInt());
    }
    for (mlir::Attribute dimension : op.getShape())
        binding.shape.push_back(static_cast<std::uint64_t>(
            llvm::cast<mlir::IntegerAttr>(dimension).getInt()));
    for (mlir::Attribute slice : op.getPlacement().getAs<mlir::ArrayAttr>("slices"))
        binding.slices.push_back(static_cast<std::uint16_t>(
            llvm::cast<mlir::IntegerAttr>(slice).getInt()));
    return binding;
}

void collect_mem(command::MemOp op, QueueMap& queues)
{
    const int64_t queue = command_integer(op, "queue");
    const int64_t waveCount =
        static_cast<int64_t>(op.getWaveCount().value_or(1));
    const int64_t waveInterval =
        static_cast<int64_t>(op.getWaveInterval().value_or(1));
    const int64_t waveAddressStride =
        static_cast<int64_t>(op.getWaveAddressStride().value_or(0));
    const int64_t address = static_cast<int64_t>(op.getAddress());
    const auto instruction = op.getOpcode() == "read"
            ? MemInstruction::Read(address, op.getPackedStream())
            : op.getOpcode() == "write_tap"
            ? MemInstruction::WriteTap(address, op.getPackedStream())
            : MemInstruction::Write(address, op.getPackedStream());
    queues[{QueueKind::Mem, queue}].push_back(CommandSequence {
            command_cycle(op),
            op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
            op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(),
            op->getAttrOfType<mlir::IntegerAttr>("address_stride").getInt(),
            mem_instruction_command(
                isa::encode_mem_instruction(instruction)),
            -1,
            op.getAddressBinding()
                ? static_cast<int64_t>(*op.getAddressBinding()) : -1,
            -1,
            false, 0, waveCount, waveInterval, waveAddressStride,
            IcuInductionTarget::MemAddress
        });
}

void collect_mxm(command::MxmOp op, QueueMap& queues)
{
    const bool is_load = op.getOpcode() == "iw";
    const bool is_accumulator_read = op.getOpcode() == "accumulator_read";
    const auto destination = op.getAccumulatorDestination() == "stream"
        ? MxmAccumulatorDestination::Stream : MxmAccumulatorDestination::Sram;
    const auto inputMode =
        op.getWeightInputMode().value_or("direct16")
            == "int8_dequant_bf16"
        ? MxmWeightInputMode::Int8DequantBf16
        : MxmWeightInputMode::Direct16;
    const auto computeMode =
        op.getComputeMode().value_or("vector") == "block8"
        ? MxmComputeMode::Block8 : MxmComputeMode::Vector;
    const auto kind = is_load ? QueueKind::MxmLoad : QueueKind::MxmCompute;
    const int64_t waveCount = op.getWaveCount().value_or(1);
    const int64_t waveInterval = op.getWaveInterval().value_or(1);
    const int64_t waveColumnStride =
        op.getWaveWeightColumnStride().value_or(0);
    const int64_t waveAccumulatorStride =
        op.getWaveAccumulatorAddressStride().value_or(0);
    const int64_t waveInductionStride = waveColumnStride != 0
        ? waveColumnStride : waveAccumulatorStride;
    const IcuInductionTarget waveInductionTarget = waveColumnStride != 0
        ? IcuInductionTarget::MxmWeightColumn
        : waveAccumulatorStride != 0
        ? IcuInductionTarget::MxmAccumulatorAddress
        : IcuInductionTarget::None;
    const int64_t groupCount = op.getGroupCount().value_or(1);
    const int64_t groupInterval = op.getGroupInterval().value_or(1);
    int64_t innerCount = op.getRepeatCount();
    int64_t innerInterval = op.getRepeatInterval();
    int64_t innerStride = 0;
    int64_t outerCount = 1;
    int64_t outerInterval = 1;
    int64_t outerStride = 0;
    IcuInductionTarget inductionTarget = IcuInductionTarget::None;
    if (groupCount > 1 && waveCount > 1) {
        if (innerCount != 1)
            throw std::runtime_error(
                "Command IR MXM requires more than two iteration dimensions");
        innerCount = waveCount;
        innerInterval = waveInterval;
        innerStride = waveInductionStride;
        outerCount = groupCount;
        outerInterval = groupInterval;
        inductionTarget = waveInductionTarget;
    } else if (groupCount > 1) {
        outerCount = groupCount;
        outerInterval = groupInterval;
    } else if (waveCount > 1) {
        outerCount = waveCount;
        outerInterval = waveInterval;
        outerStride = waveInductionStride;
        inductionTarget = waveInductionTarget;
    }
    const int64_t weightColumn = op.getWeightColumn();
        const auto instruction = is_load
            ? op.getWeightLoadMode().value_or("supercell") == "column"
                ? MxmControlInstruction::IWColumn(
                    op.getWeightBuffer(), weightColumn,
                    op.getWeightInnerColumn().value_or(0), inputMode)
                : MxmControlInstruction::IW(
                    op.getWeightBuffer(), weightColumn, inputMode)
            : is_accumulator_read
            ? MxmControlInstruction::AccumulatorRead(
                op.getAccumulatorAddress(), op.getOutputStreamBase(),
                op.getAccumulatorClear(), computeMode)
            : MxmControlInstruction::Compute(op.getWeightBuffer(),
                op.getActivationStreamBase(), op.getOutputStreamBase(),
                op.getAccumulatorAddress(), op.getAccumulatorRowStride(),
                destination,
                op.getDataFormat().value_or("fp16") == "bf16"
                    ? MxmDataFormat::BFloat16
                    : MxmDataFormat::Float16,
                computeMode, op.getAccumulatorClear(),
                op.getAccumulatorOutputFormat().value_or("fp32") == "bf16"
                    ? MxmAccumulatorOutputFormat::BFloat16
                    : MxmAccumulatorOutputFormat::Float32);
        queues[{kind, static_cast<int64_t>(op.getQueue())}]
            .push_back(CommandSequence {
                command_cycle(op), innerCount, innerInterval, innerStride,
                mxm_instruction_command(
                    isa::encode_mxm_instruction(instruction)),
                -1, -1, -1, false, 0,
                outerCount, outerInterval, outerStride, inductionTarget,
            });
}

void collect_mxm_dequant(
    command::MxmDequantOp op, QueueMap& queues)
{
    const auto instruction = MxmDequantInstruction::Scale(
        static_cast<float>(op.getScaleAttr().getValueAsDouble()));
    queues[{QueueKind::MxmDequant,
            static_cast<int64_t>(op.getQueue())}]
            .push_back(CommandSequence {
                command_cycle(op),
                static_cast<int64_t>(op.getRepeatCount()),
                static_cast<int64_t>(op.getRepeatInterval()),
                0,
                mxm_dequant_instruction_command(
                    isa::encode_mxm_dequant_instruction(instruction)),
                -1, -1, -1, false, 0,
                static_cast<int64_t>(op.getWaveCount().value_or(1)),
                static_cast<int64_t>(op.getWaveInterval().value_or(1)),
            });
}

VxmLaneOperation parse_vxm_operation(llvm::StringRef value)
{
    if (value == "pass" || value == "bypass" || value == "cast")
        return VxmAluOpcode::Bypass;
    if (value == "add") return VxmAluOpcode::Add;
    if (value == "subtract") return VxmAluOpcode::Subtract;
    if (value == "multiply") return VxmAluOpcode::Multiply;
    if (value == "negate") return VxmAluOpcode::Negate;
    if (value == "max") return VxmAluOpcode::Max;
    if (value == "exp") return VxmSpecialAluOpcode::Exp;
    if (value == "reciprocal" || value == "divide")
        return VxmSpecialAluOpcode::Reciprocal;
    if (value == "rsqrt") return VxmSpecialAluOpcode::Rsqrt;
    throw std::runtime_error(
        "Command IR VXM operation is not implemented by the current CModel");
}

VxmLaneOperand parse_vxm_operand(llvm::StringRef kind, int64_t index,
    float immediate, int64_t queue)
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
    if (kind == "stream_f16") return VxmLaneOperand::StreamFloat16();
    if (kind == "stream_bf16") return VxmLaneOperand::StreamBFloat16();
    if (kind == "immediate") return VxmLaneOperand::Imm(immediate);
    throw std::runtime_error(
        "legacy integer/FP32 VXM stream operands require BF16 legalization");
}

VxmCastTarget parse_vxm_cast_target(llvm::StringRef value)
{
    if (value == "fp32") return VxmCastTarget::Float32;
    if (value == "fp16") return VxmCastTarget::Float16;
    if (value == "bf16") return VxmCastTarget::BFloat16;
    if (value == "i8") return VxmCastTarget::Int8;
    throw std::runtime_error("unsupported Command IR VXM cast target");
}

void collect_vxm(command::VxmOp op, QueueMap& queues)
{
    const int64_t queue = op.getQueue();
    if (queue < 0 || queue >= InstructionControlUnit::kVxmQueues)
        throw std::runtime_error(
            "Command IR VXM queue exceeds the 8 compact control queues");
    const int64_t output_stream = op.getOutputStreamAttr().getInt();
    auto instruction = VxmLaneAluInstruction {};
    instruction.operation = parse_vxm_operation(op.getOpcode());
    instruction.lhs = parse_vxm_operand(op.getLhsKind(), op.getLhsIndex(),
        static_cast<float>(op.getLhsImmediateAttr().getValueAsDouble()), queue);
    instruction.rhs = parse_vxm_operand(op.getRhsKind(), op.getRhsIndex(),
        static_cast<float>(op.getRhsImmediateAttr().getValueAsDouble()), queue);
    instruction.output_type = parse_vxm_cast_target(op.getCastTarget());
    instruction.precision = VxmAluPrecision::Float32;
    instruction.repeat_count = static_cast<std::size_t>(op.getRepeatCount());
    instruction.accumulator_reset = op.getAccumulatorReset().value_or(false);
    instruction.accumulator_write = op.getAccumulatorWrite().value_or(false);
    instruction.accumulator_emit = op.getAccumulatorEmit().value_or(true);
    instruction.local_scalar_write = op.getLocalScalarWrite().value_or(false);
    if (output_stream >= 0)
        instruction.output_stream = static_cast<std::size_t>(output_stream);
    const auto depth = static_cast<VxmChainDepth>(
        op->getAttrOfType<mlir::IntegerAttr>("chain_depth")
            ? op->getAttrOfType<mlir::IntegerAttr>("chain_depth").getInt()
            : 8);
    queues[{QueueKind::Vxm, queue}].push_back(CommandSequence {
        command_cycle(op),
        1, 1, 0,
        vxm_instruction_command(
            isa::encode_vxm_instruction(queue, depth, instruction)),
        op.getScaleBinding()
            ? static_cast<int64_t>(*op.getScaleBinding()) : -1,
    });
}

void collect_sxm(command::SxmOp op, QueueMap& queues)
{
    SxmInstruction instruction {};
    instruction.opcode = op.getOpcode() == "transpose"
        ? SxmOpcode::Transpose : SxmOpcode::Permute;
    if (op.getOutputRow())
        instruction.output_row = static_cast<std::size_t>(*op.getOutputRow());
    if (op.getInputRow())
        instruction.input_row = static_cast<std::size_t>(*op.getInputRow());
    if (op.getOutputTile())
        instruction.output_tile = static_cast<std::size_t>(*op.getOutputTile());
    for (mlir::Attribute stream : op.getSourceStreams())
        instruction.src_streams.push_back(SxmStreamId {static_cast<std::size_t>(
            llvm::cast<mlir::IntegerAttr>(stream).getInt())});
    for (mlir::Attribute stream : op.getDestinationStreams())
        instruction.dst_streams.push_back(SxmStreamId {static_cast<std::size_t>(
            llvm::cast<mlir::IntegerAttr>(stream).getInt())});
    for (std::size_t lane = 0; lane < instruction.permute_map.size(); ++lane) {
        const auto value = llvm::cast<mlir::IntegerAttr>(op.getPermuteMap()[lane]).getInt();
        instruction.permute_map[lane] = value < 0 ? SxmInstruction::kZeroFill
            : static_cast<std::size_t>(value);
    }
    const auto kind = instruction.opcode == SxmOpcode::Transpose
        ? QueueKind::SxmTranspose : QueueKind::SxmPermute;
    queues[{kind, static_cast<int64_t>(op.getHemisphere())}].push_back(CommandSequence {
        command_cycle(op),
        static_cast<int64_t>(op.getRepeatCount().value_or(1)),
        static_cast<int64_t>(op.getRepeatInterval().value_or(1)), 0,
        sxm_instruction_command(instruction),
    });
}

int64_t sequence_final_cycle(const CommandSequence& sequence)
{
    return sequence.cycle
        + (sequence.outer_count - 1) * sequence.outer_interval
        + (sequence.repeat_count - 1) * sequence.repeat_interval;
}

QueueCommand apply_outer_induction(
    QueueCommand command, IcuInductionTarget target, int64_t delta)
{
    if (target == IcuInductionTarget::None) return command;
    if (target == IcuInductionTarget::MemAddress) {
        const auto encoded = static_cast<isa::EncodedMemInstruction>(
                                 command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1])
                << 32);
        const auto instruction = ftlpu::detail::apply_icu_repeat_2d_stride(
            isa::decode_mem_instruction(encoded), target, delta);
        return mem_instruction_command(
            isa::encode_mem_instruction(instruction));
    }
    if (target == IcuInductionTarget::MxmWeightColumn
        || target == IcuInductionTarget::MxmAccumulatorAddress) {
        const auto encoded = static_cast<isa::EncodedMxmInstruction>(
                                 command.words[0])
            | (static_cast<isa::EncodedMxmInstruction>(command.words[1])
                << 32);
        const auto instruction = ftlpu::detail::apply_icu_repeat_2d_stride(
            isa::decode_mxm_instruction(encoded), target, delta);
        return mxm_instruction_command(
            isa::encode_mxm_instruction(instruction));
    }
    throw std::runtime_error("unsupported Repeat2D induction target");
}

void expand_interleaved_repeat_2d(
    std::vector<CommandSequence>& sequences, bool repeat2DEnabled)
{
    std::vector<bool> expand(sequences.size(), false);
    int64_t precedingEnd = std::numeric_limits<int64_t>::min();
    for (std::size_t index = 0; index < sequences.size(); ++index) {
        const auto& candidate = sequences[index];
        const int64_t candidateEnd = sequence_final_cycle(candidate);
        if (candidate.outer_count > 1) {
            const bool overlapsPreceding = precedingEnd >= candidate.cycle;
            const bool overlapsFollowing = index + 1 < sequences.size()
                && sequences[index + 1].cycle <= candidateEnd;
            expand[index] = !repeat2DEnabled
                || overlapsPreceding || overlapsFollowing;
        }
        precedingEnd = std::max(precedingEnd, candidateEnd);
    }
    if (std::find(expand.begin(), expand.end(), true) == expand.end())
        return;
    std::vector<CommandSequence> materialized;
    for (std::size_t index = 0; index < sequences.size(); ++index) {
        if (!expand[index]) {
            materialized.push_back(std::move(sequences[index]));
            continue;
        }
        for (int64_t outer = 0;
             outer < sequences[index].outer_count; ++outer) {
            auto item = sequences[index];
            item.cycle += outer * item.outer_interval;
            item.instruction = apply_outer_induction(
                std::move(item.instruction), item.induction_target,
                outer * item.outer_stride);
            item.outer_count = 1;
            item.outer_interval = 1;
            item.outer_stride = 0;
            item.induction_target = IcuInductionTarget::None;
            materialized.push_back(std::move(item));
        }
    }
    sequences = std::move(materialized);
}

bool same_loop_instruction(const CommandSequence& first,
    const CommandSequence& next, QueueKind kind, int64_t addressStride)
{
    if (first.scale_binding != next.scale_binding
        || first.address_binding != next.address_binding
        || first.write_address_binding != next.write_address_binding
        || first.instruction.instruction_kind
            != next.instruction.instruction_kind
        || first.instruction.word_count != next.instruction.word_count
        || first.instruction.extension_words
            != next.instruction.extension_words)
        return false;
    if (kind != QueueKind::Mem)
        return first.instruction.words == next.instruction.words;

    const auto decode = [](const QueueCommand& command) {
        const auto encoded = static_cast<isa::EncodedMemInstruction>(
                                 command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1])
                << 32);
        return isa::decode_mem_instruction(encoded);
    };
    const auto lhs = decode(first.instruction);
    const auto rhs = decode(next.instruction);
    return lhs.opcode == rhs.opcode
        && lhs.stream == rhs.stream
        && lhs.map_stream == rhs.map_stream
        && lhs.preserve_stream == rhs.preserve_stream
        && static_cast<int64_t>(rhs.address)
            == static_cast<int64_t>(lhs.address) + addressStride;
}

void compress_loop_windows(
    std::vector<CommandSequence>& sequences, QueueKind kind)
{
    constexpr std::size_t kMaxWindow = 63;
    constexpr std::size_t kMaxRounds = 256;
    std::vector<CommandSequence> compressed;
    compressed.reserve(sequences.size());
    for (std::size_t index = 0; index < sequences.size();) {
        std::size_t bestWindow = 0;
        std::size_t bestRounds = 0;
        int64_t bestInterval = 0;
        int64_t bestStride = 0;
        const std::size_t remaining = sequences.size() - index;
        for (std::size_t window = 1;
             window <= std::min(kMaxWindow, remaining / 2); ++window) {
            bool simpleWindow = true;
            for (std::size_t offset = 0; offset < window; ++offset) {
                const auto& sequence = sequences[index + offset];
                simpleWindow = simpleWindow
                    && !sequence.is_loop
                    && sequence.repeat_count == 1
                    && sequence.outer_count == 1
                    && sequence.cycle
                        == sequences[index].cycle
                            + static_cast<int64_t>(offset);
            }
            if (!simpleWindow) continue;
            const int64_t interval =
                sequences[index + window].cycle
                - sequences[index].cycle;
            if (interval < static_cast<int64_t>(window)
                || interval > 255)
                continue;

            int64_t stride = 0;
            if (kind == QueueKind::Mem) {
                const auto decodeAddress = [](const QueueCommand& command) {
                    const auto encoded =
                        static_cast<isa::EncodedMemInstruction>(
                            command.words[0])
                        | (static_cast<isa::EncodedMemInstruction>(
                               command.words[1])
                            << 32);
                    return static_cast<int64_t>(
                        isa::decode_mem_instruction(encoded).address);
                };
                stride = decodeAddress(
                             sequences[index + window].instruction)
                    - decodeAddress(sequences[index].instruction);
                if (stride < -128 || stride > 127) continue;
            }
            std::size_t rounds = 1;
            while (rounds < kMaxRounds
                && index + (rounds + 1) * window <= sequences.size()) {
                bool same = true;
                for (std::size_t offset = 0; offset < window; ++offset) {
                    const auto& first = sequences[index + offset];
                    const auto& next =
                        sequences[index + rounds * window + offset];
                    if (next.is_loop
                        || next.repeat_count != 1
                        || next.outer_count != 1
                        || next.cycle
                            != first.cycle
                                + static_cast<int64_t>(rounds) * interval
                        || !same_loop_instruction(first, next, kind,
                            static_cast<int64_t>(rounds) * stride)) {
                        same = false;
                        break;
                    }
                }
                if (!same) break;
                ++rounds;
            }
            if (rounds > 1
                && rounds * window > bestRounds * bestWindow) {
                bestWindow = window;
                bestRounds = rounds;
                bestInterval = interval;
                bestStride = stride;
            }
        }
        if (bestRounds <= 1) {
            compressed.push_back(std::move(sequences[index++]));
            continue;
        }
        const int64_t loopCycle = sequences[index].cycle + bestInterval;
        for (std::size_t offset = 0; offset < bestWindow; ++offset)
            compressed.push_back(
                std::move(sequences[index + offset]));
        compressed.push_back(CommandSequence {
            loopCycle,
            static_cast<int64_t>(bestRounds - 1),
            bestInterval, bestStride,
            {}, -1, -1, -1, true,
            static_cast<int64_t>(bestWindow),
        });
        index += bestRounds * bestWindow;
    }
    sequences = std::move(compressed);
}

QueueProgram encode_queue(const QueueKey& key, std::vector<CommandSequence> sequences,
    std::size_t& max_cycle,
    std::vector<BinaryScaleRelocation>& scaleRelocations,
    std::vector<BinaryAddressRelocation>& addressRelocations,
    bool repeat2DEnabled)
{
    std::sort(sequences.begin(), sequences.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });
    expand_interleaved_repeat_2d(sequences, repeat2DEnabled);
    std::sort(sequences.begin(), sequences.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });
    if (key.first == QueueKind::Mem) {
        const auto decodeMem = [](const CommandSequence& sequence) {
            const auto encoded =
                static_cast<isa::EncodedMemInstruction>(
                    sequence.instruction.words[0])
                | (static_cast<isa::EncodedMemInstruction>(
                       sequence.instruction.words[1])
                    << 32);
            return isa::decode_mem_instruction(encoded);
        };
        std::unordered_set<int64_t> readCycles;
        std::unordered_set<int64_t> writeCycles;
        using RepeatShape = std::tuple<int64_t, int64_t, int64_t,
            int64_t, int64_t, int64_t, int64_t>;
        std::set<RepeatShape> readShapes;
        std::set<RepeatShape> writeShapes;
        for (const auto& sequence : sequences) {
            const auto instruction = decodeMem(sequence);
            auto* cycles = instruction.opcode == MemOpcode::Read
                ? &readCycles
                : instruction.opcode == MemOpcode::Write
                ? &writeCycles
                : nullptr;
            if (!cycles) continue;
            auto& shapes = instruction.opcode == MemOpcode::Read
                ? readShapes : writeShapes;
            shapes.emplace(sequence.cycle, sequence.repeat_count,
                sequence.repeat_interval, sequence.address_stride,
                sequence.outer_count, sequence.outer_interval,
                sequence.outer_stride);
            for (int64_t outer = 0;
                 outer < sequence.outer_count; ++outer)
                for (int64_t repeat = 0;
                     repeat < sequence.repeat_count; ++repeat)
                    cycles->insert(sequence.cycle
                        + outer * sequence.outer_interval
                        + repeat * sequence.repeat_interval);
        }
        std::vector<bool> expandSequence(sequences.size(), false);
        for (std::size_t index = 0; index < sequences.size(); ++index) {
            const auto instruction = decodeMem(sequences[index]);
            if (sequences[index].repeat_count <= 1
                || (instruction.opcode != MemOpcode::Read
                    && instruction.opcode != MemOpcode::Write))
                continue;
            const RepeatShape shape {sequences[index].cycle,
                sequences[index].repeat_count,
                sequences[index].repeat_interval,
                sequences[index].address_stride,
                sequences[index].outer_count,
                sequences[index].outer_interval,
                sequences[index].outer_stride};
            const bool hasDirectMate =
                instruction.opcode == MemOpcode::Read
                ? writeShapes.contains(shape)
                : readShapes.contains(shape);
            if (hasDirectMate) continue;
            const auto& oppositeCycles =
                instruction.opcode == MemOpcode::Read
                ? writeCycles : readCycles;
            for (int64_t outer = 0;
                 outer < sequences[index].outer_count
                    && !expandSequence[index]; ++outer)
                for (int64_t repeat = 0;
                     repeat < sequences[index].repeat_count; ++repeat) {
                    const int64_t cycle = sequences[index].cycle
                        + outer * sequences[index].outer_interval
                        + repeat * sequences[index].repeat_interval;
                    if (oppositeCycles.contains(cycle)) {
                        expandSequence[index] = true;
                        break;
                    }
                }
        }
        if (std::find(expandSequence.begin(), expandSequence.end(), true)
            != expandSequence.end()) {
            std::vector<CommandSequence> expanded;
            for (std::size_t index = 0; index < sequences.size(); ++index) {
                if (!expandSequence[index]) {
                    expanded.push_back(std::move(sequences[index]));
                    continue;
                }
                const auto instruction = decodeMem(sequences[index]);
                for (int64_t outer = 0;
                     outer < sequences[index].outer_count; ++outer)
                    for (int64_t repeat = 0;
                         repeat < sequences[index].repeat_count; ++repeat) {
                        CommandSequence item = sequences[index];
                        item.cycle += outer * sequences[index].outer_interval
                            + repeat * sequences[index].repeat_interval;
                        item.repeat_count = 1;
                        item.repeat_interval = 1;
                        item.address_stride = 0;
                        item.outer_count = 1;
                        item.outer_interval = 1;
                        item.outer_stride = 0;
                        item.induction_target = IcuInductionTarget::None;
                        const int64_t address = instruction.address
                            + outer * sequences[index].outer_stride
                            + repeat * sequences[index].address_stride;
                        const auto single = instruction.opcode == MemOpcode::Read
                            ? MemInstruction::Read(
                                  address, instruction.stream_id())
                            : instruction.preserve_stream
                            ? MemInstruction::WriteTap(
                                  address, instruction.stream_id())
                            : MemInstruction::Write(
                                  address, instruction.stream_id());
                        item.instruction = mem_instruction_command(
                            isa::encode_mem_instruction(single));
                        expanded.push_back(std::move(item));
                    }
            }
            sequences = std::move(expanded);
            std::sort(sequences.begin(), sequences.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.cycle < rhs.cycle;
            });
        }
    }
    // Loop control sequences are not functional MEM instructions. Perform
    // read/write collision legalization before introducing them, otherwise
    // the legalization code can decode a Loop placeholder as a MEM command
    // and discard the replay window while attempting to expand it.
    if (repeat2DEnabled)
        compress_loop_windows(sequences, key.first);
    QueueProgram queue {key.first, static_cast<std::size_t>(key.second), {}};
    int64_t cursor = 0;
    const CommandSequence* previous = nullptr;
    for (const CommandSequence& sequence : sequences) {
        if (sequence.is_loop) {
            if (sequence.cycle < cursor)
                throw std::runtime_error(
                    "Command IR Loop begins before its replay window ends");
            const auto window = static_cast<std::size_t>(
                sequence.loop_window_size);
            if (window == 0 || window > queue.commands.size())
                throw std::runtime_error(
                    "Command IR Loop window exceeds prior queue commands");
            for (std::size_t offset = 0; offset < window; ++offset) {
                const auto& command = queue.commands[
                    queue.commands.size() - window + offset];
                if (isa::decode_icu_command_opcode(command.command)
                    != isa::IcuCommandOpcode::Instruction)
                    throw std::runtime_error(
                        "Command IR Loop window must contain only contiguous instructions");
            }
            if (sequence.cycle > cursor)
                queue.commands.push_back(control_command(
                    isa::encode_icu_nop(static_cast<std::size_t>(
                        sequence.cycle - cursor))));
            queue.commands.push_back(control_command(isa::encode_icu_loop(
                IcuLoop {
                    window,
                    static_cast<std::size_t>(sequence.repeat_count),
                    static_cast<std::size_t>(sequence.repeat_interval),
                    sequence.address_stride,
                })));
            const int64_t finalCycle = sequence.cycle
                + (sequence.repeat_count - 1) * sequence.repeat_interval
                + sequence.loop_window_size - 1;
            cursor = finalCycle + 1;
            max_cycle = std::max(max_cycle,
                static_cast<std::size_t>(finalCycle));
            previous = &sequence;
            continue;
        }
        if (sequence.cycle < cursor)
            throw std::runtime_error("overlapping Command IR sequences target ICU queue kind="
                + std::to_string(static_cast<int>(key.first)) + " index="
                + std::to_string(key.second) + " at cycle="
                + std::to_string(sequence.cycle) + " (busy through "
                + std::to_string(cursor - 1) + "; previous start="
                + std::to_string(previous ? previous->cycle : -1) + " count="
                + std::to_string(previous ? previous->repeat_count : -1) + " interval="
                + std::to_string(previous ? previous->repeat_interval : -1)
                + ")");
        if (sequence.cycle > cursor)
            queue.commands.push_back(control_command(
                isa::encode_icu_nop(static_cast<std::size_t>(sequence.cycle - cursor))));
        const std::size_t instructionIndex = queue.commands.size();
        queue.commands.push_back(sequence.instruction);
        if (sequence.scale_binding >= 0) {
            scaleRelocations.push_back(BinaryScaleRelocation {
                static_cast<std::uint32_t>(sequence.scale_binding),
                0,
                key.first,
                static_cast<std::uint16_t>(key.second),
                static_cast<std::uint32_t>(instructionIndex),
                VxmImmediateOperand::Rhs,
            });
        }
        if (sequence.address_binding >= 0) {
            addressRelocations.push_back(BinaryAddressRelocation {
                static_cast<std::uint32_t>(sequence.address_binding),
                software::runtime::BindingAccess::Input,
                key.first,
                static_cast<std::uint16_t>(key.second),
                static_cast<std::uint32_t>(instructionIndex),
                false,
            });
        }
        if (sequence.write_address_binding >= 0) {
            addressRelocations.push_back(BinaryAddressRelocation {
                static_cast<std::uint32_t>(
                    sequence.write_address_binding),
                software::runtime::BindingAccess::Input,
                key.first,
                static_cast<std::uint16_t>(key.second),
                static_cast<std::uint32_t>(instructionIndex),
                true,
            });
        }
        if (sequence.outer_count > 1) {
            queue.commands.push_back(repeat_2d_command(IcuRepeat2D {
                static_cast<std::size_t>(sequence.repeat_count),
                static_cast<std::size_t>(sequence.repeat_interval),
                sequence.address_stride,
                static_cast<std::size_t>(sequence.outer_count),
                static_cast<std::size_t>(sequence.outer_interval),
                sequence.outer_stride,
                sequence.induction_target,
            }));
        } else if (sequence.repeat_count > 1) {
            queue.commands.push_back(control_command(isa::encode_icu_repeat(
                InstructionControlUnit::Repeat {
                    static_cast<std::size_t>(sequence.repeat_count - 1),
                    static_cast<std::size_t>(sequence.repeat_interval),
                    sequence.address_stride,
                })));
        }
        const int64_t final_cycle = sequence_final_cycle(sequence);
        cursor = final_cycle + 1;
        max_cycle = std::max(max_cycle, static_cast<std::size_t>(final_cycle));
        previous = &sequence;
    }
    return queue;
}

} // namespace

software::runtime::BinaryProgram translate_command_module(mlir::ModuleOp module)
{
    const auto target = LPUTargetModel::from_operation(module);
    if (mlir::failed(target))
        throw std::runtime_error("Command IR module has an invalid target");
    QueueMap queues;
    std::vector<BinaryBinding> bindings;
    std::vector<BinaryTimeline> timelines;
    module.walk([&](command::BindingOp op) { bindings.push_back(translate_binding(op)); });
    module.walk([&](command::TimelineOp op) {
        timelines.push_back(BinaryTimeline {
            op.getName().str(),
            static_cast<std::uint64_t>(op.getStart()),
            static_cast<std::uint64_t>(op.getEnd()),
        });
    });
    module.walk([&](command::MemOp op) { collect_mem(op, queues); });
    module.walk([&](command::MxmOp op) { collect_mxm(op, queues); });
    module.walk([&](command::MxmDequantOp op) {
        collect_mxm_dequant(op, queues);
    });
    module.walk([&](command::VxmOp op) { collect_vxm(op, queues); });
    module.walk([&](command::SxmOp op) { collect_sxm(op, queues); });
    module.walk([&](command::LoopOp op) { collect_loop(op, queues); });
    if (queues.empty()) throw std::runtime_error("Command IR module has no queue commands");

    // A binary program starts its ICU clock at zero. Full programs naturally
    // have an origin of zero; rebasing also makes a standalone scheduled phase
    // (for example the QK trace) runnable without materializing unrelated
    // preceding phases as tens of thousands of ICU NOPs.
    int64_t cycle_origin = std::numeric_limits<int64_t>::max();
    for (const auto& [key, sequences] : queues) {
        (void)key;
        for (const CommandSequence& sequence : sequences)
            cycle_origin = std::min(cycle_origin, sequence.cycle);
    }
    for (const BinaryTimeline& timeline : timelines)
        cycle_origin = std::min(
            cycle_origin, static_cast<int64_t>(timeline.start_cycle));
    if (cycle_origin > 0) {
        for (auto& [key, sequences] : queues) {
            (void)key;
            for (CommandSequence& sequence : sequences) sequence.cycle -= cycle_origin;
        }
        for (BinaryBinding& binding : bindings)
            binding.ready_cycle =
                binding.ready_cycle > static_cast<std::uint64_t>(cycle_origin)
                ? binding.ready_cycle
                    - static_cast<std::uint64_t>(cycle_origin)
                : 0;
        for (BinaryTimeline& timeline : timelines) {
            timeline.start_cycle -= static_cast<std::uint64_t>(cycle_origin);
            timeline.end_cycle -= static_cast<std::uint64_t>(cycle_origin);
        }
    }

    std::sort(bindings.begin(), bindings.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.access, lhs.index) < std::tie(rhs.access, rhs.index);
    });
    software::runtime::BinaryProgram program;
    program.target_name = target->name();
    program.target_abi = target->abi_fingerprint();
    auto& hardware = program.hardware;
    const auto& memory = target->memory();
    const auto& streams = target->streams();
    const auto& throughput = target->throughput();
#define COPY_MEMORY(field) \
    hardware.field = static_cast<std::uint32_t>(memory.field)
    COPY_MEMORY(hemispheres);
    COPY_MEMORY(slices_per_hemisphere);
    COPY_MEMORY(banks_per_slice);
    COPY_MEMORY(words_per_bank);
    COPY_MEMORY(bytes_per_word);
    COPY_MEMORY(sram_depth_rows);
    COPY_MEMORY(sram_read_ports_per_slice);
    COPY_MEMORY(sram_write_ports_per_slice);
#undef COPY_MEMORY
#define COPY_STREAM(field) \
    hardware.field = static_cast<std::uint32_t>(streams.field)
    COPY_STREAM(streams_per_direction);
    COPY_STREAM(encoded_streams);
    COPY_STREAM(mem_boundary_register_columns);
    COPY_STREAM(system_register_columns);
    COPY_STREAM(mem_slices_per_register_group);
#undef COPY_STREAM
#define COPY_THROUGHPUT(field) \
    hardware.field = static_cast<std::uint32_t>(throughput.field)
    COPY_THROUGHPUT(tile_rows);
    COPY_THROUGHPUT(lanes_per_tile);
    COPY_THROUGHPUT(mem_read_bytes_per_cycle);
    COPY_THROUGHPUT(mem_write_bytes_per_cycle);
    COPY_THROUGHPUT(mxm_rows);
    COPY_THROUGHPUT(mxm_columns);
    COPY_THROUGHPUT(mxm_load_streams_per_cycle);
    COPY_THROUGHPUT(mxm_int8_load_streams_per_cycle);
    COPY_THROUGHPUT(mxm_load_bytes_per_cycle);
    COPY_THROUGHPUT(mxm_activation_streams);
    COPY_THROUGHPUT(mxm_result_streams);
    COPY_THROUGHPUT(mxm_pipeline_rows);
    COPY_THROUGHPUT(mxm_block_rows);
    COPY_THROUGHPUT(mxm_local_dequant_enabled);
    COPY_THROUGHPUT(mxm_block_compute_enabled);
    COPY_THROUGHPUT(mxm_weight_activation_overlap_enabled);
    COPY_THROUGHPUT(mxm_local_load_to_compute_latency);
    COPY_THROUGHPUT(mxm_block_group_interval);
    COPY_THROUGHPUT(mxm_earliest_iw_cycle);
    COPY_THROUGHPUT(qk_iw_to_compute_latency);
    COPY_THROUGHPUT(mxms_per_hemisphere);
    COPY_THROUGHPUT(mxm_weight_buffers);
    COPY_THROUGHPUT(mxm_accumulator_blocks);
    COPY_THROUGHPUT(vxm_alus);
    COPY_THROUGHPUT(vxm_weight_to_iw_latency);
    COPY_THROUGHPUT(mem_to_sxm_latency);
    COPY_THROUGHPUT(mem_to_mxm_latency);
    COPY_THROUGHPUT(mxm0_accumulator_latency);
    COPY_THROUGHPUT(mxm1_accumulator_latency);
    COPY_THROUGHPUT(accumulator_to_vxm_latency);
    COPY_THROUGHPUT(accumulator_read_to_vxm_latency);
    COPY_THROUGHPUT(swiglu_write_latency);
#undef COPY_THROUGHPUT
    program.memory_floors = static_memory_floors(module,
        target->memory().slices_per_hemisphere,
        target->memory().banks_per_slice,
        target->memory().sram_depth_rows);
    program.bindings = std::move(bindings);
    program.timelines = std::move(timelines);
    for (auto& [key, sequences] : queues)
        program.queues.push_back(encode_queue(key, std::move(sequences),
            program.max_cycle, program.scale_relocations,
            program.address_relocations,
            target->throughput().icu_repeat_2d_enabled != 0));
    return program;
}

} // namespace ftlpu::compiler::target
