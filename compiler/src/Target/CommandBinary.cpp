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
        4,
        encoded.words,
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
    command.words[3] = 0;
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

std::vector<BinaryMemoryFloor> static_memory_floors(
    mlir::ModuleOp module, int64_t slices_per_hemisphere,
    int64_t rows_per_slice)
{
    std::map<std::pair<int64_t, int64_t>, int64_t> floors;
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
            queue / slices_per_hemisphere;
        const int64_t slice = queue % slices_per_hemisphere;
        if (queue < 0 || hemisphere >= 2)
            throw std::runtime_error(
                "Command IR MEM queue is outside the target");
        for (int64_t repeat : {int64_t {0}, repeat_count - 1})
            for (int64_t wave : {int64_t {0}, wave_count - 1}) {
                const int64_t address = base
                    + repeat * repeat_stride
                    + wave * wave_stride;
                if (address < 0 || address >= rows_per_slice)
                    throw std::runtime_error(
                        "Command IR MEM scratch address is outside the target");
                auto& floor = floors[{hemisphere, slice}];
                floor = std::max(floor, address + 1);
            }
    });
    std::vector<BinaryMemoryFloor> result;
    result.reserve(floors.size());
    for (const auto& entry : floors) {
        const auto [hemisphere, slice] = entry.first;
        const int64_t floor = entry.second;
        result.push_back(BinaryMemoryFloor {
            static_cast<std::uint16_t>(hemisphere),
            static_cast<std::uint16_t>(slice),
            static_cast<std::uint32_t>(floor),
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
    for (int64_t wave = 0; wave < waveCount; ++wave) {
        const int64_t address =
            static_cast<int64_t>(op.getAddress())
            + wave * waveAddressStride;
        const auto instruction = op.getOpcode() == "read"
            ? MemInstruction::Read(address, op.getPackedStream())
            : op.getOpcode() == "write_tap"
            ? MemInstruction::WriteTap(address, op.getPackedStream())
            : MemInstruction::Write(address, op.getPackedStream());
        queues[{QueueKind::Mem, queue}].push_back(CommandSequence {
            command_cycle(op) + wave * waveInterval,
            op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
            op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(),
            op->getAttrOfType<mlir::IntegerAttr>("address_stride").getInt(),
            mem_instruction_command(
                isa::encode_mem_instruction(instruction)),
            -1,
            op.getAddressBinding()
                ? static_cast<int64_t>(*op.getAddressBinding()) : -1,
        });
    }
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
    const auto instruction = is_load
        ? op.getWeightLoadMode().value_or("supercell") == "column"
            ? MxmControlInstruction::IWColumn(
                op.getWeightBuffer(), op.getWeightColumn(),
                op.getWeightInnerColumn().value_or(0), inputMode)
            : MxmControlInstruction::IW(
                op.getWeightBuffer(), op.getWeightColumn(), inputMode)
        : is_accumulator_read
        ? MxmControlInstruction::AccumulatorRead(op.getAccumulatorAddress(),
            op.getOutputStreamBase(), op.getAccumulatorClear(),
            computeMode)
        : MxmControlInstruction::Compute(op.getWeightBuffer(),
            op.getActivationStreamBase(), op.getOutputStreamBase(),
            op.getAccumulatorAddress(), op.getAccumulatorRowStride(),
            destination,
             op.getDataFormat().value_or("fp16") == "bf16"
                 ? MxmDataFormat::BFloat16
                 : MxmDataFormat::Float16,
             computeMode,
             op.getAccumulatorClear(),
             op.getAccumulatorOutputFormat().value_or("fp32") == "bf16"
                 ? MxmAccumulatorOutputFormat::BFloat16
                 : MxmAccumulatorOutputFormat::Float32);
    const auto kind = is_load ? QueueKind::MxmLoad : QueueKind::MxmCompute;
    queues[{kind, static_cast<int64_t>(op.getQueue())}].push_back(CommandSequence {
        command_cycle(op),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(), 0,
        mxm_instruction_command(isa::encode_mxm_instruction(instruction)),
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
        });
}

VxmAluOpcode parse_vxm_opcode(llvm::StringRef value)
{
    if (value == "pass") return VxmAluOpcode::Pass;
    if (value == "add") return VxmAluOpcode::Add;
    if (value == "subtract") return VxmAluOpcode::Subtract;
    if (value == "multiply") return VxmAluOpcode::Multiply;
    if (value == "divide") return VxmAluOpcode::Divide;
    if (value == "negate") return VxmAluOpcode::Negate;
    if (value == "abs") return VxmAluOpcode::Abs;
    if (value == "min") return VxmAluOpcode::Min;
    if (value == "max") return VxmAluOpcode::Max;
    if (value == "clamp") return VxmAluOpcode::Clamp;
    if (value == "square") return VxmAluOpcode::Square;
    if (value == "sqrt") return VxmAluOpcode::Sqrt;
    if (value == "exp") return VxmAluOpcode::Exp;
    if (value == "log") return VxmAluOpcode::Log;
    if (value == "relu") return VxmAluOpcode::Relu;
    if (value == "cast") return VxmAluOpcode::Cast;
    throw std::runtime_error("unsupported Command IR VXM opcode");
}

VxmLaneOperand parse_vxm_operand(llvm::StringRef kind, int64_t index, float immediate)
{
    if (kind == "alu") return VxmLaneOperand::Alu(static_cast<std::size_t>(index));
    if (kind == "stream_i32")
        return VxmLaneOperand::StreamInt32(static_cast<std::size_t>(index));
    if (kind == "stream_f32")
        return VxmLaneOperand::StreamFloat32(static_cast<std::size_t>(index));
    if (kind == "stream_i8")
        return VxmLaneOperand::StreamInt8(static_cast<std::size_t>(index));
    if (kind == "stream_f16")
        return VxmLaneOperand::StreamFloat16(static_cast<std::size_t>(index));
    if (kind == "stream_bf16")
        return VxmLaneOperand::StreamBFloat16(
            static_cast<std::size_t>(index));
    if (kind == "immediate") return VxmLaneOperand::Imm(immediate);
    throw std::runtime_error("unsupported Command IR VXM operand kind");
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
    const int64_t output_stream = op.getOutputStreamAttr().getInt();
    auto instruction = VxmLaneAluInstruction {};
    instruction.opcode = parse_vxm_opcode(op.getOpcode());
    instruction.lhs = parse_vxm_operand(op.getLhsKind(), op.getLhsIndex(),
        static_cast<float>(op.getLhsImmediateAttr().getValueAsDouble()));
    instruction.rhs = parse_vxm_operand(op.getRhsKind(), op.getRhsIndex(),
        static_cast<float>(op.getRhsImmediateAttr().getValueAsDouble()));
    instruction.cast_target = parse_vxm_cast_target(op.getCastTarget());
    if (output_stream >= 0)
        instruction.output_stream = static_cast<std::size_t>(output_stream);
    instruction.input_hemisphere = op.getInputHemisphere() == "east"
        ? Hemisphere::East : Hemisphere::West;
    instruction.output_hemisphere = op.getOutputHemisphere() == "east"
        ? Hemisphere::East : Hemisphere::West;
    queues[{QueueKind::Vxm, static_cast<int64_t>(op.getQueue())}].push_back(CommandSequence {
        command_cycle(op),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(), 0,
        vxm_instruction_command(isa::encode_vxm_instruction(instruction)),
        op.getScaleBinding()
            ? static_cast<int64_t>(*op.getScaleBinding()) : -1,
    });
}

void collect_sxm(command::SxmOp op, QueueMap& queues)
{
    SxmInstruction instruction {};
    instruction.opcode = op.getOpcode() == "transpose"
        ? SxmOpcode::Transpose : SxmOpcode::Permute;
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

QueueProgram encode_queue(const QueueKey& key, std::vector<CommandSequence> sequences,
    std::size_t& max_cycle,
    std::vector<BinaryScaleRelocation>& scaleRelocations,
    std::vector<BinaryAddressRelocation>& addressRelocations)
{
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
        using RepeatShape =
            std::tuple<int64_t, int64_t, int64_t, int64_t>;
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
                sequence.repeat_interval, sequence.address_stride);
            for (int64_t repeat = 0;
                 repeat < sequence.repeat_count; ++repeat)
                cycles->insert(sequence.cycle
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
                sequences[index].address_stride};
            const bool hasDirectMate =
                instruction.opcode == MemOpcode::Read
                ? writeShapes.contains(shape)
                : readShapes.contains(shape);
            if (hasDirectMate) continue;
            const auto& oppositeCycles =
                instruction.opcode == MemOpcode::Read
                ? writeCycles : readCycles;
            for (int64_t repeat = 0;
                 repeat < sequences[index].repeat_count; ++repeat) {
                const int64_t cycle = sequences[index].cycle
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
                for (int64_t repeat = 0;
                     repeat < sequences[index].repeat_count; ++repeat) {
                    CommandSequence item = sequences[index];
                    item.cycle +=
                        repeat * sequences[index].repeat_interval;
                    item.repeat_count = 1;
                    item.repeat_interval = 1;
                    item.address_stride = 0;
                    const int64_t address = instruction.address
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
        for (std::size_t index = 0; index < sequences.size(); ++index) {
            auto& first = sequences[index];
            const auto firstEncoded =
                static_cast<isa::EncodedMemInstruction>(
                    first.instruction.words[0])
                | (static_cast<isa::EncodedMemInstruction>(
                       first.instruction.words[1])
                    << 32);
            const MemInstruction firstInstruction =
                isa::decode_mem_instruction(firstEncoded);
            if (firstInstruction.opcode != MemOpcode::Read
                && firstInstruction.opcode != MemOpcode::Write)
                continue;
            for (std::size_t other = index + 1;
                 other < sequences.size()
                    && sequences[other].cycle == first.cycle;
                 ++other) {
                auto& second = sequences[other];
                if (first.repeat_count != second.repeat_count
                    || first.repeat_interval != second.repeat_interval
                    || (first.repeat_count > 1
                        && first.address_stride
                            != second.address_stride))
                    continue;
                const auto secondEncoded =
                    static_cast<isa::EncodedMemInstruction>(
                        second.instruction.words[0])
                    | (static_cast<isa::EncodedMemInstruction>(
                           second.instruction.words[1])
                        << 32);
                const MemInstruction secondInstruction =
                    isa::decode_mem_instruction(secondEncoded);
                if (firstInstruction.opcode
                        == secondInstruction.opcode
                    || (secondInstruction.opcode != MemOpcode::Read
                        && secondInstruction.opcode != MemOpcode::Write))
                    continue;
                const MemInstruction& read =
                    firstInstruction.opcode == MemOpcode::Read
                    ? firstInstruction
                    : secondInstruction;
                const MemInstruction& write =
                    firstInstruction.opcode == MemOpcode::Write
                    ? firstInstruction
                    : secondInstruction;
                if (read.address == write.address)
                    throw std::runtime_error(
                        "dual-port MEM read/write uses the same SRAM row");
                const int64_t readBinding =
                    firstInstruction.opcode == MemOpcode::Read
                    ? first.address_binding
                    : second.address_binding;
                const int64_t writeBinding =
                    firstInstruction.opcode == MemOpcode::Write
                    ? first.address_binding
                    : second.address_binding;
                const auto combined = write.preserve_stream
                    ? MemInstruction::ReadWriteTap(
                          read.address, read.stream_id(), write.address,
                          write.stream_id())
                    : MemInstruction::ReadWrite(
                          read.address, read.stream_id(), write.address,
                          write.stream_id());
                first.instruction = mem_instruction_command(
                    isa::encode_mem_instruction(combined));
                first.address_binding = readBinding;
                first.write_address_binding = writeBinding;
                sequences.erase(sequences.begin() + other);
                break;
            }
        }
    }
    QueueProgram queue {key.first, static_cast<std::size_t>(key.second), {}};
    int64_t cursor = 0;
    const CommandSequence* previous = nullptr;
    for (const CommandSequence& sequence : sequences) {
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
        if (sequence.repeat_count > 1) {
            queue.commands.push_back(control_command(isa::encode_icu_repeat(
                InstructionControlUnit::Repeat {
                    static_cast<std::size_t>(sequence.repeat_count - 1),
                    static_cast<std::size_t>(sequence.repeat_interval),
                    sequence.address_stride,
                })));
        }
        const int64_t final_cycle = sequence.cycle
            + (sequence.repeat_count - 1) * sequence.repeat_interval;
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
    program.memory_rows_per_slice = static_cast<std::uint32_t>(
        target->memory().sram_depth_rows);
    program.mxms_per_hemisphere = static_cast<std::uint32_t>(
        target->throughput().mxms_per_hemisphere);
    program.memory_floors = static_memory_floors(module,
        target->memory().slices_per_hemisphere,
        target->memory().sram_depth_rows);
    program.bindings = std::move(bindings);
    program.timelines = std::move(timelines);
    for (auto& [key, sequences] : queues)
        program.queues.push_back(encode_queue(key, std::move(sequences),
            program.max_cycle, program.scale_relocations,
            program.address_relocations));
    return program;
}

} // namespace ftlpu::compiler::target
