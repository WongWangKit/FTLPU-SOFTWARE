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

namespace ftlpu::compiler::target {
namespace {

// QueueCommand has a variable SXM payload; keep this translation unit rebuilt
// with the runtime queue ABI rather than relying on a transitive header edge.

using software::runtime::BinaryBinding;
using software::runtime::BindingAccess;
using software::runtime::BindingElementType;
using software::runtime::BindingLayout;
using software::runtime::InstructionKind;
using software::runtime::QueueCommand;
using software::runtime::QueueKind;
using software::runtime::QueueProgram;

struct CommandSequence {
    int64_t cycle{0};
    int64_t repeat_count{0};
    int64_t repeat_interval{0};
    int64_t address_stride{0};
    QueueCommand instruction;
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

QueueCommand instruction_command(InstructionKind kind, std::uint32_t word)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        kind,
        1,
        {word, 0, 0, 0},
    };
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
    command.words[3] = static_cast<std::uint32_t>(instruction.weight_layout);
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
    if (value == "w8a16_attention_weight_striped")
        return BindingLayout::W8A16AttentionWeightStriped;
    if (value == "fp16_pair_planar") return BindingLayout::Fp16PairPlanar;
    if (value == "fp32_causal_mask_tile")
        return BindingLayout::Fp32CausalMaskTile;
    throw std::runtime_error("unsupported Command IR binding layout");
}

BinaryBinding translate_binding(command::BindingOp op)
{
    BinaryBinding binding;
    binding.index = static_cast<std::uint32_t>(op.getIndex());
    binding.access = op.getAccess() == "input" ? BindingAccess::Input
        : op.getAccess() == "output" ? BindingAccess::Output
        : BindingAccess::Internal;
    binding.element_type = op.getElementType() == "i8" ? BindingElementType::I8
        : op.getElementType() == "f16" ? BindingElementType::F16
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
    const int64_t cycle = command_cycle(op);
    const auto instruction = op.getOpcode() == "read"
        ? MemInstruction::Read(op.getAddress(), op.getPackedStream())
        : op.getOpcode() == "write"
        ? MemInstruction::Write(op.getAddress(), op.getPackedStream())
        : MemInstruction::Accumulate(op.getAddress(), op.getPackedStream(),
            op.getAccumulatorDestination() == "stream"
                ? MemAccumulatorDestination::Stream : MemAccumulatorDestination::Sram);
    queues[{QueueKind::Mem, queue}].push_back(CommandSequence {
        cycle,
        op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(),
        op->getAttrOfType<mlir::IntegerAttr>("address_stride").getInt(),
        mem_instruction_command(isa::encode_mem_instruction(instruction)),
    });
}

void collect_mxm(command::MxmOp op, QueueMap& queues)
{
    const bool is_load = op.getOpcode() == "iw";
    const auto instruction = is_load
        ? MxmControlInstruction::IW(op.getWeightBuffer(), op.getWeightColumn())
        : MxmControlInstruction::Compute(op.getWeightBuffer(),
            op.getActivationStreamBase(), op.getOutputStreamBase());
    const auto kind = is_load ? QueueKind::MxmLoad : QueueKind::MxmCompute;
    queues[{kind, static_cast<int64_t>(op.getQueue())}].push_back(CommandSequence {
        command_cycle(op),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
        op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(), 0,
        instruction_command(InstructionKind::Mxm, isa::encode_mxm_instruction(instruction)),
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
    if (kind == "immediate") return VxmLaneOperand::Imm(immediate);
    throw std::runtime_error("unsupported Command IR VXM operand kind");
}

VxmCastTarget parse_vxm_cast_target(llvm::StringRef value)
{
    if (value == "fp32") return VxmCastTarget::Float32;
    if (value == "fp16") return VxmCastTarget::Float16;
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
    });
}

void collect_sxm(command::SxmOp op, QueueMap& queues)
{
    SxmInstruction instruction {};
    instruction.opcode = op.getOpcode() == "transpose"
        ? SxmOpcode::Transpose : SxmOpcode::Permute;
    instruction.weight_layout = op.getWeightLayout() == "matrix_columns"
        ? SxmWeightLayout::MatrixColumns : SxmWeightLayout::VectorColumns;
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
        1, 1, 0, sxm_instruction_command(instruction),
    });
}

QueueProgram encode_queue(const QueueKey& key, std::vector<CommandSequence> sequences,
    std::size_t& max_cycle)
{
    std::sort(sequences.begin(), sequences.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });
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
        queue.commands.push_back(sequence.instruction);
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
    module.walk([&](command::BindingOp op) { bindings.push_back(translate_binding(op)); });
    module.walk([&](command::MemOp op) { collect_mem(op, queues); });
    module.walk([&](command::MxmOp op) { collect_mxm(op, queues); });
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
    if (cycle_origin > 0) {
        for (auto& [key, sequences] : queues) {
            (void)key;
            for (CommandSequence& sequence : sequences) sequence.cycle -= cycle_origin;
        }
    }

    std::sort(bindings.begin(), bindings.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.access, lhs.index) < std::tie(rhs.access, rhs.index);
    });
    software::runtime::BinaryProgram program;
    program.target_name = target->name();
    program.target_abi = target->abi_fingerprint();
    program.bindings = std::move(bindings);
    for (auto& [key, sequences] : queues)
        program.queues.push_back(encode_queue(key, std::move(sequences), program.max_cycle));
    return program;
}

} // namespace ftlpu::compiler::target
