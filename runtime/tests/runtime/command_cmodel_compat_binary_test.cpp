#include "ftlpu/software/runtime/binary.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    if (argc != 2)
        throw std::runtime_error(
            "usage: command_cmodel_compat_binary_test program.ftlpu");

    const BinaryProgram program =
        read_binary_program(std::filesystem::path(argv[1]));
    bool sawReadBase = false;
    bool sawWriteBase = false;
    bool sawReadRepeat = false;
    bool sawWriteRepeat = false;
    bool sawDequant = false;
    bool sawInt8Load = false;
    bool sawCompute = false;
    std::size_t materializedLoopReplays = 0;
    bool sawRepeat2D = false;
    bool sawMxmAccumulatorRepeat2D = false;
    bool sawInterleavedRepeat2D = false;
    bool sawInterleavedInnerStride = false;
    bool sawRepeatedLoopCandidate = false;
    std::size_t loopQueueCommands = 0;
    std::size_t repeat2DQueueCommands = 0;
    for (const QueueProgram& queue : program.queues) {
        if (queue.kind == QueueKind::Mem && queue.index == 2)
            loopQueueCommands = queue.commands.size();
        if (queue.kind == QueueKind::Mem && queue.index == 4)
            repeat2DQueueCommands = queue.commands.size();
        for (const QueueCommand& command : queue.commands) {
            if (is_repeat_2d_command(command)) {
                const auto repeat = decode_repeat_2d_command(command);
                sawMxmAccumulatorRepeat2D |=
                    queue.kind == QueueKind::MxmCompute
                    && queue.index == 1
                    && repeat.inner_count == 4
                    && repeat.inner_interval == 1
                    && repeat.inner_stride == 0
                    && repeat.outer_count == 3
                    && repeat.outer_interval == 8
                    && repeat.outer_stride == 4
                    && repeat.induction_target
                        == IcuInductionTarget::MxmAccumulatorAddress;
                sawInterleavedRepeat2D |= queue.kind == QueueKind::Mem
                    && queue.index == 6;
                sawRepeat2D |= queue.kind == QueueKind::Mem
                    && queue.index == 4
                    && repeat.inner_count == 4
                    && repeat.inner_interval == 1
                    && repeat.inner_stride == -1
                    && repeat.outer_count == 3
                    && repeat.outer_interval == 8
                    && repeat.outer_stride == 16
                    && repeat.induction_target
                        == IcuInductionTarget::MemAddress;
                continue;
            }
            if (isa::decode_icu_command_opcode(command.command)
                == isa::IcuCommandOpcode::Repeat) {
                const auto repeat = isa::decode_icu_repeat(command.command);
                sawInterleavedInnerStride |=
                    queue.kind == QueueKind::Mem && queue.index == 6
                    && repeat.count == 3 && repeat.interval == 1
                    && repeat.address_stride == 1;
                sawRepeatedLoopCandidate |= queue.kind == QueueKind::Mem
                    && queue.index == 8 && repeat.count == 1
                    && repeat.interval == 1
                    && repeat.address_stride == 1;
                sawReadRepeat |= queue.kind == QueueKind::Mem
                    && queue.index == 0 && repeat.count == 2
                    && repeat.interval == 1
                    && repeat.address_stride == 1;
                sawWriteRepeat |= queue.kind == QueueKind::Mem
                    && queue.index == 1 && repeat.count == 2
                    && repeat.interval == 1
                    && repeat.address_stride == 2;
                continue;
            }
            if (command.instruction_kind == InstructionKind::Mem) {
                const auto word =
                    static_cast<isa::EncodedMemInstruction>(
                        command.words[0])
                    | (static_cast<isa::EncodedMemInstruction>(
                           command.words[1])
                        << 32);
                const MemInstruction instruction =
                    isa::decode_mem_instruction(word);
                sawReadBase |= queue.index == 0
                    && instruction.opcode == MemOpcode::Read
                    && instruction.address == 10
                    && instruction.stream == 0;
                sawWriteBase |= queue.index == 1
                    && instruction.opcode == MemOpcode::Write
                    && instruction.address == 20
                    && instruction.stream == 32;
                if (queue.index == 2
                    && instruction.opcode == MemOpcode::Read
                    && ((instruction.address >= 116
                            && instruction.address <= 118)
                        || (instruction.address >= 132
                            && instruction.address <= 134))) {
                    ++materializedLoopReplays;
                }
            } else if (command.instruction_kind
                == InstructionKind::MxmDequant) {
                const auto instruction =
                    isa::decode_mxm_dequant_instruction(
                        static_cast<isa::EncodedMxmDequantInstruction>(
                            command.words[0]));
                sawDequant = instruction.scale_bf16
                    == MxmDequantInstruction::Scale(0.125f).scale_bf16;
            } else if (command.instruction_kind
                == InstructionKind::Mxm) {
                const auto word =
                    static_cast<isa::EncodedMxmInstruction>(
                        command.words[0])
                    | (static_cast<isa::EncodedMxmInstruction>(
                           command.words[1])
                        << 32);
                const MxmControlInstruction instruction =
                    isa::decode_mxm_instruction(word);
                sawInt8Load |= instruction.opcode == MxmControlOpcode::IW
                    && instruction.weight_input_mode
                        == MxmWeightInputMode::Int8DequantBf16;
                sawCompute |= instruction.opcode == MxmControlOpcode::Compute;
            }
        }
    }
    require(sawReadBase && sawWriteBase
            && sawReadRepeat && sawWriteRepeat,
        "bank-separated MEM read/write repeats were not preserved");
    require(sawDequant, "MXM dequant queue scale was not preserved");
    require(program.scale_relocations.size() == 1
            && program.scale_relocations[0].binding_index == 7
            && program.scale_relocations[0].queue_kind
                == QueueKind::MxmDequant,
        "MXM dequant scale relocation was not preserved");
    require(sawInt8Load, "MXM INT8 dequant load mode was not preserved");
    require(sawCompute, "MXM compute command was not preserved");
    require(materializedLoopReplays == 6,
        "legacy Command IR Loop was not materialized correctly");
    require(loopQueueCommands == 10,
        "materialized Command IR Loop has an unexpected queue size");
    require(sawRepeat2D, "ICU Repeat2D descriptor was not preserved");
    require(sawMxmAccumulatorRepeat2D,
        "MXM accumulator-address Repeat2D descriptor was not preserved");
    require(repeat2DQueueCommands == 2,
        "ICU Repeat2D was expanded instead of encoded as one descriptor");
    require(!sawInterleavedRepeat2D,
        "interleaved outer waves cannot use blocking ICU Repeat2D");
    require(sawInterleavedInnerStride,
        "expanding interleaved outer waves lost the inner MEM address stride");
    require(sawRepeatedLoopCandidate,
        "Repeat encoding lost a command's inner repeat space");

    InstructionControlUnit icu;
    load_queue_programs_into_icu(program.queues, icu,
        program.hardware.mxms_per_hemisphere);
    std::cout << "command_cmodel_compat_binary_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "command_cmodel_compat_binary_test failed: "
              << ex.what() << '\n';
    return 1;
}
