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
    bool sawMacro = false;
    bool sawMxmAccumulatorMacro = false;
    bool sawSequentialMacro = false;
    bool sawSequentialInnerStride = false;
    bool sawRepeatedCandidate = false;
    std::size_t macroQueueCommands = 0;
    for (const QueueProgram& queue : program.queues) {
        if (queue.kind == QueueKind::Mem && queue.index == 4)
            macroQueueCommands = queue.commands.size();
        for (const QueueCommand& command : queue.commands) {
            if (is_macro_schedule_command(command)) {
                const auto repeat = decode_macro_schedule_command(command);
                sawMxmAccumulatorMacro |=
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
                sawSequentialMacro |= queue.kind == QueueKind::Mem
                    && queue.index == 6;
                sawMacro |= queue.kind == QueueKind::Mem
                    && queue.index == 4
                    && repeat.inner_count == 4
                    && repeat.inner_interval == 1
                    && repeat.inner_stride == -1
                    && repeat.outer_count == 3
                    && repeat.outer_interval == 8
                    && repeat.outer_stride == 16
                    && repeat.induction_target
                        == IcuInductionTarget::MemAddress;
                sawReadRepeat |= queue.kind==QueueKind::Mem && queue.index==0 && repeat.inner_count==3 && repeat.inner_stride==1;
                sawWriteRepeat |= queue.kind==QueueKind::Mem && queue.index==1 && repeat.inner_count==3 && repeat.inner_stride==2;
                sawSequentialInnerStride |= queue.kind==QueueKind::Mem && queue.index==6 && repeat.inner_stride==1;
                sawRepeatedCandidate |= queue.kind==QueueKind::Mem && queue.index==8 && repeat.inner_count==2;
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
    require(sawMacro, "ICU Macro descriptor was not preserved");
    require(sawMxmAccumulatorMacro,
        "MXM accumulator-address Macro descriptor was not preserved");
    require(macroQueueCommands == 1,
        "ICU Macro was expanded instead of encoded as one descriptor");
    require(sawSequentialMacro,
        "sequential command queue did not use Macro");
    require(sawSequentialInnerStride,
        "Macro lost the inner MEM address stride");
    require(sawRepeatedCandidate,
        "Macro encoding lost a command's inner repeat space");

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
