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
    bool sawReadWrite[3] = {};
    bool sawDequant = false;
    bool sawInt8Load = false;
    bool sawBlock8 = false;
    for (const QueueProgram& queue : program.queues) {
        for (const QueueCommand& command : queue.commands) {
            if (command.instruction_kind == InstructionKind::Mem) {
                const auto word =
                    static_cast<isa::EncodedMemInstruction>(
                        command.words[0])
                    | (static_cast<isa::EncodedMemInstruction>(
                           command.words[1])
                        << 32);
                const MemInstruction instruction =
                    isa::decode_mem_instruction(word);
                for (int repeat = 0; repeat < 3; ++repeat)
                    sawReadWrite[repeat] |=
                        instruction.opcode == MemOpcode::ReadWrite
                        && instruction.address == 10 + repeat
                        && instruction.write_address
                            == 20 + 2 * repeat
                        && instruction.stream == 0
                        && instruction.write_stream == 32;
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
                sawBlock8 |=
                    instruction.opcode == MxmControlOpcode::Compute
                    && instruction.compute_mode == MxmComputeMode::Block8;
            }
        }
    }
    require(sawReadWrite[0] && sawReadWrite[1]
            && sawReadWrite[2],
        "different-stride MEM repeats were not expanded and merged");
    require(sawDequant, "MXM dequant queue scale was not preserved");
    require(sawInt8Load, "MXM INT8 dequant load mode was not preserved");
    require(sawBlock8, "MXM Block8 compute mode was not preserved");

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
