#include "ftlpu/software/runtime/binary.hpp"

#include <cstdint>
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
            "usage: command_mxm_column_binary_test program.ftlpu");

    const auto program = read_binary_program(std::filesystem::path(argv[1]));
    require(program.queues.size() == 1, "expected one MXM load queue");
    const auto& queue = program.queues.front();
    require(queue.kind == QueueKind::MxmLoad && queue.index == 1,
        "serialized MXM load queue identity is incorrect");
    require(queue.commands.size() == 1, "expected one MXM load command");

    const auto& command = queue.commands.front();
    require(command.instruction_kind == InstructionKind::Mxm,
        "command does not contain an MXM instruction");
    const auto encoded = static_cast<isa::EncodedMxmInstruction>(
        static_cast<std::uint64_t>(command.words[0])
        | (static_cast<std::uint64_t>(command.words[1]) << 32));
    const auto instruction = isa::decode_mxm_instruction(encoded);
    require(instruction.opcode == MxmControlOpcode::IW,
        "decoded instruction is not IW");
    require(instruction.weight_load_mode == MxmWeightLoadMode::Column,
        "MXM column load mode was lost during binary translation");
    require(instruction.weight_buffer == 1, "wrong MXM weight buffer");
    require(instruction.weight_column == 2, "wrong MXM outer weight column");
    require(instruction.weight_inner_column == 6,
        "wrong MXM inner weight column");
    require(instruction.weight_input_mode
            == MxmWeightInputMode::Direct16,
        "legacy compiler weight path must explicitly encode Direct16");

    std::cout << "command_mxm_column_binary_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "command_mxm_column_binary_test failed: " << ex.what() << '\n';
    return 1;
}
