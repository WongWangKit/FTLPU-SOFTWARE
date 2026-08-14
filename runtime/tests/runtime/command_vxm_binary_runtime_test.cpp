#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu::software::runtime;
    if (argc != 2) throw std::runtime_error("usage: command_vxm_binary_runtime_test program.ftlpu");

    const auto path = std::filesystem::path(argv[1]);
    const auto program = read_binary_program(path);
    require(program.max_cycle == 0,
        "VXM packet repeat must not become an ICU Repeat command");
    require(program.queues.size() == 1, "expected exactly one serialized VXM queue");
    require(program.queues[0].kind == QueueKind::Vxm && program.queues[0].index == 1,
        "serialized VXM queue identity is incorrect");

    bool hasCompactInstruction = false;
    for (const auto& command : program.queues[0].commands) {
        if (command.instruction_kind != InstructionKind::Vxm
            || command.word_count != 3)
            continue;
        const auto decoded = ftlpu::isa::decode_vxm_instruction(1,
            ftlpu::isa::EncodedVxmInstruction {
                static_cast<std::uint64_t>(command.words[0])
                    | (static_cast<std::uint64_t>(command.words[1]) << 32),
                command.words[2]});
        require(decoded.chain_depth == ftlpu::VxmChainDepth::Two,
            "VXM chain depth was not encoded");
        require(decoded.instruction.repeat_count == 2,
            "VXM packet repeat count was not encoded");
        hasCompactInstruction = true;
    }
    require(hasCompactInstruction,
        "VXM command did not use the 96-bit compact codec");

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    auto runtime = CModelRuntime(*system);
    runtime.load(program);

    std::cout << "command_vxm_binary_runtime_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "command_vxm_binary_runtime_test failed: " << ex.what() << '\n';
    return 1;
}
