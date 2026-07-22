#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error("usage: compiled_sxm_command_runtime_test program.ftlpu");

    using namespace ftlpu::software::runtime;
    const auto program = read_binary_program(std::filesystem::path(argv[1]));
    require(program.queues.size() == 1, "expected one SXM queue in translated program");
    const auto& queue = program.queues.front();
    require(queue.kind == QueueKind::SxmTranspose && queue.index == 0,
        "translated SXM queue identity is incorrect");
    bool found_sxm = false;
    for (const QueueCommand& command : queue.commands) {
        if (command.instruction_kind != InstructionKind::Sxm) continue;
        require(command.word_count == 4, "SXM fixed header was not preserved");
        require(command.extension_words.size() == 2 + 4 + 2 + ftlpu::SxmInstruction::kTotalLanes,
            "SXM stream lists or lane map were not preserved");
        found_sxm = true;
    }
    require(found_sxm, "translated program has no SXM instruction");

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    CModelRuntime runtime(*system);
    runtime.load(program);
    std::cout << "compiled_sxm_command_runtime_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_sxm_command_runtime_test failed: " << ex.what() << '\n';
    return 1;
}
