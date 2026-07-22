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
    require(program.max_cycle == 1, "VXM Repeat final cycle was not serialized");
    require(program.queues.size() == 1, "expected exactly one serialized VXM queue");
    require(program.queues[0].kind == QueueKind::Vxm && program.queues[0].index == 3,
        "serialized VXM queue identity is incorrect");

    bool has_four_word_instruction = false;
    for (const auto& command : program.queues[0].commands) {
        if (command.instruction_kind == InstructionKind::Vxm && command.word_count == 4)
            has_four_word_instruction = true;
    }
    require(has_four_word_instruction, "VXM command did not use the four-word codec");

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    auto runtime = CModelRuntime(*system);
    runtime.load(program);
    std::ostringstream log;
    runtime.dispatch_icu_cycles(program.max_cycle + 1, &log);
    constexpr auto dispatch = "ICU -> VXM.alu3 mul";
    const auto first_dispatch = log.str().find(dispatch);
    const auto second_dispatch = first_dispatch == std::string::npos
        ? std::string::npos
        : log.str().find(dispatch, first_dispatch + 1);
    if (second_dispatch == std::string::npos) {
        throw std::logic_error(
            "runtime did not dispatch and repeat the VXM instruction on ALU3; ICU log:\n"
            + log.str());
    }

    std::cout << "command_vxm_binary_runtime_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "command_vxm_binary_runtime_test failed: " << ex.what() << '\n';
    return 1;
}
