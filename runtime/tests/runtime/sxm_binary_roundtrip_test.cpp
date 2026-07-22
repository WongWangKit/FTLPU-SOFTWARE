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
    if (argc != 2) throw std::runtime_error("usage: sxm_binary_roundtrip_test program.ftlpu");

    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    IcuProgram icu_program;
    auto transpose = SxmInstruction::Transpose(
        {SxmStreamId {0}, SxmStreamId {1}}, {SxmStreamId {16}, SxmStreamId {17}});
    icu_program.emit_sxm_transpose(7, Hemisphere::East, transpose);

    BinaryProgram program;
    program.max_cycle = icu_program.last_cycle();
    program.queues = icu_program.encode_queues();
    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    write_binary_program(program, path);

    const auto decoded = read_binary_program(path);
    bool found = false;
    for (const auto& queue : decoded.queues) {
        if (queue.kind != QueueKind::SxmTranspose || queue.index != 0 || queue.commands.empty())
            continue;
        const auto& command = queue.commands.back();
        require(command.instruction_kind == InstructionKind::Sxm, "SXM instruction kind was not preserved");
        require(command.extension_words.size() == 2 + 2 + 2 + SxmInstruction::kTotalLanes,
            "SXM variable payload size was not preserved");
        found = true;
    }
    require(found, "serialized SXM transpose queue is missing");

    auto system = std::make_unique<TspSliceSystem>();
    CModelRuntime runtime(*system);
    runtime.load(decoded);
    std::cout << "sxm_binary_roundtrip_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "sxm_binary_roundtrip_test failed: " << ex.what() << '\n';
    return 1;
}
