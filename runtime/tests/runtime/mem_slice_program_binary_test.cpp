#include "ftlpu/software/runtime/binary.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: mem_slice_program_binary_test program.ftlpu");

    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    const IcuMemSliceProgram sliceProgram {
        IcuStreamNdSchedule {
            2, 1, {3, 1, 1}, {2, 1, 1}, {0, 0, 0},
            IcuInductionTarget::None},
        {
            IcuMemSliceProgramEntry {
                0, {1, 0, 0}, MemInstruction::Read(100, 0)},
            IcuMemSliceProgramEntry {
                1, {2, 0, 0}, MemInstruction::Write(200, 1)},
        },
    };

    BinaryProgram program;
    program.max_cycle = 7;
    program.queues.push_back(QueueProgram {QueueKind::Mem, 0,
        {encode_mem_slice_program_command(sliceProgram)}});

    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    write_binary_program(program, path);
    const auto decoded = read_binary_program(path);
    require(decoded.queues.size() == 1
            && decoded.queues[0].commands.size() == 1,
        "MEM_SLICE_PROGRAM command count was not preserved");
    require(is_mem_slice_program_command(
                decoded.queues[0].commands.front()),
        "MEM_SLICE_PROGRAM did not survive binary round-trip");
    const auto descriptor = decode_mem_slice_program_command(
        decoded.queues[0].commands.front());
    require(descriptor.schedule.start_cycle == 2
            && descriptor.schedule.counts[0] == 3
            && descriptor.body.size() == 2
            && descriptor.body[0].instruction.address == 100
            && descriptor.body[1].instruction.address == 200,
        "MEM_SLICE_PROGRAM payload changed during binary round-trip");

    InstructionControlUnit icu;
    load_queue_programs_into_icu(decoded.queues, icu);
    std::vector<std::pair<std::size_t, MemInstruction>> issues;
    for (std::size_t cycle = 0; cycle <= 8; ++cycle) {
        if (const auto issued = icu.mem_iq(0).tick())
            issues.emplace_back(cycle, *issued);
    }
    require(issues.size() == 6
            && issues[0].first == 2
            && issues[0].second.opcode == MemOpcode::Read
            && issues[0].second.address == 100
            && issues[1].first == 3
            && issues[1].second.opcode == MemOpcode::Write
            && issues[1].second.address == 200
            && issues[4].first == 6
            && issues[4].second.address == 102
            && issues[5].first == 7
            && issues[5].second.address == 204,
        "runtime loaded MEM_SLICE_PROGRAM with incorrect issue semantics");

    std::cout << "mem_slice_program_binary_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "mem_slice_program_binary_test failed: "
              << error.what() << '\n';
    return 1;
}
