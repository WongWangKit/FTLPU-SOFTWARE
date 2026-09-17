#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: mem_execution_trace_test output.mem.csv");

    using namespace ftlpu;
    using namespace ftlpu::software::runtime;

    IcuProgram schedule;
    schedule.emit_mem(1,
        InstructionControlUnit::mem_queue(Hemisphere::East, 2, 1),
        MemInstruction::Read(17, StreamId::East(3)));
    // Slice 2 emits at the right boundary of MEM group 0. Slice 4 consumes
    // that same eastbound stream at the left boundary of group 1 one cycle
    // later, exercising the write operation without host injection.
    schedule.emit_mem(2,
        InstructionControlUnit::mem_queue(Hemisphere::East, 4, 0),
        MemInstruction::Write(41, StreamId::East(3)));
    schedule.emit_mem(3,
        InstructionControlUnit::mem_queue(Hemisphere::West, 4, 0),
        MemInstruction::Read(29, StreamId::West(5)));

    BinaryProgram program;
    program.max_cycle = schedule.last_cycle();
    program.queues = schedule.encode_queues();

    TspSliceSystem system;
    CModelRuntime runtime(system);
    runtime.enable_mem_execution_trace();
    runtime.load(program);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            system.initialize_mem_sram_lane_byte(
                Hemisphere::East, 2, 1, tile, 17, lane,
                static_cast<std::uint8_t>(0x10 + tile * 8 + lane));
            system.initialize_mem_sram_lane_byte(
                Hemisphere::West, 4, 0, tile, 29, lane,
                static_cast<std::uint8_t>(0x80 + tile * 8 + lane));
        }
    }
    runtime.run_cycles(16);

    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    runtime.write_mem_execution_trace_csv(path);

    std::ifstream input(path);
    const std::string csv {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    require(csv.starts_with(
                "cycle,hemisphere,slice,bank,port,tile,stage,action,"),
        "MEM trace CSV header is missing");
    require(csv.find(",E,2,1,read,,\"icu\",\"issue\",\"Read\",17,E,3,")
            != std::string::npos,
        "east MEM ICU issue was not traced");
    require(csv.find(",W,4,0,read,,\"icu\",\"issue\",\"Read\",29,W,5,")
            != std::string::npos,
        "west MEM ICU issue was not traced");
    require(csv.find(",E,2,1,read,0,\"pipeline\",\"execute\",\"Read\",17,E,3,")
            != std::string::npos
            && csv.find(",E,2,1,read,3,\"pipeline\",\"execute\",\"Read\",17,E,3,")
                != std::string::npos,
        "MEM tile pipeline was not traced across all stages");
    require(csv.find(",E,2,1,read,0,\"sram\",\"read_to_sr\",\"Read\",17,E,3,")
            != std::string::npos
            && csv.find("1011121314151617") != std::string::npos,
        "MEM SRAM transfer payload was not traced");
    require(csv.find(",E,4,0,write,,\"icu\",\"issue\",\"Write\",41,E,3,")
            != std::string::npos
            && csv.find(",E,4,0,write,0,\"sram\",\"write_commit\",\"Write\",41,E,3,")
                != std::string::npos,
        "MEM SRAM write commit was not traced");
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            require(system.read_mem_sram_lane_byte(
                        Hemisphere::East, 4, 0, tile, 41, lane)
                    == static_cast<std::uint8_t>(0x10 + tile * 8 + lane),
                "traced MEM write did not commit the expected SRAM byte");

    std::cout << "mem_execution_trace_test passed: " << path << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "mem_execution_trace_test failed: "
              << error.what() << '\n';
    return 1;
}
