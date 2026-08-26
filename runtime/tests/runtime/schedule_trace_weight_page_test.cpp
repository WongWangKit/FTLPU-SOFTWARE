#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using namespace ftlpu::software::runtime;

BinaryBinding weight(
    std::uint32_t index, std::string name, std::uint64_t bytes)
{
    BinaryBinding binding;
    binding.index = index;
    binding.access = BindingAccess::Input;
    binding.role = "weight";
    binding.name = std::move(name);
    binding.byte_size = bytes;
    binding.hemisphere_mask = 3;
    binding.paged_weight = true;
    binding.page_count = 2;
    binding.page_bank_count = 2;
    return binding;
}

void require_contains(const std::string& text, const std::string& expected)
{
    if (text.find(expected) == std::string::npos)
        throw std::runtime_error("trace is missing: " + expected);
}

QueueCommand mxm_command(const ftlpu::MxmControlInstruction& instruction)
{
    const auto encoded = ftlpu::isa::encode_mxm_instruction(instruction);
    return QueueCommand {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
}

} // namespace

int main()
try {
    BinaryProgram program;
    program.hardware.c2c_streams_per_direction = 8;
    program.hardware.c2c_bytes_per_stream_per_cycle = 32;
    program.hardware.lpu_clock_mhz = 500;
    program.hardware.ddr_peak_bandwidth_mbytes_per_second = 51200;
    program.hardware.ddr_scheduling_efficiency_percent = 90;
    program.hardware.ddr_read_latency_cycles = 35;
    program.hardware.ddr_read_latency_jitter_cycles = 15;
    program.bindings = {
        weight(1, "gate", 2048),
        weight(2, "up", 2048),
        weight(4, "reuse", 1024),
        weight(3, "down", 1024),
    };
    program.weight_page_uses = {
        {1, 0, 0, 200, 300},
        {2, 0, 0, 202, 302},
        {4, 0, 0, 450, 500},
        {3, 0, 0, 700, 750},
    };
    program.hardware.mxms_per_hemisphere = 1;
    program.queues.push_back(QueueProgram {
        QueueKind::MxmLoad, 0,
        {encode_macro_schedule_command(
            mxm_command(ftlpu::MxmControlInstruction::IW(0, 3)),
            ftlpu::IcuMacroSchedule {10, 3, 2, 1, 1, 1, 0,
                ftlpu::IcuInductionTarget::MxmWeightColumn})}});

    const auto path = std::filesystem::temp_directory_path()
        / "ftlpu_schedule_trace_weight_page_test.csv";
    write_schedule_trace_csv(program, path);
    std::ostringstream contents;
    {
        std::ifstream input(path);
        contents << input.rdbuf();
    }
    std::filesystem::remove(path);
    const auto trace = contents.str();

    require_contains(trace,
        "0,137,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=gate+up bytes=1024 lanes=8 bandwidth=256B/cycle "
        "deadline=200 scheduled=true\"");
    require_contains(trace,
        "0,137,\"C2C.W.Prefetch\",\"page=0 bank=0 "
        "bindings=gate+up bytes=1024 lanes=8 bandwidth=256B/cycle "
        "deadline=200 scheduled=true\"");
    require_contains(trace,
        "330,450,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=reuse bytes=256 lanes=8 bandwidth=256B/cycle "
        "deadline=450 scheduled=true\"");
    require_contains(trace,
        "580,700,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=down bytes=256 lanes=8 bandwidth=256B/cycle "
        "deadline=700 scheduled=true\"");
    if (trace.find("bindings=gate+up+reuse") != std::string::npos)
        throw std::runtime_error(
            "non-overlapping reuse page was merged with Gate/Up");
    require_contains(trace,
        "10,11,\"MXM.E0.Load\",\"IW buffer=0 column=3\"");
    require_contains(trace,
        "12,13,\"MXM.E0.Load\",\"IW buffer=0 column=4\"");
    require_contains(trace,
        "14,15,\"MXM.E0.Load\",\"IW buffer=0 column=5\"");

    std::cout << "schedule_trace_weight_page_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "schedule_trace_weight_page_test failed: "
              << error.what() << '\n';
    return 1;
}
