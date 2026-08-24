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

} // namespace

int main()
try {
    BinaryProgram program;
    program.hardware.c2c_streams_per_direction = 8;
    program.hardware.c2c_bytes_per_stream_per_cycle = 32;
    program.bindings = {
        weight(1, "gate", 2048),
        weight(2, "up", 2048),
        weight(3, "down", 1024),
    };
    program.weight_page_uses = {
        {1, 0, 0, 10, 100},
        {2, 0, 0, 12, 102},
        {3, 0, 0, 200, 240},
    };

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
        "6,10,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=gate+up bytes=1024 lanes=8 bandwidth=256B/cycle "
        "planned=true\"");
    require_contains(trace,
        "6,10,\"C2C.W.Prefetch\",\"page=0 bank=0 "
        "bindings=gate+up bytes=1024 lanes=8 bandwidth=256B/cycle "
        "planned=true\"");
    require_contains(trace,
        "199,200,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=down bytes=256 lanes=8 bandwidth=256B/cycle "
        "planned=true\"");
    if (trace.find("bindings=gate+up+down") != std::string::npos)
        throw std::runtime_error(
            "non-overlapping Down page was merged with Gate/Up");

    std::cout << "schedule_trace_weight_page_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "schedule_trace_weight_page_test failed: "
              << error.what() << '\n';
    return 1;
}
