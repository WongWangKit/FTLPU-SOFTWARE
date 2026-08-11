#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv)
try {
    if (argc != 2 && argc != 4)
        throw std::runtime_error(
            "usage: ftlpu_binary_inspect program.ftlpu "
            "[--trace schedule.csv]");
    if (argc == 4 && std::string_view(argv[2]) != "--trace")
        throw std::runtime_error("expected --trace before the CSV path");

    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    const auto cycles = program.max_cycle + 64;
    std::cout << "binary target=" << program.target_name
              << " max_cycle=" << program.max_cycle
              << " measured_cycles=" << cycles
              << " mxms_per_hemisphere="
              << program.hardware.mxms_per_hemisphere
              << " vxm_alus=" << program.hardware.vxm_alus << '\n';
    ftlpu::software::runtime::print_runtime_performance(
        program, cycles, std::cout);
    if (argc == 4)
        ftlpu::software::runtime::write_schedule_trace_csv(
            program, std::filesystem::path(argv[3]));
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu_binary_inspect failed: " << ex.what() << '\n';
    return 1;
}
