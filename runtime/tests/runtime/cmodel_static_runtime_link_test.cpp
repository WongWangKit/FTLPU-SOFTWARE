#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_link.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require_contains(const std::string& text, const std::string& pattern)
{
    if (text.find(pattern) == std::string::npos) {
        throw std::logic_error("missing CModel runtime log pattern: " + pattern);
    }
}

} // namespace

int main()
{
    try {
        using namespace ftlpu;
        using namespace ftlpu::software::runtime;

        assert(linked_cmodel_mxm_count() == 2);

        auto program = IcuProgram {};
        program.emit_mem(0, 0, MemInstruction::Read(0, 0));
        program.emit_mxm_load(2, 0, MxmControlInstruction::IW(0));
        program.emit_mxm_compute(4, 0, MxmControlInstruction::Compute(0, 0, 32));

        const auto binary_path = std::filesystem::path("cmodel_static_runtime_link_test.ftlpu");
        write_binary_program(BinaryProgram {program.last_cycle(), program.encode_queues()}, binary_path);

        auto dispatch_system = std::make_unique<TspSliceSystem>();
        auto dispatch_runtime = CModelRuntime(*dispatch_system);
        dispatch_runtime.load_file(binary_path);

        std::ostringstream dispatch_log;
        dispatch_runtime.dispatch_icu_cycles(6, &dispatch_log);
        const auto dispatch_text = dispatch_log.str();
        require_contains(dispatch_text, "ICU -> MEM q0 Read address=0 stream=0");
        require_contains(dispatch_text, "ICU -> MXM0.load IW b0");
        require_contains(dispatch_text, "ICU -> MXM0.compute Compute b0 stream=0 out=32");

        auto mem_only_program = IcuProgram {};
        mem_only_program.emit_mem(0, 0, MemInstruction::Read(0, 0));
        const auto mem_only_path = std::filesystem::path("cmodel_static_runtime_mem_tick.ftlpu");
        write_binary_program(
            BinaryProgram {mem_only_program.last_cycle(), mem_only_program.encode_queues()},
            mem_only_path);

        auto tick_system = std::make_unique<TspSliceSystem>();
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            tick_system->mem().set_sram_byte(0, 0, lane, static_cast<std::uint8_t>(lane + 1));
        }

        auto tick_runtime = CModelRuntime(*tick_system);
        tick_runtime.load_file(mem_only_path);

        std::ostringstream tick_log;
        tick_system->tick(TspSliceSystem::LogSinks {&tick_log, &tick_log, nullptr, nullptr, &tick_log, 0});
        require_contains(tick_log.str(), "c0.t0=Read(a=0,s=0)");

        std::cout << "cmodel_static_runtime_link_test passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cmodel_static_runtime_link_test failed: " << ex.what() << '\n';
        return 1;
    }
}
