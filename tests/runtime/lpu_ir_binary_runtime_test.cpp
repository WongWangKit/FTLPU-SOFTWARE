#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/lpu_ir.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool contains(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

void require_contains(const std::string& text, const std::string& pattern)
{
    if (!contains(text, pattern)) {
        throw std::logic_error("missing runtime log pattern: " + pattern);
    }
}

} // namespace

int main()
{
    try {
        using namespace ftlpu::software::runtime;

        const auto ir = std::string {
            "# hand-written LPU IR: cycle queue instruction\n"
            "mem 3 0 read 100 0\n"
            "mem 4 1 write 200 32\n"
            "mxm_load 5 0 iw 0 0\n"
            "mxm_compute 7 0 compute 0\n"
            "mxm_output 9 0 output 32\n"
            "vxm 11 0 cast stream_i32 32 fp32\n"
        };

        const auto scheduled = parse_lpu_ir(ir);
        assert(!scheduled.empty());
        assert(scheduled.last_cycle() == 11);

        const auto binary_path = std::filesystem::path("lpu_ir_binary_runtime_test.ftlpu");
        write_binary_program(BinaryProgram {scheduled.last_cycle(), scheduled.encode_queues()}, binary_path);

        const auto loaded = read_binary_program(binary_path);
        assert(loaded.max_cycle == 11);
        assert(!loaded.queues.empty());

        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        auto runtime = CModelRuntime(*system);
        runtime.load_file(binary_path);

        std::ostringstream icu_log;
        runtime.dispatch_icu_cycles(loaded.max_cycle + 1, &icu_log);
        const auto log = icu_log.str();

        require_contains(log, "ICU -> MEM q0 Read address=100 stream=0");
        require_contains(log, "ICU -> MEM q1 Write address=200 stream=32");
        require_contains(log, "ICU -> MXM0.load IW b0 col=0");
        require_contains(log, "ICU -> MXM0.compute Compute b0");
        require_contains(log, "ICU -> MXM0.output Output stream_base=32");
        require_contains(log, "ICU -> VXM alu0 cast lhs=stream[32..35]");

        std::cout << "lpu_ir_binary_runtime_test passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "lpu_ir_binary_runtime_test failed: " << ex.what() << '\n';
        return 1;
    }
}
