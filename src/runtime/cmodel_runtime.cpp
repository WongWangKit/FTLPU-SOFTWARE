#include "ftlpu/software/runtime/cmodel_runtime.hpp"

namespace ftlpu::software::runtime {

CModelRuntime::CModelRuntime(TspSliceSystem& system)
    : system_(system)
{
}

void CModelRuntime::load(const BinaryProgram& program)
{
    load_queue_programs_into_icu(program.queues, system_.icu());
    loaded_max_cycle_ = program.max_cycle;
}

void CModelRuntime::load_file(const std::filesystem::path& path)
{
    load(read_binary_program(path));
}

void CModelRuntime::dispatch_icu_cycles(std::size_t cycles, std::ostream* log)
{
    const auto count = cycles == 0 ? loaded_max_cycle_ + 1 : cycles;
    for (std::size_t cycle = 0; cycle < count; ++cycle) {
        system_.dispatch_icu_only(log);
    }
}

} // namespace ftlpu::software::runtime
