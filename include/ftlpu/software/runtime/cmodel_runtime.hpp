#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstddef>
#include <filesystem>
#include <ostream>

namespace ftlpu::software::runtime {

class CModelRuntime {
public:
    explicit CModelRuntime(TspSliceSystem& system);

    void load(const BinaryProgram& program);
    void load_file(const std::filesystem::path& path);
    void dispatch_icu_cycles(std::size_t cycles, std::ostream* log = nullptr);

private:
    TspSliceSystem& system_;
    std::size_t loaded_max_cycle_{0};
};

} // namespace ftlpu::software::runtime
