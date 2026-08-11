#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <ostream>
#include <span>
#include <vector>

namespace ftlpu::software::runtime {

class CModelRuntime {
public:
    explicit CModelRuntime(TspSliceSystem& system);

    void load(const BinaryProgram& program);
    void load_file(const std::filesystem::path& path);
    void upload_input(std::size_t index, std::span<const std::uint8_t> data);
    void upload_binding(
        const BinaryBinding& binding, std::span<const std::uint8_t> data);
    std::vector<std::uint8_t> download_output(std::size_t index) const;
    void copy_binding(
        const BinaryBinding& source, const BinaryBinding& destination);
    void dispatch_icu_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void run_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void print_datapath_performance(std::ostream& os) const;

private:
    const BinaryBinding& find_binding(BindingAccess access, std::size_t index) const;

    TspSliceSystem& system_;
    std::size_t loaded_max_cycle_{0};
    std::size_t loaded_mxms_per_hemisphere_{hw::kMxmsPerHemisphere};
    std::size_t loaded_vxm_alus_{16};
    std::vector<BinaryBinding> bindings_;
    DatapathPerformanceMonitor datapath_performance_{};
};

} // namespace ftlpu::software::runtime
