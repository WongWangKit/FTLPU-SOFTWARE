#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <ostream>
#include <span>
#include <unordered_map>
#include <vector>

namespace ftlpu::software::runtime {

class CModelRuntime {
public:
    explicit CModelRuntime(TspSliceSystem& system);
    CModelRuntime(TspSliceSystem& system,
        std::function<void(TspSliceSystem::LogSinks)> tick);

    void load(const BinaryProgram& program);
    void load_file(const std::filesystem::path& path);
    void upload_input(std::size_t index, std::span<const std::uint8_t> data);
    void upload_binding(
        const BinaryBinding& binding, std::span<const std::uint8_t> data);
    std::vector<std::uint8_t> download_binding(
        const BinaryBinding& binding) const;
    std::vector<std::uint8_t> download_output(std::size_t index) const;
    void copy_binding(
        const BinaryBinding& source, const BinaryBinding& destination);
    void dispatch_icu_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void run_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void print_datapath_performance(std::ostream& os) const;

private:
    const BinaryBinding& find_binding(BindingAccess access, std::size_t index) const;
    void load_ready_weight_pages();

    TspSliceSystem& system_;
    std::size_t loaded_max_cycle_{0};
    std::size_t loaded_mxms_per_hemisphere_{hw::kMxmsPerHemisphere};
    std::size_t loaded_vxm_alus_{16};
    std::vector<BinaryBinding> bindings_;
    ExecutableHardwareConfig hardware_{};
    std::vector<BinaryWeightPageUse> weight_page_uses_{};
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>
        paged_weight_data_{};
    std::size_t next_weight_page_use_{0};
    std::size_t executed_cycles_{0};
    DatapathPerformanceMonitor datapath_performance_{};
    std::function<void(TspSliceSystem::LogSinks)> tick_{};
};

} // namespace ftlpu::software::runtime
