#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/runtime_execution_trace.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ftlpu {
class C2cDmaSystem;
}

namespace ftlpu::software::runtime {

class CModelRuntime {
public:
    explicit CModelRuntime(TspSliceSystem& system);
    CModelRuntime(TspSliceSystem& system,
        std::function<void(TspSliceSystem::LogSinks)> tick);
    CModelRuntime(C2cDmaSystem& system,
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
    void set_weight_page_residency_checker(
        std::function<bool(const BinaryWeightPageUse&)> checker);
    void enable_execution_trace(bool enabled = true) noexcept;
    void configure_execution_trace_segment(
        std::int64_t cycle_offset, bool append) noexcept;
    void record_execution_trace_interval(std::int64_t start_cycle,
        std::int64_t end_cycle, std::string resource, std::string detail,
        std::size_t issue_count = 1);
    void write_execution_trace_csv(const std::filesystem::path& path) const;
    std::size_t physical_cycles() const noexcept { return physical_cycles_; }
    std::size_t logical_cycles() const noexcept { return executed_cycles_; }
    std::size_t instruction_prefill_cycles() const noexcept
    {
        return instruction_prefill_cycles_;
    }
    void dispatch_icu_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void run_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void print_datapath_performance(std::ostream& os) const;

private:
    const BinaryBinding& find_binding(BindingAccess access, std::size_t index) const;
    bool load_ready_weight_pages();
    void run_logical_cycles(std::size_t cycles,
        TspSliceSystem::LogSinks sinks);

    TspSliceSystem& system_;
    C2cDmaSystem* c2c_system_{nullptr};
    std::size_t loaded_max_cycle_{0};
    std::size_t loaded_mxms_per_hemisphere_{hw::kMxmsPerHemisphere};
    std::size_t loaded_vxm_alus_{hw::kVxmAluCount};
    std::vector<BinaryBinding> bindings_;
    ExecutableHardwareConfig hardware_{};
    std::vector<BinaryWeightPageUse> weight_page_uses_{};
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>
        paged_weight_data_{};
    std::size_t next_weight_page_use_{0};
    std::size_t executed_cycles_{0};
    std::size_t physical_cycles_{0};
    std::size_t instruction_prefill_cycles_{0};
    std::optional<BinaryWeightPageUse> waiting_weight_page_use_{};
    DatapathPerformanceMonitor datapath_performance_{};
    RuntimeExecutionTrace execution_trace_{};
    bool execution_trace_enabled_{false};
    std::int64_t execution_trace_cycle_offset_{0};
    bool execution_trace_append_on_load_{false};
    std::function<void(TspSliceSystem::LogSinks)> tick_{};
    std::function<bool(const BinaryWeightPageUse&)>
        weight_page_residency_checker_{};
};

} // namespace ftlpu::software::runtime
