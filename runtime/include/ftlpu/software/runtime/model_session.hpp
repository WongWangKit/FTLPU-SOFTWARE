#pragma once

#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/session_memory_planner.hpp"
#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ftlpu::software::runtime {

struct ModelSessionStats {
    std::size_t resident_uploads{0};
    std::size_t resident_upload_bytes{0};
    std::size_t state_initializations{0};
    std::size_t state_initialization_bytes{0};
    std::size_t host_uploads{0};
    std::size_t host_downloads{0};
    std::size_t c2c_ingress_bytes{0};
    std::size_t c2c_ingress_cycles{0};
    std::size_t c2c_egress_bytes{0};
    std::size_t c2c_egress_cycles{0};
    std::size_t device_aliases{0};
    std::size_t device_copies{0};
    std::size_t device_copy_bytes{0};
    std::size_t host_operations{0};
    std::size_t weight_page_prefetches{0};
    std::size_t weight_page_prefetch_bytes{0};
    std::size_t weight_page_wait_cycles{0};
    std::size_t weight_page_initial_wait_cycles{0};
    std::size_t weight_page_boundary_wait_cycles{0};
    std::size_t weight_page_hidden_prefetches{0};
    std::size_t weight_page_deferred_prefetches{0};
    std::size_t weight_page_runtime_wait_cycles{0};
};

class ModelSession {
public:
    explicit ModelSession(TspSliceSystem& system);
    explicit ModelSession(C2cDmaSystem& system);

    void load(ModelPackage package);
    void load_file(const std::filesystem::path& path);
    void set_input(std::string name, std::span<const std::uint8_t> data);
    void run(std::size_t drain_cycles = 64);
    void run_invocation(std::size_t index, std::size_t drain_cycles = 64);
    void enable_execution_trace(bool enabled = true) noexcept;
    void write_execution_trace_csv(const std::filesystem::path& path) const;
    void set_ddr_peak_bandwidth_mbytes_per_second(
        std::uint32_t bandwidth);

    const ModelPackage& package() const;
    const std::vector<std::uint8_t>& value(const std::string& name) const;
    const SessionMemoryPlan& memory_plan() const;
    const ModelSessionStats& stats() const;
    std::vector<WeightPrefetchPlan> executable_weight_prefetch_plans() const;

private:
    struct DeviceValue {
        BinaryBinding binding{};
        std::uint64_t target_abi{0};
    };

    struct ExecutableWeightTransfer {
        WeightPrefetchPlan plan{};
        C2cWeightPage page{};
        C2cWeightPageFence fence{};
        std::vector<BinaryWeightPageUse> uses{};
        std::size_t launch_event_tag{0};
        std::optional<std::int64_t> actual_start_cycle{};
        std::optional<std::int64_t> actual_ready_cycle{};
        std::size_t pre_execution_cycles{0};
        bool launch_released{false};
        bool trace_recorded{false};
        bool ready_before_execution{false};
    };

    const std::vector<std::uint8_t>& resolve_value(const std::string& name) const;
    const ModelValue* find_value_metadata(const std::string& name) const;
    void run_embedding_lookups();
    void run_host_lm_heads();
    void prepare_weight_pages();
    void ensure_weight_page(std::uint32_t page_index);
    void start_weight_page(std::uint32_t page_index);
    void observe_weight_page_tick();
    void release_due_executable_weight_pages();
    void observe_executable_weight_page_tick();
    void record_weight_page_trace(ExecutableWeightTransfer& transfer);
    void prepare_executable_weight_pages(
        const BinaryProgram& program, const ModelInvocation& invocation);
    void schedule_executable_weight_pages();
    bool executable_weight_page_ready(
        const BinaryWeightPageUse& use) const;
    void configure_external_transport(
        const ExecutableHardwareConfig& hardware);
    ExecutableHardwareConfig effective_external_transport(
        const ExecutableHardwareConfig& hardware) const;
    void upload_binding_through_c2c(
        const BinaryBinding& binding, std::span<const std::uint8_t> data,
        const ExecutableHardwareConfig& hardware);
    std::vector<std::uint8_t> download_binding_through_c2c(
        const BinaryBinding& binding,
        const ExecutableHardwareConfig& hardware);

    CModelRuntime runtime_;
    C2cDmaSystem* c2c_system_{nullptr};
    std::unique_ptr<C2cWeightPager> weight_pager_{};
    std::vector<C2cWeightPage> c2c_pages_{};
    std::optional<std::uint32_t> ready_weight_page_{};
    std::optional<std::uint32_t> inflight_weight_page_{};
    std::vector<ExecutableWeightTransfer> executable_weight_transfers_{};
    std::uint64_t executable_cycle_{0};
    std::uint64_t executable_ddr4_address_{0};
    std::size_t c2c_bytes_per_stream_per_cycle_{0};
    std::optional<std::uint32_t>
        ddr_peak_bandwidth_mbytes_per_second_override_{};
    bool executable_clock_active_{false};
    bool execution_trace_enabled_{false};
    bool execution_trace_has_segment_{false};
    std::int64_t execution_trace_cycle_cursor_{0};
    ModelPackage package_{};
    SessionMemoryPlan memory_plan_{};
    ModelSessionStats stats_{};
    ModelSessionStats load_stats_{};
    bool loaded_{false};
    bool completed_invocation_{false};
    std::unordered_map<std::string, std::vector<std::uint8_t>> values_{};
    std::unordered_map<std::string, DeviceValue> device_values_{};
};

} // namespace ftlpu::software::runtime
