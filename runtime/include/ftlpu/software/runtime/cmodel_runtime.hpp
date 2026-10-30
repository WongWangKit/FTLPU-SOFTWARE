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
#include <unordered_set>
#include <vector>

namespace ftlpu {
class C2cDmaSystem;
}

namespace ftlpu::software::runtime {

struct QueueExecutionProgress {
    QueueKind kind{};
    std::size_t index{}, invocation_index{}, queue_cycle{}, physical_cycles{}, wait_cycles{}, issued{};
    bool local_cycle{false}, done{false};
};

class CModelRuntime {
public:
    explicit CModelRuntime(TspSliceSystem& system);
    explicit CModelRuntime(C2cDmaSystem& system);
    CModelRuntime(TspSliceSystem& system,
        std::function<void(TspSliceSystem::LogSinks)> tick);
    CModelRuntime(C2cDmaSystem& system,
        std::function<void(TspSliceSystem::LogSinks)> tick);

    void load(const BinaryProgram& program,
        bool initialize_internal_data = true);
    void initialize_internal_data(const BinaryProgram& program);
    // Legacy load() directly installs semantic/physical queues and primes IQ.
    // load_timed() submits a real DDR image and returns before any i-MEM write.
    void load_direct(const BinaryProgram& program) { load(program); }
    void load_timed(const BinaryProgram& program,
        std::uint64_t image_base = 0x1'0000'0000ULL);
    // Startup-only dependency gate; common barriers still govern page transitions.
    void set_program_start_dependency_checker(std::function<bool()> checker)
    { program_start_dependency_checker_ = std::move(checker); }
    bool all_images_committed() const;
    bool ready_to_start() const;
    const ProgramLoadStatistics& program_load_statistics() const;
    void print_program_load_performance(std::ostream& os) const;
    void load_file(const std::filesystem::path& path);
    void upload_input(std::size_t index, std::span<const std::uint8_t> data);
    void upload_binding(
        const BinaryBinding& binding, std::span<const std::uint8_t> data);
    std::vector<std::uint8_t> download_binding(
        const BinaryBinding& binding) const;
    std::vector<std::uint8_t> download_output(std::size_t index) const;
    void copy_binding(
        const BinaryBinding& source, const BinaryBinding& destination);
    void enable_execution_trace(bool enabled = true) noexcept;
    void configure_execution_trace_segment(
        std::int64_t cycle_offset, bool append) noexcept;
    // Trace a standalone transport ICU image that runs before or after the
    // main executable.  ModelSession uses this for cold-start/page/input C2C
    // traffic so the exported physical timeline begins at the first DMA/RX/
    // MEM_WRITE_SYNC issue rather than at Session.Invocation.
    void begin_external_execution_trace_segment(const BinaryProgram& program,
        std::int64_t cycle_offset, bool append);
    void sample_external_execution_trace_cycle(
        std::uint64_t physical_cycle, bool program_issue_enabled = true);
    void record_execution_trace_interval(std::int64_t start_cycle,
        std::int64_t end_cycle, std::string resource, std::string detail,
        std::size_t issue_count = 1);
    void write_execution_trace_csv(const std::filesystem::path& path) const;
    void enable_mem_execution_trace(bool enabled = true) noexcept;
    void stream_mem_execution_trace_csv(const std::filesystem::path& path);
    void write_mem_execution_trace_csv(
        const std::filesystem::path& path) const;
    std::uint64_t execution_generation() const noexcept { return execution_generation_; }
    void notify_common_page_commit(std::uint32_t instance);
    void notify_common_page_commit(std::uint64_t generation, std::uint32_t instance);
    bool common_transport_launched(std::uint32_t launch_event) const;
    bool common_transport_committed(std::uint32_t launch_event) const;
    const std::vector<QueueExecutionProgress>& queue_execution_progress() const noexcept { return queue_progress_; }
    bool queue_release_complete(std::uint32_t event) const;
    bool queue_control_protocol() const noexcept { return queue_control_protocol_; }
    bool queue_completion_ready(std::uint32_t event) const { return system_.icu().queue_completion_ready(event); }
    bool common_barrier_protocol() const noexcept { return common_barrier_protocol_; }
    struct PagingOverlapStatistics {
        std::size_t compute_active{}, compute_dma{}, compute_rx{}, compute_commit{}, commit_segments{};
    };
    const PagingOverlapStatistics& paging_overlap_statistics() const noexcept { return paging_overlap_; }
    std::size_t physical_cycles() const noexcept { return physical_cycles_; }
    std::size_t logical_cycles() const noexcept { return executed_cycles_; }
    std::size_t instruction_prefill_cycles() const noexcept
    {
        return instruction_prefill_cycles_;
    }
    void dispatch_icu_cycles(std::size_t cycles, std::ostream* log = nullptr);
    void run_cycles(std::size_t cycles, std::ostream* log = nullptr);
    IcuFrontendStatistics icu_frontend_statistics() const noexcept;
    std::vector<IcuMacroQueueFrontendStatistics> icu_macro_queue_frontend_statistics() const;
    void print_icu_frontend_performance(std::ostream& os) const;
    void print_datapath_performance(std::ostream& os) const;

private:
    void load_impl(const BinaryProgram& program, bool timed, std::uint64_t image_base,
        bool initialize_internal_data);
    void upload_binding_impl(const BinaryBinding& binding, std::span<const std::uint8_t> data);
    bool inputs_ready() const;
    const BinaryBinding& find_binding(BindingAccess access, std::size_t index) const;
    void load_ready_weight_pages();
    void configure_common_transport(const BinaryProgram& program);
    void configure_queue_transport(const BinaryProgram& program);
    void advance_common_transport();
    void reload_drained_compute_imem();
    std::vector<BinaryProgram> compute_imem_segments_{};
    std::vector<std::pair<QueueKind, std::size_t>> compute_imem_queues_{};
    std::size_t compute_imem_segment_{0}, compute_imem_boundary_{0};
    void advance_queue_releases();
    void sample_queue_progress();
    std::vector<QueueExecutionProgress> queue_progress_{};
    struct QueueRelease {
        IcuLocation location;
        std::size_t phase_offset{0};
        bool activation_producer{false};
    };
    struct QueueReleaseEpoch {
        std::uint32_t event_tag{0};
        std::vector<QueueRelease> releases{};
        std::size_t next_release{0};
        std::optional<std::int64_t> actual_ready_cycle{}, release_origin_cycle{};
        bool released{false};
    };
    std::vector<QueueReleaseEpoch> queue_releases_{};
    struct CommonTransportPage {
        std::uint32_t launch_event{0}, stop_cycle{0};
        std::uint32_t launch_cycle{0};
        std::vector<std::pair<std::size_t, std::size_t>> mem_commit_targets{}, mem_descriptors{};
        std::uint16_t hemisphere_mask{0};
        std::vector<std::uint32_t> instances{};
        std::vector<std::pair<std::size_t, std::size_t>> mem_issue_targets{};
        std::optional<std::size_t> writes_issued_cycle{};
        bool launched{false}, committed{false};
    };
    std::vector<CommonTransportPage> common_transports_{};
    void run_logical_cycles(std::size_t cycles,
        TspSliceSystem::LogSinks sinks);

    TspSliceSystem& system_;
    C2cDmaSystem* c2c_system_{nullptr};
    std::uint64_t execution_generation_{0};
    bool queue_control_protocol_{false};
    bool common_barrier_protocol_{false};
    bool transport_overlap_protocol_{false};
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
    PagingOverlapStatistics paging_overlap_{};
    std::size_t instruction_prefill_cycles_{0};
    bool timed_program_load_{false};
    std::function<bool()> program_start_dependency_checker_{};
    std::unordered_set<std::size_t> uploaded_inputs_{};
    DatapathPerformanceMonitor datapath_performance_{};
    RuntimeExecutionTrace execution_trace_{};
    MemExecutionTrace mem_execution_trace_{};
    bool execution_trace_enabled_{false};
    bool mem_execution_trace_enabled_{false};
    std::int64_t execution_trace_cycle_offset_{0};
    bool execution_trace_append_on_load_{false};
    std::function<void(TspSliceSystem::LogSinks)> tick_{};
};

} // namespace ftlpu::software::runtime
