#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ftlpu {
class TspSliceSystem;
}

namespace ftlpu::software::runtime {

// Records instructions that were actually issued by the modeled ICUs. This
// is deliberately separate from the static binary schedule trace: runtime
// stalls, DDR jitter, and event releases are represented in physical cycles.
class RuntimeExecutionTrace {
public:
    void reset(const BinaryProgram& program);
    void begin_segment(const BinaryProgram& program,
        std::int64_t cycle_offset, bool append);
    void sample(TspSliceSystem& system, std::uint64_t physical_cycle,
        bool program_issue_enabled,
        const BinaryWeightPageUse* waiting_page = nullptr);
    void record_interval(std::int64_t start_cycle, std::int64_t end_cycle,
        std::string resource, std::string detail,
        std::size_t issue_count = 1);
    void write_csv(const std::filesystem::path& path) const;
    bool empty() const noexcept { return events_.empty(); }

private:
    struct QueueRef {
        QueueKind kind{QueueKind::Mem};
        std::size_t index{0};
    };

    struct Event {
        std::int64_t start_cycle{0};
        std::int64_t end_cycle{0};
        std::string resource{};
        std::string detail{};
        std::size_t issue_count{0};
        std::size_t repeat_count{1};
        std::int64_t repeat_interval{0};
        std::size_t outer_count{1};
        std::int64_t outer_interval{0};
        std::size_t sequence{0};
    };

    std::vector<QueueRef> queues_{};
    std::vector<Event> events_{};
    std::unordered_map<std::string, std::size_t> last_event_by_resource_{};
    std::size_t mxms_per_hemisphere_{1};
    std::size_t sequence_{0};
    std::int64_t cycle_offset_{0};
};

// Sparse, cycle-accurate MEM trace. Idle queues are implicit; every ICU state
// transition, active pipeline tile and completed SRAM transfer is retained.
// Keeping the CSV sparse makes full decoder-layer traces practical while the
// viewer can still label each operation as a read or write at every selected
// cycle.
class MemExecutionTrace {
public:
    void begin_segment(std::int64_t cycle_offset, bool append);
    // Stream events directly to disk for traces too large to retain in RAM.
    // Event order is already cycle-major because sample() is called once per
    // physical cycle.
    void stream_csv(const std::filesystem::path& path);
    void sample(TspSliceSystem& system, std::uint64_t physical_cycle,
        bool program_issue_enabled);
    void write_csv(const std::filesystem::path& path) const;
    bool empty() const noexcept { return events_.empty(); }

private:
    struct Event {
        std::int64_t cycle{0};
        std::uint16_t hemisphere{0};
        std::uint16_t slice{0};
        std::uint16_t bank{0};
        std::uint16_t port{0};
        std::int16_t tile{-1};
        std::string stage{};
        std::string action{};
        std::string opcode{};
        std::int64_t address{-1};
        std::string stream_direction{};
        std::int32_t stream_index{-1};
        std::int64_t sr_column{-1};
        std::uint64_t vector_tag{0};
        bool has_vector_tag{false};
        std::string data_hex{};
        std::string source{};
        std::int64_t pc{-1};
        std::size_t iq_before{0};
        std::size_t iq_after{0};
    };

    static void write_header(std::ostream& output);
    static void write_event(std::ostream& output, const Event& event);
    void record(Event event);

    std::vector<Event> events_{};
    std::int64_t cycle_offset_{0};
    std::vector<char> stream_buffer_{};
    mutable std::ofstream stream_output_{};
    std::filesystem::path stream_path_{};
};

} // namespace ftlpu::software::runtime
