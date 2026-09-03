#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
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
};

} // namespace ftlpu::software::runtime
