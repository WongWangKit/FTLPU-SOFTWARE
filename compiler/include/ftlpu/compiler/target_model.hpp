#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ftlpu::compiler {

enum class Hemisphere {
    West = 0,
    East = 1,
};

enum class StreamDirection {
    East,
    West,
};

struct MemoryAddress {
    std::size_t device{0};
    Hemisphere hemisphere{Hemisphere::East};
    std::size_t slice{0};
    std::size_t bank{0};
    std::size_t word{0};
    std::size_t byte{0};
};

struct MemoryAllocation {
    std::string symbol;
    MemoryAddress base{};
    std::size_t bytes{0};
    std::size_t words{0};
};

class MemoryAllocator {
public:
    explicit MemoryAllocator(Hemisphere hemisphere = Hemisphere::East);

    MemoryAllocation allocate(std::string symbol, std::size_t bytes);
    const std::vector<MemoryAllocation>& allocations() const;

private:
    Hemisphere hemisphere_{Hemisphere::East};
    std::size_t next_word_{0};
    std::vector<MemoryAllocation> allocations_{};
};

struct StreamEndpoint {
    std::string kind;
    std::string name;
    std::optional<std::string> port{};
};

struct StreamBinding {
    std::string symbol;
    std::size_t stream_id{0};
    StreamDirection direction{StreamDirection::East};
    StreamEndpoint producer{};
    StreamEndpoint consumer{};
    std::size_t produce_cycle{0};
    std::size_t consume_cycle{0};
    std::size_t latency{0};
};

class StreamManager {
public:
    StreamBinding allocate(
        std::string symbol,
        StreamDirection direction,
        StreamEndpoint producer,
        StreamEndpoint consumer,
        std::size_t produce_cycle,
        std::size_t consume_cycle,
        std::size_t preferred_stream);

    const std::vector<StreamBinding>& bindings() const;

private:
    bool conflicts(std::size_t stream_id, std::size_t produce_cycle, std::size_t consume_cycle) const;

    std::vector<StreamBinding> bindings_{};
};

class ResourceScheduler {
public:
    std::size_t reserve(std::string resource, std::size_t earliest_cycle, std::size_t duration = 1);
    std::size_t available_after(const std::string& resource) const;

private:
    struct Reservation {
        std::string resource;
        std::size_t start{0};
        std::size_t end{0};
    };

    bool is_free(const std::string& resource, std::size_t start, std::size_t end) const;

    std::vector<Reservation> reservations_{};
};

std::size_t align_up(std::size_t value, std::size_t alignment);
std::size_t mem_to_mxm_latency(std::size_t mem_slice);
std::size_t mxm_to_vxm_latency();
std::size_t vxm_pipeline_latency(std::size_t stages);
std::size_t vxm_to_mem_latency(std::size_t mem_slice);
std::string format_address(const MemoryAddress& address);
std::string format_allocation(const MemoryAllocation& allocation);
std::string stream_direction_name(StreamDirection direction);
std::string format_endpoint(const StreamEndpoint& endpoint);

} // namespace ftlpu::compiler
