#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ftlpu::compiler::stream {

enum class Direction {
    East,
    West,
};

struct Endpoint {
    std::string kind;
    std::string name;
    std::optional<std::string> port{};
};

struct Binding {
    std::string symbol;
    std::size_t stream_id{0};
    Direction direction{Direction::East};
    Endpoint producer{};
    Endpoint consumer{};
    std::size_t produce_cycle{0};
    std::size_t consume_cycle{0};
    std::size_t latency{0};
};

class StreamAllocator {
public:
    Binding allocate(
        std::string symbol,
        Direction direction,
        Endpoint producer,
        Endpoint consumer,
        std::size_t produce_cycle,
        std::size_t consume_cycle,
        std::size_t preferred_stream);

    const std::vector<Binding>& bindings() const;

private:
    bool conflicts(std::size_t stream_id, std::size_t produce_cycle, std::size_t consume_cycle) const;

    std::vector<Binding> bindings_{};
};

std::string direction_name(Direction direction);
std::string format_endpoint(const Endpoint& endpoint);

} // namespace ftlpu::compiler::stream
