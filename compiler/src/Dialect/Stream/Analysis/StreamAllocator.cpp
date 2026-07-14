#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_allocator.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace ftlpu::compiler::stream {

Binding StreamAllocator::allocate(
    std::string symbol,
    Direction direction,
    Endpoint producer,
    Endpoint consumer,
    std::size_t produce_cycle,
    std::size_t consume_cycle,
    std::size_t preferred_stream)
{
    if (consume_cycle < produce_cycle) {
        throw std::runtime_error("stream consume cycle is earlier than produce cycle");
    }

    for (std::size_t offset = 0; offset < 64; ++offset) {
        const auto candidate = (preferred_stream + offset) % 64;
        if (!conflicts(candidate, produce_cycle, consume_cycle)) {
            auto binding = Binding {
                std::move(symbol),
                candidate,
                direction,
                std::move(producer),
                std::move(consumer),
                produce_cycle,
                consume_cycle,
                consume_cycle - produce_cycle,
            };
            bindings_.push_back(binding);
            return binding;
        }
    }
    throw std::runtime_error("no stream id is free for requested lifetime");
}

const std::vector<Binding>& StreamAllocator::bindings() const
{
    return bindings_;
}

bool StreamAllocator::conflicts(std::size_t stream_id, std::size_t produce_cycle, std::size_t consume_cycle) const
{
    for (const auto& binding : bindings_) {
        if (binding.stream_id != stream_id) {
            continue;
        }
        if (produce_cycle < binding.consume_cycle && consume_cycle > binding.produce_cycle) {
            return true;
        }
    }
    return false;
}

std::string direction_name(Direction direction)
{
    return direction == Direction::East ? "E" : "W";
}

std::string format_endpoint(const Endpoint& endpoint)
{
    std::ostringstream os;
    os << endpoint.kind << ":" << endpoint.name;
    if (endpoint.port.has_value()) {
        os << ":" << *endpoint.port;
    }
    return os.str();
}

} // namespace ftlpu::compiler::stream
