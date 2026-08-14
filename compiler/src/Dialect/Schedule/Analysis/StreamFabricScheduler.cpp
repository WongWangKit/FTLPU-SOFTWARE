#include "ftlpu/compiler/Dialect/Schedule/Analysis/stream_fabric_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ftlpu::compiler::schedule {

StreamFabricScheduler::StreamFabricScheduler(
    int64_t column_count, int64_t streams_per_direction)
    : column_count_(column_count)
    , streams_per_direction_(streams_per_direction)
{
    if (column_count <= 0 || streams_per_direction <= 0)
        throw std::invalid_argument(
            "stream fabric dimensions must be positive");
}

std::size_t StreamFabricScheduler::CellKeyHash::operator()(
    const CellKey& key) const noexcept
{
    std::size_t value = static_cast<std::size_t>(key.column);
    value ^= static_cast<std::size_t>(key.stream) + 0x9e3779b9u
        + (value << 6) + (value >> 2);
    value ^= static_cast<std::size_t>(key.direction)
        + 0x9e3779b9u + (value << 6) + (value >> 2);
    return value;
}

llvm::SmallVector<int64_t> StreamFabricScheduler::path_columns(
    int64_t source_column, int64_t destination_column)
{
    llvm::SmallVector<int64_t> result;
    const int64_t step = source_column <= destination_column ? 1 : -1;
    for (int64_t column = source_column;; column += step) {
        result.push_back(column);
        if (column == destination_column) break;
    }
    return result;
}

std::uint64_t StreamFabricScheduler::beat_token(
    const StreamRouteWindow& route, int64_t beat)
{
    // Keep adjacent beats distinct while preserving deliberate multicast of
    // the same base token and beat through multiple consumers.
    return route.token_id + static_cast<std::uint64_t>(beat);
}

std::optional<StreamFabricConflict> StreamFabricScheduler::validate_route(
    int64_t start_cycle, const StreamRouteWindow& route) const
{
    if (start_cycle < 0 || route.source_column < 0
        || route.source_column >= column_count_
        || route.destination_column < 0
        || route.destination_column >= column_count_
        || route.stream_base < 0 || route.stream_count <= 0
        || route.stream_base + route.stream_count
            > streams_per_direction_
        || route.beat_count <= 0 || route.beat_interval <= 0) {
        throw std::invalid_argument("invalid stream fabric route window");
    }
    if (route.direction == target::StreamDirection::East
        && route.destination_column < route.source_column)
        throw std::invalid_argument("east stream route moves west");
    if (route.direction == target::StreamDirection::West
        && route.destination_column > route.source_column)
        throw std::invalid_argument("west stream route moves east");

    const auto columns = path_columns(
        route.source_column, route.destination_column);
    for (int64_t beat = 0; beat < route.beat_count; ++beat) {
        const auto token = beat_token(route, beat);
        const int64_t beat_cycle = start_cycle + beat * route.beat_interval;
        for (int64_t stream = route.stream_base;
             stream < route.stream_base + route.stream_count; ++stream) {
            for (std::size_t hop = 0; hop < columns.size(); ++hop) {
                const int64_t cycle = beat_cycle + static_cast<int64_t>(hop);
                const CellKey key {columns[hop], route.direction, stream};
                const auto cycle_it = reservations_.find(cycle);
                if (cycle_it == reservations_.end()) continue;
                const auto use_it = cycle_it->second.find(key);
                if (use_it == cycle_it->second.end()) continue;
                const auto& existing = use_it->second;
                if (existing.token_id != token) {
                    return StreamFabricConflict {cycle, columns[hop], stream,
                        route.direction, existing.token_id, token,
                        "different tokens occupy one SR cell"};
                }
                if (hop + 1 < columns.size() && existing.consumed) {
                    return StreamFabricConflict {cycle, columns[hop], stream,
                        route.direction, existing.token_id, token,
                        "token was consumed before the requested destination"};
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<StreamFabricConflict> StreamFabricScheduler::conflict(
    int64_t start_cycle, const StreamRouteWindow& route) const
{
    return validate_route(start_cycle, route);
}

bool StreamFabricScheduler::reserve_at(int64_t start_cycle,
    const StreamRouteWindow& route, StreamFabricConflict* conflict_out)
{
    if (auto found = validate_route(start_cycle, route)) {
        if (conflict_out != nullptr) *conflict_out = *found;
        return false;
    }

    const auto columns = path_columns(
        route.source_column, route.destination_column);
    for (int64_t beat = 0; beat < route.beat_count; ++beat) {
        const auto token = beat_token(route, beat);
        const int64_t beat_cycle = start_cycle + beat * route.beat_interval;
        for (int64_t stream = route.stream_base;
             stream < route.stream_base + route.stream_count; ++stream) {
            for (std::size_t hop = 0; hop < columns.size(); ++hop) {
                const int64_t cycle = beat_cycle + static_cast<int64_t>(hop);
                auto& use = reservations_[cycle][CellKey {
                    columns[hop], route.direction, stream}];
                use.token_id = token;
                if (hop + 1 == columns.size()) {
                    if (route.consumer_mode == StreamConsumerMode::Consume)
                        use.consumed = true;
                    else
                        use.tapped = true;
                }
            }
        }
    }
    return true;
}

int64_t StreamFabricScheduler::reserve(
    int64_t earliest_cycle, const StreamRouteWindow& route)
{
    if (earliest_cycle < 0)
        throw std::invalid_argument("stream route cannot start before cycle zero");
    for (int64_t cycle = earliest_cycle;; ++cycle) {
        if (reserve_at(cycle, route)) return cycle;
        if (cycle == std::numeric_limits<int64_t>::max())
            throw std::overflow_error("stream route cycle search overflowed");
    }
}

bool StreamFabricScheduler::is_reserved(int64_t cycle, int64_t column,
    target::StreamDirection direction, int64_t stream) const
{
    const auto cycle_it = reservations_.find(cycle);
    if (cycle_it == reservations_.end()) return false;
    return cycle_it->second.contains(CellKey {column, direction, stream});
}

std::uint64_t StreamFabricScheduler::allocate_token_range(int64_t beat_count)
{
    if (beat_count <= 0)
        throw std::invalid_argument("stream token range must be positive");
    const auto count = static_cast<std::uint64_t>(beat_count);
    if (next_token_id_ > std::numeric_limits<std::uint64_t>::max() - count)
        throw std::overflow_error("stream token identifier overflowed");
    const auto first = next_token_id_;
    next_token_id_ += count;
    return first;
}

} // namespace ftlpu::compiler::schedule
