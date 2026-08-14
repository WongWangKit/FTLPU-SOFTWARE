#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

namespace ftlpu::compiler::schedule {

enum class StreamConsumerMode {
    Consume,
    Tap,
};

// One vector beat follows a stream through one SR column per cycle. The
// current Schedule IR does not expose tile/lane skew, so a complete physical
// vector beat is the smallest schedulable unit here.
struct StreamRouteWindow {
    target::StreamDirection direction{target::StreamDirection::East};
    int64_t source_column{0};
    int64_t destination_column{0};
    int64_t stream_base{0};
    int64_t stream_count{1};
    int64_t beat_count{1};
    int64_t beat_interval{1};
    std::uint64_t token_id{0};
    StreamConsumerMode consumer_mode{StreamConsumerMode::Consume};
};

struct StreamFabricConflict {
    int64_t cycle{0};
    int64_t column{0};
    int64_t stream{0};
    target::StreamDirection direction{target::StreamDirection::East};
    std::uint64_t existing_token{0};
    std::uint64_t requested_token{0};
    std::string reason;
};

class StreamFabricScheduler {
public:
    explicit StreamFabricScheduler(int64_t column_count,
        int64_t streams_per_direction);

    std::optional<StreamFabricConflict> conflict(
        int64_t start_cycle, const StreamRouteWindow& route) const;
    bool reserve_at(int64_t start_cycle, const StreamRouteWindow& route,
        StreamFabricConflict* conflict = nullptr);
    int64_t reserve(int64_t earliest_cycle, const StreamRouteWindow& route);

    bool is_reserved(int64_t cycle, int64_t column,
        target::StreamDirection direction, int64_t stream) const;
    std::uint64_t allocate_token_range(int64_t beat_count);

    static llvm::SmallVector<int64_t> path_columns(
        int64_t source_column, int64_t destination_column);

private:
    struct CellKey {
        int64_t column{0};
        target::StreamDirection direction{target::StreamDirection::East};
        int64_t stream{0};

        bool operator==(const CellKey&) const = default;
    };

    struct CellKeyHash {
        std::size_t operator()(const CellKey& key) const noexcept;
    };

    struct CellUse {
        std::uint64_t token_id{0};
        bool consumed{false};
        bool tapped{false};
    };

    using CycleUses = std::unordered_map<CellKey, CellUse, CellKeyHash>;

    std::optional<StreamFabricConflict> validate_route(
        int64_t start_cycle, const StreamRouteWindow& route) const;
    static std::uint64_t beat_token(
        const StreamRouteWindow& route, int64_t beat);

    int64_t column_count_{0};
    int64_t streams_per_direction_{0};
    std::map<int64_t, CycleUses> reservations_;
    std::uint64_t next_token_id_{1};
};

} // namespace ftlpu::compiler::schedule
