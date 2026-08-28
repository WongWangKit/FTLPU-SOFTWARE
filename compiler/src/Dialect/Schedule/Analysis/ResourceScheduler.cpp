#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"

#include <algorithm>
#include <vector>

namespace ftlpu::compiler::schedule {

int64_t ResourceScheduler::find_earliest(int64_t earliest_cycle,
    llvm::ArrayRef<ResourceWindow> windows) const
{
    int64_t anchor = earliest_cycle;
    for (;;) {
        int64_t next_anchor = anchor;
        for (const auto& window : windows) {
            const int64_t start = anchor + window.offset;
            const int64_t end = start + window.duration;
            const auto reservations = reservations_.find(window.resource);
            if (reservations == reservations_.end()) continue;
            const auto& intervals = reservations->second;
            auto position = intervals.lower_bound(start);
            if (position != intervals.begin()) --position;
            for (; position != intervals.end() && position->first < end;
                 ++position) {
                if (start < position->second && end > position->first)
                    next_anchor = std::max(next_anchor,
                        position->second - window.offset);
            }
        }
        if (next_anchor == anchor) break;
        anchor = next_anchor;
    }
    return anchor;
}

bool ResourceScheduler::is_free_at(int64_t cycle,
    llvm::ArrayRef<ResourceWindow> windows) const
{
    if (has_internal_conflict(windows)) return false;
    return std::all_of(windows.begin(), windows.end(),
        [&](const ResourceWindow& window) { return is_free(window, cycle); });
}

bool ResourceScheduler::has_internal_conflict(
    llvm::ArrayRef<ResourceWindow> windows) const
{
    std::unordered_map<std::string, std::vector<std::pair<int64_t, int64_t>>>
        intervalsByResource;
    for (const ResourceWindow& window : windows) {
        if (window.duration <= 0) return true;
        intervalsByResource[window.resource].push_back(
            {window.offset, window.offset + window.duration});
    }
    for (auto& [resource, intervals] : intervalsByResource) {
        (void)resource;
        std::sort(intervals.begin(), intervals.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.first != rhs.first
                    ? lhs.first < rhs.first : lhs.second < rhs.second;
            });
        int64_t occupiedUntil = intervals.front().second;
        for (std::size_t index = 1; index < intervals.size(); ++index) {
            if (intervals[index].first < occupiedUntil)
                return true;
            occupiedUntil = std::max(occupiedUntil, intervals[index].second);
        }
    }
    return false;
}

bool ResourceScheduler::try_reserve_at(int64_t cycle,
    llvm::ArrayRef<ResourceWindow> windows)
{
    if (!is_free_at(cycle, windows)) return false;
    reserve_at(cycle, windows);
    return true;
}

std::optional<int64_t> ResourceScheduler::try_reserve(
    int64_t earliest_cycle, llvm::ArrayRef<ResourceWindow> windows)
{
    if (has_internal_conflict(windows)) return std::nullopt;
    const int64_t cycle = find_earliest(earliest_cycle, windows);
    reserve_at(cycle, windows);
    return cycle;
}

int64_t ResourceScheduler::reserve(int64_t earliest_cycle,
    llvm::ArrayRef<ResourceWindow> windows)
{
    const int64_t anchor = find_earliest(earliest_cycle, windows);
    reserve_at(anchor, windows);
    return anchor;
}

void ResourceScheduler::reserve_at(int64_t cycle,
    llvm::ArrayRef<ResourceWindow> windows)
{
    for (const auto& window : windows) {
        const int64_t start = cycle + window.offset;
        const int64_t end = start + window.duration;
        auto& intervals = reservations_[window.resource];
        auto [position, inserted] = intervals.emplace(start, end);
        if (!inserted) position->second = std::max(position->second, end);
    }
}

int64_t ResourceScheduler::minimum_non_overlapping_shift(
    int64_t minimum_shift) const
{
    int64_t shift = std::max<int64_t>(1, minimum_shift);
    for (;;) {
        int64_t nextShift = shift;
        for (const auto& [resource, intervals] : reservations_) {
            (void)resource;
            for (const auto& [start, end] : intervals) {
                const int64_t shiftedStart = start + shift;
                const int64_t shiftedEnd = end + shift;
                auto position = intervals.lower_bound(shiftedStart);
                if (position != intervals.begin()) --position;
                for (; position != intervals.end()
                       && position->first < shiftedEnd;
                     ++position) {
                    if (shiftedStart < position->second
                        && shiftedEnd > position->first)
                        nextShift = std::max(
                            nextShift, position->second - start);
                }
            }
        }
        if (nextShift == shift) return shift;
        shift = nextShift;
    }
}

bool ResourceScheduler::is_free(const ResourceWindow& window, int64_t anchor) const
{
    const int64_t start = anchor + window.offset;
    const int64_t end = start + window.duration;
    const auto reservations = reservations_.find(window.resource);
    if (reservations == reservations_.end()) return true;
    const auto& intervals = reservations->second;
    auto position = intervals.lower_bound(start);
    if (position != intervals.begin()) --position;
    for (; position != intervals.end() && position->first < end; ++position)
        if (start < position->second && end > position->first) return false;
    return true;
}

} // namespace ftlpu::compiler::schedule
