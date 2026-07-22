#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {

int64_t ResourceScheduler::reserve(int64_t earliest_cycle,
    llvm::ArrayRef<ResourceWindow> windows)
{
    int64_t anchor = earliest_cycle;
    for (;;) {
        int64_t next_anchor = anchor;
        for (const auto& window : windows) {
            const int64_t start = anchor + window.offset;
            const int64_t end = start + window.duration;
            const auto reservations = reservations_.find(window.resource);
            if (reservations == reservations_.end()) continue;
            for (const auto& reservation : reservations->second) {
                if (start < reservation.end && end > reservation.start) {
                    next_anchor = std::max(next_anchor, reservation.end - window.offset);
                }
            }
        }
        if (next_anchor == anchor) break;
        anchor = next_anchor;
    }
    reserve_at(anchor, windows);
    return anchor;
}

void ResourceScheduler::reserve_at(int64_t cycle,
    llvm::ArrayRef<ResourceWindow> windows)
{
    for (const auto& window : windows) {
        reservations_[window.resource].push_back(
            {cycle + window.offset, cycle + window.offset + window.duration});
    }
}

bool ResourceScheduler::is_free(const ResourceWindow& window, int64_t anchor) const
{
    const int64_t start = anchor + window.offset;
    const int64_t end = start + window.duration;
    const auto reservations = reservations_.find(window.resource);
    if (reservations == reservations_.end()) return true;
    for (const auto& reservation : reservations->second)
        if (start < reservation.end && end > reservation.start) return false;
    return true;
}

} // namespace ftlpu::compiler::schedule
