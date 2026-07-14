#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"

#include <algorithm>
#include <utility>

namespace ftlpu::compiler::schedule {

std::size_t ResourceScheduler::reserve(std::string resource, std::size_t earliest_cycle, std::size_t duration)
{
    auto start = earliest_cycle;
    while (!is_free(resource, start, start + duration)) {
        ++start;
    }
    reservations_.push_back(Reservation {std::move(resource), start, start + duration});
    return start;
}

std::size_t ResourceScheduler::available_after(const std::string& resource) const
{
    std::size_t cycle = 0;
    for (const auto& reservation : reservations_) {
        if (reservation.resource == resource) {
            cycle = std::max(cycle, reservation.end);
        }
    }
    return cycle;
}

bool ResourceScheduler::is_free(const std::string& resource, std::size_t start, std::size_t end) const
{
    for (const auto& reservation : reservations_) {
        if (reservation.resource == resource && start < reservation.end && end > reservation.start) {
            return false;
        }
    }
    return true;
}

} // namespace ftlpu::compiler::schedule
