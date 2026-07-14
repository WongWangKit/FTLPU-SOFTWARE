#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ftlpu::compiler::schedule {

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

} // namespace ftlpu::compiler::schedule
