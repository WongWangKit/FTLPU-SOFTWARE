#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace ftlpu::compiler::schedule {

struct ResourceWindow {
    std::string resource;
    int64_t offset;
    int64_t duration;
};

class ResourceScheduler {
public:
    int64_t reserve(int64_t earliest_cycle, llvm::ArrayRef<ResourceWindow> windows);
    void reserve_at(int64_t cycle, llvm::ArrayRef<ResourceWindow> windows);

private:
    struct Reservation {
        int64_t start;
        int64_t end;
    };

    bool is_free(const ResourceWindow& window, int64_t anchor) const;
    std::unordered_map<std::string, llvm::SmallVector<Reservation, 32>> reservations_;
};

} // namespace ftlpu::compiler::schedule
