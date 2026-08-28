#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <optional>
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
    int64_t find_earliest(int64_t earliest_cycle,
        llvm::ArrayRef<ResourceWindow> windows) const;
    bool is_free_at(int64_t cycle,
        llvm::ArrayRef<ResourceWindow> windows) const;
    bool has_internal_conflict(
        llvm::ArrayRef<ResourceWindow> windows) const;
    bool try_reserve_at(int64_t cycle,
        llvm::ArrayRef<ResourceWindow> windows);
    std::optional<int64_t> try_reserve(int64_t earliest_cycle,
        llvm::ArrayRef<ResourceWindow> windows);
    int64_t reserve(int64_t earliest_cycle, llvm::ArrayRef<ResourceWindow> windows);
    void reserve_at(int64_t cycle, llvm::ArrayRef<ResourceWindow> windows);
    int64_t minimum_non_overlapping_shift(int64_t minimum_shift) const;

private:
    bool is_free(const ResourceWindow& window, int64_t anchor) const;
    std::unordered_map<std::string, std::map<int64_t, int64_t>> reservations_;
};

} // namespace ftlpu::compiler::schedule
