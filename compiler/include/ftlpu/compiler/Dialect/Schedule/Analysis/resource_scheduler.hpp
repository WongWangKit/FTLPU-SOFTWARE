#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace ftlpu::compiler::schedule {

enum class PhysicalResourceKind : std::uint8_t {
    Legacy,
    MemIcuQueue,
    MemReadPort,
    MemWritePort,
    MxmLoadQueue,
    MxmComputeQueue,
    MxmDequantQueue,
    MxmWeightBuffer,
    VxmQueue,
    VxmAlu,
    SxmTransposeQueue,
    SxmPermuteQueue,
    SxmUnit,
    StreamChannel,
};

struct PhysicalResourceId {
    PhysicalResourceKind kind{PhysicalResourceKind::Legacy};
    int64_t first{-1};
    int64_t second{-1};
    int64_t third{-1};
    std::string legacy_name;

    PhysicalResourceId() = default;
    PhysicalResourceId(const char* name) : legacy_name(name) {}
    PhysicalResourceId(std::string name) : legacy_name(std::move(name)) {}
    PhysicalResourceId(PhysicalResourceKind resource_kind,
        int64_t first_id, int64_t second_id = -1, int64_t third_id = -1)
        : kind(resource_kind), first(first_id), second(second_id), third(third_id)
    {
    }

    bool operator==(const PhysicalResourceId&) const = default;
};

struct PhysicalResourceIdHash {
    std::size_t operator()(const PhysicalResourceId& resource) const noexcept;
};

std::string physical_resource_name(const PhysicalResourceId& resource);

struct ResourceWindow {
    PhysicalResourceId resource;
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
    // Imports externally occupied intervals. Unlike reserve_at(), overlapping
    // constraints are intentional here and form the union of unavailable time.
    void block_at(int64_t cycle, llvm::ArrayRef<ResourceWindow> windows);
    int64_t minimum_non_overlapping_shift(int64_t minimum_shift) const;

private:
    void commit_at(int64_t cycle, llvm::ArrayRef<ResourceWindow> windows);
    bool is_free(const ResourceWindow& window, int64_t anchor) const;
    std::unordered_map<PhysicalResourceId, std::map<int64_t, int64_t>,
        PhysicalResourceIdHash> reservations_;
};

} // namespace ftlpu::compiler::schedule
