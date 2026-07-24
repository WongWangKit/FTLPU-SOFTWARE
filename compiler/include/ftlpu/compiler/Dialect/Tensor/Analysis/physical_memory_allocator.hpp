#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <string>

namespace ftlpu::compiler::tensor {

struct PhysicalAllocation {
    std::string name;
    llvm::SmallVector<int64_t, 16> slices;
    int64_t base_row;
    int64_t rows;
    int64_t live_start;
    int64_t live_end;
    bool reserve_slice_port = true;
};

struct PhysicalAllocationRequest {
    std::string name;
    int64_t slice_count;
    int64_t base_row;
    int64_t rows;
    int64_t live_start;
    int64_t live_end;
    llvm::ArrayRef<int64_t> candidate_slices;
    bool reserve_slice_port = true;
};

// Allocates byte planes with interval reuse. Lifetimes are half-open and are
// compiler stages rather than cycles; cycle-level queue conflicts are handled
// by the Schedule verifier.
class PhysicalMemoryAllocator {
public:
    explicit PhysicalMemoryAllocator(const target::LPUTargetModel& target)
        : target_(target) {}

    mlir::LogicalResult reserve(PhysicalAllocation allocation);
    mlir::FailureOr<PhysicalAllocation> allocate(
        const PhysicalAllocationRequest& request);

    llvm::ArrayRef<PhysicalAllocation> allocations() const {
        return allocations_;
    }

private:
    bool valid(const PhysicalAllocation& allocation) const;
    bool conflicts(llvm::ArrayRef<int64_t> slices,
        int64_t base_row, int64_t rows,
        int64_t live_start, int64_t live_end,
        bool reserve_slice_port) const;

    const target::LPUTargetModel& target_;
    llvm::SmallVector<PhysicalAllocation, 32> allocations_;
};

} // namespace ftlpu::compiler::tensor
