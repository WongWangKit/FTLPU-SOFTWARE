#include "ftlpu/compiler/Dialect/Tensor/Analysis/physical_memory_allocator.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace ftlpu::compiler::tensor {

bool PhysicalMemoryAllocator::valid(const PhysicalAllocation& allocation) const
{
    if (allocation.name.empty() || allocation.slices.empty()
        || allocation.base_row < 0 || allocation.rows <= 0
        || allocation.live_start < 0 || allocation.live_end <= allocation.live_start)
        return false;
    llvm::SmallDenseSet<int64_t, 16> unique;
    for (int64_t slice : allocation.slices) {
        if (slice < 0 || slice >= target_.memory().accumulator_slice_base
            || !unique.insert(slice).second)
            return false;
    }
    return allocation.base_row + allocation.rows
        <= target_.memory().words_per_bank * target_.memory().banks_per_slice;
}

bool PhysicalMemoryAllocator::conflicts(llvm::ArrayRef<int64_t> slices,
    int64_t base_row, int64_t rows,
    int64_t live_start, int64_t live_end,
    bool reserve_slice_port) const
{
    for (const PhysicalAllocation& allocation : allocations_) {
        if (live_start >= allocation.live_end || live_end <= allocation.live_start)
            continue;
        const bool rows_overlap =
            base_row < allocation.base_row + allocation.rows
            && base_row + rows > allocation.base_row;
        if (!rows_overlap
            && !(reserve_slice_port && allocation.reserve_slice_port))
            continue;
        for (int64_t slice : slices)
            if (llvm::is_contained(allocation.slices, slice)) return true;
    }
    return false;
}

mlir::LogicalResult PhysicalMemoryAllocator::reserve(PhysicalAllocation allocation)
{
    if (!valid(allocation)
        || conflicts(allocation.slices, allocation.base_row, allocation.rows,
            allocation.live_start, allocation.live_end,
            allocation.reserve_slice_port))
        return mlir::failure();
    allocations_.push_back(std::move(allocation));
    return mlir::success();
}

mlir::FailureOr<PhysicalAllocation> PhysicalMemoryAllocator::allocate(
    const PhysicalAllocationRequest& request)
{
    if (request.slice_count <= 0
        || static_cast<int64_t>(request.candidate_slices.size()) < request.slice_count)
        return mlir::failure();
    for (std::size_t begin = 0;
         begin + static_cast<std::size_t>(request.slice_count)
            <= request.candidate_slices.size();
         ++begin) {
        llvm::SmallVector<int64_t, 16> slices(
            request.candidate_slices.slice(begin,
                static_cast<std::size_t>(request.slice_count)));
        bool contiguous = true;
        for (std::size_t index = 1; index < slices.size(); ++index)
            contiguous &= slices[index] == slices[index - 1] + 1;
        if (!contiguous
            || conflicts(slices, request.base_row, request.rows,
                request.live_start, request.live_end,
                request.reserve_slice_port))
            continue;
        PhysicalAllocation allocation {request.name, std::move(slices),
            request.base_row, request.rows, request.live_start, request.live_end,
            request.reserve_slice_port};
        if (failed(reserve(allocation))) continue;
        return allocations_.back();
    }
    return mlir::failure();
}

} // namespace ftlpu::compiler::tensor
