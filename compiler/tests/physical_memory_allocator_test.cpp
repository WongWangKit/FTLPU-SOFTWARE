#include "ftlpu/compiler/Dialect/Tensor/Analysis/physical_memory_allocator.hpp"

#include <array>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
{
    using namespace ftlpu::compiler;
    const target::LPUTargetModel target;
    tensor::PhysicalMemoryAllocator allocator(target);
    const std::array<int64_t, 8> candidates {0, 1, 2, 3, 4, 5, 6, 7};
    auto first = allocator.allocate(
        {"first", 2, 0, 32, 0, 2, candidates, false});
    auto disjoint_rows = allocator.allocate(
        {"disjoint_rows", 2, 32, 32, 1, 3, candidates});
    auto overlapping = allocator.allocate(
        {"overlap", 2, 16, 32, 1, 3, candidates});
    auto reused = allocator.allocate(
        {"reused", 2, 0, 32, 3, 4, candidates});
    require(mlir::succeeded(first) && first->slices[0] == 0,
        "first-fit allocation failed");
    require(mlir::succeeded(disjoint_rows) && disjoint_rows->slices[0] == 0,
        "disjoint row ranges did not share storage-only slices");
    require(mlir::succeeded(overlapping) && overlapping->slices[0] == 2,
        "overlapping row range and lifetime reused live slices");
    require(mlir::succeeded(reused) && reused->slices[0] == 0,
        "expired slices were not reused");
}
