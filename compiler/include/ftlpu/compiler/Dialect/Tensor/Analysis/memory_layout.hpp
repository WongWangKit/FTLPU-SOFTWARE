#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ftlpu::compiler::tensor {

enum class Hemisphere {
    West = 0,
    East = 1,
};

struct MemoryAddress {
    std::size_t device{0};
    Hemisphere hemisphere{Hemisphere::East};
    std::size_t slice{0};
    std::size_t bank{0};
    std::size_t word{0};
    std::size_t byte{0};
};

struct MemoryAllocation {
    std::string symbol;
    MemoryAddress base{};
    std::size_t bytes{0};
    std::size_t words{0};
};

class MemoryLayout {
public:
    explicit MemoryLayout(Hemisphere hemisphere = Hemisphere::East);

    MemoryAllocation allocate(std::string symbol, std::size_t bytes);
    const std::vector<MemoryAllocation>& allocations() const;

private:
    Hemisphere hemisphere_{Hemisphere::East};
    std::size_t next_word_{0};
    std::vector<MemoryAllocation> allocations_{};
};

std::size_t align_up(std::size_t value, std::size_t alignment);
std::string format_address(const MemoryAddress& address);
std::string format_allocation(const MemoryAllocation& allocation);

} // namespace ftlpu::compiler::tensor
