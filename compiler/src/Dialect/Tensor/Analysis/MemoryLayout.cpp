#include "ftlpu/compiler/Dialect/Tensor/Analysis/memory_layout.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace ftlpu::compiler::tensor {
namespace {

constexpr std::size_t kSlicesPerHemisphere = 44;
constexpr std::size_t kBanksPerSlice = 2;
constexpr std::size_t kWordsPerBank = 4096;
constexpr std::size_t kBytesPerWord = 16;

MemoryAddress decode_word_address(Hemisphere hemisphere, std::size_t linear_word)
{
    const auto words_per_slice = kBanksPerSlice * kWordsPerBank;
    const auto slice = (linear_word / words_per_slice) % kSlicesPerHemisphere;
    const auto word_in_slice = linear_word % words_per_slice;
    return MemoryAddress {
        0,
        hemisphere,
        slice,
        word_in_slice / kWordsPerBank,
        word_in_slice % kWordsPerBank,
        0,
    };
}

const char* hemisphere_name(Hemisphere hemisphere)
{
    return hemisphere == Hemisphere::East ? "east" : "west";
}

} // namespace

MemoryLayout::MemoryLayout(Hemisphere hemisphere)
    : hemisphere_(hemisphere)
{
}

MemoryAllocation MemoryLayout::allocate(std::string symbol, std::size_t bytes)
{
    const auto words = align_up(bytes, kBytesPerWord) / kBytesPerWord;
    const auto capacity_words = kSlicesPerHemisphere * kBanksPerSlice * kWordsPerBank;
    if (next_word_ + words > capacity_words) {
        throw std::runtime_error("FTLPU MEM allocator exhausted hemisphere capacity");
    }

    auto allocation = MemoryAllocation {std::move(symbol), decode_word_address(hemisphere_, next_word_), bytes, words};
    next_word_ += words;
    allocations_.push_back(allocation);
    return allocation;
}

const std::vector<MemoryAllocation>& MemoryLayout::allocations() const
{
    return allocations_;
}

std::size_t align_up(std::size_t value, std::size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

std::string format_address(const MemoryAddress& address)
{
    std::ostringstream os;
    os << "[device = " << address.device
       << ", hemi = \"" << hemisphere_name(address.hemisphere)
       << "\", slice = " << address.slice
       << ", bank = " << address.bank
       << ", word = " << address.word
       << ", byte = " << address.byte << "]";
    return os.str();
}

std::string format_allocation(const MemoryAllocation& allocation)
{
    std::ostringstream os;
    os << "{addr = " << format_address(allocation.base)
       << ", bytes = " << allocation.bytes
       << ", words = " << allocation.words << "}";
    return os.str();
}

} // namespace ftlpu::compiler::tensor
