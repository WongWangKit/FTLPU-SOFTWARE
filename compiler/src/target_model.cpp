#include "ftlpu/compiler/target_model.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ftlpu::compiler {
namespace {

constexpr std::size_t kSlicesPerHemisphere = 44;
constexpr std::size_t kBanksPerSlice = 2;
constexpr std::size_t kWordsPerBank = 4096;
constexpr std::size_t kBytesPerWord = 16;
constexpr std::size_t kStreamRegisterColumns = 12;
constexpr std::size_t kSlicesPerStreamRegisterGroup = 4;

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

MemoryAllocator::MemoryAllocator(Hemisphere hemisphere)
    : hemisphere_(hemisphere)
{
}

MemoryAllocation MemoryAllocator::allocate(std::string symbol, std::size_t bytes)
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

const std::vector<MemoryAllocation>& MemoryAllocator::allocations() const
{
    return allocations_;
}

StreamBinding StreamManager::allocate(
    std::string symbol,
    StreamDirection direction,
    StreamEndpoint producer,
    StreamEndpoint consumer,
    std::size_t produce_cycle,
    std::size_t consume_cycle,
    std::size_t preferred_stream)
{
    if (consume_cycle < produce_cycle) {
        throw std::runtime_error("stream consume cycle is earlier than produce cycle");
    }

    for (std::size_t offset = 0; offset < 64; ++offset) {
        const auto candidate = (preferred_stream + offset) % 64;
        if (!conflicts(candidate, produce_cycle, consume_cycle)) {
            auto binding = StreamBinding {
                std::move(symbol),
                candidate,
                direction,
                std::move(producer),
                std::move(consumer),
                produce_cycle,
                consume_cycle,
                consume_cycle - produce_cycle,
            };
            bindings_.push_back(binding);
            return binding;
        }
    }
    throw std::runtime_error("no stream id is free for requested lifetime");
}

const std::vector<StreamBinding>& StreamManager::bindings() const
{
    return bindings_;
}

bool StreamManager::conflicts(std::size_t stream_id, std::size_t produce_cycle, std::size_t consume_cycle) const
{
    for (const auto& binding : bindings_) {
        if (binding.stream_id != stream_id) {
            continue;
        }
        if (produce_cycle < binding.consume_cycle && consume_cycle > binding.produce_cycle) {
            return true;
        }
    }
    return false;
}

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

std::size_t align_up(std::size_t value, std::size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

std::size_t mem_to_mxm_latency(std::size_t mem_slice)
{
    return (kStreamRegisterColumns - 1) - mem_slice / kSlicesPerStreamRegisterGroup + 1;
}

std::size_t mxm_to_vxm_latency()
{
    return kStreamRegisterColumns;
}

std::size_t vxm_pipeline_latency(std::size_t stages)
{
    return stages == 0 ? 0 : stages - 1;
}

std::size_t vxm_to_mem_latency(std::size_t mem_slice)
{
    return mem_slice / kSlicesPerStreamRegisterGroup + 2;
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

std::string stream_direction_name(StreamDirection direction)
{
    return direction == StreamDirection::East ? "E" : "W";
}

std::string format_endpoint(const StreamEndpoint& endpoint)
{
    std::ostringstream os;
    os << endpoint.kind << ":" << endpoint.name;
    if (endpoint.port.has_value()) {
        os << ":" << *endpoint.port;
    }
    return os.str();
}

} // namespace ftlpu::compiler
