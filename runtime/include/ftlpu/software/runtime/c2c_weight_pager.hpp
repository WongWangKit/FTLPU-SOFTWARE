#pragma once

#include "ftlpu/system/c2c_dma_system.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftlpu::software::runtime {

struct C2cWeightSegment {
    Hemisphere hemisphere{Hemisphere::East};
    std::uint16_t slice{0};
    std::uint16_t bank{0};
    std::uint32_t base_row{0};
    std::uint16_t stream{0};
    std::uint64_t ddr4_address{0};
    std::uint32_t vector_count{0};
};

struct C2cWeightPage {
    std::uint32_t layer{0};
    std::uint16_t bank{0};
    std::vector<C2cWeightSegment> segments{};
};

struct C2cWeightPageStats {
    std::uint32_t layer{0};
    std::uint16_t bank{0};
    std::size_t enqueue_cycle{0};
    std::size_t ready_cycle{0};
    std::size_t vectors{0};
    std::size_t bytes{0};
};

class C2cWeightPager {
public:
    explicit C2cWeightPager(C2cDmaSystem& system);

    void enqueue(const C2cWeightPage& page);
    bool busy() const noexcept;
    bool ready() const;
    void tick();
    void observe_tick();
    void wait(std::size_t max_cycles);

    const C2cWeightPageStats& stats() const noexcept { return stats_; }

private:
    C2cDmaSystem& system_;
    C2cWeightPageStats stats_{};
    std::vector<std::size_t> mem_queues_{};
    std::uint32_t drain_cycles_{0};
    bool active_{false};
};

} // namespace ftlpu::software::runtime
