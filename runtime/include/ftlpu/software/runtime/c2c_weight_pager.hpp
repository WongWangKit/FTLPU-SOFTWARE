#pragma once

#include "ftlpu/c2c/icu_instruction.hpp"
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <optional>
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
    // Ordinary westbound SR range used after the C2C receive lanes. When
    // absent, the pager retains the target's conventional high stream range.
    std::optional<std::uint16_t> fabric_stream_base{};
};

struct C2cWeightPageStats {
    std::uint32_t layer{0};
    std::uint16_t bank{0};
    std::size_t enqueue_cycle{0};
    std::size_t ready_cycle{0};
    std::size_t vectors{0};
    std::size_t bytes{0};
};

struct C2cWeightPageFence {
    std::array<std::size_t, hw::kHemispheres> dma_issues_begin{};
    std::array<std::size_t, hw::kHemispheres> dma_issues_end{};
    std::array<std::array<std::size_t, hw::kC2cStreamsPerDirection>,
               hw::kHemispheres>
        completed_segments{};
    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        completed_mem_writes{};
    // Cumulative MEM_WRITE_SYNC retirements, including the reserved ownership
    // interval after an early final write.
    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        completed_mem_reservations{};
    // MEM ICU issue precedes the final tile's SRAM write. Record when every
    // write has issued so ready() can account for the distributed MEM pipe.
    mutable std::optional<std::size_t> mem_writes_issued_cycle{};
    // The executable linker may move a transfer to the next common idle
    // interval of every target MEM ICU. This records the resolved transport
    // estimate; a finite ready cycle may reserve the MEM ICU for longer.
    std::size_t scheduled_start_cycle{0};
    std::size_t scheduled_end_cycle{0};
};

// The direct-lowering result for one C2C weight-page segment. Every member is
// already encoded in the fixed hardware packet accepted by its target ICU.
// RX carries only the ordinary-SR destination and MEM completion route; SRAM
// placement lives exclusively in mem_write_sync.
struct C2cWeightSegmentIcuProgram {
    Hemisphere hemisphere{Hemisphere::East};
    std::uint16_t lane{0};
    std::uint16_t fabric_stream{0};
    std::size_t mem_queue{0};
    std::uint32_t sync_tag{0};
    C2cEndpointIcuPacket rx{};
    InstructionControlUnit::MemIcu::EncodedSynchronizedPacket
        mem_write_sync{};
    C2cDmaIcuPacket dma{};
    std::uint32_t vector_count{0};
};

struct C2cWeightPageIcuProgram {
    std::vector<C2cWeightSegmentIcuProgram> segments{};
    // The lowering owns these ordinary DMA completion barriers. The loader
    // must not invent control instructions after the program is built.
    std::array<bool, hw::kHemispheres> dma_completion_sync{};
    std::uint32_t next_sync_tag{1};

    std::size_t physical_word_count() const noexcept
    {
        return segments.size()
            * (1
                + InstructionControlUnit::MemIcu::
                    synchronized_packet_word_count
                + C2cDmaIcuPacket::kWordCount)
            + std::count(dma_completion_sync.begin(),
                dma_completion_sync.end(), true);
    }
};

// Lowers a logical weight page directly to fixed C2C/MEM ICU packets. Tags
// are monotonically allocated from first_sync_tag and must fit the 16-bit
// field shared by RX and MEM_WRITE_SYNC.
C2cWeightPageIcuProgram lower_c2c_weight_page_to_icu(
    const C2cWeightPage& page,
    const SystemHardwareConfiguration& hardware,
    std::uint32_t first_sync_tag = 1,
    // Zero selects the minimum standalone reservation on each physical MEM
    // queue. The executable linker supplies an explicit scheduled window.
    std::size_t reservation_cycles = 0);

class C2cWeightPager {
public:
    struct PageReadyRelease {
        IcuLocation location{};
        std::size_t phase_offset{0};
    };

    explicit C2cWeightPager(C2cDmaSystem& system);

    void enqueue(const C2cWeightPage& page);
    void begin_schedule(const BinaryProgram& program);
    C2cWeightPageFence schedule(
        BinaryProgram& binary, const C2cWeightPage& page,
        std::size_t start_cycle, std::size_t transfer_end_cycle,
        std::size_t ready_cycle,
        std::size_t launch_event_tag,
        std::size_t page_ready_event_tag = 0);
    void finalize_schedule(BinaryProgram& program);
    // Apply the scheduled page-ready waits to a parallel compiler image that
    // already contains compiler-authored MEM_READ/WRITE_SYNC packets.
    void inject_page_ready_barriers(BinaryProgram& program) const;
    std::vector<PageReadyRelease> page_ready_releases(
        std::size_t event_tag) const;
    std::size_t earliest_schedule_cycle(
        const C2cWeightPage& page) const;
    bool started(const C2cWeightPageFence& fence) const;
    bool ready(const C2cWeightPageFence& fence) const;
    bool busy() const noexcept;
    bool ready() const;
    void retire();
    void tick();
    void observe_tick();
    void wait(std::size_t max_cycles);

    const C2cWeightPageStats& stats() const noexcept { return stats_; }
    const BinaryProgram& last_enqueued_program() const noexcept
    {
        return last_enqueued_program_;
    }

private:
    struct MemIdleWindow {
        std::size_t begin{0};
        std::size_t end{0};
    };

    struct LinkedMemWindow {
        std::size_t begin{0};
        std::size_t end{0};
        std::vector<InstructionControlUnit::MemIcu::
            EncodedSynchronizedPacket> packets{};
    };

    struct LinkedPageBarrier {
        std::size_t ready_cycle{0};
        std::size_t event_tag{0};
    };

    struct LinkedPageRelease {
        std::size_t event_tag{0};
        IcuLocation location{};
        std::size_t barrier_cycle{0};
    };

    C2cDmaSystem& system_;
    BinaryProgram last_enqueued_program_{};
    C2cWeightPageStats stats_{};
    std::vector<std::size_t> target_mem_queues_{};
    std::array<std::size_t, hw::kHemispheres> schedule_dma_cursor_{};
    std::array<std::size_t, hw::kHemispheres> schedule_rx_cursor_{};
    std::array<std::size_t, hw::kHemispheres> scheduled_dma_issues_{};
    std::array<std::array<std::size_t, hw::kC2cStreamsPerDirection>,
               hw::kHemispheres>
        scheduled_rx_segments_{};
    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        scheduled_mem_writes_{};
    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        scheduled_mem_reservations_{};
    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        active_mem_writes_target_{};
    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        active_mem_reservations_target_{};
    std::array<std::vector<MemIdleWindow>,
        InstructionControlUnit::kMemQueues> mem_idle_windows_{};
    std::array<std::vector<LinkedMemWindow>,
        InstructionControlUnit::kMemQueues> linked_mem_windows_{};
    std::vector<LinkedPageBarrier> linked_page_barriers_{};
    std::vector<LinkedPageRelease> linked_page_releases_{};
    std::uint32_t next_sync_tag_{1};
    std::uint32_t drain_cycles_{0};
    bool schedule_open_{false};
    bool active_{false};
};

} // namespace ftlpu::software::runtime
