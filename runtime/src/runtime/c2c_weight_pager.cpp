#include "ftlpu/software/runtime/c2c_weight_pager.hpp"

#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/slice.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace ftlpu::software::runtime {
C2cWeightPager::C2cWeightPager(C2cDmaSystem& system)
    : system_(system)
{
}

void C2cWeightPager::begin_schedule()
{
    if (active_ && !ready())
        throw std::logic_error(
            "cannot schedule executable pages while a page is in flight");
    if (active_) retire();
    schedule_dma_cursor_ = {};
    schedule_rx_cursor_ = {};
    scheduled_rx_segments_ = {};
}

C2cWeightPageFence C2cWeightPager::schedule(
    const C2cWeightPage& page, std::size_t start_cycle)
{
    auto& chip = system_.chip();
    const auto& hardware = chip.hardware_configuration();
    if (!hardware.c2c_dedicated_streams)
        throw std::logic_error(
            "executable C2C pages require the dedicated stream fabric");
    if (page.bank >= hw::kMemBanksPerSlice || page.segments.empty())
        throw std::invalid_argument("invalid executable C2C weight page");

    C2cWeightPageFence fence;
    fence.completed_segments = scheduled_rx_segments_;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        const auto segmentCount = static_cast<std::size_t>(std::count_if(
            page.segments.begin(), page.segments.end(),
            [&](const C2cWeightSegment& segment) {
                return hemisphere_index(segment.hemisphere) == side;
            }));
        if (segmentCount == 0) continue;
        if (start_cycle < schedule_dma_cursor_[side]
            || start_cycle < schedule_rx_cursor_[side])
            throw std::logic_error(
                "executable C2C page schedule overlaps an ICU queue");
        chip.icu().enqueue_c2c_dma_nop(
            hemisphere, start_cycle - schedule_dma_cursor_[side]);
        chip.icu().enqueue_c2c_rx_nop(
            hemisphere, start_cycle - schedule_rx_cursor_[side]);
        schedule_dma_cursor_[side] = start_cycle;
        schedule_rx_cursor_[side] = start_cycle;

        for (const C2cWeightSegment& segment : page.segments) {
            if (hemisphere_index(segment.hemisphere) != side) continue;
            if (segment.bank != page.bank
                || segment.slice >= hw::kMemSliceColumns
                || segment.vector_count == 0
                || static_cast<std::uint64_t>(segment.base_row)
                        + segment.vector_count
                    > hardware.sram_depth_rows)
                throw std::invalid_argument(
                    "invalid executable C2C weight-page segment");
            const auto lane = segment.stream
                % hardware.c2c_streams_per_direction;
            chip.icu().enqueue_c2c_dma(hemisphere,
                C2cDmaInstruction::Load(segment.ddr4_address,
                    segment.vector_count, hw::kPhysicalVectorBytes, 0,
                    lane));
            chip.icu().enqueue_c2c_receive(hemisphere, lane,
                segment.hemisphere, segment.slice, segment.bank, false,
                segment.base_row, segment.vector_count, 1, lane);
            ++scheduled_rx_segments_[side][lane];
            fence.completed_segments[side][lane] =
                scheduled_rx_segments_[side][lane];
        }
        schedule_dma_cursor_[side] += segmentCount;
        schedule_rx_cursor_[side] += segmentCount;
    }
    return fence;
}

bool C2cWeightPager::ready(const C2cWeightPageFence& fence) const
{
    const auto& chip = system_.chip();
    for (std::size_t side = 0; side < hw::kHemispheres; ++side)
        for (std::size_t stream = 0;
             stream < hw::kC2cStreamsPerDirection; ++stream)
            if (chip.c2c_endpoint(static_cast<Hemisphere>(side))
                    .rx().completed_instruction_count(stream)
                < fence.completed_segments[side][stream])
                return false;
    return true;
}

void C2cWeightPager::enqueue(const C2cWeightPage& page)
{
    if (active_ && !ready())
        throw std::logic_error("a C2C weight page is already in flight");
    if (page.bank >= hw::kMemBanksPerSlice)
        throw std::out_of_range("weight page bank is outside the MEM bank array");
    if (page.segments.empty())
        throw std::invalid_argument("C2C weight page has no segments");

    stats_ = {};
    stats_.layer = page.layer;
    stats_.bank = page.bank;
    stats_.enqueue_cycle = system_.cycle();
    target_mem_queues_.clear();
    drain_cycles_ = 0;

    auto& chip = system_.chip();
    const auto& hardware = chip.hardware_configuration();
    const bool sharedFabric = !hardware.c2c_dedicated_streams;
    const std::size_t sharedStreamBase =
        hw::kWestStreams - hardware.c2c_streams_per_direction;
    auto usedHemisphere = std::array<bool, hw::kHemispheres> {};
    for (const C2cWeightSegment& segment : page.segments) {
        if (segment.bank != page.bank)
            throw std::invalid_argument(
                "every weight-page segment must target the page bank");
        if (segment.slice >= hw::kMemSliceColumns
            || segment.vector_count == 0)
            throw std::invalid_argument("invalid C2C weight-page segment");
        if (static_cast<std::uint64_t>(segment.base_row)
                + segment.vector_count
            > chip.hardware_configuration().sram_depth_rows)
            throw std::out_of_range(
                "C2C weight segment exceeds its SRAM bank: base="
                + std::to_string(segment.base_row)
                + " vectors=" + std::to_string(segment.vector_count)
                + " depth=" + std::to_string(
                    chip.hardware_configuration().sram_depth_rows));

        const auto side = hemisphere_index(segment.hemisphere);
        usedHemisphere[side] = true;
        const auto lane = segment.stream
            % hardware.c2c_streams_per_direction;
        const auto fabricStream = sharedStreamBase + lane;

        chip.icu().enqueue_c2c_dma(segment.hemisphere,
            C2cDmaInstruction::Load(segment.ddr4_address,
                segment.vector_count, hw::kPhysicalVectorBytes, 0, lane));
        chip.icu().enqueue_c2c_receive(segment.hemisphere,
            lane, segment.hemisphere, segment.slice,
            segment.bank, sharedFabric, segment.base_row,
            segment.vector_count, 1, fabricStream);
        if (sharedFabric) {
            const auto queue = InstructionControlUnit::mem_queue(
                segment.hemisphere, segment.slice, segment.bank);
            if (std::find(target_mem_queues_.begin(),
                    target_mem_queues_.end(), queue)
                == target_mem_queues_.end())
                target_mem_queues_.push_back(queue);
            chip.icu().enqueue_control(
                IcuLocation::Mem(segment.hemisphere,
                    segment.slice, segment.bank),
                IcuControlInstruction::Sync());
            const auto group =
                segment.slice / hw::kMemSlicesPerGroup;
            const auto transportNops =
                hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
            chip.icu().enqueue_mem_nop(queue, transportNops);
            chip.icu().enqueue_mem(queue,
                MemInstruction::Write(segment.base_row,
                    StreamId::West(fabricStream)));
            if (segment.vector_count > 1) {
                chip.icu().enqueue_mem_repeat(queue,
                    segment.vector_count - 1, 1, 1);
            }
        }
        stats_.vectors += segment.vector_count;
    }
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        if (!usedHemisphere[side]) continue;
        chip.icu().enqueue_control(
            IcuLocation::C2cDma(static_cast<Hemisphere>(side)),
            IcuControlInstruction::Sync());
    }
    stats_.bytes = stats_.vectors * hw::kPhysicalVectorBytes;
    active_ = true;
}

bool C2cWeightPager::busy() const noexcept
{
    return active_ && !ready();
}

bool C2cWeightPager::ready() const
{
    if (!active_) return false;
    auto& chip = system_.chip();
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        if (!chip.has_c2c(hemisphere))
            throw std::logic_error(
                "C2C weight pager lost endpoint for hemisphere "
                + std::to_string(side) + " at cycle "
                + std::to_string(system_.cycle()));
        if (!chip.icu().c2c_dma_iq(hemisphere).done()
            || !chip.icu().c2c_rx_iq(hemisphere).done()
            || !system_.dma(hemisphere).idle()
            || !chip.c2c_endpoint(hemisphere).rx().idle())
            return false;
    }
    for (const auto queue : target_mem_queues_)
        if (!chip.icu().mem_iq(queue).done()) return false;
    return drain_cycles_ >= hw::kTileRows;
}

void C2cWeightPager::retire()
{
    if (!active_ || !ready())
        throw std::logic_error(
            "cannot retire an incomplete C2C weight page");
    active_ = false;
    target_mem_queues_.clear();
    drain_cycles_ = 0;
}

void C2cWeightPager::tick()
{
    system_.tick();
    observe_tick();
}

void C2cWeightPager::observe_tick()
{
    if (!active_) return;
    bool queues_done = true;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        queues_done = queues_done
            && system_.chip().icu().c2c_rx_iq(hemisphere).done();
    }
    for (const auto queue : target_mem_queues_)
        queues_done = queues_done
            && system_.chip().icu().mem_iq(queue).done();
    if (queues_done) ++drain_cycles_;
    else drain_cycles_ = 0;
    if (ready() && stats_.ready_cycle == 0)
        stats_.ready_cycle = system_.cycle();
}

void C2cWeightPager::wait(std::size_t max_cycles)
{
    for (std::size_t cycle = 0; cycle < max_cycles && !ready(); ++cycle)
        tick();
    if (!ready())
        throw std::runtime_error("C2C weight-page prefetch timed out");
}

} // namespace ftlpu::software::runtime
