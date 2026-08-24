#include "ftlpu/software/runtime/c2c_weight_pager.hpp"

#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace ftlpu::software::runtime {
C2cWeightPager::C2cWeightPager(C2cDmaSystem& system)
    : system_(system)
{
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
    drain_cycles_ = 0;

    auto& chip = system_.chip();
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
            throw std::out_of_range("C2C weight segment exceeds its SRAM bank");

        const auto side = hemisphere_index(segment.hemisphere);
        usedHemisphere[side] = true;
        const auto stream = segment.slice
            % chip.hardware_configuration().c2c_streams_per_direction;

        chip.icu().enqueue_c2c_dma(segment.hemisphere,
            C2cDmaInstruction::Load(segment.ddr4_address,
                segment.vector_count, hw::kPhysicalVectorBytes, 0, stream));
        chip.icu().enqueue_c2c_receive(segment.hemisphere,
            stream, segment.hemisphere, segment.slice,
            segment.bank, false, segment.base_row,
            segment.vector_count, 1);
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
    return drain_cycles_ >= 1;
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
