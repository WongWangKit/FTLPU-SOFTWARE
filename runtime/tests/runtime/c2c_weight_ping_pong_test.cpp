#include "ftlpu/software/runtime/c2c_weight_pager.hpp"

#include "ftlpu/icu/instruction.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

C2cVector make_vector(std::uint8_t base)
{
    C2cVector vector;
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            vector.payload[tile][lane] = static_cast<std::uint8_t>(
                base + tile * hw::kLanesPerTile + lane);
    return vector;
}

} // namespace

void run_test()
{
    constexpr auto hemisphere = Hemisphere::West;
    constexpr std::size_t slice = 16;
    constexpr std::size_t current_bank = 0;
    constexpr std::size_t next_bank = 1;
    constexpr std::size_t current_row = 7;
    constexpr std::size_t next_row = 41;
    constexpr std::size_t vectors = 3;
    constexpr std::uint64_t ddr4_address = 0x1000;

    C2cDmaSystem system(Ddr4Config {32, 2, 2, 256, 8});
    const auto current = make_vector(0x10);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            system.chip().initialize_mem_sram_lane_byte(hemisphere, slice,
                current_bank, tile, current_row, lane,
                current.payload[tile][lane]);

    for (std::size_t vector = 0; vector < vectors; ++vector)
        system.ddr4().initialize_vector(
            ddr4_address + vector * hw::kPhysicalVectorBytes,
            make_vector(static_cast<std::uint8_t>(0x40 + vector * 0x20)));

    const std::size_t current_queue = InstructionControlUnit::mem_queue(
        hemisphere, slice, current_bank);
    for (std::size_t cycle = 0; cycle < 96; ++cycle)
        system.chip().icu().enqueue_mem(current_queue,
            MemInstruction::Read(current_row, StreamId::East(23)));

    C2cWeightPager pager(system);
    pager.enqueue(C2cWeightPage {
        1,
        next_bank,
        {C2cWeightSegment {hemisphere, slice, next_bank, next_row, 5,
            ddr4_address, vectors}},
    });

    bool overlapped_bank_issue = false;
    for (std::size_t cycle = 0; cycle < 512 && !pager.ready(); ++cycle) {
        pager.tick();
        overlapped_bank_issue = overlapped_bank_issue
            || (system.chip().icu().mem_iq(current_queue).last_trace().action
                    == IcuQueueAction::FunctionalIssue
                && system.dma(hemisphere).last_beat().has_value());
    }

    require(pager.ready(), "next-layer C2C weight page did not become ready");
    require(overlapped_bank_issue,
        "current-bank reads did not overlap the dedicated C2C data path");
    require(pager.stats().layer == 1 && pager.stats().bank == next_bank,
        "weight pager lost layer or bank identity");
    require(pager.stats().vectors == vectors
            && pager.stats().bytes == vectors * hw::kPhysicalVectorBytes,
        "weight pager statistics are incorrect");

    for (std::size_t vector = 0; vector < vectors; ++vector) {
        const auto expected = make_vector(
            static_cast<std::uint8_t>(0x40 + vector * 0x20));
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto actual = system.chip().read_mem_sram_lane_byte(
                    hemisphere, slice, next_bank, tile, next_row + vector,
                    lane);
                if (actual != expected.payload[tile][lane])
                    throw std::runtime_error(
                        "C2C weight page mismatch: vector="
                        + std::to_string(vector) + " tile="
                        + std::to_string(tile) + " lane="
                        + std::to_string(lane) + " expected="
                        + std::to_string(expected.payload[tile][lane])
                        + " actual=" + std::to_string(actual));
            }
    }
}

int main()
{
    try {
        run_test();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
