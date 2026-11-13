#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/binary.hpp"

#include "ftlpu/icu/instruction.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

constexpr auto kTimingHemisphere = Hemisphere::West;
constexpr std::size_t kTimingSlice = hw::kMemSliceColumns - 1;
constexpr std::size_t kTimingBank = 1;
constexpr std::size_t kTimingWindowBegin = 3;
constexpr std::size_t kTimingVectorCount = 2;
constexpr std::size_t kTimingTransferEnd =
    kTimingWindowBegin + kTimingVectorCount;
constexpr std::size_t kTimingTrailingRow = 777;
constexpr std::uint64_t kTimingDdr4Address = 0x4000;
constexpr std::size_t kTimingLaunchEvent = 0x5a5a;
constexpr std::size_t kTimingSyncTag = 1;

std::size_t timing_mem_queue()
{
    return InstructionControlUnit::mem_queue(
        kTimingHemisphere, kTimingSlice, kTimingBank);
}

C2cWeightPage timing_page()
{
    return C2cWeightPage {
        7,
        kTimingBank,
        {C2cWeightSegment {kTimingHemisphere, kTimingSlice, kTimingBank,
            200, 0, kTimingDdr4Address, kTimingVectorCount}},
    };
}

BinaryProgram common_paging_program()
{
    BinaryProgram program;
    program.common_barrier_protocol = true;
    program.max_cycle = 2048;
    BinaryBinding binding;
    binding.index = 0;
    binding.paged_weight = true;
    binding.page_count = 2;
    binding.bank = kTimingBank;
    binding.page_banks = {kTimingBank, kTimingBank};
    binding.page_slice_group_bases = {0, 0};
    binding.page_slice_group_counts = {1, 1};
    binding.page_base_rows = {0, 16};
    binding.page_row_counts = {16, 16};
    program.bindings.push_back(binding);
    program.weight_page_uses = {
        {0, 0, kTimingBank, 512, 1024, true},
        {0, 1, kTimingBank, 512, 1024, true},
        {0, 0, kTimingBank, 1536, 2048, true}};
    program.timelines = {
        {"c2c.window.binding:0.page:0", 100, 512},
        {"c2c.window.binding:0.page:1", 100, 512},
        {"c2c.window.binding:0.page:0", 1100, 1536}};
    const auto encoded = isa::encode_mem_instruction(
        MemInstruction::Read(200, StreamId::East(0)));
    QueueCommand read;
    read.command = static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);
    read.instruction_kind = InstructionKind::Mem;
    read.word_count = 2;
    read.words[0] = static_cast<std::uint32_t>(encoded);
    read.words[1] = static_cast<std::uint32_t>(encoded >> 32);
    QueueProgram macro {QueueKind::Mem, timing_mem_queue(),
        {encode_macro_schedule_command(read,
            IcuMacroSchedule {0, 1, 1, 0, 3, 1024, 1,
                IcuInductionTarget::MemAddress})}};
    macro.requires_raw_macro = true;
    program.queues.push_back(std::move(macro));
    program.address_relocations.push_back({0, BindingAccess::Input, QueueKind::Mem,
        static_cast<std::uint16_t>(timing_mem_queue()), 0});
    const auto nativeQueue = InstructionControlUnit::mem_queue(Hemisphere::East, 0, 0);
    IcuProgram extra;
    extra.emit_mem(544,0,MemInstruction::Read(50,StreamId::East(1)));
    // A second physical queue is unrelated to the paged-weight region.
    auto extraQueues=extra.encode_queues();
    for(auto& q:extraQueues) if(!q.commands.empty()) program.queues.push_back(std::move(q));
    return program;
}

void test_common_paging_linker()
{
    auto system = std::make_unique<C2cDmaSystem>();
    auto program = common_paging_program();
    C2cWeightPager pager(*system);
    pager.begin_schedule(program);
    const std::array<std::size_t, 1> first {0}, second {1}, repeat {2};
    const auto firstFence = pager.schedule(program, timing_page(), 100, 200, 512,
        0x11000, 0x31000, 100, first);
    auto otherPage = timing_page();
    otherPage.segments.front().base_row += 16;
    static_cast<void>(pager.schedule(program, otherPage, 100, 200, 512,
        0x11001, 0x31001, 100, second));
    static_cast<void>(pager.schedule(program, timing_page(), 1100, 1200, 1536,
        0x11002, 0x31002, 1100, repeat));
    const auto aliases = pager.coalesce_page_ready_events();
    pager.finalize_schedule(program);
    require(firstFence.transport_queue && firstFence.scheduled_start_cycle == 512
            && firstFence.scheduled_end_cycle == 512,
        "common paging consumed logical computation time for its transport");
    require(program.barriers.size() == 2 && program.barriers[0].stop_cycle == 512
            && program.barriers[1].stop_cycle == 1536,
        "common pager lost logical stops or repeated page instances");
    require(program.barrier_events.size() == 2
            && program.barrier_events[0].page_use_indices == std::vector<std::uint32_t>({0, 1})
            && program.barrier_events[1].page_use_indices == std::vector<std::uint32_t>({2}),
        "same-cycle dependencies were not merged as exact all-of instances");
    require(std::ranges::find(aliases, std::pair<std::size_t, std::size_t> {0x31001, 0x31000}) != aliases.end(),
        "same-cycle event alias was not exported to the runtime");
    require(program.queues.front().requires_raw_macro
            && program.queues.front().commands.size() == 1
            && decode_macro_schedule_command(program.queues.front().commands.front()).outer_count == 3,
        "common paging split or converted a Macro crossing multiple stops");
    require(program.transport_queues.size() == 1
            && program.transport_queues.front().commands.size() == 6
            && pager.page_ready_releases(0x31000).empty(),
        "common paging did not use independent MEM control or retained queue phase offsets");

    const auto rejected = [&](BinaryProgram bad, std::size_t earliest, const char* message) {
        C2cWeightPager invalid(*system);
        invalid.begin_schedule(bad);
        bool failed = false;
        try {
            static_cast<void>(invalid.schedule(bad, timing_page(), 100, 200, 512,
                0x11000, 0x31000, earliest, first));
            invalid.finalize_schedule(bad);
        } catch (const std::logic_error&) { failed = true; }
        require(failed, message);
    };
    auto relinked = program;
    C2cWeightPager relinker(*system);
    relinker.begin_schedule(relinked);
    require(relinked.barriers.empty() && relinked.barrier_events.empty()
            && relinked.transport_queues.empty()
            && std::ranges::all_of(relinked.weight_page_uses, [](const auto& use) {
                return use.transport_launch_event_id == 0;
            }), "common relink retained stale barriers or transport instances");
    auto missing = common_paging_program();
    missing.timelines.clear();
    rejected(std::move(missing), 100, "common paging accepted an unproved ingress candidate");
    auto active = common_paging_program();
    auto schedule = decode_macro_schedule_command(active.queues.front().commands.front());
    schedule.start_cycle = 500;
    auto instruction = active.queues.front().commands.front();
    instruction.command = static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);
    instruction.extension_words.clear();
    active.queues.front().commands.front() = encode_macro_schedule_command(instruction, schedule);
    rejected(std::move(active), 100, "common paging accepted an in-flight producer in the drain window");
    auto liveStorage = common_paging_program();
    liveStorage.address_relocations.front().binding_access = BindingAccess::Internal;
    rejected(std::move(liveStorage), 100,
        "common paging overwrote stable SRAM still required by future computation");
    rejected(common_paging_program(), 513,
        "common paging accepted a launch dependent on frozen future logical time");
}

} // namespace
int main() try {
    test_common_paging_linker();
    std::cout<<"c2c_weight_ping_pong_test passed: all-of, repeated pages, relink, SRAM protection, Macro preservation\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
