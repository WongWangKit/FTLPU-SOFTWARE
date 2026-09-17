#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/binary.hpp"

#include "ftlpu/icu/instruction.hpp"

#include <algorithm>
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

std::vector<QueueCommand> raw_mem_3d_commands(
    const isa::EncodedMemIcu3DPacket& packet)
{
    std::vector<QueueCommand> commands;
    commands.reserve(packet.words.size());
    for (const auto& word : packet.words) {
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mem;
        command.word_count = isa::EncodedMemIcu3DPacket::kLanesPerWord;
        for (std::size_t lane = 0; lane < command.word_count; ++lane)
            command.words[lane] = word.lanes[lane];
        commands.push_back(std::move(command));
    }
    return commands;
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

struct LinkedTimingProgram {
    BinaryProgram binary{};
    C2cWeightPageFence fence{};
};

LinkedTimingProgram make_linked_timing_program(
    C2cDmaSystem& system, std::size_t readyCycle)
{
    require(readyCycle >= kTimingTransferEnd,
        "test MEM_WRITE_SYNC ready cycle precedes transfer end");

    BinaryProgram program;
    program.max_cycle = readyCycle + 1;
    QueueProgram memQueue {QueueKind::Mem, timing_mem_queue(), {}};
    memQueue.commands.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(readyCycle)));
    const auto trailing = raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Read3D(
            IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(kTimingTrailingRow, {0, 0, 0}),
            StreamId::East(0))));
    memQueue.commands.insert(memQueue.commands.end(),
        trailing.begin(), trailing.end());
    program.queues.push_back(std::move(memQueue));

    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    const auto fence = pager.schedule(program, timing_page(),
        kTimingWindowBegin, kTimingTransferEnd,
        readyCycle, kTimingLaunchEvent);
    pager.finalize_schedule(program);
    require(fence.scheduled_start_cycle == kTimingWindowBegin
            && fence.scheduled_end_cycle == kTimingTransferEnd,
        "linker moved the timing regression window");
    return LinkedTimingProgram {std::move(program), fence};
}

struct LinkedTimingResult {
    std::optional<std::size_t> last_sync_issue_cycle{};
    std::optional<std::size_t> trailing_mem_issue_cycle{};
    std::optional<std::size_t> pager_ready_cycle{};
    std::size_t synchronized_issue_count{0};
};

LinkedTimingResult run_controlled_linked_timing_case(
    std::size_t readyCycle,
    const std::vector<std::size_t>& notificationCycles)
{
    auto system = std::make_unique<C2cDmaSystem>();
    const auto linked = make_linked_timing_program(*system, readyCycle);
    const auto linkedMem = std::find_if(linked.binary.queues.begin(),
        linked.binary.queues.end(), [](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == timing_mem_queue();
        });
    require(linkedMem != linked.binary.queues.end(),
        "linked timing program lost its MEM queue");

    InstructionControlUnit icu;
    load_queue_programs_into_icu({*linkedMem}, icu);
    auto& mem = icu.mem_iq(timing_mem_queue());
    LinkedTimingResult result;
    for (std::size_t cycle = 0; cycle <= readyCycle + 4; ++cycle) {
        for (const auto notificationCycle : notificationCycles)
            if (notificationCycle == cycle)
                mem.notify(kTimingSyncTag);
        std::optional<MemInstruction> issue;
        try {
            issue = mem.tick();
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "controlled linked MEM timing failed at cycle "
                + std::to_string(cycle) + ": " + error.what());
        }
        if (mem.last_trace().action == IcuQueueAction::SynchronizedIssue)
            result.last_sync_issue_cycle = cycle;
        if (issue.has_value() && issue->opcode == MemOpcode::Read
            && issue->address == kTimingTrailingRow) {
            result.trailing_mem_issue_cycle = cycle;
            break;
        }
    }
    result.synchronized_issue_count = mem.synchronized_issued_count();
    return result;
}

Ddr4Config deterministic_ddr4()
{
    Ddr4Config config;
    config.beat_bytes = hw::kPhysicalVectorBytes;
    config.read_latency_cycles = 2;
    config.write_latency_cycles = 2;
    config.request_queue_depth = 32;
    config.transfer_channels = 1;
    config.peak_bandwidth_bytes_per_second =
        config.lpu_clock_hz * config.beat_bytes;
    config.read_latency_jitter_cycles = 0;
    config.write_latency_jitter_cycles = 0;
    return config;
}

LinkedTimingResult run_real_linked_timing_case(std::size_t readyCycle)
{
    auto system = std::make_unique<C2cDmaSystem>(deterministic_ddr4());
    const auto linked = make_linked_timing_program(*system, readyCycle);
    for (std::size_t vector = 0; vector < kTimingVectorCount; ++vector)
        system->ddr4().initialize_vector(
            kTimingDdr4Address + vector * hw::kPhysicalVectorBytes,
            make_vector(static_cast<std::uint8_t>(0x70 + vector * 0x10)));
    load_queue_programs_into_icu(
        linked.binary.queues, system->chip().icu());

    auto& mem = system->chip().icu().mem_iq(timing_mem_queue());
    C2cWeightPager readinessObserver(*system);
    LinkedTimingResult result;
    for (std::size_t cycle = 0; cycle <= readyCycle + 32; ++cycle) {
        if (cycle == kTimingWindowBegin) {
            system->chip().icu().notify_tagged(
                IcuLocation::C2cDma(kTimingHemisphere),
                kTimingLaunchEvent);
            system->chip().icu().notify_tagged(
                IcuLocation::C2cRx(kTimingHemisphere),
                kTimingLaunchEvent);
        }
        try {
            system->tick();
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "real linked C2C timing failed at cycle "
                + std::to_string(cycle) + ": " + error.what());
        }
        if (mem.last_trace().action == IcuQueueAction::SynchronizedIssue)
            result.last_sync_issue_cycle = cycle;
        if (!result.pager_ready_cycle
            && readinessObserver.ready(linked.fence))
            result.pager_ready_cycle = system->cycle();
        if (mem.last_trace().action == IcuQueueAction::Mem3DIssue
            && mem.last_dispatched().has_value()
            && mem.last_dispatched()->opcode == MemOpcode::Read
            && mem.last_dispatched()->address == kTimingTrailingRow) {
            result.trailing_mem_issue_cycle = cycle;
        }
        if (result.trailing_mem_issue_cycle && result.pager_ready_cycle)
            break;
    }
    result.synchronized_issue_count = mem.synchronized_issued_count();
    return result;
}

void test_linked_mem_window_preserves_trailing_3d_cycle()
{
    constexpr std::size_t readyCycle = 11;
    const auto early = run_controlled_linked_timing_case(readyCycle,
        {kTimingWindowBegin, kTimingWindowBegin + 1});
    const auto boundary = run_controlled_linked_timing_case(readyCycle,
        {readyCycle - 3, readyCycle - 2});
    require(early.synchronized_issue_count == kTimingVectorCount
            && early.last_sync_issue_cycle == kTimingWindowBegin + 2,
        "controlled early MEM_WRITE_SYNC did not issue both writes");
    require(boundary.synchronized_issue_count == kTimingVectorCount
            && boundary.last_sync_issue_cycle == readyCycle - 1
            && *boundary.last_sync_issue_cycle >= kTimingTransferEnd,
        "controlled late MEM_WRITE_SYNC did not finish on the reservation boundary");
    require(early.trailing_mem_issue_cycle == readyCycle
            && boundary.trailing_mem_issue_cycle == readyCycle,
        "linked MEM_WRITE_SYNC completion shifted the following MEM 3-D cycle");

    // First observe the deterministic end-to-end completion with ample slack,
    // then make that same final write land on the last reserved cycle. This
    // exercises WAIT_EVENT -> DMA -> RX -> MEM_WRITE_SYNC without baking the
    // transport pipeline latency into the test.
    constexpr std::size_t probeReadyCycle = 64;
    const auto realEarly = run_real_linked_timing_case(probeReadyCycle);
    require(realEarly.synchronized_issue_count == kTimingVectorCount
            && realEarly.last_sync_issue_cycle.has_value()
            && *realEarly.last_sync_issue_cycle + 1 < probeReadyCycle
            && realEarly.trailing_mem_issue_cycle == probeReadyCycle
            && realEarly.pager_ready_cycle.has_value()
            && *realEarly.pager_ready_cycle >= probeReadyCycle,
        "real linked C2C path did not complete inside its probe reservation");
    const auto realBoundaryReadyCycle =
        *realEarly.last_sync_issue_cycle + 1;
    require(realBoundaryReadyCycle > kTimingTransferEnd,
        "real C2C completion did not exercise post-transfer page slack");
    const auto realBoundary =
        run_real_linked_timing_case(realBoundaryReadyCycle);
    require(realBoundary.synchronized_issue_count == kTimingVectorCount
            && realBoundary.last_sync_issue_cycle.has_value()
            && *realBoundary.last_sync_issue_cycle + 1
                == realBoundaryReadyCycle
            && realBoundary.pager_ready_cycle.has_value()
            && *realBoundary.pager_ready_cycle
                >= realBoundaryReadyCycle,
        "real linked C2C path did not reproduce its boundary completion");
    require(realBoundary.trailing_mem_issue_cycle
            == realBoundaryReadyCycle,
        "boundary MEM_WRITE_SYNC completion delayed the following real MEM 3-D issue");
}

void test_c2c_dma_frontend_tail()
{
    constexpr std::size_t packetCount = 20;
    constexpr auto iqDepth = InstructionControlUnit::C2cDmaIcu::iq_depth;
    static_assert(iqDepth >= C2cDmaIcuPacket::kWordCount);

    InstructionControlUnit::C2cDmaIcu dmaIcu;
    for (std::size_t packetIndex = 0;
         packetIndex < packetCount; ++packetIndex) {
        dmaIcu.push_encoded_c2c_dma_packet(C2cIcuPacketCodec::encode(
            C2cDmaIcuInstruction::Load(Hemisphere::West,
                packetIndex % hw::kC2cStreamsPerDirection,
                0x1000 + packetIndex * hw::kPhysicalVectorBytes,
                1, hw::kPhysicalVectorBytes,
                static_cast<std::uint32_t>(packetIndex + 1))));
    }

    constexpr auto packetsBeforeFirstDecodeWait = iqDepth - 1;
    static_assert(packetCount > packetsBeforeFirstDecodeWait);
    constexpr auto expectedDecodeWaits =
        packetCount > packetsBeforeFirstDecodeWait
        ? packetCount - packetsBeforeFirstDecodeWait
        : 0;
    constexpr auto expectedCycles = packetCount + expectedDecodeWaits;
    std::size_t elapsedCycles = 0;
    std::size_t decodeWaitCycles = 0;
    while (!dmaIcu.done() && elapsedCycles < expectedCycles) {
        static_cast<void>(dmaIcu.dispatch_next());
        if (dmaIcu.last_trace().action
            == IcuQueueAction::ThreeDDecodeWait)
            ++decodeWaitCycles;
        ++elapsedCycles;
    }

    require(dmaIcu.done() && !dmaIcu.underflowed(),
        "20-packet C2C DMA queue did not drain without underflow");
    require(dmaIcu.issued_count() == packetCount,
        "20-packet C2C DMA queue issued the wrong packet count");
    require(decodeWaitCycles == expectedDecodeWaits
            && elapsedCycles == expectedCycles,
        "C2C DMA frontend tail did not match its finite-IQ model");
}

} // namespace

void run_test()
{
    test_c2c_dma_frontend_tail();
    test_linked_mem_window_preserves_trailing_3d_cycle();

    constexpr auto hemisphere = Hemisphere::West;
    constexpr std::size_t slice = 16;
    constexpr std::size_t current_bank = 0;
    constexpr std::size_t next_bank = 1;
    constexpr std::size_t current_row = 7;
    constexpr std::size_t next_row = 41;
    constexpr std::size_t vectors = 3;
    constexpr std::uint64_t ddr4_address = 0x1000;

    const C2cWeightPage logicalPage {
        1,
        next_bank,
        {C2cWeightSegment {hemisphere, slice, next_bank, next_row, 5,
            ddr4_address, vectors}},
    };
    const SystemHardwareConfiguration hardware;
    const auto lowered = lower_c2c_weight_page_to_icu(
        logicalPage, hardware, 37);
    require(lowered.segments.size() == 1,
        "direct C2C lowering changed the segment count");
    require(lowered.segments[0].mem_queue
            == InstructionControlUnit::mem_queue(
                hemisphere, slice, next_bank),
        "direct C2C lowering targeted the wrong unified MEM queue");
    require(lowered.physical_word_count() == 5,
        "one C2C segment must occupy five 96-bit i-MEM words");
    const auto decodedRx = C2cIcuPacketCodec::decode_rx(
        lowered.segments[0].rx);
    const auto decodedDma = C2cIcuPacketCodec::decode_dma(
        lowered.segments[0].dma);
    require(decodedRx.sync_tag == 37 && decodedDma.sync_tag == 37
            && lowered.segments[0].sync_tag == 37,
        "direct C2C lowering did not join RX/MEM/DMA by one sync tag");
    require(decodedRx.notify.enabled
            && decodedRx.notify.mem_slice == slice
            && decodedRx.notify.mem_bank == next_bank,
        "direct C2C lowering lost the RX-to-MEM completion route");
    require(decodedDma.ddr4_address == ddr4_address
            && decodedDma.vector_count == vectors,
        "direct C2C lowering changed the DMA domain");

    BinaryProgram rawProgram;
    QueueProgram rxQueue {QueueKind::C2cRx,
        hemisphere_index(hemisphere), {}};
    rxQueue.commands.push_back(
        encode_c2c_endpoint_icu_packet(lowered.segments[0].rx));
    QueueProgram dmaQueue {QueueKind::C2cDma,
        hemisphere_index(hemisphere), {}};
    const auto dmaWords = encode_c2c_dma_icu_packet(
        lowered.segments[0].dma);
    dmaQueue.commands.insert(dmaQueue.commands.end(),
        dmaWords.begin(), dmaWords.end());
    QueueProgram memQueue {QueueKind::Mem,
        lowered.segments[0].mem_queue, {}};
    const auto memWords = encode_mem_synchronized_icu_packet(
        lowered.segments[0].mem_write_sync);
    memQueue.commands.insert(memQueue.commands.end(),
        memWords.begin(), memWords.end());
    rawProgram.queues = {rxQueue, dmaQueue, memQueue};
    std::stringstream binary(
        std::ios::in | std::ios::out | std::ios::binary);
    write_binary_program(rawProgram, binary);
    binary.seekg(0);
    const auto roundTrip = read_binary_program(binary);
    auto rawIcu = std::make_unique<InstructionControlUnit>();
    load_queue_programs_into_icu(roundTrip.queues, *rawIcu);
    require(rawIcu->c2c_rx_iq(hemisphere).imem_occupancy() == 1
            && rawIcu->c2c_dma_iq(hemisphere).imem_occupancy() == 2
            && rawIcu->mem_iq(lowered.segments[0].mem_queue)
                   .imem_occupancy() == 2,
        "binary loader did not preserve physical C2C/MEM i-MEM words");

    BinaryProgram linkedProgram;
    linkedProgram.max_cycle = 128;
    linkedProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(128))}});
    auto linkSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager linkPager(*linkSystem);
    linkPager.begin_schedule(linkedProgram);
    const auto linkedFence = linkPager.schedule(linkedProgram, logicalPage,
        20, 60, 96, 0x1234);
    linkPager.finalize_schedule(linkedProgram);
    const auto linkedMem = std::find_if(linkedProgram.queues.begin(),
        linkedProgram.queues.end(), [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    require(linkedFence.scheduled_start_cycle == 20
            && linkedFence.scheduled_end_cycle == 60,
        "pre-load linker changed an already legal MEM idle window");
    require(linkedMem != linkedProgram.queues.end()
            && linkedMem->commands.size() == 4
            && is_mem_synchronized_raw_packet_header(
                linkedMem->commands[1]),
        "pre-load linker appended MEM_WRITE_SYNC instead of splitting its NOP window");
    auto linkedIcu = std::make_unique<InstructionControlUnit>();
    load_queue_programs_into_icu(linkedProgram.queues, *linkedIcu);
    require(linkedIcu->mem_iq(lowered.segments[0].mem_queue)
                .imem_occupancy() == 4,
        "linked MEM queue did not occupy one physical i-MEM");

    C2cWeightPage multiQueuePage;
    multiQueuePage.layer = 2;
    multiQueuePage.bank = next_bank;
    constexpr std::size_t distinctMemQueues = 20;
    static_assert(slice + distinctMemQueues <= hw::kMemSliceColumns);
    for (std::size_t segmentIndex = 0;
         segmentIndex < distinctMemQueues; ++segmentIndex) {
        multiQueuePage.segments.push_back(C2cWeightSegment {hemisphere,
            static_cast<std::uint16_t>(slice + segmentIndex),
            next_bank, next_row,
            static_cast<std::uint16_t>(5 + segmentIndex),
            ddr4_address + segmentIndex * hw::kPhysicalVectorBytes, 1});
    }
    constexpr auto dmaPacketsBeforeFirstDecodeWait =
        InstructionControlUnit::C2cDmaIcu::iq_depth - 1;
    constexpr auto dmaFrontendWaits = distinctMemQueues
            > dmaPacketsBeforeFirstDecodeWait
        ? distinctMemQueues - dmaPacketsBeforeFirstDecodeWait
        : 0;
    constexpr std::size_t linkedStart = 5;
    constexpr std::size_t linkedTransferEnd = 6;
    constexpr auto linkedC2cTail = linkedStart + 1
        + distinctMemQueues + dmaFrontendWaits;
    BinaryProgram shortWindowProgram;
    auto shortWindowSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager shortWindowPager(*shortWindowSystem);
    shortWindowPager.begin_schedule(shortWindowProgram);
    const auto shortWindowFence = shortWindowPager.schedule(
        shortWindowProgram, multiQueuePage, linkedStart, linkedTransferEnd,
        linkedC2cTail, 0x2345);
    require(shortWindowFence.scheduled_end_cycle == linkedTransferEnd
            && shortWindowProgram.max_cycle == linkedC2cTail,
        "linked binary max_cycle did not cover the C2C DMA/RX queue tail");

    BinaryProgram boundedShortWindowProgram;
    auto boundedShortWindowSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager boundedShortWindowPager(*boundedShortWindowSystem);
    boundedShortWindowPager.begin_schedule(boundedShortWindowProgram);
    bool rejectedLateC2cTail = false;
    try {
        static_cast<void>(boundedShortWindowPager.schedule(
            boundedShortWindowProgram, multiQueuePage,
            linkedStart, linkedTransferEnd, linkedC2cTail - 1, 0x3456));
    } catch (const std::logic_error& error) {
        rejectedLateC2cTail = std::string(error.what()).find(
            "past its first consumer") != std::string::npos;
    }
    require(rejectedLateC2cTail,
        "ready_cycle accepted a C2C DMA/RX queue tail after the consumer");

    BinaryProgram capacityProgram;
    capacityProgram.hardware.icu_mem_imem_depth = 3;
    capacityProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(128))}});
    auto capacitySystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager capacityPager(*capacitySystem);
    capacityPager.begin_schedule(capacityProgram);
    static_cast<void>(capacityPager.schedule(capacityProgram, logicalPage,
        20, 60, 96, 0x4567));
    bool rejectedMemImemOverflow = false;
    try {
        capacityPager.finalize_schedule(capacityProgram);
    } catch (const std::logic_error& error) {
        rejectedMemImemOverflow = std::string(error.what()).find(
            "MEM ICU i-MEM capacity exceeded") != std::string::npos;
    }
    require(rejectedMemImemOverflow,
        "pre-load linker exceeded the target MEM i-MEM depth");

    auto system = std::make_unique<C2cDmaSystem>(
        Ddr4Config {32, 2, 2, 256, 8});
    const auto current = make_vector(0x10);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            system->chip().initialize_mem_sram_lane_byte(hemisphere, slice,
                current_bank, tile, current_row, lane,
                current.payload[tile][lane]);

    for (std::size_t vector = 0; vector < vectors; ++vector)
        system->ddr4().initialize_vector(
            ddr4_address + vector * hw::kPhysicalVectorBytes,
            make_vector(static_cast<std::uint8_t>(0x40 + vector * 0x20)));

    const std::size_t current_queue = InstructionControlUnit::mem_queue(
        hemisphere, slice, current_bank);
    for (std::size_t cycle = 0; cycle < 96; ++cycle)
        system->chip().icu().enqueue_mem(current_queue,
            MemInstruction::Read(current_row, StreamId::East(23)));

    C2cWeightPager pager(*system);
    pager.enqueue(logicalPage);

    bool overlapped_bank_issue = false;
    for (std::size_t cycle = 0; cycle < 512 && !pager.ready(); ++cycle) {
        pager.tick();
        overlapped_bank_issue = overlapped_bank_issue
            || (system->chip().icu().mem_iq(current_queue).last_trace().action
                    == IcuQueueAction::FunctionalIssue
                && system->dma(hemisphere).last_beat().has_value());
    }

    require(pager.ready(), "next-layer C2C weight page did not become ready");
    require(overlapped_bank_issue,
        "current-bank reads did not overlap shared-SR C2C writes");
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
                const auto actual = system->chip().read_mem_sram_lane_byte(
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
