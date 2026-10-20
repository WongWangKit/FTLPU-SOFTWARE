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
    std::optional<std::size_t> transport_ready_cycle{};
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
    for (std::size_t cycle = 0; cycle <= readyCycle + 32; ++cycle) {
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
        if (!result.transport_ready_cycle
            && readinessObserver.transport_ready(linked.fence))
            result.transport_ready_cycle = system->cycle();
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

void test_short_dynamic_page_window()
{
    auto system = std::make_unique<C2cDmaSystem>();
    BinaryProgram program;
    program.max_cycle = 16;
    const auto memQueue = timing_mem_queue();
    program.queues.push_back({QueueKind::Mem, memQueue,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(5))}});
    const auto reads = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(kTimingTrailingRow, {0, 0, 0}), StreamId::East(0))));
    program.queues.front().commands.insert(program.queues.front().commands.end(), reads.begin(), reads.end());
    C2cWeightPager pager(*system);
    pager.begin_schedule(program);
    auto page = timing_page();
    page.segments[0].vector_count = 8;
    // Eight physical writes cannot fit the one-cycle gap before the consumer.
    // The compiler must arm WRITE_SYNC there and let real completion stall
    // the page event, without padding the calculation timeline.
    static_cast<void>(pager.schedule(program, page, 3, 80, 5,
        kTimingLaunchEvent, 0x7f00, 3));
    pager.finalize_schedule(program);
    const auto queue = std::ranges::find_if(program.queues, [&](const auto& q) {
        return q.kind == QueueKind::Mem && q.index == memQueue;
    });
    require(queue != program.queues.end(), "short page window lost MEM queue");
    const auto sync = std::ranges::find_if(queue->commands,
        [](const auto& c) { return is_mem_synchronized_raw_packet_header(c); });
    require(sync != queue->commands.end(), "short page window omitted WRITE_SYNC");
    require((sync->words[2] & 0xffffffU) + 1 < 8,
        "short page window inserted a fixed write-count delay");
    // Decode only this packet: each delivered token must still issue a write,
    // even though its encoded logical reservation is shorter than the count.
    const auto pc = static_cast<std::size_t>(sync - queue->commands.begin());
    QueueProgram isolated {QueueKind::Mem, memQueue,
        {queue->commands[pc], queue->commands[pc + 1]}};
    isolated.commands.insert(isolated.commands.end(), reads.begin(), reads.end());
    auto icu = std::make_unique<InstructionControlUnit>();
    load_queue_programs_into_icu({isolated}, *icu);
    auto& mem = icu->mem_iq(memQueue);
    std::optional<std::size_t> lastWrite, ordinaryRead;
    for (std::size_t cycle = 0; cycle < 96; ++cycle) {
        if (cycle >= 10 && cycle <= 66 && (cycle - 10) % 8 == 0) mem.notify(kTimingSyncTag);
        if (const auto issue = mem.tick()) {
            if (issue->opcode == MemOpcode::Write) lastWrite = cycle;
            if (issue->opcode == MemOpcode::Read) { ordinaryRead = cycle; break; }
        }
    }
    require(mem.synchronized_issued_count() == 8 && lastWrite && ordinaryRead
        && *lastWrite >= 66 && *ordinaryRead > *lastWrite,
        "short logical reservation retired before all actual writes");
}

void test_linked_mem_window_releases_after_ingress()
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
    require(early.trailing_mem_issue_cycle == readyCycle + 1,
        "early linked MEM_WRITE_SYNC did not release the port immediately after ingress");
    const auto boundaryTimingFailure =
        "late MEM_WRITE_SYNC ordering: trailing="
        + (boundary.trailing_mem_issue_cycle
            ? std::to_string(*boundary.trailing_mem_issue_cycle)
            : std::string("none"))
        + ", last sync="
        + (boundary.last_sync_issue_cycle
            ? std::to_string(*boundary.last_sync_issue_cycle)
            : std::string("none"));
    require(boundary.trailing_mem_issue_cycle.has_value()
            && *boundary.trailing_mem_issue_cycle >= readyCycle
            && *boundary.trailing_mem_issue_cycle
                > *boundary.last_sync_issue_cycle,
        boundaryTimingFailure.c_str());

    // Use the identical linked program but deliver page data after the
    // consumer's nominal cycle. Completion, rather than the predicted
    // reservation end, must keep the following ordinary read blocked.
    const auto late = run_controlled_linked_timing_case(readyCycle,
        {readyCycle + 8, readyCycle + 9});
    require(late.synchronized_issue_count == kTimingVectorCount
            && late.last_sync_issue_cycle.has_value()
            && *late.last_sync_issue_cycle > readyCycle
            && late.trailing_mem_issue_cycle.has_value()
            && *late.trailing_mem_issue_cycle > *late.last_sync_issue_cycle
            && *late.trailing_mem_issue_cycle > *early.trailing_mem_issue_cycle,
        "late completion did not dynamically delay the following MEM read");

    // First observe the deterministic end-to-end completion with ample slack,
    // then move the consumer boundary next to that completion. The MEM port is
    // reserved for ingress only; any transport delay beyond the static window
    // shifts later commands through the queue's normal WAIT_EVENT semantics.
    constexpr std::size_t probeReadyCycle = 64;
    const auto realEarly = run_real_linked_timing_case(probeReadyCycle);
    require(realEarly.synchronized_issue_count == kTimingVectorCount
            && realEarly.last_sync_issue_cycle.has_value()
            && *realEarly.last_sync_issue_cycle + 1 < probeReadyCycle
            && realEarly.transport_ready_cycle.has_value()
            && *realEarly.transport_ready_cycle < probeReadyCycle
            && realEarly.trailing_mem_issue_cycle.has_value()
            && *realEarly.trailing_mem_issue_cycle >= probeReadyCycle
            && *realEarly.trailing_mem_issue_cycle
                > *realEarly.last_sync_issue_cycle
            && realEarly.pager_ready_cycle.has_value()
            && *realEarly.pager_ready_cycle
                >= *realEarly.transport_ready_cycle,
        "real linked C2C path did not release the MEM port after ingress");
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
            && realBoundary.transport_ready_cycle.has_value()
            && realBoundary.pager_ready_cycle.has_value()
            && *realBoundary.pager_ready_cycle
                >= realBoundaryReadyCycle,
        "real linked C2C path did not reproduce its boundary completion");
    require(realBoundary.trailing_mem_issue_cycle.has_value()
            && *realBoundary.trailing_mem_issue_cycle
                >= realBoundaryReadyCycle
            && *realBoundary.trailing_mem_issue_cycle
                > *realBoundary.last_sync_issue_cycle,
        "following real MEM 3-D issue did not remain ordered after MEM_WRITE_SYNC");
}

void test_late_page_barrier_follows_its_mem_write_window()
{
    constexpr std::size_t startCycle = 3;
    constexpr std::size_t consumerCycle = 5;
    constexpr std::size_t transferEndCycle = 11;
    constexpr std::size_t pageReadyEvent = 0x6a6a;

    C2cDmaSystem system;
    BinaryProgram program;
    program.max_cycle = 24;
    program.hardware.mxms_per_hemisphere = 1;
    program.queues.push_back(QueueProgram{
        QueueKind::Mem, timing_mem_queue(),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(20))}});
    const auto crossingQueue = InstructionControlUnit::mem_queue(
        Hemisphere::East, 0, 0);
    auto crossingCommands =
        raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Read3D(
                IcuLoop3D{0, {16, 1, 1}, {1, 1, 1}},
                MemIcuAddress3D::Affine(300, {1, 0, 0}),
                StreamId::East(0))));
    crossingCommands.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(4)));
    program.queues.push_back(QueueProgram{
        QueueKind::Mem, crossingQueue, std::move(crossingCommands)});
    const auto zeroWaitQueue = InstructionControlUnit::mem_queue(
        Hemisphere::East, 1, 0);
    auto zeroWaitCommands = raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Read3D(
            IcuLoop3D{0, {16, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(400, {1, 0, 0}), StreamId::East(0))));
    auto followingDomain = raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Read3D(
            IcuLoop3D{0, {4, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(416, {1, 0, 0}), StreamId::East(0))));
    zeroWaitCommands.insert(zeroWaitCommands.end(),
        followingDomain.begin(), followingDomain.end());
    program.queues.push_back(QueueProgram {
        QueueKind::Mem, zeroWaitQueue, std::move(zeroWaitCommands)});
    const std::array consumerQueues {
        std::pair {QueueKind::MxmLoad, std::size_t {0}},
        std::pair {QueueKind::MxmDequant, std::size_t {0}},
        std::pair {QueueKind::MxmCompute, std::size_t {0}},
        std::pair {QueueKind::MxmCompute, std::size_t {1}},
        std::pair {QueueKind::Vxm, std::size_t {0}},
        std::pair {QueueKind::SxmTranspose, std::size_t {0}},
    };
    for (const auto& [kind, index] : consumerQueues)
        program.queues.push_back(QueueProgram {kind, index,
            {encode_icu_control_raw_word(
                IcuControlInstruction::Nop(20))}});

    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    static_cast<void>(pager.schedule(program, timing_page(), startCycle,
        transferEndCycle, consumerCycle, kTimingLaunchEvent,
        pageReadyEvent));
    pager.finalize_schedule(program);

    const auto releases = pager.page_ready_releases(pageReadyEvent);
    const auto crossingRelease = std::ranges::find_if(releases,
        [&](const C2cWeightPager::PageReadyRelease& release) {
            return release.location == IcuLocation::Mem(
                Hemisphere::East, 0, 0);
        });
    const auto zeroWaitRelease = std::ranges::find_if(releases,
        [](const C2cWeightPager::PageReadyRelease& release) {
            return release.location == IcuLocation::Mem(
                Hemisphere::East, 1, 0);
        });
    require(zeroWaitRelease != releases.end()
            && zeroWaitRelease->phase_offset == 12,
        "linked zero-wait SYNC issue was omitted from the release phase");
    const auto mxmRelease = std::ranges::find_if(releases,
        [](const C2cWeightPager::PageReadyRelease& release) {
            return release.location == IcuLocation::MxmLoad(0);
        });
    const auto westMxmRelease = std::ranges::find_if(releases,
        [](const C2cWeightPager::PageReadyRelease& release) {
            return release.location
                == IcuLocation::MxmCompute(hw::kMxmsPerHemisphere);
        });
    require(crossingRelease != releases.end()
            && mxmRelease != releases.end()
            && westMxmRelease != releases.end()
            && crossingRelease->phase_offset == 12
            && mxmRelease->phase_offset == 0
            && westMxmRelease->phase_offset == 0,
        "page-ready release plan did not preserve ICU wavefront phase or physical MXM mapping");

    const auto queue = std::ranges::find_if(program.queues,
        [](const QueueProgram& candidate) {
            return candidate.kind == QueueKind::Mem
                && candidate.index == timing_mem_queue();
        });
    require(queue != program.queues.end(),
        "late page barrier test lost its MEM queue");
    std::optional<std::size_t> writeSyncIndex;
    std::optional<std::size_t> pageWaitIndex;
    for (std::size_t index = 0; index < queue->commands.size(); ++index) {
        if (is_mem_synchronized_raw_packet_header(queue->commands[index]))
            writeSyncIndex = index;
        if (is_icu_control_raw_word_command(queue->commands[index])) {
            const auto control = decode_icu_control_raw_word(
                queue->commands[index]);
            if (control.opcode == IcuControlOpcode::WaitEvent
                && control.event_tag == pageReadyEvent)
                pageWaitIndex = index;
        }
    }
    require(writeSyncIndex.has_value() && pageWaitIndex.has_value()
            && *writeSyncIndex < *pageWaitIndex,
        "asynchronous MEM_WRITE_SYNC producer did not rejoin the compute epoch after its reservation");

    for (const auto& [kind, index] : consumerQueues) {
        const auto consumer = std::ranges::find_if(program.queues,
            [&](const QueueProgram& candidate) {
                return candidate.kind == kind && candidate.index == index;
            });
        require(consumer != program.queues.end(),
            "page barrier test lost a consumer ICU queue");
        const bool hasPageSync = std::ranges::any_of(consumer->commands,
            [&](const QueueCommand& command) {
                if (!is_icu_control_raw_word_command(command)) return false;
                const auto control = decode_icu_control_raw_word(command);
                return control.opcode == IcuControlOpcode::WaitEvent
                    && control.event_tag == pageReadyEvent;
            });
        require(hasPageSync,
            "page consumer ICU queue is missing its tagged SYNC");
    }

    const auto crossing = std::ranges::find_if(program.queues,
        [&](const QueueProgram& candidate) {
            return candidate.kind == QueueKind::Mem
                && candidate.index == crossingQueue;
        });
    require(crossing != program.queues.end()
            && !crossing->commands.empty()
            && is_fu_3d_raw_packet_header(crossing->commands.front()),
        "unrelated crossing MEM 3-D domain was gated by a page barrier");
    std::optional<std::size_t> deferredPageWait;
    std::size_t domainCount = 0;
    for (std::size_t index = 0; index < crossing->commands.size(); ++index) {
        if (is_fu_3d_raw_packet_header(crossing->commands[index]))
            ++domainCount;
        if (!is_icu_control_raw_word_command(crossing->commands[index]))
            continue;
        const auto control = decode_icu_control_raw_word(
            crossing->commands[index]);
        if (control.opcode == IcuControlOpcode::WaitEvent
            && control.event_tag == pageReadyEvent)
            deferredPageWait = index;
    }
    require(domainCount == 1 && deferredPageWait.has_value()
            && *deferredPageWait
                >= fu_3d_raw_packet_word_count(QueueKind::Mem),
        "page barrier split or interrupted the crossing 3-D ICU instruction");
}

void test_direct_compute_joins_late_activation_producer()
{
    C2cDmaSystem system;
    BinaryProgram program;
    program.max_cycle = 96;
    program.hardware.mxms_per_hemisphere = 1;
    program.queues.push_back(QueueProgram {QueueKind::Mem, timing_mem_queue(),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(96))}});
    program.queues.back().commands = {
        encode_icu_control_raw_word(IcuControlInstruction::Nop(41))};
    auto weightRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D{0, {4, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(200, {1, 0, 0}), StreamId::East(16))));
    program.queues.back().commands.insert(program.queues.back().commands.end(),
        weightRead.begin(), weightRead.end());
    const auto producerQueue = InstructionControlUnit::mem_queue(
        Hemisphere::East, 12, 0);
    auto reads = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D{0, {40, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(0, {1, 0, 0}), StreamId::East(0))));
    auto nextRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D{0, {4, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(40, {1, 0, 0}), StreamId::East(16))));
    reads.insert(reads.end(), nextRead.begin(), nextRead.end());
    program.queues.push_back(QueueProgram {QueueKind::Mem, producerQueue,
        std::move(reads)});
    const auto packet = isa::encode_mxm_compute_icu_3d_instruction(
        MxmComputeIcuInstruction::Compute3D(
            IcuLoop3D{0, {4, 1, 1}, {1, 1, 1}, 55}, 0,
            MxmIcuBufferMode::Fixed, 16, 0, 0, {0, 0, 0}, 1,
            MxmDataFormat::BFloat16, MxmComputeIcuMode {}));
    std::vector<QueueCommand> compute;
    for (const auto& word : packet.words) {
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mxm;
        command.word_count = packet.kLanesPerWord;
        for (std::size_t lane = 0; lane < command.word_count; ++lane)
            command.words[lane] = word.lanes[lane];
        compute.push_back(command);
    }
    program.queues.push_back(QueueProgram {QueueKind::MxmCompute, 0,
        std::move(compute)});
    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    constexpr std::size_t event = 0x7d00;
    static_cast<void>(pager.schedule(program, timing_page(), 3, 5, 16,
        0x7100, event));
    pager.finalize_schedule(program);
    const auto releases = pager.page_ready_releases(event);
    const auto producer = std::ranges::find_if(releases,
        [](const auto& release) {
            return release.location == IcuLocation::Mem(Hemisphere::East, 12, 0);
        });
    require(producer != releases.end() && producer->activation_producer
            && producer->phase_offset > 16,
        "direct BF16 COMPUTE_3D omitted its late MEM activation producer");
    const auto weight = std::ranges::find_if(releases, [](const auto& release) {
        return release.location == IcuLocation::Mem(kTimingHemisphere, kTimingSlice, kTimingBank);
    });
    require(weight != releases.end() && !weight->activation_producer,
        "weight ingress sharing the activation SR number received activation release timing");
}

void test_page_wait_preserves_partial_result_offset()
{
    C2cDmaSystem system;
    BinaryProgram program;
    program.max_cycle = 256;
    program.hardware.mxms_per_hemisphere = 1;
    program.queues.push_back(QueueProgram {QueueKind::Mem, timing_mem_queue(),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(256))}});
    constexpr std::size_t start = 64, offset = 16, count = 16, event = 0x7d20;
    const MxmComputeIcuMode mode {MxmAccumulatorDestination::Stream, true,
        MxmAccumulatorOutputFormat::BFloat16};
    const auto packet = isa::encode_mxm_compute_icu_3d_instruction(
        MxmComputeIcuInstruction::Compute3D(
            IcuLoop3D {0, {32, 1, 1}, {1, 1, 1}}, 0,
            MxmIcuBufferMode::Fixed, 16, 4, 0, {0, 0, 0}, 1,
            MxmDataFormat::BFloat16, mode,
            MxmComputeIcuInstruction::kNoTerminalDimension, mode));
    std::vector<QueueCommand> compute {
        encode_icu_control_raw_word(IcuControlInstruction::Nop(start))};
    for (const auto& word : packet.words) {
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mxm;
        command.word_count = packet.kLanesPerWord;
        for (std::size_t lane = 0; lane < command.word_count; ++lane)
            command.words[lane] = word.lanes[lane];
        compute.push_back(command);
    }
    program.queues.push_back(QueueProgram {QueueKind::MxmCompute, 0, std::move(compute)});
    const auto latency = program.hardware.mxm_pipeline_rows - 1
        + program.hardware.system_register_columns - 1;
    std::vector<QueueCommand> writer {encode_icu_control_raw_word(
        IcuControlInstruction::Nop(start + latency + offset))};
    const auto writes = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Write3D(IcuLoop3D {0, {count, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(0, {1, 0, 0}), StreamId::West(4))));
    writer.insert(writer.end(), writes.begin(), writes.end());
    const auto writerIndex = InstructionControlUnit::mem_queue(Hemisphere::East, 0, 1);
    program.queues.push_back(QueueProgram {QueueKind::Mem, writerIndex, std::move(writer)});
    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    static_cast<void>(pager.schedule(program, timing_page(), 3, 5, start, 0x7120, event));
    pager.finalize_schedule(program);
    const auto releases = pager.page_ready_releases(event);
    const auto computeRelease = std::ranges::find_if(releases, [](const auto& r) {
        return r.location == IcuLocation::MxmCompute(0);
    });
    const auto writerRelease = std::ranges::find_if(releases, [](const auto& r) {
        return r.location == IcuLocation::Mem(Hemisphere::East, 0, 1);
    });
    require(computeRelease != releases.end() && writerRelease != releases.end()
        && writerRelease->result_writer, "partial result writer missing from page release");
    const auto cq = std::ranges::find_if(program.queues, [](const auto& q) {
        return q.kind == QueueKind::MxmCompute && q.index == 0;
    });
    const auto wq = std::ranges::find_if(program.queues, [=](const auto& q) {
        return q.kind == QueueKind::Mem && q.index == writerIndex;
    });
    InstructionControlUnit icu;
    load_queue_programs_into_icu({*cq, *wq}, icu, 1);
    std::optional<std::size_t> firstCompute, firstWrite;
    std::size_t writeCount = 0;
    for (std::size_t cycle = 0; cycle < 512; ++cycle) {
        if (cycle == 128 + computeRelease->phase_offset) icu.mxm_compute_iq(0).notify(event);
        if (cycle == 128 + writerRelease->phase_offset) icu.mem_iq(writerIndex).notify(event);
        if (icu.mxm_compute_iq(0).tick() && !firstCompute) firstCompute = cycle;
        if (icu.mem_iq(writerIndex).tick()) {
            if (!firstWrite) firstWrite = cycle;
            ++writeCount;
        }
    }
    require(firstCompute && firstWrite && *firstWrite == *firstCompute + latency + offset
        && writeCount == count, "page wait lost the partial result token offset or count");
}

void test_page_wait_does_not_split_mxm_output_tile_pair(
    bool nonzeroResultStreamBase = false)
{
    C2cDmaSystem system;
    BinaryProgram program;
    program.max_cycle = 96;
    program.hardware.mxms_per_hemisphere = 1;
    program.queues.push_back(QueueProgram {QueueKind::Mem, timing_mem_queue(),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(96))}});
    const MxmComputeIcuMode streamMode {
        MxmAccumulatorDestination::Stream, true,
        MxmAccumulatorOutputFormat::BFloat16};
    const std::size_t resultStreamBase = nonzeroResultStreamBase ? 4 : 0;
    std::vector<QueueCommand> compute;
    for (std::size_t tokenTile = 0; tokenTile < 3; ++tokenTile) {
        const auto packet = isa::encode_mxm_compute_icu_3d_instruction(
            MxmComputeIcuInstruction::Compute3D(
                IcuLoop3D {0, {hw::kMxmRows, 1, 1}, {1, 1, 1}},
                1, MxmIcuBufferMode::Fixed, 16, resultStreamBase,
                tokenTile * hw::kMxmRows, {0, 0, 0}, 1,
                MxmDataFormat::BFloat16, streamMode,
                MxmComputeIcuInstruction::kNoTerminalDimension, streamMode));
        for (const auto& word : packet.words) {
            QueueCommand command;
            command.command = word.lanes[0];
            command.instruction_kind = InstructionKind::Mxm;
            command.word_count = packet.kLanesPerWord;
            for (std::size_t lane = 0; lane < command.word_count; ++lane)
                command.words[lane] = word.lanes[lane];
            compute.push_back(command);
        }
    }
    compute.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(64)));
    program.queues.push_back(QueueProgram {QueueKind::MxmCompute, 0,
        std::move(compute)});
    std::vector<QueueCommand> activations {
        encode_icu_control_raw_word(IcuControlInstruction::Nop(2))};
    for (std::size_t tile = 0; tile < 2; ++tile) {
        auto read = raw_mem_3d_commands(
            isa::encode_mem_icu_3d_instruction(
                MemIcuInstruction::Read3D(
                    IcuLoop3D {0, {hw::kMxmRows, 1, 1}, {1, 1, 1}},
                    MemIcuAddress3D::Affine(
                        tile * hw::kMxmRows, {1, 0, 0}),
                    StreamId::East(16))));
        activations.insert(activations.end(), read.begin(), read.end());
    }
    activations.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(32)));
    auto laterRead = raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Read3D(IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
                MemIcuAddress3D::Affine(64, {1, 0, 0}),
                StreamId::East(16))));
    activations.insert(activations.end(), laterRead.begin(), laterRead.end());
    auto earlyActivations = activations;
    earlyActivations.erase(earlyActivations.begin());
    program.queues.push_back(QueueProgram {QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 13, 0),
        std::move(earlyActivations)});
    program.queues.push_back(QueueProgram {QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 12, 0),
        std::move(activations)});
    for (const auto [slice, stream] : {
             std::pair {std::size_t {0}, StreamId::West(0)},
             std::pair {std::size_t {8}, StreamId::West(4)}}) {
        const auto arrival = program.hardware.mxm_pipeline_rows - 1
            + program.hardware.system_register_columns - 1
            - slice / program.hardware.mem_slices_per_register_group;
        std::vector<QueueCommand> writer {
            encode_icu_control_raw_word(IcuControlInstruction::Nop(arrival))};
        auto write = raw_mem_3d_commands(
            isa::encode_mem_icu_3d_instruction(
                MemIcuInstruction::Write3D(
                    IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
                    MemIcuAddress3D::Affine(0, {1, 0, 0}), stream)));
        for (std::size_t tile = 0; tile < 3; ++tile) {
            if (tile != 0)
                writer.push_back(encode_icu_control_raw_word(
                    IcuControlInstruction::Nop(hw::kMxmRows - 1)));
            writer.insert(writer.end(), write.begin(), write.end());
        }
        writer.push_back(encode_icu_control_raw_word(
            IcuControlInstruction::Nop(48)));
        program.queues.push_back(QueueProgram {QueueKind::Mem,
            InstructionControlUnit::mem_queue(Hemisphere::East, slice, 0),
            std::move(writer)});
    }
    // A later vector operator reuses the same westbound SR. It must not be
    // treated as an MXM result consumer and pull the whole epoch forward/back.
    std::vector<QueueCommand> laterWriter {
        encode_icu_control_raw_word(IcuControlInstruction::Nop(
            3 * hw::kMxmRows + 4 * program.hardware.system_register_columns))};
    const auto laterWrite = raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Write3D(
            IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(64, {1, 0, 0}),
            StreamId::West(resultStreamBase))));
    laterWriter.insert(laterWriter.end(), laterWrite.begin(), laterWrite.end());
    program.queues.push_back(QueueProgram {QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 32, 1),
        std::move(laterWriter)});
    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    constexpr std::size_t event = 0x7d10;
    static_cast<void>(pager.schedule(program, timing_page(), 3, 5,
        hw::kMxmRows, 0x7110, event));
    pager.finalize_schedule(program);
    const auto computeQueue = std::ranges::find_if(program.queues,
        [](const QueueProgram& queue) {
            return queue.kind == QueueKind::MxmCompute && queue.index == 0;
        });
    require(computeQueue != program.queues.end(),
        "page linker lost the MXM compute queue");
    std::size_t computePackets = 0;
    std::size_t packetsBeforeWait = 0;
    for (const auto& command : computeQueue->commands) {
        if (is_fu_3d_raw_packet_header(command)) ++computePackets;
        if (!is_icu_control_raw_word_command(command)) continue;
        const auto control = decode_icu_control_raw_word(command);
        if (control.opcode == IcuControlOpcode::WaitEvent
            && control.event_tag == event)
            packetsBeforeWait = computePackets;
    }
    require(packetsBeforeWait == 2,
        "page-ready wait split the two token tiles of one MXM output wave");
    const auto activationQueue = std::ranges::find_if(program.queues,
        [](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == InstructionControlUnit::mem_queue(
                    Hemisphere::East, 12, 0);
        });
    require(activationQueue != program.queues.end(),
        "page linker lost the activation producer queue");
    std::size_t readPackets = 0;
    std::size_t readsBeforeWait = 0;
    for (const auto& command : activationQueue->commands) {
        if (is_fu_3d_raw_packet_header(command)) ++readPackets;
        if (!is_icu_control_raw_word_command(command)) continue;
        const auto control = decode_icu_control_raw_word(command);
        if (control.opcode == IcuControlOpcode::WaitEvent
            && control.event_tag == event) {
            readsBeforeWait = readPackets;
        }
    }
    require(readsBeforeWait == 2,
        "page-ready wait split the activation reads of one MXM output wave");
    const auto releases = pager.page_ready_releases(event);
    const auto computeRelease = std::ranges::find_if(releases,
        [](const auto& release) {
            return release.location == IcuLocation::MxmCompute(0);
        });
    require(computeRelease != releases.end(),
        "output-pair compute queue was omitted from the page startup join");
    const auto resultWriter = std::ranges::find_if(releases,
        [&](const auto& release) {
            return release.location == IcuLocation::Mem(Hemisphere::East,
                nonzeroResultStreamBase ? 8 : 0, 0);
        });
    const auto otherWriter = std::ranges::find_if(releases,
        [&](const auto& release) {
            return release.location == IcuLocation::Mem(Hemisphere::East,
                nonzeroResultStreamBase ? 0 : 8, 0);
        });
    require(resultWriter != releases.end() && otherWriter != releases.end(),
        "output-pair MEM write queues were omitted from the page release");
    const auto unrelatedWriter = std::ranges::find_if(releases,
        [](const auto& release) {
            return release.location == IcuLocation::Mem(Hemisphere::East, 32, 1);
        });
    require(unrelatedWriter != releases.end(),
        "later SR reuse queue was omitted from the page release");
    require(!unrelatedWriter->result_writer,
        "later SR reuse was incorrectly linked to an earlier MXM result wave");
    if (nonzeroResultStreamBase)
        require(resultWriter->result_writer && !otherWriter->result_writer,
            "page linker did not identify the nonzero MXM result stream base");
    // Exercise real ICU WAIT_EVENT retirement, not just the linker's phase
    // labels. An arbitrarily late page must preserve physical result arrival
    // at both the first and a later MEM register group.
    const auto writerSlice = nonzeroResultStreamBase ? 8 : 0;
    const auto writerQueue = std::ranges::find_if(program.queues,
        [&](const auto& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == InstructionControlUnit::mem_queue(
                    Hemisphere::East, writerSlice, 0);
        });
    InstructionControlUnit icu;
    load_queue_programs_into_icu({*computeQueue, *writerQueue}, icu, 1);
    auto& computeIcu = icu.mxm_compute_iq(0);
    auto& writerIcu = icu.mem_iq(writerQueue->index);
    std::optional<std::size_t> firstCompute;
    std::optional<std::size_t> firstWrite;
    constexpr std::size_t latePageReady = 128;
    for (std::size_t cycle = 0; cycle < 512; ++cycle) {
        if (cycle == latePageReady + computeRelease->phase_offset)
            computeIcu.notify(event);
        if (cycle == latePageReady + resultWriter->phase_offset)
            writerIcu.notify(event);
        const auto computeIssue = computeIcu.tick();
        const auto writeIssue = writerIcu.tick();
        if (cycle >= latePageReady && computeIssue && !firstCompute)
            firstCompute = cycle;
        if (cycle >= latePageReady && writeIssue && !firstWrite)
            firstWrite = cycle;
        if (firstCompute && firstWrite) break;
    }
    const auto arrivalLatency = program.hardware.mxm_pipeline_rows - 1
        + program.hardware.system_register_columns - 1
        - writerSlice / program.hardware.mem_slices_per_register_group;
    if (!firstCompute || !firstWrite
        || *firstWrite != *firstCompute + arrivalLatency)
        throw std::runtime_error(
            "dynamic page release misaligned MXM result and MEM write at its SR column: compute="
            + std::to_string(firstCompute.value_or(0)) + " write="
            + std::to_string(firstWrite.value_or(0)) + " latency="
            + std::to_string(arrivalLatency) + " compute_phase="
            + std::to_string(computeRelease->phase_offset) + " writer_phase="
            + std::to_string(resultWriter->phase_offset) + " writer="
            + std::to_string(resultWriter->result_writer));
    std::size_t lastProducerPhase = 0;
    for (const auto slice : {12, 13}) {
        const auto producer = std::ranges::find_if(releases,
            [slice](const auto& release) {
                return release.location
                    == IcuLocation::Mem(Hemisphere::East, slice, 0);
            });
        require(producer != releases.end() && producer->activation_producer,
            "output-pair activation queue was omitted from the page startup join");
        // A nearby MEM slice can resume after the MXM ICU and still deliver
        // its vector before that vector reaches the MXM datapath. Reject
        // only a release outside the physical SR flight window.
        require(producer->phase_offset
                <= computeRelease->phase_offset
                    + program.hardware.system_register_columns,
            "activation producer was released outside the MXM SR flight window");
        require(computeRelease->phase_offset
                < producer->phase_offset
                    + program.hardware.system_register_columns,
            "MXM consumer was released after the activation SR flight window");
        lastProducerPhase = std::max(lastProducerPhase,
            producer->phase_offset);
    }
    require(computeRelease->phase_offset
            <= lastProducerPhase
                + program.hardware.mxm_pipeline_rows + 1,
        "MXM consumer was delayed beyond the activation ingress latency");
}

void test_closed_prefill_page_preserves_issue_phase()
{
    C2cDmaSystem system;
    BinaryProgram program;
    const auto rows = hw::kMxmRows;
    const auto inputLatency = program.hardware.system_register_columns;
    const auto outputLatency = program.hardware.mxm_pipeline_rows - 1
        + program.hardware.system_register_columns - 1;
    program.max_cycle = 16 * rows;
    program.queues.push_back({QueueKind::Mem, timing_mem_queue(),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(program.max_cycle))}});
    QueueProgram compute {QueueKind::MxmCompute, 0,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(inputLatency))}};
    QueueProgram read {QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 0, 1), {}};
    QueueProgram write {QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 0, 0),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(
            inputLatency + 2 * rows + outputLatency))}};
    const MxmComputeIcuMode accumulate {MxmAccumulatorDestination::Sram,
        true, MxmAccumulatorOutputFormat::Float32};
    const MxmComputeIcuMode stream {MxmAccumulatorDestination::Stream,
        true, MxmAccumulatorOutputFormat::BFloat16};
    for (std::size_t block = 0; block < 2; ++block) {
        const auto packet = isa::encode_mxm_compute_icu_3d_instruction(
            MxmComputeIcuInstruction::Compute3D(
                IcuLoop3D {0, {rows, 2, 2}, {1, rows, 2 * rows}},
                0, MxmIcuBufferMode::Fixed, 16, 0, 0,
                {0, static_cast<std::int64_t>(rows), 0}, 1,
                MxmDataFormat::BFloat16, accumulate, 2, stream));
        for (const auto& word : packet.words) {
            QueueCommand command;
            command.command = word.lanes[0];
            command.instruction_kind = InstructionKind::Mxm;
            command.word_count = packet.kLanesPerWord;
            for (std::size_t lane = 0; lane < command.word_count; ++lane)
                command.words[lane] = word.lanes[lane];
            compute.commands.push_back(command);
        }
        const auto reads = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Read3D(IcuLoop3D {0, {4 * rows, 1, 1}, {1, 1, 1}},
                MemIcuAddress3D::Affine(block * 4 * rows, {1, 0, 0}), StreamId::East(16))));
        read.commands.insert(read.commands.end(), reads.begin(), reads.end());
        if (block) write.commands.push_back(encode_icu_control_raw_word(
            IcuControlInstruction::Nop(2 * rows)));
        const auto writes = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Write3D(IcuLoop3D {0, {2 * rows, 1, 1}, {1, 1, 1}},
                MemIcuAddress3D::Affine(block * 2 * rows, {1, 0, 0}), StreamId::West(0))));
        write.commands.insert(write.commands.end(), writes.begin(), writes.end());
    }
    program.queues.push_back(compute);
    program.queues.push_back(read);
    program.queues.push_back(write);
    constexpr std::size_t event = 0x7d20;
    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    static_cast<void>(pager.schedule(program, timing_page(), 3, 5,
        inputLatency + rows, 0x7120, event));
    pager.finalize_schedule(program);
    const auto releases = pager.page_ready_releases(event);
    const auto phase = [&](const IcuLocation& location) {
        const auto found = std::ranges::find_if(releases,
            [&](const auto& item) { return item.location == location; });
        require(found != releases.end(), "closed prefill queue omitted from page join");
        return found->phase_offset;
    };
    InstructionControlUnit icu;
    std::vector<QueueProgram> queues;
    for (const auto& queue : program.queues)
        if (queue.kind == QueueKind::MxmCompute
            || (queue.kind == QueueKind::Mem
                && (queue.index == read.index || queue.index == write.index)))
            queues.push_back(queue);
    load_queue_programs_into_icu(queues, icu);
    auto& reader = icu.mem_iq(read.index);
    auto& writer = icu.mem_iq(write.index);
    auto& consumer = icu.mxm_compute_iq(0);
    std::optional<std::size_t> readCycle, computeCycle, writeCycle;
    constexpr std::size_t pageReady = 512;
    for (std::size_t cycle = 0; cycle < 2048; ++cycle) {
        if (cycle == pageReady + phase(IcuLocation::Mem(Hemisphere::East, 0, 1)))
            reader.notify(event);
        if (cycle == pageReady + phase(IcuLocation::Mem(Hemisphere::East, 0, 0)))
            writer.notify(event);
        if (cycle == pageReady + phase(IcuLocation::MxmCompute(0))) consumer.notify(event);
        const auto r = reader.tick();
        const auto c = consumer.tick();
        const auto w = writer.tick();
        if (cycle >= pageReady && r && !readCycle) readCycle = cycle;
        if (cycle >= pageReady && c && !computeCycle) computeCycle = cycle;
        if (cycle >= pageReady && w && !writeCycle) writeCycle = cycle;
        if (readCycle && computeCycle && writeCycle) break;
    }
    require(readCycle && computeCycle && writeCycle
            && *computeCycle == *readCycle + inputLatency
            && *writeCycle == *computeCycle + 2 * rows + outputLatency,
        "closed prefill page did not preserve compiler FU issue phases");
}

void test_same_boundary_pages_share_one_execution_epoch()
{
    constexpr std::size_t readyCycle = 32;
    constexpr std::size_t epochEvent = 0x7e00;
    C2cDmaSystem system;
    BinaryProgram program;
    program.max_cycle = 48;
    program.queues.push_back(QueueProgram {QueueKind::Vxm, 0,
        {encode_icu_control_raw_word(
            IcuControlInstruction::Nop(program.max_cycle))}});

    auto first = timing_page();
    auto second = timing_page();
    second.layer = first.layer + 1;
    second.segments[0].slice = kTimingSlice - 1;
    second.segments[0].ddr4_address = kTimingDdr4Address + 0x1000;

    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    static_cast<void>(pager.schedule(program, first,
        3, 5, readyCycle, 0x7100, epochEvent));
    static_cast<void>(pager.schedule(program, second,
        5, 7, readyCycle, 0x7101, epochEvent));
    pager.finalize_schedule(program);

    const auto vxm = std::ranges::find_if(program.queues,
        [](const QueueProgram& queue) {
            return queue.kind == QueueKind::Vxm && queue.index == 0;
        });
    require(vxm != program.queues.end(),
        "execution epoch test lost its VXM queue");
    const auto waits = std::ranges::count_if(vxm->commands,
        [](const QueueCommand& command) {
            if (!is_icu_control_raw_word_command(command)) return false;
            const auto control = decode_icu_control_raw_word(command);
            return control.opcode == IcuControlOpcode::WaitEvent
                && control.event_tag == epochEvent;
        });
    require(waits == 1,
        "same-boundary pages emitted more than one ICU epoch wait");
    const auto releases = pager.page_ready_releases(epochEvent);
    require(std::ranges::count_if(releases,
                [](const C2cWeightPager::PageReadyRelease& release) {
                    return release.location == IcuLocation::Vxm(0);
                }) == 1,
        "same-boundary pages emitted duplicate epoch releases");
}

void test_early_page_event_is_latched_until_consumer()
{
    constexpr std::size_t eventTag = 0x6b6b;
    InstructionControlUnit icu;
    const auto location = IcuLocation::Mem(Hemisphere::East, 0, 0);
    const auto queueIndex = InstructionControlUnit::mem_queue(
        Hemisphere::East, 0, 0);
    icu.enqueue_control(location,
        IcuControlInstruction::Nop(3));
    icu.enqueue_control(location,
        IcuControlInstruction::WaitEvent(eventTag));
    icu.enqueue_mem(queueIndex,
        MemInstruction::Read(123, StreamId::East(0)));

    auto& queue = icu.mem_iq(queueIndex);
    for (std::size_t cycle = 0;
         cycle < 16 && queue.last_trace().action != IcuQueueAction::Nop;
         ++cycle)
        static_cast<void>(queue.tick());
    require(queue.last_trace().action == IcuQueueAction::Nop,
        "page-event latch test did not enter its queue delay");
    queue.notify(eventTag);
    std::optional<MemInstruction> released;
    for (std::size_t cycle = 0; cycle < 16 && !released; ++cycle) {
        released = queue.tick();
        require(queue.last_trace().action != IcuQueueAction::EventWait,
            "early page event was lost before WAIT_EVENT reached the head");
    }
    require(released.has_value()
            && released->opcode == MemOpcode::Read
            && released->address == 123,
        "early page event was not latched until its MEM consumer");
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
    test_late_page_barrier_follows_its_mem_write_window();
    test_direct_compute_joins_late_activation_producer();
    test_page_wait_does_not_split_mxm_output_tile_pair();
    test_page_wait_does_not_split_mxm_output_tile_pair(true);
    test_closed_prefill_page_preserves_issue_phase();
    test_same_boundary_pages_share_one_execution_epoch();
    test_early_page_event_is_latched_until_consumer();
    test_c2c_dma_frontend_tail();
    test_short_dynamic_page_window();
    test_linked_mem_window_releases_after_ingress();

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
    require(lowered.physical_word_count() == 6
            && lowered.dma_completion_sync[hemisphere_index(hemisphere)]
            && !lowered.dma_completion_sync[1 - hemisphere_index(hemisphere)],
        "C2C lowering must emit a DMA completion SYNC on the used ICU");
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

    auto wrapPage = logicalPage;
    wrapPage.segments.push_back(C2cWeightSegment {
        hemisphere, slice + 1, next_bank, next_row,
        5, ddr4_address + vectors * hw::kPhysicalVectorBytes, vectors});
    const auto wrapped = lower_c2c_weight_page_to_icu(
        wrapPage, hardware, 65535);
    require(wrapped.segments[0].sync_tag == 65535
            && wrapped.segments[1].sync_tag == 1
            && wrapped.next_sync_tag == 2
            && C2cIcuPacketCodec::decode_rx(
                wrapped.segments[1].rx).sync_tag == 1
            && C2cIcuPacketCodec::decode_dma(
                wrapped.segments[1].dma).sync_tag == 1,
        "C2C tag recycling did not preserve RX/DMA/MEM identity");

    auto mappedPage = logicalPage;
    mappedPage.fabric_streams = {29};
    const auto mapped = lower_c2c_weight_page_to_icu(
        mappedPage, hardware, 51);
    const auto mappedRx = C2cIcuPacketCodec::decode_rx(
        mapped.segments[0].rx);
    require(mapped.segments[0].lane == 0
            && mapped.segments[0].fabric_stream == 29
            && mappedRx.lane == 0 && mappedRx.fabric_stream == 29,
        "C2C transport lane was not mapped to the selected idle West SR");

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
            && linkedMem->commands.size() == 3
            && is_mem_synchronized_raw_packet_header(
                linkedMem->commands[0])
            && (linkedMem->commands[0].words[2] & 0xffffffU) + 1 == 60,
        "pre-load linker did not reserve MEM_WRITE_SYNC through page ingress");
    auto linkedIcu = std::make_unique<InstructionControlUnit>();
    load_queue_programs_into_icu(linkedProgram.queues, *linkedIcu);
    require(linkedIcu->mem_iq(lowered.segments[0].mem_queue)
                .imem_occupancy() == 3,
        "linked MEM queue did not occupy one physical i-MEM");

    BinaryProgram separatedPagesProgram;
    separatedPagesProgram.max_cycle = 200;
    separatedPagesProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(200))}});
    auto separatedPagesSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager separatedPagesPager(*separatedPagesSystem);
    separatedPagesPager.begin_schedule(separatedPagesProgram);
    static_cast<void>(separatedPagesPager.schedule(separatedPagesProgram,
        logicalPage, 20, 60, 96, 0x1240));
    auto secondLogicalPage = logicalPage;
    secondLogicalPage.segments[0].base_row = next_row + vectors;
    secondLogicalPage.segments[0].ddr4_address =
        ddr4_address + vectors * hw::kPhysicalVectorBytes;
    const auto secondFence = separatedPagesPager.schedule(
        separatedPagesProgram, secondLogicalPage,
        100, 140, 160, 0x1241);
    separatedPagesPager.finalize_schedule(separatedPagesProgram);
    require(secondFence.scheduled_start_cycle == 100,
        "second separated C2C page lost its planned start cycle");
    const auto separatedDma = std::find_if(
        separatedPagesProgram.queues.begin(),
        separatedPagesProgram.queues.end(), [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::C2cDma
                && queue.index == hemisphere_index(hemisphere);
        });
    require(separatedDma != separatedPagesProgram.queues.end(),
        "separated C2C pages lost their DMA ICU queue");
    std::size_t separatedWaitEvents = 0;
    std::size_t separatedSyncs = 0;
    std::size_t separatedNops = 0;
    for (const auto& command : separatedDma->commands) {
        if (!is_icu_control_raw_word_command(command)) continue;
        const auto control = decode_icu_control_raw_word(command);
        separatedWaitEvents +=
            control.opcode == IcuControlOpcode::WaitEvent ? 1 : 0;
        separatedSyncs += control.opcode == IcuControlOpcode::Sync ? 1 : 0;
        separatedNops += control.opcode == IcuControlOpcode::Nop ? 1 : 0;
    }
    require(separatedWaitEvents == 2 && separatedSyncs == 2
            && separatedNops == 0,
        "C2C page queue encoded a redundant static gap before WAIT_EVENT");

    BinaryProgram boundaryLaunchProgram;
    boundaryLaunchProgram.max_cycle = 128;
    boundaryLaunchProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(128))}});
    auto boundaryLaunchSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager boundaryLaunchPager(*boundaryLaunchSystem);
    boundaryLaunchPager.begin_schedule(boundaryLaunchProgram);
    const auto boundaryLaunchFence = boundaryLaunchPager.schedule(
        boundaryLaunchProgram, logicalPage, 96, 136, 96, 0x2234, 0x2235);
    boundaryLaunchPager.finalize_schedule(boundaryLaunchProgram);
    const auto boundaryLaunchMem = std::find_if(
        boundaryLaunchProgram.queues.begin(),
        boundaryLaunchProgram.queues.end(),
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    const auto boundaryLaunchHasWrite =
        boundaryLaunchMem != boundaryLaunchProgram.queues.end()
        && std::ranges::any_of(boundaryLaunchMem->commands,
            [](const QueueCommand& command) {
                return is_mem_synchronized_raw_packet_header(command);
            });
    require(boundaryLaunchFence.scheduled_start_cycle == 96
            && boundaryLaunchHasWrite,
        "slow DDR page was moved behind its consumer instead of waiting in MEM_WRITE_SYNC");
    const auto boundaryWrite = std::ranges::find_if(boundaryLaunchMem->commands,
        [](const QueueCommand& command) {
            return is_mem_synchronized_raw_packet_header(command);
        });
    require(boundaryWrite != boundaryLaunchMem->commands.end()
            && (boundaryWrite->words[2] & 0xffffffU) + 1 <= 96,
        "late DDR launch did not arm its MEM_WRITE_SYNC before the consumer");

    BinaryProgram fragmentedIdleProgram;
    IcuLoop3D fragmentedLoop {0, {1, 1, 1}, {1, 1, 1}};
    fragmentedLoop.wait_cycle = 32;
    auto fragmentedRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(fragmentedLoop,
            MemIcuAddress3D::Affine(next_row, {1, 0, 0}), StreamId::East(0))));
    fragmentedRead.insert(fragmentedRead.begin(), encode_icu_control_raw_word(
        IcuControlInstruction::Nop(40)));
    fragmentedRead.insert(fragmentedRead.begin(), encode_icu_control_raw_word(
        IcuControlInstruction::Nop(28)));
    fragmentedIdleProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue, std::move(fragmentedRead)});
    auto fragmentedIdleSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager fragmentedIdlePager(*fragmentedIdleSystem);
    fragmentedIdlePager.begin_schedule(fragmentedIdleProgram);
    static_cast<void>(fragmentedIdlePager.schedule(fragmentedIdleProgram,
        logicalPage, 0, 10, 100, 0x3200, 0x3201));
    fragmentedIdlePager.finalize_schedule(fragmentedIdleProgram);
    const auto fragmentedMem = std::ranges::find_if(fragmentedIdleProgram.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    require(fragmentedMem != fragmentedIdleProgram.queues.end()
            && fragmentedMem->commands.size() == 7
            && decode_icu_control_raw_word(fragmentedMem->commands[2]).count == 89
            && decode_fu_3d_raw_packet_loop(*fragmentedMem, 4).wait_cycle == 0,
        "fragmented NOP/wait idle span introduced a false early page barrier");

    BinaryProgram groupedPagesProgram;
    auto groupedRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(next_row, {1, 0, 0}), StreamId::East(0))));
    groupedRead.insert(groupedRead.begin(), encode_icu_control_raw_word(
        IcuControlInstruction::Nop(100)));
    groupedPagesProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue, std::move(groupedRead)});
    auto groupedPagesSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager groupedPagesPager(*groupedPagesSystem);
    groupedPagesPager.begin_schedule(groupedPagesProgram);
    for (std::size_t page = 0; page < 3; ++page) {
        auto disjoint = logicalPage;
        disjoint.segments[0].slice += static_cast<std::uint16_t>(page);
        static_cast<void>(groupedPagesPager.schedule(groupedPagesProgram,
            disjoint, page * 10, page * 10 + 10, 100,
            0x3400 + page, 0x3500 + page));
    }
    groupedPagesPager.finalize_schedule(groupedPagesProgram);
    const auto groupedMem = std::ranges::find_if(groupedPagesProgram.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    require(groupedMem != groupedPagesProgram.queues.end()
            && groupedMem->commands.size() == 9
            && decode_icu_control_raw_word(groupedMem->commands[2]).count == 87
            && decode_fu_3d_raw_packet_loop(*groupedMem, 6).wait_cycle == 0,
        "three page waits at one boundary moved the following MEM wavefront");

    BinaryProgram mergedBoundaryProgram;
    auto mergedRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(next_row, {1, 0, 0}), StreamId::East(0))));
    mergedRead.insert(mergedRead.begin(), encode_icu_control_raw_word(
        IcuControlInstruction::Nop(100)));
    mergedBoundaryProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue, std::move(mergedRead)});
    auto mergedSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager mergedPager(*mergedSystem);
    mergedPager.begin_schedule(mergedBoundaryProgram);
    for (std::size_t page = 0; page < 3; ++page) {
        auto disjoint = logicalPage;
        disjoint.segments[0].slice += static_cast<std::uint16_t>(page);
        static_cast<void>(mergedPager.schedule(mergedBoundaryProgram,
            disjoint, page * 10, page * 10 + 10, 100, 0x3600 + page, 0x3700 + page));
    }
    const auto aliases = mergedPager.coalesce_page_ready_events();
    require(aliases.size() == 3 && std::ranges::all_of(aliases,
        [](const auto& alias) { return alias.second == 0x3700; }),
        "pages sharing one queue collision did not share one release event");
    mergedPager.finalize_schedule(mergedBoundaryProgram);
    const auto mergedMem = std::ranges::find_if(mergedBoundaryProgram.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem && queue.index == lowered.segments[0].mem_queue;
        });
    require(mergedMem != mergedBoundaryProgram.queues.end(), "merged page cut lost its MEM queue");
    std::size_t mergedWaits = 0;
    for (const auto& command : mergedMem->commands)
        if (is_icu_control_raw_word_command(command)) {
            const auto control = decode_icu_control_raw_word(command);
            if (control.opcode == IcuControlOpcode::WaitEvent) {
                ++mergedWaits;
                require(control.event_tag == 0x3700, "merged cut retained a stale page tag");
            }
        }
    require(mergedWaits == 1 && mergedPager.page_ready_releases(0x3701).empty(),
        "merged cut replayed multiple page rendezvous");

    BinaryProgram joinedEpochProgram;
    auto joinedRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(next_row, {1, 0, 0}), StreamId::East(0))));
    joinedRead.insert(joinedRead.begin(), encode_icu_control_raw_word(
        IcuControlInstruction::Nop(100)));
    joinedEpochProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue, std::move(joinedRead)});
    auto joinedEpochSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager joinedEpochPager(*joinedEpochSystem);
    joinedEpochPager.begin_schedule(joinedEpochProgram);
    static_cast<void>(joinedEpochPager.schedule(joinedEpochProgram,
        timing_page(), 0, 10, 20, 0x3300, 0x3301));
    static_cast<void>(joinedEpochPager.schedule(joinedEpochProgram,
        logicalPage, 20, 140, 100, 0x3302, 0x3303));
    joinedEpochPager.finalize_schedule(joinedEpochProgram);
    const auto joinedMem = std::ranges::find_if(joinedEpochProgram.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    require(joinedMem != joinedEpochProgram.queues.end()
            && joinedMem->commands.size() == 6
            && is_mem_synchronized_raw_packet_header(joinedMem->commands[0])
            && (joinedMem->commands[0].words[2] & 0xffffffU) + 1 == 98
            && decode_icu_control_raw_word(joinedMem->commands[2]).event_tag == 0x3303
            && decode_fu_3d_raw_packet_loop(*joinedMem, 3).wait_cycle == 0,
        "prefetch-only MEM queue retained an unrelated epoch rendezvous");

    BinaryProgram earlyJoinProgram;
    auto earlyRead = raw_mem_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D {0, {1, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(next_row, {1, 0, 0}), StreamId::East(0))));
    earlyRead.insert(earlyRead.begin(), encode_icu_control_raw_word(
        IcuControlInstruction::Nop(300)));
    earlyJoinProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue, std::move(earlyRead)});
    auto earlyJoinSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager earlyJoinPager(*earlyJoinSystem);
    earlyJoinPager.begin_schedule(earlyJoinProgram);
    static_cast<void>(earlyJoinPager.schedule(earlyJoinProgram,
        timing_page(), 0, 10, 80, 0x3400, 0x3401));
    // The unrelated epoch arrives before this queue's WRITE_SYNC window,
    // not inside it. There is still no consumer until this page's own join.
    static_cast<void>(earlyJoinPager.schedule(earlyJoinProgram,
        logicalPage, 120, 180, 300, 0x3402, 0x3403, 120));
    earlyJoinPager.finalize_schedule(earlyJoinProgram);
    const auto earlyMem = std::ranges::find_if(earlyJoinProgram.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    require(earlyMem != earlyJoinProgram.queues.end(),
        "early prefetch join lost its MEM queue");
    const auto findWait = [&](std::size_t tag) {
        return std::ranges::find_if(earlyMem->commands,
            [&](const QueueCommand& command) {
                if (!is_icu_control_raw_word_command(command)) return false;
                const auto control = decode_icu_control_raw_word(command);
                return control.opcode == IcuControlOpcode::WaitEvent
                    && control.event_tag == tag;
            });
    };
    require(findWait(0x3401) == earlyMem->commands.end(),
        "prefetch-only MEM queue kept an epoch preceding its write window");
    const auto ownWait = findWait(0x3403);
    require(ownWait != earlyMem->commands.end()
            && ownWait + 1 != earlyMem->commands.end()
            && is_fu_3d_raw_packet_header(*(ownWait + 1))
            && decode_fu_3d_raw_packet_loop(*earlyMem,
                static_cast<std::size_t>(ownWait + 1
                    - earlyMem->commands.begin())).wait_cycle == 0,
        "late prefetch replayed static idle time before its consumer rendezvous");

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
    const auto boundedShortWindowFence =
        boundedShortWindowPager.schedule(
            boundedShortWindowProgram, multiQueuePage,
            linkedStart, linkedTransferEnd, linkedC2cTail - 1, 0x3456);
    require(boundedShortWindowFence.scheduled_end_cycle
                == linkedTransferEnd
            && boundedShortWindowProgram.max_cycle == linkedC2cTail,
        "page-ready synchronization did not retain a late C2C queue tail");

    BinaryProgram capacityProgram;
    capacityProgram.hardware.icu_mem_imem_depth = 2;
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

    constexpr std::size_t oversizedStart = std::size_t {1} << 24;
    BinaryProgram oversizedGapProgram;
    oversizedGapProgram.queues.push_back(QueueProgram {QueueKind::Mem,
        lowered.segments[0].mem_queue,
        {encode_icu_control_raw_word(
            IcuControlInstruction::Nop(oversizedStart + 128))}});
    auto oversizedGapSystem = std::make_unique<C2cDmaSystem>();
    C2cWeightPager oversizedGapPager(*oversizedGapSystem);
    oversizedGapPager.begin_schedule(oversizedGapProgram);
    static_cast<void>(oversizedGapPager.schedule(oversizedGapProgram,
        logicalPage, oversizedStart, oversizedStart + 40,
        oversizedStart + 96, 0x5678));
    oversizedGapPager.finalize_schedule(oversizedGapProgram);
    const auto oversizedMem = std::find_if(
        oversizedGapProgram.queues.begin(),
        oversizedGapProgram.queues.end(),
        [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem
                && queue.index == lowered.segments[0].mem_queue;
        });
    require(oversizedMem != oversizedGapProgram.queues.end()
            && oversizedMem->commands.size() == 4
            && decode_icu_control_raw_word(
                oversizedMem->commands[0]).count == oversizedStart
            && is_mem_synchronized_raw_packet_header(
                oversizedMem->commands[1]),
        "oversized leading wait did not retain its NOP fallback");

    auto system = std::make_unique<C2cDmaSystem>(
        Ddr4Config {hw::kPhysicalVectorBytes, 2, 2, 256, 8});
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
    const auto& captured = pager.last_enqueued_program();
    const auto capturedDmaQueue = std::find_if(captured.queues.begin(),
        captured.queues.end(), [&](const QueueProgram& queue) {
            return queue.kind == QueueKind::C2cDma
                && queue.index == hemisphere_index(hemisphere);
        });
    require(capturedDmaQueue != captured.queues.end()
            && !capturedDmaQueue->commands.empty()
            && is_icu_control_raw_word_command(
                capturedDmaQueue->commands.back())
            && decode_icu_control_raw_word(
                capturedDmaQueue->commands.back()).opcode
                == IcuControlOpcode::Sync,
        "standalone C2C DMA program must end with lowered SYNC");
    std::ostringstream capturedBytes(std::ios::binary);
    write_binary_program(captured, capturedBytes);
    std::istringstream capturedInput(capturedBytes.str(), std::ios::binary);
    require(!read_binary_program(capturedInput).queues.empty(),
        "standalone C2C ICU program must serialize with a valid target ABI");

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

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--mxm-output-pair") {
            test_page_wait_does_not_split_mxm_output_tile_pair();
            test_page_wait_does_not_split_mxm_output_tile_pair(true);
            test_page_wait_preserves_partial_result_offset();
            test_closed_prefill_page_preserves_issue_phase();
        } else
            run_test();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
