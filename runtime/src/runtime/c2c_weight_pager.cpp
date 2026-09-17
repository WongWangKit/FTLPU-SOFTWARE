#include "ftlpu/software/runtime/c2c_weight_pager.hpp"

#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ftlpu::software::runtime {
namespace {

std::vector<std::size_t> allocate_c2c_lanes(
    const C2cWeightPage& page, std::size_t laneCount)
{
    using TargetLanes = std::array<std::array<std::array<
        std::optional<std::size_t>, hw::kMemBanksPerSlice>,
        hw::kMemSliceColumns>, hw::kHemispheres>;
    TargetLanes targetLanes{};
    std::array<std::array<std::size_t,
        hw::kC2cStreamsPerDirection>, hw::kHemispheres> laneLoads{};
    std::vector<std::size_t> assignments;
    assignments.reserve(page.segments.size());

    for (const auto& segment : page.segments) {
        const auto side = hemisphere_index(segment.hemisphere);
        auto& assigned =
            targetLanes[side][segment.slice][segment.bank];
        if (!assigned.has_value()) {
            const auto begin = laneLoads[side].begin();
            assigned = static_cast<std::size_t>(std::distance(
                begin, std::min_element(begin, begin + laneCount)));
        }
        assignments.push_back(*assigned);
        laneLoads[side][*assigned] += segment.vector_count;
    }
    return assignments;
}

std::size_t resolve_fabric_stream_base(
    const C2cWeightPage& page, std::size_t laneCount)
{
    if (laneCount == 0 || laneCount > hw::kWestStreams)
        throw std::invalid_argument(
            "invalid C2C lane count for the ordinary SR fabric");
    const std::size_t base = page.fabric_stream_base.value_or(
        static_cast<std::uint16_t>(hw::kWestStreams - laneCount));
    if (base + laneCount > hw::kWestStreams)
        throw std::out_of_range(
            "C2C fabric stream range is outside the ordinary west SR file");
    return base;
}

void validate_weight_page(
    const C2cWeightPage& page,
    const SystemHardwareConfiguration& hardware)
{
    if (page.bank >= hw::kMemBanksPerSlice || page.segments.empty())
        throw std::invalid_argument("invalid executable C2C weight page");
    if (hardware.c2c_streams_per_direction == 0
        || hardware.c2c_streams_per_direction
            > hw::kC2cStreamsPerDirection)
        throw std::invalid_argument(
            "invalid C2C lane count for weight-page lowering");
    for (const auto& segment : page.segments) {
        if (segment.bank != page.bank
            || segment.bank >= hw::kMemBanksPerSlice
            || segment.slice >= hw::kMemSliceColumns
            || segment.vector_count == 0)
            throw std::invalid_argument(
                "invalid executable C2C weight-page segment");
        if (segment.base_row >= hardware.sram_depth_rows
            || static_cast<std::uint64_t>(segment.vector_count - 1)
                > static_cast<std::uint64_t>(hardware.sram_depth_rows - 1)
                    - segment.base_row)
            throw std::out_of_range(
                "C2C weight segment exceeds its SRAM bank: base="
                + std::to_string(segment.base_row)
                + " vectors=" + std::to_string(segment.vector_count)
                + " depth="
                + std::to_string(hardware.sram_depth_rows));
    }
}

void load_c2c_weight_page_icu_program(
    InstructionControlUnit& icu,
    const C2cWeightPageIcuProgram& program)
{
    // Program every consumer before making a producer visible. This mirrors
    // the hardware boot sequence and prevents an already-running DMA queue
    // from delivering a vector before RX/MEM own matching packet contexts.
    for (const auto& segment : program.segments) {
        icu.push_c2c_rx_raw(segment.hemisphere, segment.rx);
        icu.mem_iq(segment.mem_queue)
            .push_encoded_synchronized_packet(segment.mem_write_sync);
    }
    for (const auto& segment : program.segments)
        icu.push_c2c_dma_raw(segment.hemisphere, segment.dma);
}

QueueProgram* find_queue(BinaryProgram& program, QueueKind kind,
    std::size_t index)
{
    const auto found = std::ranges::find_if(program.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == kind && queue.index == index;
        });
    return found == program.queues.end() ? nullptr : &*found;
}

const QueueProgram* find_queue(const BinaryProgram& program,
    QueueKind kind, std::size_t index)
{
    const auto found = std::ranges::find_if(program.queues,
        [&](const QueueProgram& queue) {
            return queue.kind == kind && queue.index == index;
        });
    return found == program.queues.end() ? nullptr : &*found;
}

QueueProgram& find_or_create_queue(BinaryProgram& program,
    QueueKind kind, std::size_t index)
{
    if (auto* queue = find_queue(program, kind, index)) return *queue;
    program.queues.push_back(QueueProgram {kind, index, {}});
    return program.queues.back();
}

std::optional<std::size_t> nop_cycles(const QueueCommand& command)
{
    if (isa::decode_icu_command_opcode(command.command)
        != isa::IcuCommandOpcode::Nop)
        return std::nullopt;
    if (is_icu_control_raw_word_command(command)) {
        const auto control = decode_icu_control_raw_word(command);
        if (control.opcode != IcuControlOpcode::Nop)
            throw std::logic_error("raw ICU NOP decoded as another opcode");
        return control.count;
    }
    if (command.instruction_kind != InstructionKind::None
        || command.word_count != 0 || !command.extension_words.empty())
        throw std::logic_error("malformed MEM ICU NOP in BinaryProgram");
    return isa::decode_icu_nop_cycles(command.command);
}

std::size_t loop_cycles(const IcuLoop3D& loop)
{
    std::size_t final = loop.start_cycle + loop.wait_cycle;
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension) {
        const auto steps = loop.counts[dimension] - 1;
        if (steps != 0
            && loop.cycle_strides[dimension]
                > (std::numeric_limits<std::size_t>::max() - final)
                    / steps)
            throw std::overflow_error("MEM 3-D timeline overflows size_t");
        final += steps * loop.cycle_strides[dimension];
    }
    if (final == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("MEM 3-D duration overflows size_t");
    return final + 1;
}

void set_raw_3d_wait(QueueCommand& header, std::size_t wait)
{
    if (wait >= (std::size_t {1} << 24)
        || !is_fu_3d_raw_packet_header(header))
        throw std::logic_error("MEM 3-D wait_cycle is not encodable");
    header.words[0] = (header.words[0] & 0xffU)
        | (static_cast<std::uint32_t>(wait) << 8);
    header.command = header.words[0];
}

void append_nop(std::vector<QueueCommand>& commands, std::size_t cycles)
{
    if (cycles == 0) return;
    commands.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(cycles)));
}

void append_control(std::vector<QueueCommand>& commands,
    const IcuControlInstruction& instruction)
{
    commands.push_back(encode_icu_control_raw_word(instruction));
}

void append_c2c_endpoint(std::vector<QueueCommand>& commands,
    const C2cEndpointIcuPacket& packet)
{
    commands.push_back(encode_c2c_endpoint_icu_packet(packet));
}

void append_c2c_dma(std::vector<QueueCommand>& commands,
    const C2cDmaIcuPacket& packet)
{
    const auto encoded = encode_c2c_dma_icu_packet(packet);
    commands.insert(commands.end(), encoded.begin(), encoded.end());
}

std::size_t c2c_dma_frontend_wait_cycles(std::size_t packetCount)
{
    constexpr auto iqDepth = InstructionControlUnit::C2cDmaIcu::iq_depth;
    static_assert(iqDepth >= C2cDmaIcuPacket::kWordCount);
    constexpr auto packetsBeforeFirstDecodeWait = iqDepth - 1;
    return packetCount > packetsBeforeFirstDecodeWait
        ? packetCount - packetsBeforeFirstDecodeWait
        : 0;
}

std::size_t c2c_queue_end(std::size_t startCycle,
    std::size_t packetCount, std::size_t frontendWaitCycles)
{
    constexpr auto maxCycle = std::numeric_limits<std::size_t>::max();
    if (startCycle == maxCycle || packetCount > maxCycle - startCycle - 1)
        throw std::overflow_error(
            "resolved C2C ICU queue timeline overflows size_t");
    const auto packetEnd = startCycle + 1 + packetCount;
    if (frontendWaitCycles > maxCycle - packetEnd)
        throw std::overflow_error(
            "resolved C2C ICU queue timeline overflows size_t");
    return packetEnd + frontendWaitCycles;
}

} // namespace

C2cWeightPageIcuProgram lower_c2c_weight_page_to_icu(
    const C2cWeightPage& page,
    const SystemHardwareConfiguration& hardware,
    std::uint32_t firstSyncTag,
    std::size_t reservationCycles)
{
    validate_weight_page(page, hardware);
    const bool useMinimumReservation = reservationCycles == 0;
    const auto laneCount = static_cast<std::size_t>(
        hardware.c2c_streams_per_direction);
    const auto sharedStreamBase = resolve_fabric_stream_base(page, laneCount);
    const auto laneAssignments = allocate_c2c_lanes(page, laneCount);
    constexpr auto maxSyncTag =
        static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max());
    if (firstSyncTag == 0 || firstSyncTag > maxSyncTag
        || page.segments.size() - 1 > maxSyncTag - firstSyncTag)
        throw std::overflow_error(
            "C2C weight-page sync tags exceed the 16-bit ICU field");

    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        vectorsPerQueue{};
    for (const auto& segment : page.segments) {
        const auto queue = InstructionControlUnit::mem_queue(
            segment.hemisphere, segment.slice, segment.bank);
        if (segment.vector_count
            > std::numeric_limits<std::size_t>::max()
                - vectorsPerQueue[queue])
            throw std::overflow_error(
                "C2C weight-page MEM vector count overflows size_t");
        vectorsPerQueue[queue] += segment.vector_count;
    }
    for (const auto count : vectorsPerQueue)
        if (!useMinimumReservation && count > reservationCycles)
            throw std::invalid_argument(
                "C2C weight-page reservation is shorter than its MEM write count");
    auto remainingVectors = vectorsPerQueue;

    C2cWeightPageIcuProgram program;
    program.segments.reserve(page.segments.size());
    for (std::size_t segmentIndex = 0;
         segmentIndex < page.segments.size(); ++segmentIndex) {
        const auto& segment = page.segments[segmentIndex];
        const auto lane = laneAssignments[segmentIndex];
        const auto fabricStream = sharedStreamBase + lane;
        const auto queue = InstructionControlUnit::mem_queue(
            segment.hemisphere, segment.slice, segment.bank);
        const auto group = segment.slice / hw::kMemSlicesPerGroup;
        const auto transportNops =
            hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
        const auto syncTag = firstSyncTag
            + static_cast<std::uint32_t>(segmentIndex);
        // Every packet needs at least one cycle per emitted FU write. The
        // final packet holds any page-level latency/guard slack so the whole
        // physical queue remains reserved through the linked window end.
        auto packetReservation =
            static_cast<std::size_t>(segment.vector_count);
        remainingVectors[queue] -= segment.vector_count;
        const auto queueReservation = useMinimumReservation
            ? vectorsPerQueue[queue] : reservationCycles;
        if (remainingVectors[queue] == 0)
            packetReservation += queueReservation - vectorsPerQueue[queue];

        const auto rx = C2cIcuPacketCodec::encode(
            C2cRxIcuInstruction::Receive(segment.hemisphere, lane,
                fabricStream, segment.vector_count, syncTag,
                C2cMemNotifyRoute::Mem(segment.hemisphere,
                    segment.slice, segment.bank)));
        const auto memWrite = InstructionControlUnit::MemIcu::
            encode_synchronized_raw_packet(segment.vector_count, syncTag,
                transportNops, 1,
                MemInstruction::Write(segment.base_row,
                    StreamId::West(fabricStream)),
                packetReservation);
        const auto dma = C2cIcuPacketCodec::encode(
            C2cDmaIcuInstruction::Load(segment.hemisphere, lane,
                segment.ddr4_address, segment.vector_count,
                hw::kPhysicalVectorBytes, syncTag));
        program.segments.push_back(C2cWeightSegmentIcuProgram {
            segment.hemisphere,
            static_cast<std::uint16_t>(lane),
            static_cast<std::uint16_t>(fabricStream),
            queue,
            syncTag,
            rx,
            memWrite,
            dma,
            segment.vector_count,
        });
    }
    program.next_sync_tag = firstSyncTag
        + static_cast<std::uint32_t>(program.segments.size());
    return program;
}

C2cWeightPager::C2cWeightPager(C2cDmaSystem& system)
    : system_(system)
{
}

void C2cWeightPager::begin_schedule(const BinaryProgram& program)
{
    if (active_ && !ready())
        throw std::logic_error(
            "cannot schedule executable pages while a page is in flight");
    if (active_) retire();
    schedule_dma_cursor_ = {};
    schedule_rx_cursor_ = {};
    scheduled_dma_issues_ = {};
    scheduled_rx_segments_ = {};
    scheduled_mem_writes_ = {};
    scheduled_mem_reservations_ = {};
    next_sync_tag_ = 1;
    schedule_open_ = true;
    for (auto& windows : mem_idle_windows_) windows.clear();
    for (auto& windows : linked_mem_windows_) windows.clear();

    std::array<bool, InstructionControlUnit::kMemQueues> seenMem{};
    for (const auto& queue : program.queues) {
        if ((queue.kind == QueueKind::C2cDma
                || queue.kind == QueueKind::C2cRx)
            && !queue.commands.empty())
            throw std::logic_error(
                "runtime C2C page linker requires ownership of the C2C RX/DMA queues");
        if (queue.kind != QueueKind::Mem) continue;
        if (queue.index >= InstructionControlUnit::kMemQueues)
            throw std::out_of_range(
                "BinaryProgram MEM queue is outside the hardware topology");
        if (seenMem[queue.index])
            throw std::logic_error(
                "BinaryProgram contains duplicate physical MEM queues");
        seenMem[queue.index] = true;

        std::size_t cursor = 0;
        for (std::size_t commandIndex = 0;
             commandIndex < queue.commands.size();) {
            const auto& command = queue.commands[commandIndex];
            if (const auto cycles = nop_cycles(command)) {
                if (*cycles > std::numeric_limits<std::size_t>::max()
                        - cursor)
                    throw std::overflow_error(
                        "MEM ICU NOP timeline overflows size_t");
                if (*cycles != 0)
                    mem_idle_windows_[queue.index].push_back(
                        MemIdleWindow {cursor, cursor + *cycles});
                cursor += *cycles;
                ++commandIndex;
                continue;
            }
            if (is_mem_write_read_2d_raw_packet_header(command)
                || is_fu_3d_raw_packet_header(command)) {
                const auto words = fu_3d_raw_packet_word_count(queue.kind);
                const auto wait =
                    is_mem_write_read_2d_raw_packet_header(command)
                    ? std::size_t {0}
                    : decode_fu_3d_raw_packet_loop(
                          queue, commandIndex).wait_cycle;
                if (wait != 0)
                    mem_idle_windows_[queue.index].push_back(
                        MemIdleWindow {cursor, cursor + wait});
                const auto duration =
                    is_mem_write_read_2d_raw_packet_header(command)
                    ? detail::mem_icu_write_read_2d_last_issue_cycle(
                          decode_mem_write_read_2d_raw_packet(
                              queue, commandIndex)) + 1
                    : loop_cycles(
                          decode_fu_3d_raw_packet_loop(
                              queue, commandIndex));
                if (duration > std::numeric_limits<std::size_t>::max()
                        - cursor)
                    throw std::overflow_error(
                        "MEM ICU 3-D timeline overflows size_t");
                cursor += duration;
                commandIndex += words;
                continue;
            }
            if (is_fu_3d_raw_word_command(command))
                throw std::logic_error(
                    "orphan MEM 3-D packet word in BinaryProgram");
            throw std::logic_error(
                "runtime C2C page linker requires direct MEM 3-D/NOP queues");
        }
        mem_idle_windows_[queue.index].push_back(MemIdleWindow {
            cursor, std::numeric_limits<std::size_t>::max()});
    }
    for (std::size_t queue = 0;
         queue < InstructionControlUnit::kMemQueues; ++queue)
        if (!seenMem[queue])
            mem_idle_windows_[queue].push_back(MemIdleWindow {
                0, std::numeric_limits<std::size_t>::max()});
}

C2cWeightPageFence C2cWeightPager::schedule(
    BinaryProgram& binary, const C2cWeightPage& page,
    std::size_t startCycle, std::size_t transferEndCycle,
    std::size_t readyCycle, std::size_t launchEventTag)
{
    if (!schedule_open_)
        throw std::logic_error(
            "begin_schedule must precede executable C2C page linking");
    if (transferEndCycle <= startCycle)
        throw std::invalid_argument(
            "executable C2C page has an empty transfer window");
    const auto transportDuration = transferEndCycle - startCycle;
    auto& chip = system_.chip();
    const auto& hardware = chip.hardware_configuration();

    std::vector<std::size_t> targetQueues;
    targetQueues.reserve(page.segments.size());
    for (const auto& segment : page.segments) {
        const auto queue = InstructionControlUnit::mem_queue(
            segment.hemisphere, segment.slice, segment.bank);
        if (std::ranges::find(targetQueues, queue)
            == targetQueues.end())
            targetQueues.push_back(queue);
    }

    const bool hasFiniteReadyCycle =
        readyCycle != std::numeric_limits<std::size_t>::max();
    const auto reservationEndAt = [&](std::size_t candidate) {
        if (hasFiniteReadyCycle) return readyCycle;
        if (transportDuration > std::numeric_limits<std::size_t>::max()
                - candidate)
            throw std::overflow_error(
                "resolved C2C weight-page window overflows size_t");
        return candidate + transportDuration;
    };

    std::size_t resolvedStart = std::max(
        startCycle, earliest_schedule_cycle(page));
    for (;;) {
        std::size_t commonStart = resolvedStart;
        for (const auto queue : targetQueues) {
            const auto& windows = mem_idle_windows_[queue];
            bool found = false;
            for (const auto& window : windows) {
                const auto candidate = std::max(commonStart, window.begin);
                const auto reservationEnd = reservationEndAt(candidate);
                if (candidate > reservationEnd
                    || candidate > window.end
                    || reservationEnd > window.end)
                    continue;
                commonStart = std::max(commonStart, candidate);
                found = true;
                break;
            }
            if (!found)
                throw std::logic_error(
                    "no legal MEM ICU idle window can hold a C2C weight page");
        }
        bool allFit = true;
        for (const auto queue : targetQueues) {
            const auto& windows = mem_idle_windows_[queue];
            const bool fits = std::ranges::any_of(windows,
                [&](const MemIdleWindow& window) {
                    const auto reservationEnd = reservationEndAt(commonStart);
                    return commonStart >= window.begin
                        && commonStart <= window.end
                        && commonStart <= reservationEnd
                        && reservationEnd <= window.end;
                });
            if (!fits) {
                allFit = false;
                break;
            }
        }
        resolvedStart = commonStart;
        if (allFit) break;
    }
    if (transportDuration > std::numeric_limits<std::size_t>::max()
            - resolvedStart)
        throw std::overflow_error(
            "resolved C2C weight-page window overflows size_t");
    const auto transportEnd = resolvedStart + transportDuration;
    auto segmentCounts = std::array<std::size_t, hw::kHemispheres> {};
    auto dmaLogicalEnds = std::array<std::size_t, hw::kHemispheres> {};
    auto rxLogicalEnds = std::array<std::size_t, hw::kHemispheres> {};
    auto c2cLogicalEnd = resolvedStart;
    for (const auto& segment : page.segments)
        ++segmentCounts[hemisphere_index(segment.hemisphere)];
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto segmentCount = segmentCounts[side];
        if (segmentCount == 0) continue;
        // WAIT_EVENT consumes one logical queue cycle. RX then sustains one
        // one-word packet per cycle. DMA packets occupy two physical words;
        // after the initially primed IQ burst, its one-word/cycle frontend
        // inserts one decode wait before every remaining packet.
        rxLogicalEnds[side] = c2c_queue_end(
            resolvedStart, segmentCount, 0);
        dmaLogicalEnds[side] = c2c_queue_end(resolvedStart, segmentCount,
            c2c_dma_frontend_wait_cycles(segmentCount));
        c2cLogicalEnd = std::max({c2cLogicalEnd,
            rxLogicalEnds[side], dmaLogicalEnds[side]});
    }
    if (hasFiniteReadyCycle
        && std::max(transportEnd, c2cLogicalEnd) > readyCycle)
        throw std::logic_error(
            "C2C/MEM ICU linking moved a weight page past its first consumer");

    // Transport end is an estimate derived from the external-memory model.
    // Keep the single MEM ICU owned through the first consumer so a late DDR
    // response cannot shift the ordinary static MEM program. Page-ready
    // synchronization absorbs any completion later than this boundary.
    const auto reservationEnd = hasFiniteReadyCycle
        ? readyCycle : transportEnd;
    if (reservationEnd <= resolvedStart)
        throw std::logic_error(
            "C2C/MEM ICU reservation reaches an empty consumer window");
    const auto reservationDuration = reservationEnd - resolvedStart;
    const auto lowered = lower_c2c_weight_page_to_icu(
        page, hardware, next_sync_tag_, reservationDuration);

    for (const auto queue : targetQueues) {
        auto remaining = std::vector<MemIdleWindow> {};
        for (const auto& window : mem_idle_windows_[queue]) {
            if (reservationEnd <= window.begin
                || resolvedStart >= window.end) {
                remaining.push_back(window);
                continue;
            }
            if (window.begin < resolvedStart)
                remaining.push_back(
                    MemIdleWindow {window.begin, resolvedStart});
            if (reservationEnd < window.end)
                remaining.push_back(
                    MemIdleWindow {reservationEnd, window.end});
        }
        mem_idle_windows_[queue] = std::move(remaining);
        LinkedMemWindow window;
        window.begin = resolvedStart;
        window.end = reservationEnd;
        for (const auto& segment : lowered.segments)
            if (segment.mem_queue == queue)
                window.packets.push_back(segment.mem_write_sync);
        linked_mem_windows_[queue].push_back(std::move(window));
    }

    C2cWeightPageFence fence;
    fence.dma_issues_begin = scheduled_dma_issues_;
    fence.dma_issues_end = scheduled_dma_issues_;
    fence.completed_segments = scheduled_rx_segments_;
    fence.scheduled_start_cycle = resolvedStart;
    fence.scheduled_end_cycle = transportEnd;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        const auto segmentCount = segmentCounts[side];
        if (segmentCount == 0) continue;
        if (resolvedStart < schedule_dma_cursor_[side]
            || resolvedStart < schedule_rx_cursor_[side])
            throw std::logic_error(
                "executable C2C page schedule overlaps an ICU queue");

        static_cast<void>(find_or_create_queue(
            binary, QueueKind::C2cDma, side));
        static_cast<void>(find_or_create_queue(
            binary, QueueKind::C2cRx, side));
        auto& dmaQueue = *find_queue(
            binary, QueueKind::C2cDma, side);
        auto& rxQueue = *find_queue(
            binary, QueueKind::C2cRx, side);
        append_nop(dmaQueue.commands,
            resolvedStart - schedule_dma_cursor_[side]);
        append_nop(rxQueue.commands,
            resolvedStart - schedule_rx_cursor_[side]);
        append_control(dmaQueue.commands,
            IcuControlInstruction::WaitEvent(launchEventTag));
        append_control(rxQueue.commands,
            IcuControlInstruction::WaitEvent(launchEventTag));
        for (std::size_t segmentIndex = 0;
             segmentIndex < page.segments.size(); ++segmentIndex) {
            const auto& segment = page.segments[segmentIndex];
            if (hemisphere_index(segment.hemisphere) != side) continue;
            const auto& item = lowered.segments[segmentIndex];
            append_c2c_endpoint(rxQueue.commands, item.rx);
            append_c2c_dma(dmaQueue.commands, item.dma);
            ++scheduled_dma_issues_[side];
            fence.dma_issues_end[side] = scheduled_dma_issues_[side];
            ++scheduled_rx_segments_[side][item.lane];
            fence.completed_segments[side][item.lane] =
                scheduled_rx_segments_[side][item.lane];
            scheduled_mem_writes_[item.mem_queue] += item.vector_count;
            fence.completed_mem_writes[item.mem_queue] =
                scheduled_mem_writes_[item.mem_queue];
            ++scheduled_mem_reservations_[item.mem_queue];
            fence.completed_mem_reservations[item.mem_queue] =
                scheduled_mem_reservations_[item.mem_queue];
        }
        schedule_dma_cursor_[side] = dmaLogicalEnds[side];
        schedule_rx_cursor_[side] = rxLogicalEnds[side];
    }
    next_sync_tag_ = lowered.next_sync_tag;
    binary.max_cycle = std::max(
        binary.max_cycle, std::max(transportEnd, c2cLogicalEnd));
    return fence;
}

void C2cWeightPager::finalize_schedule(BinaryProgram& program)
{
    if (!schedule_open_)
        throw std::logic_error(
            "begin_schedule must precede executable C2C page finalization");

    for (std::size_t queueIndex = 0;
         queueIndex < InstructionControlUnit::kMemQueues; ++queueIndex) {
        auto& windows = linked_mem_windows_[queueIndex];
        if (windows.empty()) continue;
        std::ranges::sort(windows,
            [](const LinkedMemWindow& lhs, const LinkedMemWindow& rhs) {
                return lhs.begin < rhs.begin;
            });
        auto& queue = find_or_create_queue(
            program, QueueKind::Mem, queueIndex);
        const auto original = queue.commands;
        std::vector<QueueCommand> linked;
        linked.reserve(original.size() + windows.size() * 3);
        std::vector<std::size_t> oldToNew(original.size(),
            std::numeric_limits<std::size_t>::max());
        std::size_t cursor = 0;
        std::size_t windowIndex = 0;

        auto emitWindows = [&](std::size_t idleEnd,
                               bool unbounded) {
            while (windowIndex < windows.size()
                && (unbounded || windows[windowIndex].begin < idleEnd)) {
                const auto& window = windows[windowIndex];
                if (window.begin < cursor
                    || (!unbounded && window.end > idleEnd))
                    throw std::logic_error(
                        "linked MEM_WRITE_SYNC escaped its reserved idle window");
                append_nop(linked, window.begin - cursor);
                for (const auto& packet : window.packets) {
                    const auto encoded =
                        encode_mem_synchronized_icu_packet(packet);
                    linked.insert(linked.end(),
                        encoded.begin(), encoded.end());
                }
                cursor = window.end;
                ++windowIndex;
            }
        };

        for (std::size_t commandIndex = 0;
             commandIndex < original.size();) {
            const auto& command = original[commandIndex];
            if (const auto cycles = nop_cycles(command)) {
                const auto idleEnd = cursor + *cycles;
                emitWindows(idleEnd, false);
                append_nop(linked, idleEnd - cursor);
                cursor = idleEnd;
                ++commandIndex;
                continue;
            }
            if (!is_fu_3d_raw_packet_header(command))
                throw std::logic_error(
                    "runtime C2C page linker encountered a non-direct MEM command during finalization");
            if (windowIndex < windows.size()
                && windows[windowIndex].begin < cursor)
                throw std::logic_error(
                    "linked MEM_WRITE_SYNC overlaps an ordinary MEM domain");
            const auto words = fu_3d_raw_packet_word_count(queue.kind);
            const auto loop =
                is_mem_write_read_2d_raw_packet_header(command)
                ? IcuLoop3D {}
                : decode_fu_3d_raw_packet_loop(queue, commandIndex);
            const auto wait = loop.wait_cycle;
            const auto idleEnd = cursor + wait;
            if (wait != 0)
                emitWindows(idleEnd, false);
            const auto remainingWait = idleEnd - cursor;
            const auto duration =
                is_mem_write_read_2d_raw_packet_header(command)
                ? detail::mem_icu_write_read_2d_last_issue_cycle(
                      decode_mem_write_read_2d_raw_packet(
                          queue, commandIndex)) + 1
                : loop_cycles(
                      decode_fu_3d_raw_packet_loop(
                          queue, commandIndex));
            for (std::size_t word = 0; word < words; ++word) {
                oldToNew[commandIndex + word] = linked.size();
                linked.push_back(original[commandIndex + word]);
            }
            if (wait != 0)
                set_raw_3d_wait(linked[oldToNew[commandIndex]],
                    remainingWait);
            cursor += duration - wait + remainingWait;
            commandIndex += words;
        }
        emitWindows(std::numeric_limits<std::size_t>::max(), true);
        const auto targetDepth = static_cast<std::size_t>(
            program.hardware.icu_mem_imem_depth);
        if (linked.size() > targetDepth)
            throw std::logic_error(
                "linked MEM ICU i-MEM capacity exceeded: queue="
                + std::to_string(queueIndex)
                + " used_slots=" + std::to_string(linked.size())
                + " depth=" + std::to_string(targetDepth));
        queue.commands = std::move(linked);

        const auto relocate = [&](auto& relocations) {
            for (auto& relocation : relocations) {
                if (relocation.queue_kind != QueueKind::Mem
                    || relocation.queue_index != queueIndex)
                    continue;
                if (relocation.command_index >= oldToNew.size()
                    || oldToNew[relocation.command_index]
                        == std::numeric_limits<std::size_t>::max())
                    throw std::logic_error(
                        "MEM relocation does not target a preserved command");
                relocation.command_index = static_cast<std::uint32_t>(
                    oldToNew[relocation.command_index]);
            }
        };
        relocate(program.address_relocations);
        relocate(program.scale_relocations);
    }
    std::ranges::sort(program.queues,
        [](const QueueProgram& lhs, const QueueProgram& rhs) {
            return std::tie(lhs.kind, lhs.index)
                < std::tie(rhs.kind, rhs.index);
        });
    schedule_open_ = false;
}

std::size_t C2cWeightPager::earliest_schedule_cycle(
    const C2cWeightPage& page) const
{
    std::size_t cycle = 0;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const bool usesSide = std::any_of(
            page.segments.begin(), page.segments.end(),
            [&](const C2cWeightSegment& segment) {
                return hemisphere_index(segment.hemisphere) == side;
            });
        if (usesSide)
            cycle = std::max(
                {cycle, schedule_dma_cursor_[side],
                 schedule_rx_cursor_[side]});
    }
    return cycle;
}

bool C2cWeightPager::started(const C2cWeightPageFence& fence) const
{
    auto& icu = system_.chip().icu();
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        if (fence.dma_issues_end[side] == fence.dma_issues_begin[side])
            continue;
        if (icu.c2c_dma_iq(static_cast<Hemisphere>(side)).issued_count()
            > fence.dma_issues_begin[side])
            return true;
    }
    return false;
}

bool C2cWeightPager::ready(const C2cWeightPageFence& fence) const
{
    auto& chip = system_.chip();
    for (std::size_t side = 0; side < hw::kHemispheres; ++side)
        for (std::size_t stream = 0;
             stream < hw::kC2cStreamsPerDirection; ++stream)
            if (chip.c2c_endpoint(static_cast<Hemisphere>(side))
                    .rx().completed_instruction_count(stream)
                < fence.completed_segments[side][stream]) {
                fence.mem_writes_issued_cycle.reset();
                return false;
            }
    for (std::size_t queue = 0;
         queue < fence.completed_mem_writes.size(); ++queue)
        if (chip.icu().mem_iq(queue).synchronized_issued_count()
            < fence.completed_mem_writes[queue]) {
            fence.mem_writes_issued_cycle.reset();
            return false;
        }

    if (!fence.mem_writes_issued_cycle)
        fence.mem_writes_issued_cycle = system_.cycle();
    const std::size_t elapsed =
        system_.cycle() - *fence.mem_writes_issued_cycle + 1;
    if (elapsed < hw::kTileRows) return false;

    for (std::size_t queue = 0;
         queue < fence.completed_mem_reservations.size(); ++queue)
        if (chip.icu().mem_iq(queue).synchronized_completed_count()
            < fence.completed_mem_reservations[queue])
            return false;
    return true;
}

void C2cWeightPager::enqueue(const C2cWeightPage& page)
{
    if (active_ && !ready())
        throw std::logic_error("a C2C weight page is already in flight");
    stats_ = {};
    stats_.layer = page.layer;
    stats_.bank = page.bank;
    stats_.enqueue_cycle = system_.cycle();
    target_mem_queues_.clear();
    active_mem_writes_target_ = {};
    active_mem_reservations_target_ = {};
    drain_cycles_ = 0;

    auto& chip = system_.chip();
    const auto& hardware = chip.hardware_configuration();
    const auto program = lower_c2c_weight_page_to_icu(page, hardware, 1);
    for (std::size_t queue = 0;
         queue < active_mem_writes_target_.size(); ++queue)
        active_mem_writes_target_[queue] =
            chip.icu().mem_iq(queue).synchronized_issued_count();
    for (std::size_t queue = 0;
         queue < active_mem_reservations_target_.size(); ++queue)
        active_mem_reservations_target_[queue] =
            chip.icu().mem_iq(queue).synchronized_completed_count();
    auto usedHemisphere = std::array<bool, hw::kHemispheres> {};
    for (std::size_t segmentIndex = 0;
         segmentIndex < page.segments.size(); ++segmentIndex) {
        const auto& segment = page.segments[segmentIndex];
        const auto side = hemisphere_index(segment.hemisphere);
        usedHemisphere[side] = true;
        const auto queue = program.segments[segmentIndex].mem_queue;
        if (std::find(target_mem_queues_.begin(),
                target_mem_queues_.end(), queue)
            == target_mem_queues_.end())
            target_mem_queues_.push_back(queue);
        active_mem_writes_target_[queue] += segment.vector_count;
        ++active_mem_reservations_target_[queue];
        stats_.vectors += segment.vector_count;
    }
    load_c2c_weight_page_icu_program(chip.icu(), program);
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
        if (chip.icu().mem_iq(queue).synchronized_issued_count()
                < active_mem_writes_target_[queue]
            || chip.icu().mem_iq(queue).synchronized_completed_count()
                < active_mem_reservations_target_[queue])
            return false;
    return drain_cycles_ >= hw::kTileRows;
}

void C2cWeightPager::retire()
{
    if (!active_ || !ready())
        throw std::logic_error(
            "cannot retire an incomplete C2C weight page");
    active_ = false;
    target_mem_queues_.clear();
    active_mem_writes_target_ = {};
    active_mem_reservations_target_ = {};
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
            && system_.chip().icu().mem_iq(queue)
                .synchronized_issued_count()
                >= active_mem_writes_target_[queue]
            && system_.chip().icu().mem_iq(queue)
                .synchronized_completed_count()
                >= active_mem_reservations_target_[queue];
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
