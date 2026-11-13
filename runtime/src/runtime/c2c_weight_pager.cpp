#include "ftlpu/software/runtime/common_transport_validation.hpp"
#include "ftlpu/software/runtime/common_overlap.hpp"
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
#include <unordered_map>

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

std::vector<std::size_t> resolve_fabric_streams(
    const C2cWeightPage& page, std::size_t hardwareLaneCount)
{
    if (hardwareLaneCount == 0 || hardwareLaneCount > hw::kWestStreams)
        throw std::invalid_argument(
            "invalid C2C lane count for the ordinary SR fabric");
    if (!page.fabric_streams.empty()) {
        if (page.fabric_streams.size() > hardwareLaneCount)
            throw std::invalid_argument(
                "C2C page selects more fabric streams than transport lanes");
        std::vector<std::size_t> streams(
            page.fabric_streams.begin(), page.fabric_streams.end());
        std::ranges::sort(streams);
        if (streams.back() >= hw::kWestStreams
            || std::adjacent_find(streams.begin(), streams.end())
                != streams.end())
            throw std::invalid_argument(
                "C2C page fabric streams must be unique West SR indices");
        return std::vector<std::size_t>(
            page.fabric_streams.begin(), page.fabric_streams.end());
    }
    const std::size_t laneCount = hardwareLaneCount;
    const std::size_t base = page.fabric_stream_base.value_or(
        static_cast<std::uint16_t>(hw::kWestStreams - laneCount));
    if (base + laneCount > hw::kWestStreams)
        throw std::out_of_range(
            "C2C fabric stream range is outside the ordinary west SR file");
    std::vector<std::size_t> streams(laneCount);
    for (std::size_t lane = 0; lane < laneCount; ++lane)
        streams[lane] = base + lane;
    return streams;
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
    for (std::size_t side = 0; side < hw::kHemispheres; ++side)
        if (program.dma_completion_sync[side])
            icu.enqueue_control(
                IcuLocation::C2cDma(static_cast<Hemisphere>(side)),
                IcuControlInstruction::Sync());
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

// Query a compact affine issue domain without expanding a decoder's complete
// execution trace. This analysis never changes the finite hardware frontend.
struct CommonIssueDomain {
    std::size_t start{0};
    std::array<std::size_t, 3> counts{1, 1, 1};
    std::array<std::size_t, 3> strides{1, 1, 1};
};

bool common_domain_intersects(const CommonIssueDomain& domain,
    std::size_t begin, std::size_t end)
{
    if (begin >= end) return false;
    std::array<std::pair<std::size_t, std::size_t>, 3> axes;
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        if (domain.counts[axis] == 0 || domain.strides[axis] == 0)
            throw std::logic_error("invalid affine issue domain at common stop");
        axes[axis] = {domain.strides[axis], domain.counts[axis]};
    }
    std::ranges::sort(axes);
    std::array<std::size_t, 4> spans{};
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        const auto [stride, count] = axes[axis];
        if (count - 1 > (std::numeric_limits<std::size_t>::max()
                - spans[axis]) / stride)
            throw std::overflow_error("common-stop affine issue span overflows");
        spans[axis + 1] = spans[axis] + (count - 1) * stride;
    }
    std::size_t probes = 0;
    const auto search = [&](auto&& self, std::size_t rank,
                            std::size_t start) -> bool {
        if (++probes > 1000000)
            throw std::logic_error("common-stop safety analysis exceeded its finite work bound");
        if (start >= end) return false;
        if (spans[rank] > std::numeric_limits<std::size_t>::max() - start)
            throw std::overflow_error("common-stop final issue cycle overflows");
        if (start + spans[rank] < begin) return false;
        if (rank == 0) return start >= begin;
        const auto [stride, count] = axes[rank - 1];
        const auto lowerSpan = spans[rank - 1];
        const auto first = begin > start + lowerSpan
            ? (begin - start - lowerSpan - 1) / stride + 1 : 0;
        const auto last = std::min(count - 1, (end - 1 - start) / stride);
        for (auto point = first; point <= last && point < count; ++point)
            if (self(self, rank - 1, start + point * stride)) return true;
        return false;
    };
    return search(search, axes.size(), domain.start);
}

std::function<void(std::size_t, std::size_t)>
make_common_drain_validator(const BinaryProgram& program)
{
    // A full fabric traversal plus the longest configured FU path is a
    // deliberately conservative drain cost. The compiler's ingress timeline
    // promises that live cross-operator values have reached stable storage.
    const auto& hwc = program.hardware;
    const auto longestFu = std::max({hwc.mem_to_sxm_latency,
        hwc.mem_to_mxm_latency, hwc.mxm0_accumulator_latency,
        hwc.mxm1_accumulator_latency, hwc.accumulator_to_vxm_latency,
        hwc.accumulator_read_to_vxm_latency, hwc.swiglu_write_latency,
        hwc.qk_iw_to_compute_latency, hwc.vxm_weight_to_iw_latency});
    const std::size_t guard = static_cast<std::size_t>(hwc.system_register_columns)
        + hwc.mem_boundary_register_columns + hwc.mxm_pipeline_rows
        + hwc.tile_rows + longestFu;
    struct Entry {
        CommonIssueDomain domain;
        QueueKind kind;
        std::size_t index, last;
    };
    std::vector<Entry> entries;
    const auto check = [&](const CommonIssueDomain& domain,
                            const QueueProgram& queue) {
        auto last = domain.start;
        for (std::size_t axis = 0; axis < domain.counts.size(); ++axis) {
            const auto count = domain.counts[axis], stride = domain.strides[axis];
            if (count == 0 || stride == 0)
                throw std::logic_error("invalid affine issue domain at common stop");
            if (count - 1 > (std::numeric_limits<std::size_t>::max() - last) / stride)
                throw std::overflow_error("common-stop final issue cycle overflows");
            last += (count - 1) * stride;
        }
        entries.push_back({domain, queue.kind, queue.index, last});
    };
    for (const auto& queue : program.queues) {
        if (queue.kind == QueueKind::C2cDma || queue.kind == QueueKind::C2cRx
            || queue.kind == QueueKind::C2cTx) continue;
        std::size_t cursor = 0, previous = 0;
        bool hasPrevious = false;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            const auto& command = queue.commands[pc];
            if (const auto idle = nop_cycles(command)) {
                if (*idle > std::numeric_limits<std::size_t>::max() - cursor)
                    throw std::overflow_error("common-stop Native NOP overflows");
                cursor += *idle; ++pc; continue;
            }
            if (is_macro_schedule_command(command)) {
                const auto macro = decode_macro_schedule_command(command);
                check({macro.start_cycle, {macro.inner_count, macro.outer_count, 1},
                    {macro.inner_interval, macro.outer_interval, 1}}, queue);
                ++pc; continue;
            }
            const auto opcode = isa::decode_icu_command_opcode(command.command);
            if (is_icu_control_raw_word_command(command)
                || opcode != isa::IcuCommandOpcode::Instruction
                || command.instruction_kind == InstructionKind::None)
                throw std::logic_error("common-stop computation contains an unmapped Native synchronization/control instruction");
            check({cursor}, queue);
            previous = cursor++; hasPrevious = true; ++pc;
        }
    }
    // Compute queues stay immutable while schedule_common appends transport
    // controls. Decode once; retain the exact affine intersection test for
    // every candidate domain rather than weakening the drained-stop proof.
    return [guard, entries = std::move(entries)](std::size_t stop, std::size_t ingressBegin) {
        if (stop < ingressBegin || stop - ingressBegin < guard)
            throw std::logic_error("unsafe common stop: exclusive compiler ingress window is shorter than the drain guard");
        const auto begin = stop - guard;
        for (const auto& entry : entries)
            if (entry.domain.start < stop && entry.last >= begin
                && common_domain_intersects(entry.domain, begin, stop))
                throw std::logic_error("unsafe common stop: computation is still in the drain window: queue="
                    + std::string(queue_kind_name(entry.kind)) + "["
                    + std::to_string(entry.index) + "] stop=" + std::to_string(stop)
                    + " drain_begin=" + std::to_string(begin));
    };
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

IcuLocation linked_queue_location(QueueKind kind, std::size_t queueIndex,
    std::size_t logicalMxmsPerHemisphere)
{
    const auto physicalMxm = [&] {
        if (logicalMxmsPerHemisphere == 0
            || logicalMxmsPerHemisphere > hw::kMxmsPerHemisphere)
            throw std::out_of_range(
                "linked MXM topology cannot be mapped onto the CModel");
        const auto logicalMxmCount =
            hw::kHemispheres * logicalMxmsPerHemisphere;
        if (queueIndex >= logicalMxmCount)
            throw std::out_of_range(
                "linked MXM queue index is outside its logical topology");
        const auto hemisphere = queueIndex / logicalMxmsPerHemisphere;
        const auto localMxm = queueIndex % logicalMxmsPerHemisphere;
        return hemisphere * hw::kMxmsPerHemisphere + localMxm;
    };
    switch (kind) {
    case QueueKind::Mem: {
        const auto hemisphere = static_cast<Hemisphere>(queueIndex
            / InstructionControlUnit::kMemQueuesPerHemisphere);
        const auto local = queueIndex
            % InstructionControlUnit::kMemQueuesPerHemisphere;
        return IcuLocation::Mem(hemisphere,
            local / hw::kMemBanksPerSlice,
            local % hw::kMemBanksPerSlice);
    }
    case QueueKind::MxmLoad:
        return IcuLocation::MxmLoad(physicalMxm());
    case QueueKind::MxmCompute:
        return IcuLocation::MxmCompute(physicalMxm());
    case QueueKind::MxmDequant:
        return IcuLocation::MxmDequant(physicalMxm());
    case QueueKind::Vxm:
        return IcuLocation::Vxm(queueIndex);
    case QueueKind::SxmTranspose:
        return IcuLocation::Sxm(
            static_cast<Hemisphere>(queueIndex), 0);
    case QueueKind::SxmPermute:
        return IcuLocation::Sxm(
            static_cast<Hemisphere>(queueIndex), 1);
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        break;
    }
    throw std::logic_error(
        "page-ready release targets an unsupported ICU queue");
}

} // namespace

C2cWeightPageIcuProgram lower_c2c_weight_page_to_icu(
    const C2cWeightPage& page,
    const SystemHardwareConfiguration& hardware,
    std::uint32_t firstSyncTag,
    std::size_t reservationCycles, bool dynamicReservation)
{
    validate_weight_page(page, hardware);
    const bool useMinimumReservation = reservationCycles == 0;
    const auto hardwareLaneCount = static_cast<std::size_t>(
        hardware.c2c_streams_per_direction);
    const auto fabricStreams =
        resolve_fabric_streams(page, hardwareLaneCount);
    const auto laneCount = fabricStreams.size();
    const auto laneAssignments = allocate_c2c_lanes(page, laneCount);
    constexpr auto maxSyncTag =
        static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max());
    if (firstSyncTag == 0 || firstSyncTag > maxSyncTag
        || page.segments.size() > maxSyncTag)
        throw std::overflow_error(
            "C2C weight-page sync tags exceed the 16-bit ICU field");

    std::array<std::size_t, InstructionControlUnit::kMemQueues>
        vectorsPerQueue{};
    std::array<std::size_t, InstructionControlUnit::kMemQueues> packetsPerQueue{};
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
    for (const auto& segment : page.segments)
        ++packetsPerQueue[InstructionControlUnit::mem_queue(
            segment.hemisphere, segment.slice, segment.bank)];
    const auto& logicalPerQueue = dynamicReservation ? packetsPerQueue : vectorsPerQueue;
    for (const auto count : logicalPerQueue)
        if (!useMinimumReservation && count > reservationCycles)
            throw std::invalid_argument("C2C reservation is shorter than its logical MEM instruction slots");
    auto remainingLogical = logicalPerQueue;

    C2cWeightPageIcuProgram program;
    program.segments.reserve(page.segments.size());
    for (std::size_t segmentIndex = 0;
         segmentIndex < page.segments.size(); ++segmentIndex) {
        const auto& segment = page.segments[segmentIndex];
        const auto lane = laneAssignments[segmentIndex];
        const auto fabricStream = fabricStreams[lane];
        const auto queue = InstructionControlUnit::mem_queue(
            segment.hemisphere, segment.slice, segment.bank);
        const auto group = segment.slice / hw::kMemSlicesPerGroup;
        const auto transportNops =
            hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
        // RX/DMA/MEM queues consume packets in order, with a DMA completion
        // SYNC after every descriptor. Recycle the nonzero 16-bit tag space
        // after a full rotation instead of imposing a 65k-segment limit on
        // an entire decoder layer.
        const auto syncTag = 1 + static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(firstSyncTag - 1) + segmentIndex)
            % maxSyncTag);
        // A dynamically synchronized packet has one logical instruction
        // slot; its actual lifetime still includes all requested FU writes.
        // Legacy standalone reservations retain the fixed write-count floor.
        auto packetReservation = dynamicReservation ? std::size_t {1}
            : static_cast<std::size_t>(segment.vector_count);
        remainingLogical[queue] -= packetReservation;
        const auto queueReservation = useMinimumReservation
            ? logicalPerQueue[queue] : reservationCycles;
        if (remainingLogical[queue] == 0)
            packetReservation += queueReservation - logicalPerQueue[queue];

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
        program.dma_completion_sync[hemisphere_index(segment.hemisphere)] =
            true;
    }
    program.next_sync_tag = 1 + static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(firstSyncTag - 1)
            + program.segments.size()) % maxSyncTag);
    return program;
}

IcuLocation queue_sync_location(QueueKind kind,std::size_t index,std::size_t logical_mxms)
{ return linked_queue_location(kind,index,logical_mxms); }

C2cWeightPager::C2cWeightPager(C2cDmaSystem& system)
    : system_(system)
{
}

void C2cWeightPager::begin_schedule(BinaryProgram& program,
                                  std::size_t overlap_analysis_budget)
{
    common_barrier_protocol_ = program.common_barrier_protocol;
    // Pure preloads and cross-invocation transfers do not need an in-program
    // barrier. schedule_common() still requires it for dynamic page insertion.
    if (active_ && !ready())
        throw std::logic_error(
            "cannot schedule executable pages while a page is in flight");
    if (active_) retire();
    if (common_barrier_protocol_) {
        const bool hadLinkedPlan = !program.barriers.empty();
        std::erase_if(program.timelines, [](const auto& value) {
            return value.name.starts_with("c2c.serial_fallback.");
        });
        program.barriers.clear();
        program.barrier_events.clear();
        for (auto& use : program.weight_page_uses) {
            use.transport_launch_event_id = 0;
            use.transport_launch_cycle = 0;
        }
        if (hadLinkedPlan) {
            std::erase_if(program.queues, [](const QueueProgram& queue) {
                return queue.kind == QueueKind::C2cDma || queue.kind == QueueKind::C2cRx;
            });
            // Preserve compiler KV page-out controls and their independent
            // relocations; remove only derived page-in writes before relink.
            for (auto& queue : program.transport_queues) {
                std::vector<QueueCommand> preserved;
                std::vector<std::size_t> oldToNew(queue.commands.size(),
                    std::numeric_limits<std::size_t>::max());
                for (std::size_t pc = 0; pc < queue.commands.size();) {
                    if (!is_mem_synchronized_raw_packet_header(queue.commands[pc]))
                        throw std::logic_error("common relink encountered an unsupported transport control");
                    const auto packet = decode_mem_synchronized_icu_packet(queue, pc);
                    const auto encoded = ((static_cast<isa::EncodedMemInstruction>(packet[1].lanes[0])
                        | (static_cast<isa::EncodedMemInstruction>(packet[1].lanes[1]) << 32)) >> 2)
                        | (static_cast<isa::EncodedMemInstruction>(packet[1].lanes[2]) << 62);
                    constexpr auto words = InstructionControlUnit::MemIcu::synchronized_packet_word_count;
                    if (isa::decode_mem_instruction(encoded).opcode == MemOpcode::Read)
                        for (std::size_t word = 0; word < words; ++word) {
                            oldToNew[pc + word] = preserved.size();
                            preserved.push_back(queue.commands[pc + word]);
                        }
                    pc += words;
                }
                for (auto& relocation : program.address_relocations) {
                    if (!relocation.transport_queue || relocation.queue_kind != QueueKind::Mem
                        || relocation.queue_index != queue.index) continue;
                    if (relocation.command_index >= oldToNew.size())
                        throw std::logic_error("common relink transport relocation is outside its image");
                    const auto next = oldToNew[relocation.command_index];
                    relocation.command_index = next == std::numeric_limits<std::size_t>::max()
                        ? std::numeric_limits<std::uint32_t>::max() : static_cast<std::uint32_t>(next);
                }
                std::erase_if(program.address_relocations, [&](const BinaryAddressRelocation& relocation) {
                    return relocation.transport_queue && relocation.queue_kind == QueueKind::Mem
                        && relocation.queue_index == queue.index
                        && relocation.command_index == std::numeric_limits<std::uint32_t>::max();
                });
                queue.commands = std::move(preserved);
            }
            std::erase_if(program.transport_queues, [](const QueueProgram& queue) {
                return queue.commands.empty();
            });
        }
    }
    schedule_dma_cursor_ = {};
    schedule_rx_cursor_ = {};
    scheduled_dma_issues_ = {};
    scheduled_rx_segments_ = {};
    scheduled_mem_writes_ = {};
    scheduled_mem_reservations_ = {};
    next_sync_tag_ = 1;
    schedule_open_ = true;
    linked_page_barriers_.clear();
    common_events_.clear();
    common_drain_validator_ = {};
    common_page_ranges_.clear();
    overlap_calendar_.reset();
    if (program.transport_overlap_protocol) {
        try {
            overlap_calendar_ = std::make_shared<CommonOverlapCalendar>(program, overlap_analysis_budget);
        } catch (const LogicalIssueAnalysisLimit&) {
            // Overlap is an optional optimization. A finite proof budget can
            // be exhausted by a valid long scan; retain its compiler-declared
            // drained common stops and synchronize real MEM commits there.
            // Do not catch malformed images, invalid resources or unsafe
            // drain windows. They are not a reason to relax the contract.
            // Persist the resolved protocol in the linked binary too. The
            // loader independently validates it, and must not retry the
            // exhausted overlap proof or interpret serial launches as early.
            program.transport_overlap_protocol = false;
            program.timelines.push_back({"c2c.serial_fallback.proof_work_bound", 0, 0});
        }
    }
    if (common_barrier_protocol_) {
        for (const auto& queue : program.queues)
            if ((queue.kind == QueueKind::C2cDma || queue.kind == QueueKind::C2cRx)
                && !queue.commands.empty())
                throw std::logic_error("common paging linker requires ownership of the C2C transport queues");
        return;
    }


}

C2cWeightPageFence C2cWeightPager::schedule(
    BinaryProgram& binary, const C2cWeightPage& page,
    std::size_t startCycle, std::size_t transferEndCycle,
    std::size_t readyCycle, std::size_t launchEventTag,
    std::size_t pageReadyEventTag, std::size_t earliestMemCycle,
    std::span<const std::size_t> pageUseIndices)
{
    if (!schedule_open_)
        throw std::logic_error(
            "begin_schedule must precede executable C2C page linking");
    if (transferEndCycle <= startCycle)
        throw std::invalid_argument(
            "executable C2C page has an empty transfer window");
    if (common_barrier_protocol_)
        return schedule_common(binary, page, readyCycle, launchEventTag,
            pageReadyEventTag, std::max(startCycle, earliestMemCycle), pageUseIndices);

    // Cross-invocation lookahead has no consumer in this executable. Its
    // destination queues must be transport-only for the entire invocation.
    // Do not split or suspend any computation Macro to insert a page.
    if (pageReadyEventTag != 0 || readyCycle != std::numeric_limits<std::size_t>::max())
        throw std::logic_error("dynamic in-program paging requires the common barrier protocol");
    const auto duration = transferEndCycle - startCycle;
    const auto producer = std::max(startCycle, earliest_schedule_cycle(page));
    if (duration > std::numeric_limits<std::size_t>::max() - producer)
        throw std::overflow_error("lookahead transfer end overflows");
    const auto end = producer + duration;
    const auto lowered = lower_c2c_weight_page_to_icu(page,
        system_.chip().hardware_configuration(), next_sync_tag_, duration);
    std::set<std::size_t> destinations;
    for (const auto& item : lowered.segments) destinations.insert(item.mem_queue);
    std::map<std::size_t, std::size_t> memCursors;
    for (const auto queue : destinations) {
        const auto* existing = find_queue(binary, QueueKind::Mem, queue);
        auto& cursor = memCursors[queue];
        if (existing) {
            if (existing->requires_raw_macro || existing->packed_macro_imem)
                throw std::logic_error("lookahead destination MEM ICU owns a Macro program");
            for (std::size_t pc = 0; pc < existing->commands.size();) {
                const auto& command = existing->commands[pc];
                std::size_t slots = 0;
                if (is_mem_synchronized_raw_packet_header(command)) {
                    (void)decode_mem_synchronized_icu_packet(*existing, pc);
                    slots = (command.words[2] & 0xffffffU) + 1;
                    pc += InstructionControlUnit::MemIcu::synchronized_packet_word_count;
                } else if (is_icu_control_raw_word_command(command)
                    && decode_icu_control_raw_word(command).opcode == IcuControlOpcode::Nop) {
                    slots = decode_icu_control_raw_word(command).count; ++pc;
                } else {
                    throw std::logic_error("lookahead destination MEM ICU is not transport-only");
                }
                if (slots > std::numeric_limits<std::size_t>::max() - cursor)
                    throw std::overflow_error("lookahead MEM cursor overflows");
                cursor += slots;
            }
        }
        if (producer < cursor)
            throw std::logic_error("lookahead overlaps a preceding MEM transfer");
    }
    C2cWeightPageFence fence;
    fence.dma_issues_begin = scheduled_dma_issues_;
    fence.dma_issues_end = scheduled_dma_issues_;
    fence.completed_segments = scheduled_rx_segments_;
    fence.scheduled_start_cycle = producer;
    fence.scheduled_end_cycle = end;
    for (const auto queue : destinations) {
        auto& commands = find_or_create_queue(binary, QueueKind::Mem, queue).commands;
        if (producer > memCursors[queue])
            append_control(commands, IcuControlInstruction::Nop(producer - memCursors[queue]));
        for (const auto& item : lowered.segments) {
            if (item.mem_queue != queue) continue;
            const auto words = encode_mem_synchronized_icu_packet(item.mem_write_sync);
            commands.insert(commands.end(), words.begin(), words.end());
        }
        memCursors[queue] = end;
    }
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto count = std::ranges::count_if(lowered.segments,
            [&](const auto& item) { return hemisphere_index(item.hemisphere) == side; });
        if (!count) continue;
        (void)find_or_create_queue(binary, QueueKind::C2cDma, side);
        (void)find_or_create_queue(binary, QueueKind::C2cRx, side);
        auto& dma = find_queue(binary, QueueKind::C2cDma, side)->commands;
        auto& rx = find_queue(binary, QueueKind::C2cRx, side)->commands;
        append_control(dma, IcuControlInstruction::WaitEvent(launchEventTag));
        append_control(rx, IcuControlInstruction::WaitEvent(launchEventTag));
        for (const auto& item : lowered.segments) {
            if (hemisphere_index(item.hemisphere) != side) continue;
            append_c2c_endpoint(rx, item.rx);
            append_c2c_dma(dma, item.dma);
            fence.dma_issues_end[side] = ++scheduled_dma_issues_[side];
            fence.completed_segments[side][item.lane] = ++scheduled_rx_segments_[side][item.lane];
            scheduled_mem_writes_[item.mem_queue] += item.vector_count;
            fence.completed_mem_writes[item.mem_queue] = scheduled_mem_writes_[item.mem_queue];
            fence.completed_mem_reservations[item.mem_queue] = ++scheduled_mem_reservations_[item.mem_queue];
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
            append_control(dma, IcuControlInstruction::Sync());
        schedule_dma_cursor_[side] = end;
        schedule_rx_cursor_[side] = end;
        binary.max_cycle = std::max<std::uint64_t>(binary.max_cycle,
            c2c_queue_end(producer, count, c2c_dma_frontend_wait_cycles(count)));
    }
    next_sync_tag_ = lowered.next_sync_tag;
    binary.max_cycle = std::max<std::uint64_t>(binary.max_cycle, end);
    return fence;
}

C2cWeightPageFence C2cWeightPager::schedule_common(
    BinaryProgram& binary, const C2cWeightPage& page,
    std::size_t readyCycle, std::size_t launchEventTag,
    std::size_t pageReadyEventTag, std::size_t earliestMemCycle,
    std::span<const std::size_t> pageUseIndices)
{
    if (!binary.common_barrier_protocol || pageReadyEventTag == 0
        || pageReadyEventTag > std::numeric_limits<std::uint32_t>::max()
        || launchEventTag == 0 || launchEventTag > std::numeric_limits<std::uint32_t>::max()
        || pageUseIndices.empty())
        throw std::logic_error("common paging requires a concrete event and nonempty transport-instance association");
    validate_weight_page(page, system_.chip().hardware_configuration());
    std::size_t ingressBegin = 0;
    std::size_t stop = readyCycle;
    for (const auto index : pageUseIndices) {
        if (index >= binary.weight_page_uses.size()
            || index > std::numeric_limits<std::uint32_t>::max())
            throw std::logic_error("common paging references an invalid weight-page use instance");
        const auto& use = binary.weight_page_uses[index];
        const auto name = "c2c.window.binding:" + std::to_string(use.binding_index)
            + ".page:" + std::to_string(use.page_index);
        auto window = binary.timelines.end();
        for (auto candidate = binary.timelines.begin(); candidate != binary.timelines.end(); ++candidate)
            if (candidate->name == name && candidate->end_cycle <= use.ready_cycle
                && (window == binary.timelines.end() || candidate->end_cycle > window->end_cycle))
                window = candidate;
        if (window == binary.timelines.end() || window->start_cycle >= window->end_cycle
            || window->end_cycle > std::numeric_limits<std::uint32_t>::max())
            throw std::logic_error("unsafe common stop: missing or invalid exclusive compiler ingress candidate: " + name);
        ingressBegin = std::max(ingressBegin, static_cast<std::size_t>(window->start_cycle));
        stop = std::min(stop, static_cast<std::size_t>(window->end_cycle));
    }
    if (stop < earliestMemCycle)
        throw std::logic_error("unsafe common stop: transport launch or SRAM reuse depends on computation after the stop: stop="
            + std::to_string(stop) + " earliest_mem=" + std::to_string(earliestMemCycle)
            + " binding=" + std::to_string(binary.weight_page_uses[pageUseIndices.front()].binding_index));
    if (!common_drain_validator_)
        common_drain_validator_ = make_common_drain_validator(binary);
    common_drain_validator_(stop, ingressBegin);
    auto linkedPage = page;
    std::size_t producerStart = stop;
    if (overlap_calendar_) {
        std::vector<std::size_t> queues;
        for (const auto& segment : page.segments)
            queues.push_back(InstructionControlUnit::mem_queue(
                segment.hemisphere, segment.slice, segment.bank));
        const auto lease = overlap_calendar_->select(static_cast<std::uint32_t>(stop),
            earliestMemCycle, queues);
        producerStart = lease.launch;
        if (producerStart == stop)
            binary.timelines.push_back({"c2c.serial_fallback."
                + std::string(earliestMemCycle == stop ? "sram_release" : "mem_sr_lease")
                + ".launch:" + std::to_string(launchEventTag), stop, stop});
        linkedPage.fabric_streams = lease.streams;
        overlap_calendar_->validate(lease.launch, static_cast<std::uint32_t>(stop),
            queues, lease.streams);
    }
    for (const auto index : pageUseIndices) {
        const auto& use = binary.weight_page_uses[index];
        for (const auto& segment : page.segments) {
            if (common_page_ranges_.size() == kBinaryCommonBarrierMaxCompletionInstances)
                throw std::logic_error("common storage proof exceeds 65535 actual MEM segments");
            common_page_ranges_.push_back({InstructionControlUnit::mem_queue(
                segment.hemisphere, segment.slice, segment.bank), segment.base_row,
                segment.vector_count, use.binding_index, static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(stop), use.ready_cycle, use.release_cycle,
                static_cast<std::uint32_t>(producerStart)});
        }
    }
    // E is independent of physical DDR latency. Same-C pages remain ordered
    // on the finite transport path and are joined by one all-of completion.
    const auto lowered = lower_c2c_weight_page_to_icu(linkedPage,
        system_.chip().hardware_configuration(), next_sync_tag_, 0, true);
    C2cWeightPageFence fence;
    fence.transport_queue = true;
    fence.dma_issues_begin = scheduled_dma_issues_;
    fence.dma_issues_end = scheduled_dma_issues_;
    fence.completed_segments = scheduled_rx_segments_;
    fence.scheduled_start_cycle = producerStart;
    fence.scheduled_end_cycle = producerStart;
    const auto getTransport = [&](std::size_t index) -> QueueProgram& {
        const auto found = std::ranges::find_if(binary.transport_queues,
            [&](const QueueProgram& queue) { return queue.kind == QueueKind::Mem && queue.index == index; });
        if (found != binary.transport_queues.end()) return *found;
        binary.transport_queues.push_back(QueueProgram {QueueKind::Mem, index, {}});
        return binary.transport_queues.back();
    };
    std::array<std::size_t, hw::kHemispheres> segmentCounts{};
    for (const auto& segment : lowered.segments)
        ++segmentCounts[hemisphere_index(segment.hemisphere)];
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        if (segmentCounts[side] == 0) continue;
        auto& dmaQueue = find_or_create_queue(binary, QueueKind::C2cDma, side);
        append_control(dmaQueue.commands, IcuControlInstruction::WaitEvent(launchEventTag));
        auto& rxQueue = find_or_create_queue(binary, QueueKind::C2cRx, side);
        append_control(rxQueue.commands, IcuControlInstruction::WaitEvent(launchEventTag));
        // find_or_create_queue may reallocate binary.queues: reacquire refs.
        for (const auto& segment : lowered.segments) {
            if (hemisphere_index(segment.hemisphere) != side) continue;
            append_c2c_endpoint(find_queue(binary, QueueKind::C2cRx, side)->commands, segment.rx);
            append_c2c_dma(find_queue(binary, QueueKind::C2cDma, side)->commands, segment.dma);
            const auto encoded = encode_mem_synchronized_icu_packet(segment.mem_write_sync);
            auto& mem = getTransport(segment.mem_queue).commands;
            // Existing KV readback packets remain intact for the runtime's
            // post-execution extraction; the page-in loader filters them.
            mem.insert(mem.end(), encoded.begin(), encoded.end());
            ++scheduled_dma_issues_[side];
            fence.dma_issues_end[side] = scheduled_dma_issues_[side];
            ++scheduled_rx_segments_[side][segment.lane];
            fence.completed_segments[side][segment.lane] = scheduled_rx_segments_[side][segment.lane];
            scheduled_mem_writes_[segment.mem_queue] += segment.vector_count;
            fence.completed_mem_writes[segment.mem_queue] = scheduled_mem_writes_[segment.mem_queue];
            ++scheduled_mem_reservations_[segment.mem_queue];
            fence.completed_mem_reservations[segment.mem_queue] = scheduled_mem_reservations_[segment.mem_queue];
        }
        for (std::size_t segment = 0; segment < segmentCounts[side]; ++segment)
            append_control(find_queue(binary, QueueKind::C2cDma, side)->commands,
                IcuControlInstruction::Sync());
    }
    next_sync_tag_ = lowered.next_sync_tag;
    linked_page_barriers_.push_back({stop, pageReadyEventTag});
    BinaryBarrierEvent association;
    association.event_id = static_cast<std::uint32_t>(pageReadyEventTag);
    for (const auto index : pageUseIndices) {
        binary.weight_page_uses[index].transport_launch_event_id = static_cast<std::uint32_t>(launchEventTag);
        binary.weight_page_uses[index].transport_launch_cycle = binary.transport_overlap_protocol
            ? static_cast<std::uint32_t>(producerStart) : 0;
        association.page_use_indices.push_back(static_cast<std::uint32_t>(index));
    }
    common_events_.push_back(std::move(association));
    return fence;
}

void C2cWeightPager::inject_page_ready_barriers(
    BinaryProgram& program) const
{
    program.common_barrier_protocol = true;
    program.barriers.clear();
    for (const auto& barrier : linked_page_barriers_)
        program.barriers.push_back(make_binary_barrier(barrier.ready_cycle,
            static_cast<std::uint32_t>(barrier.event_tag)));
    program.barrier_events = common_events_;
    validate_binary_barriers(program);
}

std::vector<C2cWeightPager::PageReadyRelease>
C2cWeightPager::page_ready_releases(std::size_t eventTag) const
{
    // Common barriers resume all participating queues at their compiled phase.
    return {};
}

std::vector<std::pair<std::size_t, std::size_t>>
C2cWeightPager::coalesce_page_ready_events()
{
    if (!schedule_open_) throw std::logic_error("page event coalescing requires an open schedule");
    std::ranges::sort(linked_page_barriers_, [](const auto& a, const auto& b) {
        return std::tie(a.ready_cycle, a.event_tag) < std::tie(b.ready_cycle, b.event_tag);
    });
    std::vector<std::pair<std::size_t, std::size_t>> aliases;
    for (std::size_t begin = 0; begin < linked_page_barriers_.size();) {
        auto end = begin + 1;
        while (end < linked_page_barriers_.size()
            && linked_page_barriers_[end].ready_cycle == linked_page_barriers_[begin].ready_cycle)
            ++end;
        const auto tag = linked_page_barriers_[begin].event_tag;
        for (auto i = begin; i < end; ++i) {
            aliases.emplace_back(linked_page_barriers_[i].event_tag, tag);
            linked_page_barriers_[i].event_tag = tag;
        }
        begin = end;
    }
    if (common_barrier_protocol_) {
        std::vector<BinaryBarrierEvent> merged;
        for (auto event : common_events_) {
            for (const auto& [oldTag, newTag] : aliases)
                if (event.event_id == oldTag) { event.event_id = static_cast<std::uint32_t>(newTag); break; }
            auto target = std::ranges::find_if(merged, [&](const BinaryBarrierEvent& value) {
                return value.event_id == event.event_id;
            });
            if (target == merged.end()) merged.push_back(std::move(event));
            else target->page_use_indices.insert(target->page_use_indices.end(),
                event.page_use_indices.begin(), event.page_use_indices.end());
        }
        for (auto& event : merged) {
            std::ranges::sort(event.page_use_indices);
            event.page_use_indices.erase(std::ranges::unique(event.page_use_indices).begin(),
                event.page_use_indices.end());
        }
        common_events_ = std::move(merged);
    }
    return aliases;
}

void C2cWeightPager::finalize_schedule(BinaryProgram& program)
{
    if (!schedule_open_)
        throw std::logic_error(
            "begin_schedule must precede executable C2C page finalization");

    if (common_barrier_protocol_)
        static_cast<void>(coalesce_page_ready_events());
    std::ranges::sort(linked_page_barriers_,
        [](const LinkedPageBarrier& lhs, const LinkedPageBarrier& rhs) {
            return std::tie(lhs.ready_cycle, lhs.event_tag)
                < std::tie(rhs.ready_cycle, rhs.event_tag);
        });
    linked_page_barriers_.erase(
        std::ranges::unique(linked_page_barriers_,
            [](const LinkedPageBarrier& lhs,
               const LinkedPageBarrier& rhs) {
                return lhs.ready_cycle == rhs.ready_cycle
                    && lhs.event_tag == rhs.event_tag;
            }).begin(),
        linked_page_barriers_.end());
    if (common_barrier_protocol_) {
        inject_page_ready_barriers(program);
        CommonPageStorageValidator storageProof(program);
        for (const auto& load : common_page_ranges_) {
            const std::array<std::uint32_t, 1> ownerBinding {load.binding_index};
            storageProof.validate(load.launch, load.queue, load.row_begin,
                load.row_count, ownerBinding, load.release, common_page_ranges_);
        }
        for (const auto& queue : program.transport_queues)
            if (queue.commands.size() > program.hardware.icu_mem_imem_depth)
                throw std::logic_error("common paging MEM transport control RAM capacity exceeded");
        schedule_open_ = false;
        return;
    }
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

bool C2cWeightPager::transport_ready(
    const C2cWeightPageFence& fence) const
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
        if (((fence.transport_queue || chip.icu().queue_control_transport_enabled(queue)) ? chip.icu().mem_transport_iq(queue)
                                 : chip.icu().mem_iq(queue)).synchronized_issued_count()
            < fence.completed_mem_writes[queue]) {
            fence.mem_writes_issued_cycle.reset();
            return false;
        }

    if(fence.commit_generation) {
        for(std::size_t queue=0;queue<fence.commit_vectors.size();++queue) {
            if(!fence.commit_vectors[queue]) continue;
            const auto h=static_cast<Hemisphere>(queue/InstructionControlUnit::kMemQueuesPerHemisphere);
            const auto local=queue%InstructionControlUnit::kMemQueuesPerHemisphere;
            if(chip.mem_array(h).transport_committed_segments(fence.commit_generation,fence.commit_launch,
                local/hw::kMemBanksPerSlice,local%hw::kMemBanksPerSlice)
                < static_cast<std::uint64_t>(fence.commit_vectors[queue])*hw::kTileRows) return false;
        }
        return true;
    }
    if (!fence.mem_writes_issued_cycle)
        fence.mem_writes_issued_cycle = system_.cycle();
    const std::size_t elapsed =
        system_.cycle() - *fence.mem_writes_issued_cycle + 1;
    if (elapsed < hw::kTileRows) return false;

    return true;
}

bool C2cWeightPager::ready(const C2cWeightPageFence& fence) const
{
    if (!transport_ready(fence)) return false;

    auto& chip = system_.chip();

    for (std::size_t queue = 0;
         queue < fence.completed_mem_reservations.size(); ++queue)
        if (((fence.transport_queue || chip.icu().queue_control_transport_enabled(queue)) ? chip.icu().mem_transport_iq(queue)
                                 : chip.icu().mem_iq(queue)).synchronized_completed_count()
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
    for (std::size_t segmentIndex = 0;
         segmentIndex < page.segments.size(); ++segmentIndex) {
        const auto& segment = page.segments[segmentIndex];
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
    // Keep the exact standalone packets sent to each physical ICU. The
    // executable linker cannot contain these pre-execution commands because
    // each page is loaded and retired before the executable starts.
    last_enqueued_program_ = BinaryProgram {};
    last_enqueued_program_.hardware.words_per_bank =
        static_cast<std::uint32_t>(hardware.sram_depth_rows);
    last_enqueued_program_.hardware.sram_depth_rows =
        static_cast<std::uint32_t>(hardware.sram_depth_rows);
    last_enqueued_program_.hardware.mxms_per_hemisphere =
        static_cast<std::uint32_t>(hardware.mxms_per_hemisphere);
    last_enqueued_program_.hardware.mxm_weight_buffers =
        static_cast<std::uint32_t>(hardware.mxm_weight_buffers);
    last_enqueued_program_.hardware.vxm_alus =
        static_cast<std::uint32_t>(hardware.vxm_alus);
    last_enqueued_program_.hardware.c2c_streams_per_direction =
        static_cast<std::uint32_t>(hardware.c2c_streams_per_direction);
    last_enqueued_program_.hardware.mxm_local_dequant_enabled =
        hardware.mxm_local_dequant_enabled;
    last_enqueued_program_.hardware.mxm_weight_activation_overlap_enabled =
        hardware.mxm_weight_activation_overlap_enabled;
    last_enqueued_program_.target_abi =
        executable_target_abi(last_enqueued_program_.hardware);
    for (const auto& segment : program.segments) {
        const auto side = hemisphere_index(segment.hemisphere);
        append_c2c_endpoint(find_or_create_queue(last_enqueued_program_,
            QueueKind::C2cRx, side).commands, segment.rx);
        append_c2c_dma(find_or_create_queue(last_enqueued_program_,
            QueueKind::C2cDma, side).commands, segment.dma);
        auto encoded = encode_mem_synchronized_icu_packet(
            segment.mem_write_sync);
        auto& mem = find_or_create_queue(last_enqueued_program_,
            QueueKind::Mem, segment.mem_queue).commands;
        mem.insert(mem.end(), encoded.begin(), encoded.end());
    }
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        if (!program.dma_completion_sync[side]) continue;
        append_control(find_or_create_queue(last_enqueued_program_,
            QueueKind::C2cDma, side).commands,
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
