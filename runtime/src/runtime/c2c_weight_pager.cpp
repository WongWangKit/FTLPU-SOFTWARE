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

MemIcuInstruction decode_mem_3d_instruction(
    const QueueProgram& queue, std::size_t commandIndex)
{
    if (queue.kind != QueueKind::Mem
        || commandIndex + fu_3d_raw_packet_word_count(queue.kind, queue.commands.at(commandIndex))
            > queue.commands.size()
        || !is_fu_3d_raw_packet_header(queue.commands[commandIndex]))
        throw std::logic_error("expected MEM READ/WRITE_3D packet");
    isa::EncodedMemIcu3DPacket packet{};
    for (std::size_t word = 0;
         word < fu_3d_raw_packet_word_count(queue.kind, queue.commands.at(commandIndex)); ++word) {
        const auto& source = queue.commands[commandIndex + word];
        if (source.instruction_kind != InstructionKind::Mem
            || source.word_count
                != isa::EncodedMemIcu3DPacket::kLanesPerWord)
            throw std::logic_error("malformed MEM READ/WRITE_3D packet");
        for (std::size_t lane = 0;
             lane < isa::EncodedMemIcu3DPacket::kLanesPerWord; ++lane)
            packet.words[word].lanes[lane] = source.words[lane];
    }
    return isa::decode_mem_icu_3d_instruction(packet);
}

struct Native4ActivationStart {
    std::size_t cycle{std::numeric_limits<std::size_t>::max()};
    std::size_t stream_base{0};
};

constexpr std::size_t kNative4ActivationStartupWindow =
    hw::kMxmBoundaryStreamRegisterColumn + 1;

using Native4ActivationStarts = std::array<Native4ActivationStart,
    hw::kHemispheres>;

std::vector<Native4ActivationStarts> find_native4_activation_starts(
    const BinaryProgram& program,
    const std::vector<std::pair<std::size_t, std::size_t>>& barriers)
{
    std::vector<Native4ActivationStarts> starts(barriers.size());
    const auto mxmsPerHemisphere =
        program.hardware.mxms_per_hemisphere;
    if (mxmsPerHemisphere == 0) return starts;
    for (const auto& queue : program.queues) {
        if (queue.kind == QueueKind::MxmCompute) {
            const auto hemisphere = queue.index / mxmsPerHemisphere;
            if (hemisphere >= hw::kHemispheres) continue;
            std::size_t cursor = 0;
            for (std::size_t pc = 0; pc < queue.commands.size();) {
                const auto& command = queue.commands[pc];
                if (const auto idle = nop_cycles(command)) {
                    cursor += *idle;
                    ++pc;
                } else if (is_fu_3d_raw_packet_header(command)) {
                    isa::EncodedMxmComputeIcu3DPacket packet {};
                    for (std::size_t word = 0; word < fu_3d_raw_packet_word_count(queue.kind, command); ++word)
                        for (std::size_t lane = 0; lane < packet.kLanesPerWord; ++lane)
                            packet.words[word].lanes[lane] = queue.commands[pc + word].words[lane];
                    const auto instruction = isa::decode_mxm_compute_icu_3d_instruction(packet);
                    const auto issue = cursor + instruction.loop.wait_cycle;
                    // Ordinary BF16 COMPUTE_3D consumes the activation stream
                    // directly, without a separate Native4 LoadActivation.
                    if (instruction.opcode == MxmComputeIcuOpcode::Compute3D
                        && instruction.data_format == MxmDataFormat::BFloat16)
                        for (std::size_t index = 0; index < barriers.size(); ++index)
                            if (issue >= barriers[index].first
                                && issue < starts[index][hemisphere].cycle)
                                starts[index][hemisphere] = Native4ActivationStart {
                                    issue, instruction.activation_stream_base};
                    cursor += loop_cycles(instruction.loop);
                    pc += fu_3d_raw_packet_word_count(queue.kind, command);
                } else {
                    ++pc;
                }
            }
            continue;
        }
        if (queue.kind != QueueKind::MxmLoad) continue;
        const auto hemisphere = queue.index / mxmsPerHemisphere;
        if (hemisphere >= hw::kHemispheres) continue;
        for (const auto& command : queue.commands) {
            if (!is_mxm_stream_nd_command(command)) continue;
            const auto schedule = decode_mxm_stream_nd_command(command);
            const auto encoded = static_cast<isa::EncodedMxmInstruction>(
                command.words[0])
                | (static_cast<isa::EncodedMxmInstruction>(
                       command.words[1])
                    << 32);
            const auto instruction = isa::decode_mxm_instruction(encoded);
            if (instruction.opcode != MxmControlOpcode::Decode
                || instruction.decode_operation
                    != MxmDecodeOperation::LoadActivation
                || instruction.decode_layout
                    != MxmDecodeLayout::Native4x4)
                continue;
            for (std::size_t index = 0; index < barriers.size(); ++index) {
                const auto ready = barriers[index].first;
                if (schedule.start_cycle < ready)
                    continue;
                auto& candidate = starts[index][hemisphere];
                if (schedule.start_cycle < candidate.cycle)
                    candidate = Native4ActivationStart {
                        schedule.start_cycle,
                        instruction.activation_stream_base};
            }
        }
    }
    return starts;
}

std::vector<std::pair<std::size_t, std::size_t>> native4_launch_producers(
    const BinaryProgram& program,
    const std::vector<std::pair<std::size_t, std::size_t>>& barriers)
{
    const auto starts = find_native4_activation_starts(program, barriers);
    std::vector<std::pair<std::size_t, std::size_t>> producers;
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::Mem) continue;
        const auto side = queue.index / InstructionControlUnit::kMemQueuesPerHemisphere;
        std::size_t cursor = 0;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            const auto& command = queue.commands[pc];
            if (const auto idle = nop_cycles(command)) {
                cursor += *idle;
                ++pc;
            } else if (is_mem_synchronized_raw_packet_header(command)) {
                cursor += (command.words[2] & 0xffffffU) + 1;
                pc += InstructionControlUnit::MemIcu::synchronized_packet_word_count;
            } else if (is_fu_3d_raw_packet_header(command)) {
                const auto instruction = decode_mem_3d_instruction(queue, pc);
                const auto issue = cursor + instruction.loop.start_cycle + instruction.loop.wait_cycle;
                if (instruction.opcode == MemIcuOpcode::Read3D)
                    for (std::size_t i = 0; i < barriers.size(); ++i) {
                        const auto& activation = starts[i][side];
                        if (activation.cycle == std::numeric_limits<std::size_t>::max()) continue;
                        const auto distance = issue > activation.cycle
                            ? issue - activation.cycle : activation.cycle - issue;
                        const auto stream = StreamId::from_packed(instruction.stream).index();
                        if (distance <= kNative4ActivationStartupWindow
                            && stream >= activation.stream_base
                            && stream < activation.stream_base + sizeof(std::uint16_t))
                            producers.emplace_back(barriers[i].second, queue.index);
                    }
                cursor += loop_cycles(instruction.loop);
                pc += fu_3d_raw_packet_word_count(queue.kind, command);
            } else if (is_mem_write_read_2d_raw_packet_header(command)) {
                cursor += detail::mem_icu_write_read_2d_last_issue_cycle(
                    decode_mem_write_read_2d_raw_packet(queue, pc)) + 1;
                pc += fu_3d_raw_packet_word_count(queue.kind, command);
            } else {
                ++cursor;
                ++pc;
            }
        }
    }
    return producers;
}

void retreat_native4_activation_producer_barriers(
    const QueueProgram& queue,
    std::vector<std::pair<std::size_t, std::size_t>>& barriers,
    const std::vector<Native4ActivationStarts>& activationStarts)
{
    if (queue.kind != QueueKind::Mem || barriers.empty()) return;
    const auto hemisphere = queue.index
        / InstructionControlUnit::kMemQueuesPerHemisphere;
    if (hemisphere >= hw::kHemispheres) return;
    std::size_t cursor = 0;
    for (std::size_t commandIndex = 0;
         commandIndex < queue.commands.size();) {
        const auto& command = queue.commands[commandIndex];
        if (const auto cycles = nop_cycles(command)) {
            cursor += *cycles;
            ++commandIndex;
            continue;
        }
        if (is_mem_synchronized_raw_packet_header(command)) {
            constexpr auto words = InstructionControlUnit::MemIcu::
                synchronized_packet_word_count;
            const auto reservation =
                (command.words[2] & 0x00ffffffU) + 1;
            cursor += reservation;
            commandIndex += words;
            continue;
        }
        if (is_icu_control_raw_word_command(command)) {
            ++commandIndex;
            continue;
        }
        if (is_mem_write_read_2d_raw_packet_header(command)) {
            cursor += detail::mem_icu_write_read_2d_last_issue_cycle(
                          decode_mem_write_read_2d_raw_packet(
                              queue, commandIndex))
                + 1;
            commandIndex += fu_3d_raw_packet_word_count(queue.kind, command);
            continue;
        }
        if (is_fu_3d_raw_packet_header(command)) {
            const auto instruction =
                decode_mem_3d_instruction(queue, commandIndex);
            const auto issueStart = cursor + instruction.loop.start_cycle
                + instruction.loop.wait_cycle;
            const auto commandEnd = cursor + loop_cycles(instruction.loop);
            if (instruction.opcode == MemIcuOpcode::Read3D) {
                const auto stream = StreamId::from_packed(
                    instruction.stream);
                constexpr std::size_t kNative4ActivationStreams =
                    sizeof(std::uint16_t);
                for (std::size_t index = 0; index < barriers.size(); ++index) {
                    const auto& activation =
                        activationStarts[index][hemisphere];
                    const auto ready = barriers[index].first;
                    if (activation.cycle
                        == std::numeric_limits<std::size_t>::max()
                        || ready < issueStart || ready >= commandEnd
                        || ready - issueStart
                            > kNative4ActivationStartupWindow
                        || activation.cycle < ready
                        || activation.cycle >= commandEnd
                        || stream.index() < activation.stream_base
                        || stream.index() >= activation.stream_base
                                + kNative4ActivationStreams)
                        continue;
                    // Native4 consumes the vector emitted just before its
                    // first activation-load issue. If a page wait lands in
                    // that producer READ_3D startup window, waiting only the
                    // MXM queue lets the vector pass while MXM is stopped.
                    // Put this MEM ICU's wait at the first issue instead; the
                    // per-ICU release phase retains the original MEM-to-MXM
                    // lead without splitting the 3-D instruction.
                    barriers[index].first = issueStart;
                }
            }
            cursor = commandEnd;
            commandIndex += fu_3d_raw_packet_word_count(queue.kind, command);
            continue;
        }
        if (is_mem_stream_nd_command(command)) {
            const auto schedule = decode_mem_stream_nd_command(command);
            std::size_t final = schedule.start_cycle;
            for (std::size_t dimension = 0;
                 dimension < schedule.rank; ++dimension)
                final += (schedule.counts[dimension] - 1)
                    * schedule.cycle_strides[dimension];
            cursor = final + 1;
            ++commandIndex;
            continue;
        }
        if (command.instruction_kind != InstructionKind::None
            && command.extension_words.empty()) {
            ++cursor;
            ++commandIndex;
            continue;
        }
        throw std::logic_error(
            "unsupported MEM command while locating native4 activation producer");
    }
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

void inject_page_ready_barriers_impl(BinaryProgram& program,
    const std::vector<std::pair<std::size_t, std::size_t>>& barriers,
    const std::array<std::vector<std::tuple<std::size_t, std::size_t,
        std::size_t>>,
        InstructionControlUnit::kMemQueues>& memWriteWindows,
    std::vector<std::tuple<std::size_t, QueueKind, std::size_t,
        std::size_t>>* insertedBarriers = nullptr)
{
    if (barriers.empty()) return;
    auto timedBarriers = barriers;
    std::ranges::stable_sort(timedBarriers,
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    for (std::size_t begin = 0; begin < timedBarriers.size();) {
        auto end = begin + 1;
        while (end < timedBarriers.size()
            && timedBarriers[end].first == timedBarriers[begin].first)
            ++end;
        const auto precedingSlots = end - begin - 1;
        if (timedBarriers[begin].first < precedingSlots)
            throw std::logic_error("page barriers have no space before their consumer");
        const auto firstSlot = timedBarriers[begin].first - precedingSlots;
        for (auto index = begin; index < end; ++index)
            timedBarriers[index].first = firstSlot;
        begin = end;
    }
    const auto native4ActivationStarts =
        find_native4_activation_starts(program, barriers);
    for (QueueProgram& queue : program.queues) {
        if (queue.kind == QueueKind::C2cDma
            || queue.kind == QueueKind::C2cRx
            || queue.kind == QueueKind::C2cTx)
            continue;

        auto queueBarriers = timedBarriers;
        std::vector<std::size_t> earlyJoinTags;
        if (queue.kind == QueueKind::Mem
            && queue.index < memWriteWindows.size()) {
            retreat_native4_activation_producer_barriers(
                queue, queueBarriers, native4ActivationStarts);
            // Prefetch-only MEM queues do not participate in intervening
            // compute epochs. Join at the next consumer barrier instead;
            // its release preserves the consumer's phase without replaying
            // static idle time after a dynamically late WRITE_SYNC.
            std::vector<std::pair<std::size_t, std::size_t>> activeWindows;
            std::size_t position = 0;
            for (std::size_t pc = 0; pc < queue.commands.size();) {
                const auto& command = queue.commands[pc];
                if (const auto cycles = nop_cycles(command)) {
                    position += *cycles;
                    ++pc;
                } else if (is_mem_synchronized_raw_packet_header(command)) {
                    position += (command.words[2] & 0x00ffffffU) + 1;
                    pc += InstructionControlUnit::MemIcu::synchronized_packet_word_count;
                } else if (is_fu_3d_raw_packet_header(command)) {
                    const auto loop = decode_fu_3d_raw_packet_loop(queue, pc);
                    activeWindows.emplace_back(position + loop.wait_cycle,
                        position + loop_cycles(loop));
                    position += loop_cycles(loop);
                    pc += fu_3d_raw_packet_word_count(queue.kind, command);
                } else {
                    // Conservatively retain fences around other commands.
                    activeWindows.emplace_back(position, position + 1);
                    ++position;
                    ++pc;
                }
            }
            std::vector<std::pair<std::size_t, std::size_t>> retained;
            for (std::size_t index = 0; index < queueBarriers.size(); ++index) {
                const auto& barrier = queueBarriers[index];
                bool asynchronous = false;
                // Include idle time before the asynchronous write begins.
                // A prefetch-only queue has no consumer for these earlier
                // epochs either. Retaining their waits makes it replay old
                // rendezvous after the page commits, skewing weight readers.
                for (const auto& [begin, end, eventTag] : memWriteWindows[queue.index])
                    asynchronous |= barrier.first < end
                        && barrier.second != eventTag;
                const auto next = index + 1;
                if (asynchronous && next < queueBarriers.size()) {
                    const auto nextCycle = queueBarriers[next].first;
                    const bool hasConsumer = std::ranges::any_of(activeWindows,
                        [&](const auto& window) {
                            return window.second > barrier.first
                                && window.first < nextCycle;
                        });
                    if (!hasConsumer) {
                        earlyJoinTags.push_back(queueBarriers[next].second);
                        continue;
                    }
                }
                retained.push_back(barrier);
            }
            queueBarriers = std::move(retained);
            // Remaining barriers protect actual consumers. A WRITE_SYNC
            // cannot be interrupted, so place these joins after its reserved
            // interval while retaining the consumer's relative phase.
            for (auto& barrier : queueBarriers) {
                bool moved = true;
                while (moved) {
                    moved = false;
                    for (const auto& [begin, end, eventTag] :
                         memWriteWindows[queue.index]) {
                        if ((barrier.first >= begin
                                && barrier.first < end)
                            || (barrier.second == eventTag
                                && barrier.first < end)) {
                            barrier.first = end;
                            moved = true;
                        }
                    }
                }
            }
            std::ranges::sort(queueBarriers,
                [](const auto& lhs, const auto& rhs) {
                    return std::tie(lhs.first, lhs.second)
                        < std::tie(rhs.first, rhs.second);
                });
        }

        const auto original = queue.commands;
        std::vector<QueueCommand> linked;
        linked.reserve(original.size() + barriers.size());
        std::vector<std::size_t> oldToNew(original.size(),
            std::numeric_limits<std::size_t>::max());
        std::size_t cursor = 0;
        std::size_t barrierIndex = 0;
        std::size_t barrierIssueDrift = 0;

        // actualCycle is the logical consumer phase, even for an early join
        // whose queue reaches WAIT_EVENT before that logical cycle. Release
        // waits for the compute participants and preserves this phase.
        const auto appendBarrier = [&](const auto& barrier,
                                       std::size_t actualCycle) {
            append_control(linked,
                IcuControlInstruction::WaitEvent(barrier.second));
            if (insertedBarriers != nullptr)
                insertedBarriers->emplace_back(
                    barrier.second, queue.kind, queue.index,
                    actualCycle >= barrierIssueDrift
                        ? actualCycle - barrierIssueDrift : 0);
        };
        auto appendDueBarriers = [&](std::size_t throughCycle) {
            while (barrierIndex < queueBarriers.size()
                && queueBarriers[barrierIndex].first <= throughCycle) {
                appendBarrier(queueBarriers[barrierIndex], cursor);
                // WAIT_EVENT occupies one ICU issue cycle even when the
                // event has already been latched.  Account for that cycle so
                // the following command can consume one fewer idle cycle
                // instead of drifting later once per injected barrier.
                ++cursor;
                ++barrierIndex;
            }
        };

        for (std::size_t commandIndex = 0;
             commandIndex < original.size();) {
            const auto& command = original[commandIndex];
            if (is_mem_synchronized_raw_packet_header(command)) {
                appendDueBarriers(cursor);
                constexpr auto words = InstructionControlUnit::MemIcu::
                    synchronized_packet_word_count;
                static_cast<void>(decode_mem_synchronized_icu_packet(
                    queue, commandIndex));
                const auto reservation =
                    (command.words[2] & 0x00ffffffU) + 1;
                for (std::size_t word = 0; word < words; ++word) {
                    oldToNew[commandIndex + word] = linked.size();
                    linked.push_back(original[commandIndex + word]);
                }
                cursor += reservation;
                commandIndex += words;
                continue;
            }
            if (const auto cycles = nop_cycles(command)) {
                // Keep relocation indices total even when an idle command is
                // split around one or more page barriers.  Parameterized
                // programs normally have no live relocation into a NOP, but
                // retaining the mapping keeps the linked image structurally
                // valid for inspection and re-serialization.
                oldToNew[commandIndex] = linked.size();
                const auto idleEnd = cursor + *cycles;
                while (barrierIndex < queueBarriers.size()
                    && queueBarriers[barrierIndex].first < idleEnd) {
                    const auto barrierCycle = std::max(
                        cursor, queueBarriers[barrierIndex].first);
                    if (std::ranges::find(earlyJoinTags,
                            queueBarriers[barrierIndex].second) == earlyJoinTags.end())
                        append_nop(linked, barrierCycle - cursor);
                    cursor = barrierCycle;
                    appendBarrier(
                        queueBarriers[barrierIndex], barrierCycle);
                    ++cursor;
                    ++barrierIndex;
                }
                if (cursor < idleEnd) {
                    append_nop(linked, idleEnd - cursor);
                    cursor = idleEnd;
                }
                ++commandIndex;
                continue;
            }
            if (is_icu_control_raw_word_command(command)) {
                appendDueBarriers(cursor);
                oldToNew[commandIndex] = linked.size();
                linked.push_back(command);
                ++commandIndex;
                continue;
            }
            if (is_mem_write_read_2d_raw_packet_header(command)
                || is_fu_3d_raw_packet_header(command)) {
                const auto words = fu_3d_raw_packet_word_count(queue.kind, command);
                const bool writeRead =
                    is_mem_write_read_2d_raw_packet_header(command);
                const auto loop = writeRead
                    ? IcuLoop3D {}
                    : decode_fu_3d_raw_packet_loop(queue, commandIndex);
                const auto wait = writeRead ? std::size_t {0}
                                            : loop.wait_cycle;
                const auto commandBase = cursor;
                const auto idleEnd = commandBase + wait;
                while (barrierIndex < queueBarriers.size()
                    && queueBarriers[barrierIndex].first <= idleEnd) {
                    const auto barrierCycle = std::max(
                        cursor, queueBarriers[barrierIndex].first);
                    if (std::ranges::find(earlyJoinTags,
                            queueBarriers[barrierIndex].second) == earlyJoinTags.end())
                        append_nop(linked, barrierCycle - cursor);
                    cursor = barrierCycle;
                    // A barrier at a zero-wait boundary adds an issue cycle
                    // to this queue. Express its release phase in the original
                    // schedule so other queues (which absorb SYNC in wait_cycle)
                    // keep their producer/consumer distance.
                    const auto extraIssue = cursor >= idleEnd ? std::size_t {1}
                                                             : std::size_t {0};
                    barrierIssueDrift += extraIssue;
                    appendBarrier(queueBarriers[barrierIndex], barrierCycle);
                    ++cursor;
                    ++barrierIndex;
                }
                const auto duration = writeRead
                    ? detail::mem_icu_write_read_2d_last_issue_cycle(
                          decode_mem_write_read_2d_raw_packet(
                              queue, commandIndex)) + 1
                    : loop_cycles(loop);
                const auto activeDuration = duration - wait;
                const auto remainingWait = cursor < idleEnd
                    ? idleEnd - cursor : std::size_t {0};
                const auto commandEnd = cursor + remainingWait
                    + activeDuration;
                // A hardware ICU cannot suspend a decoded 3-D domain halfway
                // through its nested loop. Keep a barrier crossed by the
                // active domain pending so appendDueBarriers places it at the
                // next coarse-instruction boundary. The entire command stays
                // in the old execution epoch; the following command begins
                // the new epoch.
                const auto newHeader = linked.size();
                for (std::size_t word = 0; word < words; ++word) {
                    oldToNew[commandIndex + word] = linked.size();
                    linked.push_back(original[commandIndex + word]);
                }
                if (!writeRead && remainingWait != wait)
                    set_raw_3d_wait(linked[newHeader], remainingWait);
                cursor = commandEnd;
                commandIndex += words;
                continue;
            }
            const auto appendStreamNd = [&](const auto& schedule) {
                const auto start = schedule.start_cycle;
                std::size_t final = start;
                for (std::size_t dimension = 0;
                     dimension < schedule.rank; ++dimension)
                    final += (schedule.counts[dimension] - 1)
                        * schedule.cycle_strides[dimension];
                appendDueBarriers(start);
                oldToNew[commandIndex] = linked.size();
                linked.push_back(command);
                cursor = final + 1;
                ++commandIndex;
            };
            if (is_mem_stream_nd_command(command)) {
                appendStreamNd(decode_mem_stream_nd_command(command));
                continue;
            }
            if (is_mxm_stream_nd_command(command)) {
                appendStreamNd(decode_mxm_stream_nd_command(command));
                continue;
            }
            if (is_vxm_stream_nd_command(command)) {
                appendStreamNd(
                    decode_vxm_stream_nd_command(command).schedule);
                continue;
            }
            if (command.instruction_kind != InstructionKind::None
                && command.extension_words.empty()) {
                // Native one-cycle FU instructions remain legal alongside
                // direct 3-D domains. Their absolute placement has already
                // been materialized as preceding NOPs, so a page barrier can
                // be inserted at either instruction boundary.
                oldToNew[commandIndex] = linked.size();
                linked.push_back(command);
                ++cursor;
                ++commandIndex;
                continue;
            }
            if (is_fu_3d_raw_word_command(command))
                throw std::logic_error(
                    "orphan FU 3-D word while injecting page barrier");
            throw std::logic_error(
                "page-ready barrier requires direct FU 3-D/NOP queues: "
                "kind="
                + std::to_string(static_cast<unsigned>(queue.kind))
                + ", queue=" + std::to_string(queue.index)
                + ", command_index=" + std::to_string(commandIndex)
                + ", instruction_kind="
                + std::to_string(static_cast<unsigned>(
                    command.instruction_kind))
                + ", word_count=" + std::to_string(command.word_count)
                + ", extension_words="
                + std::to_string(command.extension_words.size())
                + ", opcode="
                + std::to_string(static_cast<unsigned>(
                    isa::decode_icu_command_opcode(command.command)))
                + ", word0=" + std::to_string(command.words[0]));
        }
        queue.commands = std::move(linked);

        const auto relocate = [&](auto& relocations) {
            for (auto& relocation : relocations) {
                if (relocation.queue_kind != queue.kind
                    || relocation.queue_index != queue.index)
                    continue;
                if (relocation.command_index >= oldToNew.size()
                    || oldToNew[relocation.command_index]
                        == std::numeric_limits<std::size_t>::max())
                    throw std::logic_error(
                        "page barrier relocation target was not preserved");
                relocation.command_index = static_cast<std::uint32_t>(
                    oldToNew[relocation.command_index]);
            }
        };
        relocate(program.address_relocations);
        relocate(program.scale_relocations);
    }
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
        || page.segments.size() - 1 > maxSyncTag - firstSyncTag)
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
        const auto syncTag = firstSyncTag
            + static_cast<std::uint32_t>(segmentIndex);
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
    linked_page_barriers_.clear();
    linked_page_releases_.clear();

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
                const auto words = fu_3d_raw_packet_word_count(queue.kind, command);
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
        auto merged = std::vector<MemIdleWindow> {};
        for (const auto& window : mem_idle_windows_[queue.index]) {
            if (!merged.empty() && merged.back().end == window.begin)
                merged.back().end = window.end;
            else
                merged.push_back(window);
        }
        mem_idle_windows_[queue.index] = std::move(merged);
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
    std::size_t readyCycle, std::size_t launchEventTag,
    std::size_t pageReadyEventTag, std::size_t earliestMemCycle)
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

    auto logicalSlotsPerQueue = std::array<std::size_t,
        InstructionControlUnit::kMemQueues> {};
    std::size_t minimumReservation = 0;
    std::array<std::size_t, InstructionControlUnit::kMemQueues> physicalWritesPerQueue{};
    std::size_t minimumPhysicalWrites = 0;
    for (const auto& segment : page.segments) {
        const auto queue = InstructionControlUnit::mem_queue(
            segment.hemisphere, segment.slice, segment.bank);
        const auto slots = pageReadyEventTag != 0 ? std::size_t {1}
            : static_cast<std::size_t>(segment.vector_count);
        if (slots > std::numeric_limits<std::size_t>::max() - logicalSlotsPerQueue[queue])
            throw std::overflow_error("C2C logical MEM reservation overflows size_t");
        physicalWritesPerQueue[queue] += segment.vector_count;
        minimumPhysicalWrites = std::max(minimumPhysicalWrites, physicalWritesPerQueue[queue]);
        logicalSlotsPerQueue[queue] += slots;
        minimumReservation = std::max(minimumReservation, logicalSlotsPerQueue[queue]);
    }

    const auto reservationEndAt = [&](std::size_t candidate) {
        // The synchronized packet owns the MEM port only while the page is
        // entering SRAM.  The resident rows do not need the ICU queue after
        // the writes complete, so extending this reservation to the first
        // consumer would prevent disjoint-row pages from using the large
        // pre-consumer idle window.  WAIT_EVENT still holds every consumer
        // when real DDR/C2C latency exceeds this planned transfer window.
        // A page must enter the queue before its consumer. A slow DDR
        // estimate may exceed that idle window; the synchronized packet
        // waits for the real transfer and the page barrier stalls consumers.
        // Moving the packet after the consumer would create a dependency
        // cycle: the consumer waits for a page whose writes are behind it.
        // WAIT_EVENT itself consumes one ICU cycle. Leave that cycle before
        // the first ordinary issue so its transport wavefront is preserved.
        auto fenceCycle = readyCycle == 0 ? 0 : readyCycle - 1;
        if (pageReadyEventTag != 0) {
            const auto coveredWaits = std::ranges::count_if(
                linked_page_barriers_, [&](const LinkedPageBarrier& barrier) {
                    return barrier.ready_cycle >= candidate
                        && barrier.ready_cycle <= fenceCycle;
                });
            fenceCycle -= std::min<std::size_t>(fenceCycle, coveredWaits);
        }
        const auto availableBeforeConsumer = fenceCycle > candidate
            ? fenceCycle - candidate : std::size_t {0};
        const auto duration = std::max(minimumReservation,
            pageReadyEventTag != 0
                ? std::min(transportDuration, availableBeforeConsumer)
                : transportDuration);
        if (duration > std::numeric_limits<std::size_t>::max() - candidate)
            throw std::overflow_error(
                "resolved C2C weight-page window overflows size_t");
        return candidate + duration;
    };

    const auto earliestTransport = earliest_schedule_cycle(page);
    // Arm MEM before its consumer even when serialized DDR traffic cannot
    // launch until later. The launch token retains the transport start;
    // MEM_WRITE_SYNC is allowed to wait at its own earlier queue position.
    // Search earlier idle windows too: consecutive consumers can leave only
    // a few cycles between domains, while disjoint page rows can be written
    // in the larger gap before the first domain. The launch token prevents
    // an armed packet from writing before the requested transport time.
    const auto preferredMemStart = pageReadyEventTag != 0
        ? earliestMemCycle : startCycle;
    std::size_t resolvedStart = pageReadyEventTag != 0
        ? preferredMemStart : std::max(startCycle, earliestTransport);
    std::size_t resolvedReservationEnd = 0;
    for (;;) {
        std::size_t commonStart = resolvedStart;
        for (const auto queue : targetQueues) {
            const auto& windows = mem_idle_windows_[queue];
            bool found = false;
            for (const auto& window : windows) {
                const auto candidate = std::max(commonStart, window.begin);
                const auto reservationEnd = pageReadyEventTag != 0
                    ? std::min(reservationEndAt(candidate), window.end)
                    : reservationEndAt(candidate);
                if (candidate > reservationEnd
                    || reservationEnd - candidate < minimumReservation
                    || (pageReadyEventTag != 0
                        && (readyCycle == 0 || reservationEnd >= readyCycle))
                    || candidate > window.end
                    || reservationEnd > window.end)
                    continue;
                commonStart = std::max(commonStart, candidate);
                found = true;
                break;
            }
            if (!found)
                throw std::logic_error(
                        "no legal MEM ICU idle window can hold a C2C weight page: queue="
                        + std::to_string(queue) + " consumer="
                        + std::to_string(readyCycle) + " vectors="
                        + std::to_string(minimumReservation)
                        + " earliest_mem=" + std::to_string(earliestMemCycle)
                        + " common_start=" + std::to_string(commonStart)
                        + " transport_start=" + std::to_string(startCycle)
                        + " transport_duration=" + std::to_string(transportDuration));
        }
        bool allFit = true;
        auto commonReservationEnd = reservationEndAt(commonStart);
        for (const auto queue : targetQueues) {
            const auto& windows = mem_idle_windows_[queue];
            const bool fits = std::ranges::any_of(windows,
                [&](const MemIdleWindow& window) {
                    const auto reservationEnd = pageReadyEventTag != 0
                        ? std::min(reservationEndAt(commonStart), window.end)
                        : reservationEndAt(commonStart);
                    const bool fits = commonStart >= window.begin
                        && (pageReadyEventTag == 0
                            || (readyCycle != 0 && reservationEnd < readyCycle))
                        && commonStart <= window.end
                        && commonStart <= reservationEnd
                        && reservationEnd <= window.end
                        && reservationEnd - commonStart >= minimumReservation;
                    if (fits)
                        commonReservationEnd = std::min(commonReservationEnd, reservationEnd);
                    return fits;
                });
            if (!fits) {
                allFit = false;
                break;
            }
        }
        resolvedStart = commonStart;
        if (allFit) {
            resolvedReservationEnd = commonReservationEnd;
            break;
        }
    }
    const auto producerStart = std::max(
        {startCycle, earliestTransport, resolvedStart});
    if (transportDuration > std::numeric_limits<std::size_t>::max()
            - producerStart)
        throw std::overflow_error(
            "resolved C2C weight-page window overflows size_t");
    const auto transportEnd = producerStart + transportDuration;
    auto segmentCounts = std::array<std::size_t, hw::kHemispheres> {};
    auto dmaLogicalEnds = std::array<std::size_t, hw::kHemispheres> {};
    auto rxLogicalEnds = std::array<std::size_t, hw::kHemispheres> {};
    auto c2cLogicalEnd = producerStart;
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
            producerStart, segmentCount, 0);
        dmaLogicalEnds[side] = c2c_queue_end(producerStart, segmentCount,
            c2c_dma_frontend_wait_cycles(segmentCount));
        c2cLogicalEnd = std::max({c2cLogicalEnd,
            rxLogicalEnds[side], dmaLogicalEnds[side]});
    }
    // The planned end describes a deterministic on-chip transport window.
    // When actual DDR completion is later, MEM_WRITE_SYNC stays at the head
    // while the page-ready barrier holds every ordinary consumer. C2C DMA/RX
    // and MEM transport-only issue keep advancing until all writes retire.
    const auto reservationEnd = resolvedReservationEnd;
    const auto reservationDuration = reservationEnd - resolvedStart;
    const auto lowered = lower_c2c_weight_page_to_icu(
        page, hardware, next_sync_tag_, reservationDuration,
        pageReadyEventTag != 0 && reservationDuration < minimumPhysicalWrites);
    if (pageReadyEventTag != 0) {
        // A late asynchronous page can outlive its predicted reservation.
        // Its MEM_WRITE_SYNC then remains at the head of every target MEM
        // queue and delays the first ordinary command after the selected idle
        // window.  Fence all ordinary ICUs at that earliest collision point;
        // otherwise producers such as SXM continue on the static timeline and
        // consume data that the delayed MEM queues have not emitted yet.
        auto collisionCycle = readyCycle;
        for (const auto queue : targetQueues) {
            const auto& windows = mem_idle_windows_[queue];
            const auto selected = std::ranges::find_if(windows,
                [&](const MemIdleWindow& window) {
                    return resolvedStart >= window.begin
                        && reservationEnd <= window.end;
                });
            if (selected != windows.end()
                && selected->end != std::numeric_limits<std::size_t>::max())
                collisionCycle = std::min(collisionCycle, selected->end);
        }
        linked_page_barriers_.push_back(
            LinkedPageBarrier {collisionCycle == 0 ? 0 : collisionCycle - 1,
                pageReadyEventTag});
    }

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
        window.page_ready_event_tag = pageReadyEventTag;
        for (const auto& segment : lowered.segments)
            if (segment.mem_queue == queue)
                window.packets.push_back(segment.mem_write_sync);
        linked_mem_windows_[queue].push_back(std::move(window));
    }

    C2cWeightPageFence fence;
    fence.dma_issues_begin = scheduled_dma_issues_;
    fence.dma_issues_end = scheduled_dma_issues_;
    fence.completed_segments = scheduled_rx_segments_;
    fence.scheduled_start_cycle = producerStart;
    fence.scheduled_end_cycle = transportEnd;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        const auto segmentCount = segmentCounts[side];
        if (segmentCount == 0) continue;
        if (producerStart < schedule_dma_cursor_[side]
            || producerStart < schedule_rx_cursor_[side])
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
        // The runtime releases launchEventTag at producerStart. Encoding the
        // same absolute delay as a queue-local NOP would make every page wait
        // twice: once in the NOP and again for the runtime event.  Keep pages
        // adjacent in program order instead.  The DMA completion SYNCs below
        // keep one complete coarse page instruction in flight at a time, and
        // WAIT_EVENT supplies the compiler-planned wall-clock constraint.
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
        // A DMA descriptor is a coarse ICU instruction.  Consume every
        // completion token before the next page's WAIT_EVENT can reach the
        // queue head; otherwise descriptors from two pages can be in flight
        // together and drive the same ordinary SR lane concurrently.
        for (std::size_t segment = 0; segment < segmentCount; ++segment)
            append_control(dmaQueue.commands,
                IcuControlInstruction::Sync());
        schedule_dma_cursor_[side] = transportEnd;
        schedule_rx_cursor_[side] = transportEnd;
    }
    next_sync_tag_ = lowered.next_sync_tag;
    binary.max_cycle = std::max(
        binary.max_cycle, std::max(transportEnd, c2cLogicalEnd));
    return fence;
}

void C2cWeightPager::inject_page_ready_barriers(
    BinaryProgram& program) const
{
    std::vector<std::pair<std::size_t, std::size_t>> pageBarriers;
    pageBarriers.reserve(linked_page_barriers_.size());
    for (const LinkedPageBarrier& barrier : linked_page_barriers_)
        pageBarriers.emplace_back(barrier.ready_cycle, barrier.event_tag);
    std::array<std::vector<std::tuple<std::size_t, std::size_t,
        std::size_t>>,
        InstructionControlUnit::kMemQueues> memWriteWindows;
    for (std::size_t queue = 0; queue < linked_mem_windows_.size(); ++queue) {
        memWriteWindows[queue].reserve(linked_mem_windows_[queue].size());
        for (const LinkedMemWindow& window : linked_mem_windows_[queue])
            memWriteWindows[queue].emplace_back(window.begin, window.end,
                window.page_ready_event_tag);
    }
    inject_page_ready_barriers_impl(program, pageBarriers, memWriteWindows);
}

std::vector<C2cWeightPager::PageReadyRelease>
C2cWeightPager::page_ready_releases(std::size_t eventTag) const
{
    const auto barrier = std::ranges::find_if(linked_page_barriers_,
        [&](const LinkedPageBarrier& candidate) {
            return candidate.event_tag == eventTag;
        });
    if (barrier == linked_page_barriers_.end()) return {};
    std::size_t referenceCycle = barrier->ready_cycle;
    for (const LinkedPageRelease& release : linked_page_releases_)
        if (release.event_tag == eventTag)
            referenceCycle = std::min(
                referenceCycle, release.barrier_cycle);
    std::vector<PageReadyRelease> releases;
    for (const LinkedPageRelease& release : linked_page_releases_)
        if (release.event_tag == eventTag)
            releases.push_back(PageReadyRelease {
                release.location,
                release.barrier_cycle - referenceCycle,
                release.activation_producer});
    std::ranges::sort(releases,
        [](const PageReadyRelease& lhs, const PageReadyRelease& rhs) {
            return lhs.phase_offset < rhs.phase_offset;
        });
    return releases;
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
    for (auto& windows : linked_mem_windows_)
        for (auto& window : windows)
            for (const auto& [oldTag, newTag] : aliases)
                if (window.page_ready_event_tag == oldTag) {
                    window.page_ready_event_tag = newTag;
                    break;
                }
    return aliases;
}

void C2cWeightPager::finalize_schedule(BinaryProgram& program)
{
    if (!schedule_open_)
        throw std::logic_error(
            "begin_schedule must precede executable C2C page finalization");

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
    std::vector<std::pair<std::size_t, std::size_t>> pageBarriers;
    pageBarriers.reserve(linked_page_barriers_.size());
    for (const LinkedPageBarrier& barrier : linked_page_barriers_)
        pageBarriers.emplace_back(barrier.ready_cycle, barrier.event_tag);
    std::array<std::vector<std::tuple<std::size_t, std::size_t,
        std::size_t>>,
        InstructionControlUnit::kMemQueues> memWriteWindows;
    for (std::size_t queue = 0; queue < linked_mem_windows_.size(); ++queue) {
        memWriteWindows[queue].reserve(linked_mem_windows_[queue].size());
        for (const LinkedMemWindow& window : linked_mem_windows_[queue])
            memWriteWindows[queue].emplace_back(window.begin, window.end,
                window.page_ready_event_tag);
    }
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
        auto originalQueue = queue;
        auto& original = originalQueue.commands;
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
                const auto leadingWait = window.begin - cursor;
                // WRITE_SYNC already waits for its tagged C2C notification.
                // Let it own the preceding idle cycles too, while extending
                // its reservation so the following MEM packet stays at the
                // same scheduled cycle. The header has a 24-bit reservation.
                constexpr auto kReservationMask = std::uint32_t {0xffffff};
                const auto canAbsorbWait = !window.packets.empty()
                    && leadingWait <= kReservationMask
                        - (window.packets.front()[0].lanes[2]
                            & kReservationMask);
                if (!canAbsorbWait) append_nop(linked, leadingWait);
                for (std::size_t packetIndex = 0;
                     packetIndex < window.packets.size(); ++packetIndex) {
                    auto packet = window.packets[packetIndex];
                    if (packetIndex == 0 && canAbsorbWait) {
                        auto& header = packet[0].lanes[2];
                        header = (header & ~kReservationMask)
                            | ((header & kReservationMask)
                                + static_cast<std::uint32_t>(leadingWait));
                    }
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
            if (is_mem_synchronized_raw_packet_header(command)) {
                constexpr auto words = InstructionControlUnit::MemIcu::
                    synchronized_packet_word_count;
                const auto reservation =
                    (command.words[2] & 0x00ffffffU) + 1;
                for (std::size_t word = 0; word < words; ++word) {
                    oldToNew[commandIndex + word] = linked.size();
                    linked.push_back(original[commandIndex + word]);
                }
                cursor += reservation;
                commandIndex += words;
                continue;
            }
            if (const auto cycles = nop_cycles(command)) {
                auto idleEnd = cursor + *cycles;
                auto next = commandIndex + 1;
                while (next < original.size()) {
                    const auto nextCycles = nop_cycles(original[next]);
                    if (!nextCycles) break;
                    oldToNew[next] = linked.size();
                    idleEnd += *nextCycles;
                    ++next;
                }
                if (next < original.size()
                    && is_fu_3d_raw_packet_header(original[next])
                    && !is_mem_write_read_2d_raw_packet_header(original[next])) {
                    const auto wait = decode_fu_3d_raw_packet_loop(
                        originalQueue, next).wait_cycle;
                    idleEnd += wait;
                    if (wait != 0) set_raw_3d_wait(original[next], 0);
                }
                emitWindows(idleEnd, false);
                append_nop(linked, idleEnd - cursor);
                cursor = idleEnd;
                commandIndex = next;
                continue;
            }
            if (is_icu_control_raw_word_command(command)) {
                emitWindows(cursor, false);
                oldToNew[commandIndex] = linked.size();
                linked.push_back(command);
                ++cursor;
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
            const auto words = fu_3d_raw_packet_word_count(queue.kind, command);
            const auto loop =
                is_mem_write_read_2d_raw_packet_header(command)
                ? IcuLoop3D {}
                : decode_fu_3d_raw_packet_loop(originalQueue, commandIndex);
            const auto wait = loop.wait_cycle;
            const auto idleEnd = cursor + wait;
            if (wait != 0)
                emitWindows(idleEnd, false);
            const auto remainingWait = idleEnd - cursor;
            const auto duration =
                is_mem_write_read_2d_raw_packet_header(command)
                ? detail::mem_icu_write_read_2d_last_issue_cycle(
                      decode_mem_write_read_2d_raw_packet(
                          originalQueue, commandIndex)) + 1
                : loop_cycles(
                      decode_fu_3d_raw_packet_loop(
                          originalQueue, commandIndex));
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
    // Materialize asynchronous C2C writes first.  Page barriers are then
    // inserted into the remaining idle cycles and can be moved to the end of
    // a write reservation without accidentally being placed before the
    // MEM_WRITE_SYNC packet that owns that reservation.
    const auto activationProducers = native4_launch_producers(program, pageBarriers);
    std::vector<std::tuple<std::size_t, QueueKind, std::size_t,
        std::size_t>> insertedBarriers;
    inject_page_ready_barriers_impl(
        program, pageBarriers, memWriteWindows, &insertedBarriers);
    linked_page_releases_.clear();
    linked_page_releases_.reserve(insertedBarriers.size());
    for (const auto& [eventTag, kind, queueIndex, barrierCycle] :
         insertedBarriers)
        linked_page_releases_.push_back(LinkedPageRelease {
            eventTag,
            linked_queue_location(kind, queueIndex,
                program.hardware.mxms_per_hemisphere),
            barrierCycle,
            kind == QueueKind::Mem && std::ranges::find(activationProducers,
                std::pair {eventTag, queueIndex}) != activationProducers.end()});
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

    return true;
}

bool C2cWeightPager::ready(const C2cWeightPageFence& fence) const
{
    if (!transport_ready(fence)) return false;

    auto& chip = system_.chip();

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
