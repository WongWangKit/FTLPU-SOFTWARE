#include "ftlpu/software/runtime/common_transport_validation.hpp"
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

void validate_common_drain_window(const BinaryProgram& program,
    std::size_t stop, std::size_t ingressBegin)
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
    if (stop < ingressBegin || stop - ingressBegin < guard)
        throw std::logic_error("unsafe common stop: exclusive compiler ingress window is shorter than the drain guard");
    const auto begin = stop - guard;
    const auto check = [&](const CommonIssueDomain& domain,
                            const QueueProgram& queue) {
        if (common_domain_intersects(domain, begin, stop))
            throw std::logic_error("unsafe common stop: computation is still in the drain window: queue="
                + std::string(queue_kind_name(queue.kind)) + "["
                + std::to_string(queue.index) + "] stop=" + std::to_string(stop)
                + " drain_begin=" + std::to_string(begin));
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
            if (is_mem_write_read_2d_raw_packet_header(command))
                throw std::logic_error("common-stop proof does not support unsplit MEM WRITE_READ_2D");
            if (is_fu_3d_raw_packet_header(command)) {
                const auto loop = decode_fu_3d_raw_packet_loop(queue, pc);
                const auto duration = loop_cycles(loop);
                if (duration > std::numeric_limits<std::size_t>::max() - cursor)
                    throw std::overflow_error("common-stop Native loop overflows");
                check({cursor + loop.start_cycle + loop.wait_cycle,
                    loop.counts, loop.cycle_strides}, queue);
                cursor += duration;
                pc += fu_3d_raw_packet_word_count(queue.kind, command);
                continue;
            }
            const auto checkSchedule = [&](const auto& schedule) {
                check({schedule.start_cycle, schedule.counts,
                    schedule.cycle_strides}, queue);
            };
            if (is_mem_stream_nd_command(command)) {
                checkSchedule(decode_mem_stream_nd_command(command)); ++pc; continue;
            }
            if (is_mxm_stream_nd_command(command)) {
                checkSchedule(decode_mxm_stream_nd_command(command)); ++pc; continue;
            }
            if (is_vxm_stream_nd_command(command)) {
                checkSchedule(decode_vxm_stream_nd_command(command).schedule); ++pc; continue;
            }
            if (is_sxm_tile_program_command(command)) {
                checkSchedule(decode_sxm_tile_program_command(command).schedule); ++pc; continue;
            }
            if (is_mem_slice_program_command(command)) {
                const auto descriptor = decode_mem_slice_program_command(command);
                for (const auto& body : descriptor.body) {
                    auto schedule = descriptor.schedule;
                    schedule.start_cycle += body.cycle_offset;
                    checkSchedule(schedule);
                }
                ++pc; continue;
            }
            if (is_repeat_2d_command(command)) {
                if (!hasPrevious) throw std::logic_error("common-stop Repeat2D has no instruction");
                const auto repeat = decode_repeat_2d_command(command);
                check({previous, {repeat.inner_count, repeat.outer_count, 1},
                    {repeat.inner_interval, repeat.outer_interval, 1}}, queue);
                cursor = previous + (repeat.inner_count - 1) * repeat.inner_interval
                    + (repeat.outer_count - 1) * repeat.outer_interval + 1;
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
                    // WAIT_EVENT itself occupies one issue cycle. Use the
                    // idle slot immediately before the producer's first
                    // read so the read retains its compiler-scheduled lead
                    // over MXM after a dynamically late page release.
                    barriers[index].first = issueStart > cursor
                        ? issueStart - 1 : issueStart;
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

struct MxmOutputTail {
    std::size_t event_tag{0};
    std::size_t hemisphere{0};
    std::size_t stream_base{0};
    std::size_t last_compute_issue{0};
};
using MxmOutputTailsByEvent =
    std::unordered_map<std::size_t, std::vector<MxmOutputTail>>;

std::vector<MxmOutputTail> find_mxm_output_tails(
    const BinaryProgram& program,
    const std::vector<std::pair<std::size_t, std::size_t>>& barriers)
{
    std::vector<MxmOutputTail> tails;
    const auto mxmsPerHemisphere = program.hardware.mxms_per_hemisphere;
    if (mxmsPerHemisphere == 0) return tails;
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::MxmCompute) continue;
        const auto hemisphere = queue.index / mxmsPerHemisphere;
        if (hemisphere >= hw::kHemispheres) continue;
        std::size_t cursor = 0;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            const auto& command = queue.commands[pc];
            if (const auto idle = nop_cycles(command)) {
                cursor += *idle;
                ++pc;
            } else if (is_fu_3d_raw_packet_header(command)) {
                isa::EncodedMxmComputeIcu3DPacket packet{};
                const auto words = fu_3d_raw_packet_word_count(
                    queue.kind, command);
                for (std::size_t word = 0; word < words; ++word)
                    for (std::size_t lane = 0; lane < packet.kLanesPerWord;
                         ++lane)
                        packet.words[word].lanes[lane] =
                            queue.commands[pc + word].words[lane];
                const auto instruction =
                    isa::decode_mxm_compute_icu_3d_instruction(packet);
                const auto issueStart = cursor + instruction.loop.start_cycle
                    + instruction.loop.wait_cycle;
                const auto commandEnd = cursor + loop_cycles(instruction.loop);
                const bool streamsResult =
                    instruction.regular_mode.accumulator_destination
                            == MxmAccumulatorDestination::Stream
                    || instruction.terminal_mode.accumulator_destination
                            == MxmAccumulatorDestination::Stream;
                if (streamsResult)
                    for (const auto& [ready, eventTag] : barriers)
                        if (issueStart < ready
                            && ready < commandEnd
                                + program.hardware.accumulator_to_vxm_latency
                                + program.hardware.vxm_alus + 2)
                            tails.push_back(MxmOutputTail {eventTag,
                                hemisphere,
                                instruction.result_stream_base,
                                commandEnd - 1});
                cursor = commandEnd;
                pc += words;
            } else {
                ++cursor;
                ++pc;
            }
        }
    }
    return tails;
}

void defer_mem_barriers_past_mxm_output_tails(
    const QueueProgram& queue, const ExecutableHardwareConfig& hardware,
    const MxmOutputTailsByEvent& tailsByEvent,
    std::vector<std::pair<std::size_t, std::size_t>>& barriers)
{
    if (queue.kind != QueueKind::Mem || tailsByEvent.empty()
        || hardware.mem_slices_per_register_group == 0)
        return;
    const auto slice = (queue.index
        % InstructionControlUnit::kMemQueuesPerHemisphere)
        / hw::kMemBanksPerSlice;
    const auto group = slice / hardware.mem_slices_per_register_group;
    const auto hemisphere = queue.index
        / InstructionControlUnit::kMemQueuesPerHemisphere;
    if (hardware.system_register_columns <= group + 1) return;
    const auto localResultLatency = hardware.mxm_pipeline_rows - 1
        + hardware.system_register_columns - 1 - group;
    const auto remoteResultLatency = hardware.mxm_pipeline_rows - 1
        + hardware.accumulator_to_vxm_latency + hardware.vxm_alus
        + group;
    std::size_t cursor = 0;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (const auto idle = nop_cycles(command)) {
            cursor += *idle;
            ++pc;
        } else if (is_mem_synchronized_raw_packet_header(command)) {
            cursor += (command.words[2] & 0x00ffffffU) + 1;
            pc += InstructionControlUnit::MemIcu::synchronized_packet_word_count;
        } else if (is_mem_write_read_2d_raw_packet_header(command)) {
            cursor += detail::mem_icu_write_read_2d_last_issue_cycle(
                decode_mem_write_read_2d_raw_packet(queue, pc)) + 1;
            pc += fu_3d_raw_packet_word_count(queue.kind, command);
        } else if (is_fu_3d_raw_packet_header(command)) {
            const auto instruction = decode_mem_3d_instruction(queue, pc);
            const auto issueStart = cursor + instruction.loop.start_cycle
                + instruction.loop.wait_cycle;
            const auto commandEnd = cursor + loop_cycles(instruction.loop);
            if (instruction.opcode == MemIcuOpcode::Write3D) {
                const auto stream = StreamId::from_packed(instruction.stream);
                for (auto& [ready, eventTag] : barriers) {
                    if (issueStart < ready || ready >= commandEnd) continue;
                    const auto found = tailsByEvent.find(eventTag);
                    if (found == tailsByEvent.end()) continue;
                    for (const auto& tail : found->second) {
                        const bool local = stream.direction()
                                == StreamDirection::West
                            && hemisphere == tail.hemisphere;
                        const bool remote = stream.direction()
                                == StreamDirection::East
                            && hemisphere != tail.hemisphere;
                        if (tail.event_tag != eventTag
                            || (!local && !remote)
                            || stream.index() < tail.stream_base
                            || stream.index() >= tail.stream_base
                                + hardware.mxm_result_streams
                            || issueStart > tail.last_compute_issue
                                + (local ? localResultLatency
                                         : remoteResultLatency))
                            continue;
                        // The compute domain started before this page wait.
                        // Its last result can still be in the local stream
                        // route or the passive VXM bridge. Complete that
                        // old-page MEM write before joining the next page.
                        ready = commandEnd;
                    }
                }
            }
            cursor = commandEnd;
            pc += fu_3d_raw_packet_word_count(queue.kind, command);
        } else {
            ++cursor;
            ++pc;
        }
    }
}

struct MxmOutputTilePair {
    std::size_t event_tag;
    std::size_t queue_index;
    std::size_t activation_stream_base;
    std::size_t first_begin;
    std::size_t second_begin;
    std::size_t second_end;
};

std::optional<std::size_t> mem_result_write_delay_after_page_wait(
    const QueueProgram& queue, std::size_t eventTag,
    std::size_t resultStreamBase, std::size_t resultStreamCount,
    std::size_t* contiguousWriteCount = nullptr)
{
    if (queue.kind != QueueKind::Mem) return std::nullopt;
    bool afterWait = false;
    std::size_t delay = 0;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (const auto idle = nop_cycles(command)) {
            if (afterWait) delay += *idle;
            ++pc;
            continue;
        }
        if (is_icu_control_raw_word_command(command)) {
            const auto control = decode_icu_control_raw_word(command);
            if (control.opcode == IcuControlOpcode::WaitEvent
                && control.event_tag == eventTag) {
                afterWait = true;
                delay = 1;
            } else if (afterWait) {
                return std::nullopt;
            }
            ++pc;
            continue;
        }
        if (is_fu_3d_raw_packet_header(command)) {
            const auto words = fu_3d_raw_packet_word_count(
                queue.kind, command);
            if (afterWait) {
                const auto instruction = decode_mem_3d_instruction(queue, pc);
                const auto stream = StreamId::from_packed(
                    instruction.stream);
                if ((instruction.opcode == MemIcuOpcode::Write3D
                        || instruction.opcode == MemIcuOpcode::WriteTap3D)
                    && stream.direction() == StreamDirection::West
                    && stream.index() >= resultStreamBase
                    && stream.index() < resultStreamBase
                        + resultStreamCount)
                {
                    if (contiguousWriteCount
                        && instruction.loop.counts[1] == 1
                        && instruction.loop.counts[2] == 1
                        && instruction.loop.cycle_strides[0] == 1)
                        *contiguousWriteCount = instruction.loop.counts[0];
                    return delay + instruction.loop.start_cycle
                        + instruction.loop.wait_cycle;
                }
                return std::nullopt;
            }
            pc += words;
            continue;
        }
        if (afterWait) return std::nullopt;
        ++pc;
    }
    return std::nullopt;
}

struct MxmPageResultStream {
    std::size_t event_tag{0};
    std::size_t hemisphere{0};
    std::size_t stream_base{0};
    std::size_t first_result_delay{0};
    std::size_t last_compute_delay{0};
    std::size_t queue_index{0};
    std::size_t activation_stream_base{0};
    std::size_t first_compute_delay{1};
    std::size_t contiguous_result_count{0};
};

std::vector<MxmPageResultStream> find_mxm_page_result_streams(
    const BinaryProgram& program)
{
    std::vector<MxmPageResultStream> streams;
    if (program.hardware.mxms_per_hemisphere == 0) return streams;
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::MxmCompute) continue;
        std::optional<std::size_t> pageEvent;
        std::size_t delay = 0;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            const auto& command = queue.commands[pc];
            if (const auto idle = nop_cycles(command)) {
                if (pageEvent) delay += *idle;
                ++pc;
                continue;
            }
            if (is_icu_control_raw_word_command(command)) {
                const auto control = decode_icu_control_raw_word(command);
                if (control.opcode == IcuControlOpcode::WaitEvent) {
                    pageEvent = control.event_tag;
                    delay = 1;
                } else {
                    pageEvent.reset();
                }
                ++pc;
                continue;
            }
            if (is_fu_3d_raw_packet_header(command)) {
                const auto words = fu_3d_raw_packet_word_count(
                    queue.kind, command);
                if (pageEvent) {
                    isa::EncodedMxmComputeIcu3DPacket packet{};
                    for (std::size_t word = 0; word < words; ++word)
                        for (std::size_t lane = 0;
                             lane < packet.kLanesPerWord; ++lane)
                            packet.words[word].lanes[lane] =
                                queue.commands[pc + word].words[lane];
                    const auto instruction =
                        isa::decode_mxm_compute_icu_3d_instruction(packet);
                    const auto dimension = instruction.terminal_dimension;
                    const bool hasTerminal = dimension < IcuLoop3D::kDimensions;
                    const bool regularStream = instruction.regular_mode
                        .accumulator_destination == MxmAccumulatorDestination::Stream;
                    const bool terminalStream = hasTerminal && instruction.terminal_mode
                        .accumulator_destination == MxmAccumulatorDestination::Stream;
                    const bool firstIsTerminal = hasTerminal
                        && instruction.loop.counts[dimension] == 1;
                    std::optional<std::size_t> firstResult;
                    if (firstIsTerminal ? terminalStream : regularStream)
                        firstResult = 0;
                    else if (terminalStream)
                        firstResult = (instruction.loop.counts[dimension] - 1)
                            * instruction.loop.cycle_strides[dimension];
                    if (instruction.opcode == MxmComputeIcuOpcode::Compute3D
                        && firstResult) {
                        streams.push_back(MxmPageResultStream {
                            *pageEvent,
                            queue.index / program.hardware.mxms_per_hemisphere,
                            instruction.result_stream_base,
                            delay + instruction.loop.start_cycle
                                + instruction.loop.wait_cycle + *firstResult,
                            delay + loop_cycles(instruction.loop) - 1,
                            queue.index, instruction.activation_stream_base,
                            delay + instruction.loop.start_cycle + instruction.loop.wait_cycle,
                            instruction.loop.cycle_strides[0] == 1
                                ? instruction.loop.counts[0] : 0});
                        pageEvent.reset();
                    }
                    delay += loop_cycles(instruction.loop);
                }
                pc += words;
                continue;
            }
            if (!nop_cycles(command)) pageEvent.reset();
            ++pc;
        }
    }
    return streams;
}

std::size_t mem_activation_first_read_delay_after_page_wait(
    const QueueProgram& queue, std::size_t eventTag,
    std::size_t activationStreamBase)
{
    if (queue.kind != QueueKind::Mem) return 0;
    bool afterWait = false;
    std::size_t elapsed = 0;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (is_icu_control_raw_word_command(command)) {
            const auto control = decode_icu_control_raw_word(command);
            if (control.opcode == IcuControlOpcode::WaitEvent
                && control.event_tag == eventTag) {
                afterWait = true;
                elapsed = 1;
            } else if (afterWait && control.opcode == IcuControlOpcode::Nop) {
                elapsed += control.count;
            } else if (afterWait) {
                return 0;
            }
            ++pc;
            continue;
        }
        if (is_fu_3d_raw_packet_header(command)) {
            const auto words = fu_3d_raw_packet_word_count(
                queue.kind, command);
            if (afterWait) {
                const auto instruction = decode_mem_3d_instruction(queue, pc);
                const auto stream = StreamId::from_packed(instruction.stream);
                if (instruction.opcode == MemIcuOpcode::Read3D
                    && stream.direction() == StreamDirection::East
                    && stream.index() >= activationStreamBase
                    && stream.index() < activationStreamBase + 2)
                    return elapsed + instruction.loop.start_cycle
                        + instruction.loop.wait_cycle;
                return 0;
            }
            pc += words;
            continue;
        }
        if (afterWait) return 0;
        ++pc;
    }
    return 0;
}

void defer_mxm_barriers_past_output_tile_pairs(
    const QueueProgram& queue,
    std::vector<std::pair<std::size_t, std::size_t>>& barriers,
    std::vector<MxmOutputTilePair>* outputPairs = nullptr)
{
    if (queue.kind != QueueKind::MxmCompute || barriers.empty()) return;
    struct OutputTile {
        std::size_t begin;
        std::size_t end;
        MxmComputeIcuInstruction instruction;
    };
    std::optional<OutputTile> previous;
    std::size_t cursor = 0;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (const auto idle = nop_cycles(command)) {
            cursor += *idle;
            ++pc;
            continue;
        }
        if (!is_fu_3d_raw_packet_header(command)) {
            previous.reset();
            ++cursor;
            ++pc;
            continue;
        }
        const auto words = fu_3d_raw_packet_word_count(queue.kind, command);
        isa::EncodedMxmComputeIcu3DPacket packet{};
        for (std::size_t word = 0; word < words; ++word)
            for (std::size_t lane = 0; lane < packet.kLanesPerWord; ++lane)
                packet.words[word].lanes[lane] =
                    queue.commands[pc + word].words[lane];
        const auto current = isa::decode_mxm_compute_icu_3d_instruction(packet);
        const auto begin = cursor + current.loop.start_cycle
            + current.loop.wait_cycle;
        const auto end = cursor + loop_cycles(current.loop);
        const auto streamsResult = [](const MxmComputeIcuInstruction& value) {
            return value.opcode == MxmComputeIcuOpcode::Compute3D
                && value.regular_mode.accumulator_destination
                    == MxmAccumulatorDestination::Stream
                && value.terminal_mode.accumulator_destination
                    == MxmAccumulatorDestination::Stream
                && value.loop.counts[0] > 1
                && value.loop.counts[0] <= hw::kMxmRows
                && value.loop.counts[1] == 1
                && value.loop.counts[2] == 1;
        };
        if (previous && streamsResult(previous->instruction)
            && streamsResult(current)
            && begin == previous->end
            && previous->instruction.weight_buffer_base
                == current.weight_buffer_base
            && previous->instruction.result_stream_base
                == current.result_stream_base
            && previous->instruction.loop.counts[0]
                == current.loop.counts[0]
            && previous->instruction.accumulator_address_base
                % (2 * current.loop.counts[0]) == 0
            && current.accumulator_address_base
                == previous->instruction.accumulator_address_base
                    + current.loop.counts[0]) {
            for (auto& [ready, eventTag] : barriers) {
                // WAIT_EVENT may be scheduled one cycle before a page's
                // nominal boundary. If it lands inside the first output
                // tile, the ICU cannot pause that domain and would otherwise
                // insert the wait between the two tiles. Keep the complete
                // output wave in one execution epoch.
                if (ready > previous->begin && ready <= begin) {
                    if (outputPairs != nullptr)
                        outputPairs->push_back(MxmOutputTilePair {
                            eventTag, queue.index,
                            current.activation_stream_base,
                            previous->begin, begin, end});
                    ready = end;
                }
            }
        }
        previous = OutputTile {begin, end, current};
        cursor = end;
        pc += words;
    }
}

void align_mem_barriers_at_mxm_activation_wave_boundary(
    const QueueProgram& queue, const ExecutableHardwareConfig& hardware,
    const std::vector<MxmOutputTilePair>& pairs,
    std::vector<std::pair<std::size_t, std::size_t>>& barriers,
    std::vector<std::pair<std::size_t, std::size_t>>* activationProducers = nullptr,
    std::unordered_map<std::size_t, std::size_t>* barrierShifts = nullptr)
{
    if (queue.kind != QueueKind::Mem || pairs.empty()
        || hardware.mxms_per_hemisphere == 0
        || hardware.mem_slices_per_register_group == 0)
        return;
    const auto hemisphere = queue.index
        / InstructionControlUnit::kMemQueuesPerHemisphere;
    const auto slice = (queue.index
        % InstructionControlUnit::kMemQueuesPerHemisphere)
        / hw::kMemBanksPerSlice;
    const auto registerGroup = slice
        / hardware.mem_slices_per_register_group;
    const auto tokenLane = slice / 2;
    std::size_t cursor = 0;
    std::unordered_map<std::size_t, std::size_t> firstReadEndByEvent;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (const auto idle = nop_cycles(command)) {
            cursor += *idle;
            ++pc;
            continue;
        }
        if (is_mem_synchronized_raw_packet_header(command)) {
            cursor += (command.words[2] & 0x00ffffffU) + 1;
            pc += InstructionControlUnit::MemIcu::
                synchronized_packet_word_count;
            continue;
        }
        if (!is_fu_3d_raw_packet_header(command)) {
            ++cursor;
            ++pc;
            continue;
        }
        const auto words = fu_3d_raw_packet_word_count(queue.kind, command);
        if (is_mem_write_read_2d_raw_packet_header(command)) {
            cursor += detail::mem_icu_write_read_2d_last_issue_cycle(
                decode_mem_write_read_2d_raw_packet(queue, pc)) + 1;
            pc += words;
            continue;
        }
        const auto instruction = decode_mem_3d_instruction(queue, pc);
        const auto issueStart = cursor + instruction.loop.start_cycle
            + instruction.loop.wait_cycle;
        const auto commandEnd = cursor + loop_cycles(instruction.loop);
        if (instruction.opcode == MemIcuOpcode::Read3D) {
            const auto stream = StreamId::from_packed(instruction.stream);
            if (stream.direction() == StreamDirection::East)
                for (const auto& pair : pairs) {
                    // A slice may emit the next output wave's first vector
                    // while the previous MXM tile still computes. Classify
                    // the read by its arrival at MXM, not only its MEM issue
                    // cycle, before placing the dynamic page wait.
                    if (pair.queue_index / hardware.mxms_per_hemisphere
                            != hemisphere
                        || stream.index() < pair.activation_stream_base
                        || stream.index()
                            >= pair.activation_stream_base + 2
                        || issueStart < pair.first_begin
                        || issueStart >= pair.second_end)
                        continue;
                    if (activationProducers != nullptr)
                        activationProducers->emplace_back(
                            pair.event_tag, queue.index);
                    for (auto& [ready, eventTag] : barriers)
                        if (eventTag == pair.event_tag
                            && ready <= pair.second_end) {
                            // The final activation read for the following
                            // page may issue while the prior output tile is
                            // still computing: it needs SR flight time.
                            // Such a vector cannot cross a dynamic page
                            // wait, because it would leave the SR long
                            // before the next MXM domain resumes. Move the
                            // MEM wait before this read, not after it.
                            const bool planarActivationSlice =
                                slice < 2 * hardware.mxm_block_rows
                                && hardware.system_register_columns + 1
                                    >= registerGroup + tokenLane;
                            const auto mxmArrival = planarActivationSlice
                                ? issueStart
                                    + hardware.system_register_columns + 1
                                    - registerGroup - tokenLane
                                : 0;
                            if (planarActivationSlice
                                && mxmArrival >= pair.second_end) {
                                ready = std::min(ready, cursor);
                            } else if (ready < commandEnd) {
                                firstReadEndByEvent.try_emplace(eventTag,
                                    commandEnd);
                                ready = std::max(ready, commandEnd);
                            }
                        }
                }
        }
        cursor = commandEnd;
        pc += words;
    }
    if (barrierShifts != nullptr)
        for (const auto& [ready, eventTag] : barriers) {
            const auto first = firstReadEndByEvent.find(eventTag);
            if (first != firstReadEndByEvent.end()
                && ready > first->second)
                (*barrierShifts)[eventTag] = std::max(
                    (*barrierShifts)[eventTag], ready - first->second);
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
        std::size_t>>* insertedBarriers = nullptr,
    bool preserveIssuePhases = false)
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
    const auto mxmOutputTails = find_mxm_output_tails(program, barriers);
    MxmOutputTailsByEvent mxmOutputTailsByEvent;
    for (const auto& tail : mxmOutputTails)
        mxmOutputTailsByEvent[tail.event_tag].push_back(tail);
    std::vector<MxmOutputTilePair> mxmOutputTilePairs;
    for (const QueueProgram& queue : program.queues) {
        if (queue.kind != QueueKind::MxmCompute) continue;
        auto queueBarriers = timedBarriers;
        defer_mxm_barriers_past_output_tile_pairs(
            queue, queueBarriers, &mxmOutputTilePairs);
    }
    for (QueueProgram& queue : program.queues) {
        if (queue.kind == QueueKind::C2cDma
            || queue.kind == QueueKind::C2cRx
            || queue.kind == QueueKind::C2cTx)
            continue;

        auto queueBarriers = timedBarriers;
        std::vector<std::size_t> earlyJoinTags;
        if (queue.kind == QueueKind::MxmCompute) {
            defer_mxm_barriers_past_output_tile_pairs(queue, queueBarriers);
            std::ranges::sort(queueBarriers,
                [](const auto& lhs, const auto& rhs) {
                    return std::tie(lhs.first, lhs.second)
                        < std::tie(rhs.first, rhs.second);
                });
        }
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
            defer_mem_barriers_past_mxm_output_tails(
                queue, program.hardware, mxmOutputTailsByEvent, queueBarriers);
            align_mem_barriers_at_mxm_activation_wave_boundary(
                queue, program.hardware, mxmOutputTilePairs, queueBarriers);
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
        std::size_t originalCursor = 0;
        std::size_t barrierIndex = 0;
        std::vector<std::size_t> pendingPhaseIndices;
        const auto preserveNextIssue = [&](std::size_t originalIssue,
                                           std::size_t linkedIssue) {
            if (!preserveIssuePhases || insertedBarriers == nullptr) return;
            for (const auto index : pendingPhaseIndices) {
                auto& phase = std::get<3>((*insertedBarriers)[index]);
                const auto adjusted = static_cast<std::int64_t>(phase)
                    + static_cast<std::int64_t>(originalIssue)
                    - static_cast<std::int64_t>(linkedIssue);
                if (adjusted < 0)
                    throw std::logic_error("page wait has no nonnegative logical issue phase");
                phase = static_cast<std::size_t>(adjusted);
            }
            pendingPhaseIndices.clear();
        };

        // actualCycle is the logical consumer phase, even for an early join
        // whose queue reaches WAIT_EVENT before that logical cycle. Release
        // waits for the compute participants and preserves this phase.
        const auto appendBarrier = [&](const auto& barrier,
                                       std::size_t actualCycle) {
            append_control(linked,
                IcuControlInstruction::WaitEvent(barrier.second));
            if (insertedBarriers != nullptr) {
                insertedBarriers->emplace_back(
                    barrier.second, queue.kind, queue.index,
                    actualCycle);
                pendingPhaseIndices.push_back(insertedBarriers->size() - 1);
            }
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
                originalCursor += reservation;
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
                const auto idleEnd = originalCursor + *cycles;
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
                originalCursor = idleEnd;
                ++commandIndex;
                continue;
            }
            if (is_icu_control_raw_word_command(command)) {
                appendDueBarriers(cursor);
                oldToNew[commandIndex] = linked.size();
                linked.push_back(command);
                ++cursor;
                ++originalCursor;
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
                const auto commandBase = originalCursor;
                const auto idleEnd = commandBase + wait;
                while (barrierIndex < queueBarriers.size()
                    && queueBarriers[barrierIndex].first <= idleEnd) {
                    const auto barrierCycle = std::max(
                        cursor, queueBarriers[barrierIndex].first);
                    if (std::ranges::find(earlyJoinTags,
                            queueBarriers[barrierIndex].second) == earlyJoinTags.end())
                        append_nop(linked, barrierCycle - cursor);
                    cursor = barrierCycle;
                    // Release phases describe the linked ICU image. Previous
                    // zero-wait barriers may have added issue cycles; removing
                    // them here would release a consumer ahead of its MEM
                    // activation producers after a later dynamic page wait.
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
                // The injected control word can consume idle time or add a
                // cycle to a zero-wait queue. Release relative to the first
                // original FU issue, not that altered local cursor. This
                // preserves the compiler's producer/consumer phase for the
                // entire following interval, including MEM/VXM writeback.
                preserveNextIssue(idleEnd + loop.start_cycle,
                    cursor + remainingWait + loop.start_cycle);
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
                originalCursor = commandBase + duration;
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
                originalCursor = final + 1;
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
                ++originalCursor;
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

C2cWeightPager::C2cWeightPager(C2cDmaSystem& system)
    : system_(system)
{
}

void C2cWeightPager::begin_schedule(BinaryProgram& program)
{
    common_barrier_protocol_ = program.common_barrier_protocol;
    if (!common_barrier_protocol_ && std::ranges::any_of(program.queues,
            [](const auto& queue) { return queue.requires_raw_macro; }))
        throw std::logic_error("Raw Macro paging requires the common barrier protocol capability");
    if (active_ && !ready())
        throw std::logic_error(
            "cannot schedule executable pages while a page is in flight");
    if (active_) retire();
    if (common_barrier_protocol_) {
        const bool hadLinkedPlan = !program.barriers.empty();
        program.barriers.clear();
        program.barrier_events.clear();
        for (auto& use : program.weight_page_uses)
            use.transport_launch_event_id = 0;
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
    for (auto& windows : mem_idle_windows_) windows.clear();
    for (auto& windows : linked_mem_windows_) windows.clear();
    linked_page_barriers_.clear();
    linked_page_releases_.clear();
    linked_preserves_issue_phases_ = false;
    linked_activation_barrier_shifts_.clear();
    linked_mxm_rows_ = program.hardware.mxm_rows;
    linked_mxm_pipeline_rows_ = program.hardware.mxm_pipeline_rows;
    linked_system_register_columns_ = program.hardware.system_register_columns;
    linked_mem_slices_per_group_ =
        program.hardware.mem_slices_per_register_group;
    linked_mxms_per_hemisphere_ = program.hardware.mxms_per_hemisphere;
    common_events_.clear();
    common_page_ranges_.clear();
    if (common_barrier_protocol_) {
        for (const auto& queue : program.queues)
            if ((queue.kind == QueueKind::C2cDma || queue.kind == QueueKind::C2cRx)
                && !queue.commands.empty())
                throw std::logic_error("common paging linker requires ownership of the C2C transport queues");
        return;
    }

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
        throw std::logic_error("unsafe common stop: transport launch or SRAM reuse depends on computation after the stop");
    validate_common_drain_window(binary, stop, ingressBegin);
    for (const auto index : pageUseIndices) {
        const auto& use = binary.weight_page_uses[index];
        for (const auto& segment : page.segments) {
            if (common_page_ranges_.size() == kBinaryCommonBarrierMaxCompletionInstances)
                throw std::logic_error("common storage proof exceeds 65535 actual MEM segments");
            common_page_ranges_.push_back({InstructionControlUnit::mem_queue(
                segment.hemisphere, segment.slice, segment.bank), segment.base_row,
                segment.vector_count, use.binding_index, static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(stop), use.ready_cycle, use.release_cycle});
        }
    }
    // First implementation deliberately performs runtime page traffic while
    // held at the drained stop. Thus an arbitrarily late DDR response cannot
    // collide with computation or borrow an SR past its compiler reservation.
    // Same-C pages are serialized physically and joined by one all-of event.
    const auto producerStart = stop;
    const auto lowered = lower_c2c_weight_page_to_icu(page,
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
        association.page_use_indices.push_back(static_cast<std::uint32_t>(index));
    }
    common_events_.push_back(std::move(association));
    return fence;
}

void C2cWeightPager::inject_page_ready_barriers(
    BinaryProgram& program) const
{
    if (common_barrier_protocol_) {
        program.common_barrier_protocol = true;
        program.barriers.clear();
        for (const auto& barrier : linked_page_barriers_)
            program.barriers.push_back(make_binary_barrier(barrier.ready_cycle,
                static_cast<std::uint32_t>(barrier.event_tag)));
        program.barrier_events = common_events_;
        validate_binary_barriers(program);
        return;
    }
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
    if (common_barrier_protocol_) return {};
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
                release.activation_producer,
                release.result_writer});
    if (linked_preserves_issue_phases_) {
        std::ranges::stable_sort(releases, [](const auto& lhs, const auto& rhs) {
            return lhs.phase_offset < rhs.phase_offset;
        });
        return releases;
    }
    std::optional<std::size_t> firstMxmCompute;
    for (const PageReadyRelease& release : releases) {
        if (release.location.kind == IcuLocationKind::MxmCompute)
            firstMxmCompute = std::min(
                firstMxmCompute.value_or(release.phase_offset),
                release.phase_offset);
    }
    const auto shifted = linked_activation_barrier_shifts_.find(eventTag);
    std::optional<std::size_t> firstGroupArrival;
    for (const auto& release : linked_page_releases_)
        if (release.event_tag == eventTag
            && release.activation_first_read_delay != 0) {
            const auto tokenLane = release.location.index / 2;
            const auto arrival = release.barrier_cycle - referenceCycle
                + release.activation_first_read_delay
                + std::max<std::size_t>(2 + tokenLane,
                    linked_system_register_columns_) - 2 - tokenLane;
            firstGroupArrival = std::max(
                firstGroupArrival.value_or(arrival), arrival);
        }
    if (firstMxmCompute && (firstGroupArrival
            || shifted != linked_activation_barrier_shifts_.end())) {
        std::optional<std::size_t> lastActivationProducer;
        for (const auto& release : releases)
            if (release.activation_producer)
                lastActivationProducer = std::max(
                    lastActivationProducer.value_or(release.phase_offset),
                    release.phase_offset);
        // Group-zero MEM reads inject the first activation row at the far
        // end of the SR. Account for the wait-to-first-read interval and
        // its physical trip to the MXM ingress. Later MEM groups produce
        // later rows nearer the MXM, so their release time is not the
        // first-row arrival time. This also removes a one-cycle page-to-page
        // drift when the compiler places adjacent waits one cycle apart.
        std::size_t desiredComputePhase = *firstMxmCompute;
        if (firstGroupArrival) {
            // WAIT_EVENT retires on the preceding issue cycle. The first
            // activation reaches the MXM one cycle after the linked SR
            // arrival boundary. Anchor each page to that arrival even when
            // the compiler's nominal MXM wait is one cycle later.
            std::size_t firstIssueDelay = 1;
            for (const auto& linked : linked_page_releases_)
                if (linked.event_tag == eventTag
                    && linked.location.kind == IcuLocationKind::MxmCompute
                    && linked.barrier_cycle - referenceCycle == *firstMxmCompute)
                    firstIssueDelay = std::max(firstIssueDelay, linked.compute_first_issue_delay);
            // Arrival anchors the first FU issue, not the ICU release. The
            // domain may still carry wait_cycle after WAIT_EVENT retires.
            desiredComputePhase = *firstGroupArrival + 1 >= firstIssueDelay - 1
                ? *firstGroupArrival + 1 - (firstIssueDelay - 1) : 0;
        } else if (lastActivationProducer) {
            const auto producerBound = *lastActivationProducer
                + linked_mxm_pipeline_rows_ + 1;
            if (producerBound > desiredComputePhase)
                desiredComputePhase += std::min(shifted->second,
                    producerBound - desiredComputePhase);
        }
        // Anchor only the activation consumer. Unrelated MEM/VXM queues
        // retain their compiler timing; shifting one side of a later VXM
        // writeback would accumulate a one-cycle skew at each page join.
        // MXM result writers are aligned separately below.
        for (auto& release : releases)
            if (release.location.kind == IcuLocationKind::MxmCompute) {
                if (desiredComputePhase >= *firstMxmCompute)
                    release.phase_offset += desiredComputePhase
                        - *firstMxmCompute;
                else
                    release.phase_offset -= *firstMxmCompute
                        - desiredComputePhase;
            }
        // The MEM WAIT_EVENT retires before its following Read3D can inject
        // into the SR, while the resumed MXM domain observes the SR at the
        // start of its cycle. Keep the relocated producer wave one physical
        // cycle behind the linked issue-boundary phase so the last activation
        // vector is still present for the final MXM row.
        for (auto& release : releases)
            if (release.activation_producer)
                ++release.phase_offset;
    }
    // Resolve writers after the activation wave has anchored MXM's release.
    // A writer can precede its producer's release when it carries a long
    // local wait. Preserve that signed offset, then shift the common origin
    // if necessary so no event needs to be signalled before the page join.
    std::vector<std::pair<std::size_t, std::int64_t>> writerPhases;
    std::int64_t minimumPhase = 0;
    for (const auto& linked : linked_page_releases_) {
        if (linked.event_tag != eventTag || !linked.result_release_offset)
            continue;
        const auto producer = std::ranges::find_if(releases,
            [&](const auto& release) {
                return release.location == IcuLocation::MxmCompute(
                    linked.result_producer_index);
            });
        const auto writer = std::ranges::find_if(releases,
            [&](const auto& release) {
                return release.location == linked.location;
            });
        if (producer == releases.end() || writer == releases.end()) continue;
        const auto phase = static_cast<std::int64_t>(producer->phase_offset)
            + *linked.result_release_offset;
        minimumPhase = std::min(minimumPhase, phase);
        writerPhases.emplace_back(std::distance(releases.begin(), writer), phase);
    }
    for (auto& release : releases)
        release.phase_offset += static_cast<std::size_t>(-minimumPhase);
    for (const auto& [index, phase] : writerPhases)
        releases[index].phase_offset = static_cast<std::size_t>(phase - minimumPhase);
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
        for (const auto& load : common_page_ranges_) {
            const std::array<std::uint32_t, 1> ownerBinding {load.binding_index};
            validate_common_page_storage(program, load.stop, load.queue, load.row_begin,
                load.row_count, ownerBinding, load.release, common_page_ranges_);
        }
        for (const auto& queue : program.transport_queues)
            if (queue.commands.size() > program.hardware.icu_mem_imem_depth)
                throw std::logic_error("common paging MEM transport control RAM capacity exceeded");
        linked_page_releases_.clear();
        schedule_open_ = false;
        return;
    }
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
    auto activationProducers = native4_launch_producers(
        program, pageBarriers);
    std::vector<MxmOutputTilePair> outputPairs;
    for (const QueueProgram& queue : program.queues) {
        if (queue.kind != QueueKind::MxmCompute) continue;
        auto shiftedBarriers = pageBarriers;
        defer_mxm_barriers_past_output_tile_pairs(
            queue, shiftedBarriers, &outputPairs);
    }
    for (const QueueProgram& queue : program.queues) {
        if (queue.kind != QueueKind::Mem) continue;
        auto shiftedBarriers = pageBarriers;
        align_mem_barriers_at_mxm_activation_wave_boundary(queue, program.hardware,
            outputPairs, shiftedBarriers, &activationProducers,
            &linked_activation_barrier_shifts_);
    }
    // An SR number is local to its physical column. Weight ingress can use
    // E0/E1 just like activation ingress, but must not receive the activation
    // release adjustment. This epoch's C2C weight receivers are known here.
    std::erase_if(activationProducers, [&](const auto& producer) {
        return std::ranges::any_of(memWriteWindows[producer.second],
            [&](const auto& window) {
                return std::get<2>(window) == producer.first;
            });
    });
    std::ranges::sort(activationProducers);
    activationProducers.erase(
        std::unique(activationProducers.begin(), activationProducers.end()),
        activationProducers.end());
    std::vector<std::tuple<std::size_t, QueueKind, std::size_t,
        std::size_t>> insertedBarriers;
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::MxmCompute) continue;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            if (!is_fu_3d_raw_packet_header(queue.commands[pc])) {
                ++pc;
                continue;
            }
            const auto words = fu_3d_raw_packet_word_count(queue.kind, queue.commands[pc]);
            isa::EncodedMxmComputeIcu3DPacket packet{};
            for (std::size_t word = 0; word < words; ++word)
                for (std::size_t lane = 0; lane < packet.kLanesPerWord; ++lane)
                    packet.words[word].lanes[lane] = queue.commands[pc + word].words[lane];
            const auto instruction = isa::decode_mxm_compute_icu_3d_instruction(packet);
            linked_preserves_issue_phases_ |=
                instruction.opcode == MxmComputeIcuOpcode::Compute3D
                && instruction.loop.counts[0] > 1
                && instruction.loop.counts[1] > 1
                && instruction.terminal_dimension == 2;
            pc += words;
        }
    }
    inject_page_ready_barriers_impl(
        program, pageBarriers, memWriteWindows, &insertedBarriers,
        linked_preserves_issue_phases_);
    const auto pageResultStreams = find_mxm_page_result_streams(program);
    linked_page_releases_.clear();
    linked_page_releases_.reserve(insertedBarriers.size());
    for (const auto& [eventTag, kind, queueIndex, barrierCycle] :
         insertedBarriers)
    {
        std::optional<std::int64_t> resultReleaseOffset;
        std::size_t resultProducerIndex = 0;
        bool resultWriter = false;
        if (kind == QueueKind::Mem) {
            const auto queue = std::ranges::find_if(program.queues,
                [&](const QueueProgram& candidate) {
                    return candidate.kind == kind
                        && candidate.index == queueIndex;
                });
            if (queue != program.queues.end())
                for (const MxmPageResultStream& result : pageResultStreams)
                    if (result.event_tag == eventTag
                        && result.hemisphere == queueIndex
                            / InstructionControlUnit::
                                kMemQueuesPerHemisphere) {
                        std::size_t contiguousWriteCount = 0;
                        const auto writeDelay = mem_result_write_delay_after_page_wait(
                            *queue, eventTag, result.stream_base,
                            program.hardware.mxm_result_streams, &contiguousWriteCount);
                        if (!writeDelay) continue;
                        const auto slice = (queueIndex
                            % InstructionControlUnit::kMemQueuesPerHemisphere)
                            / hw::kMemBanksPerSlice;
                        const auto group = slice
                            / program.hardware.mem_slices_per_register_group;
                        const auto latency = program.hardware.mxm_pipeline_rows - 1
                            + program.hardware.system_register_columns - 1 - group;
                        const auto producerBarrier = std::ranges::find_if(
                            insertedBarriers, [&](const auto& barrier) {
                                return std::get<0>(barrier) == eventTag
                                    && std::get<1>(barrier) == QueueKind::MxmCompute
                                    && std::get<2>(barrier) == result.queue_index;
                            });
                        // SR IDs are reused by later operators. A write after
                        // this compute packet's final result has passed the
                        // MEM column cannot be one of its result consumers.
                        // Compare in the common nominal schedule, since page
                        // waits may have moved to different queue boundaries.
                        if (producerBarrier == insertedBarriers.end()
                            || barrierCycle + *writeDelay
                                > std::get<3>(*producerBarrier)
                                    + result.last_compute_delay + latency)
                            continue;
                        // A consumer may intentionally write only a suffix of
                        // this result tile. Preserve its token offset instead
                        // of moving the write to the first streamed token.
                        std::size_t resultOffset = 0;
                        const auto nominalFirstResult = std::get<3>(*producerBarrier)
                            + result.first_result_delay + latency;
                        const auto nominalFirstWrite = barrierCycle + *writeDelay;
                        if (contiguousWriteCount > 0
                            && contiguousWriteCount < result.contiguous_result_count
                            && nominalFirstWrite >= nominalFirstResult
                            && nominalFirstWrite - nominalFirstResult
                                <= result.contiguous_result_count - contiguousWriteCount)
                            resultOffset = nominalFirstWrite - nominalFirstResult;
                        // Both delays include retirement of the page wait.
                        // Match the first MEM write to the first streamed
                        // result at this physical MEM column. This remains
                        // valid when an earlier page inserted an extra wait
                        // into only one of the two queues.
                        resultReleaseOffset = static_cast<std::int64_t>(
                            result.first_result_delay + latency + resultOffset)
                            - static_cast<std::int64_t>(*writeDelay);
                        resultProducerIndex = result.queue_index;
                        break;
                    }
            resultWriter = resultReleaseOffset.has_value();
        }
        const bool activationProducer = kind == QueueKind::Mem
            && std::ranges::find(activationProducers,
                std::pair {eventTag, queueIndex})
                != activationProducers.end();
        std::size_t activationFirstReadDelay = 0;
        if (activationProducer) {
            const auto slice = (queueIndex
                % InstructionControlUnit::kMemQueuesPerHemisphere)
                / hw::kMemBanksPerSlice;
            if (slice < program.hardware.mem_slices_per_register_group) {
                const auto hemisphere = queueIndex
                    / InstructionControlUnit::kMemQueuesPerHemisphere;
                const auto pair = std::ranges::find_if(outputPairs,
                    [&](const MxmOutputTilePair& candidate) {
                        return candidate.event_tag == eventTag
                            && candidate.queue_index
                                / program.hardware.mxms_per_hemisphere
                                == hemisphere;
                    });
                const auto queue = std::ranges::find_if(program.queues,
                    [&](const QueueProgram& candidate) {
                        return candidate.kind == QueueKind::Mem
                            && candidate.index == queueIndex;
                    });
                const auto result = std::ranges::find_if(pageResultStreams,
                    [&](const MxmPageResultStream& candidate) {
                        return candidate.event_tag == eventTag
                            && candidate.hemisphere == hemisphere;
                    });
                if (queue != program.queues.end()
                    && (pair != outputPairs.end()
                        || (result != pageResultStreams.end()
                            && result->first_compute_delay == 1)))
                    // A non-relocated projection with an explicit startup
                    // wait already has compiler-timed weight/activation
                    // ingress. Only re-anchor an immediate consumer or a
                    // wave whose barrier was moved past an output tile pair.
                    activationFirstReadDelay =
                        mem_activation_first_read_delay_after_page_wait(
                            *queue, eventTag,
                            pair != outputPairs.end()
                                ? pair->activation_stream_base
                                : result->activation_stream_base);
            }
        }
        std::size_t computeFirstIssueDelay = 1;
        if (kind == QueueKind::MxmCompute)
            for (const auto& result : pageResultStreams)
                if (result.event_tag == eventTag && result.queue_index == queueIndex)
                    computeFirstIssueDelay = result.first_compute_delay;
        linked_page_releases_.push_back(LinkedPageRelease {
            eventTag,
            linked_queue_location(kind, queueIndex,
                program.hardware.mxms_per_hemisphere),
            barrierCycle,
            activationProducer,
            resultWriter,
            activationFirstReadDelay,
            resultReleaseOffset,
            resultProducerIndex,
            computeFirstIssueDelay});
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
        if ((fence.transport_queue ? chip.icu().mem_transport_iq(queue)
                                 : chip.icu().mem_iq(queue)).synchronized_issued_count()
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
        if ((fence.transport_queue ? chip.icu().mem_transport_iq(queue)
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
