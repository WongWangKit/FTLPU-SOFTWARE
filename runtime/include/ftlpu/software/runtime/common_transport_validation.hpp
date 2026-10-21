#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/common_page_range.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"

#include <algorithm>
#include <span>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace ftlpu::software::runtime {

// A drained pipeline does not prove SRAM liveness. Conservatively reject a
// page write if a later ordinary read can still observe the overwritten rows.
// A read must consume this weight, or follow a proved complete replacement
// after the weight's release. Bounds can reject legal but unproved aliasing;
// v1 never assumes that a partial/future write repairs a live operand.
inline void validate_common_page_storage(const BinaryProgram& program,
    std::uint32_t stop, std::size_t memQueue, std::uint32_t rowBegin,
    std::uint32_t rowCount, std::span<const std::uint32_t> weightBindings,
    std::uint64_t weightRelease, std::span<const CommonPageLoadRange> pageLoads = {})
{
    const auto rowEnd = static_cast<std::uint64_t>(rowBegin) + rowCount;
    if (rowCount == 0 || rowEnd > program.hardware.sram_depth_rows)
        throw std::invalid_argument("common page SRAM range is invalid");
    const auto bound = [](std::uint64_t start, std::size_t count, std::size_t stride) {
        if (count == 0 || stride == 0 || count - 1 > (std::numeric_limits<std::uint64_t>::max() - start) / stride)
            throw std::invalid_argument("common storage proof issue domain overflows");
        return start + (count - 1) * stride;
    };
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::Mem || queue.index != memQueue) continue;
        struct Access {
            std::int64_t low, high;
            std::uint64_t first, last;
            bool write, dense, weight;
            std::optional<std::uint32_t> input_binding;
        };
        std::vector<Access> accesses;
        std::uint64_t cursor = 0;
        bool sawMacro = false, sawNative = false;
        for (std::size_t pc = 0; pc < queue.commands.size(); ++pc) {
            const auto& command = queue.commands[pc];
            const bool macro = is_macro_schedule_command(command);
            if (macro) sawMacro = true; else sawNative = true;
            // Macro coordinates are absolute; Native PC/NOP time is relative.
            // Their compatibility lowering has a different cursor, so v1 does
            // not use this proof for a queue that combines both encodings.
            if (sawMacro && sawNative)
                throw std::invalid_argument("common storage proof must not mix Macro and Native MEM commands in one queue");
            const auto opcode = isa::decode_icu_command_opcode(command.command);
            if (opcode == isa::IcuCommandOpcode::Nop) {
                const auto count = is_icu_control_raw_word_command(command)
                    ? decode_icu_control_raw_word(command).count : isa::decode_icu_nop_cycles(command.command);
                if (count > std::numeric_limits<std::uint64_t>::max() - cursor)
                    throw std::invalid_argument("common storage proof Native time overflows");
                cursor += count;
                continue;
            }
            if ((!macro && opcode != isa::IcuCommandOpcode::Instruction)
                || command.instruction_kind != InstructionKind::Mem || command.word_count > 2
                || command.word_count == 0)
                throw std::invalid_argument("common storage proof requires affine Macro or simple Native MEM domains");
            const auto bits = static_cast<isa::EncodedMemInstruction>(command.words[0])
                | (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
            const auto instruction = isa::decode_mem_instruction(bits);
            if (instruction.opcode != MemOpcode::Read && instruction.opcode != MemOpcode::Write)
                throw std::invalid_argument("common storage proof does not support indirect gather/scatter addresses");
            std::int64_t low = instruction.address, high = instruction.address;
            auto first = cursor, last = cursor;
            bool dense = true;
            if (macro) {
                const auto schedule = decode_macro_schedule_command(command);
                first = schedule.start_cycle;
                last = bound(bound(schedule.start_cycle, schedule.inner_count, schedule.inner_interval),
                    schedule.outer_count, schedule.outer_interval);
                if (schedule.induction_target != IcuInductionTarget::None
                    && schedule.induction_target != IcuInductionTarget::MemAddress)
                    throw std::invalid_argument("common storage proof has an unsupported MEM induction target");
                const auto addRows = [&](std::size_t count, std::int64_t stride) {
                    const auto magnitude = stride < 0 ? static_cast<std::uint64_t>(-(stride + 1)) + 1
                        : static_cast<std::uint64_t>(stride);
                    if (magnitude && count - 1 > program.hardware.sram_depth_rows / magnitude)
                        throw std::invalid_argument("common storage proof SRAM induction is out of bounds");
                    const auto delta = static_cast<std::int64_t>((count - 1) * magnitude);
                    if (stride < 0) low -= delta; else high += delta;
                };
                if (schedule.induction_target == IcuInductionTarget::MemAddress) {
                    addRows(schedule.inner_count, schedule.inner_stride);
                    addRows(schedule.outer_count, schedule.outer_stride);
                    // Exact coverage of the affine address envelope is proved
                    // only for contiguous unions. Strides of zero repeat a
                    // row; a nonzero stride may not leave a hole in coverage.
                    std::array<std::pair<std::uint64_t, std::size_t>, 2> axes {{{
                        static_cast<std::uint64_t>(std::abs(schedule.inner_stride)), schedule.inner_count}, {
                        static_cast<std::uint64_t>(std::abs(schedule.outer_stride)), schedule.outer_count}}};
                    std::ranges::sort(axes);
                    std::uint64_t covered = 1;
                    for (const auto& [stride, count] : axes) {
                        if (count <= 1 || stride == 0) continue;
                        if (stride > covered) dense = false;
                        covered += (count - 1) * stride;
                    }
                }
            } else {
                cursor = bound(cursor, 2, 1);
            }
            if (low < 0 || high >= program.hardware.sram_depth_rows)
                throw std::invalid_argument("common storage proof MEM address wraps SRAM");
            if (last < stop || high < rowBegin || static_cast<std::uint64_t>(low) >= rowEnd) continue;
            const bool incomingWeightRead = std::ranges::any_of(program.address_relocations,
                [&](const auto& relocation) {
                    return !relocation.transport_queue && !relocation.write_port
                        && relocation.queue_kind == QueueKind::Mem && relocation.queue_index == queue.index
                        && relocation.command_index == pc && relocation.binding_access == BindingAccess::Input
                        && std::ranges::find(weightBindings, relocation.binding_index) != weightBindings.end();
                });
            std::optional<std::uint32_t> inputBinding;
            for (const auto& relocation : program.address_relocations)
                if (!relocation.transport_queue && !relocation.write_port
                    && relocation.queue_kind == QueueKind::Mem && relocation.queue_index == queue.index
                    && relocation.command_index == pc && relocation.binding_access == BindingAccess::Input) {
                    if (inputBinding && *inputBinding != relocation.binding_index)
                        throw std::invalid_argument("common storage proof has ambiguous input relocation");
                    inputBinding = relocation.binding_index;
                }
            accesses.push_back({low, high, first, last, instruction.opcode == MemOpcode::Write,
                dense, incomingWeightRead, inputBinding});
        }
        for (const auto& access : accesses) {
            if (access.write) {
                if (access.first < weightRelease)
                    throw std::invalid_argument("unsafe common stop: computation writes the incoming page before its release: queue="
                        + std::to_string(memQueue) + " stop=" + std::to_string(stop)
                        + " write=" + std::to_string(access.first) + " release=" + std::to_string(weightRelease));
                continue;
            }
            if (access.weight) continue;
            const auto low = std::max<std::int64_t>(rowBegin, access.low);
            const auto high = std::min<std::int64_t>(rowEnd - 1, access.high);
            const auto replacement = std::ranges::find_if(accesses, [&](const auto& write) {
                return write.write && write.dense && write.first >= weightRelease
                    && write.low <= low && write.high >= high
                    && write.last < access.first
                    && access.first - write.last >= program.hardware.tile_rows;
            });
            // A later linked page-in can also replace released storage. Its
            // exact packed segments and its own common barrier guarantee SRAM
            // commit before the reader; DDR completion alone never qualifies.
            bool replacedByPage = false;
            if (access.input_binding) {
                std::vector<std::pair<std::uint64_t, std::uint64_t>> covered;
                for (const auto& load : pageLoads)
                    if (load.queue == memQueue && load.binding_index == *access.input_binding
                        && load.stop >= weightRelease && load.stop <= access.first
                        && load.ready <= access.first && load.release > access.last)
                        covered.emplace_back(load.row_begin,
                            static_cast<std::uint64_t>(load.row_begin) + load.row_count);
                std::ranges::sort(covered);
                auto next = static_cast<std::uint64_t>(low);
                for (const auto& [begin, end] : covered) {
                    if (begin > next) break;
                    next = std::max(next, end);
                }
                replacedByPage = next > static_cast<std::uint64_t>(high);
            }
            if (replacement == accesses.end() && !replacedByPage)
                throw std::invalid_argument("unsafe common stop: page write may overwrite SRAM needed by a future computation read: queue="
                    + std::to_string(memQueue) + " stop=" + std::to_string(stop)
                    + " read=" + std::to_string(access.first) + " rows=" + std::to_string(low) + ".." + std::to_string(high));
        }
    }
}

// Validate one concrete MEM/RX/DMA triple for every page-in segment. A match
// by SRAM row/count alone is insufficient for repeated loads into ping-pong
// storage: the tag and launch domain identify the actual transport instance.
inline std::map<std::pair<std::size_t, std::size_t>, std::uint32_t>
validate_common_transport_packet_links(const BinaryProgram& program)
{
    using Location = std::pair<std::size_t, std::size_t>;
    std::map<Location, std::uint32_t> result;
    if (!program.common_barrier_protocol) return result;
    validate_binary_barriers(program);

    const auto fail = [](const char* reason) {
        throw std::invalid_argument(reason);
    };
    std::set<std::uint32_t> launches;
    for (const auto& event : program.barrier_events)
        for (const auto instance : event.page_use_indices)
            launches.insert(program.weight_page_uses.at(instance).transport_launch_event_id);

    struct MemSegment {
        Location location{};
        std::size_t count{0}, delay{0};
        MemInstruction instruction{};
    };
    struct RxSegment {
        std::uint32_t launch{0};
        C2cRxIcuInstruction instruction{};
    };
    struct DmaSegment {
        std::uint32_t launch{0};
        C2cDmaIcuInstruction instruction{};
    };
    std::map<std::uint32_t, MemSegment> mems;
    std::map<std::uint32_t, RxSegment> receives;
    std::map<std::uint32_t, DmaSegment> dmas;
    std::set<std::size_t> mem_queues;

    const auto bits = [](const auto& word, unsigned begin, unsigned width) {
        std::uint64_t value = 0;
        for (unsigned bit = 0; bit < width; ++bit)
            value |= static_cast<std::uint64_t>(
                (word.lanes[(begin + bit) / 32] >> ((begin + bit) % 32)) & 1u) << bit;
        return value;
    };

    for (const auto& queue : program.transport_queues) {
        if (queue.kind != QueueKind::Mem || queue.packed_macro_imem
            || queue.requires_raw_macro || queue.index >= InstructionControlUnit::kMemQueues
            || !mem_queues.insert(queue.index).second)
            fail("common transport contains an invalid or duplicate MEM control queue");
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            if (is_icu_control_raw_word_command(queue.commands[pc])) {
                const auto control = decode_icu_control_raw_word(queue.commands[pc]);
                if (control.opcode != IcuControlOpcode::Nop)
                    fail("common page-in MEM control may only contain NOPs and WRITE_SYNC packets");
                ++pc;
                continue;
            }
            if (!is_mem_synchronized_raw_packet_header(queue.commands[pc]))
                fail("common transport contains an orphan or unsupported MEM packet word");
            const auto packet = decode_mem_synchronized_icu_packet(queue, pc);
            const auto tag = static_cast<std::uint32_t>(bits(packet[0], 18, 16));
            const auto count = static_cast<std::size_t>(bits(packet[0], 2, 16)) + 1;
            const auto delay = static_cast<std::size_t>(bits(packet[0], 34, 16));
            if (bits(packet[0], 50, 14) != 1)
                fail("common MEM page-in address stride must be exactly one SRAM row");
            const auto native = static_cast<isa::EncodedMemInstruction>(bits(packet[1], 2, 64));
            const auto instruction = isa::decode_mem_instruction(native);
            if (instruction.opcode != MemOpcode::Write || instruction.preserve_stream)
                fail("common MEM page-in must consume a canonical WRITE stream");
            if (tag == 0 || !mems.emplace(tag, MemSegment{{queue.index, pc}, count, delay, instruction}).second)
                fail("common MEM page-in synchronization tag is zero or repeated");
            if (count > program.hardware.sram_depth_rows
                || instruction.address > program.hardware.sram_depth_rows - count)
                fail("common MEM page-in row range exceeds visible SRAM");
            if (mems.size() > kBinaryCommonBarrierMaxCompletionInstances)
                fail("common transport segment capacity exceeds 65535");
            pc += InstructionControlUnit::MemIcu::synchronized_packet_word_count;
        }
    }

    std::set<std::pair<QueueKind, std::size_t>> control_queues;
    for (const auto& queue : program.queues) {
        if (queue.kind != QueueKind::C2cDma && queue.kind != QueueKind::C2cRx) continue;
        if (queue.index >= hw::kHemispheres || queue.packed_macro_imem
            || queue.requires_raw_macro
            || !control_queues.emplace(queue.kind, queue.index).second)
            fail("common transport contains an invalid or duplicate DMA/RX queue");
        std::optional<std::uint32_t> launch;
        std::set<std::uint32_t> queue_launches;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            const auto& command = queue.commands[pc];
            if (is_icu_control_raw_word_command(command)) {
                const auto control = decode_icu_control_raw_word(command);
                if (control.opcode == IcuControlOpcode::WaitEvent) {
                    if (control.event_tag > std::numeric_limits<std::uint32_t>::max()
                        || !launches.contains(static_cast<std::uint32_t>(control.event_tag))
                        || !queue_launches.insert(static_cast<std::uint32_t>(control.event_tag)).second)
                        fail("common DMA/RX launch WAIT is orphaned or repeated");
                    launch = static_cast<std::uint32_t>(control.event_tag);
                } else if (control.opcode == IcuControlOpcode::Sync) {
                    if (queue.kind != QueueKind::C2cDma || !launch)
                        fail("common completion SYNC is outside a DMA launch domain");
                } else if (control.opcode != IcuControlOpcode::Nop) {
                    fail("common DMA/RX queue contains unsupported transport control");
                }
                ++pc;
                continue;
            }
            if (!launch)
                fail("common DMA/RX packet is not guarded by a linked launch WAIT");
            if (queue.kind == QueueKind::C2cRx) {
                const auto rx = C2cIcuPacketCodec::decode_rx(
                    decode_c2c_raw_word(command, InstructionKind::C2cEndpoint));
                if (hemisphere_index(rx.endpoint_hemisphere) != queue.index
                    || rx.lane >= program.hardware.c2c_streams_per_direction
                    || !rx.notify.enabled || rx.notify.hemisphere != rx.endpoint_hemisphere
                    || rx.sync_tag == 0
                    || !receives.emplace(rx.sync_tag, RxSegment{*launch, rx}).second)
                    fail("common RX route, lane or unique synchronization tag is invalid");
                ++pc;
            } else {
                const auto dma = C2cIcuPacketCodec::decode_dma(decode_c2c_dma_icu_packet(queue, pc));
                if (hemisphere_index(dma.endpoint_hemisphere) != queue.index
                    || dma.lane >= program.hardware.c2c_streams_per_direction
                    || dma.direction != C2cDmaDirection::Ddr4ToC2c
                    || dma.address_stride_bytes != hw::kPhysicalVectorBytes
                    || dma.sync_tag == 0
                    || !dmas.emplace(dma.sync_tag, DmaSegment{*launch, dma}).second)
                    fail("common DMA must be a uniquely tagged physical-stride page-in load");
                pc += C2cDmaIcuPacket::kWordCount;
            }
        }
    }
    if (mems.size() != receives.size() || mems.size() != dmas.size())
        fail("common transport contains an orphan MEM, RX or DMA segment");

    std::set<std::uint32_t> linked_launches;
    for (const auto& [tag, mem] : mems) {
        const auto rx_it = receives.find(tag);
        const auto dma_it = dmas.find(tag);
        if (rx_it == receives.end() || dma_it == dmas.end())
            fail("common MEM synchronization tag has no matching RX/DMA segment");
        const auto& rx = rx_it->second.instruction;
        const auto& dma = dma_it->second.instruction;
        const auto queue = InstructionControlUnit::mem_queue(
            rx.notify.hemisphere, rx.notify.mem_slice, rx.notify.mem_bank);
        const auto group = rx.notify.mem_slice / hw::kMemSlicesPerGroup;
        const auto expected_delay = hw::kMemEastBoundaryStreamRegisterColumn - (group + 1);
        if (rx_it->second.launch != dma_it->second.launch
            || rx.endpoint_hemisphere != dma.endpoint_hemisphere || rx.lane != dma.lane
            || rx.vector_count != dma.vector_count || mem.count != rx.vector_count
            || mem.location.first != queue || mem.delay != expected_delay
            || mem.instruction.stream != StreamId::West(rx.fabric_stream).packed())
            fail("common MEM/RX/DMA launch, route, stream, count or physical delay mismatch");
        result.emplace(mem.location, rx_it->second.launch);
        linked_launches.insert(rx_it->second.launch);
    }
    if (linked_launches != launches)
        fail("common launch event has no fully linked physical transport segment");
    return result;
}

} // namespace ftlpu::software::runtime
