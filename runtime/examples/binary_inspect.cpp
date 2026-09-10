#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
try {
    if (argc < 2)
        throw std::runtime_error(
            "usage: ftlpu_binary_inspect program.ftlpu "
            "[--all-queues] [--trace schedule.csv] "
            "[--compare reference.ftlpu]");
    bool reportAllQueues = false;
    std::optional<std::filesystem::path> tracePath;
    std::optional<std::filesystem::path> comparePath;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--all-queues") {
            reportAllQueues = true;
            continue;
        }
        if (argument == "--trace" || argument == "--compare") {
            if (++index >= argc)
                throw std::runtime_error(std::string(argument)
                    + " requires a path");
            if (argument == "--trace")
                tracePath = std::filesystem::path(argv[index]);
            else
                comparePath = std::filesystem::path(argv[index]);
            continue;
        }
        throw std::runtime_error(
            "unknown binary inspector option: " + std::string(argument));
    }

    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    std::ifstream binaryVersionStream(argv[1], std::ios::binary);
    binaryVersionStream.seekg(8);
    std::uint32_t binaryVersion = 0;
    binaryVersionStream.read(reinterpret_cast<char*>(&binaryVersion),
        sizeof(binaryVersion));
    if (!binaryVersionStream)
        throw std::runtime_error("failed to read FTLPU binary version");
    const auto cycles = program.max_cycle + 64;
    std::cout << "binary target=" << program.target_name
              << " max_cycle=" << program.max_cycle
              << " measured_cycles=" << cycles
              << " mxms_per_hemisphere="
              << program.hardware.mxms_per_hemisphere
              << " vxm_alus=" << program.hardware.vxm_alus << '\n';
    std::vector<const ftlpu::software::runtime::QueueProgram*> queues;
    queues.reserve(program.queues.size());
    for (const auto& queue : program.queues) queues.push_back(&queue);
    std::ranges::sort(queues, [](const auto* lhs, const auto* rhs) {
        return lhs->commands.size() > rhs->commands.size();
    });
    std::size_t total_commands = 0;
    for (const auto& queue : program.queues)
        total_commands += queue.commands.size();
    std::cout << "binary queues=" << program.queues.size()
              << " commands=" << total_commands
              << " file_bytes=" << std::filesystem::file_size(argv[1])
              << '\n';
    const auto logicalIssues =
        ftlpu::software::runtime::inspect_logical_issues(program);
    std::cout << "binary logical_issue_summary queues="
              << logicalIssues.queues.size()
              << " functional_issues=" << logicalIssues.functional_issues
              << " logical_nop_cycles="
              << logicalIssues.logical_nop_cycles << '\n';
    std::size_t totalInstructions = 0;
    std::size_t totalNops = 0;
    std::size_t totalRepeats = 0;
    std::size_t totalRepeat2D = 0;
    std::size_t totalMacros = 0;
    std::size_t totalMemStreamNd = 0;
    std::size_t totalMemSlicePrograms = 0;
    std::size_t totalMemSliceBodyEntries = 0;
    std::size_t maxMemSliceBodyEntries = 0;
    std::size_t memSlicePayloadWords = 0;
    std::size_t equivalentMemStreamPayloadWords = 0;
    std::size_t totalMxmStreamNd = 0;
    std::size_t totalVxmStreamNd = 0;
    std::size_t totalSxmTilePrograms = 0;
    std::size_t expandedInstructions = 0;
    std::size_t repeatReplayed = 0;
    std::size_t repeat2DReplayed = 0;
    std::size_t macroExpanded = 0;
    std::size_t memStreamNdExpanded = 0;
    std::size_t memSliceProgramExpanded = 0;
    std::size_t mxmStreamNdExpanded = 0;
    std::size_t vxmStreamNdExpanded = 0;
    std::size_t sxmTileProgramExpanded = 0;
    std::size_t tracePatternRows = 0;
    std::size_t serializedQueueBytes =
        program.queues.size() * (binaryVersion >= 26 ? 9 : 8);
    for (const auto& queue : program.queues) {
        const auto macroCount = std::ranges::count_if(queue.commands,
            [](const auto& command) {
                return ftlpu::software::runtime::is_macro_schedule_command(
                    command);
            });
        const bool macroQueue = binaryVersion >= 26
            && macroCount == queue.commands.size()
            && !queue.commands.empty();
        const bool memDeltaQueue = binaryVersion >= 28 && macroQueue
            && queue.kind == ftlpu::software::runtime::QueueKind::Mem;
        const bool hasExtendedDescriptor = std::ranges::any_of(
            queue.commands, [](const auto& command) {
                return ftlpu::isa::decode_icu_command_opcode(
                           command.command)
                    == ftlpu::isa::IcuCommandOpcode::Extended;
            });
        const bool nativeQueue = binaryVersion >= 26
            && macroCount == 0 && !hasExtendedDescriptor;
        if (memDeltaQueue) {
            const auto image =
                ftlpu::software::runtime::encode_mem_macro_bitstream(queue);
            serializedQueueBytes += 10
                + image.delta_count * 8 + image.bytes.size();
        } else if (macroQueue) {
            serializedQueueBytes += (queue.commands.size() + 7) / 8;
        }
        for (const auto& command : queue.commands) {
            const bool macro =
                ftlpu::software::runtime::is_macro_schedule_command(command);
            if (memDeltaQueue) {
                // Accounted once as a packed queue-level bitstream above.
            } else if (macroQueue) {
                serializedQueueBytes +=
                    command.word_count * sizeof(std::uint32_t) + 29;
            } else if (nativeQueue) {
                if (ftlpu::software::runtime::is_repeat_2d_command(command))
                    serializedQueueBytes += 3 * sizeof(std::uint32_t);
                else if (command.instruction_kind
                            == ftlpu::software::runtime::InstructionKind::Sxm)
                    serializedQueueBytes += 14 * sizeof(std::uint32_t);
                else
                    serializedQueueBytes += sizeof(std::uint32_t)
                        + command.word_count * sizeof(std::uint32_t);
            } else {
                serializedQueueBytes += 1
                    + command.word_count * sizeof(std::uint32_t)
                    + (macro ? 29 : sizeof(std::uint32_t))
                    + (!macro && !command.extension_words.empty()
                            ? sizeof(std::uint16_t)
                                + command.extension_words.size()
                                    * sizeof(std::uint32_t)
                            : 0);
            }
            if (ftlpu::software::runtime::is_vxm_stream_nd_command(command)) {
                ++totalVxmStreamNd;
                const auto stream = ftlpu::software::runtime::
                    decode_vxm_stream_nd_command(command);
                std::size_t points = 1;
                for (std::size_t dimension = 0;
                     dimension < stream.schedule.rank; ++dimension)
                    points *= stream.schedule.counts[dimension];
                vxmStreamNdExpanded += points;
                expandedInstructions += points;
                tracePatternRows += stream.schedule.rank > 2
                    ? stream.schedule.counts[2] : 1;
                continue;
            }
            if (ftlpu::software::runtime::is_sxm_tile_program_command(
                    command)) {
                ++totalSxmTilePrograms;
                const auto tile = ftlpu::software::runtime::
                    decode_sxm_tile_program_command(command);
                std::size_t points = 1;
                for (std::size_t dimension = 0;
                     dimension < tile.schedule.rank; ++dimension)
                    points *= tile.schedule.counts[dimension];
                sxmTileProgramExpanded += points;
                expandedInstructions += points;
                tracePatternRows += tile.schedule.rank > 2
                    ? tile.schedule.counts[2] : 1;
                continue;
            }
            if (ftlpu::software::runtime::is_mem_slice_program_command(
                    command)) {
                ++totalMemSlicePrograms;
                const auto sliceProgram = ftlpu::software::runtime::
                    decode_mem_slice_program_command(command);
                totalMemSliceBodyEntries += sliceProgram.body.size();
                maxMemSliceBodyEntries = std::max(
                    maxMemSliceBodyEntries, sliceProgram.body.size());
                memSlicePayloadWords += command.extension_words.size();
                for (const auto& body : sliceProgram.body) {
                    const auto encoded = ftlpu::isa::encode_mem_instruction(
                        body.instruction);
                    equivalentMemStreamPayloadWords += 12
                        + ((encoded >> 32) == 0 ? 1 : 2);
                }
                std::size_t points = sliceProgram.body.size();
                for (std::size_t dimension = 0;
                     dimension < sliceProgram.schedule.rank; ++dimension)
                    points *= sliceProgram.schedule.counts[dimension];
                memSliceProgramExpanded += points;
                expandedInstructions += points;
                tracePatternRows += sliceProgram.body.size()
                    * (sliceProgram.schedule.rank > 2
                            ? sliceProgram.schedule.counts[2] : 1);
                continue;
            }
            if (ftlpu::software::runtime::is_mem_stream_nd_command(
                    command)) {
                ++totalMemStreamNd;
                const auto stream =
                    ftlpu::software::runtime::decode_mem_stream_nd_command(
                        command);
                std::size_t points = 1;
                for (std::size_t dimension = 0;
                     dimension < stream.rank; ++dimension)
                    points *= stream.counts[dimension];
                memStreamNdExpanded += points;
                expandedInstructions += points;
                tracePatternRows += stream.rank > 2
                    ? stream.counts[2] : 1;
                continue;
            }
            if (ftlpu::software::runtime::is_mxm_stream_nd_command(
                    command)) {
                ++totalMxmStreamNd;
                const auto stream =
                    ftlpu::software::runtime::decode_mxm_stream_nd_command(
                        command);
                std::size_t points = 1;
                for (std::size_t dimension = 0;
                     dimension < stream.rank; ++dimension)
                    points *= stream.counts[dimension];
                mxmStreamNdExpanded += points;
                expandedInstructions += points;
                tracePatternRows += stream.rank > 2
                    ? stream.counts[2] : 1;
                continue;
            }
            if (macro) {
                ++totalMacros;
                const auto macro =
                    ftlpu::software::runtime::decode_macro_schedule_command(
                        command);
                macroExpanded += macro.inner_count * macro.outer_count;
                expandedInstructions += macro.inner_count * macro.outer_count;
                ++tracePatternRows;
                continue;
            }
            if (ftlpu::software::runtime::is_repeat_2d_command(command)) {
                ++totalRepeat2D;
                const auto repeat =
                    ftlpu::software::runtime::decode_repeat_2d_command(
                        command);
                repeat2DReplayed +=
                    repeat.inner_count * repeat.outer_count - 1;
                expandedInstructions +=
                    repeat.inner_count * repeat.outer_count - 1;
                if (repeat.inner_count * repeat.outer_count > 1)
                    ++tracePatternRows;
                continue;
            }
            switch (ftlpu::isa::decode_icu_command_opcode(command.command)) {
            case ftlpu::isa::IcuCommandOpcode::Instruction:
                ++totalInstructions;
                ++expandedInstructions;
                ++tracePatternRows;
                break;
            case ftlpu::isa::IcuCommandOpcode::Nop:
                ++totalNops;
                break;
            case ftlpu::isa::IcuCommandOpcode::Repeat:
                ++totalRepeats;
                {
                    const auto repeat =
                        ftlpu::isa::decode_icu_repeat(command.command);
                    repeatReplayed += repeat.count;
                    expandedInstructions += repeat.count;
                    if (repeat.count != 0) ++tracePatternRows;
                }
                break;
            default:
                break;
            }
        }
    }
    const std::size_t encodedWorkEntries = totalInstructions + totalRepeats
        + totalRepeat2D + totalMacros + totalMemStreamNd
        + totalMemSlicePrograms + totalMxmStreamNd + totalVxmStreamNd
        + totalSxmTilePrograms;
    const std::size_t savedWorkEntries = expandedInstructions
        > encodedWorkEntries ? expandedInstructions - encodedWorkEntries : 0;
    std::cout << "binary aggregate instruction=" << totalInstructions
              << " nop=" << totalNops
              << " repeat=" << totalRepeats
              << " repeat2d=" << totalRepeat2D
              << " macro=" << totalMacros
              << " mem_stream_nd=" << totalMemStreamNd
              << " mem_slice_program=" << totalMemSlicePrograms
              << " mem_slice_body=" << totalMemSliceBodyEntries
              << " mxm_stream_nd=" << totalMxmStreamNd
              << " vxm_stream_nd=" << totalVxmStreamNd
              << " sxm_tile_program=" << totalSxmTilePrograms
              << " repeat_replayed=" << repeatReplayed
              << " repeat2d_replayed=" << repeat2DReplayed
              << " macro_expanded=" << macroExpanded
              << " mem_stream_nd_expanded=" << memStreamNdExpanded
              << " mem_slice_program_expanded="
              << memSliceProgramExpanded
              << " mxm_stream_nd_expanded=" << mxmStreamNdExpanded
              << " vxm_stream_nd_expanded=" << vxmStreamNdExpanded
              << " sxm_tile_program_expanded=" << sxmTileProgramExpanded
              << " expanded_instruction=" << expandedInstructions
              << " encoded_work_entries=" << encodedWorkEntries
              << " saved_work_entries=" << savedWorkEntries
              << " trace_queue_pattern_rows=" << tracePatternRows
              << " serialized_queue_bytes=" << serializedQueueBytes
              << '\n';
    if (totalMemSlicePrograms != 0) {
        const auto savedPayloadWords =
            static_cast<std::int64_t>(equivalentMemStreamPayloadWords)
            - static_cast<std::int64_t>(memSlicePayloadWords);
        std::cout << "binary mem_slice_cost programs="
                  << totalMemSlicePrograms
                  << " body_entries=" << totalMemSliceBodyEntries
                  << " max_body_entries=" << maxMemSliceBodyEntries
                  << " average_body_entries="
                  << static_cast<double>(totalMemSliceBodyEntries)
                        / totalMemSlicePrograms
                  << " slice_payload_words=" << memSlicePayloadWords
                  << " equivalent_mem_stream_payload_words="
                  << equivalentMemStreamPayloadWords
                  << " saved_payload_words=" << savedPayloadWords
                  << '\n';
    }
    const auto imem =
        ftlpu::software::runtime::analyze_cmodel_abstract_imem(program);
    const auto savedImemWork = imem.expanded_work > imem.encoded_work_entries
        ? imem.expanded_work - imem.encoded_work_entries : 0;
    const auto workCompression = imem.expanded_work == 0 ? 0.0
        : 100.0 * static_cast<double>(savedImemWork)
            / static_cast<double>(imem.expanded_work);
    std::cout << "imem model=cmodel-abstract"
              << " deployable=" << (imem.fits() ? "yes" : "no")
              << " queues=" << imem.queues.size()
              << " overflow_queues=" << imem.overflow_queues
              << " used_slots=" << imem.used_slots
              << " used_bits=" << imem.used_bits
              << " used_bytes_ceil=" << (imem.used_bits + 7) / 8
              << " active_queue_capacity_bits="
              << imem.active_queue_capacity_bits
              << " expanded_work=" << imem.expanded_work
              << " encoded_work_entries=" << imem.encoded_work_entries
              << " saved_work_entries=" << savedImemWork
              << " work_compression_percent=" << workCompression
              << '\n';
    for (const auto kind : {
             ftlpu::software::runtime::QueueKind::Mem,
             ftlpu::software::runtime::QueueKind::MxmLoad,
             ftlpu::software::runtime::QueueKind::MxmCompute,
             ftlpu::software::runtime::QueueKind::MxmDequant,
             ftlpu::software::runtime::QueueKind::Vxm,
             ftlpu::software::runtime::QueueKind::SxmTranspose,
             ftlpu::software::runtime::QueueKind::SxmPermute}) {
        std::size_t queueCount = 0;
        std::size_t usedSlots = 0;
        std::size_t overflowQueues = 0;
        std::size_t maxUsed = 0;
        std::uint64_t usedBits = 0;
        std::uint64_t expandedWork = 0;
        std::size_t encodedWork = 0;
        std::uint32_t slotBits = 0;
        std::uint32_t depth = 0;
        for (const auto& queue : imem.queues) {
            if (queue.kind != kind) continue;
            ++queueCount;
            usedSlots += queue.used_slots;
            usedBits += queue.used_bits();
            expandedWork += queue.expanded_work;
            encodedWork += queue.instruction_entries
                + queue.repeat_entries + queue.repeat_2d_entries
                + queue.macro_entries
                + queue.coarse_program_entries;
            maxUsed = std::max(maxUsed, queue.used_slots);
            overflowQueues += queue.overflow() ? 1 : 0;
            slotBits = queue.slot_bits;
            depth = queue.depth;
        }
        if (queueCount == 0) continue;
        const auto saved = expandedWork > encodedWork
            ? expandedWork - encodedWork : 0;
        std::cout << "imem resource="
                  << ftlpu::software::runtime::queue_kind_name(kind)
                  << " slot_bits=" << slotBits
                  << " depth_per_queue=" << depth
                  << " queues=" << queueCount
                  << " used_slots=" << usedSlots
                  << " max_used_slots=" << maxUsed
                  << " max_utilization_percent="
                  << 100.0 * static_cast<double>(maxUsed) / depth
                  << " used_bits=" << usedBits
                  << " expanded_work=" << expandedWork
                  << " encoded_work_entries=" << encodedWork
                  << " saved_work_entries=" << saved
                  << " overflow_queues=" << overflowQueues << '\n';
    }
    for (const auto& queue : imem.queues) {
        if (!reportAllQueues && !queue.overflow()) continue;
        std::cout << "imem queue resource="
                  << ftlpu::software::runtime::queue_kind_name(queue.kind)
                  << " queue=" << queue.index
                  << " slot_bits=" << queue.slot_bits
                  << " depth=" << queue.depth
                  << " used_slots=" << queue.used_slots
                  << " utilization_percent="
                  << 100.0 * static_cast<double>(queue.used_slots)
                      / queue.depth
                  << " overflow_slots=" << queue.overflow_slots()
                  << " expanded_work=" << queue.expanded_work
                  << " instruction=" << queue.instruction_entries
                  << " nop=" << queue.nop_entries
                  << " repeat=" << queue.repeat_entries
                  << " repeat2d=" << queue.repeat_2d_entries
                  << " macro=" << queue.macro_entries
                  << " coarse_program=" << queue.coarse_program_entries
                  << '\n';
    }
    const auto physical =
        ftlpu::software::runtime::analyze_physical_imem(program);
    std::cout << "imem model=target-physical-v1"
              << " deployable=" << (physical.fits() ? "yes" : "no")
              << " queues=" << physical.queues.size()
              << " mem_delta_rle_queues=" << physical.mem_delta_rle_queues
              << " stream_nd_packets=" << physical.stream_nd_packets
              << " overflow_queues=" << physical.overflow_queues
              << " context_overflow_queues="
              << physical.macro_context_overflow_queues
              << " used_slots=" << physical.used_slots
              << " used_bits=" << physical.used_bits
              << " used_bytes_ceil=" << (physical.used_bits + 7) / 8
              << " active_queue_capacity_bits="
              << physical.active_queue_capacity_bits
              << " peak_macro_context_bits="
              << physical.peak_macro_context_bits
              << " provisioned_macro_context_bits="
              << physical.provisioned_macro_context_bits << '\n';
    for (const auto kind : {
             ftlpu::software::runtime::QueueKind::Mem,
             ftlpu::software::runtime::QueueKind::MxmLoad,
             ftlpu::software::runtime::QueueKind::MxmCompute,
             ftlpu::software::runtime::QueueKind::MxmDequant,
             ftlpu::software::runtime::QueueKind::Vxm,
             ftlpu::software::runtime::QueueKind::SxmTranspose,
             ftlpu::software::runtime::QueueKind::SxmPermute}) {
        std::size_t queues = 0;
        std::size_t slots = 0;
        std::size_t maxSlots = 0;
        std::size_t peakContexts = 0;
        std::size_t contextCapacity = 0;
        std::uint64_t peakContextBits = 0;
        std::uint64_t provisionedContextBits = 0;
        std::size_t runs = 0;
        std::size_t escapes = 0;
        std::size_t compactTemplateRuns = 0;
        std::size_t extendedTemplateRuns = 0;
        std::size_t streamNdPackets = 0;
        std::uint64_t bits = 0;
        for (const auto& queue : physical.queues) {
            if (queue.kind != kind) continue;
            ++queues;
            slots += queue.physical_slots;
            maxSlots = std::max(maxSlots, queue.physical_slots);
            peakContexts = std::max(peakContexts, queue.peak_macro_contexts);
            contextCapacity = std::max(
                contextCapacity, queue.macro_context_capacity);
            peakContextBits += static_cast<std::uint64_t>(
                queue.peak_macro_contexts) * queue.macro_context_bits;
            provisionedContextBits += static_cast<std::uint64_t>(
                queue.macro_context_capacity) * queue.macro_context_bits;
            runs += queue.macro_codec.run_count;
            escapes += queue.macro_codec.escaped_transitions;
            compactTemplateRuns += queue.macro_codec.compact_template_runs;
            extendedTemplateRuns += queue.macro_codec.extended_template_runs;
            streamNdPackets += queue.stream_nd_packets;
            bits += queue.physical_bits;
        }
        if (queues == 0) continue;
        std::cout << "imem physical_resource="
                  << ftlpu::software::runtime::queue_kind_name(kind)
                  << " queues=" << queues
                  << " used_slots=" << slots
                  << " max_used_slots=" << maxSlots
                  << " used_bits=" << bits
                  << " stream_nd_packets=" << streamNdPackets
                  << " peak_macro_contexts=" << peakContexts
                  << " macro_context_capacity=" << contextCapacity
                  << " peak_context_bits=" << peakContextBits
                  << " provisioned_context_bits="
                  << provisionedContextBits
                  << " template_runs=" << runs
                  << " compact_template_runs=" << compactTemplateRuns
                  << " extended_template_runs=" << extendedTemplateRuns
                  << " escaped_deltas=" << escapes << '\n';
    }
    std::map<std::pair<std::uint32_t,
        ftlpu::software::runtime::QueueKind>, std::size_t>
        scaleRelocations;
    for (const auto& relocation : program.scale_relocations)
        ++scaleRelocations[{relocation.binding_index,
            relocation.queue_kind}];
    std::cout << "binary scale_relocations="
              << program.scale_relocations.size() << '\n';
    for (const auto& [key, count] : scaleRelocations)
        std::cout << "binary scale_relocation binding=" << key.first
                  << " resource="
                  << ftlpu::software::runtime::queue_kind_name(key.second)
                  << " count=" << count << '\n';
    std::cout << "binary weight_page_uses="
              << program.weight_page_uses.size() << '\n';
    for (const auto& use : program.weight_page_uses)
        std::cout << "binary weight_page_use binding=" << use.binding_index
                  << " page=" << use.page_index
                  << " bank=" << use.bank
                  << " ready_cycle=" << use.ready_cycle
                  << " release_cycle=" << use.release_cycle << '\n';
    std::cout << "binary stream_release_cycles="
              << program.stream_release_cycles.size() << '\n';
    const auto reportStreamReleases = [&](std::string_view direction,
                                          std::size_t offset,
                                          std::size_t count) {
        for (std::size_t begin = 0; begin < count;) {
            std::size_t end = begin + 1;
            while (end < count
                && program.stream_release_cycles[offset + end]
                    == program.stream_release_cycles[offset + begin])
                ++end;
            std::cout << "binary stream_release direction=" << direction
                      << " streams=" << begin;
            if (end != begin + 1) std::cout << ".." << end - 1;
            std::cout << " release_cycle="
                      << program.stream_release_cycles[offset + begin]
                      << '\n';
            begin = end;
        }
    };
    const std::size_t directionalStreams =
        program.hardware.streams_per_direction;
    if (program.stream_release_cycles.size()
            == program.hardware.encoded_streams
        && program.hardware.encoded_streams == 2 * directionalStreams) {
        reportStreamReleases("east", 0, directionalStreams);
        reportStreamReleases(
            "west", directionalStreams, directionalStreams);
    } else {
        reportStreamReleases(
            "merged", 0, program.stream_release_cycles.size());
    }
    const std::size_t reported = reportAllQueues
        ? queues.size() : std::min<std::size_t>(queues.size(), 20);
    for (std::size_t i = 0; i < reported; ++i) {
        const auto& queue = *queues[i];
        std::size_t instructions = 0;
        std::size_t nops = 0;
        std::size_t repeats = 0;
        std::size_t repeats2d = 0;
        std::size_t macros = 0;
        std::size_t memStreamNd = 0;
        std::size_t memSlicePrograms = 0;
        std::size_t mxmStreamNd = 0;
        std::size_t vxmStreamNd = 0;
        std::size_t sxmTilePrograms = 0;
        for (const auto& command : queue.commands) {
            if (ftlpu::software::runtime::is_vxm_stream_nd_command(command)) {
                ++vxmStreamNd;
                continue;
            }
            if (ftlpu::software::runtime::is_sxm_tile_program_command(
                    command)) {
                ++sxmTilePrograms;
                continue;
            }
            if (ftlpu::software::runtime::is_mem_slice_program_command(
                    command)) {
                ++memSlicePrograms;
                continue;
            }
            if (ftlpu::software::runtime::is_mem_stream_nd_command(command)) {
                ++memStreamNd;
                continue;
            }
            if (ftlpu::software::runtime::is_mxm_stream_nd_command(command)) {
                ++mxmStreamNd;
                continue;
            }
            if (ftlpu::software::runtime::is_macro_schedule_command(command)) {
                ++macros;
                continue;
            }
            if (ftlpu::software::runtime::is_repeat_2d_command(command)) {
                ++repeats2d;
                continue;
            }
            switch (ftlpu::isa::decode_icu_command_opcode(command.command)) {
            case ftlpu::isa::IcuCommandOpcode::Instruction:
                ++instructions;
                break;
            case ftlpu::isa::IcuCommandOpcode::Nop:
                ++nops;
                break;
            case ftlpu::isa::IcuCommandOpcode::Repeat:
                ++repeats;
                break;
            default:
                break;
            }
        }
        std::cout << "binary queue_occupancy rank=" << i
                  << " resource="
                  << ftlpu::software::runtime::queue_kind_name(queue.kind)
                  << " queue=" << queue.index
                  << " commands=" << queue.commands.size()
                  << " instruction=" << instructions
                  << " nop=" << nops
                  << " repeat=" << repeats
                  << " repeat2d=" << repeats2d
                  << " macro=" << macros
                  << " mem_stream_nd=" << memStreamNd
                  << " mem_slice_program=" << memSlicePrograms
                  << " mxm_stream_nd=" << mxmStreamNd
                  << " vxm_stream_nd=" << vxmStreamNd
                  << " sxm_tile_program=" << sxmTilePrograms << '\n';
    }
    for (const auto kind : {
             ftlpu::software::runtime::QueueKind::Mem,
             ftlpu::software::runtime::QueueKind::MxmLoad,
             ftlpu::software::runtime::QueueKind::MxmCompute,
             ftlpu::software::runtime::QueueKind::MxmDequant,
             ftlpu::software::runtime::QueueKind::Vxm,
             ftlpu::software::runtime::QueueKind::SxmTranspose,
             ftlpu::software::runtime::QueueKind::SxmPermute}) {
        std::size_t kindQueues = 0;
        std::size_t kindCommands = 0;
        std::size_t kindMin = std::numeric_limits<std::size_t>::max();
        std::size_t kindMax = 0;
        for (const auto* queue : queues) {
            if (queue->kind != kind) continue;
            ++kindQueues;
            kindCommands += queue->commands.size();
            kindMin = std::min(kindMin, queue->commands.size());
            kindMax = std::max(kindMax, queue->commands.size());
        }
        if (kindQueues != 0)
            std::cout << "binary resource_summary resource="
                      << ftlpu::software::runtime::queue_kind_name(kind)
                      << " queues=" << kindQueues
                      << " commands=" << kindCommands
                      << " min=" << kindMin
                      << " max=" << kindMax
                      << " average="
                      << static_cast<double>(kindCommands) / kindQueues
                      << '\n';
        const auto found = std::ranges::find_if(queues,
            [kind](const auto* queue) { return queue->kind == kind; });
        if (found == queues.end()) continue;
        std::size_t instructionCount = 0;
        std::size_t nopCount = 0;
        std::size_t repeatCount = 0;
        std::size_t repeat2DCount = 0;
        std::size_t macroCount = 0;
        std::size_t memStreamNdCount = 0;
        std::size_t memSliceProgramCount = 0;
        std::size_t mxmStreamNdCount = 0;
        std::size_t vxmStreamNdCount = 0;
        std::size_t sxmTileProgramCount = 0;
        for (const auto& command : (*found)->commands) {
            if (ftlpu::software::runtime::is_vxm_stream_nd_command(command)) {
                ++vxmStreamNdCount;
                continue;
            }
            if (ftlpu::software::runtime::is_sxm_tile_program_command(
                    command)) {
                ++sxmTileProgramCount;
                continue;
            }
            if (ftlpu::software::runtime::is_mem_slice_program_command(
                    command)) {
                ++memSliceProgramCount;
                continue;
            }
            if (ftlpu::software::runtime::is_mem_stream_nd_command(command)) {
                ++memStreamNdCount;
                continue;
            }
            if (ftlpu::software::runtime::is_mxm_stream_nd_command(command)) {
                ++mxmStreamNdCount;
                continue;
            }
            if (ftlpu::software::runtime::is_macro_schedule_command(command)) {
                ++macroCount;
                continue;
            }
            if (ftlpu::software::runtime::is_repeat_2d_command(command)) {
                ++repeat2DCount;
                continue;
            }
            switch (ftlpu::isa::decode_icu_command_opcode(command.command)) {
            case ftlpu::isa::IcuCommandOpcode::Instruction:
                ++instructionCount;
                break;
            case ftlpu::isa::IcuCommandOpcode::Nop:
                ++nopCount;
                break;
            case ftlpu::isa::IcuCommandOpcode::Repeat:
                ++repeatCount;
                break;
            default:
                break;
            }
        }
        std::cout << "binary resource_max resource="
                  << ftlpu::software::runtime::queue_kind_name(kind)
                  << " queue=" << (*found)->index
                  << " commands=" << (*found)->commands.size()
                  << " instruction=" << instructionCount
                  << " nop=" << nopCount
                  << " repeat=" << repeatCount
                  << " repeat2d=" << repeat2DCount
                  << " macro=" << macroCount
                  << " mem_stream_nd=" << memStreamNdCount
                  << " mem_slice_program=" << memSliceProgramCount
                  << " mxm_stream_nd=" << mxmStreamNdCount
                  << " vxm_stream_nd=" << vxmStreamNdCount
                  << " sxm_tile_program=" << sxmTileProgramCount << '\n';
    }
    for (const auto& binding : program.bindings) {
        std::cout << "binary binding index=" << binding.index
                  << " access=" << static_cast<unsigned>(binding.access)
                  << " role=" << binding.role
                  << " name=" << binding.name
                  << " layout=" << static_cast<unsigned>(binding.layout)
                  << " bytes=" << binding.byte_size
                  << " page_count=" << binding.page_count
                  << " base_row=" << binding.base_row
                  << " rows=" << binding.instruction_count
                  << " bank=" << binding.bank
                  << " hemisphere_mask=" << binding.hemisphere_mask
                  << " slices=";
        for (std::size_t i = 0; i < binding.slices.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << binding.slices[i];
        }
        std::cout << '\n';
    }
    ftlpu::software::runtime::print_runtime_performance(
        program, cycles, std::cout);
    if (tracePath)
        ftlpu::software::runtime::write_schedule_trace_csv(
            program, *tracePath);
    if (comparePath) {
        const auto reference =
            ftlpu::software::runtime::read_binary_program(*comparePath);
        const auto referenceIssues =
            ftlpu::software::runtime::inspect_logical_issues(reference);
        const auto comparison =
            ftlpu::software::runtime::compare_logical_issues(
                program, reference);
        const auto leftBytes = std::filesystem::file_size(argv[1]);
        const auto rightBytes = std::filesystem::file_size(*comparePath);
        const auto byteDelta = static_cast<std::int64_t>(leftBytes)
            - static_cast<std::int64_t>(rightBytes);
        std::cout << "binary compare result="
                  << (comparison.equivalent ? "equivalent" : "mismatch")
                  << " same_target=" << comparison.same_target
                  << " same_horizon=" << comparison.same_horizon
                  << " left_functional_issues="
                  << logicalIssues.functional_issues
                  << " right_functional_issues="
                  << referenceIssues.functional_issues
                  << " left_logical_nop_cycles="
                  << logicalIssues.logical_nop_cycles
                  << " right_logical_nop_cycles="
                  << referenceIssues.logical_nop_cycles
                  << " left_file_bytes=" << leftBytes
                  << " right_file_bytes=" << rightBytes
                  << " left_minus_right_bytes=" << byteDelta << '\n';
        if (comparison.first_mismatch) {
            const auto& mismatch = *comparison.first_mismatch;
            std::cout << "binary compare_first_mismatch reason=\""
                      << mismatch.reason << "\" cycle=" << mismatch.cycle;
            if (mismatch.left || mismatch.right)
                std::cout << " queue="
                          << ftlpu::software::runtime::queue_kind_name(
                                 mismatch.kind)
                          << '[' << mismatch.index << ']';
            std::cout << " left="
                      << (mismatch.left
                              ? ftlpu::software::runtime::
                                    describe_logical_instruction(
                                        *mismatch.left)
                              : "NOP")
                      << " right="
                      << (mismatch.right
                              ? ftlpu::software::runtime::
                                    describe_logical_instruction(
                                        *mismatch.right)
                              : "NOP")
                      << '\n';
        }
        if (!comparison.equivalent) return 2;
    }
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu_binary_inspect failed: " << ex.what() << '\n';
    return 1;
}
