#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
try {
    if (argc != 2 && argc != 3 && argc != 4)
        throw std::runtime_error(
            "usage: ftlpu_binary_inspect program.ftlpu "
            "[--all-queues] [--trace schedule.csv]");
    const bool reportAllQueues = argc == 3
        && std::string_view(argv[2]) == "--all-queues";
    if (argc == 3 && !reportAllQueues)
        throw std::runtime_error("expected --all-queues");
    if (argc == 4 && std::string_view(argv[2]) != "--trace")
        throw std::runtime_error("expected --trace before the CSV path");

    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
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
    std::size_t totalInstructions = 0;
    std::size_t totalNops = 0;
    std::size_t totalRepeats = 0;
    std::size_t totalRepeat2D = 0;
    std::size_t totalLoops = 0;
    std::size_t totalMacros = 0;
    std::size_t expandedInstructions = 0;
    std::size_t repeatReplayed = 0;
    std::size_t repeat2DReplayed = 0;
    std::size_t loopReplayed = 0;
    std::size_t macroExpanded = 0;
    std::size_t tracePatternRows = 0;
    std::size_t serializedQueueBytes = program.queues.size() * 8;
    for (const auto& queue : program.queues) {
        for (const auto& command : queue.commands) {
            const bool macro =
                ftlpu::software::runtime::is_macro_schedule_command(command);
            serializedQueueBytes += 1
                + command.word_count * sizeof(std::uint32_t)
                + (macro ? 29 : sizeof(std::uint32_t))
                + (!macro && !command.extension_words.empty()
                        ? sizeof(std::uint16_t)
                            + command.extension_words.size()
                                * sizeof(std::uint32_t)
                        : 0);
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
            case ftlpu::isa::IcuCommandOpcode::Loop: {
                ++totalLoops;
                const auto loop =
                    ftlpu::isa::decode_icu_loop(command.command);
                loopReplayed += loop.window_size * loop.count;
                expandedInstructions += loop.window_size * loop.count;
                if (loop.count != 0) tracePatternRows += loop.window_size;
                break;
            }
            default:
                break;
            }
        }
    }
    const std::size_t encodedWorkEntries = totalInstructions + totalRepeats
        + totalRepeat2D + totalLoops + totalMacros;
    const std::size_t savedWorkEntries = expandedInstructions
        > encodedWorkEntries ? expandedInstructions - encodedWorkEntries : 0;
    std::cout << "binary aggregate instruction=" << totalInstructions
              << " nop=" << totalNops
              << " repeat=" << totalRepeats
              << " repeat2d=" << totalRepeat2D
              << " loop=" << totalLoops
              << " macro=" << totalMacros
              << " repeat_replayed=" << repeatReplayed
              << " repeat2d_replayed=" << repeat2DReplayed
              << " loop_replayed=" << loopReplayed
              << " macro_expanded=" << macroExpanded
              << " expanded_instruction=" << expandedInstructions
              << " encoded_work_entries=" << encodedWorkEntries
              << " saved_work_entries=" << savedWorkEntries
              << " trace_queue_pattern_rows=" << tracePatternRows
              << " serialized_queue_bytes=" << serializedQueueBytes
              << '\n';
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
    const std::size_t reported = reportAllQueues
        ? queues.size() : std::min<std::size_t>(queues.size(), 20);
    for (std::size_t i = 0; i < reported; ++i) {
        const auto& queue = *queues[i];
        std::size_t instructions = 0;
        std::size_t nops = 0;
        std::size_t repeats = 0;
        std::size_t loops = 0;
        std::size_t repeats2d = 0;
        std::size_t macros = 0;
        for (const auto& command : queue.commands) {
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
            case ftlpu::isa::IcuCommandOpcode::Loop:
                ++loops;
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
                  << " loop=" << loops
                  << " macro=" << macros << '\n';
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
        std::size_t loopCount = 0;
        std::size_t macroCount = 0;
        for (const auto& command : (*found)->commands) {
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
            case ftlpu::isa::IcuCommandOpcode::Loop:
                ++loopCount;
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
                  << " loop=" << loopCount
                  << " macro=" << macroCount << '\n';
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
    if (argc == 4)
        ftlpu::software::runtime::write_schedule_trace_csv(
            program, std::filesystem::path(argv[3]));
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu_binary_inspect failed: " << ex.what() << '\n';
    return 1;
}
