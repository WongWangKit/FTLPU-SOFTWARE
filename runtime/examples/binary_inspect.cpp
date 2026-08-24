#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
try {
    if (argc != 2 && argc != 4)
        throw std::runtime_error(
            "usage: ftlpu_binary_inspect program.ftlpu "
            "[--trace schedule.csv]");
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
              << " commands=" << total_commands << '\n';
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
    const std::size_t reported = std::min<std::size_t>(queues.size(), 20);
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
