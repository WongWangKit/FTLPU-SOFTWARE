#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;

static std::array<std::uint32_t, 4> bits(const MemInstruction& instruction)
{
    const auto value = isa::encode_mem_instruction(instruction);
    return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32), 0, 0};
}

static std::array<std::uint32_t, 4> bits(const MxmControlInstruction& instruction)
{
    const auto value = isa::encode_mxm_instruction(instruction);
    return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32), 0, 0};
}

static std::array<std::uint32_t, 4> bits(const MxmDequantInstruction& instruction)
{
    return {static_cast<std::uint32_t>(isa::encode_mxm_dequant_instruction(instruction)), 0, 0, 0};
}

template <typename Queue>
static void replay(Queue& queue, const rt::LogicalQueueTimeline& reference,
    std::size_t maxCycle)
{
    if (!queue.raw_macro_mode())
        throw std::runtime_error("requested Macro queue silently loaded as FU3D");
    std::size_t position = 0;
    for (std::size_t cycle = 0; cycle <= maxCycle + 64; ++cycle) {
        const auto issued = queue.tick();
        const bool due = position < reference.issues.size()
            && reference.issues[position].cycle == cycle;
        if (static_cast<bool>(issued) != due)
            throw std::runtime_error("physical Macro issue timing differs at cycle " + std::to_string(cycle));
        if (issued) {
            if (bits(*issued) != reference.issues[position].instruction.words)
                throw std::runtime_error("physical Macro native instruction differs");
            ++position;
        }
        if (queue.done()) break;
    }
    if (!queue.done() || position != reference.issues.size())
        throw std::runtime_error("physical Macro queue did not retire its complete trace");
}

int main(int argc, char** argv)
try {
    if (argc != 3)
        throw std::runtime_error("usage: raw_macro_cli_replay_test baseline.ftlpu raw-macro.ftlpu");
    const auto baseline = rt::read_binary_program(argv[1]);
    const auto macro = rt::read_binary_program(argv[2]);
    if (!rt::compare_logical_issues(baseline, macro).equivalent)
        throw std::runtime_error("production Macro binary changed logical issues");
    const auto checkRelocations = [&](const auto& relocations) {
        for (const auto& relocation : relocations) {
            const auto queue = std::ranges::find_if(macro.queues, [&](const auto& item) {
                return item.kind == relocation.queue_kind && item.index == relocation.queue_index;
            });
            if (queue == macro.queues.end() || relocation.command_index >= queue->commands.size())
                throw std::runtime_error("relocation does not identify a valid command");
            if (queue->requires_raw_macro
                && !rt::is_macro_schedule_command(queue->commands[relocation.command_index]))
                throw std::runtime_error("relocation lost its Macro descriptor owner");
        }
    };
    checkRelocations(macro.address_relocations);
    checkRelocations(macro.scale_relocations);
    const auto expected = rt::inspect_logical_issues(baseline);
    auto icu = std::make_unique<InstructionControlUnit>();
    const auto prime = rt::load_queue_programs_into_icu(
        macro.queues, *icu, macro.hardware.mxms_per_hemisphere);
    std::size_t tested = 0;
    std::size_t issues = 0;
    for (const auto& queue : macro.queues) {
        if (queue.commands.empty()) continue;
        if (queue.kind != rt::QueueKind::Mem && queue.kind != rt::QueueKind::MxmLoad
            && queue.kind != rt::QueueKind::MxmCompute && queue.kind != rt::QueueKind::MxmDequant)
            continue;
        if (!queue.requires_raw_macro)
            throw std::runtime_error("packed binary did not retain its required Macro mode");
        const auto reference = std::ranges::find_if(expected.queues, [&](const auto& item) {
            return item.kind == queue.kind && item.index == queue.index;
        });
        if (reference == expected.queues.end()) throw std::runtime_error("reference queue is missing");
        const auto local = macro.hardware.mxms_per_hemisphere;
        const auto physical = queue.index / local * hw::kMxmsPerHemisphere + queue.index % local;
        switch (queue.kind) {
        case rt::QueueKind::Mem: replay(icu->mem_iq(queue.index), *reference, macro.max_cycle); break;
        case rt::QueueKind::MxmLoad: replay(icu->mxm_load_iq(physical), *reference, macro.max_cycle); break;
        case rt::QueueKind::MxmCompute: replay(icu->mxm_compute_iq(physical), *reference, macro.max_cycle); break;
        case rt::QueueKind::MxmDequant: replay(icu->mxm_dequant_iq(physical), *reference, macro.max_cycle); break;
        default: break;
        }
        ++tested;
        issues += reference->issues.size();
    }
    if (tested == 0) throw std::runtime_error("production binary contains no Macro queues");
    std::cout << "production Macro frontend replay passed: queues=" << tested
              << " issues=" << issues << " prime_cycles=" << prime << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "raw_macro_cli_replay_test failed: " << error.what() << '\n';
    return 1;
}
