#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rt = ftlpu::software::runtime;

namespace {

rt::QueueCommand macro(ftlpu::MemInstruction instruction,
    const ftlpu::IcuMacroSchedule& schedule)
{
    const auto encoded = ftlpu::isa::encode_mem_instruction(instruction);
    rt::QueueCommand command {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
    return rt::encode_macro_schedule_command(std::move(command), schedule);
}

rt::QueueCommand macro(ftlpu::MxmControlInstruction instruction,
    const ftlpu::IcuMacroSchedule& schedule)
{
    const auto encoded = ftlpu::isa::encode_mxm_instruction(instruction);
    rt::QueueCommand command {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
    return rt::encode_macro_schedule_command(std::move(command), schedule);
}

void expect(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

void compare(const rt::QueueCommand& expected, const rt::QueueCommand& actual)
{
    const auto expectedSchedule = rt::decode_macro_schedule_command(expected);
    const auto actualSchedule = rt::decode_macro_schedule_command(actual);
    expect(expectedSchedule.start_cycle == actualSchedule.start_cycle,
        "start cycle changed");
    expect(expectedSchedule.inner_count == actualSchedule.inner_count,
        "inner count changed");
    expect(expectedSchedule.outer_count == actualSchedule.outer_count,
        "outer count changed");
    if (expectedSchedule.inner_count != 1) {
        expect(expectedSchedule.inner_interval == actualSchedule.inner_interval,
            "inner interval changed");
        expect(expectedSchedule.inner_stride == actualSchedule.inner_stride,
            "inner stride changed");
    }
    if (expectedSchedule.outer_count != 1) {
        expect(expectedSchedule.outer_interval == actualSchedule.outer_interval,
            "outer interval changed");
        expect(expectedSchedule.outer_stride == actualSchedule.outer_stride,
            "outer stride changed");
    }
    expect(expectedSchedule.induction_target == actualSchedule.induction_target,
        "induction target changed");
    const auto expectedInstruction = ftlpu::isa::decode_mem_instruction(
        expected.words[0] | (static_cast<std::uint64_t>(expected.words[1]) << 32));
    const auto actualInstruction = ftlpu::isa::decode_mem_instruction(
        actual.words[0] | (static_cast<std::uint64_t>(actual.words[1]) << 32));
    expect(expectedInstruction.opcode == actualInstruction.opcode,
        "MEM opcode changed");
    expect(expectedInstruction.address == actualInstruction.address,
        "MEM address changed");
    expect(expectedInstruction.stream == actualInstruction.stream,
        "MEM stream changed");
    expect(expectedInstruction.preserve_stream == actualInstruction.preserve_stream,
        "MEM preserve-stream changed");
}

struct CmodelMemRun {
    ftlpu::IcuFrontendStatistics frontend{};
    std::vector<std::pair<std::size_t, ftlpu::isa::EncodedMemInstruction>>
        issues;
};

CmodelMemRun run_compiled_mem_program(const rt::BinaryProgram& program)
{
    const rt::QueueProgram* memQueue = nullptr;
    for (const auto& queue : program.queues) {
        if (queue.commands.empty()) continue;
        expect(queue.kind == rt::QueueKind::Mem && memQueue == nullptr,
            "Macro CModel A/B fixture must contain one MEM queue");
        memQueue = &queue;
    }
    expect(memQueue != nullptr,
        "Macro CModel A/B fixture has no MEM queue");

    ftlpu::InstructionControlUnit icu;
    rt::load_queue_programs_into_icu(program.queues, icu,
        program.hardware.mxms_per_hemisphere);
    CmodelMemRun result;
    auto& queue = icu.mem_iq(memQueue->index);
    for (std::size_t cycle = 0; cycle <= program.max_cycle + 64; ++cycle) {
        if (const auto issued = queue.tick())
            result.issues.emplace_back(
                cycle, ftlpu::isa::encode_mem_instruction(*issued));
        if (queue.done()) break;
    }
    expect(queue.done(), "Macro CModel A/B queue did not complete");
    result.frontend = icu.frontend_statistics();
    return result;
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc > 3)
        throw std::runtime_error(
            "usage: macro_bitstream_test [program.ftlpu | none.ftlpu macro.ftlpu]");
    rt::QueueProgram queue {rt::QueueKind::Mem, 3, {}};
    queue.commands.push_back(macro(ftlpu::MemInstruction::Read(100, 2),
        {10, 1, 1, 0, 4, 7, 3, ftlpu::IcuInductionTarget::MemAddress}));
    queue.commands.push_back(macro(ftlpu::MemInstruction::Read(112, 2),
        {42, 1, 1, 0, 4, 7, 3, ftlpu::IcuInductionTarget::MemAddress}));
    queue.commands.push_back(macro(ftlpu::MemInstruction::Read(124, 2),
        {74, 1, 1, 0, 4, 7, 3, ftlpu::IcuInductionTarget::MemAddress}));
    queue.commands.push_back(macro(ftlpu::MemInstruction::WriteTap(88, 5),
        {106, 8, 2, -1, 3, 23, 4, ftlpu::IcuInductionTarget::MemAddress}));
    // Forces both the wide delta escape and the extended template path.
    queue.commands.push_back(macro(ftlpu::MemInstruction::Write(90, 6),
        {5000000, 4097, 3, 2, 1, 1, 0,
            ftlpu::IcuInductionTarget::MemAddress}));

    const auto image = rt::encode_mem_macro_bitstream(queue);
    const auto decoded = rt::decode_mem_macro_bitstream(image, queue.index);
    expect(decoded.commands.size() == queue.commands.size(),
        "command count changed");
    for (std::size_t i = 0; i < queue.commands.size(); ++i)
        compare(queue.commands[i], decoded.commands[i]);
    expect(image.stats.compact_template_runs != 0,
        "compact template was not exercised");
    expect(image.stats.extended_template_runs != 0,
        "extended template was not exercised");
    expect(image.stats.wide_escaped_transitions != 0,
        "wide delta escape was not exercised");

    // Exercise the complete binary -> runtime loader -> finite CModel Macro
    // context path for both supported Macro v1 functional-unit families.
    rt::BinaryProgram executionProgram;
    executionProgram.max_cycle = 13;
    executionProgram.queues.push_back(rt::QueueProgram {
        rt::QueueKind::Mem, 0,
        {
            macro(ftlpu::MemInstruction::Read(100, 0),
                {2, 3, 4, 1, 1, 1, 0,
                    ftlpu::IcuInductionTarget::MemAddress}),
            macro(ftlpu::MemInstruction::Read(200, 1),
                {3, 3, 4, 1, 1, 1, 0,
                    ftlpu::IcuInductionTarget::MemAddress}),
        }});
    executionProgram.queues.push_back(rt::QueueProgram {
        rt::QueueKind::MxmCompute, 0,
        {macro(ftlpu::MxmControlInstruction::Compute(
                   0, 0, 0, 10),
            {2, 2, 3, 1, 2, 8, 10,
                ftlpu::IcuInductionTarget::MxmAccumulatorAddress})}});

    std::stringstream binary(
        std::ios::in | std::ios::out | std::ios::binary);
    rt::write_binary_program(executionProgram, binary);
    binary.seekg(0);
    const auto restoredProgram = rt::read_binary_program(binary);
    for (const auto& restoredQueue : restoredProgram.queues)
        for (const auto& command : restoredQueue.commands) {
            expect(rt::is_macro_schedule_command(command),
                "2-D Macro changed representation during binary round-trip");
            expect(!rt::is_mem_stream_nd_command(command)
                    && !rt::is_mxm_stream_nd_command(command),
                "2-D Macro unexpectedly became STREAM_ND");
        }

    ftlpu::InstructionControlUnit icu;
    rt::load_queue_programs_into_icu(restoredProgram.queues, icu);
    std::vector<std::pair<std::size_t, std::size_t>> memIssues;
    std::vector<std::pair<std::size_t, std::size_t>> mxmIssues;
    for (std::size_t cycle = 0; cycle <= executionProgram.max_cycle;
         ++cycle) {
        if (const auto issued = icu.mem_iq(0).tick())
            memIssues.emplace_back(cycle, issued->address);
        if (const auto issued = icu.mxm_compute_iq(0).tick())
            mxmIssues.emplace_back(cycle, issued->accumulator_address);
    }
    expect(memIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {2, 100}, {3, 200}, {6, 101},
                {7, 201}, {10, 102}, {11, 202}},
        "runtime/CModel MEM Macro emitted incorrect cycles or addresses");
    expect(mxmIssues
            == std::vector<std::pair<std::size_t, std::size_t>> {
                {2, 10}, {5, 11}, {10, 20}, {13, 21}},
        "runtime/CModel MXM Macro emitted incorrect cycles or accumulators");
    expect(icu.mem_iq(0).peak_active_macros() == 2,
        "runtime/CModel did not model interleaved finite Macro contexts");
    expect(icu.mem_iq(0).done() && icu.mxm_compute_iq(0).done(),
        "runtime/CModel Macro queues did not complete");

    std::cout << "macro_bitstream_test passed: bits="
              << image.stats.physical_bits() << '\n';
    if (argc == 2) {
        const auto program = rt::read_binary_program(argv[1]);
        std::size_t queues = 0;
        std::size_t commands = 0;
        std::uint64_t bits = 0;
        for (const auto& source : program.queues) {
            if (source.kind != rt::QueueKind::Mem || source.commands.empty()
                || !std::all_of(source.commands.begin(), source.commands.end(),
                    [](const rt::QueueCommand& command) {
                        return rt::is_macro_schedule_command(command)
                            && command.instruction_kind == rt::InstructionKind::Mem;
                    }))
                continue;
            const auto encoded = rt::encode_mem_macro_bitstream(source);
            const auto restored = rt::decode_mem_macro_bitstream(
                encoded, source.index);
            expect(restored.commands.size() == source.commands.size(),
                "Qwen roundtrip command count changed");
            for (std::size_t i = 0; i < source.commands.size(); ++i)
                compare(source.commands[i], restored.commands[i]);
            ++queues;
            commands += source.commands.size();
            bits += encoded.stats.physical_bits();
        }
        std::cout << "macro_bitstream_qwen_roundtrip passed: queues="
                  << queues << " commands=" << commands
                  << " bits=" << bits << '\n';
    }
    if (argc == 3) {
        const auto baseline = rt::read_binary_program(argv[1]);
        const auto compressed = rt::read_binary_program(argv[2]);
        expect(baseline.max_cycle == compressed.max_cycle,
            "compiler-generated Macro changed the scheduled cycle count");
        std::size_t macroCommands = 0;
        for (const auto& compressedQueue : compressed.queues)
            for (const auto& command : compressedQueue.commands) {
                expect(!rt::is_mem_stream_nd_command(command)
                        && !rt::is_mxm_stream_nd_command(command),
                    "compiler-generated Macro A/B binary contains STREAM_ND");
                macroCommands += rt::is_macro_schedule_command(command)
                    ? 1 : 0;
            }
        expect(macroCommands != 0,
            "compiler-generated Macro A/B binary contains no 2-D Macro");

        const auto baselineRun = run_compiled_mem_program(baseline);
        const auto compressedRun = run_compiled_mem_program(compressed);
        expect(baselineRun.issues == compressedRun.issues,
            "compiler-generated Macro changed CModel issue semantics");
        expect(baselineRun.frontend.issued_instructions
                == compressedRun.frontend.issued_instructions,
            "compiler-generated Macro changed dynamic issue count");
        expect(compressedRun.frontend.imem_entries
                < baselineRun.frontend.imem_entries,
            "compiler-generated Macro did not reduce CModel i-MEM entries");
        expect(compressedRun.frontend.fetched_entries
                < baselineRun.frontend.fetched_entries,
            "compiler-generated Macro did not reduce CModel fetch entries");
        expect(compressedRun.frontend.macro_queues != 0
                && compressedRun.frontend.peak_macro_contexts_per_queue != 0,
            "CModel did not execute compiler-generated Macro contexts");

        const auto baselineImem = rt::analyze_physical_imem(baseline);
        const auto compressedImem = rt::analyze_physical_imem(compressed);
        expect(compressedImem.used_bits < baselineImem.used_bits,
            "compiler-generated Macro did not reduce physical MEM bits");
        std::cout << "macro_cmodel_ab_test passed"
                  << " logical_issues=" << compressedRun.issues.size()
                  << " scheduled_cycles=" << compressed.max_cycle + 1
                  << " baseline_imem_entries="
                  << baselineRun.frontend.imem_entries
                  << " macro_imem_entries="
                  << compressedRun.frontend.imem_entries
                  << " baseline_fetched_entries="
                  << baselineRun.frontend.fetched_entries
                  << " macro_fetched_entries="
                  << compressedRun.frontend.fetched_entries
                  << " baseline_physical_bits=" << baselineImem.used_bits
                  << " macro_physical_bits=" << compressedImem.used_bits
                  << " peak_macro_contexts="
                  << compressedRun.frontend.peak_macro_contexts_per_queue
                  << '\n';
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "macro_bitstream_test failed: " << error.what() << '\n';
    return 1;
}
