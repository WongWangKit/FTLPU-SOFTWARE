#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/software/runtime/binary.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

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

} // namespace

int main(int argc, char** argv)
try {
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
    return 0;
} catch (const std::exception& error) {
    std::cerr << "macro_bitstream_test failed: " << error.what() << '\n';
    return 1;
}
