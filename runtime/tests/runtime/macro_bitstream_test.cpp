#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rt = ftlpu::software::runtime;

namespace {

rt::QueueCommand macro(ftlpu::MemInstruction instruction,
                       const ftlpu::IcuMacroSchedule& schedule)
{
    const auto encoded = ftlpu::isa::encode_mem_instruction(instruction);
    rt::QueueCommand command{
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
    rt::QueueCommand command{
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
         static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
    return rt::encode_macro_schedule_command(std::move(command), schedule);
}

rt::QueueCommand macro(ftlpu::MxmDequantInstruction instruction,
                       const ftlpu::IcuMacroSchedule& schedule)
{
    rt::QueueCommand command{
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::MxmDequant,
        1,
        {ftlpu::isa::encode_mxm_dequant_instruction(instruction), 0, 0, 0},
    };
    return rt::encode_macro_schedule_command(std::move(command), schedule);
}

void expect(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <typename Fn>
void expect_throw(Fn&& function, const char* message)
{
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void compare_schedule(const rt::QueueCommand& expected,
                      const rt::QueueCommand& actual)
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
}

void compare(const rt::QueueCommand& expected, const rt::QueueCommand& actual)
{
    compare_schedule(expected, actual);
    const auto expectedInstruction = ftlpu::isa::decode_mem_instruction(
        expected.words[0] |
        (static_cast<std::uint64_t>(expected.words[1]) << 32));
    const auto actualInstruction = ftlpu::isa::decode_mem_instruction(
        actual.words[0] | (static_cast<std::uint64_t>(actual.words[1]) << 32));
    expect(expectedInstruction.opcode == actualInstruction.opcode,
           "MEM opcode changed");
    expect(expectedInstruction.address == actualInstruction.address,
           "MEM address changed");
    expect(expectedInstruction.stream == actualInstruction.stream,
           "MEM stream changed");
    expect(expectedInstruction.preserve_stream ==
               actualInstruction.preserve_stream,
           "MEM preserve-stream changed");
}

void compare_mxm(const rt::QueueCommand& expected,
                  const rt::QueueCommand& actual)
{
    compare_schedule(expected, actual);
    expect(expected.instruction_kind == actual.instruction_kind,
           "MXM instruction kind changed");
    expect(expected.word_count == actual.word_count,
           "MXM instruction width changed");
    expect(expected.words[0] == actual.words[0] &&
               expected.words[1] == actual.words[1],
           "MXM native instruction changed");
}

std::string serialize(const rt::BinaryProgram& program)
{
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    rt::write_binary_program(program, stream);
    return stream.str();
}

std::span<const std::uint8_t> bytes(const std::string& storage)
{
    return {reinterpret_cast<const std::uint8_t*>(storage.data()),
            storage.size()};
}

std::uint32_t low_u32(const std::vector<std::uint8_t>& storage)
{
    expect(storage.size() >= 4, "physical i-MEM image has no control word");
    return static_cast<std::uint32_t>(storage[0]) |
           (static_cast<std::uint32_t>(storage[1]) << 8) |
           (static_cast<std::uint32_t>(storage[2]) << 16) |
           (static_cast<std::uint32_t>(storage[3]) << 24);
}

std::uint64_t packed_bits(const std::vector<std::uint8_t>& storage,
                          std::uint64_t offset,
                          unsigned width)
{
    std::uint64_t result = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto position = offset + bit;
        expect(position < storage.size() * 8,
               "physical i-MEM bit read is out of range");
        result |= static_cast<std::uint64_t>(
                      (storage[position / 8] >> (position & 7u)) & 1u)
                  << bit;
    }
    return result;
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
        if (queue.commands.empty())
            continue;
        expect(queue.kind == rt::QueueKind::Mem && memQueue == nullptr,
               "Macro CModel A/B fixture must contain one MEM queue");
        memQueue = &queue;
    }
    expect(memQueue != nullptr, "Macro CModel A/B fixture has no MEM queue");

    ftlpu::InstructionControlUnit icu;
    rt::load_queue_programs_into_icu(program.queues, icu,
                                     program.hardware.mxms_per_hemisphere);
    CmodelMemRun result;
    auto& queue = icu.mem_iq(memQueue->index);
    for (std::size_t cycle = 0; cycle <= program.max_cycle + 64; ++cycle) {
        if (const auto issued = queue.tick())
            result.issues.emplace_back(
                cycle, ftlpu::isa::encode_mem_instruction(*issued));
        if (queue.done())
            break;
    }
    expect(queue.done(), "Macro CModel A/B queue did not complete");
    result.frontend = icu.frontend_statistics();
    return result;
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc > 3)
        throw std::runtime_error("usage: macro_bitstream_test [program.ftlpu | "
                                 "none.ftlpu macro.ftlpu]");
    rt::QueueProgram queue{rt::QueueKind::Mem, 3, {}};
    queue.commands.push_back(
        macro(ftlpu::MemInstruction::Read(100, 2),
              {10, 1, 1, 0, 4, 7, 3, ftlpu::IcuInductionTarget::MemAddress}));
    queue.commands.push_back(
        macro(ftlpu::MemInstruction::Read(112, 2),
              {42, 1, 1, 0, 4, 7, 3, ftlpu::IcuInductionTarget::MemAddress}));
    queue.commands.push_back(
        macro(ftlpu::MemInstruction::Read(124, 2),
              {74, 1, 1, 0, 4, 7, 3, ftlpu::IcuInductionTarget::MemAddress}));
    queue.commands.push_back(macro(
        ftlpu::MemInstruction::WriteTap(88, 5),
        {106, 8, 2, -1, 3, 23, 4, ftlpu::IcuInductionTarget::MemAddress}));
    // Forces both the wide delta escape and the extended template path.
    queue.commands.push_back(macro(
        ftlpu::MemInstruction::Write(90, 6),
        {5000000, 4097, 3, 2, 1, 1, 0, ftlpu::IcuInductionTarget::MemAddress}));

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

    const auto packedMem = rt::pack_mem_macro_imem(image);
    const auto memControl =
        rt::decode_icu_queue_control(low_u32(packedMem.bytes));
    expect(memControl.valid && memControl.enable,
           "MEM physical control word did not enable a valid queue");
    expect(memControl.mode == rt::IcuQueueMode::Macro,
           "MEM physical control word did not select Macro mode");
    expect(memControl.command_count == queue.commands.size(),
           "MEM physical control word has the wrong command count");
    expect(packedMem.word_bits == rt::kMemIcuImemWordBits,
           "MEM physical image has the wrong word width");
    expect(packedMem.software_valid_bit_length ==
               rt::kMemIcuImemWordBits + image.stats.physical_bits(),
           "MEM software valid bit length changed");
    expect(packedMem.word_count() ==
               1 + (image.stats.physical_bits() +
                    rt::kMemIcuImemWordBits - 1) /
                       rt::kMemIcuImemWordBits,
           "MEM physical image did not reserve word 0");
    expect(packed_bits(packedMem.bytes, rt::kMemIcuImemWordBits, 3) ==
               image.delta_count,
           "MEM dictionary count is not the first payload field");
    const auto decodedPackedMem =
        rt::decode_mem_macro_imem(packedMem, queue.index);
    expect(decodedPackedMem.commands.size() == queue.commands.size(),
           "MEM physical i-MEM command count changed");
    for (std::size_t i = 0; i < queue.commands.size(); ++i)
        compare(queue.commands[i], decodedPackedMem.commands[i]);

    expect_throw(
        []() {
            (void)rt::encode_icu_queue_control(
                {true,
                 true,
                 rt::IcuQueueMode::Macro,
                 rt::kIcuQueueControlMaxCommandCount + 1});
        },
        "oversized physical command count was accepted");
    auto invalidPackedMem = rt::pack_mem_macro_imem(image, true, false);
    expect_throw(
        [&]() { (void)rt::decode_mem_macro_imem(invalidPackedMem, queue.index); },
        "invalid MEM physical queue was decoded");

    std::vector<rt::QueueProgram> mxmQueues;
    mxmQueues.push_back(
        {rt::QueueKind::MxmLoad,
         0,
         {
             macro(ftlpu::MxmControlInstruction::IW(0, 1),
                   {10, 4, 2, 1, 2, 11, 2,
                    ftlpu::IcuInductionTarget::MxmWeightColumn}),
             macro(ftlpu::MxmControlInstruction::IW(0, 3),
                   {42, 4, 2, 1, 2, 11, 2,
                    ftlpu::IcuInductionTarget::MxmWeightColumn}),
         }});
    mxmQueues.push_back(
        {rt::QueueKind::MxmCompute,
         0,
         {
             macro(ftlpu::MxmControlInstruction::Compute(0, 0, 0, 100),
                   {20, 8, 3, 1, 4, 29, 16,
                    ftlpu::IcuInductionTarget::MxmAccumulatorAddress}),
             macro(ftlpu::MxmControlInstruction::Compute(0, 0, 0, 116),
                   {84, 8, 3, 1, 4, 29, 16,
                    ftlpu::IcuInductionTarget::MxmAccumulatorAddress}),
             macro(ftlpu::MxmControlInstruction::AccumulatorRead(200, 4),
                   {5000000, 4097, 3, 1, 1, 1, 0,
                    ftlpu::IcuInductionTarget::MxmAccumulatorAddress}),
         }});
    mxmQueues.push_back(
        {rt::QueueKind::MxmDequant,
         0,
         {
             macro(ftlpu::MxmDequantInstruction::ScaleBits(0x3f80),
                   {5, 16, 2, 0, 1, 1, 0, ftlpu::IcuInductionTarget::None}),
             macro(ftlpu::MxmDequantInstruction::ScaleBits(0x3f80),
                   {69, 16, 2, 0, 1, 1, 0, ftlpu::IcuInductionTarget::None}),
         }});
    bool sawMxmCompact = false;
    bool sawMxmExtended = false;
    bool sawMxmWideDelta = false;
    for (const auto& mxmQueue : mxmQueues) {
        const auto encoded = rt::encode_mxm_macro_bitstream(mxmQueue);
        sawMxmCompact |= encoded.stats.compact_template_runs != 0;
        sawMxmExtended |= encoded.stats.extended_template_runs != 0;
        sawMxmWideDelta |= encoded.stats.wide_escaped_transitions != 0;
        const auto restored =
            rt::decode_mxm_macro_bitstream(encoded, mxmQueue.index);
        expect(restored.kind == mxmQueue.kind, "MXM queue kind changed");
        expect(restored.commands.size() == mxmQueue.commands.size(),
               "MXM command count changed");
        for (std::size_t i = 0; i < mxmQueue.commands.size(); ++i)
            compare_mxm(mxmQueue.commands[i], restored.commands[i]);

        const auto packed = rt::pack_mxm_macro_imem(encoded);
        const auto control =
            rt::decode_icu_queue_control(low_u32(packed.bytes));
        expect(control.valid && control.enable &&
                   control.mode == rt::IcuQueueMode::Macro,
               "MXM physical control word is invalid");
        expect(control.command_count == mxmQueue.commands.size(),
               "MXM physical control word has the wrong command count");
        expect(packed.word_bits == rt::kMxmIcuImemWordBits,
               "MXM physical image has the wrong word width");
        expect(packed.software_valid_bit_length ==
                   rt::kMxmIcuImemWordBits + encoded.stats.physical_bits(),
               "MXM software valid bit length changed");
        expect(packed_bits(packed.bytes, rt::kMxmIcuImemWordBits, 3) ==
                   encoded.delta_count,
               "MXM dictionary count is not the first payload field");
        const auto restoredPacked =
            rt::decode_mxm_macro_imem(packed, mxmQueue.kind, mxmQueue.index);
        expect(restoredPacked.commands.size() == mxmQueue.commands.size(),
               "MXM physical i-MEM command count changed");
        for (std::size_t i = 0; i < mxmQueue.commands.size(); ++i)
            compare_mxm(mxmQueue.commands[i], restoredPacked.commands[i]);
    }
    expect(sawMxmCompact, "compact MXM template was not exercised");
    expect(sawMxmExtended, "extended MXM template was not exercised");
    expect(sawMxmWideDelta, "wide MXM delta escape was not exercised");

    // Prove that all three MXM queue kinds use the packed container path, not
    // only the in-memory reference codec or the legacy per-record envelope.
    rt::BinaryProgram emptyProgram;
    rt::BinaryProgram packedMxmProgram;
    packedMxmProgram.queues = mxmQueues;
    const auto emptyBinary = serialize(emptyProgram);
    const auto packedMxmBinary = serialize(packedMxmProgram);
    std::size_t expectedPackedQueueBytes = 0;
    for (const auto& mxmQueue : mxmQueues) {
        const auto encoded = rt::encode_mxm_macro_bitstream(mxmQueue);
        expectedPackedQueueBytes += 9 + 10 + encoded.delta_count * 8 +
                                    encoded.bytes.size();
    }
    expect(packedMxmBinary.size() - emptyBinary.size() ==
               expectedPackedQueueBytes,
           "MXM Macro queues were not serialized as packed images");
    constexpr std::size_t kQueueModeOffset = 4;
    expect(static_cast<std::uint8_t>(
               packedMxmBinary[emptyBinary.size() + kQueueModeOffset]) == 3,
           "MXM Macro queue did not select the packed container mode");

    std::stringstream packedMxmStream(
        packedMxmBinary, std::ios::in | std::ios::binary);
    const auto restoredMxmStream = rt::read_binary_program(packedMxmStream);
    const auto restoredMxmSpan = rt::read_binary_program(bytes(packedMxmBinary));
    for (const auto* restored : {&restoredMxmStream, &restoredMxmSpan}) {
        expect(restored->queues.size() == mxmQueues.size(),
               "packed MXM binary queue count changed");
        for (std::size_t queueIndex = 0; queueIndex < mxmQueues.size();
             ++queueIndex) {
            const auto& expectedQueue = mxmQueues[queueIndex];
            const auto& actualQueue = restored->queues[queueIndex];
            expect(actualQueue.kind == expectedQueue.kind,
                   "packed MXM binary queue kind changed");
            expect(actualQueue.commands.size() == expectedQueue.commands.size(),
                   "packed MXM binary command count changed");
            for (std::size_t commandIndex = 0;
                 commandIndex < expectedQueue.commands.size(); ++commandIndex)
                compare_mxm(expectedQueue.commands[commandIndex],
                            actualQueue.commands[commandIndex]);
        }
    }
    std::stringstream packedMxmMetadataStream(
        packedMxmBinary, std::ios::in | std::ios::binary);
    (void)rt::read_binary_program_metadata(packedMxmMetadataStream);
    (void)rt::read_binary_program_metadata(bytes(packedMxmBinary));

    auto unsupportedCodecBinary = packedMxmBinary;
    constexpr std::size_t kPackedCodecVersionOffset = 9;
    unsupportedCodecBinary[emptyBinary.size() + kPackedCodecVersionOffset] = 2;
    expect_throw(
        [&]() { (void)rt::read_binary_program(bytes(unsupportedCodecBinary)); },
        "unsupported MXM Macro codec version was accepted");
    auto truncatedMxmBinary = packedMxmBinary;
    truncatedMxmBinary.pop_back();
    expect_throw(
        [&]() { (void)rt::read_binary_program(bytes(truncatedMxmBinary)); },
        "truncated MXM Macro packed image was accepted");

    // Exercise the complete binary -> runtime loader -> finite CModel Macro
    // context path for both supported Macro v1 functional-unit families.
    rt::BinaryProgram executionProgram;
    executionProgram.max_cycle = 13;
    executionProgram.queues.push_back(rt::QueueProgram{
        rt::QueueKind::Mem,
        0,
        {
            macro(ftlpu::MemInstruction::Read(100, 0),
                  {2, 3, 4, 1, 1, 1, 0, ftlpu::IcuInductionTarget::MemAddress}),
            macro(ftlpu::MemInstruction::Read(200, 1),
                  {3, 3, 4, 1, 1, 1, 0, ftlpu::IcuInductionTarget::MemAddress}),
        }});
    executionProgram.queues.push_back(rt::QueueProgram{
        rt::QueueKind::MxmCompute,
        0,
        {macro(ftlpu::MxmControlInstruction::Compute(0, 0, 0, 10),
               {2, 2, 3, 1, 2, 8, 10,
                ftlpu::IcuInductionTarget::MxmAccumulatorAddress})}});

    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    rt::write_binary_program(executionProgram, binary);
    binary.seekg(0);
    const auto restoredProgram = rt::read_binary_program(binary);
    for (const auto& restoredQueue : restoredProgram.queues)
        for (const auto& command : restoredQueue.commands) {
            expect(rt::is_macro_schedule_command(command),
                   "2-D Macro changed representation during binary round-trip");
            expect(!rt::is_mem_stream_nd_command(command) &&
                       !rt::is_mxm_stream_nd_command(command),
                   "2-D Macro unexpectedly became STREAM_ND");
        }

    ftlpu::InstructionControlUnit icu;
    rt::load_queue_programs_into_icu(restoredProgram.queues, icu);
    std::vector<std::pair<std::size_t, std::size_t>> memIssues;
    std::vector<std::pair<std::size_t, std::size_t>> mxmIssues;
    for (std::size_t cycle = 0; cycle <= executionProgram.max_cycle; ++cycle) {
        if (const auto issued = icu.mem_iq(0).tick())
            memIssues.emplace_back(cycle, issued->address);
        if (const auto issued = icu.mxm_compute_iq(0).tick())
            mxmIssues.emplace_back(cycle, issued->accumulator_address);
    }
    expect(
        memIssues ==
            std::vector<std::pair<std::size_t, std::size_t>>{
                {2, 100}, {3, 200}, {6, 101}, {7, 201}, {10, 102}, {11, 202}},
        "runtime/CModel MEM Macro emitted incorrect cycles or addresses");
    expect(mxmIssues ==
               std::vector<std::pair<std::size_t, std::size_t>>{
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
        std::size_t memQueues = 0;
        std::size_t memCommands = 0;
        std::uint64_t memBits = 0;
        std::size_t mxmQueues = 0;
        std::size_t mxmCommands = 0;
        std::uint64_t mxmBits = 0;
        std::size_t memWords = 0;
        std::size_t mxmWords = 0;
        for (const auto& source : program.queues) {
            if (source.commands.empty())
                continue;
            const bool allMacro =
                std::all_of(source.commands.begin(), source.commands.end(),
                            [](const rt::QueueCommand& command) {
                                return rt::is_macro_schedule_command(command);
                            });
            if (!allMacro)
                continue;
            if (source.kind == rt::QueueKind::Mem) {
                const auto encoded = rt::encode_mem_macro_bitstream(source);
                const auto packed = rt::pack_mem_macro_imem(encoded);
                const auto restored =
                    rt::decode_mem_macro_imem(packed, source.index);
                expect(restored.commands.size() == source.commands.size(),
                       "Qwen MEM roundtrip command count changed");
                for (std::size_t i = 0; i < source.commands.size(); ++i)
                    compare(source.commands[i], restored.commands[i]);
                ++memQueues;
                memCommands += source.commands.size();
                memBits += encoded.stats.physical_bits();
                memWords += packed.word_count();
                continue;
            }
            if (source.kind == rt::QueueKind::MxmLoad ||
                source.kind == rt::QueueKind::MxmCompute ||
                source.kind == rt::QueueKind::MxmDequant) {
                const auto encoded = rt::encode_mxm_macro_bitstream(source);
                const auto packed = rt::pack_mxm_macro_imem(encoded);
                const auto restored = rt::decode_mxm_macro_imem(
                    packed, source.kind, source.index);
                expect(restored.commands.size() == source.commands.size(),
                       "Qwen MXM roundtrip command count changed");
                for (std::size_t i = 0; i < source.commands.size(); ++i)
                    compare_mxm(source.commands[i], restored.commands[i]);
                ++mxmQueues;
                mxmCommands += source.commands.size();
                mxmBits += encoded.stats.physical_bits();
                mxmWords += packed.word_count();
            }
        }
        std::cout << "macro_bitstream_qwen_roundtrip passed: mem_queues="
                  << memQueues << " mem_commands=" << memCommands
                  << " mem_payload_bits=" << memBits
                  << " mem_physical_words=" << memWords
                  << " mxm_queues=" << mxmQueues
                  << " mxm_commands=" << mxmCommands << " mxm_bits=" << mxmBits
                  << " mxm_physical_words=" << mxmWords
                  << '\n';
    }
    if (argc == 3) {
        const auto baseline = rt::read_binary_program(argv[1]);
        const auto compressed = rt::read_binary_program(argv[2]);
        expect(baseline.max_cycle == compressed.max_cycle,
               "compiler-generated Macro changed the scheduled cycle count");
        std::size_t macroCommands = 0;
        for (const auto& compressedQueue : compressed.queues)
            for (const auto& command : compressedQueue.commands) {
                expect(
                    !rt::is_mem_stream_nd_command(command) &&
                        !rt::is_mxm_stream_nd_command(command),
                    "compiler-generated Macro A/B binary contains STREAM_ND");
                macroCommands += rt::is_macro_schedule_command(command) ? 1 : 0;
            }
        expect(macroCommands != 0,
               "compiler-generated Macro A/B binary contains no 2-D Macro");

        const auto baselineRun = run_compiled_mem_program(baseline);
        const auto compressedRun = run_compiled_mem_program(compressed);
        expect(baselineRun.issues == compressedRun.issues,
               "compiler-generated Macro changed CModel issue semantics");
        expect(baselineRun.frontend.issued_instructions ==
                   compressedRun.frontend.issued_instructions,
               "compiler-generated Macro changed dynamic issue count");
        expect(compressedRun.frontend.imem_entries <
                   baselineRun.frontend.imem_entries,
               "compiler-generated Macro did not reduce CModel i-MEM entries");
        expect(compressedRun.frontend.fetched_entries <
                   baselineRun.frontend.fetched_entries,
               "compiler-generated Macro did not reduce CModel fetch entries");
        expect(compressedRun.frontend.macro_queues != 0 &&
                   compressedRun.frontend.peak_macro_contexts_per_queue != 0,
               "CModel did not execute compiler-generated Macro contexts");

        const auto baselineImem = rt::analyze_physical_imem(baseline);
        const auto compressedImem = rt::analyze_physical_imem(compressed);
        expect(compressedImem.used_bits < baselineImem.used_bits,
               "compiler-generated Macro did not reduce physical MEM bits");
        std::cout << "macro_cmodel_ab_test passed"
                  << " logical_issues=" << compressedRun.issues.size()
                  << " scheduled_cycles=" << compressed.max_cycle + 1
                  << " baseline_imem_entries="
                  << baselineRun.frontend.imem_entries << " macro_imem_entries="
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
