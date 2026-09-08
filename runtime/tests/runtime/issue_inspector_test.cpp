#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

ftlpu::software::runtime::QueueCommand mem_command(
    const ftlpu::MemInstruction& instruction)
{
    using namespace ftlpu::software::runtime;
    const auto encoded = ftlpu::isa::encode_mem_instruction(instruction);
    return QueueCommand {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0}};
}

} // namespace

int main()
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;

    BinaryProgram sliced;
    sliced.max_cycle = 7;
    sliced.queues.push_back(QueueProgram {QueueKind::Mem, 0,
        {encode_mem_slice_program_command(IcuMemSliceProgram {
            IcuStreamNdSchedule {2, 1, {3, 1, 1}, {2, 1, 1},
                {0, 0, 0}, IcuInductionTarget::None},
            {
                IcuMemSliceProgramEntry {0, {1, 0, 0},
                    MemInstruction::Read(100, 0)},
                IcuMemSliceProgramEntry {1, {2, 0, 0},
                    MemInstruction::Write(200, 1)},
            }})}});

    BinaryProgram streams;
    streams.max_cycle = sliced.max_cycle;
    streams.queues.push_back(QueueProgram {QueueKind::Mem, 0, {
        encode_mem_stream_nd_command(mem_command(MemInstruction::Read(100, 0)),
            IcuMemStreamNdSchedule {2, 1, {3, 1, 1}, {2, 1, 1},
                {1, 0, 0}, IcuInductionTarget::MemAddress}),
        encode_mem_stream_nd_command(mem_command(MemInstruction::Write(200, 1)),
            IcuMemStreamNdSchedule {3, 1, {3, 1, 1}, {2, 1, 1},
                {2, 0, 0}, IcuInductionTarget::MemAddress}),
    }});

    const auto inspection = inspect_logical_issues(sliced);
    require(inspection.functional_issues == 6
            && inspection.logical_nop_cycles == 2,
        "logical inspector did not count functional and NOP cycles");
    require(compare_logical_issues(sliced, streams).equivalent,
        "MEM_SLICE_PROGRAM differs from equivalent MEM_STREAM_ND commands");
    sliced.hardware.icu_mem_macro_contexts = 2;
    sliced.hardware.icu_mem_macro_context_bits = 256;
    streams.hardware = sliced.hardware;
    const auto slicedCapacity = analyze_physical_imem(sliced);
    const auto streamCapacity = analyze_physical_imem(streams);
    require(slicedCapacity.queues[0].peak_macro_contexts == 2
            && streamCapacity.queues[0].peak_macro_contexts == 2,
        "physical inspector did not count each active MEM stream context");

    BinaryProgram control;
    control.max_cycle = 4;
    control.queues.push_back(QueueProgram {QueueKind::Mem, 0, {
        mem_command(MemInstruction::Read(10, 0)),
        QueueCommand {isa::encode_icu_repeat(IcuRepeat {2, 2, 1})},
    }});
    BinaryProgram macro;
    macro.max_cycle = control.max_cycle;
    macro.queues.push_back(QueueProgram {QueueKind::Mem, 0, {
        encode_macro_schedule_command(
            mem_command(MemInstruction::Read(10, 0)),
            IcuMacroSchedule {0, 3, 2, 1, 1, 1, 0,
                IcuInductionTarget::MemAddress}),
    }});
    require(compare_logical_issues(control, macro).equivalent,
        "logical inspector disagrees between Repeat and Macro forms");
    const auto controlInspection = inspect_logical_issues(control);
    require(controlInspection.functional_issues == 3
            && controlInspection.logical_nop_cycles == 2,
        "logical inspector did not materialize Repeat gaps as logical NOPs");

    streams.queues[0].commands[1] = encode_mem_stream_nd_command(
        mem_command(MemInstruction::Write(201, 1)),
        IcuMemStreamNdSchedule {3, 1, {3, 1, 1}, {2, 1, 1},
            {2, 0, 0}, IcuInductionTarget::MemAddress});
    const auto changed = compare_logical_issues(sliced, streams);
    require(!changed.equivalent && changed.first_mismatch
            && changed.first_mismatch->cycle == 3
            && changed.first_mismatch->reason
                == "functional instruction differs",
        "logical inspector missed an instruction mismatch");

    streams = sliced;
    streams.max_cycle = sliced.max_cycle + 1;
    const auto horizon = compare_logical_issues(sliced, streams);
    require(!horizon.equivalent && horizon.first_mismatch
            && horizon.first_mismatch->cycle == 8,
        "logical inspector missed a logical NOP horizon mismatch");

    std::cout << "issue_inspector_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "issue_inspector_test failed: " << error.what() << '\n';
    return 1;
}
