#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

template <typename Packet>
std::vector<ftlpu::software::runtime::QueueCommand> raw_3d_commands(
    const Packet& packet,
    ftlpu::software::runtime::InstructionKind instructionKind)
{
    using namespace ftlpu::software::runtime;
    std::vector<QueueCommand> commands;
    commands.reserve(Packet::kWordCount);
    for (const auto& physicalWord : packet.words) {
        QueueCommand command;
        command.command = physicalWord.lanes[0];
        command.instruction_kind = instructionKind;
        command.word_count = Packet::kLanesPerWord;
        for (std::size_t lane = 0; lane < Packet::kLanesPerWord; ++lane)
            command.words[lane] = physicalWord.lanes[lane];
        commands.push_back(std::move(command));
    }
    return commands;
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
    require(slicedCapacity.queues[0].peak_fu_3d_contexts == 0
            && streamCapacity.queues[0].peak_fu_3d_contexts == 2
            && streamCapacity.queues[0].fu_3d_context_capacity
                == hw::kIcuMem3DContextDepth
            && streamCapacity.queues[0].fu_3d_context_bits
                == hw::kIcuMem3DContextBits
            && streamCapacity.queues[0].fu_3d_context_overflow()
            && streamCapacity.fu_3d_context_overflow_queues == 1
            && !streamCapacity.fits(),
        "physical inspector accepted overlapping MEM coarse instructions on a single-context ICU");
    require(streamCapacity.stream_nd_packets == 2
            && streamCapacity.queues[0].stream_nd_packets == 2
            && streamCapacity.queues[0].physical_bits
                == 2 * isa::EncodedMemIcu3DPacket::kWordCount
                    * streams.hardware.icu_mem_instruction_bits
            && streamCapacity.queues[0].physical_slots
                == 2 * isa::EncodedMemIcu3DPacket::kWordCount,
        "physical inspector did not account for FU-specific STREAM_ND packet words");

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
    const auto macroCapacity = analyze_physical_imem(macro);
    require(macroCapacity.queues[0].physical_slots
                == isa::EncodedMemIcu3DPacket::kWordCount
            && macroCapacity.queues[0].physical_bits
                == isa::EncodedMemIcu3DPacket::kWordCount
                    * macro.hardware.icu_mem_instruction_bits
            && macroCapacity.queues[0].peak_macro_contexts == 1
            && macroCapacity.queues[0].peak_fu_3d_contexts == 1
            && !macroCapacity.queues[0].fu_3d_context_overflow()
            && !macroCapacity.queues[0].mem_delta_rle
            && macroCapacity.mem_delta_rle_queues == 0,
        "physical inspector did not count an all-Macro MEM queue as raw 3-D packets");

    const IcuLoop3D rawLoop {0, {1, 1, 1}, {1, 1, 1}};
    BinaryProgram rawTypes;
    rawTypes.max_cycle = 10;
    rawTypes.queues.push_back(QueueProgram {QueueKind::Mem, 0,
        raw_3d_commands(isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Read3D(rawLoop,
                MemIcuAddress3D::Affine(0, {0, 0, 0}),
                StreamId::East(0))), InstructionKind::Mem)});
    rawTypes.queues.push_back(QueueProgram {QueueKind::MxmLoad, 0,
        raw_3d_commands(isa::encode_mxm_load_icu_3d_instruction(
            MxmLoadIcuInstruction::Load3D(rawLoop, 0,
                MxmIcuBufferMode::Fixed, 0, {0, 0, 0}, 8)),
            InstructionKind::Mxm)});
    rawTypes.queues.push_back(QueueProgram {QueueKind::MxmDequant, 0,
        raw_3d_commands(isa::encode_mxm_dequant_icu_3d_instruction(
            MxmDequantIcuInstruction::Dequant3D(rawLoop,
                MxmDequantInstruction::ScaleBits(0x3c00))),
            InstructionKind::MxmDequant)});
    rawTypes.queues.push_back(QueueProgram {QueueKind::MxmCompute, 0,
        raw_3d_commands(isa::encode_mxm_compute_icu_3d_instruction(
            MxmComputeIcuInstruction::Compute3D(rawLoop, 0,
                MxmIcuBufferMode::Fixed, 16, 12, 32, {0, 0, 0}, 1,
                MxmDataFormat::BFloat16, MxmComputeIcuMode {})),
            InstructionKind::Mxm)});
    const auto rawTypeCapacity = analyze_physical_imem(rawTypes);
    const std::array<std::uint32_t, 4> expectedContextBits {
        static_cast<std::uint32_t>(hw::kIcuMem3DContextBits),
        static_cast<std::uint32_t>(hw::kIcuMxmLoad3DContextBits),
        static_cast<std::uint32_t>(hw::kIcuMxmDequant3DContextBits),
        static_cast<std::uint32_t>(hw::kIcuMxmCompute3DContextBits),
    };
    const std::array<std::uint32_t, 4> expectedContextDepths {
        static_cast<std::uint32_t>(hw::kIcuMem3DContextDepth),
        static_cast<std::uint32_t>(hw::kIcuMxmLoad3DContextDepth),
        static_cast<std::uint32_t>(hw::kIcuMxmDequant3DContextDepth),
        static_cast<std::uint32_t>(hw::kIcuMxmCompute3DContextDepth),
    };
    require(rawTypeCapacity.queues.size() == expectedContextBits.size(),
        "raw FU 3-D context fixture lost a physical queue");
    for (std::size_t index = 0; index < expectedContextBits.size(); ++index)
        require(rawTypeCapacity.queues[index].peak_macro_contexts == 0
                && rawTypeCapacity.queues[index].peak_fu_3d_contexts == 1
                && rawTypeCapacity.queues[index].fu_3d_context_capacity
                    == expectedContextDepths[index]
                && rawTypeCapacity.queues[index].fu_3d_context_bits
                    == expectedContextBits[index]
                && !rawTypeCapacity.queues[index].fu_3d_context_overflow(),
            "raw FU 3-D context accounting used the wrong queue-specific hardware geometry");

    BinaryProgram sequentialRaw;
    constexpr std::uint32_t kSequentialMemPackets =
        static_cast<std::uint32_t>(hw::kIcuMem3DContextDepth) + 1;
    sequentialRaw.max_cycle = 256 * kSequentialMemPackets;
    sequentialRaw.queues.push_back(QueueProgram {QueueKind::Mem, 0, {}});
    for (std::uint32_t index = 0; index < kSequentialMemPackets; ++index) {
        const auto packet = raw_3d_commands(
            isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Read3D(
                IcuLoop3D {0, {16, 1, 1}, {16, 1, 1}},
                MemIcuAddress3D::Affine(16 * index, {1, 0, 0}),
                StreamId::East(static_cast<std::uint8_t>(index % 8)))),
            InstructionKind::Mem);
        sequentialRaw.queues[0].commands.insert(
            sequentialRaw.queues[0].commands.end(),
            packet.begin(), packet.end());
    }
    const auto sequentialRawCapacity =
        analyze_physical_imem(sequentialRaw);
    require(sequentialRawCapacity.queues[0].peak_macro_contexts == 0
            && sequentialRawCapacity.queues[0].peak_fu_3d_contexts == 1
            && !sequentialRawCapacity.queues[0].fu_3d_context_overflow()
            && sequentialRawCapacity.fu_3d_context_overflow_queues == 0
            && sequentialRawCapacity.overflow_queues == 0
            && sequentialRawCapacity.fits(),
        "sequential raw FU 3-D packets were treated as overlapping contexts");
    {
        auto icu = std::make_unique<InstructionControlUnit>();
        load_queue_programs_into_icu(sequentialRaw.queues, *icu);
    }

    BinaryProgram adjacentRaw;
    adjacentRaw.max_cycle = 15;
    const auto firstRaw = raw_3d_commands(
        isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Read3D(
            IcuLoop3D {0, {2, 1, 1}, {2, 1, 1}},
            MemIcuAddress3D::Affine(0, {1, 0, 0}),
            StreamId::East(0))), InstructionKind::Mem);
    adjacentRaw.queues.push_back(QueueProgram {QueueKind::Mem, 0,
        {QueueCommand {isa::encode_icu_nop(10)}}});
    adjacentRaw.queues[0].commands.insert(
        adjacentRaw.queues[0].commands.end(),
        firstRaw.begin(), firstRaw.end());
    const auto adjacentSecondRaw = raw_3d_commands(
        isa::encode_mem_icu_3d_instruction(MemIcuInstruction::Read3D(
            IcuLoop3D {0, {2, 1, 1}, {2, 1, 1}},
            MemIcuAddress3D::Affine(16, {1, 0, 0}),
            StreamId::East(1))), InstructionKind::Mem);
    adjacentRaw.queues[0].commands.insert(
        adjacentRaw.queues[0].commands.end(),
        adjacentSecondRaw.begin(), adjacentSecondRaw.end());
    {
        auto icu = std::make_unique<InstructionControlUnit>();
        load_queue_programs_into_icu(adjacentRaw.queues, *icu);
    }
    {
        auto icu = std::make_unique<InstructionControlUnit>();
        load_queue_programs_into_icu(rawTypes.queues, *icu);
    }
    auto mixedMacro = macro;
    mixedMacro.queues[0].commands.push_back(
        QueueCommand {isa::encode_icu_nop(1)});
    const auto mixedMacroCapacity = analyze_physical_imem(mixedMacro);
    require(mixedMacroCapacity.queues[0].physical_slots
                == isa::EncodedMemIcu3DPacket::kWordCount + 1
            && mixedMacroCapacity.queues[0].physical_bits
                == (isa::EncodedMemIcu3DPacket::kWordCount + 1)
                    * mixedMacro.hardware.icu_mem_instruction_bits
            && mixedMacroCapacity.queues[0].stream_nd_packets == 0,
        "physical inspector did not count a mixed legacy MEM Macro as one raw 3-D packet");
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
