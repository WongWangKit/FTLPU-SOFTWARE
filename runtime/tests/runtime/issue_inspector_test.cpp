#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"

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

template <typename Fn>
void require_rejected(Fn fn, const char* message)
{
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

ftlpu::software::runtime::QueueCommand mxm_command(const ftlpu::MxmControlInstruction& instruction)
{
    using namespace ftlpu::software::runtime;
    const auto bits = ftlpu::isa::encode_mxm_instruction(instruction);
    return {0, InstructionKind::Mxm, 2,
        {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

void test_native_fu_issues()
{
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    BinaryProgram coarse;
    coarse.max_cycle = 84;
    IcuLoop3D loop{0, {2, 2, 4}, {2, 7, 20}, 3};
    auto packets = raw_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Read3D(loop,
            MemIcuAddress3D::BlockedOuter(100, 1, 4, 2, 16, 64),
            StreamId::East(2))), InstructionKind::Mem);
    QueueProgram queue{QueueKind::Mem, 0,
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(2))}};
    queue.commands.insert(queue.commands.end(), packets.begin(), packets.end());
    IcuLoop3D nextLoop{0, {1, 1, 1}, {1, 1, 1}, 4};
    const auto next = raw_3d_commands(isa::encode_mem_icu_3d_instruction(
        MemIcuInstruction::Write3D(nextLoop, MemIcuAddress3D::Affine(300, {0, 0, 0}),
            StreamId::West(1))), InstructionKind::Mem);
    queue.commands.insert(queue.commands.end(), next.begin(), next.end());
    coarse.queues.push_back(queue);
    BinaryProgram macros;
    macros.max_cycle = coarse.max_cycle;
    macros.queues.push_back({QueueKind::Mem, 0, {}});
    for (std::size_t outer = 0; outer < 4; ++outer)
        macros.queues[0].commands.push_back(encode_macro_schedule_command(
            mem_command(MemInstruction::Read(100 + (outer / 2) * 64 + (outer % 2) * 16, 2)),
            IcuMacroSchedule{5 + outer * 20, 2, 2, 1, 2, 7, 4,
                IcuInductionTarget::MemAddress}));
    macros.queues[0].commands.push_back(encode_macro_schedule_command(
        mem_command(MemInstruction::Write(300, StreamId::West(1).packed())),
        IcuMacroSchedule{79, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress}));
    require(compare_logical_issues(coarse, macros).equivalent,
        "native MEM 3-D blocked addressing, relative waits, or packet cursor differs from Macro");
    auto changed = macros;
    changed.queues[0].commands[0].words = mem_command(MemInstruction::Read(101, 2)).words;
    const auto mismatch = compare_logical_issues(coarse, changed);
    require(!mismatch.equivalent && mismatch.first_mismatch
            && mismatch.first_mismatch->cycle == 5,
        "native FU versus Macro comparison missed an operand difference");
    changed = macros;
    auto schedule = decode_macro_schedule_command(changed.queues[0].commands[0]);
    schedule.start_cycle = 6;
    changed.queues[0].commands[0] = encode_macro_schedule_command(
        mem_command(MemInstruction::Read(100, 2)), schedule);
    require(!compare_logical_issues(coarse, changed).equivalent,
        "native FU versus Macro comparison missed a cycle difference");

    // A standalone physical native word has a two-bit ICU envelope.
    const auto bits = isa::encode_mem_instruction(MemInstruction::Read(319, 7));
    auto physical = queue;
    physical.commands.push_back(QueueCommand{static_cast<std::uint32_t>(bits << 2),
        InstructionKind::Mem, 3, {static_cast<std::uint32_t>(bits << 2),
            static_cast<std::uint32_t>(bits >> 30), static_cast<std::uint32_t>(bits >> 62), 0}});
    auto withNative = coarse;
    withNative.queues = {physical};
    macros.queues[0].commands.push_back(encode_macro_schedule_command(
        mem_command(MemInstruction::Read(319, 7)),
        IcuMacroSchedule{80, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress}));
    require(compare_logical_issues(withNative, macros).equivalent,
        "physical native envelope was confused with semantic FU bits");
    physical.commands.back().words[2] |= 4;
    withNative.queues = {physical};
    require_rejected([&] { (void)inspect_logical_issues(withNative); },
        "inspector accepted reserved physical native bits");

    IcuProgram expected;
    BinaryProgram mxm;
    mxm.max_cycle = 40;
    const IcuLoop3D loadLoop{0, {2, 2, 1}, {3, 9, 1}, 1};
    mxm.queues.push_back({QueueKind::MxmLoad, 0,
        raw_3d_commands(isa::encode_mxm_load_icu_3d_instruction(
            MxmLoadIcuInstruction::Load3D(loadLoop, 0,
                MxmIcuBufferMode::ToggleDimension0, 0, {1, 2, 0}, 8)), InstructionKind::Mxm)});
    for (std::size_t outer = 0; outer < 2; ++outer)
        for (std::size_t inner = 0; inner < 2; ++inner)
            expected.emit_mxm_load(1 + inner * 3 + outer * 9, 0,
                MxmControlInstruction::IW(inner, inner + outer * 2,
                    MxmWeightInputMode::Int8DequantBf16, 8));
    const IcuLoop3D computeLoop{0, {3, 2, 1}, {2, 9, 1}, 4};
    const MxmComputeIcuMode terminal{MxmAccumulatorDestination::Stream,
        true, MxmAccumulatorOutputFormat::BFloat16};
    mxm.queues.push_back({QueueKind::MxmCompute, 0,
        raw_3d_commands(isa::encode_mxm_compute_icu_3d_instruction(
            MxmComputeIcuInstruction::Compute3D(computeLoop, 0,
                MxmIcuBufferMode::ToggleDimension1, 16, 12, 10, {0, 4, 0}, 1,
                MxmDataFormat::BFloat16, MxmComputeIcuMode{}, 0, terminal)), InstructionKind::Mxm)});
    for (std::size_t outer = 0; outer < 2; ++outer)
        for (std::size_t inner = 0; inner < 3; ++inner) {
            const bool last = inner == 2;
            expected.emit_mxm_compute(4 + inner * 2 + outer * 9, 0,
                MxmControlInstruction::Compute(outer, 16, 12, 10 + outer * 4, 1,
                    last ? MxmAccumulatorDestination::Stream : MxmAccumulatorDestination::Sram,
                    MxmDataFormat::BFloat16, last,
                    last ? MxmAccumulatorOutputFormat::BFloat16 : MxmAccumulatorOutputFormat::Float32));
        }
    const auto scale = MxmDequantInstruction::ScaleBits(0x3f80);
    const IcuLoop3D dequantLoop{0, {2, 2, 1}, {3, 10, 1}, 2};
    mxm.queues.push_back({QueueKind::MxmDequant, 0,
        raw_3d_commands(isa::encode_mxm_dequant_icu_3d_instruction(
            MxmDequantIcuInstruction::Dequant3D(dequantLoop, scale)), InstructionKind::MxmDequant)});
    for (std::size_t outer = 0; outer < 2; ++outer)
        for (std::size_t inner = 0; inner < 2; ++inner)
            expected.emit_mxm_dequant(2 + inner * 3 + outer * 10, 0, scale);
    BinaryProgram expanded;
    expanded.max_cycle = mxm.max_cycle;
    expanded.queues = expected.encode_queues();
    require(compare_logical_issues(mxm, expanded).equivalent,
        "MXM parity, terminal compute mode, or dequant differs from native instructions");
    const auto dequantMacro = encode_macro_schedule_command(
        QueueCommand{0, InstructionKind::MxmDequant, 1,
            {static_cast<std::uint32_t>(isa::encode_mxm_dequant_instruction(scale)), 0, 0, 0}},
        IcuMacroSchedule{2, 2, 3, 0, 2, 10, 0, IcuInductionTarget::None});
    auto dequantOnly = mxm;
    dequantOnly.queues = {mxm.queues.back()};
    auto dequantEquivalent = dequantOnly;
    dequantEquivalent.queues[0].commands = {dequantMacro};
    require(compare_logical_issues(dequantOnly, dequantEquivalent).equivalent,
        "native DEQUANT_3D differs from equivalent zero-stride Macro");

    BinaryProgram rejected = coarse;
    rejected.max_cycle = 4;
    require_rejected([&] { (void)inspect_logical_issues(rejected); },
        "inspector accepted native issues beyond program max_cycle");
    rejected = coarse;
    rejected.queues[0].commands.resize(2);
    require_rejected([&] { (void)inspect_logical_issues(rejected); },
        "inspector accepted a truncated native FU packet");
    rejected = coarse;
    rejected.queues[0].commands.push_back(encode_macro_schedule_command(
        mem_command(MemInstruction::Read(100, 2)),
        IcuMacroSchedule{5, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress}));
    require_rejected([&] { (void)inspect_logical_issues(rejected); },
        "inspector accepted duplicate functional issue cycles");
    for (const auto control : {IcuControlInstruction::Sync(), IcuControlInstruction::WaitEvent(42),
             IcuControlInstruction::Notify()}) {
        rejected.queues = {{QueueKind::Mem, 0, {encode_icu_control_raw_word(control)}}};
        require_rejected([&] { (void)inspect_logical_issues(rejected); },
            "inspector silently ignored runtime-dependent native synchronization");
    }
}

void test_native_vxm_sxm_issues()
{
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    BinaryProgram packets;
    packets.max_cycle = 40;
    VxmLaneAluInstruction vxm;
    vxm.lhs = VxmLaneOperand::Imm(1.25f);
    vxm.rhs = VxmLaneOperand::Imm(1.25f);
    vxm.repeat_count = 4;
    const auto compact = isa::encode_vxm_instruction(0, VxmChainDepth::Two, vxm);
    auto vxmRun = VxmIcuRun2DInstruction::Run2D(0, {2, 2}, {3, 11}, compact);
    vxmRun.loop.wait_cycle = 2;
    packets.queues.push_back({QueueKind::Vxm, 0,
        raw_3d_commands(isa::encode_vxm_icu_run_2d_instruction(vxmRun), InstructionKind::Vxm)});
    IcuProgram expected;

    SxmInstruction::StreamList streams;
    for (std::size_t i = 0; i < 2 * hw::kLanesPerTile; ++i) streams.push_back({i});
    auto transpose = SxmInstruction::Transpose({SxmStreamId{2}, SxmStreamId{3}}, streams);
    transpose.input_row = 1;
    auto transposeRun = SxmIcuRun2DInstruction::Run2D(0, {2, 1}, {7, 1}, transpose);
    transposeRun.loop.wait_cycle = 1;
    packets.queues.push_back({QueueKind::SxmTranspose, 0,
        raw_3d_commands(isa::encode_sxm_transpose_icu_run_2d_instruction(transposeRun), InstructionKind::Sxm)});
    expected.emit_sxm_transpose(1, Hemisphere::East, transpose);
    expected.emit_sxm_transpose(8, Hemisphere::East, transpose);
    SxmInstruction::PermuteMap map;
    for (std::size_t i = 0; i < map.size(); ++i) map[i] = i;
    auto permute = SxmInstruction::Permute(streams, streams, map);
    permute.output_row = 11;
    permute.output_tile = 1;
    auto permuteRun = SxmIcuRun2DInstruction::Run2D(0, {2, 2}, {3, 11}, permute, hw::kLanesPerTile);
    permuteRun.loop.wait_cycle = 2;
    packets.queues.push_back({QueueKind::SxmPermute, 0,
        raw_3d_commands(isa::encode_sxm_permute_icu_run_2d_instruction(permuteRun), InstructionKind::Sxm)});
    for (std::size_t outer = 0; outer < 2; ++outer)
        for (std::size_t inner = 0; inner < 2; ++inner) {
            auto instruction = permute;
            const auto offset = (inner + 2 * outer) * hw::kLanesPerTile;
            for (auto& lane : instruction.permute_map)
                lane = (lane + offset) % SxmInstruction::kTotalLanes;
            expected.emit_sxm_permute(2 + inner * 3 + outer * 11,
                Hemisphere::East, std::move(instruction));
        }
    BinaryProgram native;
    native.max_cycle = packets.max_cycle;
    native.queues = expected.encode_queues();
    for (auto& queue : native.queues)
        if (queue.kind == QueueKind::Vxm && queue.index == 0)
            queue.commands = {encode_vxm_stream_nd_command(
                QueueCommand{0, InstructionKind::Vxm, 3,
                    {static_cast<std::uint32_t>(compact.control),
                        static_cast<std::uint32_t>(compact.control >> 32), compact.immediate_bits, 0}},
                IcuVxmStreamNdSchedule{2, 2, {2, 2, 1}, {3, 11, 1},
                    {0, 0, 0}, IcuInductionTarget::None})};
    require(compare_logical_issues(packets, native).equivalent,
        "native VXM/SXM RUN_2D differs from semantic native instruction encoding");
    const auto inspected = inspect_logical_issues(packets);
    require(inspected.functional_issues == 10,
        "VXM FU repeat_count was confused with ICU launch count");
}

} // namespace

int main()
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;

    test_native_fu_issues();
    test_native_vxm_sxm_issues();

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
    const bool packedMacroTarget = macro.hardware.icu_mem_instruction_bits
            == kMemIcuImemWordBits
        && macro.hardware.icu_macro_encoding_version == kMemMacroBitstreamVersion;
    if (packedMacroTarget) {
        // A singleton descriptor uses one 96-bit control word plus one packed
        // payload word (dictionary count, absolute origin, and compact 1-D
        // template). The three functional issues share one Macro context.
        require(macroCapacity.queues[0].physical_slots == 2
                && macroCapacity.queues[0].physical_bits == 192
                && macroCapacity.used_bits == 192
                && macroCapacity.queues[0].peak_macro_contexts == 1
                && macroCapacity.queues[0].peak_fu_3d_contexts == 0
                && macroCapacity.queues[0].mem_delta_rle
                && macroCapacity.mem_delta_rle_queues == 1
                && macroCapacity.queues[0].macro_codec.command_count == 1
                && macroCapacity.queues[0].macro_codec.run_count == 1
                && macroCapacity.queues[0].macro_codec.compact_template_runs == 1
                && !macroCapacity.queues[0].deployment_overflow(),
            "physical inspector did not account for the 24-byte packed MEM Macro image");
    } else {
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
            "unsupported packed-Macro target did not retain legacy FU packet accounting");
    }

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
    require_rejected([&]{(void)inspect_logical_issues(control,2);},
        "bounded analysis expanded a queue beyond its issue capacity");
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
