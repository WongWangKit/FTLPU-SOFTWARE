#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

BinaryBinding weight(
    std::uint32_t index, std::string name, std::uint64_t bytes,
    std::uint16_t slice)
{
    BinaryBinding binding;
    binding.index = index;
    binding.access = BindingAccess::Input;
    binding.role = "weight";
    binding.name = std::move(name);
    binding.byte_size = bytes;
    binding.hemisphere_mask = 3;
    binding.paged_weight = true;
    binding.page_count = 2;
    binding.page_rows = 8;
    binding.page_granularity = 1;
    binding.page_role_group_count = 1;
    binding.page_items_per_slice_group = 1;
    binding.page_bank_count = 2;
    binding.instruction_count = 8;
    binding.slices = {slice};
    binding.page_storage_slices = binding.slices;
    binding.page_banks = {0, 1};
    binding.page_slice_group_bases = {0, 0};
    binding.page_slice_group_counts = {1, 1};
    binding.page_base_rows = {0, 0};
    binding.page_row_counts = {8, 8};
    return binding;
}

void require_contains(const std::string& text, const std::string& expected)
{
    if (text.find(expected) == std::string::npos)
        throw std::runtime_error(
            "trace is missing: " + expected + "\nactual trace:\n" + text);
}

template <typename T>
void write_fixture_scalar(std::ostream& os, T value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void verify_official_v26_header_compatibility()
{
    constexpr char magic[8] {'F', 'T', 'L', 'P', 'U', 'B', '0', '1'};
    ExecutableHardwareConfig hardware;
    hardware.vxm_cross_hemisphere_streams_enabled = 0;
    hardware.vxm_fma_enabled = 1;
    const std::string targetName = "official-v26-fixture";

    std::ostringstream os(std::ios::out | std::ios::binary);
    os.write(magic, sizeof(magic));
    write_fixture_scalar<std::uint32_t>(os, 26);
    write_fixture_scalar<std::uint64_t>(os, executable_target_abi(hardware));
    write_fixture_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(targetName.size()));
    os.write(targetName.data(), static_cast<std::streamsize>(targetName.size()));
    hardware.visit_v26([&](std::uint32_t value) {
        write_fixture_scalar(os, value);
    });
    write_fixture_scalar<std::uint64_t>(os, 0);
    for (int count = 0; count < 7; ++count)
        write_fixture_scalar<std::uint32_t>(os, 0);

    const auto bytes = os.str();
    const auto decoded = read_binary_program(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
    if (decoded.target_name != targetName || !decoded.queues.empty()
        || decoded.hardware.vxm_cross_hemisphere_streams_enabled != 0
        || decoded.hardware.vxm_fma_enabled != 1)
        throw std::runtime_error("official v26 header did not round-trip");
}

QueueCommand mxm_command(const ftlpu::MxmControlInstruction& instruction)
{
    const auto encoded = ftlpu::isa::encode_mxm_instruction(instruction);
    return QueueCommand {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
}

QueueCommand mem_command(const ftlpu::MemInstruction& instruction)
{
    const auto encoded = ftlpu::isa::encode_mem_instruction(instruction);
    return QueueCommand {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
}

QueueCommand vxm_command(std::size_t queue,
    ftlpu::VxmChainDepth depth,
    const ftlpu::VxmLaneAluInstruction& instruction)
{
    const auto encoded =
        ftlpu::isa::encode_vxm_instruction(queue, depth, instruction);
    return QueueCommand {
        static_cast<ftlpu::isa::EncodedIcuCommand>(
            ftlpu::isa::IcuCommandOpcode::Instruction),
        InstructionKind::Vxm,
        3,
        {static_cast<std::uint32_t>(encoded.control),
            static_cast<std::uint32_t>(encoded.control >> 32),
            encoded.immediate_bits, 0},
    };
}

QueueCommand repeat_2d_command(const ftlpu::IcuRepeat2D& repeat)
{
    const auto encoded = ftlpu::isa::encode_icu_repeat_2d(repeat);
    return QueueCommand {
        encoded.words[0], InstructionKind::None, 3,
        {encoded.words[0], encoded.words[1], encoded.words[2], 0},
    };
}

template <typename Packet>
std::vector<QueueCommand> raw_fu_commands(
    const Packet& packet, InstructionKind instructionKind,
    std::size_t leadingNopCycles = 0)
{
    std::vector<QueueCommand> commands;
    commands.reserve(Packet::kWordCount + (leadingNopCycles != 0 ? 1 : 0));
    if (leadingNopCycles != 0)
        commands.push_back(
            QueueCommand{ftlpu::isa::encode_icu_nop(leadingNopCycles)});
    for (const auto& word : packet.words) {
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = instructionKind;
        command.word_count = Packet::kLanesPerWord;
        for (std::size_t lane = 0; lane < Packet::kLanesPerWord; ++lane)
            command.words[lane] = word.lanes[lane];
        commands.push_back(std::move(command));
    }
    return commands;
}

} // namespace

int main()
try {
    verify_official_v26_header_compatibility();
    bool rejectedInterleavedDimensions = false;
    try {
        static_cast<void>(encode_mxm_stream_nd_command(
            mxm_command(ftlpu::MxmControlInstruction::Compute(
                0, 0, 0, 4, 1,
                ftlpu::MxmAccumulatorDestination::Sram)),
            ftlpu::IcuMxmStreamNdSchedule {
                0, 3, {32, 13, 2}, {1, 3102, 1536}, {0, 0, 0},
                ftlpu::IcuInductionTarget::None}));
    } catch (const std::invalid_argument&) {
        rejectedInterleavedDimensions = true;
    }
    if (!rejectedInterleavedDimensions)
        throw std::runtime_error(
            "MXM_STREAM_ND accepted non-monotonic dimensions");

    BinaryProgram program;
    program.hardware.c2c_streams_per_direction = 8;
    program.hardware.c2c_bytes_per_stream_per_cycle = 32;
    program.hardware.lpu_clock_mhz = 500;
    program.hardware.ddr_peak_bandwidth_mbytes_per_second = 51200;
    program.hardware.ddr_scheduling_efficiency_percent = 90;
    program.hardware.ddr_read_latency_cycles = 35;
    program.hardware.ddr_read_latency_jitter_cycles = 15;
    program.hardware.icu_mem_imem_depth = 6;
    program.bindings = {
        weight(1, "gate", 2048, 20),
        weight(2, "up", 2048, 21),
        weight(4, "reuse", 1024, 20),
        weight(3, "down", 1024, 20),
    };
    program.weight_page_uses = {
        {1, 0, 0, 200, 300},
        {2, 0, 0, 202, 302},
        {4, 0, 0, 450, 500},
        {3, 0, 0, 700, 750},
    };
    program.hardware.mxms_per_hemisphere = 1;
    program.queues.push_back(QueueProgram {
        QueueKind::Mem, 0,
        {
            mem_command(ftlpu::MemInstruction::Read(100, 0)),
            QueueCommand {ftlpu::isa::encode_icu_repeat(
                ftlpu::IcuRepeat {3, 2, 1})},
            mem_command(ftlpu::MemInstruction::Read(200, 1)),
            repeat_2d_command(ftlpu::IcuRepeat2D {
                3, 2, 1, 2, 10, 100,
                ftlpu::IcuInductionTarget::MemAddress}),
            encode_macro_schedule_command(
                mem_command(ftlpu::MemInstruction::Read(300, 2)),
                ftlpu::IcuMacroSchedule {24, 3, 4, 10, 1, 1, 0,
                    ftlpu::IcuInductionTarget::MemAddress}),
            encode_macro_schedule_command(
                mem_command(ftlpu::MemInstruction::Read(301, 3)),
                ftlpu::IcuMacroSchedule {25, 3, 4, 10, 1, 1, 0,
                    ftlpu::IcuInductionTarget::MemAddress}),
            encode_mem_stream_nd_command(
                mem_command(ftlpu::MemInstruction::Read(400, 4)),
                ftlpu::IcuMemStreamNdSchedule {
                    50, 3, {3, 2, 2}, {2, 8, 24}, {1, 16, 64}}),
        }});
    program.queues.push_back(QueueProgram {
        QueueKind::MxmLoad, 0,
        {encode_macro_schedule_command(
            mxm_command(ftlpu::MxmControlInstruction::IW(0, 3)),
            ftlpu::IcuMacroSchedule {10, 3, 2, 1, 1, 1, 0,
                ftlpu::IcuInductionTarget::MxmWeightColumn})}});
    program.queues.push_back(QueueProgram {
        QueueKind::MxmCompute, 1,
        {encode_mxm_stream_nd_command(
            mxm_command(ftlpu::MxmControlInstruction::Compute(
                0, 0, 0, 4, 1,
                ftlpu::MxmAccumulatorDestination::Stream,
                ftlpu::MxmDataFormat::BFloat16, true,
                ftlpu::MxmAccumulatorOutputFormat::BFloat16)),
            ftlpu::IcuMxmStreamNdSchedule {
                40, 3, {3, 2, 2}, {1, 8, 24}, {0, 4, 16},
                ftlpu::IcuInductionTarget::MxmAccumulatorAddress})}});
    auto vxm = ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::StreamBFloat16(),
        ftlpu::VxmLaneOperand::StreamBFloat16()};
    vxm.repeat_count = 7;
    program.queues.push_back(QueueProgram {
        QueueKind::Vxm, 0,
        {vxm_command(0, ftlpu::VxmChainDepth::Two, vxm)}});
    program.address_relocations.push_back(BinaryAddressRelocation {
        1, BindingAccess::Input, QueueKind::MxmLoad, 0, 0, false,
    });

    program.target_abi = executable_target_abi(program.hardware);
    std::ostringstream binary(std::ios::out | std::ios::binary);
    write_binary_program(program, binary);
    const auto bytes = binary.str();
    const auto decoded = read_binary_program(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
    std::istringstream metadataStream(bytes,
        std::ios::in | std::ios::binary);
    const auto streamMetadata = read_binary_program_metadata(metadataStream);
    const auto spanMetadata = read_binary_program_metadata(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytes.data()),
            bytes.size()));
    const auto decodedMxm = std::find_if(decoded.queues.begin(),
        decoded.queues.end(), [](const QueueProgram& queue) {
            return queue.kind == QueueKind::MxmLoad && queue.index == 0;
        });
    const auto decodedMem = std::find_if(decoded.queues.begin(),
        decoded.queues.end(), [](const QueueProgram& queue) {
            return queue.kind == QueueKind::Mem && queue.index == 0;
        });
    const auto decodedMxmNd = std::find_if(decoded.queues.begin(),
        decoded.queues.end(), [](const QueueProgram& queue) {
            return queue.kind == QueueKind::MxmCompute
                && queue.index == 1;
        });
    if (decodedMxm == decoded.queues.end()
        || decodedMxm->commands.size() != 1
        || !is_macro_schedule_command(decodedMxm->commands[0]))
        throw std::runtime_error("compact macro binary did not round-trip");
    if (decoded.hardware.icu_mem_imem_depth != 6
        || decoded.hardware.icu_mxm_instruction_bits
            != program.hardware.icu_mxm_instruction_bits)
        throw std::runtime_error("i-MEM geometry did not round-trip");
    if (decoded.bindings.empty()
        || decoded.bindings.front().page_banks
            != std::vector<std::uint16_t>({0, 1})
        || decoded.bindings.front().page_slice_group_bases
            != std::vector<std::uint16_t>({0, 0})
        || decoded.bindings.front().page_slice_group_counts
            != std::vector<std::uint16_t>({1, 1})
        || decoded.bindings.front().page_base_rows
            != std::vector<std::uint32_t>({0, 0})
        || decoded.bindings.front().page_row_counts
            != std::vector<std::uint32_t>({8, 8}))
        throw std::runtime_error(
            "exact weight-page placement did not round-trip");
    const auto imem = analyze_cmodel_abstract_imem(decoded);
    if (imem.fits() || imem.overflow_queues != 1
        || imem.used_slots != 10 || imem.encoded_work_entries != 10
        || imem.expanded_work != 44
        || !imem.queues[0].overflow()
        || imem.queues[0].overflow_slots() != 1)
        throw std::runtime_error(
            "cmodel-abstract i-MEM report is incorrect: used_slots="
            + std::to_string(imem.used_slots)
            + " encoded_work_entries="
            + std::to_string(imem.encoded_work_entries)
            + " expanded_work=" + std::to_string(imem.expanded_work)
            + " overflow_queues="
            + std::to_string(imem.overflow_queues));
    if (decoded.queues[0].commands.size() != 7
        || !is_repeat_2d_command(decoded.queues[0].commands[3]))
        throw std::runtime_error("compact Repeat2D binary did not round-trip");
    const auto decodedRepeat2D =
        decode_repeat_2d_command(decoded.queues[0].commands[3]);
    if (decodedRepeat2D.inner_count != 3
        || decodedRepeat2D.inner_interval != 2
        || decodedRepeat2D.inner_stride != 1
        || decodedRepeat2D.outer_count != 2
        || decodedRepeat2D.outer_interval != 10
        || decodedRepeat2D.outer_stride != 100
        || decodedRepeat2D.induction_target
            != ftlpu::IcuInductionTarget::MemAddress)
        throw std::runtime_error("compact Repeat2D descriptor changed on disk");
    if (streamMetadata.target_abi != program.target_abi
        || spanMetadata.target_abi != program.target_abi
        || metadataStream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error(
            "compact macro metadata-only read did not round-trip");
    const auto decodedMacro =
        decode_macro_schedule_command(decodedMxm->commands[0]);
    if (decodedMacro.start_cycle != 10 || decodedMacro.inner_count != 3
        || decodedMacro.inner_interval != 2
        || decodedMacro.inner_stride != 1
        || decodedMacro.induction_target
            != ftlpu::IcuInductionTarget::MxmWeightColumn)
        throw std::runtime_error("compact macro descriptor changed on disk");
    if (decodedMem == decoded.queues.end())
        throw std::runtime_error("MEM queue did not round-trip");
    const auto decodedStream = std::find_if(decodedMem->commands.begin(),
        decodedMem->commands.end(), is_mem_stream_nd_command);
    if (decodedStream == decodedMem->commands.end())
        throw std::runtime_error("MEM_STREAM_ND binary did not round-trip");
    const auto stream = decode_mem_stream_nd_command(*decodedStream);
    if (stream.start_cycle != 50 || stream.rank != 3
        || stream.counts != std::array<std::size_t, 3> {3, 2, 2}
        || stream.cycle_strides
            != std::array<std::size_t, 3> {2, 8, 24}
        || stream.operand_strides
            != std::array<std::int64_t, 3> {1, 16, 64})
        throw std::runtime_error(
            "MEM_STREAM_ND descriptor changed on disk");
    if (decodedMxmNd == decoded.queues.end()
        || decodedMxmNd->commands.size() != 1
        || !is_mxm_stream_nd_command(decodedMxmNd->commands[0]))
        throw std::runtime_error(
            "MXM_STREAM_ND binary did not round-trip");
    const auto mxmStream =
        decode_mxm_stream_nd_command(decodedMxmNd->commands[0]);
    if (mxmStream.start_cycle != 40 || mxmStream.rank != 3
        || mxmStream.counts
            != std::array<std::size_t, 3> {3, 2, 2}
        || mxmStream.cycle_strides
            != std::array<std::size_t, 3> {1, 8, 24}
        || mxmStream.operand_strides
            != std::array<std::int64_t, 3> {0, 4, 16}
        || mxmStream.induction_target
            != ftlpu::IcuInductionTarget::MxmAccumulatorAddress)
        throw std::runtime_error(
            "MXM_STREAM_ND descriptor changed on disk");

    const auto path = std::filesystem::temp_directory_path()
        / "ftlpu_schedule_trace_weight_page_test.csv";
    write_schedule_trace_csv(program, path);
    std::ostringstream contents;
    {
        std::ifstream input(path);
        contents << input.rdbuf();
    }
    std::filesystem::remove(path);
    const auto trace = contents.str();

    require_contains(trace,
        "start,end,resource,detail,pattern,inner_count,inner_interval,"
        "inner_stride,outer_count,outer_interval,outer_stride,skip_first,"
        "induction,base_delta");
    require_contains(trace,
        "\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=gate bytes=512 lanes=8 bandwidth=256B/cycle "
        "consumer_cycle=200 phase=pre_execution scheduled=true\"");
    require_contains(trace,
        "\"C2C.W.Prefetch\",\"page=0 bank=0 "
        "bindings=gate bytes=512 lanes=8 bandwidth=256B/cycle "
        "consumer_cycle=200 phase=pre_execution scheduled=true\"");
    require_contains(trace,
        "\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=up bytes=512 lanes=8 bandwidth=256B/cycle "
        "consumer_cycle=202 phase=pre_execution scheduled=true\"");
    require_contains(trace,
        "\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=reuse bytes=256 lanes=8 bandwidth=256B/cycle "
        "consumer_cycle=450 phase=overlap scheduled=true\"");
    require_contains(trace,
        "\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=down bytes=256 lanes=8 bandwidth=256B/cycle "
        "consumer_cycle=700 phase=overlap scheduled=true\"");
    require_contains(trace,
        "\"SR.E.C2C.Shared\",\"page=0 bank=0 streams=W24..W31 "
        "sync=target_mem+stream_tag timing=per_vector_notification\"");
    require_contains(trace,
        "\"MEM.E.C2CWrite\",\"page=0 bank=0 streams=W24..W31 "
        "sync=target_mem+stream_tag timing=per_vector_notification\"");
    if (trace.find("bindings=gate+up+reuse") != std::string::npos)
        throw std::runtime_error(
            "non-overlapping reuse page was merged with Gate/Up");
    require_contains(trace,
        "10,11,\"MXM.E0.Load\",\"IW buffer=0 column=3\",\"repeat\","
        "3,2,1,1,1,0,0,\"mxm_weight_column\",0");
    if (trace.find("12,13,\"MXM.E0.Load\"") != std::string::npos)
        throw std::runtime_error("macro schedule was expanded in CSV v2");
    require_contains(trace,
        "2,3,\"MEM.E.Read\",\"slice=0 bank=0 operation=read addr=100 stream=E0\","
        "\"repeat\",3,2,1,1,0,0,0,\"mem_address\",1");
    require_contains(trace,
        "7,8,\"MEM.E.Read\",\"slice=0 bank=0 operation=read addr=200 stream=E1\","
        "\"repeat2d\",3,2,1,2,10,100,1,\"mem_address\",0");
    require_contains(trace,
        "24,25,\"MEM.E.Read\",\"slice=0 bank=0 operation=read addr=300 stream=E2\","
        "\"repeat\",3,4,10,1,1,0,0,\"mem_address\",0");
    require_contains(trace,
        "25,26,\"MEM.E.Read\",\"slice=0 bank=0 operation=read addr=301 stream=E3\","
        "\"repeat\",3,4,10,1,1,0,0,\"mem_address\",0");
    require_contains(trace,
        "0,7,\"VXM.C0\",\"mul depth=2 repeat=7\",\"single\","
        "1,0,0,1,0,0,0,\"none\",0");

    BinaryProgram rawProgram;
    rawProgram.hardware.mxms_per_hemisphere = 1;
    const ftlpu::IcuLoop3D rawMemLoop{
        0, {3, 2, 2}, {2, 8, 24}};
    rawProgram.queues.push_back(QueueProgram{QueueKind::Mem, 0,
        raw_fu_commands(ftlpu::isa::encode_mem_icu_3d_instruction(
            ftlpu::MemIcuInstruction::Read3D(rawMemLoop,
                ftlpu::MemIcuAddress3D::Affine(100, {1, 16, 64}),
                ftlpu::StreamId::East(0))), InstructionKind::Mem, 10)});

    rawProgram.queues[0].commands.push_back(
        QueueCommand{ftlpu::isa::encode_icu_nop(3)});
    const auto trailingMem = raw_fu_commands(
        ftlpu::isa::encode_mem_icu_3d_instruction(
            ftlpu::MemIcuInstruction::Read3D(
                ftlpu::IcuLoop3D{0, {1, 1, 1}, {1, 1, 1}},
                ftlpu::MemIcuAddress3D::Affine(999, {0, 0, 0}),
                ftlpu::StreamId::East(1))),
        InstructionKind::Mem);
    rawProgram.queues[0].commands.insert(
        rawProgram.queues[0].commands.end(),
        trailingMem.begin(), trailingMem.end());

    const ftlpu::IcuLoop3D rawMxmLoop{0, {2, 1, 1}, {2, 1, 1}};
    rawProgram.queues.push_back(QueueProgram{QueueKind::MxmLoad, 0,
        raw_fu_commands(ftlpu::isa::encode_mxm_load_icu_3d_instruction(
            ftlpu::MxmLoadIcuInstruction::Load3D(rawMxmLoop, 0,
                ftlpu::MxmIcuBufferMode::ToggleDimension0, 0,
                {1, 0, 0}, 8)), InstructionKind::Mxm, 70)});
    rawProgram.queues.push_back(QueueProgram{QueueKind::MxmDequant, 0,
        raw_fu_commands(ftlpu::isa::encode_mxm_dequant_icu_3d_instruction(
            ftlpu::MxmDequantIcuInstruction::Dequant3D(rawMxmLoop,
                ftlpu::MxmDequantInstruction::ScaleBits(0x3c00))),
            InstructionKind::MxmDequant, 70)});
    rawProgram.queues.push_back(QueueProgram{QueueKind::MxmCompute, 1,
        raw_fu_commands(ftlpu::isa::encode_mxm_compute_icu_3d_instruction(
            ftlpu::MxmComputeIcuInstruction::Compute3D(rawMxmLoop, 0,
                ftlpu::MxmIcuBufferMode::ToggleDimension0, 16, 12, 4,
                {0, 0, 0}, 1, ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeIcuMode{})), InstructionKind::Mxm, 70)});

    auto rawVxm = ftlpu::VxmLaneAluInstruction{
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::StreamBFloat16(),
        ftlpu::VxmLaneOperand::Imm(0.5f)};
    rawVxm.repeat_count = 4;
    const auto compact = ftlpu::VxmCompactInstructionCodec::encode(
        0, ftlpu::VxmChainDepth::Two, rawVxm);
    rawProgram.queues.push_back(QueueProgram{QueueKind::Vxm, 0,
        raw_fu_commands(ftlpu::isa::encode_vxm_icu_run_2d_instruction(
            ftlpu::VxmIcuRun2DInstruction::Run2D(
                0, {2, 2}, {8, 24}, compact)), InstructionKind::Vxm,
            90)});

    auto map = ftlpu::SxmInstruction::PermuteMap{};
    for (std::size_t lane = 0; lane < map.size(); ++lane)
        map[lane] = (lane + ftlpu::hw::kLanesPerTile) % map.size();
    const auto sxm = ftlpu::SxmInstruction::Permute(
        {{0}, {1}}, {{16}, {17}}, map);
    rawProgram.queues.push_back(QueueProgram{QueueKind::SxmPermute, 0,
        raw_fu_commands(ftlpu::isa::encode_sxm_icu_run_2d_instruction(
            ftlpu::SxmIcuRun2DInstruction::Run2D(
                0, {2, 2}, {2, 12}, sxm, 8)), InstructionKind::Sxm,
            120)});

    const auto rawPath = std::filesystem::temp_directory_path()
        / "ftlpu_schedule_trace_raw_fu_test.csv";
    write_schedule_trace_csv(rawProgram, rawPath);
    std::ostringstream rawContents;
    {
        std::ifstream input(rawPath);
        rawContents << input.rdbuf();
    }
    std::filesystem::remove(rawPath);
    const auto rawTrace = rawContents.str();
    require_contains(rawTrace,
        "10,11,\"MEM.E.Read3D\",\"slice=0 bank=0 operation=read addr=100 "
        "stream=E0 depth=0/2");
    require_contains(rawTrace,
        "34,35,\"MEM.E.Read3D\",\"slice=0 bank=0 operation=read addr=164 "
        "stream=E0 depth=1/2");
    require_contains(rawTrace,
        "50,51,\"MEM.E.Read3D\",\"slice=0 bank=0 operation=read addr=999 "
        "stream=E1 depth=0/1");
    require_contains(rawTrace, "\"MXM.E0.Load\",\"Load3D");
    require_contains(rawTrace, "\"MXM.E0.Dequant\",\"Dequant3D");
    require_contains(rawTrace, "\"MXM.W0.Compute\",\"Compute3D");
    require_contains(rawTrace, "\"VXM.C0\",\"mul depth=2 repeat=4 Run2D\"");
    require_contains(rawTrace, "\"SXM.E.Permute\",\"permute Run2D map_stride=8\"");

    std::ostringstream rawPerformance;
    print_runtime_performance(rawProgram, 128, rawPerformance);
    require_contains(rawPerformance.str(),
        "runtime perf resource=MEM cycles=128 active_queues=1/"
            + std::to_string(rawProgram.hardware.hemispheres
                * rawProgram.hardware.slices_per_hemisphere
                * rawProgram.hardware.banks_per_slice)
            + " issued=13");
    require_contains(rawPerformance.str(),
        "runtime perf resource=MXM.load cycles=128 active_queues=1/2 issued=2");
    require_contains(rawPerformance.str(),
        "runtime perf resource=VXM cycles=128 active_queues=1/"
            + std::to_string(rawProgram.hardware.vxm_alus)
            + " issued=4");

    // A MEM queue may mix ordinary READ3D with the three-word WRITE_READ_2D
    // packet. Both the performance walk and CSV trace must advance by the
    // complete packet before decoding the following instruction.
    ftlpu::MemIcuWriteRead2DInstruction writeRead{};
    writeRead.start_wait = 5;
    writeRead.counts = {2, 2};
    writeRead.write_cycle_strides = {8, 20};
    writeRead.read_cycle_strides = {8, 20};
    writeRead.read_start_offset = 2;
    writeRead.base_address = 30;
    writeRead.address_strides = {1, 4};
    writeRead.write_stream = ftlpu::StreamId::East(3).packed();
    writeRead.read_stream_base = ftlpu::StreamId::East(4).packed();
    writeRead.read_stream_outer_stride = 1;
    const auto rawRead = raw_fu_commands(
        ftlpu::isa::encode_mem_icu_3d_instruction(
            ftlpu::MemIcuInstruction::Read3D(
                ftlpu::IcuLoop3D{0, {1, 1, 1}, {1, 1, 1}},
                ftlpu::MemIcuAddress3D::Affine(50, {0, 0, 0}),
                ftlpu::StreamId::East(0))),
        InstructionKind::Mem);
    auto mixedCommands = rawRead;
    mixedCommands.push_back(QueueCommand{ftlpu::isa::encode_icu_nop(3)});
    const auto writeReadCommands = raw_fu_commands(
        ftlpu::isa::encode_mem_icu_write_read_2d_instruction(writeRead),
        InstructionKind::Mem);
    mixedCommands.insert(mixedCommands.end(), writeReadCommands.begin(),
        writeReadCommands.end());
    mixedCommands.insert(mixedCommands.end(), rawRead.begin(), rawRead.end());
    BinaryProgram mixedProgram;
    mixedProgram.queues.push_back(
        QueueProgram{QueueKind::Mem, 0, std::move(mixedCommands)});
    std::ostringstream mixedPerformance;
    print_runtime_performance(mixedProgram, 128, mixedPerformance);
    require_contains(mixedPerformance.str(),
        "runtime perf resource=MEM cycles=128 active_queues=1/"
            + std::to_string(mixedProgram.hardware.hemispheres
                * mixedProgram.hardware.slices_per_hemisphere
                * mixedProgram.hardware.banks_per_slice)
            + " issued=10");
    const auto mixedPath = std::filesystem::temp_directory_path()
        / "ftlpu_schedule_trace_mixed_mem_test.csv";
    write_schedule_trace_csv(mixedProgram, mixedPath);
    std::ostringstream mixedContents;
    {
        std::ifstream input(mixedPath);
        mixedContents << input.rdbuf();
    }
    std::filesystem::remove(mixedPath);
    require_contains(mixedContents.str(),
        "9,10,\"MEM.E.WRITE_READ_2D.Write\"");
    require_contains(mixedContents.str(),
        "11,12,\"MEM.E.WRITE_READ_2D.Read\"");
    require_contains(mixedContents.str(),
        "40,41,\"MEM.E.Read3D\"");

    BinaryProgram synchronizedProgram;
    const auto synchronizedPacket =
        ftlpu::InstructionControlUnit::MemIcu::
            encode_synchronized_raw_packet(
                3, 17, 2, 1,
                ftlpu::MemInstruction::Write(
                    64, ftlpu::StreamId::East(0)),
                8);
    const auto synchronizedCommands =
        encode_mem_synchronized_icu_packet(synchronizedPacket);
    std::vector<QueueCommand> synchronizedQueueCommands(
        synchronizedCommands.begin(), synchronizedCommands.end());
    synchronizedProgram.queues.push_back(
        QueueProgram {QueueKind::Mem, 0,
            std::move(synchronizedQueueCommands)});
    std::ostringstream synchronizedPerformance;
    print_runtime_performance(
        synchronizedProgram, 16, synchronizedPerformance);
    require_contains(synchronizedPerformance.str(),
        "runtime perf resource=MEM cycles=16 active_queues=1/"
            + std::to_string(
                synchronizedProgram.hardware.hemispheres
                * synchronizedProgram.hardware.slices_per_hemisphere
                * synchronizedProgram.hardware.banks_per_slice)
            + " issued=3");

    auto syncTraceProgram = synchronizedProgram;
    auto& syncTraceQueue = syncTraceProgram.queues.front().commands;
    syncTraceQueue.insert(syncTraceQueue.end(),
        rawRead.begin(), rawRead.end());
    syncTraceQueue.push_back(encode_icu_control_raw_word(
        ftlpu::IcuControlInstruction::Sync()));
    const auto syncTracePath = std::filesystem::temp_directory_path()
        / "ftlpu_schedule_trace_sync_test.csv";
    write_schedule_trace_csv(syncTraceProgram, syncTracePath);
    std::ostringstream syncTraceContents;
    {
        std::ifstream input(syncTracePath);
        syncTraceContents << input.rdbuf();
    }
    std::filesystem::remove(syncTracePath);
    require_contains(syncTraceContents.str(),
        "0,8,\"MEM.E.WriteSync\",\"opcode=MEM_WRITE_SYNC "
        "slice=0 bank=0 pc=0 sync_tag=17 vectors=3 "
        "reservation_cycles=8 addr=64 stream=E0\"");
    require_contains(syncTraceContents.str(),
        "8,9,\"MEM.E.Read3D\"");
    require_contains(syncTraceContents.str(),
        "9,10,\"MEM.E.Sync\",\"opcode=SYNC queue=mem "
        "index=0 pc=5 actual_wait=runtime_dependent\"");

    BinaryProgram rawControlProgram;
    rawControlProgram.queues.push_back(QueueProgram {QueueKind::C2cDma, 0,
        {encode_icu_control_raw_word(
            ftlpu::IcuControlInstruction::WaitEvent(23)),
         encode_icu_control_raw_word(
            ftlpu::IcuControlInstruction::Notify())}});
    std::ostringstream rawControlPerformance;
    print_runtime_performance(
        rawControlProgram, 16, rawControlPerformance);
    require_contains(rawControlPerformance.str(),
        "runtime perf resource=C2C.DMA cycles=16 active_queues=0/2 issued=0");

    std::cout << "schedule_trace_weight_page_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "schedule_trace_weight_page_test failed: "
              << error.what() << '\n';
    return 1;
}
