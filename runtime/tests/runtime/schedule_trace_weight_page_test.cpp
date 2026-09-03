#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using namespace ftlpu::software::runtime;

BinaryBinding weight(
    std::uint32_t index, std::string name, std::uint64_t bytes)
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
    binding.page_bank_count = 2;
    return binding;
}

void require_contains(const std::string& text, const std::string& expected)
{
    if (text.find(expected) == std::string::npos)
        throw std::runtime_error("trace is missing: " + expected);
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

QueueCommand repeat_2d_command(const ftlpu::IcuRepeat2D& repeat)
{
    const auto encoded = ftlpu::isa::encode_icu_repeat_2d(repeat);
    return QueueCommand {
        encoded.words[0], InstructionKind::None, 3,
        {encoded.words[0], encoded.words[1], encoded.words[2], 0},
    };
}

} // namespace

int main()
try {
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
        weight(1, "gate", 2048),
        weight(2, "up", 2048),
        weight(4, "reuse", 1024),
        weight(3, "down", 1024),
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
            mem_command(ftlpu::MemInstruction::Read(300, 2)),
            mem_command(ftlpu::MemInstruction::Read(301, 3)),
            QueueCommand {ftlpu::isa::encode_icu_loop(
                ftlpu::IcuLoop {2, 3, 4, 10})},
        }});
    program.queues.push_back(QueueProgram {
        QueueKind::MxmLoad, 0,
        {encode_macro_schedule_command(
            mxm_command(ftlpu::MxmControlInstruction::IW(0, 3)),
            ftlpu::IcuMacroSchedule {10, 3, 2, 1, 1, 1, 0,
                ftlpu::IcuInductionTarget::MxmWeightColumn})}});
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
    if (decoded.queues.size() != 2 || decoded.queues[1].commands.size() != 1
        || !is_macro_schedule_command(decoded.queues[1].commands[0]))
        throw std::runtime_error("compact macro binary did not round-trip");
    if (decoded.hardware.icu_mem_imem_depth != 6
        || decoded.hardware.icu_mxm_instruction_bits
            != program.hardware.icu_mxm_instruction_bits)
        throw std::runtime_error("i-MEM geometry did not round-trip");
    const auto imem = analyze_cmodel_abstract_imem(decoded);
    if (imem.fits() || imem.overflow_queues != 1
        || imem.used_slots != 8 || imem.encoded_work_entries != 8
        || imem.expanded_work != 21
        || !imem.queues[0].overflow()
        || imem.queues[0].overflow_slots() != 1)
        throw std::runtime_error("cmodel-abstract i-MEM report is incorrect");
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
        decode_macro_schedule_command(decoded.queues[1].commands[0]);
    if (decodedMacro.start_cycle != 10 || decodedMacro.inner_count != 3
        || decodedMacro.inner_interval != 2
        || decodedMacro.inner_stride != 1
        || decodedMacro.induction_target
            != ftlpu::IcuInductionTarget::MxmWeightColumn)
        throw std::runtime_error("compact macro descriptor changed on disk");

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
        "0,137,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=gate+up bytes=1024 lanes=8 bandwidth=256B/cycle "
        "deadline=200 scheduled=true\"");
    require_contains(trace,
        "0,137,\"C2C.W.Prefetch\",\"page=0 bank=0 "
        "bindings=gate+up bytes=1024 lanes=8 bandwidth=256B/cycle "
        "deadline=200 scheduled=true\"");
    require_contains(trace,
        "330,450,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=reuse bytes=256 lanes=8 bandwidth=256B/cycle "
        "deadline=450 scheduled=true\"");
    require_contains(trace,
        "580,700,\"C2C.E.Prefetch\",\"page=0 bank=0 "
        "bindings=down bytes=256 lanes=8 bandwidth=256B/cycle "
        "deadline=700 scheduled=true\"");
    if (trace.find("bindings=gate+up+reuse") != std::string::npos)
        throw std::runtime_error(
            "non-overlapping reuse page was merged with Gate/Up");
    require_contains(trace,
        "10,11,\"MXM.E0.Load\",\"IW buffer=0 column=3\",\"repeat\","
        "3,2,1,1,1,0,0,\"mxm_weight_column\",0");
    if (trace.find("12,13,\"MXM.E0.Load\"") != std::string::npos)
        throw std::runtime_error("macro schedule was expanded in CSV v2");
    require_contains(trace,
        "2,3,\"MEM.E.Read\",\"slice=0 bank=0 addr=100 stream=E0\","
        "\"repeat\",3,2,1,1,0,0,0,\"mem_address\",1");
    require_contains(trace,
        "7,8,\"MEM.E.Read\",\"slice=0 bank=0 addr=200 stream=E1\","
        "\"repeat2d\",3,2,1,2,10,100,1,\"mem_address\",0");
    require_contains(trace,
        "24,25,\"MEM.E.Read\",\"slice=0 bank=0 addr=300 stream=E2\","
        "\"repeat\",3,4,10,1,0,0,0,\"mem_address\",10");
    require_contains(trace,
        "25,26,\"MEM.E.Read\",\"slice=0 bank=0 addr=301 stream=E3\","
        "\"repeat\",3,4,10,1,0,0,0,\"mem_address\",10");

    std::cout << "schedule_trace_weight_page_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "schedule_trace_weight_page_test failed: "
              << error.what() << '\n';
    return 1;
}
