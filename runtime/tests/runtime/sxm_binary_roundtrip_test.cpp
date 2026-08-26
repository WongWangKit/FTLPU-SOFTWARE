#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

ftlpu::SxmInstruction::StreamList streamRange(
    std::size_t first, std::size_t count)
{
    auto streams = ftlpu::SxmInstruction::StreamList {};
    for (std::size_t index = 0; index < count; ++index)
        streams.push_back(ftlpu::SxmStreamId {first + index});
    return streams;
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2) throw std::runtime_error("usage: sxm_binary_roundtrip_test program.ftlpu");

    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    IcuProgram icu_program;
    auto source_streams = SxmInstruction::StreamList {};
    auto destination_streams = SxmInstruction::StreamList {};
    for (std::size_t stream = 0; stream < 16; ++stream) {
        source_streams.push_back(SxmStreamId {stream});
        destination_streams.push_back(SxmStreamId {16 + stream});
    }
    auto transpose = SxmInstruction::Transpose(
        std::move(source_streams), std::move(destination_streams));
    icu_program.emit_sxm_transpose(7, Hemisphere::East, transpose);

    auto permute = SxmInstruction::Permute(
        streamRange(16, 16), streamRange(32, 16),
        Permute320::identity_map());
    permute.output_row = 5;
    permute.output_tile = 2;
    icu_program.emit_sxm_permute(12, Hemisphere::East, permute);

    BinaryProgram program;
    program.hardware.mxms_per_hemisphere = 1;
    program.target_abi = lpu_32stream_target_abi(
        program.hardware.mxms_per_hemisphere);
    program.memory_floors.push_back({1, 7, 8192, 1});
    BinaryBinding binding;
    binding.index = 3;
    binding.access = BindingAccess::Internal;
    binding.role = "attention_input";
    binding.name = "rms1_result";
    binding.ready_cycle = 1234;
    binding.bank = 1;
    program.bindings.push_back(binding);
    program.timelines.push_back({"rmsnorm.transpose", 7, 13});
    program.max_cycle = icu_program.last_cycle();
    program.queues = icu_program.encode_queues();
    for (auto& queue : program.queues) {
        if (queue.kind != QueueKind::SxmPermute || queue.index != 0)
            continue;
        QueueCommand repeat;
        repeat.command = isa::encode_icu_repeat(
            InstructionControlUnit::Repeat {3, 2, 0});
        queue.commands.push_back(repeat);
        program.max_cycle += 6;
    }
    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    write_binary_program(program, path);

    const auto decoded = read_binary_program(path);
    if (decoded.bindings.size() != 1
        || decoded.bindings[0].role != "attention_input"
        || decoded.bindings[0].name != "rms1_result"
        || decoded.bindings[0].ready_cycle != 1234
        || decoded.bindings[0].bank != 1)
        return 1;
    require(decoded.timelines.size() == 1
            && decoded.timelines[0].name == "rmsnorm.transpose"
            && decoded.timelines[0].start_cycle == 7
            && decoded.timelines[0].end_cycle == 13,
        "timeline metadata was not preserved");
    require(decoded.target_name == kLpu32StreamTargetName,
        "target name was not preserved");
    require(decoded.target_abi == lpu_32stream_target_abi(1),
        "target ABI was not preserved");
    require(decoded.hardware.mxms_per_hemisphere == 1,
        "logical MXM topology was not preserved");
    require(decoded.memory_floors.size() == 1
            && decoded.memory_floors[0].hemisphere == 1
            && decoded.memory_floors[0].slice == 7
            && decoded.memory_floors[0].first_free_row == 8192
            && decoded.memory_floors[0].bank == 1,
        "per-slice MEM floor was not preserved");
    bool found = false;
    for (const auto& queue : decoded.queues) {
        if (queue.kind != QueueKind::SxmPermute || queue.index != 0 || queue.commands.empty())
            continue;
        const auto instruction = std::find_if(
            queue.commands.begin(), queue.commands.end(),
            [](const QueueCommand& command) {
                return command.instruction_kind
                    == InstructionKind::Sxm;
            });
        require(instruction != queue.commands.end(),
            "SXM instruction kind was not preserved");
        const auto& command = *instruction;
        require(command.instruction_kind == InstructionKind::Sxm, "SXM instruction kind was not preserved");
        require((command.words[3] & 0xffu) == 5
                && ((command.words[3] >> 8) & 0xffu) == 0xffu
                && ((command.words[3] >> 16) & 0xffu) == 2,
            "SXM row/tile selectors were not preserved");
        require(std::any_of(queue.commands.begin(),
                    queue.commands.end(),
                    [](const QueueCommand& candidate) {
                        return isa::decode_icu_command_opcode(
                                   candidate.command)
                            == isa::IcuCommandOpcode::Repeat;
                    }),
            "SXM repeat command was not preserved");
        require(command.extension_words.size() == 2 + 16 + 16 + SxmInstruction::kTotalLanes,
            "SXM variable payload size was not preserved");
        found = true;
    }
    require(found, "serialized SXM permute queue is missing");

    auto incompatible = decoded;
    incompatible.target_name = "incompatible-test-target";
    incompatible.target_abi ^= 1;
    auto rejected_system = std::make_unique<TspSliceSystem>();
    CModelRuntime rejected_runtime(*rejected_system);
    bool rejected = false;
    try {
        rejected_runtime.load(incompatible);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "CModel runtime accepted an incompatible target ABI");

    auto system = std::make_unique<TspSliceSystem>();
    CModelRuntime runtime(*system);
    runtime.load(decoded);
    require(system->hardware_configuration().mxms_per_hemisphere == 1,
        "CModel instance did not select the executable MXM topology");

    auto legacy = decoded;
    legacy.target_name = "test-owned-legacy-hardware";
    legacy.hardware.mxms_per_hemisphere = 2;
    legacy.hardware.mxm_local_dequant_enabled = 0;
    legacy.hardware.mxm_block_compute_enabled = 0;
    legacy.target_abi = executable_target_abi(legacy.hardware);
    runtime.load(legacy);
    require(system->hardware_configuration().mxms_per_hemisphere == 2,
        "CModel instance did not select the legacy test configuration");

    auto oversized = decoded;
    oversized.hardware.mxms_per_hemisphere =
        static_cast<std::uint32_t>(hw::kMxmsPerHemisphere + 1);
    oversized.target_abi = executable_target_abi(oversized.hardware);
    bool oversized_rejected = false;
    try {
        runtime.load(oversized);
    } catch (const std::invalid_argument&) {
        oversized_rejected = true;
    }
    require(oversized_rejected,
        "CModel runtime accepted a topology beyond physical capacity");

    auto shallow_sram = decoded;
    shallow_sram.hardware.sram_depth_rows = 1024;
    shallow_sram.target_abi = executable_target_abi(shallow_sram.hardware);
    runtime.load(shallow_sram);
    bool row_rejected = false;
    try {
        system->initialize_mem_sram_lane_byte(
            Hemisphere::East, 0, 0, 1024, 0, 1);
    } catch (const std::out_of_range&) {
        row_rejected = true;
    }
    require(row_rejected,
        "CModel SRAM did not enforce the test-selected logical depth");
    std::cout << "sxm_binary_roundtrip_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "sxm_binary_roundtrip_test failed: " << ex.what() << '\n';
    return 1;
}
