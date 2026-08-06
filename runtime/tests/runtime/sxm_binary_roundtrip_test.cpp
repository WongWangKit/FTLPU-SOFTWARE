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

    BinaryProgram program;
    program.mxms_per_hemisphere = 1;
    program.target_abi = lpu_32stream_target_abi(
        program.mxms_per_hemisphere);
    program.memory_floors.push_back({1, 7, 8192});
    BinaryBinding binding;
    binding.index = 3;
    binding.access = BindingAccess::Internal;
    binding.role = "attention_input";
    binding.name = "rms1_result";
    binding.ready_cycle = 1234;
    program.bindings.push_back(binding);
    program.timelines.push_back({"rmsnorm.transpose", 7, 13});
    program.max_cycle = icu_program.last_cycle();
    program.queues = icu_program.encode_queues();
    for (auto& queue : program.queues) {
        if (queue.kind != QueueKind::SxmTranspose || queue.index != 0)
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
        || decoded.bindings[0].ready_cycle != 1234)
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
    require(decoded.mxms_per_hemisphere == 1,
        "logical MXM topology was not preserved");
    require(decoded.memory_floors.size() == 1
            && decoded.memory_floors[0].hemisphere == 1
            && decoded.memory_floors[0].slice == 7
            && decoded.memory_floors[0].first_free_row == 8192,
        "per-slice MEM floor was not preserved");
    bool found = false;
    for (const auto& queue : decoded.queues) {
        if (queue.kind != QueueKind::SxmTranspose || queue.index != 0 || queue.commands.empty())
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
    require(found, "serialized SXM transpose queue is missing");

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
    std::cout << "sxm_binary_roundtrip_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "sxm_binary_roundtrip_test failed: " << ex.what() << '\n';
    return 1;
}
