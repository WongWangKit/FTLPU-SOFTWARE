#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

template <typename Packet>
std::vector<ftlpu::software::runtime::QueueCommand> raw_commands(
    const Packet& packet)
{
    using namespace ftlpu::software::runtime;
    std::vector<QueueCommand> commands;
    commands.reserve(Packet::kWordCount);
    for (const auto& physicalWord : packet.words) {
        QueueCommand command;
        command.command = physicalWord.lanes[0];
        command.instruction_kind = InstructionKind::Vxm;
        command.word_count = Packet::kLanesPerWord;
        for (std::size_t laneIndex = 0;
             laneIndex < Packet::kLanesPerWord; ++laneIndex)
            command.words[laneIndex] = physicalWord.lanes[laneIndex];
        commands.push_back(command);
    }
    return commands;
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: vxm_stream_nd_binary_test program.ftlpu");

    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    VxmLaneAluInstruction lane {};
    lane.operation = VxmAluOpcode::Add;
    lane.lhs = VxmLaneOperand::StreamBFloat16(1.0f, 0);
    lane.rhs = VxmLaneOperand::Imm(0.25f);
    lane.output_type = VxmCastTarget::BFloat16;
    lane.repeat_count = 32;
    const auto packet = VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Eight, lane);

    QueueCommand native;
    native.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Instruction);
    native.instruction_kind = InstructionKind::Vxm;
    native.word_count = 3;
    native.words = {static_cast<std::uint32_t>(packet.control),
        static_cast<std::uint32_t>(packet.control >> 32),
        packet.immediate_bits, 0};

    BinaryProgram program;
    program.max_cycle = 24;
    program.queues.push_back(QueueProgram {QueueKind::Vxm, 0, {
        encode_vxm_stream_nd_command(native, IcuVxmStreamNdSchedule {
            2, 2, {2, 2, 1}, {5, 17, 1}, {0, 0, 0},
            IcuInductionTarget::None}),
    }});

    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    write_binary_program(program, path);
    const auto decoded = read_binary_program(path);
    require(decoded.queues.size() == 1
            && decoded.queues[0].commands.size() == 1,
        "VXM_STREAM_ND command count was not preserved");
    require(is_vxm_stream_nd_command(decoded.queues[0].commands[0]),
        "VXM_STREAM_ND did not survive binary round-trip");
    const auto descriptor = decode_vxm_stream_nd_command(
        decoded.queues[0].commands[0]);
    require(descriptor.schedule.rank == 2
            && descriptor.instruction.words == native.words,
        "VXM_STREAM_ND payload did not survive binary round-trip");

    InstructionControlUnit icu;
    load_queue_programs_into_icu(decoded.queues, icu);
    std::vector<std::size_t> issueCycles;
    for (std::size_t cycle = 0; cycle <= 24; ++cycle) {
        if (const auto issued = icu.vxm_iq(0).tick()) {
            require(*issued == packet,
                "runtime loader changed the VXM packet");
            require(VxmCompactInstructionCodec::decode(0, *issued)
                        .instruction.repeat_count == 32,
                "runtime loader expanded the VXM datapath repeat");
            issueCycles.push_back(cycle);
        }
    }
    require(issueCycles == std::vector<std::size_t> {2, 7, 19, 24},
        "VXM_STREAM_ND issued at incorrect absolute cycles");

    const auto rawPacket = isa::encode_vxm_icu_run_2d_instruction(
        VxmIcuRun2DInstruction::Run2D(
            0, {2, 2}, {5, 17}, packet));
    BinaryProgram rawProgram;
    rawProgram.max_cycle = 24;
    auto rawCommands = raw_commands(rawPacket);
    rawCommands.insert(rawCommands.begin(),
        QueueCommand {isa::encode_icu_nop(2)});
    rawProgram.queues.push_back(QueueProgram {
        QueueKind::Vxm, 0, std::move(rawCommands)});
    write_binary_program(rawProgram, path);
    const auto rawDecoded = read_binary_program(path);
    require(rawDecoded.queues.size() == 1
            && rawDecoded.queues[0].commands.size()
                == isa::EncodedVxmIcuRun2DPacket::kWordCount + 1
            && is_fu_3d_raw_packet_header(
                rawDecoded.queues[0].commands[1]),
        "raw VXM RUN_2D did not survive binary round-trip");
    for (std::size_t word = 0; word < rawPacket.words.size(); ++word)
        require(rawDecoded.queues[0].commands[word + 1].words
                    == rawProgram.queues[0].commands[word + 1].words,
            "raw VXM RUN_2D changed a physical i-MEM word");

    const auto capacity = analyze_physical_imem(rawDecoded);
    require(capacity.fits() && capacity.queues.size() == 1
            && capacity.queues[0].physical_slots == 4
            && capacity.queues[0].nop_entries == 1
            && capacity.queues[0].coarse_program_entries == 1
            && capacity.queues[0].expanded_work == 4
            && capacity.queues[0].peak_fu_3d_contexts == 1
            && capacity.queues[0].fu_3d_context_capacity
                == hw::kIcuVxmRun2DContextDepth
            && capacity.queues[0].fu_3d_context_bits
                == hw::kIcuVxmRun2DContextBits,
        "raw VXM RUN_2D capacity accounting is incorrect");

    InstructionControlUnit rawIcu;
    load_queue_programs_into_icu(rawDecoded.queues, rawIcu);
    issueCycles.clear();
    for (std::size_t cycle = 0; cycle <= 24; ++cycle) {
        if (const auto issued = rawIcu.vxm_iq(0).tick()) {
            require(*issued == packet,
                "raw runtime loader changed the VXM compact config");
            issueCycles.push_back(cycle);
        }
    }
    require(issueCycles == std::vector<std::size_t> {2, 7, 19, 24},
        "raw VXM RUN_2D issued at incorrect absolute cycles");

    std::cout << "vxm_stream_nd_binary_test passed: legacy_adapter=1 "
              << "raw_run_2d_words=3 launches=4 run_length=32\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "vxm_stream_nd_binary_test failed: "
              << error.what() << '\n';
    return 1;
}
