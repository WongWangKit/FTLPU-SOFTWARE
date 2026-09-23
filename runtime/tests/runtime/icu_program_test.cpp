#include "ftlpu/software/runtime/icu_program.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Packet>
std::vector<QueueCommand> raw_mem_3d_commands(const Packet& packet)
{
    std::vector<QueueCommand> commands;
    commands.reserve(packet.words.size());
    for (const auto& word : packet.words) {
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mem;
        command.word_count =
            Packet::kLanesPerWord;
        for (std::size_t lane = 0; lane < command.word_count; ++lane)
            command.words[lane] = word.lanes[lane];
        commands.push_back(std::move(command));
    }
    return commands;
}

} // namespace

int main()
try {
    constexpr std::size_t kWaitTag = 0x1234;
    constexpr std::size_t kWriteTag = 0x2345;
    constexpr std::size_t kReadTag = 0x3456;
    const IcuLoop3D onePoint {0, {1, 1, 1}, {1, 1, 1}, 0};

    QueueProgram queue {QueueKind::Mem, 0, {}};
    const auto append = [&](const auto& commands) {
        queue.commands.insert(queue.commands.end(),
            commands.begin(), commands.end());
    };
    append(raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Read3D(onePoint,
                MemIcuAddress3D::Affine(10, {0, 0, 0}),
                StreamId::East(1)))));
    append(raw_mem_3d_commands(
        isa::encode_mem_icu_3d_instruction(
            MemIcuInstruction::Write3D(onePoint,
                MemIcuAddress3D::Affine(20, {0, 0, 0}),
                StreamId::East(2)))));

    const auto wait = encode_icu_control_raw_word(
        IcuControlInstruction::WaitEvent(kWaitTag));
    require(is_icu_control_raw_word_command(wait)
            && !is_repeat_2d_command(wait),
        "raw WAIT_EVENT was mistaken for Repeat2D");
    queue.commands.push_back(wait);

    const auto synchronized =
        InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(
            2, kWriteTag, 1, 1,
            MemInstruction::Write(30, StreamId::East(3)));
    const auto synchronizedCommands =
        encode_mem_synchronized_icu_packet(synchronized);
    require(is_mem_synchronized_raw_packet_header(
                synchronizedCommands[0])
            && is_mem_synchronized_raw_word_command(
                synchronizedCommands[1])
            && !is_repeat_2d_command(synchronizedCommands[0]),
        "MEM_WRITE_SYNC raw packet was not recognized structurally");
    append(synchronizedCommands);

    const auto synchronizedRead =
        InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(
            2, kReadTag, 0, 1,
            MemInstruction::Read(40, StreamId::East(4)));
    const auto synchronizedReadCommands =
        encode_mem_synchronized_icu_packet(synchronizedRead);
    require(is_mem_synchronized_raw_packet_header(
                synchronizedReadCommands[0])
            && is_mem_synchronized_raw_word_command(
                synchronizedReadCommands[1]),
        "MEM_READ_SYNC raw packet was not recognized structurally");
    append(synchronizedReadCommands);

    const auto expectedWords =
        2 * isa::EncodedMemIcu3DPacket::kWordCount + 1
        + 2 * InstructionControlUnit::MemIcu::
            synchronized_packet_word_count;

    InstructionControlUnit icu;
    load_queue_programs_into_icu({queue}, icu);
    auto& mem = icu.mem_iq(0);
    require(mem.imem_occupancy() == expectedWords,
        "MEM loader did not place every command in one local iMEM");

    mem.configure_all();
    for (std::size_t cycle = 0; cycle <= expectedWords; ++cycle)
    mem.prefetch_only();
    mem.notify(kWaitTag);
    mem.notify(kWriteTag);
    mem.notify(kWriteTag);
    mem.notify(kReadTag);
    mem.notify(kReadTag);
    std::vector<MemInstruction> issued;
    for (std::size_t cycle = 0; cycle < 512 && !mem.done(); ++cycle) {
        if (const auto instruction = mem.tick())
            issued.push_back(*instruction);
    }

    if (!mem.done())
        throw std::runtime_error(
            "unified MEM ICU program did not retire: fetch_pc="
            + std::to_string(mem.fetch_pc())
            + " iq=" + std::to_string(mem.iq_occupancy())
            + " pending=" + std::to_string(mem.pending_fetch_count())
            + " issued=" + std::to_string(issued.size())
            + " action=" + std::to_string(static_cast<unsigned>(
                mem.last_trace().action)));
    require(mem.fetch_pc() == expectedWords,
        "unified MEM ICU PC did not traverse the complete program");
    require(issued.size() == 6,
        "unified MEM ICU emitted the wrong number of FU instructions");
    require(issued[0].opcode == MemOpcode::Read
            && issued[0].address == 10
            && issued[1].opcode == MemOpcode::Write
            && issued[1].address == 20
            && issued[2].opcode == MemOpcode::Write
            && issued[2].address == 30
            && issued[3].opcode == MemOpcode::Write
            && issued[3].address == 31
            && issued[4].opcode == MemOpcode::Read
            && issued[4].address == 40
            && issued[5].opcode == MemOpcode::Read
            && issued[5].address == 41,
        "Read3D, Write3D, MEM_WRITE_SYNC, and MEM_READ_SYNC did not execute in iMEM order");
    require(mem.synchronized_issued_count() == 4,
        "MEM synchronized issue accounting is incorrect");

    MemIcuWriteRead2DInstruction writeRead{};
    writeRead.start_wait = 5;
    writeRead.counts = {2, 2};
    writeRead.write_cycle_strides = {8, 20};
    writeRead.read_cycle_strides = {8, 20};
    writeRead.read_start_offset = 2;
    writeRead.base_address = 30;
    writeRead.address_strides = {1, 4};
    writeRead.write_stream = StreamId::East(3).packed();
    writeRead.read_stream_base = StreamId::East(4).packed();
    writeRead.read_stream_outer_stride = 1;
    QueueCommand nop{};
    nop.command = isa::encode_icu_nop(3);
    QueueProgram writeReadQueue {QueueKind::Mem, 0, {nop}};
    const auto rawWriteRead = raw_mem_3d_commands(
        isa::encode_mem_icu_write_read_2d_instruction(writeRead));
    writeReadQueue.commands.insert(writeReadQueue.commands.end(),
        rawWriteRead.begin(), rawWriteRead.end());
    InstructionControlUnit writeReadIcu;
    load_queue_programs_into_icu({writeReadQueue}, writeReadIcu);
    auto& writeReadMem = writeReadIcu.mem_iq(0);
    require(writeReadMem.imem_occupancy() == 4,
        "WRITE_READ_2D must occupy three MEM iMEM words plus NOP");
    writeReadMem.configure_all();
    for (std::size_t cycle = 0; cycle < 5; ++cycle)
        writeReadMem.prefetch_only();
    std::vector<std::size_t> issueCycles;
    std::vector<MemInstruction> writeReadIssued;
    for (std::size_t cycle = 0; cycle < 64 && !writeReadMem.done();
         ++cycle) {
        if (const auto instruction = writeReadMem.tick()) {
            issueCycles.push_back(cycle);
            writeReadIssued.push_back(*instruction);
        }
    }
    require(writeReadMem.done() && writeReadIssued.size() == 8,
        "WRITE_READ_2D did not retire eight FU issues");
    require(issueCycles == std::vector<std::size_t>({8, 10, 16, 18,
                28, 30, 36, 38}),
        "WRITE_READ_2D start_wait shifted FU issue cycles");
    for (std::size_t event = 0; event < writeReadIssued.size(); ++event) {
        const auto& instruction = writeReadIssued[event];
        const std::size_t address = std::array<std::size_t, 4>{
            30, 31, 34, 35}[event / 2];
        require(instruction.address == address
                && instruction.opcode ==
                    (event % 2 == 0 ? MemOpcode::Write : MemOpcode::Read),
            "WRITE_READ_2D FU operation or address mismatch");
    }

    std::cout << "icu_program_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_program_test failed: " << error.what() << '\n';
    return 1;
}
