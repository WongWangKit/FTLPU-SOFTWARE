#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;

    constexpr std::size_t kVectors = 7;
    const auto packet =
        InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(
            kVectors, 17, 3, 2,
            MemInstruction::Write(64, StreamId::West(5)));
    const auto commands = encode_mem_synchronized_icu_packet(packet);
    const auto wait = encode_icu_control_raw_word(
        IcuControlInstruction::WaitEvent(17));

    BinaryProgram program;
    program.queues.push_back(QueueProgram {QueueKind::Mem, 0,
        {wait, commands[0], commands[1]}});

    const auto abstract = analyze_cmodel_abstract_imem(program);
    require(abstract.queues.size() == 1
            && abstract.queues[0].used_slots == 3
            && abstract.queues[0].coarse_program_entries == 1
            && abstract.queues[0].repeat_2d_entries == 0
            && abstract.queues[0].expanded_work == kVectors
            && abstract.encoded_work_entries == 1,
        "abstract iMEM accounting did not treat MEM_WRITE_SYNC as one coarse instruction");

    const auto physical = analyze_physical_imem(program);
    require(physical.queues.size() == 1
            && physical.queues[0].physical_slots == 3
            && physical.queues[0].physical_bits
                == 3 * program.hardware.icu_mem_instruction_bits
            && physical.used_slots == 3,
        "physical iMEM accounting did not preserve the two-word MEM_WRITE_SYNC packet");

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
    const auto raw = isa::encode_mem_icu_write_read_2d_instruction(writeRead);
    BinaryProgram writeReadProgram;
    QueueProgram writeReadQueue {QueueKind::Mem, 0, {}};
    for (const auto& word : raw.words) {
        QueueCommand command{};
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mem;
        command.word_count = raw.kLanesPerWord;
        for (std::size_t lane = 0; lane < raw.kLanesPerWord; ++lane)
            command.words[lane] = word.lanes[lane];
        writeReadQueue.commands.push_back(command);
    }
    writeReadProgram.queues.push_back(std::move(writeReadQueue));
    const auto writeReadAbstract =
        analyze_cmodel_abstract_imem(writeReadProgram);
    require(writeReadAbstract.queues[0].used_slots == 3
            && writeReadAbstract.queues[0].coarse_program_entries == 1
            && writeReadAbstract.queues[0].expanded_work == 8,
        "WRITE_READ_2D iMEM accounting must count one 3-word packet and eight FU issues");

    std::cout << "imem_capacity_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "imem_capacity_test failed: " << error.what() << '\n';
    return 1;
}
