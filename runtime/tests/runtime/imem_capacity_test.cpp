#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include <memory>

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

    // A drain split must preserve absolute issue cycles and raw header bits.
    BinaryProgram segmented;
    segmented.max_cycle=25;
    segmented.timelines.push_back({"node.imem.drain.test",0,20});
    VxmLaneAluInstruction lane{};
    lane.operation=VxmAluOpcode::Bypass;
    lane.lhs=VxmLaneOperand::Imm(1.f);
    lane.rhs=VxmLaneOperand::Imm(1.f);
    const auto instruction=isa::encode_vxm_instruction(0,VxmChainDepth::Two,lane);
    QueueProgram queue{QueueKind::Vxm,0,{}};
    for(auto waitCycles:{3u,21u}) {
        auto run=VxmIcuRun2DInstruction::Run2D(0,{1,1},{1,1},instruction);
        run.loop.wait_cycle=waitCycles;
        const auto encoded=isa::encode_vxm_icu_run_2d_instruction(run);
        for(const auto& word:encoded.words) {
            QueueCommand c{};c.command=word.lanes[0];
            c.instruction_kind=InstructionKind::Vxm;c.word_count=encoded.kLanesPerWord;
            for(std::size_t i=0;i<encoded.kLanesPerWord;++i)c.words[i]=word.lanes[i];
            queue.commands.push_back(c);
        }
    }
    segmented.hardware.icu_vxm_imem_depth=queue.commands.size()/2;
    segmented.queues.push_back(queue);
    const auto parts=split_resident_program_at_drains(segmented);
    require(parts.size()==2,"expected two resident segments");
    std::vector<std::size_t> issued;
    std::size_t offset=0;
    for(const auto& part:parts) {
        require(analyze_physical_imem(part).fits(),"segment exceeds iMEM");
        InstructionControlUnit unit;
        load_queue_programs_into_icu(part.queues,unit);
        for(std::size_t cycle=0;cycle<part.max_cycle;++cycle)
            if(unit.vxm_iq(0).tick())issued.push_back(offset+cycle);
        offset+=part.max_cycle;
    }
    require(issued==std::vector<std::size_t>{3,25},"drain changed issue timing");
    segmented.weight_page_uses.emplace_back();
    bool rejected=false;
    try { static_cast<void>(split_resident_program_at_drains(segmented)); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"resident split must not bypass paged execution");

    // Common-clock compute reload must leave an unsplit MEM queue running.
    BinaryProgram common;
    common.common_barrier_protocol=true;common.max_cycle=2200;
    common.timelines={{"imem.drain.compute",0,1100}};
    QueueProgram waits{QueueKind::Vxm,0,{}};
    const auto appendRun=[](QueueProgram& q,std::size_t delay,bool tail) {
        VxmLaneAluInstruction lane{};
        lane.operation=VxmAluOpcode::Bypass;
        lane.lhs=tail?VxmLaneOperand::Previous():VxmLaneOperand::Imm(1.f);
        lane.rhs=VxmLaneOperand::Imm(1.f);
        lane.output_type=VxmCastTarget::BFloat16;
        if(tail)lane.output_stream=0;
        const auto instruction=isa::encode_vxm_instruction(q.index,VxmChainDepth::Two,lane);
        auto run=VxmIcuRun2DInstruction::Run2D(0,{1,1},{1,1},instruction);
        run.loop.wait_cycle=delay;
        const auto raw=isa::encode_vxm_icu_run_2d_instruction(run);
        for(const auto& word:raw.words) {
            QueueCommand c{};c.command=word.lanes[0];c.instruction_kind=InstructionKind::Vxm;
            c.word_count=raw.kLanesPerWord;
            for(std::size_t j=0;j<raw.kLanesPerWord;++j)c.words[j]=word.lanes[j];
            q.commands.push_back(c);
        }
    };
    appendRun(waits,3,false);
    for(std::size_t i=0;i<2100;++i)
        waits.commands.push_back(encode_icu_control_raw_word(IcuControlInstruction::Nop(1)));
    appendRun(waits,0,false);
    common.queues.push_back(waits);
    QueueProgram tail{QueueKind::Vxm,1,{}};
    appendRun(tail,3,true);appendRun(tail,2100,true);
    common.queues.push_back(tail);
    QueueCommand read{};
    read.command=static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);
    read.instruction_kind=InstructionKind::Mem;read.word_count=2;
    const auto bits=isa::encode_mem_instruction(MemInstruction::Read(0,StreamId::East(0)));
    read.words[0]=static_cast<std::uint32_t>(bits);read.words[1]=static_cast<std::uint32_t>(bits>>32);
    common.queues.push_back({QueueKind::Mem,0,{
        encode_icu_control_raw_word(IcuControlInstruction::Nop(100)),read,
        encode_icu_control_raw_word(IcuControlInstruction::Nop(1999)),read}});
    auto system=std::make_unique<TspSliceSystem>();
    CModelRuntime runtime(*system);
    runtime.load(common);runtime.run_cycles(1100);
    require(system->icu().vxm_iq(0).issued_count()==1,"first compute segment did not issue");
    runtime.run_cycles(1100);
    require(system->icu().vxm_iq(0).issued_count()==1 && system->icu().vxm_iq(1).issued_count()==2,
        "compute reload lost its issue or reset the independent tail queue");
    require(system->icu().mem_iq(0).issued_count()==2,
        "compute reload reset or repeated the independent MEM program");
    require(runtime.logical_cycles()==2200,"compute reload changed the global logical clock");
    const auto& activity=system->vxm_unit().superlane(0).lane(0).statistics();
    require(activity.cycles>=2200 && activity.timeline.size()==1,
        "runtime must retain cumulative VXM statistics without unbounded cycle history");
    std::cout << "imem_capacity_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "imem_capacity_test failed: " << error.what() << '\n';
    return 1;
}
