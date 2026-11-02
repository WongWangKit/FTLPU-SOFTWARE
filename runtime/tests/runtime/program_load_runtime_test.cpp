#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"
#include "ftlpu/core/bf16.hpp"
#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;
namespace {
void require(bool value, const char* message)
{ if (!value) throw std::runtime_error(message); }
template <typename F> void rejects(F action)
{
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("unsupported format was accepted");
}
rt::QueueCommand macro(MemInstruction instruction, std::size_t start,
    std::size_t count = 4)
{
    const auto encoded = isa::encode_mem_instruction(instruction);
    rt::QueueCommand command{static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mem, 2,
        {static_cast<std::uint32_t>(encoded),static_cast<std::uint32_t>(encoded >> 32),0,0}};
    return rt::encode_macro_schedule_command(std::move(command),
        IcuMacroSchedule{start,count,1,1,1,1,0,IcuInductionTarget::MemAddress});
}
rt::BinaryProgram copy_program()
{
    // Slice 0's westward read reaches the passive bridge after one cycle;
    // the eastward destination write follows one further cycle later.
    rt::BinaryProgram program;
    const auto src = InstructionControlUnit::mem_queue(Hemisphere::East, 0, 0);
    const auto dst = InstructionControlUnit::mem_queue(Hemisphere::West, 0, 0);
    program.queues.push_back({rt::QueueKind::Mem,src,
        {macro(MemInstruction::Read(100,StreamId::West(0)),4)}});
    program.queues.push_back({rt::QueueKind::Mem,dst,
        {macro(MemInstruction::Write(200,StreamId::East(0)),6)}});
    program.max_cycle = 16;
    rt::BinaryBinding input;
    input.index = 0; input.access = rt::BindingAccess::Input;
    input.element_type = rt::BindingElementType::I8;
    input.layout = rt::BindingLayout::Vector;
    input.shape = {4,32}; input.byte_size = 128;
    input.base_row = 100; input.address_stride = 1;
    input.slices = {0}; input.hemisphere_mask = 1;
    program.bindings.push_back(input);
    return program;
}
std::vector<std::uint8_t> output(TspSliceSystem& system)
{
    std::vector<std::uint8_t> bytes;
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t b = 0; b < 32; ++b)
            bytes.push_back(system.read_mem_sram_lane_byte(Hemisphere::West,
                0,0,b/hw::kLanesPerTile,200+row,b%hw::kLanesPerTile));
    return bytes;
}

rt::QueueCommand mxm_macro(MxmControlInstruction instruction, std::size_t start,
    std::size_t count, std::int64_t stride, IcuInductionTarget target)
{
    const auto encoded = isa::encode_mxm_instruction(instruction);
    rt::QueueCommand command{static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mxm,2,
        {static_cast<std::uint32_t>(encoded),static_cast<std::uint32_t>(encoded >> 32),0,0}};
    return rt::encode_macro_schedule_command(std::move(command),
        IcuMacroSchedule{start,count,1,stride,1,1,0,target});
}

void numerical_mxm_regression()
{
    rt::BinaryProgram program;
    const auto latency = [](std::size_t slice) {
        return hw::kMemGroups + hw::kC2cToSxmStreamRegisterColumns + 1
            - slice / hw::kMemSlicesPerGroup;
    };
    for (std::size_t slice = 0; slice < 16; ++slice)
        program.queues.push_back({rt::QueueKind::Mem,
            InstructionControlUnit::mem_queue(Hemisphere::East,slice,0),
            {macro(MemInstruction::Read(64,StreamId::East(slice)),20-latency(slice),4)}});
    for (std::size_t slice = 0; slice < hw::kMxmInt8LoadStreamsPerCycle; ++slice)
        program.queues[slice].commands.push_back(
            macro(MemInstruction::Read(90,StreamId::East(slice)),35-latency(slice),1));
    for (std::size_t byte = 0; byte < 2; ++byte)
        program.queues.push_back({rt::QueueKind::Mem,
            InstructionControlUnit::mem_queue(Hemisphere::East,50+byte,0),
            {macro(MemInstruction::Read(80,StreamId::East(byte)),30-latency(50+byte),1)}});
    program.queues.push_back({rt::QueueKind::MxmLoad,0,
        {mxm_macro(MxmControlInstruction::IWDirect16(0,0),20,4,1,
            IcuInductionTarget::MxmWeightColumn),
         mxm_macro(MxmControlInstruction::IW(1,0),35,1,0,IcuInductionTarget::None)}});
    program.queues.push_back({rt::QueueKind::MxmCompute,0,
        {mxm_macro(MxmControlInstruction::Compute(0,0,0,hw::kMxmAccumulatorRows/2,1,
            MxmAccumulatorDestination::Sram,MxmDataFormat::BFloat16),30,1,0,
            IcuInductionTarget::None)}});
    rt::QueueCommand scale{static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::MxmDequant,1,
        {isa::encode_mxm_dequant_instruction(MxmDequantInstruction::Scale(0.125f)),0,0,0}};
    program.queues.push_back({rt::QueueKind::MxmDequant,0,
        {rt::encode_macro_schedule_command(std::move(scale),IcuMacroSchedule{35})}});
    program.max_cycle = 48;
    const auto weight = [](std::size_t row, std::size_t column) {
        return static_cast<float>(static_cast<int>((row*5+column*3)%23)-11)/16.0f;
    };
    const auto activation = [](std::size_t row) {
        return static_cast<float>(static_cast<int>((row*7)%17)-8)/8.0f;
    };
    const auto initialize = [&](TspSliceSystem& system) {
        for (std::size_t row = 0; row < hw::kMxmRows; ++row) {
            for (std::size_t col = 0; col < hw::kMxmColumns; ++col) {
                const auto bits = Bf16::from_float(weight(row,col)).bits();
                for (std::size_t byte = 0; byte < 2; ++byte)
                    system.initialize_mem_sram_lane_byte(Hemisphere::East,
                        (col%hw::kMxmSupercellColumns)*2+byte,0,
                        row/hw::kLanesPerTile,64+col/hw::kMxmSupercellColumns,
                        row%hw::kLanesPerTile,static_cast<std::uint8_t>(bits>>(byte*8)));
            }
            const auto bits = Bf16::from_float(activation(row)).bits();
            for (std::size_t byte = 0; byte < 2; ++byte)
                system.initialize_mem_sram_lane_byte(Hemisphere::East,50+byte,0,
                    row/hw::kLanesPerTile,80,row%hw::kLanesPerTile,
                    static_cast<std::uint8_t>(bits>>(byte*8)));
            for (std::size_t col = 0; col < hw::kMxmSupercellColumns; ++col)
                system.initialize_mem_sram_lane_byte(Hemisphere::East,col,0,
                    row/hw::kLanesPerTile,90,row%hw::kLanesPerTile,
                    static_cast<std::uint8_t>(static_cast<int>((row*13+col*7)%31)-15));
        }
    };
    auto direct = std::make_unique<TspSliceSystem>();
    rt::CModelRuntime reference(*direct);
    reference.load_direct(program); initialize(*direct); reference.run_cycles(49);
    auto system = std::make_unique<C2cDmaSystem>();
    rt::CModelRuntime timed(*system);
    timed.load_timed(program); initialize(system->chip()); timed.run_cycles(49);
    for (std::size_t col = 0; col < hw::kMxmColumns; ++col) {
        float expected = 0;
        for (std::size_t row = 0; row < hw::kMxmRows; ++row)
            expected += Bf16::from_float(weight(row,col)).to_float()
                * Bf16::from_float(activation(row)).to_float();
        const auto actual = system->chip().mxm_unit(0).accumulator().value(hw::kMxmAccumulatorRows/2,col);
        require(std::fabs(actual-expected) <= 1.0e-5f
            && actual == direct->mxm_unit(0).accumulator().value(hw::kMxmAccumulatorRows/2,col),
            "timed MEM/MXM BF16 matrix-vector numerical regression failed");
    }
    require(timed.program_load_statistics().images.size() == 21,
        "numerical program did not load all MEM/MXM/dequant queues");
    for (std::size_t row = 0; row < hw::kMxmRows; ++row)
        for (std::size_t col = 0; col < hw::kMxmSupercellColumns; ++col) {
            const auto expected = Bf16::from_float(
                static_cast<float>(static_cast<int>((row*13+col*7)%31)-15)*0.125f).bits();
            require(system->chip().mxm_unit(0).array().cell(row/hw::kLanesPerTile,0)
                    .weight_bits(1,row%hw::kLanesPerTile,col) == expected,
                "timed MXM INT8 dequant numerical regression failed");
        }
}
void all_physical_formats() {
    rt::BinaryProgram program;
    for(unsigned kind=0;kind<10;++kind) {
        rt::QueueProgram queue{static_cast<rt::QueueKind>(kind),0,{}};
        for(unsigned i=0;i<5;++i) queue.commands.push_back(rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(3)));
        if(queue.kind==rt::QueueKind::C2cDma) {
            const auto packet=rt::encode_c2c_dma_icu_packet(C2cIcuPacketCodec::encode(
                C2cDmaIcuInstruction::Load(Hemisphere::East,0,0x8000,2)));
            queue.commands.insert(queue.commands.begin()+1,packet.begin(),packet.end());
        }
        program.queues.push_back(std::move(queue));
    }
    program.transport_queues={{rt::QueueKind::Mem,1,{rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(4)),
        rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(2)),rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(1))}}};
    const auto images=rt::prepare_program_images(program,0x10000);
    require(images.size()==11,"not all Native/transport formats were physicalized");
    auto system=std::make_unique<C2cDmaSystem>();
    std::vector<ProgramImage> descriptors;
    for(const auto& image:images) {
        descriptors.push_back(image.descriptor);
        for(std::size_t offset=0;offset<image.bytes.size();offset+=hw::kPhysicalVectorBytes) {
            C2cVector vector{};
            for(std::size_t b=0;b<hw::kPhysicalVectorBytes && offset+b<image.bytes.size();++b)
                vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=image.bytes[offset+b];
            system->ddr4().initialize_vector(image.descriptor.source_address+offset,vector);
        }
    }
    system->program_loader().submit(std::move(descriptors),0);
    auto& loader=system->program_loader();loader.start();loader.set_dependencies_ready(true);
    while(!loader.ready_to_start()) {
        require(system->cycle()<5000,"all-format load timed out");system->tick();
        system->chip().icu().for_each_program_queue([](auto& q){
            require(q.issued_count()==0 && q.cycle()==0,"new queue executed during load/prime");
        });
        require(system->dma(Hemisphere::East).queued_instruction_count()==0,"C2C ran before unified start");
    }
    std::size_t effective=0,dma=0,words=0;
    for(const auto& image:images) {
        require(system->chip().icu().read_program_image(image.descriptor)==image.bytes,"Native/transport readback mismatch");
        effective+=image.bytes.size();dma+=image.descriptor.vector_count()*hw::kPhysicalVectorBytes;words+=image.descriptor.word_count();
        if(image.descriptor.kind==IcuMacroQueueKind::C2cDma)
            require(image.bytes.size()==7*12,"DMA packet was not two physical words");
    }
    const auto& stats=loader.statistics();
    require(stats.effective_bytes==effective && stats.dma_bytes==dma && stats.written_words==words,"physical byte accounting mismatch");
    require(*stats.frontend_ready>*stats.all_images_committed,"Native IQ prefetch took no time");
}
void pending_write_resubmit() {
    auto program=copy_program();auto system=std::make_unique<C2cDmaSystem>();rt::CModelRuntime runtime(*system);
    runtime.load_timed(program);std::vector<std::uint8_t> input(128,0x73);runtime.upload_input(0,input);
    bool observed=false;
    for(std::size_t n=0;!runtime.kernel_done();++n) {
        require(n<2000,"copy completion timed out");runtime.run_until_complete(1);
        if(system->chip().icu().program_queues_done() && !system->chip().program_resources_drained()) {
            observed=true;const auto gen=runtime.execution_generation();const auto stats=runtime.program_load_statistics();
            const auto physical=system->cycle();const auto requests=system->ddr4().read_bytes_transferred();
            rejects([&]{runtime.load_timed(program,0x400000);});rejects([&]{runtime.load_direct(program);});
            require(runtime.execution_generation()==gen && runtime.program_load_statistics().request_accepted==stats.request_accepted
                && system->cycle()==physical && system->ddr4().read_bytes_transferred()==requests,
                "pending write rejection changed old task");
        }
    }
    require(observed && runtime.can_load_program(),"completion ignored in-flight writes or drain window was absent");
}

void barr_transport_roundtrip() {
    rt::BinaryBinding binding;binding.index=0;binding.access=rt::BindingAccess::Input;
    binding.element_type=rt::BindingElementType::I8;binding.layout=rt::BindingLayout::W8A16AttentionWeightStriped;
    binding.shape={32,128};binding.byte_size=4096;binding.slices={20,21,22,23,24,25,26,27};
    binding.hemisphere_mask=3;binding.bank=1;binding.paged_weight=true;binding.page_count=1;
    binding.page_rows=8;binding.instruction_count=8;binding.address_stride=1;
    binding.page_granularity=1;binding.page_role_group_count=1;binding.page_items_per_slice_group=1;
    binding.page_bank_count=2;binding.page_storage_slices=binding.slices;binding.role="weight";binding.name="test.weight";
    std::vector<std::uint8_t> values(4096);for(std::size_t i=0;i<values.size();++i)values[i]=static_cast<std::uint8_t>(i*13+19);
    const auto packed=rt::pack_weight_binding_page(binding,0,values,rt::ExecutableHardwareConfig{});
    auto direct=std::make_unique<C2cDmaSystem>();auto system=std::make_unique<C2cDmaSystem>();
    rt::BinaryProgram program;program.common_barrier_protocol=true;program.max_cycle=256;program.bindings={binding};
    program.weight_page_uses={{0,0,1,128,256,true}};program.timelines={{"c2c.window.binding:0.page:0",32,128}};
    rt::C2cWeightPage page;page.bank=1;
    for(const auto& segment:packed.segments)page.segments.push_back({static_cast<Hemisphere>(segment.hemisphere),segment.slice,1,
        segment.base_row,0,0x10000+segment.byte_offset,segment.vector_count});
    rt::C2cWeightPager pager(*system);pager.begin_schedule(program);
    const std::array<std::size_t,1> instances{0};
    pager.schedule(program,page,32,48,128,0x10000,0x30000,32,instances);pager.finalize_schedule(program);
    const auto stage=[&](C2cDmaSystem& device) {
        for(std::size_t offset=0;offset<packed.data.size();offset+=hw::kPhysicalVectorBytes) {
            C2cVector vector{};for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)
                vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=packed.data[offset+b];
            device.ddr4().initialize_vector(0x10000+offset,vector);
        }
    };
    stage(*direct);stage(*system);
    rt::CModelRuntime reference(*direct),timed(*system);
    reference.load_direct(program);reference.upload_input(0,values);
    timed.load_timed(program);timed.upload_input(0,values);
    require(reference.run_until_complete(10000) && timed.run_until_complete(10000),"BARR transport task did not drain");
    require(system->chip().icu().common_barriers_done() && timed.common_transport_committed(program.weight_page_uses[0].transport_launch_event_id),
        "BARR completion was not initialized from committed event bindings");
    for(const auto& segment:packed.segments) for(std::size_t row=0;row<segment.vector_count;++row)
        for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b) {
            const auto h=static_cast<Hemisphere>(segment.hemisphere);
            const auto actual=system->chip().read_mem_sram_lane_byte(h,segment.slice,1,b/hw::kLanesPerTile,segment.base_row+row,b%hw::kLanesPerTile);
            require(actual==packed.data[segment.byte_offset+row*hw::kPhysicalVectorBytes+b]
                && actual==direct->chip().read_mem_sram_lane_byte(h,segment.slice,1,b/hw::kLanesPerTile,segment.base_row+row,b%hw::kLanesPerTile),
                "BARR direct/timed transport output differs");
        }
    const auto oldGeneration=timed.execution_generation();
    rt::BinaryProgram empty;empty.common_barrier_protocol=true;timed.load_timed(empty,0x200000);
    timed.notify_common_page_commit(oldGeneration,0);
    while(!system->program_loader().ready_to_start())system->tick();
    timed.notify_common_page_commit(oldGeneration,0);
    require(timed.run_until_complete(100),"empty BARR partition leaked previous task");
    require(timed.program_load_statistics().dma_bytes==0,"empty BARR/config RAM issued DMA");
}

void qctl_and_empty_controls() {
    auto program=copy_program();program.queue_control_protocol=true;
    for(auto& q:program.queues) {
        q.requires_raw_macro=true;
        program.queue_sync_plan.partitions.push_back({q.kind,static_cast<std::uint16_t>(q.index),{{0,IcuControlOpcode::WaitEvent,false,7,0}},16});
    }
    program.queue_sync_plan.events={{7,rt::QueueEventSource::External,{}}};
    std::vector<std::uint8_t> input(128);for(std::size_t b=0;b<input.size();++b)input[b]=static_cast<std::uint8_t>(b*19+3);
    auto direct=std::make_unique<C2cDmaSystem>();rt::CModelRuntime reference(*direct);
    reference.load_direct(program);reference.upload_input(0,input);
    direct->chip().icu().notify_tagged(IcuLocation::Mem(Hemisphere::East,0,0),7);
    // Destination is on the other hemisphere; both waits receive one token.
    direct->chip().icu().notify_tagged(IcuLocation::Mem(Hemisphere::West,0,0),7);
    require(reference.run_until_complete(1000),"direct QCTL copy did not complete");
    auto system=std::make_unique<C2cDmaSystem>();rt::CModelRuntime timed(*system);
    timed.load_timed(program);timed.upload_input(0,input);
    const auto accepted=timed.program_load_statistics().request_accepted;
    const auto generation=timed.execution_generation();
    rejects([&]{timed.load_timed(program,0x200000);});
    require(timed.execution_generation()==generation && timed.program_load_statistics().request_accepted==accepted,"busy resubmit mutated task");
    while(!system->program_loader().ready_to_start()) {
        require(system->cycle()<2000,"QCTL load timed out");system->tick();
        for(const auto& q:program.queues) require(system->chip().icu().mem_iq(q.index).issued_count()==0,"QCTL gate leaked execution");
    }
    auto images=rt::prepare_program_images(program,0x1'0000'0000ULL);
    require(system->chip().icu().queue_control_pool().used()==program.queue_sync_plan.partitions.size(),
        "timed QCTL did not commit into the shared pool");
    require(system->chip().icu().queue_control_pool().statistics().prime_reads==program.queue_sync_plan.partitions.size(),
        "timed QCTL PRIME bypassed shared RAM reads");
    for(const auto& image:images) {
        require(system->chip().icu().read_program_image(image.descriptor)==image.bytes,"QCTL/event RAM bytes differ");
        if(image.descriptor.storage==ProgramStorage::Qctl) require(image.bytes.size()==12,"QCTL host metadata was transferred");
    }
    // Run with no external event: timeout must retain a resumable task.
    require(!timed.run_until_complete(2),"QCTL wait completed without its token");
    system->chip().icu().notify_tagged(IcuLocation::Mem(Hemisphere::East,0,0),7);
    system->chip().icu().notify_tagged(IcuLocation::Mem(Hemisphere::West,0,0),7);
    require(timed.run_until_complete(1000),"timed QCTL copy did not complete");
    require(output(system->chip())==input && output(direct->chip())==input,"QCTL direct/timed output mismatch");
    for(const auto& q:program.queues) require(system->chip().icu().mem_iq(q.index).cycle()==direct->chip().icu().mem_iq(q.index).cycle(),"QCTL local release phase changed");
    const auto physical=system->cycle();
    for(auto& partition:program.queue_sync_plan.partitions)partition.entries.clear();
    program.queue_sync_plan.events.clear();
    timed.load_timed(program,0x200000);timed.upload_input(0,input);
    require(timed.run_until_complete(1000),"empty QCTL partition did not complete");
    require(system->cycle()>physical && output(system->chip())==input,"empty control partition leaked old waits");
    for(const auto& image:timed.program_load_statistics().images) if(image.image.storage==ProgramStorage::Qctl || image.image.storage==ProgramStorage::QueueEvents)
        require(image.written_words==0 && image.dma_bytes==0 && !image.first_ddr_request,"empty control partition issued DMA");
    rt::BinaryProgram only;only.queue_control_protocol=true;
    only.queues={{rt::QueueKind::Mem,0,{}}};only.queues[0].requires_raw_macro=true;
    only.queue_sync_plan.partitions={{rt::QueueKind::Mem,0,{{0,IcuControlOpcode::WaitEvent,false,17,0}},0}};
    only.queue_sync_plan.events={{17,rt::QueueEventSource::External,{}}};
    const auto oldControlGeneration=timed.execution_generation();
    timed.load_timed(only,0x300000);
    timed.notify_common_page_commit(oldControlGeneration,0);
    while(!system->program_loader().ready_to_start())system->tick();
    require(!timed.run_until_complete(1),"control-only queue ended before WAIT retired");
    system->chip().icu().notify_tagged(IcuLocation::Mem(Hemisphere::East,0,0),17);
    require(timed.run_until_complete(100),"control-only Macro queue did not complete");
    require(timed.program_load_statistics().images[0].effective_bytes==12,"empty Macro header lost its physical length");

}

}

int main() try
{
    const auto check=[](const char* name,auto test) {
        try {test();}catch(const std::exception& ex){throw std::runtime_error(std::string(name)+": "+ex.what());}
    };
    check("pending writes",pending_write_resubmit);
    check("all formats",all_physical_formats);
    check("BARR transport",barr_transport_roundtrip);
    check("QCTL",qctl_and_empty_controls);
    check("MXM numerical",numerical_mxm_regression);
    auto program = copy_program();
    const auto images = rt::prepare_program_images(program.queues,0x100000);
    require(images.size() == 2 && images[0].bytes.size() % 12 == 0,
        "runtime image preparation lost physical word boundaries");
    std::vector<std::uint8_t> input(128);
    for (std::size_t b = 0; b < input.size(); ++b) input[b] = static_cast<std::uint8_t>(b * 37 + 11);

    auto direct = std::make_unique<TspSliceSystem>();
    rt::CModelRuntime legacy(*direct);
    legacy.load_direct(program); legacy.upload_input(0,input); legacy.run_cycles(17);
    require(output(*direct) == input, "reference copy program produced incorrect values");

    auto system = std::make_unique<C2cDmaSystem>();
    rt::CModelRuntime runtime(*system,[&](TspSliceSystem::LogSinks sinks) { system->tick(sinks); });
    runtime.load_timed(program);
    require(system->cycle() == 0 && !runtime.all_images_committed(),
        "submission preloaded i-MEM or advanced time synchronously");
    // Preparation and prefetch may finish while required input is still absent.
    for (std::size_t timeout = 0; !system->program_loader().ready_to_start(); ++timeout) {
        require(timeout < 1000,"runtime loader did not prime"); system->tick();
    }
    require(!runtime.ready_to_start(), "missing input did not gate start");
    runtime.upload_input(0,input);
    std::size_t weightWaitTicks = 0;
    // A first-batch weight dependency must also gate logical cycle zero.
    auto weightProgram = program;
    // External controller reports initial weight residency through the startup gate.
    runtime.run_cycles(17);
    require(output(system->chip()) == input, "timed copy program differs from reference");
    const auto first = runtime.program_load_statistics();
    require(first.program_start && *first.program_start > *first.frontend_ready,
        "program started before frontend was ready");
    require(runtime.logical_cycles() == 17 && runtime.physical_cycles() == system->cycle(),
        "runtime lost physical load/prefetch cycles");
    require(runtime.run_until_complete(1000),"first copy did not drain");
    const auto physical = system->cycle();
    runtime.load_timed(weightProgram,0x200000);
    runtime.upload_input(0,input);
    runtime.set_program_start_dependency_checker([&]() { return ++weightWaitTicks > 100; });
    runtime.run_cycles(17);
    require(runtime.program_load_statistics().request_accepted == physical
        && *runtime.program_load_statistics().program_start >= physical+100,
        "kernel switch reset physical time or bypassed initial weights");
    require(output(system->chip()) == input, "second timed kernel changed numerical output");

    require(runtime.run_until_complete(1000),"second copy did not drain");
    auto invalid = program;
    invalid.queues.push_back({rt::QueueKind::Vxm,InstructionControlUnit::kVxmQueues,{rt::QueueCommand{}}});
    const auto before = system->cycle();
    rejects([&] { runtime.load_timed(invalid); });
    require(system->cycle() == before && system->program_loader().started(),
        "unsupported queue mutated active program");
    auto shortProgram = program;
    shortProgram.queues.resize(1);
    runtime.load_timed(shortProgram,0x300000);
    runtime.upload_input(0,input); runtime.run_cycles(17);
    require(runtime.program_load_statistics().images.size() == 1,
        "previous queue notifications leaked into new kernel");
    std::cout << "program_load_runtime_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "program_load_runtime_test failed: " << ex.what() << '\n'; return 1;
}
