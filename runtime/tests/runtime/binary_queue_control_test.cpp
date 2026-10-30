#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/queue_transport_order.hpp"
#include <algorithm>
#include "ftlpu/system/c2c_dma_system.hpp"
#include "ftlpu/software/runtime/fu_3d_macro_lowering.hpp"
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace ftlpu;
using namespace ftlpu::software::runtime;
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool failed=false;try {f();}catch(const std::exception&){failed=true;}require(failed,"invalid QCTL accepted"); }
std::string encode(const BinaryProgram& p) { std::ostringstream s(std::ios::binary);write_binary_program(p,s);return s.str(); }
std::span<const std::uint8_t> bytes(const std::string& s) {return {reinterpret_cast<const std::uint8_t*>(s.data()),s.size()};}
void put(std::string& s,std::size_t offset,std::uint32_t x) {for(int b=0;b<4;++b) s[offset+b]=static_cast<char>(x>>(8*b));}
void all_readers(const std::string& s,bool valid) {
    for(unsigned path=0;path<4;++path) {
        auto read=[&]{
            std::istringstream stream(s,std::ios::binary);
            auto p=path==0 ? read_binary_program(stream):path==1 ? read_binary_program(bytes(s)):
                path==2 ? read_binary_program_metadata(stream):read_binary_program_metadata(bytes(s));
            require(p.queue_control_protocol,"QCTL flag was lost");
            return p;
        };
        if(valid) read(); else rejects(read);
    }
}
BinaryProgram fixture() {
    BinaryProgram p;p.queue_control_protocol=true;p.max_cycle=40;
    QueueProgram q{QueueKind::Mem,0,{encode_icu_control_raw_word(IcuControlInstruction::Nop(20)),
        encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(7)),
        encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(8)),
        encode_icu_control_raw_word(IcuControlInstruction::Sync()),
        encode_icu_control_raw_word(IcuControlInstruction::Notify())}};
    const auto bits=isa::encode_mem_instruction(MemInstruction::Read(10,0));
    q.commands.push_back({static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,2,{static_cast<std::uint32_t>(bits),static_cast<std::uint32_t>(bits>>32),0,0}});
    auto lowered=lower_fu_3d_queue_to_macro(q,65536,true);
    require(lowered.controls.entries.size()==4,"lowering lost controls");
    require(decode_macro_schedule_command(lowered.queue.commands[0]).start_cycle==20,"control slots advanced Lq");
    lowered.queue.requires_raw_macro=true;
    p.queues.push_back(lowered.queue);
    p.queue_sync_plan.partitions.push_back(lowered.controls);
    p.queue_sync_plan.events={{7,QueueEventSource::External,{}},{8,QueueEventSource::External,{}}};
    // Native queue retains its actual control; this directory is a binding only.
    q.index=1;p.queues.push_back(q);
    auto binding=lowered.controls;binding.index=1;
    for(auto& c:binding.entries) c.native_binding=true;
    p.queue_sync_plan.partitions.push_back(binding);
    return p;
}
void sync_projection_relocations() {
    BinaryProgram p;p.max_cycle=40;
    const auto packet=InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(
        1,31,0,1,MemInstruction::Read(10,0),5);
    auto words=encode_mem_synchronized_icu_packet(packet);
    QueueProgram q{QueueKind::Mem,0,{words.begin(),words.end()}};
    const auto bits=isa::encode_mem_instruction(MemInstruction::Read(20,0));
    q.commands.push_back({static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,2,{static_cast<std::uint32_t>(bits),static_cast<std::uint32_t>(bits>>32),0,0}});
    p.queues.push_back(q);
    p.address_relocations={{7,BindingAccess::Input,QueueKind::Mem,0,0},{8,BindingAccess::Input,QueueKind::Mem,0,2}};
    auto projected=without_compiled_mem_read_sync(p);
    require(projected.address_relocations.size()==1 && projected.address_relocations[0].binding_index==8
        && projected.address_relocations[0].command_index==1,"sync projection did not preserve only live relocations");
    externalize_native_queue_controls(projected);
    require(projected.address_relocations.size()==1 && projected.address_relocations[0].command_index==0,
        "surviving relocation was not mapped to Macro");
    externalize_native_queue_controls(p);
    require(p.transport_queues.size()==1 && p.address_relocations[0].transport_queue,
        "transport relocation did not follow Native packet store");
    all_readers(encode(p),true);
    auto bad=p;bad.queue_sync_plan.partitions[0].entries[0].operand=2;
    rejects([&]{encode(bad);});
    bad=p;bad.transport_queues.clear();rejects([&]{encode(bad);});
}
void link_source_roundtrip() {
    auto f=fixture();BinaryProgram native;native.max_cycle=f.max_cycle;
    native.queues={f.queues[1]};
    native.address_relocations={{7,BindingAccess::Input,QueueKind::Mem,1,5}};
    auto encoded=native;
    externalize_native_queue_controls(encoded);
    encoded.queue_link_source={true,native.queues,native.address_relocations,{}};
    auto data=encode(encoded);all_readers(data,true);
    // A candidate's page events are resolved by final linking. The parser
    // must read QLNK.required before applying final-image association checks.
    auto candidate=encoded;
    candidate.weight_page_uses={{0,0,0,0,20,true}};
    all_readers(encode(candidate),true);
    candidate.queue_link_source.required=false;
    rejects([&]{encode(candidate);});

    const auto restored=native_queue_link_input(read_binary_program(bytes(data)));
    require(encode(restored)==encode(native),"QLNK changed compact Native plan on roundtrip");
    auto relinked=restored;externalize_native_queue_controls(relinked);
    auto original=encoded;original.queue_link_source={};
    require(encode(relinked)==encode(original),"checked relink changed Macro control mapping");
    const auto offset=data.find("QLNK");require(offset!=std::string::npos,"QLNK absent");
    for(unsigned i=0;i<7;++i) {
        auto bad=data;
        if(i==0)bad[offset]='X';
        if(i==1)bad[offset+4]=2;
        if(i==2)bad[offset+6]=2;
        if(i==3)put(bad,offset+8,4097);
        if(i==4)put(bad,offset+20,1);
        if(i==5)put(bad,offset+28,1000001);
        if(i==6)bad.resize(offset+30);
        all_readers(bad,false);
    }
    auto bad=encoded;bad.queue_link_source.address_relocations[0].command_index=99;
    rejects([&]{encode(bad);});
}
void transport_dependency_order() {
    BinaryProgram p;p.queue_control_protocol=true;
    p.weight_page_uses.resize(2);
    p.weight_page_uses[0].transport_launch_event_id=101;
    p.weight_page_uses[1].transport_launch_event_id=102;
    p.queue_sync_plan.events={{7,QueueEventSource::PageCommit,{0}}};
    const auto packet=static_cast<IcuControlOpcode>(8);
    p.queue_sync_plan.partitions={{QueueKind::Mem,0,{
        {0,packet,false,1,0},{0,IcuControlOpcode::WaitEvent,false,7,2},{0,packet,false,2,3}},1},
        {QueueKind::C2cRx,0,{{0,IcuControlOpcode::WaitEvent,true,101,0},
            {1,IcuControlOpcode::WaitEvent,true,102,1}},2}};
    const std::map<std::pair<std::size_t,std::size_t>,std::uint32_t> packets={{{0,0},101},{{0,2},102}};
    validate_queue_transport_order(p,packets);
    auto bad=p;std::swap(bad.queue_sync_plan.partitions[0].entries[0],bad.queue_sync_plan.partitions[0].entries[1]);
    rejects([&]{validate_queue_transport_order(bad,packets);}); // waits for its own future MEM packet
    bad=p;std::swap(bad.queue_sync_plan.partitions[1].entries[0],bad.queue_sync_plan.partitions[1].entries[1]);
    rejects([&]{validate_queue_transport_order(bad,packets);}); // opposite RX/MEM page order
}
void early_event_after_load() {
    for(const bool raw:{false,true}) {
        const auto f=fixture();BinaryProgram p;p.max_cycle=f.max_cycle;p.queues={f.queues[1]};
        externalize_native_queue_controls(p,raw);
        auto system=std::make_unique<TspSliceSystem>();CModelRuntime runtime(*system);
        runtime.load(p);
        auto& q=system->icu().mem_iq(1);
        q.notify(7);q.notify(8);q.notify(); // arrival before physical cycle zero
        runtime.run_cycles(30);
        require(q.issued_count()==1 && q.done(),"load lost a cycle-zero Native/pseudo notification");
    }
}
int main(int argc,char** argv) try {
    if(argc==3 && std::string_view(argv[1])=="--validate-final") {
        const auto program=read_binary_program(argv[2]);
        auto system=std::make_unique<C2cDmaSystem>();
        CModelRuntime runtime(*system,[&](auto sinks){system->tick(sinks);});
        runtime.load(program);
        std::cout<<"final image load passed: queue boundaries, transport dependency graph, SRAM targets and finite PRIME\n";
        return 0;
    }

    if(argc==4 && std::string_view(argv[1])=="--compare-final") {
        auto native=read_binary_program(argv[2]);auto actual=read_binary_program(argv[3]);
        const auto plan=native.queue_sync_plan;
        externalize_native_queue_controls(native,true,[&](QueueSyncPlan& mapped) {
            for(auto& p:mapped.partitions) {
                const auto before=std::ranges::find_if(plan.partitions,[&](const auto& q){return q.kind==p.kind && q.index==p.index;});
                if(before==plan.partitions.end()) {require(p.entries.empty(),"unexpected final control partition");continue;}
                require(before->entries.size()==p.entries.size(),"final control count changed");
                for(std::size_t i=0;i<p.entries.size();++i) {
                    p.entries[i].release_group=before->entries[i].release_group;
                    p.entries[i].phase_offset=before->entries[i].phase_offset;
                    p.entries[i].activation_producer=before->entries[i].activation_producer;
                }
            }
        });
        // Host-only recipes do not enter execution. Everything else, including
        // packed instruction bytes, relocation, event and phase plans, must match.
        native.queue_link_source={};actual.queue_link_source={};
        require(encode(native)==encode(actual),"final Native and Macro executable images are not equivalent");
        std::cout<<"final image equivalence passed: computation, transport, operands, relocations, events and release phases\n";
        return 0;
    }
    if(argc==3) {
        auto p=without_compiled_mem_read_sync(read_binary_program(argv[1]));
        p.hardware.queue_control_version=hw::kQueueControlVersion;
        p.hardware.queue_control_capacity=hw::kQueueControlCapacity;
        p.hardware.queue_control_cache_depth=hw::kQueueControlCacheDepth;
        p.hardware.queue_control_read_latency=hw::kQueueControlReadLatency;
        p.hardware.queue_control_read_ports=hw::kQueueControlReadPorts;
        p.target_abi=executable_target_abi(p.hardware);
        externalize_native_queue_controls(p);write_binary_program(p,argv[2]);
        auto icu=std::make_unique<InstructionControlUnit>();
        load_transport_queue_programs_into_icu(p.transport_queues,*icu);
        const auto prime=load_queue_programs_into_icu(p.queues,*icu,p.hardware.mxms_per_hemisphere,&p);
        std::cout<<"final encoding/physical frontend priming passed: "<<prime<<" cycles\n";return 0;
    }
    sync_projection_relocations();link_source_roundtrip();early_event_after_load();transport_dependency_order();
    BinaryProgram empty;empty.queue_control_protocol=true;all_readers(encode(empty),true);
    auto p=fixture();auto data=encode(p);all_readers(data,true);
    auto zero=p;
    zero.queue_sync_plan.events[0].event_id=0;
    for(auto& partition:zero.queue_sync_plan.partitions)partition.entries[0].operand=0;
    zero.queues[1].commands[1]=encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0));
    all_readers(encode(zero),true); // tagged zero is legal Native WAIT, distinct from SYNC

    const auto qctl=data.find("QCTL");require(qctl!=std::string::npos,"missing QCTL");
    const auto decoded=read_binary_program(bytes(data));
    require(decoded.queue_sync_plan.partitions[0].entries.size()==4 &&
        decoded.queue_sync_plan.partitions[1].entries[0].native_binding,"binding changed on roundtrip");
    for(unsigned mutation=0;mutation<12;++mutation) {
        auto bad=p;auto& c=bad.queue_sync_plan.partitions[0].entries[0];
        if(mutation==0) bad.hardware.queue_control_version=0;
        if(mutation==1) bad.hardware.queue_control_capacity=3;
        if(mutation==2) c.local_cycle=std::numeric_limits<std::uint32_t>::max();
        if(mutation==3) c.opcode=IcuControlOpcode::Repeat;
        if(mutation==4) c.operand=99;
        if(mutation==5) bad.queue_sync_plan.partitions[0].index=65535;
        if(mutation==6) c.source_command=2;
        if(mutation==7) c.native_binding=true;
        if(mutation==8) bad.queue_sync_plan.partitions[1].entries[0].source_command=0;
        if(mutation==9) bad.common_barrier_protocol=true;
        if(mutation==10) bad.max_cycle=std::uint64_t{1}<<32;
        if(mutation==11) bad.queue_sync_plan.events[0].source=QueueEventSource::PageCommit;
        rejects([&]{encode(bad);});
    }
    for(unsigned mutation=0;mutation<8;++mutation) {
        auto bad=data;
        if(mutation==0) bad[qctl]='X';
        if(mutation==1) bad[qctl+4]=2;
        if(mutation==2) bad[qctl+6]=23;
        if(mutation==3) put(bad,qctl+8,5000);
        if(mutation==4) put(bad,qctl+20,257);
        if(mutation==5) bad[qctl+28+6]=4;
        if(mutation==6) put(bad,qctl+28+8,99);
        if(mutation==7) bad.resize(qctl+30);
        all_readers(bad,false);
    }
    std::cout<<"QCTL passed: four readers, empty/mixed partitions, control stripping, malformed data and bindings\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
