#include "ftlpu/software/runtime/macro_queue_link.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"
#include <iostream>
#include <stdexcept>
using namespace ftlpu;
using namespace ftlpu::software::runtime;
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
QueueCommand read(std::size_t address,std::size_t start){
    const auto raw=isa::encode_mem_instruction(MemInstruction::Read(address,0));
    QueueCommand native{0,InstructionKind::Mem,2,{std::uint32_t(raw),std::uint32_t(raw>>32),0,0}};
    return encode_macro_schedule_command(native,{start,4,2,1,1,1,0,IcuInductionTarget::MemAddress});
}
int main()try{
    QueueProgram source{QueueKind::Mem,0,{read(100,10),
        encode_icu_control_raw_word(IcuControlInstruction::Sync()),
        encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(77)),read(200,30)}};
    auto result=link_macro_queue_controls(source,16,true);
    require(result.queue.commands.size()==2 && result.controls.entries.size()==2,"linker lost Macro or controls");
    require(result.controls.entries[0].local_cycle==17 && result.controls.entries[1].local_cycle==17,"controls changed logical phase");
    require(result.command_indices[0]==std::vector<std::size_t>{0} && result.command_indices[3]==std::vector<std::size_t>{1},"relocation map changed");
    require(result.queue.commands[0].extension_words==source.commands[0].extension_words
        && result.queue.commands[1].extension_words==source.commands[3].extension_words,"linker rewrote Macro domains");
    bool rejected=false;try{(void)link_macro_queue_controls(source);}catch(const std::exception&){rejected=true;}
    require(rejected,"controls were silently discarded");
    source.commands={read(100,10),read(200,11)};
    rejected=false;try{(void)link_macro_queue_controls(source);}catch(const std::exception&){rejected=true;}
    require(rejected,"interleaving Macro domains accepted");
    std::cout<<"macro_queue_link_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
