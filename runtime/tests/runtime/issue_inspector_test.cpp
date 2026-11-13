#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace ftlpu;
using namespace ftlpu::software::runtime;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
QueueCommand mem(MemInstruction instruction,IcuMacroSchedule schedule){
    const auto raw=isa::encode_mem_instruction(instruction);
    return encode_macro_schedule_command({0,InstructionKind::Mem,2,{std::uint32_t(raw),std::uint32_t(raw>>32),0,0}},schedule);
}
int main() try {
    BinaryProgram program;program.max_cycle=90;
    IcuProgram expected;
    QueueProgram queue{QueueKind::Mem,0,{}};
    for(std::size_t block=0;block<4;++block){
        auto address=100+(block/2)*64+(block%2)*16;
        queue.commands.push_back(mem(MemInstruction::Read(address,2),
            {5+block*20,2,2,1,2,7,4,IcuInductionTarget::MemAddress}));
        for(std::size_t outer=0;outer<2;++outer)for(std::size_t inner=0;inner<2;++inner)
            expected.emit_mem(5+block*20+outer*7+inner*2,0,
                MemInstruction::Read(address+outer*4+inner,2));
    }
    program.queues.push_back(queue);
    // Both SXM queues preserve the complete FU payload, including high map lanes.
    SxmInstruction::StreamList streams;for(std::size_t i=0;i<16;++i)streams.push_back({i});
    auto transpose=SxmInstruction::Transpose({{2},{3}},streams);transpose.input_row=1;
    program.queues.push_back({QueueKind::SxmTranspose,0,{encode_macro_schedule_command(
        encode_sxm_command(transpose),{2,2,7,0,1,1,0,IcuInductionTarget::None})}});
    expected.emit_sxm_transpose(2,Hemisphere::East,transpose);expected.emit_sxm_transpose(9,Hemisphere::East,transpose);
    SxmInstruction::PermuteMap map{};for(std::size_t i=0;i<map.size();++i)map[i]=map.size()-1-i;
    auto permute=SxmInstruction::Permute(streams,streams,map);permute.output_row=11;permute.output_tile=1;
    program.queues.push_back({QueueKind::SxmPermute,0,{encode_macro_schedule_command(
        encode_sxm_command(permute),{3,2,11,0,1,1,0,IcuInductionTarget::None})}});
    expected.emit_sxm_permute(3,Hemisphere::East,permute);expected.emit_sxm_permute(14,Hemisphere::East,permute);
    BinaryProgram expanded;expanded.max_cycle=program.max_cycle;expanded.queues=expected.encode_queues();
    require(compare_logical_issues(program,expanded).equivalent,"Macro inspection changed FU values or cycles");
    require(inspect_logical_issues(program).functional_issues==20,"expanded issue count incorrect");
    auto changed=program;changed.queues[0].commands[0].words[0]^=1u<<4;
    require(!compare_logical_issues(program,changed).equivalent,"operand difference missed");
    changed=program;changed.max_cycle++;
    require(!compare_logical_issues(program,changed).equivalent,"horizon difference missed");
    bool rejected=false;try{(void)inspect_logical_issues(program,2);}catch(const std::exception&){rejected=true;}
    require(rejected,"inspection ignored expansion bound");
    const auto capacity=analyze_physical_imem(program);
    require(capacity.fits() && capacity.queues[1].macro_context_bits==256
        && capacity.queues[2].macro_context_bits==384,"SXM context limits changed");
    changed=program;changed.queues[0].commands.push_back(queue.commands.front());
    rejected=false;try{(void)analyze_physical_imem(changed);}catch(const std::exception&){rejected=true;}
    require(rejected,"interleaved Macro accepted by capacity validation");
    std::stringstream binary;write_binary_program(program,binary);
    require(compare_logical_issues(program,read_binary_program(binary)).equivalent,"packed binary changed logical issues");
    std::cout<<"issue_inspector_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
