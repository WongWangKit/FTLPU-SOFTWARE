#pragma once
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/affine_domains.hpp"
#include <map>
namespace ftlpu::test {
class DirectMacroFixture {
    using QueueKey=std::pair<software::runtime::QueueKind,std::size_t>;
    std::map<QueueKey,software::runtime::QueueProgram> queues;
    void add(QueueKey key,software::runtime::QueueCommand command){
        auto& q=queues[key];q.kind=key.first;q.index=key.second;q.commands.push_back(std::move(command));
    }
public:
    void mem(std::size_t queue,std::size_t cycle,const compiler::MemAddressDomain& domain){
        // These standalone kernel fixtures use affine rectangles. The production
        // compiler handles blocked and higher-rank domains in its materializer.
        if(domain.loop.counts[2]!=1)throw std::invalid_argument("fixture expects a rectangle");
        const auto instruction=compiler::detail::mem_domain_instruction(domain,{});
        const auto raw=isa::encode_mem_instruction(instruction);
        software::runtime::QueueCommand native{0,software::runtime::InstructionKind::Mem,2,
            {std::uint32_t(raw),std::uint32_t(raw>>32),0,0}};
        add({software::runtime::QueueKind::Mem,queue},software::runtime::encode_macro_schedule_command(native,
            {cycle,domain.loop.counts[0],domain.loop.cycle_strides[0],domain.address.inner_stride,
             domain.loop.counts[1],domain.loop.cycle_strides[1],domain.address.middle_stride,IcuInductionTarget::MemAddress}));
    }
    void sxm(std::size_t side,std::size_t cycle,std::size_t count,std::size_t interval,const SxmInstruction& instruction){
        const auto kind=instruction.opcode==SxmOpcode::Permute?software::runtime::QueueKind::SxmPermute:software::runtime::QueueKind::SxmTranspose;
        add({kind,side},software::runtime::encode_macro_schedule_command(software::runtime::encode_sxm_command(instruction),
            {cycle,count,interval,0,1,1,0,IcuInductionTarget::None}));
    }
    void load(InstructionControlUnit& icu){
        std::vector<software::runtime::QueueProgram> program;
        for(auto& [key,queue]:queues)program.push_back(std::move(queue));
        software::runtime::load_queue_programs_into_icu(program,icu);
    }
};
}
