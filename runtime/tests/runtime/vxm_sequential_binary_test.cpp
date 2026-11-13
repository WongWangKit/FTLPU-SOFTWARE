#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace ftlpu;
using namespace ftlpu::software::runtime;
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
int main(int argc,char** argv)try{
    require(argc==2,"usage: vxm_sequential_binary_test program.ftlpu");
    VxmLaneAluInstruction lane{};lane.operation=VxmAluOpcode::Multiply;
    lane.lhs=VxmLaneOperand::StreamBFloat16();lane.rhs=VxmLaneOperand::Imm(0.25f);
    lane.repeat_count=32;lane.output_type=VxmCastTarget::BFloat16;
    const auto config=isa::encode_vxm_instruction(0,VxmChainDepth::Two,lane);
    QueueCommand native{0,InstructionKind::Vxm,3,{std::uint32_t(config.control),std::uint32_t(config.control>>32),config.immediate_bits,0}};
    BinaryProgram program;program.max_cycle=73;
    program.queues.push_back({QueueKind::Vxm,0,{{isa::encode_icu_nop(2)},native,{isa::encode_icu_nop(39)},native}});
    auto path=std::filesystem::path(argv[1]);std::filesystem::create_directories(path.parent_path());
    write_binary_program(program,path);const auto decoded=read_binary_program(path);
    const auto report=analyze_physical_imem(decoded);
    require(report.queues.size()==1 && report.used_slots==4,"one config must cost one physical word");
    require(report.queues[0].peak_macro_contexts==0,"VXM must have no loop context");
    auto images=prepare_program_images(decoded.queues,0,decoded.hardware.mxms_per_hemisphere);
    require(images.size()==1 && images[0].descriptor.byte_count==48,"physical image must contain four 96-bit words");
    InstructionControlUnit icu;load_queue_programs_into_icu(decoded.queues,icu);
    require(icu.vxm_iq(0).read_committed_imem(1).lanes==isa::encode_vxm_icu_instruction(config).lanes,"physical config bytes differ");
    std::vector<std::size_t> cycles;
    for(std::size_t i=0;i<80;++i)if(auto got=icu.vxm_iq(0).tick()){
        require(*got==config,"decoded configuration differs");cycles.push_back(i);
    }
    require(cycles==std::vector<std::size_t>{2,42},"WAIT changed sequential issue timing");
    auto obsolete=native;obsolete.command=3;obsolete.extension_words={0x56534e44u,0,2,1,1,1,1,1,1};
    bool rejected=false;try{InstructionControlUnit invalid;load_queue_programs_into_icu({{QueueKind::Vxm,0,{obsolete}}},invalid);}catch(const std::exception&){rejected=true;}
    require(rejected,"obsolete VXM launch descriptor accepted");
    std::cout<<"vxm_sequential_binary_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
