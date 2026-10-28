#include "ftlpu/compiler/Target/gdn_native_schedule.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

using namespace ftlpu;
void store(TspSliceSystem& s,std::size_t pair,std::size_t index,float v){
    const auto bits=VxmDataFormat::float_to_bf16_bits(v);
    for(std::size_t h=0;h<2;++h)for(std::size_t b=0;b<2;++b)
        s.initialize_mem_sram_lane_byte(static_cast<Hemisphere>(h),pair+b,
            index%32/hw::kLanesPerTile,index/32,index%hw::kLanesPerTile,std::uint8_t(bits>>(8*b)));
}
std::vector<float> load(const std::filesystem::path& p,std::size_t size){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=static_cast<std::streamoff>(size*sizeof(float)))throw std::runtime_error("bad input "+p.string());
    f.seekg(0);std::vector<float> v(size);f.read(reinterpret_cast<char*>(v.data()),size*sizeof(float));return v;
}
void dump(TspSliceSystem& s,const std::filesystem::path& p,std::size_t pair,std::size_t count){
    std::ofstream f(p,std::ios::binary);
    for(std::size_t i=0;i<count;++i){
        std::uint16_t bits=0;
        for(std::size_t b=0;b<2;++b)bits|=std::uint16_t(s.read_mem_sram_lane_byte(Hemisphere::East,pair+b,
            i%32/hw::kLanesPerTile,i/32,i%hw::kLanesPerTile))<<(8*b);
        const auto v=VxmDataFormat::bf16_bits_to_float(bits);f.write(reinterpret_cast<const char*>(&v),4);
        std::uint16_t other=0;
        for(std::size_t b=0;b<2;++b)other|=std::uint16_t(s.read_mem_sram_lane_byte(Hemisphere::West,pair+b,
            i%32/hw::kLanesPerTile,i/32,i%hw::kLanesPerTile))<<(8*b);
        if(other!=bits)throw std::runtime_error("mirrored hemispheres disagree");
    }
}
int main(int argc,char** argv) try {
    if(argc!=3 && argc!=4)throw std::runtime_error("usage: gdn_native_chunkwise_test inputs_dir output_dir [tokens]");
    const std::size_t tokens=argc==4?std::stoul(argv[3]):64;
    if(!tokens || tokens%32 || tokens>1024)throw std::runtime_error("invalid chunk32 token count");
    const std::filesystem::path in=argv[1],out=argv[2];std::filesystem::create_directories(out);
    auto system=std::make_unique<TspSliceSystem>();
    for(std::size_t h=0;h<2;++h){
        system->vxm_unit().configure_input_group_source(h*8,static_cast<Hemisphere>(h));
        system->vxm_unit().configure_input_group_source(h*8+1,static_cast<Hemisphere>(h));
        system->vxm_unit().configure_output_block_destination(h*4,static_cast<Hemisphere>(h));
    }
    for(auto [name,pair,count]:std::vector<std::tuple<const char*,std::size_t,std::size_t>>{
        {"q",10,tokens*128},{"k",12,tokens*128},{"v",14,tokens*128},{"state",16,16384},{"alpha",20,tokens},{"beta",22,tokens}}){
        const auto values=load(in/(std::string(name)+".bin"),count);
        for(std::size_t i=0;i<count;++i)store(*system,pair,i,values[i]);
    }
    // Scratch is deliberately nonzero: all constants and K transpose must
    // be initialized by the emitted program, as in a reused whole-layer arena.
    for(std::size_t i=0;i<33*32;++i)store(*system,6,i,-7.f);
    for(std::size_t i=32*32;i<33*32;++i)store(*system,26,i,-7.f);
    const auto epochs=compiler::GdnNativeLowering{}.lower(tokens);
    std::ofstream spans(out/"epochs.csv"),trace(out/"runtime.pipeline.csv");
    spans<<"epoch,start,end,name\n";trace<<"start,end,resource,detail\n";
    std::size_t global=0,number=0;
    for(const auto& epoch:epochs){
        if(!system->vxm_unit().idle())throw std::runtime_error("VXM not drained at epoch boundary");
        system->icu().reset();
        std::array<std::size_t,InstructionControlUnit::kMemQueues> mc{};
        std::array<std::size_t,8> vc{};std::size_t sc[2][2]{};
        for(const auto& i:epoch.program.mem){
            const auto q=InstructionControlUnit::mem_queue(static_cast<Hemisphere>(i.side),i.slice);
            if(i.cycle<mc[q])throw std::runtime_error(epoch.name+": overlapping MEM domains");
            auto ins=i.instruction;ins.loop.wait_cycle=i.cycle-mc[q];system->icu().enqueue_mem_3d(q,ins);
            mc[q]=i.cycle+(ins.loop.counts[0]-1)*ins.loop.cycle_strides[0]+1;
        }
        for(const auto& i:epoch.program.vxm){
            auto run=VxmIcuRun2DInstruction::Run2D(0,{1,1},{1,1},i.instruction);
            run.loop.wait_cycle=i.cycle-vc[i.queue];system->icu().enqueue_vxm_run_2d(i.queue,run);vc[i.queue]=i.cycle+1;
        }
        for(const auto& i:epoch.program.sxm){
            const bool perm=i.instruction.opcode==SxmOpcode::Permute;
            auto run=SxmIcuRun2DInstruction::Run2D(0,{i.count,1},{i.interval,1},i.instruction);
            run.loop.wait_cycle=i.cycle-sc[i.side][perm];
            if(perm)system->icu().sxm_permute_iq(static_cast<Hemisphere>(i.side)).push_sxm_run_2d(run);
            else system->icu().sxm_transpose_iq(static_cast<Hemisphere>(i.side)).push_sxm_run_2d(run);
            sc[i.side][perm]=i.cycle+(i.count-1)*i.interval+1;
        }
        spans<<number<<','<<global<<','<<global+epoch.program.end_cycle<<','<<epoch.name<<'\n';
        std::size_t sxm_issued[2][2]{};
        for(std::size_t cycle=0;cycle<epoch.program.end_cycle;++cycle){
            TspSliceSystem::LogSinks sinks;sinks.capture_mem_trace=true;
            try{system->tick(sinks);}catch(const std::exception& e){throw std::runtime_error(epoch.name+" cycle="+std::to_string(cycle)+": "+e.what());}
            for(std::size_t h=0;h<2;++h)for(const auto& tr:system->mem_array(static_cast<Hemisphere>(h)).executed_transfers()){
                if(tr.tile)continue;
                trace<<global+cycle<<','<<global+cycle+1<<",MEM."<<(h?"W":"E")<<".S"<<tr.mem_slice<<".T0,"
                    <<(tr.kind==MemArrayModel::MemTransfer::Kind::StoreStreamToSram?"write":"read")<<" address="<<tr.address<<" epoch="<<number<<'\n';
            }
            for(auto stage:{0u,1u,8u,9u})if(system->vxm_unit().superlane(0).lane(0).last_trace()[stage].state==VxmLaneAluTraceState::Executed)
                trace<<global+cycle<<','<<global+cycle+1<<",VXM.C"<<stage<<",execute epoch="<<number<<'\n';
            for(std::size_t h=0;h<2;++h)for(std::size_t p=0;p<2;++p){
                const auto hemi=static_cast<Hemisphere>(h);
                const auto count=p?system->icu().sxm_permute_iq(hemi).issued_count():system->icu().sxm_transpose_iq(hemi).issued_count();
                if(count!=sxm_issued[h][p]){
                    sxm_issued[h][p]=count;
                    trace<<global+cycle<<','<<global+cycle+1<<",SXM."<<(h?"W":"E")
                        <<(p?".Permute.ICU":".Transpose.ICU")<<",issue epoch="<<number<<'\n';
                }
            }
        }
        for(const auto& i:epoch.program.mem)
            if(!system->icu().mem_iq(InstructionControlUnit::mem_queue(static_cast<Hemisphere>(i.side),i.slice)).done())
                throw std::runtime_error("MEM epoch did not retire");
        global+=epoch.program.end_cycle;++number;
        if(number%20==0)std::cout<<number<<"/"<<epochs.size()<<" epochs; "<<epoch.name<<std::endl;
    }
    dump(*system,out/"output.bin",40,tokens*128);dump(*system,out/"state.bin",16,16384);
    std::ofstream summary(out/"execution.json");
    summary<<"{\"backend\":\"native MEM/ICU/SXM/VXM\",\"cycles\":"<<global<<",\"epochs\":"<<number
        <<",\"host_intermediate_transfers\":0,\"host_key_transpose\":false,\"native_constant_initialization\":true,\"matmul_backend\":\"VXM FP32 accumulation\",\"storage\":\"BF16\"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
