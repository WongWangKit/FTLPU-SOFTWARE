#include "ftlpu/compiler/Target/vector32_triangular_schedule.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

using namespace ftlpu;
float round_bf16(float x) {
    return VxmDataFormat::bf16_bits_to_float(VxmDataFormat::float_to_bf16_bits(x));
}
void store(TspSliceSystem& system, std::size_t side, std::size_t pair,
           std::size_t address, std::size_t c, float x) {
    const auto bits=VxmDataFormat::float_to_bf16_bits(x);
    for (std::size_t byte=0; byte<2; ++byte)
        system.initialize_mem_sram_lane_byte(static_cast<Hemisphere>(side),pair+byte,
            c/hw::kLanesPerTile,address,c%hw::kLanesPerTile,
            static_cast<std::uint8_t>(bits>>(byte*8)));
}
int main(int argc, char** argv) try {
    constexpr std::size_t width=128, rows=32, blocks=width/64;
    const std::filesystem::path out=argc>1?argv[1]:"gdn_vector32_solve_system";
    const bool packed=argc>2 && std::string(argv[2])=="--packed-coefficients";
    const std::filesystem::path fixture=argc>3?argv[3]:"";
    std::filesystem::create_directories(out);
    std::ofstream pipeline(out/"runtime.pipeline.csv"), program(out/"compiled_icu_program.csv"),
        dispatch(out/"runtime.icu_dispatch.csv");
    pipeline << "start,end,resource,detail\n";
    program << "queue,cycle,opcode,count,address,stride\n";
    dispatch << "cycle,queue,opcode,address\n";
    auto system=std::make_unique<TspSliceSystem>();
    for (std::size_t side=0; side<2; ++side) {
        system->vxm_unit().configure_input_group_source(side*8,static_cast<Hemisphere>(side));
        system->vxm_unit().configure_input_group_source(side*8+1,static_cast<Hemisphere>(side));
        system->vxm_unit().configure_output_block_destination(side*4,static_cast<Hemisphere>(side));
    }
    std::vector<float> lower(rows*rows), rhs(rows*width), expected(rows*width);
    if (!fixture.empty()) {
        auto load=[&](const char* name, std::vector<float>& values) {
            std::ifstream file(fixture/name,std::ios::binary|std::ios::ate);
            if (!file || file.tellg()!=static_cast<std::streamoff>(values.size()*sizeof(float)))
                throw std::runtime_error(std::string("invalid fixture size: ")+name);
            file.seekg(0);
            file.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float));
            for (auto& v:values) {
                if (!std::isfinite(v)) throw std::runtime_error("non-finite solve fixture");
                v=round_bf16(v);
            }
        };
        load("lower.f32.bin",lower);
        load("rhs.f32.bin",rhs);
    }
    std::mt19937 rng(2364);
    std::normal_distribution<float> normal(0,1);
    for (std::size_t i=0; i<rows; ++i) {
        if (fixture.empty())
            for (std::size_t j=0; j<i; ++j) lower[i*rows+j]=round_bf16(normal(rng)*.08f);
        for (std::size_t c=0; c<width; ++c) {
            if (fixture.empty()) rhs[i*width+c]=round_bf16(normal(rng)*.2f);
            float sum=0;
            for (std::size_t j=0; j<i; ++j) sum += -lower[i*rows+j]*expected[j*width+c];
            expected[i*width+c]=round_bf16(sum+rhs[i*width+c]);
            store(*system,(c/32)%2,2,i*blocks+c/64,c%32,rhs[i*width+c]);
        }
        if (packed) {
            for (std::size_t side=0; side<2; ++side)
                for (std::size_t j=0; j<32; ++j)
                    store(*system,side,8,i,j,-lower[i*rows+j]);
        } else for (std::size_t j=0; j<i; ++j)
            for (std::size_t side=0; side<2; ++side)
                for (std::size_t c=0; c<32; ++c)
                    store(*system,side,4,i*(i-1)/2+j,c,-lower[i*rows+j]);
    }
    for (std::size_t side=0; side<2; ++side)
        for (std::size_t c=0; c<32; ++c) store(*system,side,6,0,c,1);

    const auto plan=compiler::schedule_vector32_triangular_solve(width,packed);
    std::array<std::size_t,InstructionControlUnit::kMemQueues> mem_cursor{};
    std::array<std::size_t,8> vxm_cursor{};
    for (const auto& issue : plan.mem) {
        const auto q=InstructionControlUnit::mem_queue(static_cast<Hemisphere>(issue.side),issue.slice);
        if (issue.cycle<mem_cursor[q]) throw std::logic_error("overlapping MEM domains");
        auto instruction=issue.instruction;
        instruction.loop.wait_cycle=issue.cycle-mem_cursor[q];
        system->icu().enqueue_mem_3d(q,instruction);
        mem_cursor[q]=issue.cycle+(instruction.loop.counts[0]-1)*instruction.loop.cycle_strides[0]+1;
        program << "MEM." << issue.side << '.' << issue.slice << ',' << issue.cycle << ','
            << (instruction.opcode==MemIcuOpcode::Read3D?"READ3D":"WRITE3D") << ','
            << instruction.loop.counts[0] << ',' << instruction.address.base_address << ','
            << instruction.address.inner_stride << '\n';
    }
    std::size_t sxm_cursor[2][2]{};
    for (const auto& issue : plan.sxm) {
        const bool permute=issue.instruction.opcode==SxmOpcode::Permute;
        auto run=SxmIcuRun2DInstruction::Run2D(0,{issue.count,1},{issue.interval,1},issue.instruction);
        run.loop.wait_cycle=issue.cycle-sxm_cursor[issue.side][permute];
        if (permute) system->icu().sxm_permute_iq(static_cast<Hemisphere>(issue.side)).push_sxm_run_2d(run);
        else system->icu().sxm_transpose_iq(static_cast<Hemisphere>(issue.side)).push_sxm_run_2d(run);
        sxm_cursor[issue.side][permute]=issue.cycle+(issue.count-1)*issue.interval+1;
        program << "SXM." << issue.side << '.' << (permute?"Permute":"Transpose")
            << ',' << issue.cycle << ",RUN2D," << issue.count << ",0,0\n";
    }
    for (const auto& issue : plan.vxm) {
        auto run=VxmIcuRun2DInstruction::Run2D(0,{1,1},{1,1},issue.instruction);
        run.loop.wait_cycle=issue.cycle-vxm_cursor[issue.queue];
        system->icu().enqueue_vxm_run_2d(issue.queue,run);
        vxm_cursor[issue.queue]=issue.cycle+1;
        program << "VXM." << issue.queue << ',' << issue.cycle << ",RUN2D,1,0,0\n";
    }
    std::ofstream debug(out/"execution.log");
    std::array<std::size_t,InstructionControlUnit::kMemQueues> observed_mem{};
    std::array<std::size_t,2> observed_vxm{};
    std::size_t observed_sxm[2][2]{};
    std::size_t multiply=0, add=0, mem_reads=0, mem_writes=0;
    for (std::size_t cycle=0; cycle<plan.end_cycle; ++cycle) {
        TspSliceSystem::LogSinks sinks;
        sinks.capture_mem_trace=true;
        if (cycle<80) { sinks.vxm=&debug; sinks.mem=&debug; sinks.icu=&debug; sinks.vxm_log_tile=0; sinks.mem_log_tile=0; }
        try { system->tick(sinks); }
        catch (const std::exception& e) {
            throw std::runtime_error("cycle="+std::to_string(cycle)+": "+e.what());
        }
        for (std::size_t side=0; side<2; ++side) {
            const auto hemi=static_cast<Hemisphere>(side);
            for (const auto& transfer : system->mem_array(hemi).executed_transfers()) {
                const bool write=transfer.kind==MemArrayModel::MemTransfer::Kind::StoreStreamToSram;
                (write?mem_writes:mem_reads)++;
                pipeline << cycle << ',' << cycle+1 << ",MEM." << (side?"W":"E")
                    << ".S" << transfer.mem_slice << ".B" << transfer.bank << ".T" << transfer.tile
                    << ',' << (write?"write":"read") << " address=" << transfer.address << '\n';
            }
            for (std::size_t slice=0; slice<10; ++slice) {
                const auto queue=InstructionControlUnit::mem_queue(hemi,slice);
                const auto count=system->icu().mem_iq(queue).issued_count();
                if (count==observed_mem[queue]) continue;
                observed_mem[queue]=count;
                if (const auto& inst=system->icu().mem_iq(queue).last_dispatched(); inst)
                    dispatch << cycle << ",MEM." << side << '.' << slice << ','
                        << (inst->opcode==MemOpcode::Read?"read":"write") << ',' << inst->address << '\n';
            }
            for (std::size_t p=0; p<2; ++p) {
                const auto issued=p?system->icu().sxm_permute_iq(hemi).issued_count():system->icu().sxm_transpose_iq(hemi).issued_count();
                if (issued!=observed_sxm[side][p]) {
                    observed_sxm[side][p]=issued;
                    pipeline << cycle << ',' << cycle+1 << ",SXM." << (side?"W":"E")
                        << (p?".Permute.ICU":".Transpose.ICU") << ",issue\n";
                    dispatch << cycle << ",SXM." << side << ',' << (p?"permute":"transpose") << ",0\n";
                }
            }
        }
        for (std::size_t queue=0; queue<2; ++queue) {
            const auto count=system->icu().vxm_iq(queue).issued_count();
            if (count==observed_vxm[queue]) continue;
            observed_vxm[queue]=count;
            if (const auto& inst=system->icu().vxm_iq(queue).last_dispatched(); inst)
                dispatch << cycle << ",VXM." << queue << ','
                    << VxmLane::operation_name(VxmCompactInstructionCodec::decode(queue,*inst).instruction.operation)
                    << ",0\n";
        }
        for (auto stage : {0u,1u,8u,9u}) {
            const auto& trace=system->vxm_unit().superlane(0).lane(0).last_trace()[stage];
            if (trace.state==VxmLaneAluTraceState::Executed) {
                (stage%8==0?multiply:add)++;
                pipeline << cycle << ',' << cycle+1 << ",VXM.C" << stage << ','
                    << (stage%8==0?"multiply":"add_acc") << '\n';
            }
        }
    }
    std::size_t mismatches=0;
    std::ofstream actual(out/"actual.f32.bin",std::ios::binary);
    for (std::size_t i=0; i<rows; ++i)
        for (std::size_t c=0; c<width; ++c) {
            std::uint16_t bits=0;
            for (std::size_t b=0; b<2; ++b)
                bits |= std::uint16_t(system->read_mem_sram_lane_byte(
                    static_cast<Hemisphere>((c/32)%2),b,(c%32)/hw::kLanesPerTile,
                    i*blocks+c/64,c%hw::kLanesPerTile))<<(8*b);
            const float got=VxmDataFormat::bf16_bits_to_float(bits);
            actual.write(reinterpret_cast<const char*>(&got),sizeof(got));
            if (got!=expected[i*width+c]) {
                if (mismatches<8) std::cerr << "row="<<i<<" c="<<c<<" got="<<got<<" expected="<<expected[i*width+c]<<'\n';
                ++mismatches;
            }
        }
    for (const auto& issue : plan.mem)
        if (!system->icu().mem_iq(InstructionControlUnit::mem_queue(
                static_cast<Hemisphere>(issue.side),issue.slice)).done())
            throw std::runtime_error("MEM program has not retired");
    if (!system->vxm_unit().idle()) throw std::runtime_error("VXM has not drained");
    if (mem_writes!=rows*width/32*hw::kTileRows*2+(packed?496*2*hw::kTileRows*2:0))
        throw std::runtime_error("unexpected partial-result SRAM writes");
    std::ofstream summary(out/"summary.json");
    summary << "{\"status\":\"" << (mismatches?"failed":"passed") << "\",\"mismatches\":" << mismatches
        << ",\"cycles\":" << plan.end_cycle
        << ",\"mem_tile_byte_reads\":" << mem_reads << ",\"mem_tile_byte_writes\":" << mem_writes
        << ",\"host_intermediate_transfers\":0,\"result_vector_writes\":128"
        << ",\"native_coefficient_expansion\":" << (packed?"true":"false")
        << ",\"backend\":\"resident MEM/ICU/SXM/VXM; no C2C\"}\n";
    if (mismatches) throw std::runtime_error("SRAM solve mismatches="+std::to_string(mismatches));
    std::cout << "MEM/ICU/VXM chunk32 solve passed; cycles="<<plan.end_cycle<<"\n";
    return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
