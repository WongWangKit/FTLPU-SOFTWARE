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
#include "ftlpu/software/runtime/schedule_trace.hpp"
#include <filesystem>
#include <fstream>
template <typename T>
void write_fixture_scalar(std::ostream& os, T value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void verify_official_v26_header_compatibility()
{
    constexpr char magic[8] {'F', 'T', 'L', 'P', 'U', 'B', '0', '1'};
    ExecutableHardwareConfig hardware;
    hardware.vxm_cross_hemisphere_streams_enabled = 0;
    hardware.vxm_reserved_feature = 0;
    const std::string targetName = "official-v26-fixture";

    std::ostringstream os(std::ios::out | std::ios::binary);
    os.write(magic, sizeof(magic));
    write_fixture_scalar<std::uint32_t>(os, 26);
    write_fixture_scalar<std::uint64_t>(os, executable_target_abi(hardware));
    write_fixture_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(targetName.size()));
    os.write(targetName.data(), static_cast<std::streamsize>(targetName.size()));
    hardware.visit_v26([&](std::uint32_t value) {
        write_fixture_scalar(os, value);
    });
    write_fixture_scalar<std::uint64_t>(os, 0);
    for (int count = 0; count < 7; ++count)
        write_fixture_scalar<std::uint32_t>(os, 0);

    const auto bytes = os.str();
    const auto decoded = read_binary_program(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
    if (decoded.target_name != targetName || !decoded.queues.empty()
        || decoded.hardware.vxm_cross_hemisphere_streams_enabled != 0
        || decoded.hardware.vxm_reserved_feature != 0)
        throw std::runtime_error("official v26 header did not round-trip");
}

int main()try{
    verify_official_v26_header_compatibility();
    BinaryProgram p;p.max_cycle=100;
    p.queues.push_back({QueueKind::Mem,0,{mem(MemInstruction::Read(100,2),
        {10,4,2,1,2,20,16,IcuInductionTarget::MemAddress})}});
    BinaryBinding weight;weight.index=1;weight.name="gate";weight.role="weight";weight.access=BindingAccess::Input;
    weight.paged_weight=true;weight.page_count=2;weight.page_banks={0,1};weight.page_base_rows={0,16};
    weight.page_row_counts={8,8};weight.page_slice_group_bases={0,0};weight.page_slice_group_counts={1,1};
    p.bindings.push_back(weight);p.weight_page_uses.push_back({1,0,0,10,37});
    p.timelines.push_back({"gate",10,37});
    std::stringstream binary;write_binary_program(p,binary);auto decoded=read_binary_program(binary);
    require(decoded.bindings[0].page_banks==weight.page_banks
        && decoded.bindings[0].page_base_rows==weight.page_base_rows,"page placement did not round trip");
    require(compare_logical_issues(p,decoded).equivalent,"binary changed Macro issue domain");
    auto path=std::filesystem::temp_directory_path()/"ftlpu-macro-trace-test.csv";
    write_schedule_trace_csv(decoded,path);
    std::ifstream input(path);std::string csv((std::istreambuf_iterator<char>(input)),{});
    require(csv.find("MEM.E.Read")!=std::string::npos && csv.find("repeat2d")!=std::string::npos,
        "pipeline omitted the Macro MEM read domain");
    input.close();
    std::filesystem::remove(path);
    require(analyze_cmodel_abstract_imem(decoded).expanded_work==8,"capacity did not count Macro issues");
    std::cout<<"schedule_trace_weight_page_test passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
