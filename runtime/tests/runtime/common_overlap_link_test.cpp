#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"
#include <iostream>
#include <memory>
#include <ranges>

// Preflight only. Executes the production final linker and image loader with
// layout-sized zero payloads, without claiming numeric or transport execution.
int main(int argc,char** argv) try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    if (argc!=2) throw std::invalid_argument("expected Raw paging container");
    auto program=read_binary_program(std::filesystem::path(argv[1]));
    auto system=std::make_unique<C2cDmaSystem>();
    auto plans=plan_weight_prefetches(program);
    schedule_weight_prefetches(program,plans);
    C2cWeightPager pager(*system);
    pager.begin_schedule(program);
    std::size_t index=0;
    for (const auto& plan:plans) {
        if (plan.pre_execution) continue;
        const auto use=program.weight_page_uses.at(plan.use_indices.front());
        const auto binding=std::ranges::find(program.bindings,use.binding_index,&BinaryBinding::index);
        if (binding==program.bindings.end()) throw std::invalid_argument("missing binding");
        const std::vector<std::uint8_t> zeros(binding->byte_size);
        const auto packed=pack_weight_binding_page(*binding,use.page_index,zeros,program.hardware);
        C2cWeightPage page;
        page.bank=plan.bank;
        page.fabric_streams=plan.fabric_streams;
        for(const auto& s:packed.segments)
            page.segments.push_back({static_cast<Hemisphere>(s.hemisphere),s.slice,plan.bank,
                s.base_row,0,0x1000000ULL*(index+1)+s.byte_offset,s.vector_count});
        const auto fence=pager.schedule(program,page,plan.start_cycle,plan.transfer_end_cycle,
            plan.ready_cycle,0x10000+index,0x30000+index,plan.earliest_mem_cycle,plan.use_indices);
        std::cout<<"page="<<index<<" E="<<fence.scheduled_start_cycle<<" ready="<<plan.ready_cycle<<'\n';
        ++index;
    }
    pager.finalize_schedule(program);
    CModelRuntime runtime(*system,[&](TspSliceSystem::LogSinks sinks){system->tick(sinks);});
    runtime.load(program);
    for (const auto& barrier:program.barriers) std::cout<<"C="<<barrier.stop_cycle<<'\n';
    std::cout<<"final link + loader preflight passed; execution not tested\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
