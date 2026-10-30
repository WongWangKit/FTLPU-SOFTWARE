#pragma once
#include "ftlpu/software/runtime/binary.hpp"
#include <map>
#include <set>
#include <stdexcept>

namespace ftlpu::software::runtime {
// Prove queue order without inventing a shared Lq. Nodes are concrete page
// launches; edges come only from physical queue order and completion waits.
inline void validate_queue_transport_order(const BinaryProgram& program,
    const std::map<std::pair<std::size_t,std::size_t>,std::uint32_t>& packets)
{
    if(!program.queue_control_protocol) return;
    std::map<std::uint32_t,std::set<std::uint32_t>> predecessors;
    const auto edge=[&](std::uint32_t before,std::uint32_t after) {
        predecessors[before];predecessors[after].insert(before);
    };
    std::map<std::size_t,std::vector<std::pair<std::size_t,std::uint32_t>>> mem;
    for(const auto& [location,launch]:packets) {
        predecessors[launch];
        auto& q=mem[location.first];
        if(!q.empty() && q.back().second!=launch) edge(q.back().second,launch);
        q.emplace_back(location.second,launch);
    }
    std::map<std::uint32_t,std::set<std::uint32_t>> events;
    for(const auto& event:program.queue_sync_plan.events)
        if(event.source==QueueEventSource::PageCommit)
            for(const auto instance:event.page_use_indices)
                events[event.event_id].insert(program.weight_page_uses.at(instance).transport_launch_event_id);
    for(const auto& partition:program.queue_sync_plan.partitions) {
        if(partition.kind==QueueKind::C2cDma || partition.kind==QueueKind::C2cRx) {
            std::uint32_t previous=0;
            for(const auto& control:partition.entries) {
                if(control.opcode!=IcuControlOpcode::WaitEvent || !predecessors.contains(control.operand)) continue;
                if(previous && previous!=control.operand) edge(previous,control.operand);
                previous=control.operand;
            }
        }
        if(partition.kind!=QueueKind::Mem) continue;
        std::set<std::uint32_t> required;
        for(const auto& control:partition.entries) {
            if(control.opcode==IcuControlOpcode::WaitEvent) {
                const auto event=events.find(control.operand);
                if(event!=events.end()) required.insert(event->second.begin(),event->second.end());
            }
            if(control.opcode!=static_cast<IcuControlOpcode>(8)) continue;
            const auto queue=mem.find(partition.index);
            if(queue==mem.end()) throw std::invalid_argument("QCTL MEM control has no linked transport");
            const auto& descriptors=queue->second;
            std::uint32_t launch=0;
            if(control.native_binding) {
                for(const auto& [pc,owner]:descriptors) if(pc==control.source_command) launch=owner;
            } else if(control.operand && control.operand<=descriptors.size()) {
                launch=descriptors[control.operand-1].second;
            }
            if(!launch) throw std::invalid_argument("QCTL transport boundary has no concrete launch");
            for(const auto dependency:required) edge(dependency,launch);
        }
    }
    std::set<std::uint32_t> done;
    while(done.size()!=predecessors.size()) {
        bool progress=false;
        for(const auto& [launch,dependencies]:predecessors) {
            if(done.contains(launch)) continue;
            bool ready=true;
            for(const auto dependency:dependencies) ready &= done.contains(dependency);
            if(ready) {done.insert(launch);progress=true;}
        }
        if(!progress) throw std::invalid_argument("QCTL transport order depends on a future packet or completion wait");
    }
}
} // namespace ftlpu::software::runtime
