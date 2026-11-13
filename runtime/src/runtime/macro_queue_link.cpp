#include "ftlpu/software/runtime/macro_queue_link.hpp"


#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ftlpu::software::runtime {
namespace {
std::size_t checked_add(std::size_t a,std::size_t b) {
    if(b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("queue cycle overflow");
    return a+b;
}
}

MacroQueueLinkResult link_macro_queue_controls(
    const QueueProgram& queue, std::size_t max_descriptors, bool externalize_controls)
{
    MacroQueueLinkResult result;
    result.queue.kind = queue.kind;
    result.queue.index = queue.index;
    result.controls.kind = queue.kind;
    result.controls.index = static_cast<std::uint16_t>(queue.index);
    result.command_indices.resize(queue.commands.size());
    result.transport_command_indices.resize(queue.commands.size());
    result.transport.kind=queue.kind;result.transport.index=queue.index;
    std::uint32_t transport_ordinal=0;
    std::vector<std::pair<std::size_t,QueueCommand>> output;
    std::size_t cursor = 0;
    for (std::size_t pc = 0; pc < queue.commands.size();) {
        const auto& command = queue.commands[pc];
        if (is_mem_synchronized_raw_packet_header(command)) {
            if (!externalize_controls)
                throw std::invalid_argument("synchronized MEM transfers require the native ICU path");
            if (cursor>std::numeric_limits<std::uint32_t>::max() || pc>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("queue transport boundary exceeds uint32");
            const auto packet=decode_mem_synchronized_icu_packet(queue,pc);
            const auto words=encode_mem_synchronized_icu_packet(packet);
            result.transport_command_indices[pc]=result.transport.commands.size();
            result.transport.commands.insert(result.transport.commands.end(),words.begin(),words.end());
            result.controls.entries.push_back({static_cast<std::uint32_t>(cursor),kQueueControlMemPacket,
                false,++transport_ordinal,static_cast<std::uint32_t>(pc),0,0});
            pc+=words.size();
            continue;
        }
        if (is_icu_control_raw_word_command(command)) {
            const auto control = decode_icu_control_raw_word(command);
            if (control.opcode == IcuControlOpcode::Nop) {
                cursor = checked_add(cursor, control.count);
            } else {
                if (!externalize_controls || (control.opcode != IcuControlOpcode::Sync
                    && control.opcode != IcuControlOpcode::WaitEvent && control.opcode != IcuControlOpcode::Notify))
                    throw std::invalid_argument("event/SYNC control requires explicit queue-local synchronization mapping");
                if (cursor > std::numeric_limits<std::uint32_t>::max()
                    || pc > std::numeric_limits<std::uint32_t>::max()
                    || control.event_tag > std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("queue control position or operand exceeds uint32");
                result.controls.entries.push_back({static_cast<std::uint32_t>(cursor), control.opcode,
                    false, static_cast<std::uint32_t>(control.event_tag), static_cast<std::uint32_t>(pc), 0, 0});
            }
            ++pc;
            continue;
        }
        const auto opcode = isa::decode_icu_command_opcode(command.command);
        if (opcode == isa::IcuCommandOpcode::Nop) {
            if (command.instruction_kind != InstructionKind::None
                || command.word_count != 0 || !command.extension_words.empty())
                throw std::invalid_argument("malformed Macro source NOP");
            cursor = checked_add(cursor, isa::decode_icu_nop_cycles(command.command));
            ++pc;
            continue;
        }
        if(is_macro_schedule_command(command)) {
            const auto schedule=decode_macro_schedule_command(command);
            if(schedule.start_cycle<cursor) throw std::invalid_argument("Macro starts before its queue boundary");
            output.push_back({pc,command});
            cursor=schedule.start_cycle+(schedule.inner_count-1)*schedule.inner_interval
                +(schedule.outer_count-1)*schedule.outer_interval+1;
            ++pc;continue;
        }
        throw std::invalid_argument("Macro queue recipe contains a retired loop instruction or native FU template");
    }
    if(cursor>UINT32_MAX) throw std::overflow_error("queue completion coordinate exceeds uint32");
    result.controls.end_cycle=static_cast<std::uint32_t>(cursor);
    if(output.size()>max_descriptors) throw std::invalid_argument("Macro queue descriptor limit exceeded");
    for(auto& [source,command]:output) {
        result.command_indices[source].push_back(result.queue.commands.size());
        result.queue.commands.push_back(std::move(command));
    }
    return result;
}

namespace {
BinaryQueueControlPartition native_control_boundaries(const QueueProgram& q)
{
    BinaryQueueControlPartition p{q.kind,static_cast<std::uint16_t>(q.index),{}};
    std::size_t cycle=0;
    std::uint32_t transport=0;
    const auto control=[&](std::size_t pc,IcuControlOpcode opcode,std::size_t operand) {
        if (cycle>UINT32_MAX || pc>UINT32_MAX || operand>UINT32_MAX)
            throw std::overflow_error("queue synchronization plan exceeds uint32");
        p.entries.push_back({static_cast<std::uint32_t>(cycle),opcode,true,
            static_cast<std::uint32_t>(operand),static_cast<std::uint32_t>(pc),0,0});
    };
    for(std::size_t pc=0;pc<q.commands.size();) {
        const auto& command=q.commands[pc];
        if(is_mem_synchronized_raw_packet_header(command)) {
            control(pc,kQueueControlMemPacket,++transport);
            pc+=InstructionControlUnit::MemIcu::synchronized_packet_word_count;
        } else if(is_icu_control_raw_word_command(command)) {
            const auto c=decode_icu_control_raw_word(command);
            if(c.opcode==IcuControlOpcode::Nop) cycle=checked_add(cycle,c.count);
            else if(c.opcode==IcuControlOpcode::WaitEvent || c.opcode==IcuControlOpcode::Sync || c.opcode==IcuControlOpcode::Notify)
                control(pc,c.opcode,c.event_tag);
            else throw std::invalid_argument("queue sync planning does not support repeat controls");
            ++pc;
        } else if(is_macro_schedule_command(command)) {
            const auto s=decode_macro_schedule_command(command);
            if(s.start_cycle<cycle) throw std::invalid_argument("Macro overlaps queue control boundary");
            cycle=s.start_cycle+(s.inner_count-1)*s.inner_interval+(s.outer_count-1)*s.outer_interval+1;
            ++pc;
        } else if(isa::decode_icu_command_opcode(command.command)==isa::IcuCommandOpcode::Nop) {
            cycle=checked_add(cycle,isa::decode_icu_nop_cycles(command.command));++pc;
        } else if(command.instruction_kind==InstructionKind::C2cDma) {
            (void)decode_c2c_dma_icu_packet(q,pc);
            cycle=checked_add(cycle,1);pc+=C2cDmaIcuPacket::kWordCount;
        } else if(command.instruction_kind==InstructionKind::C2cEndpoint
            || isa::decode_icu_command_opcode(command.command)==isa::IcuCommandOpcode::Instruction) {
            cycle=checked_add(cycle,1);++pc;
        } else throw std::invalid_argument("unrecognized Native boundary in synchronization plan");
    }
    if(cycle>UINT32_MAX) throw std::overflow_error("queue completion coordinate exceeds uint32");
    p.end_cycle=static_cast<std::uint32_t>(cycle);
    return p;
}
}

BinaryProgram macro_queue_link_input(const BinaryProgram& program)
{
    if(!program.queue_control_protocol || program.queue_link_source.queues.empty())
        throw std::invalid_argument("QCTL relink requires its compact Native boundary recipe");
    auto result=program;
    result.queues=program.queue_link_source.queues;
    result.address_relocations=program.queue_link_source.address_relocations;
    result.scale_relocations=program.queue_link_source.scale_relocations;
    result.transport_queues.clear();result.queue_sync_plan={};result.queue_link_source={};
    result.queue_control_protocol=false;
    return result;
}

void link_queue_controls(BinaryProgram& program, bool encode_macro,
    const std::function<void(QueueSyncPlan&)>& link_releases)
{
    if(program.common_barrier_protocol || program.transport_overlap_protocol || !program.transport_queues.empty())
        throw std::invalid_argument("queue-local conversion requires a Native synchronization plan");
    if(program.hardware.queue_control_version!=1)
        throw std::invalid_argument("target does not support queue-local synchronization");
    auto result=program;
    result.queue_control_protocol=true;
    result.queue_sync_plan.partitions.clear();
    std::map<std::pair<QueueKind,std::size_t>,MacroQueueLinkResult> lowered;
    std::set<std::uint32_t> tags;
    // Establish every boundary in the Native plan before lowering any queue.
    for(const auto& queue:program.queues) {
        if(queue.requires_raw_macro || queue.packed_macro_imem)
            throw std::invalid_argument("queue sync plan must precede Macro encoding");
        auto partition=native_control_boundaries(queue);
        // Paging inserts Native NOP/control boundaries after the compiler's
        // max_cycle was recorded. Derive the final extent from that actual
        // queue, before either encoding, rather than weakening QCTL bounds.
        if(partition.end_cycle)
            result.max_cycle=std::max(result.max_cycle,static_cast<std::uint64_t>(partition.end_cycle)-1);

        for(const auto& c:partition.entries)
            if(c.opcode==IcuControlOpcode::WaitEvent) tags.insert(c.operand);
        if(!partition.entries.empty()) result.queue_sync_plan.partitions.push_back(std::move(partition));
    }
    for(const auto tag:tags)
        if(std::ranges::none_of(result.queue_sync_plan.events,[&](const auto& e){return e.event_id==tag;}))
            result.queue_sync_plan.events.push_back({tag,QueueEventSource::External,{}});
    for(auto& queue:result.queues) {
        if(!encode_macro) continue;
        if(queue.kind!=QueueKind::Mem && queue.kind!=QueueKind::MxmLoad
            && queue.kind!=QueueKind::MxmCompute && queue.kind!=QueueKind::MxmDequant
            && queue.kind!=QueueKind::SxmTranspose && queue.kind!=QueueKind::SxmPermute) continue;
        auto mapped=link_macro_queue_controls(queue,65536,true);
        // A transport-only queue remains genuinely Native, with real fetch.
        if(mapped.queue.commands.empty()) continue;
        auto partition=std::ranges::find_if(result.queue_sync_plan.partitions,[&](const auto& p){return p.kind==queue.kind && p.index==queue.index;});
        if(partition!=result.queue_sync_plan.partitions.end()) {
            if(partition->entries.size()!=mapped.controls.entries.size())
                throw std::logic_error("control lowering diverged from Native synchronization plan");
            for(std::size_t i=0;i<partition->entries.size();++i) {
                auto& a=partition->entries[i];const auto& b=mapped.controls.entries[i];
                if(a.local_cycle!=b.local_cycle || a.opcode!=b.opcode || a.operand!=b.operand || a.source_command!=b.source_command)
                    throw std::logic_error("control position changed during Macro lowering");
                a.native_binding=false;
            }
        }
        if(partition==result.queue_sync_plan.partitions.end())
            result.queue_sync_plan.partitions.push_back(mapped.controls);
        if(!mapped.transport.commands.empty()) result.transport_queues.push_back(mapped.transport);
        mapped.queue.requires_raw_macro=true;
        const auto key=std::pair{queue.kind,queue.index};
        queue=mapped.queue;
        lowered.emplace(key,std::move(mapped));
    }
    result.address_relocations.clear();
    for(const auto& relocation:program.address_relocations) {
        const auto it=lowered.find({relocation.queue_kind,relocation.queue_index});
        if(it==lowered.end()) {result.address_relocations.push_back(relocation);continue;}
        const auto& map=it->second;
        if(relocation.command_index>=map.command_indices.size())
            throw std::invalid_argument("relocation is outside Native source queue");
        if(const auto transport=map.transport_command_indices[relocation.command_index]) {
            auto target=relocation;target.transport_queue=true;target.command_index=static_cast<std::uint32_t>(*transport);
            result.address_relocations.push_back(target);
        } else {
            if(map.command_indices[relocation.command_index].empty())
                throw std::invalid_argument("relocation names a nonfunctional queue boundary");
            for(const auto index:map.command_indices[relocation.command_index]) {
                auto target=relocation;target.command_index=static_cast<std::uint32_t>(index);
                result.address_relocations.push_back(target);
            }
        }
    }
    result.scale_relocations.clear();
    for(const auto& relocation:program.scale_relocations) {
        const auto it=lowered.find({relocation.queue_kind,relocation.queue_index});
        if(it==lowered.end()) {result.scale_relocations.push_back(relocation);continue;}
        if(relocation.command_index>=it->second.command_indices.size() || it->second.command_indices[relocation.command_index].empty())
            throw std::invalid_argument("scale relocation has no functional Macro source");
        for(const auto index:it->second.command_indices[relocation.command_index]) {
            auto target=relocation;target.command_index=static_cast<std::uint32_t>(index);
            result.scale_relocations.push_back(target);
        }
    }
    if(link_releases) link_releases(result.queue_sync_plan);
    validate_binary_barriers(result);validate_queue_sync_plan(result);
    program=std::move(result);
}

} // namespace ftlpu::software::runtime
