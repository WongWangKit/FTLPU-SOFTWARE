#include "ftlpu/software/runtime/imem_capacity.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

std::pair<std::uint32_t, std::uint32_t>
geometry_for(QueueKind kind, const ExecutableHardwareConfig& hardware)
{
    switch (kind) {
    case QueueKind::Mem:
        return {hardware.icu_mem_instruction_bits, hardware.icu_mem_imem_depth};
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return {hardware.icu_mxm_instruction_bits, hardware.icu_mxm_imem_depth};
    case QueueKind::Vxm:
        return {hardware.icu_vxm_instruction_bits, hardware.icu_vxm_imem_depth};
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return {hardware.icu_sxm_instruction_bits,
            hardware.icu_sxm_imem_depth};
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return {static_cast<std::uint32_t>(hw::kIcuC2cInstructionBits),
            static_cast<std::uint32_t>(hw::kIcuC2cImemDepth)};
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

void checked_add(std::uint64_t& destination, std::uint64_t value)
{
    if (value > std::numeric_limits<std::uint64_t>::max() - destination)
        throw std::overflow_error("i-MEM capacity statistic overflow");
    destination += value;
}

std::uint64_t synchronized_mem_points(const QueueCommand& header)
{
    if (!is_mem_synchronized_raw_packet_header(header))
        throw std::logic_error(
            "queue command is not a synchronized MEM header");
    return static_cast<std::uint64_t>((header.words[0] >> 2) & 0xffffU)
        + 1;
}

using ContextInterval = std::pair<std::uint64_t, std::uint64_t>;

std::size_t peak_contexts(std::vector<ContextInterval> intervals)
{
    std::sort(intervals.begin(), intervals.end());
    std::vector<std::uint64_t> activeEnds;
    std::size_t peak = 0;
    for (const auto [start, end] : intervals) {
        activeEnds.erase(std::remove_if(activeEnds.begin(), activeEnds.end(),
            [start](std::uint64_t activeEnd) { return activeEnd < start; }),
            activeEnds.end());
        activeEnds.push_back(end);
        peak = std::max(peak, activeEnds.size());
    }
    return peak;
}

ContextInterval macro_interval(const IcuMacroSchedule& schedule)
{
    const auto end = static_cast<std::uint64_t>(schedule.start_cycle)
        + static_cast<std::uint64_t>(schedule.inner_count - 1)
            * schedule.inner_interval
        + static_cast<std::uint64_t>(schedule.outer_count - 1)
            * schedule.outer_interval;
    return {schedule.start_cycle, end};
}

std::size_t peak_macro_contexts(const QueueProgram& queue)
{
    std::vector<ContextInterval> intervals;
    for (const auto& command : queue.commands)
        if (is_macro_schedule_command(command))
            intervals.push_back(macro_interval(decode_macro_schedule_command(command)));
    return peak_contexts(std::move(intervals));
}

bool is_all_macro_mem(const QueueProgram& queue)
{
    return queue.kind == QueueKind::Mem && !queue.commands.empty() &&
           std::all_of(queue.commands.begin(), queue.commands.end(),
                       [](const QueueCommand& command) {
                           return is_macro_schedule_command(command) &&
                                  command.instruction_kind ==
                                      InstructionKind::Mem;
                       });
}

bool is_all_macro_mxm(const QueueProgram& queue)
{
    const bool mxmKind = queue.kind == QueueKind::MxmLoad ||
                         queue.kind == QueueKind::MxmCompute ||
                         queue.kind == QueueKind::MxmDequant || queue.kind == QueueKind::SxmTranspose || queue.kind == QueueKind::SxmPermute;
    return mxmKind && !queue.commands.empty() &&
           std::all_of(queue.commands.begin(), queue.commands.end(),
                       [&](const QueueCommand& command) {
                           const auto expected =
                               queue.kind == QueueKind::MxmDequant ? InstructionKind::MxmDequant
                                   : (queue.kind == QueueKind::SxmTranspose || queue.kind == QueueKind::SxmPermute)
                                       ? InstructionKind::Sxm : InstructionKind::Mxm;
                           return is_macro_schedule_command(command) &&
                                  command.instruction_kind == expected;
                       });
}

std::size_t macro_context_capacity_for(QueueKind kind,
                                       const ExecutableHardwareConfig& hardware)
{
    switch (kind) {
    case QueueKind::Mem:
        return hardware.icu_mem_macro_contexts;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return hardware.icu_mxm_macro_contexts;
    case QueueKind::Vxm:
        return 0;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: return hw::kIcuSxmMacroContextDepth;
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return 0;
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

std::uint32_t macro_context_bits_for(QueueKind kind,
                                     const ExecutableHardwareConfig& hardware)
{
    switch (kind) {
    case QueueKind::Mem:
        return hardware.icu_mem_macro_context_bits;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant:
        return hardware.icu_mxm_macro_context_bits;
    case QueueKind::Vxm:
        return 0;
    case QueueKind::SxmTranspose: return IcuSxmActiveMacroContextCodec<true>::bit_count;
    case QueueKind::SxmPermute: return IcuSxmActiveMacroContextCodec<false>::bit_count;
    case QueueKind::C2cDma:
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return 0;
    }
    throw std::invalid_argument("unknown ICU queue kind");
}

} // namespace

CmodelAbstractImemReport
analyze_cmodel_abstract_imem(const BinaryProgram& program)
{
    CmodelAbstractImemReport report;
    for (const auto& queue : program.queues) {
        CmodelAbstractImemQueue capacity;
        capacity.kind=queue.kind; capacity.index=queue.index;
        const auto [bits,depth]=geometry_for(queue.kind,program.hardware);
        capacity.slot_bits=bits; capacity.depth=depth;
        capacity.used_slots=queue.commands.size();
        for(std::size_t pc=0;pc<queue.commands.size();++pc) {
            const auto& command=queue.commands[pc];
            if(is_macro_schedule_command(command)) {
                const auto m=decode_macro_schedule_command(command);
                ++capacity.macro_entries;
                checked_add(capacity.expanded_work,std::uint64_t(m.inner_count)*m.outer_count);
            } else if(is_mem_synchronized_raw_packet_header(command)) {
                ++capacity.coarse_program_entries;
                checked_add(capacity.expanded_work,synchronized_mem_points(command));
                pc+=InstructionControlUnit::MemIcu::synchronized_packet_word_count-1;
            } else {
                const auto op=isa::decode_icu_command_opcode(command.command);
                if(op==isa::IcuCommandOpcode::Nop) ++capacity.nop_entries;
                else if(op==isa::IcuCommandOpcode::Instruction) {
                    ++capacity.instruction_entries; ++capacity.expanded_work;
                } else if(!is_icu_control_raw_word_command(command))
                    throw std::invalid_argument("unsupported instruction encoding in iMEM accounting");
            }
        }
        report.used_slots+=capacity.used_slots;
        report.encoded_work_entries+=capacity.instruction_entries+capacity.macro_entries+capacity.coarse_program_entries;
        checked_add(report.expanded_work,capacity.expanded_work);
        checked_add(report.used_bits,capacity.used_bits());
        checked_add(report.active_queue_capacity_bits,std::uint64_t(bits)*depth);
        if(capacity.overflow())++report.overflow_queues;
        report.queues.push_back(capacity);
    }
    return report;
}

PhysicalImemReport analyze_physical_imem(const BinaryProgram& program)
{
    const auto abstract = analyze_cmodel_abstract_imem(program);
    PhysicalImemReport report;
    report.queues.reserve(program.queues.size());
    for (std::size_t i = 0; i < program.queues.size(); ++i) {
        const auto& queue = program.queues[i];
        PhysicalImemQueue physical;
        static_cast<CmodelAbstractImemQueue&>(physical) = abstract.queues[i];
        physical.peak_macro_contexts = peak_macro_contexts(queue);
        const bool rawMem = is_all_macro_mem(queue)
            && physical.slot_bits == kMemIcuImemWordBits
            && program.hardware.icu_macro_encoding_version == kMemMacroBitstreamVersion;
        const bool rawMxm = is_all_macro_mxm(queue)
            && physical.slot_bits == ((queue.kind==QueueKind::SxmTranspose || queue.kind==QueueKind::SxmPermute)?96:kMxmIcuImemWordBits)
            && program.hardware.icu_macro_encoding_version == kMxmMacroBitstreamVersion;

        physical.macro_context_capacity = macro_context_capacity_for(
            queue.kind, program.hardware);
        physical.macro_context_bits = macro_context_bits_for(
            queue.kind, program.hardware);
        if (physical.slot_bits == 0)
            throw std::invalid_argument("target i-MEM slot width must be non-zero");
        if (rawMem) {
            if (program.hardware.icu_macro_encoding_version !=
                kMemMacroBitstreamVersion)
                throw std::invalid_argument(
                    "target does not support MEM Macro bitstream v1");
            const auto image = encode_mem_macro_bitstream(queue);
            const auto packed = pack_mem_macro_imem(image);
            physical.mem_delta_rle = true;
            physical.macro_codec = image.stats;
            physical.payload_bits = image.stats.physical_bits();
            physical.physical_slots = packed.word_count();
            ++report.mem_delta_rle_queues;
        } else if (rawMxm) {
            if (program.hardware.icu_macro_encoding_version !=
                kMxmMacroBitstreamVersion)
                throw std::invalid_argument(
                    "target does not support MXM Macro bitstream v1");
            const auto image = encode_mxm_macro_bitstream(queue);
            const auto packed = pack_mxm_macro_imem(image);
            physical.mxm_delta_rle = true;
            physical.macro_codec = image.stats;
            physical.payload_bits = image.stats.physical_bits();
            physical.physical_slots = packed.word_count();
            ++report.mxm_delta_rle_queues;
        } else {
            if (std::ranges::any_of(queue.commands,is_macro_schedule_command))
                throw std::invalid_argument("Macro queue must use the packed physical encoding");
            physical.physical_slots=queue.commands.size();
            physical.physical_bits=std::uint64_t(physical.physical_slots)*physical.slot_bits;

        }
        if (physical.mem_delta_rle || physical.mxm_delta_rle)
            physical.physical_bits = static_cast<std::uint64_t>(physical.physical_slots) * physical.slot_bits;
        checked_add(report.used_bits, physical.physical_bits);
        report.used_slots += physical.physical_slots;
        checked_add(report.peak_macro_context_bits,
            static_cast<std::uint64_t>(physical.peak_macro_contexts)
                * physical.macro_context_bits);
        checked_add(report.provisioned_macro_context_bits,
            static_cast<std::uint64_t>(physical.macro_context_capacity)
                * physical.macro_context_bits);
        checked_add(report.active_queue_capacity_bits,
                    static_cast<std::uint64_t>(physical.depth) *
                        physical.slot_bits);
        if (physical.macro_context_overflow())
            ++report.macro_context_overflow_queues;
        if (physical.deployment_overflow()) ++report.overflow_queues;
        report.queues.push_back(std::move(physical));
    }
    return report;
}

std::vector<BinaryProgram> split_resident_program_at_drains(const BinaryProgram& program)
{
    if(analyze_physical_imem(program).fits())return {program};
    if(program.common_barrier_protocol || !program.transport_queues.empty() ||
       !program.weight_page_uses.empty() || !program.barriers.empty() ||
       !program.scale_relocations.empty())
        throw std::invalid_argument("resident iMEM segmentation cannot bypass paging/barrier/scale contracts");
    std::vector<std::size_t> ends;
    for(const auto& timeline:program.timelines)
        if(timeline.name.find("imem.drain.")!=std::string::npos && timeline.end_cycle>0 && timeline.end_cycle<program.max_cycle)
            ends.push_back(timeline.end_cycle);
    // max_cycle names the last issue cycle; intervals here are half-open.
    ends.push_back(program.max_cycle+1);
    std::sort(ends.begin(),ends.end());ends.erase(std::unique(ends.begin(),ends.end()),ends.end());
    if(ends.size()==1)throw std::invalid_argument("iMEM overflow has no compiler-declared drain boundaries");
    auto metadata=program;
    metadata.queues.clear();metadata.timelines.clear();metadata.address_relocations.clear();
    std::vector<BinaryProgram> result(ends.size(),metadata);
    for(std::size_t n=0;n<ends.size();++n){
        result[n].max_cycle=ends[n]-(n?ends[n-1]:0);
        result[n].queues.clear();result[n].timelines.clear();result[n].address_relocations.clear();
    }
    for(const auto& queue:program.queues){
        std::vector<QueueProgram> parts(ends.size());std::vector<std::size_t> local(ends.size());
        for(auto& part:parts){part.kind=queue.kind;part.index=queue.index;}
        std::size_t cursor=0;
        for(std::size_t pc=0;pc<queue.commands.size();){
            const auto& command=queue.commands[pc];
            if(isa::decode_icu_command_opcode(command.command)==isa::IcuCommandOpcode::Nop){
                cursor+=is_icu_control_raw_word_command(command)?decode_icu_control_raw_word(command).count:isa::decode_icu_nop_cycles(command.command);
                ++pc;continue;
            }
            if(queue.kind==QueueKind::Vxm && command.command==0
                && command.word_count==3 && command.extension_words.empty()) {
                const auto config=isa::decode_vxm_instruction(queue.index,{
                    std::uint64_t(command.words[0])|(std::uint64_t(command.words[1])<<32),command.words[2]});
                const auto part=std::upper_bound(ends.begin(),ends.end(),cursor)-ends.begin();
                if(part>=ends.size() || config.instruction.repeat_count>ends[part]-cursor)
                    throw std::invalid_argument("VXM FU run crosses compiler-declared iMEM drain");
                const auto begin=part?ends[part-1]:0;
                if(cursor<begin+local[part])throw std::invalid_argument("overlapping sequential VXM configurations");
                const auto wait=cursor-begin-local[part];
                if(wait)parts[part].commands.push_back({isa::encode_icu_nop(wait)});
                parts[part].commands.push_back(command);
                local[part]=cursor+1-begin;++cursor;++pc;continue;
            }
            if(!is_macro_schedule_command(command))
                throw std::invalid_argument("resident iMEM segmentation requires Macro or VXM CONFIG");
            const auto macro=decode_macro_schedule_command(command);
            const auto [first,last]=macro_interval(macro);
            const auto end=last+1;
            const auto part=std::upper_bound(ends.begin(),ends.end(),first)-ends.begin();
            if(part>=ends.size() || end>ends[part])
                throw std::invalid_argument("Macro crosses compiler-declared iMEM drain");
            const auto begin=part?ends[part-1]:0;
            if(first<begin+local[part])throw std::invalid_argument("interleaved Macro at iMEM drain");
            auto relocated=command;
            relocated.extension_words[1]=static_cast<std::uint32_t>(first-begin);
            parts[part].commands.push_back(std::move(relocated));
            parts[part].requires_raw_macro=true;
            local[part]=end-begin;cursor=end;++pc;
        }
        for(std::size_t n=0;n<parts.size();++n)if(!parts[n].commands.empty())result[n].queues.push_back(std::move(parts[n]));
    }
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto report = analyze_physical_imem(result[index]);
        if (report.fits()) continue;
        const auto begin = index ? ends[index - 1] : 0;
        std::string detail = "compiler-declared iMEM segment still exceeds physical capacity: segment="
            + std::to_string(index) + " begin=" + std::to_string(begin)
            + " end_exclusive=" + std::to_string(ends[index]);
        for (const auto& queue : report.queues) {
            if (!queue.deployment_overflow()) continue;
            detail += " queue=" + std::string(queue_kind_name(queue.kind))
                + "[" + std::to_string(queue.index) + "]"
                + " physical_slots=" + std::to_string(queue.physical_slots)
                + " depth=" + std::to_string(queue.depth)
                + " peak_macro_contexts=" + std::to_string(queue.peak_macro_contexts)
                + " macro_context_capacity=" + std::to_string(queue.macro_context_capacity);
        }
        throw std::invalid_argument(detail);
    }
    return result;
}

std::vector<BinaryProgram> plan_compute_imem_segments(const BinaryProgram& program)
{
    const auto report = analyze_physical_imem(program);
    if (report.fits()) return {};
    if (!program.common_barrier_protocol)
        throw std::invalid_argument("compute iMEM reload requires the common barrier protocol");
    BinaryProgram subset;
    subset.target_name = program.target_name;
    subset.target_abi = program.target_abi;
    subset.hardware = program.hardware;
    subset.max_cycle = program.max_cycle;
    subset.timelines = program.timelines;
    for (const auto& queue : report.queues) {
        if (!queue.deployment_overflow()) continue;
        if (!queue.physical_overflow()
            || queue.macro_context_overflow()
            || (queue.kind != QueueKind::Vxm && queue.kind != QueueKind::SxmTranspose
                && queue.kind != QueueKind::SxmPermute))
            throw std::invalid_argument("iMEM reload cannot split MEM/MXM or bypass Active Context capacity");
        for (const auto& source : program.queues)
            if (source.kind == queue.kind && source.index == queue.index)
                subset.queues.push_back(source);
    }
    return split_resident_program_at_drains(subset);
}

} // namespace ftlpu::software::runtime
