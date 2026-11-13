#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"
#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
try {
    if (argc < 2)
        throw std::runtime_error(
            "usage: ftlpu_binary_inspect program.ftlpu "
            "[--all-queues] [--physical-only] [--storage-only] [--trace schedule.csv] "
            "[--compare reference.ftlpu]");
    bool reportAllQueues = false;
    bool physicalOnly = false;
    bool storageOnly = false;
    std::optional<std::filesystem::path> tracePath;
    std::optional<std::filesystem::path> comparePath;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--physical-only") {
            physicalOnly = true;
            continue;
        }
        if (argument == "--storage-only") { storageOnly=true; continue; }
        if (argument == "--all-queues") {
            reportAllQueues = true;
            continue;
        }
        if (argument == "--trace" || argument == "--compare") {
            if (++index >= argc)
                throw std::runtime_error(std::string(argument)
                    + " requires a path");
            if (argument == "--trace")
                tracePath = std::filesystem::path(argv[index]);
            else
                comparePath = std::filesystem::path(argv[index]);
            continue;
        }
        throw std::runtime_error(
            "unknown binary inspector option: " + std::string(argument));
    }

    if (physicalOnly && (comparePath || tracePath))
        throw std::runtime_error("--physical-only cannot be combined with logical comparison or trace");
    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    if(storageOnly) {
        using namespace ftlpu::software::runtime;
        const auto main=analyze_physical_imem(program);
        auto transport=program;transport.queues=program.transport_queues;transport.transport_queues.clear();
        const auto tx=analyze_physical_imem(transport);
        std::size_t records=0,pseudo=0,peak=0,pseudoPeak=0,nativePeak=0,qctlBytes=16;
        for(const auto& partition:program.queue_sync_plan.partitions) {
            records+=partition.entries.size();peak=std::max(peak,partition.entries.size());
            auto& kindPeak=!partition.entries.empty() && partition.entries.front().native_binding ? nativePeak : pseudoPeak;
            kindPeak=std::max(kindPeak,partition.entries.size());
            qctlBytes+=12+24*partition.entries.size();
            for(const auto& c:partition.entries)pseudo+=!c.native_binding;
        }
        for(const auto& e:program.queue_sync_plan.events)qctlBytes+=12+4*e.page_use_indices.size();
        auto noSource=program;noSource.queue_link_source={};
        std::ostringstream with(std::ios::binary),without(std::ios::binary);
        write_binary_program(program,with);write_binary_program(noSource,without);
        std::cout<<"storage file_bytes="<<std::filesystem::file_size(argv[1])
            <<" main_imem_bytes="<<(main.used_bits+7)/8<<" transport_imem_bytes="<<(tx.used_bits+7)/8
            <<" qctl_bytes="<<qctlBytes<<" qctl_records="<<records<<" pseudo_records="<<pseudo
            <<" peak_records_per_queue="<<peak
            <<" peak_pseudo_records_per_queue="<<pseudoPeak<<" peak_native_bindings_per_queue="<<nativePeak
            <<" qctl_storage_version="<<program.hardware.queue_control_storage_version
            <<" qctl_pool_capacity="<<program.hardware.queue_control_pool_capacity
            <<" qctl_pool_read_ii="<<program.hardware.queue_control_pool_read_ii<<" qctl_capacity="<<program.hardware.queue_control_capacity
            <<" qctl_cache="<<program.hardware.queue_control_cache_depth
            <<" qctl_latency="<<program.hardware.queue_control_read_latency
            <<" qctl_ports="<<program.hardware.queue_control_read_ports
            <<" qlnk_bytes="<<24+with.str().size()-without.str().size()
            <<" final_link_required="<<program.queue_link_source.required<<'\n';
        return 0;
    }
    std::ifstream binaryVersionStream(argv[1], std::ios::binary);
    binaryVersionStream.seekg(8);
    std::uint32_t binaryVersion = 0;
    binaryVersionStream.read(reinterpret_cast<char*>(&binaryVersion),
        sizeof(binaryVersion));
    if (!binaryVersionStream)
        throw std::runtime_error("failed to read FTLPU binary version");
    const auto cycles = program.max_cycle + 64;
    std::cout << "binary target=" << program.target_name
              << " max_cycle=" << program.max_cycle
              << " measured_cycles=" << cycles
              << " mxms_per_hemisphere="
              << program.hardware.mxms_per_hemisphere
              << " tile_rows=" << program.hardware.tile_rows
              << " mxm_rows=" << program.hardware.mxm_rows
              << " bytes_per_word=" << program.hardware.bytes_per_word
              << " vxm_alus=" << program.hardware.vxm_alus << '\n';
    std::vector<const ftlpu::software::runtime::QueueProgram*> queues;
    queues.reserve(program.queues.size());
    for (const auto& queue : program.queues) queues.push_back(&queue);
    std::ranges::sort(queues, [](const auto* lhs, const auto* rhs) {
        return lhs->commands.size() > rhs->commands.size();
    });
    std::size_t total_commands = 0;
    for (const auto& queue : program.queues)
        total_commands += queue.commands.size();
    std::cout << "binary queues=" << program.queues.size()
              << " commands=" << total_commands
              << " file_bytes=" << std::filesystem::file_size(argv[1])
              << '\n';
    // Logical issue comparison concerns only computation. DMA/RX and the
    // independently loaded MEM transport have physical schedules of their own.
    const auto computationOnly = [](auto image) {
        using ftlpu::software::runtime::QueueKind;
        std::erase_if(image.queues, [](const auto& queue) {
            return queue.kind == QueueKind::C2cDma
                || queue.kind == QueueKind::C2cTx || queue.kind == QueueKind::C2cRx;
        });
        image.transport_queues.clear();
        return image;
    };
    std::optional<ftlpu::software::runtime::LogicalIssueProgram>
        logicalIssues;
    if (physicalOnly || !comparePath) {
        // Physical inspection stays cheap for large coarse programs. Explicit
        // comparison uses the logical inspector's native FU packet expansion.
        std::cout << "binary logical_issue_summary mode="
                  << (physicalOnly ? "physical_only" : "packed_macro")
                  << " status=not_expanded\n";
    } else {
        logicalIssues =
            ftlpu::software::runtime::inspect_logical_issues(computationOnly(program));
        std::cout << "binary logical_issue_summary queues="
                  << logicalIssues->queues.size()
                  << " functional_issues="
                  << logicalIssues->functional_issues
                  << " logical_nop_cycles="
                  << logicalIssues->logical_nop_cycles << '\n';
    }
    const auto imem =
        ftlpu::software::runtime::analyze_cmodel_abstract_imem(program);
    const auto savedImemWork = imem.expanded_work > imem.encoded_work_entries
        ? imem.expanded_work - imem.encoded_work_entries : 0;
    const auto workCompression = imem.expanded_work == 0 ? 0.0
        : 100.0 * static_cast<double>(savedImemWork)
            / static_cast<double>(imem.expanded_work);
    std::cout << "imem model=cmodel-abstract"
              << " deployable=" << (imem.fits() ? "yes" : "no")
              << " queues=" << imem.queues.size()
              << " overflow_queues=" << imem.overflow_queues
              << " used_slots=" << imem.used_slots
              << " used_bits=" << imem.used_bits
              << " used_bytes_ceil=" << (imem.used_bits + 7) / 8
              << " active_queue_capacity_bits="
              << imem.active_queue_capacity_bits
              << " expanded_work=" << imem.expanded_work
              << " encoded_work_entries=" << imem.encoded_work_entries
              << " saved_work_entries=" << savedImemWork
              << " work_compression_percent=" << workCompression
              << '\n';
    for (const auto kind : {
             ftlpu::software::runtime::QueueKind::Mem,
             ftlpu::software::runtime::QueueKind::MxmLoad,
             ftlpu::software::runtime::QueueKind::MxmCompute,
             ftlpu::software::runtime::QueueKind::MxmDequant,
             ftlpu::software::runtime::QueueKind::Vxm,
             ftlpu::software::runtime::QueueKind::SxmTranspose,
             ftlpu::software::runtime::QueueKind::SxmPermute,
             ftlpu::software::runtime::QueueKind::C2cDma,
             ftlpu::software::runtime::QueueKind::C2cTx,
             ftlpu::software::runtime::QueueKind::C2cRx}) {
        std::size_t queueCount = 0;
        std::size_t usedSlots = 0;
        std::size_t overflowQueues = 0;
        std::size_t maxUsed = 0;
        std::uint64_t usedBits = 0;
        std::uint64_t expandedWork = 0;
        std::size_t encodedWork = 0;
        std::uint32_t slotBits = 0;
        std::uint32_t depth = 0;
        for (const auto& queue : imem.queues) {
            if (queue.kind != kind) continue;
            ++queueCount;
            usedSlots += queue.used_slots;
            usedBits += queue.used_bits();
            expandedWork += queue.expanded_work;
            encodedWork += queue.instruction_entries
                + queue.repeat_entries
                + queue.macro_entries
                + queue.coarse_program_entries;
            maxUsed = std::max(maxUsed, queue.used_slots);
            overflowQueues += queue.overflow() ? 1 : 0;
            slotBits = queue.slot_bits;
            depth = queue.depth;
        }
        if (queueCount == 0) continue;
        const auto saved = expandedWork > encodedWork
            ? expandedWork - encodedWork : 0;
        std::cout << "imem resource="
                  << ftlpu::software::runtime::queue_kind_name(kind)
                  << " slot_bits=" << slotBits
                  << " depth_per_queue=" << depth
                  << " queues=" << queueCount
                  << " used_slots=" << usedSlots
                  << " max_used_slots=" << maxUsed
                  << " max_utilization_percent="
                  << 100.0 * static_cast<double>(maxUsed) / depth
                  << " used_bits=" << usedBits
                  << " expanded_work=" << expandedWork
                  << " encoded_work_entries=" << encodedWork
                  << " saved_work_entries=" << saved
                  << " overflow_queues=" << overflowQueues << '\n';
    }
    for (const auto& queue : imem.queues) {
        if (!reportAllQueues && !queue.overflow()) continue;
        std::cout << "imem queue resource="
                  << ftlpu::software::runtime::queue_kind_name(queue.kind)
                  << " queue=" << queue.index
                  << " slot_bits=" << queue.slot_bits
                  << " depth=" << queue.depth
                  << " used_slots=" << queue.used_slots
                  << " utilization_percent="
                  << 100.0 * static_cast<double>(queue.used_slots)
                      / queue.depth
                  << " overflow_slots=" << queue.overflow_slots()
                  << " expanded_work=" << queue.expanded_work
                  << " instruction=" << queue.instruction_entries
                  << " nop=" << queue.nop_entries
                  << " repeat=" << queue.repeat_entries
                  << " macro=" << queue.macro_entries
                  << " coarse_program=" << queue.coarse_program_entries
                  << '\n';
    }
    // Report physical computation and live transport separately. Container
    // metadata costs are explicit and are not counted as hardware i-MEM.
    auto computeStorageProgram = program;
    auto transportStorageProgram = program;
    computeStorageProgram.queues.clear();
    transportStorageProgram.queues = program.transport_queues;
    computeStorageProgram.transport_queues.clear();
    transportStorageProgram.transport_queues.clear();
    for (const auto& queue : program.queues) {
        using ftlpu::software::runtime::QueueKind;
        if (queue.kind == QueueKind::C2cDma || queue.kind == QueueKind::C2cTx
            || queue.kind == QueueKind::C2cRx)
            transportStorageProgram.queues.push_back(queue);
        else
            computeStorageProgram.queues.push_back(queue);
    }
    const auto computeStorage = ftlpu::software::runtime::analyze_physical_imem(computeStorageProgram);
    const auto transportStorage = ftlpu::software::runtime::analyze_physical_imem(transportStorageProgram);
    const auto computeBytes = (computeStorage.used_bits + 7) / 8;
    const auto transportBytes = (transportStorage.used_bits + 7) / 8;
    const auto barrierRamBytes = program.barriers.size() * std::uint64_t{8};
    const auto barrierBlockBytes = binaryVersion >= 37 ? std::uint64_t{16} + barrierRamBytes : 0;
    std::uint64_t associationBytes = 0;
    std::uint64_t completionInstances = 0;
    for (const auto& event : program.barrier_events) {
        associationBytes += 8 + event.page_use_indices.size() * std::uint64_t{4};
        completionInstances += event.page_use_indices.size();
    }
    std::cout << "binary common_barrier_storage protocol=" << program.common_barrier_protocol
              << " overlap_protocol=" << program.transport_overlap_protocol
              << " overlap_version=" << program.hardware.common_transport_overlap_version
              << " entries=" << program.barriers.size()
              << " barrier_ram_bytes=" << barrierRamBytes
              << " barrier_block_bytes=" << barrierBlockBytes
              << " event_association_bytes=" << associationBytes
              << " completion_instances=" << completionInstances
              << " use_launch_metadata_bytes=" << (binaryVersion >= 37 ? program.weight_page_uses.size() * (binaryVersion >= 38 ? 8 : 4) : 0)
              << " compute_imem_bytes=" << computeBytes
              << " transport_native_words=" << transportStorage.used_slots
              << " transport_native_bytes=" << transportBytes
              << " total_imem_bytes=" << computeBytes + transportBytes
              << " imem_and_barrier_ram_bytes=" << computeBytes + transportBytes + barrierRamBytes
              << " total_file_bytes=" << std::filesystem::file_size(argv[1]) << '\n';
    for (const auto& barrier : program.barriers) {
        const auto event = std::ranges::find(program.barrier_events, barrier.event_id,
            &ftlpu::software::runtime::BinaryBarrierEvent::event_id);
        std::cout << "binary common_barrier stop_cycle=" << barrier.stop_cycle
                  << " event_id=" << barrier.event_id << " page_use_instances=";
        for (std::size_t i = 0; i < event->page_use_indices.size(); ++i)
            std::cout << (i ? "," : "") << event->page_use_indices[i];
        std::cout << '\n';
    }
    for (std::size_t i=0;i<program.weight_page_uses.size();++i) {
        const auto& use=program.weight_page_uses[i];
        std::cout << "binary transport_instance=" << i << " launch_event=" << use.transport_launch_event_id
                  << " launch_cycle=" << use.transport_launch_cycle << '\n';
    }
    const auto physical =
        ftlpu::software::runtime::analyze_physical_imem(program);
    std::cout << "imem model=target-physical-v1"
              << " deployable=" << (physical.fits() ? "yes" : "no")
              << " queues=" << physical.queues.size()
              << " mem_delta_rle_queues=" << physical.mem_delta_rle_queues
              << " overflow_queues=" << physical.overflow_queues
              << " macro_context_overflow_queues="
              << physical.macro_context_overflow_queues
              << " used_slots=" << physical.used_slots
              << " used_bits=" << physical.used_bits
              << " used_bytes_ceil=" << (physical.used_bits + 7) / 8
              << " active_queue_capacity_bits="
              << physical.active_queue_capacity_bits
              << " peak_macro_context_bits="
              << physical.peak_macro_context_bits
              << " provisioned_macro_context_bits="
              << physical.provisioned_macro_context_bits
              << '\n';
    for (const auto kind : {
             ftlpu::software::runtime::QueueKind::Mem,
             ftlpu::software::runtime::QueueKind::MxmLoad,
             ftlpu::software::runtime::QueueKind::MxmCompute,
             ftlpu::software::runtime::QueueKind::MxmDequant,
             ftlpu::software::runtime::QueueKind::Vxm,
             ftlpu::software::runtime::QueueKind::SxmTranspose,
             ftlpu::software::runtime::QueueKind::SxmPermute,
             ftlpu::software::runtime::QueueKind::C2cDma,
             ftlpu::software::runtime::QueueKind::C2cTx,
             ftlpu::software::runtime::QueueKind::C2cRx}) {
        std::size_t queues = 0;
        std::size_t slots = 0;
        std::size_t maxSlots = 0;
        std::size_t peakContexts = 0;
        std::size_t contextCapacity = 0;
        std::uint64_t peakContextBits = 0;
        std::uint64_t provisionedContextBits = 0;
        std::size_t runs = 0;
        std::size_t escapes = 0;
        std::size_t compactTemplateRuns = 0;
        std::size_t extendedTemplateRuns = 0;
        std::uint64_t bits = 0;
        for (const auto& queue : physical.queues) {
            if (queue.kind != kind) continue;
            ++queues;
            slots += queue.physical_slots;
            maxSlots = std::max(maxSlots, queue.physical_slots);
            peakContexts = std::max(peakContexts, queue.peak_macro_contexts);
            contextCapacity = std::max(
                contextCapacity, queue.macro_context_capacity);
            peakContextBits += static_cast<std::uint64_t>(
                queue.peak_macro_contexts) * queue.macro_context_bits;
            provisionedContextBits += static_cast<std::uint64_t>(
                queue.macro_context_capacity) * queue.macro_context_bits;
            runs += queue.macro_codec.run_count;
            escapes += queue.macro_codec.escaped_transitions;
            compactTemplateRuns += queue.macro_codec.compact_template_runs;
            extendedTemplateRuns += queue.macro_codec.extended_template_runs;
            bits += queue.physical_bits;
        }
        if (queues == 0) continue;
        std::cout << "imem physical_resource="
                  << ftlpu::software::runtime::queue_kind_name(kind)
                  << " queues=" << queues
                  << " used_slots=" << slots
                  << " max_used_slots=" << maxSlots
                  << " used_bits=" << bits
                  << " peak_macro_contexts=" << peakContexts
                  << " macro_context_capacity=" << contextCapacity
                  << " peak_macro_context_bits=" << peakContextBits
                  << " provisioned_macro_context_bits="
                  << provisionedContextBits
                  << " template_runs=" << runs
                  << " compact_template_runs=" << compactTemplateRuns
                  << " extended_template_runs=" << extendedTemplateRuns
                  << " escaped_deltas=" << escapes << '\n';
    }
    std::map<std::pair<std::uint32_t,
        ftlpu::software::runtime::QueueKind>, std::size_t>
        scaleRelocations;
    for (const auto& relocation : program.scale_relocations)
        ++scaleRelocations[{relocation.binding_index,
            relocation.queue_kind}];
    std::cout << "binary scale_relocations="
              << program.scale_relocations.size() << '\n';
    for (const auto& [key, count] : scaleRelocations)
        std::cout << "binary scale_relocation binding=" << key.first
                  << " resource="
                  << ftlpu::software::runtime::queue_kind_name(key.second)
                  << " count=" << count << '\n';
    for (const auto& timeline : program.timelines)
        std::cout << "binary timeline name=" << timeline.name
                  << " start_cycle=" << timeline.start_cycle
                  << " end_cycle=" << timeline.end_cycle << '\n';
    std::map<std::pair<ftlpu::software::runtime::BindingAccess,
        std::uint32_t>, std::size_t> addressRelocations;
    for (const auto& relocation : program.address_relocations)
        ++addressRelocations[{relocation.binding_access,
            relocation.binding_index}];
    for (const auto& [key, count] : addressRelocations)
        std::cout << "binary address_relocation access="
                  << static_cast<unsigned>(key.first)
                  << " binding=" << key.second
                  << " count=" << count << '\n';
    std::cout << "binary weight_page_uses="
              << program.weight_page_uses.size() << '\n';
    for (const auto& use : program.weight_page_uses)
        std::cout << "binary weight_page_use binding=" << use.binding_index
                  << " page=" << use.page_index
                  << " bank=" << use.bank
                  << " runtime_prefetch=" << use.runtime_prefetch
                  << " ready_cycle=" << use.ready_cycle
                  << " release_cycle=" << use.release_cycle << '\n';
    const auto weightPrefetches =
        ftlpu::software::runtime::plan_weight_prefetches(program);
    std::map<std::pair<std::uint16_t, std::uint32_t>, std::size_t>
        residencyFragments;
    std::map<std::pair<std::uint16_t, std::uint32_t>, std::uint64_t>
        residencyVectors;
    std::map<std::uint16_t,
        std::set<std::pair<std::size_t, std::uint16_t>>> weightLanes;
    for (const auto& plan : weightPrefetches) {
        ++residencyFragments[{plan.bank, plan.residency_page_index}];
        for (const auto& region : plan.regions) {
            const auto rows = static_cast<std::uint64_t>(
                region.row_end - region.row_begin);
            for (std::size_t side = 0; side < 2; ++side) {
                if ((region.hemisphere_mask & (1u << side)) == 0) continue;
                residencyVectors[{plan.bank, plan.residency_page_index}]
                    += rows;
                if (!region.wildcard_slice)
                    weightLanes[plan.bank].insert({side, region.slice});
            }
        }
    }
    std::cout << "binary weight_residency_pages="
              << residencyFragments.size()
              << " transfer_fragments=" << weightPrefetches.size() << '\n';
    for (const auto& [key, fragments] : residencyFragments) {
        const std::uint64_t capacityVectors =
            weightLanes[key.first].size()
            * static_cast<std::uint64_t>(program.hardware.sram_depth_rows);
        const std::uint64_t usedVectors = residencyVectors[key];
        const double utilization = capacityVectors == 0 ? 0.0
            : 100.0 * static_cast<double>(usedVectors)
                / static_cast<double>(capacityVectors);
        std::cout << "binary weight_residency_page bank=" << key.first
                  << " page=" << key.second
                  << " fragments=" << fragments
                  << " used_bytes="
                  << usedVectors * ftlpu::hw::kPhysicalVectorBytes
                  << " capacity_bytes="
                  << capacityVectors * ftlpu::hw::kPhysicalVectorBytes
                  << " utilization_percent=" << utilization << '\n';
    }
    for (const auto& plan : weightPrefetches) {
        std::cout << "binary weight_prefetch_plan page=" << plan.page_index
                  << " residency_page=" << plan.residency_page_index
                  << " bank=" << plan.bank
                  << " pre_execution=" << plan.pre_execution
                  << " start_cycle=" << plan.start_cycle
                  << " transfer_end_cycle=" << plan.transfer_end_cycle
                  << " ready_cycle=" << plan.ready_cycle
                  << " release_cycle=" << plan.release_cycle
                  << " uses=" << plan.use_indices.size() << '\n';
    }
    std::cout << "binary stream_release_cycles="
              << program.stream_release_cycles.size() << '\n';
    const auto reportStreamReleases = [&](std::string_view direction,
                                          std::size_t offset,
                                          std::size_t count) {
        for (std::size_t begin = 0; begin < count;) {
            std::size_t end = begin + 1;
            while (end < count
                && program.stream_release_cycles[offset + end]
                    == program.stream_release_cycles[offset + begin])
                ++end;
            std::cout << "binary stream_release direction=" << direction
                      << " streams=" << begin;
            if (end != begin + 1) std::cout << ".." << end - 1;
            std::cout << " release_cycle="
                      << program.stream_release_cycles[offset + begin]
                      << '\n';
            begin = end;
        }
    };
    const std::size_t directionalStreams =
        program.hardware.streams_per_direction;
    if (program.stream_release_cycles.size()
            == program.hardware.encoded_streams
        && program.hardware.encoded_streams == 2 * directionalStreams) {
        reportStreamReleases("east", 0, directionalStreams);
        reportStreamReleases(
            "west", directionalStreams, directionalStreams);
    } else {
        reportStreamReleases(
            "merged", 0, program.stream_release_cycles.size());
    }
    const std::size_t reported = reportAllQueues
        ? queues.size() : std::min<std::size_t>(queues.size(), 20);
    for (std::size_t i = 0; i < reported; ++i) {
        const auto& queue = *queues[i];
        std::size_t instructions = 0;
        std::size_t nops = 0;
        std::size_t repeats = 0;
        std::size_t macros = 0;
        for (const auto& command : queue.commands) {

            if (ftlpu::software::runtime::is_macro_schedule_command(command)) {
                ++macros;
                continue;
            }

            switch (ftlpu::isa::decode_icu_command_opcode(command.command)) {
            case ftlpu::isa::IcuCommandOpcode::Instruction:
                ++instructions;
                break;
            case ftlpu::isa::IcuCommandOpcode::Nop:
                ++nops;
                break;
            case ftlpu::isa::IcuCommandOpcode::Repeat:
                ++repeats;
                break;
            default:
                break;
            }
        }
        std::cout << "binary queue_occupancy rank=" << i
                  << " resource="
                  << ftlpu::software::runtime::queue_kind_name(queue.kind)
                  << " queue=" << queue.index
                  << " commands=" << queue.commands.size()
                  << " instruction=" << instructions
                  << " nop=" << nops
                  << " repeat=" << repeats
                  << " macro=" << macros
                  << '\n';
    }
    for (const auto kind : {
             ftlpu::software::runtime::QueueKind::Mem,
             ftlpu::software::runtime::QueueKind::MxmLoad,
             ftlpu::software::runtime::QueueKind::MxmCompute,
             ftlpu::software::runtime::QueueKind::MxmDequant,
             ftlpu::software::runtime::QueueKind::Vxm,
             ftlpu::software::runtime::QueueKind::SxmTranspose,
             ftlpu::software::runtime::QueueKind::SxmPermute,
             ftlpu::software::runtime::QueueKind::C2cDma,
             ftlpu::software::runtime::QueueKind::C2cTx,
             ftlpu::software::runtime::QueueKind::C2cRx}) {
        std::size_t kindQueues = 0;
        std::size_t kindCommands = 0;
        std::size_t kindMin = std::numeric_limits<std::size_t>::max();
        std::size_t kindMax = 0;
        for (const auto* queue : queues) {
            if (queue->kind != kind) continue;
            ++kindQueues;
            kindCommands += queue->commands.size();
            kindMin = std::min(kindMin, queue->commands.size());
            kindMax = std::max(kindMax, queue->commands.size());
        }
        if (kindQueues != 0)
            std::cout << "binary resource_summary resource="
                      << ftlpu::software::runtime::queue_kind_name(kind)
                      << " queues=" << kindQueues
                      << " commands=" << kindCommands
                      << " min=" << kindMin
                      << " max=" << kindMax
                      << " average="
                      << static_cast<double>(kindCommands) / kindQueues
                      << '\n';
        const auto found = std::ranges::find_if(queues,
            [kind](const auto* queue) { return queue->kind == kind; });
        if (found == queues.end()) continue;
        std::size_t instructionCount = 0;
        std::size_t nopCount = 0;
        std::size_t repeatCount = 0;
        std::size_t macroCount = 0;
        for (const auto& command : (*found)->commands) {

            if (ftlpu::software::runtime::is_macro_schedule_command(command)) {
                ++macroCount;
                continue;
            }

            switch (ftlpu::isa::decode_icu_command_opcode(command.command)) {
            case ftlpu::isa::IcuCommandOpcode::Instruction:
                ++instructionCount;
                break;
            case ftlpu::isa::IcuCommandOpcode::Nop:
                ++nopCount;
                break;
            case ftlpu::isa::IcuCommandOpcode::Repeat:
                ++repeatCount;
                break;
            default:
                break;
            }
        }
        std::cout << "binary resource_max resource="
                  << ftlpu::software::runtime::queue_kind_name(kind)
                  << " queue=" << (*found)->index
                  << " commands=" << (*found)->commands.size()
                  << " instruction=" << instructionCount
                  << " nop=" << nopCount
                  << " repeat=" << repeatCount
                  << " macro=" << macroCount
                  << '\n';
    }
    for (const auto& binding : program.bindings) {
        std::cout << "binary binding index=" << binding.index
                  << " access=" << static_cast<unsigned>(binding.access)
                  << " role=" << binding.role
                  << " name=" << binding.name
                  << " layout=" << static_cast<unsigned>(binding.layout)
                  << " bytes=" << binding.byte_size
                  << " page_count=" << binding.page_count
                  << " base_row=" << binding.base_row
                  << " rows=" << binding.instruction_count
                  << " bank=" << binding.bank
                  << " hemisphere_mask=" << binding.hemisphere_mask
                  << " slices=";
        for (std::size_t i = 0; i < binding.slices.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << binding.slices[i];
        }
        if (binding.paged_weight) {
            std::cout << " page_granularity=" << binding.page_granularity
                      << " page_items_per_slice_group="
                      << binding.page_items_per_slice_group
                      << " page_storage_slices=";
            for (std::size_t i = 0;
                 i < binding.page_storage_slices.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << binding.page_storage_slices[i];
            }
            std::cout << " page_placements=";
            for (std::uint32_t page = 0; page < binding.page_count; ++page) {
                if (page != 0) std::cout << ';';
                const auto placement =
                    ftlpu::software::runtime::resolve_weight_page_placement(
                        binding, page);
                std::cout << "bank" << placement.bank
                          << ":groups" << placement.slice_group_base
                          << '+' << placement.slice_group_count
                          << ":rows" << placement.base_row
                          << '+' << placement.row_count;
            }
        }
        std::cout << '\n';
    }
    if (!physicalOnly)
        ftlpu::software::runtime::print_runtime_performance(
            program, cycles, std::cout);
    if (tracePath)
        ftlpu::software::runtime::write_schedule_trace_csv(
            program, *tracePath);
    if (comparePath) {
        const auto reference =
            ftlpu::software::runtime::read_binary_program(*comparePath);
        const auto referenceIssues =
            ftlpu::software::runtime::inspect_logical_issues(computationOnly(reference));
        const auto comparison =
            ftlpu::software::runtime::compare_logical_issues(
                program, reference);
        const auto leftBytes = std::filesystem::file_size(argv[1]);
        const auto rightBytes = std::filesystem::file_size(*comparePath);
        const auto byteDelta = static_cast<std::int64_t>(leftBytes)
            - static_cast<std::int64_t>(rightBytes);
        std::cout << "binary compare result="
                  << (comparison.equivalent ? "equivalent" : "mismatch")
                  << " same_target=" << comparison.same_target
                  << " same_horizon=" << comparison.same_horizon
                  << " left_functional_issues="
                  << logicalIssues->functional_issues
                  << " right_functional_issues="
                  << referenceIssues.functional_issues
                  << " left_logical_nop_cycles="
                  << logicalIssues->logical_nop_cycles
                  << " right_logical_nop_cycles="
                  << referenceIssues.logical_nop_cycles
                  << " left_file_bytes=" << leftBytes
                  << " right_file_bytes=" << rightBytes
                  << " left_minus_right_bytes=" << byteDelta << '\n';
        if (comparison.first_mismatch) {
            const auto& mismatch = *comparison.first_mismatch;
            std::cout << "binary compare_first_mismatch reason=\""
                      << mismatch.reason << "\" cycle=" << mismatch.cycle;
            if (mismatch.left || mismatch.right)
                std::cout << " queue="
                          << ftlpu::software::runtime::queue_kind_name(
                                 mismatch.kind)
                          << '[' << mismatch.index << ']';
            std::cout << " left="
                      << (mismatch.left
                              ? ftlpu::software::runtime::
                                    describe_logical_instruction(
                                        *mismatch.left)
                              : "NOP")
                      << " right="
                      << (mismatch.right
                              ? ftlpu::software::runtime::
                                    describe_logical_instruction(
                                        *mismatch.right)
                              : "NOP")
                      << '\n';
        }
        if (!comparison.equivalent) return 2;
    }
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu_binary_inspect failed: " << ex.what() << '\n';
    return 1;
}
