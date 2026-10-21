#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

BinaryBinding make_weight_binding(std::uint16_t bank = 0)
{
    BinaryBinding binding;
    binding.index = 0;
    binding.access = BindingAccess::Input;
    binding.element_type = BindingElementType::I8;
    binding.layout = BindingLayout::W8A16AttentionWeightStriped;
    binding.byte_size = 32 * 128;
    binding.base_row = 0;
    binding.instruction_count = 8;
    binding.address_stride = 1;
    binding.shape = {32, 128};
    binding.slices = {20, 21, 22, 23, 24, 25, 26, 27};
    binding.role = "weight";
    binding.name = "q_proj.weight";
    binding.hemisphere_mask = 3;
    binding.bank = bank;
    binding.paged_weight = true;
    binding.page_count = 1;
    binding.page_rows = 8;
    binding.page_granularity = 1;
    binding.page_role_group_base = 0;
    binding.page_role_group_count = 1;
    binding.page_items_per_slice_group = 1;
    binding.page_bank_count = 2;
    binding.page_storage_slices = binding.slices;
    return binding;
}

} // namespace

int main()
try {
    BinaryProgram preloadProgram;
    auto firstPreload = make_weight_binding();
    firstPreload.bank = 1;
    auto alternateBankPreload = make_weight_binding();
    alternateBankPreload.index = 1;
    alternateBankPreload.bank = 0;
    alternateBankPreload.slices = {28, 29, 30, 31, 32, 33, 34, 35};
    alternateBankPreload.page_storage_slices =
        alternateBankPreload.slices;
    preloadProgram.bindings = {firstPreload, alternateBankPreload};
    preloadProgram.weight_page_uses = {
        {0, 0, 1, 200, 220},
        {1, 0, 0, 400, 420},
    };
    const auto preloadPlans = plan_weight_prefetches(preloadProgram);
    if (preloadPlans.size() != 2 || !preloadPlans[0].pre_execution
        || !preloadPlans[1].pre_execution)
        throw std::runtime_error(
            "disjoint alternate-bank weight page was not preloaded");
    auto runtimePrefetchProgram = preloadProgram;
    runtimePrefetchProgram.weight_page_uses[1].runtime_prefetch = true;
    std::ostringstream runtimePrefetchBinary(
        std::ios::out | std::ios::binary);
    write_binary_program(runtimePrefetchProgram, runtimePrefetchBinary);
    std::istringstream runtimePrefetchStream(
        runtimePrefetchBinary.str(), std::ios::in | std::ios::binary);
    const auto decodedRuntimePrefetch =
        read_binary_program(runtimePrefetchStream);
    if (decodedRuntimePrefetch.weight_page_uses.size() != 2
        || !decodedRuntimePrefetch.weight_page_uses[1].runtime_prefetch)
        throw std::runtime_error(
            "runtime-prefetch policy did not survive binary round-trip");
    const auto runtimePrefetchPlans =
        plan_weight_prefetches(runtimePrefetchProgram);
    {
        auto tagged = runtimePrefetchProgram;
        tagged.queues.push_back({QueueKind::Vxm, 0,
            {encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0x30000)),
             encode_icu_control_raw_word(IcuControlInstruction::Sync()),
             encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0x2ffff)),
             encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0x40000))}});
        const auto unlinked = without_compiled_mem_write_sync(tagged);
        const auto& controls = unlinked.queues.back().commands;
        if (decode_icu_control_raw_word(controls[0]).opcode != IcuControlOpcode::Nop
            || decode_icu_control_raw_word(controls[0]).count != 1
            || decode_icu_control_raw_word(controls[1]).opcode != IcuControlOpcode::Sync
            || decode_icu_control_raw_word(controls[2]).opcode != IcuControlOpcode::WaitEvent
            || decode_icu_control_raw_word(controls[2]).event_tag != 0x2ffff
            || decode_icu_control_raw_word(controls[3]).opcode != IcuControlOpcode::WaitEvent
            || decode_icu_control_raw_word(controls[3]).event_tag != 0x40000)
            throw std::runtime_error("compiler page-event rebind changed ordinary SYNC");
    }
    if (runtimePrefetchPlans.size() != 2
        || !runtimePrefetchPlans[0].pre_execution
        || runtimePrefetchPlans[1].pre_execution
        || runtimePrefetchPlans[1].start_cycle != 0)
        throw std::runtime_error(
            "explicit next-epoch page was not left on the runtime timeline");
    auto scratchReuseProgram = runtimePrefetchProgram;
    QueueProgram scratchQueue{QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 28, 0), {}};
    scratchQueue.commands.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(80)));
    const auto scratchPacket = isa::encode_mem_icu_shortest_instruction(
        MemIcuInstruction::Write3D(IcuLoop3D{0, {4,1,1}, {5,1,1}},
            MemIcuAddress3D::Affine(0, {1,0,0}), StreamId::East(0)));
    for (std::size_t w = 0; w < isa::fu_loop_packet_word_count(scratchPacket); ++w) {
        const auto& word = scratchPacket.words[w];
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mem;
        command.word_count = isa::EncodedMemIcu3DPacket::kLanesPerWord;
        for (std::size_t lane = 0; lane < command.word_count; ++lane)
            command.words[lane] = word.lanes[lane];
        scratchQueue.commands.push_back(command);
    }
    scratchReuseProgram.queues.push_back(scratchQueue);
    BinaryBinding sharedScratch;
    sharedScratch.access = BindingAccess::Internal;
    sharedScratch.bank = 0;
    sharedScratch.hemisphere_mask = 1;
    sharedScratch.slices = {28};
    sharedScratch.instruction_count = 4;
    sharedScratch.address_stride = 1;
    scratchReuseProgram.bindings.push_back(sharedScratch);
    const auto scratchReusePlans = plan_weight_prefetches(scratchReuseProgram);
    if (scratchReusePlans[1].earliest_mem_cycle != 96
        || scratchReusePlans[1].start_cycle < 96)
        throw std::runtime_error("C2C page overwrote a live native scratch domain");
    auto initializedOverlapProgram = preloadProgram;
    BinaryBinding rope;
    rope.access = BindingAccess::Internal;
    rope.element_type = BindingElementType::F16;
    rope.layout = BindingLayout::Fp16RopeTable;
    rope.base_row = 0;
    rope.instruction_count = 8;
    rope.address_stride = 1;
    rope.slices = alternateBankPreload.slices;
    rope.hemisphere_mask = 3;
    rope.bank = 0;
    rope.initializer = BindingInitializer::RopeTable;
    initializedOverlapProgram.bindings.push_back(rope);
    initializedOverlapProgram.timelines.push_back({"rope", 20, 40});
    const auto initializedOverlapPlans =
        plan_weight_prefetches(initializedOverlapProgram);
    if (initializedOverlapPlans.size() != 2
        || !initializedOverlapPlans[0].pre_execution
        || initializedOverlapPlans[1].pre_execution
        || initializedOverlapPlans[1].start_cycle != 40
        || initializedOverlapPlans[1].earliest_mem_cycle != 40)
        throw std::runtime_error(
            "weight page ignored initialized SRAM release timeline");
    auto lowBandwidthHardware = preloadProgram.hardware;
    lowBandwidthHardware.lpu_clock_mhz = 1000;
    lowBandwidthHardware.ddr_peak_bandwidth_mbytes_per_second = 6400;
    lowBandwidthHardware.ddr_read_latency_cycles = 2500;
    lowBandwidthHardware.ddr_read_latency_jitter_cycles = 500;
    const auto requireDdrIndependentPlans = [&](const BinaryProgram& program) {
        const auto normal = plan_weight_prefetches(program);
        const auto slow = plan_weight_prefetches(program, lowBandwidthHardware);
        if (normal.size() != slow.size())
            throw std::runtime_error("DDR changed the page plan count");
        for (std::size_t index = 0; index < normal.size(); ++index) {
            const auto& a = normal[index];
            const auto& b = slow[index];
            if (a.start_cycle != b.start_cycle
                || a.transfer_end_cycle != b.transfer_end_cycle
                || a.earliest_mem_cycle != b.earliest_mem_cycle
                || a.ready_cycle != b.ready_cycle
                || a.pre_execution != b.pre_execution
                || a.fabric_streams != b.fabric_streams)
                throw std::runtime_error(
                    "DDR prediction leaked into the logical ICU page timeline");
        }
    };
    requireDdrIndependentPlans(preloadProgram);
    requireDdrIndependentPlans(runtimePrefetchProgram);
    requireDdrIndependentPlans(initializedOverlapProgram);

    BinaryProgram coalescedProgram;
    auto coalescedFirst = make_weight_binding(1);
    auto coalescedDisjoint = make_weight_binding(1);
    coalescedDisjoint.index = 1;
    coalescedDisjoint.base_row = 16;
    auto coalescedReplacement = make_weight_binding(1);
    coalescedReplacement.index = 2;
    coalescedProgram.bindings = {
        coalescedFirst, coalescedDisjoint, coalescedReplacement};
    coalescedProgram.weight_page_uses = {
        {0, 0, 1, 100, 200},
        {1, 0, 1, 300, 400},
        {2, 0, 1, 500, 600},
    };
    const auto coalescedPlans =
        plan_weight_prefetches(coalescedProgram);
    if (coalescedPlans.size() != 3
        || coalescedPlans[0].residency_page_index != 0
        || coalescedPlans[1].residency_page_index != 0
        || coalescedPlans[2].residency_page_index != 1
        || coalescedPlans[0].ready_cycle != 100
        || coalescedPlans[0].release_cycle != 200
        || !coalescedPlans[0].pre_execution
        || !coalescedPlans[1].pre_execution
        || coalescedPlans[2].pre_execution
        || coalescedPlans[2].start_cycle != 200
        || coalescedPlans[2].earliest_mem_cycle != 200)
        throw std::runtime_error(
            "disjoint fragments did not share a residency page while retaining independent transfers");

    BinaryProgram overlappingProgram;
    auto resident = make_weight_binding();
    auto replacement = make_weight_binding();
    replacement.index = 1;
    overlappingProgram.bindings = {resident, replacement};
    overlappingProgram.weight_page_uses = {
        {0, 0, 0, 100, 320},
        {1, 0, 0, 1000, 1100},
    };
    const auto overlappingPlans =
        plan_weight_prefetches(overlappingProgram);
    if (overlappingPlans.size() != 2
        || !overlappingPlans[0].pre_execution
        || overlappingPlans[1].pre_execution
        || overlappingPlans[1].start_cycle != 320)
        throw std::runtime_error(
            "overlapping weight replacement did not launch at release");
    auto sharedStreamProgram = overlappingProgram;
    sharedStreamProgram.stream_release_cycles.assign(
        sharedStreamProgram.hardware.encoded_streams, 0);
    const auto westBase =
        sharedStreamProgram.hardware.streams_per_direction;
    for (std::size_t stream = 0;
         stream < sharedStreamProgram.hardware.streams_per_direction;
         ++stream)
        sharedStreamProgram.stream_release_cycles[westBase + stream] = 400;
    // Four high West streams are already idle at the SRAM reuse boundary.
    // The linker must start immediately with those four transport lanes
    // instead of waiting for a fixed contiguous eight-stream range.
    for (std::size_t stream = 28; stream < 32; ++stream)
        sharedStreamProgram.stream_release_cycles[westBase + stream] = 0;
    const auto sharedStreamPlans =
        plan_weight_prefetches(sharedStreamProgram);
    if (sharedStreamPlans.size() != 2
        || !sharedStreamPlans[0].pre_execution
        || sharedStreamPlans[1].start_cycle != 320
        || sharedStreamPlans[1].fabric_streams
            != std::vector<std::uint16_t>({31, 30, 29, 28}))
        throw std::runtime_error(
            "runtime C2C prefetch did not use the idle West SR subset");

    BinaryProgram chainedOverlapProgram;
    chainedOverlapProgram.hardware = overlappingProgram.hardware;
    std::vector<WeightPrefetchPlan> chainedPlans(3);
    chainedPlans[0].ready_cycle = 100;
    chainedPlans[0].release_cycle = 300;
    chainedPlans[0].bytes[0] = 128;
    chainedPlans[0].regions = {{0, 1, 20, 0, 8, false}};
    chainedPlans[1].ready_cycle = 400;
    chainedPlans[1].release_cycle = 600;
    chainedPlans[1].bytes[0] = 128;
    chainedPlans[1].regions = {
        {0, 1, 20, 0, 8, false},
        {0, 1, 28, 0, 8, false},
    };
    chainedPlans[2].ready_cycle = 700;
    chainedPlans[2].release_cycle = 800;
    chainedPlans[2].bytes[0] = 128;
    chainedPlans[2].regions = {{0, 1, 28, 0, 8, false}};
    schedule_weight_prefetches(chainedOverlapProgram, chainedPlans);
    if (!chainedPlans[0].pre_execution
        || chainedPlans[1].pre_execution
        || chainedPlans[2].pre_execution
        || chainedPlans[1].start_cycle != 300
        || chainedPlans[2].start_cycle != 600)
        throw std::runtime_error(
            "chained weight-region reuse was scheduled before release");

    std::vector<WeightPrefetchPlan> sharedQueuePlans(3);
    sharedQueuePlans[0].ready_cycle = 100;
    sharedQueuePlans[0].release_cycle = 200;
    sharedQueuePlans[0].bytes[0] = 128;
    sharedQueuePlans[0].regions = {{1, 1, 20, 100, 108, false}};
    sharedQueuePlans[1].ready_cycle = 300;
    sharedQueuePlans[1].release_cycle = 600;
    sharedQueuePlans[1].bytes[0] = 128;
    sharedQueuePlans[1].regions = {{1, 1, 20, 0, 8, false}};
    sharedQueuePlans[2].ready_cycle = 700;
    sharedQueuePlans[2].release_cycle = 800;
    sharedQueuePlans[2].bytes[0] = 128;
    sharedQueuePlans[2].regions = {{1, 1, 20, 100, 108, false}};
    schedule_weight_prefetches(chainedOverlapProgram, sharedQueuePlans);
    if (!sharedQueuePlans[0].pre_execution
        || !sharedQueuePlans[1].pre_execution
        || sharedQueuePlans[2].pre_execution
        || sharedQueuePlans[2].start_cycle != 200)
        throw std::runtime_error(
            "disjoint-row C2C write waited for an unrelated MEM row");

    std::vector<WeightPrefetchPlan> earlyAlternateBankPlans(4);
    earlyAlternateBankPlans[0].ready_cycle = 10;
    earlyAlternateBankPlans[0].release_cycle = 300;
    earlyAlternateBankPlans[0].bytes[0] = 128;
    earlyAlternateBankPlans[0].regions = {{1, 1, 20, 0, 8, false}};
    earlyAlternateBankPlans[1].ready_cycle = 20;
    earlyAlternateBankPlans[1].release_cycle = 100;
    earlyAlternateBankPlans[1].bytes[0] = 128;
    earlyAlternateBankPlans[1].regions = {{0, 1, 20, 0, 8, false}};
    earlyAlternateBankPlans[2].ready_cycle = 400;
    earlyAlternateBankPlans[2].release_cycle = 500;
    earlyAlternateBankPlans[2].bytes[0] = 128;
    earlyAlternateBankPlans[2].regions = {{1, 1, 20, 0, 8, false}};
    earlyAlternateBankPlans[3].ready_cycle = 500;
    earlyAlternateBankPlans[3].release_cycle = 600;
    earlyAlternateBankPlans[3].bytes[0] = 128;
    earlyAlternateBankPlans[3].regions = {{0, 1, 20, 0, 8, false}};
    schedule_weight_prefetches(
        chainedOverlapProgram, earlyAlternateBankPlans);
    if (earlyAlternateBankPlans[2].pre_execution
        || earlyAlternateBankPlans[3].pre_execution
        || earlyAlternateBankPlans[2].start_cycle != 300
        || earlyAlternateBankPlans[3].start_cycle
            < earlyAlternateBankPlans[2].transfer_end_cycle)
        throw std::runtime_error(
            "later alternate-bank page bypassed its earlier consumer");

    std::vector<WeightPrefetchPlan> sharedTransportPlans(4);
    sharedTransportPlans[0].ready_cycle = 100;
    sharedTransportPlans[0].release_cycle = 300;
    sharedTransportPlans[0].bytes[0] = 128;
    sharedTransportPlans[0].regions = {{1, 1, 20, 0, 8, false}};
    sharedTransportPlans[1].ready_cycle = 110;
    sharedTransportPlans[1].release_cycle = 300;
    sharedTransportPlans[1].bytes[0] = 128;
    sharedTransportPlans[1].regions = {{1, 1, 28, 0, 8, false}};
    sharedTransportPlans[2].ready_cycle = 400;
    sharedTransportPlans[2].release_cycle = 500;
    sharedTransportPlans[2].bytes[0] = 128;
    sharedTransportPlans[2].regions = {{1, 1, 20, 0, 8, false}};
    sharedTransportPlans[3].ready_cycle = 410;
    sharedTransportPlans[3].release_cycle = 510;
    sharedTransportPlans[3].bytes[0] = 128;
    sharedTransportPlans[3].regions = {{1, 1, 28, 0, 8, false}};
    schedule_weight_prefetches(
        chainedOverlapProgram, sharedTransportPlans);
    if (sharedTransportPlans[2].start_cycle != 300
        || sharedTransportPlans[3].start_cycle
            < sharedTransportPlans[2].transfer_end_cycle)
        throw std::runtime_error(
            "different C2C coarse instructions overlapped one ICU queue");

    auto slowSharedTransportPlans = sharedTransportPlans;
    schedule_weight_prefetches(chainedOverlapProgram,
        slowSharedTransportPlans, lowBandwidthHardware);
    for (std::size_t index = 0; index < sharedTransportPlans.size(); ++index)
        if (sharedTransportPlans[index].start_cycle
                != slowSharedTransportPlans[index].start_cycle
            || sharedTransportPlans[index].transfer_end_cycle
                != slowSharedTransportPlans[index].transfer_end_cycle)
            throw std::runtime_error("DDR prediction serialized later C2C pages");

    // Consumption order wins over SRAM availability for both single-page
    // and multi-page bindings. The earlier consumer reuses SRAM at 150,
    // while the later consumer's bank is already free at 100.
    BinaryProgram projectionPriorityProgram = chainedOverlapProgram;
    projectionPriorityProgram.bindings = {make_weight_binding(), make_weight_binding()};
    projectionPriorityProgram.bindings[0].page_count = 2;
    projectionPriorityProgram.bindings[1].index = 1;
    projectionPriorityProgram.weight_page_uses = {
        {0, 0, 0, 1000, 1100}, {1, 0, 1, 250, 300}};
    std::vector<WeightPrefetchPlan> priorityPlans(4);
    priorityPlans[0].ready_cycle = 10;
    priorityPlans[0].release_cycle = 100;
    priorityPlans[0].bytes[0] = 128;
    priorityPlans[0].regions = {{0, 1, 20, 0, 8, false}};
    priorityPlans[1].ready_cycle = 20;
    priorityPlans[1].release_cycle = 150;
    priorityPlans[1].bytes[0] = 128;
    priorityPlans[1].regions = {{1, 1, 20, 0, 8, false}};
    priorityPlans[2].ready_cycle = 1000;
    priorityPlans[2].release_cycle = 1100;
    priorityPlans[2].bytes[0] = 131072;
    priorityPlans[2].regions = priorityPlans[0].regions;
    priorityPlans[2].use_indices = {0};
    priorityPlans[3].ready_cycle = 250;
    priorityPlans[3].release_cycle = 300;
    priorityPlans[3].bytes[0] = 128;
    priorityPlans[3].regions = priorityPlans[1].regions;
    for (const auto pageCount : {1u, 2u}) {
        projectionPriorityProgram.bindings[1].page_count = pageCount;
        schedule_weight_prefetches(projectionPriorityProgram, priorityPlans);
        if (priorityPlans[2].ready_cycle != 250
            || priorityPlans[2].start_cycle != 150
            || priorityPlans[3].ready_cycle != 1000
            || priorityPlans[3].start_cycle < priorityPlans[2].transfer_end_cycle
            || priorityPlans[3].earliest_mem_cycle < priorityPlans[2].transfer_end_cycle)
            throw std::runtime_error("page consumption order was changed");
    }

    // A free late Up bank must not be preloaded before the first Gate page
    // whose bank still holds attention weights. Gate0 -> Gate1 -> Up0 also
    // remains the DMA order when Gate1's SRAM becomes reusable first.
    std::vector<WeightPrefetchPlan> prefixPlans(4);
    prefixPlans[0].ready_cycle = 10;
    prefixPlans[0].release_cycle = 300;
    prefixPlans[0].bytes[0] = 128;
    prefixPlans[0].regions = {{0, 1, 20, 0, 8, false}};
    prefixPlans[1].ready_cycle = 400;
    prefixPlans[1].release_cycle = 500;
    prefixPlans[1].bytes[0] = 128;
    prefixPlans[1].regions = prefixPlans[0].regions;
    prefixPlans[2].ready_cycle = 600;
    prefixPlans[2].release_cycle = 700;
    prefixPlans[2].bytes[0] = 128;
    prefixPlans[2].regions = {{1, 1, 28, 0, 8, false}};
    prefixPlans[3].ready_cycle = 800;
    prefixPlans[3].release_cycle = 900;
    prefixPlans[3].bytes[0] = 128;
    prefixPlans[3].regions = {{1, 1, 36, 0, 8, false}};
    schedule_weight_prefetches(chainedOverlapProgram, prefixPlans);
    if (!prefixPlans[0].pre_execution || prefixPlans[1].pre_execution
        || prefixPlans[2].pre_execution || prefixPlans[3].pre_execution
        || prefixPlans[1].start_cycle != 300
        || prefixPlans[2].start_cycle < prefixPlans[1].transfer_end_cycle
        || prefixPlans[3].start_cycle < prefixPlans[2].transfer_end_cycle)
        throw std::runtime_error("late free-bank page bypassed the preload prefix");

    BinaryProgram nativeWeightProgram;
    QueueProgram nativeMem {QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 20, 0), {}};
    nativeMem.commands.push_back(encode_icu_control_raw_word(
        IcuControlInstruction::Nop(40)));
    const auto nativePacket = isa::encode_mem_icu_shortest_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D {0, {16, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(100, {1, 0, 0}), StreamId::East(0)));
    for (std::size_t w = 0; w < isa::fu_loop_packet_word_count(nativePacket); ++w) {
        const auto& word = nativePacket.words[w];
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = InstructionKind::Mem;
        command.word_count = isa::EncodedMemIcu3DPacket::kLanesPerWord;
        for (std::size_t lane = 0; lane < command.word_count; ++lane)
            command.words[lane] = word.lanes[lane];
        nativeMem.commands.push_back(command);
    }
    nativeWeightProgram.queues.push_back(std::move(nativeMem));
    std::vector<WeightPrefetchPlan> nativePlans(1);
    nativePlans[0].runtime_prefetch = true;
    nativePlans[0].ready_cycle = 100;
    nativePlans[0].release_cycle = 120;
    nativePlans[0].bytes[0] = 128;
    nativePlans[0].regions = {{0, 1, 20, 100, 108, false}};
    schedule_weight_prefetches(nativeWeightProgram, nativePlans);
    if (nativePlans[0].earliest_mem_cycle != 0)
        throw std::runtime_error("ordinary native weight reads deferred prefetch");
    BinaryBinding scratch;
    scratch.access = BindingAccess::Internal;
    scratch.bank = 0;
    scratch.hemisphere_mask = 1;
    scratch.slices = {20};
    nativeWeightProgram.bindings.push_back(scratch);
    schedule_weight_prefetches(nativeWeightProgram, nativePlans);
    if (nativePlans[0].earliest_mem_cycle != 56)
        throw std::runtime_error("native scratch release did not protect page reuse");

    // A compiler SYNC may consume an idle slot in one place and add an
    // extra issue slot at a zero-wait boundary in another. Rebinding must
    // recover both original phases, rather than turn every old SYNC into NOP.
    auto beforeSync = nativeWeightProgram;
    const auto secondPacket = isa::encode_mem_icu_shortest_instruction(
        MemIcuInstruction::Read3D(IcuLoop3D{0, {4, 1, 1}, {1, 1, 1}},
            MemIcuAddress3D::Affine(200, {1, 0, 0}), StreamId::East(0)));
    const auto appendPacket = [](auto& commands, const auto& packet) {
        for (std::size_t w = 0; w < isa::fu_loop_packet_word_count(packet); ++w) {
        const auto& word = packet.words[w];
            QueueCommand command;
            command.command = word.lanes[0];
            command.instruction_kind = InstructionKind::Mem;
            command.word_count = isa::EncodedMemIcu3DPacket::kLanesPerWord;
            for (std::size_t lane = 0; lane < command.word_count; ++lane)
                command.words[lane] = word.lanes[lane];
            commands.push_back(command);
        }
    };
    appendPacket(beforeSync.queues[0].commands, secondPacket);
    auto withSync = beforeSync;
    auto& syncCommands = withSync.queues[0].commands;
    syncCommands.clear();
    syncCommands.push_back(encode_icu_control_raw_word(IcuControlInstruction::Nop(39)));
    syncCommands.push_back(encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0x30000)));
    appendPacket(syncCommands, nativePacket);
    syncCommands.push_back(encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0x30001)));
    appendPacket(syncCommands, secondPacket);
    BinaryAddressRelocation phaseRelocation;
    phaseRelocation.queue_kind = QueueKind::Mem;
    phaseRelocation.queue_index = withSync.queues[0].index;
    phaseRelocation.command_index = 2;
    withSync.address_relocations.push_back(phaseRelocation);
    record_compiler_sync_origins(withSync, beforeSync);
    const auto pageOut = encode_mem_synchronized_icu_packet(
        InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(4, 1, 0, 1,
            MemInstruction::Read(300, StreamId::East(0)), 4));
    syncCommands.insert(syncCommands.end(), pageOut.begin(), pageOut.end());
    std::ostringstream phaseBinary(std::ios::out | std::ios::binary);
    write_binary_program(withSync, phaseBinary);
    std::istringstream phaseInput(phaseBinary.str(), std::ios::in | std::ios::binary);
    const auto restoredPhases = without_compiled_mem_write_sync(read_binary_program(phaseInput));
    const auto& restoredQueue = restoredPhases.queues[0];
    if (decode_fu_3d_raw_packet_loop(restoredQueue, 0).wait_cycle != 40
        || decode_fu_3d_raw_packet_loop(restoredQueue,
            fu_3d_raw_packet_word_count(QueueKind::Mem, restoredQueue.commands[0])).wait_cycle != 0
        || !is_mem_synchronized_raw_packet_header(restoredQueue.commands[
            2 * fu_3d_raw_packet_word_count(QueueKind::Mem, restoredQueue.commands[0])])
        || restoredPhases.address_relocations[0].command_index != 0)
        throw std::runtime_error("compiler SYNC rebind changed domain timing or relocation");

    // A long recurrent phase can leave an unrelated queue idle beyond the
    // 24-bit domain wait field. Restoring compiler timing must preserve the
    // complete gap and relocate the packet past the required leading NOP.
    auto longSync = withSync;
    constexpr std::uint64_t longGap = std::uint64_t{1} << 24;
    std::erase_if(longSync.timelines, [](const auto& timeline) {
        return timeline.name.starts_with("icu.sync_origin.");
    });
    const auto longOriginPrefix = "icu.sync_origin."
        + std::to_string(static_cast<unsigned>(longSync.queues[0].kind)) + "."
        + std::to_string(longSync.queues[0].index) + ".";
    longSync.timelines.push_back({longOriginPrefix + "0", longGap + 40, longGap + 56});
    longSync.timelines.push_back({longOriginPrefix + "1", longGap + 56, longGap + 60});
    longSync.timelines.push_back({longOriginPrefix + "end", longGap + 60, longGap + 61});
    const auto restoredLong = without_compiled_mem_write_sync(longSync);
    const auto& longQueue = restoredLong.queues[0];
    if (decode_icu_control_raw_word(longQueue.commands[0]).count != 41
        || decode_fu_3d_raw_packet_loop(longQueue, 1).wait_cycle != longGap - 1
        || restoredLong.address_relocations[0].command_index != 1)
        throw std::runtime_error("long compiler SYNC wait lost timing or relocation");

    BinaryProgram wideningStreamProgram = overlappingProgram;
    wideningStreamProgram.stream_release_cycles.assign(
        wideningStreamProgram.hardware.encoded_streams, 400);
    for (std::size_t stream = 28; stream < 32; ++stream)
        wideningStreamProgram.stream_release_cycles[westBase + stream] = 0;
    for (std::size_t stream = 24; stream < 28; ++stream)
        wideningStreamProgram.stream_release_cycles[westBase + stream] = 8;
    std::vector<WeightPrefetchPlan> wideningPlans(2);
    wideningPlans[0].ready_cycle = 100;
    wideningPlans[0].release_cycle = 0;
    wideningPlans[0].bytes[0] = 128;
    wideningPlans[0].regions = {{1, 1, 20, 0, 8, false}};
    wideningPlans[1].ready_cycle = 1000;
    wideningPlans[1].release_cycle = 1100;
    wideningPlans[1].bytes[0] = 8192;
    wideningPlans[1].regions = {{1, 1, 20, 0, 8, false}};
    schedule_weight_prefetches(wideningStreamProgram, wideningPlans);
    if (wideningPlans[1].start_cycle != 8
        || wideningPlans[1].fabric_streams
            != std::vector<std::uint16_t>(
                {31, 30, 29, 28, 27, 26, 25, 24}))
        throw std::runtime_error(
            "C2C planner did not wait briefly for the faster wide stream set");

    // Future SR use cannot delay a page past its consumer when the compiler
    // explicitly reserved an exclusive ingress window before that consumer.
    BinaryProgram windowProgram;
    windowProgram.stream_release_cycles.assign(
        windowProgram.hardware.encoded_streams, 1000000);
    windowProgram.weight_page_uses.push_back({0, 0, 0, 1000, 1100, true});
    windowProgram.timelines.push_back({"c2c.window.binding:0.page:0", 100, 900});
    std::vector<WeightPrefetchPlan> windowPlans(1);
    windowPlans[0].use_indices = {0};
    windowPlans[0].runtime_prefetch = true;
    windowPlans[0].ready_cycle = 1000;
    windowPlans[0].release_cycle = 1100;
    windowPlans[0].bytes[0] = 128;
    windowPlans[0].regions = {{0, 1, 20, 0, 8, false}};
    schedule_weight_prefetches(windowProgram, windowPlans);
    if (windowPlans[0].start_cycle != 100
        || windowPlans[0].earliest_mem_cycle != 100
        || windowPlans[0].fabric_streams.size()
            != windowProgram.hardware.c2c_streams_per_direction)
        throw std::runtime_error("compiler C2C ingress reservation was ignored");

    // Physical transfer time must not advance the common computation
    // coordinate. Both disjoint pages enter at the same proven stop even
    // when the first page takes longer than the entire logical program.
    auto commonWindowProgram = windowProgram;
    commonWindowProgram.common_barrier_protocol = true;
    commonWindowProgram.weight_page_uses.push_back({1, 0, 0, 1000, 1100, true});
    commonWindowProgram.timelines.push_back({"c2c.window.binding:1.page:0", 100, 900});
    auto commonWindowPlans = windowPlans;
    commonWindowPlans[0].bytes[0] = 1'000'000;
    auto secondCommonPlan = commonWindowPlans[0];
    secondCommonPlan.use_indices = {1};
    secondCommonPlan.regions = {{0, 1, 21, 0, 8, false}};
    commonWindowPlans.push_back(secondCommonPlan);
    schedule_weight_prefetches(commonWindowProgram, commonWindowPlans);
    if (commonWindowPlans[0].earliest_mem_cycle != 100
        || commonWindowPlans[1].earliest_mem_cycle != 100
        || commonWindowPlans[1].start_cycle != 100)
        throw std::runtime_error("physical paging duration consumed common logical time");

    BinaryProgram program;
    program.bindings.push_back(make_weight_binding());
    program.weight_page_uses.push_back({0, 0, 0, 200, 220});
    program.max_cycle = 512;
    program.stream_release_cycles.assign(
        program.hardware.encoded_streams, program.max_cycle + 1);
    const auto westBegin = program.stream_release_cycles.begin()
        + program.hardware.streams_per_direction;
    std::fill(westBegin + 16, westBegin + 24, 0);

    std::vector<std::uint8_t> logical(program.bindings[0].byte_size);
    for (std::size_t index = 0; index < logical.size(); ++index)
        logical[index] = static_cast<std::uint8_t>(index * 17 + 3);
    std::vector<std::uint8_t> nextLogical(program.bindings[0].byte_size);
    for (std::size_t index = 0; index < nextLogical.size(); ++index)
        nextLogical[index] = static_cast<std::uint8_t>(index * 29 + 11);
    const PackedWeightImage expected = pack_weight_binding_page(
        program.bindings[0], 0, logical, program.hardware);
    BinaryProgram nextProgram = program;
    nextProgram.bindings[0] = make_weight_binding(1);
    nextProgram.weight_page_uses[0].bank = 1;
    const PackedWeightImage nextExpected = pack_weight_binding_page(
        nextProgram.bindings[0], 0, nextLogical, nextProgram.hardware);

    ModelPackage package;
    package.model_name = "executable-c2c-weight";
    package.architecture = "test";
    package.tensors.push_back(ModelTensor {
        "q_proj.weight", BindingElementType::I8, {32, 128}, logical});
    package.executables.push_back({"projection.0", program, {}});
    package.executables.push_back({"projection.1", nextProgram, {}});
    package.invocations.push_back(ModelInvocation {
        "projection.0", 0, {{0, "q_proj.weight"}}, {}, {}});
    package.tensors.push_back(ModelTensor {
        "q_proj.next.weight", BindingElementType::I8, {32, 128},
        nextLogical});
    package.invocations.push_back(ModelInvocation {
        "projection.1", 1, {{0, "q_proj.next.weight"}}, {}, {}});

    std::vector<std::uint8_t> nextModelPageData(hw::kPhysicalVectorBytes);
    for (std::size_t index = 0; index < nextModelPageData.size(); ++index)
        nextModelPageData[index] = static_cast<std::uint8_t>(index + 1);
    ModelTensor nextModelPageTensor {
        "layer.1.gamma.sram_page.1", BindingElementType::I8,
        {nextModelPageData.size()}, nextModelPageData};
    nextModelPageTensor.encoding =
        ModelTensorEncoding::TargetPackedSramVectors;
    package.tensors.push_back(std::move(nextModelPageTensor));
    ModelWeightPage nextModelPage;
    nextModelPage.layer = 1;
    nextModelPage.bank = 1;
    nextModelPage.tensors = {"layer.1.gamma.sram_page.1"};
    nextModelPage.segments.push_back(ModelWeightPage::Segment {
        "layer.1.gamma.sram_page.1", 0, 0, 17, 8191, 1, 0});
    package.weight_pages.push_back(std::move(nextModelPage));
    package.invocations.back().weight_page = 0;

    C2cDmaSystem system;
    ModelSession session(system);
    session.set_ddr_peak_bandwidth_mbytes_per_second(12800);
    session.load(std::move(package));
    if (system.ddr4().config().peak_bandwidth_bytes_per_second
        != 12'800'000'000ULL)
        throw std::runtime_error(
            "ModelSession did not apply the runtime DDR bandwidth");
    session.run(0);

    const auto& stats = session.stats();
    if (stats.weight_page_prefetches != 3
        || stats.weight_page_prefetch_bytes
            != expected.data.size() + nextExpected.data.size()
                + nextModelPageData.size()
        || stats.weight_page_initial_wait_cycles == 0
        || stats.weight_page_boundary_wait_cycles != 0
        || stats.weight_page_hidden_prefetches != 2)
        throw std::runtime_error(
            "next executable page was not hidden by C2C lookahead: "
            "prefetches=" + std::to_string(stats.weight_page_prefetches)
            + " bytes=" + std::to_string(stats.weight_page_prefetch_bytes)
            + " initial_wait="
            + std::to_string(stats.weight_page_initial_wait_cycles)
            + " boundary_wait="
            + std::to_string(stats.weight_page_boundary_wait_cycles)
            + " hidden="
            + std::to_string(stats.weight_page_hidden_prefetches));

    const auto verify = [&](const PackedWeightImage& image,
                            std::size_t bank) {
      for (const PackedWeightSegment& segment : image.segments) {
        for (std::uint32_t row = 0; row < segment.vector_count; ++row) {
            const std::size_t offset =
                static_cast<std::size_t>(segment.byte_offset)
                + static_cast<std::size_t>(row) * hw::kPhysicalVectorBytes;
            for (std::size_t column = 0;
                 column < hw::kPhysicalVectorBytes; ++column) {
                const auto actual = system.chip().read_mem_sram_lane_byte(
                    static_cast<Hemisphere>(segment.hemisphere),
                    segment.slice, bank, column / hw::kLanesPerTile,
                    segment.base_row + row, column % hw::kLanesPerTile);
                if (actual != image.data[offset + column])
                    throw std::runtime_error(
                        "C2C page payload does not match packed weight SRAM");
            }
        }
      }
    };
    verify(expected, 0);
    verify(nextExpected, 1);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto actual = system.chip().read_mem_sram_lane_byte(
                Hemisphere::East, 17, 1, tile, 8191, lane);
            const auto expectedByte =
                nextModelPageData[tile * hw::kLanesPerTile + lane];
            if (actual != expectedByte)
                throw std::runtime_error(
                    "lookahead model page became ready before SRAM commit");
        }

    {
        // Exercise common INIT ordering through the actual session: one
        // preloaded bank plus a second bank filled while computation holds.
        BinaryProgram commonProgram;
        commonProgram.common_barrier_protocol = true;
        commonProgram.max_cycle = 512;
        auto startup = make_weight_binding(0);
        auto refill = make_weight_binding(1);
        refill.index = 1;
        refill.name = "next.weight";
        commonProgram.bindings = {startup, refill};
        commonProgram.weight_page_uses = {{0, 0, 0, 128, 256, false}, {1, 0, 1, 256, 512, true}};
        commonProgram.timelines = {{"c2c.window.binding:1.page:0", 128, 256}};
        ModelPackage commonPackage;
        commonPackage.model_name = "common-bootstrap-and-refill";
        commonPackage.architecture = "test";
        commonPackage.tensors = {{startup.name, BindingElementType::I8, {32, 128}, logical},
            {refill.name, BindingElementType::I8, {32, 128}, nextLogical}};
        commonPackage.executables.push_back({"main", commonProgram, {}});
        commonPackage.invocations.push_back({"main", 0, {{0, startup.name}, {1, refill.name}}, {}, {}});
        auto commonSystem = std::make_unique<C2cDmaSystem>();
        ModelSession commonSession(*commonSystem);
        commonSession.load(std::move(commonPackage));
        commonSession.run(0);
        const auto& commonStats = commonSession.stats();
        const auto& linked = commonSession.last_linked_program();
        if (commonStats.weight_page_prefetches != 2 || commonStats.weight_page_initial_wait_cycles == 0
            || commonStats.weight_page_runtime_wait_cycles == 0 || linked.barriers.size() != 1
            || linked.barriers.front().stop_cycle != 256 || linked.transport_queues.empty())
            throw std::runtime_error("common session did not separate startup transport and logical refill waiting");
        for (const auto& [weight, bytes] : std::array{std::pair{startup, logical}, std::pair{refill, nextLogical}}) {
            const auto packed = pack_weight_binding_page(weight, 0, bytes, commonProgram.hardware);
            for (const auto& segment : packed.segments)
                for (std::size_t row = 0; row < segment.vector_count; ++row)
                    for (std::size_t byte = 0; byte < hw::kPhysicalVectorBytes; ++byte)
                        if (commonSystem->chip().read_mem_sram_lane_byte(static_cast<Hemisphere>(segment.hemisphere),
                            segment.slice, weight.bank, byte / hw::kLanesPerTile, segment.base_row + row,
                            byte % hw::kLanesPerTile) != packed.data[segment.byte_offset + row * hw::kPhysicalVectorBytes + byte])
                            throw std::runtime_error("common session resumed before its initial/refill SRAM contents committed");
        }
    }

    std::cout << "model_session_executable_c2c_weight_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "model_session_executable_c2c_weight_test failed: "
              << error.what() << '\n';
    return 1;
}
