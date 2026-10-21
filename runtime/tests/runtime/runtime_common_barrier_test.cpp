#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace ftlpu;
using namespace ftlpu::software::runtime;
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function&& function, const char* message)
{
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
BinaryBinding binding()
{
    BinaryBinding value;
    value.index = 0;
    value.access = BindingAccess::Input;
    value.element_type = BindingElementType::I8;
    value.layout = BindingLayout::W8A16AttentionWeightStriped;
    value.byte_size = 32 * 128;
    value.base_row = 0;
    value.instruction_count = 8;
    value.address_stride = 1;
    value.shape = {32, 128};
    value.slices = {20, 21, 22, 23, 24, 25, 26, 27};
    value.role = "weight";
    value.name = "q_proj.weight";
    value.hemisphere_mask = 3;
    value.bank = 1;
    value.paged_weight = true;
    value.page_count = 1;
    value.page_rows = 8;
    value.page_granularity = 1;
    value.page_role_group_count = 1;
    value.page_items_per_slice_group = 1;
    value.page_bank_count = 2;
    value.page_storage_slices = value.slices;
    return value;
}
constexpr std::size_t kStop = 128;
constexpr std::size_t kBudget = 516;
constexpr std::uint64_t kDdrBase = 0x10000;
constexpr std::size_t macro_queue()
{
    return InstructionControlUnit::mem_queue(Hemisphere::East, 0, 0);
}
PackedWeightImage image()
{
    std::vector<std::uint8_t> values(binding().byte_size);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<std::uint8_t>(index * 13 + 19);
    return pack_weight_binding_page(binding(), 0, values, ExecutableHardwareConfig{});
}
BinaryProgram linked_program(C2cDmaSystem& system, const PackedWeightImage& packed)
{
    BinaryProgram program;
    program.common_barrier_protocol = true;
    program.max_cycle = kBudget;
    program.bindings = {binding()};
    program.weight_page_uses = {{0, 0, 1, kStop, kBudget, true}};
    program.timelines = {{"c2c.window.binding:0.page:0", 32, kStop}};
    const auto encoded = isa::encode_mem_instruction(MemInstruction::Read(600, StreamId::East(0)));
    QueueCommand read;
    read.command = static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);
    read.instruction_kind = InstructionKind::Mem;
    read.word_count = 2;
    read.words[0] = static_cast<std::uint32_t>(encoded);
    read.words[1] = static_cast<std::uint32_t>(encoded >> 32);
    program.queues.push_back({QueueKind::Mem, macro_queue(),
        {encode_macro_schedule_command(read,
            IcuMacroSchedule {0, 1, 1, 0, 3, 256, 1, IcuInductionTarget::MemAddress})}});
    program.queues.back().requires_raw_macro = true;
    // A Native computation queue must retain its NOP execution state through
    // the same wait and keep the original logical phase.
    const auto nativeRead = [&](std::size_t address) {
        auto command = read;
        const auto bits = isa::encode_mem_instruction(MemInstruction::Read(address, StreamId::East(1)));
        command.words[0] = static_cast<std::uint32_t>(bits);
        command.words[1] = static_cast<std::uint32_t>(bits >> 32);
        return command;
    };
    program.queues.push_back({QueueKind::Mem,
        InstructionControlUnit::mem_queue(Hemisphere::East, 1, 0),
        {encode_icu_control_raw_word(IcuControlInstruction::Nop(32)), nativeRead(700),
         encode_icu_control_raw_word(IcuControlInstruction::Nop(223)), nativeRead(701),
         encode_icu_control_raw_word(IcuControlInstruction::Nop(256)), nativeRead(702)}});
    C2cWeightPage page;
    page.bank = 1;
    for (const auto& segment : packed.segments)
        page.segments.push_back({static_cast<Hemisphere>(segment.hemisphere),
            segment.slice, 1, segment.base_row, 0,
            kDdrBase + segment.byte_offset, segment.vector_count});
    C2cWeightPager pager(system);
    pager.begin_schedule(program);
    const std::array<std::size_t, 1> instances {0};
    static_cast<void>(pager.schedule(program, page, 32, 48, kStop,
        0x10000, 0x30000, 32, instances));
    pager.finalize_schedule(program);
    // Execute the actual linked container, including launch mapping and the
    // independent MEM control area, after a full writer/reader round trip.
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    write_binary_program(program, stream);
    stream.seekg(0);
    return read_binary_program(stream);
}
void initialize_ddr(C2cDmaSystem& system, const PackedWeightImage& packed)
{
    for (std::size_t offset = 0; offset < packed.data.size(); offset += hw::kPhysicalVectorBytes) {
        C2cVector vector;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                vector.payload[tile][lane] = packed.data[offset + tile * hw::kLanesPerTile + lane];
        system.ddr4().initialize_vector(kDdrBase + offset, vector);
    }
}
void verify_commit(C2cDmaSystem& system, const PackedWeightImage& packed)
{
    for (const auto& segment : packed.segments)
        for (std::size_t row = 0; row < segment.vector_count; ++row)
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    const auto actual = system.chip().read_mem_sram_lane_byte(
                        static_cast<Hemisphere>(segment.hemisphere), segment.slice,
                        1, tile, segment.base_row + row, lane);
                    const auto expected = packed.data[segment.byte_offset
                        + row * hw::kPhysicalVectorBytes + tile * hw::kLanesPerTile + lane];
                    require(actual == expected, "computation resumed before the page committed to visible SRAM");
                }
}
void reject_corrupt_transport(CModelRuntime& runtime, const BinaryProgram& program)
{
    // These fields remain individually legal packet encodings. The loader
    // must reject their broken DMA -> RX -> MEM -> page-instance association.
    for (unsigned mutation = 0; mutation < 10; ++mutation) {
        auto bad = program;
        if (mutation < 4) {
            auto& queue = bad.transport_queues.front();
            const auto packet = decode_mem_synchronized_icu_packet(queue, 0);
            const auto& header = packet[0].lanes;
            const auto& native = packet[1].lanes;
            const auto count = ((header[0] >> 2) & 0xffffU) + 1;
            auto tag = ((header[0] >> 18) | (header[1] << 14)) & 0xffffU;
            const auto delay = (header[1] >> 2) & 0xffffU;
            const auto reservation = (header[2] & 0xffffffU) + 1;
            const auto bits = ((static_cast<isa::EncodedMemInstruction>(native[0])
                | (static_cast<isa::EncodedMemInstruction>(native[1]) << 32)) >> 2)
                | (static_cast<isa::EncodedMemInstruction>(native[2]) << 62);
            auto instruction = isa::decode_mem_instruction(bits);
            auto stride = std::int64_t{1};
            if (mutation == 0) stride = 0;
            if (mutation == 1) ++tag;
            if (mutation == 2) instruction.stream = StreamId::East(0).packed();
            if (mutation == 3) instruction.preserve_stream = true;
            const auto changed = encode_mem_synchronized_icu_packet(
                InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(
                    count, tag, delay, stride, instruction, reservation));
            for (std::size_t word = 0; word < changed.size(); ++word)
                queue.commands[word] = changed[word];
        } else if (mutation < 7) {
            auto queue = std::find_if(bad.queues.begin(), bad.queues.end(),
                [](const auto& value) { return value.kind == QueueKind::C2cDma; });
            require(queue != bad.queues.end(), "test has no real DMA control queue");
            auto dma = C2cIcuPacketCodec::decode_dma(decode_c2c_dma_icu_packet(*queue, 1));
            if (mutation == 4) dma.address_stride_bytes *= 2;
            if (mutation == 5) ++dma.vector_count;
            if (mutation == 6) dma.sync_tag += 100;
            const auto changed = encode_c2c_dma_icu_packet(C2cIcuPacketCodec::encode(dma));
            for (std::size_t word = 0; word < changed.size(); ++word)
                queue->commands[1 + word] = changed[word];
        } else {
            auto queue = std::find_if(bad.queues.begin(), bad.queues.end(),
                [](const auto& value) { return value.kind == QueueKind::C2cRx; });
            require(queue != bad.queues.end(), "test has no real RX control queue");
            if (mutation == 9) {
                queue->commands[0] = encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(0x10001));
            } else {
                auto rx = C2cIcuPacketCodec::decode_rx(decode_c2c_raw_word(queue->commands[1], InstructionKind::C2cEndpoint));
                if (mutation == 7) rx.notify.mem_slice = (rx.notify.mem_slice + 1) % hw::kMemSliceColumns;
                if (mutation == 8) rx.sync_tag += 100;
                queue->commands[1] = encode_c2c_endpoint_icu_packet(C2cIcuPacketCodec::encode(rx));
            }
        }
        rejects([&] { runtime.load(bad); }, "loader accepted a corrupt physical page completion chain");
    }
}
struct Result {
    std::size_t logical{}, physical{};
    std::vector<std::pair<std::size_t, std::size_t>> issues{}, native_issues{};
};
Result run(std::size_t latency, bool cycleZero = false)
{
    Ddr4Config config;
    config.read_latency_cycles = latency;
    config.read_latency_jitter_cycles = 0;
    config.write_latency_jitter_cycles = 0;
    auto system = std::make_unique<C2cDmaSystem>(config);
    const auto packed = image();
    auto program = linked_program(*system, packed);
    if (cycleZero) program.barriers.front().stop_cycle = 0;
    if (latency == 3 && !cycleZero)
        if (const char* output = std::getenv("FTLPU_COMMON_TEST_LINKED_PROGRAM"))
            write_binary_program(program, std::filesystem::path(output));
    Result result;
    CModelRuntime* current = nullptr;
    bool checked_resumption = false;
    CModelRuntime runtime(*system, [&](TspSliceSystem::LogSinks sinks) {
        system->tick(sinks);
        auto& icu = system->chip().icu();
        if (icu.compute_enabled() && current->logical_cycles() >= (cycleZero ? 0 : kStop) && !checked_resumption) {
            verify_commit(*system, packed);
            checked_resumption = true;
        }
        auto& queue = icu.mem_iq(macro_queue());
        if (icu.compute_enabled() && queue.last_trace().action == IcuQueueAction::MacroIssue) {
            require(queue.last_dispatched().has_value(), "Macro trace lost its dispatched instruction");
            result.issues.emplace_back(current->logical_cycles(), queue.last_dispatched()->address);
        }
        auto& native = icu.mem_iq(InstructionControlUnit::mem_queue(Hemisphere::East, 1, 0));
        if (icu.compute_enabled() && native.last_trace().action == IcuQueueAction::FunctionalIssue) {
            require(native.last_dispatched().has_value(), "Native trace lost its dispatched instruction");
            result.native_issues.emplace_back(current->logical_cycles(), native.last_dispatched()->address);
        }
    });
    current = &runtime;
    // Reload changes the generation before any transfer may launch. A stale
    // task's completion must not release the new task's same numeric event.
    runtime.load(program);
    const auto old_generation = runtime.execution_generation();
    runtime.load(program);
    require(runtime.execution_generation() != old_generation, "reload reused execution generation");
    runtime.notify_common_page_commit(old_generation, 0);
    initialize_ddr(*system, packed);
    runtime.run_cycles(kBudget);
    require(checked_resumption, "test never resumed computation after its common stop");
    require(runtime.logical_cycles() == kBudget, "physical waiting consumed the requested logical budget");
    require(runtime.physical_cycles() > runtime.logical_cycles(), "stale completion released the new task or real transport never waited");
    require(system->chip().icu().mem_iq(macro_queue()).done(), "Macro context did not finish across the stop");
    result.logical = runtime.logical_cycles();
    result.physical = runtime.physical_cycles();
    verify_commit(*system, packed);
    auto missing = program;
    missing.barriers.clear();
    missing.barrier_events.clear();
    rejects([&] { runtime.load(missing); }, "loader accepted a paging candidate without a final common table");
    auto no_transport = program;
    no_transport.transport_queues.clear();
    rejects([&] { runtime.load(no_transport); }, "loader accepted barrier events without their actual MEM commits");
    if (latency == 3 && !cycleZero) {
        reject_corrupt_transport(runtime, program);
        // All transient resources can drain while a stable SRAM operand is
        // still live. A page may not overwrite it merely because C is idle.
        auto alias = program;
        const auto& segment = packed.segments.front();
        const auto aliasQueue = InstructionControlUnit::mem_queue(
            static_cast<Hemisphere>(segment.hemisphere), segment.slice, 1);
        const auto bits = isa::encode_mem_instruction(MemInstruction::Read(segment.base_row, StreamId::East(2)));
        QueueCommand read;
        read.command = static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);
        read.instruction_kind = InstructionKind::Mem;
        read.word_count = 2;
        read.words[0] = static_cast<std::uint32_t>(bits);
        read.words[1] = static_cast<std::uint32_t>(bits >> 32);
        alias.queues.push_back({QueueKind::Mem, aliasQueue,
            {encode_macro_schedule_command(read, IcuMacroSchedule{256, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress})}});
        alias.queues.back().requires_raw_macro = true;
        rejects([&] { runtime.load(alias); }, "page write overwrote a stable operand needed after resume");
        alias.address_relocations.push_back({0, BindingAccess::Internal, QueueKind::Mem,
            static_cast<std::uint16_t>(aliasQueue), 0});
        rejects([&] { runtime.load(alias); }, "internal SRAM read was mistaken for the incoming weight page");
        alias.address_relocations.back().binding_access = BindingAccess::Input;
        runtime.load(alias); // The incoming page's own linked read is legal.
        // A mixed queue can take the Native compatibility loader. Its first
        // Macro runs at L256, so a following Native read really runs after the
        // page-in; the proof must not mistake it for cursor zero and skip it.
        for (const auto& native : {read,
                encode_icu_control_raw_word(IcuControlInstruction::Nop(1))}) {
            auto mixed = alias;
            mixed.queues.back().requires_raw_macro = false;
            mixed.queues.back().commands.push_back(native);
            rejects([&] { runtime.load(mixed); },
                "loader accepted mixed Macro/Native MEM timing in its common storage proof");
        }
        auto earlyWrite = alias;
        auto write = read;
        const auto writeBits = isa::encode_mem_instruction(MemInstruction::Write(segment.base_row, StreamId::West(2)));
        write.words[0] = static_cast<std::uint32_t>(writeBits);
        write.words[1] = static_cast<std::uint32_t>(writeBits >> 32);
        earlyWrite.queues.back().commands.push_back(encode_macro_schedule_command(write,
            {300, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress}));
        rejects([&] { runtime.load(earlyWrite); }, "computation overwrote a weight page before its declared release");
        auto replaced = program;
        replaced.max_cycle = 700;
        replaced.queues.push_back({QueueKind::Mem, aliasQueue, {
            encode_macro_schedule_command(write, {530, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress}),
            encode_macro_schedule_command(read, {600, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress})}});
        replaced.queues.back().requires_raw_macro = true;
        runtime.load(replaced); // Released storage was completely replaced before its new reader.
        auto holes = replaced;
        const auto holeRead = isa::encode_mem_instruction(MemInstruction::Read(segment.base_row + 1, StreamId::East(2)));
        auto readHole = read;
        readHole.words[0] = static_cast<std::uint32_t>(holeRead);
        readHole.words[1] = static_cast<std::uint32_t>(holeRead >> 32);
        holes.queues.back().commands = {
            encode_macro_schedule_command(write, {530, 2, 1, 2, 1, 1, 0, IcuInductionTarget::MemAddress}),
            encode_macro_schedule_command(readHole, {600, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress})};
        rejects([&] { runtime.load(holes); }, "a hole in the future write domain was mistaken for complete SRAM replacement");
        auto overlap = program;
        auto secondBinding = overlap.bindings.front();
        secondBinding.index = 1;
        overlap.bindings.push_back(secondBinding);
        auto secondUse = overlap.weight_page_uses.front();
        secondUse.binding_index = 1;
        secondUse.ready_cycle = 256;
        ++secondUse.transport_launch_event_id;
        overlap.weight_page_uses.push_back(secondUse);
        overlap.barriers.push_back({129, 0x30001});
        overlap.barrier_events.push_back({0x30001, {1}});
        bool rejectedReuse = false;
        try { runtime.load(overlap); }
        catch (const std::invalid_argument& error) {
            rejectedReuse = std::string(error.what()).find("SRAM reuse") != std::string::npos;
        }
        require(rejectedReuse, "final table accepted a second overlapping page before the first page's release");
        // The same rows may instead receive a later, fully linked weight
        // instance after release. Its own barrier proves a visible MEM commit.
        auto futurePage = program;
        futurePage.max_cycle = 700;
        futurePage.bindings.push_back(secondBinding);
        secondUse.ready_cycle = 600;
        secondUse.release_cycle = 700;
        futurePage.weight_page_uses.push_back(secondUse);
        futurePage.timelines.push_back({"c2c.window.binding:1.page:0", 520, 590});
        futurePage.queues.push_back({QueueKind::Mem, aliasQueue, {
            encode_macro_schedule_command(read, {600, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress})}});
        futurePage.queues.back().requires_raw_macro = true;
        futurePage.address_relocations.push_back({1, BindingAccess::Input, QueueKind::Mem,
            static_cast<std::uint16_t>(aliasQueue), 0});
        C2cWeightPage physicalPage;
        physicalPage.bank = 1;
        for (const auto& item : packed.segments)
            physicalPage.segments.push_back({static_cast<Hemisphere>(item.hemisphere), item.slice, 1,
                item.base_row, 0, kDdrBase + item.byte_offset, item.vector_count});
        C2cWeightPager futurePager(*system);
        futurePager.begin_schedule(futurePage);
        const std::array<std::size_t, 1> firstUse {0}, nextUse {1};
        static_cast<void>(futurePager.schedule(futurePage, physicalPage, 32, 48, kStop,
            0x10000, 0x30000, 32, firstUse));
        static_cast<void>(futurePager.schedule(futurePage, physicalPage, 520, 540, 600,
            0x10001, 0x30001, 516, nextUse));
        futurePager.finalize_schedule(futurePage);
        runtime.load(futurePage);
        auto unrelated = futurePage;
        unrelated.address_relocations.back().binding_index = 2;
        rejects([&] { runtime.load(unrelated); }, "a future page for another binding released an unrelated stable operand");
    }
    return result;
}
}
int main()
try {
    const auto early = run(3);
    const auto late = run(53);
    const auto atZero = run(3, true);
    require(atZero.issues == early.issues && atZero.native_issues == early.native_issues
            && atZero.logical == early.logical,
        "cycle-zero launch lost its event or changed computation");
    require(early.logical == late.logical && early.issues == late.issues
            && early.native_issues == late.native_issues,
        "DDR latency changed the logical Macro issue content or sequence");
    require(early.issues == std::vector<std::pair<std::size_t, std::size_t>>({{0, 600}, {256, 601}, {512, 602}}),
        "Macro had repeated or omitted computation across the common stop");
    require(early.native_issues == std::vector<std::pair<std::size_t, std::size_t>>({{32, 700}, {256, 701}, {513, 702}}),
        "Native phase shifted or an instruction repeated/disappeared across the common stop");
    require(late.physical > early.physical, "the second transport latency did not add physical waiting");
    std::cout << "runtime_common_barrier_test passed logical=" << early.logical
              << " early_physical=" << early.physical << " late_physical=" << late.physical
              << " issues=" << early.issues.size() << " native_issues=" << early.native_issues.size() << " generation=isolated real_mem_commit=verified corrupt_transport_cases=10 cycle_zero=verified stable_sram=protected\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "runtime_common_barrier_test failed: " << error.what() << '\n';
    return 1;
}
