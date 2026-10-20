#include "ftlpu/software/runtime/fu_3d_macro_lowering.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ftlpu;
using namespace ftlpu::software::runtime;
constexpr auto kInstruction =
    static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction);

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

template <typename Action>
void rejects(Action action, const char* reason)
{
    try {
        action();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(reason) != std::string::npos,
            "rejection did not explain the unsupported condition");
        return;
    }
    throw std::runtime_error("invalid source unexpectedly accepted");
}

QueueCommand native(const MemInstruction& x)
{
    const auto bits = isa::encode_mem_instruction(x);
    return {kInstruction, InstructionKind::Mem, 2,
        {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

QueueCommand native(const MxmControlInstruction& x)
{
    const auto bits = isa::encode_mxm_instruction(x);
    return {kInstruction, InstructionKind::Mxm, 2,
        {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32), 0, 0}};
}

QueueCommand native(const MxmDequantInstruction& x)
{
    return {kInstruction, InstructionKind::MxmDequant, 1,
        {static_cast<std::uint32_t>(isa::encode_mxm_dequant_instruction(x)), 0, 0, 0}};
}

template <typename Packet>
std::size_t append(QueueProgram& q, const Packet& packet)
{
    const auto start = q.commands.size();
    const auto kind = q.kind == QueueKind::Mem ? InstructionKind::Mem
        : q.kind == QueueKind::MxmDequant ? InstructionKind::MxmDequant : InstructionKind::Mxm;
    for (const auto& word : packet.words) {
        QueueCommand command;
        command.command = word.lanes[0];
        command.instruction_kind = kind;
        command.word_count = Packet::kLanesPerWord;
        std::copy(word.lanes.begin(), word.lanes.end(), command.words.begin());
        q.commands.push_back(command);
    }
    q.commands.resize(start + fu_3d_raw_packet_word_count(q.kind, q.commands[start]));
    return start;
}

template <typename Coarse, typename Expand>
std::vector<LogicalIssue> expected(const Coarse& coarse,
    std::size_t cursor, Expand expand)
{
    std::vector<LogicalIssue> issues;
    IcuCoordinate3D coordinate;
    do {
        issues.push_back({cursor + detail::icu_loop_3d_issue_cycle(coarse.loop, coordinate),
            native(expand(coarse, coordinate))});
    } while (detail::advance_icu_coordinate_3d(coarse.loop, coordinate));
    std::sort(issues.begin(), issues.end(),
        [](const auto& a, const auto& b) { return a.cycle < b.cycle; });
    return issues;
}

void equivalent(const QueueProgram& macro, const std::vector<LogicalIssue>& source)
{
    BinaryProgram p;
    p.max_cycle = source.empty() ? 0 : source.back().cycle;
    p.queues = {macro};
    const auto compare = [&](const QueueProgram& q) {
        p.queues = {q};
        const auto expanded = inspect_logical_issues(p);
        require(expanded.queues.size() == 1, "converted queue identity changed");
        const auto& actual = expanded.queues[0].issues;
        require(actual.size() == source.size(), "coarse conversion changed issue count");
        for (std::size_t i = 0; i < source.size(); ++i)
            require(actual[i].cycle == source[i].cycle
                    && actual[i].instruction.instruction_kind == source[i].instruction.instruction_kind
                    && actual[i].instruction.words == source[i].instruction.words,
                "coarse conversion changed an issue cycle or native operand");
    };
    compare(macro);
    if (macro.kind == QueueKind::Mem) {
        compare(decode_mem_macro_imem(
            pack_mem_macro_imem(encode_mem_macro_bitstream(macro)), macro.index));
    } else {
        compare(decode_mxm_macro_imem(
            pack_mxm_macro_imem(encode_mxm_macro_bitstream(macro)), macro.kind, macro.index));
    }
}

void mem_mapping_and_waits()
{
    IcuLoop3D loop;
    loop.counts = {3, 4, 2};
    loop.cycle_strides = {2, 12, 64};
    loop.wait_cycle = 3;
    const auto coarse = MemIcuInstruction::Read3D(loop,
        MemIcuAddress3D::Affine(20, {1, 8, 64}), StreamId::West(4));
    QueueProgram source {QueueKind::Mem, 7, {}};
    source.commands.push_back(encode_icu_control_raw_word(IcuControlInstruction::Nop(5)));
    const auto header = append(source, isa::encode_mem_icu_3d_instruction(coarse));
    source.commands.push_back({isa::encode_icu_nop(7)});
    const auto single = source.commands.size();
    source.commands.push_back(native(MemInstruction::WriteTap(91, StreamId::East(6))));
    const auto converted = lower_fu_3d_queue_to_macro(source);
    auto trace = expected(coarse, 5, detail::expand_mem_icu_instruction);
    trace.push_back({120, source.commands[single]});
    equivalent(converted.queue, trace);
    require(converted.queue.index == 7 && converted.command_indices.size() == source.commands.size(),
        "source mapping changed queue identity or size");
    require(converted.command_indices[header].size() == 2
            && converted.command_indices[header + 1].empty()
            && converted.command_indices[header + 2].empty()
            && converted.command_indices[0].empty()
            && converted.command_indices[single - 1].empty()
            && converted.command_indices[single].size() == 1,
        "one-to-many header mapping or NOP mapping is wrong");
    rejects([&] { lower_fu_3d_queue_to_macro(source, 1); }, "budget");
}

void blocked_outer_tail()
{
    IcuLoop3D loop;
    loop.counts = {2, 2, 5};
    loop.cycle_strides = {1, 4, 16};
    loop.wait_cycle = 50;
    const auto coarse = MemIcuInstruction::WriteTap3D(loop,
        MemIcuAddress3D::BlockedOuter(50, 1, 4, 2, 16, 50), StreamId::East(2));
    QueueProgram source {QueueKind::Mem, 2, {}};
    append(source, isa::encode_mem_icu_3d_instruction(coarse));
    const auto converted = lower_fu_3d_queue_to_macro(source);
    equivalent(converted.queue, expected(coarse, 0, detail::expand_mem_icu_instruction));
    require(converted.command_indices[0].size() > 1,
        "blocked groups and ragged tail were not partitioned");
}

void mxm_buffer_and_terminal()
{
    IcuLoop3D loop;
    loop.counts = {4, 3, 2};
    loop.cycle_strides = {1, 8, 40};
    loop.wait_cycle = 100;
    const auto compute = MxmComputeIcuInstruction::Compute3D(loop, 0,
        MxmIcuBufferMode::ToggleDimension1, StreamId::East(0).packed(),
        // Counter zero advances the MXM's row wave via row_stride; only
        // middle/outer counters may change its accumulator base address.
        4, 4, {0, 8, 32}, 1,
        MxmDataFormat::BFloat16,
        {MxmAccumulatorDestination::Sram, false, MxmAccumulatorOutputFormat::Float32},
        0, {MxmAccumulatorDestination::Stream, true, MxmAccumulatorOutputFormat::BFloat16});
    QueueProgram source {QueueKind::MxmCompute, 1, {}};
    append(source, isa::encode_mxm_compute_icu_shortest_instruction(compute));
    auto converted = lower_fu_3d_queue_to_macro(source);
    equivalent(converted.queue, expected(compute, 0, detail::expand_mxm_compute_icu_instruction));
    require(converted.command_indices[0].size() >= 4,
        "terminal and parity changes were folded into one constant native template");

    // Terminal and parity on the same axis must also handle odd lengths.
    auto sameAxis = compute;
    sameAxis.loop.counts = {5, 2, 1};
    sameAxis.loop.cycle_strides = {1, 8, 40};
    sameAxis.weight_buffer_mode = MxmIcuBufferMode::ToggleDimension0;
    source.commands.clear();
    append(source, isa::encode_mxm_compute_icu_3d_instruction(sameAxis));
    converted = lower_fu_3d_queue_to_macro(source);
    equivalent(converted.queue, expected(sameAxis, 0, detail::expand_mxm_compute_icu_instruction));
}

void load_parity_and_dequant()
{
    IcuLoop3D loop;
    loop.counts = {2, 2, 3};
    loop.cycle_strides = {1, 20, 80};
    loop.wait_cycle = 100;
    const auto load = MxmLoadIcuInstruction::Load3D(loop, 0,
        MxmIcuBufferMode::ToggleDimension2, 0, {1, 2, 0}, StreamId::East(0).packed());
    QueueProgram source {QueueKind::MxmLoad, 0, {}};
    append(source, isa::encode_mxm_load_icu_shortest_instruction(load));
    equivalent(lower_fu_3d_queue_to_macro(source).queue,
        expected(load, 0, detail::expand_mxm_load_icu_instruction));
    const auto dequant = MxmDequantIcuInstruction::Dequant3D(loop, MxmDequantInstruction::Scale(0.125f));
    source = {QueueKind::MxmDequant, 0, {}};
    append(source, isa::encode_mxm_dequant_icu_shortest_instruction(dequant));
    equivalent(lower_fu_3d_queue_to_macro(source).queue,
        expected(dequant, 0, [](const auto& x, const auto&) { return x.instruction; }));
}

void large_closed_form_and_limits()
{
    // 262144 work points become one two-axis descriptor without constructing
    // a point list. Adjacent axes flatten only while counts remain <=65536.
    IcuLoop3D loop;
    loop.counts = {64, 64, 64};
    loop.cycle_strides = {1, 64, 4096};
    const auto coarse = MemIcuInstruction::Read3D(loop,
        MemIcuAddress3D::Affine(0, {0, 0, 0}), StreamId::East(0));
    QueueProgram source {QueueKind::Mem, 0, {}};
    append(source, isa::encode_mem_icu_3d_instruction(coarse));
    const auto lowered = lower_fu_3d_queue_to_macro(source, 1);
    require(lowered.queue.commands.size() == 1,
        "closed-form work was unnecessarily split into issue-sized descriptors");
    const auto shape = decode_macro_schedule_command(lowered.queue.commands[0]);
    require(shape.inner_count * shape.outer_count == 262144,
        "large closed-form domain lost work points");
    source.commands.clear();
    for (unsigned i = 0; i < 5; ++i)
        source.commands.push_back({isa::encode_icu_nop(0x3fffffff)});
    source.commands.push_back(native(MemInstruction::Read(0, StreamId::East(0))));
    rejects([&] { lower_fu_3d_queue_to_macro(source); }, "Active Context");

    QueueProgram noWork {QueueKind::Mem, 0, {}};
    require(lower_fu_3d_queue_to_macro(noWork, 0).queue.commands.empty(),
        "empty queue gained a Macro or control word");
    noWork.commands.push_back({isa::encode_icu_nop(17)});
    require(lower_fu_3d_queue_to_macro(noWork, 0).queue.commands.empty(),
        "NOP-only queue gained a Macro");
}

void unsupported_and_malformed()
{
    QueueProgram source {QueueKind::Mem, 0,
        {encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(17))}};
    rejects([&] { lower_fu_3d_queue_to_macro(source); }, "event/SYNC");
    const auto sync = InstructionControlUnit::MemIcu::encode_synchronized_raw_packet(
        4, 17, 3, 2, MemInstruction::Write(64, StreamId::West(5)));
    const auto syncWords = encode_mem_synchronized_icu_packet(sync);
    source.commands = {syncWords[0], syncWords[1]};
    rejects([&] { lower_fu_3d_queue_to_macro(source); }, "synchronized MEM");
    MemIcuWriteRead2DInstruction wr;
    wr.counts = {2, 1};
    wr.write_cycle_strides = {4, 1};
    wr.read_cycle_strides = {4, 1};
    wr.read_start_offset = 2;
    wr.address_strides = {1, 0};
    source.commands.clear();
    append(source, isa::encode_mem_icu_write_read_2d_instruction(wr));
    rejects([&] { lower_fu_3d_queue_to_macro(source); }, "WRITE_READ_2D");
    source = {QueueKind::MxmLoad, 0, {}};
    const auto decode = MxmLoadIcuInstruction::DecodeLoadActivation3D({}, 0,
        MxmIcuBufferMode::Fixed, StreamId::East(0).packed(),
        MxmDataFormat::BFloat16, MxmDecodeLayout::Native4x4);
    append(source, isa::encode_mxm_load_icu_3d_instruction(decode));
    rejects([&] { lower_fu_3d_queue_to_macro(source); }, "native4");
    source = {QueueKind::Mem, 0, {}};
    append(source, isa::encode_mem_icu_3d_instruction(MemIcuInstruction {}));
    source.commands.pop_back();
    rejects([&] { lower_fu_3d_queue_to_macro(source); }, "truncated");
}

} // namespace

int main()
try {
    mem_mapping_and_waits();
    blocked_outer_tail();
    mxm_buffer_and_terminal();
    load_parity_and_dequant();
    large_closed_form_and_limits();
    unsupported_and_malformed();
    std::cout << "fu_3d_macro_lowering_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "fu_3d_macro_lowering_test failed: " << error.what() << '\n';
    return 1;
}
