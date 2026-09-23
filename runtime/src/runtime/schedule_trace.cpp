#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include "ftlpu/core/instruction_codec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ftlpu::software::runtime {
namespace {

std::string csv_field(const std::string& value)
{
    std::string result = "\"";
    for (const char ch : value) {
        if (ch == '"') result += '"';
        result += ch;
    }
    return result + '"';
}

std::string stream_name(std::size_t packed)
{
    const auto stream = StreamId::from_packed(packed);
    return std::string(stream.direction() == StreamDirection::East ? "E" : "W")
        + std::to_string(stream.index());
}

const char* mem_opcode_name(MemOpcode opcode)
{
    switch (opcode) {
    case MemOpcode::Read: return "Read";
    case MemOpcode::Write: return "Write";
    case MemOpcode::Gather: return "Gather";
    case MemOpcode::Scatter: return "Scatter";
    }
    return "Unknown";
}

struct EventDescription {
    std::string resource;
    std::string detail;
};

MemInstruction decode_mem_instruction_for_target(
    isa::EncodedMemInstruction word, std::size_t sram_depth_rows)
{
    if (sram_depth_rows == 0 || sram_depth_rows > (1u << 16)
        || (sram_depth_rows & (sram_depth_rows - 1)) != 0)
        throw std::logic_error(
            "runtime trace requires a power-of-two target SRAM depth");
    const auto opcode = static_cast<MemOpcode>(word & 0x7);
    const auto stream = static_cast<std::size_t>((word >> 3) & 0x3f);
    const auto map_stream = static_cast<std::size_t>((word >> 9) & 0x3f);
    const auto address_mask = static_cast<std::uint64_t>(sram_depth_rows - 1);
    const auto address = static_cast<std::size_t>((word >> 15) & address_mask);
    const bool preserve_stream = ((word >> 31) & 0x1) != 0;
    const auto used_mask = 0x80007fffull | (address_mask << 15);
    if ((word & ~used_mask) != 0)
        throw std::logic_error(
            "encoded MEM instruction exceeds target SRAM address width");
    if (preserve_stream && opcode != MemOpcode::Write)
        throw std::logic_error(
            "only encoded MEM Write can preserve its input stream");
    switch (opcode) {
    case MemOpcode::Read: return MemInstruction::Read(address, stream);
    case MemOpcode::Write:
        return preserve_stream
            ? MemInstruction::WriteTap(address, stream)
            : MemInstruction::Write(address, stream);
    case MemOpcode::Gather: return MemInstruction::Gather(stream, map_stream);
    case MemOpcode::Scatter: return MemInstruction::Scatter(stream, map_stream);
    }
    throw std::logic_error("unknown encoded MEM opcode");
}

EventDescription describe(const QueueProgram& queue, const QueueCommand& command,
    std::int64_t address_delta, std::size_t mxms_per_hemisphere,
    std::size_t sram_depth_rows,
    IcuInductionTarget induction_target = IcuInductionTarget::None)
{
    const auto east = queue.index < InstructionControlUnit::kMemQueuesPerHemisphere;
    switch (queue.kind) {
    case QueueKind::Mem: {
        if (induction_target != IcuInductionTarget::None
            && induction_target != IcuInductionTarget::MemAddress)
            throw std::logic_error(
                "runtime trace has a non-MEM induction target on a MEM queue");
        const std::size_t localQueue =
            queue.index % InstructionControlUnit::kMemQueuesPerHemisphere;
        const std::size_t slice = localQueue / hw::kMemBanksPerSlice;
        const std::size_t bank = localQueue % hw::kMemBanksPerSlice;
        const auto encoded = static_cast<isa::EncodedMemInstruction>(command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
        auto instruction = decode_mem_instruction_for_target(
            encoded, sram_depth_rows);
        const auto address =
            static_cast<std::int64_t>(instruction.address) + address_delta;
        if (address < 0)
            throw std::logic_error(
                "runtime trace MEM induction underflows the address");
        std::ostringstream detail;
        detail << "slice=" << slice << " bank=" << bank
               << " operation="
               << (instruction.opcode == MemOpcode::Read ? "read" : "write")
               << " addr=" << address << " stream=" << stream_name(instruction.stream);
        const char* opcodeName = instruction.preserve_stream
            ? "WriteTap"
            : mem_opcode_name(instruction.opcode);
        return {std::string("MEM.") + (east ? "E." : "W.")
                + opcodeName, detail.str()};
    }
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: {
        const auto encoded =
            static_cast<isa::EncodedMxmInstruction>(command.words[0])
            | (static_cast<isa::EncodedMxmInstruction>(command.words[1])
                << 32);
        auto instruction = isa::decode_mxm_instruction(encoded);
        auto target = induction_target;
        // Repeat predates explicit induction targets. Preserve its
        // opcode-selected MXM stride semantics while honoring the typed target
        // carried by Repeat2D and macro schedule commands.
        if (target == IcuInductionTarget::None && address_delta != 0) {
            target = instruction.opcode == MxmControlOpcode::IW
                ? IcuInductionTarget::MxmWeightColumn
                : IcuInductionTarget::MxmAccumulatorAddress;
        }
        if (target == IcuInductionTarget::MxmWeightColumn) {
            if (instruction.opcode != MxmControlOpcode::IW)
                throw std::logic_error(
                    "runtime trace MXM column induction requires IW");
            const auto column =
                static_cast<std::int64_t>(instruction.weight_column)
                + address_delta;
            if (column < 0
                || column >= static_cast<std::int64_t>(hw::kMxmColumns))
                throw std::logic_error(
                    "runtime trace MXM column induction is out of range");
            instruction.weight_column = static_cast<std::size_t>(column);
        } else if (target == IcuInductionTarget::MxmAccumulatorAddress) {
            if (instruction.opcode != MxmControlOpcode::Compute
                && instruction.opcode != MxmControlOpcode::AccumulatorRead)
                throw std::logic_error(
                    "runtime trace MXM accumulator induction requires compute or accumulator-read");
            const auto address =
                static_cast<std::int64_t>(instruction.accumulator_address)
                + address_delta;
            if (address < 0)
                throw std::logic_error(
                    "runtime trace MXM accumulator induction underflows the address");
            instruction.accumulator_address =
                static_cast<std::size_t>(address);
        } else if (target != IcuInductionTarget::None) {
            throw std::logic_error(
                "runtime trace has a non-MXM induction target on an MXM queue");
        }
        const auto per_hemisphere = mxms_per_hemisphere;
        const auto side = queue.index < per_hemisphere ? "E" : "W";
        std::ostringstream detail;
        if (instruction.opcode == MxmControlOpcode::IW) {
            detail << "IW buffer=" << instruction.weight_buffer
                   << " column=" << instruction.weight_column;
        } else if (instruction.opcode == MxmControlOpcode::Compute) {
            detail << "Compute buffer=" << instruction.weight_buffer
                   << " act=" << stream_name(instruction.activation_stream_base)
                   << " out=" << stream_name(instruction.stream_base)
                   << " acc=" << instruction.accumulator_address
                   << " stride=" << instruction.accumulator_row_stride
                   << " format="
                   << mxm_data_format_name(instruction.data_format)
                   << (instruction.accumulator_destination
                               == MxmAccumulatorDestination::Stream
                           ? " dst=stream"
                           : " dst=sram");
            if (instruction.accumulator_destination
                == MxmAccumulatorDestination::Stream)
                detail << " acc_output="
                       << (instruction.accumulator_output_format
                                   == MxmAccumulatorOutputFormat::BFloat16
                               ? "bf16"
                               : "fp32");
        } else {
            detail << "AccumulatorRead acc=" << instruction.accumulator_address
                   << " out=" << stream_name(instruction.stream_base)
                   << (instruction.accumulator_clear ? " clear" : " keep");
        }
        return {std::string("MXM.") + side + std::to_string(queue.index % per_hemisphere)
                + (queue.kind == QueueKind::MxmLoad ? ".Load" : ".Compute"), detail.str()};
    }
    case QueueKind::MxmDequant: {
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct an MXM dequant instruction");
        const auto instruction =
            isa::decode_mxm_dequant_instruction(
                static_cast<isa::EncodedMxmDequantInstruction>(
                    command.words[0]));
        const auto perHemisphere = mxms_per_hemisphere;
        const auto side =
            queue.index < perHemisphere ? "E" : "W";
        std::ostringstream detail;
        detail << "scale=" << instruction.scale();
        return {std::string("MXM.") + side
                + std::to_string(queue.index % perHemisphere)
                + ".Dequant",
            detail.str()};
    }
    case QueueKind::Vxm: {
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct a VXM instruction");
        const auto decoded = isa::decode_vxm_instruction(queue.index,
            isa::EncodedVxmInstruction {
                static_cast<std::uint64_t>(command.words[0])
                    | (static_cast<std::uint64_t>(command.words[1]) << 32),
                command.words[2]});
        const auto& instruction = decoded.instruction;
        std::ostringstream detail;
        detail << VxmLane::operation_name(instruction.operation)
               << " depth=" << static_cast<std::size_t>(decoded.chain_depth)
               << " repeat=" << instruction.repeat_count;
        if (instruction.output_stream)
            detail << " -> S" << *instruction.output_stream;
        return {"VXM.C" + std::to_string(queue.index), detail.str()};
    }
    case QueueKind::SxmTranspose:
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct an SXM transpose instruction");
        return {std::string("SXM.") + (queue.index == 0 ? "E.Transpose" : "W.Transpose"),
            "transpose"};
    case QueueKind::SxmPermute:
        if (induction_target != IcuInductionTarget::None
            || address_delta != 0)
            throw std::logic_error(
                "runtime trace cannot induct an SXM permute instruction");
        return {std::string("SXM.") + (queue.index == 0 ? "E.Permute" : "W.Permute"),
            "permute"};
    }
    throw std::logic_error("unknown ICU queue kind in schedule trace");
}

struct EventPattern {
    std::string_view kind{"single"};
    std::size_t inner_count{1};
    std::size_t inner_interval{0};
    std::int64_t inner_stride{0};
    std::size_t outer_count{1};
    std::size_t outer_interval{0};
    std::int64_t outer_stride{0};
    bool skip_first{false};
    IcuInductionTarget induction_target{IcuInductionTarget::None};
    std::int64_t base_delta{0};
};

InstructionKind raw_instruction_kind(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return InstructionKind::Mem;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute: return InstructionKind::Mxm;
    case QueueKind::MxmDequant: return InstructionKind::MxmDequant;
    case QueueKind::Vxm: return InstructionKind::Vxm;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: return InstructionKind::Sxm;
    }
    throw std::logic_error("unknown raw FU queue kind in schedule trace");
}

template <typename Packet>
Packet read_raw_trace_packet(
    const QueueProgram& queue, std::size_t commandIndex)
{
    if (commandIndex > queue.commands.size()
        || Packet::kWordCount > queue.commands.size() - commandIndex)
        throw std::logic_error("truncated raw FU packet in schedule trace");
    const auto expectedKind = raw_instruction_kind(queue.kind);
    Packet packet{};
    for (std::size_t word = 0; word < Packet::kWordCount; ++word) {
        const auto& command = queue.commands[commandIndex + word];
        const auto wordIndexMask = expectedKind == InstructionKind::Sxm
            ? std::uint32_t{0x7} : std::uint32_t{0x3};
        if (command.instruction_kind != expectedKind
            || command.word_count != Packet::kLanesPerWord
            || !command.extension_words.empty()
            || command.command != command.words[0]
            || !is_fu_3d_raw_word_command(command)
            || ((command.words[0] >> 2) & wordIndexMask) != word)
            throw std::logic_error(
                "malformed raw FU packet in schedule trace");
        for (std::size_t lane = 0; lane < Packet::kLanesPerWord; ++lane)
            packet.words[word].lanes[lane] = command.words[lane];
    }
    return packet;
}

EventPattern loop_pattern(const IcuLoop3D& loop,
    std::int64_t innerStride = 0, std::int64_t outerStride = 0,
    IcuInductionTarget induction = IcuInductionTarget::None)
{
    return {loop.counts[1] > 1 ? "repeat2d"
            : loop.counts[0] > 1 ? "repeat" : "single",
        loop.counts[0], loop.cycle_strides[0], innerStride,
        loop.counts[1], loop.cycle_strides[1], outerStride, false,
        induction, 0};
}

const char* buffer_mode_name(MxmIcuBufferMode mode)
{
    switch (mode) {
    case MxmIcuBufferMode::Fixed: return "fixed";
    case MxmIcuBufferMode::ToggleDimension0: return "toggle_d0";
    case MxmIcuBufferMode::ToggleDimension1: return "toggle_d1";
    case MxmIcuBufferMode::ToggleDimension2: return "toggle_d2";
    }
    return "unknown";
}

std::string mxm_resource(const QueueProgram& queue,
    std::size_t mxmsPerHemisphere, std::string_view suffix)
{
    if (mxmsPerHemisphere == 0)
        throw std::logic_error(
            "schedule trace requires at least one MXM per hemisphere");
    const auto side = queue.index < mxmsPerHemisphere ? "E" : "W";
    const auto local = queue.index % mxmsPerHemisphere;
    return "MXM." + std::string(side) + std::to_string(local)
        + "." + std::string(suffix);
}

std::size_t event_duration(
    const QueueProgram& queue, const QueueCommand& command)
{
    if (queue.kind != QueueKind::Vxm) return 1;
    const auto decoded = isa::decode_vxm_instruction(queue.index,
        isa::EncodedVxmInstruction {
            static_cast<std::uint64_t>(command.words[0])
                | (static_cast<std::uint64_t>(command.words[1]) << 32),
            command.words[2]});
    return decoded.instruction.repeat_count;
}

IcuInductionTarget resolve_induction_target(const QueueProgram& queue,
    const QueueCommand& command, IcuInductionTarget requested,
    const EventPattern& pattern)
{
    if (requested != IcuInductionTarget::None) return requested;
    if (pattern.base_delta == 0 && pattern.inner_stride == 0
        && pattern.outer_stride == 0)
        return IcuInductionTarget::None;
    if (queue.kind == QueueKind::Mem)
        return IcuInductionTarget::MemAddress;
    if (queue.kind == QueueKind::MxmLoad
        || queue.kind == QueueKind::MxmCompute) {
        const auto encoded =
            static_cast<isa::EncodedMxmInstruction>(command.words[0])
            | (static_cast<isa::EncodedMxmInstruction>(command.words[1]) << 32);
        const auto instruction = isa::decode_mxm_instruction(encoded);
        return instruction.opcode == MxmControlOpcode::IW
            ? IcuInductionTarget::MxmWeightColumn
            : IcuInductionTarget::MxmAccumulatorAddress;
    }
    return IcuInductionTarget::None;
}

std::string_view induction_name(IcuInductionTarget target)
{
    switch (target) {
    case IcuInductionTarget::None: return "none";
    case IcuInductionTarget::MemAddress: return "mem_address";
    case IcuInductionTarget::MxmWeightColumn: return "mxm_weight_column";
    case IcuInductionTarget::MxmAccumulatorAddress:
        return "mxm_accumulator_address";
    }
    throw std::logic_error("unknown ICU induction target in schedule trace");
}

void write_event(std::ostream& output, std::int64_t start, std::int64_t end,
    const EventDescription& event, const EventPattern& pattern = {})
{
    output << start << ',' << end << ',' << csv_field(event.resource) << ','
           << csv_field(event.detail) << ',' << csv_field(std::string(pattern.kind))
           << ',' << pattern.inner_count << ',' << pattern.inner_interval
           << ',' << pattern.inner_stride << ',' << pattern.outer_count
           << ',' << pattern.outer_interval << ',' << pattern.outer_stride
           << ',' << (pattern.skip_first ? 1 : 0) << ','
           << csv_field(std::string(induction_name(pattern.induction_target)))
           << ',' << pattern.base_delta << '\n';
}

std::size_t raw_loop_final_offset(const IcuLoop3D& loop)
{
    std::size_t offset = loop.wait_cycle;
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension)
        offset += (loop.counts[dimension] - 1)
            * loop.cycle_strides[dimension];
    return offset;
}

std::size_t write_raw_fu_packet(std::ostream& output,
    const QueueProgram& queue, std::size_t commandIndex,
    std::size_t packetIssueCycle, std::size_t mxmsPerHemisphere)
{
    const auto writeDepthRows = [&](const IcuLoop3D& loop,
                                    const EventPattern& pattern,
                                    std::size_t duration,
                                    const auto& describeDepth) {
        for (std::size_t depth = 0; depth < loop.counts[2]; ++depth) {
            const auto start = packetIssueCycle + loop.start_cycle
                + loop.wait_cycle
                + depth * loop.cycle_strides[2];
            write_event(output, static_cast<std::int64_t>(start),
                static_cast<std::int64_t>(start + duration),
                describeDepth(depth), pattern);
        }
    };

    switch (queue.kind) {
    case QueueKind::Mem: {
        if (is_mem_write_read_2d_raw_packet_header(
                queue.commands[commandIndex])) {
            const auto instruction =
                decode_mem_write_read_2d_raw_packet(queue, commandIndex);
            const auto local = queue.index
                % InstructionControlUnit::kMemQueuesPerHemisphere;
            const auto slice = local / hw::kMemBanksPerSlice;
            const auto bank = local % hw::kMemBanksPerSlice;
            const auto side = queue.index
                < InstructionControlUnit::kMemQueuesPerHemisphere
                ? "E" : "W";
            std::ostringstream writeDetail;
            writeDetail << "slice=" << slice << " bank=" << bank
                        << " operation=write addr="
                        << instruction.base_address << " stream="
                        << stream_name(instruction.write_stream);
            std::ostringstream readDetail;
            readDetail << "slice=" << slice << " bank=" << bank
                       << " operation=read addr="
                       << instruction.base_address << " stream_base="
                       << stream_name(instruction.read_stream_base)
                       << " stream_outer_stride="
                       << instruction.read_stream_outer_stride;
            const auto pattern = [&](const auto& cycleStrides) {
                return EventPattern {"repeat2d", instruction.counts[0],
                    cycleStrides[0], instruction.address_strides[0],
                    instruction.counts[1], cycleStrides[1],
                    instruction.address_strides[1], false,
                    IcuInductionTarget::MemAddress, 0};
            };
            const auto writeStart = packetIssueCycle
                + instruction.start_wait;
            const auto readStart = writeStart
                + instruction.read_start_offset;
            write_event(output, static_cast<std::int64_t>(writeStart),
                static_cast<std::int64_t>(writeStart + 1),
                {std::string("MEM.") + side + ".WRITE_READ_2D.Write",
                    writeDetail.str()},
                pattern(instruction.write_cycle_strides));
            write_event(output, static_cast<std::int64_t>(readStart),
                static_cast<std::int64_t>(readStart + 1),
                {std::string("MEM.") + side + ".WRITE_READ_2D.Read",
                    readDetail.str()},
                pattern(instruction.read_cycle_strides));
            return packetIssueCycle
                + ftlpu::detail::mem_icu_write_read_2d_last_issue_cycle(
                    instruction) + 1;
        }
        const auto instruction = isa::decode_mem_icu_3d_instruction(
            read_raw_trace_packet<isa::EncodedMemIcu3DPacket>(
                queue, commandIndex));
        const auto local = queue.index
            % InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto slice = local / hw::kMemBanksPerSlice;
        const auto bank = local % hw::kMemBanksPerSlice;
        const auto east = queue.index
            < InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto opcode = instruction.opcode == MemIcuOpcode::Read3D
            ? "Read3D"
            : instruction.opcode == MemIcuOpcode::Write3D
                ? "Write3D" : "WriteTap3D";
        const auto pattern = loop_pattern(instruction.loop,
            instruction.address.inner_stride,
            instruction.address.middle_stride,
            IcuInductionTarget::MemAddress);
        writeDepthRows(instruction.loop, pattern, 1,
            [&](std::size_t depth) {
                const auto address = ftlpu::detail::mem_icu_address_3d(
                    instruction,
                    IcuCoordinate3D{{0, 0, depth}});
                std::ostringstream detail;
                detail << "slice=" << slice << " bank=" << bank
                       << " operation="
                       << (instruction.opcode == MemIcuOpcode::Read3D
                               ? "read" : "write")
                       << " addr=" << address
                       << " stream=" << stream_name(instruction.stream)
                       << " depth=" << depth << '/'
                       << instruction.loop.counts[2]
                       << " d2_cycle_stride="
                       << instruction.loop.cycle_strides[2]
                       << " outer_group_size="
                       << instruction.address.outer_group_size
                       << " outer_inner_stride="
                       << instruction.address.outer_inner_stride
                       << " outer_group_stride="
                       << instruction.address.outer_group_stride;
                return EventDescription{
                    std::string("MEM.") + (east ? "E." : "W.")
                        + opcode,
                    detail.str()};
            });
        return packetIssueCycle
            + raw_loop_final_offset(instruction.loop) + 1;
    }
    case QueueKind::MxmLoad: {
        const auto instruction = isa::decode_mxm_load_icu_3d_instruction(
            read_raw_trace_packet<isa::EncodedMxmLoadIcu3DPacket>(
                queue, commandIndex));
        const auto pattern = loop_pattern(instruction.loop,
            instruction.weight_column_strides[0],
            instruction.weight_column_strides[1],
            IcuInductionTarget::MxmWeightColumn);
        writeDepthRows(instruction.loop, pattern, 1,
            [&](std::size_t depth) {
                const auto column = ftlpu::detail::checked_icu_3d_operand(
                    instruction.weight_column_base,
                    instruction.weight_column_strides,
                    IcuCoordinate3D{{0, 0, depth}},
                    "trace MXM weight column");
                std::ostringstream detail;
                detail << "Load3D buffer="
                       << instruction.weight_buffer_base
                       << " buffer_mode="
                       << buffer_mode_name(instruction.weight_buffer_mode)
                       << " column=" << column
                       << " stream="
                       << stream_name(instruction.weight_stream_base)
                       << " input_mode="
                       << static_cast<unsigned>(
                              instruction.weight_input_mode)
                       << " depth=" << depth << '/'
                       << instruction.loop.counts[2]
                       << " d2_column_stride="
                       << instruction.weight_column_strides[2];
                return EventDescription{
                    mxm_resource(queue, mxmsPerHemisphere, "Load"),
                    detail.str()};
            });
        return packetIssueCycle
            + raw_loop_final_offset(instruction.loop) + 1;
    }
    case QueueKind::MxmDequant: {
        const auto instruction = isa::decode_mxm_dequant_icu_3d_instruction(
            read_raw_trace_packet<isa::EncodedMxmDequantIcu3DPacket>(
                queue, commandIndex));
        const auto pattern = loop_pattern(instruction.loop);
        writeDepthRows(instruction.loop, pattern, 1,
            [&](std::size_t depth) {
                std::ostringstream detail;
                detail << "Dequant3D scale="
                       << instruction.instruction.scale()
                       << " depth=" << depth << '/'
                       << instruction.loop.counts[2];
                return EventDescription{
                    mxm_resource(queue, mxmsPerHemisphere, "Dequant"),
                    detail.str()};
            });
        return packetIssueCycle
            + raw_loop_final_offset(instruction.loop) + 1;
    }
    case QueueKind::MxmCompute: {
        const auto instruction = isa::decode_mxm_compute_icu_3d_instruction(
            read_raw_trace_packet<isa::EncodedMxmComputeIcu3DPacket>(
                queue, commandIndex));
        const auto pattern = loop_pattern(instruction.loop,
            instruction.accumulator_address_strides[0],
            instruction.accumulator_address_strides[1],
            IcuInductionTarget::MxmAccumulatorAddress);
        writeDepthRows(instruction.loop, pattern, 1,
            [&](std::size_t depth) {
                const auto accumulator =
                    ftlpu::detail::checked_icu_3d_operand(
                    instruction.accumulator_address_base,
                    instruction.accumulator_address_strides,
                    IcuCoordinate3D{{0, 0, depth}},
                    "trace MXM accumulator address");
                std::ostringstream detail;
                if (instruction.opcode
                    == MxmComputeIcuOpcode::AccumulatorRead3D) {
                    detail << "AccumulatorRead3D out="
                           << stream_name(instruction.result_stream_base)
                           << " acc=" << accumulator;
                } else {
                    detail << "Compute3D buffer="
                           << instruction.weight_buffer_base
                           << " buffer_mode="
                           << buffer_mode_name(
                                  instruction.weight_buffer_mode)
                           << " act="
                           << stream_name(
                                  instruction.activation_stream_base)
                           << " out="
                           << stream_name(instruction.result_stream_base)
                           << " acc=" << accumulator
                           << " row_stride="
                           << instruction.accumulator_row_stride
                           << " format="
                           << mxm_data_format_name(instruction.data_format)
                           << " terminal_dimension="
                           << instruction.terminal_dimension;
                }
                detail << " depth=" << depth << '/'
                       << instruction.loop.counts[2]
                       << " d2_acc_stride="
                       << instruction.accumulator_address_strides[2];
                return EventDescription{
                    mxm_resource(queue, mxmsPerHemisphere, "Compute"),
                    detail.str()};
            });
        return packetIssueCycle
            + raw_loop_final_offset(instruction.loop) + 1;
    }
    case QueueKind::Vxm: {
        const auto run = isa::decode_vxm_icu_run_2d_instruction(
            read_raw_trace_packet<isa::EncodedVxmIcuRun2DPacket>(
                queue, commandIndex));
        const auto decoded = VxmCompactInstructionCodec::decode(
            queue.index, run.instruction);
        std::ostringstream detail;
        detail << VxmLane::operation_name(decoded.instruction.operation)
               << " depth="
               << static_cast<std::size_t>(decoded.chain_depth)
               << " repeat=" << decoded.instruction.repeat_count
               << " Run2D";
        write_event(output,
            static_cast<std::int64_t>(packetIssueCycle
                + run.loop.start_cycle + run.loop.wait_cycle),
            static_cast<std::int64_t>(packetIssueCycle
                + run.loop.start_cycle + run.loop.wait_cycle
                + decoded.instruction.repeat_count),
            {"VXM.C" + std::to_string(queue.index), detail.str()},
            loop_pattern(run.loop));
        return packetIssueCycle + raw_loop_final_offset(run.loop) + 1;
    }
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: {
        const auto run = isa::decode_sxm_icu_run_2d_instruction(
            read_raw_trace_packet<isa::EncodedSxmIcuRun2DPacket>(
                queue, commandIndex));
        std::ostringstream detail;
        detail << (queue.kind == QueueKind::SxmTranspose
                       ? "transpose" : "permute")
               << " Run2D map_stride=" << run.permute_map_stride;
        const auto side = queue.index == 0 ? "E" : "W";
        write_event(output,
            static_cast<std::int64_t>(packetIssueCycle
                + run.loop.start_cycle + run.loop.wait_cycle),
            static_cast<std::int64_t>(packetIssueCycle
                + run.loop.start_cycle + run.loop.wait_cycle + 1),
            {std::string("SXM.") + side
                    + (queue.kind == QueueKind::SxmTranspose
                            ? ".Transpose" : ".Permute"),
                detail.str()},
            loop_pattern(run.loop));
        return packetIssueCycle + raw_loop_final_offset(run.loop) + 1;
    }
    }
    throw std::logic_error("unknown raw FU packet in schedule trace");
}

void write_pattern(std::ostream& output, const QueueProgram& queue,
    const QueueCommand& command, std::int64_t start,
    std::size_t mxms_per_hemisphere, std::size_t sram_depth_rows,
    EventPattern pattern)
{
    pattern.induction_target = resolve_induction_target(
        queue, command, pattern.induction_target, pattern);
    const auto event = describe(queue, command, 0, mxms_per_hemisphere,
        sram_depth_rows, pattern.induction_target);
    write_event(output, start,
        start + static_cast<std::int64_t>(event_duration(queue, command)),
        event, pattern);
}

const BinaryBinding& find_paged_weight(
    const BinaryProgram& program, std::uint32_t index)
{
    const auto found = std::ranges::find_if(program.bindings,
        [index](const BinaryBinding& binding) {
            return binding.access == BindingAccess::Input
                && binding.index == index && binding.paged_weight;
        });
    if (found == program.bindings.end())
        throw std::logic_error(
            "weight-page trace references a missing paged binding");
    return *found;
}

void write_weight_prefetches(
    std::ostream& output, const BinaryProgram& program,
    std::span<const WeightPrefetchPlan> physicalPrefetches)
{
    if (program.weight_page_uses.empty()) return;
    const std::uint64_t lanes =
        program.hardware.c2c_streams_per_direction;
    const std::uint64_t bytesPerLane =
        program.hardware.c2c_bytes_per_stream_per_cycle;
    if (lanes == 0 || bytesPerLane == 0)
        throw std::logic_error(
            "paged weight trace requires non-zero C2C bandwidth");

    const std::uint64_t bandwidth = lanes * bytesPerLane;
    std::vector<WeightPrefetchPlan> computedPrefetches;
    if (physicalPrefetches.empty())
        computedPrefetches = plan_weight_prefetches(program);
    const std::span<const WeightPrefetchPlan> prefetches =
        physicalPrefetches.empty()
        ? std::span<const WeightPrefetchPlan>(computedPrefetches)
        : physicalPrefetches;
    std::int64_t preExecutionCursor = 0;
    for (const auto& prefetch : prefetches)
        if (prefetch.pre_execution)
            preExecutionCursor -= static_cast<std::int64_t>(
                prefetch.transfer_end_cycle - prefetch.start_cycle);
    for (const auto& prefetch : prefetches) {
        std::vector<std::string> bindings;
        for (const std::size_t useIndex : prefetch.use_indices) {
            const auto& use = program.weight_page_uses[useIndex];
            const auto& binding =
                find_paged_weight(program, use.binding_index);
            bindings.push_back(binding.name.empty()
                ? std::to_string(binding.index) : binding.name);
        }
        for (std::size_t side = 0; side < prefetch.bytes.size(); ++side) {
            if (prefetch.bytes[side] == 0) continue;
            std::ostringstream detail;
            detail << "page=" << prefetch.page_index
                   << " bank=" << prefetch.bank << " bindings=";
            for (std::size_t index = 0; index < bindings.size(); ++index) {
                if (index != 0) detail << '+';
                detail << bindings[index];
            }
            detail << " bytes=" << prefetch.bytes[side]
                   << " lanes=" << lanes
                   << " bandwidth=" << bandwidth
                   << "B/cycle consumer_cycle=" << prefetch.ready_cycle
                   << " phase="
                   << (prefetch.pre_execution ? "pre_execution" : "overlap")
                   << " scheduled=true";
            const auto duration = static_cast<std::int64_t>(
                prefetch.transfer_end_cycle - prefetch.start_cycle);
            const auto start = prefetch.pre_execution
                ? preExecutionCursor
                : static_cast<std::int64_t>(prefetch.start_cycle);
            const auto end = prefetch.pre_execution
                ? preExecutionCursor + duration
                : static_cast<std::int64_t>(
                    prefetch.transfer_end_cycle);
            write_event(output,
                start, end,
                {std::string("C2C.") + (side == 0 ? "E" : "W")
                        + ".Prefetch",
                    detail.str()});
            const auto sideName = side == 0 ? "E" : "W";
            std::ostringstream pathDetail;
            pathDetail << "page=" << prefetch.page_index
                       << " bank=" << prefetch.bank
                       << " streams=W" << (hw::kWestStreams - lanes)
                       << "..W" << (hw::kWestStreams - 1)
                       << " sync=target_mem+stream_tag"
                       << " timing=per_vector_notification";
            write_event(output, start, end,
                {std::string("SR.") + sideName + ".C2C.Shared",
                    pathDetail.str()});
            write_event(output, start, end,
                {std::string("MEM.") + sideName + ".C2CWrite",
                    pathDetail.str()});
        }
        if (prefetch.pre_execution)
            preExecutionCursor += static_cast<std::int64_t>(
                prefetch.transfer_end_cycle - prefetch.start_cycle);
    }
}

} // namespace

void write_schedule_trace_csv(const BinaryProgram& program,
    const std::filesystem::path& path,
    std::span<const WeightPrefetchPlan> physicalPrefetches)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open runtime schedule trace: " + path.string());
    output << "start,end,resource,detail,pattern,inner_count,inner_interval,"
              "inner_stride,outer_count,outer_interval,outer_stride,skip_first,"
              "induction,base_delta\n";
    write_weight_prefetches(output, program, physicalPrefetches);

    for (const auto& queue : program.queues) {
        std::size_t cursor = 0;
        std::size_t previous_cycle = 0;
        const QueueCommand* previous = nullptr;
        std::deque<std::pair<const QueueCommand*, std::size_t>> history;
        for (std::size_t commandIndex = 0;
             commandIndex < queue.commands.size(); ++commandIndex) {
            const auto& command = queue.commands[commandIndex];
            if (is_mem_synchronized_raw_packet_header(command)) {
                if (queue.kind != QueueKind::Mem)
                    throw std::logic_error(
                        "synchronized MEM packet is outside a MEM queue");
                const auto packet = decode_mem_synchronized_icu_packet(
                    queue, commandIndex);
                const auto& header = packet[0].lanes;
                const auto& native = packet[1].lanes;
                const std::size_t reservation =
                    (header[2] & 0x00ffffffU) + 1;
                const std::size_t vectors =
                    ((header[0] >> 2) & 0xffffU) + 1;
                const std::size_t tag =
                    ((header[0] >> 18) | (header[1] << 14)) & 0xffffU;
                const std::size_t localQueue = queue.index
                    % InstructionControlUnit::kMemQueuesPerHemisphere;
                const auto side = queue.index
                        < InstructionControlUnit::kMemQueuesPerHemisphere
                    ? "E" : "W";
                const auto encoded =
                    (static_cast<isa::EncodedMemInstruction>(native[0])
                     | (static_cast<isa::EncodedMemInstruction>(native[1])
                        << 32)) >> 2
                    | (static_cast<isa::EncodedMemInstruction>(native[2])
                       << 62);
                const auto transfer = decode_mem_instruction_for_target(
                    encoded, program.hardware.sram_depth_rows);
                const bool readSync = transfer.opcode == MemOpcode::Read;
                std::ostringstream detail;
                detail << "opcode="
                       << (readSync ? "MEM_READ_SYNC" : "MEM_WRITE_SYNC")
                       << " slice="
                       << localQueue / hw::kMemBanksPerSlice
                       << " bank=" << localQueue % hw::kMemBanksPerSlice
                       << " pc=" << commandIndex
                       << " sync_tag=" << tag
                       << " vectors=" << vectors
                       << " reservation_cycles=" << reservation
                       << " addr=" << transfer.address
                       << " stream=" << stream_name(
                              transfer.stream_id().packed());
                write_event(output, cursor, cursor + reservation,
                    {std::string("MEM.") + side
                         + (readSync ? ".ReadSync" : ".WriteSync"),
                     detail.str()});
                cursor += reservation;
                commandIndex += InstructionControlUnit::MemIcu::
                    synchronized_packet_word_count - 1;
                previous = nullptr;
                history.clear();
                continue;
            }
            if (is_icu_control_raw_word_command(command)) {
                const auto control = decode_icu_control_raw_word(command);
                if (control.opcode == IcuControlOpcode::Sync) {
                    const auto side = queue.kind == QueueKind::Mem
                        ? (queue.index
                                < InstructionControlUnit::
                                    kMemQueuesPerHemisphere ? "E" : "W")
                        : (queue.index == 0 ? "E" : "W");
                    const auto resource = queue.kind == QueueKind::Mem
                        ? std::string("MEM.") + side + ".Sync"
                        : std::string("C2C.") + side + ".Sync";
                    std::ostringstream detail;
                    detail << "opcode=SYNC queue="
                           << queue_kind_name(queue.kind)
                           << " index=" << queue.index
                           << " pc=" << commandIndex
                           << " actual_wait=runtime_dependent";
                    write_event(output, cursor, cursor + 1,
                        {resource, detail.str()});
                    ++cursor;
                    previous = nullptr;
                    history.clear();
                    continue;
                }
            }
            if (queue.kind == QueueKind::C2cDma
                && command.instruction_kind == InstructionKind::C2cDma) {
                const auto packet = decode_c2c_dma_icu_packet(
                    queue, commandIndex);
                const auto instruction = C2cIcuPacketCodec::decode_dma(packet);
                std::ostringstream detail;
                detail << (instruction.direction
                                   == C2cDmaDirection::Ddr4ToC2c
                               ? "load" : "store")
                       << " lane=" << instruction.lane
                       << " vectors=" << instruction.vector_count
                       << " stride=" << instruction.address_stride_bytes
                       << " sync=" << instruction.sync_tag;
                const auto side = queue.index == 0 ? "E" : "W";
                write_event(output, cursor,
                    cursor + static_cast<std::int64_t>(
                        instruction.vector_count),
                    {std::string("C2C.") + side + ".DMA",
                        detail.str()});
                commandIndex += C2cDmaIcuPacket::kWordCount - 1;
                ++cursor;
                previous = nullptr;
                history.clear();
                continue;
            }
            if ((queue.kind == QueueKind::C2cTx
                    || queue.kind == QueueKind::C2cRx)
                && command.instruction_kind
                    == InstructionKind::C2cEndpoint) {
                const auto packet = decode_c2c_raw_word(
                    command, InstructionKind::C2cEndpoint);
                const auto rx = queue.kind == QueueKind::C2cRx;
                std::size_t lane = 0;
                std::size_t fabricStream = 0;
                std::size_t vectorCount = 0;
                std::uint32_t syncTag = 0;
                if (rx) {
                    const auto instruction =
                        C2cIcuPacketCodec::decode_rx(packet);
                    lane = instruction.lane;
                    fabricStream = instruction.fabric_stream;
                    vectorCount = instruction.vector_count;
                    syncTag = instruction.sync_tag;
                } else {
                    const auto instruction =
                        C2cIcuPacketCodec::decode_tx(packet);
                    lane = instruction.lane;
                    fabricStream = instruction.fabric_stream;
                    vectorCount = instruction.vector_count;
                    syncTag = instruction.sync_tag;
                }
                std::ostringstream detail;
                detail << "lane=" << lane
                       << " sr=" << fabricStream
                       << " vectors=" << vectorCount
                       << " sync=" << syncTag;
                const auto side = queue.index == 0 ? "E" : "W";
                write_event(output, cursor,
                    cursor + static_cast<std::int64_t>(
                        vectorCount),
                    {std::string("C2C.") + side
                            + (rx ? ".RX" : ".TX"),
                        detail.str()});
                ++cursor;
                previous = nullptr;
                history.clear();
                continue;
            }
            if (is_fu_3d_raw_packet_header(command)) {
                cursor = write_raw_fu_packet(
                    output, queue, commandIndex, cursor,
                    program.hardware.mxms_per_hemisphere);
                commandIndex += fu_3d_raw_packet_word_count(queue.kind) - 1;
                previous = nullptr;
                history.clear();
                continue;
            }
            if (queue.kind == QueueKind::Mem
                && is_mem_synchronized_raw_word_command(command))
                throw std::logic_error(
                    "schedule trace found an orphan MEM_WRITE_SYNC word");
            if (is_fu_3d_raw_word_command(command))
                throw std::logic_error(
                    "runtime trace found an orphan raw FU packet word");
            if (is_vxm_stream_nd_command(command)) {
                const auto descriptor =
                    decode_vxm_stream_nd_command(command);
                const auto& stream = descriptor.schedule;
                const auto depthCount = stream.rank > 2
                    ? stream.counts[2] : std::size_t {1};
                for (std::size_t depth = 0; depth < depthCount; ++depth) {
                    const auto outerCount = stream.rank > 1
                        ? stream.counts[1] : std::size_t {1};
                    const auto outerInterval = stream.rank > 1
                        ? stream.cycle_strides[1] : std::size_t {1};
                    write_pattern(output, queue, descriptor.instruction,
                        stream.start_cycle
                            + depth * stream.cycle_strides[2],
                        program.hardware.mxms_per_hemisphere,
                        program.hardware.sram_depth_rows,
                        {outerCount > 1 ? "repeat2d"
                                : stream.counts[0] > 1
                                ? "repeat" : "single",
                            stream.counts[0],
                            stream.cycle_strides[0], 0,
                            outerCount, outerInterval, 0, false,
                            IcuInductionTarget::None, 0});
                }
                std::size_t finalCycle = stream.start_cycle;
                for (std::size_t dimension = 0;
                     dimension < stream.rank; ++dimension)
                    finalCycle += (stream.counts[dimension] - 1)
                        * stream.cycle_strides[dimension];
                cursor = std::max(cursor, finalCycle + 1);
                continue;
            }
            if (is_sxm_tile_program_command(command)) {
                const auto tile = decode_sxm_tile_program_command(command);
                const auto& stream = tile.schedule;
                const auto depthCount = stream.rank > 2
                    ? stream.counts[2] : std::size_t {1};
                for (std::size_t depth = 0; depth < depthCount; ++depth) {
                    const auto outerCount = stream.rank > 1
                        ? stream.counts[1] : std::size_t {1};
                    const auto outerInterval = stream.rank > 1
                        ? stream.cycle_strides[1] : std::size_t {1};
                    write_pattern(output, queue, tile.instruction,
                        stream.start_cycle
                            + depth * stream.cycle_strides[2],
                        program.hardware.mxms_per_hemisphere,
                        program.hardware.sram_depth_rows,
                        {outerCount > 1 ? "repeat2d"
                                : stream.counts[0] > 1
                                ? "repeat" : "single",
                            stream.counts[0],
                            stream.cycle_strides[0], 0,
                            outerCount, outerInterval, 0, false,
                            IcuInductionTarget::None, 0});
                }
                std::size_t finalCycle = stream.start_cycle;
                for (std::size_t dimension = 0;
                     dimension < stream.rank; ++dimension)
                    finalCycle += (stream.counts[dimension] - 1)
                        * stream.cycle_strides[dimension];
                cursor = std::max(cursor, finalCycle + 1);
                continue;
            }
            if (is_mem_slice_program_command(command)) {
                const auto sliceProgram =
                    decode_mem_slice_program_command(command);
                for (const auto& body : sliceProgram.body) {
                    const auto encoded =
                        isa::encode_mem_instruction(body.instruction);
                    const QueueCommand native {
                        static_cast<isa::EncodedIcuCommand>(
                            isa::IcuCommandOpcode::Instruction),
                        InstructionKind::Mem,
                        static_cast<std::uint16_t>(
                            (encoded >> 32) == 0 ? 1 : 2),
                        {static_cast<std::uint32_t>(encoded),
                            static_cast<std::uint32_t>(encoded >> 32),
                            0, 0},
                    };
                    const auto& stream = sliceProgram.schedule;
                    const auto depthCount = stream.rank > 2
                        ? stream.counts[2] : std::size_t {1};
                    for (std::size_t depth = 0; depth < depthCount;
                         ++depth) {
                        const auto outerCount = stream.rank > 1
                            ? stream.counts[1] : std::size_t {1};
                        const auto outerInterval = stream.rank > 1
                            ? stream.cycle_strides[1] : std::size_t {1};
                        const auto outerStride = stream.rank > 1
                            ? body.operand_strides[1]
                            : std::int64_t {0};
                        write_pattern(output, queue, native,
                            stream.start_cycle + body.cycle_offset
                                + depth * stream.cycle_strides[2],
                            program.hardware.mxms_per_hemisphere,
                            program.hardware.sram_depth_rows,
                            {outerCount > 1 ? "repeat2d"
                                    : stream.counts[0] > 1
                                    ? "repeat" : "single",
                                stream.counts[0],
                                stream.cycle_strides[0],
                                body.operand_strides[0], outerCount,
                                outerInterval, outerStride, false,
                                IcuInductionTarget::MemAddress,
                                static_cast<std::int64_t>(depth)
                                    * body.operand_strides[2]});
                    }
                    std::size_t finalCycle = stream.start_cycle
                        + body.cycle_offset;
                    for (std::size_t dimension = 0;
                         dimension < stream.rank; ++dimension)
                        finalCycle += (stream.counts[dimension] - 1)
                            * stream.cycle_strides[dimension];
                    cursor = std::max(cursor, finalCycle + 1);
                }
                continue;
            }
            if (is_mem_stream_nd_command(command)
                || is_mxm_stream_nd_command(command)) {
                const auto stream = is_mem_stream_nd_command(command)
                    ? decode_mem_stream_nd_command(command)
                    : decode_mxm_stream_nd_command(command);
                const auto depthCount = stream.rank > 2
                    ? stream.counts[2] : std::size_t {1};
                for (std::size_t depth = 0; depth < depthCount; ++depth) {
                    const auto outerCount = stream.rank > 1
                        ? stream.counts[1] : std::size_t {1};
                    const auto outerInterval = stream.rank > 1
                        ? stream.cycle_strides[1] : std::size_t {1};
                    const auto outerStride = stream.rank > 1
                        ? stream.operand_strides[1] : std::int64_t {0};
                    write_pattern(output, queue, command,
                        stream.start_cycle
                            + depth * stream.cycle_strides[2],
                        program.hardware.mxms_per_hemisphere,
                        program.hardware.sram_depth_rows,
                        {outerCount > 1 ? "repeat2d"
                                : stream.counts[0] > 1
                                ? "repeat" : "single",
                            stream.counts[0],
                            stream.cycle_strides[0],
                            stream.operand_strides[0], outerCount,
                            outerInterval, outerStride, false,
                            stream.induction_target,
                            static_cast<std::int64_t>(depth)
                                * stream.operand_strides[2]});
                }
                std::size_t finalCycle = stream.start_cycle;
                for (std::size_t dimension = 0;
                     dimension < stream.rank; ++dimension)
                    finalCycle += (stream.counts[dimension] - 1)
                        * stream.cycle_strides[dimension];
                cursor = std::max(cursor, finalCycle + 1);
                continue;
            }
            if (is_macro_schedule_command(command)) {
                const auto macro = decode_macro_schedule_command(command);
                write_pattern(output, queue, command, macro.start_cycle,
                    program.hardware.mxms_per_hemisphere,
                    program.hardware.sram_depth_rows,
                    {macro.outer_count > 1 ? "repeat2d"
                            : macro.inner_count > 1 ? "repeat" : "single",
                        macro.inner_count, macro.inner_interval,
                        macro.inner_stride, macro.outer_count,
                        macro.outer_interval, macro.outer_stride, false,
                        macro.induction_target, 0});
                cursor = std::max(cursor, macro.start_cycle
                    + (macro.outer_count - 1) * macro.outer_interval
                    + (macro.inner_count - 1) * macro.inner_interval + 1);
                continue;
            }
            if (is_repeat_2d_command(command)) {
                if (!previous)
                    throw std::logic_error(
                        "runtime trace found Repeat2D without instruction");
                const auto repeat = decode_repeat_2d_command(command);
                if (repeat.outer_count * repeat.inner_count > 1)
                    write_pattern(output, queue, *previous, previous_cycle,
                        program.hardware.mxms_per_hemisphere,
                        program.hardware.sram_depth_rows,
                        {"repeat2d", repeat.inner_count,
                            repeat.inner_interval, repeat.inner_stride,
                            repeat.outer_count, repeat.outer_interval,
                            repeat.outer_stride, true,
                            repeat.induction_target, 0});
                cursor = previous_cycle
                    + (repeat.outer_count - 1) * repeat.outer_interval
                    + (repeat.inner_count - 1) * repeat.inner_interval + 1;
                continue;
            }
            const auto opcode = isa::decode_icu_command_opcode(command.command);
            if (opcode == isa::IcuCommandOpcode::Nop) {
                cursor += isa::decode_icu_nop_cycles(command.command);
                continue;
            }
            if (opcode == isa::IcuCommandOpcode::Repeat) {
                if (!previous) throw std::logic_error("runtime trace found Repeat without instruction");
                const auto repeat = isa::decode_icu_repeat(command.command);
                if (repeat.count != 0) {
                    const auto first = previous_cycle + repeat.interval;
                    const auto last = previous_cycle + repeat.count * repeat.interval;
                    write_pattern(output, queue, *previous, first,
                        program.hardware.mxms_per_hemisphere,
                        program.hardware.sram_depth_rows,
                        {"repeat", repeat.count, repeat.interval,
                            repeat.address_stride, 1, 0, 0, false,
                            IcuInductionTarget::None,
                            repeat.address_stride});
                    cursor = last + 1;
                }
                continue;
            }
            if (opcode != isa::IcuCommandOpcode::Instruction)
                throw std::logic_error("runtime trace found unsupported ICU command");
            const auto event = describe(
                queue, command, 0, program.hardware.mxms_per_hemisphere,
                program.hardware.sram_depth_rows);
            write_event(output, cursor,
                cursor + static_cast<std::int64_t>(
                    event_duration(queue, command)),
                event);
            previous = &command;
            previous_cycle = cursor;
            history.emplace_back(&command, cursor);
            if (history.size() > 63) history.pop_front();
            ++cursor;
        }
    }
}

void write_schedule_trace_csv(
    const BinaryProgram& program, const std::filesystem::path& path)
{
    write_schedule_trace_csv(program, path, {});
}

} // namespace ftlpu::software::runtime
