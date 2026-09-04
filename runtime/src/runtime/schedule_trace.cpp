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
        for (const auto& command : queue.commands) {
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
