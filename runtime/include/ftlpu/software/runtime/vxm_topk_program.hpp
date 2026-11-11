#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/system/stream_topology_builder.hpp"

#include <map>

namespace ftlpu::software::runtime {

// The input lane is a row of the logical matrix, and stream 4*c+b
// contains byte b of column c within a block of eight columns. Output
// lanes keep the same row mapping, with one SRAM row per output rank.
struct VxmTopKWorkspace {
    std::array<RoutingPlane, RouteVxmSelectSchedule::streams> input{};
    std::uint32_t input_batch_stride{}; // zero uses the tight block count
    std::uint32_t mask_offset{};
    std::array<RoutingPlane, 4> indices{}, values{};
};

// Emits one TOPK and the ordinary MEM Macros that supply its packed input
// and store its outputs. A caller assembling its own stream schedule can
// use encode_vxm_topk_command directly.
inline std::uint32_t append_vxm_topk(BinaryProgram& program, VxmTopKInstruction instruction,
    const VxmTopKWorkspace& workspace, std::size_t first_bundle_cycle,
    std::size_t bundle_interval = 1)
{
    const auto command = encode_vxm_topk_command(instruction);
    if (program.hardware.vxm_alus != hw::kVxmAluCount || !bundle_interval ||
        bundle_interval > 255 || first_bundle_cycle > UINT32_MAX)
        throw std::invalid_argument("invalid VXM TOPK hardware or bundle schedule");

    const auto layout = make_configured_stream_layout();
    const auto blocks = (instruction.columns + VxmRouteTopK::chains - 1) / VxmRouteTopK::chains;
    const auto batches = (instruction.rows + VxmRouteTopK::lanes - 1) / VxmRouteTopK::lanes;
    const auto stride = workspace.input_batch_stride ? workspace.input_batch_stride : blocks;
    if (stride < blocks)
        throw std::invalid_argument("VXM TOPK input batches overlap");
    const auto input_rows = (batches - 1) * stride + blocks;
    const auto parts = instruction.has_mask ? 2u : 1u;
    if (instruction.has_mask ? workspace.mask_offset < input_rows : workspace.mask_offset != 0)
        throw std::invalid_argument("invalid VXM TOPK mask offset");

    using Macros = std::map<std::size_t, std::vector<QueueCommand>>;
    Macros mem;
    const auto reserve = [&](const RoutingPlane& plane, std::uint64_t rows) {
        if (plane.hemisphere >= hw::kHemispheres || plane.slice >= hw::kMemSliceColumns ||
            plane.bank >= hw::kMemBanksPerSlice || plane.base_row + rows > program.hardware.sram_depth_rows)
            throw std::invalid_argument("invalid VXM TOPK SRAM plane");
        const auto queue = InstructionControlUnit::mem_queue(
            static_cast<Hemisphere>(plane.hemisphere), plane.slice, plane.bank);
        if (!mem.try_emplace(queue).second)
            throw std::invalid_argument("VXM TOPK workspace requires independent SRAM banks");
        return queue;
    };
    const auto distance = [](std::size_t from, std::size_t to, StreamDirection direction) {
        if ((from > to && direction == StreamDirection::East) ||
            (from < to && direction == StreamDirection::West))
            throw std::invalid_argument("VXM TOPK SRAM plane is outside its SR path");
        return from > to ? from - to : to - from;
    };
    const auto stream_id = [](std::size_t stream, StreamDirection direction) {
        return direction == StreamDirection::East ? StreamId::East(stream) : StreamId::West(stream);
    };
    const auto emit = [&](std::size_t queue, std::size_t cycle, std::size_t count,
                          std::size_t interval, MemInstruction operation) {
        IcuMacroSchedule schedule;
        schedule.start_cycle = cycle;
        schedule.inner_count = count;
        schedule.inner_interval = interval;
        schedule.inner_stride = 1;
        schedule.induction_target = IcuInductionTarget::MemAddress;
        const auto bits = isa::encode_mem_instruction(operation);
        mem.at(queue).push_back(encode_macro_schedule_command(
            {0, InstructionKind::Mem, static_cast<std::uint16_t>(bits >> 32 ? 2 : 1),
             {std::uint32_t(bits), std::uint32_t(bits >> 32)}}, schedule));
    };

    std::array<std::size_t, RouteVxmSelectSchedule::streams> read_queues{}, read_latencies{};
    for (std::size_t stream = 0; stream < workspace.input.size(); ++stream) {
        const auto& plane = workspace.input[stream];
        const auto group = stream / VxmLane::kStreamGroupBytes;
        if (plane.hemisphere != ((instruction.west_input_groups >> group) & 1))
            throw std::invalid_argument("VXM TOPK input plane does not match the group source");
        read_queues[stream] = reserve(plane, std::uint64_t(workspace.mask_offset) + input_rows);
        read_latencies[stream] = distance(layout.mem_ports.output_column(plane.slice, layout.vxm_input.direction),
            layout.vxm_input.column, layout.vxm_input.direction) + 1;
        if (first_bundle_cycle < read_latencies[stream] || first_bundle_cycle == 0)
            throw std::invalid_argument("VXM TOPK input precedes MEM latency");
    }

    struct OutputPlane {
        std::size_t queue, latency, stream, row;
    };
    std::array<OutputPlane, 8> outputs{};
    for (std::size_t byte = 0; byte < 4; ++byte) {
        for (const auto base : {VxmTopKInstruction::index_stream, VxmTopKInstruction::value_stream}) {
            const auto& plane = base == VxmTopKInstruction::index_stream
                ? workspace.indices[byte] : workspace.values[byte];
            // Runtime's standard mux sends both four-byte result groups west.
            if (plane.hemisphere != hemisphere_index(Hemisphere::West))
                throw std::invalid_argument("VXM TOPK builder outputs use the west MEM fabric");
            const auto queue = reserve(plane, batches * instruction.k);
            const auto latency = distance(layout.vxm_output.column,
                layout.mem_ports.input_column(plane.slice, layout.vxm_output.direction),
                layout.vxm_output.direction) + 1;
            outputs[base + byte] = {queue, latency, base + byte, plane.base_row};
        }
    }
    auto arrival = first_bundle_cycle;
    auto end = std::size_t{0};
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t rank = 0; rank < instruction.k; ++rank) {
            for (std::size_t stream = 0; stream < workspace.input.size(); ++stream) {
                const auto row = workspace.input[stream].base_row + batch * stride;
                const auto issue = arrival - read_latencies[stream];
                const auto sr = stream_id(stream, layout.vxm_input.direction);
                if (instruction.has_mask)
                    emit(read_queues[stream], issue, blocks, parts * bundle_interval,
                         MemInstruction::Read(row + workspace.mask_offset, sr));
                emit(read_queues[stream], issue + (parts - 1) * bundle_interval,
                     blocks, parts * bundle_interval, MemInstruction::Read(row, sr));
            }
            arrival += blocks * parts * bundle_interval;
            if (rank + 1 != instruction.k) arrival += VxmRouteTopK::merge_input_gap;
        }
        const auto first_output = arrival - bundle_interval
            + (hw::kTileRows - 1) + VxmRouteTopK::scan_to_result_cycles;
        for (const auto& plane : outputs) {
            const auto issue = first_output + plane.latency;
            emit(plane.queue, issue, instruction.k, 1,
                MemInstruction::Write(plane.row + batch * instruction.k,
                    stream_id(plane.stream, layout.vxm_output.direction)));
            end = std::max(end, issue + instruction.k + hw::kTileRows + 1);
        }
        arrival = first_output + instruction.k + hw::kTileRows + 1;
    }
    if (end > UINT32_MAX)
        throw std::invalid_argument("VXM TOPK timeline exceeds 32 bits");

    for (const auto& queue : program.queues) {
        if (queue.kind == QueueKind::Vxm && queue.index == VxmTopKInstruction::queue)
            throw std::invalid_argument("VXM TOPK builder requires VXM queue 0");
        const auto next = mem.find(queue.index);
        if (queue.kind != QueueKind::Mem || next == mem.end()) continue;
        if (queue.packed_macro_imem || !std::ranges::all_of(queue.commands, is_macro_schedule_command))
            throw std::invalid_argument("VXM TOPK requires an editable MEM Macro schedule");
        const auto first = decode_macro_schedule_command(next->second.front()).start_cycle;
        for (const auto& existing : queue.commands) {
            const auto schedule = decode_macro_schedule_command(existing);
            const auto last = schedule.start_cycle + (schedule.outer_count - 1) * schedule.outer_interval
                + (schedule.inner_count - 1) * schedule.inner_interval;
            if (last >= first || first - last < hw::kTileRows)
                throw std::invalid_argument("VXM TOPK workspace is still in use by an earlier MEM Macro");
        }
    }

    QueueProgram vxm{QueueKind::Vxm, VxmTopKInstruction::queue, {}};
    if (first_bundle_cycle > 2)
        vxm.commands.push_back({isa::encode_icu_nop(first_bundle_cycle - 2)});
    vxm.commands.push_back(command);
    program.queues.push_back(std::move(vxm));
    for (auto& queue : program.queues) {
        if (queue.kind != QueueKind::Mem) continue;
        const auto next = mem.find(queue.index);
        if (next == mem.end()) continue;
        queue.commands.insert(queue.commands.end(), next->second.begin(), next->second.end());
        queue.requires_raw_macro = true;
        mem.erase(next);
    }
    for (auto& [index, commands] : mem) {
        QueueProgram queue{QueueKind::Mem, static_cast<std::uint16_t>(index), std::move(commands)};
        queue.requires_raw_macro = true;
        program.queues.push_back(std::move(queue));
    }
    program.max_cycle = std::max(program.max_cycle, end);
    return static_cast<std::uint32_t>(end);
}

} // namespace ftlpu::software::runtime
