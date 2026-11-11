#pragma once
#include "ftlpu/routing/control_unit.hpp"
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/system/stream_topology_builder.hpp"
#include "ftlpu/vxm/route_topk.hpp"

#include <map>

namespace ftlpu::software::runtime {

// Builds device commands only. It never evaluates scores or reads a route plan.
class RoutingProgramBuilder {
public:
  explicit RoutingProgramBuilder(BinaryProgram &program) : program_(program) {
    program_.hardware.routing_isa_version =
        std::max(1u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
  }
  std::uint32_t select(RouteSelectDescriptor d, std::uint32_t completion) {
    RoutingDescriptor x;
    x.select = std::move(d);
    return append(std::move(x), completion);
  }
  // Emits ordinary MEM and SXM operations. The returned first bundle cycle
  // leaves the packer drained before the VXM takes ownership of its inputs.
  RouteVxmSelectSchedule pack_vxm_scores(const RouteSelectDescriptor& select,
                                         RouteVxmSelectSchedule schedule,
                                         std::size_t start_cycle) {
    RoutingDescriptor descriptor;
    descriptor.opcode = RoutingOpcode::SelectVxm;
    descriptor.select = select;
    descriptor.vxm_select = schedule;
    descriptor.vxm_select.first_bundle_cycle =
        std::max<std::uint32_t>(4, schedule.first_bundle_cycle);
    RoutingControlUnit::validate_descriptor(descriptor, program_.hardware.sram_depth_rows);
    if (hw::kLanesPerTile != schedule.chains || hw::kTileRows != 4)
      throw std::invalid_argument("VXM scoring packer requires the lpu32 tile geometry");

    const auto layout = make_configured_stream_layout();
    const auto east = StreamDirection::East, west = StreamDirection::West;
    const auto pitch = schedule.workspace_blocks(select.experts);
    const auto batches = (select.tokens + VxmRouteTopK::lanes - 1) / VxmRouteTopK::lanes;
    std::map<std::size_t, std::vector<QueueCommand>> mem;
    IcuProgram sxm;
    auto cycle = start_cycle + hw::kSystemStreamRegisterColumns + 4;
    const auto emit_mem = [&](const RoutingPlane& p, std::size_t issue,
                              MemInstruction instruction) {
      const auto index = InstructionControlUnit::mem_queue(
          static_cast<Hemisphere>(p.hemisphere), p.slice, p.bank);
      IcuMacroSchedule op;
      op.start_cycle = issue;
      op.induction_target = IcuInductionTarget::MemAddress;
      mem[index].push_back(mem_macro(instruction, op));
    };
    SxmInstruction::StreamList transpose_streams, write_streams;
    for (std::size_t i = 0; i < 2 * hw::kLanesPerTile; ++i) {
      transpose_streams.push_back({StreamId::East(16 + i).packed()});
      write_streams.push_back({StreamId::West(i).packed()});
    }
    const auto pack = [&](const RoutingTensorView& source, std::size_t offset) {
      for (const auto& src : source.planes)
        for (const auto& dst : schedule.score_planes)
          if (src.hemisphere == dst.hemisphere && src.slice == dst.slice &&
              src.bank == dst.bank && src.base_row < dst.base_row + offset + batches * pitch &&
              dst.base_row + offset <= source.row(source.rows - 1, source.columns - 1,
                                                  &src - source.planes.data()))
            throw std::invalid_argument("VXM scoring workspace overlaps its source");
      for (std::size_t p = 0; p < 4; p += 2) {
        const auto side = static_cast<Hemisphere>(source.planes[p].hemisphere);
        if (source.planes[p + 1].hemisphere != source.planes[p].hemisphere)
          throw std::invalid_argument("SXM scoring pack requires co-located byte pairs");
        for (std::size_t j = 0; j < schedule.chains; ++j)
          for (std::size_t byte = 0; byte < 2; ++byte)
            if (schedule.score_planes[j * 4 + p + byte].hemisphere !=
                source.planes[p].hemisphere)
              throw std::invalid_argument("SXM scoring pack requires local workspace banks");
        for (std::size_t t0 = 0; t0 < select.tokens; t0 += hw::kLanesPerTile) {
          const auto batch = t0 / VxmRouteTopK::lanes;
          const auto tile = (t0 % VxmRouteTopK::lanes) / hw::kLanesPerTile;
          for (std::size_t e0 = 0; e0 < select.experts; e0 += hw::kPhysicalVectorBytes) {
            for (std::size_t r = 0; r < hw::kLanesPerTile; ++r) {
              auto transpose = SxmInstruction::Transpose(
                  {{StreamId::East(0).packed()}, {StreamId::East(1).packed()}},
                  transpose_streams);
              transpose.input_row = r;
              sxm.emit_sxm_transpose(cycle, side, transpose);
              for (std::size_t byte = 0; byte < 2; ++byte) {
                const auto& plane = source.planes[p + byte];
                const auto latency = stream_distance(layout.mem_ports.output_column(plane.slice, east),
                    layout.sxm_ports.input_column(east), east) + 1;
                const auto token = std::min(t0 + r, std::size_t(select.tokens - 1));
                emit_mem(plane, cycle - latency,
                    MemInstruction::Read(source.row(token, e0, p + byte), StreamId::East(byte)));
              }
              cycle += 4;
            }
            cycle += hw::kTileRows;
            for (std::size_t src_tile = 0; src_tile < hw::kTileRows; ++src_tile) {
              auto map = Permute320::identity_map();
              for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                std::swap(map[tile * hw::kLanesPerTile + lane],
                          map[src_tile * hw::kLanesPerTile + lane]);
              auto permute = SxmInstruction::Permute(transpose_streams, write_streams, map);
              permute.output_tile = tile;
              sxm.emit_sxm_permute(cycle, side, permute);
              for (std::size_t j = 0; j < schedule.chains; ++j)
                for (std::size_t byte = 0; byte < 2; ++byte) {
                  const auto& plane = schedule.score_planes[j * 4 + p + byte];
                  const auto row = plane.base_row + offset + batch * pitch +
                                   e0 / schedule.chains + src_tile;
                  const auto latency = stream_distance(layout.sxm_ports.output_column(west),
                      layout.mem_ports.input_column(plane.slice, west), west) + 1;
                  emit_mem(plane, cycle + latency - tile,
                      MemInstruction::WriteTap(row, StreamId::West(2 * j + byte)));
                }
              cycle += hw::kTileRows + 2;
            }
            // Writes to the preceding block can share a bank with the
            // source of the next block, whose read is issued before capture.
            cycle += 2 * hw::kSystemStreamRegisterColumns + hw::kTileRows + 4;
          }
        }
      }
    };
    pack(select.choice_scores, 0);
    if (select.eligibility.present()) pack(select.eligibility, schedule.eligibility_offset);
    if (select.raw_scores.present() && select.raw_scores != select.choice_scores)
      pack(select.raw_scores, schedule.raw_offset);
    auto sxm_queues = sxm.encode_queues();
    for (const auto& q : sxm_queues) {
      if (q.commands.empty()) continue;
      for (const auto& existing : program_.queues)
        if (existing.kind == q.kind && existing.index == q.index && !existing.commands.empty())
          throw std::invalid_argument("VXM score packing requires a dedicated SXM schedule");
    }
    for (const auto& [index, commands] : mem) validate_macro_queue(index);
    cycle += hw::kSystemStreamRegisterColumns + 4;
    if (cycle > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("VXM pack schedule cycle overflow");
    for (auto& q : sxm_queues)
      if (!q.commands.empty()) queue(q.kind, q.index) = std::move(q);
    for (const auto& [index, commands] : mem) append_mem_macros(index, commands);
    schedule.first_bundle_cycle = std::max(schedule.first_bundle_cycle,
                                          static_cast<std::uint32_t>(cycle));
    return schedule;
  }

  std::uint32_t select_vxm(RouteSelectDescriptor select, RouteVxmSelectSchedule schedule,
                           std::uint32_t completion) {
    RoutingDescriptor descriptor;
    descriptor.opcode = RoutingOpcode::SelectVxm;
    descriptor.select = std::move(select);
    descriptor.vxm_select = schedule;
    RoutingControlUnit::validate_descriptor(descriptor, program_.hardware.sram_depth_rows);
    const auto layout = make_configured_stream_layout();
    const auto direction = layout.vxm_input.direction;
    const auto blocks = (descriptor.select.experts + schedule.chains - 1) / schedule.chains;
    const auto pitch = schedule.workspace_blocks(descriptor.select.experts);
    const auto batches = (descriptor.select.tokens + VxmRouteTopK::lanes - 1) / VxmRouteTopK::lanes;
    const bool mask = descriptor.select.eligibility.present();
    const bool raw = descriptor.select.raw_scores.present() &&
                     descriptor.select.raw_scores != descriptor.select.choice_scores;
    std::array<std::vector<QueueCommand>, RouteVxmSelectSchedule::streams> commands;
    std::array<std::size_t, RouteVxmSelectSchedule::streams> queues{};
    std::array<std::size_t, RouteVxmSelectSchedule::streams> latency{};
    for (std::size_t stream = 0; stream < schedule.streams; ++stream) {
      const auto& plane = schedule.score_planes[stream];
      queues[stream] = InstructionControlUnit::mem_queue(
          static_cast<Hemisphere>(plane.hemisphere), plane.slice, plane.bank);
      const auto column = layout.mem_ports.output_column(plane.slice, direction);
      latency[stream] = stream_distance(column, layout.vxm_input.column, direction) + 1;
      validate_macro_queue(queues[stream]);
    }

    auto arrival = std::size_t(schedule.first_bundle_cycle);
    for (std::size_t batch = 0; batch < batches; ++batch) {
      for (std::size_t rank = 0; rank < descriptor.select.k; ++rank) {
        std::array<std::size_t, 3> offsets{};
        std::size_t parts = 0;
        if (mask) offsets[parts++] = schedule.eligibility_offset;
        if (raw && rank == 0) offsets[parts++] = schedule.raw_offset;
        offsets[parts++] = 0;
        for (std::size_t stream = 0; stream < schedule.streams; ++stream) {
          const auto sr = direction == StreamDirection::East ? StreamId::East(stream)
                                                             : StreamId::West(stream);
          for (std::size_t i = 0; i < parts; ++i) {
            const auto bundle = arrival + i * schedule.bundle_interval;
            if (bundle < latency[stream])
              throw std::invalid_argument("VXM scoring window precedes MEM latency");
            IcuMacroSchedule read;
            read.start_cycle = bundle - latency[stream];
            read.inner_count = blocks;
            read.inner_interval = parts * schedule.bundle_interval;
            read.inner_stride = 1;
            read.induction_target = IcuInductionTarget::MemAddress;
            const auto row = schedule.score_planes[stream].base_row + batch * pitch + offsets[i];
            commands[stream].push_back(mem_macro(MemInstruction::Read(row, sr), read));
          }
        }
        arrival += blocks * parts * schedule.bundle_interval + VxmRouteTopK::merge_input_gap;
      }
      const auto valid_rows = std::min(VxmRouteTopK::lanes,
          std::size_t(descriptor.select.tokens) - batch * VxmRouteTopK::lanes);
      arrival += 2 * valid_rows * descriptor.select.k + hw::kTileRows + 4;
    }

    const auto index = append(std::move(descriptor), completion);
    for (std::size_t stream = 0; stream < schedule.streams; ++stream)
      append_mem_macros(queues[stream], commands[stream]);
    program_.max_cycle = std::max(program_.max_cycle, arrival);
    program_.hardware.routing_isa_version = std::max(6u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
    return index;
  }

  // Read the rank-major SRAM results of a separately issued VXM TOPK.
  // earliest_cycle comes from the producer's MEM write schedule.
  std::uint32_t import_topk(RouteSelectDescriptor select, std::uint32_t earliest_cycle,
                            std::uint32_t completion) {
    RoutingDescriptor descriptor;
    descriptor.opcode = RoutingOpcode::ImportTopK;
    descriptor.select = std::move(select);
    descriptor.import_after_cycle = earliest_cycle;
    const auto index = append(std::move(descriptor), completion);
    program_.hardware.routing_isa_version = std::max(7u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
    return index;
  }

  std::uint32_t plan(RoutePlanDescriptor d, std::uint32_t completion) {
    RoutingDescriptor x;
    x.opcode = RoutingOpcode::Plan;
    x.plan = std::move(d);
    return append(std::move(x), completion);
  }
  std::uint32_t address(RouteAddressDescriptor d, std::uint32_t completion) {
    RoutingDescriptor x;
    x.opcode = RoutingOpcode::Address;
    x.address = std::move(d);
    return append(std::move(x), completion);
  }
  // Registers a reusable address template without issuing a ROUTE_ADDR.
  std::uint32_t address_template(RouteAddressDescriptor d) {
    RoutingDescriptor x;
    x.opcode = RoutingOpcode::Address;
    x.address = std::move(d);
    RoutingControlUnit::validate_descriptor(x,
                                            program_.hardware.sram_depth_rows);
    if (program_.routing_descriptors.size() >= RoutingConfig::descriptor_depth)
      throw std::invalid_argument("routing descriptor capacity");
    const auto index =
        static_cast<std::uint32_t>(program_.routing_descriptors.size());
    program_.routing_descriptors.push_back(std::move(x));
    return index;
  }
  std::uint32_t dispatch(RouteTaskDispatchDescriptor d,
                         std::uint32_t completion) {
    if (d.address_template_descriptor_id >= program_.routing_descriptors.size())
      throw std::invalid_argument("missing dispatch template");
    const auto &a =
        program_.routing_descriptors[d.address_template_descriptor_id];
    if (a.opcode != RoutingOpcode::Address ||
        a.address.context_id != d.context_id ||
        a.address.max_rows < d.block_rows)
      throw std::invalid_argument("invalid dispatch template");
    RoutingDescriptor x;
    x.opcode = RoutingOpcode::Dispatch;
    x.dispatch = d;
    const auto index = append(std::move(x), completion);
    program_.hardware.routing_isa_version =
        std::max(2u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
    return index;
  }
  std::uint32_t weight_table(std::uint32_t workset,
                             const std::vector<ExpertWeightEntry> &entries,
                             const std::vector<ExpertWeightSegment> &segments) {
    RoutingDescriptor d;
    d.opcode = RoutingOpcode::WeightTable;
    d.weight_table = make_expert_weight_table(workset, entries, segments);
    RoutingControlUnit::validate_descriptor(d,
                                            program_.hardware.sram_depth_rows);
    if (program_.routing_descriptors.size() >= RoutingConfig::descriptor_depth)
      throw std::invalid_argument("routing descriptor capacity");
    auto candidate = program_.routing_descriptors;
    candidate.push_back(d);
    validate_routing_descriptor_references(
        candidate, std::max(3u, program_.hardware.routing_isa_version));
    (void)encode_routing_descriptors(candidate);
    const auto index =
        static_cast<std::uint32_t>(program_.routing_descriptors.size());
    program_.routing_descriptors.push_back(std::move(d));
    program_.hardware.routing_isa_version =
        std::max(3u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
    return index;
  }
  std::uint32_t expert_template(ExpertMacroTemplateDescriptor descriptor) {
    RoutingDescriptor d;
    d.opcode = RoutingOpcode::ExpertTemplate;
    d.expert_template = std::move(descriptor);
    RoutingControlUnit::validate_descriptor(d,
                                            program_.hardware.sram_depth_rows);
    auto candidate = program_.routing_descriptors;
    candidate.push_back(d);
    validate_routing_descriptor_references(
        candidate, std::max(4u, program_.hardware.routing_isa_version));
    (void)encode_routing_descriptors(candidate);
    const auto index =
        static_cast<std::uint32_t>(program_.routing_descriptors.size());
    program_.routing_descriptors.push_back(std::move(d));
    program_.hardware.routing_isa_version =
        std::max(4u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
    return index;
  }
  std::uint32_t combine(RouteCombineDescriptor descriptor,
                        std::uint32_t completion) {
    RoutingDescriptor d;
    d.opcode = RoutingOpcode::Combine;
    d.combine = std::move(descriptor);
    auto candidate = program_.routing_descriptors;
    candidate.push_back(d);
    validate_routing_descriptor_references(candidate, std::max(5u, program_.hardware.routing_isa_version));
    (void)encode_routing_descriptors(candidate);
    const auto index = append(std::move(d), completion);
    program_.hardware.routing_isa_version = std::max(5u, program_.hardware.routing_isa_version);
    program_.target_abi = executable_target_abi(program_.hardware);
    return index;
  }
  void completion_event(std::uint32_t event,
                        std::vector<std::uint32_t> instances) {
    if (!event || instances.empty() ||
        std::ranges::any_of(program_.queue_sync_plan.events,
                            [&](const auto &e) { return e.event_id == event; }))
      throw std::invalid_argument("duplicate/empty routing event");
    program_.queue_control_protocol = true;
    program_.queue_sync_plan.events.push_back(
        {event, QueueEventSource::Routing, std::move(instances)});
  }
  void wait_for(QueueKind kind, std::uint16_t index, std::uint32_t event) {
    auto &q = queue(kind, index);
    const auto command = static_cast<std::uint32_t>(q.commands.size());
    q.commands.push_back(
        encode_icu_control_raw_word(IcuControlInstruction::WaitEvent(event)));
    auto &parts = program_.queue_sync_plan.partitions;
    auto it = std::ranges::find_if(parts, [&](const auto &p) {
      return p.kind == kind && p.index == index;
    });
    if (it == parts.end()) {
      parts.push_back({kind, index, {}});
      it = std::prev(parts.end());
    }
    BinaryQueueControl c;
    c.opcode = IcuControlOpcode::WaitEvent;
    c.native_binding = true;
    c.operand = event;
    c.source_command = command;
    it->entries.push_back(c);
    program_.max_cycle = std::max<std::size_t>(program_.max_cycle, 1);
  }

private:
  static std::size_t stream_distance(std::size_t source, std::size_t destination,
                                     StreamDirection direction) {
    if ((direction == StreamDirection::East && source > destination) ||
        (direction == StreamDirection::West && source < destination))
      throw std::invalid_argument("routing workspace is outside its stream path");
    return source < destination ? destination - source : source - destination;
  }

  static QueueCommand mem_macro(MemInstruction instruction, const IcuMacroSchedule& schedule) {
    const auto bits = isa::encode_mem_instruction(instruction);
    return encode_macro_schedule_command(
        {static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
         InstructionKind::Mem, static_cast<std::uint16_t>(bits >> 32 ? 2 : 1),
         {std::uint32_t(bits), std::uint32_t(bits >> 32), 0, 0}}, schedule);
  }

  void validate_macro_queue(std::size_t index) const {
    for (const auto& q : program_.queues) {
      if (q.kind != QueueKind::Mem || q.index != index) continue;
      if (q.packed_macro_imem || !std::ranges::all_of(q.commands, is_macro_schedule_command))
        throw std::invalid_argument("VXM scoring requires an editable MEM macro queue");
    }
  }

  void append_mem_macros(std::size_t index, const std::vector<QueueCommand>& commands) {
    auto& q = queue(QueueKind::Mem, index);
    q.commands.insert(q.commands.end(), commands.begin(), commands.end());
    std::stable_sort(q.commands.begin(), q.commands.end(), [](const auto& a, const auto& b) {
      return decode_macro_schedule_command(a).start_cycle <
             decode_macro_schedule_command(b).start_cycle;
    });
    q.requires_raw_macro = true;
  }

  QueueProgram &queue(QueueKind kind, std::uint16_t index) {
    auto it = std::ranges::find_if(program_.queues, [&](const auto &q) {
      return q.kind == kind && q.index == index;
    });
    if (it == program_.queues.end()) {
      program_.queues.push_back({kind, index, {}});
      return program_.queues.back();
    }
    return *it;
  }
  std::uint32_t append(RoutingDescriptor d, std::uint32_t completion) {
    RoutingControlUnit::validate_descriptor(d,
                                            program_.hardware.sram_depth_rows);
    if (program_.routing_descriptors.size() >= RoutingConfig::descriptor_depth)
      throw std::invalid_argument("routing descriptor capacity");
    const auto index =
        static_cast<std::uint32_t>(program_.routing_descriptors.size());
    const auto word = encode_routing_instruction({d.opcode, index, completion});
    auto &q = queue(QueueKind::Routing, 0);
    if (q.commands.size() >= RoutingConfig::imem_depth)
      throw std::invalid_argument("routing i-MEM capacity");
    for (const auto &c : q.commands)
      if (c.instruction_kind == InstructionKind::Routing &&
          c.words[2] == completion)
        throw std::invalid_argument("duplicate Routing completion instance");
    q.commands.push_back({0,
                          InstructionKind::Routing,
                          3,
                          {word.lanes[0], word.lanes[1], word.lanes[2], 0}});
    program_.routing_descriptors.push_back(std::move(d));
    program_.max_cycle = std::max<std::size_t>(program_.max_cycle, 1);
    return index;
  }
  BinaryProgram &program_;
};
} // namespace ftlpu::software::runtime
