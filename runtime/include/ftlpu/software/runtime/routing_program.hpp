#pragma once
#include "ftlpu/routing/control_unit.hpp"
#include "ftlpu/software/runtime/binary.hpp"

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
    validate_routing_descriptor_references(candidate, 5);
    (void)encode_routing_descriptors(candidate);
    const auto index = append(std::move(d), completion);
    program_.hardware.routing_isa_version = 5;
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
