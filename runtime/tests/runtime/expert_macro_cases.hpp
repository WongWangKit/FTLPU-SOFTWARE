#pragma once
#include "ftlpu/core/bf16.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

// Included by routing_runtime_test after its existing SRAM/assertion helpers.
template <class Instruction>
static rt::QueueCommand task_macro(Instruction instruction,
                                   IcuMacroSchedule schedule) {
  std::uint64_t bits{};
  rt::InstructionKind kind{};
  if constexpr (std::is_same_v<Instruction, MemInstruction>) {
    bits = isa::encode_mem_instruction(instruction);
    kind = rt::InstructionKind::Mem;
  } else {
    bits = isa::encode_mxm_instruction(instruction);
    kind = rt::InstructionKind::Mxm;
  }
  rt::QueueCommand c{
      static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
      kind,
      static_cast<std::uint16_t>((bits >> 32) ? 2 : 1),
      {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32),
       0, 0}};
  return rt::encode_macro_schedule_command(c, schedule);
}
static IcuMacroSchedule row_schedule(std::size_t cycle, std::size_t rows,
                                     std::int64_t stride,
                                     IcuInductionTarget target) {
  IcuMacroSchedule s;
  s.start_cycle = cycle;
  s.outer_count = rows;
  s.outer_stride = stride;
  s.induction_target = target;
  return s;
}
static void macro_case(bool dynamic, bool top2 = false) {
  const std::uint32_t tokens = top2      ? 3
                               : dynamic ? 42
                                         : 1,
                      rows = top2      ? 2
                             : dynamic ? 32
                                       : 1;
  const auto vxm_arrival = 120 + hw::kMxmBoundaryStreamRegisterColumn + 1;
  rt::BinaryProgram p;
  rt::RoutingProgramBuilder b(p);
  RouteSelectDescriptor select;
  select.tokens = tokens;
  select.experts = 4;
  select.k = top2 ? 2 : 1;
  select.choice_scores = view(RoutingElement::F32, tokens, 4, 28, 0);
  b.select(select, 1);
  RoutePlanDescriptor plan;
  if (top2)
    plan.weights = view(RoutingElement::F32, tokens, 2, 36, 0);
  b.plan(plan, 2);
  RouteAddressDescriptor a;
  a.max_rows = rows;
  a.source = view(RoutingElement::BF16, tokens, 32, 44, 0);
  a.destination = view(RoutingElement::BF16, rows, 32, 50, 64);
  auto address = b.address_template(a);
  b.dispatch({1, address, rows}, 3);
  std::vector<ExpertWeightSegment> segments;
  for (unsigned expert = 0; expert < 2; ++expert) {
    for (unsigned slice = 0; slice < 16; ++slice)
      segments.push_back(
          {0x200000 + segments.size() * 0x1000, 4, 0, slice, 1, 64, 30});
    for (unsigned slice = 4; slice < 6; ++slice)
      segments.push_back(
          {0x200000 + segments.size() * 0x1000, 1, 1, slice, 1, 64, 30});
  }
  b.weight_table(1, {{0, 18}, {0, 18}, {0, 18}, {18, 18}}, segments);
  ExpertMacroTemplateDescriptor t;
  t.output = view(RoutingElement::BF16, tokens * select.k, 32, 0, 0, 1);
  RouteCombineDescriptor combine;
  combine.source = t.output;
  combine.destination = view(RoutingElement::BF16, tokens, 32, 8, 0, 1);
  const auto count_binding = [](unsigned record) {
    return MacroTaskBinding{
        record, MacroBindingTarget::OuterCount, MacroTaskParameter::Rows, 1, 0,
        false};
  };
  const auto base_binding = [](MacroTaskParameter slot) {
    return MacroTaskBinding{0, MacroBindingTarget::Operand, slot, 1, 0, true};
  };
  const auto add = [&](rt::QueueKind kind, std::uint32_t index,
                       std::vector<rt::QueueCommand> commands,
                       std::vector<MacroTaskBinding> bindings,
                       std::uint32_t weight_base = 0) {
    rt::QueueProgram q{kind, static_cast<std::uint16_t>(index),
                       std::move(commands)};
    q.requires_raw_macro = true;
    p.queues.push_back(std::move(q));
    t.queues.push_back({static_cast<std::uint32_t>(kind), index,
                        std::move(bindings), weight_base});
  };
  for (unsigned slice = 0; slice < 16; ++slice) {
    IcuMacroSchedule schedule;
    schedule.start_cycle = 30 - (hw::kMxmBoundaryStreamRegisterColumn -
                                 slice / hw::kMemSlicesPerGroup);
    schedule.inner_count = 4;
    schedule.inner_stride = 1;
    schedule.induction_target = IcuInductionTarget::MemAddress;
    add(rt::QueueKind::Mem,
        static_cast<std::uint32_t>(
            InstructionControlUnit::mem_queue(Hemisphere::East, slice, 1)),
        {task_macro(
            MemInstruction::Read(dynamic ? 0 : 64, StreamId::East(slice)),
            schedule)},
        dynamic ? std::vector{base_binding(MacroTaskParameter::WeightBase)}
                : std::vector<MacroTaskBinding>{},
        64);
  }
  for (unsigned plane = 0; plane < 2; ++plane) {
    const auto slice = 50 + plane;
    const auto start = 50 - (hw::kMxmBoundaryStreamRegisterColumn -
                             slice / hw::kMemSlicesPerGroup);
    std::vector<MacroTaskBinding> bindings;
    if (dynamic)
      bindings = {count_binding(0),
                  base_binding(MacroTaskParameter::InputBase)};
    add(rt::QueueKind::Mem,
        static_cast<std::uint32_t>(
            InstructionControlUnit::mem_queue(Hemisphere::East, slice, 0)),
        {task_macro(
            MemInstruction::Read(dynamic ? 0 : 64, StreamId::East(plane)),
            row_schedule(start, rows, 1, IcuInductionTarget::MemAddress))},
        bindings);
    bindings.clear();
    if (dynamic)
      bindings = {count_binding(0),
                  base_binding(MacroTaskParameter::WeightBase)};
    add(rt::QueueKind::Mem,
        static_cast<std::uint32_t>(
            InstructionControlUnit::mem_queue(Hemisphere::West, 4 + plane, 1)),
        {task_macro(
            MemInstruction::Read(dynamic ? 0 : 64, StreamId::West(16 + plane)),
            row_schedule(vxm_arrival - 2, rows, 0,
                         IcuInductionTarget::MemAddress))},
        bindings, 64);
  }
  IcuMacroSchedule load;
  load.start_cycle = 30;
  load.inner_count = 4;
  load.inner_stride = 1;
  load.induction_target = IcuInductionTarget::MxmWeightColumn;
  add(rt::QueueKind::MxmLoad, 0,
      {task_macro(MxmControlInstruction::IWDirect16(0, 0), load)}, {});
  auto compute = MxmControlInstruction::Compute(
      0, 0, 0, 0, 1, MxmAccumulatorDestination::Sram, MxmDataFormat::BFloat16);
  auto read = MxmControlInstruction::AccumulatorRead(
      0, 0, true, MxmAccumulatorOutputFormat::BFloat16);
  add(rt::QueueKind::MxmCompute, 0,
      {task_macro(compute, row_schedule(50, rows, 0, IcuInductionTarget::None)),
       task_macro(read,
                  row_schedule(120, rows, 1,
                               IcuInductionTarget::MxmAccumulatorAddress))},
      dynamic ? std::vector{count_binding(0), count_binding(1)}
              : std::vector<MacroTaskBinding>{});
  rt::IcuProgram vxm;
  VxmLaneAluInstruction head;
  head.operation = VxmAluOpcode::Multiply;
  head.lhs = VxmLaneOperand::StreamBFloat16();
  head.rhs = VxmLaneOperand::Imm(0.5f);
  head.precision = VxmAluPrecision::Float32;
  head.output_type = VxmCastTarget::BFloat16;
  head.repeat_count = rows;
  auto tail = head;
  tail.operation = VxmAluOpcode::Bypass;
  tail.lhs = VxmLaneOperand::Previous();
  tail.rhs = VxmLaneOperand::Imm(0.f);
  tail.output_stream = 0;
  vxm.emit_vxm(vxm_arrival - 1, 0, VxmChainDepth::Two, head);
  vxm.emit_vxm(vxm_arrival - 1, 1, VxmChainDepth::Two, tail);
  for (auto q : vxm.encode_queues()) {
    if (q.kind != rt::QueueKind::Vxm || q.commands.empty())
      continue;
    std::vector<MacroTaskBinding> bindings;
    if (dynamic)
      bindings = {{1, MacroBindingTarget::NativeRepeatCount,
                   MacroTaskParameter::Rows, 1, 0, false}};
    t.queues.push_back({4, static_cast<std::uint32_t>(q.index), bindings, 0});
    p.queues.push_back(std::move(q));
  }
  for (unsigned plane = 0; plane < 2; ++plane) {
    std::vector<MacroTaskBinding> bindings;
    if (dynamic)
      bindings = {count_binding(0),
                  base_binding(MacroTaskParameter::OutputBase)};
    add(rt::QueueKind::Mem,
        static_cast<std::uint32_t>(
            InstructionControlUnit::mem_queue(Hemisphere::West, plane, 0)),
        {task_macro(MemInstruction::Write(0, StreamId::East(plane)),
                    row_schedule(vxm_arrival + 3, rows, 1,
                                 IcuInductionTarget::MemAddress))},
        bindings);
  }
  b.expert_template(t);
  b.combine(combine, 4);
  b.completion_event(77, {4});
  b.wait_for(rt::QueueKind::Mem,
             static_cast<std::uint16_t>(
                 InstructionControlUnit::mem_queue(Hemisphere::West, 30, 0)),
             77);
  p.max_cycle = 200;
  if (dynamic) {
    rejects([&] {
      validate_routing_descriptor_references(p.routing_descriptors, 3);
    });
    rejects([&] {
      validate_routing_descriptor_references(p.routing_descriptors, 4);
    });
    auto invalid = p;
    for (auto &d : invalid.routing_descriptors)
      if (d.opcode == RoutingOpcode::ExpertTemplate)
        d.expert_template.queues.front().bindings.front().record = 999;
    rejects([&] {
      std::ostringstream discarded;
      rt::write_binary_program(invalid, discarded);
    });
    invalid = p;
    for (auto &q : invalid.queues)
      if (q.kind == rt::QueueKind::MxmCompute)
        q.commands[1] = q.commands[0];
    rejects([&] {
      std::ostringstream discarded;
      rt::write_binary_program(invalid, discarded);
    });
  }
  std::ostringstream os(std::ios::binary);
  rt::write_binary_program(p, os);
  const auto bytes = os.str();
  auto program = rt::read_binary_program(std::span(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  SystemHardwareConfiguration cfg;
  cfg.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, cfg);
  auto &chip = system->chip();
  rt::CModelRuntime runtime(*system);
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned e = 0; e < 4; ++e)
      put(chip, select.choice_scores, token, e,
          std::bit_cast<std::uint32_t>(top2 ? (e == 3   ? 4.f
                                               : e == 1 ? 3.f
                                                        : 0.f)
                                       : e == (token < 37 ? 3u : 1u) ? 1.f
                                                                     : 0.f));
    if (top2) {
      put(chip, plan.weights, token, 0, std::bit_cast<std::uint32_t>(0.1f));
      put(chip, plan.weights, token, 1, std::bit_cast<std::uint32_t>(0.9f));
    }
    for (unsigned c = 0; c < 32; ++c)
      put(chip, a.source, token, c,
          Bf16::from_float(static_cast<float>(token + c) / 64).bits());
  }
  for (std::size_t index = 0; index < segments.size(); ++index) {
    const auto &segment = segments[index];
    const auto local = index % 18;
    const auto scale = index < 18 ? 2.f : 1.f;
    for (unsigned row = 0; row < segment.vector_count; ++row) {
      C2cVector v{};
      if (local < 16)
        for (unsigned tile = 0; tile < hw::kTileRows; ++tile)
          for (unsigned lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto k = tile * hw::kLanesPerTile + lane;
            const auto column = row * 8 + local / 2;
            const auto bits =
                Bf16::from_float(k == column ? scale : 0.f).bits();
            v.payload[tile][lane] =
                static_cast<std::uint8_t>(bits >> (8 * (local % 2)));
          }
      system->ddr4().initialize_vector(
          segment.ddr_address + row * hw::kPhysicalVectorBytes, v);
    }
  }
  auto sentinel = t.output;
  sentinel.rows = t.output.rows + 1;
  auto final_sentinel = combine.destination;
  final_sentinel.rows = tokens + 1;
  for (unsigned c = 0; c < 32; ++c) {
    put(chip, sentinel, t.output.rows, c, 0xdeadu);
    put(chip, final_sentinel, tokens, c, 0xbeefu);
  }
  if (dynamic) {
    runtime.load_timed(program);
    check(!runtime.all_images_committed() &&
              chip.expert_macros().trace().empty(),
          "expert template ran before timed image commit");
  } else {
    runtime.load(program);
  }
  bool injected = false;
  std::uint64_t issues = 0;
  for (unsigned n = 0; n < 200000 && !runtime.kernel_done(); ++n) {
    try {
      runtime.run_until_complete(1);
    } catch (...) {
      const auto &tr = chip.expert_macros().trace();
      std::cerr << "macro failure physical=" << system->cycle() << " local="
                << (tr.empty() ? 0 : chip.cycle() - tr.back().start_cycle)
                << '\n';
      for (unsigned col = 0; col < 3; ++col)
        std::cerr << "col=" << col << " east0="
                  << chip.stream_fabric(Hemisphere::East)
                         .cell(col, 0, 0, StreamId::West(0))
                         .valid
                  << " west16="
                  << chip.stream_fabric(Hemisphere::West)
                         .cell(col, 0, 0, StreamId::West(16))
                         .valid
                  << '\n';
      throw;
    }
    const auto &trace = chip.expert_macros().trace();
    if(!chip.routing_combine().idle()) {
      check(!runtime.can_load_program() && !chip.icu().queue_completion_ready(77),
            "combine released completion or program switching before drain");
    }
    if (!trace.empty() && !trace.back().finish_cycle) {
      check(!runtime.can_load_program(),
            "active expert macro allowed program replacement");
      for (const auto &q : t.queues)
        chip.icu().visit_program_queue(
            static_cast<IcuMacroQueueKind>(q.kind), q.index, false,
            [&](const auto &queue) {
              const auto &event = queue.last_trace();
              if (event.action == IcuQueueAction::MacroIssue ||
                  event.action == IcuQueueAction::FunctionalIssue) {
                check(system->cycle() - 1 ==
                          trace.back().start_cycle + event.cycle,
                      "task queues lost common epoch");
                ++issues;
              }
            });
    }
    if (dynamic && trace.size() == 2 && !injected) {
      rejects([&] { chip.expert_macros().input_ready(trace.front().task); });
      injected = true;
    }
  }
  check(runtime.kernel_done() && runtime.can_load_program(),
        "expert macro pipeline timeout");
  const auto &trace = chip.expert_macros().trace();
  check(trace.size() == (top2      ? 4u
                         : dynamic ? 3u
                                   : 1u) &&
            issues > 0,
        "expert macro task trace missing");
  for (const auto &entry : trace)
    check(entry.finish_cycle > entry.start_cycle &&
              entry.output_segments == entry.task.valid_rows * 8,
          "TASK_DONE before output commit");
  for (unsigned token = 0; token < tokens; ++token)
    for (unsigned c = 0; c < 32; ++c) {
      const auto expert = token < 37 ? 3u : 1u;
      const auto assignment = top2      ? tokens + token
                              : dynamic ? (expert == 3 ? 5 + token : token - 37)
                                        : 0;
      const auto expected =
          (static_cast<float>(token + c) / 64) * (expert == 3 ? 0.5f : 1.f);
      check(get(chip, t.output, assignment, c) ==
                Bf16::from_float(expected).bits(),
            "expert MXM/VXM numeric mismatch");
      auto final_expected = expected;
      if (top2) {
        const auto x = static_cast<float>(token + c) / 64;
        check(get(chip, t.output, token, c) == Bf16::from_float(x).bits(),
              "second expert output mismatch");
        const auto first = Bf16::from_float(x * 0.5f).to_float() * 0.1f;
        const auto second = Bf16::from_float(x).to_float() * 0.9f;
        final_expected = first + second;
      }
      check(get(chip, combine.destination, token, c) ==
                Bf16::from_float(final_expected).bits(),
            "device weighted combine mismatch");
    }
  for (unsigned c = 0; c < 32; ++c)
    check(get(chip, sentinel, t.output.rows, c) == 0xdeadu &&
              get(chip, final_sentinel, tokens, c) == 0xbeefu,
          "expert macro tail overwrite");
  if (dynamic)
    check(injected && chip.expert_weights().statistics().residency_hits ==
                          (top2 ? 2u : 1u),
          "dynamic task identity/residency not exercised");
  check(
      chip.routing_combine().statistics().contributions == tokens * select.k &&
          chip.routing_combine().statistics().committed_segments == tokens * 8,
      "combine physical accounting");
  std::cout << "macro+combine "
            << (top2      ? "top2"
                : dynamic ? "dynamic"
                          : "static")
            << " tasks=" << trace.size() << " cycles=" << system->cycle()
            << " numeric exact\n";
}
