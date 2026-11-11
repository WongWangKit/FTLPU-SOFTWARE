#include "ftlpu/core/bf16.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/software/runtime/routing_program.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"
#include "moe_cycle_trace.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;
namespace {
constexpr unsigned T = 5, E = 4, K = 2, H = 32, BLOCK = 2;
using Matrix = std::array<float, H * H>;
float bf(float x) { return Bf16::from_float(x).to_float(); }
void check(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
RoutingTensorView view(RoutingElement type, unsigned rows, unsigned columns,
                       unsigned side, unsigned slice, unsigned bank = 0,
                       unsigned base = 0) {
  RoutingTensorView v;
  v.element = type;
  v.rows = rows;
  v.columns = columns;
  v.feature_stride = rows;
  for (unsigned p = 0; p < v.bytes(); ++p)
    v.planes[p] = {side, slice + p, bank, base};
  return v;
}
void put(TspSliceSystem &chip, const RoutingTensorView &v, unsigned r,
         unsigned c, std::uint32_t bits) {
  for (unsigned p = 0; p < v.bytes(); ++p) {
    const auto q = v.planes[p];
    chip.mem_array(static_cast<Hemisphere>(q.hemisphere))
        .set_sram_byte(q.slice, q.bank, v.row(r, c, p), c % 32,
                       static_cast<std::uint8_t>(bits >> (8 * p)));
  }
}
std::uint32_t get(const TspSliceSystem &chip, const RoutingTensorView &v,
                  unsigned r, unsigned c) {
  std::uint32_t bits = 0;
  for (unsigned p = 0; p < v.bytes(); ++p) {
    const auto q = v.planes[p];
    bits |=
        std::uint32_t(chip.mem_array(static_cast<Hemisphere>(q.hemisphere))
                          .sram_byte(q.slice, q.bank, v.row(r, c, p), c % 32))
        << (8 * p);
  }
  return bits;
}
template <class Instruction>
rt::QueueCommand macro(Instruction i, IcuMacroSchedule s) {
  std::uint64_t bits;
  rt::InstructionKind kind;
  if constexpr (std::is_same_v<Instruction, MemInstruction>) {
    bits = isa::encode_mem_instruction(i);
    kind = rt::InstructionKind::Mem;
  } else {
    bits = isa::encode_mxm_instruction(i);
    kind = rt::InstructionKind::Mxm;
  }
  return rt::encode_macro_schedule_command(
      {static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
       kind,
       static_cast<std::uint16_t>(bits >> 32 ? 2 : 1),
       {std::uint32_t(bits), std::uint32_t(bits >> 32), 0, 0}},
      s);
}
IcuMacroSchedule rows(unsigned cycle, unsigned count, std::int64_t stride,
                      IcuInductionTarget target) {
  IcuMacroSchedule s;
  s.start_cycle = cycle;
  s.outer_count = count;
  s.outer_stride = stride;
  s.induction_target = target;
  return s;
}
unsigned memq(unsigned side, unsigned slice, unsigned bank) {
  return static_cast<unsigned>(InstructionControlUnit::mem_queue(
      static_cast<Hemisphere>(side), slice, bank));
}
struct Fixture {
  rt::BinaryProgram program;
  RoutingTensorView input = view(RoutingElement::BF16, T, H, 0, 44),
                    router_input =
                        view(RoutingElement::BF16, T, H, 1, 50, 0, 300);
  RoutingTensorView scores = view(RoutingElement::F32, T, E, 1, 40, 0, 500),
                    packed = view(RoutingElement::BF16, BLOCK, H, 0, 50, 0, 64);
  RoutingTensorView expert_output = view(RoutingElement::BF16, T *K, H, 1, 0),
                    output = view(RoutingElement::BF16, T, H, 1, 8);
  std::array<float, T * H> x{};
  Matrix router{};
  std::array<std::array<Matrix, 3>, E> weights{};
  std::vector<ExpertWeightSegment> segments;
  explicit Fixture(const std::filesystem::path &checkpoint_tiles = {}) {
    if (!checkpoint_tiles.empty()) {
      load_checkpoint_tiles(checkpoint_tiles);
      build(true);
      return;
    }
    for (unsigned t = 0; t < T; ++t)
      for (unsigned c = 0; c < H; ++c)
        x[t * H + c] = bf(float(int((t * 11 + c * 5) % 23) - 11) / 32);
    for (unsigned t = 0; t < T; ++t) {
      x[t * H] = bf((t % 2 ? -1.f : 1.f) * (0.25f + t / 32.f));
      x[t * H + 1] = bf((t % 3 ? -1.f : 1.f) * 0.1875f);
      x[t * H + 31] = 1.f;
    }
    for (unsigned e = 0; e < E; ++e) {
      router[31 * H + e] = 0.5f;
      router[(e / 2) * H + e] = (e % 2 ? -1.f : 1.f) * 0.25f;
    }
    for (unsigned e = 0; e < E; ++e)
      for (unsigned projection = 0; projection < 3; ++projection)
        for (unsigned r = 0; r < H; ++r)
          for (unsigned c = 0; c < H; ++c) {
            const float diagonal = projection == 0   ? 0.5f + e / 8.f
                                   : projection == 1 ? 0.75f + e / 16.f
                                                     : 0.5f + e / 16.f;
            weights[e][projection][r * H + c] = bf(
                (r == c ? diagonal : 0.f) +
                float(int((r * 3 + c * 5 + e + projection) % 7) - 3) / 128.f);
          }
    build(false);
  }
  void load_checkpoint_tiles(const std::filesystem::path &path) {
    std::ifstream source(path, std::ios::binary);
    if (!source)
      throw std::runtime_error("cannot open checkpoint tiles");
    const auto read = [&] {
      std::array<unsigned char, 2> bytes{};
      source.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
      if (!source)
        throw std::runtime_error("truncated checkpoint tile payload");
      return Bf16::from_bits(std::uint16_t(bytes[0] | (std::uint16_t(bytes[1]) << 8))).to_float();
    };
    for (auto &value : x) value = read();
    for (unsigned expert = 0; expert < E; ++expert)
      for (unsigned hidden = 0; hidden < H; ++hidden)
        router[hidden * H + expert] = read();
    for (auto &expert : weights)
      for (auto &projection : expert)
        for (unsigned output = 0; output < H; ++output)
          for (unsigned input = 0; input < H; ++input)
            projection[input * H + output] = read();
    if (source.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("extra data in checkpoint tile payload");
  }
  void build(bool use_vxm_topk) {
    if (use_vxm_topk) build_router();
    rt::RoutingProgramBuilder builder(program);
    RouteSelectDescriptor select;
    select.tokens = T;
    select.experts = E;
    select.k = K;
    select.choice_scores = scores;
    if (use_vxm_topk) {
      RouteVxmSelectSchedule schedule;
      schedule.bundle_interval = 4;
      for (unsigned stream = 0; stream < schedule.streams; ++stream)
        schedule.score_planes[stream] = {1, 16 + stream, 1, 800};
      schedule = builder.pack_vxm_scores(select, schedule, 200);
      builder.select_vxm(select, schedule, 1);
    } else {
      builder.select(select, 1);
    }
    builder.plan({}, 2);
    RouteAddressDescriptor address;
    address.max_rows = BLOCK;
    address.source = input;
    address.destination = packed;
    const auto address_id = builder.address_template(address);
    builder.dispatch({1, address_id, BLOCK}, 3);
    std::vector<ExpertWeightEntry> entries;
    for (unsigned e = 0; e < E; ++e) {
      entries.push_back({static_cast<std::uint32_t>(segments.size()), 20});
      for (unsigned s = 0; s < 16; ++s)
        segments.push_back(
            {0x200000 + segments.size() * 0x1000, 12, 0, s, 1, 64, 30});
      for (unsigned s = 4; s < 8; ++s)
        segments.push_back(
            {0x200000 + segments.size() * 0x1000, 1, 1, s, 1, 64, 30});
    }
    builder.weight_table(1, entries, segments);
    ExpertMacroTemplateDescriptor descriptor;
    descriptor.output = expert_output;
    // Append commands to one resident image per participating queue. Binding
    // record numbers refer to logical packed-Macro records, not byte offsets.
    const auto add = [&](rt::QueueKind kind, unsigned index, auto instruction,
                         IcuMacroSchedule schedule, bool dynamic_rows = false,
                         std::optional<MacroTaskParameter> base = {},
                         unsigned relative_base = 0) {
      auto q = std::find_if(
          program.queues.begin(), program.queues.end(),
          [&](const auto &q) { return q.kind == kind && q.index == index; });
      if (q == program.queues.end()) {
        program.queues.push_back({kind, static_cast<std::uint16_t>(index), {}});
        q = std::prev(program.queues.end());
        q->requires_raw_macro = true;
      }
      auto spec = std::find_if(descriptor.queues.begin(),
                               descriptor.queues.end(), [&](const auto &q) {
                                 return q.kind == static_cast<unsigned>(kind) &&
                                        q.index == index;
                               });
      if (spec == descriptor.queues.end()) {
        descriptor.queues.push_back(
            {static_cast<unsigned>(kind), index, {}, relative_base});
        spec = std::prev(descriptor.queues.end());
      }
      const auto record = static_cast<unsigned>(q->commands.size());
      q->commands.push_back(macro(instruction, schedule));
      if (dynamic_rows)
        spec->bindings.push_back({record, MacroBindingTarget::OuterCount,
                                  MacroTaskParameter::Rows, 1, 0, false});
      if (base)
        spec->bindings.push_back(
            {record, MacroBindingTarget::Operand, *base, 1, 0, true});
    };
    const auto layout = make_configured_stream_layout();
    const auto weight_latency = [&](unsigned s) {
      return unsigned(layout.mxm_weight_input.column -
                      layout.mem_ports.output_column(s, StreamDirection::East) +
                      1);
    };
    const auto input_latency = [&](unsigned s) {
      return unsigned(layout.mxm_activation_input.column -
                      layout.mem_ports.output_column(s, StreamDirection::East) +
                      1);
    };
    const auto vxm_latency = [&](unsigned s) {
      return unsigned(layout.mem_ports.output_column(s, StreamDirection::West) -
                      layout.vxm_input.column + 1);
    };
    for (unsigned phase = 0; phase < 3; ++phase) {
      const unsigned load = phase == 0   ? 30
                            : phase == 1 ? 200
                                         : 380,
                     compute = phase == 0   ? 50
                               : phase == 1 ? 220
                                            : 400,
                     read = phase == 0   ? 120
                            : phase == 1 ? 290
                                         : 470;
      for (unsigned s = 0; s < 16; ++s) {
        IcuMacroSchedule schedule;
        schedule.start_cycle = load - weight_latency(s);
        schedule.inner_count = 4;
        schedule.inner_stride = 1;
        schedule.induction_target = IcuInductionTarget::MemAddress;
        add(rt::QueueKind::Mem, memq(0, s, 1),
            MemInstruction::Read(phase * 4, StreamId::East(s)), schedule, false,
            MacroTaskParameter::WeightBase, 64);
      }
      IcuMacroSchedule load_schedule;
      load_schedule.start_cycle = load;
      load_schedule.inner_count = 4;
      load_schedule.inner_stride = 1;
      load_schedule.induction_target = IcuInductionTarget::MxmWeightColumn;
      add(rt::QueueKind::MxmLoad, 0, MxmControlInstruction::IWDirect16(0, 0),
          load_schedule);
      for (unsigned p = 0; p < 2; ++p) {
        const auto s = phase < 2 ? 50 + p : 48 + p;
        add(rt::QueueKind::Mem, memq(0, s, 0),
            MemInstruction::Read(phase < 2 ? 0 : 220, StreamId::East(p)),
            rows(compute - input_latency(s), BLOCK, 1,
                 IcuInductionTarget::MemAddress),
            true,
            phase < 2 ? std::optional(MacroTaskParameter::InputBase)
                      : std::nullopt);
      }
      add(rt::QueueKind::MxmCompute, 0,
          MxmControlInstruction::Compute(0, 0, 0, 0, 1,
                                         MxmAccumulatorDestination::Sram,
                                         MxmDataFormat::BFloat16),
          rows(compute, BLOCK, 0, IcuInductionTarget::None), true);
      add(rt::QueueKind::MxmCompute, 0,
          MxmControlInstruction::AccumulatorRead(
              0, phase == 1 ? 2 : 0, true,
              MxmAccumulatorOutputFormat::BFloat16),
          rows(read, BLOCK, 1, IcuInductionTarget::MxmAccumulatorAddress),
          true);
    }
    // Gate spills BF16; Up streams directly into the second fixed VXM group.
    for (unsigned p = 0; p < 2; ++p) {
      const auto s = 40 + p;
      const auto arrival =
          120 +
          unsigned(layout.mxm_result_output.column -
                   layout.mem_ports.input_column(s, StreamDirection::West)) +
          1;
      add(rt::QueueKind::Mem, memq(0, s, 0),
          MemInstruction::Write(200, StreamId::West(p)),
          rows(arrival, BLOCK, 1, IcuInductionTarget::MemAddress), true);
      add(rt::QueueKind::Mem, memq(0, s, 0),
          MemInstruction::Read(200, StreamId::West(p)),
          rows(306 - vxm_latency(s), BLOCK, 1, IcuInductionTarget::MemAddress),
          true);
      const auto hidden_write = 306 + 18 +
                                unsigned(layout.mem_ports.input_column(
                                             48 + p, StreamDirection::East) -
                                         layout.vxm_output.column);
      // Keep logical Macro records in issue order for the packed delta chain.
      auto q = std::find_if(program.queues.begin(), program.queues.end(),
                            [&](const auto &q) {
                              return q.kind == rt::QueueKind::Mem &&
                                     q.index == memq(0, 48 + p, 0);
                            });
      check(q != program.queues.end(), "missing hidden input reader");
      auto spec =
          std::find_if(descriptor.queues.begin(), descriptor.queues.end(),
                       [&](const auto &x) {
                         return x.kind == 0 && x.index == memq(0, 48 + p, 0);
                       });
      q->commands.insert(
          q->commands.begin(),
          macro(MemInstruction::Write(220, StreamId::East(6 + p)),
                rows(hidden_write, BLOCK, 1, IcuInductionTarget::MemAddress)));
      for (auto &binding : spec->bindings)
        ++binding.record;
      spec->bindings.push_back({0, MacroBindingTarget::OuterCount,
                                MacroTaskParameter::Rows, 1, 0, false});
      add(rt::QueueKind::Mem, memq(1, p, 0),
          MemInstruction::Write(0, StreamId::East(p)),
          rows(486 + 2 +
                   unsigned(
                       layout.mem_ports.input_column(p, StreamDirection::East) -
                       layout.vxm_output.column),
               BLOCK, 1, IcuInductionTarget::MemAddress),
          true, MacroTaskParameter::OutputBase);
    }
    for (unsigned p = 0; p < 4; ++p) {
      add(rt::QueueKind::Mem, memq(1, 4 + p, 1),
          MemInstruction::Read(0, StreamId::West(16 + p)),
          rows(306 - vxm_latency(4 + p), BLOCK, 0,
               IcuInductionTarget::MemAddress),
          true, MacroTaskParameter::WeightBase, 64);
      if (p < 2)
        add(rt::QueueKind::Mem, memq(1, 4 + p, 1),
            MemInstruction::Read(0, StreamId::West(16 + p)),
            rows(486 - vxm_latency(4 + p), BLOCK, 0,
                 IcuInductionTarget::MemAddress),
            true, MacroTaskParameter::WeightBase, 64);
    }
    rt::IcuProgram vxm;
    for (unsigned stage = 0; stage < 8; ++stage) {
      VxmLaneAluInstruction op;
      op.precision = VxmAluPrecision::Float32;
      op.output_type = VxmCastTarget::BFloat16;
      op.repeat_count = BLOCK;
      op.lhs = stage == 0 ? VxmLaneOperand::StreamBFloat16()
                          : VxmLaneOperand::Previous();
      if (stage == 0) {
        op.operation = VxmAluOpcode::Negate;
        op.rhs = VxmLaneOperand::StreamBFloat16();
      }
      if (stage == 1)
        op.operation = VxmSpecialAluOpcode::Exp;
      if (stage == 2) {
        op.operation = VxmAluOpcode::Add;
        op.rhs = VxmLaneOperand::Imm(1.f);
      }
      if (stage == 3)
        op.operation = VxmSpecialAluOpcode::Reciprocal;
      if (stage == 4) {
        op.operation = VxmAluOpcode::Multiply;
        op.rhs = VxmLaneOperand::Original();
      }
      if (stage == 5) {
        op.operation = VxmAluOpcode::Multiply;
        op.rhs = VxmLaneOperand::Aux();
      }
      if (stage == 7)
        op.output_stream = 6;
      vxm.emit_vxm(305, stage, VxmChainDepth::Eight, op);
    }
    for (unsigned stage = 0; stage < 2; ++stage) {
      VxmLaneAluInstruction op;
      op.precision = VxmAluPrecision::Float32;
      op.output_type = VxmCastTarget::BFloat16;
      op.repeat_count = BLOCK;
      op.lhs = stage == 0 ? VxmLaneOperand::StreamBFloat16()
                          : VxmLaneOperand::Previous();
      if (stage == 1)
        op.output_stream = 0;
      vxm.emit_vxm(485, stage, VxmChainDepth::Two, op);
    }
    for (auto q : vxm.encode_queues())
      if (q.kind == rt::QueueKind::Vxm && !q.commands.empty()) {
        ExpertTemplateQueue spec{4, static_cast<unsigned>(q.index), {}, 0};
        unsigned pc = 0;
        for (const auto &c : q.commands) {
          if (c.instruction_kind == rt::InstructionKind::Vxm) {
            spec.bindings.push_back({pc, MacroBindingTarget::NativeRepeatCount,
                                     MacroTaskParameter::Rows, 1, 0, false});
            pc += 3;
          } else
            ++pc;
        }
        descriptor.queues.push_back(std::move(spec));
        program.queues.push_back(std::move(q));
      }
    builder.expert_template(descriptor);
    builder.combine({1, expert_output, output}, 4);
    builder.completion_event(77, {4});
    builder.wait_for(rt::QueueKind::Mem,
                     static_cast<std::uint16_t>(memq(1, 30, 0)), 77);
    if (!use_vxm_topk) {
      build_router();
      for (auto &q : program.queues)
        if (q.kind == rt::QueueKind::Routing)
          q.commands.insert(q.commands.begin(), {isa::encode_icu_nop(200)});
    }
    program.max_cycle = std::max<std::size_t>(program.max_cycle, 600);
  }
  void build_router() {
    // A separate physical West MXM computes router logits once, before SELECT.
    // Its queues are not owned by the dynamically restarted expert template.
    const auto layout = make_configured_stream_layout();
    const auto latency = [&](unsigned slice, std::size_t endpoint) {
      return unsigned(endpoint - layout.mem_ports.output_column(slice, StreamDirection::East) + 1);
    };
    const auto global = [&](rt::QueueKind kind, unsigned index,
                            auto instruction, IcuMacroSchedule schedule) {
      auto q = std::find_if(
          program.queues.begin(), program.queues.end(),
          [&](const auto &q) { return q.kind == kind && q.index == index; });
      if (q == program.queues.end()) {
        program.queues.push_back({kind, static_cast<std::uint16_t>(index), {}});
        q = std::prev(program.queues.end());
        q->requires_raw_macro = true;
      }
      q->commands.push_back(macro(instruction, schedule));
    };
    for (unsigned s = 0; s < 16; ++s) {
      IcuMacroSchedule schedule;
      schedule.start_cycle = 30 - latency(s, layout.mxm_weight_input.column);
      schedule.inner_count = 4;
      schedule.inner_stride = 1;
      schedule.induction_target = IcuInductionTarget::MemAddress;
      global(rt::QueueKind::Mem, memq(1, s, s >= 4 && s < 8 ? 0 : 1),
             MemInstruction::Read(128, StreamId::East(s)), schedule);
    }
    const unsigned west_mxm = program.hardware.mxms_per_hemisphere;
    IcuMacroSchedule load;
    load.start_cycle = 30;
    load.inner_count = 4;
    load.inner_stride = 1;
    load.induction_target = IcuInductionTarget::MxmWeightColumn;
    global(rt::QueueKind::MxmLoad, west_mxm,
           MxmControlInstruction::IWDirect16(0, 0), load);
    for (unsigned p = 0; p < 2; ++p)
      global(rt::QueueKind::Mem, memq(1, 50 + p, 0),
             MemInstruction::Read(300, StreamId::East(p)),
             rows(50 - latency(50 + p, layout.mxm_activation_input.column), T, 1,
                  IcuInductionTarget::MemAddress));
    global(rt::QueueKind::MxmCompute, west_mxm,
           MxmControlInstruction::Compute(0, 0, 0, 0, 1,
                                          MxmAccumulatorDestination::Sram,
                                          MxmDataFormat::BFloat16),
           rows(50, T, 0, IcuInductionTarget::None));
    global(rt::QueueKind::MxmCompute, west_mxm,
           MxmControlInstruction::AccumulatorRead(
               0, 0, true, MxmAccumulatorOutputFormat::Float32),
           rows(120, T, 1, IcuInductionTarget::MxmAccumulatorAddress));
    for (unsigned p = 0; p < 4; ++p) {
      const auto arrival = 120 +
                           unsigned(layout.mxm_result_output.column -
                                    layout.mem_ports.input_column(
                                        40 + p, StreamDirection::West)) +
                           1;
      global(rt::QueueKind::Mem, memq(1, 40 + p, 0),
             MemInstruction::Write(500, StreamId::West(p)),
             rows(arrival, T, 1, IcuInductionTarget::MemAddress));
    }
  }
  void initialize(C2cDmaSystem &system) {
    auto &chip = system.chip();
    for (unsigned t = 0; t < T; ++t)
      for (unsigned c = 0; c < H; ++c) {
        const auto bits = Bf16::from_float(x[t * H + c]).bits();
        put(chip, input, t, c, bits);
        put(chip, router_input, t, c, bits);
      }
    for (unsigned r = 0; r < H; ++r)
      for (unsigned c = 0; c < H; ++c) {
        const auto bits = Bf16::from_float(router[r * H + c]).bits();
        for (unsigned p = 0; p < 2; ++p) {
          const auto s = (c % 8) * 2 + p;
          chip.mem_array(Hemisphere::West)
              .set_sram_byte(s, s >= 4 && s < 8 ? 0 : 1, 128 + c / 8, r,
                             static_cast<std::uint8_t>(bits >> (8 * p)));
        }
      }
    for (unsigned index = 0; index < segments.size(); ++index) {
      const auto &segment = segments[index];
      const unsigned expert = index / 20, slice = index % 20;
      for (unsigned row = 0; row < segment.vector_count; ++row) {
        C2cVector vector{};
        if (slice < 16)
          for (unsigned r = 0; r < H; ++r) {
            const auto projection = row / 4, column = (row % 4) * 8 + slice / 2;
            const auto bits =
                Bf16::from_float(weights[expert][projection][r * H + column])
                    .bits();
            vector.payload[r / hw::kLanesPerTile][r % hw::kLanesPerTile] =
                static_cast<std::uint8_t>(bits >> (8 * (slice % 2)));
          }
        system.ddr4().initialize_vector(
            segment.ddr_address + row * hw::kPhysicalVectorBytes, vector);
      }
    }
  }
};
std::array<float, H> dot(const std::array<float, H> &x, const Matrix &w,
                         bool cast) {
  std::array<float, H> y{};
  for (unsigned c = 0; c < H; ++c) {
    float sum = 0;
    for (unsigned r = 0; r < H; ++r)
      sum += x[r] * w[r * H + c];
    y[c] = cast ? bf(sum) : sum;
  }
  return y;
}
VxmSpecialAlu reference_lut() {
  VxmSpecialAlu alu;
  constexpr float ln2 = 0.6931471805599453f;
  const auto make = [&](auto opcode, float begin, float step, auto fn) {
    std::vector<VxmLutEntry> entries;
    for (unsigned i = 0; i < 256; ++i) {
      const float x = begin + i * step;
      const float y = fn(x);
      entries.push_back(VxmLutEntry::from_float((fn(x + step) - y) / step, y));
    }
    alu.configure_lut(opcode, {begin, step}, std::move(entries));
  };
  make(VxmSpecialAluOpcode::Exp, -ln2 / 2, ln2 / 256,
       [](float x) { return std::exp(x); });
  make(VxmSpecialAluOpcode::Reciprocal, 1, 1.f / 256,
       [](float x) { return 1.f / x; });
  return alu;
}
struct Reference {
  std::array<float, T * H> quantized{}, mathematical{};
  std::array<std::array<unsigned, K>, T> experts{};
  std::array<float, T * K> route_weights{};
  std::array<unsigned, E> counts{};
};
Reference reference(const Fixture &f) {
  Reference result;
  auto lut = reference_lut();
  for (unsigned t = 0; t < T; ++t) {
    std::array<float, H> x;
    std::copy_n(f.x.begin() + t * H, H, x.begin());
    const auto score = dot(x, f.router, false);
    std::array<unsigned, E> order;
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
      return score[a] == score[b] ? a < b : score[a] > score[b];
    });
    for (unsigned slot = 0; slot < K; ++slot) {
      const auto e = order[slot];
      result.experts[t][slot] = e;
      result.route_weights[t * K + slot] = score[e];
      ++result.counts[e];
      const auto g = dot(x, f.weights[e][0], true),
                 u = dot(x, f.weights[e][1], true);
      auto hidden = g;
      for (unsigned c = 0; c < H; ++c) {
        const float sigmoid =
            lut.execute(VxmSpecialAluOpcode::Reciprocal,
                        1.f + lut.execute(VxmSpecialAluOpcode::Exp, -g[c]));
        hidden[c] = bf((sigmoid * g[c]) * u[c]);
      }
      const auto y = dot(hidden, f.weights[e][2], true);
      const auto exact_g = dot(x, f.weights[e][0], false),
                 exact_u = dot(x, f.weights[e][1], false);
      auto exact_hidden = exact_g;
      for (unsigned c = 0; c < H; ++c)
        exact_hidden[c] =
            (exact_g[c] / (1.f + std::exp(-exact_g[c]))) * exact_u[c];
      const auto exact_y = dot(exact_hidden, f.weights[e][2], false);
      for (unsigned c = 0; c < H; ++c) {
        result.quantized[t * H + c] += score[e] * y[c];
        result.mathematical[t * H + c] += score[e] * exact_y[c];
      }
    }
    for (unsigned c = 0; c < H; ++c)
      result.quantized[t * H + c] = bf(result.quantized[t * H + c]);
  }
  return result;
}
void observed_trace(rt::CModelRuntime &runtime, TspSliceSystem &chip,
                    std::uint64_t cycle) {
  const auto record = [&](std::string resource, std::string detail) {
    runtime.record_execution_trace_interval(
        cycle, cycle + 1, std::move(resource), std::move(detail));
  };
  if (!chip.expert_tasks().idle())
    record("MoE.Dispatch",
           "device task traversal / Gather / weights / expert; source=runtime");
  if (!chip.routing_combine().idle())
    record("MoE.Combine", "inverse index / Gather / VXM FP32 accumulate / MEM "
                          "commit; source=runtime");
  for (unsigned side = 0; side < 2; ++side)
    for (const auto &trace : chip.mem_array(static_cast<Hemisphere>(side))
                                 .executed_instructions()) {
      if (trace.instruction.routing_owner == ExpertWeightLoader::owner)
        record("MoE.WeightWrite",
               std::string(side ? "W" : "E") +
                   " slice=" + std::to_string(trace.mem_slice) +
                   " bank=" + std::to_string(trace.bank) +
                   " tile=" + std::to_string(trace.tile) + " source=runtime");
    }
  const auto &lane = chip.vxm_unit().superlane(0).lane(0);
  for (unsigned stage = 0; stage < 8; ++stage)
    if (lane.last_trace()[stage].state == VxmLaneAluTraceState::Executed)
      record("VXM.Datapath.C" + std::to_string(stage),
             "actual lane=0 tile=0 source=runtime");
}
} // namespace
int main(int argc, char **argv) try {
  const std::filesystem::path out = argc > 1 ? argv[1] : "moe_ffn_demo_output";
  const std::filesystem::path checkpoint_tiles = argc > 2 ? argv[2] : "";
  const std::string_view mode = argc > 3 ? argv[3] : "serial";
  const bool double_buffered = mode == "double-buffered";
  const bool overlap = mode == "overlap" || double_buffered;
  if (argc > 4 || (!overlap && mode != "serial"))
    throw std::invalid_argument("usage: moe_ffn_demo [output [checkpoint_tiles [serial|overlap|double-buffered]]]");
  std::filesystem::create_directories(out);
  Fixture fixture(checkpoint_tiles);
  const auto golden = reference(fixture);
  rt::write_binary_program(fixture.program, out / "program.ftlpu");
  auto program = rt::read_binary_program(out / "program.ftlpu");
  SystemHardwareConfiguration config;
  config.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, config);
  fixture.initialize(*system);
  auto &chip = system->chip();
  rt::CModelRuntime runtime(*system);
  runtime.enable_execution_trace();
  runtime.enable_mem_execution_trace();
  runtime.stream_mem_execution_trace_csv(out / "runtime.mem.csv");
  runtime.load_timed(program);
  chip.expert_tasks().set_input_weight_overlap(overlap);
  chip.expert_weights().set_double_buffered(double_buffered);
  // Initial, fixed mux configuration: the 8-stage SwiGLU tail feeds the
  // East hidden buffer; the 2-stage final projection still writes West.
  chip.configure_vxm_output_block_destination(3, Hemisphere::East);
  MoeCycleTrace instructions(out / "instructions.cycles.csv", program);
  instructions.enable(*system);
  struct ObservedRoute {
    RoutingOpcode opcode;
    std::uint64_t start, end;
  };
  std::vector<ObservedRoute> observed_routes;
  std::size_t route_sequence = 0;
  for (unsigned n = 0; n < 300000 && !runtime.kernel_done(); ++n) {
    runtime.run_until_complete(1);
    instructions.sample(*system);
    observed_trace(runtime, chip, system->cycle() - 1);
    if (chip.routing().busy() && !chip.routing().trace().empty()) {
      if (route_sequence != chip.routing().trace().size()) {
        route_sequence = chip.routing().trace().size();
        observed_routes.push_back({chip.routing().trace().back().opcode,
                                   system->cycle() - 1, system->cycle()});
      } else
        observed_routes.back().end = system->cycle();
    }
  }
  runtime.write_execution_trace_csv(out / "runtime.pipeline.csv");
  check(runtime.kernel_done() && runtime.can_load_program(),
        "MoE device pipeline timeout");
  std::ofstream tasks(out / "expert_tasks.csv"),
      operations(out / "rcu_operations.csv");
  tasks << "task,expert,rows,row_offset,assignment_base,start,end,weight_buffer\n";
  operations << "opcode,start,end\n";
  const auto &load_stats = runtime.program_load_statistics();
  if (load_stats.request_accepted && load_stats.program_start)
    runtime.record_execution_trace_interval(
        *load_stats.request_accepted, *load_stats.program_start,
        "MoE.ProgramLoad",
        "actual timed instruction and descriptor DMA; inputs already resident; "
        "source=runtime");
  // RCU internal clock is held during timed program load. Capture operations
  // in the same physical clock domain as DMA, MEM and the task-local epochs.
  for (const auto &operation : observed_routes) {
    operations << routing_opcode_name(operation.opcode) << ','
               << operation.start << ',' << operation.end << '\n';
    runtime.record_execution_trace_interval(
        operation.start, operation.end,
        operation.opcode == RoutingOpcode::Address
            ? "MoE.InputGather"
            : "MoE.Router." +
                  std::string(routing_opcode_name(operation.opcode)),
        "source=runtime actual RCU state interval");
  }
  for (const auto &task : chip.expert_macros().trace())
    tasks << task.task.identity.task_id << ',' << task.task.expert_id << ','
          << task.task.valid_rows << ',' << task.task.row_offset << ','
          << task.task.assignment_base << ',' << task.start_cycle << ','
          << task.finish_cycle + 1 << ',' << task.weight_buffer << '\n';
  for (const auto &task : chip.expert_macros().trace())
    runtime.record_execution_trace_interval(
        task.start_cycle, task.finish_cycle + 1,
        "MoE.Expert.E" + std::to_string(task.task.expert_id),
        "task=" + std::to_string(task.task.identity.task_id) +
            " rows=" + std::to_string(task.task.valid_rows) + " offset=" +
            std::to_string(task.task.row_offset) + " source=runtime");
  runtime.write_execution_trace_csv(out / "runtime.pipeline.csv");
  check(chip.routing().counts() ==
            std::vector<std::uint32_t>(golden.counts.begin(),
                                       golden.counts.end()),
        "device expert counts mismatch");
  for (unsigned t = 0; t < T; ++t)
    for (unsigned s = 0; s < K; ++s) {
      check(chip.routing().selected_experts()[t * K + s] ==
                golden.experts[t][s],
            "device Top-2 differs from router reference");
      check(chip.routing().selected_score_bits()[t * K + s] ==
                std::bit_cast<std::uint32_t>(golden.route_weights[t * K + s]),
            "FP32 router score mismatch");
    }
  std::ofstream values(out / "output.csv"),
      raw(out / "actual.bf16.bin", std::ios::binary),
      summary(out / "summary.json"), routes(out / "routing.csv");
  values << "token,feature,actual,reference_bf16_lut,reference_math\n";
  routes << "token,slot,expert,weight\n";
  double square = 0, scale = 0, math_square = 0, math_scale = 0;
  float maximum = 0;
  unsigned mismatches = 0;
  for (unsigned t = 0; t < T; ++t) {
    for (unsigned s = 0; s < K; ++s)
      routes << t << ',' << s << ','
             << chip.routing().selected_experts()[t * K + s] << ','
             << std::bit_cast<float>(
                    chip.routing().selected_score_bits()[t * K + s])
             << '\n';
    for (unsigned c = 0; c < H; ++c) {
      const auto bits =
          static_cast<std::uint16_t>(get(chip, fixture.output, t, c));
      const float actual = Bf16::from_bits(bits).to_float();
      const auto index = t * H + c;
      const float delta = actual - golden.quantized[index];
      maximum = std::max(maximum, std::fabs(delta));
      square += double(delta) * delta;
      scale += double(golden.quantized[index]) * golden.quantized[index];
      const float math_delta = actual - golden.mathematical[index];
      math_square += double(math_delta) * math_delta;
      math_scale +=
          double(golden.mathematical[index]) * golden.mathematical[index];
      mismatches += bits != Bf16::from_float(golden.quantized[index]).bits();
      raw.write(reinterpret_cast<const char *>(&bits), 2);
      values << t << ',' << c << ',' << std::setprecision(10) << actual << ','
             << golden.quantized[index] << ',' << golden.mathematical[index]
             << '\n';
    }
  }
  const auto rel = std::sqrt(square / std::max(scale, 1e-30));
  const auto weight_stats = chip.expert_weights().statistics();
  summary << "{\n  \"case\": \""
          << (checkpoint_tiles.empty() ? "constructed linear-router Top2 SwiGLU MoE"
                                       : "Qwen3-30B-A3B layer 0 BF16 weight tiles")
          << "\",\n  \"input_weight_overlap\": " << (overlap ? "true" : "false")
          << ",\n  \"weight_buffers\": " << (double_buffered ? 2 : 1)
          << ",\n  \"prefetches\": " << weight_stats.prefetches
          << ",\n  \"prefetch_hits\": " << weight_stats.prefetch_hits
          << ",\n  \"dma_segments\": " << weight_stats.dma_segments
          << ",\n  \"committed_weight_vectors\": " << weight_stats.committed_vectors
          << ",\n  \"topk_unit\": \"" << (checkpoint_tiles.empty() ? "RCU" : "VXM")
          << "\",\n  "
             "\"tokens\": 5, \"experts\": 4, \"top_k\": 2, \"hidden\": 32, "
             "\"intermediate\": 32,\n  \"router_weights\": \"selected "
             "FP32 logits; no softmax or renormalization\",\n  "
             "\"expert_dtype\": \"BF16 storage; existing VXM FP32 basic ALUs "
             "and FP16 LUT special functions\",\n  \"physical_cycles\": "
          << system->cycle()
          << ",\n  \"expert_tasks\": " << chip.expert_macros().trace().size()
          << ",\n  \"max_abs_error_vs_bf16_lut\": " << maximum
          << ",\n  \"relative_l2_vs_bf16_lut\": " << rel
          << ",\n  \"bit_mismatches\": " << mismatches
          << ",\n  \"relative_l2_vs_mathematical\": "
          << std::sqrt(math_square / std::max(math_scale, 1e-30))
          << ",\n  \"counts\": [";
  for (unsigned e = 0; e < E; ++e)
    summary << (e ? ", " : "") << golden.counts[e];
  summary << "],\n  \"status\": \"" << (mismatches == 0 ? "passed" : "failed")
          << "\"\n}\n";
  summary.flush();
  check(mismatches == 0, "MoE numeric reference failed; see output.csv");
  std::cout << "MoE SwiGLU T=" << T << " E=" << E << " K=" << K << " H=F=" << H
            << " tasks=" << chip.expert_macros().trace().size()
            << " cycles=" << system->cycle() << " max_abs=" << maximum
            << " relative_l2=" << rel << " bit_mismatches=" << mismatches
            << "\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << "moe_ffn_demo: " << e.what() << '\n';
  return 1;
}
