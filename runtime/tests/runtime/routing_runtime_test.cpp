#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/moe_router.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/routing_program.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"
#include "ftlpu/software/runtime/vxm_topk_program.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;
static void check(bool x, const char *m) {
  if (!x)
    throw std::runtime_error(m);
}
template <class F> static void rejects(F f) {
  try {
    f();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("missing rejection");
}
static RoutingTensorView view(RoutingElement f, std::uint32_t rows,
                              std::uint32_t cols, std::uint32_t slice,
                              std::uint32_t base, std::uint32_t h = 0,
                              std::uint32_t bank = 0) {
  RoutingTensorView v;
  v.element = f;
  v.rows = rows;
  v.columns = cols;
  v.feature_stride = rows;
  for (std::size_t p = 0; p < v.bytes(); ++p)
    v.planes[p] = {h, slice + static_cast<std::uint32_t>(p), bank, base};
  return v;
}
static void put(TspSliceSystem &s, const RoutingTensorView &v, std::size_t r,
                std::size_t c, std::uint32_t value) {
  for (std::size_t b = 0; b < v.bytes(); ++b) {
    auto p = v.planes[b];
    auto lane = c % hw::kPhysicalVectorBytes;
    s.initialize_mem_sram_lane_byte(
        static_cast<Hemisphere>(p.hemisphere), p.slice, p.bank,
        lane / hw::kLanesPerTile, v.row(r, c, b), lane % hw::kLanesPerTile,
        static_cast<std::uint8_t>(value >> (8 * b)));
  }
}
static std::uint32_t get(TspSliceSystem &s, const RoutingTensorView &v,
                         std::size_t r, std::size_t c) {
  std::uint32_t x = 0;
  for (std::size_t b = 0; b < v.bytes(); ++b) {
    auto p = v.planes[b];
    auto lane = c % hw::kPhysicalVectorBytes;
    x |=
        std::uint32_t(s.read_mem_sram_lane_byte(
            static_cast<Hemisphere>(p.hemisphere), p.slice, p.bank,
            lane / hw::kLanesPerTile, v.row(r, c, b), lane % hw::kLanesPerTile))
        << (8 * b);
  }
  return x;
}
static void scenario(bool timed) {
  rt::BinaryProgram p;
  rt::RoutingProgramBuilder b(p);
  RouteSelectDescriptor select;
  select.tokens = 3;
  select.experts = 4;
  select.k = 2;
  select.choice_scores = view(RoutingElement::F32, 3, 4, 0, 100);
  select.raw_scores = view(RoutingElement::F32, 3, 4, 4, 100);
  select.selected_scores = view(RoutingElement::F32, 3, 2, 8, 100);
  b.select(select, 1);
  b.plan({}, 2);
  RouteAddressDescriptor addr;
  addr.max_rows = 2;
  addr.source = view(RoutingElement::BF16, 3, 32, 12, 200);
  addr.destination = view(RoutingElement::BF16, 2, 32, 14, 300, 1, 1);
  b.address(addr, 3);
  b.completion_event(77, {3});
  const auto waiting = static_cast<std::uint16_t>(
      InstructionControlUnit::mem_queue(Hemisphere::West, 14, 1));
  b.wait_for(rt::QueueKind::Mem, waiting, 77);
  const auto independent_queue =
      InstructionControlUnit::mem_queue(Hemisphere::East, 30, 1);
  rt::IcuProgram independent;
  independent.emit_mem(200, independent_queue,
                       MemInstruction::Read(17, StreamId::East(15)));
  for (auto queue : independent.encode_queues())
    p.queues.push_back(std::move(queue));
  p.max_cycle = 201;
  std::ostringstream out(std::ios::binary);
  rt::write_binary_program(p, out);
  const auto bytes = out.str();
  const auto span = std::span(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
  auto loaded = rt::read_binary_program(span);
  auto meta = rt::read_binary_program_metadata(span);
  std::istringstream stream(bytes, std::ios::binary);
  auto streamLoaded = rt::read_binary_program(stream);
  check(loaded.hardware.routing_isa_version == 1 &&
            loaded.routing_descriptors.size() == 3 &&
            meta.routing_descriptors.size() == 3 &&
            streamLoaded.routing_descriptors.size() == 3,
        "v41 reader paths");
  SystemHardwareConfiguration hwconfig;
  hwconfig.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, hwconfig);
  rt::CModelRuntime runtime(*system);
  runtime.enable_execution_trace(true);
  for (std::size_t t = 0; t < 3; ++t) {
    for (std::size_t e = 0; e < 4; ++e) {
      put(system->chip(), select.choice_scores, t, e,
          std::bit_cast<std::uint32_t>(static_cast<float>(e)));
      put(system->chip(), select.raw_scores, t, e,
          0x3f000000u + static_cast<std::uint32_t>(t * 4 + e));
    }
    for (std::size_t c = 0; c < 32; ++c)
      put(system->chip(), addr.source, t, c,
          static_cast<std::uint32_t>(100 * t + c));
  }
  if (timed)
    runtime.load_timed(loaded);
  else
    runtime.load(loaded);
  if (timed)
    check(!runtime.all_images_committed() &&
              system->chip().routing().statistics().comparisons == 0,
          "Routing issued before timed commit");
  bool observed_busy = false, observed_gather = false,
       observed_unrelated = false;
  for (std::size_t n = 0; n < 100000 && !runtime.kernel_done(); ++n) {
    runtime.run_until_complete(1);
    if (system->chip().routing().busy()) {
      observed_busy = true;
      if (system->chip().icu().mem_iq(independent_queue).issued_count())
        observed_unrelated = true;
      check(!runtime.can_load_program(),
            "busy RCU allowed program replacement");
    }
    if (system->chip().routing().block_info().valid_rows &&
        system->chip().routing().statistics().commit_segments < 16) {
      observed_gather = true;
      check(!runtime.kernel_done(), "Gather not drained at kernel done");
    }
    if (runtime.all_images_committed() &&
        system->chip().icu().queue_completion_ready(77))
      check(system->chip().routing().statistics().commit_segments == 16,
            "INPUT_READY before last physical commit");
  }
  check(observed_busy && observed_gather && observed_unrelated &&
            runtime.kernel_done(),
        "Routing runtime did not complete or skipped physical phases");
  check(runtime.kernel_done() && runtime.can_load_program(),
        "Routing resources did not drain");
  check(system->chip().routing().counts() ==
            std::vector<std::uint32_t>({0, 0, 3, 3}),
        "device TopK/counts");
  check(system->chip().routing().block_info().expert_id == 3,
        "expert was not selected on device");
  for (std::size_t t = 0; t < 2; ++t)
    for (std::size_t c = 0; c < 32; ++c)
      check(get(system->chip(), addr.destination, t, c) == 100 * t + c,
            "runtime real Gather mismatch");
  for (std::size_t t = 0; t < 3; ++t)
    check(get(system->chip(), select.selected_scores, t, 0) ==
              0x3f000000u + t * 4 + 3,
          "raw bits not preserved");
  check(system->chip().routing().statistics().commit_segments == 16,
        "early INPUT_READY");
  check(system->chip().icu().mem_iq(waiting).done(),
        "device Routing completion did not release WAIT");
  if (timed) {
    bool desc = false;
    for (const auto &im : runtime.program_load_statistics().images)
      if (im.image.storage == ProgramStorage::RoutingDescriptors) {
        desc = true;
        check(im.image_committed.has_value(), "descriptors not committed");
      }
    check(desc, "descriptor DMA image missing");
  }
  std::cout
      << (timed ? "timed" : "direct") << " routing cycles=" << system->cycle()
      << " comparisons=" << system->chip().routing().statistics().comparisons
      << " metadata=" << system->chip().routing().statistics().metadata_accesses
      << " io=" << system->chip().routing().statistics().control_requests
      << " descriptor_words="
      << system->chip().routing().statistics().descriptor_words
      << " bank_wait=" << system->chip().routing().statistics().bank_wait_cycles
      << " address_wait="
      << system->chip().routing().statistics().address_wait_cycles
      << " sr_wait=" << system->chip().routing().statistics().stream_wait_cycles
      << " fifo_peak=" << system->chip().routing().statistics().peak_addresses
      << " commits=" << system->chip().routing().statistics().commit_segments
      << '\n';
  std::ostringstream performance;
  rt::print_runtime_performance(loaded, system->cycle(), performance);
  check(performance.str().find("resource=RCU") != std::string::npos,
        "RCU missing from performance report");
  check(system->chip().icu().frontend_statistics().issued_instructions == 4,
        "frontend omitted Routing issues");
  const auto trace_dir = std::filesystem::path("test-output");
  std::filesystem::create_directories(trace_dir);
  const auto trace_file =
      trace_dir / (timed ? "routing-timed.csv" : "routing-direct.csv");
  runtime.write_execution_trace_csv(trace_file);
  std::ifstream input(trace_file);
  const std::string trace_text((std::istreambuf_iterator<char>(input)), {});
  check(trace_text.find("RCU.Command") != std::string::npos &&
            trace_text.find("RCU.Execute") != std::string::npos,
        "Routing execution trace omitted device work");
  std::istringstream trace_rows(trace_text);
  std::string row;
  std::size_t issued_commands = 0;
  while (std::getline(trace_rows, row))
    if (row.find("RCU.Command") != std::string::npos) {
      ++issued_commands;
      check(row.find("issues=1\"") != std::string::npos,
            "held frontend repeated its issue trace");
    }
  check(issued_commands == 3, "Routing trace lost a command");
  rt::write_schedule_trace_csv(loaded, trace_dir / "routing-schedule.csv");
  const auto old_generation = system->chip().icu().routing_generation();
  system->chip().icu().routing_completion(old_generation,
                                          3); // Duplicate is idempotent.
  check(system->chip().icu().queue_completion_ready(77),
        "duplicate notification broke ready event");
  rejects(
      [&] { system->chip().icu().routing_completion(old_generation + 1, 3); });
  runtime.load(loaded);
  system->chip().icu().routing_completion(old_generation, 3);
  check(!system->chip().icu().queue_completion_ready(77),
        "old generation released new program");
  check(runtime.run_until_complete(100000),
        "reloaded Routing program did not drain");
  SystemHardwareConfiguration disabled_config;
  disabled_config.routing.enabled = false;
  auto disabled = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, disabled_config);
  rt::CModelRuntime disabledRuntime(*disabled);
  rejects([&] { disabledRuntime.load(loaded); });
}
static void vxm_producer() {
  rt::IcuProgram producer;
  VxmLaneAluInstruction head;
  head.operation = VxmAluOpcode::Bypass;
  head.lhs = VxmLaneOperand::Imm(0.75f);
  head.rhs = VxmLaneOperand::Imm(0.75f);
  head.precision = VxmAluPrecision::Float32;
  head.output_type = VxmCastTarget::Float32;
  auto tail = head;
  tail.lhs = VxmLaneOperand::Previous();
  tail.output_stream = 0;
  producer.emit_vxm(0, 0, VxmChainDepth::Two, head);
  producer.emit_vxm(0, 1, VxmChainDepth::Two, tail);
  // Conditional taps capture the physical VXM result at its MEM boundary;
  // all subsequent routing runs after this statically drained producer phase.
  for (std::size_t byte = 0; byte < 4; ++byte)
    for (std::size_t cycle = 1; cycle < 24; ++cycle)
      producer.emit_mem(
          cycle, InstructionControlUnit::mem_queue(Hemisphere::West, byte, 0),
          MemInstruction::WriteTap(400, StreamId::East(byte)));
  rt::BinaryProgram p;
  p.queues = producer.encode_queues();
  p.max_cycle = 64;
  rt::RoutingProgramBuilder builder(p);
  RouteSelectDescriptor select;
  select.tokens = 1;
  select.experts = 4;
  select.k = 2;
  select.choice_scores = view(RoutingElement::F32, 1, 4, 0, 400, 1);
  builder.select(select, 1);
  builder.plan({}, 2);
  RouteAddressDescriptor a;
  a.source = view(RoutingElement::F16, 1, 32, 12, 500);
  a.destination = view(RoutingElement::F16, 1, 32, 14, 600, 1, 1);
  a.max_rows = 1;
  builder.address(a, 3);
  auto q = std::ranges::find_if(
      p.queues, [](const auto &v) { return v.kind == rt::QueueKind::Routing; });
  q->commands.insert(
      q->commands.begin(),
      {isa::encode_icu_nop(64), rt::InstructionKind::None, 0, {}});
  SystemHardwareConfiguration hardware;
  hardware.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, hardware);
  rt::CModelRuntime runtime(*system);
  for (std::size_t c = 0; c < 32; ++c)
    put(system->chip(), a.source, 0, c, static_cast<std::uint32_t>(c + 1000));
  runtime.load(p);
  check(runtime.run_until_complete(100000), "VXM to RCU pipeline timeout");
  check(get(system->chip(), select.choice_scores, 0, 0) ==
            std::bit_cast<std::uint32_t>(0.75f),
        "VXM did not commit FP32 route scores");
  check(system->chip().routing().selected_experts() ==
            std::vector<std::uint32_t>({0, 1}),
        "device VXM TopK tie rule");
  for (std::size_t c = 0; c < 32; ++c)
    check(get(system->chip(), a.destination, 0, c) == c + 1000,
          "VXM to RCU to Gather payload");
  std::cout << "VXM -> SRAM -> RCU -> Gather passed\n";
}
static void reference_and_invalid() {
  constexpr std::size_t T = 5, E = 7, H = 4, K = 3;
  std::vector<float> hidden(T * H), weights(E * H);
  for (std::size_t n = 0; n < hidden.size(); ++n)
    hidden[n] = static_cast<float>(int((n * 7) % 13) - 6) / 8;
  for (std::size_t n = 0; n < weights.size(); ++n)
    weights[n] = static_cast<float>(int((n * 3) % 11) - 5) / 4;
  const auto reference =
      rt::plan_qwen3_moe_routes(hidden, weights, {T, H, E, K, true});
  rt::BinaryProgram p;
  rt::RoutingProgramBuilder b(p);
  RouteSelectDescriptor select;
  select.tokens = T;
  select.experts = E;
  select.k = K;
  select.choice_scores = view(RoutingElement::F32, T, E, 0, 64);
  b.select(select, 1);
  RoutePlanDescriptor plan;
  plan.weights = view(RoutingElement::F32, T, K, 4, 64);
  b.plan(plan, 2);
  SystemHardwareConfiguration cfg;
  cfg.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, cfg);
  rt::CModelRuntime runtime(*system);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t e = 0; e < E; ++e) {
      float logit = 0;
      for (std::size_t f = 0; f < H; ++f)
        logit = std::fma(hidden[t * H + f], weights[e * H + f], logit);
      put(system->chip(), select.choice_scores, t, e,
          std::bit_cast<std::uint32_t>(logit));
    }
    for (std::size_t k = 0; k < K; ++k)
      put(system->chip(), plan.weights, t, k,
          std::bit_cast<std::uint32_t>(reference.routes[t * K + k].weight));
  }
  runtime.load(p);
  check(runtime.run_until_complete(100000), "CPU reference comparison timeout");
  const auto &rcu = system->chip().routing();
  check(rcu.active_experts() == reference.active_experts,
        "CPU router active order differs");
  for (std::size_t i = 0; i < T * K; ++i) {
    check(rcu.selected_experts()[i] == reference.routes[i].expert,
          "CPU router TopK differs");
    const auto a = rcu.assignments()[i];
    const auto b = reference.assignments[i];
    check(a.token == b.token && a.slot == b.slot &&
              a.weight_bits == std::bit_cast<std::uint32_t>(b.weight),
          "CPU router stable assignment/weight differs");
  }
  for (std::size_t i = 0; i <= E; ++i)
    check(rcu.offsets()[i] == reference.expert_offsets[i],
          "CPU router CSR differs");
  for (unsigned mutation = 0; mutation < 7; ++mutation) {
    auto bad = p;
    switch (mutation) {
    case 0:
      bad.hardware.routing_isa_version = 0;
      break;
    case 1:
      bad.queues[0].commands[0].words[1] = 31;
      break;
    case 2:
      bad.queues[0].commands[1].words[2] = 1;
      break;
    case 3:
      bad.queues[0].index = 1;
      break;
    case 4:
      bad.queues[0].requires_raw_macro = true;
      break;
    case 5:
      bad.routing_descriptors[0].select.tokens = 129;
      break;
    case 6:
      bad.queues[0].commands[0].words[2] = 0x80000000u;
      break;
    }
    rejects([&] {
      std::ostringstream out;
      rt::write_binary_program(bad, out);
    });
    rejects([&] { runtime.load(bad); });
  }
  std::ostringstream good;
  rt::write_binary_program(p, good);
  auto data = good.str();
  const auto magic = data.find("RCU1");
  check(magic != std::string::npos, "RCU section missing");
  data[magic] = 'X';
  for (unsigned reader = 0; reader < 4; ++reader)
    rejects([&] {
      auto span = std::span(reinterpret_cast<const std::uint8_t *>(data.data()),
                            data.size());
      std::istringstream in(data);
      switch (reader) {
      case 0:
        (void)rt::read_binary_program(span);
        break;
      case 1:
        (void)rt::read_binary_program(in);
        break;
      case 2:
        (void)rt::read_binary_program_metadata(span);
        break;
      case 3:
        (void)rt::read_binary_program_metadata(in);
        break;
      }
    });
  // A v40 fixture is the v41 non-Routing executable without the optional
  // section.
  rt::BinaryProgram capacity;
  rt::RoutingProgramBuilder bounded(capacity);
  RouteSelectDescriptor small;
  small.tokens = small.experts = small.k = 1;
  small.choice_scores = view(RoutingElement::F32, 1, 1, 0, 0);
  for (std::uint32_t n = 1; n <= 32; ++n)
    bounded.select(small, n);
  rejects([&] { bounded.select(small, 33); });
  check(capacity.routing_descriptors.size() == 32 &&
            capacity.queues[0].commands.size() == 32,
        "descriptor overflow partially appended a command");
  auto command = capacity.queues[0].commands[0];
  for (std::uint32_t n = 33; n <= 256; ++n) {
    command.words[2] = n;
    capacity.queues[0].commands.push_back(command);
  }
  std::ostringstream full;
  rt::write_binary_program(capacity, full);
  std::istringstream full_input(full.str());
  check(rt::read_binary_program(full_input).queues[0].commands.size() == 256,
        "full Routing iMEM roundtrip");
  command.words[2] = 257;
  capacity.queues[0].commands.push_back(command);
  rejects([&] {
    std::ostringstream bad;
    rt::write_binary_program(capacity, bad);
  });
  rt::BinaryProgram legacy;
  std::ostringstream old;
  rt::write_binary_program(legacy, old);
  auto v40 = old.str();
  const auto start = v40.find("RCU1");
  v40.erase(start - 8, 16);
  v40[8] = 40;
  auto oldspan =
      std::span(reinterpret_cast<const std::uint8_t *>(v40.data()), v40.size());
  auto read = rt::read_binary_program(oldspan);
  check(!read.hardware.routing_isa_version && read.routing_descriptors.empty(),
        "v40 did not disable Routing");
  std::cout
      << "existing CPU router stages and v40/v41 rejection checks passed\n";
}
static void native_sync() {
  for (bool routing_producer : {false, true}) {
    rt::BinaryProgram p;
    rt::RoutingProgramBuilder builder(p);
    RouteSelectDescriptor s;
    s.tokens = s.experts = s.k = 1;
    s.choice_scores = view(RoutingElement::F32, 1, 1, 0, 0);
    builder.select(s, 1);
    const auto control = rt::encode_icu_control_raw_word(
        routing_producer ? IcuControlInstruction::Notify()
                         : IcuControlInstruction::Sync());
    p.queues.front().commands.insert(p.queues.front().commands.begin(),
                                     control);
    rt::QueueProgram mem{
        rt::QueueKind::Mem,
        static_cast<std::uint16_t>(
            InstructionControlUnit::mem_queue(Hemisphere::East, 30, 1)),
        {}};
    if (!routing_producer)
      mem.commands.push_back(
          rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(10)));
    mem.commands.push_back(rt::encode_icu_control_raw_word(
        routing_producer ? IcuControlInstruction::Sync()
                         : IcuControlInstruction::Notify()));
    p.queues.push_back(mem);
    p.max_cycle = 12;
    SystemHardwareConfiguration cfg;
    cfg.routing.enabled = true;
    auto system = std::make_unique<C2cDmaSystem>(
        Ddr4Config{}, hw::config::kDmaFifoDepthVectors, cfg);
    rt::CModelRuntime runtime(*system);
    runtime.load(p);
    check(runtime.run_until_complete(100000),
          "Routing native NOTIFY/SYNC did not propagate");
    check(system->chip().routing().statistics().comparisons == 1,
          "Routing SYNC bypassed/lost its command");
  }
}
// Test-only clocked endpoint: no runtime reads a route result to choose a task
// or issue a per-task command. SRAM inspection below is assertion-only.
struct DispatchConsumer : ExpertTaskConsumer {
  TspSliceSystem &chip;
  RoutingTensorView destination;
  int group;
  std::vector<ExpertTaskContext> accepted;
  ExpertTaskContext offered{}, current{};
  std::vector<std::uint32_t> before;
  std::uint64_t committed{};
  unsigned offers{}, countdown{};
  bool input{}, sent_old{};
  DispatchConsumer(TspSliceSystem &s, RoutingTensorView dst, int g)
      : chip(s), destination(dst), group(g) {}
  bool accept_task(const ExpertTaskContext &task) override {
    if (!offers)
      offered = task;
    check(offered == task, "task changed under accept backpressure");
    if (++offers < (group == 0 ? 7u : 1u))
      return false;
    offers = 0;
    current = task;
    const auto index = accepted.size();
    check(index < 3, "duplicate task");
    const std::uint32_t experts[] = {3, 3, 1}, rows[] = {0, 32, 0},
                        valid[] = {32, 5, 5};
    check(task.expert_id == experts[index] && task.row_offset == rows[index] &&
              task.valid_rows == valid[index] && task.more == (index == 0),
          "task sequence");
    check(task.assignment_base == (index == 0   ? 5u
                                   : index == 1 ? 37u
                                                : 0u),
          "assignment base");
    accepted.push_back(task);
    before.clear();
    for (std::size_t r = task.valid_rows; r < 32; ++r)
      for (std::size_t c = 0; c < 32; ++c)
        before.push_back(get(chip, destination, r, c));
    return true;
  }
  void input_ready(const ExpertTaskContext &task) override {
    check(current == task && !input, "wrong/duplicate input notification");
    committed += task.valid_rows * 8;
    check(chip.routing().statistics().commit_segments == committed,
          "input ready before physical commit");
    const auto first_token = task.expert_id == 3 ? task.row_offset : 37u;
    for (std::size_t r = 0; r < task.valid_rows; ++r)
      for (std::size_t c = 0; c < 32; ++c)
        check(get(chip, destination, r, c) == (first_token + r) * 100 + c,
              "dispatch real Gather bytes");
    std::size_t at = 0;
    for (std::size_t r = task.valid_rows; r < 32; ++r)
      for (std::size_t c = 0; c < 32; ++c)
        check(get(chip, destination, r, c) == before.at(at++),
              "tail overwritten");
    input = true;
    countdown = 5;
    sent_old = false;
  }
  std::optional<ExpertTaskIdentity> poll_done() override {
    if (!input)
      return {};
    if (group == 2 && current.identity.task_id > 1 && !sent_old) {
      sent_old = true;
      auto old = current.identity;
      --old.task_id;
      return old;
    }
    if (--countdown)
      return {};
    input = false;
    return current.identity;
  }
};

static void dispatch_group(int group) {
  rt::BinaryProgram p;
  rt::RoutingProgramBuilder b(p);
  RouteSelectDescriptor s;
  s.tokens = 42;
  s.experts = 4;
  s.k = 1;
  s.choice_scores = view(RoutingElement::F32, 42, 4, 0, 0);
  b.select(s, 1);
  b.plan({}, 2);
  RouteAddressDescriptor a;
  a.source = view(RoutingElement::BF16, 42, 32, 12, 0);
  a.destination = view(RoutingElement::BF16, 32, 32, 14, 0, 0, 1);
  const auto template_id = b.address_template(a);
  b.dispatch({1, template_id, 32}, 3);
  b.completion_event(77, {3});
  b.wait_for(rt::QueueKind::Mem,
             static_cast<std::uint16_t>(
                 InstructionControlUnit::mem_queue(Hemisphere::East, 14, 1)),
             77);
  std::ostringstream out(std::ios::binary);
  rt::write_binary_program(p, out);
  const auto bytes = out.str();
  auto loaded = rt::read_binary_program(std::span(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  check(loaded.hardware.routing_isa_version == 2 &&
            loaded.routing_descriptors.size() == 4,
        "dispatch codec/capability");
  if (group == 2) {
    auto bad = loaded;
    bad.hardware.routing_isa_version = 1;
    rejects([&] {
      std::ostringstream os;
      rt::write_binary_program(bad, os);
    });
    bad = loaded;
    bad.routing_descriptors.back().dispatch.block_rows = 0;
    rejects([&] {
      std::ostringstream os;
      rt::write_binary_program(bad, os);
    });
  }
  SystemHardwareConfiguration cfg;
  cfg.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, cfg);
  auto &chip = system->chip();
  DispatchConsumer consumer(chip, a.destination, group);
  chip.expert_tasks().connect(&consumer);
  rt::CModelRuntime runtime(*system);
  for (std::size_t t = 0; t < 42; ++t) {
    for (std::size_t e = 0; e < 4; ++e)
      put(chip, s.choice_scores, t, e,
          std::bit_cast<std::uint32_t>(e == (t < 37 ? 3u : 1u) ? 1.f : 0.f));
    for (std::size_t c = 0; c < 32; ++c)
      put(chip, a.source, t, c, static_cast<std::uint32_t>(100 * t + c));
  }
  runtime.load(loaded);
  bool guarded = false;
  for (std::size_t n = 0; n < 100000 && !runtime.kernel_done(); ++n) {
    runtime.run_until_complete(1);
    if (chip.routing().plan_in_use()) {
      check(!runtime.can_load_program(), "leased plan allowed program switch");
      if (group == 2 && !guarded) {
        RoutingDescriptor select;
        select.select = s;
        rejects([&] {
          chip.routing().start({RoutingOpcode::Select, 0, 20}, select,
                               chip.icu().routing_generation());
        });
        rejects([&] { chip.reset_program(); });
        guarded = true;
      }
      if (consumer.input)
        check(!chip.icu().queue_completion_ready(77),
              "input ready mistaken for dispatch done");
    }
  }
  check(runtime.kernel_done() && runtime.can_load_program() &&
            consumer.accepted.size() == 3 &&
            chip.expert_tasks().statistics().completed == 3 &&
            !chip.routing().plan_in_use(),
        "dispatch failed to drain");
  if (group == 2)
    check(guarded, "lifecycle guard not tested");
  check(chip.icu().queue_completion_ready(77),
        "dispatch event did not release WAIT");
  std::cout << "dispatch group=" << group
            << " tasks=3 cycles=" << system->cycle() << '\n';
}

struct WeightConsumer final : DispatchConsumer {
  const ExpertWeightTableDescriptor &table;
  unsigned notifications{};
  WeightConsumer(TspSliceSystem &s, RoutingTensorView dst, int g,
                 const ExpertWeightTableDescriptor &t)
      : DispatchConsumer(s, dst, g), table(t) {}
  void input_ready(const ExpertTaskContext &task) override {
    DispatchConsumer::input_ready(task);
    input = false; // Expert execution cannot complete before WEIGHT_READY.
  }
  void weights_ready(const ExpertTaskContext &task) override {
    check(task == current && chip.expert_weights().ready(task.identity),
          "wrong weight ready identity");
    const auto &stats = chip.expert_weights().statistics();
    const auto loads = task.expert_id == 3 ? 2u : 4u;
    check(stats.dma_segments == loads &&
              stats.committed_vectors == (loads / 2) * 3,
          "weight ready before final SRAM write");
    check(stats.residency_hits == (notifications ? 1u : 0u),
          "expert residency was not reused");
    for (std::size_t j = 0; j < 2; ++j) {
      const auto segment =
          expert_weight_segment(table, table.words[2 * task.expert_id] + j);
      for (std::size_t r = 0; r < segment.vector_count; ++r)
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
          for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto expected = static_cast<std::uint8_t>(
                task.expert_id * 40 + j * 20 + r * 7 + tile * 8 + lane);
            check(chip.read_mem_sram_lane_byte(
                      static_cast<Hemisphere>(segment.hemisphere),
                      segment.slice, segment.bank, tile, segment.base_row + r,
                      lane) == expected,
                  "indirect weight DMA bytes");
          }
    }
    if (group == 2) {
      auto old = task.identity;
      if (old.task_id > 1) {
        --old.task_id;
        check(!chip.expert_weights().ready(old),
              "old weight event released new task");
      }
      auto future = task.identity;
      ++future.task_id;
      rejects([&] { chip.expert_weights().release_task(future); });
    }
    if (group == 1) {
      const auto segment =
          expert_weight_segment(table, table.words[2 * task.expert_id]);
      auto &mem = chip.mem_array(static_cast<Hemisphere>(segment.hemisphere));
      rejects([&] {
        mem.enqueue_instruction(
            segment.slice, segment.bank,
            MemInstruction::Write(segment.base_row, StreamId::East(29)));
      });
      mem.enqueue_instruction(
          segment.slice, segment.bank,
          MemInstruction::Read(segment.base_row, StreamId::East(29)));
    }
    ++notifications;
    input = true;
    countdown = 5;
    sent_old = false;
  }
};

static void weight_group(int group) {
  rt::BinaryProgram p;
  rt::RoutingProgramBuilder b(p);
  RouteSelectDescriptor s;
  s.tokens = 42;
  s.experts = 4;
  s.k = 1;
  s.choice_scores = view(RoutingElement::F32, 42, 4, 0, 0);
  b.select(s, 1);
  b.plan({}, 2);
  RouteAddressDescriptor a;
  a.source = view(RoutingElement::BF16, 42, 32, 12, 0);
  a.destination = view(RoutingElement::BF16, 32, 32, 14, 0, 0, 1);
  auto address = b.address_template(a);
  b.dispatch({1, address, 32}, 3);
  std::vector<ExpertWeightEntry> entries{{0, 2}, {0, 2}, {0, 2}, {2, 2}};
  std::vector<ExpertWeightSegment> segments{{0x20000, 2, 0, 20, 1, 10, 30},
                                            {0x21000, 1, 0, 21, 1, 20, 30},
                                            {0x22000, 2, 0, 20, 1, 10, 30},
                                            {0x23000, 1, 0, 21, 1, 20, 30}};
  b.weight_table(1, entries, segments);
  b.completion_event(77, {3});
  b.wait_for(rt::QueueKind::Mem,
             static_cast<std::uint16_t>(
                 InstructionControlUnit::mem_queue(Hemisphere::East, 14, 1)),
             77);
  if (group == 2) {
    auto bad = segments;
    bad[0].vector_count = 0;
    rejects([&] {
      validate_expert_weight_table(make_expert_weight_table(1, entries, bad));
    });
    bad = segments;
    bad[0].ddr_address = UINT64_MAX - 31;
    rejects([&] {
      validate_expert_weight_table(make_expert_weight_table(1, entries, bad));
    });
    bad = segments;
    bad[1].slice = bad[0].slice;
    bad[1].base_row = bad[0].base_row;
    rejects([&] {
      validate_expert_weight_table(make_expert_weight_table(1, entries, bad));
    });
    auto invalid = p;
    invalid.hardware.routing_isa_version = 2;
    rejects([&] {
      std::ostringstream os;
      rt::write_binary_program(invalid, os);
    });
  }
  std::ostringstream os(std::ios::binary);
  rt::write_binary_program(p, os);
  const auto bytes = os.str();
  auto loaded = rt::read_binary_program(std::span(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  check(loaded.hardware.routing_isa_version == 3, "weight table capability");
  SystemHardwareConfiguration cfg;
  cfg.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, cfg);
  auto &chip = system->chip();
  const auto &table = loaded.routing_descriptors.back().weight_table;
  WeightConsumer consumer(chip, a.destination, group, table);
  chip.expert_tasks().connect(&consumer);
  rt::CModelRuntime runtime(*system);
  for (std::size_t t = 0; t < 42; ++t) {
    for (std::size_t e = 0; e < 4; ++e)
      put(chip, s.choice_scores, t, e,
          std::bit_cast<std::uint32_t>(e == (t < 37 ? 3u : 1u) ? 1.f : 0.f));
    for (std::size_t c = 0; c < 32; ++c)
      put(chip, a.source, t, c, static_cast<std::uint32_t>(100 * t + c));
  }
  for (std::size_t i = 0; i < segments.size(); ++i) {
    const auto &seg = segments[i];
    const auto expert = i < 2 ? 1u : 3u;
    const auto j = i % 2;
    for (std::size_t r = 0; r < seg.vector_count; ++r) {
      C2cVector vector{};
      for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
          vector.payload[tile][lane] = static_cast<std::uint8_t>(
              expert * 40 + j * 20 + r * 7 + tile * 8 + lane);
      system->ddr4().initialize_vector(
          seg.ddr_address + r * hw::kPhysicalVectorBytes, vector);
    }
  }
  runtime.load(loaded);
  bool guarded = false;
  for (std::size_t n = 0; n < 100000 && !runtime.kernel_done(); ++n) {
    runtime.run_until_complete(1);
    if (!chip.expert_weights().idle()) {
      check(!runtime.can_load_program() &&
                !chip.icu().queue_completion_ready(77),
            "weights allowed early program switch/completion");
      if (group == 2 && !guarded) {
        rejects([&] { chip.reset_program(); });
        guarded = true;
      }
    }
  }
  check(runtime.kernel_done() && runtime.can_load_program() &&
            consumer.notifications == 3 && chip.expert_weights().idle(),
        "indirect weight pipeline did not drain");
  const auto &stats = chip.expert_weights().statistics();
  check(stats.dma_segments == 4 && stats.committed_vectors == 6 &&
            stats.residency_hits == 1 && stats.table_reads == 68,
        "weight load/reuse accounting");
  if (group == 2)
    check(guarded, "weight lifecycle guard untested");
  std::cout << "weight group=" << group
            << " loads=4 vectors=6 hits=1 cycles=" << system->cycle() << '\n';
}

static void vxm_topk_submission(bool timed, bool pack_on_device = false) {
  constexpr std::size_t E = 17, K = 5;
  const std::size_t T = pack_on_device ? 9 : 35;
  constexpr std::size_t pitch = RouteVxmSelectSchedule::workspace_blocks(E);
  RouteSelectDescriptor select;
  select.tokens = T;
  select.experts = E;
  select.k = K;
  select.choice_scores = view(RoutingElement::F32, T, E, 0, 400, 1);
  select.raw_scores = view(RoutingElement::F32, T, E, 4, 400, 1);
  select.eligibility = view(RoutingElement::U32, T, E, 8, 400, 1);
  select.selected_ids = view(RoutingElement::U32, T, K, 12, 400, 1);
  select.selected_scores = view(RoutingElement::F32, T, K, 16, 400, 1);
  RouteVxmSelectSchedule schedule;
  schedule.first_bundle_cycle = 600;
  schedule.bundle_interval = 4;
  schedule.raw_offset = 2 * pitch;
  schedule.eligibility_offset = 4 * pitch;
  for (std::size_t stream = 0; stream < schedule.streams; ++stream)
    schedule.score_planes[stream] = {pack_on_device ? 1u : 0u,
                                     static_cast<std::uint32_t>(stream), 0, 900};

  RouteAddressDescriptor block;
  block.max_rows = 4;
  block.source = view(RoutingElement::BF16, T, 32, 40, 300, 0);
  block.destination = view(RoutingElement::BF16, 4, 32, 42, 400, 0);
  rt::BinaryProgram program;
  rt::RoutingProgramBuilder builder(program);
  if (pack_on_device) schedule = builder.pack_vxm_scores(select, schedule, 10);
  builder.select_vxm(select, schedule, 1);
  builder.completion_event(77, {1});
  builder.wait_for(rt::QueueKind::Routing, 0, 77);
  builder.plan({}, 2);
  builder.address(block, 3);
  if (pack_on_device) {
    for (const auto& queue : program.queues) {
      std::set<std::size_t> issues;
      for (const auto& command : queue.commands) {
        if (!rt::is_macro_schedule_command(command)) continue;
        const auto op = rt::decode_macro_schedule_command(command);
        for (std::size_t j = 0; j < op.inner_count; ++j)
          if (!issues.insert(op.start_cycle + j * op.inner_interval).second)
            throw std::logic_error("packing MEM overlap queue=" + std::to_string(queue.index)
                                   + " cycle=" + std::to_string(op.start_cycle + j * op.inner_interval));
      }
    }
  }
  std::stringstream image(std::ios::in | std::ios::out | std::ios::binary);
  rt::write_binary_program(program, image);
  image.seekg(0);
  const auto loaded = rt::read_binary_program(image);
  check(loaded.hardware.routing_isa_version == 6, "VXM SELECT capability roundtrip");

  SystemHardwareConfiguration hardware;
  hardware.routing.enabled = true;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, hardware);
  rt::CModelRuntime runtime(*system);
  auto& chip = system->chip();
  std::vector<std::uint32_t> ids, raw, active, counts(E);
  const auto write_workspace = [&](std::size_t t, std::size_t e,
                                   std::size_t offset, std::uint32_t bits) {
    for (std::size_t byte = 0; byte < 4; ++byte) {
      const auto& p = schedule.score_planes[4 * (e % 8) + byte];
      chip.mem_array(static_cast<Hemisphere>(p.hemisphere)).set_sram_byte(
          p.slice, p.bank, p.base_row + (t / 32) * pitch + e / 8 + offset,
          t % 32, static_cast<std::uint8_t>(bits >> (8 * byte)));
    }
  };
  for (std::size_t t = 0; t < T; ++t) {
    std::vector<std::pair<float, std::uint32_t>> candidates;
    for (std::size_t e = 0; e < E; ++e) {
      const bool eligible = (e + t) % 5 != 1;
      const float score = e == 16 ? std::numeric_limits<float>::infinity()
          : float(int((e * 3 + t * 7) % 9) - 4);
      const float original = float(t * E + e + 1) / 1024;
      const auto bits = std::bit_cast<std::uint32_t>(original);
      put(chip, select.raw_scores, t, e, bits);
      if (pack_on_device) {
        put(chip, select.choice_scores, t, e, std::bit_cast<std::uint32_t>(score));
        put(chip, select.eligibility, t, e, eligible ? 0x80000000u : 0u);
      } else {
        write_workspace(t, e, 0, std::bit_cast<std::uint32_t>(score));
        write_workspace(t, e, schedule.raw_offset, bits);
        write_workspace(t, e, schedule.eligibility_offset, eligible ? 0x80000000u : 0u);
      }
      if (eligible) candidates.emplace_back(score, static_cast<std::uint32_t>(e));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    for (std::size_t r = 0; r < K; ++r) {
      const auto e = candidates[r].second;
      ids.push_back(e);
      raw.push_back(std::bit_cast<std::uint32_t>(float(t * E + e + 1) / 1024));
      if (!counts[e]++) active.push_back(e);
    }
    for (std::size_t c = 0; c < 32; ++c)
      put(chip, block.source, t, c, static_cast<std::uint32_t>(t * 32 + c + 1));
  }

  if (timed) {
    runtime.load_timed(loaded);
  } else {
    runtime.load(loaded);
  }
  try {
    check(runtime.run_until_complete(40000), "VXM Top-K program did not drain");
  } catch (...) {
    if (pack_on_device) std::cerr << "pack cycle=" << chip.cycle() << '\n';
    throw;
  }
  const auto& rcu = chip.routing();
  check(rcu.selected_experts() == ids && rcu.selected_score_bits() == raw,
        "VXM selected IDs or raw score bits differ");
  check(rcu.counts() == counts && rcu.active_experts() == active,
        "VXM selection changed the RCU plan order");
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t r = 0; r < K; ++r) {
      check(get(chip, select.selected_ids, t, r) == ids[t * K + r], "VXM selected ID export");
      check(get(chip, select.selected_scores, t, r) == raw[t * K + r], "VXM selected score export");
    }
  const auto expert = active.front();
  const auto rows = std::min<std::uint32_t>(4, counts[expert]);
  std::vector<std::size_t> expected_tokens;
  for (std::size_t i = 0; i < ids.size(); ++i)
    if (ids[i] == expert) expected_tokens.push_back(i / K);
  check(rcu.block_info().valid_rows == rows, "VXM route block shape");
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < 32; ++c)
      check(get(chip, block.destination, r, c) == expected_tokens[r] * 32 + c + 1,
            "VXM Top-K Gather payload");
  const auto& stats = chip.vxm_unit().topk().statistics();
  check(stats.batches == (T + 31) / 32 && stats.submitted_words == 2 * T * K &&
        chip.icu().queue_completion_ready(77), "VXM selection commit accounting");
  std::cout << "VXM Top-K timed=" << timed << " device_pack=" << pack_on_device
            << " cycles=" << chip.cycle()
            << " score_bundles=" << stats.score_bundles
            << " submit_words=" << stats.submitted_words << '\n';
}

static void standalone_vxm_topk(bool timed, bool import = false) {
  constexpr std::uint32_t columns = 17, k = 5;
  const std::uint32_t rows = import ? 9 : 35;
  const VxmTopKInstruction instruction{
      rows, columns, k, static_cast<std::uint16_t>(import ? 0xffff : 0xff00), timed};
  constexpr auto blocks = (columns + VxmRouteTopK::chains - 1) / VxmRouteTopK::chains;
  rt::VxmTopKWorkspace workspace;
  for (std::size_t i = 0; i < workspace.input.size(); ++i)
    workspace.input[i] = {import || i >= 16 ? 1u : 0u, static_cast<std::uint32_t>(i), 0, 90};
  workspace.input_batch_stride = import ? RouteVxmSelectSchedule::workspace_blocks(columns) : blocks;
  workspace.mask_offset = instruction.has_mask ? (import ? workspace.input_batch_stride : 2 * blocks) : 0;
  const auto indices = view(RoutingElement::U32, k, rows, 32, 200, 1);
  const auto values = view(RoutingElement::F32, k, rows, 36, 200, 1);
  workspace.indices = indices.planes;
  workspace.values = values.planes;

  rt::BinaryProgram program;
  const auto choice = view(RoutingElement::F32, rows, columns, 0, 400, 1);
  const auto mask = view(RoutingElement::U32, rows, columns, 4, 400, 1);
  std::optional<rt::RoutingProgramBuilder> routing;
  auto first_bundle = std::uint32_t{64};
  if (import) {
    routing.emplace(program);
    RouteSelectDescriptor scores;
    scores.tokens = rows;
    scores.experts = columns;
    scores.k = k;
    scores.choice_scores = choice;
    scores.eligibility = mask;
    RouteVxmSelectSchedule schedule;
    schedule.score_planes = workspace.input;
    schedule.eligibility_offset = workspace.mask_offset;
    first_bundle = routing->pack_vxm_scores(scores, schedule, 10).first_bundle_cycle;
  }
  const auto outputs_ready = rt::append_vxm_topk(program, instruction, workspace, first_bundle, 2);
  rejects([&] { rt::append_vxm_topk(program, instruction, workspace, first_bundle, 2); });
  const auto raw = view(RoutingElement::F32, rows, columns, 36, 300);
  const auto activations = view(RoutingElement::BF16, rows, 32, 40, 300);
  const auto gathered = view(RoutingElement::BF16, 4, 32, 42, 300);
  if (import) {
    RouteSelectDescriptor input;
    input.tokens = rows;
    input.experts = columns;
    input.k = k;
    input.selected_ids = indices;
    input.selected_scores = values;
    input.raw_scores = raw;
    routing->import_topk(input, outputs_ready, 1);
    routing->plan({}, 2);
    routing->address({1, 0, 0, 4, activations, gathered}, 3);
  }
  auto corrupt = rt::encode_vxm_topk_command(instruction);
  corrupt.words[2] = 1;
  rejects([&] { (void)rt::decode_vxm_topk_command(corrupt); });
  std::stringstream image(std::ios::in | std::ios::out | std::ios::binary);
  rt::write_binary_program(program, image);
  image.seekg(0);
  const auto loaded = rt::read_binary_program(image);
  check(loaded.hardware.routing_isa_version == (import ? 7u : 0u) &&
        loaded.routing_descriptors.size() == (import ? 3u : 0u),
        "VXM TOPK and Routing import capability");
  const auto vxm_queue = std::ranges::find_if(loaded.queues, [](const auto& queue) {
    return queue.kind == rt::QueueKind::Vxm;
  });
  check(vxm_queue != loaded.queues.end() &&
        rt::decode_vxm_topk_command(vxm_queue->commands.back()) == instruction,
        "standalone VXM TOPK codec roundtrip");
  const auto images = rt::prepare_program_images(loaded, 0);
  const auto vxm_image = std::ranges::find_if(images, [](const auto& entry) {
    return entry.descriptor.kind == IcuMacroQueueKind::Vxm;
  });
  check(vxm_image != images.end() && vxm_image->descriptor.word_count() == 2,
        "VXM TOPK must occupy one physical i-MEM word");

  SystemHardwareConfiguration hardware;
  hardware.routing.enabled = import;
  auto system = std::make_unique<C2cDmaSystem>(
      Ddr4Config{}, hw::config::kDmaFifoDepthVectors, hardware);
  rt::CModelRuntime runtime(*system);
  auto& chip = system->chip();
  using Candidate = std::pair<float, std::uint32_t>;
  std::vector<std::vector<Candidate>> expected(rows);
  const auto write = [&](std::size_t row, std::size_t col, std::size_t offset, std::uint32_t bits) {
    for (std::size_t byte = 0; byte < 4; ++byte) {
      const auto& plane = workspace.input[4 * (col % VxmRouteTopK::chains) + byte];
      chip.mem_array(static_cast<Hemisphere>(plane.hemisphere)).set_sram_byte(
          plane.slice, plane.bank, plane.base_row + offset + row / VxmRouteTopK::lanes * blocks
            + col / VxmRouteTopK::chains, row % VxmRouteTopK::lanes,
          static_cast<std::uint8_t>(bits >> (8 * byte)));
    }
  };
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::uint32_t col = 0; col < columns; ++col) {
      const auto eligible = !instruction.has_mask || (row + col) % 4 != 1;
      const auto value = col == 16 ? std::numeric_limits<float>::infinity()
          : float(int((row + 3 * col) % 7) - 3);
      if (import) {
        put(chip, choice, row, col, std::bit_cast<std::uint32_t>(value));
        put(chip, mask, row, col, eligible ? 1 : 0);
        put(chip, raw, row, col, std::bit_cast<std::uint32_t>(float(row * columns + col) / 1024));
      } else {
        write(row, col, 0, std::bit_cast<std::uint32_t>(value));
        if (instruction.has_mask) write(row, col, workspace.mask_offset, eligible ? 1 : 0);
      }
      if (eligible) expected[row].emplace_back(value, col);
    }
    std::ranges::sort(expected[row], [](const Candidate& a, const Candidate& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    if (import)
      for (std::size_t feature = 0; feature < 32; ++feature)
        put(chip, activations, row, feature, static_cast<std::uint32_t>(1 + row * 32 + feature));
  }
  if (timed) runtime.load_timed(loaded);
  else runtime.load(loaded);
  check(runtime.run_until_complete(40000), "standalone VXM TOPK did not drain");
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t rank = 0; rank < k; ++rank) {
      const auto& [value, col] = expected[row][rank];
      check(get(chip, indices, rank, row) == col, "standalone VXM TOPK index output");
      check(get(chip, values, rank, row) == std::bit_cast<std::uint32_t>(value),
            "standalone VXM TOPK value output");
    }
  const auto& stats = chip.vxm_unit().topk().statistics();
  const auto batches = (rows + VxmRouteTopK::lanes - 1) / VxmRouteTopK::lanes;
  check(stats.batches == batches && stats.output_tiles == batches * k * hw::kTileRows &&
        stats.submitted_words == 0, "standalone VXM TOPK output accounting");
  if (import) {
    const auto& rcu = chip.routing();
    const auto first = expected.front().front().second;
    std::vector<std::size_t> expert_rows;
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t rank = 0; rank < k; ++rank) {
        const auto col = expected[row][rank].second;
        check(rcu.selected_experts()[row * k + rank] == col, "RCU imported index");
        check(rcu.selected_score_bits()[row * k + rank] ==
            std::bit_cast<std::uint32_t>(float(row * columns + col) / 1024),
            "RCU imported original score");
        if (col == first) expert_rows.push_back(row);
      }
    const auto gathered_rows = std::min<std::size_t>(4, expert_rows.size());
    check(rcu.block_info().expert_id == first && rcu.block_info().valid_rows == gathered_rows,
          "imported TOPK expert block");
    for (std::size_t row = 0; row < gathered_rows; ++row)
      for (std::size_t feature = 0; feature < 32; ++feature)
        check(get(chip, gathered, row, feature) == 1 + expert_rows[row] * 32 + feature,
              "imported TOPK Gather payload");
  }
  std::cout << "standalone VXM TOPK timed=" << timed << " import=" << import
            << " cycles=" << chip.cycle() << '\n';
}

#include "expert_macro_cases.hpp"
int main() {
  try {
    scenario(false);
    scenario(true);
    vxm_producer();
    vxm_topk_submission(false);
    vxm_topk_submission(true);
    vxm_topk_submission(false, true);
    standalone_vxm_topk(false);
    standalone_vxm_topk(true);
    standalone_vxm_topk(true, true);
    reference_and_invalid();
    native_sync();
    dispatch_group(0); // traversal, stable handshake, delayed consumer
    dispatch_group(1); // codec, real Gather, tail and SRAM commit
    dispatch_group(2); // stale identity, plan lifetime, invalid capability
    weight_group(0);   // expert [3,3,1], single-slot residency
    weight_group(1);   // real two-segment DMA/RX/MEM and physical commit
    weight_group(2);   // stale identity, malformed tables and busy switch
    macro_case(false);
    macro_case(true);
    macro_case(true,true);
    std::cout << "routing runtime passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
