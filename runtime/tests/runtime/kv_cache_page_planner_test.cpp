#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

BinaryBinding binding(std::uint32_t index, BindingLayout layout,
                      std::int64_t base_row, std::int64_t rows,
                      std::vector<std::uint16_t> slices,
                      std::string role, std::string name) {
  BinaryBinding result;
  result.index = index;
  result.access = BindingAccess::Internal;
  result.element_type = BindingElementType::BF16;
  result.layout = layout;
  result.byte_size = 32 * 2 * 128 * 2;
  result.base_row = base_row;
  result.instruction_count = rows;
  result.address_stride = 1;
  result.shape = {32, 2, 128};
  result.slices = std::move(slices);
  result.role = std::move(role);
  result.name = std::move(name);
  result.hemisphere_mask = 3;
  result.bank = 1;
  return result;
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::logic_error(message);
}

const BinaryBinding &find_state_binding(const BinaryProgram &program,
                                        const std::string &role) {
  for (const BinaryBinding &binding : program.bindings)
    if (binding.access == BindingAccess::Internal && binding.role == role)
      return binding;
  throw std::logic_error("executable is missing " + role);
}

void verify_real_executable(const std::filesystem::path &path,
                            std::uint32_t layers) {
  BinaryProgram program = read_binary_program(path);
  const BinaryBinding &key = find_state_binding(program, "state.kv.key");
  const BinaryBinding &value = find_state_binding(program, "state.kv.value");
  require(key.shape == value.shape && key.shape.size() == 3,
          "real executable has incompatible K/V windows");
  const auto resident_tokens = static_cast<std::uint32_t>(key.shape.front());
  constexpr std::uint32_t kCapacity = 256;
  auto logical_shape = key.shape;
  logical_shape.front() = kCapacity;

  ModelPackage package;
  package.model_name = "Qwen2.5-1.5B";
  package.architecture = "Qwen2ForCausalLM";
  package.executables.push_back({"decoder", std::move(program), {}});
  const BinaryBinding &packaged_key = find_state_binding(
      package.executables[0].program, "state.kv.key");
  const BinaryBinding &packaged_value = find_state_binding(
      package.executables[0].program, "state.kv.value");
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    package.states.push_back(
        {prefix + ".key_cache", ModelStateKind::KvKey,
         packaged_key.element_type,
         logical_shape, layer, kCapacity,
         package.executables[0].program.hardware.mxm_rows, resident_tokens});
    package.states.push_back(
        {prefix + ".value_cache", ModelStateKind::KvValue,
         packaged_value.element_type, logical_shape, layer, kCapacity,
         package.executables[0].program.hardware.mxm_rows, resident_tokens});
    package.invocations.push_back(
        {prefix, 0, {}, {},
         {{packaged_key.index, prefix + ".key_cache"},
          {packaged_value.index, prefix + ".value_cache"}}});
  }
  validate_model_package(package);
  const SessionMemoryPlan plan = SessionMemoryPlanner::plan(package);
  require(plan.persistent_states.size() == 2 * layers,
          "real executable cannot place every layer's K/V page");
  std::cout << "real Qwen executable KV plan passed layers=" << layers
            << " resident_tokens=" << resident_tokens
            << " states=" << plan.persistent_states.size() << '\n';
}

} // namespace

int main(int argc, char **argv)
try {
  constexpr std::uint32_t kLayers = 28;
  constexpr std::int64_t kScratchFloor = 3536;
  constexpr std::int64_t kRopeBegin = 7000;
  constexpr std::int64_t kRopeEnd = 7064;

  BinaryProgram program;
  program.hardware.hemispheres = 2;
  program.hardware.slices_per_hemisphere = 52;
  program.hardware.banks_per_slice = 2;
  program.hardware.words_per_bank = 8192;
  program.hardware.sram_depth_rows = 8192;

  BinaryBinding rope = binding(
      100, BindingLayout::Fp16RopeTable, kRopeBegin,
      kRopeEnd - kRopeBegin, {16, 17, 18, 19}, "constant",
      "rope.cos_sin");
  rope.byte_size = 64 * 32 * 2;
  rope.shape = {64, 32, 2};
  rope.initializer = BindingInitializer::RopeTable;
  BinaryBinding key = binding(
      65536, BindingLayout::Fp16HeadPlanar, 8064, 128,
      {16, 17, 18, 19}, "state.kv.key", "attention.key_cache");
  BinaryBinding value = binding(
      65537, BindingLayout::Fp16ValueX16, 8160, 32,
      {0, 1, 2, 3, 4, 5, 6, 7,
       8, 9, 10, 11, 12, 13, 14, 15},
      "state.kv.value", "attention.value_cache");
  program.bindings = {std::move(rope), std::move(key), std::move(value)};
  for (std::uint16_t hemisphere = 0; hemisphere < 2; ++hemisphere)
    for (std::uint16_t slice = 16; slice < 20; ++slice)
      program.memory_floors.push_back(
          {hemisphere, slice, kScratchFloor, 1});
  BinaryProgram mirrored_program = program;
  for (BinaryBinding &entry : mirrored_program.bindings)
    entry.bank = 0;
  for (BinaryMemoryFloor &floor : mirrored_program.memory_floors)
    floor.bank = 0;

  ModelPackage package;
  package.model_name = "Qwen2.5-1.5B";
  package.architecture = "Qwen2ForCausalLM";
  package.executables.push_back({"decoder", program, {}});
  package.executables.push_back(
      {"decoder.mirrored", std::move(mirrored_program), {}});
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer);
    package.states.push_back(
        {prefix + ".key_cache", ModelStateKind::KvKey,
         BindingElementType::BF16, {256, 2, 128}, layer, 256, 32, 32});
    package.states.push_back(
        {prefix + ".value_cache", ModelStateKind::KvValue,
         BindingElementType::BF16, {256, 2, 128}, layer, 256, 32, 32});
    package.invocations.push_back(
        {prefix, layer % 2, {}, {},
         {{65536, prefix + ".key_cache"},
          {65537, prefix + ".value_cache"}}});
  }

  validate_model_package(package);
  const SessionMemoryPlan plan = SessionMemoryPlanner::plan(package);
  require(plan.persistent_states.size() == 2 * kLayers,
          "planner did not allocate every layer's K/V page");

  std::set<std::pair<std::uint16_t, std::int64_t>> key_slots;
  std::set<std::pair<std::uint16_t, std::int64_t>> value_slots;
  for (const auto &state : plan.persistent_states) {
    const bool is_key = state.binding.role == "state.kv.key";
    const std::int64_t begin = state.binding.base_row;
    const std::int64_t end = begin + state.binding.instruction_count;
    if (is_key) {
      require(end <= kRopeBegin || begin >= kRopeEnd,
              "K page overlaps the RoPE constant interval");
      require(begin >= kScratchFloor,
              "K page overlaps anonymous decoder scratch");
      require(end == 8192,
              "K page is not isolated at the high end of SRAM");
      key_slots.insert({state.binding.bank, begin});
    } else {
      require(end == 8192,
              "V page is not isolated at the high end of SRAM");
      value_slots.insert({state.binding.bank, begin});
    }
  }
  require(key_slots.size() == 2 && value_slots.size() == 2,
          "planner did not reuse one K/V staging pair per ping-pong bank");
  for (const SessionInvocationPlan &invocation : plan.invocations) {
    require(invocation.states.size() == 2,
            "decoder invocation did not resolve both K/V states");
    require(key_slots.contains(
                {invocation.states[0].resolved_binding.bank,
                 invocation.states[0].resolved_binding.base_row}) &&
            value_slots.contains(
                {invocation.states[1].resolved_binding.bank,
                 invocation.states[1].resolved_binding.base_row}),
            "decoder invocation does not use a shared K/V staging pair");
  }

  ModelPackage overlapping = package;
  BinaryBinding late_scratch = binding(
      42, BindingLayout::Fp16HeadPlanar, 8064, 128,
      {16, 17, 18, 19}, "workspace", "late.ffn.scratch");
  overlapping.executables[0].program.bindings.push_back(
      std::move(late_scratch));
  bool rejected_overlap = false;
  try {
    (void)SessionMemoryPlanner::plan(overlapping);
  } catch (const std::invalid_argument &error) {
    rejected_overlap =
        std::string(error.what()).find("state SRAM staging overlaps") !=
        std::string::npos;
  }
  require(rejected_overlap,
          "planner accepted an executable that overwrites persistent state");

  ModelPackage floor_overlapping = package;
  floor_overlapping.executables[0].program.memory_floors.push_back(
      {0, 16, 8096, 1});
  bool rejected_floor_overlap = false;
  try {
    (void)SessionMemoryPlanner::plan(floor_overlapping);
  } catch (const std::invalid_argument &error) {
    rejected_floor_overlap =
        std::string(error.what()).find("anonymous command scratch") !=
        std::string::npos;
  }
  require(rejected_floor_overlap,
          "planner accepted persistent state inside an anonymous scratch "
          "reservation");

  std::cout << "kv_cache_page_planner_test passed layers=" << kLayers
            << " states=" << plan.persistent_states.size() << '\n';
  if (argc == 2)
    verify_real_executable(argv[1], kLayers);
  else if (argc > 2)
    throw std::invalid_argument(
        "usage: kv_cache_page_planner_test [decoder.ftlpu]");
  return 0;
} catch (const std::exception &error) {
  std::cerr << "kv_cache_page_planner_test failed: " << error.what() << '\n';
  return 1;
}
