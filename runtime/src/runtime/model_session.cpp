#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/c2c/dma_instruction.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"
#include "ftlpu/system/hardware_configuration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

template <typename System>
void configure_instruction_data_arbiter_if_supported(
    System& system, std::size_t issueWidth)
{
  if constexpr (requires(System& candidate) {
                  candidate.ddr_arbiter().set_issue_width(issueWidth);
                })
    system.ddr_arbiter().set_issue_width(issueWidth);
}

bool is_16bit_float(BindingElementType type) {
  return type == BindingElementType::F16 || type == BindingElementType::BF16;
}

float decode_16bit_float(std::uint16_t bits, BindingElementType type) {
  if (type == BindingElementType::BF16)
    return Bf16::from_bits(bits).to_float();
  if (type == BindingElementType::F16)
    return Fp16::from_bits(bits).to_float();
  throw std::logic_error("model value is not a 16-bit float");
}

std::uint16_t encode_16bit_float(float value, BindingElementType type) {
  if (type == BindingElementType::BF16)
    return Bf16::from_float(value).bits();
  if (type == BindingElementType::F16)
    return Fp16::from_float(value).bits();
  throw std::logic_error("model value is not a 16-bit float");
}

const BinaryBinding &find_binding(const BinaryProgram &program,
                                  BindingAccess access, std::uint32_t index) {
  for (const BinaryBinding &binding : program.bindings)
    if (binding.access == access && binding.index == index)
      return binding;
  throw std::logic_error(
      "session plan references a missing executable binding");
}

const ModelTensor &find_tensor(const ModelPackage &package,
                               const std::string &name) {
  for (const ModelTensor &tensor : package.tensors)
    if (tensor.name == name)
      return tensor;
  throw std::logic_error(
      "executable scale relocation requires a model tensor input");
}

const ModelBindingRef &find_input_ref(const ModelInvocation &invocation,
                                      std::uint32_t binding_index) {
  for (const ModelBindingRef &input : invocation.inputs)
    if (input.binding_index == binding_index)
      return input;
  throw std::logic_error(
      "executable scale relocation references an unbound input");
}

std::size_t element_count(const std::vector<std::uint64_t> &shape) {
  std::size_t result = 1;
  for (const std::uint64_t dimension : shape) {
    if (dimension > std::numeric_limits<std::size_t>::max() / result)
      throw std::overflow_error("model tensor shape is too large");
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

std::size_t element_size(BindingElementType type) {
  switch (type) {
  case BindingElementType::I8:
    return 1;
  case BindingElementType::F16:
  case BindingElementType::BF16:
    return 2;
  case BindingElementType::I32:
  case BindingElementType::F32:
    return 4;
  }
  throw std::invalid_argument("model state has an unknown element type");
}

BinaryProgram parameterize_program(const ModelPackage &package,
                                   const ModelInvocation &invocation,
                                   const SessionInvocationPlan &invocation_plan,
                                   BinaryProgram program) {
  const bool report_progress = std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
  for (const BinaryScaleRelocation &relocation : program.scale_relocations) {
    const ModelBindingRef &input =
        find_input_ref(invocation, relocation.binding_index);
    const ModelTensor &tensor = find_tensor(package, input.value);
    if (tensor.encoding == ModelTensorEncoding::Raw ||
        relocation.scale_index >= tensor.scales.size())
      throw std::logic_error(
          "executable scale relocation requires quantized tensor metadata");
    auto queue =
        std::find_if(program.queues.begin(), program.queues.end(),
                     [&](const QueueProgram &candidate) {
                       return candidate.kind == relocation.queue_kind &&
                              candidate.index == relocation.queue_index;
                     });
    if (queue == program.queues.end() ||
        relocation.command_index >= queue->commands.size())
      throw std::logic_error(
          "executable scale relocation references a missing command");
    QueueCommand &command = queue->commands[relocation.command_index];
    if (command.instruction_kind == InstructionKind::MxmDequant &&
        command.word_count == 1) {
      command.words[0] = static_cast<std::uint32_t>(
          isa::encode_mxm_dequant_instruction(MxmDequantInstruction::Scale(
              tensor.scales[relocation.scale_index])));
      continue;
    }
    if (command.instruction_kind != InstructionKind::Vxm ||
        command.word_count != 3)
      throw std::logic_error(
          "scale relocation target is not a VXM instruction");
    isa::EncodedVxmInstruction encoded{
        static_cast<std::uint64_t>(command.words[0]) |
            (static_cast<std::uint64_t>(command.words[1]) << 32),
        command.words[2]};
    auto decoded = isa::decode_vxm_instruction(queue->index, encoded);
    VxmLaneAluInstruction &instruction = decoded.instruction;
    VxmLaneOperand &operand = relocation.operand == VxmImmediateOperand::Lhs
                                  ? instruction.lhs
                                  : instruction.rhs;
    if (operand.kind != VxmLaneOperandKind::Immediate)
      throw std::logic_error(
          "scale relocation target is not an immediate operand");
    operand = VxmLaneOperand::Imm(tensor.scales[relocation.scale_index]);
    encoded = isa::encode_vxm_instruction(queue->index, decoded.chain_depth,
                                          instruction);
    command.words = {static_cast<std::uint32_t>(encoded.control),
                     static_cast<std::uint32_t>(encoded.control >> 32),
                     encoded.immediate_bits, 0};
  }
  std::unordered_set<std::uint64_t> relocated_bindings;
  std::unordered_map<std::uint64_t, std::size_t> relocation_counts;
  std::unordered_map<std::uint64_t, std::pair<std::int64_t, std::int64_t>>
      relocation_address_ranges;
  const auto relocation_key = [](BindingAccess access, std::uint32_t index) {
    return (static_cast<std::uint64_t>(access) << 32) | index;
  };
  for (const BinaryAddressRelocation &relocation :
       program.address_relocations) {
    const auto input = std::find_if(
        invocation_plan.inputs.begin(), invocation_plan.inputs.end(),
        [&](const SessionInputPlan &candidate) {
          return relocation.binding_access == BindingAccess::Input &&
                 candidate.binding_index == relocation.binding_index;
        });
    const auto state = std::find_if(
        invocation_plan.states.begin(), invocation_plan.states.end(),
        [&](const SessionStatePlan &candidate) {
          return relocation.binding_access == BindingAccess::Internal &&
                 candidate.binding_index == relocation.binding_index;
        });
    const bool resident_input =
        input != invocation_plan.inputs.end() &&
        input->transfer == SessionTransferKind::Resident;
    if (!resident_input && state == invocation_plan.states.end())
      continue;
    const BindingAccess access = relocation.binding_access;
    const std::uint64_t key = relocation_key(access, relocation.binding_index);
    relocated_bindings.insert(key);
    ++relocation_counts[key];
    const BinaryBinding &original_binding =
        find_binding(program, access, relocation.binding_index);
    const BinaryBinding &resolved_binding =
        resident_input ? input->resolved_binding : state->resolved_binding;
    const std::int64_t delta =
        resolved_binding.base_row - original_binding.base_row;
    auto queue =
        std::find_if(program.queues.begin(), program.queues.end(),
                     [&](const QueueProgram &candidate) {
                       return candidate.kind == relocation.queue_kind &&
                              candidate.index == relocation.queue_index;
                     });
    if (queue == program.queues.end() ||
        relocation.command_index >= queue->commands.size())
      throw std::logic_error("address relocation references a missing command");
    QueueCommand &command = queue->commands[relocation.command_index];
    if (relocation.write_port)
      throw std::logic_error(
          "legacy MEM ReadWrite relocation is not supported by "
          "the banked MEM ISA");
    const auto relocate_address = [&](std::size_t sourceAddress) {
      auto [range, inserted] = relocation_address_ranges.try_emplace(
          key, static_cast<std::int64_t>(sourceAddress),
          static_cast<std::int64_t>(sourceAddress));
      if (!inserted) {
        range->second.first = std::min(
            range->second.first, static_cast<std::int64_t>(sourceAddress));
        range->second.second = std::max(
            range->second.second, static_cast<std::int64_t>(sourceAddress));
      }
      const std::int64_t relocated =
          static_cast<std::int64_t>(sourceAddress) + delta;
      if (relocated < 0 || relocated >= program.hardware.sram_depth_rows)
        throw std::logic_error(
            "resident address relocation exceeds physical MEM: binding=" +
            std::to_string(relocation.binding_index) +
            " original_base=" + std::to_string(original_binding.base_row) +
            " resolved_base=" + std::to_string(resolved_binding.base_row) +
            " command_address=" + std::to_string(sourceAddress) +
            " relocated=" + std::to_string(relocated));
      return static_cast<std::size_t>(relocated);
    };
    if (is_mem_slice_program_command(command)) {
      auto sliceProgram = decode_mem_slice_program_command(command);
      for (auto &entry : sliceProgram.body)
        entry.instruction.address =
            relocate_address(entry.instruction.address);
      command = encode_mem_slice_program_command(sliceProgram);
      continue;
    }
    if (command.instruction_kind != InstructionKind::Mem ||
        command.word_count == 0 || command.word_count > 2)
      throw std::logic_error(
          "address relocation target is not a MEM instruction");
    const isa::EncodedMemInstruction encoded =
        static_cast<isa::EncodedMemInstruction>(command.words[0]) |
        (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
    MemInstruction instruction = isa::decode_mem_instruction(encoded);
    const std::size_t sourceAddress = instruction.address;
    instruction.address = relocate_address(sourceAddress);
    const isa::EncodedMemInstruction patched =
        isa::encode_mem_instruction(instruction);
    command.words[0] = static_cast<std::uint32_t>(patched);
    command.words[1] = static_cast<std::uint32_t>(patched >> 32);
    command.word_count =
        static_cast<std::uint16_t>((patched >> 32) == 0 ? 1 : 2);
  }
  for (const SessionInputPlan &input : invocation_plan.inputs) {
    if (input.transfer != SessionTransferKind::Resident)
      continue;
    const BinaryBinding &original_binding =
        find_binding(program, BindingAccess::Input, input.binding_index);
    const std::uint64_t key =
        relocation_key(BindingAccess::Input, input.binding_index);
    if (report_progress) {
      const auto range = relocation_address_ranges.find(key);
      std::clog << "FTLPU resident relocation: binding=" << input.binding_index
                << " name=" << original_binding.name
                << " base=" << original_binding.base_row << "->"
                << input.resolved_binding.base_row
                << " commands=" << relocation_counts[key];
      if (range != relocation_address_ranges.end())
        std::clog << " address_range=[" << range->second.first << ','
                  << range->second.second << ']';
      std::clog << std::endl;
    }
    if (input.resolved_binding.base_row != original_binding.base_row &&
        !relocated_bindings.contains(key))
      throw std::logic_error(
          "resident binding moved without MEM address relocation: "
          "binding=" +
          std::to_string(input.binding_index) +
          " original_base=" + std::to_string(original_binding.base_row) +
          " resolved_base=" + std::to_string(input.resolved_binding.base_row));
    auto binding =
        std::find_if(program.bindings.begin(), program.bindings.end(),
                     [&](const BinaryBinding &candidate) {
                       return candidate.access == BindingAccess::Input &&
                              candidate.index == input.binding_index;
                     });
    if (binding == program.bindings.end())
      throw std::logic_error(
          "resident session input references a missing binding");
    *binding = input.resolved_binding;
  }
  for (const SessionStatePlan &state : invocation_plan.states) {
    const BinaryBinding &original_binding =
        find_binding(program, BindingAccess::Internal, state.binding_index);
    if (state.resolved_binding.base_row != original_binding.base_row &&
        !relocated_bindings.contains(
            relocation_key(BindingAccess::Internal, state.binding_index)))
      throw std::logic_error(
          "persistent state moved without MEM address relocation: "
          "binding=" +
          std::to_string(state.binding_index));
    auto binding =
        std::find_if(program.bindings.begin(), program.bindings.end(),
                     [&](const BinaryBinding &candidate) {
                       return candidate.access == BindingAccess::Internal &&
                              candidate.index == state.binding_index;
                     });
    if (binding == program.bindings.end())
      throw std::logic_error(
          "persistent session state references a missing binding");
    *binding = state.resolved_binding;
  }
  return program;
}

bool weight_page_overlaps_program(const C2cWeightPage &page,
                                  const BinaryProgram &program) {
  for (const C2cWeightSegment &segment : page.segments) {
    const std::uint64_t segmentBegin = segment.base_row;
    const std::uint64_t segmentEnd = segmentBegin + segment.vector_count;
    const std::uint16_t hemisphereBit =
        static_cast<std::uint16_t>(1u << hemisphere_index(segment.hemisphere));
    for (const BinaryBinding &binding : program.bindings) {
      if (binding.bank != segment.bank ||
          (binding.hemisphere_mask & hemisphereBit) == 0 ||
          std::find(binding.slices.begin(), binding.slices.end(),
                    static_cast<std::uint16_t>(segment.slice)) ==
              binding.slices.end())
        continue;
      const std::uint64_t bindingBegin = binding.base_row;
      const std::uint64_t bindingEnd = bindingBegin + binding.instruction_count;
      if (segmentBegin < bindingEnd && bindingBegin < segmentEnd)
        return true;
    }
  }
  return false;
}

bool region_overlaps_binding(const WeightResidencyRegion &region,
                             const BinaryBinding &binding) {
  if (binding.bank != region.bank ||
      (binding.hemisphere_mask & region.hemisphere_mask) == 0 ||
      binding.instruction_count <= 0)
    return false;
  if (!region.wildcard_slice && !binding.slices.empty() &&
      std::find(binding.slices.begin(), binding.slices.end(), region.slice) ==
          binding.slices.end())
    return false;
  const std::uint64_t bindingBegin = static_cast<std::uint64_t>(
      std::max<std::int64_t>(0, binding.base_row));
  const std::uint64_t stride = static_cast<std::uint64_t>(
      std::max<std::int64_t>(1, binding.address_stride));
  const std::uint64_t bindingEnd = bindingBegin +
      static_cast<std::uint64_t>(binding.instruction_count) * stride;
  return region.row_begin < bindingEnd && bindingBegin < region.row_end;
}

std::uint64_t safe_inter_invocation_prefetch_cycle(
    const BinaryProgram &program,
    std::span<const WeightPrefetchPlan> currentPlans,
    const WeightPrefetchPlan &nextPlan) {
  std::uint64_t safeCycle = 0;
  for (const WeightPrefetchPlan &current : currentPlans)
    if (weight_prefetch_plans_overlap(current, nextPlan))
      safeCycle = std::max(safeCycle, current.release_cycle);

  // Paged bindings have exact release intervals above. Other allocations do
  // not yet carry a release cycle in the binary ABI, so an overlapping range
  // remains live through the executable. This is conservative and keeps the
  // same binary correct when DDR latency changes at runtime.
  for (const BinaryBinding &binding : program.bindings) {
    if (binding.paged_weight)
      continue;
    const bool overlaps = std::any_of(
        nextPlan.regions.begin(), nextPlan.regions.end(),
        [&](const WeightResidencyRegion &region) {
          return region_overlaps_binding(region, binding);
        });
    if (overlaps)
      safeCycle = std::max<std::uint64_t>(safeCycle, program.max_cycle);
  }
  return safeCycle;
}

struct C2cFabricLease {
  std::uint16_t stream_base{0};
  std::uint64_t start_cycle{0};
};

C2cFabricLease select_inter_invocation_c2c_fabric(
    const BinaryProgram &program, std::uint64_t earliestCycle) {
  const std::size_t streamCount = program.hardware.streams_per_direction;
  const std::size_t laneCount =
      program.hardware.c2c_streams_per_direction;
  if (laneCount == 0 || laneCount > streamCount ||
      streamCount > hw::kWestStreams)
    throw std::logic_error(
        "invalid target stream geometry for inter-invocation C2C");

  const std::size_t defaultBase = streamCount - laneCount;
  std::size_t releaseOffset = 0;
  if (program.stream_release_cycles.size() ==
      program.hardware.encoded_streams) {
    if (program.hardware.encoded_streams != 2 * streamCount)
      throw std::logic_error(
          "directional stream-release metadata has invalid geometry");
    // C2C RX enters the ordinary fabric through the West stream file.
    releaseOffset = streamCount;
  } else if (program.stream_release_cycles.size() != streamCount) {
    return {static_cast<std::uint16_t>(defaultBase),
            std::max<std::uint64_t>(
                earliestCycle,
                static_cast<std::uint64_t>(program.max_cycle) + 1)};
  }

  const auto availableAt = [&](std::size_t base) {
    std::uint64_t cycle = earliestCycle;
    for (std::size_t lane = 0; lane < laneCount; ++lane)
      cycle = std::max(
          cycle,
          program.stream_release_cycles[releaseOffset + base + lane]);
    return cycle;
  };
  std::size_t bestBase = defaultBase;
  std::uint64_t bestCycle = availableAt(bestBase);
  for (std::size_t base = 0; base + laneCount <= streamCount; ++base) {
    const std::uint64_t cycle = availableAt(base);
    if (cycle < bestCycle) {
      bestBase = base;
      bestCycle = cycle;
    }
  }
  return {static_cast<std::uint16_t>(bestBase), bestCycle};
}

WeightPrefetchPlan model_page_plan(const C2cWeightPage &page,
                                   std::uint32_t pageIndex) {
  WeightPrefetchPlan plan;
  plan.page_index = pageIndex;
  plan.bank = page.bank;
  plan.pre_execution = true;
  for (const C2cWeightSegment &segment : page.segments) {
    const auto side = hemisphere_index(segment.hemisphere);
    plan.bytes[side] += static_cast<std::uint64_t>(segment.vector_count) *
                        hw::kPhysicalVectorBytes;
    plan.regions.push_back(WeightResidencyRegion{
        segment.bank,
        static_cast<std::uint16_t>(1u << side),
        segment.slice,
        segment.base_row,
        static_cast<std::uint32_t>(segment.base_row + segment.vector_count),
        false});
  }
  return plan;
}

void verify_c2c_weight_page_residency(C2cDmaSystem &system,
                                      const C2cWeightPage &page) {
  for (std::size_t segmentIndex = 0; segmentIndex < page.segments.size();
       ++segmentIndex) {
    const C2cWeightSegment &segment = page.segments[segmentIndex];
    for (std::uint32_t row = 0; row < segment.vector_count; ++row) {
      const C2cVector expected = system.ddr4().read_vector(
          segment.ddr4_address +
          static_cast<std::uint64_t>(row) * hw::kPhysicalVectorBytes);
      for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
          const std::uint8_t actual = system.chip().read_mem_sram_lane_byte(
              segment.hemisphere, segment.slice, segment.bank, tile,
              segment.base_row + row, lane);
          if (actual != expected.payload[tile][lane])
            throw std::runtime_error(
                "inter-invocation C2C residency verification failed: layer=" +
                std::to_string(page.layer) + " segment=" +
                std::to_string(segmentIndex) + " hemisphere=" +
                std::to_string(hemisphere_index(segment.hemisphere)) +
                " slice=" + std::to_string(segment.slice) + " bank=" +
                std::to_string(segment.bank) + " row=" +
                std::to_string(segment.base_row + row) + " tile=" +
                std::to_string(tile) + " lane=" + std::to_string(lane));
        }
    }
  }
}

} // namespace

ModelSession::ModelSession(TspSliceSystem &system) : runtime_(system) {}

ModelSession::ModelSession(C2cDmaSystem &system)
    : runtime_(system,
               [this, &system](TspSliceSystem::LogSinks sinks) {
                 system.tick(sinks);
                 observe_weight_page_tick();
                 observe_executable_weight_page_tick();
                 release_due_executable_weight_pages();
                 if (executable_clock_active_) {
                   if (system.chip().icu().program_issue_enabled())
                     ++executable_cycle_;
                   else
                     ++stats_.weight_page_runtime_wait_cycles;
                 }
               }),
      c2c_system_(&system),
      weight_pager_(std::make_unique<C2cWeightPager>(system)) {
  runtime_.set_weight_page_residency_checker(
      [this](const BinaryWeightPageUse &use) {
        return executable_weight_page_ready(use);
      });
}

void ModelSession::set_ddr_peak_bandwidth_mbytes_per_second(
    std::uint32_t bandwidth) {
  if (loaded_)
    throw std::logic_error(
        "DDR bandwidth must be configured before loading a model package");
  if (bandwidth == 0)
    throw std::invalid_argument("DDR bandwidth must be non-zero");
  ddr_peak_bandwidth_mbytes_per_second_override_ = bandwidth;
}

ExecutableHardwareConfig ModelSession::effective_external_transport(
    const ExecutableHardwareConfig &hardware) const {
  ExecutableHardwareConfig effective = hardware;
  if (ddr_peak_bandwidth_mbytes_per_second_override_)
    effective.ddr_peak_bandwidth_mbytes_per_second =
        *ddr_peak_bandwidth_mbytes_per_second_override_;
  return effective;
}

void ModelSession::enable_execution_trace(bool enabled) noexcept {
  execution_trace_enabled_ = enabled;
  execution_trace_has_segment_ = false;
  execution_trace_cycle_cursor_ = 0;
  runtime_.enable_execution_trace(enabled);
}

void ModelSession::write_execution_trace_csv(
    const std::filesystem::path &path) const {
  runtime_.write_execution_trace_csv(path);
}

void ModelSession::configure_external_transport(
    const ExecutableHardwareConfig &hardware) {
  if (c2c_system_ == nullptr)
    throw std::logic_error(
        "ModelSession external transfers require C2cDmaSystem");

  c2c_bytes_per_stream_per_cycle_ =
      hardware.c2c_bytes_per_stream_per_cycle;

  SystemHardwareConfiguration cmodelHardware;
  cmodelHardware.sram_depth_rows = hardware.sram_depth_rows;
  cmodelHardware.mxms_per_hemisphere = hardware.mxms_per_hemisphere;
  cmodelHardware.mxm_weight_buffers = hardware.mxm_weight_buffers;
  cmodelHardware.vxm_alus = hardware.vxm_alus;
  cmodelHardware.c2c_streams_per_direction =
      hardware.c2c_streams_per_direction;
  cmodelHardware.mxm_local_dequant_enabled =
      hardware.mxm_local_dequant_enabled != 0;
  cmodelHardware.mxm_weight_activation_overlap_enabled =
      hardware.mxm_weight_activation_overlap_enabled != 0;
  c2c_system_->chip().configure_hardware(cmodelHardware);

  Ddr4Config ddr;
  ddr.beat_bytes = hardware.c2c_bytes_per_stream_per_cycle;
  ddr.read_latency_cycles = hardware.ddr_read_latency_cycles;
  ddr.write_latency_cycles = hardware.ddr_write_latency_cycles;
  ddr.read_latency_jitter_cycles =
      hardware.ddr_read_latency_jitter_cycles;
  ddr.write_latency_jitter_cycles =
      hardware.ddr_write_latency_jitter_cycles;
  ddr.request_queue_depth = hardware.ddr_request_queue_depth;
  ddr.transfer_channels = static_cast<std::size_t>(hardware.hemispheres) *
                          hardware.c2c_streams_per_direction;
  ddr.lpu_clock_hz =
      static_cast<std::uint64_t>(hardware.lpu_clock_mhz) * 1'000'000;
  ddr.peak_bandwidth_bytes_per_second =
      static_cast<std::uint64_t>(
          hardware.ddr_peak_bandwidth_mbytes_per_second) *
      1'000'000;
  ddr.latency_random_seed = hardware.ddr_latency_random_seed;
  c2c_system_->ddr4().configure(ddr);
  configure_instruction_data_arbiter_if_supported(
      *c2c_system_, ddr.transfer_channels);
}

void ModelSession::upload_binding_through_c2c(
    const BinaryBinding &binding, std::span<const std::uint8_t> data,
    const ExecutableHardwareConfig &hardware) {
  if (c2c_system_ == nullptr || !weight_pager_)
    throw std::logic_error(
        "LPU input cannot bypass C2C in this ModelSession");
  PackedWeightImage image;
  try {
    image = pack_binding_image(binding, data, hardware);
  } catch (const std::exception &error) {
    throw std::runtime_error(
        "failed to pack C2C input binding=" +
        std::to_string(binding.index) + " role=" + binding.role +
        " base_row=" + std::to_string(binding.base_row) + " rows=" +
        std::to_string(binding.instruction_count) + ": " + error.what());
  }
  if (image.segments.empty())
    throw std::logic_error("packed C2C input has no physical SRAM segments");

  C2cWeightPage page;
  page.bank = binding.bank;
  std::array<std::array<std::uint64_t, hw::kC2cStreamsPerDirection>,
             hw::kHemispheres>
      laneLoads{};
  const std::size_t laneCount = hardware.c2c_streams_per_direction;
  for (const PackedWeightSegment &source : image.segments) {
    const auto side = static_cast<std::size_t>(source.hemisphere);
    if (side >= hardware.hemispheres || laneCount == 0)
      throw std::logic_error("packed C2C input has an invalid hemisphere");
    const auto lane = static_cast<std::size_t>(std::distance(
        laneLoads[side].begin(),
        std::min_element(laneLoads[side].begin(),
                         laneLoads[side].begin() + laneCount)));
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(source.vector_count) *
        hw::kPhysicalVectorBytes;
    const std::uint64_t ddrAddress = executable_ddr4_address_;
    for (std::uint32_t vector = 0; vector < source.vector_count; ++vector) {
      C2cVector payload;
      const std::size_t offset =
          static_cast<std::size_t>(source.byte_offset) +
          static_cast<std::size_t>(vector) * hw::kPhysicalVectorBytes;
      for (std::size_t byte = 0; byte < hw::kPhysicalVectorBytes; ++byte)
        payload.payload[byte / hw::kLanesPerTile]
                       [byte % hw::kLanesPerTile] = image.data[offset + byte];
      c2c_system_->ddr4().initialize_vector(
          ddrAddress + vector * hw::kPhysicalVectorBytes, payload);
    }
    page.segments.push_back(C2cWeightSegment{
        static_cast<Hemisphere>(source.hemisphere), source.slice,
        binding.bank, source.base_row, static_cast<std::uint16_t>(lane),
        ddrAddress, source.vector_count});
    laneLoads[side][lane] += source.vector_count;
    executable_ddr4_address_ += bytes;
  }

  c2c_system_->reset_execution_state();
  weight_pager_->enqueue(page);
  const std::size_t beginCycle = c2c_system_->cycle();
  std::size_t vectors = 0;
  for (const auto &segment : page.segments)
    vectors += segment.vector_count;
  weight_pager_->wait(std::max<std::size_t>(4096, vectors * 64));
  stats_.c2c_ingress_cycles += c2c_system_->cycle() - beginCycle;
  stats_.c2c_ingress_bytes += vectors * hw::kPhysicalVectorBytes;
  weight_pager_->retire();
}

std::vector<std::uint8_t> ModelSession::download_binding_through_c2c(
    const BinaryBinding &binding,
    const ExecutableHardwareConfig &hardware) {
  if (c2c_system_ == nullptr)
    throw std::logic_error(
        "LPU output cannot bypass C2C in this ModelSession");
  const std::vector<std::uint8_t> zero(
      static_cast<std::size_t>(binding.byte_size), 0);
  PackedWeightImage image = pack_binding_image(binding, zero, hardware);
  if (image.segments.empty())
    throw std::logic_error("packed C2C output has no physical SRAM segments");

  const ExecutableHardwareConfig runtimeHardware =
      effective_external_transport(hardware);

  const std::size_t laneCount = hardware.c2c_streams_per_direction;
  std::array<std::vector<std::size_t>, hw::kHemispheres> byHemisphere;
  for (std::size_t index = 0; index < image.segments.size(); ++index) {
    const auto side = static_cast<std::size_t>(image.segments[index].hemisphere);
    if (side >= hardware.hemispheres)
      throw std::logic_error("packed C2C output has an invalid hemisphere");
    byHemisphere[side].push_back(index);
  }
  std::array<std::size_t, hw::kHemispheres> cursor{};
  std::size_t transferredVectors = 0;
  while (cursor[0] < byHemisphere[0].size() ||
         cursor[1] < byHemisphere[1].size()) {
    c2c_system_->reset_execution_state();
    struct BatchSegment {
      std::size_t image_index{0};
      std::uint64_t ddr_address{0};
      std::size_t lane{0};
    };
    std::vector<BatchSegment> batch;
    std::vector<std::size_t> memQueues;
    std::size_t batchVectors = 0;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
      const auto hemisphere = static_cast<Hemisphere>(side);
      for (std::size_t lane = 0;
           lane < laneCount && cursor[side] < byHemisphere[side].size();
           ++lane, ++cursor[side]) {
        const std::size_t imageIndex = byHemisphere[side][cursor[side]];
        const PackedWeightSegment &segment = image.segments[imageIndex];
        const std::uint64_t ddrAddress = executable_ddr4_address_;
        executable_ddr4_address_ +=
            static_cast<std::uint64_t>(segment.vector_count) *
            hw::kPhysicalVectorBytes;
        batch.push_back(BatchSegment{imageIndex, ddrAddress, lane});
        transferredVectors += segment.vector_count;
        batchVectors += segment.vector_count;
      }
    }

    const std::uint64_t effectiveBandwidth =
        static_cast<std::uint64_t>(
            runtimeHardware.ddr_peak_bandwidth_mbytes_per_second) *
        runtimeHardware.ddr_scheduling_efficiency_percent;
    if (effectiveBandwidth == 0)
      throw std::logic_error("C2C output requires non-zero DDR bandwidth");
    const std::uint64_t producerDemand =
        static_cast<std::uint64_t>(batch.size()) *
        hw::kPhysicalVectorBytes * runtimeHardware.lpu_clock_mhz * 100;
    const std::size_t memReadInterval = static_cast<std::size_t>(
        std::max<std::uint64_t>(
            1, (producerDemand + effectiveBandwidth - 1) /
                   effectiveBandwidth));
    std::array<std::size_t, hw::kHemispheres> dmaTransfers{};
    for (const BatchSegment &copy : batch) {
      const PackedWeightSegment &segment = image.segments[copy.image_index];
      const auto hemisphere =
          static_cast<Hemisphere>(segment.hemisphere);
      const std::size_t fabricStream = copy.lane;
      const auto queue = InstructionControlUnit::mem_queue(
          hemisphere, segment.slice, binding.bank);
      c2c_system_->chip().icu().enqueue_mem_nop(queue, laneCount);
      c2c_system_->chip().icu().enqueue_mem(
          queue, MemInstruction::Read(segment.base_row,
                     StreamId::East(fabricStream)));
      if (segment.vector_count > 1)
        c2c_system_->chip().icu().enqueue_mem_repeat(
            queue, segment.vector_count - 1, memReadInterval, 1);
      c2c_system_->chip().icu().enqueue_c2c_send(
          hemisphere, copy.lane, segment.vector_count, fabricStream);
      c2c_system_->chip().icu().enqueue_c2c_dma(
          hemisphere,
          C2cDmaInstruction::Store(copy.ddr_address, segment.vector_count,
                                   hw::kPhysicalVectorBytes, copy.lane));
      ++dmaTransfers[static_cast<std::size_t>(segment.hemisphere)];
      memQueues.push_back(queue);
    }
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
      const auto hemisphere = static_cast<Hemisphere>(side);
      for (std::size_t transfer = 0; transfer < dmaTransfers[side]; ++transfer)
        c2c_system_->chip().icu().enqueue_control(
            IcuLocation::C2cDma(hemisphere),
            IcuControlInstruction::Sync());
    }

    bool ready = false;
    const std::size_t maxCycles = std::max<std::size_t>(
        4096, batchVectors * 64 + runtimeHardware.ddr_write_latency_cycles +
                  runtimeHardware.ddr_write_latency_jitter_cycles);
    for (std::size_t cycle = 0; cycle < maxCycles; ++cycle) {
      c2c_system_->tick();
      ready = c2c_system_->ddr4().idle();
      for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        ready = ready && c2c_system_->chip().icu().c2c_tx_iq(hemisphere).done()
                && c2c_system_->chip().icu().c2c_dma_iq(hemisphere).done()
                && c2c_system_->chip().c2c_endpoint(hemisphere).tx().idle()
                && c2c_system_->dma(hemisphere).idle();
      }
      for (const auto queue : memQueues)
        ready = ready && c2c_system_->chip().icu().mem_iq(queue).done();
      if (ready) break;
    }
    if (!ready) {
      std::string detail =
          "C2C output transfer timed out: vectors=" +
          std::to_string(batchVectors) + " segments=" +
          std::to_string(batch.size()) + " max_cycles=" +
          std::to_string(maxCycles) + " ddr_write_bytes=" +
          std::to_string(c2c_system_->ddr4().write_bytes_transferred()) +
          " ddr_idle=" + std::to_string(c2c_system_->ddr4().idle());
      for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        detail += " h" + std::to_string(side) + "{tx_iq=" +
                  std::to_string(
                      c2c_system_->chip().icu().c2c_tx_iq(hemisphere).done()) +
                  ",dma_iq=" +
                  std::to_string(
                      c2c_system_->chip().icu().c2c_dma_iq(hemisphere).done()) +
                  ",tx_idle=" +
                  std::to_string(
                      c2c_system_->chip().c2c_endpoint(hemisphere).tx().idle()) +
                  ",dma_idle=" +
                  std::to_string(c2c_system_->dma(hemisphere).idle());
        for (std::size_t lane = 0; lane < laneCount; ++lane)
          detail += ",lane" + std::to_string(lane) + "=" +
                    std::to_string(c2c_system_->dma(hemisphere)
                                       .outbound_queue_size(lane));
        detail += "}";
      }
      detail += " pending_mem=";
      for (const auto queue : memQueues)
        if (!c2c_system_->chip().icu().mem_iq(queue).done())
          detail += std::to_string(queue) + ",";
      throw std::runtime_error(detail);
    }
    stats_.c2c_egress_cycles += c2c_system_->cycle();

    for (const BatchSegment &copy : batch) {
      const PackedWeightSegment &segment = image.segments[copy.image_index];
      for (std::uint32_t vector = 0; vector < segment.vector_count; ++vector) {
        const C2cVector payload = c2c_system_->ddr4().read_vector(
            copy.ddr_address + vector * hw::kPhysicalVectorBytes);
        const std::size_t offset =
            static_cast<std::size_t>(segment.byte_offset) +
            static_cast<std::size_t>(vector) * hw::kPhysicalVectorBytes;
        for (std::size_t byte = 0; byte < hw::kPhysicalVectorBytes; ++byte)
          image.data[offset + byte] =
              payload.payload[byte / hw::kLanesPerTile]
                             [byte % hw::kLanesPerTile];
      }
    }
  }
  stats_.c2c_egress_bytes +=
      transferredVectors * hw::kPhysicalVectorBytes;
  return unpack_binding_image(binding, image, hardware);
}

void ModelSession::prepare_weight_pages() {
  c2c_pages_.clear();
  ready_weight_page_.reset();
  inflight_weight_page_.reset();
  executable_ddr4_address_ = 0;
  if (package_.weight_pages.empty())
    return;
  if (c2c_system_ == nullptr)
    throw std::logic_error(
        "paged model weights require ModelSession(C2cDmaSystem&)");

  std::uint64_t nextDdrAddress = 0;
  for (const ModelWeightPage &source : package_.weight_pages) {
    C2cWeightPage page;
    page.layer = source.layer;
    page.bank = source.bank;
    for (const ModelWeightPage::Segment &segment : source.segments) {
      const ModelTensor &tensor = find_tensor(package_, segment.tensor);
      const std::uint64_t bytes =
          static_cast<std::uint64_t>(segment.vector_count) *
          hw::kPhysicalVectorBytes;
      const std::uint64_t ddrAddress = nextDdrAddress;
      for (std::uint32_t vector = 0; vector < segment.vector_count; ++vector) {
        C2cVector payload;
        const std::size_t sourceOffset = static_cast<std::size_t>(
            segment.byte_offset +
            static_cast<std::uint64_t>(vector) * hw::kPhysicalVectorBytes);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
          for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            payload.payload[tile][lane] =
                tensor.data[sourceOffset + tile * hw::kLanesPerTile + lane];
        c2c_system_->ddr4().initialize_vector(
            ddrAddress +
                static_cast<std::uint64_t>(vector) * hw::kPhysicalVectorBytes,
            payload);
      }
      page.segments.push_back(
          C2cWeightSegment{static_cast<Hemisphere>(segment.hemisphere),
                           segment.slice, source.bank, segment.base_row,
                           segment.stream, ddrAddress, segment.vector_count});
      nextDdrAddress += bytes;
    }
    c2c_pages_.push_back(std::move(page));
  }
  executable_ddr4_address_ = nextDdrAddress;
}

void ModelSession::start_weight_page(std::uint32_t page_index) {
  if (!weight_pager_ || page_index >= c2c_pages_.size())
    throw std::logic_error("model weight page is unavailable");
  if (ready_weight_page_ == page_index || inflight_weight_page_ == page_index)
    return;
  if (inflight_weight_page_)
    throw std::logic_error(
        "cannot start a weight page while another page is in flight");
  weight_pager_->enqueue(c2c_pages_[page_index]);
  ++stats_.weight_page_prefetches;
  stats_.weight_page_prefetch_bytes += weight_pager_->stats().bytes;
  inflight_weight_page_ = page_index;
  ready_weight_page_.reset();
}

void ModelSession::observe_weight_page_tick() {
  if (!weight_pager_)
    return;
  weight_pager_->observe_tick();
  if (!weight_pager_->ready())
    return;
  if (inflight_weight_page_) {
    ready_weight_page_ = *inflight_weight_page_;
    inflight_weight_page_.reset();
  }
  weight_pager_->retire();
}

std::vector<ModelSession::ExecutableWeightTransfer>
ModelSession::build_executable_weight_pages(
    const BinaryProgram &program, const ModelInvocation &invocation) {
  std::vector<ExecutableWeightTransfer> transfers;
  if (program.weight_page_uses.empty())
    return transfers;
  if (c2c_system_ == nullptr || !weight_pager_)
    throw std::logic_error(
        "executable-local paged weights require ModelSession(C2cDmaSystem&)");

  std::uint64_t nextDdrAddress = executable_ddr4_address_;
  const ExecutableHardwareConfig runtimeHardware =
      effective_external_transport(program.hardware);
  auto plans = plan_weight_prefetches(program, runtimeHardware);
  transfers.reserve(plans.size());
  for (const WeightPrefetchPlan &plan : plans) {
    ExecutableWeightTransfer transfer;
    transfer.plan = plan;
    transfer.page.layer = invocation.executable_index;
    transfer.page.bank = plan.bank;
    std::array<std::array<std::uint64_t, hw::kC2cStreamsPerDirection>,
               hw::kHemispheres>
        streamLoads{};
    std::array<std::array<std::array<std::optional<std::size_t>,
        hw::kMemBanksPerSlice>, hw::kMemSliceColumns>,
        hw::kHemispheres> targetLanes{};
    for (const std::size_t useIndex : plan.use_indices) {
      const BinaryWeightPageUse &use = program.weight_page_uses.at(useIndex);
      const BinaryBinding &binding =
          find_binding(program, BindingAccess::Input, use.binding_index);
      const ModelBindingRef &input =
          find_input_ref(invocation, use.binding_index);
      PackedWeightImage image;
      try {
        image = pack_weight_binding_page(
            binding, use.page_index, resolve_value(input.value),
            program.hardware);
      } catch (const std::exception &error) {
        throw std::runtime_error(
            "failed to pack executable C2C weight binding=" +
            std::to_string(binding.index) + " page=" +
            std::to_string(use.page_index) + " base_row=" +
            std::to_string(binding.base_row) + " page_rows=" +
            std::to_string(binding.page_rows) + ": " + error.what());
      }
      transfer.uses.push_back(use);
      for (const PackedWeightSegment &segment : image.segments) {
        const auto side = static_cast<std::size_t>(segment.hemisphere);
        if (side >= hw::kHemispheres)
          throw std::logic_error(
              "packed executable page has an invalid hemisphere");
        const auto laneCount = static_cast<std::size_t>(
            program.hardware.c2c_streams_per_direction);
        auto &targetLane =
            targetLanes[side][segment.slice][use.bank];
        if (!targetLane.has_value()) {
          const auto begin = streamLoads[side].begin();
          targetLane = static_cast<std::size_t>(std::distance(
              begin, std::min_element(begin, begin + laneCount)));
        }
        const auto lane = *targetLane;
        const std::uint64_t ddrAddress = nextDdrAddress;
        for (std::uint32_t vector = 0; vector < segment.vector_count;
             ++vector) {
          C2cVector payload;
          const std::size_t sourceOffset =
              static_cast<std::size_t>(segment.byte_offset) +
              static_cast<std::size_t>(vector) * hw::kPhysicalVectorBytes;
          for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t localLane = 0;
                 localLane < hw::kLanesPerTile; ++localLane)
              payload.payload[tile][localLane] =
                  image.data[sourceOffset +
                             tile * hw::kLanesPerTile + localLane];
          c2c_system_->ddr4().initialize_vector(
              ddrAddress + static_cast<std::uint64_t>(vector) *
                               hw::kPhysicalVectorBytes,
              payload);
        }
        transfer.page.segments.push_back(C2cWeightSegment{
            static_cast<Hemisphere>(segment.hemisphere), segment.slice,
            use.bank, segment.base_row, static_cast<std::uint16_t>(lane),
            ddrAddress, segment.vector_count});
        streamLoads[side][lane] += segment.vector_count;
        nextDdrAddress += static_cast<std::uint64_t>(segment.vector_count) *
                          hw::kPhysicalVectorBytes;
      }
    }
    transfer.plan.bytes = {};
    for (const C2cWeightSegment &segment : transfer.page.segments)
      transfer.plan.bytes[hemisphere_index(segment.hemisphere)] +=
          static_cast<std::uint64_t>(segment.vector_count) *
          hw::kPhysicalVectorBytes;
    transfers.push_back(std::move(transfer));
  }
  for (std::size_t index = 0; index < plans.size(); ++index)
    plans[index] = transfers[index].plan;
  schedule_weight_prefetches(program, plans, runtimeHardware);
  for (std::size_t index = 0; index < plans.size(); ++index) {
    transfers[index].plan = plans[index];
    if (std::getenv("FTLPU_SESSION_PROGRESS") == nullptr) continue;
    const auto &transfer = transfers[index];
    std::size_t vectors = 0;
    for (const C2cWeightSegment &segment : transfer.page.segments)
      vectors += segment.vector_count;
    std::clog << "FTLPU executable page plan: binding="
              << program.weight_page_uses[
                     transfer.plan.use_indices.front()].binding_index
              << " page=" << transfer.plan.page_index
              << " bank=" << transfer.plan.bank
              << " pre_execution=" << transfer.plan.pre_execution
              << " segments=" << transfer.page.segments.size()
              << " vectors=" << vectors
              << " physical_bytes="
              << vectors * hw::kPhysicalVectorBytes
              << " start=" << transfer.plan.start_cycle
              << " end=" << transfer.plan.transfer_end_cycle
              << " ready=" << transfer.plan.ready_cycle << std::endl;
  }
  executable_ddr4_address_ = nextDdrAddress;
  return transfers;
}

void ModelSession::prepare_executable_weight_pages(
    const BinaryProgram &program, const ModelInvocation &invocation,
    std::size_t invocationIndex) {
  executable_weight_transfers_.clear();
  executable_cycle_ = 0;
  executable_clock_active_ = false;
  if (lookahead_invocation_index_ == invocationIndex) {
    executable_weight_transfers_ =
        std::move(lookahead_executable_weight_transfers_);
    lookahead_invocation_index_.reset();
    lookahead_model_weight_transfer_.reset();
    for (ExecutableWeightTransfer &transfer :
         executable_weight_transfers_) {
      transfer.inter_invocation_lookahead = false;
      if (transfer.plan.pre_execution && transfer.ready_before_execution)
        ++stats_.weight_page_hidden_prefetches;
    }
  } else {
    lookahead_executable_weight_transfers_.clear();
    lookahead_model_weight_transfer_.reset();
    lookahead_invocation_index_.reset();
    executable_weight_transfers_ =
        build_executable_weight_pages(program, invocation);
  }

  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_) {
    if (!transfer.plan.pre_execution || transfer.ready_before_execution)
      continue;
    c2c_system_->reset_execution_state();
    weight_pager_->enqueue(transfer.page);
    ++stats_.weight_page_prefetches;
    std::size_t vectors = 0;
    for (const C2cWeightSegment &segment : transfer.page.segments)
      vectors += segment.vector_count;
    stats_.weight_page_prefetch_bytes +=
        vectors * hw::kPhysicalVectorBytes;
    const std::size_t beginCycle = c2c_system_->cycle();
    weight_pager_->wait(std::max<std::size_t>(4096, vectors * 64));
    const std::size_t waitCycles = c2c_system_->cycle() - beginCycle;
    transfer.pre_execution_cycles = waitCycles;
    stats_.weight_page_wait_cycles += waitCycles;
    if (completed_invocation_)
      stats_.weight_page_boundary_wait_cycles += waitCycles;
    else
      stats_.weight_page_initial_wait_cycles += waitCycles;
    weight_pager_->retire();
    transfer.ready_before_execution = true;
  }
}

void ModelSession::prepare_executable_weight_lookahead(
    std::size_t invocationIndex, const BinaryProgram &program) {
  lookahead_executable_weight_transfers_.clear();
  lookahead_model_weight_transfer_.reset();
  lookahead_invocation_index_.reset();
  if (std::getenv("FTLPU_SESSION_STOP_CYCLE") != nullptr ||
      invocationIndex + 1 >= package_.invocations.size())
    return;

  const std::size_t nextIndex = invocationIndex + 1;
  const ModelInvocation &nextInvocation = package_.invocations[nextIndex];
  const ModelExecutable &nextExecutable =
      package_.executables.at(nextInvocation.executable_index);
  const BinaryProgram nextProgram = parameterize_program(
      package_, nextInvocation, memory_plan_.invocations.at(nextIndex),
      materialize_model_executable(nextExecutable));
  lookahead_executable_weight_transfers_ =
      build_executable_weight_pages(nextProgram, nextInvocation);

  std::vector<WeightPrefetchPlan> currentPlans;
  currentPlans.reserve(executable_weight_transfers_.size());
  for (const ExecutableWeightTransfer &transfer :
       executable_weight_transfers_)
    currentPlans.push_back(transfer.plan);
  for (ExecutableWeightTransfer &transfer :
       lookahead_executable_weight_transfers_) {
    if (!transfer.plan.pre_execution)
      continue;
    transfer.inter_invocation_lookahead = true;
    const auto fabric = select_inter_invocation_c2c_fabric(
        program, safe_inter_invocation_prefetch_cycle(
                     program, currentPlans, transfer.plan));
    transfer.page.fabric_stream_base = fabric.stream_base;
    transfer.plan.start_cycle = fabric.start_cycle;
  }

  const std::uint32_t nextPage = nextInvocation.weight_page;
  const std::uint32_t currentPage =
      package_.invocations[invocationIndex].weight_page;
  if (nextPage != 0xffffffffu && nextPage != currentPage) {
    if (nextPage >= c2c_pages_.size())
      throw std::logic_error("next model weight page is unavailable");
    ExecutableWeightTransfer transfer;
    transfer.page = c2c_pages_[nextPage];
    transfer.plan = model_page_plan(transfer.page, nextPage);
    const auto fabric = select_inter_invocation_c2c_fabric(
        program, safe_inter_invocation_prefetch_cycle(
                     program, currentPlans, transfer.plan));
    transfer.page.fabric_stream_base = fabric.stream_base;
    transfer.plan.start_cycle = fabric.start_cycle;
    transfer.plan.ready_cycle = program.max_cycle;
    transfer.inter_invocation_lookahead = true;
    transfer.model_page_index = nextPage;
    if (transfer.plan.start_cycle >= program.max_cycle)
      ++stats_.weight_page_deferred_prefetches;
    lookahead_model_weight_transfer_ = std::move(transfer);
  }
  lookahead_invocation_index_ = nextIndex;
}

void ModelSession::schedule_executable_weight_pages() {
  if (executable_weight_transfers_.empty() &&
      lookahead_executable_weight_transfers_.empty() &&
      !lookahead_model_weight_transfer_)
    return;
  std::int64_t preExecutionCursor = 0;
  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_)
    if (transfer.plan.pre_execution && !transfer.trace_recorded)
      preExecutionCursor -= static_cast<std::int64_t>(
          transfer.pre_execution_cycles);
  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_) {
    if (!transfer.plan.pre_execution || transfer.trace_recorded)
      continue;
    transfer.actual_start_cycle = preExecutionCursor;
    transfer.actual_ready_cycle = preExecutionCursor
        + static_cast<std::int64_t>(transfer.pre_execution_cycles);
    record_weight_page_trace(transfer);
    preExecutionCursor += static_cast<std::int64_t>(
        transfer.pre_execution_cycles);
  }
  std::optional<std::size_t> debugStopCycle;
  if (const char *stop = std::getenv("FTLPU_SESSION_STOP_CYCLE"))
    debugStopCycle = static_cast<std::size_t>(std::stoull(stop));
  weight_pager_->begin_schedule();
  for (std::size_t transferIndex = 0;
       transferIndex < executable_weight_transfers_.size(); ++transferIndex) {
    ExecutableWeightTransfer &transfer =
        executable_weight_transfers_[transferIndex];
    if (transfer.plan.pre_execution)
      continue;
    // A bounded diagnostic run must not inject traffic for a page whose first
    // consumer is beyond the stop point. Besides saving time, this keeps
    // stage captures isolated from future C2C/MEM traffic.
    if (debugStopCycle && transfer.plan.ready_cycle > *debugStopCycle)
      continue;
    transfer.launch_event_tag = 0x10000u + transferIndex;
    transfer.fence = weight_pager_->schedule(transfer.page,
        static_cast<std::size_t>(transfer.plan.start_cycle),
        transfer.launch_event_tag);
    ++stats_.weight_page_prefetches;
    for (const C2cWeightSegment &segment : transfer.page.segments)
      stats_.weight_page_prefetch_bytes +=
          static_cast<std::size_t>(segment.vector_count) *
          hw::kPhysicalVectorBytes;
  }

  std::vector<ExecutableWeightTransfer *> lookahead;
  for (ExecutableWeightTransfer &transfer :
       lookahead_executable_weight_transfers_)
    if (transfer.plan.pre_execution)
      lookahead.push_back(&transfer);
  if (lookahead_model_weight_transfer_)
    lookahead.push_back(&*lookahead_model_weight_transfer_);
  std::ranges::sort(lookahead, [](const auto *lhs, const auto *rhs) {
    return lhs->plan.start_cycle < rhs->plan.start_cycle;
  });
  for (std::size_t index = 0; index < lookahead.size(); ++index) {
    ExecutableWeightTransfer &transfer = *lookahead[index];
    const std::uint64_t duration = transfer.plan.transfer_end_cycle;
    transfer.plan.start_cycle = std::max<std::uint64_t>(
        transfer.plan.start_cycle,
        weight_pager_->earliest_schedule_cycle(transfer.page));
    transfer.plan.transfer_end_cycle = transfer.plan.start_cycle + duration;
    transfer.launch_event_tag = 0x20000u + index;
    transfer.fence = weight_pager_->schedule(
        transfer.page, static_cast<std::size_t>(transfer.plan.start_cycle),
        transfer.launch_event_tag);
    ++stats_.weight_page_prefetches;
    for (const C2cWeightSegment &segment : transfer.page.segments)
      stats_.weight_page_prefetch_bytes +=
          static_cast<std::size_t>(segment.vector_count) *
          hw::kPhysicalVectorBytes;
  }
  executable_clock_active_ = true;
}

void ModelSession::release_due_executable_weight_pages() {
  if (!executable_clock_active_ || c2c_system_ == nullptr)
    return;
  const auto release = [&](ExecutableWeightTransfer &transfer) {
    if (transfer.launch_event_tag == 0 || transfer.launch_released ||
        transfer.plan.start_cycle > executable_cycle_)
      return;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
      if (transfer.fence.dma_issues_end[side] ==
          transfer.fence.dma_issues_begin[side])
        continue;
      const auto hemisphere = static_cast<Hemisphere>(side);
      c2c_system_->chip().icu().notify_tagged(
          IcuLocation::C2cDma(hemisphere), transfer.launch_event_tag);
      c2c_system_->chip().icu().notify_tagged(
          IcuLocation::C2cRx(hemisphere), transfer.launch_event_tag);
    }
    transfer.launch_released = true;
  };
  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_)
    release(transfer);
  for (ExecutableWeightTransfer &transfer :
       lookahead_executable_weight_transfers_)
    release(transfer);
  if (lookahead_model_weight_transfer_)
    release(*lookahead_model_weight_transfer_);
}

void ModelSession::observe_executable_weight_page_tick() {
  if (!executable_clock_active_ || !weight_pager_)
    return;
  const auto physicalCycle =
      static_cast<std::int64_t>(runtime_.physical_cycles());
  const auto observe = [&](ExecutableWeightTransfer &transfer) {
    if (transfer.launch_event_tag == 0 || !transfer.launch_released ||
        transfer.trace_recorded)
      return;
    if (!transfer.actual_start_cycle &&
        weight_pager_->started(transfer.fence))
      transfer.actual_start_cycle = physicalCycle;
    if (!weight_pager_->ready(transfer.fence))
      return;
    if (!transfer.actual_start_cycle)
      transfer.actual_start_cycle = physicalCycle;
    transfer.actual_ready_cycle = physicalCycle + 1;
    record_weight_page_trace(transfer);
  };
  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_)
    observe(transfer);
  for (ExecutableWeightTransfer &transfer :
       lookahead_executable_weight_transfers_)
    observe(transfer);
  if (lookahead_model_weight_transfer_)
    observe(*lookahead_model_weight_transfer_);
}

std::size_t ModelSession::settle_executable_weight_lookahead() {
  if (!weight_pager_ || !c2c_system_ || !lookahead_invocation_index_)
    return 0;
  std::vector<ExecutableWeightTransfer *> transfers;
  for (ExecutableWeightTransfer &transfer :
       lookahead_executable_weight_transfers_)
    if (transfer.plan.pre_execution && transfer.launch_event_tag != 0)
      transfers.push_back(&transfer);
  if (lookahead_model_weight_transfer_ &&
      lookahead_model_weight_transfer_->launch_event_tag != 0)
    transfers.push_back(&*lookahead_model_weight_transfer_);
  if (transfers.empty())
    return 0;

  // Once the current executable has drained, every current-layer residency
  // interval is over. Release any lookahead event that was intentionally
  // placed at the tail and then wait only for real unfinished transport.
  for (ExecutableWeightTransfer *transfer : transfers) {
    if (transfer->launch_released)
      continue;
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
      if (transfer->fence.dma_issues_end[side] ==
          transfer->fence.dma_issues_begin[side])
        continue;
      const auto hemisphere = static_cast<Hemisphere>(side);
      c2c_system_->chip().icu().notify_tagged(
          IcuLocation::C2cDma(hemisphere), transfer->launch_event_tag);
      c2c_system_->chip().icu().notify_tagged(
          IcuLocation::C2cRx(hemisphere), transfer->launch_event_tag);
    }
    transfer->launch_released = true;
  }

  std::size_t maximumCycles = 4096;
  for (const ExecutableWeightTransfer *transfer : transfers)
    for (const C2cWeightSegment &segment : transfer->page.segments) {
      const std::size_t vectors = segment.vector_count;
      if (vectors > (std::numeric_limits<std::size_t>::max() -
                     maximumCycles) /
                        64)
        throw std::overflow_error("lookahead C2C timeout overflows size_t");
      maximumCycles += vectors * 64;
    }
  const std::int64_t physicalBase =
      static_cast<std::int64_t>(runtime_.physical_cycles());
  std::size_t waited = 0;
  const auto allReady = [&] {
    return std::ranges::all_of(transfers, [&](const auto *transfer) {
      return weight_pager_->ready(transfer->fence);
    });
  };
  while (!allReady() && waited < maximumCycles) {
    c2c_system_->tick();
    ++waited;
    const std::int64_t physicalCycle =
        physicalBase + static_cast<std::int64_t>(waited);
    for (ExecutableWeightTransfer *transfer : transfers) {
      if (!transfer->actual_start_cycle &&
          weight_pager_->started(transfer->fence))
        transfer->actual_start_cycle = physicalCycle;
      if (!transfer->trace_recorded &&
          weight_pager_->ready(transfer->fence)) {
        if (!transfer->actual_start_cycle)
          transfer->actual_start_cycle = physicalCycle;
        transfer->actual_ready_cycle = physicalCycle + 1;
        record_weight_page_trace(*transfer);
      }
    }
  }
  if (!allReady())
    throw std::runtime_error(
        "inter-invocation C2C weight lookahead timed out");
  if (std::getenv("FTLPU_VERIFY_LOOKAHEAD_WEIGHTS") != nullptr)
    for (const ExecutableWeightTransfer *transfer : transfers)
      verify_c2c_weight_page_residency(*c2c_system_, transfer->page);
  for (ExecutableWeightTransfer *transfer : transfers) {
    if (!transfer->actual_start_cycle)
      transfer->actual_start_cycle = physicalBase;
    if (!transfer->actual_ready_cycle)
      transfer->actual_ready_cycle = physicalBase +
          static_cast<std::int64_t>(waited) + 1;
    record_weight_page_trace(*transfer);
    transfer->ready_before_execution = true;
  }
  if (lookahead_model_weight_transfer_) {
    ready_weight_page_ =
        *lookahead_model_weight_transfer_->model_page_index;
    inflight_weight_page_.reset();
  }
  if (waited != 0) {
    stats_.weight_page_wait_cycles += waited;
    stats_.weight_page_boundary_wait_cycles += waited;
  }
  return waited;
}

void ModelSession::record_weight_page_trace(
    ExecutableWeightTransfer &transfer) {
  if (transfer.trace_recorded || !transfer.actual_start_cycle ||
      !transfer.actual_ready_cycle)
    return;
  std::ostringstream bindings;
  for (std::size_t index = 0; index < transfer.uses.size(); ++index) {
    if (index != 0)
      bindings << '+';
    bindings << transfer.uses[index].binding_index;
  }
  const std::uint64_t bandwidth =
      static_cast<std::uint64_t>(
          c2c_system_->chip().hardware_configuration()
              .c2c_streams_per_direction) *
      c2c_bytes_per_stream_per_cycle_;
  const std::size_t c2cStreamCount =
      c2c_system_->chip().hardware_configuration()
          .c2c_streams_per_direction;
  const std::size_t fabricStreamBase =
      transfer.page.fabric_stream_base.value_or(
          static_cast<std::uint16_t>(hw::kWestStreams - c2cStreamCount));
  for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
    if (transfer.plan.bytes[side] == 0)
      continue;
    const char *sideName = side == 0 ? "E" : "W";
    std::ostringstream detail;
    detail << "page=" << transfer.plan.page_index
           << " bank=" << transfer.plan.bank
           << " bindings=" << bindings.str()
           << " bytes=" << transfer.plan.bytes[side]
           << " bandwidth=" << bandwidth << "B/cycle"
           << " fabric_streams=" << fabricStreamBase << ".."
           << (fabricStreamBase + c2cStreamCount - 1)
           << " consumer_cycle=" << transfer.plan.ready_cycle
           << " actual_ready=" << *transfer.actual_ready_cycle
           << " phase=";
    if (transfer.inter_invocation_lookahead)
      detail << "inter_invocation_lookahead";
    else if (transfer.plan.pre_execution)
      detail << "pre_execution";
    else
      detail << "overlap";
    if (transfer.model_page_index)
      detail << " scope=model invocation_page="
             << *transfer.model_page_index;
    runtime_.record_execution_trace_interval(*transfer.actual_start_cycle,
        *transfer.actual_ready_cycle,
        std::string("C2C.") + sideName + ".Prefetch", detail.str());
    runtime_.record_execution_trace_interval(*transfer.actual_start_cycle,
        *transfer.actual_ready_cycle,
        std::string("SR.") + sideName + ".C2C.Shared",
        "page=" + std::to_string(transfer.plan.page_index) +
            " bank=" + std::to_string(transfer.plan.bank) +
            " streams=" + std::to_string(fabricStreamBase) + ".." +
            std::to_string(fabricStreamBase + c2cStreamCount - 1));
    runtime_.record_execution_trace_interval(*transfer.actual_start_cycle,
        *transfer.actual_ready_cycle,
        std::string("MEM.") + sideName + ".C2CWrite",
        "page=" + std::to_string(transfer.plan.page_index) +
            " bank=" + std::to_string(transfer.plan.bank));
  }
  transfer.trace_recorded = true;
}

bool ModelSession::executable_weight_page_ready(
    const BinaryWeightPageUse &use) const {
  for (const ExecutableWeightTransfer &transfer :
       executable_weight_transfers_) {
    for (const BinaryWeightPageUse &candidate : transfer.uses)
      if (candidate.binding_index == use.binding_index &&
          candidate.page_index == use.page_index &&
          candidate.bank == use.bank) {
        if (transfer.ready_before_execution)
          return true;
        const bool ready = weight_pager_->ready(transfer.fence);
        if (!ready && std::getenv("FTLPU_SESSION_PROGRESS") != nullptr) {
          std::clog << "FTLPU executable page wait: binding="
                    << use.binding_index << " page=" << use.page_index
                    << " cycle=" << executable_cycle_
                    << " planned_start=" << transfer.plan.start_cycle
                    << " planned_end=" << transfer.plan.transfer_end_cycle
                    << " ddr_read_bytes="
                    << c2c_system_->ddr4().read_bytes_transferred();
          for (std::size_t side = 0; side < hw::kHemispheres; ++side)
            for (std::size_t lane = 0;
                 lane < c2c_system_->chip()
                            .hardware_configuration()
                            .c2c_streams_per_direction;
                 ++lane)
              if (transfer.fence.completed_segments[side][lane] != 0)
                std::clog << " s" << side << "l" << lane << '='
                          << c2c_system_->chip()
                                 .c2c_endpoint(
                                     static_cast<Hemisphere>(side))
                                 .rx()
                                 .completed_instruction_count(lane)
                          << '/'
                          << transfer.fence.completed_segments[side][lane];
          std::clog << std::endl;
        }
        return ready;
      }
  }
  return false;
}

void ModelSession::ensure_weight_page(std::uint32_t page_index) {
  if (ready_weight_page_ == page_index) {
    ++stats_.weight_page_hidden_prefetches;
    return;
  }
  if (inflight_weight_page_ != page_index) {
    if (inflight_weight_page_)
      throw std::logic_error(
          "cannot replace a model weight page while another page is in flight");
    // Standalone C2C transactions, including resident uploads during session
    // load, leave their ICU programs launched. Start each unscheduled model
    // page from a fresh execution context; SRAM and DDR contents survive.
    c2c_system_->reset_execution_state();
    start_weight_page(page_index);
  }
  const std::size_t beginCycle = c2c_system_->cycle();
  weight_pager_->wait(
      std::max<std::size_t>(1024, weight_pager_->stats().vectors * 64));
  const std::size_t waitCycles = c2c_system_->cycle() - beginCycle;
  stats_.weight_page_wait_cycles += waitCycles;
  if (completed_invocation_)
    stats_.weight_page_boundary_wait_cycles += waitCycles;
  else
    stats_.weight_page_initial_wait_cycles += waitCycles;
  ready_weight_page_ = page_index;
  inflight_weight_page_.reset();
  weight_pager_->retire();
}

void ModelSession::load(ModelPackage package) {
  validate_model_package(package);
  memory_plan_ = SessionMemoryPlanner::plan(package);
  package_ = std::move(package);
  values_.clear();
  host_input_overrides_.clear();
  device_values_.clear();
  state_backing_.clear();
  stats_ = {};
  load_stats_ = {};
  executable_weight_transfers_.clear();
  lookahead_executable_weight_transfers_.clear();
  lookahead_model_weight_transfer_.reset();
  lookahead_invocation_index_.reset();
  completed_invocation_ = false;
  execution_trace_has_segment_ = false;
  execution_trace_cycle_cursor_ = 0;
  const ExecutableHardwareConfig *sessionHardware = nullptr;
  const bool usesLpu = !package_.invocations.empty() ||
                       !memory_plan_.resident_tensors.empty() ||
                       !memory_plan_.persistent_states.empty();
  if (usesLpu && !package_.executables.empty()) {
    sessionHardware = &package_.executables.front().program.hardware;
    for (const ModelExecutable &executable : package_.executables) {
      const auto &candidate = executable.program.hardware;
      if (candidate.lpu_clock_mhz != sessionHardware->lpu_clock_mhz ||
          candidate.ddr_peak_bandwidth_mbytes_per_second !=
              sessionHardware->ddr_peak_bandwidth_mbytes_per_second ||
          candidate.ddr_scheduling_efficiency_percent !=
              sessionHardware->ddr_scheduling_efficiency_percent ||
          candidate.ddr_read_latency_cycles !=
              sessionHardware->ddr_read_latency_cycles ||
          candidate.ddr_write_latency_cycles !=
              sessionHardware->ddr_write_latency_cycles ||
          candidate.ddr_read_latency_jitter_cycles !=
              sessionHardware->ddr_read_latency_jitter_cycles ||
          candidate.ddr_write_latency_jitter_cycles !=
              sessionHardware->ddr_write_latency_jitter_cycles ||
          candidate.ddr_request_queue_depth !=
              sessionHardware->ddr_request_queue_depth ||
          candidate.ddr_latency_random_seed !=
              sessionHardware->ddr_latency_random_seed)
        throw std::invalid_argument(
            "all executables in one session must share the external-memory "
            "target configuration");
    }
    configure_external_transport(
        effective_external_transport(*sessionHardware));
  }
  prepare_weight_pages();
  const bool report_progress = std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
  if (report_progress)
    std::clog << "FTLPU session resident uploads: "
              << memory_plan_.resident_tensors.size() << std::endl;
  for (const auto &resident : memory_plan_.resident_tensors) {
    const std::size_t upload_index = stats_.resident_uploads + 1;
    if (report_progress &&
        (memory_plan_.resident_tensors.size() <= 16 || upload_index == 1 ||
         upload_index == memory_plan_.resident_tensors.size() ||
         upload_index % 25 == 0))
      std::clog << "FTLPU resident upload " << upload_index << '/'
                << memory_plan_.resident_tensors.size() << ": "
                << resident.value << " base_row=" << resident.binding.base_row
                << " bytes=" << resident.binding.byte_size << std::endl;
    if (sessionHardware == nullptr)
      throw std::logic_error(
          "resident LPU data requires an executable hardware target");
    upload_binding_through_c2c(resident.binding, resolve_value(resident.value),
                               *sessionHardware);
    ++stats_.resident_uploads;
    stats_.resident_upload_bytes +=
        static_cast<std::size_t>(resident.binding.byte_size);
  }
  for (const ModelState &state : package_.states) {
    const std::size_t elements = element_count(state.shape);
    const std::size_t bytesPerElement = element_size(state.element_type);
    if (elements > std::numeric_limits<std::size_t>::max() / bytesPerElement)
      throw std::overflow_error("persistent state backing is too large");
    const std::size_t logicalBytes = elements * bytesPerElement;
    const auto [_, inserted] = state_backing_.emplace(
        state.name, std::vector<std::uint8_t>(logicalBytes, 0));
    if (!inserted)
      throw std::logic_error("duplicate persistent state backing: " +
                             state.name);
    ++stats_.state_initializations;
    stats_.state_initialization_bytes += logicalBytes;
  }
  load_stats_ = stats_;
  loaded_ = true;
}

void ModelSession::load_file(const std::filesystem::path &path) {
  load(read_model_package(path, ModelPackageLoadMode::LazyExecutables));
}

std::vector<std::uint8_t> ModelSession::read_state(const std::string &name) {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  const auto state = state_backing_.find(name);
  if (state == state_backing_.end())
    throw std::out_of_range("unknown FTLPU model state: " + name);
  return state->second;
}

void ModelSession::reset_states() {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  for (auto &[name, state] : state_backing_) {
    (void)name;
    std::fill(state.begin(), state.end(), 0);
    ++stats_.state_initializations;
    stats_.state_initialization_bytes += state.size();
  }
}

const ModelValue *
ModelSession::find_value_metadata(const std::string &name) const {
  for (const auto &value : package_.values)
    if (value.name == name)
      return &value;
  return nullptr;
}

void ModelSession::set_input(std::string name,
                             std::span<const std::uint8_t> data) {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  const ModelValue *metadata = find_value_metadata(name);
  if (!metadata || !metadata->external_input)
    throw std::invalid_argument(
        "FTLPU model input is not declared as external");
  host_input_overrides_.insert(name);
  values_[std::move(name)] =
      std::vector<std::uint8_t>(data.begin(), data.end());
}

const std::vector<std::uint8_t> &
ModelSession::resolve_value(const std::string &name) const {
  if (const auto value = values_.find(name); value != values_.end())
    return value->second;
  for (const auto &tensor : package_.tensors)
    if (tensor.name == name)
      return tensor.data;
  throw std::out_of_range("FTLPU model value is not available: " + name);
}

void ModelSession::run_invocation(std::size_t index, std::size_t drain_cycles) {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  if (index >= package_.invocations.size())
    throw std::out_of_range("FTLPU model invocation index is out of range");
  const bool reportProgress = std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
  const auto report = [&](const char *phase) {
    if (reportProgress)
      std::clog << "FTLPU invocation " << index << ": " << phase << std::endl;
  };
  report("begin");
  const auto &invocation = package_.invocations[index];
  const auto &executable = package_.executables.at(invocation.executable_index);
  const SessionInvocationPlan &invocation_plan =
      memory_plan_.invocations.at(index);
  const std::size_t pageWaitBefore = stats_.weight_page_wait_cycles;
  if (invocation.weight_page != 0xffffffffu)
    ensure_weight_page(invocation.weight_page);
  report("model weight page ready");
  const std::size_t modelPageWaitCycles =
      stats_.weight_page_wait_cycles - pageWaitBefore;
  const BinaryProgram program =
      parameterize_program(package_, invocation, invocation_plan,
                           materialize_model_executable(executable));
  report("program parameterized");

  const std::size_t stateIngressCyclesBefore = stats_.c2c_ingress_cycles;
  const std::size_t stateIngressBytesBefore = stats_.c2c_ingress_bytes;
  for (const SessionStatePlan &state : invocation_plan.states) {
    const BinaryBinding &binding =
        find_binding(program, BindingAccess::Internal, state.binding_index);
    const auto backing = state_backing_.find(state.state);
    if (backing == state_backing_.end())
      throw std::logic_error("persistent state backing is unavailable: " +
                             state.state);
    const std::size_t residentBytes =
        static_cast<std::size_t>(binding.byte_size);
    if (backing->second.size() < residentBytes)
      throw std::logic_error(
          "persistent state backing is smaller than its SRAM window: " +
          state.state);
    upload_binding_through_c2c(
        binding,
        std::span<const std::uint8_t>(backing->second.data(), residentBytes),
        program.hardware);
    ++stats_.state_page_ins;
  }
  const std::size_t statePageInCycles =
      stats_.c2c_ingress_cycles - stateIngressCyclesBefore;
  const std::size_t statePageInBytes =
      stats_.c2c_ingress_bytes - stateIngressBytesBefore;
  stats_.state_page_in_cycles += statePageInCycles;
  stats_.state_page_in_bytes += statePageInBytes;
  report("persistent state paged in");

  const std::size_t ingressBefore = stats_.c2c_ingress_cycles;
  for (const SessionInputPlan &input : invocation_plan.inputs) {
    if (input.transfer == SessionTransferKind::Resident ||
        input.transfer == SessionTransferKind::WeightPage)
      continue;
    if (input.transfer == SessionTransferKind::HostUpload ||
        host_input_overrides_.contains(input.value)) {
      upload_binding_through_c2c(
          find_binding(program, BindingAccess::Input, input.binding_index),
          resolve_value(input.value), program.hardware);
      ++stats_.host_uploads;
      continue;
    }
    const auto source = device_values_.find(input.value);
    if (source == device_values_.end())
      throw std::logic_error("device-resident model value is not available");
    if (source->second.target_abi != program.target_abi)
      throw std::logic_error("device-resident value cannot cross target ABIs");
    const BinaryBinding &destination =
        find_binding(program, BindingAccess::Input, input.binding_index);
    if (input.transfer == SessionTransferKind::DeviceAlias) {
      if (!bindings_physically_alias(source->second.binding, destination))
        throw std::logic_error(
            "session device alias no longer matches executable ABI");
      ++stats_.device_aliases;
    } else {
      runtime_.copy_binding(source->second.binding, destination);
      ++stats_.device_copies;
      stats_.device_copy_bytes +=
          static_cast<std::size_t>(destination.byte_size);
    }
    if (input.release_after_transfer)
      device_values_.erase(source);
  }
  const std::size_t inputTransferCycles =
      stats_.c2c_ingress_cycles - ingressBefore;
  report("inputs ready");
  const std::size_t executablePageWaitBefore = stats_.weight_page_wait_cycles;
  prepare_executable_weight_pages(program, invocation, index);
  const std::size_t executablePreExecutionCycles =
      stats_.weight_page_wait_cycles - executablePageWaitBefore;
  report("executable weight pages prepared");
  const auto traceOrigin =
      execution_trace_cycle_cursor_ +
      static_cast<std::int64_t>(modelPageWaitCycles) +
      static_cast<std::int64_t>(statePageInCycles) +
      static_cast<std::int64_t>(inputTransferCycles) +
      static_cast<std::int64_t>(executablePreExecutionCycles);
  if (execution_trace_enabled_)
    runtime_.configure_execution_trace_segment(traceOrigin,
                                               execution_trace_has_segment_);
  report("loading runtime program");
  runtime_.load(program);
  report("runtime program loaded");
  if (execution_trace_enabled_) {
    std::int64_t localCursor = -static_cast<std::int64_t>(
        modelPageWaitCycles + statePageInCycles + inputTransferCycles +
        executablePreExecutionCycles);
    if (modelPageWaitCycles != 0) {
      std::ostringstream detail;
      detail << "invocation=" << index;
      if (invocation.weight_page != 0xffffffffu)
        detail << " page=" << invocation.weight_page;
      detail << " phase="
             << (completed_invocation_ ? "layer_boundary" : "initial");
      runtime_.record_execution_trace_interval(
          localCursor,
          localCursor + static_cast<std::int64_t>(modelPageWaitCycles),
          "C2C.ModelWeightPage", detail.str());
      localCursor += static_cast<std::int64_t>(modelPageWaitCycles);
    }
    if (statePageInCycles != 0) {
      runtime_.record_execution_trace_interval(
          localCursor,
          localCursor + static_cast<std::int64_t>(statePageInCycles),
          "C2C.StatePageIn",
          "invocation=" + std::to_string(index) +
              " states=" + std::to_string(invocation_plan.states.size()) +
              " bytes=" + std::to_string(statePageInBytes));
      localCursor += static_cast<std::int64_t>(statePageInCycles);
    }
    if (inputTransferCycles != 0)
      runtime_.record_execution_trace_interval(
          localCursor,
          localCursor + static_cast<std::int64_t>(inputTransferCycles),
          "C2C.HostInput", "invocation=" + std::to_string(index));
  }
  if (!program.weight_page_uses.empty())
    prepare_executable_weight_lookahead(index, program);
  schedule_executable_weight_pages();
  if (index + 1 < package_.invocations.size()) {
    const auto nextPage = package_.invocations[index + 1].weight_page;
    if (nextPage != 0xffffffffu && nextPage != invocation.weight_page &&
        !lookahead_model_weight_transfer_) {
      if (nextPage >= c2c_pages_.size())
        throw std::logic_error("next model weight page is unavailable");
      if (!program.weight_page_uses.empty() ||
          weight_page_overlaps_program(c2c_pages_[nextPage], program))
        ++stats_.weight_page_deferred_prefetches;
      else
        start_weight_page(nextPage);
    }
  }
  std::size_t executionCycles = program.max_cycle + drain_cycles;
  if (const char *stop = std::getenv("FTLPU_SESSION_STOP_CYCLE"))
    executionCycles =
        std::min(executionCycles, static_cast<std::size_t>(std::stoull(stop)));
  report("executing runtime program");
  const char *traceStartText = std::getenv("FTLPU_SESSION_TRACE_START");
  const char *traceCyclesText = std::getenv("FTLPU_SESSION_TRACE_CYCLES");
  if (traceStartText != nullptr && traceCyclesText != nullptr) {
    const auto traceStart =
        static_cast<std::size_t>(std::stoull(traceStartText));
    const auto traceCycles =
        static_cast<std::size_t>(std::stoull(traceCyclesText));
    if (traceStart + traceCycles > executionCycles)
      throw std::logic_error(
          "session trace window exceeds invocation execution");
    runtime_.run_cycles(traceStart);
    runtime_.run_cycles(traceCycles, &std::cerr);
    runtime_.run_cycles(executionCycles - traceStart - traceCycles);
  } else {
    runtime_.run_cycles(executionCycles);
  }
  report("runtime program completed");
  const std::size_t invocationPhysicalCycles = runtime_.physical_cycles();
  const std::size_t lookaheadBoundaryCycles =
      settle_executable_weight_lookahead();
  if (execution_trace_enabled_)
    runtime_.record_execution_trace_interval(
        0, static_cast<std::int64_t>(invocationPhysicalCycles),
        "Session.Invocation",
        "index=" + std::to_string(index) + " name=" + invocation.name);
  executable_clock_active_ = false;
  completed_invocation_ = true;
  if (std::getenv("FTLPU_SESSION_TRACE_BINDINGS") != nullptr) {
    for (const BinaryBinding &binding : program.bindings) {
      if (binding.shape.size() != 2 || !is_16bit_float(binding.element_type) ||
          binding.role == "weight")
        continue;
      std::vector<std::uint8_t> data;
      try {
        data = download_binding_through_c2c(binding, program.hardware);
      } catch (const std::exception &error) {
        std::clog << "FTLPU binding trace: cycle=" << executionCycles
                  << " access=" << static_cast<int>(binding.access)
                  << " index=" << binding.index << " role=" << binding.role
                  << " name=" << binding.name << " bank=" << binding.bank
                  << " base=" << binding.base_row
                  << " layout=" << static_cast<int>(binding.layout)
                  << " unavailable=" << error.what() << std::endl;
        continue;
      }
      std::size_t nonFinite = 0;
      float minimum = std::numeric_limits<float>::infinity();
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t element = 0;
           element < data.size() / sizeof(std::uint16_t); ++element) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, data.data() + element * sizeof(std::uint16_t),
                    sizeof(bits));
        const float value = decode_16bit_float(bits, binding.element_type);
        if (!std::isfinite(value)) {
          ++nonFinite;
          continue;
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
      std::clog << "FTLPU binding trace: cycle=" << executionCycles
                << " access=" << static_cast<int>(binding.access)
                << " index=" << binding.index << " role=" << binding.role
                << " name=" << binding.name << " bank=" << binding.bank
                << " base=" << binding.base_row
                << " layout=" << static_cast<int>(binding.layout)
                << " non_finite=" << nonFinite << " min=" << minimum
                << " max=" << maximum << std::endl;
    }
  }
  const std::size_t stateEgressCyclesBefore = stats_.c2c_egress_cycles;
  const std::size_t stateEgressBytesBefore = stats_.c2c_egress_bytes;
  for (const SessionStatePlan &state : invocation_plan.states) {
    const BinaryBinding &binding =
        find_binding(program, BindingAccess::Internal, state.binding_index);
    std::vector<std::uint8_t> resident =
        download_binding_through_c2c(binding, program.hardware);
    auto backing = state_backing_.find(state.state);
    if (backing == state_backing_.end() ||
        backing->second.size() < resident.size())
      throw std::logic_error(
          "persistent state backing cannot receive its SRAM window: " +
          state.state);
    std::copy(resident.begin(), resident.end(), backing->second.begin());
    ++stats_.state_page_outs;
  }
  const std::size_t statePageOutCycles =
      stats_.c2c_egress_cycles - stateEgressCyclesBefore;
  const std::size_t statePageOutBytes =
      stats_.c2c_egress_bytes - stateEgressBytesBefore;
  stats_.state_page_out_cycles += statePageOutCycles;
  stats_.state_page_out_bytes += statePageOutBytes;
  report("persistent state paged out");
  if (execution_trace_enabled_ && statePageOutCycles != 0)
    runtime_.record_execution_trace_interval(
        static_cast<std::int64_t>(invocationPhysicalCycles +
                                  lookaheadBoundaryCycles),
        static_cast<std::int64_t>(invocationPhysicalCycles +
                                  lookaheadBoundaryCycles + statePageOutCycles),
        "C2C.StatePageOut",
        "invocation=" + std::to_string(index) +
            " states=" + std::to_string(invocation_plan.states.size()) +
            " bytes=" + std::to_string(statePageOutBytes));

  const std::size_t egressBefore = stats_.c2c_egress_cycles;
  const bool validate_fp16 =
      std::getenv("FTLPU_SESSION_VALIDATE_FP16") != nullptr;
  for (const SessionOutputPlan &output : invocation_plan.outputs) {
    const BinaryBinding &binding =
        find_binding(program, BindingAccess::Output, output.binding_index);
    std::optional<std::vector<std::uint8_t>> hostData;
    if (validate_fp16 || output.download_to_host)
      hostData = download_binding_through_c2c(binding, program.hardware);
    if (validate_fp16 && is_16bit_float(binding.element_type)) {
      const auto &data = *hostData;
      for (std::size_t element = 0;
           element < data.size() / sizeof(std::uint16_t); ++element) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, data.data() + element * sizeof(std::uint16_t),
                    sizeof(bits));
        if (!std::isfinite(decode_16bit_float(bits, binding.element_type)))
          throw std::logic_error("model invocation produced non-finite "
                                 "16-bit float: " +
                                 invocation.name +
                                 " element=" + std::to_string(element));
      }
    }
    if (output.retain_on_device) {
      device_values_[output.value] = DeviceValue{
          binding,
          program.target_abi,
      };
    }
    if (output.download_to_host) {
      values_[output.value] = std::move(*hostData);
      ++stats_.host_downloads;
    }
  }
  const std::size_t outputTransferCycles =
      stats_.c2c_egress_cycles - egressBefore;
  report("outputs resolved");
  if (execution_trace_enabled_ && outputTransferCycles != 0)
    runtime_.record_execution_trace_interval(
        static_cast<std::int64_t>(invocationPhysicalCycles +
                                  lookaheadBoundaryCycles + statePageOutCycles),
        static_cast<std::int64_t>(invocationPhysicalCycles +
                                  lookaheadBoundaryCycles + statePageOutCycles +
                                  outputTransferCycles),
        "C2C.HostOutput", "invocation=" + std::to_string(index));
  execution_trace_cycle_cursor_ =
      traceOrigin + static_cast<std::int64_t>(invocationPhysicalCycles) +
      static_cast<std::int64_t>(lookaheadBoundaryCycles) +
      static_cast<std::int64_t>(statePageOutCycles) +
      static_cast<std::int64_t>(outputTransferCycles);
  execution_trace_has_segment_ = true;
  report("complete");
}

void ModelSession::run_embedding_lookups() {
  for (const ModelEmbeddingLookup &lookup : package_.embedding_lookups) {
    const ModelValue *ids_metadata = find_value_metadata(lookup.token_ids);
    const ModelValue *output_metadata = find_value_metadata(lookup.output);
    const ModelTensor &table = find_tensor(package_, lookup.table);
    if (!ids_metadata ||
        ids_metadata->element_type != BindingElementType::I32 ||
        ids_metadata->shape.size() != 1 || !output_metadata ||
        !is_16bit_float(output_metadata->element_type) ||
        output_metadata->shape.size() != 2 ||
        table.element_type != output_metadata->element_type ||
        table.encoding != ModelTensorEncoding::Raw || table.shape.size() != 2 ||
        output_metadata->shape[0] != ids_metadata->shape[0] ||
        output_metadata->shape[1] != table.shape[1])
      throw std::logic_error(
          "embedding lookup requires i32 ids and a matching raw "
          "rank-2 16-bit float table");
    const auto &ids = resolve_value(lookup.token_ids);
    if (ids.size() !=
            element_count(ids_metadata->shape) * sizeof(std::int32_t) ||
        table.data.size() != element_count(table.shape) * sizeof(std::uint16_t))
      throw std::logic_error("embedding lookup tensor byte size mismatch");

    const std::size_t row_bytes =
        static_cast<std::size_t>(table.shape[1]) * sizeof(std::uint16_t);
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(ids_metadata->shape[0]) * row_bytes);
    for (std::size_t row = 0; row < ids_metadata->shape[0]; ++row) {
      std::int32_t token = 0;
      std::memcpy(&token, ids.data() + row * sizeof(token), sizeof(token));
      if (token < 0 || static_cast<std::uint64_t>(token) >= table.shape[0])
        throw std::out_of_range("embedding token id is out of range");
      std::memcpy(output.data() + row * row_bytes,
                  table.data.data() +
                      static_cast<std::size_t>(token) * row_bytes,
                  row_bytes);
    }
    values_[lookup.output] = std::move(output);
    ++stats_.host_operations;
  }
}

void ModelSession::run_host_lm_heads() {
  for (const ModelHostLmHead &lm_head : package_.host_lm_heads) {
    const ModelValue *hidden_metadata = find_value_metadata(lm_head.hidden);
    const ModelValue *output_metadata = find_value_metadata(lm_head.output);
    const ModelTensor &weight = find_tensor(package_, lm_head.weight);
    if (!hidden_metadata || !is_16bit_float(hidden_metadata->element_type) ||
        hidden_metadata->shape.size() != 2 || !output_metadata ||
        output_metadata->shape.size() != 2 ||
        (output_metadata->element_type != hidden_metadata->element_type &&
         output_metadata->element_type != BindingElementType::F32) ||
        weight.element_type != hidden_metadata->element_type ||
        weight.encoding != ModelTensorEncoding::Raw ||
        weight.shape.size() != 2 ||
        hidden_metadata->shape[1] != weight.shape[1] ||
        output_metadata->shape[0] !=
            (lm_head.last_token_only ? 1 : hidden_metadata->shape[0]) ||
        output_metadata->shape[1] != weight.shape[0])
      throw std::logic_error(
          "host LM head requires matching [tokens, hidden] and "
          "[vocab, hidden] 16-bit float tensors");

    const auto &hidden = resolve_value(lm_head.hidden);
    if (hidden.size() !=
            element_count(hidden_metadata->shape) * sizeof(std::uint16_t) ||
        weight.data.size() !=
            element_count(weight.shape) * sizeof(std::uint16_t))
      throw std::logic_error("host LM head tensor byte size mismatch");

    const std::size_t hidden_size = static_cast<std::size_t>(weight.shape[1]);
    const std::size_t vocabulary = static_cast<std::size_t>(weight.shape[0]);
    const std::size_t output_rows =
        static_cast<std::size_t>(output_metadata->shape[0]);
    const std::size_t first_hidden_row =
        lm_head.last_token_only
            ? static_cast<std::size_t>(hidden_metadata->shape[0] - 1)
            : 0;
    const std::size_t output_element_bytes =
        output_metadata->element_type == BindingElementType::F32
            ? sizeof(float)
            : sizeof(std::uint16_t);
    std::vector<std::uint8_t> output(output_rows * vocabulary *
                                     output_element_bytes);

    const auto read_16bit = [&](const std::uint8_t *source) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, source, sizeof(bits));
      return decode_16bit_float(bits, hidden_metadata->element_type);
    };
    for (std::size_t row = 0; row < output_rows; ++row) {
      const std::size_t hidden_row = first_hidden_row + row;
      for (std::size_t token = 0; token < vocabulary; ++token) {
        float accumulator = 0.0f;
        for (std::size_t column = 0; column < hidden_size; ++column) {
          const std::size_t hidden_index = hidden_row * hidden_size + column;
          const std::size_t weight_index = token * hidden_size + column;
          accumulator +=
              read_16bit(hidden.data() + hidden_index * sizeof(std::uint16_t)) *
              read_16bit(weight.data.data() +
                         weight_index * sizeof(std::uint16_t));
        }
        const std::size_t output_index = row * vocabulary + token;
        if (output_metadata->element_type == BindingElementType::F32) {
          std::memcpy(output.data() + output_index * sizeof(float),
                      &accumulator, sizeof(accumulator));
        } else {
          const std::uint16_t bits =
              encode_16bit_float(accumulator, output_metadata->element_type);
          std::memcpy(output.data() + output_index * sizeof(bits), &bits,
                      sizeof(bits));
        }
      }
    }
    values_[lm_head.output] = std::move(output);
    ++stats_.host_operations;
  }
}

void ModelSession::run(std::size_t drain_cycles) {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  device_values_.clear();
  stats_ = load_stats_;
  executable_weight_transfers_.clear();
  lookahead_executable_weight_transfers_.clear();
  lookahead_model_weight_transfer_.reset();
  lookahead_invocation_index_.reset();
  completed_invocation_ = false;
  execution_trace_has_segment_ = false;
  execution_trace_cycle_cursor_ = 0;
  run_embedding_lookups();
  const bool report_progress = std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
  for (std::size_t index = 0; index < package_.invocations.size(); ++index) {
    if (report_progress)
      std::clog << "FTLPU session invocation " << (index + 1) << '/'
                << package_.invocations.size() << ": "
                << package_.invocations[index].name << std::endl;
    run_invocation(index, drain_cycles);
  }
  run_host_lm_heads();
}

const ModelPackage &ModelSession::package() const {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  return package_;
}

const std::vector<std::uint8_t> &
ModelSession::value(const std::string &name) const {
  return resolve_value(name);
}

const SessionMemoryPlan &ModelSession::memory_plan() const {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  return memory_plan_;
}

const ModelSessionStats &ModelSession::stats() const {
  if (!loaded_)
    throw std::logic_error("no FTLPU model package is loaded");
  return stats_;
}

std::vector<WeightPrefetchPlan>
ModelSession::executable_weight_prefetch_plans() const {
  std::vector<WeightPrefetchPlan> plans;
  plans.reserve(executable_weight_transfers_.size());
  for (const ExecutableWeightTransfer &transfer : executable_weight_transfers_)
    plans.push_back(transfer.plan);
  return plans;
}

} // namespace ftlpu::software::runtime
