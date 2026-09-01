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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

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
    if (command.instruction_kind != InstructionKind::Mem ||
        command.word_count == 0 || command.word_count > 2)
      throw std::logic_error(
          "address relocation target is not a MEM instruction");
    const isa::EncodedMemInstruction encoded =
        static_cast<isa::EncodedMemInstruction>(command.words[0]) |
        (static_cast<isa::EncodedMemInstruction>(command.words[1]) << 32);
    MemInstruction instruction = isa::decode_mem_instruction(encoded);
    if (relocation.write_port)
      throw std::logic_error(
          "legacy MEM ReadWrite relocation is not supported by "
          "the banked MEM ISA");
    const std::size_t sourceAddress = instruction.address;
    auto [range, inserted] = relocation_address_ranges.try_emplace(
        key, static_cast<std::int64_t>(sourceAddress),
        static_cast<std::int64_t>(sourceAddress));
    if (!inserted) {
      range->second.first = std::min(range->second.first,
                                     static_cast<std::int64_t>(sourceAddress));
      range->second.second = std::max(range->second.second,
                                      static_cast<std::int64_t>(sourceAddress));
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
    instruction.address = static_cast<std::size_t>(relocated);
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

} // namespace

ModelSession::ModelSession(TspSliceSystem &system) : runtime_(system) {}

ModelSession::ModelSession(C2cDmaSystem &system)
    : runtime_(system.chip(),
               [this, &system](TspSliceSystem::LogSinks sinks) {
                 system.tick(sinks);
                 observe_weight_page_tick();
                 if (executable_clock_active_)
                   ++executable_cycle_;
               }),
      c2c_system_(&system),
      weight_pager_(std::make_unique<C2cWeightPager>(system)) {
  runtime_.set_weight_page_residency_checker(
      [this](const BinaryWeightPageUse &use) {
        return executable_weight_page_ready(use);
      });
}

void ModelSession::configure_external_transport(
    const ExecutableHardwareConfig &hardware) {
  if (c2c_system_ == nullptr)
    throw std::logic_error(
        "ModelSession external transfers require C2cDmaSystem");

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
            hardware.ddr_peak_bandwidth_mbytes_per_second) *
        hardware.ddr_scheduling_efficiency_percent;
    if (effectiveBandwidth == 0)
      throw std::logic_error("C2C output requires non-zero DDR bandwidth");
    const std::uint64_t producerDemand =
        static_cast<std::uint64_t>(batch.size()) *
        hw::kPhysicalVectorBytes * hardware.lpu_clock_mhz * 100;
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
        4096, batchVectors * 64 + hardware.ddr_write_latency_cycles +
                  hardware.ddr_write_latency_jitter_cycles);
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

void ModelSession::prepare_executable_weight_pages(
    const BinaryProgram &program, const ModelInvocation &invocation) {
  executable_weight_transfers_.clear();
  executable_cycle_ = 0;
  executable_clock_active_ = false;
  if (program.weight_page_uses.empty())
    return;
  if (c2c_system_ == nullptr || !weight_pager_)
    throw std::logic_error(
        "executable-local paged weights require ModelSession(C2cDmaSystem&)");

  std::uint64_t nextDdrAddress = executable_ddr4_address_;
  auto plans = plan_weight_prefetches(program);
  executable_weight_transfers_.reserve(plans.size());
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
    executable_weight_transfers_.push_back(std::move(transfer));
  }
  for (std::size_t index = 0; index < plans.size(); ++index)
    plans[index] = executable_weight_transfers_[index].plan;
  schedule_weight_prefetches(program, plans);
  for (std::size_t index = 0; index < plans.size(); ++index) {
    executable_weight_transfers_[index].plan = plans[index];
    if (std::getenv("FTLPU_SESSION_PROGRESS") == nullptr) continue;
    const auto &transfer = executable_weight_transfers_[index];
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
  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_) {
    if (!transfer.plan.pre_execution)
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
    stats_.weight_page_wait_cycles += waitCycles;
    stats_.weight_page_initial_wait_cycles += waitCycles;
    weight_pager_->retire();
    transfer.ready_before_execution = true;
  }
}

void ModelSession::schedule_executable_weight_pages() {
  if (executable_weight_transfers_.empty())
    return;
  std::optional<std::size_t> debugStopCycle;
  if (const char *stop = std::getenv("FTLPU_SESSION_STOP_CYCLE"))
    debugStopCycle = static_cast<std::size_t>(std::stoull(stop));
  weight_pager_->begin_schedule();
  for (ExecutableWeightTransfer &transfer : executable_weight_transfers_) {
    if (transfer.plan.pre_execution)
      continue;
    // A bounded diagnostic run must not inject traffic for a page whose first
    // consumer is beyond the stop point. Besides saving time, this keeps
    // stage captures isolated from future C2C/MEM traffic.
    if (debugStopCycle && transfer.plan.ready_cycle > *debugStopCycle)
      continue;
    transfer.fence = weight_pager_->schedule(
        transfer.page,
        static_cast<std::size_t>(transfer.plan.start_cycle));
    ++stats_.weight_page_prefetches;
    for (const C2cWeightSegment &segment : transfer.page.segments)
      stats_.weight_page_prefetch_bytes +=
          static_cast<std::size_t>(segment.vector_count) *
          hw::kPhysicalVectorBytes;
  }
  executable_clock_active_ = true;
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
          std::clog << "FTLPU executable page miss: binding="
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
    // A deferred page is loaded at an invocation boundary. The previous
    // executable has already launched its ICU queues, so clear execution
    // state before enqueueing C2C commands. SRAM and DDR contents remain
    // intact across this reset.
    if (completed_invocation_)
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
  device_values_.clear();
  stats_ = {};
  load_stats_ = {};
  completed_invocation_ = false;
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
    configure_external_transport(*sessionHardware);
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
    upload_binding_through_c2c(
        resident.binding, resolve_value(resident.value), *sessionHardware);
    ++stats_.resident_uploads;
    stats_.resident_upload_bytes +=
        static_cast<std::size_t>(resident.binding.byte_size);
  }
  for (const auto &state : memory_plan_.persistent_states) {
    const std::vector<std::uint8_t> zero(
        static_cast<std::size_t>(state.binding.byte_size), 0);
    if (sessionHardware == nullptr)
      throw std::logic_error(
          "persistent LPU state requires an executable hardware target");
    upload_binding_through_c2c(state.binding, zero, *sessionHardware);
    ++stats_.state_initializations;
    stats_.state_initialization_bytes += zero.size();
  }
  load_stats_ = stats_;
  loaded_ = true;
}

void ModelSession::load_file(const std::filesystem::path &path) {
  load(read_model_package(path, ModelPackageLoadMode::LazyExecutables));
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
  const auto &invocation = package_.invocations[index];
  const auto &executable = package_.executables.at(invocation.executable_index);
  const SessionInvocationPlan &invocation_plan =
      memory_plan_.invocations.at(index);
  if (invocation.weight_page != 0xffffffffu)
    ensure_weight_page(invocation.weight_page);
  const BinaryProgram program =
      parameterize_program(package_, invocation, invocation_plan,
                           materialize_model_executable(executable));

  for (const SessionInputPlan &input : invocation_plan.inputs) {
    if (input.transfer == SessionTransferKind::Resident ||
        input.transfer == SessionTransferKind::WeightPage)
      continue;
    if (input.transfer == SessionTransferKind::HostUpload) {
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
  prepare_executable_weight_pages(program, invocation);
  runtime_.load(program);
  schedule_executable_weight_pages();
  if (index + 1 < package_.invocations.size()) {
    const auto nextPage = package_.invocations[index + 1].weight_page;
    if (nextPage != 0xffffffffu && nextPage != invocation.weight_page) {
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
