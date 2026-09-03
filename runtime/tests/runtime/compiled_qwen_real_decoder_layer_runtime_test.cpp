#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(stream), {}};
}

void write_bytes(const std::filesystem::path &path,
                 const std::vector<std::uint8_t> &data) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot write " + path.string());
  stream.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

const BinaryBinding &find_binding(const BinaryProgram &program,
                                  BindingAccess access, std::uint32_t index) {
  const auto binding = std::find_if(
      program.bindings.begin(), program.bindings.end(),
      [&](const BinaryBinding &candidate) {
        return candidate.access == access && candidate.index == index;
      });
  if (binding == program.bindings.end())
    throw std::logic_error("Qwen decoder binary is missing binding " +
                           std::to_string(index));
  return *binding;
}

void print_weight_page_residency_error(
    const ftlpu::TspSliceSystem &system, const BinaryBinding &binding,
    std::span<const std::uint8_t> logical,
    const ExecutableHardwareConfig &hardware) {
  const PackedWeightImage expected =
      pack_weight_binding_page(binding, 0, logical, hardware);
  std::size_t mismatches = 0;
  bool printed_first = false;
  for (const PackedWeightSegment &segment : expected.segments) {
    for (std::uint32_t vector = 0; vector < segment.vector_count; ++vector) {
      for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
          const std::size_t offset =
              static_cast<std::size_t>(segment.byte_offset) +
              static_cast<std::size_t>(vector) *
                  ftlpu::hw::kPhysicalVectorBytes +
              tile * ftlpu::hw::kLanesPerTile + lane;
          const auto actual = system.read_mem_sram_lane_byte(
              static_cast<ftlpu::Hemisphere>(segment.hemisphere),
              segment.slice, binding.bank, tile, segment.base_row + vector,
              lane);
          if (actual == expected.data[offset])
            continue;
          ++mismatches;
          if (!printed_first) {
            std::cout << "Qwen weight-page first mismatch: binding="
                      << binding.index << " hemisphere=" << segment.hemisphere
                      << " slice=" << segment.slice << " bank="
                      << binding.bank << " row=" << segment.base_row + vector
                      << " tile=" << tile << " lane=" << lane
                      << " actual=" << static_cast<unsigned>(actual)
                      << " expected="
                      << static_cast<unsigned>(expected.data[offset]) << '\n';
            printed_first = true;
          }
        }
      }
    }
  }
  std::cout << "Qwen weight-page residency summary: binding=" << binding.index
            << " segments=" << expected.segments.size()
            << " bytes=" << expected.data.size()
            << " mismatches=" << mismatches << '\n';
}

const BinaryBinding &find_latest_internal_binding(
    const BinaryProgram &program, BindingLayout layout, std::uint16_t bank,
    std::vector<std::uint16_t> slices,
    std::vector<std::uint64_t> shape = {32, 1536}) {
  const BinaryBinding *latest = nullptr;
  for (const BinaryBinding &binding : program.bindings) {
    if (binding.access != BindingAccess::Internal ||
        binding.layout != layout || binding.bank != bank ||
        binding.slices != slices || binding.shape != shape)
      continue;
    if (latest == nullptr || binding.ready_cycle > latest->ready_cycle)
      latest = &binding;
  }
  if (latest == nullptr)
    throw std::logic_error("Qwen decoder is missing an internal stage binding");
  return *latest;
}

const BinaryBinding &find_earliest_internal_binding(
    const BinaryProgram &program, BindingLayout layout,
    std::vector<std::uint64_t> shape = {32, 1536}) {
  const BinaryBinding *earliest = nullptr;
  for (const BinaryBinding &binding : program.bindings) {
    if (binding.access != BindingAccess::Internal ||
        binding.layout != layout || binding.shape != shape)
      continue;
    if (earliest == nullptr || binding.ready_cycle < earliest->ready_cycle)
      earliest = &binding;
  }
  if (earliest == nullptr)
    throw std::logic_error("Qwen decoder is missing an internal stage binding");
  return *earliest;
}

const BinaryBinding &find_internal_binding_by_name(
    const BinaryProgram &program, std::string_view name) {
  const auto binding = std::find_if(
      program.bindings.begin(), program.bindings.end(),
      [&](const BinaryBinding &candidate) {
        return candidate.access == BindingAccess::Internal &&
               candidate.name == name;
      });
  if (binding == program.bindings.end())
    throw std::logic_error("Qwen decoder is missing internal binding " +
                           std::string(name));
  return *binding;
}

std::vector<std::uint8_t> download_swiglu_stage(
    CModelRuntime &runtime, const BinaryBinding &distributed_template) {
  BinaryBinding binding = distributed_template;
  binding.shape = {32, 8960};
  binding.byte_size = 32 * 8960 * sizeof(std::uint16_t);
  // Fused FFN keeps the normalized activation live while Swish starts. The
  // Tensor memory planner places hidden immediately after that distributed
  // input allocation when both use the same bank and slices.
  binding.base_row = distributed_template.base_row
      + distributed_template.instruction_count;
  binding.instruction_count = 1120;
  binding.address_stride = 1;
  binding.bank = 0;
  binding.slices = {0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15};
  binding.role = "result";
  binding.name = "ffn.swiglu";
  return runtime.download_binding(binding);
}

std::array<float, 7> read_quant_scales(const std::filesystem::path &fixture) {
  const auto bytes = read_bytes(fixture / "quant_scales.f32.bin");
  std::array<float, 7> scales{};
  if (bytes.size() != sizeof(scales))
    throw std::logic_error(
        "quant_scales.f32.bin must contain query, key, value, output, "
        "gate, up, and down FP32 scales");
  std::copy(bytes.begin(), bytes.end(),
            reinterpret_cast<std::uint8_t *>(scales.data()));
  return scales;
}

ModelTensor make_tensor(const BinaryProgram &program,
                        std::uint32_t binding_index, std::string name,
                        const std::filesystem::path &path,
                        ModelTensorEncoding encoding,
                        std::vector<float> scales = {}) {
  const BinaryBinding &binding =
      find_binding(program, BindingAccess::Input, binding_index);
  ModelTensor tensor;
  tensor.name = std::move(name);
  tensor.element_type = binding.element_type;
  tensor.shape = binding.shape;
  tensor.data = read_bytes(path);
  tensor.encoding = encoding;
  tensor.scales = std::move(scales);
  if (tensor.data.size() != binding.byte_size)
    throw std::logic_error("fixture byte size does not match binding " +
                           std::to_string(binding_index) +
                           ": fixture=" + std::to_string(tensor.data.size()) +
                           " binding=" + std::to_string(binding.byte_size));
  return tensor;
}

void append_packed_tensor(ModelPackage &package, ModelWeightPage &page,
                          const BinaryProgram &program,
                          std::uint32_t binding_index, std::string name,
                          const std::filesystem::path &path,
                          std::uint16_t &next_stream) {
  const BinaryBinding &binding =
      find_binding(program, BindingAccess::Input, binding_index);
  const auto logical = read_bytes(path);
  if (logical.size() != binding.byte_size)
    throw std::logic_error("packed tensor byte size does not match binding " +
                           std::to_string(binding_index));
  PackedWeightImage image =
      pack_weight_binding(binding, logical, program.hardware);
  const std::string tensor_name = name;
  package.tensors.push_back(
      ModelTensor{std::move(name),
                  BindingElementType::I8,
                  {static_cast<std::uint64_t>(image.data.size())},
                  std::move(image.data),
                  ModelTensorEncoding::TargetPackedSramVectors});
  page.tensors.push_back(tensor_name);
  for (const PackedWeightSegment &segment : image.segments) {
    page.segments.push_back(ModelWeightPage::Segment{
        tensor_name, segment.byte_offset, segment.hemisphere, segment.slice,
        segment.base_row, segment.vector_count, next_stream});
    next_stream = static_cast<std::uint16_t>(
        (next_stream + 1) % program.hardware.c2c_streams_per_direction);
  }
}

float bf16_at(const std::vector<std::uint8_t> &data, std::size_t index) {
  return ftlpu::Bf16::from_bits(
             static_cast<std::uint16_t>(data[2 * index]) |
             (static_cast<std::uint16_t>(data[2 * index + 1]) << 8))
      .to_float();
}

void print_pair_planar_blocks(const char *label,
                              const std::vector<std::uint8_t> &data,
                              std::size_t rows, std::size_t columns) {
  constexpr std::size_t kBlockColumns = 32;
  for (std::size_t block = 0; block < columns / kBlockColumns; ++block) {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    float maximum_absolute = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = block * kBlockColumns;
           column < (block + 1) * kBlockColumns; ++column) {
        const float value = bf16_at(data, row * columns + column);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        maximum_absolute = std::max(maximum_absolute, std::fabs(value));
      }
    }
    std::cout << "Qwen operand block: name=" << label
              << " block=" << block << " min=" << minimum
              << " max=" << maximum
              << " max_abs=" << maximum_absolute << '\n';
  }
}

void print_stage_error(const std::string &label,
                       const std::vector<std::uint8_t> &observed,
                       const std::vector<std::uint8_t> &expected) {
  if (observed.size() != expected.size())
    throw std::logic_error(label + " stage byte-size mismatch: observed=" +
                           std::to_string(observed.size()) +
                           " expected=" + std::to_string(expected.size()));
  double absolute_error = 0.0;
  float maximum_error = 0.0f;
  std::size_t maximum_index = 0;
  std::size_t mismatches = 0;
  std::size_t printed = 0;
  std::array<double, 8> block_absolute_error{};
  std::array<float, 8> block_maximum_error{};
  std::array<std::size_t, 8> block_mismatches{};
  for (std::size_t index = 0; index < observed.size() / 2; ++index) {
    const float actual = bf16_at(observed, index);
    const float golden = bf16_at(expected, index);
    const float error = std::fabs(actual - golden);
    const std::size_t columns = observed.size() / 2 == 32 * 256 ? 256 : 1536;
    const std::size_t block = (index % columns) / 32;
    absolute_error += error;
    if (block < block_absolute_error.size()) {
      block_absolute_error[block] += error;
      block_maximum_error[block] =
          std::max(block_maximum_error[block], error);
    }
    if (error > maximum_error) {
      maximum_error = error;
      maximum_index = index;
    }
    if (error > 0.25f + 0.05f * std::fabs(golden)) {
      ++mismatches;
      if (block < block_mismatches.size())
        ++block_mismatches[block];
      if (printed++ < 8)
        std::cout << "Qwen stage mismatch: name=" << label
                  << " index=" << index << " actual=" << actual
                  << " golden=" << golden << " error=" << error << '\n';
    }
  }
  std::cout << "Qwen stage summary: name=" << label
            << " values=" << observed.size() / 2
            << " mismatches=" << mismatches
            << " mae=" << absolute_error / (observed.size() / 2)
            << " max_error=" << maximum_error
            << " max_error_index=" << maximum_index << '\n';
  if (observed.size() / 2 == 32 * 256) {
    for (std::size_t index = 0; index < 32; ++index)
      std::cout << "Qwen value sample: index=" << index
                << " actual=" << bf16_at(observed, index)
                << " golden=" << bf16_at(expected, index) << '\n';
    for (std::size_t block = 0; block < block_absolute_error.size(); ++block)
      std::cout << "Qwen value block: block=" << block
                << " mismatches=" << block_mismatches[block]
                << " mae=" << block_absolute_error[block] / (32 * 32)
                << " max_error=" << block_maximum_error[block] << '\n';
  }
}

std::vector<std::uint8_t> download_value_stage(
    const ftlpu::TspSliceSystem &system, std::size_t bank,
    bool source_hemisphere = false) {
  constexpr std::size_t kSeqLen = 32;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kHeadBlock = 32;
  constexpr std::size_t kTokenLanes = 8;
  constexpr std::size_t kValueBaseRow = 672;
  std::vector<std::uint8_t> result(kSeqLen * kKvHeads * kHeadDim * 2);
  for (std::size_t token = 0; token < kSeqLen; ++token) {
    const std::size_t token_lane = token % kTokenLanes;
    const std::size_t token_wave = token / kTokenLanes;
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto hemisphere = static_cast<ftlpu::Hemisphere>(head % 2);
      for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
        const std::size_t feature_block = feature / kHeadBlock;
        const auto physical_hemisphere = source_hemisphere
            ? static_cast<ftlpu::Hemisphere>(
                  ((head * (kHeadDim / kHeadBlock) + feature_block) % 4) / 2)
            : hemisphere;
        const std::size_t address = kValueBaseRow +
            (head * (kHeadDim / kHeadBlock) + feature_block) * 4 + token_wave;
        const std::size_t column = feature % kHeadBlock;
        const std::size_t logical =
            (token * kKvHeads * kHeadDim + head * kHeadDim + feature) * 2;
        result[logical] = system.read_mem_sram_lane_byte(
            physical_hemisphere, 2 * token_lane, bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
        result[logical + 1] = system.read_mem_sram_lane_byte(
            physical_hemisphere, 2 * token_lane + 1, bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> download_projection_staging(
    const ftlpu::TspSliceSystem &system, std::size_t bank) {
  constexpr std::size_t kRows = 32;
  constexpr std::size_t kColumns = 1536;
  constexpr std::size_t kBaseRow = 512;
  std::vector<std::uint8_t> result(kRows * kColumns * 2);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t column = 0; column < kColumns; ++column) {
      const std::size_t address = kBaseRow + (column / 32) * kRows + row;
      const std::size_t logical = (row * kColumns + column) * 2;
      result[logical] = system.read_mem_sram_lane_byte(
          ftlpu::Hemisphere::East, 18, bank,
          (column % 32) / ftlpu::hw::kLanesPerTile, address,
          column % ftlpu::hw::kLanesPerTile);
      result[logical + 1] = system.read_mem_sram_lane_byte(
          ftlpu::Hemisphere::East, 19, bank,
          (column % 32) / ftlpu::hw::kLanesPerTile, address,
          column % ftlpu::hw::kLanesPerTile);
    }
  }
  return result;
}

std::vector<std::uint8_t> download_context_stage(
    const ftlpu::TspSliceSystem &system, const BinaryBinding &binding,
    int hemisphere_mode = 0) {
  constexpr std::size_t kSeqLen = 32;
  constexpr std::size_t kQueryHeads = 12;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kHeadBlock = 32;
  constexpr std::size_t kHidden = kQueryHeads * kHeadDim;
  const bool head_block_packed =
      binding.layout == BindingLayout::Fp16HeadBlockPacked;
  if ((head_block_packed && binding.slices.size() < 4)
      || (!head_block_packed
          && binding.slices.size() < 2 * (kHeadDim / kHeadBlock)))
    throw std::logic_error("Qwen context binding has too few slices");
  std::vector<std::uint8_t> result(kSeqLen * kHidden * 2);
  for (std::size_t token = 0; token < kSeqLen; ++token) {
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
      const auto hemisphere = hemisphere_mode == 1
          ? static_cast<ftlpu::Hemisphere>((head / 6) % 2)
          : hemisphere_mode == 2 ? ftlpu::Hemisphere::West
                                 : ftlpu::Hemisphere::East;
      for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
        const std::size_t head_block = feature / kHeadBlock;
        const std::size_t address = binding.base_row
            + (head_block_packed
                ? (head * (kHeadDim / kHeadBlock) + head_block) * kSeqLen
                : head * kSeqLen)
            + token;
        const std::size_t slice_base =
            head_block_packed
                ? ((head / (kQueryHeads / 2)) % 2) * 2
                : 2 * head_block;
        const std::size_t column = feature % kHeadBlock;
        const std::size_t logical =
            (token * kHidden + head * kHeadDim + feature) * 2;
        result[logical] = system.read_mem_sram_lane_byte(
            hemisphere, binding.slices[slice_base], binding.bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
        result[logical + 1] = system.read_mem_sram_lane_byte(
            hemisphere, binding.slices[slice_base + 1], binding.bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> download_query_stage(
    const ftlpu::TspSliceSystem &system, std::size_t base_bank) {
  constexpr std::size_t kSeqLen = 32;
  constexpr std::size_t kQueryHeads = 12;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kBlockColumns = 32;
  constexpr std::size_t kQueryBaseRow = 576;
  constexpr std::size_t kHidden = kQueryHeads * kHeadDim;
  std::vector<std::uint8_t> result(kSeqLen * kHidden * 2);
  for (std::size_t token = 0; token < kSeqLen; ++token) {
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
      const auto hemisphere =
          static_cast<ftlpu::Hemisphere>((head / 6) % 2);
      for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
        const std::size_t reduction = feature / kBlockColumns;
        const std::size_t column = feature % kBlockColumns;
        const std::size_t bank = (base_bank + reduction / 2) % 2;
        const std::size_t address = kQueryBaseRow +
            (head * 2 + reduction % 2) * 4 + token / 8;
        const std::size_t slice = 2 * (token % 8);
        const std::size_t logical =
            (token * kHidden + head * kHeadDim + feature) * 2;
        result[logical] = system.read_mem_sram_lane_byte(
            hemisphere, slice, bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
        result[logical + 1] = system.read_mem_sram_lane_byte(
            hemisphere, slice + 1, bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> download_key_stage(
    const ftlpu::TspSliceSystem &system, std::size_t bank,
    bool east_only = false) {
  constexpr std::size_t kSeqLen = 32;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kBlockColumns = 32;
  constexpr std::size_t kColumns = kKvHeads * kHeadDim;
  std::vector<std::uint8_t> result(kSeqLen * kColumns * 2);
  for (std::size_t token = 0; token < kSeqLen; ++token) {
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto hemisphere = east_only
          ? ftlpu::Hemisphere::East
          : static_cast<ftlpu::Hemisphere>(head % 2);
      for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
        const std::size_t reduction = feature / kBlockColumns;
        const std::size_t column = feature % kBlockColumns;
        const std::size_t address =
            (head * 2 + reduction % 2) * kSeqLen + token;
        const std::size_t slice = 16 + 2 * (reduction / 2);
        const std::size_t logical =
            (token * kColumns + head * kHeadDim + feature) * 2;
        result[logical] = system.read_mem_sram_lane_byte(
            hemisphere, slice, bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
        result[logical + 1] = system.read_mem_sram_lane_byte(
            hemisphere, slice + 1, bank,
            column / ftlpu::hw::kLanesPerTile, address,
            column % ftlpu::hw::kLanesPerTile);
      }
    }
  }
  return result;
}

std::vector<std::uint8_t> download_distributed16_stage(
    const ftlpu::TspSliceSystem &system, std::size_t bank,
    ftlpu::Hemisphere hemisphere) {
  constexpr std::size_t kRows = 32;
  constexpr std::size_t kColumns = 1536;
  constexpr std::size_t kBlockColumns = 32;
  std::vector<std::uint8_t> result(kRows * kColumns * 2);
  for (std::size_t row = 0; row < kRows; ++row) {
    const std::size_t token_wave = row / 8;
    const std::size_t token_lane = row % 8;
    for (std::size_t column = 0; column < kColumns; ++column) {
      const std::size_t hidden_block = column / kBlockColumns;
      const std::size_t feature_wave = (column % kBlockColumns) / 8;
      const std::size_t feature_lane = column % 8;
      const std::size_t address = hidden_block * 4 + token_wave;
      const std::size_t logical = (row * kColumns + column) * 2;
      result[logical] = system.read_mem_sram_lane_byte(
          hemisphere, 2 * token_lane, bank, feature_wave, address,
          feature_lane);
      result[logical + 1] = system.read_mem_sram_lane_byte(
          hemisphere, 2 * token_lane + 1, bank, feature_wave, address,
          feature_lane);
    }
  }
  return result;
}

std::vector<std::uint8_t> download_ffn_projection_stage(
    const ftlpu::TspSliceSystem &system, bool up_projection) {
  constexpr std::size_t kRows = 32;
  constexpr std::size_t kColumns = 8960;
  constexpr std::size_t kBlockColumns = 32;
  constexpr std::size_t kPairsPerTempGroup = 64;
  constexpr std::size_t kUpSliceBase = 8;
  std::vector<std::uint8_t> result(kRows * kColumns * 2);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t column = 0; column < kColumns; ++column) {
      const std::size_t block = column / kBlockColumns;
      const std::size_t block_in_quad = block % 4;
      const std::size_t hemisphere = block_in_quad / 2;
      const std::size_t pair = (block / 4) * 2 + block_in_quad % 2;
      const std::size_t temp_group = pair / kPairsPerTempGroup;
      const std::size_t slice = (up_projection ? kUpSliceBase : 0) +
                                2 * temp_group;
      const std::size_t address =
          (pair % kPairsPerTempGroup) * kRows + row;
      const std::size_t tile = (column % kBlockColumns) / 8;
      const std::size_t lane = column % 8;
      const std::size_t logical = (row * kColumns + column) * 2;
      const auto owner = hemisphere == 0 ? ftlpu::Hemisphere::East
                                         : ftlpu::Hemisphere::West;
      result[logical] =
          system.read_mem_sram_lane_byte(owner, slice, 1, tile, address, lane);
      result[logical + 1] = system.read_mem_sram_lane_byte(
          owner, slice + 1, 1, tile, address, lane);
    }
  }
  return result;
}

std::vector<std::uint8_t> download_key_staging(
    const ftlpu::TspSliceSystem &system, bool east = false) {
  constexpr std::size_t kSeqLen = 32;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kBlockColumns = 32;
  std::vector<std::uint8_t> result(kSeqLen * kHeadDim * 2);
  for (std::size_t token = 0; token < kSeqLen; ++token) {
    const std::size_t token_lane = token % 8;
    const std::size_t token_row = token / 8;
    for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
      const std::size_t head_block = feature / kBlockColumns;
      const std::size_t column = feature % kBlockColumns;
      const std::size_t address = head_block * kSeqLen + token_row;
      const std::size_t slice =
          (2 * token_lane + 2 * head_block) % 16;
      const std::size_t logical = (token * kHeadDim + feature) * 2;
      result[logical] = system.read_mem_sram_lane_byte(
          east ? ftlpu::Hemisphere::East : ftlpu::Hemisphere::West, slice, 0,
          column / ftlpu::hw::kLanesPerTile, address,
          column % ftlpu::hw::kLanesPerTile);
      result[logical + 1] = system.read_mem_sram_lane_byte(
          east ? ftlpu::Hemisphere::East : ftlpu::Hemisphere::West,
          slice + 1, 0,
          column / ftlpu::hw::kLanesPerTile, address,
          column % ftlpu::hw::kLanesPerTile);
    }
  }
  return result;
}

std::vector<std::uint8_t> download_key_products(
    const ftlpu::TspSliceSystem &system, bool west = false) {
  constexpr std::size_t kSeqLen = 32;
  constexpr std::size_t kPairBlocks = 2;
  constexpr std::size_t kProducts = 4;
  constexpr std::size_t kColumns = 32;
  constexpr std::size_t kProductBaseSlice = 20;
  constexpr std::size_t kGlobalHead = 13;
  constexpr std::size_t kRowsPerVector = 16;
  std::vector<std::uint8_t> result(
      kSeqLen * kPairBlocks * kProducts * kColumns * 2);
  const auto hemisphere =
      west ? ftlpu::Hemisphere::West : ftlpu::Hemisphere::East;
  for (std::size_t token = 0; token < kSeqLen; ++token) {
    for (std::size_t pair = 0; pair < kPairBlocks; ++pair) {
      for (std::size_t product = 0; product < kProducts; ++product) {
        const std::size_t address =
            ((kGlobalHead * kPairBlocks + pair) * kProducts + product) *
                kRowsPerVector +
            token / 2;
        const std::size_t slice =
            kProductBaseSlice + (token % 2) * 8 + product * 2;
        for (std::size_t column = 0; column < kColumns; ++column) {
          const std::size_t logical =
              (((token * kPairBlocks + pair) * kProducts + product) *
                    kColumns +
                column) *
              2;
          result[logical] = system.read_mem_sram_lane_byte(
              hemisphere, slice, 1,
              column / ftlpu::hw::kLanesPerTile, address,
              column % ftlpu::hw::kLanesPerTile);
          result[logical + 1] = system.read_mem_sram_lane_byte(
              hemisphere, slice + 1, 1,
              column / ftlpu::hw::kLanesPerTile, address,
              column % ftlpu::hw::kLanesPerTile);
        }
      }
    }
  }
  return result;
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 3)
    throw std::runtime_error(
        "usage: compiled_qwen_real_decoder_layer_runtime_test "
        "program.ftlpu fixture_dir");

  const std::filesystem::path fixture(argv[2]);
  BinaryProgram program = read_binary_program(std::filesystem::path(argv[1]));
  const std::uint32_t compiled_ddr_bandwidth =
      program.hardware.ddr_peak_bandwidth_mbytes_per_second;
  std::uint32_t runtime_ddr_bandwidth = compiled_ddr_bandwidth;
  if (program.weight_page_uses.empty())
    throw std::logic_error(
        "Qwen decoder binary has no intra-executable weight pages");
  const auto scales = read_quant_scales(fixture);

  ModelPackage package;
  package.model_name = "Qwen2.5-1.5B-layer0-seq32";
  package.architecture = "Qwen2ForCausalLM";
  package.tensors = {
      make_tensor(program, 2, "self_attn.q_proj.weight",
                  fixture / "query.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[0]}),
      make_tensor(program, 3, "self_attn.k_proj.weight", fixture / "key.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[1]}),
      make_tensor(program, 4, "self_attn.v_proj.weight",
                  fixture / "value.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[2]}),
      make_tensor(program, 5, "self_attn.o_proj.weight",
                  fixture / "output.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[3]}),
      make_tensor(program, 7, "mlp.gate_proj.weight", fixture / "gate.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[4]}),
      make_tensor(program, 8, "mlp.up_proj.weight", fixture / "up.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[5]}),
      make_tensor(program, 9, "mlp.down_proj.weight", fixture / "down.i8.bin",
                  ModelTensorEncoding::SymmetricPerTensorI8, {scales[6]}),
  };
  ModelWeightPage parameter_page;
  parameter_page.layer = 0;
  parameter_page.bank = find_binding(
      program, BindingAccess::Input, 1).bank;
  std::uint16_t next_stream = 0;
  append_packed_tensor(package, parameter_page, program, 1,
                       "input_layernorm.weight",
                       fixture / "input_layernorm.bf16.bin", next_stream);
  append_packed_tensor(
      package, parameter_page, program, 6, "post_attention_layernorm.weight",
      fixture / "post_attention_layernorm.bf16.bin", next_stream);
  package.weight_pages.push_back(std::move(parameter_page));

  const BinaryBinding &input = find_binding(program, BindingAccess::Input, 0);
  const BinaryBinding &output = find_binding(program, BindingAccess::Output, 0);
  package.values = {
      {"hidden.0", input.element_type, input.shape, true, false},
      {"hidden.1", output.element_type, output.shape, false, true},
  };
  package.executables.push_back({"decoder.layer0", std::move(program), {}});
  package.invocations.push_back(
      ModelInvocation{"decoder.layer0",
                      0,
                      {{0, "hidden.0"},
                       {1, "input_layernorm.weight"},
                       {2, "self_attn.q_proj.weight"},
                       {3, "self_attn.k_proj.weight"},
                       {4, "self_attn.v_proj.weight"},
                       {5, "self_attn.o_proj.weight"},
                       {6, "post_attention_layernorm.weight"},
                       {7, "mlp.gate_proj.weight"},
                       {8, "mlp.up_proj.weight"},
                       {9, "mlp.down_proj.weight"}},
                      {{0, "hidden.1"}},
                      {},
                      0});

  ftlpu::C2cDmaSystem system;
  ModelSession session(system);
  if (const char *bandwidth = std::getenv("FTLPU_DDR_BANDWIDTH_MBPS")) {
    const auto parsed = std::stoull(bandwidth);
    if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("invalid FTLPU_DDR_BANDWIDTH_MBPS");
    runtime_ddr_bandwidth = static_cast<std::uint32_t>(parsed);
    session.set_ddr_peak_bandwidth_mbytes_per_second(
        runtime_ddr_bandwidth);
  }
  session.load(std::move(package));
  const auto input_bytes = read_bytes(fixture / "input.bf16.bin");
  session.set_input("hidden.0", input_bytes);
  const char *trace_path = std::getenv("FTLPU_QWEN_PIPELINE_CSV");
  if (trace_path != nullptr)
    session.enable_execution_trace();
  session.run();

  if (trace_path != nullptr)
    session.write_execution_trace_csv(trace_path);

  if (const char *binding_text =
          std::getenv("FTLPU_TRACE_QWEN_WEIGHT_PAGE")) {
    const BinaryProgram &loaded_program =
        session.package().executables[0].program;
    const std::uint32_t binding_index =
        static_cast<std::uint32_t>(std::stoul(binding_text));
    const std::array<const char *, 10> fixture_names = {
        "input.bf16.bin", "input_layernorm.bf16.bin", "query.i8.bin",
        "key.i8.bin", "value.i8.bin", "output.i8.bin",
        "post_attention_layernorm.bf16.bin", "gate.i8.bin", "up.i8.bin",
        "down.i8.bin"};
    if (binding_index >= fixture_names.size())
      throw std::logic_error("invalid Qwen weight-page binding index");
    print_weight_page_residency_error(
        system.chip(),
        find_binding(loaded_program, BindingAccess::Input, binding_index),
        read_bytes(fixture / fixture_names[binding_index]),
        loaded_program.hardware);
  }

  if (const char *stage = std::getenv("FTLPU_TRACE_QWEN_STAGE")) {
    const BinaryProgram &loaded_program =
        session.package().executables[0].program;
    const std::size_t weight_bank =
        find_binding(loaded_program, BindingAccess::Input, 1).bank;
    const std::size_t scratch_bank =
        (weight_bank + 1) % loaded_program.hardware.banks_per_slice;
    const auto distributed_binding = std::find_if(
        loaded_program.bindings.begin(), loaded_program.bindings.end(),
        [](const BinaryBinding &binding) {
          return binding.access == BindingAccess::Internal &&
                 binding.layout == BindingLayout::Fp16MxmDistributed16 &&
                 binding.bank == 0 && binding.slices.size() == 16 &&
                 binding.shape == std::vector<std::uint64_t>{32, 1536};
        });
    if (distributed_binding == loaded_program.bindings.end())
      throw std::logic_error(
          "Qwen decoder has no distributed attention-stage binding");
    CModelRuntime observer(system.chip());
    const std::string stage_name(stage);
    if (stage_name == "all") {
      const auto capture = [&](const std::string &name,
                               std::vector<std::uint8_t> observed,
                               bool compare = true) {
        write_bytes(fixture / ("debug." + name + ".actual.bf16.bin"),
                    observed);
        if (compare)
          print_stage_error(
              name, observed,
              read_bytes(fixture / ("golden." + name + ".bf16.bin")));
        else
          std::cout << "Qwen stage capture: name=" << name
                    << " values=" << observed.size() / 2 << '\n';
      };
      capture("query", download_query_stage(system.chip(), scratch_bank));
      capture("key", download_key_stage(system.chip(), weight_bank));
      capture("value", download_value_stage(system.chip(), scratch_bank));
      capture("context", download_context_stage(
                             system.chip(), find_internal_binding_by_name(
                                                loaded_program,
                                                "attention.context")));
      capture("attention",
              observer.download_binding(find_earliest_internal_binding(
                  loaded_program, BindingLayout::Fp16PairPlanar)));
      capture("residual0",
              observer.download_binding(find_latest_internal_binding(
                  loaded_program, BindingLayout::Fp16MxmDistributed16, 1,
                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})));
      capture("norm1",
              observer.download_binding(find_latest_internal_binding(
                  loaded_program, BindingLayout::Fp16MxmDistributed16, 0,
                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})));
      capture("gate", download_ffn_projection_stage(system.chip(), false),
              false);
      capture("up", download_ffn_projection_stage(system.chip(), true),
              false);
      capture("swiglu",
              download_swiglu_stage(observer, *distributed_binding));
      capture("down",
              observer.download_binding(find_latest_internal_binding(
                  loaded_program, BindingLayout::Fp16PairPlanar, 0,
                  {16, 17, 18, 19})));
    } else {
      const auto capture_qkv = [&]() {
        auto query = download_query_stage(system.chip(), scratch_bank);
        auto key = download_key_stage(system.chip(), weight_bank);
        auto value = download_value_stage(system.chip(), scratch_bank);
        query.insert(query.end(), key.begin(), key.end());
        query.insert(query.end(), value.begin(), value.end());
        return query;
      };
      const auto capture_resnorm = [&]() {
        auto residual = download_distributed16_stage(
            system.chip(), scratch_bank, ftlpu::Hemisphere::East);
        auto norm = download_distributed16_stage(
            system.chip(), weight_bank, ftlpu::Hemisphere::East);
        residual.insert(residual.end(), norm.begin(), norm.end());
        return residual;
      };
      const auto observed = stage_name == "resnorm"
                              ? capture_resnorm()
                          : stage_name == "input"
                              ? observer.download_binding(find_binding(
                                    loaded_program, BindingAccess::Input, 0))
                          : stage_name == "qkv"
                              ? capture_qkv()
                          : stage_name == "attention"
                              ? observer.download_binding(
                                    find_earliest_internal_binding(
                                        loaded_program,
                                        BindingLayout::Fp16PairPlanar))
                          : stage_name == "residual0"
                              ? download_distributed16_stage(
                                    system.chip(), scratch_bank,
                                    ftlpu::Hemisphere::East)
                          : stage_name == "residual0_west"
                              ? download_distributed16_stage(
                                    system.chip(), scratch_bank,
                                    ftlpu::Hemisphere::West)
                          : stage_name == "norm1"
                              ? download_distributed16_stage(
                                    system.chip(), weight_bank,
                                    ftlpu::Hemisphere::East)
                          : stage_name == "down"
                              ? observer.download_binding(
                                    find_latest_internal_binding(
                                        loaded_program,
                                        BindingLayout::Fp16PairPlanar, 0,
                                        {16, 17, 18, 19}))
                          : stage_name == "swiglu"
                              ? download_swiglu_stage(
                                    observer, *distributed_binding)
                          : stage_name == "gate"
                              ? download_ffn_projection_stage(
                                    system.chip(), false)
                          : stage_name == "up"
                              ? download_ffn_projection_stage(
                                    system.chip(), true)
                          : stage_name == "value"
                              ? download_value_stage(
                                    system.chip(), scratch_bank)
                          : stage_name == "value_source"
                              ? download_value_stage(
                                    system.chip(), scratch_bank, true)
                          : stage_name == "context"
                              ? download_context_stage(
                                    system.chip(),
                                    find_internal_binding_by_name(
                                        loaded_program, "attention.context"))
                          : stage_name == "context_source"
                              ? download_context_stage(
                                    system.chip(),
                                    find_internal_binding_by_name(
                                        loaded_program, "attention.context"),
                                    1)
                          : stage_name == "context_west"
                              ? download_context_stage(
                                    system.chip(),
                                    find_internal_binding_by_name(
                                        loaded_program, "attention.context"),
                                    2)
                          : stage_name == "norm0"
                              ? download_distributed16_stage(
                                    system.chip(), weight_bank,
                                    ftlpu::Hemisphere::East)
                          : stage_name == "query"
                              ? download_query_stage(
                                    system.chip(), scratch_bank)
                          : stage_name == "key"
                              ? download_key_stage(
                                    system.chip(), weight_bank)
                          : stage_name == "key_east"
                              ? download_key_stage(
                                    system.chip(), weight_bank, true)
                          : stage_name == "key_staging"
                              ? download_key_staging(system.chip())
                          : stage_name == "key_staging_east"
                              ? download_key_staging(system.chip(), true)
                          : stage_name == "key_products"
                              ? download_key_products(system.chip())
                          : stage_name == "key_products_west"
                              ? download_key_products(system.chip(), true)
                          : stage_name == "staging"
                              ? download_projection_staging(
                                    system.chip(), weight_bank)
                              : observer.download_binding(*distributed_binding);
    write_bytes(fixture / ("debug." + stage_name + ".actual.bf16.bin"),
                observed);
    if (stage_name == "gate" || stage_name == "up") {
      if (stage_name == "gate") {
        write_bytes(fixture / "debug.up.actual.bf16.bin",
                    download_ffn_projection_stage(system.chip(), true));
      }
      std::cout << "Qwen stage capture: name=" << stage_name
                << " values=" << observed.size() / 2 << '\n';
    } else {
      const auto expected = read_bytes(fixture /
        (stage_name == "input" ? "input.bf16.bin"
         : stage_name == "staging" || stage_name == "norm0"
             ? "golden.norm0.bf16.bin"
         : stage_name == "value_source" ? "golden.value.bf16.bin"
         : stage_name == "context_source" || stage_name == "context_west"
             ? "golden.context.bf16.bin"
         : stage_name == "residual0_west" ? "golden.residual0.bf16.bin"
         : stage_name == "key_east" ? "golden.key.bf16.bin"
         : stage_name == "key_staging_east"
             ? "golden.key_staging.bf16.bin"
         : stage_name == "key_products_west"
             ? "golden.key_products.bf16.bin"
                                 : "golden." + stage_name + ".bf16.bin"));
      print_stage_error(stage, observed, expected);
    }
    }
  }

  if (std::getenv("FTLPU_TRACE_QWEN_OPERANDS") != nullptr) {
    const BinaryProgram &loaded_program =
        session.package().executables[0].program;
    const auto pair_binding = std::find_if(
        loaded_program.bindings.begin(), loaded_program.bindings.end(),
        [](const BinaryBinding &binding) {
          return binding.access == BindingAccess::Internal
              && binding.layout == BindingLayout::Fp16PairPlanar
              && binding.bank == 0 && binding.slices.size() == 4
              && binding.slices[0] == 16 && binding.slices[1] == 17
              && binding.slices[2] == 18 && binding.slices[3] == 19
              && binding.shape == std::vector<std::uint64_t>{32, 1536};
        });
    if (pair_binding == loaded_program.bindings.end())
      throw std::logic_error("Qwen decoder has no final pair-planar operand");
    CModelRuntime observer(system.chip());
    const auto down = observer.download_binding(*pair_binding);
    BinaryBinding residual_binding = *pair_binding;
    residual_binding.bank = 1;
    const auto residual = observer.download_binding(residual_binding);
    print_pair_planar_blocks("residual", residual, 32, 1536);
    print_pair_planar_blocks("down", down, 32, 1536);
  }

  const auto &actual = session.value("hidden.1");
  const auto golden = read_bytes(fixture / "golden.bf16.bin");
  if (actual.size() != golden.size())
    throw std::logic_error("Qwen decoder output size differs from golden");

  float maximum_error = 0.0f;
  std::size_t maximum_index = 0;
  double sum_absolute_error = 0.0;
  double sum_squared_error = 0.0;
  std::size_t nonzero = 0;
  std::size_t mismatches = 0;
  constexpr std::size_t kHidden = 1536;
  constexpr std::size_t kBlockColumns = 32;
  constexpr std::size_t kHiddenBlocks = kHidden / kBlockColumns;
  std::array<std::size_t, kHiddenBlocks> block_mismatches{};
  std::array<double, kHiddenBlocks> block_absolute_error{};
  std::array<float, kHiddenBlocks> block_maximum_error{};
  std::vector<float> errors;
  errors.reserve(actual.size() / 2);
  for (std::size_t index = 0; index < actual.size() / 2; ++index) {
    const float observed = bf16_at(actual, index);
    const float expected = bf16_at(golden, index);
    if (!std::isfinite(observed) || !std::isfinite(expected))
      throw std::logic_error("Qwen decoder produced a non-finite value at " +
                             std::to_string(index));
    const float error = std::fabs(observed - expected);
    const std::size_t hidden_block = (index % kHidden) / kBlockColumns;
    block_absolute_error[hidden_block] += error;
    block_maximum_error[hidden_block] =
        std::max(block_maximum_error[hidden_block], error);
    errors.push_back(error);
    sum_absolute_error += error;
    sum_squared_error += static_cast<double>(error) * error;
    if (error > maximum_error) {
      maximum_error = error;
      maximum_index = index;
    }
    if (std::fabs(observed) > 1.0e-6f)
      ++nonzero;
    if (error > 0.25f + 0.05f * std::fabs(expected)) {
      ++block_mismatches[hidden_block];
      if (mismatches < 12)
        std::cerr << "mismatch index=" << index << " actual=" << observed
                  << " golden=" << expected << " error=" << error << '\n';
      ++mismatches;
    }
  }
  if (nonzero == 0)
    throw std::logic_error("Qwen decoder produced an all-zero output");
  std::sort(errors.begin(), errors.end());
  const auto values = actual.size() / 2;
  const double mean_absolute_error =
      sum_absolute_error / static_cast<double>(values);
  const double root_mean_squared_error =
      std::sqrt(sum_squared_error / static_cast<double>(values));
  const float p99 = errors[static_cast<std::size_t>(
      0.99 * static_cast<double>(errors.size() - 1))];
  std::cout << "Qwen decoder numerical summary: values=" << values
            << " mismatches=" << mismatches << " mae=" << mean_absolute_error
            << " rmse=" << root_mean_squared_error << " p99=" << p99
            << " max_error=" << maximum_error
            << " max_error_index=" << maximum_index << '\n';
  if (mismatches != 0) {
    constexpr double kValuesPerBlock = 32.0 * kBlockColumns;
    for (std::size_t block = 0; block < kHiddenBlocks; ++block) {
      std::cout << "Qwen decoder block summary: block=" << block
                << " east_source="
                << (((block / 2) % 2) == 0 ? "mirrored" : "direct")
                << " mismatches=" << block_mismatches[block]
                << " mae=" << block_absolute_error[block] / kValuesPerBlock
                << " max_error=" << block_maximum_error[block] << '\n';
    }
  }
  const double mismatch_fraction =
      static_cast<double>(mismatches) / static_cast<double>(values);
  if (mismatch_fraction > 0.001 || p99 > 0.5f
      || mean_absolute_error > 0.075 || maximum_error > 32.0f)
    throw std::logic_error(
        "Qwen decoder golden mismatch count=" +
        std::to_string(mismatches) +
        " fraction=" + std::to_string(mismatch_fraction));

  const auto &stats = session.stats();
  std::cout << "Qwen2.5-1.5B layer0 real decoder passed: values=" << values
            << " pages="
            << session.package().executables[0].program.weight_page_uses.size()
            << " cycles="
            << session.package().executables[0].program.max_cycle + 64
            << " resident_uploads=" << stats.resident_uploads
            << " host_uploads=" << stats.host_uploads
            << " host_downloads=" << stats.host_downloads
            << " compiled_ddr_mbytes=" << compiled_ddr_bandwidth
            << " runtime_ddr_mbytes=" << runtime_ddr_bandwidth
            << " runtime_page_wait_cycles="
            << stats.weight_page_runtime_wait_cycles
            << " nonzero=" << nonzero << " max_error=" << maximum_error << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "compiled_qwen_real_decoder_layer_runtime_test failed: "
            << error.what() << '\n';
  return 1;
}
