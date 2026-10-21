#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"
#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <utility>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

void require(bool condition, const std::string& message);

void test_attention16_paged_weight()
{
    ExecutableHardwareConfig hardware;
    hardware.bytes_per_word = 16;
    hardware.mxm_rows = 16;
    hardware.mxm_columns = 16;
    hardware.sram_depth_rows = 1024;
    hardware.words_per_bank = 1024;
    BinaryBinding weight;
    weight.element_type = BindingElementType::I8;
    weight.layout = BindingLayout::W8A16AttentionWeightStriped;
    weight.role = "weight";
    weight.shape = {32, 128};
    weight.byte_size = 32 * 128;
    weight.address_stride = 1;
    weight.slices = {20, 21, 22, 23, 24, 25, 26, 27};
    weight.paged_weight = true;
    weight.page_count = 2;
    weight.page_rows = 1024;
    weight.page_granularity = 1;
    weight.page_items_per_slice_group = 1;
    weight.page_bank_count = 2;
    weight.page_storage_slices = weight.slices;
    weight.page_banks = {0, 1};
    weight.page_slice_group_bases = {0, 0};
    weight.page_slice_group_counts = {1, 1};
    weight.page_base_rows = {0, 0};
    weight.page_row_counts = {8, 8};
    std::vector<std::uint8_t> data(weight.byte_size);
    for (std::size_t k = 0; k < 32; ++k)
        for (std::size_t n = 0; n < 128; ++n)
            data[k * 128 + n] = static_cast<std::uint8_t>(
                (k * 29 + n * 7 + k * n) & 0xff);
    for (std::uint32_t page = 0; page < 2; ++page) {
        const auto image = pack_weight_binding_page(
            weight, page, data, hardware);
        require(!image.segments.empty() && image.data.size() % 16 == 0,
            "16-byte attention page has no physical vectors");
        require(image.segments.size() == 16
                && std::all_of(image.segments.begin(), image.segments.end(),
                    [](const auto& segment) {
                        return segment.base_row == 0
                            && segment.vector_count == 8;
                    }),
            "16-byte attention page did not pack each slice contiguously");
        for (std::size_t k = 0; k < 32; ++k)
            for (std::size_t localN = 0; localN < 64; ++localN) {
                const std::size_t n = page * 64 + localN;
                const std::uint16_t hemisphere =
                    static_cast<std::uint16_t>((n / 32) % 2);
                const std::uint16_t slice = weight.slices[n % 8];
                const std::uint32_t row = static_cast<std::uint32_t>(
                    (k / 16) * 4 + (n % 32) / 16 * 2
                    + 1 - (n % 16) / 8);
                const auto segment = std::find_if(image.segments.begin(),
                    image.segments.end(), [&](const auto& candidate) {
                        return candidate.hemisphere == hemisphere
                            && candidate.slice == slice
                            && row >= candidate.base_row
                            && row < candidate.base_row
                                + candidate.vector_count;
                    });
                require(segment != image.segments.end(),
                    "16-byte attention weight has no physical row");
                const auto offset = segment->byte_offset
                    + (row - segment->base_row) * 16 + k % 16;
                require(image.data.at(offset) == data[k * 128 + n],
                    "16-byte attention page changed a weight byte");
            }
    }
}

void test_channel_tiled_ffn_down_page()
{
    ExecutableHardwareConfig hardware;
    hardware.bytes_per_word = 16;
    hardware.mxm_rows = 16;
    hardware.mxm_columns = 16;
    hardware.tile_rows = 2;
    hardware.sram_depth_rows = 1024;
    hardware.words_per_bank = 1024;
    BinaryBinding weight;
    weight.element_type = BindingElementType::I8;
    weight.layout = BindingLayout::W8A16MxmWeightWaveStriped;
    weight.role = "weight";
    // A channel-tiled Down matrix can be wider than it is tall.
    weight.shape = {64, 128};
    weight.byte_size = 64 * 128;
    weight.address_stride = 1;
    weight.slices = {20, 21, 22, 23, 24, 25, 26, 27};
    weight.paged_weight = true;
    weight.page_count = 2;
    weight.page_rows = 1024;
    weight.page_granularity = 8;
    weight.page_items_per_slice_group = 8;
    weight.page_bank_count = 2;
    weight.page_storage_slices = weight.slices;
    weight.page_banks = {0, 1};
    weight.page_slice_group_bases = {0, 0};
    weight.page_slice_group_counts = {1, 1};
    weight.page_base_rows = {0, 0};
    weight.page_row_counts = {16, 16};
    std::vector<std::uint8_t> data(weight.byte_size);
    for (std::size_t k = 0; k < 64; ++k)
        for (std::size_t n = 0; n < 128; ++n)
            data[k * 128 + n] = static_cast<std::uint8_t>(
                (k * 31 + n * 7) & 0xff);
    for (std::uint32_t page = 0; page < 2; ++page) {
        const auto image = pack_weight_binding_page(
            weight, page, data, hardware);
        require(image.segments.size() == 16,
            "channel-tiled Down page has the wrong slice count");
        for (std::size_t k = 0; k < 64; ++k)
            for (std::size_t localN = 0; localN < 64; ++localN) {
                const std::size_t n = page * 64 + localN;
                const std::uint16_t hemisphere =
                    static_cast<std::uint16_t>((n / 32) % 2);
                const std::uint16_t slice = weight.slices[n % 8];
                const std::uint32_t row = static_cast<std::uint32_t>(
                    (k / 16) * 4 + ((n % 32) / 16) * 2
                    + 1 - ((n % 16) / 8));
                const auto segment = std::find_if(image.segments.begin(),
                    image.segments.end(), [&](const auto& candidate) {
                        return candidate.hemisphere == hemisphere
                            && candidate.slice == slice
                            && row >= candidate.base_row
                            && row < candidate.base_row
                                + candidate.vector_count;
                    });
                require(segment != image.segments.end(),
                    "channel-tiled Down weight has no physical row");
                const auto offset = segment->byte_offset
                    + (row - segment->base_row) * 16 + k % 16;
                require(image.data.at(offset) == data[k * 128 + n],
                    "channel-tiled Down page changed a weight byte");
            }
    }
}

void test_native4_16_paged_weight()
{
    ExecutableHardwareConfig hardware;
    hardware.bytes_per_word = 16;
    hardware.mxm_rows = 16;
    hardware.mxm_columns = 16;
    hardware.sram_depth_rows = 1024;
    hardware.words_per_bank = 1024;
    BinaryBinding weight;
    weight.element_type = BindingElementType::I8;
    weight.layout = BindingLayout::W8A16Native4Weight;
    weight.role = "weight";
    weight.shape = {32, 64};
    weight.byte_size = 32 * 64;
    weight.address_stride = 1;
    for (std::uint16_t slice = 20; slice < 36; ++slice)
        weight.slices.push_back(slice);
    weight.paged_weight = true;
    weight.page_count = 2;
    weight.page_rows = 1024;
    weight.page_granularity = 1;
    weight.page_items_per_slice_group = 1;
    weight.page_bank_count = 2;
    weight.page_storage_slices = weight.slices;
    weight.page_banks = {0, 1};
    weight.page_slice_group_bases = {0, 0};
    weight.page_slice_group_counts = {1, 1};
    weight.page_base_rows = {0, 0};
    weight.page_row_counts = {2, 2};
    std::vector<std::uint8_t> data(weight.byte_size);
    for (std::size_t k = 0; k < 32; ++k)
        for (std::size_t n = 0; n < 64; ++n)
            data[k * 64 + n] = static_cast<std::uint8_t>(
                (k * 37 + n * 13 + k * n) & 0xff);
    for (std::uint32_t page = 0; page < 2; ++page) {
        const auto image = pack_weight_binding_page(
            weight, page, data, hardware);
        require(image.segments.size() == 32
                && std::all_of(image.segments.begin(), image.segments.end(),
                    [](const auto& segment) {
                        return segment.base_row == 0
                            && segment.vector_count == 2;
                    }),
            "16-byte Native4 page did not pack each slice contiguously");
        for (std::size_t k = 0; k < 32; ++k)
            for (std::size_t localN = 0; localN < 32; ++localN) {
                const std::size_t n = page * 32 + localN;
                const std::uint16_t hemisphere =
                    static_cast<std::uint16_t>((n / 16) % 2);
                const std::uint16_t slice = weight.slices[n % 16];
                const std::uint32_t row =
                    static_cast<std::uint32_t>(k / 16);
                const auto segment = std::find_if(image.segments.begin(),
                    image.segments.end(), [&](const auto& candidate) {
                        return candidate.hemisphere == hemisphere
                            && candidate.slice == slice;
                    });
                require(segment != image.segments.end(),
                    "16-byte Native4 weight has no physical segment");
                const auto offset = segment->byte_offset + row * 16 + k % 16;
                require(image.data.at(offset) == data[k * 64 + n],
                    "16-byte Native4 page changed a weight byte");
            }
    }
}

void test_projection_bias16_eight_slices()
{
    ExecutableHardwareConfig hardware;
    hardware.bytes_per_word = 16;
    hardware.mxm_rows = 16;
    hardware.mxm_columns = 16;
    hardware.sram_depth_rows = 1024;
    hardware.words_per_bank = 1024;
    BinaryBinding bias;
    bias.element_type = BindingElementType::BF16;
    bias.layout = BindingLayout::Fp16ProjectionBiasX4;
    bias.shape = {128};
    bias.byte_size = 256;
    bias.instruction_count = 2;
    bias.address_stride = 1;
    bias.base_row = 40;
    bias.hemisphere_mask = 3;
    bias.slices = {8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<std::uint8_t> data(bias.byte_size);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<std::uint8_t>((i * 37 + i / 5) & 0xff);
    const auto image = pack_binding_image(bias, data, hardware);
    for (std::uint16_t hemisphere = 0; hemisphere < 2; ++hemisphere)
        for (std::size_t column = 0; column < 128; ++column) {
            const std::size_t block = column / 16;
            const std::size_t pair = (block / 2) % 4;
            const std::uint32_t row = 40 + block % 2;
            for (std::size_t byte = 0; byte < 2; ++byte) {
                const std::uint16_t slice = bias.slices[2 * pair + byte];
                const auto segment = std::find_if(image.segments.begin(),
                    image.segments.end(), [&](const auto& candidate) {
                        return candidate.hemisphere == hemisphere
                            && candidate.slice == slice
                            && row >= candidate.base_row
                            && row < candidate.base_row + candidate.vector_count;
                    });
                require(segment != image.segments.end(),
                    "16-column projection bias has no physical row");
                require(image.data.at(segment->byte_offset
                            + (row - segment->base_row) * 16 + column % 16)
                        == data[column * 2 + byte],
                    "16-column projection bias changed a byte");
            }
        }
}

void test_planar_transport(bool includeKernel = true)
{
    for (const auto layout : {BindingLayout::Fp16PairPlanar,
             BindingLayout::Fp32QuadPlanar,
             BindingLayout::Fp16ScalarBroadcastPlanar,
             BindingLayout::Fp32ScalarBroadcastPlanar,
             BindingLayout::Fp16DepthwiseKernel4}) {
        const bool fp32 = layout == BindingLayout::Fp32QuadPlanar
            || layout == BindingLayout::Fp32ScalarBroadcastPlanar;
        const bool scalar = layout == BindingLayout::Fp16ScalarBroadcastPlanar
            || layout == BindingLayout::Fp32ScalarBroadcastPlanar;
        const bool kernel = layout == BindingLayout::Fp16DepthwiseKernel4;
        if (kernel && !includeKernel) continue;
        const std::size_t width = ftlpu::hw::kPhysicalVectorBytes;
        BinaryBinding b;
        b.access = BindingAccess::Input;
        b.element_type = fp32 ? BindingElementType::F32 : BindingElementType::BF16;
        b.layout = layout;
        b.shape = kernel ? std::vector<std::uint64_t>{64, 4}
                         : std::vector<std::uint64_t>{3, 2 * width};
        b.byte_size = b.shape[0] * b.shape[1] * (fp32 ? 4 : 2);
        b.base_row = 19;
        b.bank = 1;
        b.hemisphere_mask = 2;
        b.address_stride = 1;
        b.instruction_count = kernel ? 8 : scalar ? 6 * width : 6;
        for (std::uint16_t i = 0; i < (kernel ? 8 : fp32 ? 4 : 2); ++i)
            b.slices.push_back(32 + i);
        std::vector<std::uint8_t> bytes(b.byte_size);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<std::uint8_t>(i * 37 + i / 7);
        BinaryProgram p;
        p.bindings.push_back(b);
        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        CModelRuntime runtime(*system);
        runtime.load(p);
        runtime.upload_input(0, bytes);
        const auto image = pack_binding_image(b, bytes, p.hardware);
        for (const auto& segment : image.segments)
            for (std::size_t row = 0; row < segment.vector_count; ++row)
                for (std::size_t lane = 0; lane < width; ++lane)
                    require(image.data[segment.byte_offset + row * width + lane]
                        == system->read_mem_sram_lane_byte(
                            static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                            segment.slice, b.bank, lane / 8,
                            segment.base_row + row, lane % 8),
                        "planar C2C packing differs from native upload");
        if (!kernel)
            require(unpack_binding_image(b, image, p.hardware) == bytes,
                "planar C2C output failed bitwise roundtrip");
    }
}

BinaryBinding make_weight(std::uint16_t bank)
{
    BinaryBinding binding;
    binding.index = 1;
    binding.access = BindingAccess::Input;
    binding.element_type = BindingElementType::I8;
    binding.layout = BindingLayout::W8A16AttentionWeightStriped;
    binding.byte_size = 64 * 128;
    binding.base_row = 32;
    binding.instruction_count = 128;
    binding.address_stride = 1;
    binding.shape = {64, 128};
    binding.slices = {0, 4, 8, 12, 16, 20, 24, 28};
    binding.role = "weight";
    binding.name = "query.weight";
    binding.hemisphere_mask = 3;
    binding.bank = bank;
    return binding;
}

BinaryProgram make_program(std::uint16_t bank)
{
    BinaryProgram program;
    program.bindings.push_back(make_weight(bank));
    return program;
}

std::vector<std::uint8_t> logical_weight()
{
    std::vector<std::uint8_t> result(64 * 128);
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = static_cast<std::uint8_t>(index * 17 + 3);
    return result;
}

BinaryBinding make_kv_cache(BindingLayout layout, std::uint32_t index,
    std::string name, std::vector<std::uint16_t> slices,
    std::int64_t base_row, std::int64_t instruction_count)
{
    BinaryBinding binding;
    binding.index = index;
    binding.access = BindingAccess::Internal;
    binding.element_type = BindingElementType::BF16;
    binding.layout = layout;
    binding.byte_size = 256 * 2 * 128 * 2;
    binding.base_row = base_row;
    binding.instruction_count = instruction_count;
    binding.address_stride = 1;
    binding.shape = {256, 2, 128};
    binding.slices = std::move(slices);
    binding.role = layout == BindingLayout::Fp16HeadPlanar
        ? "state.kv.key" : "state.kv.value";
    binding.name = std::move(name);
    binding.hemisphere_mask = 3;
    binding.bank = 0;
    return binding;
}

void test_kv_cache_roundtrip()
{
    std::vector<std::uint8_t> logical(256 * 2 * 128 * 2);
    for (std::size_t index = 0; index < logical.size(); ++index)
        logical[index] = static_cast<std::uint8_t>(
            (index * 29 + index / 257 + 11) & 0xff);

    std::vector<std::uint16_t> value_slices;
    for (std::uint16_t slice = 0; slice < 16; ++slice)
        value_slices.push_back(slice);
    const std::vector<BinaryBinding> bindings = {
        make_kv_cache(BindingLayout::Fp16HeadPlanar, 65536,
            "attention.key_cache", {16, 17, 18, 19}, 512, 1024),
        make_kv_cache(BindingLayout::Fp16ValueX16, 65537,
            "attention.value_cache", std::move(value_slices), 1024, 256),
    };
    for (const BinaryBinding& binding : bindings) {
        const PackedWeightImage packed = pack_binding_image(
            binding, logical, ExecutableHardwareConfig {});
        require(!packed.data.empty(),
            "KV cache packing produced an empty image");
        require(unpack_binding_image(binding, packed,
                    ExecutableHardwareConfig {}) == logical,
            "KV cache physical layout did not round-trip exactly");
    }
}

void test_segmented_gamma_broadcast()
{
    for (const std::size_t segmentRows : {std::size_t{1024}, std::size_t{768}}) {
        BinaryBinding gamma;
        gamma.index = 7;
        gamma.access = BindingAccess::Input;
        gamma.element_type = BindingElementType::BF16;
        gamma.layout = BindingLayout::Fp16VxmGammaBroadcast;
        gamma.shape = {1536};
        gamma.byte_size = 1536 * 2;
        gamma.base_row = segmentRows == 1024 ? 0
            : ftlpu::hw::kSramDepthRows - segmentRows;
        gamma.instruction_count = segmentRows;
        gamma.address_stride = 1;
        gamma.slices = {16, 17, 18, 19};
        gamma.hemisphere_mask = 3;
        gamma.bank = 1;
        std::vector<std::uint8_t> logical(gamma.byte_size);
        for (std::size_t i = 0; i < logical.size(); ++i)
            logical[i] = static_cast<std::uint8_t>((i * 37 + i / 19) & 0xff);

        BinaryProgram program;
        program.bindings.push_back(gamma);
        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        CModelRuntime runtime(*system);
        runtime.load(program);
        runtime.upload_input(gamma.index, logical);
        const auto packed = pack_binding_image(gamma, logical, program.hardware);
        require(packed.segments.size() == 8,
            "segmented gamma should use two slice pairs in both hemispheres");
        for (const std::size_t feature : {std::size_t{0}, segmentRows - 1,
                 segmentRows, std::size_t{1535}}) {
            const std::size_t segment = feature / segmentRows;
            const std::size_t row = gamma.base_row + feature % segmentRows;
            for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere)
                for (std::size_t lane = 0; lane < program.hardware.bytes_per_word; ++lane)
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        const auto actual = system->read_mem_sram_lane_byte(
                            static_cast<ftlpu::Hemisphere>(hemisphere),
                            gamma.slices[2 * segment + byte], gamma.bank,
                            lane / program.hardware.lanes_per_tile, row,
                            lane % program.hardware.lanes_per_tile);
                        require(actual == logical[2 * feature + byte],
                            "segmented gamma upload lost a feature at the bank-depth boundary");
                    }
        }
        for (const auto& segment : packed.segments)
            for (std::size_t row = 0; row < segment.vector_count; ++row)
                for (std::size_t lane = 0; lane < program.hardware.bytes_per_word; ++lane)
                    require(packed.data[segment.byte_offset
                            + row * program.hardware.bytes_per_word + lane]
                        == system->read_mem_sram_lane_byte(
                            static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                            segment.slice, gamma.bank, lane / program.hardware.lanes_per_tile,
                            segment.base_row + row, lane % program.hardware.lanes_per_tile),
                        "segmented gamma offline image differs from CModel upload");
    }
}

void test_vxm_distributed_roundtrip()
{
    BinaryProgram program;
    if (program.hardware.bytes_per_word != 32) return;
    for (auto type : {BindingElementType::BF16, BindingElementType::F16})
        for (std::uint16_t mask : {1, 2, 3}) {
            BinaryBinding binding;
            binding.element_type = type;
            binding.layout = BindingLayout::Fp16VxmDistributed16;
            binding.shape = {64, 128};
            binding.byte_size = 64 * 128 * 2;
            binding.base_row = 7;
            binding.address_stride = 2;
            binding.bank = 1;
            binding.hemisphere_mask = mask;
            for (std::uint16_t slice = 0; slice < 16; ++slice)
                binding.slices.push_back(slice);
            std::vector<std::uint8_t> logical(binding.byte_size);
            for (std::size_t i = 0; i < logical.size(); ++i)
                logical[i] = static_cast<std::uint8_t>(i * 37 + i / 113);
            const auto packed = pack_binding_image(binding, logical, program.hardware);
            require(unpack_binding_image(binding, packed, program.hardware) == logical,
                "VXM distributed16 C2C output did not round-trip");
        }
}

void test_four_slice_pair_planar_roundtrip()
{
    BinaryProgram program;
    for (const std::uint16_t mask : {std::uint16_t {1},
             std::uint16_t {2}, std::uint16_t {3}}) {
        BinaryBinding binding;
        binding.access = BindingAccess::Output;
        binding.element_type = BindingElementType::BF16;
        binding.layout = BindingLayout::Fp16PairPlanar;
        binding.shape = {32, 4 * program.hardware.bytes_per_word};
        binding.byte_size = binding.shape[0] * binding.shape[1] * 2;
        binding.base_row = 3;
        binding.address_stride = 1;
        binding.bank = 1;
        binding.hemisphere_mask = mask;
        binding.slices = {16, 17, 18, 19};
        std::vector<std::uint8_t> logical(binding.byte_size);
        for (std::size_t index = 0; index < logical.size(); ++index)
            logical[index] = static_cast<std::uint8_t>(
                (index * 29 + index / 17) & 0xff);
        const auto packed = pack_binding_image(
            binding, logical, program.hardware);
        require(unpack_binding_image(binding, packed, program.hardware)
                == logical,
            "four-slice pair-planar C2C image did not round-trip");
    }
}

void test_distributed_activation_target_width()
{
    BinaryProgram program;
    BinaryBinding activation;
    activation.index = 0;
    activation.access = BindingAccess::Input;
    activation.element_type = BindingElementType::BF16;
    activation.layout = BindingLayout::Fp16MxmDistributed16;
    activation.shape = {1, 64};
    activation.byte_size = 128;
    activation.base_row = 4;
    activation.instruction_count = 64;
    activation.address_stride = 1;
    activation.bank = 0;
    activation.hemisphere_mask = 3;
    for (std::uint16_t slice = 0; slice < 16; ++slice)
        activation.slices.push_back(slice);
    program.bindings.push_back(activation);
    std::vector<std::uint8_t> logical(activation.byte_size);
    for (std::size_t index = 0; index < logical.size(); ++index)
        logical[index] = static_cast<std::uint8_t>(index * 17 + 13);

    const auto packed = pack_binding_image(activation, logical, program.hardware);
    require(unpack_binding_image(activation, packed, program.hardware) == logical,
        "distributed activation did not round-trip at target physical width");
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(activation.index, logical);
    for (const auto& segment : packed.segments)
        for (std::size_t row = 0; row < segment.vector_count; ++row)
            for (std::size_t lane = 0;
                 lane < program.hardware.bytes_per_word; ++lane)
                require(packed.data[segment.byte_offset
                            + row * program.hardware.bytes_per_word + lane]
                    == system->read_mem_sram_lane_byte(
                        static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                        segment.slice, activation.bank, lane / 8,
                        segment.base_row + row, lane % 8),
                    "distributed activation C2C image differs from direct upload");
}

void test_token_partition_planar()
{
    for (const auto type : {BindingElementType::BF16, BindingElementType::F32}) {
        const std::size_t bytes = type == BindingElementType::F32 ? 4 : 2;
        BinaryProgram program;
        BinaryBinding b;
        b.index = 0; b.access = BindingAccess::Input; b.element_type = type;
        b.layout = BindingLayout::TokenPartitionPlanar;
        b.shape = {5,65}; b.byte_size = 5 * 65 * bytes;
        b.bank = 1; b.base_row = 7; b.address_stride = 2;
        b.hemisphere_mask = 3; b.instruction_count = 18;
        for (std::size_t i = 0; i < bytes; ++i) b.slices.push_back(8+i);
        program.bindings.push_back(b);
        auto output = b; output.access = BindingAccess::Output;
        program.bindings.push_back(output);
        std::vector<std::uint8_t> logical(b.byte_size);
        for (std::size_t i = 0; i < logical.size(); ++i) logical[i] = (i * 17 + i / 65) % 251;
        const auto image = pack_binding_image(b, logical, program.hardware);
        require(unpack_binding_image(b, image, program.hardware) == logical,
            "token-partitioned image failed roundtrip");
        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        CModelRuntime runtime(*system);
        runtime.load(program); runtime.upload_input(0, logical);
        require(runtime.download_output(0) == logical, "token partition direct runtime roundtrip failed");
        for (std::size_t row = 0; row < 5; ++row)
            for (std::size_t col = 0; col < 65; ++col)
                for (std::size_t byte = 0; byte < bytes; ++byte) {
                    const auto side = row < 3 ? ftlpu::Hemisphere::West : ftlpu::Hemisphere::East;
                    const auto address = 7 + ((col / program.hardware.bytes_per_word) * 3 + row % 3) * 2;
                    const auto lane = col % program.hardware.bytes_per_word;
                    require(system->read_mem_sram_lane_byte(side, 8+byte, 1, lane/8, address, lane%8)
                        == logical[(row*65+col)*bytes+byte], "token partition physical ownership mismatch");
                }
    }
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

using ImageRows = std::map<std::tuple<std::uint16_t, std::uint16_t,
    std::uint32_t>, std::vector<std::uint8_t>>;

ImageRows physical_rows(const PackedWeightImage& image, std::size_t width)
{
    ImageRows rows;
    for (const auto& segment : image.segments)
        for (std::uint32_t row = 0; row < segment.vector_count; ++row) {
            std::vector<std::uint8_t> data(width);
            std::copy_n(image.data.begin() + segment.byte_offset + row * width,
                width, data.begin());
            require(rows.emplace(std::make_tuple(segment.hemisphere,
                        segment.slice, segment.base_row + row), data).second,
                "page contains overlapping physical vectors");
        }
    return rows;
}

void test_paged_output_tail(BindingLayout layout, std::size_t rows,
    std::size_t columns, std::size_t width = 32)
{
    ExecutableHardwareConfig hardware;
    hardware.bytes_per_word = hardware.mxm_rows = hardware.mxm_columns = width;
    hardware.tile_rows = width / 8;
    hardware.sram_depth_rows = hardware.words_per_bank = 8192;
    const auto waveColumns = 4 * width;
    const auto paddedColumns = ((columns + waveColumns - 1) / waveColumns) * waveColumns;
    std::vector<std::uint8_t> logical(rows * columns);
    std::vector<std::uint8_t> padded(rows * paddedColumns, 0);
    for (std::size_t k = 0; k < rows; ++k)
        for (std::size_t n = 0; n < columns; ++n)
            padded[k * paddedColumns + n] = logical[k * columns + n] =
                static_cast<std::uint8_t>(k * 29 + n * 17 + 3);
    BinaryBinding binding = make_weight(1);
    binding.layout = layout;
    binding.shape = {rows, columns};
    binding.byte_size = logical.size();
    binding.base_row = 17;
    binding.address_stride = 2;
    binding.slices = {8, 9, 10, 11, 12, 13, 14, 15};
    binding.page_storage_slices = binding.slices;
    binding.paged_weight = true;
    binding.page_bank_count = 2;
    binding.page_rows = 8192;
    const bool down = layout == BindingLayout::W8A16MxmWeightWaveStriped;
    binding.page_count = down ? paddedColumns / waveColumns : 1;
    binding.page_granularity = down ? rows / width : paddedColumns / (2 * width);
    binding.page_items_per_slice_group = binding.page_granularity;
    binding.page_role_group_count = 1;
    binding.instruction_count = down ? rows / width * (2 * hardware.tile_rows)
        : rows / width * (layout == BindingLayout::W8A16AttentionWeightStriped
              ? paddedColumns / waveColumns * (2 * hardware.tile_rows)
              : paddedColumns / (2 * width) * hardware.tile_rows);
    auto aligned = binding;
    aligned.shape[1] = paddedColumns;
    aligned.byte_size = padded.size();
    if (layout == BindingLayout::W8A16AttentionWeightStriped && width == 32) {
        auto resident = binding;
        auto residentAligned = aligned;
        resident.paged_weight = residentAligned.paged_weight = false;
        const auto tail = pack_binding_image(resident, logical, hardware);
        const auto standard = physical_rows(
            pack_binding_image(residentAligned, padded, hardware), width);
        for (const auto& [address, bytes] : physical_rows(tail, width))
            require(standard.at(address) == bytes,
                "resident attention tail differs from its zero-padded image");
        require(tail.data.size() == logical.size(),
            "resident attention tail dropped or duplicated weight bytes");
    }
    std::size_t packedBytes = 0;
    for (std::uint32_t page = 0; page < binding.page_count; ++page) {
        const auto tail = pack_weight_binding_page(binding, page, logical, hardware);
        const auto standard = physical_rows(
            pack_weight_binding_page(aligned, page, padded, hardware), width);
        for (const auto& [address, data] : physical_rows(tail, width))
            require(standard.at(address) == data,
                "tail weight differs from aligned zero-padded physical image");
        packedBytes += tail.data.size();
    }
    require(packedBytes == logical.size(),
        "paged tail dropped or duplicated a logical weight byte");
    binding.shape[1] = columns - 1;
    binding.byte_size = rows * (columns - 1);
    logical.resize(binding.byte_size);
    bool rejected = false;
    try { (void)pack_weight_binding_page(binding, 0, logical, hardware); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unaligned output tail was accepted");
    if (layout != BindingLayout::W8A16AttentionWeightStriped) {
        binding.shape[1] = columns - width;
        binding.byte_size = rows * (columns - width);
        logical.resize(binding.byte_size);
        rejected = false;
        try { (void)pack_weight_binding_page(binding, 0, logical, hardware); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unimplemented single-tile O/FFN-down tail was accepted");
    }
}

} // namespace

int main(int argc, char** argv)
try {
    test_vxm_distributed_roundtrip();
    test_token_partition_planar();
    if (argc == 2 && std::string(argv[1]) == "--paged-output-tail") {
        for (const std::size_t width : {16, 32, 64}) {
            test_paged_output_tail(BindingLayout::W8A16AttentionWeightStriped,
                18 * width, 18 * width, width);
            test_paged_output_tail(BindingLayout::W8A16AttentionWeightStriped,
                18 * width, 6 * width, width);
            test_paged_output_tail(BindingLayout::W8A16MxmWeightStriped,
                18 * width, 18 * width, width);
            test_paged_output_tail(BindingLayout::W8A16MxmWeightWaveStriped,
                48 * width, 18 * width, width);
        }
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--generic-planar") {
        test_planar_transport(false);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--segmented-gamma") {
        test_segmented_gamma_broadcast();
        test_distributed_activation_target_width();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--attention16-page") {
        test_attention16_paged_weight();
        test_projection_bias16_eight_slices();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--native4-16-page") {
        test_native4_16_paged_weight();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--tiled-ffn-down-page") {
        test_channel_tiled_ffn_down_page();
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--four-slice-pair-planar") {
        test_four_slice_pair_planar_roundtrip();
        return 0;
    }
    if (argc == 2) {
        const auto compiled = read_binary_program(argv[1]);
        int projectionCount = 0, tableCount = 0;
        for (const auto& binding : compiled.bindings) {
            if (binding.role == "weight" && binding.rope_rotary_dim == 96
                && binding.rope_head_dim == 128) ++projectionCount;
            if (binding.initializer == BindingInitializer::RopeTable) {
                require(binding.rope_rotary_dim == 96 && binding.rope_head_dim == 128
                    && std::abs(binding.rope_attention_factor - std::sqrt(1.0f + 5.0f / 12.0f)) < 1.0e-5f,
                    "compiled Phi RoPE table metadata lost partial width or LongRoPE amplitude");
                ++tableCount;
            }
        }
        require(projectionCount == 2 && tableCount >= 1,
            "compiled Phi Q/K packing metadata does not match partial RoPE");
    }
    {
        BinaryBinding table;
        table.access = BindingAccess::Internal;
        table.element_type = BindingElementType::BF16;
        table.layout = BindingLayout::Fp16RopeTable;
        table.role = "constant";
        table.initializer = BindingInitializer::RopeTable;
        table.shape = {32, 64, 2};
        table.byte_size = 32 * 64 * 2 * 2;
        table.slices = {0, 1, 2, 3};
        table.address_stride = 1;
        table.instruction_count = 64;
        table.hemisphere_mask = 3;
        table.rope_theta = 10000.0f;
        table.rope_head_dim = 128;
        table.rope_rotary_dim = 96;
        table.rope_attention_factor = 1.19f;
        for (int d = 0; d < 48; ++d)
            table.rope_inverse_frequencies.push_back(0.01f * (d + 1));
        BinaryProgram program;
        program.bindings.push_back(table);
        std::stringstream encoded(std::ios::in | std::ios::out | std::ios::binary);
        write_binary_program(program, encoded);
        const auto decoded = read_binary_program(encoded);
        require(decoded.bindings[0].rope_rotary_dim == 96
            && decoded.bindings[0].rope_attention_factor == 1.19f
            && decoded.bindings[0].rope_inverse_frequencies == table.rope_inverse_frequencies,
            "binary round-trip dropped partial RoPE parameters");
        const auto bytes = encoded.str();
        const auto metadata = read_binary_program_metadata(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
        require(metadata.bindings[0].rope_rotary_dim == 96
            && metadata.bindings[0].rope_inverse_frequencies == table.rope_inverse_frequencies,
            "binary metadata reader dropped partial RoPE parameters");
        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        CModelRuntime runtime(*system);
        runtime.load(decoded);
        for (std::size_t d = 0; d < 64; ++d) {
            const float angle = d < 48 ? 7.0f * table.rope_inverse_frequencies[d] : 0.0f;
            const float factor = d < 48 ? 1.19f : 1.0f;
            const std::uint16_t expected[] = {
                ftlpu::Bf16::from_float(std::cos(angle) * factor).bits(),
                ftlpu::Bf16::from_float(std::sin(angle) * factor).bits()};
            for (int hemisphere = 0; hemisphere < 2; ++hemisphere)
                for (int trig = 0; trig < 2; ++trig)
                    for (int byte = 0; byte < 2; ++byte)
                        require(system->read_mem_sram_lane_byte(static_cast<ftlpu::Hemisphere>(hemisphere),
                            trig * 2 + byte, 0, (d % 32) / 8, (d / 32) * 32 + 7, d % 8)
                                == std::uint8_t(expected[trig] >> (byte * 8)),
                            "partial RoPE table has wrong frequency, amplitude or unchanged tail");
        }
    }
    {
        auto partial = make_weight(1);
        partial.rope_head_dim = 128;
        partial.rope_rotary_dim = 96;
        auto source = logical_weight();
        const auto canonical = pack_rope_projection_data(partial, source);
        for (std::size_t k = 0; k < 64; ++k)
            for (std::size_t n = 0; n < 128; ++n) {
                const std::size_t expected = n < 48 ? n : n < 64 ? n + 48
                    : n < 112 ? n - 16 : n;
                require(canonical[k * 128 + n] == source[k * 128 + expected],
                    "partial Q/K packing mixes rotary or unchanged tail features");
            }
        const auto packedPartial = pack_weight_binding(partial, source, ExecutableHardwareConfig{});
        partial.rope_rotary_dim = 0;
        require(packedPartial.data == pack_weight_binding(partial, canonical, ExecutableHardwareConfig{}).data,
            "partial Q/K offline image did not apply the compiler-selected permutation");
    }
    {
        BinaryBinding gamma;
        gamma.shape = {256};
        gamma.element_type = BindingElementType::BF16;
        gamma.byte_size = 512;
        gamma.rope_head_dim = 256;
        gamma.rope_rotary_dim = 64;
        std::vector<std::uint8_t> source(512);
        for (std::size_t i = 0; i < 256; ++i) {
            source[2*i] = static_cast<std::uint8_t>(i);
            source[2*i+1] = 0x3f;
        }
        const auto physical = pack_rope_projection_data(gamma, source);
        for (std::size_t lane = 0; lane < 256; ++lane) {
            const auto expected = lane < 32 ? lane : lane < 128 ? lane+32
                : lane < 160 ? lane-96 : lane;
            require(physical[2*lane] == source[2*expected]
                    && physical[2*lane+1] == source[2*expected+1],
                "partial RoPE must apply the Q/K feature permutation to BF16 norm gamma");
        }
    }
    test_kv_cache_roundtrip();
    test_distributed_activation_target_width();
    if (ExecutableHardwareConfig {}.sram_depth_rows == 1024)
        test_segmented_gamma_broadcast();
    test_planar_transport();
    test_paged_output_tail(BindingLayout::W8A16AttentionWeightStriped,
        576, 576);
    test_paged_output_tail(BindingLayout::W8A16AttentionWeightStriped,
        576, 192);
    test_paged_output_tail(BindingLayout::W8A16MxmWeightStriped, 576, 576);
    test_paged_output_tail(BindingLayout::W8A16MxmWeightWaveStriped,
        1536, 576);
    const auto data = logical_weight();
    const BinaryBinding binding = make_weight(1);
    const PackedWeightImage packed = pack_weight_binding(
        binding, data, ExecutableHardwareConfig {});
    require(!packed.data.empty() && packed.data.size() % 32 == 0,
        "packed image is not a sequence of SRAM vectors");
    require(packed.segments.size() == 16,
        "attention packing should produce eight slices per hemisphere");

    auto host_system = std::make_unique<ftlpu::TspSliceSystem>();
    auto packed_system = std::make_unique<ftlpu::TspSliceSystem>();
    CModelRuntime host_runtime(*host_system);
    host_runtime.load(make_program(1));
    host_runtime.upload_binding(binding, data);
    for (const PackedWeightSegment& segment : packed.segments)
        for (std::uint32_t vector = 0; vector < segment.vector_count;
             ++vector)
            for (std::uint32_t byte = 0; byte < 32; ++byte) {
                const std::size_t offset = static_cast<std::size_t>(
                    segment.byte_offset) + vector * 32 + byte;
                packed_system->initialize_mem_sram_lane_byte(
                    static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                    segment.slice, 1, byte / 8,
                    segment.base_row + vector, byte % 8,
                    packed.data[offset]);
                const auto expected = host_system->read_mem_sram_lane_byte(
                    static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                    segment.slice, 1, byte / 8,
                    segment.base_row + vector, byte % 8);
                const auto actual = packed_system->read_mem_sram_lane_byte(
                    static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                    segment.slice, 1, byte / 8,
                    segment.base_row + vector, byte % 8);
                require(actual == expected,
                    "offline page image differs from CModel host upload");
                require(host_system->read_mem_sram_lane_byte(
                            static_cast<ftlpu::Hemisphere>(segment.hemisphere),
                            segment.slice, 0, byte / 8,
                            segment.base_row + vector, byte % 8)
                        == 0,
                    "bank1 host upload modified bank0");
            }

    ModelPackage package;
    package.model_name = "qwen-weight-page-builder-test";
    package.architecture = "Qwen2ForCausalLM";
    package.executables = {
        {"layer.bank0", make_program(0), {}},
        {"layer.bank1", make_program(1), {}},
    };
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        const std::string name = "layers." + std::to_string(layer + 4)
            + ".query.weight";
        package.tensors.push_back(ModelTensor {name,
            BindingElementType::I8, {64, 128}, data,
            ModelTensorEncoding::SymmetricPerTensorI8, -1, 0, {0.01f}});
        package.invocations.push_back(ModelInvocation {
            "layers." + std::to_string(layer + 4), layer, {{1, name}}, {}, {}});
    }

    build_weight_pages(package);
    require(package.weight_pages.size() == 2
            && package.weight_pages[0].bank == 0
            && package.weight_pages[1].bank == 1
            && package.weight_pages[0].layer == 4
            && package.weight_pages[1].layer == 5,
        "builder did not create alternating weight pages");
    require(package.invocations[0].weight_page == 0
            && package.invocations[1].weight_page == 1,
        "builder did not bind invocations to their pages");
    require(package.tensors.size() == 2
            && package.tensors[0].encoding
                == ModelTensorEncoding::TargetPackedSramVectors,
        "builder retained logical weights or missed packed encoding");
    require(package.weight_pages[0].segments.size() == 16
            && package.weight_pages[1].segments.size() == 16,
        "builder emitted the wrong physical segment count");
    std::cout << "weight_page_builder_test passed packed_bytes="
              << packed.data.size() << " segments="
              << packed.segments.size() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "weight_page_builder_test failed: "
              << error.what() << '\n';
    return 1;
}
