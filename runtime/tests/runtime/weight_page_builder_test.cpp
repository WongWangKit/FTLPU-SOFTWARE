#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"
#include "ftlpu/core/bf16.hpp"

#include <cstdint>
#include <cmath>
#include <sstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

void require(bool condition, const std::string& message);

void test_planar_transport()
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
        BinaryBinding b;
        b.access = BindingAccess::Input;
        b.element_type = fp32 ? BindingElementType::F32 : BindingElementType::BF16;
        b.layout = layout;
        b.shape = kernel ? std::vector<std::uint64_t>{64, 4}
                         : std::vector<std::uint64_t>{3, 64};
        b.byte_size = b.shape[0] * b.shape[1] * (fp32 ? 4 : 2);
        b.base_row = 19;
        b.bank = 1;
        b.hemisphere_mask = 2;
        b.address_stride = 1;
        b.instruction_count = kernel ? 8 : scalar ? 192 : 6;
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
                for (std::size_t lane = 0; lane < 32; ++lane)
                    require(image.data[segment.byte_offset + row * 32 + lane]
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

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
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
    test_planar_transport();
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
