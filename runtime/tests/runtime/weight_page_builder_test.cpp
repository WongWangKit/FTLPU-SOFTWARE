#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

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

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
try {
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
