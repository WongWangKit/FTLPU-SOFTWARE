#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

BinaryBinding make_binding(std::uint32_t index, std::string name,
    BindingElementType type, BindingLayout layout,
    std::vector<std::uint64_t> shape, std::int64_t base_row,
    std::int64_t rows, std::vector<std::uint16_t> slices)
{
    BinaryBinding binding;
    binding.index = index;
    binding.access = BindingAccess::Input;
    binding.element_type = type;
    binding.layout = layout;
    binding.byte_size = 1;
    for (std::uint64_t dimension : shape) binding.byte_size *= dimension;
    if (type == BindingElementType::BF16) binding.byte_size *= 2;
    binding.base_row = base_row;
    binding.instruction_count = rows;
    binding.address_stride = 1;
    binding.shape = std::move(shape);
    binding.slices = std::move(slices);
    binding.role = "weight";
    binding.name = std::move(name);
    binding.hemisphere_mask = 3;
    binding.bank = 1;
    return binding;
}

std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint8_t seed)
{
    std::vector<std::uint8_t> data(bytes);
    for (std::size_t index = 0; index < bytes; ++index)
        data[index] = static_cast<std::uint8_t>(seed + index * 29);
    return data;
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
try {
    constexpr std::uint64_t hiddenSize = 256;
    constexpr std::uint64_t kvSize = 64;
    constexpr std::uint64_t intermediateSize = 512;
    const std::vector<std::uint16_t> attentionSlices {
        22, 23, 25, 26, 27, 29, 30, 31};
    const std::vector<std::uint16_t> gateSlices {
        0, 4, 8, 12, 16, 20, 24, 28};
    const std::vector<std::uint16_t> upSlices {
        3, 7, 11, 13, 14, 15, 17, 18};
    const std::vector<std::uint16_t> downSlices {
        1, 2, 5, 6, 9, 10, 19, 21};
    std::vector<std::uint16_t> norm0Slices;
    std::vector<std::uint16_t> norm1Slices;
    for (std::uint16_t slice = 16; slice < 32; ++slice)
        norm0Slices.push_back(slice);
    for (std::uint16_t slice = 32; slice < 48; ++slice)
        norm1Slices.push_back(slice);

    BinaryProgram program;
    program.hardware.mxms_per_hemisphere = 1;
    const auto normBaseRow = static_cast<std::int64_t>(
        program.hardware.sram_depth_rows - hiddenSize);
    constexpr std::int64_t queryRows = hiddenSize * hiddenSize / 512;
    constexpr std::int64_t kvRows = hiddenSize * 128 / 512;
    constexpr std::int64_t outputRows = hiddenSize * hiddenSize / 512;
    constexpr std::int64_t ffnRows =
        hiddenSize * intermediateSize / 512;
    program.bindings = {
        make_binding(0, "input_layernorm.weight", BindingElementType::BF16,
            BindingLayout::Fp16VxmRowParallel8, {hiddenSize}, normBaseRow,
            hiddenSize,
            norm0Slices),
        make_binding(1, "query.weight", BindingElementType::I8,
            BindingLayout::W8A16AttentionWeightStriped,
            {hiddenSize, hiddenSize}, 0, queryRows, attentionSlices),
        make_binding(2, "key.weight", BindingElementType::I8,
            BindingLayout::W8A16AttentionWeightStriped,
            {hiddenSize, kvSize}, queryRows, kvRows, attentionSlices),
        make_binding(3, "value.weight", BindingElementType::I8,
            BindingLayout::W8A16AttentionWeightStriped,
            {hiddenSize, kvSize}, queryRows + kvRows, kvRows,
            attentionSlices),
        make_binding(4, "output.weight", BindingElementType::I8,
            BindingLayout::W8A16MxmWeightStriped,
            {hiddenSize, hiddenSize}, queryRows + 2 * kvRows,
            outputRows, attentionSlices),
        make_binding(5, "post_attention_layernorm.weight",
            BindingElementType::BF16,
            BindingLayout::Fp16VxmRowParallel8, {hiddenSize}, normBaseRow,
            hiddenSize,
            norm1Slices),
        make_binding(6, "gate.weight", BindingElementType::I8,
            BindingLayout::W8A16Block8WeightWaveStriped,
            {hiddenSize, intermediateSize}, 0, ffnRows, gateSlices),
        make_binding(7, "up.weight", BindingElementType::I8,
            BindingLayout::W8A16Block8WeightWaveStriped,
            {hiddenSize, intermediateSize}, 0, ffnRows, upSlices),
        make_binding(8, "down.weight", BindingElementType::I8,
            BindingLayout::W8A16Block8WeightWaveStriped,
            {intermediateSize, hiddenSize}, 0, ffnRows, downSlices),
    };

    const auto replicated = make_binding(9, "replicated.weight",
        BindingElementType::I8,
        BindingLayout::W8A16MxmWeightReplicated,
        {32, 32}, program.hardware.sram_depth_rows - 16, 16, gateSlices);
    const auto replicatedData = pattern(32 * 32, 7);
    const auto replicatedImage = pack_weight_binding(
        replicated, replicatedData, program.hardware);
    require(!replicatedImage.segments.empty(),
        "replicated W8A16 weight page is empty");
    for (const auto& segment : replicatedImage.segments) {
        const auto peer = std::find_if(replicatedImage.segments.begin(),
            replicatedImage.segments.end(), [&](const auto& candidate) {
                return candidate.hemisphere != segment.hemisphere
                    && candidate.slice == segment.slice
                    && candidate.base_row == segment.base_row
                    && candidate.vector_count == segment.vector_count;
            });
        require(peer != replicatedImage.segments.end(),
            "replicated W8A16 weight has no peer hemisphere segment");
        const auto bytes = static_cast<std::size_t>(
            segment.vector_count) * 32;
        require(std::equal(
                replicatedImage.data.begin() + segment.byte_offset,
                replicatedImage.data.begin() + segment.byte_offset + bytes,
                replicatedImage.data.begin() + peer->byte_offset),
            "replicated W8A16 weight differs across hemispheres");
    }

    ModelPackage package;
    package.model_name = "qwen2.5-1.5b-layer-page";
    package.architecture = "Qwen2ForCausalLM";
    package.executables.push_back({"layer.bank1", program, {}});
    ModelInvocation invocation;
    invocation.name = "layers.7";
    invocation.executable_index = 0;
    for (const BinaryBinding& binding : program.bindings) {
        const std::string tensorName = "layers.7." + binding.name;
        package.tensors.push_back(ModelTensor {tensorName,
            binding.element_type, binding.shape,
            pattern(static_cast<std::size_t>(binding.byte_size),
                static_cast<std::uint8_t>(binding.index * 17 + 1)),
            binding.element_type == BindingElementType::I8
                ? ModelTensorEncoding::SymmetricPerTensorI8
                : ModelTensorEncoding::Raw,
            -1, 0, binding.element_type == BindingElementType::I8
                ? std::vector<float> {0.01f} : std::vector<float> {}});
        invocation.inputs.push_back({binding.index, tensorName});
    }
    package.invocations.push_back(std::move(invocation));

    build_weight_pages(package, {.first_bank = 1});
    require(package.weight_pages.size() == 1
            && package.weight_pages[0].layer == 7
            && package.weight_pages[0].bank == 1,
        "Qwen page metadata is incorrect");
    require(package.weight_pages[0].tensors.size() == 9,
        "Qwen page does not contain every layer weight");
    std::uint64_t packedBytes = 0;
    for (const ModelTensor& tensor : package.tensors) {
        require(tensor.encoding
                == ModelTensorEncoding::TargetPackedSramVectors,
            "Qwen logical weight was not replaced by target-packed data");
        packedBytes += tensor.data.size();
    }
    constexpr std::uint64_t expectedPackedBytes =
        hiddenSize * hiddenSize + 2 * hiddenSize * kvSize
        + hiddenSize * hiddenSize
        + 3 * hiddenSize * intermediateSize
        + 2 * hiddenSize * 2 * 16 * 32;
    require(packedBytes == expectedPackedBytes,
        "Qwen layer page has an unexpected physical byte count");
    for (const auto& segment : package.weight_pages[0].segments)
        require(static_cast<std::uint64_t>(segment.base_row)
                + segment.vector_count <= program.hardware.words_per_bank,
            "Qwen page segment exceeds one SRAM bank");

    std::cout << "qwen_weight_page_builder_test passed bytes="
              << packedBytes << " segments="
              << package.weight_pages[0].segments.size() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "qwen_weight_page_builder_test failed: "
              << error.what() << '\n';
    return 1;
}
