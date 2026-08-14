#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;

BinaryProgram make_program(std::uint16_t bank)
{
    BinaryProgram program;
    program.hardware.sram_depth_rows = 32768;
    for (std::uint32_t index = 0; index < 2; ++index) {
        BinaryBinding binding;
        binding.index = index;
        binding.access = BindingAccess::Input;
        binding.element_type = index == 0
            ? BindingElementType::BF16 : BindingElementType::I8;
        binding.layout = index == 0
            ? BindingLayout::Fp16MxmDistributed16
            : BindingLayout::W8A16Block8WeightWaveStriped;
        binding.byte_size = index == 0 ? 4096 : 65536;
        binding.base_row = index == 0 ? 4096 : 0;
        binding.instruction_count = index == 0 ? 16 : 256;
        binding.address_stride = 1;
        binding.shape = index == 0
            ? std::vector<std::uint64_t> {32, 64}
            : std::vector<std::uint64_t> {256, 256};
        binding.slices = index == 0
            ? std::vector<std::uint16_t> {16, 17, 18, 19, 20, 21, 22, 23,
                  24, 25, 26, 27, 28, 29, 30, 31}
            : std::vector<std::uint16_t> {0, 4, 8, 12, 16, 20, 24, 28};
        binding.role = index == 0 ? "activation" : "weight";
        binding.name = index == 0 ? "hidden" : "q_proj.weight";
        binding.hemisphere_mask = 3;
        binding.bank = bank;
        program.bindings.push_back(std::move(binding));
    }
    BinaryBinding output = program.bindings.front();
    output.access = BindingAccess::Output;
    output.index = 0;
    output.base_row = 8192;
    output.role = "result";
    program.bindings.push_back(std::move(output));
    return program;
}

} // namespace

int main()
try {
    ModelPackage package;
    package.model_name = "qwen2.5-1.5b-two-layer-page-test";
    package.architecture = "Qwen2ForCausalLM";
    package.executables = {
        {"layer.bank0", make_program(0), {}},
        {"layer.bank1", make_program(1), {}},
    };
    package.values = {
        {"hidden.0", BindingElementType::BF16, {32, 64}, true, false},
        {"hidden.1", BindingElementType::BF16, {32, 64}, false, false},
        {"hidden.2", BindingElementType::BF16, {32, 64}, false, true},
    };
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        const std::string weight = "layers." + std::to_string(layer)
            + ".q_proj.weight.packed";
        package.tensors.push_back(ModelTensor {
            weight, BindingElementType::I8, {256, 256},
            std::vector<std::uint8_t>(65536), ModelTensorEncoding::Raw});
        package.weight_pages.push_back(
            ModelWeightPage {layer, static_cast<std::uint16_t>(layer),
                {weight}});
        package.invocations.push_back(ModelInvocation {
            "layers." + std::to_string(layer), layer,
            {{0, "hidden." + std::to_string(layer)}, {1, weight}},
            {{0, "hidden." + std::to_string(layer + 1)}}, {}, layer});
    }

    validate_model_package(package);
    const SessionMemoryPlan plan = SessionMemoryPlanner::plan(package);
    if (plan.weight_pages.size() != 2 || !plan.resident_tensors.empty())
        throw std::runtime_error(
            "paged weights were incorrectly allocated as resident tensors");
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const auto& input = plan.invocations[layer].inputs[1];
        if (input.transfer != SessionTransferKind::WeightPage
            || input.resolved_binding.bank != layer)
            throw std::runtime_error(
                "invocation did not resolve its alternating weight bank");
    }
    std::cout << "weight_page_planner_test passed pages=2 resident=0\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "weight_page_planner_test failed: " << error.what() << '\n';
    return 1;
}
