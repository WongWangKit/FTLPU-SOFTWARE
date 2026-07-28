#include "ftlpu/software/runtime/model_package.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: model_package_roundtrip_test model.ftlpum");

    using namespace ftlpu::software::runtime;
    BinaryProgram program;
    program.max_cycle = 17;
    program.bindings = {
        BinaryBinding {
            0, BindingAccess::Input, BindingElementType::F16,
            BindingLayout::Fp16PairPlanar, 8, 12, 1, 1, {2, 2}, {0, 1},
            "activation", "input", 0, 1},
        BinaryBinding {
            0, BindingAccess::Output, BindingElementType::F16,
            BindingLayout::Fp16PairPlanar, 8, 24, 1, 1, {2, 2},
            {0, 1, 2, 3}, "result", "output", 17, 1},
    };
    BinaryBinding rope;
    rope.index = 2;
    rope.access = BindingAccess::Internal;
    rope.element_type = BindingElementType::F16;
    rope.layout = BindingLayout::Fp16RopeTable;
    rope.byte_size = 128 * 32 * 2 * sizeof(std::uint16_t);
    rope.base_row = 7000;
    rope.instruction_count = 128;
    rope.address_stride = 1;
    rope.shape = {128, 32, 2};
    rope.slices = {4, 5, 6, 7};
    rope.role = "constant";
    rope.name = "rope.cos_sin";
    rope.hemisphere_mask = 3;
    rope.initializer = BindingInitializer::RopeTable;
    rope.rope_theta = 100000.0f;
    rope.rope_head_dim = 64;
    program.bindings.push_back(rope);

    ModelPackage package;
    package.model_name = "roundtrip";
    package.architecture = "LlamaForCausalLM";
    package.tensors.push_back(ModelTensor {
        "layers.0.q_proj.weight", BindingElementType::I8, {2, 2},
        {1, 2, 3, 4}, ModelTensorEncoding::SymmetricPerAxisI8, 1, 0,
        {0.25f, 0.5f}});
    package.values = {
        ModelValue {"hidden.0", BindingElementType::F16, {2, 2}, true, false},
        ModelValue {"hidden.1", BindingElementType::F16, {2, 2}, false, true},
    };
    package.executables.push_back({"decoder_layer", program});
    package.invocations.push_back(ModelInvocation {
        "layers.0", 0, {{0, "hidden.0"}}, {{0, "hidden.1"}}});

    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    write_model_package(package, path);
    const ModelPackage decoded = read_model_package(path);

    require(decoded.model_name == package.model_name,
        "model name was not preserved");
    require(decoded.architecture == package.architecture,
        "architecture was not preserved");
    require(decoded.tensors.size() == 1
            && decoded.tensors[0].scales == package.tensors[0].scales
            && decoded.tensors[0].data == package.tensors[0].data,
        "quantized tensor was not preserved");
    require(decoded.executables.size() == 1
            && decoded.executables[0].program.max_cycle == 17
            && decoded.executables[0].program.bindings.size() == 3,
        "embedded executable was not preserved");
    const BinaryBinding& decoded_rope =
        decoded.executables[0].program.bindings[2];
    require(decoded_rope.initializer == BindingInitializer::RopeTable
            && decoded_rope.layout == BindingLayout::Fp16RopeTable
            && decoded_rope.name == "rope.cos_sin"
            && decoded_rope.rope_theta == 100000.0f
            && decoded_rope.rope_head_dim == 64,
        "internal RoPE initializer metadata was not preserved");
    require(decoded.invocations.size() == 1
            && decoded.invocations[0].inputs[0].value == "hidden.0"
            && decoded.invocations[0].outputs[0].value == "hidden.1",
        "invocation bindings were not preserved");

    std::cout << "model_package_roundtrip_test passed\n";
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "model_package_roundtrip_test failed: "
              << exception.what() << '\n';
    return 1;
}
