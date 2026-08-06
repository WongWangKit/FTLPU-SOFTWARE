#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/session_memory_planner.hpp"

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
    program.memory_floors.push_back({0, 3, 6144});
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
    BinaryBinding key_cache;
    key_cache.index = 10;
    key_cache.access = BindingAccess::Internal;
    key_cache.element_type = BindingElementType::F16;
    key_cache.layout = BindingLayout::Fp16MxmDistributed16;
    key_cache.byte_size = 2048 * 3 * 64 * sizeof(std::uint16_t);
    key_cache.base_row = 6000;
    key_cache.instruction_count = 1536;
    key_cache.address_stride = 1;
    key_cache.shape = {2048, 3, 64};
    key_cache.slices = {
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31};
    key_cache.role = "state.kv.key";
    key_cache.name = "layers.0.key_cache";
    key_cache.hemisphere_mask = 1;
    key_cache.initializer = BindingInitializer::None;
    program.bindings.push_back(key_cache);
    program.scale_relocations.push_back(BinaryScaleRelocation {
        0, 1, QueueKind::Vxm, 7, 11, VxmImmediateOperand::Rhs});
    program.timelines.push_back({"decoder", 0, 17});

    ModelPackage package;
    package.model_name = "roundtrip";
    package.architecture = "LlamaForCausalLM";
    package.tensors.push_back(ModelTensor {
        "layers.0.q_proj.weight", BindingElementType::I8, {2, 2},
        {1, 2, 3, 4}, ModelTensorEncoding::SymmetricPerAxisI8, 1, 0,
        {0.25f, 0.5f}});
    package.tensors.push_back(ModelTensor {
        "model.embed_tokens.weight", BindingElementType::F16, {2, 2},
        {0, 0, 0, 0, 0, 0, 0, 0}});
    package.values = {
        ModelValue {"token_ids", BindingElementType::I32, {2}, true, false},
        ModelValue {"hidden.0", BindingElementType::F16, {2, 2}, true, false},
        ModelValue {"hidden.1", BindingElementType::F16, {2, 2}, false, true},
        ModelValue {"logits", BindingElementType::F32, {1, 2}, false, true},
    };
    package.embedding_lookups.push_back(
        {"embedding", "token_ids", "model.embed_tokens.weight", "hidden.0"});
    package.host_lm_heads.push_back(
        {"lm_head", "hidden.1", "model.embed_tokens.weight", "logits", true});
    package.states.push_back(ModelState {
        "layers.0.key_cache", ModelStateKind::KvKey,
        BindingElementType::F16, {2048, 3, 64}, 0, 2048});
    package.executables.push_back({"decoder_layer", program, {}});
    package.invocations.push_back(ModelInvocation {
        "layers.0", 0, {{0, "hidden.0"}}, {{0, "hidden.1"}},
        {{10, "layers.0.key_cache"}}});
    require(package.executables.size() == 1
            && package.executables[0].name == "decoder_layer",
        "executable name was invalid before serialization");

    const auto path = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(path.parent_path());
    write_model_package(package, path);
    const ModelPackage decoded = read_model_package(path);
    require(decoded.executables.size() == 1
            && decoded.executables[0].name == "decoder_layer",
        "executable name was not preserved");

    require(decoded.model_name == package.model_name,
        "model name was not preserved");
    require(decoded.architecture == package.architecture,
        "architecture was not preserved");
    require(decoded.tensors.size() == 2
            && decoded.tensors[0].scales == package.tensors[0].scales
            && decoded.tensors[0].data == package.tensors[0].data,
        "quantized tensor was not preserved");
    require(decoded.executables.size() == 1
            && decoded.executables[0].program.max_cycle == 17
            && decoded.executables[0].program.memory_floors.size() == 1
            && decoded.executables[0].program.memory_floors[0]
                   .first_free_row == 6144
            && decoded.executables[0].program.bindings.size() == 4
            && decoded.executables[0].program.timelines.size() == 1
            && decoded.executables[0].program.timelines[0].name
                == "decoder"
            && decoded.executables[0].program.scale_relocations.size() == 1
            && decoded.executables[0]
                   .program.scale_relocations[0].scale_index == 1,
        "embedded executable was not preserved");
    const ModelPackage lazy = read_model_package(
        path, ModelPackageLoadMode::LazyExecutables);
    require(lazy.executables.size() == 1
            && lazy.executables[0].program.timelines.size() == 1
            && lazy.executables[0].program.memory_floors.size() == 1
            && lazy.executables[0].program.memory_floors[0]
                   .first_free_row == 6144
            && lazy.executables[0].program.timelines[0].end_cycle == 17
            && !lazy.executables[0].serialized_program.empty(),
        "lazy executable timeline metadata was not preserved");
    require(decoded.embedding_lookups.size() == 1
            && decoded.embedding_lookups[0].token_ids == "token_ids"
            && decoded.embedding_lookups[0].output == "hidden.0",
        "embedding lookup was not preserved");
    require(decoded.host_lm_heads.size() == 1
            && decoded.host_lm_heads[0].hidden == "hidden.1"
            && decoded.host_lm_heads[0].weight
                == "model.embed_tokens.weight"
            && decoded.host_lm_heads[0].last_token_only,
        "host LM head was not preserved");
    require(decoded.states.size() == 1
            && decoded.states[0].kind == ModelStateKind::KvKey
            && decoded.states[0].shape
                == std::vector<std::uint64_t>({2048, 3, 64})
            && decoded.states[0].max_tokens == 2048
            && decoded.invocations[0].states.size() == 1
            && decoded.invocations[0].states[0].binding_index == 10,
        "persistent KV state metadata was not preserved");
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
    const SessionMemoryPlan state_plan =
        SessionMemoryPlanner::plan(decoded);
    require(state_plan.persistent_states.size() == 1
            && state_plan.persistent_states[0].state
                == "layers.0.key_cache"
            && state_plan.invocations[0].states.size() == 1
            && state_plan.invocations[0].states[0].resolved_binding.shape
                == std::vector<std::uint64_t>({2048, 3, 64}),
        "persistent KV state was not physically allocated");

    ModelPackage fanout = decoded;
    fanout.states.clear();
    fanout.embedding_lookups.clear();
    fanout.host_lm_heads.clear();
    fanout.values.push_back(
        {"hidden.2", BindingElementType::F16, {2, 2}, false, true});
    fanout.invocations = {
        {"producer", 0, {{0, "model.embed_tokens.weight"}},
            {{0, "hidden.0"}}},
        {"consumer.0", 0, {{0, "hidden.0"}}, {{0, "hidden.1"}}},
        {"consumer.1", 0, {{0, "hidden.0"}}, {{0, "hidden.2"}}},
    };
    const SessionMemoryPlan fanoutPlan =
        SessionMemoryPlanner::plan(fanout);
    require(fanoutPlan.lifetimes.size() == 1
            && fanoutPlan.lifetimes[0].last_consumer == 2
            && !fanoutPlan.invocations[1]
                    .inputs[0].release_after_transfer
            && fanoutPlan.invocations[2]
                    .inputs[0].release_after_transfer,
        "fan-out lifetime did not retain the value through its last consumer");

    std::cout << "model_package_roundtrip_test passed\n";
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "model_package_roundtrip_test failed: "
              << exception.what() << '\n';
    return 1;
}
