#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> makeBf16(
    std::initializer_list<float> values)
{
    std::vector<std::uint8_t> result;
    result.reserve(values.size() * sizeof(std::uint16_t));
    for (const float value : values) {
        const std::uint16_t bits =
            ftlpu::Bf16::from_float(value).bits();
        result.push_back(static_cast<std::uint8_t>(bits));
        result.push_back(static_cast<std::uint8_t>(bits >> 8));
    }
    return result;
}

float readF32(
    const std::vector<std::uint8_t>& data, std::size_t index)
{
    float result = 0.0f;
    std::memcpy(
        &result, data.data() + index * sizeof(result), sizeof(result));
    return result;
}

float readBf16(
    const std::vector<std::uint8_t>& data, std::size_t index)
{
    const std::uint16_t bits =
        static_cast<std::uint16_t>(data[index * 2])
        | (static_cast<std::uint16_t>(data[index * 2 + 1]) << 8);
    return ftlpu::Bf16::from_bits(bits).to_float();
}

void requireEqual(float actual, float expected, const char* message)
{
    if (actual != expected)
        throw std::logic_error(
            std::string(message) + ": actual="
            + std::to_string(actual)
            + " expected=" + std::to_string(expected));
}

} // namespace

int main()
try {
    using namespace ftlpu::software::runtime;
    ModelPackage package;
    package.model_name = "host-lm-head-test";
    package.architecture = "LlamaForCausalLM";
    package.tensors.push_back(ModelTensor {
        "lm_head.weight", BindingElementType::BF16, {4, 3},
        makeBf16({
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
        })});
    package.values = {
        {"hidden", BindingElementType::BF16, {2, 3}, true, false},
        {"last_logits", BindingElementType::F32, {1, 4}, false, true},
        {"all_logits", BindingElementType::BF16, {2, 4}, false, true},
    };
    package.host_lm_heads = {
        {"last_token", "hidden", "lm_head.weight", "last_logits", true},
        {"all_tokens", "hidden", "lm_head.weight", "all_logits", false},
    };
    package.executables.push_back({"unused", BinaryProgram {}, {}});

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ModelSession session(*system);
    session.load(std::move(package));
    const auto hidden = makeBf16({
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 2.0f,
    });
    session.set_input("hidden", hidden);
    session.run();

    const auto& last = session.value("last_logits");
    const std::array<float, 4> expectedLast {
        -1.0f, 0.5f, 2.0f, 1.5f};
    if (last.size() != expectedLast.size() * sizeof(float))
        throw std::logic_error("last-token F32 logits size mismatch");
    for (std::size_t index = 0; index < expectedLast.size(); ++index)
        requireEqual(
            readF32(last, index), expectedLast[index],
            "last-token F32 logit mismatch");

    const auto& all = session.value("all_logits");
    const std::array<float, 8> expectedAll {
        1.0f, 2.0f, 3.0f, 6.0f,
        -1.0f, 0.5f, 2.0f, 1.5f};
    if (all.size() != expectedAll.size() * sizeof(std::uint16_t))
        throw std::logic_error("all-token BF16 logits size mismatch");
    for (std::size_t index = 0; index < expectedAll.size(); ++index)
        requireEqual(
            readBf16(all, index), expectedAll[index],
            "all-token BF16 logit mismatch");

    const ModelSessionStats& stats = session.stats();
    if (stats.host_operations != 2 || stats.host_uploads != 0
        || stats.host_downloads != 0 || stats.device_aliases != 0
        || stats.device_copies != 0)
        throw std::logic_error(
            "standalone host LM head unexpectedly touched the device");

    std::cout << "host_lm_head_test passed"
              << " last_logits=[-1,0.5,2,1.5]"
              << " all_token_logits=8"
              << " host_operations=" << stats.host_operations << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "host_lm_head_test failed: "
              << exception.what() << '\n';
    return 1;
}
