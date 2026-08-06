#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void appendBf16(std::vector<std::uint8_t>& data, float value)
{
    const std::uint16_t bits = ftlpu::Bf16::from_float(value).bits();
    data.push_back(static_cast<std::uint8_t>(bits));
    data.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float readBf16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(data[index * 2])
        | (static_cast<std::uint16_t>(data[index * 2 + 1]) << 8))
        .to_float();
}

} // namespace

int main()
try {
    using namespace ftlpu::software::runtime;
    ModelPackage package;
    package.model_name = "embedding-test";
    package.architecture = "LlamaForCausalLM";
    std::vector<std::uint8_t> table;
    for (float value : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
        appendBf16(table, value);
    package.tensors.push_back(ModelTensor {
        "model.embed_tokens.weight", BindingElementType::BF16, {3, 2},
        table});
    package.values = {
        {"token_ids", BindingElementType::I32, {2}, true, false},
        {"hidden.0", BindingElementType::BF16, {2, 2}, false, true},
        {"logits", BindingElementType::F32, {1, 3}, false, true},
    };
    package.embedding_lookups.push_back(
        {"embedding", "token_ids", "model.embed_tokens.weight", "hidden.0"});
    package.host_lm_heads.push_back(
        {"lm_head", "hidden.0", "model.embed_tokens.weight", "logits", true});
    package.executables.push_back({"unused", BinaryProgram {}, {}});

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ModelSession session(*system);
    session.load(std::move(package));
    const std::array<std::int32_t, 2> ids {2, 0};
    session.set_input("token_ids",
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(ids.data()),
            sizeof(ids)));
    session.run();

    const auto& hidden = session.value("hidden.0");
    if (hidden.size() != 8 || readBf16(hidden, 0) != 5.0f
        || readBf16(hidden, 1) != 6.0f
        || readBf16(hidden, 2) != 1.0f
        || readBf16(hidden, 3) != 2.0f)
        throw std::logic_error("host embedding lookup produced wrong values");
    const auto& logits = session.value("logits");
    const auto readFloat = [&](std::size_t index) {
        float value = 0.0f;
        std::memcpy(&value, logits.data() + index * sizeof(float),
            sizeof(value));
        return value;
    };
    if (logits.size() != 3 * sizeof(float)
        || readFloat(0) != 5.0f || readFloat(1) != 11.0f
        || readFloat(2) != 17.0f)
        throw std::logic_error("host tied-weight LM head produced wrong logits");
    if (session.stats().host_uploads != 0
        || session.stats().host_downloads != 0
        || session.stats().host_operations != 2)
        throw std::logic_error("embedding-only session touched the device");

    std::cout << "model_session_embedding_test passed\n";
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "model_session_embedding_test failed: "
              << exception.what() << '\n';
    return 1;
}
