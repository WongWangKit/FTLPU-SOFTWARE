#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

float readBf16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(data[2 * index])
        | (static_cast<std::uint16_t>(data[2 * index + 1]) << 8))
        .to_float();
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: hf_decoder_layer_model_session_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ModelSession session(*system);
    session.load_file(std::filesystem::path(argv[1]));
    session.set_input("hidden.0", session.value("golden.input"));
    session.run();

    const auto& actual = session.value("hidden.1");
    const auto& expected = session.value("golden.output");
    if (actual.size() != expected.size())
        throw std::logic_error("decoder layer result size mismatch");
    float maximumError = 0.0f;
    double meanError = 0.0;
    std::size_t maximumIndex = 0;
    for (std::size_t index = 0; index < actual.size() / 2; ++index) {
        const float actualValue = readBf16(actual, index);
        const float expectedValue = readBf16(expected, index);
        if (!std::isfinite(actualValue)
            || !std::isfinite(expectedValue))
            throw std::logic_error(
                "HF decoder layer produced a non-finite value index="
                + std::to_string(index)
                + " actual=" + std::to_string(actualValue)
                + " expected=" + std::to_string(expectedValue));
        const float error = std::fabs(actualValue - expectedValue);
        meanError += error;
        if (error > maximumError) {
            maximumError = error;
            maximumIndex = index;
        }
    }
    meanError /= actual.size() / 2;
    if (!std::isfinite(maximumError) || !std::isfinite(meanError)
        || maximumError > 0.35f
        || meanError > 0.035)
        throw std::logic_error(
            "HF decoder layer golden mismatch max_error="
            + std::to_string(maximumError)
            + " mean_error=" + std::to_string(meanError)
            + " index=" + std::to_string(maximumIndex));
    std::cout << "hf_decoder_layer_model_session_test passed max_error="
              << maximumError << " mean_error=" << meanError << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_decoder_layer_model_session_test failed: "
              << exception.what() << '\n';
    return 1;
}
