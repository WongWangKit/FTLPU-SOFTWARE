#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float readBf16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(data[2 * index])
        | (static_cast<std::uint16_t>(data[2 * index + 1]) << 8))
        .to_float();
}

struct Error {
    float maximum{0.0f};
    double mean{0.0};
    std::size_t maximumIndex{0};
    float actualAtMaximum{0.0f};
    float expectedAtMaximum{0.0f};
    float maximumToleranceRatio{0.0f};
    std::size_t toleranceViolations{0};
};

float bf16Tolerance(float expected)
{
    const float magnitude = std::fabs(expected);
    const std::uint16_t bits =
        ftlpu::Bf16::from_float(magnitude).bits();
    const float next =
        ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(bits + 1))
            .to_float();
    const float ulp = next - magnitude;
    return std::max(0.25f, 4.0f * ulp);
}

Error compare(const std::vector<std::uint8_t>& actual,
    const std::vector<std::uint8_t>& expected)
{
    if (actual.size() != expected.size() || actual.size() % 2 != 0)
        throw std::logic_error("decoder checkpoint size mismatch");
    Error result;
    const std::size_t elements = actual.size() / 2;
    for (std::size_t index = 0; index < elements; ++index) {
        const float actualValue = readBf16(actual, index);
        const float expectedValue = readBf16(expected, index);
        if (!std::isfinite(actualValue) || !std::isfinite(expectedValue))
            throw std::logic_error(
                "decoder checkpoint contains a non-finite value");
        const float error = std::fabs(actualValue - expectedValue);
        const float tolerance = bf16Tolerance(expectedValue);
        result.mean += error;
        result.maximumToleranceRatio = std::max(
            result.maximumToleranceRatio, error / tolerance);
        result.toleranceViolations += error > tolerance;
        if (error > result.maximum) {
            result.maximum = error;
            result.maximumIndex = index;
            result.actualAtMaximum = actualValue;
            result.expectedAtMaximum = expectedValue;
        }
    }
    result.mean /= static_cast<double>(elements);
    return result;
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: hf_decoder_stack_checkpoint_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    auto system = std::make_unique<ftlpu::C2cDmaSystem>();
    ModelSession session(*system);
    session.load_file(std::filesystem::path(argv[1]));
    session.set_input("hidden.0", session.value("golden.input"));

    bool passed = true;
    const std::size_t decoderLayers =
        session.package().invocations.size();
    for (std::size_t index = 0; index < decoderLayers; ++index) {
        session.run_invocation(index);
        const std::string hidden =
            "hidden." + std::to_string(index + 1);
        const std::string golden =
            "golden.hidden." + std::to_string(index + 1);
        const Error error =
            compare(session.value(hidden), session.value(golden));
        std::cout << "checkpoint layer=" << index
                  << " max_error=" << error.maximum
                  << " mean_error=" << error.mean
                  << " index=" << error.maximumIndex
                  << " actual=" << error.actualAtMaximum
                  << " expected=" << error.expectedAtMaximum
                  << " max_tolerance_ratio="
                  << error.maximumToleranceRatio
                  << " tolerance_violations="
                  << error.toleranceViolations << std::endl;
        const bool checkpointPassed = error.toleranceViolations == 0
            && error.mean <= 0.04;
        passed &= checkpointPassed;
        if (!checkpointPassed
            && std::getenv("FTLPU_CHECKPOINT_FAIL_FAST") != nullptr)
            break;
    }
    if (!passed)
        throw std::logic_error(
            "one or more decoder checkpoints exceeded tolerance");
    std::cout << "hf_decoder_stack_checkpoint_test passed layers="
              << decoderLayers << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_decoder_stack_checkpoint_test failed: "
              << exception.what() << '\n';
    return 1;
}
