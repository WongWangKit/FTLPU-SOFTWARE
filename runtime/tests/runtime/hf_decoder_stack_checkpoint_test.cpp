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
#include <utility>
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
    std::size_t maximumToleranceIndex{0};
    float actualAtMaximumTolerance{0.0f};
    float expectedAtMaximumTolerance{0.0f};
    float toleranceAtMaximumTolerance{0.0f};
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
        const float toleranceRatio = error / tolerance;
        result.mean += error;
        if (toleranceRatio > result.maximumToleranceRatio) {
            result.maximumToleranceRatio = toleranceRatio;
            result.maximumToleranceIndex = index;
            result.actualAtMaximumTolerance = actualValue;
            result.expectedAtMaximumTolerance = expectedValue;
            result.toleranceAtMaximumTolerance = tolerance;
        }
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

std::pair<std::string, std::string> restartBoundary(
    const ftlpu::software::runtime::ModelInvocation& invocation)
{
    using namespace ftlpu::software::runtime;
    const auto input = std::find_if(invocation.inputs.begin(),
        invocation.inputs.end(), [](const ModelBindingRef& candidate) {
            return candidate.value.starts_with("hidden.");
        });
    if (input == invocation.inputs.end())
        throw std::logic_error(
            "checkpoint invocation has no restartable hidden input");
    return { input->value, input->value == "hidden.0"
            ? "golden.input"
            : "golden." + input->value };
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: hf_decoder_stack_checkpoint_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    auto system = std::make_unique<ftlpu::C2cDmaSystem>();
    {
        ModelSession session(*system);
        session.load_file(std::filesystem::path(argv[1]));

        bool passed = true;
        const bool reportProgress
            = std::getenv("FTLPU_CHECKPOINT_PROGRESS") != nullptr;
        const bool chained
            = std::getenv("FTLPU_CHECKPOINT_CHAINED") != nullptr;
        const std::size_t checkpointInvocations
            = session.package().invocations.size();
        std::size_t firstLayer = 0;
        if (const char* value = std::getenv("FTLPU_CHECKPOINT_START_LAYER"))
            firstLayer = static_cast<std::size_t>(std::stoull(value));
        if (firstLayer >= checkpointInvocations)
            throw std::out_of_range("checkpoint start layer is out of range");
        std::size_t lastLayer = checkpointInvocations;
        if (const char* value = std::getenv("FTLPU_CHECKPOINT_LAYER_COUNT")) {
            const std::size_t layerCount
                = static_cast<std::size_t>(std::stoull(value));
            if (layerCount == 0)
                throw std::invalid_argument(
                    "checkpoint layer count must be non-zero");
            lastLayer
                = std::min(checkpointInvocations, firstLayer + layerCount);
        }
        for (std::size_t index = firstLayer; index < lastLayer; ++index) {
            if (!chained || index == firstLayer) {
                const auto [hidden, golden] = restartBoundary(
                    session.package().invocations[index]);
                session.set_input(hidden, session.value(golden));
            }
            if (reportProgress)
                std::clog << "checkpoint layer=" << index
                          << " entering invocation" << std::endl;
            session.run_invocation(index);
            if (reportProgress)
                std::clog << "checkpoint layer=" << index
                          << " invocation returned" << std::endl;
            const ModelInvocation& invocation
                = session.package().invocations[index];
            if (invocation.outputs.size() != 1)
                throw std::logic_error(
                    "checkpoint invocation must have one model output");
            const std::string& hidden = invocation.outputs.front().value;
            const std::string golden = hidden == "final_hidden"
                ? "golden.final_hidden"
                : "golden." + hidden;
            const auto& actual = session.value(hidden);
            const auto& expected = session.value(golden);
            if (reportProgress)
                std::clog << "checkpoint layer=" << index
                          << " comparing actual_bytes=" << actual.size()
                          << " expected_bytes=" << expected.size() << std::endl;
            const Error error = compare(actual, expected);
            std::cout << "checkpoint layer=" << index
                      << " max_error=" << error.maximum
                      << " mean_error=" << error.mean
                      << " index=" << error.maximumIndex
                      << " actual=" << error.actualAtMaximum
                      << " expected=" << error.expectedAtMaximum
                      << " max_tolerance_ratio=" << error.maximumToleranceRatio
                      << " tolerance_index=" << error.maximumToleranceIndex
                      << " tolerance_actual=" << error.actualAtMaximumTolerance
                      << " tolerance_expected="
                      << error.expectedAtMaximumTolerance
                      << " tolerance=" << error.toleranceAtMaximumTolerance
                      << " tolerance_violations=" << error.toleranceViolations
                      << std::endl;
            const bool checkpointPassed
                = error.toleranceViolations == 0 && error.mean <= 0.04;
            passed &= checkpointPassed;
            if (!checkpointPassed
                && std::getenv("FTLPU_CHECKPOINT_FAIL_FAST") != nullptr)
                break;
        }
        if (!passed)
            throw std::logic_error(
                "one or more decoder checkpoints exceeded tolerance");
        std::cout << "hf_decoder_stack_checkpoint_test passed layers="
                  << (lastLayer - firstLayer) << " range=[" << firstLayer << ','
                  << lastLayer << ") mode="
                  << (chained ? "chained" : "independent") << std::endl;
    }
    if (std::getenv("FTLPU_CHECKPOINT_PROGRESS") != nullptr)
        std::clog << "checkpoint session destroyed" << std::endl;
    system.reset();
    if (std::getenv("FTLPU_CHECKPOINT_PROGRESS") != nullptr)
        std::clog << "checkpoint system destroyed" << std::endl;
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_decoder_stack_checkpoint_test failed: " << exception.what()
              << '\n';
    return 1;
}
