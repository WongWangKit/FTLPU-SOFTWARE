#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

float readBf16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(data[2 * index])
        | (static_cast<std::uint16_t>(data[2 * index + 1]) << 8))
        .to_float();
}

float bf16Tolerance(float expected)
{
    const float magnitude = std::fabs(expected);
    const std::uint16_t bits =
        ftlpu::Bf16::from_float(magnitude).bits();
    const float next =
        ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(bits + 1))
            .to_float();
    return std::max(0.25f, 4.0f * (next - magnitude));
}

std::vector<std::size_t> topKIndices(
    const std::vector<float>& values, std::size_t count)
{
    std::vector<std::size_t> indices(values.size());
    for (std::size_t index = 0; index < indices.size(); ++index)
        indices[index] = index;
    count = std::min(count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count,
        indices.end(), [&](std::size_t lhs, std::size_t rhs) {
            return values[lhs] > values[rhs];
        });
    indices.resize(count);
    return indices;
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: hf_model_boundaries_session_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ModelSession session(*system);
    if (std::getenv("FTLPU_SESSION_PROGRESS"))
        std::clog << "FTLPU session loading model package" << std::endl;
    session.load_file(std::filesystem::path(argv[1]));

    const auto& package = session.package();
    if (package.values.empty())
        throw std::logic_error("model package has no values");
    std::size_t sequence_length = 0;
    for (const ModelValue& value : package.values)
        if (value.name == "token_ids" && value.shape.size() == 1)
            sequence_length = static_cast<std::size_t>(value.shape[0]);
    if (sequence_length == 0)
        throw std::logic_error("model package has no token_ids input");
    std::vector<std::int32_t> token_ids(sequence_length);
    for (std::size_t index = 0; index < sequence_length; ++index)
        token_ids[index] = static_cast<std::int32_t>(index);
    session.set_input("token_ids",
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(token_ids.data()),
            token_ids.size() * sizeof(token_ids[0])));
    session.run();

    const auto& decoderGolden = session.value("golden.output");
    const auto& finalNormWeight = session.value("model.norm.weight");
    const auto& finalHidden = session.value("final_hidden");
    constexpr std::size_t hiddenSize = 576;
    if (decoderGolden.size() != sequence_length * hiddenSize * 2
        || finalNormWeight.size() != hiddenSize * 2
        || finalHidden.size() != decoderGolden.size())
        throw std::logic_error("unexpected full-prefill BF16 tensor size");
    float maximumNormError = 0.0f;
    double meanNormError = 0.0;
    float maximumNormToleranceRatio = 0.0f;
    std::size_t normToleranceViolations = 0;
    double normSquaredError = 0.0;
    double normActualSquared = 0.0;
    double normExpectedSquared = 0.0;
    double normDot = 0.0;
    std::vector<float> expectedLastHidden(hiddenSize);
    for (std::size_t row = 0; row < sequence_length; ++row) {
        double sumSquares = 0.0;
        for (std::size_t column = 0; column < hiddenSize; ++column) {
            const float value =
                readBf16(decoderGolden, row * hiddenSize + column);
            sumSquares += static_cast<double>(value) * value;
        }
        const float inverseRms = 1.0f / std::sqrt(
            static_cast<float>(sumSquares / hiddenSize) + 1.0e-5f);
        for (std::size_t column = 0; column < hiddenSize; ++column) {
            const std::size_t index = row * hiddenSize + column;
            const float expected = ftlpu::Bf16::from_float(
                readBf16(decoderGolden, index) * inverseRms
                * readBf16(finalNormWeight, column))
                                       .to_float();
            const float actual = readBf16(finalHidden, index);
            if (!std::isfinite(actual))
                throw std::logic_error(
                    "LPU final RMSNorm produced a non-finite value");
            const float error = std::fabs(actual - expected);
            maximumNormError = std::max(maximumNormError, error);
            meanNormError += error;
            normSquaredError += static_cast<double>(error) * error;
            normActualSquared += static_cast<double>(actual) * actual;
            normExpectedSquared += static_cast<double>(expected) * expected;
            normDot += static_cast<double>(actual) * expected;
            const float tolerance = bf16Tolerance(expected);
            maximumNormToleranceRatio = std::max(
                maximumNormToleranceRatio, error / tolerance);
            normToleranceViolations += error > tolerance;
            if (row + 1 == sequence_length)
                expectedLastHidden[column] = expected;
        }
    }
    meanNormError /= sequence_length * hiddenSize;
    const double normRelativeL2 =
        std::sqrt(normSquaredError / normExpectedSquared);
    const double normCosine = normDot
        / std::sqrt(normActualSquared * normExpectedSquared);
    const double normViolationFraction =
        static_cast<double>(normToleranceViolations)
        / (sequence_length * hiddenSize);

    const auto& logits = session.value("logits");
    const auto& embedding = session.value("model.embed_tokens.weight");
    if (logits.size() != 49152 * sizeof(float))
        throw std::logic_error("unexpected SmolLM2 logits size");
    if (embedding.size() != 49152 * hiddenSize * 2)
        throw std::logic_error("unexpected SmolLM2 embedding size");
    float minimum = 0.0f;
    float maximum = 0.0f;
    bool initialized = false;
    std::vector<float> actualLogits(logits.size() / sizeof(float));
    std::vector<float> expectedLogits(actualLogits.size());
    double logitActualSquared = 0.0;
    double logitExpectedSquared = 0.0;
    double logitDot = 0.0;
    for (std::size_t index = 0; index < logits.size() / sizeof(float);
         ++index) {
        float value = 0.0f;
        std::memcpy(&value, logits.data() + index * sizeof(float),
            sizeof(value));
        if (!std::isfinite(value))
            throw std::logic_error("host LM head produced non-finite logits");
        actualLogits[index] = value;
        float expected = 0.0f;
        for (std::size_t column = 0; column < hiddenSize; ++column)
            expected += expectedLastHidden[column]
                * readBf16(embedding, index * hiddenSize + column);
        expectedLogits[index] = expected;
        logitActualSquared += static_cast<double>(value) * value;
        logitExpectedSquared += static_cast<double>(expected) * expected;
        logitDot += static_cast<double>(value) * expected;
        if (!initialized) {
            minimum = maximum = value;
            initialized = true;
        } else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    const double logitCosine = logitDot
        / std::sqrt(logitActualSquared * logitExpectedSquared);
    const auto actualTop5 = topKIndices(actualLogits, 5);
    const auto expectedTop5 = topKIndices(expectedLogits, 5);
    const std::size_t top5Overlap = std::count_if(
        expectedTop5.begin(), expectedTop5.end(),
        [&](std::size_t index) {
            return std::find(actualTop5.begin(), actualTop5.end(), index)
                != actualTop5.end();
        });
    if (maximum - minimum < 1.0e-3f)
        throw std::logic_error("host LM head logits are constant");
    if (session.stats().host_operations != 2)
        throw std::logic_error(
            "model did not execute host embedding and host LM head");
    if (meanNormError > 0.04 || normRelativeL2 > 0.08
        || normCosine < 0.995 || normViolationFraction > 0.005
        || maximumNormToleranceRatio > 8.0f
        || logitCosine < 0.995 || top5Overlap < 3)
        throw std::logic_error(
            "full-prefill numerical golden mismatch max_error="
            + std::to_string(maximumNormError)
            + " mean_error=" + std::to_string(meanNormError)
            + " relative_l2=" + std::to_string(normRelativeL2)
            + " cosine=" + std::to_string(normCosine)
            + " max_tolerance_ratio="
            + std::to_string(maximumNormToleranceRatio)
            + " violation_fraction="
            + std::to_string(normViolationFraction)
            + " logit_cosine=" + std::to_string(logitCosine)
            + " top5_overlap=" + std::to_string(top5Overlap));

    std::cout << "hf_model_boundaries_session_test passed"
              << " logits=" << logits.size() / sizeof(float)
              << " min=" << minimum << " max=" << maximum
              << " final_norm_max_error=" << maximumNormError
              << " final_norm_mean_error=" << meanNormError
              << " final_norm_max_tolerance_ratio="
              << maximumNormToleranceRatio
              << " final_norm_tolerance_violations="
              << normToleranceViolations
              << " final_norm_relative_l2=" << normRelativeL2
              << " final_norm_cosine=" << normCosine
              << " final_norm_violation_fraction="
              << normViolationFraction
              << " logit_cosine=" << logitCosine
              << " expected_top1=" << expectedTop5.front()
              << " actual_top1=" << actualTop5.front()
              << " top5_overlap=" << top5Overlap
              << " host_operations=" << session.stats().host_operations
              << " device_aliases=" << session.stats().device_aliases
              << " device_copies=" << session.stats().device_copies << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_model_boundaries_session_test failed: "
              << exception.what() << '\n';
    return 1;
}
