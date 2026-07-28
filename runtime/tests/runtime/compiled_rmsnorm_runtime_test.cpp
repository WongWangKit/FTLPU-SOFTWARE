#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kHidden = 576;
constexpr float kEpsilon = 1.0e-5f;

float toFp16(float value)
{
    return ftlpu::Fp16::from_float(value).to_float();
}

float inputValue(std::size_t row, std::size_t column)
{
    const int value =
        static_cast<int>((row * 13 + column * 7) % 31) - 15;
    return toFp16(static_cast<float>(value) / 32.0f);
}

float gammaValue(std::size_t column)
{
    return toFp16(0.75f
        + static_cast<float>(column % 9) / 32.0f);
}

void appendFp16(std::vector<std::uint8_t>& bytes, float value)
{
    const std::uint16_t bits = ftlpu::Fp16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float readFp16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8))
        .to_float();
}

float readMemFp16(const ftlpu::TspSliceSystem& system,
    ftlpu::Hemisphere hemisphere, std::size_t lowSlice,
    std::size_t highSlice, std::size_t address, std::size_t lane)
{
    const auto tile = lane / ftlpu::hw::kLanesPerTile;
    const auto localLane = lane % ftlpu::hw::kLanesPerTile;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, lowSlice, tile, address, localLane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, highSlice, tile, address, localLane);
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8))
        .to_float();
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 3)
        throw std::runtime_error(
            "usage: compiled_rmsnorm_runtime_test program.ftlpu strategy");
    const bool feedback = std::string_view(argv[2]) == "vxm-feedback";
    const auto program =
        ftlpu::software::runtime::read_binary_program(
            std::filesystem::path(argv[1]));
    if (program.bindings.size() != 3
        || program.max_cycle < (feedback ? 5000 : 7000))
        throw std::logic_error(
            "RMSNorm binary is missing bindings or scheduled commands");

    std::vector<std::uint8_t> input;
    input.reserve(kRows * kHidden * 2);
    for (std::size_t row = 0; row < kRows; ++row)
        for (std::size_t column = 0; column < kHidden; ++column)
            appendFp16(input, inputValue(row, column));

    std::vector<std::uint8_t> gamma;
    gamma.reserve(kHidden * 2);
    for (std::size_t column = 0; column < kHidden; ++column)
        appendFp16(gamma, gammaValue(column));

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, input);
    runtime.upload_input(1, gamma);
    runtime.run_cycles(program.max_cycle + 64);
    const auto output = runtime.download_output(0);

        std::size_t checked = 0;
    std::size_t nonzero = 0;
    float maxError = 0.0f;
    for (std::size_t row = 0; row < kRows; ++row) {
        float meanSquare = 0.0f;
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float value = inputValue(row, column);
            meanSquare += feedback
                ? value * value / static_cast<float>(kHidden)
                : toFp16(value * value)
                    * toFp16(1.0f / kHidden);
        }
        const float factor = feedback
            ? 1.0f / std::sqrt(meanSquare + kEpsilon)
            : toFp16(1.0f / std::sqrt(meanSquare + kEpsilon));
        const float physicalFactor = feedback
            ? 0.0f
            : readMemFp16(*system, ftlpu::Hemisphere::East,
                22, 23, row, 0);
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float expected = toFp16(
                inputValue(row, column) * factor * gammaValue(column));
            const float actual =
                readFp16(output, row * kHidden + column);
            const float error = std::fabs(actual - expected);
            maxError = std::max(maxError, error);
            if (std::fabs(actual) > 1.0e-5f) ++nonzero;
            if (error > 8.0e-3f)
            {
                float feedbackMean = 0.0f;
                float physicalFeedbackMean = 0.0f;
                float feedbackMaxError = 0.0f;
                std::size_t feedbackNonzero = 0;
                std::string feedbackZeroIndices;
                std::string feedbackSamples;
                std::string feedbackFirstMismatch;
                std::string feedbackInputFirstMismatch;
                std::string reverseMatches;
                for (std::size_t index = 0; index < kHidden; ++index) {
                    const float feedbackInput = readMemFp16(*system,
                        ftlpu::Hemisphere::East, 36, 37,
                        (row / 32) * kHidden + index, row % 32);
                    const float feedbackWeight = readMemFp16(*system,
                        ftlpu::Hemisphere::East, 10, 11,
                        index, row % 32);
                    const float feedbackOutput = readMemFp16(*system,
                        ftlpu::Hemisphere::East, 22, 23,
                        (row / 32) * kHidden + index, row % 32);
                    feedbackMean += inputValue(row, index)
                        * inputValue(row, index)
                        / static_cast<float>(kHidden);
                    physicalFeedbackMean += feedbackInput * feedbackInput
                        / static_cast<float>(kHidden);
                    if (feedbackInputFirstMismatch.empty()
                        && feedbackInput != inputValue(row, index))
                        feedbackInputFirstMismatch = std::to_string(index)
                            + ":" + std::to_string(feedbackInput)
                            + "/" + std::to_string(inputValue(row, index));
                    if (feedbackInput != 0.0f) {
                        ++feedbackNonzero;
                    } else if (feedbackZeroIndices.size() < 80) {
                        feedbackZeroIndices += std::to_string(index) + ",";
                    }
                    const float feedbackExpected = toFp16(
                        inputValue(row, index) * factor
                        * gammaValue(index));
                    feedbackMaxError = std::max(feedbackMaxError,
                        std::fabs(feedbackOutput - feedbackExpected));
                    if (feedbackFirstMismatch.empty()
                        && std::fabs(
                            feedbackOutput - feedbackExpected) > 8.0e-3f)
                        feedbackFirstMismatch = std::to_string(index)
                            + ":" + std::to_string(feedbackOutput)
                            + "/" + std::to_string(feedbackExpected);
                    if (index < 8) {
                        feedbackSamples += "["
                            + std::to_string(feedbackInput) + ","
                            + std::to_string(feedbackWeight) + ","
                            + std::to_string(feedbackOutput) + ","
                            + std::to_string(feedbackExpected) + "]";
                    }
                }
                for (std::size_t feedbackLane = 0;
                     feedbackLane < 32 && reverseMatches.size() < 120;
                     ++feedbackLane) {
                    for (std::size_t index = 0;
                         index < kHidden && reverseMatches.size() < 120;
                         ++index) {
                        const float candidate = readMemFp16(*system,
                            ftlpu::Hemisphere::East, 22, 23,
                            index, feedbackLane);
                        if (std::fabs(candidate - actual) < 1.0e-4f)
                            reverseMatches += std::to_string(feedbackLane)
                                + ":" + std::to_string(index) + ",";
                    }
                }
                throw std::logic_error(
                    "RMSNorm CPU baseline mismatch at row="
                    + std::to_string(row) + " column="
                    + std::to_string(column) + " actual="
                    + std::to_string(actual) + " expected="
                    + std::to_string(expected) + " factor="
                    + std::to_string(physicalFactor) + " expected_factor="
                    + std::to_string(factor) + " mean="
                    + std::to_string(meanSquare) + " weight00="
                    + std::to_string(static_cast<float>(
                        system->mxm_unit(0).array().weight(
                            0, 0, 0, 0, 0)))
                    + " feedback_mean="
                    + std::to_string(feedbackMean)
                    + " physical_feedback_mean="
                    + std::to_string(physicalFeedbackMean)
                    + " feedback_output_max_error="
                    + std::to_string(feedbackMaxError)
                    + " feedback_nonzero="
                    + std::to_string(feedbackNonzero)
                    + " feedback_zero_indices="
                    + feedbackZeroIndices
                    + " feedback_samples="
                    + feedbackSamples
                    + " feedback_first_mismatch="
                    + feedbackFirstMismatch
                    + " feedback_input_first_mismatch="
                    + feedbackInputFirstMismatch
                    + " reverse_matches="
                    + reverseMatches);
            }
            ++checked;
        }
    }
    if (nonzero == 0)
        throw std::logic_error("RMSNorm produced only zero output values");
    std::cout << "RMSNorm StableHLO -> binary -> runtime -> CModel passed: "
              << checked << " values, nonzero=" << nonzero
              << ", max_error=" << maxError
              << ", strategy=" << argv[2]
              << ", max_cycle=" << program.max_cycle << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_rmsnorm_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
