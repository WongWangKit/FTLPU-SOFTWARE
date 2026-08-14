#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5f;

float toBf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

float inputValue(std::size_t row, std::size_t column)
{
    const int value =
        static_cast<int>((row * 13 + column * 7) % 31) - 15;
    return toBf16(static_cast<float>(value) / 32.0f);
}

float gammaValue(std::size_t column)
{
    return toBf16(0.75f
        + static_cast<float>(column % 9) / 32.0f);
}

void appendBf16(std::vector<std::uint8_t>& bytes, float value)
{
    const std::uint16_t bits = ftlpu::Bf16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float readBf16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8))
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
    if (program.bindings.size() != 3 || program.max_cycle < 100)
        throw std::logic_error(
            "RMSNorm binary is missing bindings or scheduled commands");
    const auto& activation = program.bindings[0];
    if (activation.shape.size() != 2)
        throw std::logic_error("RMSNorm activation binding must be rank 2");
    const std::size_t rows = activation.shape[0];
    const std::size_t hidden = activation.shape[1];

    std::vector<std::uint8_t> input;
    input.reserve(rows * hidden * 2);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < hidden; ++column)
            appendBf16(input, inputValue(row, column));

    std::vector<std::uint8_t> gamma;
    gamma.reserve(hidden * 2);
    for (std::size_t column = 0; column < hidden; ++column)
        appendBf16(gamma, gammaValue(column));

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, input);
    runtime.upload_input(1, gamma);
    auto trace = std::ofstream {};
    std::ostream* traceStream = nullptr;
    if (const char* path = std::getenv("FTLPU_RMSNORM_TRACE")) {
        trace.open(path, std::ios::trunc);
        traceStream = &trace;
    }
    runtime.run_cycles(program.max_cycle + 64, traceStream);
    const auto output = runtime.download_output(0);

    std::size_t checked = 0;
    std::size_t nonzero = 0;
    float maxError = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        float meanSquare = 0.0f;
        for (std::size_t column = 0; column < hidden; ++column) {
            const float value = inputValue(row, column);
            meanSquare += feedback
                ? value * value / static_cast<float>(hidden)
                : toBf16(value * value)
                    * toBf16(1.0f / hidden);
        }
        const float factor = feedback
            ? 1.0f / std::sqrt(meanSquare + kEpsilon)
            : toBf16(1.0f / std::sqrt(meanSquare + kEpsilon));
        for (std::size_t column = 0; column < hidden; ++column) {
            const float expected = toBf16(
                inputValue(row, column) * factor * gammaValue(column));
            const float actual =
                readBf16(output, row * hidden + column);
            const float error = std::fabs(actual - expected);
            maxError = std::max(maxError, error);
            if (std::fabs(actual) > 1.0e-5f) ++nonzero;
            if (error > 8.0e-3f) {
                throw std::logic_error(
                    "RMSNorm CPU baseline mismatch at row="
                    + std::to_string(row) + " column="
                    + std::to_string(column) + " actual="
                    + std::to_string(actual) + " expected="
                    + std::to_string(expected));
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
