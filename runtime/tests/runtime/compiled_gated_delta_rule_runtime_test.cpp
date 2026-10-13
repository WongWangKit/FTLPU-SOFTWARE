#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kHeads = 2;
constexpr std::size_t kKeyDim = 32;
constexpr std::size_t kValueDim = 32;

float bf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

std::vector<std::uint8_t> encodeBf16(const std::vector<float>& values)
{
    std::vector<std::uint8_t> result(values.size() * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto bits = ftlpu::Bf16::from_float(values[i]).bits();
        result[2 * i] = static_cast<std::uint8_t>(bits);
        result[2 * i + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    return result;
}

std::vector<std::uint8_t> encodeF32(const std::vector<float>& values)
{
    std::vector<std::uint8_t> result(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto bits = std::bit_cast<std::uint32_t>(values[i]);
        for (std::size_t byte = 0; byte < 4; ++byte)
            result[4 * i + byte] =
                static_cast<std::uint8_t>(bits >> (8 * byte));
    }
    return result;
}

float decodeBf16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const auto bits = static_cast<std::uint16_t>(bytes[2 * index])
        | static_cast<std::uint16_t>(bytes[2 * index + 1]) << 8;
    return ftlpu::Bf16::from_bits(bits).to_float();
}

float decodeF32(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    std::uint32_t bits = 0;
    for (std::size_t byte = 0; byte < 4; ++byte)
        bits |= static_cast<std::uint32_t>(bytes[4 * index + byte])
            << (8 * byte);
    return std::bit_cast<float>(bits);
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu::software::runtime;
    if (argc != 2 && argc != 3)
        throw std::runtime_error(
            "usage: compiled_gated_delta_rule_runtime_test program.ftlpu [tokens]");
    const std::size_t tokens = argc == 3
        ? static_cast<std::size_t>(std::stoull(argv[2])) : 1;
    const auto program = read_binary_program(std::filesystem::path(argv[1]));

    std::vector<float> query(tokens * kHeads * kKeyDim);
    std::vector<float> key(query.size());
    std::vector<float> value(tokens * kHeads * kValueDim);
    std::vector<float> decay(tokens * kHeads);
    std::vector<float> beta(tokens * kHeads);
    std::vector<float> state(kHeads * kKeyDim * kValueDim);
    for (std::size_t i = 0; i < query.size(); ++i) {
        query[i] = bf16((static_cast<int>(i % 13) - 6) / 64.0f);
        key[i] = bf16((static_cast<int>((i * 3) % 11) - 5) / 64.0f);
    }
    for (std::size_t i = 0; i < value.size(); ++i)
        value[i] = bf16((static_cast<int>((i * 5) % 17) - 8) / 32.0f);
    for (std::size_t i = 0; i < decay.size(); ++i) {
        decay[i] = 0.97f - static_cast<float>(i % 3) * 0.01f;
        beta[i] = 0.125f + static_cast<float>(i % 2) * 0.0625f;
    }
    for (std::size_t i = 0; i < state.size(); ++i)
        state[i] = (static_cast<int>(i % 7) - 3) / 256.0f;

    auto expectedState = state;
    std::vector<float> expectedOutput(value.size());
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < kHeads; ++head) {
            const float a = decay[token * kHeads + head];
            const float b = beta[token * kHeads + head];
            std::vector<float> remembered(kValueDim, 0.0f);
            for (std::size_t k = 0; k < kKeyDim; ++k) {
                const auto stateRow = (head * kKeyDim + k) * kValueDim;
                const float keyValue =
                    key[(token * kHeads + head) * kKeyDim + k];
                for (std::size_t v = 0; v < kValueDim; ++v) {
                    expectedState[stateRow + v] *= a;
                    remembered[v] += expectedState[stateRow + v] * keyValue;
                }
            }
            for (std::size_t v = 0; v < kValueDim; ++v) {
                const float delta =
                    (value[(token * kHeads + head) * kValueDim + v]
                        - remembered[v]) * b;
                float output = 0.0f;
                for (std::size_t k = 0; k < kKeyDim; ++k) {
                    const auto stateIndex =
                        (head * kKeyDim + k) * kValueDim + v;
                    const auto vectorIndex =
                        (token * kHeads + head) * kKeyDim + k;
                    expectedState[stateIndex] += key[vectorIndex] * delta;
                    output += expectedState[stateIndex] * query[vectorIndex];
                }
                expectedOutput[(token * kHeads + head) * kValueDim + v]
                    = bf16(output);
            }
        }
    }

    auto* system = new ftlpu::TspSliceSystem();
    CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, encodeBf16(query));
    runtime.upload_input(1, encodeBf16(key));
    runtime.upload_input(2, encodeBf16(value));
    runtime.upload_input(3, encodeF32(decay));
    runtime.upload_input(4, encodeF32(beta));
    runtime.upload_input(5, encodeF32(state));
    runtime.run_cycles(program.max_cycle + 128);
    const auto output = runtime.download_output(0);
    const auto finalState = runtime.download_output(1);
    float maxOutputError = 0.0f;
    float maxStateError = 0.0f;
    std::size_t maxOutputIndex = 0;
    std::size_t maxStateIndex = 0;
    for (std::size_t i = 0; i < expectedOutput.size(); ++i) {
        const float error =
            std::fabs(decodeBf16(output, i) - expectedOutput[i]);
        if (error > maxOutputError) {
            maxOutputError = error;
            maxOutputIndex = i;
        }
    }
    for (std::size_t i = 0; i < expectedState.size(); ++i) {
        const float error =
            std::fabs(decodeF32(finalState, i) - expectedState[i]);
        if (error > maxStateError) {
            maxStateError = error;
            maxStateIndex = i;
        }
    }
    if (maxOutputError > 0.03f || maxStateError > 0.01f)
        throw std::logic_error(
            "gated delta-rule mismatch output_error="
            + std::to_string(maxOutputError)
            + " output_index=" + std::to_string(maxOutputIndex)
            + " output_actual="
            + std::to_string(decodeBf16(output, maxOutputIndex))
            + " output_expected=" + std::to_string(expectedOutput[maxOutputIndex])
            + " state_error=" + std::to_string(maxStateError)
            + " state_index=" + std::to_string(maxStateIndex)
            + " state_actual="
            + std::to_string(decodeF32(finalState, maxStateIndex))
            + " state_expected=" + std::to_string(expectedState[maxStateIndex]));
    std::cout << "compiled_gated_delta_rule_runtime_test passed output_error="
              << maxOutputError << " output_index=" << maxOutputIndex
              << " output_actual=" << decodeBf16(output, maxOutputIndex)
              << " output_expected=" << expectedOutput[maxOutputIndex]
              << " state_error=" << maxStateError
              << " state_index=" << maxStateIndex
              << " state_actual=" << decodeF32(finalState, maxStateIndex)
              << " state_expected=" << expectedState[maxStateIndex] << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "compiled_gated_delta_rule_runtime_test failed: "
              << error.what() << '\n';
    return 1;
}
