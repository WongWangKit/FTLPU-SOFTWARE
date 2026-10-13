#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kKernel = 4;

float bf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

std::vector<std::uint8_t> encodeBf16(const std::vector<float>& values)
{
    std::vector<std::uint8_t> bytes(values.size() * 2);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto bits = ftlpu::Bf16::from_float(values[index]).bits();
        bytes[2 * index] = static_cast<std::uint8_t>(bits);
        bytes[2 * index + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    return bytes;
}

float decodeBf16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const auto bits = static_cast<std::uint16_t>(bytes[2 * index])
        | static_cast<std::uint16_t>(bytes[2 * index + 1]) << 8;
    return ftlpu::Bf16::from_bits(bits).to_float();
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu::software::runtime;
    if (argc != 2 && argc != 4 && argc != 5)
        throw std::runtime_error(
            "usage: compiled_depthwise_conv1d_runtime_test program.ftlpu [tokens channels [none|silu]]");
    const std::size_t kTokens = argc >= 4 ? std::stoull(argv[2]) : 32;
    const std::size_t kChannels = argc >= 4 ? std::stoull(argv[3]) : 32;
    const bool silu = argc != 5 || std::string(argv[4]) == "silu";
    if (argc == 5 && !silu && std::string(argv[4]) != "none")
        throw std::runtime_error("activation must be none or silu");

    const auto program = read_binary_program(std::filesystem::path(argv[1]));
    std::vector<float> input(kTokens * kChannels);
    std::vector<float> weight(kChannels * kKernel);
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            const int value = static_cast<int>(
                (token * 3 + channel * 5) % 17) - 8;
            input[token * kChannels + channel] =
                bf16(static_cast<float>(value) / 32.0f);
        }
    }
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        constexpr float taps[kKernel] = {0.125f, -0.25f, 0.375f, 0.5f};
        const float scale = 1.0f
            + static_cast<float>(static_cast<int>(channel % 5) - 2) / 32.0f;
        for (std::size_t tap = 0; tap < kKernel; ++tap)
            weight[channel * kKernel + tap] = bf16(taps[tap] * scale);
    }

    // Keep the full-chip object alive until process exit; destroying its large
    // queue graph is disproportionately expensive on Windows test runners.
    auto* system = new ftlpu::TspSliceSystem();
    CModelRuntime runtime(*system);
    const char* tracePrefix = std::getenv("FTLPU_DEPTHWISE_TRACE_PREFIX");
    if (tracePrefix) {
        runtime.enable_execution_trace();
        runtime.enable_mem_execution_trace();
    }
    runtime.load(program);
    runtime.upload_input(0, encodeBf16(input));
    runtime.upload_input(1, encodeBf16(weight));
    if (std::getenv("FTLPU_DEPTHWISE_COLUMN_TRACE")) {
        for (std::size_t cycle = 0; cycle < program.max_cycle + 64; ++cycle) {
            runtime.run_cycles(1);
            const auto& executed = system->vxm_unit().last_execution();
            std::cout << "cycle=" << cycle << " columns=";
            for (std::size_t column = 0; column < 16; ++column)
                if (executed[0][column]) std::cout << column << ',';
            std::cout << '\n';
        }
    } else {
        runtime.run_cycles(program.max_cycle + 64,
            std::getenv("FTLPU_CMODEL_LOG") != nullptr ? &std::cout : nullptr);
    }
    const auto actual = runtime.download_output(0);
    if (tracePrefix) {
        runtime.write_execution_trace_csv(std::string(tracePrefix) + ".pipeline.csv");
        runtime.write_mem_execution_trace_csv(std::string(tracePrefix) + ".mem.csv");
    }
    if (actual.size() != input.size() * 2)
        throw std::logic_error("depthwise Conv1D output byte count is wrong");

    float maxError = 0.0f;
    std::size_t worstToken = 0;
    std::size_t worstChannel = 0;
    float worstActual = 0.0f;
    float worstExpected = 0.0f;
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            float partial0 = 0.0f;
            float partial1 = 0.0f;
            float partial2 = 0.0f;
            float partial3 = 0.0f;
            float* partials[kKernel] = {
                &partial0, &partial1, &partial2, &partial3};
            for (std::size_t tap = 0; tap < kKernel; ++tap) {
                const int inputToken = static_cast<int>(token)
                    - static_cast<int>(kKernel - 1) + static_cast<int>(tap);
                const float value = inputToken < 0 ? 0.0f
                    : input[static_cast<std::size_t>(inputToken) * kChannels
                        + channel];
                *partials[tap] = value * weight[channel * kKernel + tap];
            }
            const float sum01 = partial0 + partial1;
            const float sum23 = partial2 + partial3;
            const float sum = bf16(sum01 + sum23);
            const float expected = silu
                ? bf16(sum / (1.0f + std::exp(-sum))) : sum;
            const float observed =
                decodeBf16(actual, token * kChannels + channel);
            if (!std::isfinite(observed))
                throw std::logic_error("non-finite depthwise Conv1D output");
            const float error = std::fabs(observed - expected);
            if (error > maxError) {
                maxError = error;
                worstToken = token;
                worstChannel = channel;
                worstActual = observed;
                worstExpected = expected;
            }
        }
    }
    if (maxError > (silu ? 0.002f : 0.0f))
        throw std::logic_error(
            "depthwise Conv1D mismatch token=" + std::to_string(worstToken)
            + " channel=" + std::to_string(worstChannel)
            + " actual=" + std::to_string(worstActual)
            + " expected=" + std::to_string(worstExpected)
            + " max_error=" + std::to_string(maxError));

    std::cout << "compiled_depthwise_conv1d_runtime_test passed max_error="
              << maxError << " cycles=" << runtime.physical_cycles()
              << " binary_bytes=" << std::filesystem::file_size(argv[1]) << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "compiled_depthwise_conv1d_runtime_test failed: "
              << error.what() << '\n';
    return 1;
}
