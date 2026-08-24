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
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

float bf16_at(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(data[2 * index])
        | (static_cast<std::uint16_t>(data[2 * index + 1]) << 8))
        .to_float();
}
} // namespace

int main(int argc, char** argv)
try {
    if (argc != 3)
        throw std::runtime_error(
            "usage: compiled_qwen_real_ffn_runtime_test program.ftlpu fixture_dir");
    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    const std::filesystem::path fixture(argv[2]);
    if (program.weight_page_uses.empty())
        throw std::logic_error(
            "Qwen FFN binary has no intra-executable weight pages");
    auto* system = new ftlpu::TspSliceSystem();
    const bool reportProgress =
        std::getenv("FTLPU_QWEN_FFN_PROGRESS") != nullptr;
    ftlpu::software::runtime::CModelRuntime runtime(*system,
        [system](ftlpu::TspSliceSystem::LogSinks sinks) {
            system->tick(sinks);
        });
    runtime.load(program);
    runtime.upload_input(0, read_bytes(fixture / "input.bf16.bin"));
    runtime.upload_input(1, read_bytes(fixture / "gate.i8.bin"));
    runtime.upload_input(2, read_bytes(fixture / "up.i8.bin"));
    runtime.upload_input(3, read_bytes(fixture / "down.i8.bin"));
    constexpr std::size_t kProgressChunkCycles = 16 * 1024;
    const std::size_t totalCycles = program.max_cycle + 64;
    const auto traceStartText = std::getenv("FTLPU_QWEN_FFN_TRACE_START");
    const auto traceCyclesText = std::getenv("FTLPU_QWEN_FFN_TRACE_CYCLES");
    if (traceStartText != nullptr && traceCyclesText != nullptr) {
        const auto traceStart = static_cast<std::size_t>(
            std::stoull(traceStartText));
        const auto traceCycles = static_cast<std::size_t>(
            std::stoull(traceCyclesText));
        if (traceStart + traceCycles > totalCycles)
            throw std::logic_error("requested Qwen FFN trace exceeds program");
        runtime.run_cycles(traceStart);
        runtime.run_cycles(traceCycles, &std::cerr);
        runtime.run_cycles(totalCycles - traceStart - traceCycles);
    } else {
    for (std::size_t executed = 0; executed < totalCycles;) {
        const std::size_t chunk =
            std::min(kProgressChunkCycles, totalCycles - executed);
        runtime.run_cycles(chunk);
        executed += chunk;
        if (reportProgress)
            std::cerr << "Qwen FFN CModel progress " << executed << '/'
                      << totalCycles << " cycles\n";
    }
    }

    const auto actual = runtime.download_output(0);
    const auto golden = read_bytes(fixture / "golden.bf16.bin");
    if (actual.size() != golden.size())
        throw std::logic_error("Qwen FFN output size differs from golden");
    float maxError = 0.0f;
    float maxActual = 0.0f;
    float maxGolden = 0.0f;
    std::size_t maxErrorIndex = 0;
    double sumAbsoluteError = 0.0;
    double sumSquaredError = 0.0;
    std::vector<float> errors;
    errors.reserve(actual.size() / 2);
    std::size_t mismatches = 0;
    std::size_t nonzero = 0;
    for (std::size_t index = 0; index < actual.size() / 2; ++index) {
        const float observed = bf16_at(actual, index);
        const float expected = bf16_at(golden, index);
        const float error = std::fabs(observed - expected);
        if (error > maxError) {
            maxError = error;
            maxErrorIndex = index;
        }
        errors.push_back(error);
        sumAbsoluteError += error;
        sumSquaredError += static_cast<double>(error) * error;
        maxActual = std::max(maxActual, std::fabs(observed));
        maxGolden = std::max(maxGolden, std::fabs(expected));
        if (std::fabs(observed) > 1.0e-6f) ++nonzero;
        if (error > 0.125f + 0.025f * std::fabs(expected)) {
            if (mismatches < 12)
                std::cerr << "mismatch index=" << index
                          << " actual=" << observed
                          << " golden=" << expected
                          << " error=" << error << '\n';
            ++mismatches;
        }
    }
    if (nonzero == 0)
        throw std::logic_error("Qwen FFN produced an all-zero output");
    std::sort(errors.begin(), errors.end());
    const auto percentile = [&](double fraction) {
        const auto index = static_cast<std::size_t>(
            fraction * static_cast<double>(errors.size() - 1));
        return errors[index];
    };
    const auto values = actual.size() / 2;
    const auto meanAbsoluteError =
        sumAbsoluteError / static_cast<double>(values);
    const auto rootMeanSquaredError = std::sqrt(
        sumSquaredError / static_cast<double>(values));
    std::cout << "Qwen FFN numerical summary: values=" << values
              << " mismatches=" << mismatches
              << " mae=" << meanAbsoluteError
              << " rmse=" << rootMeanSquaredError
              << " p99=" << percentile(0.99)
              << " p999=" << percentile(0.999)
              << " max_error=" << maxError
              << " max_error_index=" << maxErrorIndex
              << " max_abs(actual/golden)=" << maxActual << "/"
              << maxGolden << '\n';
    if (mismatches != 0)
        throw std::logic_error("Qwen FFN golden mismatch count="
            + std::to_string(mismatches));
    std::cout << "Qwen2.5-1.5B layer0 real FFN passed: values="
              << values
              << " pages=" << program.weight_page_uses.size()
              << " max_cycle=" << program.max_cycle
              << " nonzero=" << nonzero
              << " max_error=" << maxError << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "compiled_qwen_real_ffn_runtime_test failed: "
              << error.what() << '\n';
    return 1;
}
