#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/core/fp16.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
#ifndef FTLPU_TEST_SEQUENCE_LENGTH
#define FTLPU_TEST_SEQUENCE_LENGTH 32
#endif

constexpr std::size_t kM = FTLPU_TEST_SEQUENCE_LENGTH;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;

float activation(std::size_t row, std::size_t k)
{
    const int value = static_cast<int>((row * 7 + k * 5) % 17) - 8;
    return ftlpu::Fp16::from_float(static_cast<float>(value) / 16.0f).to_float();
}

std::size_t gate_k(std::size_t h) { return (h * 7 + 1) % kHidden; }
std::size_t up_k(std::size_t h) { return (h * 11 + 3) % kHidden; }
std::int8_t gate_sign(std::size_t h) { return (h & 1) ? -1 : 1; }
std::int8_t up_sign(std::size_t h) { return (h & 2) ? -1 : 1; }

void append_fp16(std::vector<std::uint8_t>& bytes, float value)
{
    const auto bits = ftlpu::Fp16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float read_fp16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const auto bits = static_cast<std::uint16_t>(bytes[index * 2])
        | (static_cast<std::uint16_t>(bytes[index * 2 + 1]) << 8);
    return ftlpu::Fp16::from_bits(bits).to_float();
}
} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error("usage: compiled_smollm2_135m_ffn_runtime_test program.ftlpu");
    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    if (program.bindings.size() != 5)
        throw std::logic_error("SmolLM2 FFN binary must contain four inputs and one output");
    if (program.max_cycle == 0)
        throw std::logic_error("SmolLM2 FFN binary has no scheduled ICU commands");
    if (const auto* trace_path = std::getenv("FTLPU_SCHEDULE_TRACE")) {
        ftlpu::software::runtime::write_schedule_trace_csv(program, trace_path);
    }

    std::vector<std::uint8_t> x;
    x.reserve(kM * kHidden * 2);
    for (std::size_t row = 0; row < kM; ++row)
        for (std::size_t k = 0; k < kHidden; ++k)
            append_fp16(x, activation(row, k));

    std::vector<std::uint8_t> gate_w(kHidden * kIntermediate, 0);
    std::vector<std::uint8_t> up_w(kHidden * kIntermediate, 0);
    for (std::size_t h = 0; h < kIntermediate; ++h) {
        gate_w[gate_k(h) * kIntermediate + h] = static_cast<std::uint8_t>(gate_sign(h));
        up_w[up_k(h) * kIntermediate + h] = static_cast<std::uint8_t>(up_sign(h));
    }

    std::vector<std::uint8_t> down_w(kIntermediate * kHidden, 0);
    for (std::size_t n = 0; n < kHidden; ++n) {
        const std::size_t h0 = (n * 5 + 17) % kIntermediate;
        const std::size_t h1 = (h0 + 37) % kIntermediate;
        down_w[h0 * kHidden + n] = 1;
        down_w[h1 * kHidden + n] = static_cast<std::uint8_t>(-1);
    }

    // This full-chip model is intentionally process-lifetime in the test. On
    // Windows its very large nested queue graph is costly to tear down.
    auto* system = new ftlpu::TspSliceSystem();
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, x);
    runtime.upload_input(1, gate_w);
    runtime.upload_input(2, up_w);
    runtime.upload_input(3, down_w);
    runtime.run_cycles(program.max_cycle + 64);
    const auto actual = runtime.download_output(0);

    std::vector<float> hidden(kIntermediate);
    std::size_t checked = 0;
    std::size_t actual_nonzero = 0;
    std::size_t expected_nonzero = 0;
    float actual_max_abs = 0.0f;
    float expected_max_abs = 0.0f;
    float sample_actual[3] {};
    float sample_expected[3] {};
    constexpr std::size_t kSampleRows[] = {0, kM / 2, kM - 1};
    constexpr std::size_t kSampleColumns[] = {0, 191, 575};
    for (std::size_t row = 0; row < kM; ++row) {
        const auto read_hidden = [&](std::size_t h) {
            const std::size_t address = (h / 32) * kM + row;
            const std::size_t column = h % 32;
            const auto low = system->read_mem_sram_lane_byte(ftlpu::Hemisphere::East,
                21, column / 8, address, column % 8);
            const auto high = system->read_mem_sram_lane_byte(ftlpu::Hemisphere::East,
                22, column / 8, address, column % 8);
            return ftlpu::Fp16::from_bits(static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        };
        for (std::size_t h = 0; h < kIntermediate; ++h) {
            const float gate = activation(row, gate_k(h)) * gate_sign(h);
            const float up = activation(row, up_k(h)) * up_sign(h);
            hidden[h] = ftlpu::Fp16::from_float(
                gate * (1.0f / (1.0f + std::exp(-gate))) * up).to_float();
        }
        for (std::size_t n = 0; n < kHidden; ++n) {
            const std::size_t h0 = (n * 5 + 17) % kIntermediate;
            const std::size_t h1 = (h0 + 37) % kIntermediate;
            const float expected = ftlpu::Fp16::from_float(hidden[h0] - hidden[h1]).to_float();
            const float observed = read_fp16(actual, row * kHidden + n);
            actual_max_abs = std::max(actual_max_abs, std::fabs(observed));
            expected_max_abs = std::max(expected_max_abs, std::fabs(expected));
            if (std::fabs(observed) > 0.0005f) ++actual_nonzero;
            if (std::fabs(expected) > 0.0005f) ++expected_nonzero;
            for (std::size_t sample = 0; sample < std::size(kSampleRows); ++sample) {
                if (row == kSampleRows[sample] && n == kSampleColumns[sample]) {
                    sample_actual[sample] = observed;
                    sample_expected[sample] = expected;
                }
            }
            if (std::fabs(observed - expected) > 0.004f) {
                const auto read_temp_fp32 =
                    [&](const std::array<std::size_t, 4>& slices,
                        std::size_t h) {
                        std::uint32_t raw = 0;
                        const std::size_t column = h % 32;
                        for (std::size_t byte = 0; byte < slices.size();
                             ++byte) {
                            raw |= static_cast<std::uint32_t>(
                                system->read_mem_sram_lane_byte(
                                    ftlpu::Hemisphere::East, slices[byte],
                                    column / 8, row, column % 8))
                                << (byte * 8);
                        }
                        return std::bit_cast<float>(raw);
                    };
                float hidden_max_error = 0.0f;
                for (std::size_t index = 0; index < kIntermediate; ++index)
                    hidden_max_error = std::max(hidden_max_error,
                        std::fabs(read_hidden(index) - hidden[index]));
                const float device_hidden_expected = ftlpu::Fp16::from_float(
                    read_hidden(h0) - read_hidden(h1)).to_float();
                throw std::logic_error("SmolLM2 FFN mismatch row=" + std::to_string(row)
                    + " column=" + std::to_string(n) + " actual="
                    + std::to_string(observed) + " expected=" + std::to_string(expected)
                    + " hidden0.actual=" + std::to_string(read_hidden(h0))
                    + " hidden0.cpu=" + std::to_string(hidden[h0])
                    + " hidden1.actual=" + std::to_string(read_hidden(h1))
                    + " hidden1.cpu=" + std::to_string(hidden[h1])
                    + " gate_temp=" + std::to_string(read_temp_fp32(
                        {1, 5, 9, 13}, h0))
                    + " up_temp=" + std::to_string(read_temp_fp32(
                        {2, 6, 10, 14}, h0))
                    + " hidden.max_error=" + std::to_string(hidden_max_error)
                    + " down.from_device_hidden="
                    + std::to_string(device_hidden_expected));
            }
            ++checked;
        }
    }
    if (actual_nonzero == 0 || expected_nonzero == 0)
        throw std::logic_error("SmolLM2 FFN numeric test unexpectedly produced only zero outputs");
    ftlpu::software::runtime::print_runtime_performance(
        program, program.max_cycle + 64, std::cout);
    std::cout << "SmolLM2-135M complete FFN passed: " << checked
              << " FP16 values, max_cycle=" << program.max_cycle
              << ", nonzero(actual/reference)=" << actual_nonzero << "/" << expected_nonzero
              << ", max_abs(actual/reference)=" << actual_max_abs << "/" << expected_max_abs
              << ", samples [r0,c0]=" << sample_actual[0] << "/" << sample_expected[0]
              << " [rmid,c191]=" << sample_actual[1] << "/" << sample_expected[1]
              << " [rlast,c575]=" << sample_actual[2] << "/" << sample_expected[2]
              << std::endl;
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_smollm2_135m_ffn_runtime_test failed: " << ex.what() << '\n';
    return 1;
}
