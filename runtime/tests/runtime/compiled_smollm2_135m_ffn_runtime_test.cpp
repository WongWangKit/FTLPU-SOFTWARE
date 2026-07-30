#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/binary_program_adapter.hpp"
#include "ftlpu/software/runtime/binding_transfer.hpp"
#include "ftlpu/software/runtime/cmodel_device_backend.hpp"

#include "ftlpu/core/fp16.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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

const ftlpu::software::runtime::BinaryBinding& find_binding(
    const std::vector<ftlpu::software::runtime::BinaryBinding>& bindings,
    ftlpu::software::runtime::BindingAccess access,
    std::uint32_t index)
{
    const auto found = std::find_if(
        bindings.begin(), bindings.end(),
        [&](const auto& binding) {
            return binding.access == access
                && binding.index == index;
        });
    if (found == bindings.end()) {
        throw std::logic_error(
            "SmolLM2 FFN binary is missing a required binding");
    }
    return *found;
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
    auto* backend =
        new ftlpu::software::runtime::CModelDeviceBackend();
    const auto adapted =
        ftlpu::software::runtime::adapt_binary_program(
            program, backend->target(), 63);
    const std::array<std::vector<std::uint8_t>, 4> inputs {
        std::move(x),
        std::move(gate_w),
        std::move(up_w),
        std::move(down_w),
    };
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto& input_binding = find_binding(
            adapted.bindings,
            ftlpu::software::runtime::BindingAccess::Input,
            static_cast<std::uint32_t>(index));
        const auto transfers =
            ftlpu::software::runtime::pack_binding(
                input_binding,
                inputs[index],
                backend->target());
        for (const auto& transfer : transfers) {
            backend->upload(
                transfer.address,
                transfer.bytes,
                transfer.purpose);
        }
    }
    backend->load(adapted.device_program);
    backend->launch();
    backend->wait();

    std::vector<ftlpu::software::runtime::BindingTransfer>
        output_transfers;
    const auto& output_binding = find_binding(
        adapted.bindings,
        ftlpu::software::runtime::BindingAccess::Output,
        0);
    for (const auto& region :
        ftlpu::software::runtime::plan_binding_regions(
            output_binding, backend->target())) {
        output_transfers.push_back({
            region.address,
            region.purpose,
            backend->download(
                region.address,
                region.byte_size,
                region.purpose),
        });
    }
    const auto actual =
        ftlpu::software::runtime::unpack_binding(
            output_binding,
            output_transfers,
            backend->target());

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
                throw std::logic_error("SmolLM2 FFN mismatch row=" + std::to_string(row)
                    + " column=" + std::to_string(n) + " actual="
                    + std::to_string(observed) + " expected="
                    + std::to_string(expected));
            }
            ++checked;
        }
    }
    if (actual_nonzero == 0 || expected_nonzero == 0)
        throw std::logic_error("SmolLM2 FFN numeric test unexpectedly produced only zero outputs");
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
