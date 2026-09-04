#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
#ifndef FTLPU_TEST_SEQUENCE_LENGTH
#define FTLPU_TEST_SEQUENCE_LENGTH 32
#endif
#ifndef FTLPU_TEST_HIDDEN_SIZE
#define FTLPU_TEST_HIDDEN_SIZE 576
#endif
#ifndef FTLPU_TEST_INTERMEDIATE_SIZE
#define FTLPU_TEST_INTERMEDIATE_SIZE 1536
#endif
#ifndef FTLPU_TEST_OUTPUT_SIZE
#define FTLPU_TEST_OUTPUT_SIZE FTLPU_TEST_HIDDEN_SIZE
#endif
#ifndef FTLPU_TEST_MODEL_NAME
#define FTLPU_TEST_MODEL_NAME "SmolLM2-135M"
#endif

constexpr std::size_t kM = FTLPU_TEST_SEQUENCE_LENGTH;
constexpr std::size_t kHidden = FTLPU_TEST_HIDDEN_SIZE;
constexpr std::size_t kIntermediate = FTLPU_TEST_INTERMEDIATE_SIZE;
constexpr std::size_t kOutput = FTLPU_TEST_OUTPUT_SIZE;

float activation(std::size_t row, std::size_t k)
{
    const int value = static_cast<int>((row * 7 + k * 5) % 17) - 8;
    return ftlpu::Bf16::from_float(static_cast<float>(value) / 16.0f).to_float();
}

std::size_t gate_k(std::size_t h) { return (h * 7 + 1) % kHidden; }
std::size_t up_k(std::size_t h) { return (h * 11 + 3) % kHidden; }
std::int8_t gate_sign(std::size_t h) { return (h & 1) ? -1 : 1; }
std::int8_t up_sign(std::size_t h) { return (h & 2) ? -1 : 1; }

void append_bf16(std::vector<std::uint8_t>& bytes, float value)
{
    const auto bits = ftlpu::Bf16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float read_bf16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const auto bits = static_cast<std::uint16_t>(bytes[index * 2])
        | (static_cast<std::uint16_t>(bytes[index * 2 + 1]) << 8);
    return ftlpu::Bf16::from_bits(bits).to_float();
}
} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2 && argc != 3)
        throw std::runtime_error(
            "usage: compiled_smollm2_135m_ffn_runtime_test program.ftlpu "
            "[--require-icu-macro]");
    const bool requireIcuMacro = argc == 3
        && std::string_view(argv[2]) == "--require-icu-macro";
    if (argc == 3 && !requireIcuMacro)
        throw std::runtime_error("unknown FFN runtime test option");
    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    if (program.bindings.size() != 5)
        throw std::logic_error("SmolLM2 FFN binary must contain four inputs and one output");
    if (program.max_cycle == 0)
        throw std::logic_error("SmolLM2 FFN binary has no scheduled ICU commands");
    if (requireIcuMacro) {
        std::size_t macroCount = 0;
        for (const auto& queue : program.queues) {
            const bool macroQueue =
                queue.kind == ftlpu::software::runtime::QueueKind::Mem
                || queue.kind
                    == ftlpu::software::runtime::QueueKind::MxmLoad
                || queue.kind
                    == ftlpu::software::runtime::QueueKind::MxmCompute
                || queue.kind
                    == ftlpu::software::runtime::QueueKind::MxmDequant;
            if (!macroQueue) continue;
            for (const auto& command : queue.commands) {
                const bool coarseMem = queue.kind
                        == ftlpu::software::runtime::QueueKind::Mem
                    && (ftlpu::software::runtime::is_mem_stream_nd_command(
                            command)
                        || ftlpu::software::runtime::
                            is_mem_slice_program_command(command));
                const bool coarseMxm = queue.kind
                        != ftlpu::software::runtime::QueueKind::Mem
                    && ftlpu::software::runtime::is_mxm_stream_nd_command(
                        command);
                if (!coarseMem && !coarseMxm
                    && !ftlpu::software::runtime::is_macro_schedule_command(
                        command))
                    throw std::logic_error(
                        "macro FFN contains a fine-grained MEM/MXM command");
                ++macroCount;
            }
        }
        if (macroCount == 0)
            throw std::logic_error("macro FFN contains no ICU macro commands");
    }
    if (const auto* trace_path = std::getenv("FTLPU_SCHEDULE_TRACE")) {
        ftlpu::software::runtime::write_schedule_trace_csv(program, trace_path);
    }
    std::vector<std::uint8_t> x;
    x.reserve(kM * kHidden * 2);
    for (std::size_t row = 0; row < kM; ++row)
        for (std::size_t k = 0; k < kHidden; ++k)
            append_bf16(x, activation(row, k));

    std::vector<std::uint8_t> gate_w(kHidden * kIntermediate, 0);
    std::vector<std::uint8_t> up_w(kHidden * kIntermediate, 0);
    for (std::size_t h = 0; h < kIntermediate; ++h) {
        gate_w[gate_k(h) * kIntermediate + h] = static_cast<std::uint8_t>(gate_sign(h));
        up_w[up_k(h) * kIntermediate + h] = static_cast<std::uint8_t>(up_sign(h));
    }

    std::vector<std::uint8_t> down_w(kIntermediate * kOutput, 0);
    for (std::size_t n = 0; n < kOutput; ++n) {
        const std::size_t h0 = (n * 5 + 17) % kIntermediate;
        const std::size_t h1 = (h0 + 37) % kIntermediate;
        down_w[h0 * kOutput + n] = 1;
        down_w[h1 * kOutput + n] = static_cast<std::uint8_t>(-1);
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
    std::ofstream cmodel_log;
    std::ostream* cmodel_log_sink = nullptr;
    if (const auto* path = std::getenv("FTLPU_CMODEL_LOG")) {
        cmodel_log.open(path, std::ios::trunc);
        if (!cmodel_log)
            throw std::runtime_error("cannot open FTLPU_CMODEL_LOG path");
        cmodel_log_sink = &cmodel_log;
    }
    const auto total_cycles = program.max_cycle + 64;
    const auto stop_cycle = std::getenv("FTLPU_STOP_CYCLE") != nullptr
        ? std::strtoull(std::getenv("FTLPU_STOP_CYCLE"), nullptr, 10)
        : 0;
    const auto run_limit = stop_cycle != 0 ? stop_cycle : total_cycles;
    if (cmodel_log_sink != nullptr
        && std::getenv("FTLPU_CMODEL_LOG_START") != nullptr) {
        const auto log_start = std::min<std::uint64_t>(
            std::strtoull(std::getenv("FTLPU_CMODEL_LOG_START"), nullptr, 10),
            run_limit);
        const auto requested = std::getenv("FTLPU_CMODEL_LOG_CYCLES") != nullptr
            ? std::strtoull(std::getenv("FTLPU_CMODEL_LOG_CYCLES"), nullptr, 10)
            : 192;
        const auto log_cycles = std::min<std::uint64_t>(
            requested, run_limit - log_start);
        runtime.run_cycles(log_start);
        runtime.run_cycles(log_cycles, cmodel_log_sink);
        const auto remaining = run_limit - log_start - log_cycles;
        if (remaining != 0) runtime.run_cycles(remaining);
    } else if (cmodel_log_sink != nullptr && run_limit > 128) {
        runtime.run_cycles(run_limit - 128);
        runtime.run_cycles(128, cmodel_log_sink);
    } else {
        runtime.run_cycles(run_limit, cmodel_log_sink);
    }
    ftlpu::software::runtime::print_runtime_performance(
        program, program.max_cycle + 64, std::cout);
    runtime.print_datapath_performance(std::cout);
    const auto output_binding = std::find_if(program.bindings.begin(),
        program.bindings.end(), [](const auto& binding) {
            return binding.access
                == ftlpu::software::runtime::BindingAccess::Output;
        });
    if (output_binding == program.bindings.end())
        throw std::logic_error(FTLPU_TEST_MODEL_NAME " FFN has no output binding");
    const bool is_block8 = output_binding->layout
        == ftlpu::software::runtime::BindingLayout::Fp16MxmBlock8Distributed16;
    if (is_block8 && output_binding->slices.size() != 16)
        throw std::logic_error(
            FTLPU_TEST_MODEL_NAME " FFN has no distributed16 output binding");
    const auto physical_bf16 = [&](ftlpu::Hemisphere hemisphere,
                                   std::size_t base_row,
                                   std::size_t width,
                                   std::size_t row,
                                   std::size_t column,
                                   std::size_t bank) {
        const std::size_t blocks = width / 32;
        const std::size_t address = base_row
            + ((row / 32) * blocks + column / 32) * 4
            + (row % 32) / 8;
        const std::size_t stream = row % 8;
        const std::size_t lane = ((column % 32) / 8) * 8
            + column % 8;
        const auto low = system->read_mem_sram_lane_byte(hemisphere,
            output_binding->slices[2 * stream], bank,
            lane / 8, address, lane % 8);
        const auto high = system->read_mem_sram_lane_byte(hemisphere,
            output_binding->slices[2 * stream + 1], bank,
            lane / 8, address, lane % 8);
        return ftlpu::Bf16::from_bits(
            static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8)).to_float();
    };
    const auto expected_hidden = [&](std::size_t row, std::size_t h) {
        const float gate = activation(row, gate_k(h)) * gate_sign(h);
        const float up = activation(row, up_k(h)) * up_sign(h);
        return ftlpu::Bf16::from_float(
            gate * (1.0f / (1.0f + std::exp(-gate))) * up).to_float();
    };
    const std::size_t hidden_rows = is_block8
        ? (kM / 32) * (kIntermediate / 32) * 4 : 0;
    if (is_block8 && output_binding->base_row < hidden_rows)
        throw std::logic_error("FFN output does not follow hidden workspace");
    const std::size_t hidden_base = output_binding->base_row - hidden_rows;
    if (stop_cycle != 0) {
        if (!is_block8)
            throw std::logic_error(
                "FTLPU_STOP_CYCLE checkpoints require Block8 layout");
        const std::vector<std::size_t> scratch_slices(
            output_binding->slices.begin(), output_binding->slices.end());
        if (scratch_slices.size() != 16)
            throw std::logic_error("cannot reconstruct FFN scratch slices");
        const auto scratch_bf16 = [&](ftlpu::Hemisphere hemisphere,
                                      std::size_t bank,
                                      std::size_t row,
                                      std::size_t h) {
            const std::size_t pair = h / 64;
            const std::size_t address = output_binding->base_row
                + pair * 4 + (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            const std::size_t lane = ((h % 32) / 8) * 8 + h % 8;
            const auto low = system->read_mem_sram_lane_byte(hemisphere,
                scratch_slices[2 * token_lane], bank, lane / 8,
                address, lane % 8);
            const auto high = system->read_mem_sram_lane_byte(hemisphere,
                scratch_slices[2 * token_lane + 1], bank, lane / 8,
                address, lane % 8);
            return ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        };
        for (std::size_t h = 64; h < 96; ++h) {
            const auto source = static_cast<ftlpu::Hemisphere>((h / 32) % 2);
            const auto owner = static_cast<ftlpu::Hemisphere>(1 - (h / 32) % 2);
            std::cout << "checkpoint h=" << h
                      << " gate=" << scratch_bf16(source, 0, 0, h)
                      << "/" << activation(0, gate_k(h)) * gate_sign(h)
                      << " up=" << scratch_bf16(source, 1, 0, h)
                      << "/" << activation(0, up_k(h)) * up_sign(h)
                      << " swish=" << physical_bf16(owner, hidden_base,
                             kIntermediate, 0, h, 0)
                      << "/" << expected_hidden(0, h) << '\n';
        }
        return 0;
    }
    if (is_block8) {
        for (std::size_t row = 0; row < kM; ++row) {
            for (std::size_t h = 0; h < kIntermediate; ++h) {
                const float expected = expected_hidden(row, h);
                const float east = physical_bf16(ftlpu::Hemisphere::East,
                    hidden_base, kIntermediate, row, h, 0);
                const float west = physical_bf16(ftlpu::Hemisphere::West,
                    hidden_base, kIntermediate, row, h, 0);
                if (std::fabs(east - expected) > 0.004f
                    || std::fabs(west - expected) > 0.004f)
                    throw std::logic_error(
                        FTLPU_TEST_MODEL_NAME " SwiGLU checkpoint mismatch row="
                        + std::to_string(row) + " hidden=" + std::to_string(h)
                        + " east=" + std::to_string(east)
                        + " west=" + std::to_string(west)
                        + " expected=" + std::to_string(expected));
            }
        }
        for (std::size_t row = 0; row < kM; ++row) {
            for (std::size_t n = 0; n < kOutput; ++n) {
                const std::size_t h0 = (n * 5 + 17) % kIntermediate;
                const std::size_t h1 = (h0 + 37) % kIntermediate;
                const float expected = ftlpu::Bf16::from_float(
                    expected_hidden(row, h0) - expected_hidden(row, h1))
                                           .to_float();
                const auto owner = static_cast<ftlpu::Hemisphere>(
                    (n / 32) % 2);
                const auto replica = static_cast<ftlpu::Hemisphere>(
                    1 - (n / 32) % 2);
                const float owner_value = physical_bf16(owner,
                    output_binding->base_row, kOutput, row, n,
                    output_binding->bank);
                const float replica_value = physical_bf16(replica,
                    output_binding->base_row, kOutput, row, n,
                    output_binding->bank);
                if (std::fabs(owner_value - expected) > 0.004f
                    || std::fabs(replica_value - expected) > 0.004f)
                    throw std::logic_error(
                        FTLPU_TEST_MODEL_NAME " down checkpoint mismatch row="
                        + std::to_string(row) + " column=" + std::to_string(n)
                        + " owner=" + std::to_string(
                            static_cast<std::size_t>(owner))
                        + " owner_actual=" + std::to_string(owner_value)
                        + " replica_actual=" + std::to_string(replica_value)
                        + " expected=" + std::to_string(expected)
                        + " h0=" + std::to_string(h0)
                        + " h1=" + std::to_string(h1));
            }
        }
    }
    const auto actual = runtime.download_output(0);

    std::vector<float> hidden(kIntermediate);
    std::size_t checked = 0;
    std::size_t actual_nonzero = 0;
    std::size_t expected_nonzero = 0;
    std::size_t mismatch_count = 0;
    float actual_max_abs = 0.0f;
    float expected_max_abs = 0.0f;
    float sample_actual[3] {};
    float sample_expected[3] {};
    constexpr std::size_t kSampleRows[] = {0, kM / 2, kM - 1};
    constexpr std::size_t kSampleColumns[] = {0, 191, kOutput - 1};
    for (std::size_t row = 0; row < kM; ++row) {
        for (std::size_t h = 0; h < kIntermediate; ++h) {
            const float gate = activation(row, gate_k(h)) * gate_sign(h);
            const float up = activation(row, up_k(h)) * up_sign(h);
            hidden[h] = ftlpu::Bf16::from_float(
                gate * (1.0f / (1.0f + std::exp(-gate))) * up).to_float();
        }
        for (std::size_t n = 0; n < kOutput; ++n) {
            const std::size_t h0 = (n * 5 + 17) % kIntermediate;
            const std::size_t h1 = (h0 + 37) % kIntermediate;
            const float expected = ftlpu::Bf16::from_float(hidden[h0] - hidden[h1]).to_float();
            const float observed = read_bf16(actual, row * kOutput + n);
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
                if (mismatch_count < 16)
                    std::cerr << "mismatch row=" << row
                              << " column=" << n
                              << " actual=" << observed
                              << " expected=" << expected << '\n';
                ++mismatch_count;
            }
            ++checked;
        }
    }
    if (mismatch_count != 0)
        throw std::logic_error(FTLPU_TEST_MODEL_NAME " FFN mismatch count="
            + std::to_string(mismatch_count));
    if (actual_nonzero == 0 || expected_nonzero == 0)
        throw std::logic_error(FTLPU_TEST_MODEL_NAME
            " FFN numeric test unexpectedly produced only zero outputs");
    std::cout << FTLPU_TEST_MODEL_NAME " complete FFN passed: " << checked
              << " BF16 values, max_cycle=" << program.max_cycle
              << ", nonzero(actual/reference)=" << actual_nonzero << "/" << expected_nonzero
              << ", max_abs(actual/reference)=" << actual_max_abs << "/" << expected_max_abs
              << ", samples [r0,c0]=" << sample_actual[0] << "/" << sample_expected[0]
              << " [rmid,c191]=" << sample_actual[1] << "/" << sample_expected[1]
              << " [rlast,clast]=" << sample_actual[2] << "/" << sample_expected[2]
              << std::endl;
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_smollm2_135m_ffn_runtime_test failed: " << ex.what() << '\n';
    return 1;
}
