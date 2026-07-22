#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/fp16.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kQueryHeads = 9;
constexpr std::size_t kKvHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kTile = 32;
constexpr std::size_t kProjectionEndCycle = 32508;
constexpr float kRopeTheta = 100000.0f;
constexpr std::array<std::array<std::size_t, 16>, 2> kQueryIwSlices {{
    {{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
    {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
}};

enum class Projection : std::size_t { Query, Key, Value };

std::size_t width(Projection projection)
{
    return (projection == Projection::Query ? kQueryHeads : kKvHeads) * kHeadDim;
}

float activation(std::size_t token, std::size_t hidden)
{
    const int value = static_cast<int>((token * 7 + hidden * 5) % 23) - 11;
    return ftlpu::Fp16::from_float(static_cast<float>(value) / 16.0f).to_float();
}

std::size_t source_hidden(Projection projection, std::size_t column)
{
    return (column * 7 + static_cast<std::size_t>(projection) * 13 + 3) % kHidden;
}

float projected(Projection projection, std::size_t token, std::size_t column)
{
    const float sign = ((column + static_cast<std::size_t>(projection)) & 1) ? -1.0f : 1.0f;
    return activation(token, source_hidden(projection, column)) * sign;
}

void append_fp16(std::vector<std::uint8_t>& bytes, float value)
{
    const auto bits = ftlpu::Fp16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

std::vector<std::uint8_t> make_weight(Projection projection)
{
    std::vector<std::uint8_t> weight(kHidden * width(projection), 0);
    for (std::size_t column = 0; column < width(projection); ++column) {
        const std::int8_t value = ((column + static_cast<std::size_t>(projection)) & 1) ? -1 : 1;
        weight[source_hidden(projection, column) * width(projection) + column]
            = static_cast<std::uint8_t>(value);
    }
    return weight;
}

void initialize_rope(ftlpu::TspSliceSystem& system)
{
    for (std::size_t h = 0; h < 2; ++h) {
        const auto hemisphere = static_cast<ftlpu::Hemisphere>(h);
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t dimension = 0; dimension < kTile; ++dimension) {
                const float inverse_frequency = 1.0f / std::pow(
                    kRopeTheta, static_cast<float>(2 * dimension) / kHeadDim);
                const float angle = static_cast<float>(token) * inverse_frequency;
                const auto cos_bits = ftlpu::Fp16::from_float(std::cos(angle)).bits();
                const auto sin_bits = ftlpu::Fp16::from_float(std::sin(angle)).bits();
                const std::size_t tile = dimension / 8;
                const std::size_t lane = dimension % 8;
                system.initialize_mem_sram_lane_byte(hemisphere, 4, tile, 7000 + token,
                    lane, static_cast<std::uint8_t>(cos_bits));
                system.initialize_mem_sram_lane_byte(hemisphere, 5, tile, 7000 + token,
                    lane, static_cast<std::uint8_t>(cos_bits >> 8));
                system.initialize_mem_sram_lane_byte(hemisphere, 6, tile, 7000 + token,
                    lane, static_cast<std::uint8_t>(sin_bits));
                system.initialize_mem_sram_lane_byte(hemisphere, 7, tile, 7000 + token,
                    lane, static_cast<std::uint8_t>(sin_bits >> 8));
            }
        }
    }
}

float read_projection(const ftlpu::TspSliceSystem& system, Projection projection,
    std::size_t token, std::size_t column)
{
    const std::size_t head = column / kHeadDim;
    const std::size_t dimension = column % kHeadDim;
    const std::size_t physical = dimension % kTile;
    const std::size_t tile = physical / 8;
    const std::size_t lane = physical % 8;
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(head % 2);
    std::size_t low_slice = dimension < kTile ? 0 : 2;
    std::size_t high_slice = low_slice + 1;
    std::size_t address = (projection == Projection::Value ? kKvHeads * kSeqLen : 0)
        + head * kSeqLen + token;
    if (projection == Projection::Query) {
        const std::size_t reduction = dimension / kTile;
        const std::size_t local_token = token % kTile;
        const std::size_t stream = (local_token % 8) * 2;
        low_slice = kQueryIwSlices[reduction][stream];
        high_slice = kQueryIwSlices[reduction][stream + 1];
        address = 7600 + (head * (kSeqLen / kTile) + token / kTile) * 4
            + local_token / 8;
    }
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, low_slice, tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, high_slice, tile, address, lane);
    return ftlpu::Fp16::from_bits(static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float expected(Projection projection, std::size_t token, std::size_t column)
{
    if (projection == Projection::Value)
        return ftlpu::Fp16::from_float(projected(projection, token, column)).to_float();
    const std::size_t dimension = column % kHeadDim;
    const std::size_t pair = dimension % kTile;
    const std::size_t head_base = column - dimension;
    const float lo = projected(projection, token, head_base + pair);
    const float hi = projected(projection, token, head_base + pair + kTile);
    const float inverse_frequency = 1.0f / std::pow(
        kRopeTheta, static_cast<float>(2 * pair) / kHeadDim);
    const float angle = static_cast<float>(token) * inverse_frequency;
    const float cosine = ftlpu::Fp16::from_float(std::cos(angle)).to_float();
    const float sine = ftlpu::Fp16::from_float(std::sin(angle)).to_float();
    return ftlpu::Fp16::from_float(dimension < kTile
        ? lo * cosine - hi * sine : hi * cosine + lo * sine).to_float();
}

float read_probability(const ftlpu::TspSliceSystem& system, std::size_t query_head,
    std::size_t query, std::size_t key)
{
    const std::size_t query_block = query / kTile;
    const std::size_t physical = query % kTile;
    const std::size_t tile = physical / 8;
    const std::size_t lane = physical % 8;
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(kv_head % 2);
    const std::size_t address = (query_head * (kSeqLen / kTile) + query_block)
        * kSeqLen + key;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, 16, tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, 17, tile, address, lane);
    return ftlpu::Fp16::from_bits(static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}
} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error("usage: compiled_smollm2_attention_softmax_runtime_test program.ftlpu");
    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    if (program.bindings.size() != 6 || program.max_cycle <= kProjectionEndCycle)
        throw std::logic_error("attention binary is missing bindings or projection commands");

    std::vector<std::uint8_t> input;
    input.reserve(kSeqLen * kHidden * 2);
    for (std::size_t token = 0; token < kSeqLen; ++token)
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden)
            append_fp16(input, activation(token, hidden));
    const auto query_weight = make_weight(Projection::Query);
    const auto key_weight = make_weight(Projection::Key);
    const auto value_weight = make_weight(Projection::Value);
    const auto output_weight = std::vector<std::uint8_t>(kHidden * kHidden, 0);

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    initialize_rope(*system);
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, input);
    runtime.upload_input(1, query_weight);
    runtime.upload_input(2, key_weight);
    runtime.upload_input(3, value_weight);
    runtime.upload_input(4, output_weight);
    runtime.run_cycles(kProjectionEndCycle);

    constexpr std::size_t sample_tokens[] = {0, 31, 32, 63, 64, 95, 127};
    std::size_t checked = 0;
    std::size_t nonzero = 0;
    for (Projection projection : {Projection::Query, Projection::Key, Projection::Value}) {
        for (std::size_t token : sample_tokens) {
            for (std::size_t column = 0; column < width(projection); ++column) {
                const float actual = read_projection(*system, projection, token, column);
                const float reference = expected(projection, token, column);
                if (std::fabs(actual) > 0.0005f) ++nonzero;
                if (std::fabs(actual - reference) > 0.003f)
                    throw std::logic_error("attention projection mismatch p="
                        + std::to_string(static_cast<std::size_t>(projection))
                        + " token=" + std::to_string(token)
                        + " column=" + std::to_string(column)
                        + " actual=" + std::to_string(actual)
                        + " expected=" + std::to_string(reference));
                ++checked;
            }
        }
    }
    if (nonzero == 0) throw std::logic_error("attention projection produced only zero data");
    runtime.run_cycles(program.max_cycle + 64 - kProjectionEndCycle);
    constexpr std::size_t sample_queries[] = {0, 17, 31, 32, 79, 127};
    std::size_t probability_nonzero = 0;
    std::size_t probability_rows = 0;
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        for (std::size_t query : sample_queries) {
            float sum = 0.0f;
            for (std::size_t key = 0; key < kSeqLen; ++key) {
                const float probability = read_probability(*system, head, query, key);
                if (!std::isfinite(probability) || probability < 0.0f)
                    throw std::logic_error("attention softmax produced an invalid probability");
                if (probability > 0.00001f) ++probability_nonzero;
                sum += probability;
            }
            if (std::fabs(sum - 1.0f) > 0.03f)
                throw std::logic_error("attention softmax row is not normalized: head="
                    + std::to_string(head) + " query=" + std::to_string(query)
                    + " sum=" + std::to_string(sum));
            ++probability_rows;
        }
    }
    if (probability_nonzero == 0)
        throw std::logic_error("attention softmax produced only zero probabilities");
    std::cout << "Compiled attention Q/K/V projection + RoPE passed: " << checked
              << " sampled FP16 values, nonzero=" << nonzero
              << ", projection_end_cycle=" << kProjectionEndCycle
              << "; softmax passed: rows=" << probability_rows
              << ", nonzero=" << probability_nonzero
              << ", max_cycle=" << program.max_cycle << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_smollm2_attention_softmax_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
