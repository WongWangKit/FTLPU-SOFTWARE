#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/performance.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/core/bf16.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
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
constexpr float kRopeTheta = 100000.0f;
constexpr std::array<std::size_t, 6> kSampleQueries {0, 17, 31, 32, 79, 127};
constexpr std::array<std::array<std::size_t, 16>, 2> kQueryIwSlices {{
    {{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33}},
    {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
}};
constexpr std::array<std::array<std::size_t, 16>, 2> kValuePackSlices {{
    {{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35}},
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
    return ftlpu::Bf16::from_float(static_cast<float>(value) / 16.0f).to_float();
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

void append_bf16(std::vector<std::uint8_t>& bytes, float value)
{
    const auto bits = ftlpu::Bf16::from_float(value).bits();
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

std::vector<std::uint8_t> make_output_weight()
{
    std::vector<std::uint8_t> weight(kHidden * kHidden, 0);
    for (std::size_t column = 0; column < kHidden; ++column) {
        for (std::size_t block = 0; block < kHidden / kTile; ++block) {
            const std::size_t hidden =
                block * kTile + (column * 7 + block * 3) % kTile;
            const std::int8_t value =
                ((column + block) & 1) ? -1 : 1;
            weight[hidden * kHidden + column] =
                static_cast<std::uint8_t>(value);
        }
    }
    return weight;
}

float output_weight_value(std::size_t hidden, std::size_t column)
{
    const std::size_t block = hidden / kTile;
    const std::size_t selected =
        block * kTile + (column * 7 + block * 3) % kTile;
    if (hidden != selected) return 0.0f;
    return ((column + block) & 1) ? -1.0f : 1.0f;
}

float bf16_at(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    return ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8)).to_float();
}

float read_projection(const ftlpu::TspSliceSystem& system, Projection projection,
    std::size_t token, std::size_t column)
{
    const std::size_t head = column / kHeadDim;
    const std::size_t dimension = column % kHeadDim;
    const std::size_t physical = dimension % kTile;
    const std::size_t tile = physical / 8;
    const std::size_t lane = physical % 8;
    const std::size_t placement_head = projection == Projection::Query
        ? head / (kQueryHeads / kKvHeads)
        : head;
    const auto hemisphere =
        static_cast<ftlpu::Hemisphere>(placement_head % 2);
    std::size_t low_slice = dimension < kTile ? 4 : 6;
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
    } else if (projection == Projection::Value) {
        const std::size_t reduction = dimension / kTile;
        const std::size_t local_token = token % kTile;
        const std::size_t stream = (local_token % 8) * 2;
        low_slice = kValuePackSlices[reduction][stream];
        high_slice = kValuePackSlices[reduction][stream + 1];
        address = 7800 + ((head * 2 + reduction) * (kSeqLen / kTile)
                            + token / kTile)
                * 4
            + local_token / 8;
    }
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, low_slice, tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, high_slice, tile, address, lane);
    return ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float expected(Projection projection, std::size_t token, std::size_t column)
{
    if (projection == Projection::Value)
        return ftlpu::Bf16::from_float(projected(projection, token, column)).to_float();
    const std::size_t dimension = column % kHeadDim;
    const std::size_t pair = dimension % kTile;
    const std::size_t head_base = column - dimension;
    const float lo = projected(projection, token, head_base + pair);
    const float hi = projected(projection, token, head_base + pair + kTile);
    const float inverse_frequency = 1.0f / std::pow(
        kRopeTheta, static_cast<float>(2 * pair) / kHeadDim);
    const float angle = static_cast<float>(token) * inverse_frequency;
    const float cosine = ftlpu::Bf16::from_float(std::cos(angle)).to_float();
    const float sine = ftlpu::Bf16::from_float(std::sin(angle)).to_float();
    return ftlpu::Bf16::from_float(dimension < kTile
        ? lo * cosine - hi * sine : hi * cosine + lo * sine).to_float();
}

std::vector<float> reference_probability(
    std::size_t query_head, std::size_t query)
{
    const std::size_t kv_head =
        query_head / (kQueryHeads / kKvHeads);
    std::vector<float> scores(query + 1);
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t key = 0; key <= query; ++key) {
        float score = 0.0f;
        for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
            score += expected(Projection::Query, query,
                         query_head * kHeadDim + dimension)
                * expected(Projection::Key, key,
                    kv_head * kHeadDim + dimension);
        }
        score /= std::sqrt(static_cast<float>(kHeadDim));
        scores[key] = score;
        maximum = std::max(maximum, score);
    }

    float denominator = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }
    std::vector<float> probability(kSeqLen, 0.0f);
    for (std::size_t key = 0; key <= query; ++key) {
        probability[key] = ftlpu::Bf16::from_float(
            scores[key] / denominator).to_float();
    }
    return probability;
}

float reference_context(const std::vector<float>& probability,
    std::size_t query_head, std::size_t dimension)
{
    const std::size_t kv_head =
        query_head / (kQueryHeads / kKvHeads);
    float value = 0.0f;
    for (std::size_t key = 0; key < kSeqLen; ++key) {
        value += probability[key]
            * expected(Projection::Value, key,
                kv_head * kHeadDim + dimension);
    }
    return ftlpu::Bf16::from_float(value).to_float();
}

float read_packed_probability(const ftlpu::TspSliceSystem& system,
    std::size_t query_head, std::size_t query, std::size_t key)
{
    const std::size_t query_block = query / kTile;
    const std::size_t physical = query % kTile;
    const std::size_t tile = physical / 8;
    const std::size_t lane = physical % 8;
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(kv_head % 2);
    const std::size_t stream = (key % 8) * 2;
    const std::size_t address = 6000
        + (query_head * (kSeqLen / kTile) + query_block) * (kSeqLen / 8)
        + key / 8;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, kQueryIwSlices[1][stream], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, kQueryIwSlices[1][stream + 1], tile, address, lane);
    return ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float read_diagonal_probability(const ftlpu::TspSliceSystem& system,
    std::size_t query_head, std::size_t query, std::size_t key)
{
    const std::size_t query_block = query / kTile;
    const std::size_t query_row = query % 8;
    const std::size_t diagonal = (query % kTile) / 8;
    const std::size_t key_block = key / kTile;
    const std::size_t physical_key = key % kTile;
    const std::size_t tile = physical_key / 8;
    const std::size_t lane = physical_key % 8;
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(kv_head % 2);
    const std::size_t stream = query_row * 2;
    const std::size_t address = 7000
        + ((query_head * (kSeqLen / kTile) + query_block) * (kSeqLen / kTile)
              + key_block)
            * 4
        + diagonal;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, kQueryIwSlices[0][stream], tile, address, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, kQueryIwSlices[0][stream + 1], tile, address, lane);
    return ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}

float read_context(const ftlpu::TspSliceSystem& system, std::size_t query_head,
    std::size_t token, std::size_t dimension)
{
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const auto hemisphere = static_cast<ftlpu::Hemisphere>(kv_head % 2);
    const std::size_t physical = dimension % kTile;
    const std::size_t tile = physical / 8;
    const std::size_t lane = physical % 8;
    const std::size_t low_slice = dimension < kTile ? 44 : 46;
    const auto low = system.read_mem_sram_lane_byte(
        hemisphere, low_slice, tile, 2000 + query_head * kSeqLen + token, lane);
    const auto high = system.read_mem_sram_lane_byte(
        hemisphere, low_slice + 1, tile, 2000 + query_head * kSeqLen + token, lane);
    return ftlpu::Bf16::from_bits(static_cast<std::uint16_t>(low)
        | (static_cast<std::uint16_t>(high) << 8)).to_float();
}
} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error("usage: compiled_smollm2_attention_runtime_test program.ftlpu");
    const auto program = ftlpu::software::runtime::read_binary_program(
        std::filesystem::path(argv[1]));
    const auto projectionTimeline = std::find_if(
        program.timelines.begin(), program.timelines.end(),
        [](const auto& timeline) { return timeline.name == "qkv"; });
    if (projectionTimeline == program.timelines.end())
        throw std::logic_error(
            "attention binary is missing the qkv timeline");
    const std::size_t projectionEndCycle =
        static_cast<std::size_t>(projectionTimeline->end_cycle);
    if ((program.bindings.size() != 9 && program.bindings.size() != 11)
        || program.max_cycle <= projectionEndCycle)
        throw std::logic_error("attention binary is missing bindings or projection commands");
    std::size_t causal_mask_bindings = 0;
    std::size_t rope_bindings = 0;
    std::vector<std::uint16_t> first_mask_plane;
    for (const auto& binding : program.bindings) {
        if (binding.access != ftlpu::software::runtime::BindingAccess::Internal)
            continue;
        if (binding.initializer
            == ftlpu::software::runtime::BindingInitializer::RopeTable) {
            if (binding.layout
                    != ftlpu::software::runtime::BindingLayout::Fp16RopeTable
                || binding.element_type
                    != ftlpu::software::runtime::BindingElementType::BF16
                || binding.shape
                    != std::vector<std::uint64_t>({kSeqLen, kHeadDim / 2, 2})
                || binding.slices != std::vector<std::uint16_t>({2, 18, 30, 31})
                || binding.base_row != 7000
                || binding.rope_head_dim != kHeadDim
                || std::fabs(binding.rope_theta - 100000.0f) > 0.5f)
                throw std::logic_error(
                    "attention binary has an invalid RoPE-table binding");
            ++rope_bindings;
            continue;
        }
        if (binding.initializer
            != ftlpu::software::runtime::BindingInitializer::CausalMask)
            throw std::logic_error(
                "attention binary has an unknown internal initializer");
        const bool valid_mask_plane = binding.slices.size() == sizeof(float)
            && std::adjacent_find(binding.slices.begin(), binding.slices.end(),
                   [](std::uint16_t lhs, std::uint16_t rhs) {
                       return rhs != lhs + 1;
                   })
                == binding.slices.end();
        if (binding.layout
                != ftlpu::software::runtime::BindingLayout::Fp32CausalMaskTile
            || binding.shape != std::vector<std::uint64_t>({kTile - 1, kTile})
            || !valid_mask_plane || binding.base_row != 8128)
            throw std::logic_error("attention binary has an invalid causal-mask binding");
        if (first_mask_plane.empty())
            first_mask_plane = binding.slices;
        else if (std::any_of(binding.slices.begin(), binding.slices.end(),
                     [&](std::uint16_t slice) {
                         return std::find(first_mask_plane.begin(),
                                    first_mask_plane.end(), slice)
                             != first_mask_plane.end();
                     }))
            throw std::logic_error("attention causal masks overlap");
        ++causal_mask_bindings;
    }
    if (causal_mask_bindings != 2 && causal_mask_bindings != 4)
        throw std::logic_error("attention binary is missing its internal causal mask");
    if (rope_bindings != 1)
        throw std::logic_error("attention binary is missing its internal RoPE table");
    if (const auto* trace_path = std::getenv("FTLPU_SCHEDULE_TRACE")) {
        ftlpu::software::runtime::write_schedule_trace_csv(program, trace_path);
    }
    if (std::getenv("FTLPU_TRACE_ONLY")) return 0;

    std::vector<std::uint8_t> input;
    input.reserve(kSeqLen * kHidden * 2);
    for (std::size_t token = 0; token < kSeqLen; ++token)
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden)
            append_bf16(input, activation(token, hidden));
    const auto query_weight = make_weight(Projection::Query);
    const auto key_weight = make_weight(Projection::Key);
    const auto value_weight = make_weight(Projection::Value);
    const auto output_weight = make_output_weight();

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, input);
    runtime.upload_input(1, query_weight);
    runtime.upload_input(2, key_weight);
    runtime.upload_input(3, value_weight);
    runtime.upload_input(4, output_weight);
    runtime.run_cycles(projectionEndCycle);

    constexpr std::size_t sample_tokens[] = {
        0, 17, 31, 32, 63, 64, 95, 127};
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
    runtime.run_cycles(program.max_cycle + 64 - projectionEndCycle);
    std::array<std::array<std::vector<float>, kSampleQueries.size()>,
        kQueryHeads>
        reference_probabilities;
    std::size_t probability_nonzero = 0;
    std::size_t probability_rows = 0;
    std::size_t causal_zero_checked = 0;
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        for (std::size_t query_index = 0;
             query_index < kSampleQueries.size(); ++query_index) {
            const std::size_t query = kSampleQueries[query_index];
            auto& reference_row =
                reference_probabilities[head][query_index];
            reference_row = reference_probability(head, query);
            float sum = 0.0f;
            for (std::size_t key = 0; key < kSeqLen; ++key) {
                const float probability =
                    read_packed_probability(*system, head, query, key);
                const float diagonal = read_diagonal_probability(*system, head, query, key);
                const float reference = reference_row[key];
                if (!std::isfinite(probability) || probability < 0.0f)
                    throw std::logic_error("attention softmax produced an invalid probability");
                if (std::fabs(probability - reference) > 0.015f)
                    throw std::logic_error("attention softmax CPU baseline mismatch: head="
                        + std::to_string(head) + " query=" + std::to_string(query)
                        + " key=" + std::to_string(key)
                        + " actual=" + std::to_string(probability)
                        + " expected=" + std::to_string(reference));
                if (key > query) {
                    if (probability != 0.0f)
                        throw std::logic_error("causal softmax produced a nonzero future probability");
                    ++causal_zero_checked;
                }
                if (diagonal != probability)
                    throw std::logic_error("SXM diagonal probability mismatch: head="
                        + std::to_string(head) + " query=" + std::to_string(query)
                        + " key=" + std::to_string(key)
                        + " actual=" + std::to_string(diagonal)
                        + " expected=" + std::to_string(probability));
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
    if (causal_zero_checked == 0)
        throw std::logic_error("attention test did not check any causal-mask entries");
    std::array<std::array<std::array<float, kHeadDim>,
                   kSampleQueries.size()>,
        kQueryHeads>
        reference_contexts {};
    std::size_t context_checked = 0;
    std::size_t context_nonzero = 0;
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        for (std::size_t query_index = 0;
             query_index < kSampleQueries.size(); ++query_index) {
            const std::size_t query = kSampleQueries[query_index];
            for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
                const float reference = reference_context(
                    reference_probabilities[head][query_index],
                    head, dimension);
                reference_contexts[head][query_index][dimension] =
                    reference;
                const float actual = read_context(*system, head, query, dimension);
                if (std::fabs(actual) > 0.0005f) ++context_nonzero;
                if (std::fabs(actual - reference) > 0.02f)
                    throw std::logic_error("PV context CPU baseline mismatch: head="
                        + std::to_string(head) + " query=" + std::to_string(query)
                        + " dimension=" + std::to_string(dimension)
                        + " actual=" + std::to_string(actual)
                        + " expected=" + std::to_string(reference));
                ++context_checked;
            }
        }
    }
    if (context_nonzero == 0)
        throw std::logic_error("PV produced only zero context values");
    const auto output = runtime.download_output(0);
    std::size_t output_checked = 0;
    std::size_t output_nonzero = 0;
    for (std::size_t query_index = 0;
         query_index < kSampleQueries.size(); ++query_index) {
        const std::size_t token = kSampleQueries[query_index];
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float actual = bf16_at(output, token * kHidden + column);
            float reference = 0.0f;
            for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
                reference +=
                    reference_contexts[hidden / kHeadDim][query_index]
                                      [hidden % kHeadDim]
                    * output_weight_value(hidden, column);
            }
            reference =
                ftlpu::Bf16::from_float(reference).to_float();
            if (std::fabs(actual) > 0.0005f) ++output_nonzero;
            if (std::fabs(actual - reference) > 0.02f) {
                throw std::logic_error("O projection CPU baseline mismatch: token="
                    + std::to_string(token) + " column=" + std::to_string(column)
                    + " actual=" + std::to_string(actual)
                    + " expected=" + std::to_string(reference));
            }
            ++output_checked;
        }
    }
    if (output_nonzero == 0)
        throw std::logic_error("O projection produced only zero output values");
    ftlpu::software::runtime::print_runtime_performance(
        program, program.max_cycle + 64, std::cout);
    std::cout << "Complete Attention CPU baseline passed; Q/K/V projection + RoPE: " << checked
              << " sampled BF16 values, nonzero=" << nonzero
              << ", projection_end_cycle=" << projectionEndCycle
              << "; softmax + SXM diagonal layout passed: rows=" << probability_rows
              << ", nonzero=" << probability_nonzero
              << ", causal_zero_checked=" << causal_zero_checked
              << "; PV context passed: values=" << context_checked
              << ", nonzero=" << context_nonzero
              << "; O projection passed: values=" << output_checked
              << ", nonzero=" << output_nonzero
              << ", max_cycle=" << program.max_cycle << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_smollm2_attention_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
