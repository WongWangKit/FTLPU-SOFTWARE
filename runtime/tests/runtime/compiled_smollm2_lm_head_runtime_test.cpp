#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kRows = 32;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kShard = 4096;
constexpr std::size_t kVocab = 49152;
constexpr std::size_t kShardCount = kVocab / kShard;

std::vector<std::uint8_t> readBytes(
    const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error(
            "cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

float readBf16(
    const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    const auto bits = static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
    return ftlpu::Bf16::from_bits(bits).to_float();
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 5 && argc != 6)
        throw std::runtime_error(
            "usage: compiled_smollm2_lm_head_runtime_test "
            "program.ftlpu activation.bf16 weight.i8 golden.bf16 "
            "[shard_count]");
    const std::size_t shardCount = argc == 6
        ? static_cast<std::size_t>(std::stoul(argv[5]))
        : kShardCount;
    if (shardCount == 0 || shardCount > kShardCount)
        throw std::runtime_error("shard_count must be in [1, 12]");
    const auto program =
        ftlpu::software::runtime::read_binary_program(argv[1]);
    const auto activation = readBytes(argv[2]);
    const auto weight = readBytes(argv[3]);
    const auto golden = readBytes(argv[4]);
    const auto hasShape = [](const auto& binding,
                              std::size_t rows,
                              std::size_t columns) {
        return binding.shape.size() == 2
            && static_cast<std::size_t>(binding.shape[0]) == rows
            && static_cast<std::size_t>(binding.shape[1]) == columns;
    };
    if (program.bindings.size() != 3
        || !hasShape(program.bindings[0], kRows, kHidden)
        || !hasShape(program.bindings[1], kHidden, kShard)
        || !hasShape(program.bindings[2], kRows, kShard))
        throw std::logic_error(
            "LM-head shard binary has unexpected bindings");
    if (activation.size() != kRows * kHidden * 2
        || weight.size() != kHidden * kVocab
        || golden.size() != kRows * kVocab * 2)
        throw std::logic_error(
            "LM-head real-data file has an unexpected size");

    auto* system = new ftlpu::TspSliceSystem();
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    std::vector<std::uint8_t> weightShard(kHidden * kShard);
    const std::size_t processedVocab = shardCount * kShard;
    std::vector<float> actualLogits(kRows * processedVocab);
    std::size_t nonzero = 0;
    for (std::size_t shard = 0; shard < shardCount; ++shard) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto source = weight.begin()
                + static_cast<std::ptrdiff_t>(
                    hidden * kVocab + shard * kShard);
            std::copy_n(source, kShard,
                weightShard.begin()
                    + static_cast<std::ptrdiff_t>(hidden * kShard));
        }
        runtime.load(program);
        runtime.upload_input(0, activation);
        runtime.upload_input(1, weightShard);
        runtime.run_cycles(program.max_cycle + 64);
        const auto output = runtime.download_output(0);
        for (std::size_t row = 0; row < kRows; ++row) {
            for (std::size_t column = 0; column < kShard; ++column) {
                const float value =
                    readBf16(output, row * kShard + column);
                actualLogits[row * processedVocab
                    + shard * kShard + column] = value;
                if (std::fabs(value) > 1.0e-6f) ++nonzero;
            }
        }
        std::cout << "LM-head CModel shard " << shard + 1
                  << "/" << shardCount
                  << " passed through ICU" << std::endl;
    }
    if (nonzero == 0)
        throw std::logic_error(
            "LM-head CModel produced only zero logits");

    double dot = 0.0;
    double actualNorm = 0.0;
    double goldenNorm = 0.0;
    double absoluteError = 0.0;
    float maxError = 0.0f;
    std::size_t maxErrorIndex = 0;
    for (std::size_t index = 0;
         index < actualLogits.size(); ++index) {
        const std::size_t row = index / processedVocab;
        const std::size_t column = index % processedVocab;
        const float expected =
            readBf16(golden, row * kVocab + column);
        const float observed = actualLogits[index];
        if (!std::isfinite(observed))
            throw std::logic_error(
                "LM-head CModel produced a non-finite logit");
        const float error = std::fabs(observed - expected);
        absoluteError += error;
        dot += static_cast<double>(observed) * expected;
        actualNorm += static_cast<double>(observed) * observed;
        goldenNorm += static_cast<double>(expected) * expected;
        if (error > maxError) {
            maxError = error;
            maxErrorIndex = index;
        }
    }
    const double cosine =
        dot / std::sqrt(actualNorm * goldenNorm);
    const double meanAbsoluteError =
        absoluteError / actualLogits.size();
    const std::size_t sampleRow = kRows - 1;
    const auto actualTop = std::max_element(
        actualLogits.begin() + sampleRow * processedVocab,
        actualLogits.begin() + (sampleRow + 1) * processedVocab);
    std::size_t goldenTop = 0;
    float goldenTopValue = -std::numeric_limits<float>::infinity();
    for (std::size_t column = 0;
         column < processedVocab; ++column) {
        const float value =
            readBf16(golden, sampleRow * kVocab + column);
        if (value > goldenTopValue) {
            goldenTopValue = value;
            goldenTop = column;
        }
    }
    const std::size_t actualTopIndex = static_cast<std::size_t>(
        actualTop - actualLogits.begin())
        - sampleRow * processedVocab;
    if (cosine < 0.995 || meanAbsoluteError > 0.02)
        throw std::logic_error(
            "LM-head CModel numeric baseline failed: cosine="
            + std::to_string(cosine) + " mean_abs_error="
            + std::to_string(meanAbsoluteError)
            + " max_error=" + std::to_string(maxError)
            + " max_error_row="
            + std::to_string(maxErrorIndex / processedVocab)
            + " max_error_column="
            + std::to_string(maxErrorIndex % processedVocab));

    std::cout << "SmolLM2-135M real INT8 LM head passed: "
              << actualLogits.size() << " BF16 logits, "
              << shardCount << " CModel executions, cycles="
              << shardCount * (program.max_cycle + 64)
              << ", cosine=" << cosine
              << ", mean_abs_error=" << meanAbsoluteError
              << ", max_error=" << maxError
              << ", row31 top1(actual/golden)="
              << actualTopIndex << "/" << goldenTop
              << ", nonzero=" << nonzero << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_smollm2_lm_head_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
