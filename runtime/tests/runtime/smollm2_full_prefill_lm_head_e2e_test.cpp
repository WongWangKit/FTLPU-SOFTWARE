#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kLmRows = 32;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kShard = 4096;
constexpr std::size_t kVocab = 49152;
constexpr std::size_t kShardCount = kVocab / kShard;
constexpr std::size_t kDrainCycles = 64;

std::vector<std::uint8_t> readBytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

float readBf16(
    const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * sizeof(std::uint16_t);
    if (offset + sizeof(std::uint16_t) > bytes.size())
        throw std::out_of_range("BF16 read exceeds tensor size");
    const auto bits = static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
    return ftlpu::Bf16::from_bits(bits).to_float();
}

void writeBf16(
    std::vector<std::uint8_t>& bytes, std::size_t index, float value)
{
    const std::uint16_t bits = ftlpu::Bf16::from_float(value).bits();
    const std::size_t offset = index * sizeof(bits);
    bytes[offset] = static_cast<std::uint8_t>(bits);
    bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
}

const ftlpu::software::runtime::ModelValue& findValue(
    const ftlpu::software::runtime::ModelPackage& package,
    const std::string& name)
{
    const auto value = std::find_if(package.values.begin(),
        package.values.end(), [&](const auto& candidate) {
            return candidate.name == name;
        });
    if (value == package.values.end())
        throw std::logic_error("model package has no value " + name);
    return *value;
}

std::vector<std::size_t> topKIndices(
    const std::vector<float>& values, std::size_t count)
{
    std::vector<std::size_t> indices(values.size());
    for (std::size_t index = 0; index < indices.size(); ++index)
        indices[index] = index;
    count = std::min(count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count,
        indices.end(), [&](std::size_t lhs, std::size_t rhs) {
            return values[lhs] > values[rhs];
        });
    indices.resize(count);
    return indices;
}

struct Similarity {
    double cosine{0.0};
    double meanAbsoluteError{0.0};
    float maximumAbsoluteError{0.0f};
};

Similarity compare(
    const std::vector<float>& actual,
    const std::vector<float>& expected)
{
    if (actual.size() != expected.size() || actual.empty())
        throw std::logic_error("cannot compare mismatched tensors");
    double dot = 0.0;
    double actualNorm = 0.0;
    double expectedNorm = 0.0;
    double absoluteError = 0.0;
    float maximumAbsoluteError = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])
            || !std::isfinite(expected[index]))
            throw std::logic_error(
                "end-to-end comparison contains a non-finite value");
        const float error = std::fabs(actual[index] - expected[index]);
        absoluteError += error;
        maximumAbsoluteError =
            std::max(maximumAbsoluteError, error);
        dot += static_cast<double>(actual[index]) * expected[index];
        actualNorm +=
            static_cast<double>(actual[index]) * actual[index];
        expectedNorm +=
            static_cast<double>(expected[index]) * expected[index];
    }
    return {
        dot / std::sqrt(actualNorm * expectedNorm),
        absoluteError / actual.size(),
        maximumAbsoluteError,
    };
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 5)
        throw std::runtime_error(
            "usage: smollm2_full_prefill_lm_head_e2e_test "
            "model.ftlpum lm_head.ftlpu lm_head_weight.i8 "
            "weight_scale");

    using namespace ftlpu::software::runtime;
    ModelPackage package = read_model_package(
        std::filesystem::path(argv[1]),
        ModelPackageLoadMode::LazyExecutables);
    if (package.host_lm_heads.size() != 1
        || package.host_lm_heads.front().hidden != "final_hidden")
        throw std::logic_error(
            "full-prefill package must contain the replaceable host LM head");
    const ModelValue& tokenIdsMetadata = findValue(package, "token_ids");
    const ModelValue& finalHiddenMetadata =
        findValue(package, "final_hidden");
    if (tokenIdsMetadata.shape.size() != 1
        || finalHiddenMetadata.shape.size() != 2
        || finalHiddenMetadata.shape[0] != tokenIdsMetadata.shape[0]
        || finalHiddenMetadata.shape[1] != kHidden)
        throw std::logic_error(
            "full-prefill package has unexpected model dimensions");
    const std::size_t sequenceLength =
        static_cast<std::size_t>(tokenIdsMetadata.shape[0]);

    std::size_t prefillCycles = 0;
    for (const ModelInvocation& invocation : package.invocations) {
        const BinaryProgram& program =
            package.executables.at(invocation.executable_index).program;
        prefillCycles += program.max_cycle + kDrainCycles;
    }
    package.host_lm_heads.clear();

    const BinaryProgram lmProgram =
        read_binary_program(std::filesystem::path(argv[2]));
    const auto hasShape = [](const BinaryBinding& binding,
                              std::size_t rows,
                              std::size_t columns) {
        return binding.shape.size() == 2
            && static_cast<std::size_t>(binding.shape[0]) == rows
            && static_cast<std::size_t>(binding.shape[1]) == columns;
    };
    if (lmProgram.bindings.size() != 3
        || !hasShape(lmProgram.bindings[0], kLmRows, kHidden)
        || !hasShape(lmProgram.bindings[1], kHidden, kShard)
        || !hasShape(lmProgram.bindings[2], kLmRows, kShard))
        throw std::logic_error(
            "LM-head shard binary has unexpected bindings");

    const std::vector<std::uint8_t> quantizedWeight =
        readBytes(std::filesystem::path(argv[3]));
    if (quantizedWeight.size() != kHidden * kVocab)
        throw std::logic_error(
            "LM-head INT8 weight has an unexpected size");
    const float weightScale = std::stof(argv[4]);
    if (!std::isfinite(weightScale) || weightScale <= 0.0f)
        throw std::invalid_argument(
            "LM-head weight scale must be finite and positive");

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ModelSession session(*system);
    session.load(std::move(package));
    std::vector<std::int32_t> tokenIds(sequenceLength);
    for (std::size_t index = 0; index < sequenceLength; ++index)
        tokenIds[index] = static_cast<std::int32_t>(index);
    session.set_input("token_ids",
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(tokenIds.data()),
            tokenIds.size() * sizeof(tokenIds[0])));
    session.run(kDrainCycles);

    const auto& finalHidden = session.value("final_hidden");
    const auto& decoderGolden = session.value("golden.output");
    const auto& finalNormWeight = session.value("model.norm.weight");
    const auto& embedding = session.value("model.embed_tokens.weight");
    if (finalHidden.size() != sequenceLength * kHidden * 2
        || decoderGolden.size() != finalHidden.size()
        || finalNormWeight.size() != kHidden * 2
        || embedding.size() != kVocab * kHidden * 2)
        throw std::logic_error(
            "full-prefill package has unexpected BF16 tensor data");
    if (session.stats().host_operations != 1)
        throw std::logic_error(
            "combined test must run only embedding on the host");

    const std::size_t lastRow = sequenceLength - 1;
    double lastRowSumSquares = 0.0;
    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        const float value =
            readBf16(decoderGolden, lastRow * kHidden + hidden);
        lastRowSumSquares += static_cast<double>(value) * value;
    }
    const float inverseRms = 1.0f / std::sqrt(
        static_cast<float>(lastRowSumSquares / kHidden) + 1.0e-5f);
    std::vector<float> expectedFinalHidden(kHidden);
    std::vector<float> actualFinalHidden(kHidden);
    for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
        expectedFinalHidden[hidden] = ftlpu::Bf16::from_float(
            readBf16(decoderGolden, lastRow * kHidden + hidden)
            * inverseRms * readBf16(finalNormWeight, hidden))
                                              .to_float();
        actualFinalHidden[hidden] =
            readBf16(finalHidden, lastRow * kHidden + hidden);
    }
    const Similarity finalNormSimilarity =
        compare(actualFinalHidden, expectedFinalHidden);
    if (finalNormSimilarity.cosine < 0.995
        || finalNormSimilarity.meanAbsoluteError > 0.04)
        throw std::logic_error(
            "full-prefill final RMSNorm baseline failed");

    // The LM-head executable consumes one 32-row hardware tile. Place the
    // prefill result in its final row and leave the padded rows at zero.
    std::vector<std::uint8_t> lmActivation(
        kLmRows * kHidden * sizeof(std::uint16_t), 0);
    for (std::size_t hidden = 0; hidden < kHidden; ++hidden)
        writeBf16(lmActivation,
            (kLmRows - 1) * kHidden + hidden,
            actualFinalHidden[hidden]);

    CModelRuntime lmRuntime(*system);
    std::vector<std::uint8_t> weightShard(kHidden * kShard);
    std::vector<float> lpuLogits(kVocab);
    std::size_t nonzeroLogits = 0;
    float maximumPaddedLogit = 0.0f;
    for (std::size_t shard = 0; shard < kShardCount; ++shard) {
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto source = quantizedWeight.begin()
                + static_cast<std::ptrdiff_t>(
                    hidden * kVocab + shard * kShard);
            std::copy_n(source, kShard,
                weightShard.begin()
                    + static_cast<std::ptrdiff_t>(hidden * kShard));
        }
        lmRuntime.load(lmProgram);
        lmRuntime.upload_input(0, lmActivation);
        lmRuntime.upload_input(1, weightShard);
        lmRuntime.run_cycles(lmProgram.max_cycle + kDrainCycles);
        const auto output = lmRuntime.download_output(0);
        for (std::size_t row = 0; row < kLmRows; ++row) {
            for (std::size_t column = 0; column < kShard; ++column) {
                const float value =
                    readBf16(output, row * kShard + column);
                if (row + 1 == kLmRows) {
                    lpuLogits[shard * kShard + column] = value;
                    if (std::fabs(value) > 1.0e-6f)
                        ++nonzeroLogits;
                } else {
                    maximumPaddedLogit = std::max(
                        maximumPaddedLogit, std::fabs(value));
                }
            }
        }
        std::cout << "combined prefill LM-head shard "
                  << shard + 1 << '/' << kShardCount
                  << " completed through ICU" << std::endl;
    }
    if (nonzeroLogits == 0)
        throw std::logic_error(
            "combined LM head produced only zero logits");
    if (maximumPaddedLogit > 1.0e-3f)
        throw std::logic_error(
            "zero-padded LM-head rows produced nonzero logits");

    std::vector<float> quantizedGolden(kVocab);
    std::vector<float> modelGolden(kVocab);
    for (std::size_t token = 0; token < kVocab; ++token) {
        float quantizedAccumulator = 0.0f;
        float modelAccumulator = 0.0f;
        for (std::size_t hidden = 0; hidden < kHidden; ++hidden) {
            const auto quantized = static_cast<std::int8_t>(
                quantizedWeight[hidden * kVocab + token]);
            const float dequantized = ftlpu::Bf16::from_float(
                static_cast<float>(quantized) * weightScale)
                                          .to_float();
            quantizedAccumulator +=
                actualFinalHidden[hidden] * dequantized;
            modelAccumulator += expectedFinalHidden[hidden]
                * readBf16(embedding, token * kHidden + hidden);
        }
        quantizedGolden[token] =
            ftlpu::Bf16::from_float(quantizedAccumulator).to_float();
        modelGolden[token] = modelAccumulator;
    }

    const Similarity lmSimilarity =
        compare(lpuLogits, quantizedGolden);
    const Similarity endToEndSimilarity =
        compare(lpuLogits, modelGolden);
    const auto actualTop5 = topKIndices(lpuLogits, 5);
    const auto expectedTop5 = topKIndices(modelGolden, 5);
    const std::size_t top5Overlap = std::count_if(
        expectedTop5.begin(), expectedTop5.end(),
        [&](std::size_t index) {
            return std::find(actualTop5.begin(), actualTop5.end(), index)
                != actualTop5.end();
        });
    if (lmSimilarity.cosine < 0.995
        || lmSimilarity.meanAbsoluteError > 0.02)
        throw std::logic_error(
            "LPU LM-head quantized golden mismatch");
    if (endToEndSimilarity.cosine < 0.99 || top5Overlap < 3)
        throw std::logic_error(
            "combined prefill-to-logits model baseline mismatch");

    const std::size_t lmHeadCycles =
        kShardCount * (lmProgram.max_cycle + kDrainCycles);
    std::cout
        << "SmolLM2-135M seq128 prefill + LPU LM head passed"
        << " prefill_cycles=" << prefillCycles
        << " lm_head_cycles=" << lmHeadCycles
        << " total_cycles=" << prefillCycles + lmHeadCycles
        << " final_norm_cosine=" << finalNormSimilarity.cosine
        << " final_norm_mean_abs_error="
        << finalNormSimilarity.meanAbsoluteError
        << " lm_quantized_cosine=" << lmSimilarity.cosine
        << " lm_quantized_mean_abs_error="
        << lmSimilarity.meanAbsoluteError
        << " e2e_model_cosine=" << endToEndSimilarity.cosine
        << " expected_top1=" << expectedTop5.front()
        << " actual_top1=" << actualTop5.front()
        << " top5_overlap=" << top5Overlap
        << " nonzero_logits=" << nonzeroLogits
        << " padded_max_abs=" << maximumPaddedLogit
        << " host_operations=" << session.stats().host_operations
        << " device_aliases=" << session.stats().device_aliases
        << " device_copies=" << session.stats().device_copies
        << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr
        << "smollm2_full_prefill_lm_head_e2e_test failed: "
        << exception.what() << '\n';
    return 1;
}
