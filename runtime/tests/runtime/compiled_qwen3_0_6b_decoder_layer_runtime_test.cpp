#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 32;
constexpr std::size_t kHidden = 1024;
constexpr std::size_t kIntermediate = 3072;
constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kKvHeads = 8;
constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kQueryWidth = kQueryHeads * kHeadDim;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
constexpr float kEpsilon = 1.0e-6f;
constexpr float kRopeTheta = 1'000'000.0f;

float bf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

void appendBf16(std::vector<std::uint8_t>& bytes, float value)
{
    const std::uint16_t bits = ftlpu::Bf16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float readBf16(const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8))
        .to_float();
}

float inputValue(std::size_t row, std::size_t column)
{
    const int value = static_cast<int>((row * 13 + column * 7) % 31) - 15;
    return bf16(static_cast<float>(value) / 32.0f);
}

float layerGamma(std::size_t column, bool postAttention)
{
    const float base = postAttention ? 0.875f : 0.75f;
    return bf16(base + static_cast<float>(column % 9) / 32.0f);
}

float qkGamma(std::size_t column, bool key)
{
    return bf16(0.0625f
        + static_cast<float>((column + (key ? 3 : 0)) % 7) / 512.0f);
}

std::vector<std::uint8_t> encodeBf16(const std::vector<float>& values)
{
    std::vector<std::uint8_t> result;
    result.reserve(values.size() * 2);
    for (float value : values) appendBf16(result, value);
    return result;
}

std::vector<float> rmsNorm(const std::vector<float>& input,
    std::size_t rows, std::size_t width, bool postAttention)
{
    std::vector<float> output(input.size());
    const float meanWeight = bf16(1.0f / static_cast<float>(width));
    for (std::size_t row = 0; row < rows; ++row) {
        float meanSquare = 0.0f;
        for (std::size_t column = 0; column < width; ++column) {
            const float value = input[row * width + column];
            meanSquare += bf16(value * value) * meanWeight;
        }
        const float factor = bf16(
            1.0f / std::sqrt(meanSquare + kEpsilon));
        for (std::size_t column = 0; column < width; ++column)
            output[row * width + column] = bf16(
                input[row * width + column] * factor
                * layerGamma(column, postAttention));
    }
    return output;
}

void headRmsNorm(std::vector<float>& values, std::size_t heads, bool key)
{
    const float meanWeight = bf16(1.0f / static_cast<float>(kHeadDim));
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t base =
                (token * heads + head) * kHeadDim;
            float meanSquare = 0.0f;
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension) {
                const float value = values[base + dimension];
                meanSquare += bf16(value * value) * meanWeight;
            }
            const float factor = bf16(
                1.0f / std::sqrt(meanSquare + kEpsilon));
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension)
                values[base + dimension] = bf16(
                    values[base + dimension] * factor
                    * qkGamma(dimension, key));
        }
    }
}

std::vector<float> rope(const std::vector<float>& input, std::size_t heads)
{
    std::vector<float> output(input.size());
    const std::size_t half = kHeadDim / 2;
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t base =
                (token * heads + head) * kHeadDim;
            for (std::size_t pair = 0; pair < half; ++pair) {
                const float inverse = 1.0f / std::pow(kRopeTheta,
                    static_cast<float>(2 * pair) / kHeadDim);
                const float angle = static_cast<float>(token) * inverse;
                const float cosine = bf16(std::cos(angle));
                const float sine = bf16(std::sin(angle));
                const float low = input[base + pair];
                const float high = input[base + pair + half];
                output[base + pair] = bf16(
                    low * cosine - high * sine);
                output[base + pair + half] = bf16(
                    high * cosine + low * sine);
            }
        }
    }
    return output;
}

const ftlpu::software::runtime::BinaryTimeline& timeline(
    const ftlpu::software::runtime::BinaryProgram& program,
    const char* name, std::size_t occurrence = 0)
{
    for (const auto& candidate : program.timelines) {
        if (candidate.name != name) continue;
        if (occurrence == 0) return candidate;
        --occurrence;
    }
    throw std::logic_error(
        std::string("cannot locate binary timeline: ") + name);
}

bool hasInputBinding(const ftlpu::software::runtime::BinaryProgram& program,
    std::size_t index, const char* name = nullptr)
{
    return std::any_of(program.bindings.begin(), program.bindings.end(),
        [&](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Input
                && binding.index == index
                && (!name || binding.name == name);
        });
}

const ftlpu::software::runtime::BinaryBinding& inputBinding(
    const ftlpu::software::runtime::BinaryProgram& program,
    std::uint32_t index)
{
    for (const auto& binding : program.bindings)
        if (binding.access
                == ftlpu::software::runtime::BindingAccess::Input
            && binding.index == index)
            return binding;
    throw std::logic_error(
        "Qwen3 binary is missing input binding " + std::to_string(index));
}

ftlpu::software::runtime::ModelTensor makeTensor(
    const ftlpu::software::runtime::BinaryProgram& program,
    std::uint32_t bindingIndex, std::string name,
    std::vector<std::uint8_t> data,
    ftlpu::software::runtime::ModelTensorEncoding encoding,
    std::vector<float> scales = {})
{
    const auto& binding = inputBinding(program, bindingIndex);
    if (data.size() != binding.byte_size)
        throw std::logic_error(
            "Qwen3 tensor byte size does not match binding "
            + std::to_string(bindingIndex));
    ftlpu::software::runtime::ModelTensor tensor;
    tensor.name = std::move(name);
    tensor.element_type = binding.element_type;
    tensor.shape = binding.shape;
    tensor.data = std::move(data);
    tensor.encoding = encoding;
    tensor.scales = std::move(scales);
    return tensor;
}

void appendPackedTensor(
    ftlpu::software::runtime::ModelPackage& package,
    ftlpu::software::runtime::ModelWeightPage& page,
    const ftlpu::software::runtime::BinaryProgram& program,
    std::uint32_t bindingIndex, std::string name,
    std::vector<std::uint8_t> logical, std::uint16_t& nextStream)
{
    using namespace ftlpu::software::runtime;
    const auto& binding = inputBinding(program, bindingIndex);
    if (logical.size() != binding.byte_size)
        throw std::logic_error(
            "Qwen3 packed tensor byte size does not match binding "
            + std::to_string(bindingIndex));
    PackedWeightImage image =
        pack_binding_image(binding, logical, program.hardware);
    const std::string tensorName = name;
    package.tensors.push_back(ModelTensor{
        std::move(name), BindingElementType::I8,
        {static_cast<std::uint64_t>(image.data.size())},
        std::move(image.data), ModelTensorEncoding::TargetPackedSramVectors});
    page.tensors.push_back(tensorName);
    for (const auto& segment : image.segments) {
        page.segments.push_back(ModelWeightPage::Segment{
            tensorName, segment.byte_offset, segment.hemisphere,
            segment.slice, segment.base_row, segment.vector_count,
            nextStream});
        nextStream = static_cast<std::uint16_t>(
            (nextStream + 1)
            % program.hardware.c2c_streams_per_direction);
    }
}

std::size_t sourceHidden(std::size_t projection, std::size_t column)
{
    return (column * 7 + projection * 13 + 3) % kHidden;
}

float projectionSign(std::size_t projection, std::size_t column)
{
    return ((column + projection) & 1) ? -1.0f : 1.0f;
}

std::size_t gateK(std::size_t h) { return (h * 7 + 1) % kHidden; }
std::size_t upK(std::size_t h) { return (h * 11 + 3) % kHidden; }
float gateSign(std::size_t h) { return (h & 1) ? -1.0f : 1.0f; }
float upSign(std::size_t h) { return (h & 2) ? -1.0f : 1.0f; }

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: compiled_qwen3_decoder_layer_runtime_test program.ftlpu");
    const auto program =
        ftlpu::software::runtime::read_binary_program(
            std::filesystem::path(argv[1]));
    if (program.target_name != "ftlpu-lpu32" || program.max_cycle == 0)
        throw std::logic_error("Qwen3 binary has the wrong target metadata");
    if (!hasInputBinding(program, 6, "query_norm_weight")
        || !hasInputBinding(program, 7, "key_norm_weight")
        || !hasInputBinding(program, 11))
        throw std::logic_error(
            "Qwen3 binary has incomplete twelve-input layer bindings");
    for (const char* stage : {"qkv", "qk", "softmax", "pv", "o_proj",
             "ffn.down.vector"})
        (void)timeline(program, stage);

    std::vector<float> inputValues(kSeqLen * kHidden);
    for (std::size_t row = 0; row < kSeqLen; ++row)
        for (std::size_t column = 0; column < kHidden; ++column)
            inputValues[row * kHidden + column] = inputValue(row, column);

    std::vector<float> inputGamma(kHidden);
    std::vector<float> postGamma(kHidden);
    std::vector<float> queryNormGamma(kHeadDim);
    std::vector<float> keyNormGamma(kHeadDim);
    for (std::size_t column = 0; column < kHidden; ++column) {
        inputGamma[column] = layerGamma(column, false);
        postGamma[column] = layerGamma(column, true);
    }
    for (std::size_t column = 0; column < kHeadDim; ++column) {
        queryNormGamma[column] = qkGamma(column, false);
        keyNormGamma[column] = qkGamma(column, true);
    }

    std::vector<std::uint8_t> queryWeight(kHidden * kQueryWidth, 0);
    std::vector<std::uint8_t> keyWeight(kHidden * kKvWidth, 0);
    std::vector<std::uint8_t> valueWeight(kHidden * kKvWidth, 0);
    for (std::size_t column = 0; column < kQueryWidth; ++column)
        queryWeight[sourceHidden(0, column) * kQueryWidth + column] =
            static_cast<std::uint8_t>(
                static_cast<std::int8_t>(projectionSign(0, column)));
    for (std::size_t column = 0; column < kKvWidth; ++column) {
        keyWeight[sourceHidden(1, column) * kKvWidth + column] =
            static_cast<std::uint8_t>(
                static_cast<std::int8_t>(projectionSign(1, column)));
        valueWeight[sourceHidden(2, column) * kKvWidth + column] =
            static_cast<std::uint8_t>(
                static_cast<std::int8_t>(projectionSign(2, column)));
    }

    const auto contextK = [](std::size_t column) {
        return (column * 17 + 5) % kQueryWidth;
    };
    const auto outputSign = [](std::size_t column) {
        return (column & 2) ? -1.0f : 1.0f;
    };
    std::vector<std::uint8_t> outputWeight(kQueryWidth * kHidden, 0);
    for (std::size_t column = 0; column < kHidden; ++column)
        outputWeight[contextK(column) * kHidden + column] =
            static_cast<std::uint8_t>(
                static_cast<std::int8_t>(outputSign(column)));

    std::vector<std::uint8_t> gateWeight(kHidden * kIntermediate, 0);
    std::vector<std::uint8_t> upWeight(kHidden * kIntermediate, 0);
    for (std::size_t h = 0; h < kIntermediate; ++h) {
        gateWeight[gateK(h) * kIntermediate + h] =
            static_cast<std::uint8_t>(
                static_cast<std::int8_t>(gateSign(h)));
        upWeight[upK(h) * kIntermediate + h] =
            static_cast<std::uint8_t>(
                static_cast<std::int8_t>(upSign(h)));
    }
    std::vector<std::uint8_t> downWeight(kIntermediate * kHidden, 0);
    for (std::size_t column = 0; column < kHidden; ++column) {
        const std::size_t h0 = (column * 5 + 17) % kIntermediate;
        const std::size_t h1 = (h0 + 37) % kIntermediate;
        downWeight[h0 * kHidden + column] = 1;
        downWeight[h1 * kHidden + column] =
            static_cast<std::uint8_t>(static_cast<std::int8_t>(-1));
    }

    using namespace ftlpu::software::runtime;
    const auto& input = inputBinding(program, 0);
    const auto& output = std::ranges::find_if(program.bindings,
        [](const BinaryBinding& binding) {
            return binding.access == BindingAccess::Output
                && binding.index == 0;
        });
    if (output == program.bindings.end())
        throw std::logic_error("Qwen3 binary is missing output binding 0");
    const std::uint64_t maxCycle = program.max_cycle;

    ModelPackage package;
    package.model_name = "Qwen3-0.6B-layer0-seq32";
    package.architecture = "Qwen3ForCausalLM";
    package.tensors.reserve(11);
    ModelWeightPage parameterPage;
    parameterPage.layer = 0;
    parameterPage.bank = inputBinding(program, 1).bank;
    std::uint16_t nextParameterStream = 0;
    appendPackedTensor(package, parameterPage, program, 1,
        "input_layernorm.weight", encodeBf16(inputGamma),
        nextParameterStream);
    appendPackedTensor(package, parameterPage, program, 8,
        "post_attention_layernorm.weight", encodeBf16(postGamma),
        nextParameterStream);
    package.tensors.push_back(makeTensor(program, 2,
        "self_attn.q_proj.weight", queryWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.tensors.push_back(makeTensor(program, 3,
        "self_attn.k_proj.weight", keyWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.tensors.push_back(makeTensor(program, 4,
        "self_attn.v_proj.weight", valueWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.tensors.push_back(makeTensor(program, 5,
        "self_attn.o_proj.weight", outputWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.tensors.push_back(makeTensor(program, 6,
        "self_attn.q_norm.weight", encodeBf16(queryNormGamma),
        ModelTensorEncoding::Raw));
    package.tensors.push_back(makeTensor(program, 7,
        "self_attn.k_norm.weight", encodeBf16(keyNormGamma),
        ModelTensorEncoding::Raw));
    package.tensors.push_back(makeTensor(program, 9,
        "mlp.gate_proj.weight", gateWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.tensors.push_back(makeTensor(program, 10,
        "mlp.up_proj.weight", upWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.tensors.push_back(makeTensor(program, 11,
        "mlp.down_proj.weight", downWeight,
        ModelTensorEncoding::SymmetricPerTensorI8, {1.0f}));
    package.weight_pages.push_back(std::move(parameterPage));
    package.values = {
        {"hidden.0", input.element_type, input.shape, true, false},
        {"hidden.1", output->element_type, output->shape, false, true},
    };
    package.executables.push_back(
        {"decoder.layer0", program, {}});
    package.invocations.push_back(ModelInvocation{
        "decoder.layer0", 0,
        {{0, "hidden.0"},
         {1, "input_layernorm.weight"},
         {2, "self_attn.q_proj.weight"},
         {3, "self_attn.k_proj.weight"},
         {4, "self_attn.v_proj.weight"},
         {5, "self_attn.o_proj.weight"},
         {6, "self_attn.q_norm.weight"},
         {7, "self_attn.k_norm.weight"},
         {8, "post_attention_layernorm.weight"},
         {9, "mlp.gate_proj.weight"},
         {10, "mlp.up_proj.weight"},
         {11, "mlp.down_proj.weight"}},
        {{0, "hidden.1"}}, {}, 0});

    const char* tracePath = std::getenv("FTLPU_QWEN_PIPELINE_CSV");
    std::vector<std::uint8_t> actual;
    ModelSessionStats sessionStats;
    if (std::getenv("FTLPU_QWEN_DIRECT_RUNTIME") != nullptr) {
        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        CModelRuntime runtime(*system);
        if (tracePath != nullptr) runtime.enable_execution_trace();
        runtime.load(program);
        runtime.upload_input(0, encodeBf16(inputValues));
        runtime.upload_input(1, encodeBf16(inputGamma));
        runtime.upload_input(2, queryWeight);
        runtime.upload_input(3, keyWeight);
        runtime.upload_input(4, valueWeight);
        runtime.upload_input(5, outputWeight);
        runtime.upload_input(6, encodeBf16(queryNormGamma));
        runtime.upload_input(7, encodeBf16(keyNormGamma));
        runtime.upload_input(8, encodeBf16(postGamma));
        runtime.upload_input(9, gateWeight);
        runtime.upload_input(10, upWeight);
        runtime.upload_input(11, downWeight);
        runtime.run_cycles(maxCycle + 64);
        actual = runtime.download_output(0);
        if (tracePath != nullptr)
            runtime.write_execution_trace_csv(tracePath);
    } else {
        ftlpu::C2cDmaSystem system;
        ModelSession session(system);
        session.load(std::move(package));
        session.set_input("hidden.0", encodeBf16(inputValues));
        if (tracePath != nullptr) session.enable_execution_trace();
        session.run();
        actual = session.value("hidden.1");
        if (tracePath != nullptr)
            session.write_execution_trace_csv(tracePath);
        sessionStats = session.stats();
        if (sessionStats.weight_page_prefetches < 4
            || sessionStats.c2c_ingress_bytes == 0
            || sessionStats.c2c_egress_bytes == 0)
            throw std::logic_error(
                "Qwen3 ModelSession did not exercise the complete C2C path");
        if (sessionStats.weight_page_runtime_wait_cycles != 0)
            throw std::logic_error(
                "Qwen3 Down tile prefetch did not hide runtime page waits");
    }
    if (actual.size() != kSeqLen * kHidden * 2)
        throw std::logic_error("Qwen3 output binding has the wrong size");

    const auto normalized0 = rmsNorm(
        inputValues, kSeqLen, kHidden, false);
    std::vector<float> query(kSeqLen * kQueryWidth);
    std::vector<float> key(kSeqLen * kKvWidth);
    std::vector<float> value(kSeqLen * kKvWidth);
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t column = 0; column < kQueryWidth; ++column)
            query[token * kQueryWidth + column] = bf16(
                normalized0[token * kHidden + sourceHidden(0, column)]
                * projectionSign(0, column));
        for (std::size_t column = 0; column < kKvWidth; ++column) {
            key[token * kKvWidth + column] = bf16(
                normalized0[token * kHidden + sourceHidden(1, column)]
                * projectionSign(1, column));
            value[token * kKvWidth + column] = bf16(
                normalized0[token * kHidden + sourceHidden(2, column)]
                * projectionSign(2, column));
        }
    }
    headRmsNorm(query, kQueryHeads, false);
    headRmsNorm(key, kKvHeads, true);
    const auto rotatedQuery = rope(query, kQueryHeads);
    const auto rotatedKey = rope(key, kKvHeads);

    std::vector<float> context(kSeqLen * kQueryWidth);
    std::vector<float> scores(kSeqLen);
    std::vector<float> probabilities(kSeqLen);
    for (std::size_t queryToken = 0;
         queryToken < kSeqLen; ++queryToken) {
        for (std::size_t queryHead = 0;
             queryHead < kQueryHeads; ++queryHead) {
            const std::size_t kvHead = queryHead
                / (kQueryHeads / kKvHeads);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t keyToken = 0;
                 keyToken <= queryToken; ++keyToken) {
                float score = 0.0f;
                for (std::size_t dimension = 0;
                     dimension < kHeadDim; ++dimension)
                    score += rotatedQuery[
                        (queryToken * kQueryHeads + queryHead)
                            * kHeadDim + dimension]
                        * rotatedKey[(keyToken * kKvHeads + kvHead)
                            * kHeadDim + dimension];
                score /= std::sqrt(static_cast<float>(kHeadDim));
                scores[keyToken] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (std::size_t keyToken = 0;
                 keyToken <= queryToken; ++keyToken) {
                probabilities[keyToken] =
                    std::exp(scores[keyToken] - maximum);
                denominator += probabilities[keyToken];
            }
            for (std::size_t keyToken = 0;
                 keyToken <= queryToken; ++keyToken)
                probabilities[keyToken] = bf16(
                    probabilities[keyToken] / denominator);
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension) {
                float sum = 0.0f;
                for (std::size_t keyToken = 0;
                     keyToken <= queryToken; ++keyToken)
                    sum += probabilities[keyToken]
                        * value[(keyToken * kKvHeads + kvHead)
                            * kHeadDim + dimension];
                context[queryToken * kQueryWidth
                    + queryHead * kHeadDim + dimension] = bf16(sum);
            }
        }
    }

    std::vector<float> residual1(kSeqLen * kHidden);
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float projected = bf16(
                context[token * kQueryWidth + contextK(column)]
                * outputSign(column));
            residual1[token * kHidden + column] = bf16(
                inputValues[token * kHidden + column] + projected);
        }
    }
    const auto normalized1 = rmsNorm(
        residual1, kSeqLen, kHidden, true);

    std::vector<float> hidden(kIntermediate);
    float maxError = 0.0f;
    float maxExpectedMagnitude = 0.0f;
    std::size_t mismatchCount = 0;
    std::size_t maxIndex = 0;
    float maxActual = 0.0f;
    float maxExpected = 0.0f;
    constexpr float kTolerance = 0.125f;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t h = 0; h < kIntermediate; ++h) {
            const float gate = normalized1[row * kHidden + gateK(h)]
                * gateSign(h);
            const float up = normalized1[row * kHidden + upK(h)]
                * upSign(h);
            hidden[h] = bf16(
                gate * (1.0f / (1.0f + std::exp(-gate))) * up);
        }
        for (std::size_t column = 0; column < kHidden; ++column) {
            const std::size_t h0 = (column * 5 + 17) % kIntermediate;
            const std::size_t h1 = (h0 + 37) % kIntermediate;
            const float ffn = bf16(hidden[h0] - hidden[h1]);
            const float expected = bf16(
                residual1[row * kHidden + column] + ffn);
            const std::size_t index = row * kHidden + column;
            const float observed = readBf16(actual, index);
            const float error = std::fabs(observed - expected);
            maxExpectedMagnitude = std::max(
                maxExpectedMagnitude, std::fabs(expected));
            if (error > kTolerance) {
                if (mismatchCount < 16)
                    std::cerr << "Qwen3 mismatch row=" << row
                              << " column=" << column
                              << " actual=" << observed
                              << " expected=" << expected
                              << " error=" << error << '\n';
                ++mismatchCount;
            }
            if (error > maxError) {
                maxError = error;
                maxIndex = index;
                maxActual = observed;
                maxExpected = expected;
            }
        }
    }
    if (mismatchCount != 0)
        throw std::logic_error(
            "Qwen3 layer numerical mismatch: max_error="
            + std::to_string(maxError)
            + " row=" + std::to_string(maxIndex / kHidden)
            + " column=" + std::to_string(maxIndex % kHidden)
            + " actual=" + std::to_string(maxActual)
            + " expected=" + std::to_string(maxExpected)
            + " mismatches=" + std::to_string(mismatchCount));

    std::cout << "Complete Qwen3-0.6B seq32 prefill layer passed: "
              << kSeqLen * kHidden << " BF16 outputs, max_error="
              << maxError << ", tolerance=" << kTolerance
              << ", max_expected_magnitude=" << maxExpectedMagnitude
              << ", max_cycle=" << maxCycle
              << ", C2C_ingress_bytes=" << sessionStats.c2c_ingress_bytes
              << ", C2C_egress_bytes=" << sessionStats.c2c_egress_bytes
              << ", C2C_weight_pages="
              << sessionStats.weight_page_prefetches
              << ", C2C_page_wait_cycles="
              << sessionStats.weight_page_wait_cycles
              << ", C2C_runtime_page_wait_cycles="
              << sessionStats.weight_page_runtime_wait_cycles << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_qwen3_decoder_layer_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
