#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;
constexpr std::size_t kQueryHeads = 9;
constexpr std::size_t kKvHeads = 3;
constexpr std::size_t kHeadDim = 64;
constexpr float kEpsilon = 1.0e-5f;

float bf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

float inputValue(std::size_t row, std::size_t column)
{
    const int value =
        static_cast<int>((row * 13 + column * 7) % 31) - 15;
    return bf16(static_cast<float>(value) / 32.0f);
}

float gammaValue(std::size_t column, std::size_t stage)
{
    return bf16((stage == 0 ? 0.75f : 0.875f)
        + static_cast<float>((column + stage * 3) % 9) / 32.0f);
}

float ropeValue(const std::vector<float>& projection,
    std::size_t token, std::size_t head, std::size_t dimension)
{
    const std::size_t pair = dimension % 32;
    const std::size_t base = token * kHidden + head * kHeadDim;
    const float low = projection[base + pair];
    const float high = projection[base + pair + 32];
    constexpr float theta = 100000.0f;
    const float inverse = 1.0f / std::pow(
        theta, static_cast<float>(2 * pair) / kHeadDim);
    const float angle = static_cast<float>(token) * inverse;
    const float cosine = bf16(std::cos(angle));
    const float sine = bf16(std::sin(angle));
    return bf16(dimension < 32
        ? low * cosine - high * sine
        : high * cosine + low * sine);
}

void appendBf16(std::vector<std::uint8_t>& bytes, float value)
{
    const auto bits = ftlpu::Bf16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

float readBf16(
    const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8))
        .to_float();
}

std::size_t firstBindingReadCycle(
    const ftlpu::software::runtime::BinaryProgram& program,
    std::size_t bindingIndex)
{
    const auto binding = std::find_if(program.bindings.begin(),
        program.bindings.end(), [&](const auto& candidate) {
            return candidate.index == bindingIndex;
        });
    if (binding == program.bindings.end() || binding->slices.empty())
        throw std::logic_error("cannot locate FFN binding");

    const std::size_t queueIndex = binding->slices.front();
    for (const auto& queue : program.queues) {
        if (queue.kind != ftlpu::software::runtime::QueueKind::Mem
            || queue.index != queueIndex)
            continue;
        std::size_t cycle = 0;
        for (const auto& command : queue.commands) {
            const auto opcode =
                ftlpu::isa::decode_icu_command_opcode(command.command);
            if (opcode == ftlpu::isa::IcuCommandOpcode::Nop) {
                cycle += ftlpu::isa::decode_icu_nop_cycles(
                    command.command);
                continue;
            }
            if (opcode == ftlpu::isa::IcuCommandOpcode::Repeat) {
                const auto repeat =
                    ftlpu::isa::decode_icu_repeat(command.command);
                cycle += repeat.count * repeat.interval;
                continue;
            }
            if (opcode != ftlpu::isa::IcuCommandOpcode::Instruction
                || command.instruction_kind
                    != ftlpu::software::runtime::InstructionKind::Mem)
                continue;
            const auto encoded =
                static_cast<ftlpu::isa::EncodedMemInstruction>(
                    command.words[0])
                | (static_cast<ftlpu::isa::EncodedMemInstruction>(
                       command.words[1])
                    << 32);
            const auto instruction =
                ftlpu::isa::decode_mem_instruction(encoded);
            if (instruction.opcode == ftlpu::MemOpcode::Read
                && instruction.address
                    == static_cast<std::size_t>(binding->base_row))
                return cycle;
            ++cycle;
        }
    }
    throw std::logic_error("cannot locate first FFN binding read");
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

std::array<std::size_t, 2> writeSlicesAtAddress(
    const ftlpu::software::runtime::BinaryProgram& program,
    std::size_t address)
{
    std::vector<std::size_t> queues;
    for (const auto& queue : program.queues) {
        if (queue.kind != ftlpu::software::runtime::QueueKind::Mem)
            continue;
        const bool writesAddress = std::any_of(
            queue.commands.begin(), queue.commands.end(),
            [&](const auto& command) {
                if (command.instruction_kind
                        != ftlpu::software::runtime::InstructionKind::Mem
                    || command.word_count == 0)
                    return false;
                const auto encoded =
                    static_cast<ftlpu::isa::EncodedMemInstruction>(
                        command.words[0])
                    | (static_cast<ftlpu::isa::EncodedMemInstruction>(
                           command.words[1])
                        << 32);
                const auto instruction =
                    ftlpu::isa::decode_mem_instruction(encoded);
                return instruction.opcode == ftlpu::MemOpcode::Write
                    && instruction.address == address;
            });
        if (writesAddress) queues.push_back(queue.index);
    }
    std::sort(queues.begin(), queues.end());
    if (queues.size() != 4)
        throw std::logic_error(
            "cannot infer dual-hemisphere prepack slices");
    return {queues[0], queues[1]};
}

std::vector<float> rmsNorm(
    const std::vector<float>& input, std::size_t stage)
{
    std::vector<float> output(input.size());
    const float meanWeight = bf16(1.0f / kHidden);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        float meanSquare = 0.0f;
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float value = input[row * kHidden + column];
            meanSquare += bf16(value * value) * meanWeight;
        }
        const float factor =
            bf16(1.0f / std::sqrt(meanSquare + kEpsilon));
        for (std::size_t column = 0; column < kHidden; ++column)
            output[row * kHidden + column] = bf16(
                input[row * kHidden + column] * factor
                * gammaValue(column, stage));
    }
    return output;
}

std::size_t gateK(std::size_t h) { return (h * 7 + 1) % kHidden; }
std::size_t upK(std::size_t h) { return (h * 11 + 3) % kHidden; }
std::int8_t gateSign(std::size_t h) { return (h & 1) ? -1 : 1; }
std::int8_t upSign(std::size_t h) { return (h & 2) ? -1 : 1; }

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: compiled_smollm2_decoder_layer_runtime_test program.ftlpu");
    const auto program =
        ftlpu::software::runtime::read_binary_program(
            std::filesystem::path(argv[1]));
    if (program.target_abi
            != ftlpu::software::runtime::kLpu32StreamTargetAbi
        || program.max_cycle < 120000)
        throw std::logic_error(
            "decoder layer binary has the wrong target or schedule: max_cycle="
            + std::to_string(program.max_cycle));
    if (const auto* tracePath = std::getenv("FTLPU_SCHEDULE_TRACE"))
        ftlpu::software::runtime::write_schedule_trace_csv(
            program, tracePath);

    std::vector<float> inputValues(kSeqLen * kHidden);
    std::vector<std::uint8_t> input;
    input.reserve(inputValues.size() * 2);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float value = inputValue(row, column);
            inputValues[row * kHidden + column] = value;
            appendBf16(input, value);
        }
    }
    std::vector<std::uint8_t> gamma0;
    std::vector<std::uint8_t> gamma1;
    for (std::size_t column = 0; column < kHidden; ++column) {
        appendBf16(gamma0, gammaValue(column, 0));
        appendBf16(gamma1, gammaValue(column, 1));
    }

    std::vector<std::uint8_t> queryWeight(kHidden * kHidden, 0);
    std::vector<std::uint8_t> keyWeight(
        kHidden * kKvHeads * kHeadDim, 0);
    std::vector<std::uint8_t> valueWeight(
        kHidden * kKvHeads * kHeadDim, 0);
    const auto sourceHidden = [](std::size_t projection,
                                 std::size_t column) {
        return (column * 7 + projection * 13 + 3) % kHidden;
    };
    const auto projectionSign = [](std::size_t projection,
                                   std::size_t column) {
        return ((column + projection) & 1) ? -1 : 1;
    };
    for (std::size_t column = 0; column < kHidden; ++column)
        queryWeight[sourceHidden(0, column) * kHidden + column] =
            static_cast<std::uint8_t>(projectionSign(0, column));
    std::array<std::size_t, kKvHeads> valueMismatchCounts {};
    std::array<float, kKvHeads> valueMaxErrors {};
    std::array<std::vector<std::size_t>, kKvHeads>
        valueMismatchColumns;
    std::array<std::uint64_t, kKvHeads> valueMismatchMasks {};
    for (std::size_t column = 0;
         column < kKvHeads * kHeadDim; ++column) {
        keyWeight[sourceHidden(1, column)
                * (kKvHeads * kHeadDim) + column] =
            static_cast<std::uint8_t>(projectionSign(1, column));
        valueWeight[sourceHidden(2, column)
                * (kKvHeads * kHeadDim) + column] =
            static_cast<std::uint8_t>(projectionSign(2, column));
    }
    std::vector<std::uint8_t> outputWeight(kHidden * kHidden, 0);
    for (std::size_t column = 0; column < kHidden; ++column) {
        for (std::size_t block = 0; block < kHidden / 32; ++block) {
            const std::size_t hidden =
                block * 32 + (column * 7 + block * 3) % 32;
            outputWeight[hidden * kHidden + column] =
                static_cast<std::uint8_t>(
                    ((column + block) & 1) ? -1 : 1);
        }
    }

    std::vector<std::uint8_t> gateWeight(
        kHidden * kIntermediate, 0);
    std::vector<std::uint8_t> upWeight(
        kHidden * kIntermediate, 0);
    for (std::size_t h = 0; h < kIntermediate; ++h) {
        gateWeight[gateK(h) * kIntermediate + h] =
            static_cast<std::uint8_t>(gateSign(h));
        upWeight[upK(h) * kIntermediate + h] =
            static_cast<std::uint8_t>(upSign(h));
    }
    std::vector<std::uint8_t> downWeight(
        kIntermediate * kHidden, 0);
    for (std::size_t column = 0; column < kHidden; ++column) {
        const std::size_t h0 = (column * 5 + 17) % kIntermediate;
        const std::size_t h1 = (h0 + 37) % kIntermediate;
        downWeight[h0 * kHidden + column] = 1;
        downWeight[h1 * kHidden + column] =
            static_cast<std::uint8_t>(-1);
    }

    auto* system = new ftlpu::TspSliceSystem();
    ftlpu::software::runtime::CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, input);
    runtime.upload_input(1, gamma0);
    runtime.upload_input(2, queryWeight);
    runtime.upload_input(3, keyWeight);
    runtime.upload_input(4, valueWeight);
    runtime.upload_input(5, outputWeight);
    runtime.upload_input(6, gamma1);
    runtime.upload_input(7, gateWeight);
    runtime.upload_input(8, upWeight);
    runtime.upload_input(9, downWeight);
    const auto physicalBf16 = [&](ftlpu::Hemisphere hemisphere,
                                  std::size_t lowSlice,
                                  std::size_t highSlice,
                                  std::size_t address,
                                  std::size_t column) {
        const auto low = system->read_mem_sram_lane_byte(
            hemisphere, lowSlice, column / 8, address, column % 8);
        const auto high = system->read_mem_sram_lane_byte(
            hemisphere, highSlice, column / 8, address, column % 8);
        return ftlpu::Bf16::from_bits(
            static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8))
            .to_float();
    };
    const auto inputBinding = std::find_if(program.bindings.begin(),
        program.bindings.end(), [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Input
                && binding.index == 0;
        });
    if (inputBinding == program.bindings.end()
        || inputBinding->layout
            != ftlpu::software::runtime::BindingLayout::Fp16MxmDistributed16
        || inputBinding->slices.size() != 16)
        throw std::logic_error(
            "decoder input must use fp16_mxm_distributed_16");
    const auto residualBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [&](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.layout
                    == ftlpu::software::runtime::BindingLayout::
                        Fp16MxmDistributed16
                && binding.byte_size == inputBinding->byte_size
                && binding.base_row == inputBinding->base_row
                && binding.slices != inputBinding->slices;
        });
    if (residualBinding == program.bindings.end()
        || residualBinding->slices.size() != 16)
        throw std::logic_error(
            "cannot locate attention residual physical binding");
    const auto& residualSlices = residualBinding->slices;
    const auto readMxmDistributed = [&](const std::vector<std::uint16_t>& slices,
                                        std::size_t baseRow,
                                        std::size_t row,
                                        std::size_t column) {
        const std::size_t hiddenBlocks = kHidden / 32;
        const std::size_t tokenBlock = row / 32;
        const std::size_t tokenWave = (row % 32) / 8;
        const std::size_t tokenLane = row % 8;
        const std::size_t hiddenBlock = column / 32;
        const std::size_t featureWave = (column % 32) / 8;
        const std::size_t featureLane = column % 8;
        const std::size_t address = baseRow
            + (tokenBlock * hiddenBlocks + hiddenBlock) * 4 + tokenWave;
        return physicalBf16(ftlpu::Hemisphere::East,
            slices[2 * tokenLane], slices[2 * tokenLane + 1],
            address, featureWave * 8 + featureLane);
    };
    const auto normalized0 = rmsNorm(inputValues, 0);
    const std::size_t attentionPrepackEndCycle =
        firstBindingReadCycle(program, 2);
    runtime.run_cycles(attentionPrepackEndCycle);
    const auto prepackSlices = writeSlicesAtAddress(program, 63232);
    float prepackMaxError = 0.0f;
    std::size_t prepackMaxRow = 0;
    std::size_t prepackMaxColumn = 0;
    std::size_t prepackMaxHemisphere = 0;
    float prepackMaxActual = 0.0f;
    float prepackMaxExpected = 0.0f;
    std::size_t prepackMismatchCount = 0;
    std::size_t prepackLargeMismatchCount = 0;
    std::size_t prepackUnexpectedZeroCount = 0;
    std::size_t prepackFirstZeroRow = 0;
    std::size_t prepackFirstZeroColumn = 0;
    std::size_t prepackLastZeroRow = 0;
    std::size_t prepackLastZeroColumn = 0;
    for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
      for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const std::size_t address = 63232
                + (column / 32) * kSeqLen + row;
            const float observed = physicalBf16(
                static_cast<ftlpu::Hemisphere>(hemisphere),
                prepackSlices[0], prepackSlices[1],
                address, column % 32);
            const float expected = normalized0[row * kHidden + column];
            const float error = std::fabs(observed - expected);
            if (error > 0.0f) {
                ++prepackMismatchCount;
                if (error > 0.04f)
                    ++prepackLargeMismatchCount;
                if (observed == 0.0f && expected != 0.0f) {
                    if (prepackUnexpectedZeroCount == 0) {
                        prepackFirstZeroRow = row;
                        prepackFirstZeroColumn = column;
                    }
                    ++prepackUnexpectedZeroCount;
                    prepackLastZeroRow = row;
                    prepackLastZeroColumn = column;
                }
            }
            if (error > prepackMaxError) {
                prepackMaxError = error;
                prepackMaxRow = row;
                prepackMaxColumn = column;
                prepackMaxHemisphere = hemisphere;
                prepackMaxActual = observed;
                prepackMaxExpected = expected;
        }
      }
    }
    }
    if (prepackMaxError > 0.04f)
        throw std::logic_error(
            "RMS1-to-attention prepack mismatch max_error="
            + std::to_string(prepackMaxError)
            + " row=" + std::to_string(prepackMaxRow)
            + " column=" + std::to_string(prepackMaxColumn)
            + " hemisphere=" + std::to_string(prepackMaxHemisphere)
            + " actual=" + std::to_string(prepackMaxActual)
            + " expected=" + std::to_string(prepackMaxExpected)
            + " mismatches=" + std::to_string(prepackMismatchCount)
            + " large_mismatches="
                + std::to_string(prepackLargeMismatchCount)
            + " unexpected_zeros="
                + std::to_string(prepackUnexpectedZeroCount)
            + " first_zero=(" + std::to_string(prepackFirstZeroRow)
                + "," + std::to_string(prepackFirstZeroColumn) + ")"
            + " last_zero=(" + std::to_string(prepackLastZeroRow)
                + "," + std::to_string(prepackLastZeroColumn) + ")"
            + " source=" + std::to_string(readMxmDistributed(
                inputBinding->slices,
                static_cast<std::size_t>(inputBinding->base_row) + 1536,
                prepackMaxRow, prepackMaxColumn)));
    const std::size_t qkvEndCycle =
        static_cast<std::size_t>(timeline(program, "qk").start_cycle);
    runtime.run_cycles(qkvEndCycle - attentionPrepackEndCycle);
    constexpr std::array<std::array<std::size_t, 16>, 2>
        kValuePackSlices {{
            {{4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 32, 33}},
            {{18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 34, 35}},
        }};
    for (std::size_t column = 0;
         column < kKvHeads * kHeadDim; ++column) {
        const std::size_t head = column / kHeadDim;
        const std::size_t dimension = column % kHeadDim;
        const std::size_t reduction = dimension / 32;
        const std::size_t stream = 0;
        const std::size_t address =
            7800 + (head * 2 + reduction) * 16;
        const float observed = physicalBf16(
            static_cast<ftlpu::Hemisphere>(head % 2),
            kValuePackSlices[reduction][stream],
            kValuePackSlices[reduction][stream + 1],
            address, dimension % 32);
        const float expected = bf16(
            normalized0[sourceHidden(2, column)]
            * projectionSign(2, column));
        const float error = std::fabs(observed - expected);
        valueMaxErrors[head] = std::max(valueMaxErrors[head], error);
        if (error > 0.04f) {
            ++valueMismatchCounts[head];
            valueMismatchColumns[head].push_back(column % kHeadDim);
            valueMismatchMasks[head] |=
                std::uint64_t {1} << (column % kHeadDim);
        }
    }
    if (std::ranges::any_of(
            valueMismatchCounts, [](std::size_t count) {
                return count != 0;
            }))
        throw std::logic_error(
            "RMS1-to-value projection mismatch counts=["
            + std::to_string(valueMismatchCounts[0]) + ","
            + std::to_string(valueMismatchCounts[1]) + ","
            + std::to_string(valueMismatchCounts[2]) + "] max_errors=["
            + std::to_string(valueMaxErrors[0]) + ","
            + std::to_string(valueMaxErrors[1]) + ","
            + std::to_string(valueMaxErrors[2]) + "] first_dimensions=["
            + std::to_string(valueMismatchColumns[1].empty()
                    ? 999 : valueMismatchColumns[1].front())
            + ","
            + std::to_string(valueMismatchColumns[2].empty()
                    ? 999 : valueMismatchColumns[2].front())
            + "] masks=["
            + std::to_string(valueMismatchMasks[0]) + ","
            + std::to_string(valueMismatchMasks[1]) + ","
            + std::to_string(valueMismatchMasks[2]) + "]");

    std::vector<float> checkpointQuery(inputValues.size(), 0.0f);
    std::vector<float> checkpointKey(inputValues.size(), 0.0f);
    std::vector<float> checkpointValue(inputValues.size(), 0.0f);
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t column = 0; column < kHidden; ++column)
            checkpointQuery[token * kHidden + column] = bf16(
                normalized0[token * kHidden + sourceHidden(0, column)]
                * projectionSign(0, column));
        for (std::size_t column = 0;
             column < kKvHeads * kHeadDim; ++column) {
            checkpointKey[token * kHidden + column] = bf16(
                normalized0[token * kHidden + sourceHidden(1, column)]
                * projectionSign(1, column));
            checkpointValue[token * kHidden + column] = bf16(
                normalized0[token * kHidden + sourceHidden(2, column)]
                * projectionSign(2, column));
        }
    }
    std::vector<float> checkpointContext(inputValues.size(), 0.0f);
    std::vector<float> checkpointScores(kSeqLen);
    std::vector<float> checkpointProbabilities(kSeqLen);
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t queryHead = 0;
             queryHead < kQueryHeads; ++queryHead) {
            const std::size_t kvHead =
                queryHead / (kQueryHeads / kKvHeads);
            float maximum =
                -std::numeric_limits<float>::infinity();
            for (std::size_t key = 0; key <= query; ++key) {
                float score = 0.0f;
                for (std::size_t dimension = 0;
                     dimension < kHeadDim; ++dimension)
                    score += ropeValue(checkpointQuery, query,
                                 queryHead, dimension)
                        * ropeValue(checkpointKey, key,
                            kvHead, dimension);
                score /= std::sqrt(static_cast<float>(kHeadDim));
                checkpointScores[key] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (std::size_t key = 0; key <= query; ++key) {
                checkpointProbabilities[key] =
                    std::exp(checkpointScores[key] - maximum);
                denominator += checkpointProbabilities[key];
            }
            for (std::size_t key = 0; key <= query; ++key)
                checkpointProbabilities[key] = bf16(
                    checkpointProbabilities[key] / denominator);
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension) {
                float context = 0.0f;
                for (std::size_t key = 0; key <= query; ++key)
                    context += checkpointProbabilities[key]
                        * checkpointValue[key * kHidden
                            + kvHead * kHeadDim + dimension];
                checkpointContext[query * kHidden
                    + queryHead * kHeadDim + dimension] =
                    bf16(context);
            }
        }
    }
    const std::size_t pvEndCycle =
        static_cast<std::size_t>(timeline(program, "o_proj").start_cycle);
    runtime.run_cycles(pvEndCycle - qkvEndCycle);
    float contextMaxError = 0.0f;
    std::size_t contextMaxHemisphere = 0;
    std::size_t contextMaxRow = 0;
    std::size_t contextMaxColumn = 0;
    float contextMaxActual = 0.0f;
    float contextMaxExpected = 0.0f;
    for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        for (std::size_t row = 0; row < kSeqLen; ++row) {
            for (std::size_t column = 0; column < kHidden; ++column) {
                const std::size_t head = column / kHeadDim;
                const std::size_t dimension = column % kHeadDim;
                const std::size_t headBlock = dimension / 32;
                const float observed = physicalBf16(
                    static_cast<ftlpu::Hemisphere>(hemisphere),
                    20 + headBlock * 2, 21 + headBlock * 2,
                    2000 + head * kSeqLen + row,
                    dimension % 32);
                const float expected =
                    checkpointContext[row * kHidden + column];
                const float error = std::fabs(observed - expected);
                if (error > contextMaxError) {
                    contextMaxError = error;
                    contextMaxHemisphere = hemisphere;
                    contextMaxRow = row;
                    contextMaxColumn = column;
                    contextMaxActual = observed;
                    contextMaxExpected = expected;
                }
            }
        }
    }
    if (contextMaxError > 0.04f)
        throw std::logic_error(
            "PV context mismatch max_error="
            + std::to_string(contextMaxError)
            + " hemisphere="
            + std::to_string(contextMaxHemisphere)
            + " row=" + std::to_string(contextMaxRow)
            + " column=" + std::to_string(contextMaxColumn)
            + " actual=" + std::to_string(contextMaxActual)
            + " expected=" + std::to_string(contextMaxExpected));

    const std::size_t outputProjectionEndCycle =
        static_cast<std::size_t>(
            timeline(program, "elementwise.add").start_cycle);
    runtime.run_cycles(outputProjectionEndCycle - pvEndCycle);
    float outputProjectionMaxError = 0.0f;
    std::size_t outputProjectionMaxRow = 0;
    std::size_t outputProjectionMaxColumn = 0;
    float outputProjectionMaxActual = 0.0f;
    float outputProjectionMaxExpected = 0.0f;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            float expected = 0.0f;
            for (std::size_t reduction = 0;
                 reduction < kHidden / 32; ++reduction) {
                const std::size_t hidden = reduction * 32
                    + (column * 7 + reduction * 3) % 32;
                expected += checkpointContext[row * kHidden + hidden]
                    * (((column + reduction) & 1) ? -1.0f : 1.0f);
            }
            expected = bf16(expected);
            const std::size_t pair = (column % 64) / 32;
            const float observed = physicalBf16(
                ftlpu::Hemisphere::East, 28 + pair * 2,
                29 + pair * 2, (column / 64) * kSeqLen + row,
                column % 32);
            const float error = std::fabs(observed - expected);
            if (error > outputProjectionMaxError) {
                outputProjectionMaxError = error;
                outputProjectionMaxRow = row;
                outputProjectionMaxColumn = column;
                outputProjectionMaxActual = observed;
                outputProjectionMaxExpected = expected;
            }
        }
    }
    if (outputProjectionMaxError > 0.04f)
        throw std::logic_error(
            "O projection checkpoint mismatch max_error="
            + std::to_string(outputProjectionMaxError)
            + " row=" + std::to_string(outputProjectionMaxRow)
            + " column=" + std::to_string(outputProjectionMaxColumn)
            + " actual=" + std::to_string(outputProjectionMaxActual)
            + " expected=" + std::to_string(outputProjectionMaxExpected));

    const std::size_t attentionResidualEndCycle =
        static_cast<std::size_t>(
            timeline(program, "rmsnorm.feedback", 1).start_cycle);
    runtime.run_cycles(
        attentionResidualEndCycle - outputProjectionEndCycle);
    float residualCheckpointMaxError = 0.0f;
    std::size_t residualCheckpointMaxRow = 0;
    std::size_t residualCheckpointMaxColumn = 0;
    float residualCheckpointMaxActual = 0.0f;
    float residualCheckpointMaxExpected = 0.0f;
    std::vector<float> residual1(inputValues.size());
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            float projection = 0.0f;
            for (std::size_t reduction = 0;
                 reduction < kHidden / 32; ++reduction) {
                const std::size_t hidden = reduction * 32
                    + (column * 7 + reduction * 3) % 32;
                projection += checkpointContext[row * kHidden + hidden]
                    * (((column + reduction) & 1) ? -1.0f : 1.0f);
            }
            const float expected =
                bf16(inputValues[row * kHidden + column]
                    + bf16(projection));
            const float observed = readMxmDistributed(
                residualSlices,
                static_cast<std::size_t>(residualBinding->base_row),
                row, column);
            residual1[row * kHidden + column] = observed;
            const float error = std::fabs(observed - expected);
            if (error > residualCheckpointMaxError) {
                residualCheckpointMaxError = error;
                residualCheckpointMaxRow = row;
                residualCheckpointMaxColumn = column;
                residualCheckpointMaxActual = observed;
                residualCheckpointMaxExpected = expected;
            }
        }
    }
    if (residualCheckpointMaxError > 0.04f)
        throw std::logic_error(
            "attention residual checkpoint mismatch max_error="
            + std::to_string(residualCheckpointMaxError)
            + " row=" + std::to_string(residualCheckpointMaxRow)
            + " column=" + std::to_string(residualCheckpointMaxColumn)
            + " actual=" + std::to_string(residualCheckpointMaxActual)
            + " expected="
            + std::to_string(residualCheckpointMaxExpected));

    const std::size_t ffnStartCycle =
        firstBindingReadCycle(program, 7);
    runtime.run_cycles(ffnStartCycle - attentionResidualEndCycle);
    const std::size_t rms2BaseRow =
        static_cast<std::size_t>(inputBinding->base_row) + 1536;
    const float rms2Value = readMxmDistributed(
        inputBinding->slices, rms2BaseRow, 0, 0);
    if (!std::isfinite(rms2Value) || rms2Value == 0.0f)
        throw std::logic_error(
            "RMS2 stage produced invalid data value="
            + std::to_string(rms2Value)
            + " input=" + std::to_string(readMxmDistributed(
                residualSlices,
                static_cast<std::size_t>(residualBinding->base_row), 0, 0)));
    std::vector<float> rms2Output(kSeqLen * kHidden);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            rms2Output[row * kHidden + column] = readMxmDistributed(
                inputBinding->slices, rms2BaseRow, row, column);
        }
    }
    const std::size_t finalResidualStartCycle =
        static_cast<std::size_t>(
            timeline(program, "elementwise.add", 1).start_cycle);
    runtime.run_cycles(finalResidualStartCycle - ffnStartCycle);
    const float preFinalResidualSample = readMxmDistributed(
        residualSlices,
        static_cast<std::size_t>(residualBinding->base_row), 1, 0);
    const float expectedResidualSample = residual1[kHidden];
    if (preFinalResidualSample != expectedResidualSample)
        throw std::logic_error(
            "FFN overwrote live attention residual before final add"
            " actual=" + std::to_string(preFinalResidualSample)
            + " expected=" + std::to_string(expectedResidualSample));
    runtime.run_cycles(
        program.max_cycle + 64 - finalResidualStartCycle);
    const auto actual = runtime.download_output(0);

    std::vector<float> queryProjection(inputValues.size(), 0.0f);
    std::vector<float> keyProjection(inputValues.size(), 0.0f);
    std::vector<float> valueProjection(inputValues.size(), 0.0f);
    for (std::size_t token = 0; token < kSeqLen; ++token) {
        for (std::size_t column = 0; column < kHidden; ++column)
            queryProjection[token * kHidden + column] = bf16(
                normalized0[token * kHidden + sourceHidden(0, column)]
                * projectionSign(0, column));
        for (std::size_t column = 0;
             column < kKvHeads * kHeadDim; ++column) {
            keyProjection[token * kHidden + column] = bf16(
                normalized0[token * kHidden + sourceHidden(1, column)]
                * projectionSign(1, column));
            valueProjection[token * kHidden + column] = bf16(
                normalized0[token * kHidden + sourceHidden(2, column)]
                * projectionSign(2, column));
        }
    }
    std::vector<float> referenceResidual1(inputValues.size());
    std::vector<float> attentionContext(inputValues.size());
    std::vector<float> scores(kSeqLen);
    std::vector<float> probabilities(kSeqLen);
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        for (std::size_t queryHead = 0;
             queryHead < kQueryHeads; ++queryHead) {
            const std::size_t kvHead =
                queryHead / (kQueryHeads / kKvHeads);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t key = 0; key <= query; ++key) {
                float score = 0.0f;
                for (std::size_t dimension = 0;
                     dimension < kHeadDim; ++dimension)
                    score += ropeValue(queryProjection, query,
                                 queryHead, dimension)
                        * ropeValue(keyProjection, key,
                            kvHead, dimension);
                score /= std::sqrt(static_cast<float>(kHeadDim));
                scores[key] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (std::size_t key = 0; key <= query; ++key) {
                probabilities[key] =
                    std::exp(scores[key] - maximum);
                denominator += probabilities[key];
            }
            for (std::size_t key = 0; key <= query; ++key)
                probabilities[key] =
                    bf16(probabilities[key] / denominator);
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension) {
                float context = 0.0f;
                for (std::size_t key = 0; key <= query; ++key)
                    context += probabilities[key]
                        * valueProjection[key * kHidden
                            + kvHead * kHeadDim + dimension];
                const std::size_t column =
                    queryHead * kHeadDim + dimension;
                attentionContext[query * kHidden + column] =
                    bf16(context);
            }
        }
        for (std::size_t column = 0; column < kHidden; ++column) {
            float projected = 0.0f;
            for (std::size_t block = 0;
                 block < kHidden / 32; ++block) {
                const std::size_t hidden =
                    block * 32 + (column * 7 + block * 3) % 32;
                projected += attentionContext[
                    query * kHidden + hidden]
                    * (((column + block) & 1) ? -1.0f : 1.0f);
            }
            referenceResidual1[query * kHidden + column] = bf16(
                inputValues[query * kHidden + column]
                + bf16(projected));
        }
    }
    float attentionResidualMaxError = 0.0f;
    std::size_t residualMaxRow = 0;
    std::size_t residualMaxColumn = 0;
    float residualMaxActual = 0.0f;
    float residualMaxExpected = 0.0f;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float observed =
                residual1[row * kHidden + column];
            const float expected =
                referenceResidual1[row * kHidden + column];
            const float error = std::fabs(observed - expected);
            if (error > attentionResidualMaxError) {
                attentionResidualMaxError = error;
                residualMaxRow = row;
                residualMaxColumn = column;
                residualMaxActual = observed;
                residualMaxExpected = expected;
            }
        }
    }
    if (attentionResidualMaxError > 0.04f) {
        const std::size_t outputBlock = residualMaxColumn / 32;
        const std::size_t outputPair = outputBlock % 2;
        const std::size_t outputRow =
            (outputBlock / 2) * kSeqLen + residualMaxRow;
        const float observedProjection = physicalBf16(
            ftlpu::Hemisphere::East,
            28 + outputPair * 2, 29 + outputPair * 2,
            outputRow, residualMaxColumn % 32);
        const float observedInput = readMxmDistributed(
            inputBinding->slices,
            static_cast<std::size_t>(inputBinding->base_row),
            residualMaxRow, residualMaxColumn);
        const std::size_t finalReduction = kHidden / 32 - 1;
        const std::size_t finalHidden = finalReduction * 32
            + (residualMaxColumn * 7 + finalReduction * 3) % 32;
        const float finalContribution = attentionContext[
            residualMaxRow * kHidden + finalHidden]
            * (((residualMaxColumn + finalReduction) & 1)
                    ? -1.0f : 1.0f);
        std::size_t closestReduction = 0;
        float closestContribution = 0.0f;
        float closestError =
            std::numeric_limits<float>::infinity();
        for (std::size_t reduction = 0;
             reduction < kHidden / 32; ++reduction) {
            const std::size_t hidden = reduction * 32
                + (residualMaxColumn * 7 + reduction * 3) % 32;
            const float contribution = attentionContext[
                residualMaxRow * kHidden + hidden]
                * (((residualMaxColumn + reduction) & 1)
                        ? -1.0f : 1.0f);
            const float error =
                std::fabs(contribution - observedProjection);
            if (error < closestError) {
                closestError = error;
                closestReduction = reduction;
                closestContribution = contribution;
            }
        }
        throw std::logic_error(
            "attention residual mismatch max_error="
            + std::to_string(attentionResidualMaxError)
            + " row=" + std::to_string(residualMaxRow)
            + " column=" + std::to_string(residualMaxColumn)
            + " actual=" + std::to_string(residualMaxActual)
            + " expected=" + std::to_string(residualMaxExpected)
            + " input=" + std::to_string(observedInput)
            + " projection=" + std::to_string(observedProjection)
            + " expected_projection="
                + std::to_string(
                    residualMaxExpected
                    - inputValues[
                        residualMaxRow * kHidden
                        + residualMaxColumn])
            + " final_reduction_contribution="
                + std::to_string(finalContribution)
            + " closest_reduction="
                + std::to_string(closestReduction)
            + " closest_contribution="
                + std::to_string(closestContribution));
    }
    auto normalized1 = rmsNorm(residual1, 1);
    float rms2MaxError = 0.0f;
    for (std::size_t index = 0; index < normalized1.size(); ++index) {
        rms2MaxError = std::max(
            rms2MaxError,
            std::fabs(normalized1[index] - rms2Output[index]));
        normalized1[index] = rms2Output[index];
    }
    if (rms2MaxError > 0.04f) {
        throw std::logic_error(
            "second RMSNorm mismatch max_error="
            + std::to_string(rms2MaxError));
    }
    std::vector<float> hidden(kIntermediate);
    std::size_t nonzero = 0;
    float maxError = 0.0f;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t h = 0; h < kIntermediate; ++h) {
            const float gate =
                normalized1[row * kHidden + gateK(h)] * gateSign(h);
            const float up =
                normalized1[row * kHidden + upK(h)] * upSign(h);
            hidden[h] = bf16(
                gate * (1.0f / (1.0f + std::exp(-gate))) * up);
        }
        for (std::size_t column = 0; column < kHidden; ++column) {
            const std::size_t h0 = (column * 5 + 17) % kIntermediate;
            const std::size_t h1 = (h0 + 37) % kIntermediate;
            const float ffn = bf16(hidden[h0] - hidden[h1]);
            const float expected = bf16(
                residual1[row * kHidden + column] + ffn);
            const float observed =
                readBf16(actual, row * kHidden + column);
            const float error = std::fabs(observed - expected);
            maxError = std::max(maxError, error);
            if (std::fabs(observed) > 1.0e-4f) ++nonzero;
            if (error > 0.04f) {
                throw std::logic_error(
                    "decoder layer mismatch row=" + std::to_string(row)
                    + " column=" + std::to_string(column)
                    + " actual=" + std::to_string(observed)
                    + " expected=" + std::to_string(expected)
                    + " cpu_ffn=" + std::to_string(ffn)
                    + " hidden0.cpu=" + std::to_string(hidden[h0])
                    + " hidden1.cpu=" + std::to_string(hidden[h1])
                    + " residual=" + std::to_string(
                        residual1[row * kHidden + column]));
            }
        }
    }
    if (nonzero == 0) {
        throw std::logic_error(
            "decoder layer unexpectedly produced only zero output"
            " residual1.east="
            + std::to_string(physicalBf16(
                ftlpu::Hemisphere::East, 16, 17, 0, 0))
            + " ffn.east=" + std::to_string(physicalBf16(
                ftlpu::Hemisphere::East, 24, 25, 0, 0))
            + " rms2.factor=" + std::to_string(physicalBf16(
                ftlpu::Hemisphere::East, 10, 11, 0, 0))
            + " final.east=" + std::to_string(physicalBf16(
                ftlpu::Hemisphere::East, 32, 33, 0, 0))
            + " final.west=" + std::to_string(physicalBf16(
                ftlpu::Hemisphere::West, 32, 33, 0, 0)));
    }
    std::cout << "Complete SmolLM2 decoder layer passed: "
              << kSeqLen * kHidden << " BF16 values, nonzero="
              << nonzero << ", max_error=" << maxError
              << ", attention_residual_max_error="
              << attentionResidualMaxError
              << ", rms2_max_error=" << rms2MaxError
              << ", max_cycle=" << program.max_cycle << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_smollm2_decoder_layer_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
