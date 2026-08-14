#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/instruction_codec.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kM = 32;
constexpr std::size_t kN = 64;
constexpr std::size_t kK = 64;
constexpr float kScale = 0.0625f;

void appendBf16(std::vector<std::uint8_t>& bytes, float value)
{
    const std::uint16_t bits = ftlpu::Bf16::from_float(value).bits();
    bytes.push_back(static_cast<std::uint8_t>(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
}

std::uint16_t readBf16(
    const std::vector<std::uint8_t>& bytes, std::size_t index)
{
    const std::size_t offset = index * 2;
    return static_cast<std::uint16_t>(bytes[offset])
        | (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu;
    using namespace ftlpu::software::runtime;
    if (argc != 2)
        throw std::runtime_error(
            "usage: block8_linear_projection_runtime_test program.ftlpu");

    const BinaryProgram program =
        read_binary_program(std::filesystem::path(argv[1]));
    bool sawLocalDequant = false;
    bool sawBlock8Compute = false;
    bool sawDirectBlock8Result = false;
    for (const QueueProgram& queue : program.queues) {
        for (const QueueCommand& command : queue.commands) {
            if (command.instruction_kind
                == InstructionKind::MxmDequant)
                sawLocalDequant = true;
            if (command.instruction_kind != InstructionKind::Mxm)
                continue;
            const auto encoded =
                static_cast<isa::EncodedMxmInstruction>(
                    command.words[0])
                | (static_cast<isa::EncodedMxmInstruction>(
                       command.words[1])
                    << 32);
            const auto instruction =
                isa::decode_mxm_instruction(encoded);
            sawBlock8Compute |=
                instruction.opcode == MxmControlOpcode::Compute
                && instruction.compute_mode
                    == MxmComputeMode::Block8;
            sawDirectBlock8Result |=
                instruction.opcode == MxmControlOpcode::Compute
                && instruction.compute_mode
                    == MxmComputeMode::Block8
                && instruction.accumulator_destination
                    == MxmAccumulatorDestination::Stream
                && instruction.accumulator_clear
                && instruction.accumulator_output_format
                    == MxmAccumulatorOutputFormat::BFloat16;
        }
    }
    if (!sawLocalDequant || !sawBlock8Compute
        || !sawDirectBlock8Result)
        throw std::logic_error(
            "binary did not preserve the selected MXM strategy");

    std::vector<float> activation(kM * kK);
    std::vector<std::uint8_t> activationBytes;
    activationBytes.reserve(kM * kK * 2);
    for (std::size_t row = 0; row < kM; ++row) {
        for (std::size_t k = 0; k < kK; ++k) {
            const float value = static_cast<float>(
                static_cast<int>((row * 3 + k * 5) % 17) - 8)
                * 0.125f;
            activation[row * kK + k] =
                Bf16::from_float(value).to_float();
            appendBf16(activationBytes, value);
        }
    }

    std::vector<std::int8_t> weights(kK * kN);
    std::vector<std::uint8_t> weightBytes(kK * kN);
    for (std::size_t k = 0; k < kK; ++k) {
        for (std::size_t column = 0; column < kN; ++column) {
            const auto value = static_cast<std::int8_t>(
                static_cast<int>((k * 7 + column * 3) % 15) - 7);
            weights[k * kN + column] = value;
            weightBytes[k * kN + column] =
                static_cast<std::uint8_t>(value);
        }
    }

    TspSliceSystem system;
    CModelRuntime runtime(system);
    runtime.load(program);
    runtime.upload_input(0, activationBytes);
    runtime.upload_input(1, weightBytes);
    runtime.run_cycles(program.max_cycle + 64);
    const auto output = runtime.download_output(0);

    std::size_t mismatches = 0;
    float maxError = 0.0f;
    std::size_t nonzero = 0;
    for (std::size_t row = 0; row < kM; ++row) {
        for (std::size_t column = 0; column < kN; ++column) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < kK; ++k) {
                const float weight = Bf16::from_float(
                    static_cast<float>(
                        weights[k * kN + column])
                    * kScale).to_float();
                sum += activation[row * kK + k] * weight;
            }
            const std::uint16_t expected =
                Bf16::from_float(sum).bits();
            const std::uint16_t observed =
                readBf16(output, row * kN + column);
            const float error = std::fabs(
                Bf16::from_bits(observed).to_float()
                - Bf16::from_bits(expected).to_float());
            maxError = std::max(maxError, error);
            if (observed != 0) ++nonzero;
            if (observed != expected) ++mismatches;
        }
    }
    if (nonzero == 0 || mismatches != 0)
        throw std::logic_error(
            "Block8 CModel numeric mismatch count="
            + std::to_string(mismatches)
            + " max_error=" + std::to_string(maxError)
            + " nonzero=" + std::to_string(nonzero));

    std::cout
        << "Block8 linear projection passed: 32x64x64, outputs="
        << kM * kN << ", cycles=" << program.max_cycle + 64
        << ", mismatches=0, nonzero=" << nonzero << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "block8_linear_projection_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
