#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef FTLPU_TEST_SEQUENCE_LENGTH
#define FTLPU_TEST_SEQUENCE_LENGTH 128
#endif
#ifndef FTLPU_TEST_HIDDEN_SIZE
#define FTLPU_TEST_HIDDEN_SIZE 576
#endif
#ifndef FTLPU_TEST_INTERMEDIATE_SIZE
#define FTLPU_TEST_INTERMEDIATE_SIZE 1536
#endif
#ifndef FTLPU_TEST_QUERY_HEADS
#define FTLPU_TEST_QUERY_HEADS 9
#endif
#ifndef FTLPU_TEST_KV_HEADS
#define FTLPU_TEST_KV_HEADS 3
#endif
#ifndef FTLPU_TEST_HEAD_DIM
#define FTLPU_TEST_HEAD_DIM 64
#endif
#ifndef FTLPU_TEST_RMS_EPSILON
#define FTLPU_TEST_RMS_EPSILON 1.0e-5f
#endif
#ifndef FTLPU_TEST_ROPE_THETA
#define FTLPU_TEST_ROPE_THETA 100000.0f
#endif
#ifndef FTLPU_TEST_MODEL_NAME
#define FTLPU_TEST_MODEL_NAME "SmolLM2"
#endif
#ifndef FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES
#define FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES 0
#endif

constexpr std::size_t kSeqLen = FTLPU_TEST_SEQUENCE_LENGTH;
constexpr std::size_t kHidden = FTLPU_TEST_HIDDEN_SIZE;
constexpr std::size_t kIntermediate = FTLPU_TEST_INTERMEDIATE_SIZE;
constexpr std::size_t kQueryHeads = FTLPU_TEST_QUERY_HEADS;
constexpr std::size_t kKvHeads = FTLPU_TEST_KV_HEADS;
constexpr std::size_t kHeadDim = FTLPU_TEST_HEAD_DIM;
constexpr std::size_t kHeadBlocks = kHeadDim / 32;
constexpr std::size_t kTokenBlocks = kSeqLen / 32;
constexpr std::size_t kTileRows = 4;
constexpr float kEpsilon = FTLPU_TEST_RMS_EPSILON;
constexpr float kRopeTheta = FTLPU_TEST_ROPE_THETA;

static_assert(kQueryHeads % kKvHeads == 0);
static_assert(kHeadDim % 32 == 0);

template <typename T, std::size_t N>
std::string arrayValues(const std::array<T, N>& values)
{
    std::string result = "[";
    for (std::size_t index = 0; index < N; ++index) {
        if (index != 0) result += ",";
        result += std::to_string(values[index]);
    }
    return result + "]";
}

float bf16(float value)
{
    return ftlpu::Bf16::from_float(value).to_float();
}

std::size_t bf16UlpDistance(float lhs, float rhs)
{
    const auto ordered = [](float value) {
        const std::uint16_t bits =
            ftlpu::Bf16::from_float(value).bits();
        return bits & 0x8000u
            ? static_cast<std::uint32_t>(0x8000u - (bits & 0x7fffu))
            : static_cast<std::uint32_t>(0x8000u + bits);
    };
    const std::uint32_t lhsOrdered = ordered(lhs);
    const std::uint32_t rhsOrdered = ordered(rhs);
    return lhsOrdered > rhsOrdered
        ? lhsOrdered - rhsOrdered : rhsOrdered - lhsOrdered;
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
    const std::size_t halfHead = kHeadDim / 2;
    const std::size_t pair = dimension % halfHead;
    const std::size_t base = token * kHidden + head * kHeadDim;
    const float low = projection[base + pair];
    const float high = projection[base + pair + halfHead];
    const float inverse = 1.0f / std::pow(
        kRopeTheta, static_cast<float>(2 * pair) / kHeadDim);
    const float angle = static_cast<float>(token) * inverse;
    const float cosine = bf16(std::cos(angle));
    const float sine = bf16(std::sin(angle));
    return bf16(dimension < halfHead
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
    std::size_t firstCycle = std::numeric_limits<std::size_t>::max();
    for (const auto& relocation : program.address_relocations) {
        if (relocation.binding_index != bindingIndex
            || relocation.binding_access
                != ftlpu::software::runtime::BindingAccess::Input
            || relocation.queue_kind
                != ftlpu::software::runtime::QueueKind::Mem
            || relocation.write_port)
            continue;
        const auto queue = std::find_if(program.queues.begin(),
            program.queues.end(), [&](const auto& candidate) {
                return candidate.kind == relocation.queue_kind
                    && candidate.index == relocation.queue_index;
            });
        if (queue == program.queues.end()
            || relocation.command_index >= queue->commands.size())
            throw std::logic_error("invalid binding address relocation");
        std::size_t cycle = 0;
        for (std::size_t commandIndex = 0;
             commandIndex <= relocation.command_index; ++commandIndex) {
            const auto& command = queue->commands[commandIndex];
            if (ftlpu::software::runtime::
                    is_mem_write_read_2d_raw_packet_header(command)) {
                constexpr std::size_t packetWords =
                    ftlpu::isa::EncodedMemIcuWriteRead2DPacket::kWordCount;
                if (commandIndex + packetWords > queue->commands.size())
                    throw std::logic_error(
                        "truncated WRITE_READ_2D packet before binding relocation");
                if (relocation.command_index >= commandIndex
                    && relocation.command_index < commandIndex + packetWords)
                    throw std::logic_error(
                        "binding relocation references a WRITE_READ_2D packet");
                const auto instruction = ftlpu::software::runtime::
                    decode_mem_write_read_2d_raw_packet(*queue, commandIndex);
                cycle += ftlpu::detail::
                    mem_icu_write_read_2d_last_issue_cycle(instruction) + 1;
                commandIndex += packetWords - 1;
                continue;
            }
            if (ftlpu::software::runtime::is_fu_3d_raw_packet_header(
                    command)) {
                constexpr std::size_t packetWords =
                    ftlpu::isa::EncodedMemIcu3DPacket::kWordCount;
                if (commandIndex + packetWords > queue->commands.size())
                    throw std::logic_error(
                        "truncated raw MEM 3-D binding relocation");
                if (relocation.command_index > commandIndex
                    && relocation.command_index < commandIndex + packetWords)
                    throw std::logic_error(
                        "binding relocation references the middle of a raw MEM 3-D packet");

                ftlpu::isa::EncodedMemIcu3DPacket packet{};
                for (std::size_t word = 0; word < packetWords; ++word) {
                    const auto& source = queue->commands[commandIndex + word];
                    if (source.instruction_kind
                            != ftlpu::software::runtime::InstructionKind::Mem
                        || source.word_count
                            != ftlpu::isa::EncodedMemIcu3DPacket::kLanesPerWord)
                        throw std::logic_error(
                            "raw MEM 3-D binding relocation has an invalid physical word");
                    for (std::size_t lane = 0;
                         lane < ftlpu::isa::EncodedMemIcu3DPacket::kLanesPerWord;
                         ++lane)
                        packet.words[word].lanes[lane] = source.words[lane];
                }
                const auto instruction =
                    ftlpu::isa::decode_mem_icu_3d_instruction(packet);
                if (commandIndex == relocation.command_index) {
                    if (instruction.opcode != ftlpu::MemIcuOpcode::Read3D)
                        throw std::logic_error(
                            "binding relocation does not reference a MEM read");
                    // Account for queue-local wait_cycle when locating the
                    // first binding read after NOP removal.
                    firstCycle = std::min(firstCycle,
                        cycle + instruction.loop.start_cycle
                            + instruction.loop.wait_cycle);
                    break;
                }
                std::size_t finalCycle = cycle
                    + instruction.loop.start_cycle
                    + instruction.loop.wait_cycle;
                for (std::size_t dimension = 0;
                     dimension < ftlpu::IcuLoop3D::kDimensions; ++dimension)
                    finalCycle += (instruction.loop.counts[dimension] - 1)
                        * instruction.loop.cycle_strides[dimension];
                cycle = finalCycle + 1;
                commandIndex += packetWords - 1;
                continue;
            }
            if (ftlpu::software::runtime::is_mem_slice_program_command(
                    command)) {
                const auto sliceProgram = ftlpu::software::runtime::
                    decode_mem_slice_program_command(command);
                if (commandIndex == relocation.command_index) {
                    bool foundRead = false;
                    for (const auto& body : sliceProgram.body) {
                        if (body.instruction.opcode != ftlpu::MemOpcode::Read)
                            continue;
                        firstCycle = std::min(firstCycle,
                            sliceProgram.schedule.start_cycle
                                + body.cycle_offset);
                        foundRead = true;
                    }
                    if (!foundRead)
                        throw std::logic_error(
                            "binding relocation does not reference a MEM read");
                    break;
                }
                std::size_t finalCycle = sliceProgram.schedule.start_cycle;
                for (const auto& body : sliceProgram.body) {
                    std::size_t bodyFinalCycle =
                        sliceProgram.schedule.start_cycle + body.cycle_offset;
                    for (std::size_t dimension = 0;
                         dimension < sliceProgram.schedule.rank; ++dimension)
                        bodyFinalCycle +=
                            (sliceProgram.schedule.counts[dimension] - 1)
                            * sliceProgram.schedule.cycle_strides[dimension];
                    finalCycle = std::max(finalCycle, bodyFinalCycle);
                }
                cycle = std::max(cycle, finalCycle + 1);
                continue;
            }
            if (ftlpu::software::runtime::is_mem_stream_nd_command(
                    command)) {
                const auto schedule = ftlpu::software::runtime::
                    decode_mem_stream_nd_command(command);
                if (commandIndex == relocation.command_index) {
                    const auto encoded =
                        static_cast<ftlpu::isa::EncodedMemInstruction>(
                            command.words[0])
                        | (static_cast<ftlpu::isa::EncodedMemInstruction>(
                               command.words[1])
                            << 32);
                    if (ftlpu::isa::decode_mem_instruction(encoded).opcode
                        != ftlpu::MemOpcode::Read)
                        throw std::logic_error(
                            "binding relocation does not reference a MEM read");
                    firstCycle = std::min(
                        firstCycle, schedule.start_cycle);
                    break;
                }
                std::size_t finalCycle = schedule.start_cycle;
                for (std::size_t dimension = 0;
                     dimension < schedule.rank; ++dimension)
                    finalCycle += (schedule.counts[dimension] - 1)
                        * schedule.cycle_strides[dimension];
                cycle = std::max(cycle, finalCycle + 1);
                continue;
            }
            if (ftlpu::software::runtime::is_macro_schedule_command(
                    command)) {
                const auto schedule =
                    ftlpu::software::runtime::decode_macro_schedule_command(
                        command);
                if (commandIndex == relocation.command_index) {
                    if (command.instruction_kind
                        != ftlpu::software::runtime::InstructionKind::Mem)
                        throw std::logic_error(
                            "binding relocation does not reference a MEM macro");
                    const auto encoded =
                        static_cast<ftlpu::isa::EncodedMemInstruction>(
                            command.words[0])
                        | (static_cast<ftlpu::isa::EncodedMemInstruction>(
                               command.words[1])
                            << 32);
                    if (ftlpu::isa::decode_mem_instruction(encoded).opcode
                        != ftlpu::MemOpcode::Read)
                        throw std::logic_error(
                            "binding relocation does not reference a MEM read");
                    firstCycle =
                        std::min(firstCycle, schedule.start_cycle);
                    break;
                }
                cycle = std::max(cycle,
                    schedule.start_cycle
                        + (schedule.outer_count - 1)
                            * schedule.outer_interval
                        + (schedule.inner_count - 1)
                            * schedule.inner_interval
                        + 1);
                continue;
            }
            if (ftlpu::software::runtime::is_repeat_2d_command(command)) {
                const auto repeat =
                    ftlpu::software::runtime::decode_repeat_2d_command(
                        command);
                cycle += (repeat.outer_count - 1)
                        * repeat.outer_interval
                    + (repeat.inner_count - 1)
                        * repeat.inner_interval;
                continue;
            }
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
            if (commandIndex == relocation.command_index) {
                if (instruction.opcode != ftlpu::MemOpcode::Read)
                    throw std::logic_error(
                        "binding relocation does not reference a MEM read");
                firstCycle = std::min(firstCycle, cycle);
                break;
            }
            ++cycle;
        }
    }
    if (firstCycle != std::numeric_limits<std::size_t>::max())
        return firstCycle;
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
            "usage: compiled_decoder_layer_runtime_test program.ftlpu");
    const auto program =
        ftlpu::software::runtime::read_binary_program(
            std::filesystem::path(argv[1]));
    const auto directProgram = ftlpu::software::runtime::
        without_compiled_mem_read_sync(
            ftlpu::software::runtime::without_compiled_mem_write_sync(
                program));
    const auto expectedAbi =
        ftlpu::software::runtime::executable_target_abi(
            program.hardware);
    if (program.target_abi != expectedAbi
        || program.max_cycle == 0 || program.timelines.empty())
        throw std::logic_error(
            "decoder layer binary has the wrong target or schedule: target_abi="
            + std::to_string(program.target_abi)
            + " expected_abi="
            + std::to_string(expectedAbi)
            + " timelines=" + std::to_string(program.timelines.size())
            + " max_cycle=" + std::to_string(program.max_cycle));
    const bool hasAttentionBias = std::any_of(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Input
                && binding.index == 6 && binding.role == "bias";
        });
    const std::size_t postAttentionNormBinding =
        hasAttentionBias ? 9 : 6;
    const std::size_t gateBinding = hasAttentionBias ? 10 : 7;
    const std::size_t upBinding = hasAttentionBias ? 11 : 8;
    const std::size_t downBinding = hasAttentionBias ? 12 : 9;
    const auto firstFfnRead = std::min({
        firstBindingReadCycle(directProgram, gateBinding),
        firstBindingReadCycle(directProgram, upBinding),
        firstBindingReadCycle(directProgram, downBinding),
    });
    if (firstFfnRead
            <= timeline(program, "rmsnorm.feedback", 1).start_cycle
        || firstFfnRead > program.max_cycle)
            throw std::logic_error(
                "FFN first read is outside the compiled FFN window");
    for (const auto& stage : program.timelines) {
        if (stage.end_cycle <= stage.start_cycle
            || stage.end_cycle > program.max_cycle + 64)
            throw std::logic_error(
                "decoder layer binary has an invalid timeline: "
                + stage.name + " start="
                + std::to_string(stage.start_cycle) + " end="
                + std::to_string(stage.end_cycle) + " max_cycle="
                + std::to_string(program.max_cycle));
    }
    if (const auto* tracePath = std::getenv("FTLPU_SCHEDULE_TRACE"))
        ftlpu::software::runtime::write_schedule_trace_csv(
            program, tracePath);
    if (std::getenv("FTLPU_TRACE_ONLY")) return 0;

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

    std::vector<std::uint8_t> queryBias;
    std::vector<std::uint8_t> keyBias;
    std::vector<std::uint8_t> valueBias;
    if (hasAttentionBias) {
        queryBias.resize(2 * kHidden, 0);
        keyBias.resize(2 * kKvHeads * kHeadDim, 0);
        valueBias.resize(2 * kKvHeads * kHeadDim, 0);
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
    std::array<std::size_t, kKvHeads> valueNonzeroCounts {};
    std::array<float, 8> valueObservedSamples {};
    std::array<float, 8> valueExpectedSamples {};
    std::array<std::size_t, kKvHeads * kHeadBlocks>
        valueSourceNonzeroCounts {};
    std::array<std::size_t, kKvHeads * kHeadBlocks>
        valueDestinationNonzeroCounts {};
    std::array<float, kKvHeads * kHeadBlocks> valueSourceMaxErrors {};
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

    auto systemOwner = std::make_unique<ftlpu::TspSliceSystem>();
    auto* system = systemOwner.get();
    auto runtimeOwner = std::make_unique<
        ftlpu::software::runtime::CModelRuntime>(*system);
    auto& runtime = *runtimeOwner;
    const char* memTracePath = std::getenv("FTLPU_QWEN_MEM_CSV");
    if (memTracePath != nullptr)
        runtime.stream_mem_execution_trace_csv(memTracePath);
    // The direct golden run has no C2C transport. Recover the compiler's
    // reserved idle slots while the ModelSession run below uses the original
    // executable with its MEM_WRITE_SYNC packets.
    runtime.load(directProgram);
    runtime.upload_input(0, input);
    runtime.upload_input(1, gamma0);
    runtime.upload_input(2, queryWeight);
    runtime.upload_input(3, keyWeight);
    runtime.upload_input(4, valueWeight);
    runtime.upload_input(5, outputWeight);
    if (hasAttentionBias) {
        runtime.upload_input(6, queryBias);
        runtime.upload_input(7, keyBias);
        runtime.upload_input(8, valueBias);
    }
    runtime.upload_input(postAttentionNormBinding, gamma1);
    // Paged inputs are staged in host memory here; CModelRuntime installs
    // each page at its compiler-specified ready_cycle. Some pages become
    // ready before their first MEM read, so provide the logical tensor now.
    const auto uploadIfPaged = [&](std::size_t index,
                                   const std::vector<std::uint8_t>& weight) {
        const auto binding = std::find_if(program.bindings.begin(),
            program.bindings.end(), [&](const auto& candidate) {
                return candidate.access
                        == ftlpu::software::runtime::BindingAccess::Input
                    && candidate.index == index;
            });
        if (binding != program.bindings.end() && binding->paged_weight)
            runtime.upload_input(index, weight);
    };
    uploadIfPaged(gateBinding, gateWeight);
    uploadIfPaged(upBinding, upWeight);
    uploadIfPaged(downBinding, downWeight);
    const auto physicalBf16 = [&](ftlpu::Hemisphere hemisphere,
                                  std::size_t lowSlice,
                                  std::size_t highSlice,
                                  std::size_t address,
                                  std::size_t column,
                                  std::size_t bank = 0) {
        const auto low = system->read_mem_sram_lane_byte(
            hemisphere, lowSlice, bank, column / 8, address, column % 8);
        const auto high = system->read_mem_sram_lane_byte(
            hemisphere, highSlice, bank, column / 8, address, column % 8);
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
                && binding.name == "elementwise.add.0"
                && binding.layout
                    == ftlpu::software::runtime::BindingLayout::
                        Fp16MxmDistributed16
                && binding.byte_size == inputBinding->byte_size
                && binding.base_row == inputBinding->base_row;
        });
    if (residualBinding == program.bindings.end()
        || residualBinding->slices.size() != 16)
        throw std::logic_error(
            "cannot locate attention residual physical binding");
    const auto& residualSlices = residualBinding->slices;
    const auto probabilityDiagonalBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "attention.probability_diagonal";
        });
    if (probabilityDiagonalBinding == program.bindings.end()
        || probabilityDiagonalBinding->role != "workspace"
        || probabilityDiagonalBinding->layout
            != ftlpu::software::runtime::BindingLayout::
                Fp16ProbabilityDiagonal
        || probabilityDiagonalBinding->slices.size() != 16)
        throw std::logic_error(
            "decoder binary is missing diagonal probability metadata");
    const auto readMxmDistributed = [&](const std::vector<std::uint16_t>& slices,
                                        std::size_t baseRow,
                                        std::size_t row,
                                        std::size_t column,
                                        ftlpu::Hemisphere hemisphere =
                                            ftlpu::Hemisphere::East,
                                        std::size_t bank = 0) {
        const std::size_t hiddenBlocks = kHidden / 32;
        const std::size_t tokenBlock = row / 32;
        const std::size_t tokenWave = (row % 32) / 8;
        const std::size_t tokenLane = row % 8;
        const std::size_t hiddenBlock = column / 32;
        const std::size_t featureWave = (column % 32) / 8;
        const std::size_t featureLane = column % 8;
        const std::size_t address = baseRow
            + (tokenBlock * hiddenBlocks + hiddenBlock) * 4 + tokenWave;
        return physicalBf16(hemisphere,
            slices[2 * tokenLane], slices[2 * tokenLane + 1],
            address, featureWave * 8 + featureLane, bank);
    };
    const auto normalized0 = rmsNorm(inputValues, 0);
    const auto rms1Binding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "rmsnorm.result.0";
        });
    if (rms1Binding == program.bindings.end()
        || rms1Binding->layout
            != ftlpu::software::runtime::BindingLayout::
                Fp16MxmDistributed16
        || rms1Binding->slices.size() != 16)
        throw std::logic_error(
            "cannot locate first RMSNorm physical binding");
    const std::size_t attentionPrepackEndCycle =
        firstBindingReadCycle(directProgram, 2);
    std::ofstream cmodelLog;
    std::ostream* cmodelLogSink = nullptr;
    if (const auto* path = std::getenv("FTLPU_CMODEL_LOG")) {
        cmodelLog.open(path);
        if (!cmodelLog)
            throw std::runtime_error("cannot open FTLPU_CMODEL_LOG path");
        cmodelLogSink = &cmodelLog;
    }
    const bool scopedCmodelLog =
        std::getenv("FTLPU_CMODEL_LOG_START") != nullptr;
    runtime.run_cycles(attentionPrepackEndCycle,
        scopedCmodelLog ? nullptr : cmodelLogSink);
    float prepackMaxError = 0.0f;
    std::size_t prepackMaxRow = 0;
    std::size_t prepackMaxColumn = 0;
    float prepackMaxActual = 0.0f;
    float prepackMaxExpected = 0.0f;
    std::size_t prepackMismatchCount = 0;
    std::size_t prepackLargeMismatchCount = 0;
    std::size_t prepackUnexpectedZeroCount = 0;
    std::size_t prepackFirstZeroRow = 0;
    std::size_t prepackFirstZeroColumn = 0;
    std::size_t prepackLastZeroRow = 0;
    std::size_t prepackLastZeroColumn = 0;
    std::array<std::size_t, 2> prepackHemisphereMismatchCounts {};
    std::array<float, 2> prepackHemisphereMaxErrors {};
    const std::size_t rms1BaseRow =
        static_cast<std::size_t>(rms1Binding->base_row);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const float observed = readMxmDistributed(
                rms1Binding->slices, rms1BaseRow, row, column,
                ftlpu::Hemisphere::East, rms1Binding->bank);
            const float expected = normalized0[row * kHidden + column];
            const float error = std::fabs(observed - expected);
            for (std::size_t side = 0; side < 2; ++side) {
                const float sideObserved = readMxmDistributed(
                    rms1Binding->slices, rms1BaseRow, row, column,
                    static_cast<ftlpu::Hemisphere>(side),
                    rms1Binding->bank);
                const float sideError = std::fabs(sideObserved - expected);
                prepackHemisphereMaxErrors[side] = std::max(
                    prepackHemisphereMaxErrors[side], sideError);
                if (sideError > 0.04f)
                    ++prepackHemisphereMismatchCounts[side];
            }
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
                prepackMaxActual = observed;
                prepackMaxExpected = expected;
            }
        }
    }
    if (prepackMaxError > 0.04f)
        throw std::logic_error(
            "RMS1 distributed result mismatch max_error="
            + std::to_string(prepackMaxError)
            + " row=" + std::to_string(prepackMaxRow)
            + " column=" + std::to_string(prepackMaxColumn)
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
                rms1Binding->slices, rms1BaseRow,
                prepackMaxRow, prepackMaxColumn,
                ftlpu::Hemisphere::East, rms1Binding->bank))
            + " hemisphere_mismatches="
                + arrayValues(prepackHemisphereMismatchCounts)
            + " hemisphere_max_errors="
                + arrayValues(prepackHemisphereMaxErrors));
    if (std::ranges::any_of(prepackHemisphereMismatchCounts,
            [](std::size_t count) { return count != 0; }))
        throw std::logic_error(
            "RMS1 hemisphere replication mismatch counts="
            + arrayValues(prepackHemisphereMismatchCounts)
            + " max_errors=" + arrayValues(prepackHemisphereMaxErrors));
    const std::size_t qkvEndCycle =
        static_cast<std::size_t>(timeline(program, "qk").start_cycle);
    runtime.run_cycles(
        qkvEndCycle - attentionPrepackEndCycle,
        scopedCmodelLog ? nullptr : cmodelLogSink);
    const auto valueBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && (binding.name == "attention.value"
                    || (binding.name == "attention.value_cache"
                        && binding.role == "state.kv.value"));
        });
    if (valueBinding == program.bindings.end()
        || valueBinding->layout
            != ftlpu::software::runtime::BindingLayout::Fp16ValueX16
        || valueBinding->slices.empty()
        || valueBinding->slices.size() % 16 != 0)
        throw std::logic_error(
            "decoder binary is missing attention value metadata");
    const auto keyBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "attention.key_cache"
                && binding.role == "state.kv.key";
        });
    const std::size_t valueSliceGroups =
        valueBinding->slices.size() / 16;
    const auto readPackedValue = [&](std::size_t head,
                                     std::size_t token,
                                     std::size_t dimension,
                                     ftlpu::Hemisphere hemisphere) {
        const std::size_t headBlock = dimension / 32;
        const std::size_t sliceGroup =
            (headBlock % valueSliceGroups) * 16;
        const std::size_t packedStream = (token % 8) * 2;
        const std::size_t address =
            static_cast<std::size_t>(valueBinding->base_row)
            + (head * kHeadBlocks + headBlock)
                * kTokenBlocks * kTileRows
            + (token % 32) / 8;
        return physicalBf16(hemisphere,
            valueBinding->slices[sliceGroup + packedStream],
            valueBinding->slices[sliceGroup + packedStream + 1],
            address, dimension % 32, valueBinding->bank);
    };
    const auto contextBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "attention.context";
        });
    if (contextBinding == program.bindings.end()
        || (contextBinding->layout
                != ftlpu::software::runtime::BindingLayout::Fp16HeadPlanar
            && contextBinding->layout
                != ftlpu::software::runtime::BindingLayout::
                    Fp16HeadBlockPacked)
        || (contextBinding->layout
                == ftlpu::software::runtime::BindingLayout::
                    Fp16HeadBlockPacked
                ? contextBinding->slices.size() < 4
                : contextBinding->slices.size() < 2 * kHeadBlocks))
        throw std::logic_error(
            "decoder binary is missing attention context metadata");
    float valueTensorMaxError = 0.0f;
    std::size_t valueTensorMaxHead = 0;
    std::size_t valueTensorMaxToken = 0;
    std::size_t valueTensorMaxDimension = 0;
    float valueTensorMaxActual = 0.0f;
    float valueTensorMaxExpected = 0.0f;
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension) {
                const float actual = readPackedValue(head, token, dimension,
                    static_cast<ftlpu::Hemisphere>(head % 2));
                const std::size_t column = head * kHeadDim + dimension;
                const float expected = bf16(
                    normalized0[token * kHidden + sourceHidden(2, column)]
                    * projectionSign(2, column));
                const float error = std::fabs(actual - expected);
                if (error > valueTensorMaxError) {
                    valueTensorMaxError = error;
                    valueTensorMaxHead = head;
                    valueTensorMaxToken = token;
                    valueTensorMaxDimension = dimension;
                    valueTensorMaxActual = actual;
                    valueTensorMaxExpected = expected;
                }
            }
        }
    }
    if (valueTensorMaxError > 0.04f)
        throw std::logic_error(
            "full attention value checkpoint mismatch max_error="
            + std::to_string(valueTensorMaxError)
            + " head=" + std::to_string(valueTensorMaxHead)
            + " token=" + std::to_string(valueTensorMaxToken)
            + " dimension=" + std::to_string(valueTensorMaxDimension)
            + " actual=" + std::to_string(valueTensorMaxActual)
            + " expected=" + std::to_string(valueTensorMaxExpected));
    for (std::size_t column = 0;
         column < kKvHeads * kHeadDim; ++column) {
        const std::size_t head = column / kHeadDim;
        const std::size_t dimension = column % kHeadDim;
        const std::size_t reduction = dimension / 32;
        const std::size_t stream = 0;
        const std::size_t address =
            static_cast<std::size_t>(valueBinding->base_row)
            + (head * kHeadBlocks + reduction)
                * kTokenBlocks * kTileRows;
        const std::size_t sliceGroup =
            (reduction % valueSliceGroups) * 16;
        const float observed = physicalBf16(
            static_cast<ftlpu::Hemisphere>(head % 2),
            valueBinding->slices[sliceGroup + stream],
            valueBinding->slices[sliceGroup + stream + 1],
            address, dimension % 32, valueBinding->bank);
        const float expected = bf16(
            normalized0[sourceHidden(2, column)]
            * projectionSign(2, column));
        const auto block = head * kHeadBlocks + reduction;
        const float sourceObserved = physicalBf16(
            static_cast<ftlpu::Hemisphere>((block % 4) / 2),
            valueBinding->slices[sliceGroup + stream],
            valueBinding->slices[sliceGroup + stream + 1],
            address, dimension % 32, valueBinding->bank);
        if (sourceObserved != 0.0f) ++valueSourceNonzeroCounts[block];
        if (observed != 0.0f) ++valueDestinationNonzeroCounts[block];
        valueSourceMaxErrors[block] = std::max(
            valueSourceMaxErrors[block],
            std::fabs(sourceObserved - expected));
        if (observed != 0.0f) ++valueNonzeroCounts[head];
        if (column < valueObservedSamples.size()) {
            valueObservedSamples[column] = observed;
            valueExpectedSamples[column] = expected;
        }
        const float error = std::fabs(observed - expected);
        valueMaxErrors[head] = std::max(valueMaxErrors[head], error);
        if (error > 0.04f) {
            ++valueMismatchCounts[head];
            valueMismatchColumns[head].push_back(column % kHeadDim);
            valueMismatchMasks[head] |=
                std::uint64_t {1} << (column % 64);
        }
    }
    std::array<std::size_t, 8> vxmRemaining {};
    for (std::size_t stage = 0; stage < vxmRemaining.size(); ++stage)
        vxmRemaining[stage] = system->vxm_unit()
                                   .superlane(0)
                                   .remaining_executions(stage);
    const auto copiedBlockSlices = valueBinding->slices.begin();
    const float copiedToken4 = physicalBf16(
        ftlpu::Hemisphere::East,
        copiedBlockSlices[8], copiedBlockSlices[9],
        static_cast<std::size_t>(valueBinding->base_row)
            + 2 * kTokenBlocks * kTileRows,
        0, valueBinding->bank);
    const float copiedToken4Expected = bf16(
        normalized0[4 * kHidden + sourceHidden(2, 64)]
        * projectionSign(2, 64));
    const std::array<float, 2> remoteSourceSamples {{
        physicalBf16(ftlpu::Hemisphere::West,
            copiedBlockSlices[0], copiedBlockSlices[1],
            static_cast<std::size_t>(valueBinding->base_row)
                + 2 * kTokenBlocks * kTileRows,
            0, valueBinding->bank),
        physicalBf16(ftlpu::Hemisphere::East,
            copiedBlockSlices[0], copiedBlockSlices[1],
            static_cast<std::size_t>(valueBinding->base_row)
                + 4 * kTokenBlocks * kTileRows,
            0, valueBinding->bank),
    }};
    const std::array<std::size_t, 2> passiveBridgeCounts {{
        system->passive_bridge_transfer_count(
            ftlpu::Hemisphere::East, 0),
        system->passive_bridge_transfer_count(
            ftlpu::Hemisphere::West, 0),
    }};
    const auto bridgeCycleValue = [&](ftlpu::Hemisphere source) {
        const auto cycle = system->last_passive_bridge_cycle(source, 0);
        return cycle.has_value() ? *cycle : std::size_t {0};
    };
    const std::array<std::size_t, 2> passiveBridgeLastCycles {{
        bridgeCycleValue(ftlpu::Hemisphere::East),
        bridgeCycleValue(ftlpu::Hemisphere::West),
    }};
    if (std::ranges::any_of(
            valueMismatchCounts, [](std::size_t count) {
                return count != 0;
            }))
        throw std::logic_error(
            "RMS1-to-value projection mismatch counts=["
            + arrayValues(valueMismatchCounts)
            + " max_errors=" + arrayValues(valueMaxErrors)
            + " masks=" + arrayValues(valueMismatchMasks)
            + " nonzero=" + arrayValues(valueNonzeroCounts)
            + " observed0=" + arrayValues(valueObservedSamples)
            + " expected0=" + arrayValues(valueExpectedSamples)
            + " vxm_remaining=" + arrayValues(vxmRemaining)
            + " copied_token4=" + std::to_string(copiedToken4)
            + "/" + std::to_string(copiedToken4Expected)
            + " remote_sources=" + arrayValues(remoteSourceSamples)
            + " bridge_counts=" + arrayValues(passiveBridgeCounts)
            + " bridge_last_cycles="
                + arrayValues(passiveBridgeLastCycles)
            + " source_nonzero=" + arrayValues(valueSourceNonzeroCounts)
            + " destination_nonzero="
                + arrayValues(valueDestinationNonzeroCounts)
            + " source_max_errors=" + arrayValues(valueSourceMaxErrors));

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
    if constexpr (kSeqLen == 32 && kHeadDim == 128) {
        if (keyBinding == program.bindings.end()
            || keyBinding->layout
                != ftlpu::software::runtime::BindingLayout::Fp16HeadPlanar
            || keyBinding->shape.empty() || keyBinding->slices.size() < 4)
            throw std::logic_error(
                "decoder binary is missing attention key-cache metadata");
        float queryRopeMaxError = 0.0f;
        float keyRopeMaxError = 0.0f;
        std::size_t queryRopeMismatchCount = 0;
        std::size_t keyRopeMismatchCount = 0;
        std::array<std::size_t, 4> queryLocation {};
        std::array<std::size_t, 4> keyLocation {};
        std::array<float, 2> queryValues {};
        std::array<float, 2> keyValues {};
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (std::size_t token = 0; token < kSeqLen; ++token) {
                const std::size_t tokenLane = token % 8;
                const std::size_t tokenWave = (token % 32) / 8;
                for (std::size_t head = 0; head < kQueryHeads; ++head) {
                    for (std::size_t dimension = 0;
                         dimension < kHeadDim; ++dimension) {
                        const std::size_t reduction = dimension / 32;
                        const std::size_t address = 576
                            + (head * 2 + reduction % 2) * 4
                            + tokenWave;
                        const float actual = physicalBf16(
                            static_cast<ftlpu::Hemisphere>(hemisphere),
                            2 * tokenLane, 2 * tokenLane + 1, address,
                            dimension % 32, reduction / 2);
                        const float expected = ropeValue(
                            checkpointQuery, token, head, dimension);
                        const float error = std::fabs(actual - expected);
                        if (error > 0.02f) {
                            if (queryRopeMismatchCount < 16)
                                std::cerr << "Q RoPE mismatch hemisphere="
                                          << hemisphere << " head=" << head
                                          << " token=" << token
                                          << " dimension=" << dimension
                                          << " actual=" << actual
                                          << " expected=" << expected
                                          << " error=" << error << '\n';
                            ++queryRopeMismatchCount;
                        }
                        if (error > queryRopeMaxError) {
                            queryRopeMaxError = error;
                            queryLocation = {
                                hemisphere, head, token, dimension};
                            queryValues = {actual, expected};
                        }
                    }
                }
                for (std::size_t head = 0; head < kKvHeads; ++head) {
                    for (std::size_t dimension = 0;
                         dimension < kHeadDim; ++dimension) {
                        const std::size_t reduction = dimension / 32;
                        const std::size_t blocksPerRotaryHalf =
                            std::max<std::size_t>(1, kHeadBlocks / 2);
                        const std::size_t sliceGroup =
                            reduction / blocksPerRotaryHalf;
                        const std::size_t slice =
                            keyBinding->slices[2 * sliceGroup];
                        const std::size_t storageTokens =
                            static_cast<std::size_t>(keyBinding->shape[0]);
                        const std::size_t address =
                            static_cast<std::size_t>(keyBinding->base_row)
                            + (head * blocksPerRotaryHalf
                                  + reduction % blocksPerRotaryHalf)
                                * storageTokens
                            + token;
                        const float actual = physicalBf16(
                            static_cast<ftlpu::Hemisphere>(hemisphere),
                            slice, slice + 1, address, dimension % 32,
                            keyBinding->bank);
                        const float expected = ropeValue(
                            checkpointKey, token, head, dimension);
                        const float error = std::fabs(actual - expected);
                        if (error > 0.02f) {
                            if (keyRopeMismatchCount < 16)
                                std::cerr << "K RoPE mismatch hemisphere="
                                          << hemisphere << " head=" << head
                                          << " token=" << token
                                          << " dimension=" << dimension
                                          << " actual=" << actual
                                          << " expected=" << expected
                                          << " error=" << error << '\n';
                            ++keyRopeMismatchCount;
                        }
                        if (error > keyRopeMaxError) {
                            keyRopeMaxError = error;
                            keyLocation = {
                                hemisphere, head, token, dimension};
                            keyValues = {actual, expected};
                        }
                    }
                }
            }
        }
        if (queryRopeMaxError > 0.02f || keyRopeMaxError > 0.02f)
            throw std::logic_error(
                "direct RoPE physical checkpoint mismatch query_error="
                + std::to_string(queryRopeMaxError)
                + " query_mismatches="
                + std::to_string(queryRopeMismatchCount)
                + " query_location=" + arrayValues(queryLocation)
                + " query_values=" + arrayValues(queryValues)
                + " key_error=" + std::to_string(keyRopeMaxError)
                + " key_mismatches="
                + std::to_string(keyRopeMismatchCount)
                + " key_location=" + arrayValues(keyLocation)
                + " key_values=" + arrayValues(keyValues));
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
    const std::size_t pvStartCycle =
        static_cast<std::size_t>(timeline(program, "pv").start_cycle);
    runtime.run_cycles(pvStartCycle - qkvEndCycle,
        scopedCmodelLog ? nullptr : cmodelLogSink);
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        const std::size_t kvHead =
            head / (kQueryHeads / kKvHeads);
        const float probability = physicalBf16(
            static_cast<ftlpu::Hemisphere>(kvHead % 2),
            probabilityDiagonalBinding->slices[0],
            probabilityDiagonalBinding->slices[1],
            static_cast<std::size_t>(
                probabilityDiagonalBinding->base_row)
                + head * kTokenBlocks * kTokenBlocks * kTileRows,
            0, probabilityDiagonalBinding->bank);
        if (std::fabs(probability - 1.0f) > 0.01f)
            throw std::logic_error(
                "softmax probability checkpoint mismatch head="
                + std::to_string(head) + " actual="
                + std::to_string(probability) + " expected=1.0");
    }
    const auto& probabilitySlices =
        probabilityDiagonalBinding->slices;
    const std::size_t probabilityBaseRow =
        static_cast<std::size_t>(
            probabilityDiagonalBinding->base_row);
    const std::size_t pvEndCycle =
        static_cast<std::size_t>(timeline(program, "o_proj").start_cycle);
    constexpr std::size_t kCheckpointDrainCycles = 64;
    const std::size_t pvCheckpointCycles =
        pvEndCycle - pvStartCycle + kCheckpointDrainCycles;
    if (scopedCmodelLog) {
        const std::size_t logStart = static_cast<std::size_t>(
            std::strtoull(std::getenv("FTLPU_CMODEL_LOG_START"), nullptr, 10));
        const auto* logCyclesText = std::getenv("FTLPU_CMODEL_LOG_CYCLES");
        const std::size_t requestedLogCycles = logCyclesText
            ? static_cast<std::size_t>(
                  std::strtoull(logCyclesText, nullptr, 10))
            : 64;
        if (logStart < pvStartCycle
            || logStart >= pvStartCycle + pvCheckpointCycles)
            throw std::logic_error(
                "FTLPU_CMODEL_LOG_START is outside the PV checkpoint");
        const std::size_t prefixCycles = logStart - pvStartCycle;
        const std::size_t logCycles = std::min(requestedLogCycles,
            pvCheckpointCycles - prefixCycles);
        runtime.run_cycles(prefixCycles);
        runtime.run_cycles(logCycles, cmodelLogSink);
        runtime.run_cycles(pvCheckpointCycles - prefixCycles - logCycles);
    } else {
        runtime.run_cycles(pvCheckpointCycles, cmodelLogSink);
    }
    // Softmax and PV intentionally overlap across head waves. Read the final
    // probability tensor after PV has drained; a snapshot at pv.start would
    // observe the tail heads before their last causal entries are written.
    for (std::size_t query = 0; query < kSeqLen; ++query) {
        const std::size_t queryBlock = query / 32;
        const std::size_t queryRow = query % 8;
        const std::size_t diagonal = (query % 32) / 8;
        for (std::size_t head = 0; head < kQueryHeads; ++head) {
            const std::size_t kvHead =
                head / (kQueryHeads / kKvHeads);
            for (std::size_t dimension = 0;
                 dimension < kHeadDim; ++dimension) {
                float context = 0.0f;
                for (std::size_t key = 0; key <= query; ++key) {
                    const std::size_t keyBlock = key / 32;
                    const std::size_t address = probabilityBaseRow
                        + ((head * kTokenBlocks + queryBlock)
                                * kTokenBlocks
                            + keyBlock)
                            * kTileRows
                        + diagonal;
                    const float probability = physicalBf16(
                        static_cast<ftlpu::Hemisphere>(kvHead % 2),
                        probabilitySlices[queryRow * 2],
                        probabilitySlices[queryRow * 2 + 1], address,
                        key % 32, probabilityDiagonalBinding->bank);
                    context += probability
                        * checkpointValue[key * kHidden
                            + kvHead * kHeadDim + dimension];
                }
                checkpointContext[query * kHidden
                    + head * kHeadDim + dimension] = bf16(context);
            }
        }
    }
    float contextMaxError = 0.0f;
    std::size_t contextMaxHemisphere = 0;
    std::size_t contextMaxRow = 0;
    std::size_t contextMaxColumn = 0;
    float contextMaxActual = 0.0f;
    float contextMaxExpected = 0.0f;
    std::array<std::size_t, kQueryHeads> contextMismatchCounts {};
    std::array<std::size_t, kQueryHeads> contextUnexpectedZeroCounts {};
    std::array<std::size_t, kQueryHeads * kHeadBlocks>
        contextBlockMismatchCounts {};
    std::array<std::size_t, kQueryHeads> contextFirstMismatchRows {};
    std::array<std::size_t, kQueryHeads> contextLastMismatchRows {};
    contextFirstMismatchRows.fill(kSeqLen);
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
                const std::size_t head = column / kHeadDim;
                const std::size_t dimension = column % kHeadDim;
                const std::size_t headBlock = dimension / 32;
                const std::size_t kvHead =
                    head / (kQueryHeads / kKvHeads);
                const std::size_t hemisphere = kvHead % 2;
                const bool headBlockPacked =
                    contextBinding->layout
                    == ftlpu::software::runtime::BindingLayout::
                        Fp16HeadBlockPacked;
                const std::size_t sliceBase =
                    headBlockPacked ? hemisphere * 2 : headBlock * 2;
                const std::size_t contextAddress =
                    static_cast<std::size_t>(contextBinding->base_row)
                    + (headBlockPacked
                        ? (head * kHeadBlocks + headBlock) * kSeqLen
                        : head * kSeqLen)
                    + row;
                const float observed = physicalBf16(
                    static_cast<ftlpu::Hemisphere>(hemisphere),
                    contextBinding->slices[sliceBase],
                    contextBinding->slices[sliceBase + 1],
                    contextAddress,
                    dimension % 32, contextBinding->bank);
                const float expected =
                    checkpointContext[row * kHidden + column];
                const float error = std::fabs(observed - expected);
                if (error > 0.04f) {
                    ++contextMismatchCounts[head];
                    ++contextBlockMismatchCounts[
                        head * kHeadBlocks + headBlock];
                    contextFirstMismatchRows[head] = std::min(
                        contextFirstMismatchRows[head], row);
                    contextLastMismatchRows[head] = row;
                    if (observed == 0.0f && expected != 0.0f)
                        ++contextUnexpectedZeroCounts[head];
                }
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
    std::size_t contextClosestHead = 0;
    std::size_t contextClosestRow = 0;
    float contextClosestMeanError =
        std::numeric_limits<float>::infinity();
    if (contextMaxError > 0.04f) {
        const std::size_t failingHead = contextMaxColumn / kHeadDim;
        const std::size_t failingKvHead =
            failingHead / (kQueryHeads / kKvHeads);
        std::array<float, kSeqLen> failingProbabilities {};
        std::array<float, kSeqLen> failingValues {};
        std::array<float, kSeqLen> failingExpectedValues {};
        const std::size_t failingQueryBlock = contextMaxRow / 32;
        const std::size_t failingQueryRow = contextMaxRow % 8;
        const std::size_t failingDiagonal = (contextMaxRow % 32) / 8;
        for (std::size_t key = 0; key <= contextMaxRow; ++key) {
            const std::size_t keyBlock = key / 32;
            const std::size_t address = probabilityBaseRow
                + ((failingHead * kTokenBlocks + failingQueryBlock)
                        * kTokenBlocks
                    + keyBlock)
                    * kTileRows
                + failingDiagonal;
            failingProbabilities[key] = physicalBf16(
                static_cast<ftlpu::Hemisphere>(failingKvHead % 2),
                probabilitySlices[failingQueryRow * 2],
                probabilitySlices[failingQueryRow * 2 + 1], address,
                key % 32, probabilityDiagonalBinding->bank);
            failingValues[key] = readPackedValue(failingKvHead, key,
                contextMaxColumn % kHeadDim,
                static_cast<ftlpu::Hemisphere>(failingKvHead % 2));
            failingExpectedValues[key] = checkpointValue[key * kHidden
                + failingKvHead * kHeadDim
                + contextMaxColumn % kHeadDim];
        }
        const bool headBlockPacked =
            contextBinding->layout
            == ftlpu::software::runtime::BindingLayout::
                Fp16HeadBlockPacked;
        std::array<float, kHeadDim> observedVector {};
        for (std::size_t dimension = 0;
             dimension < kHeadDim; ++dimension) {
            const std::size_t headBlock = dimension / 32;
            const std::size_t sliceBase = headBlockPacked
                ? (failingKvHead % 2) * 2 : headBlock * 2;
            const std::size_t address =
                static_cast<std::size_t>(contextBinding->base_row)
                + (headBlockPacked
                    ? (failingHead * kHeadBlocks + headBlock) * kSeqLen
                    : failingHead * kSeqLen)
                + contextMaxRow;
            observedVector[dimension] = physicalBf16(
                static_cast<ftlpu::Hemisphere>(failingKvHead % 2),
                contextBinding->slices[sliceBase],
                contextBinding->slices[sliceBase + 1], address,
                dimension % 32, contextBinding->bank);
        }
        const std::size_t firstHead =
            failingKvHead * (kQueryHeads / kKvHeads);
        const std::size_t lastHead =
            firstHead + kQueryHeads / kKvHeads;
        for (std::size_t candidateHead = firstHead;
             candidateHead < lastHead; ++candidateHead) {
            for (std::size_t candidateRow = 0;
                 candidateRow < kSeqLen; ++candidateRow) {
                float totalError = 0.0f;
                for (std::size_t dimension = 0;
                     dimension < kHeadDim; ++dimension)
                    totalError += std::fabs(observedVector[dimension]
                        - checkpointContext[candidateRow * kHidden
                            + candidateHead * kHeadDim + dimension]);
                const float meanError =
                    totalError / static_cast<float>(kHeadDim);
                if (meanError < contextClosestMeanError) {
                    contextClosestMeanError = meanError;
                    contextClosestHead = candidateHead;
                    contextClosestRow = candidateRow;
                }
            }
        }
        throw std::logic_error(
            "PV context mismatch max_error="
            + std::to_string(contextMaxError)
            + " hemisphere="
            + std::to_string(contextMaxHemisphere)
            + " row=" + std::to_string(contextMaxRow)
            + " column=" + std::to_string(contextMaxColumn)
            + " actual=" + std::to_string(contextMaxActual)
            + " expected=" + std::to_string(contextMaxExpected)
            + " expected_prev_row=" + std::to_string(
                checkpointContext[(contextMaxRow == 0
                        ? contextMaxRow : contextMaxRow - 1) * kHidden
                    + contextMaxColumn])
            + " expected_next_row=" + std::to_string(
                checkpointContext[(contextMaxRow + 1 < kSeqLen
                        ? contextMaxRow + 1 : contextMaxRow) * kHidden
                    + contextMaxColumn])
            + " expected_prev_head=" + std::to_string(
                checkpointContext[contextMaxRow * kHidden
                    + (contextMaxColumn >= kHeadDim
                        ? contextMaxColumn - kHeadDim
                        : contextMaxColumn)])
            + " closest_expected=(head="
                + std::to_string(contextClosestHead)
                + ",row=" + std::to_string(contextClosestRow)
                + ",mean_error="
                + std::to_string(contextClosestMeanError) + ")"
            + " mismatch_counts=" + arrayValues(contextMismatchCounts)
            + " block_mismatch_counts="
                + arrayValues(contextBlockMismatchCounts)
            + " first_mismatch_rows="
                + arrayValues(contextFirstMismatchRows)
            + " last_mismatch_rows="
                + arrayValues(contextLastMismatchRows)
            + " zero_counts="
                + arrayValues(contextUnexpectedZeroCounts)
            + " probabilities=" + arrayValues(failingProbabilities)
            + " values=" + arrayValues(failingValues)
            + " expected_values=" + arrayValues(failingExpectedValues));
    }

    const std::size_t outputProjectionEndCycle =
        static_cast<std::size_t>(
            timeline(program, "elementwise.add").start_cycle);
    if (outputProjectionEndCycle
        < pvEndCycle + kCheckpointDrainCycles)
        throw std::logic_error(
            "PV checkpoint drain overlaps the O projection checkpoint");
    runtime.run_cycles(outputProjectionEndCycle - pvEndCycle
        - kCheckpointDrainCycles, cmodelLogSink);
    float outputProjectionMaxError = 0.0f;
    std::size_t outputProjectionMaxRow = 0;
    std::size_t outputProjectionMaxColumn = 0;
    float outputProjectionMaxActual = 0.0f;
    float outputProjectionMaxExpected = 0.0f;
    float outputProjectionReplicaMaxError = 0.0f;
    std::size_t outputProjectionReplicaMaxHemisphere = 0;
    std::size_t outputProjectionReplicaMaxRow = 0;
    std::size_t outputProjectionReplicaMaxColumn = 0;
    float outputProjectionReplicaMaxActual = 0.0f;
    const auto attentionResultBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "attention.result";
        });
    if (attentionResultBinding == program.bindings.end())
        throw std::logic_error(
            "decoder binary is missing attention result metadata");
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
            float observed = 0.0f;
            if (attentionResultBinding->layout
                == ftlpu::software::runtime::BindingLayout::
                    Fp16MxmBlock8Distributed16) {
                const std::size_t hiddenBlocks = kHidden / 32;
                const std::size_t tokenBlock = row / 32;
                const std::size_t tokenWave = (row % 32) / 8;
                const std::size_t tokenLane = row % 8;
                const std::size_t hiddenBlock = column / 32;
                const std::size_t featureWave = (column % 32) / 8;
                const std::size_t featureLane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(
                        attentionResultBinding->base_row)
                    + (tokenBlock * hiddenBlocks + hiddenBlock) * 4
                    + tokenWave;
                const std::size_t mxmsPerHemisphere =
                    program.hardware.mxms_per_hemisphere;
                const auto hemisphere = static_cast<ftlpu::Hemisphere>(
                    (hiddenBlock / mxmsPerHemisphere) % 2);
                observed = physicalBf16(hemisphere,
                    attentionResultBinding->slices[2 * tokenLane],
                    attentionResultBinding->slices[2 * tokenLane + 1],
                    address, featureWave * 8 + featureLane,
                    attentionResultBinding->bank);
                for (std::size_t side = 0; side < 2; ++side) {
                    const float replica = physicalBf16(
                        static_cast<ftlpu::Hemisphere>(side),
                        attentionResultBinding->slices[2 * tokenLane],
                        attentionResultBinding->slices[2 * tokenLane + 1],
                        address, featureWave * 8 + featureLane,
                        attentionResultBinding->bank);
                    const float replicaError =
                        std::fabs(replica - expected);
                    if (replicaError > outputProjectionReplicaMaxError) {
                        outputProjectionReplicaMaxError = replicaError;
                        outputProjectionReplicaMaxHemisphere = side;
                        outputProjectionReplicaMaxRow = row;
                        outputProjectionReplicaMaxColumn = column;
                        outputProjectionReplicaMaxActual = replica;
                    }
                }
            } else {
                if (attentionResultBinding->layout
                        != ftlpu::software::runtime::BindingLayout::
                            Fp16PairPlanar
                    || attentionResultBinding->slices.size() < 4)
                    throw std::logic_error(
                        "attention result has an unsupported vector layout");
                const std::size_t pair = (column % 64) / 32;
                observed = physicalBf16(
                    ftlpu::Hemisphere::East,
                    attentionResultBinding->slices[pair * 2],
                    attentionResultBinding->slices[pair * 2 + 1],
                    static_cast<std::size_t>(
                        attentionResultBinding->base_row)
                        + (column / 64) * kSeqLen + row,
                    column % 32, attentionResultBinding->bank);
            }
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
    if (outputProjectionReplicaMaxError > 0.04f)
        throw std::logic_error(
            "O projection hemisphere replication mismatch max_error="
            + std::to_string(outputProjectionReplicaMaxError)
            + " hemisphere="
            + std::to_string(outputProjectionReplicaMaxHemisphere)
            + " row=" + std::to_string(outputProjectionReplicaMaxRow)
            + " column="
            + std::to_string(outputProjectionReplicaMaxColumn)
            + " actual="
            + std::to_string(outputProjectionReplicaMaxActual));

    const std::size_t attentionResidualEndCycle =
        static_cast<std::size_t>(
            timeline(program, "rmsnorm.feedback", 1).start_cycle);
    runtime.run_cycles(
        attentionResidualEndCycle - outputProjectionEndCycle,
        cmodelLogSink);
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

    const std::size_t ffnStartCycle = std::min({
        firstBindingReadCycle(directProgram, gateBinding),
        firstBindingReadCycle(directProgram, upBinding),
        firstBindingReadCycle(directProgram, downBinding),
    });
    if (ffnStartCycle <= attentionResidualEndCycle
        || ffnStartCycle > program.max_cycle)
        throw std::logic_error(
            "FFN first read is outside the compiled execution window");
    runtime.run_cycles(
        ffnStartCycle - attentionResidualEndCycle - 1, cmodelLogSink);
    // Paged executables reuse the resident weight bank as scratch between
    // stages. Model the C2C handoff by making the FFN page resident immediately
    // before its first binding read instead of uploading every page at cycle 0.
    runtime.upload_input(gateBinding, gateWeight);
    runtime.upload_input(upBinding, upWeight);
    runtime.upload_input(downBinding, downWeight);
    runtime.run_cycles(1, cmodelLogSink);
    const auto rms2Binding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "rmsnorm.result.1";
        });
    if (rms2Binding == program.bindings.end()
        || rms2Binding->layout
            != ftlpu::software::runtime::BindingLayout::
                Fp16MxmDistributed16
        || rms2Binding->slices.size() != 16)
        throw std::logic_error(
            "cannot locate second RMSNorm physical binding");
    const std::size_t rms2BaseRow =
        static_cast<std::size_t>(rms2Binding->base_row);
    const float rms2Value = readMxmDistributed(
        rms2Binding->slices, rms2BaseRow, 0, 0,
        ftlpu::Hemisphere::East, rms2Binding->bank);
    if (!std::isfinite(rms2Value) || rms2Value == 0.0f)
        throw std::logic_error(
            "RMS2 stage produced invalid data value="
            + std::to_string(rms2Value)
            + " input=" + std::to_string(readMxmDistributed(
                residualSlices,
                static_cast<std::size_t>(residualBinding->base_row), 0, 0)));
    std::vector<float> rms2Output(kSeqLen * kHidden);
    const auto expectedRms2Output = rmsNorm(residual1, 1);
    float rms2CheckpointMaxError = 0.0f;
    std::size_t rms2CheckpointMaxIndex = 0;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const std::size_t index = row * kHidden + column;
            rms2Output[index] = readMxmDistributed(
                rms2Binding->slices, rms2BaseRow, row, column,
                ftlpu::Hemisphere::East, rms2Binding->bank);
            const float error = std::fabs(
                rms2Output[index] - expectedRms2Output[index]);
            if (error > rms2CheckpointMaxError) {
                rms2CheckpointMaxError = error;
                rms2CheckpointMaxIndex = index;
            }
        }
    }
    if (rms2CheckpointMaxError > 0.04f)
        throw std::logic_error(
            "second RMSNorm checkpoint mismatch max_error="
            + std::to_string(rms2CheckpointMaxError)
            + " row="
            + std::to_string(rms2CheckpointMaxIndex / kHidden)
            + " column="
            + std::to_string(rms2CheckpointMaxIndex % kHidden)
            + " actual="
            + std::to_string(rms2Output[rms2CheckpointMaxIndex])
            + " expected="
            + std::to_string(expectedRms2Output[rms2CheckpointMaxIndex]));
    std::cout << FTLPU_TEST_MODEL_NAME
              << " decoder checkpoint passed: attention + RMS2; "
              << "starting paged FFN" << std::endl;
    const std::size_t finalResidualStartCycle =
        static_cast<std::size_t>(
            timeline(program, "elementwise.add", 1).start_cycle);
    runtime.run_cycles(
        finalResidualStartCycle - ffnStartCycle, cmodelLogSink);
    const auto ffnBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "ffn.result";
        });
    if (ffnBinding == program.bindings.end())
        throw std::logic_error(
            "cannot locate FFN down-projection physical binding");
    const auto hiddenValue = [&](std::size_t row, std::size_t h) {
        const float gate =
            rms2Output[row * kHidden + gateK(h)] * gateSign(h);
        const float up =
            rms2Output[row * kHidden + upK(h)] * upSign(h);
        return bf16(gate * (1.0f / (1.0f + std::exp(-gate))) * up);
    };
    std::vector<float> ffnOutput(kSeqLen * kHidden);
    float ffnCheckpointMaxError = 0.0f;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const std::size_t h0 =
                (column * 5 + 17) % kIntermediate;
            const std::size_t h1 = (h0 + 37) % kIntermediate;
            const float expected =
                bf16(hiddenValue(row, h0) - hiddenValue(row, h1));
            const std::size_t outputBlock = column / 32;
            float ownerValue = 0.0f;
            float replicaValue = 0.0f;
            auto owner = ftlpu::Hemisphere::East;
            auto replica = ftlpu::Hemisphere::West;
            if (ffnBinding->layout
                == ftlpu::software::runtime::BindingLayout::
                    Fp16MxmBlock8Distributed16) {
                if (ffnBinding->slices.size() != 16)
                    throw std::logic_error(
                        "FFN Block8 result requires 16 slices");
                owner = static_cast<ftlpu::Hemisphere>(outputBlock % 2);
                replica = static_cast<ftlpu::Hemisphere>(1 - outputBlock % 2);
                ownerValue = readMxmDistributed(
                    ffnBinding->slices,
                    static_cast<std::size_t>(ffnBinding->base_row),
                    row, column, owner, ffnBinding->bank);
                replicaValue = readMxmDistributed(
                    ffnBinding->slices,
                    static_cast<std::size_t>(ffnBinding->base_row),
                    row, column, replica, ffnBinding->bank);
            } else if (ffnBinding->layout
                == ftlpu::software::runtime::BindingLayout::Fp16PairPlanar) {
                if (ffnBinding->slices.size() < 4)
                    throw std::logic_error(
                        "FFN vector result requires four planar slices");
                owner = static_cast<ftlpu::Hemisphere>(
                    (outputBlock % 4) / 2);
                replica = static_cast<ftlpu::Hemisphere>(1
                    - static_cast<std::size_t>(owner));
                const std::size_t pair = outputBlock % 2;
                const std::size_t address =
                    static_cast<std::size_t>(ffnBinding->base_row)
                    + (outputBlock / 4) * kSeqLen + row;
                ownerValue = physicalBf16(owner,
                    ffnBinding->slices[pair * 2],
                    ffnBinding->slices[pair * 2 + 1], address,
                    column % 32, ffnBinding->bank);
                // Vector down partitions each four-block wave across the two
                // hemispheres. The opposite side owns a different output
                // block at the same local address; it is not a replica.
                replicaValue = expected;
            } else {
                throw std::logic_error(
                    "FFN result has an unsupported physical layout");
            }
            ffnOutput[row * kHidden + column] = ownerValue;
            const float ownerError = std::fabs(ownerValue - expected);
            const float replicaError = std::fabs(replicaValue - expected);
            ffnCheckpointMaxError = std::max(
                ffnCheckpointMaxError, std::max(ownerError, replicaError));
            if (ownerError > 0.04f || replicaError > 0.04f)
                throw std::logic_error(
                    "FFN down checkpoint mismatch row="
                    + std::to_string(row)
                    + " column=" + std::to_string(column)
                    + " output_block=" + std::to_string(outputBlock)
                    + " owner="
                    + std::to_string(static_cast<std::size_t>(owner))
                    + " owner_actual=" + std::to_string(ownerValue)
                    + " replica_actual=" + std::to_string(replicaValue)
                    + " expected=" + std::to_string(expected)
                    + " h0=" + std::to_string(h0)
                    + " h1=" + std::to_string(h1)
                    + " bank="
                    + std::to_string(ffnBinding->bank)
                    + " base_row="
                    + std::to_string(ffnBinding->base_row));
        }
    }
    const float preFinalResidualSample = readMxmDistributed(
        residualSlices,
        static_cast<std::size_t>(residualBinding->base_row),
        kSeqLen - 1, 0);
    const float expectedResidualSample = residual1[(kSeqLen - 1) * kHidden];
    if (preFinalResidualSample != expectedResidualSample)
        throw std::logic_error(
            "FFN overwrote live attention residual before final add"
            " actual=" + std::to_string(preFinalResidualSample)
            + " expected=" + std::to_string(expectedResidualSample));
    runtime.run_cycles(
        program.max_cycle + kCheckpointDrainCycles
        - finalResidualStartCycle);
    if (memTracePath != nullptr)
        runtime.write_mem_execution_trace_csv(memTracePath);
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
    float hostAttentionContextMaxError = 0.0f;
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
            const std::size_t index = query * kHidden + column;
            hostAttentionContextMaxError = std::max(
                hostAttentionContextMaxError,
                std::fabs(attentionContext[index]
                    - checkpointContext[index]));
            // QK is accumulated in 32x32 hardware blocks, so its FP32 add
            // order differs from the linear host loop above. Use the actual
            // BF16 probability SRAM golden for strict downstream checks.
            attentionContext[index] = checkpointContext[index];
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
    if (hostAttentionContextMaxError > 0.1f)
        throw std::logic_error(
            "host/hardware attention context drift max_error="
            + std::to_string(hostAttentionContextMaxError));
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
    std::size_t maxUlpDistance = 0;
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
            const std::size_t index = row * kHidden + column;
            const float cpuExpected = bf16(
                residual1[row * kHidden + column] + ffn);
            const float stageExpected = bf16(
                residual1[index] + ffnOutput[index]);
            const float observed = readBf16(actual, index);
            const float stageError = std::fabs(observed - stageExpected);
            const float cpuError = std::fabs(observed - cpuExpected);
            maxError = std::max(maxError, cpuError);
            maxUlpDistance = std::max(
                maxUlpDistance,
                bf16UlpDistance(observed, cpuExpected));
            if (std::fabs(observed) > 1.0e-4f) ++nonzero;
            if (stageError != 0.0f) {
                throw std::logic_error(
                    "final residual mismatch row=" + std::to_string(row)
                    + " column=" + std::to_string(column)
                    + " actual=" + std::to_string(observed)
                    + " expected=" + std::to_string(stageExpected)
                    + " down.actual=" + std::to_string(ffnOutput[index])
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
#if FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES
    using namespace ftlpu::software::runtime;
    if (program.weight_page_uses.size() != 20)
        throw std::logic_error(
            "Qwen dynamic C2C test expected 20 executable weight-page uses, got "
            + std::to_string(program.weight_page_uses.size()));

    const auto inputBindingByIndex = [&](std::uint32_t index)
        -> const BinaryBinding& {
        const auto binding = std::ranges::find_if(
            program.bindings, [&](const BinaryBinding& candidate) {
                return candidate.access == BindingAccess::Input
                    && candidate.index == index;
            });
        if (binding == program.bindings.end())
            throw std::logic_error(
                "Qwen dynamic C2C test is missing input binding "
                + std::to_string(index));
        return *binding;
    };
    const auto logicalWeight = [&](std::uint32_t index)
        -> const std::vector<std::uint8_t>& {
        switch (index) {
        case 2: return queryWeight;
        case 3: return keyWeight;
        case 4: return valueWeight;
        case 5: return outputWeight;
        default:
            if (index == gateBinding) return gateWeight;
            if (index == upBinding) return upWeight;
            if (index == downBinding) return downWeight;
            throw std::logic_error(
                "Qwen dynamic C2C page references a non-matrix binding "
                + std::to_string(index));
        }
    };
    std::size_t expectedPagedBytes = 0;
    std::vector<std::size_t> syncInstructionsPerUse(
        program.weight_page_uses.size());
    std::vector<std::size_t> synchronizedWritesPerUse(
        program.weight_page_uses.size());
    for (std::size_t useIndex = 0;
         useIndex < program.weight_page_uses.size(); ++useIndex) {
        const BinaryWeightPageUse& use =
            program.weight_page_uses[useIndex];
        const auto image = pack_weight_binding_page(
            inputBindingByIndex(use.binding_index), use.page_index,
            logicalWeight(use.binding_index), program.hardware);
        for (const PackedWeightSegment& segment : image.segments) {
            expectedPagedBytes += static_cast<std::size_t>(
                segment.vector_count) * ftlpu::hw::kPhysicalVectorBytes;
            ++syncInstructionsPerUse[useIndex];
            synchronizedWritesPerUse[useIndex] += segment.vector_count;
        }
    }
    const auto expectedPlans = plan_weight_prefetches(program);
    std::size_t expectedPrefetches = expectedPlans.size();
    std::size_t expectedSyncInstructions = 0;
    std::size_t expectedSynchronizedWrites = 0;
    for (const WeightPrefetchPlan& plan : expectedPlans) {
        if (plan.pre_execution)
            continue;
        for (const std::size_t useIndex : plan.use_indices) {
            expectedSyncInstructions += syncInstructionsPerUse.at(useIndex);
            expectedSynchronizedWrites +=
                synchronizedWritesPerUse.at(useIndex);
        }
    }
    if (expectedPrefetches == 0 || expectedPagedBytes == 0
        || expectedSyncInstructions == 0)
        throw std::logic_error(
            "Qwen dynamic C2C page plan is unexpectedly empty");

    ModelPackage dynamicPackage;
    dynamicPackage.model_name = FTLPU_TEST_MODEL_NAME;
    dynamicPackage.architecture = "Qwen2ForCausalLM";
    const auto appendTensor = [&](std::uint32_t bindingIndex,
                                  std::string name,
                                  const std::vector<std::uint8_t>& data,
                                  bool quantized = false) {
        const BinaryBinding& binding = inputBindingByIndex(bindingIndex);
        ModelTensor tensor;
        tensor.name = std::move(name);
        tensor.element_type = binding.element_type;
        tensor.shape = binding.shape;
        tensor.data = data;
        if (quantized) {
            tensor.encoding = ModelTensorEncoding::SymmetricPerTensorI8;
            tensor.scales = {1.0f};
        }
        dynamicPackage.tensors.push_back(std::move(tensor));
    };
    ModelWeightPage parameterPage;
    parameterPage.layer = 0;
    parameterPage.bank = inputBindingByIndex(1).bank;
    std::uint16_t nextParameterStream = 0;
    const auto appendPackedParameter = [&] (
        std::uint32_t bindingIndex, std::string name,
        const std::vector<std::uint8_t>& logical) {
        const BinaryBinding& binding = inputBindingByIndex(bindingIndex);
        if (binding.bank != parameterPage.bank)
            throw std::logic_error(
                "Qwen parameter page spans multiple MEM banks");
        PackedWeightImage image =
            pack_binding_image(binding, logical, program.hardware);
        const std::string tensorName = name;
        dynamicPackage.tensors.push_back(ModelTensor{
            std::move(name), BindingElementType::I8,
            {static_cast<std::uint64_t>(image.data.size())},
            std::move(image.data),
            ModelTensorEncoding::TargetPackedSramVectors});
        parameterPage.tensors.push_back(tensorName);
        for (const PackedWeightSegment& segment : image.segments) {
            parameterPage.segments.push_back(ModelWeightPage::Segment{
                tensorName, segment.byte_offset, segment.hemisphere,
                segment.slice, segment.base_row, segment.vector_count,
                nextParameterStream});
            expectedPagedBytes += static_cast<std::size_t>(
                segment.vector_count) * ftlpu::hw::kPhysicalVectorBytes;
            nextParameterStream = static_cast<std::uint16_t>(
                (nextParameterStream + 1)
                % program.hardware.c2c_streams_per_direction);
        }
    };
    appendPackedParameter(1, "input_layernorm.weight", gamma0);
    if (hasAttentionBias) {
        appendPackedParameter(6, "self_attn.q_proj.bias", queryBias);
        appendPackedParameter(7, "self_attn.k_proj.bias", keyBias);
        appendPackedParameter(8, "self_attn.v_proj.bias", valueBias);
    }
    appendPackedParameter(
        static_cast<std::uint32_t>(postAttentionNormBinding),
        "post_attention_layernorm.weight", gamma1);
    dynamicPackage.weight_pages.push_back(std::move(parameterPage));
    ++expectedPrefetches;

    appendTensor(2, "self_attn.q_proj.weight", queryWeight, true);
    appendTensor(3, "self_attn.k_proj.weight", keyWeight, true);
    appendTensor(4, "self_attn.v_proj.weight", valueWeight, true);
    appendTensor(5, "self_attn.o_proj.weight", outputWeight, true);
    appendTensor(static_cast<std::uint32_t>(gateBinding),
        "mlp.gate_proj.weight", gateWeight, true);
    appendTensor(static_cast<std::uint32_t>(upBinding),
        "mlp.up_proj.weight", upWeight, true);
    appendTensor(static_cast<std::uint32_t>(downBinding),
        "mlp.down_proj.weight", downWeight, true);

    const BinaryBinding& dynamicInput = inputBindingByIndex(0);
    const auto dynamicOutput = std::ranges::find_if(
        program.bindings, [](const BinaryBinding& binding) {
            return binding.access == BindingAccess::Output
                && binding.index == 0;
        });
    if (dynamicOutput == program.bindings.end())
        throw std::logic_error(
            "Qwen dynamic C2C test is missing output binding 0");
    dynamicPackage.values = {
        {"hidden.0", dynamicInput.element_type, dynamicInput.shape,
         true, false},
        {"hidden.1", dynamicOutput->element_type, dynamicOutput->shape,
         false, true},
    };

    std::vector<ModelStateBindingRef> stateRefs;
    for (const BinaryBinding& binding : program.bindings) {
        ModelStateKind kind{};
        std::string name;
        if (binding.access != BindingAccess::Internal)
            continue;
        if (binding.role == "state.kv.key") {
            kind = ModelStateKind::KvKey;
            name = "layers.0.key_cache";
        } else if (binding.role == "state.kv.value") {
            kind = ModelStateKind::KvValue;
            name = "layers.0.value_cache";
        } else {
            continue;
        }
        auto logicalShape = binding.shape;
        if (logicalShape.empty())
            throw std::logic_error(
                "Qwen dynamic C2C state binding has an empty shape");
        const auto residentTokens = static_cast<std::uint32_t>(
            logicalShape.front());
        const auto capacity = std::max<std::uint32_t>(256, residentTokens);
        logicalShape.front() = capacity;
        dynamicPackage.states.push_back(ModelState{
            name, kind, binding.element_type, std::move(logicalShape), 0,
            capacity, program.hardware.mxm_rows, residentTokens});
        stateRefs.push_back({binding.index, std::move(name)});
    }

    dynamicPackage.executables.push_back(
        {"decoder.layer0", program, {}});
    ModelInvocation dynamicInvocation;
    dynamicInvocation.name = "decoder.layer0";
    dynamicInvocation.executable_index = 0;
    dynamicInvocation.inputs = {
        {0, "hidden.0"},
        {1, "input_layernorm.weight"},
        {2, "self_attn.q_proj.weight"},
        {3, "self_attn.k_proj.weight"},
        {4, "self_attn.v_proj.weight"},
        {5, "self_attn.o_proj.weight"},
    };
    if (hasAttentionBias) {
        dynamicInvocation.inputs.push_back(
            {6, "self_attn.q_proj.bias"});
        dynamicInvocation.inputs.push_back(
            {7, "self_attn.k_proj.bias"});
        dynamicInvocation.inputs.push_back(
            {8, "self_attn.v_proj.bias"});
    }
    dynamicInvocation.inputs.push_back({
        static_cast<std::uint32_t>(postAttentionNormBinding),
        "post_attention_layernorm.weight"});
    dynamicInvocation.inputs.push_back({
        static_cast<std::uint32_t>(gateBinding), "mlp.gate_proj.weight"});
    dynamicInvocation.inputs.push_back({
        static_cast<std::uint32_t>(upBinding), "mlp.up_proj.weight"});
    dynamicInvocation.inputs.push_back({
        static_cast<std::uint32_t>(downBinding), "mlp.down_proj.weight"});
    dynamicInvocation.outputs = {{0, "hidden.1"}};
    dynamicInvocation.states = std::move(stateRefs);
    dynamicInvocation.weight_page = 0;
    dynamicPackage.invocations.push_back(std::move(dynamicInvocation));

    // The direct run has completed and `actual` owns its downloaded output.
    // Release its large chip model before constructing the C2C-backed one.
    runtimeOwner.reset();
    systemOwner.reset();
    system = nullptr;
    ftlpu::C2cDmaSystem dynamicSystem;
    ModelSession dynamicSession(dynamicSystem);
    std::uint32_t runtimeDdrBandwidth =
        program.hardware.ddr_peak_bandwidth_mbytes_per_second;
    const bool hasRuntimeDdrOverride =
        std::getenv("FTLPU_DDR_BANDWIDTH_MBPS") != nullptr;
    if (const char* bandwidth =
            std::getenv("FTLPU_DDR_BANDWIDTH_MBPS")) {
        const auto parsed = std::stoull(bandwidth);
        if (parsed == 0
            || parsed > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "invalid FTLPU_DDR_BANDWIDTH_MBPS");
        runtimeDdrBandwidth = static_cast<std::uint32_t>(parsed);
        dynamicSession.set_ddr_peak_bandwidth_mbytes_per_second(
            runtimeDdrBandwidth);
    }
    const char* dynamicPipelineTracePath =
        std::getenv("FTLPU_QWEN_C2C_PIPELINE_CSV");
    if (dynamicPipelineTracePath != nullptr)
        dynamicSession.enable_execution_trace();
    if (const char* dynamicMemTrace =
            std::getenv("FTLPU_QWEN_C2C_MEM_CSV"))
        dynamicSession.stream_mem_execution_trace_csv(dynamicMemTrace);
    dynamicSession.load(std::move(dynamicPackage));
    dynamicSession.set_input("hidden.0", input);
    try {
        dynamicSession.run(kCheckpointDrainCycles);
    } catch (...) {
        const auto executionError = std::current_exception();
        try {
            if (const char* linkedPath =
                    std::getenv("FTLPU_QWEN_C2C_LINKED_BINARY"))
                dynamicSession.write_last_linked_program(linkedPath);
        } catch (...) {
            // Preserve the execution error when linking did not get far enough
            // to publish a diagnostic image.
        }
        try {
            if (const char* preExecutionDirectory =
                    std::getenv("FTLPU_QWEN_C2C_PRE_EXECUTION_DIR"))
                dynamicSession.write_pre_execution_icu_programs(
                    preExecutionDirectory);
        } catch (...) {
            // Diagnostic export must not hide the runtime failure.
        }
        std::rethrow_exception(executionError);
    }
    if (dynamicPipelineTracePath != nullptr)
        dynamicSession.write_execution_trace_csv(dynamicPipelineTracePath);
    const auto& dynamicActual = dynamicSession.value("hidden.1");
    if (dynamicActual.size() != actual.size())
        throw std::logic_error(
            "Qwen dynamic C2C output byte size differs from direct golden: "
            + std::to_string(dynamicActual.size()) + " vs "
            + std::to_string(actual.size()));
    if (dynamicActual != actual) {
        std::size_t firstMismatch = 0;
        while (firstMismatch < actual.size()
            && dynamicActual[firstMismatch] == actual[firstMismatch])
            ++firstMismatch;
        throw std::logic_error(
            "Qwen dynamic C2C output differs from direct golden at byte "
            + std::to_string(firstMismatch));
    }
    const ModelSessionStats& dynamicStats = dynamicSession.stats();
    if (dynamicStats.weight_page_prefetches != expectedPrefetches
        || dynamicStats.weight_page_prefetch_bytes != expectedPagedBytes)
        throw std::logic_error(
            "Qwen dynamic C2C transfer accounting mismatch: prefetches="
            + std::to_string(dynamicStats.weight_page_prefetches)
            + " expected_prefetches=" + std::to_string(expectedPrefetches)
            + " bytes="
            + std::to_string(dynamicStats.weight_page_prefetch_bytes)
            + " expected_bytes=" + std::to_string(expectedPagedBytes));
    if (!hasRuntimeDdrOverride && runtimeDdrBandwidth >= 51'200
        && dynamicStats.weight_page_runtime_wait_cycles != 0)
        throw std::logic_error(
            "Qwen dynamic C2C execution stalled after program start: "
            + std::to_string(dynamicStats.weight_page_runtime_wait_cycles)
            + " cycles");
    const BinaryProgram& linkedProgram =
        dynamicSession.last_linked_program();
    std::array<bool, ftlpu::InstructionControlUnit::kMemQueues>
        seenLinkedMemQueue{};
    std::size_t linkedWriteSyncInstructions = 0;
    std::size_t linkedReadSyncInstructions = 0;
    for (const QueueProgram& queue : linkedProgram.queues) {
        if (queue.kind != QueueKind::Mem)
            continue;
        if (queue.index >= seenLinkedMemQueue.size()
            || seenLinkedMemQueue[queue.index])
            throw std::logic_error(
                "Qwen dynamic linker produced a duplicate MEM ICU queue");
        seenLinkedMemQueue[queue.index] = true;
        for (std::size_t pc = 0; pc < queue.commands.size();) {
            if (is_mem_synchronized_raw_packet_header(
                    queue.commands[pc])) {
                const auto packet =
                    decode_mem_synchronized_icu_packet(queue, pc);
                const auto encoded =
                    ((static_cast<ftlpu::isa::EncodedMemInstruction>(
                          packet[1].lanes[0])
                         | (static_cast<ftlpu::isa::EncodedMemInstruction>(
                                packet[1].lanes[1])
                             << 32))
                        >> 2)
                    | (static_cast<ftlpu::isa::EncodedMemInstruction>(
                           packet[1].lanes[2])
                        << 62);
                if (ftlpu::isa::decode_mem_instruction(encoded).opcode
                    == ftlpu::MemOpcode::Read)
                    ++linkedReadSyncInstructions;
                else
                    ++linkedWriteSyncInstructions;
                pc += ftlpu::InstructionControlUnit::MemIcu::
                          synchronized_packet_word_count;
                continue;
            }
            if (is_fu_3d_raw_packet_header(queue.commands[pc])) {
                if (is_mem_write_read_2d_raw_packet_header(
                        queue.commands[pc]))
                    static_cast<void>(
                        decode_mem_write_read_2d_raw_packet(queue, pc));
                else
                    static_cast<void>(decode_fu_3d_raw_packet_loop(
                        queue, pc));
                pc += fu_3d_raw_packet_word_count(queue.kind);
                continue;
            }
            ++pc;
        }
    }
    if (linkedWriteSyncInstructions != expectedSyncInstructions)
        throw std::logic_error(
            "Qwen dynamic linker MEM_WRITE_SYNC count mismatch: linked="
            + std::to_string(linkedWriteSyncInstructions)
            + " expected=" + std::to_string(expectedSyncInstructions));
    if (linkedReadSyncInstructions == 0)
        throw std::logic_error(
            "Qwen compiler image contains no MEM_READ_SYNC for KV page-out");
    if (dynamicStats.weight_page_synchronized_writes
        != expectedSynchronizedWrites)
        throw std::logic_error(
            "Qwen dynamic C2C synchronized MEM FU write mismatch: issued="
            + std::to_string(
                dynamicStats.weight_page_synchronized_writes)
            + " expected=" + std::to_string(expectedSynchronizedWrites));
    if (const char* linkedPath =
            std::getenv("FTLPU_QWEN_C2C_LINKED_BINARY"))
        dynamicSession.write_last_linked_program(linkedPath);
    if (const char* preExecutionDirectory =
            std::getenv("FTLPU_QWEN_C2C_PRE_EXECUTION_DIR"))
        dynamicSession.write_pre_execution_icu_programs(
            preExecutionDirectory);
    std::cout << "Qwen dynamic C2C pages passed: uses="
              << program.weight_page_uses.size()
              << ", prefetches=" << dynamicStats.weight_page_prefetches
              << ", bytes=" << dynamicStats.weight_page_prefetch_bytes
              << ", linked_mem_write_sync=" << linkedWriteSyncInstructions
              << ", linked_mem_read_sync=" << linkedReadSyncInstructions
              << ", synchronized_fu_writes="
              << dynamicStats.weight_page_synchronized_writes
              << ", initial_wait_cycles="
              << dynamicStats.weight_page_initial_wait_cycles
              << ", runtime_wait_cycles="
              << dynamicStats.weight_page_runtime_wait_cycles
              << ", compiled_ddr_mbytes="
              << program.hardware.ddr_peak_bandwidth_mbytes_per_second
              << ", runtime_ddr_mbytes=" << runtimeDdrBandwidth << '\n';
#endif
    std::cout << "Complete " FTLPU_TEST_MODEL_NAME
                 " decoder layer passed: "
              << kSeqLen * kHidden << " BF16 values, nonzero="
              << nonzero << ", max_error=" << maxError
              << ", max_ulp_distance=" << maxUlpDistance
              << ", ffn_checkpoint_max_error="
              << ffnCheckpointMaxError
              << ", host_attention_context_max_error="
              << hostAttentionContextMaxError
              << ", attention_residual_max_error="
              << attentionResidualMaxError
              << ", rms2_max_error=" << rms2MaxError
              << ", max_cycle=" << program.max_cycle << '\n';
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "compiled_decoder_layer_runtime_test failed: "
              << ex.what() << '\n';
    return 1;
}
