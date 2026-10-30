#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/schedule_trace.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"
#include "ftlpu/vxm/special_alu.hpp"

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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
constexpr std::size_t kMxmBlockRows = 8;
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

template <typename Fn>
std::vector<ftlpu::VxmLutEntry> makeVxmLut(
    float inputMin, float segmentWidth, std::size_t count, Fn fn)
{
    std::vector<ftlpu::VxmLutEntry> entries;
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float x0 = inputMin
            + static_cast<float>(index) * segmentWidth;
        const float y0 = fn(x0);
        entries.push_back(ftlpu::VxmLutEntry::from_float(
            (fn(x0 + segmentWidth) - y0) / segmentWidth, y0));
    }
    return entries;
}

ftlpu::VxmSpecialAlu makeReferenceVxmSpecialAlu()
{
    constexpr std::size_t entries = 256;
    constexpr float ln2 = 0.6931471805599453f;
    ftlpu::VxmSpecialAlu alu;
    const float expWidth = ln2 / static_cast<float>(entries);
    alu.configure_lut(ftlpu::VxmSpecialAluOpcode::Exp,
        {-ln2 / 2.0f, expWidth},
        makeVxmLut(-ln2 / 2.0f, expWidth, entries,
            [](float x) { return std::exp(x); }));
    const float reciprocalWidth = 1.0f / static_cast<float>(entries);
    alu.configure_lut(ftlpu::VxmSpecialAluOpcode::Reciprocal,
        {1.0f, reciprocalWidth},
        makeVxmLut(1.0f, reciprocalWidth, entries,
            [](float x) { return 1.0f / x; }));
    const float rsqrtWidth = 3.0f / static_cast<float>(entries);
    alu.configure_lut(ftlpu::VxmSpecialAluOpcode::Rsqrt,
        {1.0f, rsqrtWidth},
        makeVxmLut(1.0f, rsqrtWidth, entries,
            [](float x) { return 1.0f / std::sqrt(x); }));
    return alu;
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
                const std::size_t packetWords =
                    ftlpu::software::runtime::fu_3d_raw_packet_word_count(queue->kind, command);
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
    if (argc != 2 && !(FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES && argc == 3))
        throw std::runtime_error(
            "usage: compiled_decoder_layer_runtime_test baseline.ftlpu [raw-program.ftlpu]");
    const bool skipNumericChecks =
        std::getenv("FTLPU_SKIP_QWEN_NUMERIC_CHECK") != nullptr;
    std::size_t decoderLayerCount = 1;
    if (const char* layers = std::getenv("FTLPU_QWEN_DECODER_LAYERS")) {
        decoderLayerCount = std::stoull(layers);
        if (decoderLayerCount == 0 || decoderLayerCount > 2)
            throw std::invalid_argument(
                "FTLPU_QWEN_DECODER_LAYERS must be 1 or 2");
    }
    if (decoderLayerCount > 1 && !skipNumericChecks)
        throw std::invalid_argument(
            "two-layer Qwen scheduling runs require "
            "FTLPU_SKIP_QWEN_NUMERIC_CHECK=1");
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
    if (const char* path = std::getenv("FTLPU_QWEN_DIRECT_BINARY"))
        ftlpu::software::runtime::write_binary_program(directProgram,
            std::filesystem::path(path));
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

#if FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES
    using namespace ftlpu::software::runtime;
    auto dynamicProgram = argc == 3
        ? read_binary_program(std::filesystem::path(argv[2])) : program;
    if (argc == 3) {
        // Raw compilation can charge explicit drain intervals before safe
        // operator boundaries. Check the retiming exactly, rather than
        // accepting arbitrary phase changes or requiring those costs to vanish.
        auto costs = dynamicProgram.timelines;
        std::erase_if(costs, [](const auto& value) { return value.name != "common_barrier.drain_cost" && value.name != "common_barrier.schedule_cost"; });
        std::ranges::sort(costs, {}, &BinaryTimeline::start_cycle);
        const auto& h = dynamicProgram.hardware;
        const std::uint64_t guard = h.system_register_columns + h.mem_boundary_register_columns
            + h.mxm_pipeline_rows + h.tile_rows
            + std::max({h.mem_to_sxm_latency, h.mem_to_mxm_latency, h.mxm0_accumulator_latency,
                h.mxm1_accumulator_latency, h.accumulator_to_vxm_latency, h.accumulator_read_to_vxm_latency,
                h.swiglu_write_latency, h.qk_iw_to_compute_latency, h.vxm_weight_to_iw_latency});
        std::uint64_t previousEnd = 0;
        for (const auto& cost : costs) {
            if (cost.start_cycle < previousEnd || cost.end_cycle <= cost.start_cycle
                || cost.end_cycle - cost.start_cycle != (cost.name == "common_barrier.schedule_cost" ? 2 * guard : guard) || cost.end_cycle > dynamicProgram.max_cycle)
                throw std::logic_error("Raw decoder has an invalid charged drain interval");
            previousEnd = cost.end_cycle;
        }
        const auto baselineCycle = [&](std::uint64_t cycle) {
            auto result = cycle;
            for (const auto& cost : costs) {
                if (cost.start_cycle < cycle && cycle < cost.end_cycle)
                    throw std::logic_error("Raw decoder places a consumer inside its charged drain interval");
                if (cost.end_cycle <= cycle) result -= cost.end_cycle - cost.start_cycle;
            }
            return result;
        };
        if (dynamicProgram.queue_control_protocol && !costs.empty())
            throw std::logic_error("QCTL changed Native scheduling with global barrier padding");
        if ((!dynamicProgram.common_barrier_protocol && !dynamicProgram.queue_control_protocol)
            || baselineCycle(dynamicProgram.max_cycle) != program.max_cycle
            || dynamicProgram.bindings.size() != program.bindings.size()
            || dynamicProgram.weight_page_uses.size() != program.weight_page_uses.size())
            throw std::logic_error("Raw decoder does not preserve the baseline paging contract");
        for (std::size_t index = 0; index < program.bindings.size(); ++index) {
            const auto& before = program.bindings[index];
            const auto& after = dynamicProgram.bindings[index];
            if (before.index != after.index || before.access != after.access
                || before.element_type != after.element_type || before.layout != after.layout
                || before.base_row != after.base_row || before.bank != after.bank
                || before.shape != after.shape || before.slices != after.slices
                || before.page_count != after.page_count || before.paged_weight != after.paged_weight)
                throw std::logic_error("Raw decoder changed a baseline binding placement");
        }
        for (std::size_t index = 0; index < program.weight_page_uses.size(); ++index) {
            const auto& before = program.weight_page_uses[index];
            const auto& after = dynamicProgram.weight_page_uses[index];
            if (before.binding_index != after.binding_index || before.page_index != after.page_index
                || before.bank != after.bank || before.ready_cycle != baselineCycle(after.ready_cycle)
                || before.release_cycle != baselineCycle(after.release_cycle))
                throw std::logic_error("Raw decoder changed a baseline page transport instance");
        }
        const bool rawMacro = std::ranges::any_of(dynamicProgram.queues,
            [](const auto& queue) { return queue.requires_raw_macro; });
        for (const auto& queue : dynamicProgram.queues)
            if (!queue.commands.empty() && (queue.kind == QueueKind::Mem
                    || queue.kind == QueueKind::MxmLoad || queue.kind == QueueKind::MxmCompute
                    || queue.kind == QueueKind::MxmDequant)
                && rawMacro && !queue.requires_raw_macro)
                throw std::logic_error("Raw decoder silently retained a non-Macro computation queue");
    }
    // Identical computation/safe-stop schedule; only disable early launch for
    // the measured serial control. The production linker never reads this knob.
    if (std::getenv("FTLPU_TEST_SERIAL_PAGING")) dynamicProgram.transport_overlap_protocol = false;
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
    // Both native layouts use the same executable page plan and C2C package.
    const auto prepareDynamic = [&]() {
        const std::size_t declaredExecutableWeightPages =
            std::accumulate(dynamicProgram.bindings.begin(), dynamicProgram.bindings.end(),
                std::size_t {0}, [](std::size_t total,
                    const BinaryBinding& binding) {
                    return total + (binding.access == BindingAccess::Input
                            && binding.paged_weight
                        ? binding.page_count : 0);
                });
        if (dynamicProgram.weight_page_uses.size()
            != declaredExecutableWeightPages)
            throw std::logic_error(
                "Qwen dynamic C2C test expected one executable use per declared "
                "weight page: declared="
                + std::to_string(declaredExecutableWeightPages)
                + ", uses="
                + std::to_string(dynamicProgram.weight_page_uses.size()));

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
        std::size_t expectedExecutablePagedBytes = 0;
        std::vector<std::size_t> syncInstructionsPerUse(
            dynamicProgram.weight_page_uses.size());
        std::vector<std::size_t> synchronizedWritesPerUse(
            dynamicProgram.weight_page_uses.size());
        for (std::size_t useIndex = 0;
             useIndex < dynamicProgram.weight_page_uses.size(); ++useIndex) {
            const BinaryWeightPageUse& use =
                dynamicProgram.weight_page_uses[useIndex];
            const auto image = pack_weight_binding_page(
                inputBindingByIndex(use.binding_index), use.page_index,
                logicalWeight(use.binding_index), dynamicProgram.hardware);
            for (const PackedWeightSegment& segment : image.segments) {
                expectedExecutablePagedBytes += static_cast<std::size_t>(
                    segment.vector_count) * ftlpu::hw::kPhysicalVectorBytes;
                ++syncInstructionsPerUse[useIndex];
                synchronizedWritesPerUse[useIndex] += segment.vector_count;
            }
        }
        const auto expectedPlans = plan_weight_prefetches(dynamicProgram);
        const std::size_t expectedExecutablePrefetches = expectedPlans.size();
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
        if (expectedExecutablePrefetches == 0
            || expectedExecutablePagedBytes == 0
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
        std::size_t expectedParameterPagedBytes = 0;
        std::uint16_t nextParameterStream = 0;
        const auto appendPackedParameter = [&] (
            std::uint32_t bindingIndex, std::string name,
            const std::vector<std::uint8_t>& logical) {
            const BinaryBinding& binding = inputBindingByIndex(bindingIndex);
            if (binding.bank != parameterPage.bank) {
                // A ModelWeightPage has one destination bank. Parameters in
                // another bank use ModelSession's ordinary C2C ingress.
                dynamicPackage.values.push_back(ModelValue{
                    std::move(name), binding.element_type, binding.shape, true, false});
                return;
            }
            PackedWeightImage image =
                pack_binding_image(binding, logical, dynamicProgram.hardware);
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
                expectedParameterPagedBytes += static_cast<std::size_t>(
                    segment.vector_count) * ftlpu::hw::kPhysicalVectorBytes;
                nextParameterStream = static_cast<std::uint16_t>(
                    (nextParameterStream + 1)
                    % dynamicProgram.hardware.c2c_streams_per_direction);
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
        const std::size_t expectedPrefetches = 1
            + decoderLayerCount * expectedExecutablePrefetches;
        const std::size_t expectedPagedBytes = expectedParameterPagedBytes
            + decoderLayerCount * expectedExecutablePagedBytes;

        const auto layerTensorName = [](std::size_t layer,
                                        std::string_view suffix) {
            return "layers." + std::to_string(layer) + "." +
                std::string(suffix);
        };
        for (std::size_t layer = 0; layer < decoderLayerCount; ++layer) {
            appendTensor(2, layerTensorName(layer, "self_attn.q_proj.weight"),
                queryWeight, true);
            appendTensor(3, layerTensorName(layer, "self_attn.k_proj.weight"),
                keyWeight, true);
            appendTensor(4, layerTensorName(layer, "self_attn.v_proj.weight"),
                valueWeight, true);
            appendTensor(5, layerTensorName(layer, "self_attn.o_proj.weight"),
                outputWeight, true);
            appendTensor(static_cast<std::uint32_t>(gateBinding),
                layerTensorName(layer, "mlp.gate_proj.weight"), gateWeight,
                true);
            appendTensor(static_cast<std::uint32_t>(upBinding),
                layerTensorName(layer, "mlp.up_proj.weight"), upWeight, true);
            appendTensor(static_cast<std::uint32_t>(downBinding),
                layerTensorName(layer, "mlp.down_proj.weight"), downWeight,
                true);
        }

        const BinaryBinding& dynamicInput = inputBindingByIndex(0);
        const auto dynamicOutput = std::ranges::find_if(
            dynamicProgram.bindings, [](const BinaryBinding& binding) {
                return binding.access == BindingAccess::Output
                    && binding.index == 0;
            });
        if (dynamicOutput == dynamicProgram.bindings.end())
            throw std::logic_error(
                "Qwen dynamic C2C test is missing output binding 0");
        for (std::size_t layer = 0; layer <= decoderLayerCount; ++layer) {
            dynamicPackage.values.push_back(ModelValue{
                "hidden." + std::to_string(layer),
                layer == 0 ? dynamicInput.element_type
                           : dynamicOutput->element_type,
                layer == 0 ? dynamicInput.shape : dynamicOutput->shape,
                layer == 0, layer == decoderLayerCount});
        }

        dynamicPackage.executables.push_back({"decoder.layer", dynamicProgram, {}});
        for (std::size_t layer = 0; layer < decoderLayerCount; ++layer) {
            std::vector<ModelStateBindingRef> stateRefs;
            for (const BinaryBinding& binding : dynamicProgram.bindings) {
                ModelStateKind kind{};
                std::string suffix;
                if (binding.access != BindingAccess::Internal)
                    continue;
                if (binding.role == "state.kv.key") {
                    kind = ModelStateKind::KvKey;
                    suffix = "key_cache";
                } else if (binding.role == "state.kv.value") {
                    kind = ModelStateKind::KvValue;
                    suffix = "value_cache";
                } else {
                    continue;
                }
                auto logicalShape = binding.shape;
                if (logicalShape.empty())
                    throw std::logic_error(
                        "Qwen dynamic C2C state binding has an empty shape");
                const auto residentTokens = static_cast<std::uint32_t>(
                    logicalShape.front());
                const auto capacity = std::max<std::uint32_t>(
                    256, residentTokens);
                logicalShape.front() = capacity;
                std::string name = "layers." + std::to_string(layer) + "." +
                    suffix;
                dynamicPackage.states.push_back(ModelState{
                    name, kind, binding.element_type, std::move(logicalShape), 0,
                    capacity, dynamicProgram.hardware.mxm_rows, residentTokens});
                stateRefs.push_back({binding.index, std::move(name)});
            }

            ModelInvocation dynamicInvocation;
            dynamicInvocation.name = "decoder.layer" + std::to_string(layer);
            dynamicInvocation.executable_index = 0;
            dynamicInvocation.inputs = {
                {0, "hidden." + std::to_string(layer)},
                {1, "input_layernorm.weight"},
                {2, layerTensorName(layer, "self_attn.q_proj.weight")},
                {3, layerTensorName(layer, "self_attn.k_proj.weight")},
                {4, layerTensorName(layer, "self_attn.v_proj.weight")},
                {5, layerTensorName(layer, "self_attn.o_proj.weight")},
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
                static_cast<std::uint32_t>(gateBinding),
                layerTensorName(layer, "mlp.gate_proj.weight")});
            dynamicInvocation.inputs.push_back({
                static_cast<std::uint32_t>(upBinding),
                layerTensorName(layer, "mlp.up_proj.weight")});
            dynamicInvocation.inputs.push_back({
                static_cast<std::uint32_t>(downBinding),
                layerTensorName(layer, "mlp.down_proj.weight")});
            dynamicInvocation.outputs = {
                {0, "hidden." + std::to_string(layer + 1)}};
            dynamicInvocation.states = std::move(stateRefs);
            dynamicInvocation.weight_page = 0;
            dynamicPackage.invocations.push_back(std::move(dynamicInvocation));
        }

        return std::tuple{std::move(dynamicPackage), expectedPrefetches,
            expectedPagedBytes, expectedSyncInstructions,
            expectedSynchronizedWrites};
    };
    const auto verifyDynamic = [&](ModelSession& dynamicSession,
        std::size_t expectedPrefetches, std::size_t expectedPagedBytes,
        std::size_t expectedSyncInstructions,
        std::size_t expectedSynchronizedWrites,
        std::uint32_t runtimeDdrBandwidth) {
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
        // Page-ready waits are legal even at high DDR bandwidth: bank reuse and
        // consumer epochs also constrain prefetch. Report them as performance
        // metrics below, rather than treating a bandwidth threshold as a proof
        // of zero waits. ModelSession's completion watchdog detects deadlock;
        // the exact output comparison above detects incorrect resumption.
        const BinaryProgram& linkedProgram =
            dynamicSession.last_linked_program();
        if(const auto path=std::getenv("FTLPU_QUEUE_PROGRESS_CSV")) {
            std::ofstream stream(path);
            if(!stream)throw std::runtime_error("cannot write queue progress report");
            stream<<"invocation,kind,index,local_cycle,queue_cycle,physical_cycles,wait_cycles,issued,done\n";
            for(const auto& q:dynamicStats.queue_progress)
                stream<<q.invocation_index<<','<<static_cast<unsigned>(q.kind)<<','<<q.index<<','
                      <<q.local_cycle<<','<<q.queue_cycle<<','<<q.physical_cycles<<','<<q.wait_cycles<<','<<q.issued<<','<<q.done<<'\n';
        }
        std::size_t linkedWriteSyncInstructions = 0;
        std::size_t linkedReadSyncInstructions = 0;
        const auto inspectLinkedQueues = [&](const std::vector<QueueProgram>& queues) {
          std::array<bool, ftlpu::InstructionControlUnit::kMemQueues> seenLinkedMemQueue{};
          for (const QueueProgram& queue : queues) {
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
                    pc += fu_3d_raw_packet_word_count(queue.kind, queue.commands[pc]);
                    continue;
                }
                ++pc;
            }
        }
        };
        inspectLinkedQueues(linkedProgram.queues);
        inspectLinkedQueues(linkedProgram.transport_queues);
        if (argc == 3 && !linkedProgram.queue_control_protocol && (linkedProgram.barriers.empty()
                || linkedProgram.transport_queues.empty()))
            throw std::logic_error("Raw decoder did not execute a linked common barrier and MEM transport image");
        if (linkedProgram.queue_control_protocol) {
            validate_queue_sync_plan(linkedProgram);
            if (linkedProgram.common_barrier_protocol || linkedProgram.queue_sync_plan.partitions.empty()
                || linkedProgram.queue_sync_plan.events.empty() || linkedProgram.queue_link_source.required)
                throw std::logic_error("decoder did not load a completed queue-control image");
            if (argc==3 && linkedProgram.transport_queues.empty())
                throw std::logic_error("Raw decoder did not execute the Native MEM transport store");
            // KV page-out is a distinct executable. Its compiler controls
            // remain in the compact relink recipe, never in the main image.
            const auto mainWrites=linkedWriteSyncInstructions;
            inspectLinkedQueues(linkedProgram.queue_link_source.queues);
            linkedWriteSyncInstructions=mainWrites;
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
            != decoderLayerCount * expectedSynchronizedWrites)
            throw std::logic_error(
                "Qwen dynamic C2C synchronized MEM FU write mismatch: issued="
                + std::to_string(
                    dynamicStats.weight_page_synchronized_writes)
                + " expected=" + std::to_string(
                    decoderLayerCount * expectedSynchronizedWrites));
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
                  << (linkedProgram.queue_control_protocol ? ", completion_tail_cycles=" : ", runtime_wait_cycles=")
                  << dynamicStats.weight_page_runtime_wait_cycles
                  << ", compiled_ddr_mbytes="
                  << program.hardware.ddr_peak_bandwidth_mbytes_per_second
                  << ", runtime_ddr_mbytes=" << runtimeDdrBandwidth
                  << (linkedProgram.queue_control_protocol ? ", queue_execution_ticks=" : ", L=") << dynamicStats.main_logical_cycles
                  << ", P=" << dynamicStats.main_physical_cycles
                  << ", PRIME=" << dynamicStats.instruction_prime_cycles
                  << ", compute_dma_overlap=" << dynamicStats.compute_dma_overlap_cycles
                  << ", compute_rx_overlap=" << dynamicStats.compute_rx_overlap_cycles
                  << ", compute_commit_overlap=" << dynamicStats.compute_commit_overlap_cycles
                  << ", commit_segments=" << dynamicStats.transport_commit_segments
                  << ", queue_control=" << linkedProgram.queue_control_protocol
                  << ", qctl_partitions=" << linkedProgram.queue_sync_plan.partitions.size()
                  << ", common_barriers=" << linkedProgram.barriers.size()
                  << ", barrier_table_bytes=" << (16 + 8 * linkedProgram.barriers.size())
                  << '\n';
    };
#endif

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
    // The later legacy checkpoints encode the LPU32 32x8x4 layout. LPU64
    // has separate stage readers for its split-bank 64x8x8 layout, followed
    // by an independent whole-layer BF16/LUT reference comparison.
    if (program.hardware.mxm_rows == 64) {
#if FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES
        if (decoderLayerCount != 1)
            throw std::invalid_argument("LPU64 numerical acceptance requires one layer");
        auto [dynamicPackage, expectedPrefetches, expectedPagedBytes,
            expectedSyncInstructions, expectedSynchronizedWrites] = prepareDynamic();
#endif
        const bool dynamicOnly = std::getenv("FTLPU_LPU64_DYNAMIC_ONLY") != nullptr;
#if !FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES
        if (dynamicOnly) throw std::invalid_argument("dynamic C2C is not enabled in this target");
#endif
        std::ofstream scopedLog;
        std::size_t logStart = 0, logEnd = 0;
        if (const char* logPath = std::getenv("FTLPU_CMODEL_LOG")) {
            logStart = std::stoull(
                std::getenv("FTLPU_CMODEL_LOG_START") != nullptr
                    ? std::getenv("FTLPU_CMODEL_LOG_START") : "0");
            const auto logCycles = std::stoull(
                std::getenv("FTLPU_CMODEL_LOG_CYCLES") != nullptr
                    ? std::getenv("FTLPU_CMODEL_LOG_CYCLES") : "64");
            logEnd = logStart + logCycles;
            scopedLog.open(logPath);
            if (!scopedLog) throw std::runtime_error("cannot open CModel log");
        }
        const bool reportProgress = std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
        const auto advanceTo = [&](std::size_t end) {
            if (dynamicOnly) return;
            while (runtime.logical_cycles() < end) {
                const auto start = runtime.logical_cycles();
                const bool logging = scopedLog.is_open()
                    && start >= logStart && start < logEnd;
                auto stop = logging ? std::min(end, logEnd)
                    : start < logStart ? std::min(end, logStart) : end;
                if (reportProgress)
                    stop = std::min(stop, start + 25000);
                runtime.run_cycles(stop - start, logging ? &scopedLog : nullptr);
                if (reportProgress)
                    std::clog << "LPU64 direct execution cycle=" << stop
                              << "/" << program.max_cycle + 64 << std::endl;
            }
        };
        const auto checkStage = [&](const char* name,
                                    const std::vector<float>& expected,
                                    std::size_t readyCycle) {
            if (dynamicOnly) return;
            advanceTo(readyCycle);
            const auto binding = std::find_if(program.bindings.begin(),
                program.bindings.end(), [&](const auto& b) {
                    return b.access == ftlpu::software::runtime::BindingAccess::Internal
                        && b.name == name;
                });
            if (binding == program.bindings.end())
                throw std::logic_error(std::string("missing stage binding: ") + name);
            std::vector<std::uint8_t> bytes;
            using ftlpu::software::runtime::BindingLayout;
            const auto readByte = [&](std::size_t hemisphere, std::size_t slice,
                                      std::size_t row, std::size_t lane,
                                      int bank = -1) {
                return system->read_mem_sram_lane_byte(
                    static_cast<ftlpu::Hemisphere>(hemisphere), slice,
                    bank < 0 ? binding->bank : bank,
                    lane / program.hardware.lanes_per_tile,
                    row, lane % program.hardware.lanes_per_tile);
            };
            if (binding->layout == BindingLayout::Fp16HeadBlockPacked) {
                const auto width = program.hardware.bytes_per_word;
                bytes.resize(binding->byte_size);
                for (std::size_t token = 0; token < kSeqLen; ++token)
                    for (std::size_t head = 0; head < kQueryHeads; ++head)
                        for (std::size_t d = 0; d < kHeadDim; ++d) {
                            const auto hemisphere = (head / (kQueryHeads / kKvHeads)) % 2;
                            const bool splitBanks =
                                program.hardware.mxms_per_hemisphere == 2
                                && kHeadDim / width == 2;
                            const auto row = binding->base_row
                                + (splitBanks ? head
                                    : head * (kHeadDim / width) + d / width) * kSeqLen + token;
                            const auto index = (token * kHidden + head * kHeadDim + d) * 2;
                            for (std::size_t byte = 0; byte < 2; ++byte)
                                bytes[index + byte] = readByte(hemisphere,
                                    binding->slices[2 * hemisphere + byte], row, d % width,
                                    splitBanks
                                        ? (binding->bank + d / width)
                                            % program.hardware.banks_per_slice
                                        : binding->bank);
                        }
            } else if (binding->layout == BindingLayout::Fp16ProbabilityDiagonal) {
                bytes.resize(binding->byte_size);
                const auto width = program.hardware.bytes_per_word;
                const auto lanes = program.hardware.lanes_per_tile;
                for (std::size_t head = 0; head < kQueryHeads; ++head)
                    for (std::size_t query = 0; query < kSeqLen; ++query)
                        for (std::size_t key = 0; key < kSeqLen; ++key) {
                            const auto row = binding->base_row
                                + ((head * (kSeqLen / width) + query / width)
                                        * (kSeqLen / width) + key / width)
                                    * program.hardware.tile_rows
                                + (query % width) / lanes;
                            const auto index = (head * kSeqLen * kSeqLen
                                + query * kSeqLen + key) * 2;
                            for (std::size_t byte = 0; byte < 2; ++byte)
                                bytes[index + byte] = readByte(
                                    (head / (kQueryHeads / kKvHeads)) % 2,
                                    binding->slices[(query % lanes) * 2 + byte],
                                    row, key % width);
                        }
            } else if (binding->layout == BindingLayout::Fp16HeadPlanar
                || binding->layout == BindingLayout::Fp16ValueX16) {
                std::vector<std::uint8_t> zero(binding->byte_size);
                auto image = ftlpu::software::runtime::pack_binding_image(
                    *binding, zero, program.hardware);
                for (const auto& segment : image.segments)
                    for (std::size_t row = 0; row < segment.vector_count; ++row) {
                        int bank = binding->bank;
                        // The dual-MXM attention fixture stores the two V
                        // feature blocks in alternating banks.
                        if (binding->layout == BindingLayout::Fp16ValueX16
                            && program.hardware.mxms_per_hemisphere == 2) {
                            const auto block = ((segment.base_row + row - binding->base_row)
                                / program.hardware.tile_rows
                                / (kSeqLen / program.hardware.mxm_rows))
                                % (kHeadDim / program.hardware.bytes_per_word);
                            if (block % 2) bank = (bank + 1) % program.hardware.banks_per_slice;
                        }
                        for (std::size_t lane = 0; lane < program.hardware.bytes_per_word; ++lane)
                            image.data[segment.byte_offset
                                + row * program.hardware.bytes_per_word + lane] =
                                readByte(segment.hemisphere, segment.slice,
                                    segment.base_row + row, lane, bank);
                    }
                bytes = ftlpu::software::runtime::unpack_binding_image(
                    *binding, image, program.hardware);
            } else {
                bytes = runtime.download_binding(*binding);
            }
            std::string replicaFailure;
            if (binding->layout == BindingLayout::Fp16MxmDistributed16
                && binding->hemisphere_mask == 3) {
                for (std::size_t row = 0; row < kSeqLen; ++row)
                    for (std::size_t c = 0; c < kHidden; ++c) {
                        const auto address = binding->base_row
                            + ((row / program.hardware.mxm_rows)
                                    * (kHidden / program.hardware.bytes_per_word)
                                + c / program.hardware.bytes_per_word) * program.hardware.tile_rows
                            + (row % program.hardware.mxm_rows) / program.hardware.lanes_per_tile;
                        const auto slice = 2 * (row % program.hardware.lanes_per_tile);
                        for (std::size_t byte = 0; byte < 2; ++byte)
                            if (readByte(1, binding->slices[slice + byte], address,
                                    c % program.hardware.bytes_per_word)
                                != bytes[(row * kHidden + c) * 2 + byte])
                                if (replicaFailure.empty())
                                    replicaFailure = std::string("hemisphere replication mismatch: ")
                                        + name + " row=" + std::to_string(row)
                                        + " column=" + std::to_string(c);
                    }
            }
            if (bytes.size() != expected.size() * 2)
                throw std::logic_error(std::string("stage size mismatch: ") + name);
            double squared = 0;
            float maximum = 0;
            std::size_t worst = 0;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const float observed = readBf16(bytes, i);
                if (!std::isfinite(observed))
                    throw std::logic_error(std::string("non-finite stage output: ") + name);
                const float error = std::fabs(observed - expected[i]);
                squared += static_cast<double>(error) * error;
                if (error > maximum) { maximum = error; worst = i; }
            }
            const auto rmse = std::sqrt(squared / expected.size());
            std::cout << "LPU64 stage=" << name << " cycle=" << readyCycle
                      << " max_error=" << maximum << " rmse=" << rmse
                      << " index=" << worst << " observed=" << readBf16(bytes, worst)
                      << " expected=" << expected[worst] << std::endl;
            if (maximum > 0.1f || rmse > 0.01 || !replicaFailure.empty()) {
                if (const char* directory = std::getenv("FTLPU_NUMERIC_DUMP_DIR")) {
                    std::filesystem::create_directories(directory);
                    const auto base = std::filesystem::path(directory) / name;
                    std::ofstream observedFile(base.string() + ".actual.bf16", std::ios::binary);
                    observedFile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    std::ofstream expectedFile(base.string() + ".expected.f32", std::ios::binary);
                    expectedFile.write(reinterpret_cast<const char*>(expected.data()),
                        expected.size() * sizeof(float));
                }
                throw std::logic_error(replicaFailure.empty()
                    ? std::string("LPU64 stage numeric mismatch: ") + name
                    : replicaFailure);
            }
        };
        const auto specialReference = makeReferenceVxmSpecialAlu();
        const auto rmsNormSr = [&](const std::vector<float>& values, std::size_t stage) {
            std::vector<float> normalized(values.size());
            for (std::size_t row = 0; row < kSeqLen; ++row) {
                float sum = 0.0f;
                for (std::size_t c = 0; c < kHidden; ++c)
                    sum += values[row * kHidden + c] * values[row * kHidden + c];
                // The SR reduction spills a BF16 sum; mean and Rsqrt stay
                // FP32, as does x*gamma before multiplying the local factor.
                const float factor = specialReference.execute(ftlpu::VxmSpecialAluOpcode::Rsqrt,
                    bf16(sum) * (1.0f / static_cast<float>(kHidden)) + kEpsilon);
                for (std::size_t c = 0; c < kHidden; ++c)
                    normalized[row * kHidden + c] = bf16(
                        (values[row * kHidden + c] * gammaValue(c, stage)) * factor);
            }
            return normalized;
        };
        const auto normalized0 = rmsNormSr(inputValues, 0);
        checkStage("rmsnorm.result.0", normalized0,
            timeline(program, "qkv").start_cycle);
        std::vector<float> queryProjection(kSeqLen * kHidden, 0.0f);
        std::vector<float> keyProjection(kSeqLen * kHidden, 0.0f);
        std::vector<float> valueProjection(keyProjection.size(), 0.0f);
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
        std::vector<float> keyCache(kSeqLen * kKvHeads * kHeadDim);
        std::vector<float> valueCache(keyCache.size());
        for (std::size_t token = 0; token < kSeqLen; ++token)
            for (std::size_t head = 0; head < kKvHeads; ++head)
                for (std::size_t d = 0; d < kHeadDim; ++d) {
                    const auto index = (token * kKvHeads + head) * kHeadDim + d;
                    keyCache[index] = ropeValue(keyProjection, token, head, d);
                    valueCache[index] = valueProjection[token * kHidden + head * kHeadDim + d];
                }
        checkStage("attention.value_cache", valueCache, timeline(program, "qk").start_cycle);
        checkStage("attention.key_cache", keyCache, timeline(program, "qk").start_cycle);
        const auto referenceExp = [&](float value) {
            return specialReference.execute(ftlpu::VxmSpecialAluOpcode::Exp, value);
        };
        const auto referenceReciprocal = [&](float value) {
            return specialReference.execute(ftlpu::VxmSpecialAluOpcode::Reciprocal, value);
        };
        // Model the declared BF16 SRAM boundaries and configured LUT math.
        // No observed intermediate result is fed into this reference.
        std::cout << "LPU64 reference=bf16_boundaries_configured_lut" << std::endl;
        std::vector<float> context(kSeqLen * kHidden, 0.0f);
        std::vector<float> expectedProbabilities(kQueryHeads * kSeqLen * kSeqLen, 0.0f);
        std::vector<float> scores(kSeqLen);
        std::vector<float> probabilities(kSeqLen);
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t head = 0; head < kQueryHeads; ++head) {
                const std::size_t kvHead = head / (kQueryHeads / kKvHeads);
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t key = 0; key <= token; ++key) {
                    float score = 0.0f;
                    for (std::size_t dimension = 0;
                         dimension < kHeadDim; ++dimension)
                        score += ropeValue(queryProjection, token, head,
                                     dimension)
                            * ropeValue(keyProjection, key, kvHead,
                                dimension);
                    score = bf16(bf16(score)
                        * (1.0f / std::sqrt(static_cast<float>(kHeadDim))));
                    scores[key] = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0.0f;
                for (std::size_t key = 0; key <= token; ++key) {
                    probabilities[key] = referenceExp(scores[key] - maximum);
                    denominator += probabilities[key];
                }
                const float inverseDenominator = referenceReciprocal(denominator);
                for (std::size_t key = 0; key <= token; ++key)
                    expectedProbabilities[(head * kSeqLen + token) * kSeqLen + key]
                        = bf16(probabilities[key] * inverseDenominator);
                for (std::size_t dimension = 0;
                     dimension < kHeadDim; ++dimension) {
                    float sum = 0.0f;
                    for (std::size_t key = 0; key <= token; ++key)
                        sum += bf16(probabilities[key] * inverseDenominator)
                            * valueProjection[
                                key * kHidden + kvHead * kHeadDim
                                + dimension];
                    context[token * kHidden + head * kHeadDim + dimension]
                        = bf16(sum);
                }
            }
        }
        checkStage("attention.probability_diagonal", expectedProbabilities,
            timeline(program, "o_proj").start_cycle);
        checkStage("attention.context", context,
            timeline(program, "o_proj").start_cycle);
        std::vector<float> residual(kSeqLen * kHidden);
        std::vector<float> projectedValues(kSeqLen * kHidden);
        for (std::size_t token = 0; token < kSeqLen; ++token)
            for (std::size_t column = 0; column < kHidden; ++column) {
                float projected = 0.0f;
                for (std::size_t block = 0; block < kHidden / 32; ++block) {
                    const auto source = block * 32
                        + (column * 7 + block * 3) % 32;
                    projected += context[token * kHidden + source]
                        * (((column + block) & 1) ? -1.0f : 1.0f);
                }
                const auto index = token * kHidden + column;
                projectedValues[index] = bf16(projected);
                residual[index] = bf16(inputValues[index]
                    + bf16(projected));
            }
        checkStage("attention.result", projectedValues,
            timeline(program, "elementwise.add", 0).start_cycle);
        if (!dynamicOnly && std::any_of(program.timelines.begin(), program.timelines.end(),
                [](const auto& t) { return t.name == "rmsnorm.materialize_gamma"; })) {
            advanceTo(timeline(program, "rmsnorm.materialize_gamma").start_cycle);
            const auto gammaBinding = std::find_if(program.bindings.begin(),
                program.bindings.end(), [&](const auto& binding) {
                    return binding.access == ftlpu::software::runtime::BindingAccess::Input
                        && binding.index == postAttentionNormBinding;
                });
            for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere)
                for (std::size_t column = 0; column < kHidden; ++column) {
                    const auto segment = column / gammaBinding->instruction_count;
                    const auto row = gammaBinding->base_row + column % gammaBinding->instruction_count;
                    std::uint16_t bits = 0;
                    for (std::size_t byte = 0; byte < 2; ++byte)
                        bits |= static_cast<std::uint16_t>(system->read_mem_sram_lane_byte(
                            static_cast<ftlpu::Hemisphere>(hemisphere),
                            gammaBinding->slices[2 * segment + byte],
                            gammaBinding->bank, 0, row, 0)) << (8 * byte);
                    if (ftlpu::Bf16::from_bits(bits).to_float() != gammaValue(column, 1))
                        throw std::logic_error("resident gamma overwritten column=" + std::to_string(column));
                }
            std::cout << "LPU64 resident gamma exact PASS" << std::endl;
        }
        checkStage("elementwise.add.0", residual,
            timeline(program, "rmsnorm.transpose_input", 1).start_cycle);
        if (!dynamicOnly && std::any_of(program.timelines.begin(), program.timelines.end(),
                [](const auto& t) { return t.name == "rmsnorm.materialize_gamma"; })) {
            for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere)
                for (std::size_t column = 0; column < kHidden; ++column)
                    for (std::size_t lane = 0; lane < program.hardware.bytes_per_word; ++lane) {
                        const auto row = program.hardware.sram_depth_rows - kHidden + column;
                        std::uint16_t bits = 0;
                        for (std::size_t byte = 0; byte < 2; ++byte)
                            bits |= static_cast<std::uint16_t>(system->read_mem_sram_lane_byte(
                                static_cast<ftlpu::Hemisphere>(hemisphere), 16 + byte, 0,
                                lane / program.hardware.lanes_per_tile, row,
                                lane % program.hardware.lanes_per_tile)) << (8 * byte);
                        const float observed = ftlpu::Bf16::from_bits(bits).to_float();
                        if (observed != gammaValue(column, 1))
                            throw std::logic_error("materialized gamma mismatch hemisphere="
                                + std::to_string(hemisphere) + " column=" + std::to_string(column)
                                + " lane=" + std::to_string(lane) + " observed=" + std::to_string(observed)
                                + " expected=" + std::to_string(gammaValue(column, 1)));
                    }
            std::cout << "LPU64 materialized gamma exact PASS" << std::endl;
        }
        const auto normalized1 = rmsNormSr(residual, 1);
        checkStage("rmsnorm.result.1", normalized1,
            timeline(program, "rmsnorm.restore_layout", 1).end_cycle + 8);
        std::uint64_t projectionReady = 0;
        for (const auto& use : program.weight_page_uses)
            if (use.binding_index == gateBinding || use.binding_index == upBinding)
                projectionReady = std::max(projectionReady, use.release_cycle);
        // Weight release precedes the final compute/result drain. Two full
        // tiles cover that drain while the tail SwiGLU scratch copies have
        // not yet reclaimed any original Gate rows.
        advanceTo(projectionReady + 2 * program.hardware.mxm_rows);
        for (std::size_t projection = 0; projection < (dynamicOnly ? 0u : 2u); ++projection)
            for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                std::vector<float> observed(kSeqLen * kIntermediate);
                std::vector<float> expected(observed.size());
                float maximum = 0.0f;
                std::size_t worst = 0;
                for (std::size_t token = 0; token < kSeqLen; ++token)
                    for (std::size_t h = 0; h < kIntermediate; ++h) {
                        const auto block = h / program.hardware.bytes_per_word;
                        const auto pairsPerGroup = program.hardware.sram_depth_rows / kSeqLen;
                        const auto slice = projection * 10 + (block / pairsPerGroup) * 2;
                        const auto row = (block % pairsPerGroup) * kSeqLen + token;
                        const auto lane = h % program.hardware.bytes_per_word;
                        std::uint16_t bits = 0;
                        for (std::size_t byte = 0; byte < 2; ++byte)
                            bits |= static_cast<std::uint16_t>(system->read_mem_sram_lane_byte(
                                static_cast<ftlpu::Hemisphere>(hemisphere), slice + byte, 0,
                                lane / program.hardware.lanes_per_tile, row,
                                lane % program.hardware.lanes_per_tile)) << (8 * byte);
                        const auto index = token * kIntermediate + h;
                        observed[index] = ftlpu::Bf16::from_bits(bits).to_float();
                        if (!std::isfinite(observed[index]))
                            throw std::logic_error("non-finite LPU64 FFN projection");
                        expected[index] = bf16(normalized1[token * kHidden
                                + (projection == 0 ? gateK(h) : upK(h))]
                            * (projection == 0 ? gateSign(h) : upSign(h)));
                        const auto error = std::fabs(observed[index] - expected[index]);
                        if (error > maximum) { maximum = error; worst = index; }
                    }
                const auto name = std::string(projection == 0 ? "ffn.gate." : "ffn.up.")
                    + std::to_string(hemisphere);
                std::cout << "LPU64 stage=" << name << " max_error=" << maximum
                          << " token=" << worst / kIntermediate << " feature=" << worst % kIntermediate
                          << " observed=" << observed[worst] << " expected=" << expected[worst] << std::endl;
                if (!std::isfinite(maximum) || maximum > 0.01f) {
                    if (const char* directory = std::getenv("FTLPU_NUMERIC_DUMP_DIR")) {
                        for (const auto& output : {std::pair{".actual.f32", &observed},
                                 std::pair{".expected.f32", &expected}}) {
                            std::ofstream file(std::filesystem::path(directory) / (name + output.first),
                                std::ios::binary);
                            file.write(reinterpret_cast<const char*>(output.second->data()),
                                output.second->size() * sizeof(float));
                        }
                    }
                    throw std::logic_error("LPU64 projection numeric mismatch: " + name);
                }
            }
        std::vector<float> expectedHidden(kSeqLen * kIntermediate);
        for (std::size_t token = 0; token < kSeqLen; ++token)
            for (std::size_t h = 0; h < kIntermediate; ++h) {
                const float gate = bf16(normalized1[token * kHidden + gateK(h)]
                    * gateSign(h));
                const float up = bf16(normalized1[token * kHidden + upK(h)]
                    * upSign(h));
                expectedHidden[token * kIntermediate + h] = bf16(gate
                    * referenceReciprocal(1.0f + referenceExp(-gate)) * up);
            }
        advanceTo(timeline(program, "ffn.down.vector").start_cycle);
        const auto lanesPerTile = program.hardware.lanes_per_tile;
        const auto vectorWidth = program.hardware.bytes_per_word;
        const auto tokenGroups = kSeqLen / lanesPerTile;
        // The fixture reserves the input's distributed rows first, followed
        // by the hidden tensor in the same bank.
        const auto hiddenBase = (kHidden / vectorWidth) * tokenGroups;
        for (std::size_t hemisphere = 0; hemisphere < (dynamicOnly ? 0u : 2u); ++hemisphere) {
            std::vector<float> observed(expectedHidden.size());
            float maximum = 0.0f;
            std::size_t worst = 0;
            for (std::size_t token = 0; token < kSeqLen; ++token)
                for (std::size_t h = 0; h < kIntermediate; ++h) {
                    const auto slice = 2 * (token % lanesPerTile);
                    const auto row = hiddenBase + (h / vectorWidth) * tokenGroups
                        + token / lanesPerTile;
                    const auto lane = h % vectorWidth;
                    std::uint16_t bits = 0;
                    for (std::size_t byte = 0; byte < 2; ++byte)
                        bits |= static_cast<std::uint16_t>(system->read_mem_sram_lane_byte(
                            static_cast<ftlpu::Hemisphere>(hemisphere), slice + byte, 1,
                            lane / lanesPerTile, row, lane % lanesPerTile)) << (8 * byte);
                    const auto index = token * kIntermediate + h;
                    observed[index] = ftlpu::Bf16::from_bits(bits).to_float();
                    if (!std::isfinite(observed[index]))
                        throw std::logic_error("non-finite LPU64 SwiGLU output");
                    const float error = std::fabs(observed[index] - expectedHidden[index]);
                    if (error > maximum) { maximum = error; worst = index; }
                }
            const auto name = "ffn.hidden." + std::to_string(hemisphere);
            std::cout << "LPU64 stage=" << name << " max_error=" << maximum
                      << " token=" << worst / kIntermediate << " feature=" << worst % kIntermediate
                      << " observed=" << observed[worst] << " expected=" << expectedHidden[worst]
                      << std::endl;
            if (maximum > 0.01f) {
                if (const char* directory = std::getenv("FTLPU_NUMERIC_DUMP_DIR"))
                    for (const auto& output : {std::pair{".actual.f32", &observed},
                             std::pair{".expected.f32", &expectedHidden}}) {
                        std::ofstream file(std::filesystem::path(directory) / (name + output.first),
                            std::ios::binary);
                        file.write(reinterpret_cast<const char*>(output.second->data()),
                            output.second->size() * sizeof(float));
                    }
                throw std::logic_error("LPU64 SwiGLU numeric mismatch: " + name);
            }
        }
        std::vector<float> expectedDown(kSeqLen * kHidden);
        for (std::size_t token = 0; token < kSeqLen; ++token)
            for (std::size_t column = 0; column < kHidden; ++column) {
                const std::size_t h0 = (column * 5 + 17) % kIntermediate;
                const std::size_t h1 = (h0 + 37) % kIntermediate;
                expectedDown[token * kHidden + column] = bf16(
                    expectedHidden[token * kIntermediate + h0]
                    - expectedHidden[token * kIntermediate + h1]);
            }
        const auto finalAddStart = timeline(program, "elementwise.add", 1).start_cycle;
        checkStage("ffn.result", expectedDown, finalAddStart);
        checkStage("elementwise.add.0.persistent", residual, finalAddStart);
        advanceTo(program.max_cycle + 64);
        std::vector<std::uint8_t> referenceOutput;
        for (std::size_t i = 0; i < residual.size(); ++i)
            appendBf16(referenceOutput, bf16(residual[i] + expectedDown[i]));
        std::cout << "LPU64 independent CPU reference prepared: BF16 values="
                  << referenceOutput.size() / 2 << std::endl;
        if (!dynamicOnly) {
        const auto actual = runtime.download_output(0);
        double squaredError = 0.0;
        float maxError = 0.0f;
        std::size_t maxIndex = 0;
        for (std::size_t token = 0; token < kSeqLen; ++token) {
            for (std::size_t column = 0; column < kHidden; ++column) {
                const std::size_t h0 = (column * 5 + 17) % kIntermediate;
                const std::size_t h1 = (h0 + 37) % kIntermediate;
                const auto index = token * kHidden + column;
                const float expected = bf16(residual[index]
                    + bf16(expectedHidden[token * kIntermediate + h0]
                        - expectedHidden[token * kIntermediate + h1]));
                const float observed = readBf16(actual, index);
                if (!std::isfinite(observed) || !std::isfinite(expected))
                    throw std::logic_error("non-finite LPU64 decoder output");
                const float error = std::fabs(observed - expected);
                squaredError += static_cast<double>(error) * error;
                if (error > maxError) {
                    maxError = error;
                    maxIndex = index;
                }
            }
        }
        const double rmse = std::sqrt(squaredError / (kSeqLen * kHidden));
        std::cout << "LPU64 decoder output max_error=" << maxError
                  << " rmse=" << rmse << " row=" << maxIndex / kHidden
                  << " column=" << maxIndex % kHidden << '\n';
        if (!std::isfinite(maxError) || !std::isfinite(rmse)
            || maxError > 0.1f || rmse > 0.01)
            throw std::logic_error("LPU64 decoder output numeric mismatch");
        }
#if FTLPU_TEST_DYNAMIC_C2C_WEIGHT_PAGES
        runtimeOwner.reset();
        systemOwner.reset();
        system = nullptr;
        auto dynamicSystem = std::make_unique<ftlpu::C2cDmaSystem>();
        ModelSession dynamicSession(*dynamicSystem);
        auto runtimeDdrBandwidth = program.hardware.ddr_peak_bandwidth_mbytes_per_second;
        if (const char* bandwidth = std::getenv("FTLPU_DDR_BANDWIDTH_MBPS")) {
            const auto parsed = std::stoull(bandwidth);
            if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("invalid FTLPU_DDR_BANDWIDTH_MBPS");
            runtimeDdrBandwidth = static_cast<std::uint32_t>(parsed);
            dynamicSession.set_ddr_peak_bandwidth_mbytes_per_second(runtimeDdrBandwidth);
        }
        const char* tracePath = std::getenv("FTLPU_QWEN_C2C_PIPELINE_CSV");
        if (tracePath) dynamicSession.enable_execution_trace();
        if (const char* memPath = std::getenv("FTLPU_QWEN_C2C_MEM_CSV"))
            dynamicSession.stream_mem_execution_trace_csv(memPath);
        dynamicSession.load(std::move(dynamicPackage));
        dynamicSession.set_input("hidden.0", input);
        if (inputBindingByIndex(static_cast<std::uint32_t>(postAttentionNormBinding)).bank
            != inputBindingByIndex(1).bank)
            dynamicSession.set_input("post_attention_layernorm.weight", gamma1);
        std::cout << "LPU64 dynamic C2C start: DDR_MBps=" << runtimeDdrBandwidth
                  << " pages=" << expectedPrefetches << std::endl;
        try {
            dynamicSession.run(64);
        } catch (...) {
            const auto error = std::current_exception();
            try {
                if (tracePath) dynamicSession.write_execution_trace_csv(tracePath);
                if (const char* path = std::getenv("FTLPU_QWEN_C2C_LINKED_BINARY"))
                    dynamicSession.write_last_linked_program(path);
            } catch (...) { }
            std::rethrow_exception(error);
        }
        if (tracePath) dynamicSession.write_execution_trace_csv(tracePath);
        // Save the linked program even when a numerical comparison fails.
        if (const char* path = std::getenv("FTLPU_QWEN_C2C_LINKED_BINARY"))
            dynamicSession.write_last_linked_program(path);
        const auto& dynamicActual = dynamicSession.value("hidden.1");
        if (dynamicActual != referenceOutput) {
            std::size_t first = 0;
            while (first < std::min(referenceOutput.size(), dynamicActual.size())
                && referenceOutput[first] == dynamicActual[first]) ++first;
            throw std::logic_error("LPU64 dynamic C2C differs from independent CPU reference: byte="
                + std::to_string(first) + " reference_bytes=" + std::to_string(referenceOutput.size())
                + " dynamic_bytes=" + std::to_string(dynamicActual.size()));
        }
        verifyDynamic(dynamicSession, expectedPrefetches, expectedPagedBytes,
            expectedSyncInstructions, expectedSynchronizedWrites, runtimeDdrBandwidth);
        std::cout << "LPU64 dynamic C2C output: exact_match=" << dynamicActual.size() / 2
                  << " BF16 values, final_transport_segment_cycles="
                  << dynamicSystem->cycle() << std::endl;
#endif
        return 0;
    }
    const auto physicalBf16 = [&](ftlpu::Hemisphere hemisphere,
                                  std::size_t lowSlice,
                                  std::size_t highSlice,
                                  std::size_t address,
                                  std::size_t column,
                                  std::size_t bank = 0) {
        try {
            const auto low = system->read_mem_sram_lane_byte(
                hemisphere, lowSlice, bank, column / 8, address, column % 8);
            const auto high = system->read_mem_sram_lane_byte(
                hemisphere, highSlice, bank, column / 8, address, column % 8);
            return ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8))
                .to_float();
        } catch (const std::out_of_range& error) {
            throw std::out_of_range(std::string(error.what())
                + " hemisphere=" + std::to_string(static_cast<int>(hemisphere))
                + " slices=" + std::to_string(lowSlice) + ","
                + std::to_string(highSlice) + " bank=" + std::to_string(bank)
                + " row=" + std::to_string(address)
                + " column=" + std::to_string(column));
        }
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
    const auto probabilityPackBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "attention.probability_pack";
        });
    if (probabilityPackBinding == program.bindings.end()
        || probabilityPackBinding->slices.size() != 16)
        throw std::logic_error(
            "decoder binary is missing packed probability metadata");
    const auto readMxmDistributed = [&](const std::vector<std::uint16_t>& slices,
                                        std::size_t baseRow,
                                        std::size_t row,
                                        std::size_t column,
                                        ftlpu::Hemisphere hemisphere =
                                            ftlpu::Hemisphere::East,
                                        std::size_t bank = 0) {
        const std::size_t block = program.hardware.mxm_rows;
        const std::size_t lanes = program.hardware.lanes_per_tile;
        const std::size_t waves = program.hardware.tile_rows;
        const std::size_t hiddenBlocks = kHidden / block;
        const std::size_t tokenBlock = row / block;
        const std::size_t tokenWave = (row % block) / lanes;
        const std::size_t tokenLane = row % lanes;
        const std::size_t hiddenBlock = column / block;
        const std::size_t address = baseRow
            + (tokenBlock * hiddenBlocks + hiddenBlock) * waves + tokenWave;
        return physicalBf16(hemisphere,
            slices[2 * tokenLane], slices[2 * tokenLane + 1],
            address, column % block, bank);
    };
    const auto readMxmDistributedWidth = [&physicalBf16, &program](
                                             const std::vector<std::uint16_t>& slices,
                                             std::size_t baseRow,
                                             std::size_t row,
                                             std::size_t column,
                                             std::size_t logicalColumns,
                                             ftlpu::Hemisphere hemisphere,
                                             std::size_t bank) {
        const std::size_t block = program.hardware.mxm_rows;
        const std::size_t lanes = program.hardware.lanes_per_tile;
        const std::size_t waves = program.hardware.tile_rows;
        const std::size_t columnBlocks = logicalColumns / block;
        const std::size_t tokenBlock = row / block;
        const std::size_t tokenWave = (row % block) / lanes;
        const std::size_t tokenLane = row % lanes;
        const std::size_t columnBlock = column / block;
        const std::size_t address = baseRow
            + (tokenBlock * columnBlocks + columnBlock) * waves + tokenWave;
        return physicalBf16(hemisphere,
            slices[2 * tokenLane], slices[2 * tokenLane + 1],
            address, column % block, bank);
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
    const std::size_t scopedCmodelLogStart = scopedCmodelLog
        ? static_cast<std::size_t>(std::strtoull(
              std::getenv("FTLPU_CMODEL_LOG_START"), nullptr, 10))
        : 0;
    const auto* scopedCmodelLogCyclesText =
        std::getenv("FTLPU_CMODEL_LOG_CYCLES");
    const std::size_t scopedCmodelLogCycles = scopedCmodelLogCyclesText
        ? static_cast<std::size_t>(
              std::strtoull(scopedCmodelLogCyclesText, nullptr, 10))
        : 64;
    const bool scopedPrepackLog = scopedCmodelLog
        && scopedCmodelLogStart < attentionPrepackEndCycle;
    if (scopedPrepackLog) {
        runtime.run_cycles(scopedCmodelLogStart);
        const std::size_t logCycles = std::min(
            scopedCmodelLogCycles,
            attentionPrepackEndCycle - scopedCmodelLogStart);
        runtime.run_cycles(logCycles, cmodelLogSink);
        runtime.run_cycles(
            attentionPrepackEndCycle - scopedCmodelLogStart - logCycles);
    } else {
        runtime.run_cycles(attentionPrepackEndCycle,
            scopedCmodelLog ? nullptr : cmodelLogSink);
    }
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
    const bool scopedQkvLog = scopedCmodelLog
        && scopedCmodelLogStart >= attentionPrepackEndCycle
        && scopedCmodelLogStart < qkvEndCycle;
    if (scopedQkvLog) {
        runtime.run_cycles(
            scopedCmodelLogStart - attentionPrepackEndCycle);
        const std::size_t logCycles = std::min(
            scopedCmodelLogCycles, qkvEndCycle - scopedCmodelLogStart);
        runtime.run_cycles(logCycles, cmodelLogSink);
        runtime.run_cycles(
            qkvEndCycle - scopedCmodelLogStart - logCycles);
    } else {
        runtime.run_cycles(
            qkvEndCycle - attentionPrepackEndCycle,
            scopedCmodelLog ? nullptr : cmodelLogSink);
    }
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
    const auto queryBiasBinding = std::find_if(
        program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access
                    == ftlpu::software::runtime::BindingAccess::Input
                && binding.name == "query_bias";
        });
    const std::size_t valueSliceGroups =
        valueBinding->slices.size() / 16;
    const std::size_t valueTokenBlocks = static_cast<std::size_t>(valueBinding->shape.at(0)) / 32;
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
            + ((head * kHeadBlocks + headBlock) * valueTokenBlocks
                + token / 32) * kTileRows
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
    if (valueTensorMaxError > 0.04f) {
        throw std::logic_error(
            "full attention value checkpoint mismatch max_error="
            + std::to_string(valueTensorMaxError)
            + " head=" + std::to_string(valueTensorMaxHead)
            + " token=" + std::to_string(valueTensorMaxToken)
            + " dimension=" + std::to_string(valueTensorMaxDimension)
            + " actual=" + std::to_string(valueTensorMaxActual)
            + " expected=" + std::to_string(valueTensorMaxExpected));
    }
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
    std::array<std::size_t, ftlpu::VxmSlice::kAluQueues> vxmRemaining {};
    for (std::size_t column = 0; column < vxmRemaining.size(); ++column) {
        for (std::size_t tile = 0; tile < ftlpu::VxmSlice::kTileCount;
             ++tile) {
            vxmRemaining[column] +=
                system->vxm_unit().superlane(tile).config_entry_count(column);
        }
    }
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
            || queryBiasBinding == program.bindings.end()
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
                            dimension % 32,
                            (queryBiasBinding->bank + reduction / 2) % 2);
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
    const bool scopedAttentionGapLog = scopedCmodelLog
        && scopedCmodelLogStart >= qkvEndCycle
        && scopedCmodelLogStart < pvStartCycle;
    if (scopedAttentionGapLog) {
        runtime.run_cycles(scopedCmodelLogStart - qkvEndCycle);
        const std::size_t logCycles = std::min(
            scopedCmodelLogCycles, pvStartCycle - scopedCmodelLogStart);
        runtime.run_cycles(logCycles, cmodelLogSink);
        runtime.run_cycles(
            pvStartCycle - scopedCmodelLogStart - logCycles);
    } else {
        runtime.run_cycles(pvStartCycle - qkvEndCycle,
            scopedCmodelLog ? nullptr : cmodelLogSink);
    }
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
                + std::to_string(probability) + " expected=1.0"
                + " pack_east=" + std::to_string(physicalBf16(
                    ftlpu::Hemisphere::East,
                    probabilityPackBinding->slices[0],
                    probabilityPackBinding->slices[1],
                    static_cast<std::size_t>(
                        probabilityPackBinding->base_row)
                        + head * kTokenBlocks * kTokenBlocks * kTileRows,
                    0, probabilityPackBinding->bank))
                + " pack_west=" + std::to_string(physicalBf16(
                    ftlpu::Hemisphere::West,
                    probabilityPackBinding->slices[0],
                    probabilityPackBinding->slices[1],
                    static_cast<std::size_t>(
                        probabilityPackBinding->base_row)
                        + head * kTokenBlocks * kTokenBlocks * kTileRows,
                    0, probabilityPackBinding->bank))
                + " diagonal8=" + std::to_string(physicalBf16(
                    static_cast<ftlpu::Hemisphere>(kvHead % 2),
                    probabilityDiagonalBinding->slices[0],
                    probabilityDiagonalBinding->slices[1],
                    static_cast<std::size_t>(
                        probabilityDiagonalBinding->base_row)
                        + head * kTokenBlocks * kTokenBlocks * kTileRows,
                    8, probabilityDiagonalBinding->bank))
                + " diagonal16=" + std::to_string(physicalBf16(
                    static_cast<ftlpu::Hemisphere>(kvHead % 2),
                    probabilityDiagonalBinding->slices[0],
                    probabilityDiagonalBinding->slices[1],
                    static_cast<std::size_t>(
                        probabilityDiagonalBinding->base_row)
                        + head * kTokenBlocks * kTokenBlocks * kTileRows,
                    16, probabilityDiagonalBinding->bank))
                + " diagonal24=" + std::to_string(physicalBf16(
                    static_cast<ftlpu::Hemisphere>(kvHead % 2),
                    probabilityDiagonalBinding->slices[0],
                    probabilityDiagonalBinding->slices[1],
                    static_cast<std::size_t>(
                        probabilityDiagonalBinding->base_row)
                        + head * kTokenBlocks * kTokenBlocks * kTileRows,
                    24, probabilityDiagonalBinding->bank)));
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
    if (scopedCmodelLog && !scopedPrepackLog && !scopedQkvLog
        && !scopedAttentionGapLog) {
        const std::size_t logStart = scopedCmodelLogStart;
        const std::size_t requestedLogCycles = scopedCmodelLogCycles;
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
        runtime.run_cycles(pvCheckpointCycles,
            (scopedPrepackLog || scopedQkvLog || scopedAttentionGapLog)
                ? nullptr : cmodelLogSink);
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
                row, column, ftlpu::Hemisphere::East,
                residualBinding->bank);
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
                static_cast<std::size_t>(residualBinding->base_row), 0, 0,
                ftlpu::Hemisphere::East, residualBinding->bank)));
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
    if (finalResidualStartCycle < ffnStartCycle)
        throw std::logic_error("final residual precedes FFN execution");
    const bool reportFfnProgress =
        std::getenv("FTLPU_SESSION_PROGRESS") != nullptr;
    for (std::size_t cycle = ffnStartCycle;
         cycle < finalResidualStartCycle;) {
        const auto count = std::min<std::size_t>(
            100'000, finalResidualStartCycle - cycle);
        runtime.run_cycles(count, cmodelLogSink);
        cycle += count;
        if (reportFfnProgress)
            std::cout << FTLPU_TEST_MODEL_NAME << " FFN cycle=" << cycle
                      << "/" << finalResidualStartCycle << std::endl;
    }
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
    const auto referenceSpecialAlu = makeReferenceVxmSpecialAlu();
    const auto hardwareHiddenValue = [&](std::size_t row, std::size_t h) {
        const float gate =
            rms2Output[row * kHidden + gateK(h)] * gateSign(h);
        const float up =
            rms2Output[row * kHidden + upK(h)] * upSign(h);
        const float exponential = referenceSpecialAlu.execute(
            ftlpu::VxmSpecialAluOpcode::Exp, -gate);
        const float reciprocal = referenceSpecialAlu.execute(
            ftlpu::VxmSpecialAluOpcode::Reciprocal,
            exponential + 1.0f);
        return bf16(reciprocal * gate * up);
    };
    const auto hiddenBinding = std::find_if(program.bindings.begin(), program.bindings.end(),
        [](const auto& binding) {
            return binding.access == ftlpu::software::runtime::BindingAccess::Internal
                && binding.name == "ffn.hidden";
        });
    if (hiddenBinding == program.bindings.end()
        || hiddenBinding->shape != std::vector<std::uint64_t>{kSeqLen, kIntermediate})
        throw std::logic_error("cannot locate actual FFN hidden placement");
    const auto hiddenLocation = [&](std::size_t row, std::size_t column) {
        using ftlpu::software::runtime::BindingLayout;
        const auto base = static_cast<std::size_t>(hiddenBinding->base_row);
        std::size_t pair = 0;
        std::size_t address = 0;
        if (hiddenBinding->layout == BindingLayout::Fp16MxmDistributed16
            && hiddenBinding->slices.size() == 16) {
            pair = row % kMxmBlockRows;
            address = base + ((row / 32) * (kIntermediate / 32) + column / 32)
                * kTileRows + (row % 32) / kMxmBlockRows;
        } else if (hiddenBinding->layout == BindingLayout::Fp16PairPlanar
            && hiddenBinding->slices.size() == 4) {
            pair = (column / 32) % 2;
            address = base + (column / 64) * kSeqLen + row;
        } else {
            throw std::logic_error("unsupported FFN hidden checkpoint placement");
        }
        return std::array<std::size_t, 3>{
            hiddenBinding->slices[2 * pair], hiddenBinding->slices[2 * pair + 1], address};
    };
    const auto readHidden = [&](const auto& readPhysical, ftlpu::Hemisphere hemisphere,
                               std::size_t row, std::size_t column) {
        const auto location = hiddenLocation(row, column);
        return readPhysical(hemisphere, location[0], location[1], location[2],
            column % 32, hiddenBinding->bank);
    };
    const auto unpackedHidden = runtime.download_binding(*hiddenBinding);
    std::vector<float> physicalHidden(kSeqLen * kIntermediate);
    std::size_t hiddenMismatchCount = 0;
    float hiddenMaxError = 0.0f;
    std::size_t hardwareHiddenMismatchCount = 0;
    float hardwareHiddenMaxError = 0.0f;
    float hiddenReplicaMaxError = 0.0f;
    for (std::size_t row = 0; row < kSeqLen; ++row) {
        for (std::size_t h = 0; h < kIntermediate; ++h) {
            const std::size_t index = row * kIntermediate + h;
            const float east = readHidden(physicalBf16, ftlpu::Hemisphere::East, row, h);
            const float west = readHidden(physicalBf16, ftlpu::Hemisphere::West, row, h);
            if (east != readBf16(unpackedHidden, index))
                throw std::logic_error("FFN hidden metadata unpacking differs from physical SRAM");
            const float expected = hiddenValue(row, h);
            const float hardwareExpected = hardwareHiddenValue(row, h);
            physicalHidden[index] = east;
            const float error = std::fabs(east - expected);
            const float hardwareError = std::fabs(east - hardwareExpected);
            hiddenMaxError = std::max(hiddenMaxError, error);
            hardwareHiddenMaxError = std::max(
                hardwareHiddenMaxError, hardwareError);
            hiddenReplicaMaxError = std::max(
                hiddenReplicaMaxError, std::fabs(east - west));
            if (error > 0.04f) ++hiddenMismatchCount;
            if (hardwareError > 0.0f) {
                if (hardwareHiddenMismatchCount < 16) {
                    const float gate = rms2Output[
                        row * kHidden + gateK(h)] * gateSign(h);
                    const float up = rms2Output[
                        row * kHidden + upK(h)] * upSign(h);
                    const float swapped = bf16(
                        up * (1.0f / (1.0f + std::exp(-up))) * gate);
                    std::cerr << "FFN hidden mismatch row=" << row
                              << " h=" << h << " actual=" << east
                              << " mathematical=" << expected
                              << " hardware=" << hardwareExpected
                              << " swapped=" << swapped
                              << " gate=" << gate << " up=" << up
                              << '\n';
                }
                ++hardwareHiddenMismatchCount;
            }
        }
    }
    std::cout << "FFN hidden checkpoint summary: mismatches="
              << hiddenMismatchCount << " max_error=" << hiddenMaxError
              << " hardware_mismatches=" << hardwareHiddenMismatchCount
              << " hardware_max_error=" << hardwareHiddenMaxError
              << " replica_max_error=" << hiddenReplicaMaxError << '\n';
    if (!skipNumericChecks && (hiddenMismatchCount != 0
            || hardwareHiddenMismatchCount != 0 || hiddenReplicaMaxError != 0.0f))
        throw std::logic_error("FFN hidden checkpoint disagrees with its hardware reference or replica");
    std::vector<float> ffnOutput(kSeqLen * kHidden);
    float ffnCheckpointMaxError = 0.0f;
    std::size_t ffnCheckpointMismatchCount = 0;
    float ffnPhysicalInputMaxError = 0.0f;
    std::size_t ffnPhysicalInputMismatchCount = 0;
    std::array<std::size_t, kHidden / 32>
        ffnCheckpointBlockMismatchCounts {};
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
            const float expectedFromPhysicalHidden = bf16(
                physicalHidden[row * kIntermediate + h0]
                - physicalHidden[row * kIntermediate + h1]);
            const float physicalInputError =
                std::fabs(ownerValue - expectedFromPhysicalHidden);
            ffnPhysicalInputMaxError = std::max(
                ffnPhysicalInputMaxError, physicalInputError);
            if (physicalInputError > 0.04f)
                ++ffnPhysicalInputMismatchCount;
            ffnCheckpointMaxError = std::max(
                ffnCheckpointMaxError, std::max(ownerError, replicaError));
            if (ownerError > 0.04f || replicaError > 0.04f) {
                ++ffnCheckpointBlockMismatchCounts[outputBlock];
                if (ffnCheckpointMismatchCount < 16)
                    std::cerr << "FFN down checkpoint mismatch row="
                              << row << " column=" << column
                              << " output_block=" << outputBlock
                              << " owner="
                              << static_cast<std::size_t>(owner)
                              << " owner_actual=" << ownerValue
                              << " replica_actual=" << replicaValue
                              << " expected=" << expected
                              << " physical_hidden_expected="
                              << expectedFromPhysicalHidden
                              << " h0=" << h0 << " h1=" << h1
                              << " bank=" << ffnBinding->bank
                              << " base_row=" << ffnBinding->base_row
                              << '\n';
                ++ffnCheckpointMismatchCount;
            }
        }
    }
    if (!skipNumericChecks && ffnCheckpointMismatchCount != 0) {
        std::ostringstream message;
        message << "FFN down checkpoint mismatch count="
                << ffnCheckpointMismatchCount
                << " max_error=" << ffnCheckpointMaxError
                << " physical_hidden_mismatches="
                << ffnPhysicalInputMismatchCount
                << " physical_hidden_max_error="
                << ffnPhysicalInputMaxError
                << " block_counts=";
        for (std::size_t block = 0;
             block < ffnCheckpointBlockMismatchCounts.size(); ++block) {
            if (block != 0) message << ',';
            message << ffnCheckpointBlockMismatchCounts[block];
        }
        throw std::logic_error(message.str());
    }
    const float preFinalResidualSample = readMxmDistributed(
        residualSlices,
        static_cast<std::size_t>(residualBinding->base_row),
        kSeqLen - 1, 0, ftlpu::Hemisphere::East,
        residualBinding->bank);
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
    auto [dynamicPackage, expectedPrefetches, expectedPagedBytes,
        expectedSyncInstructions, expectedSynchronizedWrites] = prepareDynamic();

    // Preserve the direct projection temporaries for the bounded dynamic-C2C
    // checkpoint.  Gate and Up share activation slices 0..15, live in the
    // bank opposite the normalized input, and occupy one 32-row window per
    // output pair.  Keeping raw bytes avoids baking a logical layout into the
    // diagnostic and tells us whether corruption precedes fused Swish.
    std::vector<std::uint8_t> directFfnTempBytes;
    constexpr std::size_t kFfnTempSlices = 16;
    constexpr std::size_t kFfnTempColumns = 32;
    const std::size_t ffnTempRows = (kIntermediate / 64) * kSeqLen;
    const std::size_t ffnTempBank = 1 - rms2Binding->bank;
    if (std::getenv("FTLPU_QWEN_HIDDEN_CHECKPOINT_ONLY") != nullptr) {
        directFfnTempBytes.reserve(
            2 * kFfnTempSlices * ffnTempRows * kFfnTempColumns);
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (std::size_t slice = 0; slice < kFfnTempSlices; ++slice) {
                for (std::size_t address = 0; address < ffnTempRows;
                     ++address) {
                    for (std::size_t column = 0; column < kFfnTempColumns;
                         ++column) {
                        directFfnTempBytes.push_back(
                            system->read_mem_sram_lane_byte(
                                static_cast<ftlpu::Hemisphere>(hemisphere),
                                slice, ffnTempBank, column / 8, address,
                                column % 8));
                    }
                }
            }
        }
    }

    // The direct run has completed and `actual` owns its downloaded output.
    // Release its large chip model before constructing the C2C-backed one.
    runtimeOwner.reset();
    systemOwner.reset();
    system = nullptr;
    ftlpu::C2cDmaSystem dynamicSystem;
    ModelSession dynamicSession(dynamicSystem);
    if(const auto latency=std::getenv("FTLPU_DDR_EXTRA_READ_LATENCY")) {
        const auto extra=std::stoull(latency);
        if(extra>UINT32_MAX)throw std::invalid_argument("DDR latency overflow");
        dynamicSession.set_ddr_latency_perturbation(static_cast<std::uint32_t>(extra),20261029);
    }
    std::uint32_t runtimeDdrBandwidth =
        program.hardware.ddr_peak_bandwidth_mbytes_per_second;
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
    if (inputBindingByIndex(static_cast<std::uint32_t>(postAttentionNormBinding)).bank
        != inputBindingByIndex(1).bank)
        dynamicSession.set_input("post_attention_layernorm.weight", gamma1);
    try {
        dynamicSession.run(kCheckpointDrainCycles);
    } catch (...) {
        const auto executionError = std::current_exception();
        try {
            if (dynamicPipelineTracePath != nullptr)
                dynamicSession.write_execution_trace_csv(dynamicPipelineTracePath);
        } catch (...) {
            // Keep the original failure; partial traces are diagnostic only.
        }
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
    if (const char* linkedPath =
            std::getenv("FTLPU_QWEN_C2C_LINKED_BINARY"))
        dynamicSession.write_last_linked_program(linkedPath);
    if (std::getenv("FTLPU_QWEN_HIDDEN_CHECKPOINT_ONLY") != nullptr) {
        const auto checkpointPhysicalBf16 = [&](
                ftlpu::Hemisphere hemisphere, std::size_t lowSlice,
                std::size_t highSlice, std::size_t address,
                std::size_t column, std::size_t bank) {
            const auto low = dynamicSystem.chip().read_mem_sram_lane_byte(
                hemisphere, lowSlice, bank, column / 8, address, column % 8);
            const auto high = dynamicSystem.chip().read_mem_sram_lane_byte(
                hemisphere, highSlice, bank, column / 8, address, column % 8);
            return ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        };
        std::size_t tempMismatches = 0;
        std::array<std::size_t, kFfnTempSlices> tempSliceMismatches {};
        std::size_t tempByteIndex = 0;
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (std::size_t slice = 0; slice < kFfnTempSlices; ++slice) {
                for (std::size_t address = 0; address < ffnTempRows;
                     ++address) {
                    for (std::size_t column = 0; column < kFfnTempColumns;
                         ++column, ++tempByteIndex) {
                        const auto dynamicByte =
                            dynamicSystem.chip().read_mem_sram_lane_byte(
                                static_cast<ftlpu::Hemisphere>(hemisphere),
                                slice, ffnTempBank, column / 8, address,
                                column % 8);
                        if (dynamicByte == directFfnTempBytes[tempByteIndex])
                            continue;
                        if (tempMismatches < 32)
                            std::cerr << "dynamic FFN temp mismatch hemisphere="
                                      << hemisphere << " slice=" << slice
                                      << " bank=" << ffnTempBank
                                      << " address=" << address
                                      << " column=" << column
                                      << " dynamic="
                                      << static_cast<unsigned>(dynamicByte)
                                      << " direct=" << static_cast<unsigned>(
                                             directFfnTempBytes[tempByteIndex])
                                      << '\n';
                        ++tempMismatches;
                        ++tempSliceMismatches[slice];
                    }
                }
            }
        }
        std::cout << "dynamic FFN temp checkpoint: mismatches="
                  << tempMismatches << " slice_counts=";
        for (std::size_t slice = 0; slice < tempSliceMismatches.size();
             ++slice) {
            if (slice != 0) std::cout << ',';
            std::cout << tempSliceMismatches[slice];
        }
        std::cout << '\n';

        std::size_t mismatches = 0;
        float maxError = 0.0f;
        std::array<std::size_t, kIntermediate / 32> blockMismatches {};
        for (std::size_t row = 0; row < kSeqLen; ++row) {
            for (std::size_t h = 0; h < kIntermediate; ++h) {
                const std::size_t block = h / 32;
                const auto location = hiddenLocation(row, h);
                const auto address = location[2];
                const float actualValue = readHidden(checkpointPhysicalBf16,
                    ftlpu::Hemisphere::East, row, h);
                const float expectedValue =
                    physicalHidden[row * kIntermediate + h];
                const float error = std::fabs(actualValue - expectedValue);
                maxError = std::max(maxError, error);
                if (error == 0.0f) continue;
                if (mismatches < 16)
                    std::cerr << "dynamic hidden checkpoint mismatch row="
                              << row << " h=" << h << " block=" << block
                              << " address=" << address
                              << " actual=" << actualValue
                              << " direct=" << expectedValue << '\n';
                ++mismatches;
                ++blockMismatches[block];
            }
        }
        std::cout << "dynamic hidden checkpoint: mismatches=" << mismatches
                  << " max_error=" << maxError << " block_counts=";
        for (std::size_t block = 0; block < blockMismatches.size(); ++block) {
            if (block != 0) std::cout << ',';
            std::cout << blockMismatches[block];
        }
        std::cout << '\n';
        return mismatches == 0 && tempMismatches == 0 ? 0 : 1;
    }
    const auto& dynamicActual = dynamicSession.value(
        "hidden." + std::to_string(decoderLayerCount));
    if (dynamicActual.size() != actual.size())
        throw std::logic_error(
            "Qwen dynamic C2C output byte size differs from direct golden: "
            + std::to_string(dynamicActual.size()) + " vs "
            + std::to_string(actual.size()));
    if (!skipNumericChecks && dynamicActual != actual) {
        std::size_t firstMismatch = 0;
        std::size_t outputMismatchCount = 0;
        float outputMaxError = 0.0f;
        while (firstMismatch < actual.size()
            && dynamicActual[firstMismatch] == actual[firstMismatch])
            ++firstMismatch;
        for (std::size_t offset = 0; offset + 1 < actual.size(); offset += 2) {
            const float directValue = readBf16(actual, offset / 2);
            const float dynamicValue = readBf16(dynamicActual, offset / 2);
            const float error = std::fabs(directValue - dynamicValue);
            if (error != 0.0f) ++outputMismatchCount;
            outputMaxError = std::max(outputMaxError, error);
        }
        const auto dynamicPhysicalBf16 = [&](ftlpu::Hemisphere hemisphere,
                                             std::size_t lowSlice,
                                             std::size_t highSlice,
                                             std::size_t address,
                                             std::size_t column,
                                             std::size_t bank) {
            const auto low = dynamicSystem.chip().read_mem_sram_lane_byte(
                hemisphere, lowSlice, bank, column / 8, address, column % 8);
            const auto high = dynamicSystem.chip().read_mem_sram_lane_byte(
                hemisphere, highSlice, bank, column / 8, address, column % 8);
            return ftlpu::Bf16::from_bits(
                static_cast<std::uint16_t>(low)
                | (static_cast<std::uint16_t>(high) << 8)).to_float();
        };
        const auto dynamicHidden = [&](std::size_t row, std::size_t column) {
            return readHidden(dynamicPhysicalBf16, ftlpu::Hemisphere::East, row, column);
        };
        std::size_t dynamicHiddenMismatches = 0;
        float dynamicHiddenMaxError = 0.0f;
        for (std::size_t row = 0; row < kSeqLen; ++row) {
            for (std::size_t h = 0; h < kIntermediate; ++h) {
                const std::size_t index = row * kIntermediate + h;
                const float error = std::fabs(
                    dynamicHidden(row, h)
                    - physicalHidden[index]);
                if (error != 0.0f) ++dynamicHiddenMismatches;
                dynamicHiddenMaxError = std::max(dynamicHiddenMaxError, error);
            }
        }
        std::size_t dynamicFfnMismatches = 0;
        float dynamicFfnMaxError = 0.0f;
        std::array<std::size_t, kHidden / 32>
            dynamicFfnBlockMismatchCounts {};
        if (ffnBinding->layout
            == ftlpu::software::runtime::BindingLayout::Fp16PairPlanar) {
            for (std::size_t row = 0; row < kSeqLen; ++row) {
                for (std::size_t column = 0; column < kHidden; ++column) {
                    const std::size_t outputBlock = column / 32;
                    const auto owner = static_cast<ftlpu::Hemisphere>(
                        (outputBlock % 4) / 2);
                    const std::size_t pair = outputBlock % 2;
                    const std::size_t address =
                        static_cast<std::size_t>(ffnBinding->base_row)
                        + (outputBlock / 4) * kSeqLen + row;
                    const float dynamicValue = dynamicPhysicalBf16(owner,
                        ffnBinding->slices[pair * 2],
                        ffnBinding->slices[pair * 2 + 1], address,
                        column % 32, ffnBinding->bank);
                    const float error = std::fabs(dynamicValue
                        - ffnOutput[row * kHidden + column]);
                    if (error != 0.0f) {
                        if (dynamicFfnMismatches < 16)
                        {
                            const std::size_t h0 =
                                (column * 5 + 17) % kIntermediate;
                            const std::size_t h1 =
                                (h0 + 37) % kIntermediate;
                            std::cerr << "dynamic FFN mismatch row=" << row
                                      << " column=" << column
                                      << " block=" << outputBlock
                                      << " owner="
                                      << static_cast<std::size_t>(owner)
                                      << " actual=" << dynamicValue
                                      << " direct="
                                      << ffnOutput[row * kHidden + column]
                                      << " error=" << error
                                      << " h0=" << h0
                                      << " h0.dynamic="
                                      << dynamicHidden(row, h0)
                                      << " h0.direct=" << physicalHidden[
                                             row * kIntermediate + h0]
                                      << " h1=" << h1
                                      << " h1.dynamic="
                                      << dynamicHidden(row, h1)
                                      << " h1.direct=" << physicalHidden[
                                             row * kIntermediate + h1]
                                      << '\n';
                        }
                        ++dynamicFfnMismatches;
                        ++dynamicFfnBlockMismatchCounts[outputBlock];
                    }
                    dynamicFfnMaxError = std::max(dynamicFfnMaxError, error);
                }
            }
        }
        std::vector<std::size_t> downPageByteMismatches(
            inputBindingByIndex(static_cast<std::uint32_t>(downBinding))
                .page_count);
        for (std::uint32_t page = 0;
             page < downPageByteMismatches.size(); ++page) {
            const auto image = pack_weight_binding_page(
                inputBindingByIndex(static_cast<std::uint32_t>(downBinding)),
                page, downWeight, program.hardware);
            const auto placement = resolve_weight_page_placement(
                inputBindingByIndex(static_cast<std::uint32_t>(downBinding)),
                page);
            for (const PackedWeightSegment& segment : image.segments)
                for (std::uint32_t vector = 0;
                     vector < segment.vector_count; ++vector)
                    for (std::uint32_t byte = 0;
                         byte < ftlpu::hw::kPhysicalVectorBytes; ++byte) {
                        const auto expected = image.data.at(
                            static_cast<std::size_t>(segment.byte_offset)
                            + static_cast<std::size_t>(vector)
                                * ftlpu::hw::kPhysicalVectorBytes + byte);
                        const auto resident =
                            dynamicSystem.chip().read_mem_sram_lane_byte(
                                static_cast<ftlpu::Hemisphere>(
                                    segment.hemisphere),
                                segment.slice, placement.bank, byte / 8,
                                segment.base_row + vector, byte % 8);
                        if (resident != expected)
                            ++downPageByteMismatches[page];
                    }
        }
        const auto joinCounts = [](const auto& counts) {
            std::ostringstream text;
            for (std::size_t index = 0; index < counts.size(); ++index) {
                if (index != 0) text << ',';
                text << counts[index];
            }
            return text.str();
        };
        const std::size_t firstElement = firstMismatch / 2;
        const ModelSessionStats& mismatchStats = dynamicSession.stats();
        throw std::logic_error(
            "Qwen dynamic C2C output differs from direct golden at byte "
            + std::to_string(firstMismatch)
            + " row=" + std::to_string(firstElement / kHidden)
            + " column=" + std::to_string(firstElement % kHidden)
            + " direct=" + std::to_string(readBf16(actual, firstElement))
            + " dynamic="
                + std::to_string(readBf16(dynamicActual, firstElement))
            + " output_mismatches=" + std::to_string(outputMismatchCount)
            + " output_max_error=" + std::to_string(outputMaxError)
            + " hidden_mismatches="
                + std::to_string(dynamicHiddenMismatches)
            + " hidden_max_error=" + std::to_string(dynamicHiddenMaxError)
            + " ffn_mismatches=" + std::to_string(dynamicFfnMismatches)
            + " ffn_max_error=" + std::to_string(dynamicFfnMaxError)
            + " ffn_block_mismatches="
                + joinCounts(dynamicFfnBlockMismatchCounts)
            + " down_page_byte_mismatches="
                + joinCounts(downPageByteMismatches)
            + " initial_wait_cycles="
                + std::to_string(
                    mismatchStats.weight_page_initial_wait_cycles)
            + " runtime_wait_cycles="
                + std::to_string(
                    mismatchStats.weight_page_runtime_wait_cycles));
    }
    verifyDynamic(dynamicSession, expectedPrefetches, expectedPagedBytes,
        expectedSyncInstructions, expectedSynchronizedWrites,
        runtimeDdrBandwidth);
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
