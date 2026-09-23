// Keep this translation unit rebuilt when BinaryProgram or binding ABI evolves.
#include "ftlpu/compiler/Target/command_binary.hpp"

#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Target/icu_compression.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/icu/sxm_run_2d.hpp"
#include "ftlpu/icu/vxm_run_2d.hpp"
#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace ftlpu::compiler::target {
namespace {

// QueueCommand has a variable SXM payload; keep this translation unit rebuilt
// with the runtime queue ABI rather than relying on a transitive header edge.

using software::runtime::BinaryBinding;
using software::runtime::BinaryAddressRelocation;
using software::runtime::BinaryMemoryFloor;
using software::runtime::BinaryTimeline;
using software::runtime::BinaryWeightPageUse;
using software::runtime::BindingAccess;
using software::runtime::BindingElementType;
using software::runtime::BindingLayout;
using software::runtime::BinaryScaleRelocation;
using software::runtime::InstructionKind;
using software::runtime::QueueCommand;
using software::runtime::QueueKind;
using software::runtime::QueueProgram;
using software::runtime::VxmImmediateOperand;

struct CommandSequence {
    int64_t cycle{0};
    int64_t repeat_count{0};
    int64_t repeat_interval{0};
    int64_t address_stride{0};
    QueueCommand instruction;
    int64_t scale_binding{-1};
    int64_t address_binding{-1};
    int64_t write_address_binding{-1};
    int64_t outer_count{1};
    int64_t outer_interval{1};
    int64_t outer_stride{0};
    IcuInductionTarget induction_target{IcuInductionTarget::None};
    int64_t depth_count{1};
    int64_t depth_interval{1};
    int64_t depth_stride{0};
    BindingAccess address_binding_access{BindingAccess::Input};
    BindingAccess write_address_binding_access{BindingAccess::Input};
};

BindingAccess address_binding_access(mlir::Operation* operation)
{
    const auto access = operation->getAttrOfType<mlir::StringAttr>(
        "address_binding_access");
    return access && access.getValue() == "internal"
        ? BindingAccess::Internal : BindingAccess::Input;
}

int64_t command_cycle(mlir::Operation* op)
{
    return llvm::cast<mlir::IntegerAttr>(op->getAttrDictionary().get("cycle")).getInt();
}

int64_t command_integer(mlir::Operation* op, llvm::StringRef name)
{
    return llvm::cast<mlir::IntegerAttr>(op->getAttrDictionary().get(name)).getInt();
}

using QueueKey = std::pair<QueueKind, int64_t>;
using QueueMap = std::map<QueueKey, std::vector<CommandSequence>>;

struct Raw3DPacketSequence {
    std::size_t start_cycle{0};
    std::size_t final_cycle{0};
    std::vector<QueueCommand> physical_words;
    int64_t address_binding{-1};
    BindingAccess address_binding_access{BindingAccess::Input};
    int64_t scale_binding{-1};
};

using Raw3DQueueMap =
    std::map<QueueKey, std::vector<Raw3DPacketSequence>>;

bool can_encode_packet_wait(const Raw3DPacketSequence& packet,
    std::size_t wait)
{
    if (packet.physical_words.empty() || wait >= (std::size_t {1} << 24))
        return false;
    const auto kind = packet.physical_words.front().instruction_kind;
    if (kind != InstructionKind::Mem && kind != InstructionKind::Mxm
        && kind != InstructionKind::MxmDequant
        && kind != InstructionKind::Vxm && kind != InstructionKind::Sxm)
        return false;
    // FU 3-D and VXM/SXM RUN_2D use bits 8..31 of word zero for the
    // queue-local wait_cycle. WRITE_READ_2D has a separate packet layout.
    return ((packet.physical_words.front().words[2] >> 24) & 0xfU) == 2U;
}

void set_packet_wait(Raw3DPacketSequence& packet, std::size_t wait)
{
    if (!can_encode_packet_wait(packet, wait))
        throw std::runtime_error("FU 3-D wait_cycle is not encodable");
    auto& header = packet.physical_words.front();
    header.words[0] = (header.words[0] & 0xffU)
        | (static_cast<std::uint32_t>(wait) << 8);
    header.command = header.words[0];
}

const char* raw_loop_queue_name(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return "mem";
    case QueueKind::MxmLoad: return "mxm_load";
    case QueueKind::MxmCompute: return "mxm_compute";
    case QueueKind::MxmDequant: return "mxm_dequant";
    case QueueKind::Vxm: return "vxm";
    case QueueKind::SxmTranspose: return "sxm_transpose";
    case QueueKind::SxmPermute: return "sxm_permute";
    }
    return "unknown";
}

std::size_t raw_loop_iq_depth(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem: return hw::kIcuMemIqDepth;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
    case QueueKind::MxmDequant: return hw::kIcuMxmIqDepth;
    case QueueKind::Vxm: return hw::kIcuVxmIqDepth;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute: return hw::kIcuSxmIqDepth;
    }
    throw std::logic_error("raw FU loop has no physical ICU IQ depth");
}

// The physical ICU preloads one full IQ before cycle zero. Each consumed word
// opens one IQ slot, and the single local-iMEM frontend can start one ordered
// word fetch per cycle. A fetch started after dispatch in cycle T is visible at
// the beginning of T + fetch_latency. Raw multiword packets are decoded
// atomically, so every packet word must be resident at its exact scheduled
// launch cycle; silently waiting for the rest would break cross-FU alignment.
void validate_raw_3d_frontend(const QueueKey& key,
    const std::vector<Raw3DPacketSequence>& packets)
{
    constexpr std::size_t refillWordsPerCycle = 1;
    static_assert(hw::kIcuFetchLatencyCycles > 0);

    const auto checkedAdd = [](std::size_t lhs, std::size_t rhs,
                                const char* description) {
        if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
            throw std::runtime_error(description);
        return lhs + rhs;
    };

    std::size_t totalWords = 0;
    std::size_t scheduleCursor = 0;
    for (const Raw3DPacketSequence& packet : packets) {
        if (packet.start_cycle < scheduleCursor)
            throw std::runtime_error(
                "overlapping FU loop packets target a single-context ICU "
                "queue: resource="
                + std::string(raw_loop_queue_name(key.first))
                + ", queue=" + std::to_string(key.second)
                + ", start_cycle=" + std::to_string(packet.start_cycle)
                + ", busy_through=" + std::to_string(scheduleCursor - 1));
        if (packet.start_cycle > scheduleCursor
            && !can_encode_packet_wait(
                packet, packet.start_cycle - scheduleCursor))
            totalWords = checkedAdd(totalWords, 1,
                "raw FU loop queue word count overflows");
        totalWords = checkedAdd(totalWords, packet.physical_words.size(),
            "raw FU loop queue word count overflows");
        if (packet.final_cycle == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("FU loop packet end cycle overflows");
        scheduleCursor = packet.final_cycle + 1;
    }

    const std::size_t iqDepth = raw_loop_iq_depth(key.first);
    std::size_t iqWords = std::min(iqDepth, totalWords);
    std::size_t scheduledWords = iqWords;
    std::size_t nextFetchStartCycle = 0;
    std::vector<std::size_t> fetchReadyCycles;
    fetchReadyCycles.reserve(totalWords - scheduledWords);
    std::size_t nextReadyFetch = 0;

    const auto consume = [&](std::size_t cycle, std::size_t requiredWords,
                             const char* commandName) {
        while (nextReadyFetch < fetchReadyCycles.size()
               && fetchReadyCycles[nextReadyFetch] <= cycle) {
            ++iqWords;
            ++nextReadyFetch;
        }
        if (iqWords < requiredWords) {
            throw std::runtime_error(
                "raw FU loop schedule exceeds ICU frontend bandwidth: "
                "resource=" + std::string(raw_loop_queue_name(key.first))
                + ", queue=" + std::to_string(key.second)
                + ", cycle=" + std::to_string(cycle)
                + ", command=" + commandName
                + ", required_words=" + std::to_string(requiredWords)
                + ", available_words=" + std::to_string(iqWords)
                + ", iq_depth=" + std::to_string(iqDepth)
                + ", refill_words_per_cycle="
                + std::to_string(refillWordsPerCycle)
                + ", fetch_latency="
                + std::to_string(hw::kIcuFetchLatencyCycles));
        }
        iqWords -= requiredWords;

        // Every consumed resident word opens one slot. Reserve those slots for
        // the next sequential i-MEM words at one fetch start per cycle.
        for (std::size_t word = 0;
             word < requiredWords && scheduledWords < totalWords; ++word) {
            const std::size_t fetchStart =
                std::max(cycle, nextFetchStartCycle);
            if (fetchStart > std::numeric_limits<std::size_t>::max()
                    - hw::kIcuFetchLatencyCycles
                || fetchStart == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error(
                    "raw FU loop frontend fetch cycle overflows");
            fetchReadyCycles.push_back(
                fetchStart + hw::kIcuFetchLatencyCycles);
            ++scheduledWords;
            nextFetchStartCycle = fetchStart + refillWordsPerCycle;
        }
    };

    scheduleCursor = 0;
    for (const Raw3DPacketSequence& packet : packets) {
        const auto wait = packet.start_cycle - scheduleCursor;
        if (wait != 0 && !can_encode_packet_wait(packet, wait))
            consume(scheduleCursor, 1, "NOP");
        consume(wait != 0 && can_encode_packet_wait(packet, wait)
                ? scheduleCursor : packet.start_cycle,
            packet.physical_words.size(), "packet");
        scheduleCursor = packet.final_cycle + 1;
    }
}

std::size_t loop_final_cycle(const IcuLoop3D& loop)
{
    std::size_t cycle = loop.start_cycle;
    for (std::size_t dimension = 0;
         dimension < IcuLoop3D::kDimensions; ++dimension)
        cycle += (loop.counts[dimension] - 1)
            * loop.cycle_strides[dimension];
    return cycle;
}

int64_t raw_loop_absolute_final_cycle(mlir::Operation* operation,
    const IcuLoop3D& loop, std::size_t trailingCycles = 0)
{
    const int64_t startCycle = command_cycle(operation);
    if (startCycle < 0)
        throw std::runtime_error(
            "FU loop has a negative compiler schedule cycle");
    const std::size_t localFinal = loop_final_cycle(loop);
    constexpr auto int64Max = static_cast<std::size_t>(
        std::numeric_limits<int64_t>::max());
    if (localFinal > int64Max || trailingCycles > int64Max - localFinal
        || localFinal + trailingCycles
            > int64Max - static_cast<std::size_t>(startCycle))
        throw std::runtime_error("FU loop absolute final cycle overflows");
    return startCycle
        + static_cast<int64_t>(localFinal + trailingCycles);
}

template <typename Packet>
Packet raw_3d_packet(mlir::Operation* operation, mlir::ArrayAttr words)
{
    constexpr std::size_t expected =
        Packet::kWordCount * Packet::kLanesPerWord;
    if (words.size() != expected)
        throw std::runtime_error(
            "FU 3-D Command op has an invalid physical packet width");
    Packet packet {};
    for (std::size_t index = 0; index < expected; ++index) {
        const auto word = llvm::dyn_cast<mlir::IntegerAttr>(words[index]);
        if (!word || word.getValue().isNegative()
            || word.getValue().getActiveBits() > 32)
            throw std::runtime_error(
                "FU 3-D Command op contains a non-u32 packet word");
        packet.words[index / Packet::kLanesPerWord]
            .lanes[index % Packet::kLanesPerWord] =
            static_cast<std::uint32_t>(word.getValue().getZExtValue());
    }
    (void)operation;
    return packet;
}

template <typename Packet>
std::vector<QueueCommand> raw_3d_queue_words(
    const Packet& packet, InstructionKind instructionKind)
{
    std::vector<QueueCommand> result;
    result.reserve(Packet::kWordCount);
    for (const auto& physicalWord : packet.words) {
        QueueCommand command;
        command.command = physicalWord.lanes[0];
        command.instruction_kind = instructionKind;
        command.word_count = static_cast<std::uint16_t>(
            Packet::kLanesPerWord);
        for (std::size_t lane = 0;
             lane < Packet::kLanesPerWord; ++lane)
            command.words[lane] = physicalWord.lanes[lane];
        result.push_back(std::move(command));
    }
    return result;
}

template <typename Packet, typename Instruction, typename Decode>
void collect_raw_3d(mlir::Operation* operation, int64_t queue,
    mlir::ArrayAttr words, QueueKind queueKind,
    InstructionKind instructionKind, Decode&& decode,
    Raw3DQueueMap& queues)
{
    const Packet packet = raw_3d_packet<Packet>(operation, words);
    Instruction instruction;
    try {
        instruction = decode(packet);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("invalid FU 3-D Command packet: ")
            + error.what());
    }
    const int64_t commandCycle = command_integer(operation, "cycle");
    if (commandCycle < 0)
        throw std::runtime_error(
            "FU 3-D Command op has a negative compiler schedule cycle");
    if (instruction.loop.start_cycle != 0)
        throw std::runtime_error(
            "FU 3-D hardware packet contains an absolute start cycle");
    const std::size_t startCycle = static_cast<std::size_t>(commandCycle);
    const std::size_t localFinalCycle = loop_final_cycle(instruction.loop);
    if (localFinalCycle
        > std::numeric_limits<std::size_t>::max() - startCycle)
        throw std::runtime_error("FU 3-D Command cycle range overflows");
    queues[{queueKind, queue}].push_back(Raw3DPacketSequence {
        startCycle,
        startCycle + localFinalCycle,
        raw_3d_queue_words(packet, instructionKind),
        -1,
        BindingAccess::Input,
    });
}

QueueCommand mem_instruction_command(isa::EncodedMemInstruction encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    };
}

QueueCommand mxm_instruction_command(isa::EncodedMxmInstruction encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mxm,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {
            static_cast<std::uint32_t>(encoded),
            static_cast<std::uint32_t>(encoded >> 32),
            0,
            0,
        },
    };
}

QueueCommand mxm_dequant_instruction_command(
    isa::EncodedMxmDequantInstruction encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction),
        InstructionKind::MxmDequant,
        1,
        {static_cast<std::uint32_t>(encoded), 0, 0, 0},
    };
}

QueueCommand vxm_instruction_command(const isa::EncodedVxmInstruction& encoded)
{
    return QueueCommand {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Vxm,
        3,
        {
            static_cast<std::uint32_t>(encoded.control),
            static_cast<std::uint32_t>(encoded.control >> 32),
            encoded.immediate_bits,
            0,
        },
    };
}

QueueCommand sxm_instruction_command(const SxmInstruction& instruction)
{
    QueueCommand command {
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Sxm, 4, {},
    };
    command.words[0] = static_cast<std::uint32_t>(instruction.opcode);
    command.words[1] = static_cast<std::uint32_t>(instruction.shift_source);
    command.words[2] = static_cast<std::uint32_t>(instruction.shift_distance);
    command.words[3] =
        (instruction.output_row == SxmInstruction::kAllOutputRows ? 0xffu
            : static_cast<std::uint32_t>(instruction.output_row))
        | ((instruction.input_row == SxmInstruction::kAllInputRows ? 0xffu
            : static_cast<std::uint32_t>(instruction.input_row)) << 8)
        | ((instruction.output_tile == SxmInstruction::kAllOutputTiles ? 0xffu
            : static_cast<std::uint32_t>(instruction.output_tile)) << 16);
    command.extension_words.push_back(static_cast<std::uint32_t>(instruction.src_streams.size()));
    command.extension_words.push_back(static_cast<std::uint32_t>(instruction.dst_streams.size()));
    for (const auto stream : instruction.src_streams)
        command.extension_words.push_back(static_cast<std::uint32_t>(stream.stream));
    for (const auto stream : instruction.dst_streams)
        command.extension_words.push_back(static_cast<std::uint32_t>(stream.stream));
    for (const auto lane : instruction.permute_map)
        command.extension_words.push_back(lane == SxmInstruction::kZeroFill
            ? UINT32_MAX : static_cast<std::uint32_t>(lane));
    return command;
}

QueueCommand control_command(isa::EncodedIcuCommand command)
{
    return QueueCommand {command, InstructionKind::None, 0, {}};
}

QueueCommand repeat_2d_command(const IcuRepeat2D& repeat)
{
    const auto encoded = isa::encode_icu_repeat_2d(repeat);
    return QueueCommand {
        encoded.words[0], InstructionKind::None, 3,
        {encoded.words[0], encoded.words[1], encoded.words[2], 0},
    };
}

std::vector<BinaryMemoryFloor> static_memory_floors(
    mlir::ModuleOp module, int64_t slices_per_hemisphere,
    int64_t banks_per_slice, int64_t rows_per_bank)
{
    std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> floors;
    const auto reserveAddress = [&](int64_t queue, int64_t address) {
        if (queue < 0)
            throw std::runtime_error(
                "Command IR MEM queue is outside the target");
        const int64_t queuesPerHemisphere =
            slices_per_hemisphere * banks_per_slice;
        const int64_t hemisphere = queue / queuesPerHemisphere;
        const int64_t localQueue = queue % queuesPerHemisphere;
        const int64_t slice = localQueue / banks_per_slice;
        const int64_t bank = localQueue % banks_per_slice;
        if (hemisphere >= 2)
            throw std::runtime_error(
                "Command IR MEM queue is outside the target");
        if (address < 0 || address >= rows_per_bank)
            throw std::runtime_error(
                "Command IR MEM scratch address is outside the target");
        auto& floor = floors[{hemisphere, slice, bank}];
        floor = std::max(floor, address + 1);
    };
    const auto reserveFloor = [&](int64_t queue, int64_t base,
                                  int64_t repeatCount,
                                  int64_t repeatStride,
                                  int64_t waveCount,
                                  int64_t waveStride) {
        for (int64_t repeat : {int64_t {0}, repeatCount - 1})
            for (int64_t wave : {int64_t {0}, waveCount - 1}) {
                const int64_t address = base
                    + repeat * repeatStride + wave * waveStride;
                reserveAddress(queue, address);
            }
    };
    module.walk([&](command::MemOp op) {
        if (op.getAddressBinding()) return;
        const auto integer = [&](llvm::StringRef name, int64_t fallback) {
            if (const auto attr =
                    op->getAttrOfType<mlir::IntegerAttr>(name))
                return attr.getInt();
            return fallback;
        };
        const int64_t base = op.getAddress();
        const int64_t repeat_count = op.getRepeatCount();
        const int64_t repeat_stride = op.getAddressStride();
        const int64_t wave_count = integer("wave_count", 1);
        const int64_t wave_stride =
            integer("wave_address_stride", 0);
        reserveFloor(op.getQueue(), base, repeat_count, repeat_stride,
            wave_count, wave_stride);
    });
    module.walk([&](command::MemBundleOp op) {
        if (op.getAddressBinding()) return;
        const int64_t waveCount = op.getWaveCount().value_or(1);
        const int64_t waveStride =
            op.getWaveAddressStride().value_or(0);
        for (std::size_t index = 0;
             index < op.getQueues().size(); ++index) {
            reserveFloor(
                llvm::cast<mlir::IntegerAttr>(
                    op.getQueues()[index]).getInt(),
                llvm::cast<mlir::IntegerAttr>(
                    op.getAddresses()[index]).getInt(),
                op.getRepeatCount(), op.getAddressStride(),
                waveCount, waveStride);
        }
    });
    module.walk([&](command::Mem3DOp op) {
        if (op.getAddressBinding()) return;
        const auto packet = raw_3d_packet<isa::EncodedMemIcu3DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mem_icu_3d_instruction(packet);

        // The blocked outer address is affine within each power-of-two
        // group. Its extrema therefore occur at the first or last point of
        // the first or final group; the two inner affine dimensions only
        // need their endpoints as well.
        const std::size_t lastOuter = instruction.loop.counts[2] - 1;
        const std::size_t groupSize =
            instruction.address.outer_group_size;
        const std::array<std::size_t, 4> outerCandidates {
            0,
            std::min(lastOuter, groupSize - 1),
            (lastOuter / groupSize) * groupSize,
            lastOuter,
        };
        for (const auto inner : {std::size_t {0},
                 instruction.loop.counts[0] - 1}) {
            for (const auto middle : {std::size_t {0},
                     instruction.loop.counts[1] - 1}) {
                for (const auto outer : outerCandidates) {
                    const auto address = ::ftlpu::detail::mem_icu_address_3d(
                        instruction,
                        IcuCoordinate3D {{inner, middle, outer}});
                    reserveAddress(op.getQueue(),
                        static_cast<int64_t>(address));
                }
            }
        }
    });
    module.walk([&](command::MemWriteRead2DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedMemIcuWriteRead2DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mem_icu_write_read_2d_instruction(packet);
        for (const auto i : {std::size_t {0}, instruction.counts[0] - 1})
            for (const auto j : {std::size_t {0}, instruction.counts[1] - 1})
                reserveAddress(op.getQueue(), static_cast<int64_t>(
                    instruction.base_address)
                    + static_cast<int64_t>(i) * instruction.address_strides[0]
                    + static_cast<int64_t>(j) * instruction.address_strides[1]);
    });
    std::vector<BinaryMemoryFloor> result;
    result.reserve(floors.size());
    for (const auto& entry : floors) {
        const auto [hemisphere, slice, bank] = entry.first;
        const int64_t floor = entry.second;
        result.push_back(BinaryMemoryFloor {
            static_cast<std::uint16_t>(hemisphere),
            static_cast<std::uint16_t>(slice),
            static_cast<std::uint32_t>(floor),
            static_cast<std::uint16_t>(bank),
        });
    }
    return result;
}

struct StreamReleaseSummary {
    std::vector<std::uint64_t> cycles;
};

StreamReleaseSummary stream_release_cycles(
    mlir::ModuleOp module, const LPUTargetModel& target)
{
    const int64_t streamCount = target.streams().streams_per_direction;
    const int64_t encodedStreamCount = target.streams().encoded_streams;
    if (streamCount <= 0 || encodedStreamCount != 2 * streamCount)
        throw std::runtime_error(
            "target must encode matching East and West ordinary streams");
    std::vector<std::uint64_t> releases(
        static_cast<std::size_t>(encodedStreamCount), 0);
    const int64_t fabricDrain = std::max<int64_t>(
        1, target.streams().system_register_columns);

    const auto sequenceEnd = [](mlir::Operation* operation,
                                 int64_t firstCycle) {
        int64_t end = firstCycle;
        const auto addAxis = [&](llvm::StringRef countName,
                                 llvm::StringRef intervalName) {
            const auto count =
                operation->getAttrOfType<mlir::IntegerAttr>(countName);
            const auto interval =
                operation->getAttrOfType<mlir::IntegerAttr>(intervalName);
            if (count && interval && count.getInt() > 1)
                end += (count.getInt() - 1) * interval.getInt();
        };
        addAxis("repeat_count", "repeat_interval");
        addAxis("wave_count", "wave_interval");
        addAxis("group_count", "group_interval");
        return end;
    };
    const auto mark = [&](int64_t stream, int64_t endCycle) {
        if (stream < 0 || stream >= encodedStreamCount)
            throw std::runtime_error(
                "Command IR stream is outside the target SR file");
        const auto release = static_cast<std::uint64_t>(
            std::max<int64_t>(0, endCycle + fabricDrain + 1));
        auto& current = releases[static_cast<std::size_t>(stream)];
        current = std::max(current, release);
    };
    const auto markRange = [&](int64_t base, int64_t count,
                               int64_t endCycle) {
        if (count < 0 || base < 0 || base + count > encodedStreamCount)
            throw std::runtime_error(
                "Command IR stream range is outside the target SR file");
        for (int64_t offset = 0; offset < count; ++offset)
            mark(base + offset, endCycle);
    };
    const auto markPacked = [&](int64_t packed, int64_t endCycle) {
        if (packed < 0 || packed >= encodedStreamCount)
            throw std::runtime_error(
                "Command IR packed stream is outside the target encoding");
        mark(packed, endCycle);
    };

    module.walk([&](command::MemOp op) {
        markPacked(op.getPackedStream(),
            sequenceEnd(op, command_cycle(op)));
    });
    module.walk([&](command::MemBundleOp op) {
        for (std::size_t index = 0; index < op.getCycles().size(); ++index) {
            const int64_t cycle = llvm::cast<mlir::IntegerAttr>(
                op.getCycles()[index]).getInt();
            const int64_t packed = llvm::cast<mlir::IntegerAttr>(
                op.getPackedStreams()[index]).getInt();
            markPacked(packed, sequenceEnd(op, cycle));
        }
    });
    module.walk([&](command::MxmOp op) {
        const int64_t end = sequenceEnd(op, command_cycle(op));
        if (op.getOpcode() == "iw") {
            const bool int8 = op.getWeightInputMode().value_or("direct16")
                == "int8_dequant_bf16";
            // CModel MXM weight ingress always consumes the East SR file.
            markRange(op.getWeightStreamBase().value_or(0),
                int8
                    ? target.throughput().mxm_int8_load_streams_per_cycle
                    : target.throughput().mxm_load_streams_per_cycle,
                end);
            return;
        }
        if (op.getOpcode() == "compute") {
            // MXM ingress is East-facing; accumulator output is West-facing.
            markRange(op.getActivationStreamBase() % streamCount,
                target.throughput().mxm_activation_streams, end);
            if (op.getAccumulatorDestination() == "stream")
                markRange(streamCount
                        + op.getOutputStreamBase() % streamCount,
                    target.throughput().mxm_result_streams, end);
            return;
        }
        if (op.getOpcode() == "accumulator_read")
            markRange(streamCount + op.getOutputStreamBase() % streamCount,
                target.throughput().mxm_result_streams, end);
    });
    module.walk([&](command::VxmOp op) {
        const int64_t end = sequenceEnd(op, command_cycle(op));
        const auto markOperand = [&](llvm::StringRef kind, int64_t index) {
            if (!kind.starts_with("stream_")) return;
            // VXM external operands are captured from the MEM West edge;
            // stream_source selects a hemisphere, not a travel direction.
            mark(streamCount + index % streamCount, end);
            mark(streamCount + (index + 1) % streamCount, end);
        };
        markOperand(op.getLhsKind(), op.getLhsIndex());
        markOperand(op.getRhsKind(), op.getRhsIndex());
        const int64_t output = op.getOutputStreamAttr().getInt();
        if (output < 0) return;
        const int64_t width = op.getCastTarget() == "i8" ? 1
            : op.getCastTarget() == "fp32" ? 4 : 2;
        // VXM outputs are injected at the MEM East edge.
        markRange(output % streamCount, width, end);
    });
    module.walk([&](command::SxmOp op) {
        const int64_t end = sequenceEnd(op, command_cycle(op));
        for (mlir::Attribute stream : op.getSourceStreams())
            markPacked(llvm::cast<mlir::IntegerAttr>(stream).getInt(), end);
        for (mlir::Attribute stream : op.getDestinationStreams())
            markPacked(llvm::cast<mlir::IntegerAttr>(stream).getInt(), end);
    });
    module.walk([&](command::Mem3DOp op) {
        const auto packet = raw_3d_packet<isa::EncodedMemIcu3DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mem_icu_3d_instruction(packet);
        markPacked(static_cast<int64_t>(instruction.stream),
            raw_loop_absolute_final_cycle(
                op.getOperation(), instruction.loop));
    });
    module.walk([&](command::MemWriteRead2DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedMemIcuWriteRead2DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mem_icu_write_read_2d_instruction(packet);
        const int64_t end = op.getCycle() + static_cast<int64_t>(
            ::ftlpu::detail::mem_icu_write_read_2d_last_issue_cycle(
                instruction));
        markPacked(static_cast<int64_t>(instruction.write_stream), end);
        for (std::size_t outer = 0; outer < instruction.counts[1]; ++outer)
            markPacked(static_cast<int64_t>(instruction.read_stream_base)
                + static_cast<int64_t>(outer)
                    * instruction.read_stream_outer_stride, end);
    });
    module.walk([&](command::MxmLoad3DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedMxmLoadIcu3DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mxm_load_icu_3d_instruction(packet);
        const int64_t streamWidth = instruction.weight_input_mode
                == MxmWeightInputMode::Int8DequantBf16
            ? target.throughput().mxm_int8_load_streams_per_cycle
            : target.throughput().mxm_load_streams_per_cycle;
        markRange(static_cast<int64_t>(instruction.weight_stream_base),
            streamWidth,
            raw_loop_absolute_final_cycle(
                op.getOperation(), instruction.loop));
    });
    module.walk([&](command::MxmCompute3DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedMxmComputeIcu3DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mxm_compute_icu_3d_instruction(packet);
        const int64_t end = raw_loop_absolute_final_cycle(
            op.getOperation(), instruction.loop);
        if (instruction.opcode == MxmComputeIcuOpcode::Compute3D) {
            markRange(static_cast<int64_t>(
                          instruction.activation_stream_base),
                target.throughput().mxm_activation_streams, end);
        }
        const bool regularStream =
            instruction.regular_mode.accumulator_destination
            == MxmAccumulatorDestination::Stream;
        const bool terminalStream = instruction.opcode
                == MxmComputeIcuOpcode::Compute3D
            && instruction.terminal_dimension
                < IcuLoop3D::kDimensions
            && instruction.terminal_mode.accumulator_destination
                == MxmAccumulatorDestination::Stream;
        if (regularStream || terminalStream)
            markRange(streamCount
                    + static_cast<int64_t>(
                        instruction.result_stream_base),
                target.throughput().mxm_result_streams, end);
    });
    module.walk([&](command::VxmRun2DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedVxmIcuRun2DPacket>(
            op.getOperation(), op.getWords());
        const auto run =
            isa::decode_vxm_icu_run_2d_instruction(packet);
        const auto decoded = isa::decode_vxm_instruction(
            static_cast<std::size_t>(op.getQueue()), run.instruction);
        const auto& instruction = decoded.instruction;
        const int64_t end = raw_loop_absolute_final_cycle(
            op.getOperation(), run.loop, instruction.repeat_count - 1);
        const auto markOperand = [&](const VxmLaneOperand& operand,
                                     bool rhsPort) {
            if (operand.kind != VxmLaneOperandKind::StreamFloat16
                && operand.kind != VxmLaneOperandKind::StreamBFloat16)
                return;
            const auto group = VxmLane::input_group_for_operand(
                static_cast<std::size_t>(op.getQueue()), rhsPort, operand);
            const int64_t packedBase = group < 8
                ? static_cast<int64_t>(group * 2)
                : streamCount
                    + static_cast<int64_t>((group - 8) * 2);
            markRange(packedBase, 2, end);
        };
        markOperand(instruction.lhs, false);
        markOperand(instruction.rhs, true);
        if (instruction.output_stream) {
            const auto block = VxmLane::block_for_stage(
                static_cast<std::size_t>(op.getQueue()));
            const int64_t packedBase = block < 4
                ? streamCount
                    + static_cast<int64_t>(*instruction.output_stream)
                : static_cast<int64_t>(*instruction.output_stream);
            const int64_t width =
                instruction.output_type == VxmCastTarget::Int8 ? 1
                : instruction.output_type == VxmCastTarget::Float32 ? 4
                : 2;
            markRange(packedBase, width, end);
        }
    });
    module.walk([&](command::SxmRun2DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedSxmIcuRun2DPacket>(
            op.getOperation(), op.getWords());
        const auto run =
            isa::decode_sxm_icu_run_2d_instruction(packet);
        const int64_t end = raw_loop_absolute_final_cycle(
            op.getOperation(), run.loop);
        for (const auto stream : run.instruction.src_streams)
            markPacked(static_cast<int64_t>(stream.stream), end);
        for (const auto stream : run.instruction.dst_streams)
            markPacked(static_cast<int64_t>(stream.stream), end);
    });
    return {std::move(releases)};
}

BindingLayout parse_layout(llvm::StringRef value)
{
    if (value == "vector") return BindingLayout::Vector;
    if (value == "mxm_weight_striped") return BindingLayout::MxmWeightStriped;
    if (value == "int32_byte_planar") return BindingLayout::Int32BytePlanar;
    if (value == "fp16_byte_planar") return BindingLayout::Fp16BytePlanar;
    if (value == "fp16_mxm_activation_planar") return BindingLayout::Fp16MxmActivationPlanar;
    if (value == "w8a16_mxm_weight_striped") return BindingLayout::W8A16MxmWeightStriped;
    if (value == "w8a16_mxm_weight_replicated")
        return BindingLayout::W8A16MxmWeightReplicated;
    if (value == "w8a16_mxm_weight_wave_striped")
        return BindingLayout::W8A16MxmWeightWaveStriped;
    if (value == "w8a16_attention_weight_striped")
        return BindingLayout::W8A16AttentionWeightStriped;
    if (value == "w8a16_native4_weight")
        return BindingLayout::W8A16Native4Weight;
    if (value == "fp16_pair_planar") return BindingLayout::Fp16PairPlanar;
    if (value == "fp32_causal_mask_tile")
        return BindingLayout::Fp32CausalMaskTile;
    if (value == "fp16_causal_mask_tile")
        return BindingLayout::Fp16CausalMaskTile;
    if (value == "fp16_sxm_distributed_16")
        return BindingLayout::Fp16SxmDistributed16;
    if (value == "fp16_vxm_distributed_16")
        return BindingLayout::Fp16VxmDistributed16;
    if (value == "fp16_vxm_row_parallel_8")
        return BindingLayout::Fp16VxmRowParallel8;
    if (value == "fp16_vxm_gamma_broadcast")
        return BindingLayout::Fp16VxmGammaBroadcast;
    if (value == "fp16_mxm_distributed_16")
        return BindingLayout::Fp16MxmDistributed16;
    if (value == "fp16_rope_table")
        return BindingLayout::Fp16RopeTable;
    if (value == "fp16_probability_x16")
        return BindingLayout::Fp16ProbabilityX16;
    if (value == "fp16_probability_diagonal")
        return BindingLayout::Fp16ProbabilityDiagonal;
    if (value == "fp16_value_x16")
        return BindingLayout::Fp16ValueX16;
    if (value == "fp16_head_planar")
        return BindingLayout::Fp16HeadPlanar;
    if (value == "fp16_head_block_packed")
        return BindingLayout::Fp16HeadBlockPacked;
    if (value == "fp16_projection_bias_x4")
        return BindingLayout::Fp16ProjectionBiasX4;
    throw std::runtime_error("unsupported Command IR binding layout");
}

BinaryBinding translate_binding(command::BindingOp op)
{
    BinaryBinding binding;
    binding.index = static_cast<std::uint32_t>(op.getIndex());
    binding.role = op.getRole().str();
    binding.name =
        op->getAttrOfType<mlir::StringAttr>("name").getValue().str();
    binding.ready_cycle = static_cast<std::uint64_t>(
        op->getAttrOfType<mlir::IntegerAttr>("ready_cycle").getInt());
    binding.access = op.getAccess() == "input" ? BindingAccess::Input
        : op.getAccess() == "output" ? BindingAccess::Output
        : BindingAccess::Internal;
    binding.element_type = op.getElementType() == "i8" ? BindingElementType::I8
        : op.getElementType() == "f16" ? BindingElementType::F16
        : op.getElementType() == "bf16" ? BindingElementType::BF16
        : op.getElementType() == "f32" ? BindingElementType::F32
        : BindingElementType::I32;
    binding.byte_size = static_cast<std::uint64_t>(op.getBytes());
    binding.layout = parse_layout(op.getPlacement().getAs<mlir::StringAttr>("kind").getValue());
    auto hemisphere = op.getPlacement().getAs<mlir::StringAttr>("hemisphere");
    binding.hemisphere_mask = !hemisphere || hemisphere.getValue() == "east" ? 1
        : hemisphere.getValue() == "west" ? 2 : 3;
    if (auto bank = op.getPlacement().getAs<mlir::IntegerAttr>("bank"))
        binding.bank = static_cast<std::uint16_t>(bank.getInt());
    binding.base_row = op.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt();
    binding.instruction_count = op.getPlacement().getAs<mlir::IntegerAttr>("instruction_count").getInt();
    binding.address_stride = op.getPlacement().getAs<mlir::IntegerAttr>("address_stride").getInt();
    const llvm::StringRef initializer = op.getInitializer();
    binding.initializer = initializer == "zero"
        ? software::runtime::BindingInitializer::Zero
        : initializer == "causal_mask"
        ? software::runtime::BindingInitializer::CausalMask
        : initializer == "rope_table"
        ? software::runtime::BindingInitializer::RopeTable
        : software::runtime::BindingInitializer::None;
    if (binding.initializer
        == software::runtime::BindingInitializer::RopeTable) {
        const auto config = op.getInitializerConfig();
        binding.rope_theta = static_cast<float>(
            config.getAs<mlir::FloatAttr>("theta").getValueAsDouble());
        binding.rope_head_dim = static_cast<std::uint32_t>(
            config.getAs<mlir::IntegerAttr>("head_dim").getInt());
    }
    for (mlir::Attribute dimension : op.getShape())
        binding.shape.push_back(static_cast<std::uint64_t>(
            llvm::cast<mlir::IntegerAttr>(dimension).getInt()));
    for (mlir::Attribute slice : op.getPlacement().getAs<mlir::ArrayAttr>("slices"))
        binding.slices.push_back(static_cast<std::uint16_t>(
            llvm::cast<mlir::IntegerAttr>(slice).getInt()));
    const auto placement = op.getPlacement();
    if (const auto paged = placement.getAs<mlir::BoolAttr>("paged_weight"))
        binding.paged_weight = paged.getValue();
    const auto copyPageInteger = [&](llvm::StringRef name,
                                     std::uint32_t& destination) {
        if (const auto value = placement.getAs<mlir::IntegerAttr>(name))
            destination = static_cast<std::uint32_t>(value.getInt());
    };
    copyPageInteger("page_count", binding.page_count);
    copyPageInteger("page_rows", binding.page_rows);
    copyPageInteger("page_granularity", binding.page_granularity);
    copyPageInteger("page_role_group_base",
        binding.page_role_group_base);
    copyPageInteger("page_role_group_count",
        binding.page_role_group_count);
    copyPageInteger("page_items_per_slice_group",
        binding.page_items_per_slice_group);
    copyPageInteger("page_bank_count", binding.page_bank_count);
    if (const auto storage =
            placement.getAs<mlir::ArrayAttr>("page_storage_slices"))
        for (mlir::Attribute slice : storage)
            binding.page_storage_slices.push_back(
                static_cast<std::uint16_t>(
                    llvm::cast<mlir::IntegerAttr>(slice).getInt()));
    const auto copyPageArray = [&]<typename T>(llvm::StringRef name,
                                   std::vector<T>& destination) {
        if (const auto values = placement.getAs<mlir::ArrayAttr>(name))
            for (mlir::Attribute value : values)
                destination.push_back(static_cast<T>(
                    llvm::cast<mlir::IntegerAttr>(value).getInt()));
    };
    copyPageArray("page_banks", binding.page_banks);
    copyPageArray("page_slice_group_bases",
        binding.page_slice_group_bases);
    copyPageArray("page_slice_group_counts",
        binding.page_slice_group_counts);
    copyPageArray("page_base_rows", binding.page_base_rows);
    copyPageArray("page_row_counts", binding.page_row_counts);
    return binding;
}

void collect_mem(command::MemOp op, QueueMap& queues)
{
    const int64_t queue = command_integer(op, "queue");
    const int64_t waveCount =
        static_cast<int64_t>(op.getWaveCount().value_or(1));
    const int64_t waveInterval =
        static_cast<int64_t>(op.getWaveInterval().value_or(1));
    const int64_t waveAddressStride =
        static_cast<int64_t>(op.getWaveAddressStride().value_or(0));
    const int64_t address = static_cast<int64_t>(op.getAddress());
    const auto instruction = op.getOpcode() == "read"
            ? MemInstruction::Read(address, op.getPackedStream())
            : op.getOpcode() == "write_tap"
            ? MemInstruction::WriteTap(address, op.getPackedStream())
            : MemInstruction::Write(address, op.getPackedStream());
    CommandSequence sequence {
            command_cycle(op),
            op->getAttrOfType<mlir::IntegerAttr>("repeat_count").getInt(),
            op->getAttrOfType<mlir::IntegerAttr>("repeat_interval").getInt(),
            op->getAttrOfType<mlir::IntegerAttr>("address_stride").getInt(),
            mem_instruction_command(
                isa::encode_mem_instruction(instruction)),
            -1,
            op.getAddressBinding()
                ? static_cast<int64_t>(*op.getAddressBinding()) : -1,
            -1,
            waveCount, waveInterval, waveAddressStride,
            IcuInductionTarget::MemAddress
        };
    sequence.address_binding_access = address_binding_access(op);
    queues[{QueueKind::Mem, queue}].push_back(std::move(sequence));
}

void collect_mem_bundle(command::MemBundleOp op, QueueMap& queues)
{
    const int64_t waveCount = op.getWaveCount().value_or(1);
    const int64_t waveInterval = op.getWaveInterval().value_or(1);
    const int64_t waveStride =
        op.getWaveAddressStride().value_or(0);
    for (std::size_t index = 0; index < op.getQueues().size(); ++index) {
        const int64_t cycle = llvm::cast<mlir::IntegerAttr>(
            op.getCycles()[index]).getInt();
        const int64_t queue = llvm::cast<mlir::IntegerAttr>(
            op.getQueues()[index]).getInt();
        const int64_t address = llvm::cast<mlir::IntegerAttr>(
            op.getAddresses()[index]).getInt();
        const int64_t packedStream = llvm::cast<mlir::IntegerAttr>(
            op.getPackedStreams()[index]).getInt();
        const auto instruction = op.getOpcode() == "read"
            ? MemInstruction::Read(address, packedStream)
            : op.getOpcode() == "write_tap"
            ? MemInstruction::WriteTap(address, packedStream)
            : MemInstruction::Write(address, packedStream);
        CommandSequence sequence {
            cycle, static_cast<int64_t>(op.getRepeatCount()),
            static_cast<int64_t>(op.getRepeatInterval()),
            static_cast<int64_t>(op.getAddressStride()),
            mem_instruction_command(
                isa::encode_mem_instruction(instruction)),
            -1,
            op.getAddressBinding()
                ? static_cast<int64_t>(*op.getAddressBinding()) : -1,
            -1, waveCount, waveInterval, waveStride,
            IcuInductionTarget::MemAddress,
        };
        sequence.address_binding_access = address_binding_access(op);
        queues[{QueueKind::Mem, queue}].push_back(std::move(sequence));
    }
}

void collect_mxm(command::MxmOp op, QueueMap& queues)
{
    const bool is_load = op.getOpcode() == "iw";
    const bool is_accumulator_read = op.getOpcode() == "accumulator_read";
    const auto destination = op.getAccumulatorDestination() == "stream"
        ? MxmAccumulatorDestination::Stream : MxmAccumulatorDestination::Sram;
    const auto inputMode =
        op.getWeightInputMode().value_or("direct16")
            == "int8_dequant_bf16"
        ? MxmWeightInputMode::Int8DequantBf16
        : MxmWeightInputMode::Direct16;
    const auto kind = is_load ? QueueKind::MxmLoad : QueueKind::MxmCompute;
    const int64_t waveCount = op.getWaveCount().value_or(1);
    const int64_t waveInterval = op.getWaveInterval().value_or(1);
    const int64_t waveColumnStride =
        op.getWaveWeightColumnStride().value_or(0);
    const int64_t waveAccumulatorStride =
        op.getWaveAccumulatorAddressStride().value_or(0);
    const int64_t waveInductionStride = waveColumnStride != 0
        ? waveColumnStride : waveAccumulatorStride;
    const IcuInductionTarget waveInductionTarget = waveColumnStride != 0
        ? IcuInductionTarget::MxmWeightColumn
        : waveAccumulatorStride != 0
        ? IcuInductionTarget::MxmAccumulatorAddress
        : IcuInductionTarget::None;
    const int64_t groupCount = op.getGroupCount().value_or(1);
    const int64_t groupInterval = op.getGroupInterval().value_or(1);
    int64_t innerCount = op.getRepeatCount();
    int64_t innerInterval = op.getRepeatInterval();
    int64_t innerStride = 0;
    int64_t outerCount = 1;
    int64_t outerInterval = 1;
    int64_t outerStride = 0;
    IcuInductionTarget inductionTarget = IcuInductionTarget::None;
    if (groupCount > 1 && waveCount > 1) {
        if (innerCount != 1)
            throw std::runtime_error(
                "Command IR MXM requires more than two iteration dimensions");
        innerCount = waveCount;
        innerInterval = waveInterval;
        innerStride = waveInductionStride;
        outerCount = groupCount;
        outerInterval = groupInterval;
        inductionTarget = waveInductionTarget;
    } else if (groupCount > 1) {
        outerCount = groupCount;
        outerInterval = groupInterval;
    } else if (waveCount > 1) {
        outerCount = waveCount;
        outerInterval = waveInterval;
        outerStride = waveInductionStride;
        inductionTarget = waveInductionTarget;
    }
    const int64_t weightColumn = op.getWeightColumn();
        const auto instruction = is_load
            ? op.getWeightLoadMode().value_or("supercell") == "column"
                ? MxmControlInstruction::IWColumn(
                    op.getWeightBuffer(), weightColumn,
                    op.getWeightInnerColumn().value_or(0), inputMode,
                    op.getWeightStreamBase().value_or(0))
                : MxmControlInstruction::IW(
                    op.getWeightBuffer(), weightColumn, inputMode,
                    op.getWeightStreamBase().value_or(0))
            : is_accumulator_read
            ? MxmControlInstruction::AccumulatorRead(
                op.getAccumulatorAddress(), op.getOutputStreamBase(),
                op.getAccumulatorClear(),
                op.getAccumulatorOutputFormat().value_or("fp32") == "bf16"
                    ? MxmAccumulatorOutputFormat::BFloat16
                    : MxmAccumulatorOutputFormat::Float32,
                destination)
            : MxmControlInstruction::Compute(op.getWeightBuffer(),
                op.getActivationStreamBase(), op.getOutputStreamBase(),
                op.getAccumulatorAddress(), op.getAccumulatorRowStride(),
                destination,
                op.getDataFormat().value_or("fp16") == "bf16"
                    ? MxmDataFormat::BFloat16
                    : MxmDataFormat::Float16,
                op.getAccumulatorClear(),
                op.getAccumulatorOutputFormat().value_or("fp32") == "bf16"
                    ? MxmAccumulatorOutputFormat::BFloat16
                    : MxmAccumulatorOutputFormat::Float32);
        queues[{kind, static_cast<int64_t>(op.getQueue())}]
            .push_back(CommandSequence {
                command_cycle(op), innerCount, innerInterval, innerStride,
                mxm_instruction_command(
                    isa::encode_mxm_instruction(instruction)),
                -1, -1, -1,
                outerCount, outerInterval, outerStride, inductionTarget,
            });
}

void collect_mxm_dequant(
    command::MxmDequantOp op, QueueMap& queues)
{
    const auto instruction = MxmDequantInstruction::Scale(
        static_cast<float>(op.getScaleAttr().getValueAsDouble()));
    queues[{QueueKind::MxmDequant,
            static_cast<int64_t>(op.getQueue())}]
            .push_back(CommandSequence {
                command_cycle(op),
                static_cast<int64_t>(op.getRepeatCount()),
                static_cast<int64_t>(op.getRepeatInterval()),
                0,
                mxm_dequant_instruction_command(
                    isa::encode_mxm_dequant_instruction(instruction)),
                op.getScaleBinding()
                    ? static_cast<int64_t>(*op.getScaleBinding()) : -1,
                -1, -1,
                static_cast<int64_t>(op.getWaveCount().value_or(1)),
                static_cast<int64_t>(op.getWaveInterval().value_or(1)),
            });
}

VxmLaneOperation parse_vxm_operation(llvm::StringRef value)
{
    if (value == "pass" || value == "bypass" || value == "cast")
        return VxmAluOpcode::Bypass;
    if (value == "add") return VxmAluOpcode::Add;
    if (value == "subtract") return VxmAluOpcode::Subtract;
    if (value == "multiply") return VxmAluOpcode::Multiply;
    if (value == "fma") return VxmAluOpcode::FusedMultiplyAdd;
    if (value == "fms") return VxmAluOpcode::FusedMultiplySubtract;
    if (value == "negate") return VxmAluOpcode::Negate;
    if (value == "max") return VxmAluOpcode::Max;
    if (value == "exp") return VxmSpecialAluOpcode::Exp;
    if (value == "reciprocal" || value == "divide")
        return VxmSpecialAluOpcode::Reciprocal;
    if (value == "rsqrt") return VxmSpecialAluOpcode::Rsqrt;
    throw std::runtime_error(
        "Command IR VXM operation is not implemented by the current CModel");
}

VxmStreamSource parse_vxm_stream_source(llvm::StringRef value)
{
    if (value.empty() || value == "local") return VxmStreamSource::Local;
    if (value == "east") return VxmStreamSource::East;
    if (value == "west") return VxmStreamSource::West;
    throw std::runtime_error(
        "Command IR VXM stream source must be local, east, or west");
}

VxmLaneOperand parse_vxm_operand(llvm::StringRef kind, int64_t index,
    float immediate, int64_t queue, llvm::StringRef streamSource)
{
    if (kind == "previous") return VxmLaneOperand::Previous();
    if (kind == "original") return VxmLaneOperand::Original();
    if (kind == "auxiliary") return VxmLaneOperand::Aux();
    if (kind == "accumulator") return VxmLaneOperand::Acc();
    if (kind == "feedback") return VxmLaneOperand::Feedback();
    if (kind == "alu") {
        if (index == queue - 1) return VxmLaneOperand::Previous();
        throw std::runtime_error(
            "arbitrary VXM alu(N) references require chain legalization");
    }
    const auto streamGroup = [&]() -> std::int32_t {
        if (index < 0 || index >= 64 || index % 2 != 0)
            throw std::runtime_error(
                "VXM 16-bit stream operand requires an even packed stream index");
        return static_cast<std::int32_t>(((index % 32) / 2) % 8);
    };
    if (kind == "stream_f16")
        return VxmLaneOperand::StreamFloat16(
            1.0f, streamGroup(), parse_vxm_stream_source(streamSource));
    if (kind == "stream_bf16")
        return VxmLaneOperand::StreamBFloat16(
            1.0f, streamGroup(), parse_vxm_stream_source(streamSource));
    if (kind == "immediate") return VxmLaneOperand::Imm(immediate);
    throw std::runtime_error(
        "legacy integer/FP32 VXM stream operands require BF16 legalization");
}

VxmCastTarget parse_vxm_cast_target(llvm::StringRef value)
{
    if (value == "fp32") return VxmCastTarget::Float32;
    if (value == "fp16") return VxmCastTarget::Float16;
    if (value == "bf16") return VxmCastTarget::BFloat16;
    if (value == "i8") return VxmCastTarget::Int8;
    throw std::runtime_error("unsupported Command IR VXM cast target");
}

void collect_vxm(command::VxmOp op, QueueMap& queues)
{
    const int64_t queue = op.getQueue();
    if (queue < 0 || queue >= InstructionControlUnit::kVxmQueues)
        throw std::runtime_error(
            "Command IR VXM queue exceeds the 8 compact control queues");
    const int64_t output_stream = op.getOutputStreamAttr().getInt();
    auto instruction = VxmLaneAluInstruction {};
    instruction.operation = parse_vxm_operation(op.getOpcode());
    const auto lhsSource =
        op->getAttrOfType<mlir::StringAttr>("lhs_stream_source");
    const auto rhsSource =
        op->getAttrOfType<mlir::StringAttr>("rhs_stream_source");
    instruction.lhs = parse_vxm_operand(op.getLhsKind(), op.getLhsIndex(),
        static_cast<float>(op.getLhsImmediateAttr().getValueAsDouble()), queue,
        lhsSource ? lhsSource.getValue() : llvm::StringRef{});
    instruction.rhs = parse_vxm_operand(op.getRhsKind(), op.getRhsIndex(),
        static_cast<float>(op.getRhsImmediateAttr().getValueAsDouble()), queue,
        rhsSource ? rhsSource.getValue() : llvm::StringRef{});
    instruction.output_type = parse_vxm_cast_target(op.getCastTarget());
    instruction.precision = VxmAluPrecision::Float32;
    instruction.repeat_count = static_cast<std::size_t>(op.getRepeatCount());
    instruction.accumulator_reset = op.getAccumulatorReset().value_or(false);
    instruction.accumulator_write = op.getAccumulatorWrite().value_or(false);
    instruction.accumulator_emit = op.getAccumulatorEmit().value_or(true);
    instruction.local_scalar_write = op.getLocalScalarWrite().value_or(false);
    if (output_stream >= 0)
        instruction.output_stream = static_cast<std::size_t>(output_stream);
    const auto depth = static_cast<VxmChainDepth>(
        op->getAttrOfType<mlir::IntegerAttr>("chain_depth")
            ? op->getAttrOfType<mlir::IntegerAttr>("chain_depth").getInt()
            : 8);
    try {
        queues[{QueueKind::Vxm, queue}].push_back(CommandSequence {
            command_cycle(op),
            1, 1, 0,
            vxm_instruction_command(
                isa::encode_vxm_instruction(queue, depth, instruction)),
            op.getScaleBinding()
                ? static_cast<int64_t>(*op.getScaleBinding()) : -1,
        });
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "invalid Command IR VXM instruction at cycle "
            + std::to_string(command_cycle(op)) + ", queue "
            + std::to_string(queue) + ", chain_depth "
            + std::to_string(static_cast<int64_t>(depth))
            + ", output_stream " + std::to_string(output_stream)
            + ": " + error.what());
    }
}

void collect_sxm(command::SxmOp op, QueueMap& queues)
{
    SxmInstruction instruction {};
    instruction.opcode = op.getOpcode() == "transpose"
        ? SxmOpcode::Transpose : SxmOpcode::Permute;
    if (op.getOutputRow())
        instruction.output_row = static_cast<std::size_t>(*op.getOutputRow());
    if (op.getInputRow())
        instruction.input_row = static_cast<std::size_t>(*op.getInputRow());
    if (op.getOutputTile())
        instruction.output_tile = static_cast<std::size_t>(*op.getOutputTile());
    for (mlir::Attribute stream : op.getSourceStreams())
        instruction.src_streams.push_back(SxmStreamId {static_cast<std::size_t>(
            llvm::cast<mlir::IntegerAttr>(stream).getInt())});
    for (mlir::Attribute stream : op.getDestinationStreams())
        instruction.dst_streams.push_back(SxmStreamId {static_cast<std::size_t>(
            llvm::cast<mlir::IntegerAttr>(stream).getInt())});
    for (std::size_t lane = 0; lane < instruction.permute_map.size(); ++lane) {
        const auto value = llvm::cast<mlir::IntegerAttr>(op.getPermuteMap()[lane]).getInt();
        instruction.permute_map[lane] = value < 0 ? SxmInstruction::kZeroFill
            : static_cast<std::size_t>(value);
    }
    const auto kind = instruction.opcode == SxmOpcode::Transpose
        ? QueueKind::SxmTranspose : QueueKind::SxmPermute;
    queues[{kind, static_cast<int64_t>(op.getHemisphere())}].push_back(CommandSequence {
        command_cycle(op),
        static_cast<int64_t>(op.getRepeatCount().value_or(1)),
        static_cast<int64_t>(op.getRepeatInterval().value_or(1)), 0,
        sxm_instruction_command(instruction),
    });
}

int64_t sequence_final_cycle(const CommandSequence& sequence)
{
    return sequence.cycle
        + (sequence.depth_count - 1) * sequence.depth_interval
        + (sequence.outer_count - 1) * sequence.outer_interval
        + (sequence.repeat_count - 1) * sequence.repeat_interval;
}

QueueCommand apply_outer_induction(
    QueueCommand command, IcuInductionTarget target, int64_t delta)
{
    if (target == IcuInductionTarget::None) return command;
    if (target == IcuInductionTarget::MemAddress) {
        const auto encoded = static_cast<isa::EncodedMemInstruction>(
                                 command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1])
                << 32);
        const auto instruction = ftlpu::detail::apply_icu_repeat_2d_stride(
            isa::decode_mem_instruction(encoded), target, delta);
        return mem_instruction_command(
            isa::encode_mem_instruction(instruction));
    }
    if (target == IcuInductionTarget::MxmWeightColumn
        || target == IcuInductionTarget::MxmAccumulatorAddress) {
        const auto encoded = static_cast<isa::EncodedMxmInstruction>(
                                 command.words[0])
            | (static_cast<isa::EncodedMxmInstruction>(command.words[1])
                << 32);
        const auto instruction = ftlpu::detail::apply_icu_repeat_2d_stride(
            isa::decode_mxm_instruction(encoded), target, delta);
        return mxm_instruction_command(
            isa::encode_mxm_instruction(instruction));
    }
    throw std::runtime_error("unsupported Repeat2D induction target");
}

void expand_interleaved_repeat_2d(
    std::vector<CommandSequence>& sequences, bool repeat2DEnabled)
{
    std::vector<bool> expand(sequences.size(), false);
    int64_t precedingEnd = std::numeric_limits<int64_t>::min();
    for (std::size_t index = 0; index < sequences.size(); ++index) {
        const auto& candidate = sequences[index];
        const int64_t candidateEnd = sequence_final_cycle(candidate);
        if (candidate.outer_count > 1) {
            const bool overlapsPreceding = precedingEnd >= candidate.cycle;
            const bool overlapsFollowing = index + 1 < sequences.size()
                && sequences[index + 1].cycle <= candidateEnd;
            expand[index] = !repeat2DEnabled
                || overlapsPreceding || overlapsFollowing;
        }
        precedingEnd = std::max(precedingEnd, candidateEnd);
    }
    if (std::find(expand.begin(), expand.end(), true) == expand.end())
        return;
    std::vector<CommandSequence> materialized;
    for (std::size_t index = 0; index < sequences.size(); ++index) {
        if (!expand[index]) {
            materialized.push_back(std::move(sequences[index]));
            continue;
        }
        for (int64_t outer = 0;
             outer < sequences[index].outer_count; ++outer) {
            auto item = sequences[index];
            item.cycle += outer * item.outer_interval;
            item.instruction = apply_outer_induction(
                std::move(item.instruction), item.induction_target,
                outer * item.outer_stride);
            item.outer_count = 1;
            item.outer_interval = 1;
            item.outer_stride = 0;
            // The induction target is shared by the inner and outer Repeat2D
            // dimensions. Expanding only the outer dimension must retain it
            // while the inner repeat still advances an encoded field.
            if (item.repeat_count <= 1 || item.address_stride == 0)
                item.induction_target = IcuInductionTarget::None;
            materialized.push_back(std::move(item));
        }
    }
    sequences = std::move(materialized);
}

void legalize_encoded_repeat_limits(
    std::vector<CommandSequence>& sequences)
{
    constexpr int64_t kRepeat2DCountMax = 1023;
    constexpr int64_t kRepeat2DIntervalMax = 65535;
    constexpr int64_t kRepeat2DStrideMin = -32768;
    constexpr int64_t kRepeat2DStrideMax = 32767;
    constexpr int64_t kRepeatCountMax = 1024;
    constexpr int64_t kRepeatIntervalMax = 255;
    constexpr int64_t kRepeatStrideMin = -2048;
    constexpr int64_t kRepeatStrideMax = 2047;

    const auto repeat2DEncodable = [&](const CommandSequence& sequence) {
        return sequence.repeat_count <= kRepeat2DCountMax
            && sequence.outer_count <= kRepeat2DCountMax
            && sequence.repeat_interval <= kRepeat2DIntervalMax
            && sequence.outer_interval <= kRepeat2DIntervalMax
            && sequence.address_stride >= kRepeat2DStrideMin
            && sequence.address_stride <= kRepeat2DStrideMax
            && sequence.outer_stride >= kRepeat2DStrideMin
            && sequence.outer_stride <= kRepeat2DStrideMax;
    };
    const auto repeatEncodable = [&](const CommandSequence& sequence) {
        return sequence.repeat_count <= kRepeatCountMax
            && sequence.repeat_interval <= kRepeatIntervalMax
            && sequence.address_stride >= kRepeatStrideMin
            && sequence.address_stride <= kRepeatStrideMax;
    };

    std::vector<CommandSequence> legalized;
    const auto appendInner = [&](CommandSequence base) {
        if (base.repeat_count <= 1 || repeatEncodable(base)) {
            legalized.push_back(std::move(base));
            return;
        }
        if (base.induction_target == IcuInductionTarget::None
            && base.address_stride != 0)
            throw std::runtime_error(
                "cannot materialize a repeated command without an induction target");
        const int64_t chunkLimit =
            base.repeat_interval <= kRepeatIntervalMax
                && base.address_stride >= kRepeatStrideMin
                && base.address_stride <= kRepeatStrideMax
            ? kRepeatCountMax : 1;
        for (int64_t start = 0; start < base.repeat_count;
             start += chunkLimit) {
            auto chunk = base;
            chunk.cycle += start * base.repeat_interval;
            chunk.instruction = apply_outer_induction(
                std::move(chunk.instruction), base.induction_target,
                start * base.address_stride);
            chunk.repeat_count = std::min(
                chunkLimit, base.repeat_count - start);
            if (chunk.repeat_count == 1) {
                chunk.repeat_interval = 1;
                chunk.address_stride = 0;
                chunk.induction_target = IcuInductionTarget::None;
            }
            legalized.push_back(std::move(chunk));
        }
    };

    for (const CommandSequence& sequence : sequences) {
        if (sequence.outer_count > 1
            && repeat2DEncodable(sequence)) {
            legalized.push_back(sequence);
            continue;
        }
        if (sequence.outer_count <= 1) {
            appendInner(sequence);
            continue;
        }
        if (sequence.induction_target == IcuInductionTarget::None
            && (sequence.address_stride != 0
                || sequence.outer_stride != 0))
            throw std::runtime_error(
                "cannot materialize Repeat2D without an induction target");
        for (int64_t outer = 0; outer < sequence.outer_count; ++outer) {
            auto row = sequence;
            row.cycle += outer * sequence.outer_interval;
            row.instruction = apply_outer_induction(
                std::move(row.instruction), sequence.induction_target,
                outer * sequence.outer_stride);
            row.outer_count = 1;
            row.outer_interval = 1;
            row.outer_stride = 0;
            appendInner(std::move(row));
        }
    }
    sequences = std::move(legalized);
}

bool same_affine_instruction(const CommandSequence& first,
    const CommandSequence& next, QueueKind kind, int64_t addressStride)
{
    if (first.scale_binding != next.scale_binding
        || first.address_binding != next.address_binding
        || first.address_binding_access != next.address_binding_access
        || first.write_address_binding != next.write_address_binding
        || first.write_address_binding_access
            != next.write_address_binding_access
        || first.instruction.instruction_kind
            != next.instruction.instruction_kind
        || first.instruction.word_count != next.instruction.word_count
        || first.instruction.extension_words
            != next.instruction.extension_words)
        return false;
    if (kind != QueueKind::Mem
        && kind != QueueKind::MxmLoad
        && kind != QueueKind::MxmCompute)
        return first.instruction.words == next.instruction.words;

    if (kind == QueueKind::MxmLoad
        || kind == QueueKind::MxmCompute) {
        const auto decode = [](const QueueCommand& command) {
            const auto encoded = static_cast<isa::EncodedMxmInstruction>(
                                     command.words[0])
                | (static_cast<isa::EncodedMxmInstruction>(
                       command.words[1])
                    << 32);
            return isa::decode_mxm_instruction(encoded);
        };
        try {
            auto expected = ftlpu::detail::apply_icu_repeat_stride(
                decode(first.instruction), addressStride, 1);
            return isa::encode_mxm_instruction(expected)
                == isa::encode_mxm_instruction(decode(next.instruction));
        } catch (const std::exception&) {
            return false;
        }
    }

    const auto decode = [](const QueueCommand& command) {
        const auto encoded = static_cast<isa::EncodedMemInstruction>(
                                 command.words[0])
            | (static_cast<isa::EncodedMemInstruction>(command.words[1])
                << 32);
        return isa::decode_mem_instruction(encoded);
    };
    const auto lhs = decode(first.instruction);
    const auto rhs = decode(next.instruction);
    return lhs.opcode == rhs.opcode
        && lhs.stream == rhs.stream
        && lhs.map_stream == rhs.map_stream
        && lhs.preserve_stream == rhs.preserve_stream
        && static_cast<int64_t>(rhs.address)
            == static_cast<int64_t>(lhs.address) + addressStride;
}

IcuInductionTarget macro_induction_target(QueueKind kind)
{
    if (kind == QueueKind::Mem)
        return IcuInductionTarget::MemAddress;
    if (kind == QueueKind::MxmLoad)
        return IcuInductionTarget::MxmWeightColumn;
    if (kind == QueueKind::MxmCompute)
        return IcuInductionTarget::MxmAccumulatorAddress;
    return IcuInductionTarget::None;
}

IcuStreamNdSchedule canonicalize_stream_nd_dimensions(
    IcuStreamNdSchedule schedule)
{
    std::array<std::size_t, IcuStreamNdSchedule::kMaxRank> order {
        0, 1, 2};
    std::stable_sort(order.begin(), order.begin() + schedule.rank,
        [&](std::size_t lhs, std::size_t rhs) {
            return schedule.cycle_strides[lhs]
                < schedule.cycle_strides[rhs];
        });

    const auto counts = schedule.counts;
    const auto cycleStrides = schedule.cycle_strides;
    const auto operandStrides = schedule.operand_strides;
    for (std::size_t dimension = 0; dimension < schedule.rank;
         ++dimension) {
        const auto source = order[dimension];
        schedule.counts[dimension] = counts[source];
        schedule.cycle_strides[dimension] = cycleStrides[source];
        schedule.operand_strides[dimension] = operandStrides[source];
    }
    software::runtime::validate_stream_nd_iteration_space(
        schedule, "STREAM_ND affine schedule");
    return schedule;
}

void expand_control_sequences(std::vector<CommandSequence>& sequences)
{
    // Materialize both Repeat dimensions into native instructions. NOP gaps
    // remain duration encoded so the baseline measures functional compression.
    std::vector<CommandSequence> expanded;
    for (const CommandSequence& sequence : sequences) {
        for (int64_t outer = 0; outer < sequence.outer_count; ++outer)
            for (int64_t inner = 0;
                 inner < sequence.repeat_count; ++inner) {
                CommandSequence item = sequence;
                item.cycle += outer * sequence.outer_interval
                    + inner * sequence.repeat_interval;
                item.instruction = apply_outer_induction(
                    std::move(item.instruction),
                    sequence.induction_target,
                    outer * sequence.outer_stride
                        + inner * sequence.address_stride);
                item.repeat_count = 1;
                item.repeat_interval = 1;
                item.address_stride = 0;
                item.outer_count = 1;
                item.outer_interval = 1;
                item.outer_stride = 0;
                item.induction_target = IcuInductionTarget::None;
                expanded.push_back(std::move(item));
            }
    }
    std::sort(expanded.begin(), expanded.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.cycle < rhs.cycle;
        });
    sequences = std::move(expanded);
}

int64_t macro_instruction_stride(const CommandSequence& first,
    const CommandSequence& next, QueueKind kind)
{
    if (kind == QueueKind::Mem) {
        const auto decodeAddress = [](const QueueCommand& command) {
            const auto encoded =
                static_cast<isa::EncodedMemInstruction>(command.words[0])
                | (static_cast<isa::EncodedMemInstruction>(
                       command.words[1])
                    << 32);
            return static_cast<int64_t>(
                isa::decode_mem_instruction(encoded).address);
        };
        return decodeAddress(next.instruction)
            - decodeAddress(first.instruction);
    }
    if (kind == QueueKind::MxmLoad
        || kind == QueueKind::MxmCompute) {
        const auto decode = [](const QueueCommand& command) {
            const auto encoded =
                static_cast<isa::EncodedMxmInstruction>(command.words[0])
                | (static_cast<isa::EncodedMxmInstruction>(
                       command.words[1])
                    << 32);
            return isa::decode_mxm_instruction(encoded);
        };
        const auto lhs = decode(first.instruction);
        const auto rhs = decode(next.instruction);
        if (lhs.opcode != rhs.opcode)
            return std::numeric_limits<int64_t>::max();
        if (kind == QueueKind::MxmLoad
            && lhs.opcode == MxmControlOpcode::IW)
            return static_cast<int64_t>(rhs.weight_column)
                - static_cast<int64_t>(lhs.weight_column);
        if (kind == QueueKind::MxmCompute
            && (lhs.opcode == MxmControlOpcode::Compute
                || lhs.opcode == MxmControlOpcode::AccumulatorRead))
            return static_cast<int64_t>(rhs.accumulator_address)
                - static_cast<int64_t>(lhs.accumulator_address);
        return isa::encode_mxm_instruction(lhs)
                == isa::encode_mxm_instruction(rhs)
            ? 0 : std::numeric_limits<int64_t>::max();
    }
    return first.instruction.words == next.instruction.words ? 0
        : std::numeric_limits<int64_t>::max();
}

void compress_interleaved_macro_windows(
    std::vector<CommandSequence>& sequences, QueueKind kind)
{
    // The physical ICU has one active loop state. With a depth of one this
    // pass can only fold consecutive same-body descriptors into an outer
    // loop; it cannot construct an interleaved launch window.
    constexpr std::size_t maxWindow = 1;
    std::vector<CommandSequence> compressed;
    compressed.reserve(sequences.size());
    for (std::size_t index = 0; index < sequences.size();) {
        std::size_t bestWindow = 0;
        std::size_t bestRounds = 0;
        int64_t bestInterval = 0;
        int64_t bestStride = 0;
        const std::size_t remaining = sequences.size() - index;
        for (std::size_t window = 1;
             window <= std::min(maxWindow, remaining / 2); ++window) {
            const auto& first = sequences[index];
            const auto& next = sequences[index + window];
            if (first.outer_count != 1 || next.outer_count != 1)
                continue;
            const int64_t interval = next.cycle - first.cycle;
            if (interval <= 0
                || interval
                    <= (first.repeat_count - 1)
                        * first.repeat_interval)
                continue;
            const int64_t stride =
                macro_instruction_stride(first, next, kind);
            if (stride == std::numeric_limits<int64_t>::max()) continue;

            std::size_t rounds = 1;
            while (index + (rounds + 1) * window <= sequences.size()) {
                bool same = true;
                for (std::size_t offset = 0; offset < window; ++offset) {
                    const auto& base = sequences[index + offset];
                    const auto& candidate =
                        sequences[index + rounds * window + offset];
                    if (base.outer_count != 1
                        || candidate.outer_count != 1
                        || candidate.repeat_count != base.repeat_count
                        || candidate.repeat_interval
                            != base.repeat_interval
                        || candidate.address_stride
                            != base.address_stride
                        || candidate.induction_target
                            != base.induction_target
                        || candidate.cycle
                            != base.cycle
                                + static_cast<int64_t>(rounds)
                                    * interval
                        || !same_affine_instruction(base, candidate, kind,
                            static_cast<int64_t>(rounds) * stride)) {
                        same = false;
                        break;
                    }
                }
                if (!same) break;
                ++rounds;
            }
            if (rounds > 1
                && rounds * window > bestRounds * bestWindow) {
                bestWindow = window;
                bestRounds = rounds;
                bestInterval = interval;
                bestStride = stride;
            }
        }
        if (bestRounds <= 1) {
            compressed.push_back(std::move(sequences[index++]));
            continue;
        }
        for (std::size_t offset = 0; offset < bestWindow; ++offset) {
            auto sequence = std::move(sequences[index + offset]);
            sequence.outer_count = static_cast<int64_t>(bestRounds);
            sequence.outer_interval = bestInterval;
            sequence.outer_stride = bestStride;
            if (bestStride != 0)
                sequence.induction_target = macro_induction_target(kind);
            compressed.push_back(std::move(sequence));
        }
        index += bestRounds * bestWindow;
    }
    sequences = std::move(compressed);
}

// STREAM_ND has one more affine counter than the legacy macro. Fold repeated
// two-dimensional descriptors into that third dimension while preserving any
// interleaved descriptor window on the same functional-unit queue.
void compress_stream_nd_depth(
    std::vector<CommandSequence>& sequences, QueueKind kind)
{
    // A hardware ICU owns one live loop context, so only a consecutive
    // same-body sequence can be folded into the depth dimension.
    constexpr std::size_t maxWindow = 1;
    std::vector<CommandSequence> compressed;
    compressed.reserve(sequences.size());
    for (std::size_t index = 0; index < sequences.size();) {
        std::size_t bestWindow = 0;
        std::size_t bestRounds = 0;
        int64_t bestInterval = 0;
        int64_t bestStride = 0;
        const std::size_t remaining = sequences.size() - index;
        for (std::size_t window = 1;
             window <= std::min(maxWindow, remaining / 2); ++window) {
            const auto& first = sequences[index];
            const auto& next = sequences[index + window];
            if (first.depth_count != 1 || next.depth_count != 1)
                continue;
            const int64_t interval = next.cycle - first.cycle;
            const int64_t span =
                (first.outer_count - 1) * first.outer_interval
                + (first.repeat_count - 1) * first.repeat_interval;
            if (interval <= span) continue;
            const int64_t stride =
                macro_instruction_stride(first, next, kind);
            if (stride == std::numeric_limits<int64_t>::max()) continue;

            std::size_t rounds = 1;
            while (index + (rounds + 1) * window <= sequences.size()) {
                bool same = true;
                for (std::size_t offset = 0; offset < window; ++offset) {
                    const auto& base = sequences[index + offset];
                    const auto& candidate =
                        sequences[index + rounds * window + offset];
                    if (base.depth_count != 1
                        || candidate.depth_count != 1
                        || candidate.repeat_count != base.repeat_count
                        || candidate.repeat_interval
                            != base.repeat_interval
                        || candidate.address_stride != base.address_stride
                        || candidate.outer_count != base.outer_count
                        || candidate.outer_interval != base.outer_interval
                        || candidate.outer_stride != base.outer_stride
                        || candidate.induction_target
                            != base.induction_target
                        || candidate.cycle
                            != base.cycle
                                + static_cast<int64_t>(rounds) * interval
                        || !same_affine_instruction(base, candidate,
                            kind,
                            static_cast<int64_t>(rounds) * stride)) {
                        same = false;
                        break;
                    }
                }
                if (!same) break;
                ++rounds;
            }
            if (rounds > 1
                && rounds * window > bestRounds * bestWindow) {
                bestWindow = window;
                bestRounds = rounds;
                bestInterval = interval;
                bestStride = stride;
            }
        }
        if (bestRounds <= 1) {
            compressed.push_back(std::move(sequences[index++]));
            continue;
        }
        for (std::size_t offset = 0; offset < bestWindow; ++offset) {
            auto sequence = std::move(sequences[index + offset]);
            sequence.depth_count = static_cast<int64_t>(bestRounds);
            sequence.depth_interval = bestInterval;
            sequence.depth_stride = bestStride;
            if (bestStride != 0) {
                const auto inductionTarget = macro_induction_target(kind);
                if (inductionTarget == IcuInductionTarget::None)
                    throw std::runtime_error(
                        "STREAM_ND depth stride is unsupported for this queue kind");
                if (sequence.induction_target
                        != IcuInductionTarget::None
                    && sequence.induction_target != inductionTarget)
                    throw std::runtime_error(
                        "STREAM_ND depth stride conflicts with the existing induction target");
                sequence.induction_target = inductionTarget;
            }
            compressed.push_back(std::move(sequence));
        }
        index += bestRounds * bestWindow;
    }
    sequences = std::move(compressed);
}

bool is_native4_decode(command::MxmOp op)
{
    return (op.getOpcode() == "decode_load_activation"
            || op.getOpcode() == "decode_stream_compute")
        && op.getDecodeLayout().value_or("") == "native4";
}

void collect_native4_decode(
    command::MxmOp op, Raw3DQueueMap& queues)
{
    const bool load = op.getOpcode() == "decode_load_activation";
    const auto format = op.getDataFormat().value_or("bf16") == "bf16"
        ? MxmDataFormat::BFloat16 : MxmDataFormat::Float16;
    const auto destination = op.getAccumulatorDestination() == "stream"
        ? MxmAccumulatorDestination::Stream
        : MxmAccumulatorDestination::Sram;
    const auto instruction = load
        ? MxmControlInstruction::DecodeLoadActivation(
            static_cast<std::size_t>(op.getWeightBuffer()),
            static_cast<std::size_t>(op.getActivationStreamBase()), format,
            MxmDecodeLayout::Native4x4)
        : MxmControlInstruction::DecodeStreamCompute(
            static_cast<std::size_t>(op.getWeightBuffer()),
            static_cast<std::size_t>(op.getOutputStreamBase()), format,
            static_cast<std::size_t>(op.getAccumulatorAddress()),
            static_cast<std::size_t>(op.getWeightColumn()), destination,
            op.getAccumulatorClear(), MxmDecodeLayout::Native4x4);
    const std::size_t repeatCount =
        static_cast<std::size_t>(op.getRepeatCount());
    const std::size_t waveCount = static_cast<std::size_t>(
        op.getWaveCount().value_or(1));
    const std::size_t groupCount = static_cast<std::size_t>(
        op.getGroupCount().value_or(1));
    const std::size_t repeatInterval =
        static_cast<std::size_t>(op.getRepeatInterval());
    const std::size_t waveInterval = static_cast<std::size_t>(
        op.getWaveInterval().value_or(1));
    const std::size_t groupInterval = static_cast<std::size_t>(
        op.getGroupInterval().value_or(1));
    const int64_t repeatStride = op->getAttrOfType<mlir::IntegerAttr>(
        "repeat_accumulator_address_stride")
        ? op->getAttrOfType<mlir::IntegerAttr>(
              "repeat_accumulator_address_stride").getInt()
        : 0;
    const int64_t waveStride =
        op.getWaveAccumulatorAddressStride().value_or(0);
    const auto induction = repeatStride != 0 || waveStride != 0
        ? IcuInductionTarget::MxmAccumulatorAddress
        : IcuInductionTarget::None;
    IcuMxmStreamNdSchedule schedule {
        static_cast<std::size_t>(command_cycle(op)), 3,
        {repeatCount, waveCount, groupCount},
        {repeatInterval, waveInterval, groupInterval},
        {repeatStride, waveStride, 0}, induction};
    auto encoded = software::runtime::encode_mxm_stream_nd_command(
        mxm_instruction_command(isa::encode_mxm_instruction(instruction)),
        schedule);
    const std::size_t start = static_cast<std::size_t>(command_cycle(op));
    const std::size_t final = start
        + (repeatCount - 1) * repeatInterval
        + (waveCount - 1) * waveInterval
        + (groupCount - 1) * groupInterval;
    const auto kind = load ? QueueKind::MxmLoad : QueueKind::MxmCompute;
    queues[{kind, static_cast<int64_t>(op.getQueue())}].push_back(
        Raw3DPacketSequence {start, final, {std::move(encoded)}, -1,
            BindingAccess::Input});
}

QueueProgram encode_queue(const QueueKey& key, std::vector<CommandSequence> sequences,
    std::size_t& max_cycle,
    std::vector<BinaryScaleRelocation>& scaleRelocations,
    std::vector<BinaryAddressRelocation>& addressRelocations,
    bool repeat2DEnabled,
    IcuCompressionMode compressionMode,
    bool memSliceProgramEnabled)
{
    const bool controlCompressionEnabled =
        compressionMode != IcuCompressionMode::None;
    const bool macroScheduleEnabled =
        compressionMode == IcuCompressionMode::Macro;
    std::sort(sequences.begin(), sequences.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });
    if (!controlCompressionEnabled)
        expand_control_sequences(sequences);
    else if (!macroScheduleEnabled)
        expand_interleaved_repeat_2d(sequences, repeat2DEnabled);
    std::sort(sequences.begin(), sequences.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });
    if (key.first == QueueKind::Mem) {
        const auto decodeMem = [](const CommandSequence& sequence) {
            const auto encoded =
                static_cast<isa::EncodedMemInstruction>(
                    sequence.instruction.words[0])
                | (static_cast<isa::EncodedMemInstruction>(
                       sequence.instruction.words[1])
                    << 32);
            return isa::decode_mem_instruction(encoded);
        };
        std::unordered_set<int64_t> readCycles;
        std::unordered_set<int64_t> writeCycles;
        using RepeatShape = std::tuple<int64_t, int64_t, int64_t,
            int64_t, int64_t, int64_t, int64_t>;
        std::set<RepeatShape> readShapes;
        std::set<RepeatShape> writeShapes;
        for (const auto& sequence : sequences) {
            const auto instruction = decodeMem(sequence);
            auto* cycles = instruction.opcode == MemOpcode::Read
                ? &readCycles
                : instruction.opcode == MemOpcode::Write
                ? &writeCycles
                : nullptr;
            if (!cycles) continue;
            auto& shapes = instruction.opcode == MemOpcode::Read
                ? readShapes : writeShapes;
            shapes.emplace(sequence.cycle, sequence.repeat_count,
                sequence.repeat_interval, sequence.address_stride,
                sequence.outer_count, sequence.outer_interval,
                sequence.outer_stride);
            for (int64_t outer = 0;
                 outer < sequence.outer_count; ++outer)
                for (int64_t repeat = 0;
                     repeat < sequence.repeat_count; ++repeat)
                    cycles->insert(sequence.cycle
                        + outer * sequence.outer_interval
                        + repeat * sequence.repeat_interval);
        }
        std::vector<bool> expandSequence(sequences.size(), false);
        for (std::size_t index = 0; index < sequences.size(); ++index) {
            const auto instruction = decodeMem(sequences[index]);
            if (sequences[index].repeat_count <= 1
                || (instruction.opcode != MemOpcode::Read
                    && instruction.opcode != MemOpcode::Write))
                continue;
            const RepeatShape shape {sequences[index].cycle,
                sequences[index].repeat_count,
                sequences[index].repeat_interval,
                sequences[index].address_stride,
                sequences[index].outer_count,
                sequences[index].outer_interval,
                sequences[index].outer_stride};
            const bool hasDirectMate =
                instruction.opcode == MemOpcode::Read
                ? writeShapes.contains(shape)
                : readShapes.contains(shape);
            if (hasDirectMate) continue;
            const auto& oppositeCycles =
                instruction.opcode == MemOpcode::Read
                ? writeCycles : readCycles;
            for (int64_t outer = 0;
                 outer < sequences[index].outer_count
                    && !expandSequence[index]; ++outer)
                for (int64_t repeat = 0;
                     repeat < sequences[index].repeat_count; ++repeat) {
                    const int64_t cycle = sequences[index].cycle
                        + outer * sequences[index].outer_interval
                        + repeat * sequences[index].repeat_interval;
                    if (oppositeCycles.contains(cycle)) {
                        expandSequence[index] = true;
                        break;
                    }
                }
        }
        if (std::find(expandSequence.begin(), expandSequence.end(), true)
            != expandSequence.end()) {
            std::vector<CommandSequence> expanded;
            for (std::size_t index = 0; index < sequences.size(); ++index) {
                if (!expandSequence[index]) {
                    expanded.push_back(std::move(sequences[index]));
                    continue;
                }
                const auto instruction = decodeMem(sequences[index]);
                for (int64_t outer = 0;
                     outer < sequences[index].outer_count; ++outer)
                    for (int64_t repeat = 0;
                         repeat < sequences[index].repeat_count; ++repeat) {
                        CommandSequence item = sequences[index];
                        item.cycle += outer * sequences[index].outer_interval
                            + repeat * sequences[index].repeat_interval;
                        item.repeat_count = 1;
                        item.repeat_interval = 1;
                        item.address_stride = 0;
                        item.outer_count = 1;
                        item.outer_interval = 1;
                        item.outer_stride = 0;
                        item.induction_target = IcuInductionTarget::None;
                        const int64_t address = instruction.address
                            + outer * sequences[index].outer_stride
                            + repeat * sequences[index].address_stride;
                        const auto single = instruction.opcode == MemOpcode::Read
                            ? MemInstruction::Read(
                                  address, instruction.stream_id())
                            : instruction.preserve_stream
                            ? MemInstruction::WriteTap(
                                  address, instruction.stream_id())
                            : MemInstruction::Write(
                                  address, instruction.stream_id());
                        item.instruction = mem_instruction_command(
                            isa::encode_mem_instruction(single));
                        expanded.push_back(std::move(item));
                    }
            }
            sequences = std::move(expanded);
            std::sort(sequences.begin(), sequences.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.cycle < rhs.cycle;
            });
        }
    }
    const bool macroKind = key.first == QueueKind::Mem
        || key.first == QueueKind::MxmLoad
        || key.first == QueueKind::MxmCompute
        || key.first == QueueKind::MxmDequant
        || key.first == QueueKind::Vxm
        || key.first == QueueKind::SxmTranspose
        || key.first == QueueKind::SxmPermute;
    const bool macroQueue = macroScheduleEnabled && macroKind;
    if (macroQueue) {
        // This compatibility encoder may only fold already-serial affine
        // regions.  It must never repair an invalid schedule by exposing loop
        // coordinates: hardware owns one active context, so an interleaved
        // schedule has to be rejected and regenerated by direct FU lowering.
        std::stable_sort(sequences.begin(), sequences.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cycle < rhs.cycle;
            });
        compress_interleaved_macro_windows(sequences, key.first);
        std::stable_sort(sequences.begin(), sequences.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cycle < rhs.cycle;
            });
        compress_stream_nd_depth(sequences, key.first);
        std::stable_sort(sequences.begin(), sequences.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cycle < rhs.cycle;
            });

        int64_t activeUntil = std::numeric_limits<int64_t>::min();
        for (const auto& sequence : sequences) {
            if (sequence.cycle <= activeUntil)
                throw std::runtime_error(
                    "direct FU lowering required: legacy Schedule IR produced "
                    "overlapping coarse instructions for single-context ICU "
                    "queue kind="
                    + std::to_string(static_cast<int>(key.first))
                    + " index=" + std::to_string(key.second)
                    + " start=" + std::to_string(sequence.cycle)
                    + " active_until=" + std::to_string(activeUntil));
            activeUntil = sequence_final_cycle(sequence);
        }
    }
    if (macroQueue) {
        QueueProgram queue {
            key.first, static_cast<std::size_t>(key.second), {}};
        std::unordered_set<int64_t> issueCycles;
        const auto validateIssueCycles = [&](const CommandSequence& sequence) {
            for (int64_t depth = 0; depth < sequence.depth_count;
                 ++depth)
                for (int64_t outer = 0; outer < sequence.outer_count;
                     ++outer)
                    for (int64_t inner = 0;
                         inner < sequence.repeat_count; ++inner) {
                        const int64_t issueCycle = sequence.cycle
                            + depth * sequence.depth_interval
                            + outer * sequence.outer_interval
                            + inner * sequence.repeat_interval;
                        if (!issueCycles.insert(issueCycle).second)
                            throw std::runtime_error(
                                "overlapping Command IR coarse issue on ICU queue kind="
                                + std::to_string(static_cast<int>(key.first))
                                + " index=" + std::to_string(key.second)
                                + " at cycle=" + std::to_string(issueCycle));
                    }
        };

        if (key.first == QueueKind::Mem) {
            constexpr int64_t kMaxProgramCycleOffset = 65535;
            const auto scheduleFor = [](const CommandSequence& sequence) {
                const std::size_t rank = sequence.depth_count > 1 ? 3
                    : sequence.outer_count > 1 ? 2 : 1;
                return IcuMemStreamNdSchedule {
                    static_cast<std::size_t>(sequence.cycle),
                    rank,
                    {static_cast<std::size_t>(sequence.repeat_count),
                        static_cast<std::size_t>(sequence.outer_count),
                        static_cast<std::size_t>(sequence.depth_count)},
                    {static_cast<std::size_t>(sequence.repeat_interval),
                        static_cast<std::size_t>(sequence.outer_interval),
                        static_cast<std::size_t>(sequence.depth_interval)},
                    {sequence.address_stride, sequence.outer_stride,
                        sequence.depth_stride},
                    IcuInductionTarget::MemAddress,
                };
            };
            const auto sameDomain = [&](const CommandSequence& lhs,
                                        const CommandSequence& rhs) {
                const auto left = scheduleFor(lhs);
                const auto right = scheduleFor(rhs);
                return left.rank == right.rank
                    && left.counts == right.counts
                    && left.cycle_strides == right.cycle_strides
                    && lhs.scale_binding == rhs.scale_binding
                    && lhs.address_binding == rhs.address_binding
                    && lhs.address_binding_access
                        == rhs.address_binding_access
                    && lhs.write_address_binding
                        == rhs.write_address_binding
                    && lhs.write_address_binding_access
                        == rhs.write_address_binding_access;
            };
            const auto decodeMem = [](const QueueCommand& command) {
                const auto encoded =
                    static_cast<isa::EncodedMemInstruction>(
                        command.words[0])
                    | (static_cast<isa::EncodedMemInstruction>(
                           command.words[1])
                        << 32);
                return isa::decode_mem_instruction(encoded);
            };

            if (!memSliceProgramEnabled) {
                for (const auto& sequence : sequences) {
                    validateIssueCycles(sequence);
                    max_cycle = std::max(max_cycle,
                        static_cast<std::size_t>(
                            sequence_final_cycle(sequence)));
                    const std::size_t instructionIndex =
                        queue.commands.size();
                    queue.commands.push_back(
                        software::runtime::encode_mem_stream_nd_command(
                            sequence.instruction, scheduleFor(sequence)));
                    if (sequence.address_binding >= 0) {
                        addressRelocations.push_back(
                            BinaryAddressRelocation {
                                static_cast<std::uint32_t>(
                                    sequence.address_binding),
                                sequence.address_binding_access,
                                key.first,
                                static_cast<std::uint16_t>(key.second),
                                static_cast<std::uint32_t>(
                                    instructionIndex),
                                false,
                            });
                    }
                    if (sequence.write_address_binding >= 0) {
                        addressRelocations.push_back(
                            BinaryAddressRelocation {
                                static_cast<std::uint32_t>(
                                    sequence.write_address_binding),
                                sequence.write_address_binding_access,
                                key.first,
                                static_cast<std::uint16_t>(key.second),
                                static_cast<std::uint32_t>(
                                    instructionIndex),
                                true,
                            });
                    }
                }
                return queue;
            }

            for (const auto& sequence : sequences) {
                validateIssueCycles(sequence);
                max_cycle = std::max(max_cycle,
                    static_cast<std::size_t>(
                        sequence_final_cycle(sequence)));
            }

            std::vector<bool> consumed(sequences.size(), false);
            for (std::size_t seed = 0; seed < sequences.size(); ++seed) {
                if (consumed[seed]) continue;
                consumed[seed] = true;
                std::vector<std::size_t> members {seed};
                for (std::size_t candidate = seed + 1;
                     candidate < sequences.size()
                     && members.size()
                         < IcuMemSliceProgram::kMaxBodyEntries;
                     ++candidate) {
                    if (consumed[candidate]
                        || sequences[candidate].cycle
                                - sequences[seed].cycle
                            > kMaxProgramCycleOffset
                        || !sameDomain(
                            sequences[seed], sequences[candidate]))
                        continue;
                    consumed[candidate] = true;
                    members.push_back(candidate);
                }

                // A one-entry slice program carries an eleven-word shared
                // header and is larger than the equivalent MEM_STREAM_ND.
                // Keep singletons in the simpler descriptor form.
                if (members.size() == 1) {
                    const auto& sequence = sequences[seed];
                    const std::size_t instructionIndex =
                        queue.commands.size();
                    queue.commands.push_back(
                        software::runtime::encode_mem_stream_nd_command(
                            sequence.instruction,
                            scheduleFor(sequence)));
                    if (sequence.address_binding >= 0) {
                        addressRelocations.push_back(
                            BinaryAddressRelocation {
                                static_cast<std::uint32_t>(
                                    sequence.address_binding),
                                sequence.address_binding_access,
                                key.first,
                                static_cast<std::uint16_t>(key.second),
                                static_cast<std::uint32_t>(
                                    instructionIndex),
                                false,
                            });
                    }
                    if (sequence.write_address_binding >= 0) {
                        addressRelocations.push_back(
                            BinaryAddressRelocation {
                                static_cast<std::uint32_t>(
                                    sequence.write_address_binding),
                                sequence.write_address_binding_access,
                                key.first,
                                static_cast<std::uint16_t>(key.second),
                                static_cast<std::uint32_t>(
                                    instructionIndex),
                                true,
                            });
                    }
                    continue;
                }

                auto launch = scheduleFor(sequences[seed]);
                launch.operand_strides = {0, 0, 0};
                launch.induction_target = IcuInductionTarget::None;
                IcuMemSliceProgram program {launch, {}};
                program.body.reserve(members.size());
                for (const std::size_t member : members) {
                    const auto bodySchedule = scheduleFor(sequences[member]);
                    program.body.push_back(IcuMemSliceProgramEntry {
                        static_cast<std::size_t>(
                            sequences[member].cycle
                            - sequences[seed].cycle),
                        bodySchedule.operand_strides,
                        decodeMem(sequences[member].instruction),
                    });
                }

                const std::size_t instructionIndex = queue.commands.size();
                queue.commands.push_back(
                    software::runtime::encode_mem_slice_program_command(
                        program));
                const auto& sequence = sequences[seed];
                if (sequence.address_binding >= 0) {
                    addressRelocations.push_back(BinaryAddressRelocation {
                        static_cast<std::uint32_t>(
                            sequence.address_binding),
                        sequence.address_binding_access,
                        key.first,
                        static_cast<std::uint16_t>(key.second),
                        static_cast<std::uint32_t>(instructionIndex),
                        false,
                    });
                }
                if (sequence.write_address_binding >= 0) {
                    addressRelocations.push_back(BinaryAddressRelocation {
                        static_cast<std::uint32_t>(
                            sequence.write_address_binding),
                        sequence.write_address_binding_access,
                        key.first,
                        static_cast<std::uint16_t>(key.second),
                        static_cast<std::uint32_t>(instructionIndex),
                        true,
                    });
                }
            }
            return queue;
        }

        for (const CommandSequence& sequence : sequences) {
            validateIssueCycles(sequence);
            std::size_t instructionIndex = queue.commands.size();
            const std::size_t rank = sequence.depth_count > 1 ? 3
                : sequence.outer_count > 1 ? 2 : 1;
            if (key.first == QueueKind::MxmLoad
                || key.first == QueueKind::MxmCompute
                || key.first == QueueKind::MxmDequant) {
                auto inductionTarget = sequence.induction_target;
                if (sequence.address_stride == 0
                    && sequence.outer_stride == 0
                    && sequence.depth_stride == 0)
                    inductionTarget = IcuInductionTarget::None;
                const auto schedule = canonicalize_stream_nd_dimensions(
                    IcuMxmStreamNdSchedule {
                        static_cast<std::size_t>(sequence.cycle),
                        rank,
                        {static_cast<std::size_t>(
                             sequence.repeat_count),
                            static_cast<std::size_t>(
                                sequence.outer_count),
                            static_cast<std::size_t>(
                                sequence.depth_count)},
                        {static_cast<std::size_t>(
                             sequence.repeat_interval),
                            static_cast<std::size_t>(
                                sequence.outer_interval),
                            static_cast<std::size_t>(
                                sequence.depth_interval)},
                        {sequence.address_stride,
                            sequence.outer_stride,
                            sequence.depth_stride},
                        inductionTarget,
                    });
                queue.commands.push_back(
                    software::runtime::encode_mxm_stream_nd_command(
                        sequence.instruction, schedule));
            } else if (key.first == QueueKind::Vxm) {
                const auto schedule = canonicalize_stream_nd_dimensions(
                    IcuVxmStreamNdSchedule {
                        static_cast<std::size_t>(sequence.cycle),
                        rank,
                        {static_cast<std::size_t>(
                             sequence.repeat_count),
                            static_cast<std::size_t>(
                                sequence.outer_count),
                            static_cast<std::size_t>(
                                sequence.depth_count)},
                        {static_cast<std::size_t>(
                             sequence.repeat_interval),
                            static_cast<std::size_t>(
                                sequence.outer_interval),
                            static_cast<std::size_t>(
                                sequence.depth_interval)},
                        {0, 0, 0},
                        IcuInductionTarget::None,
                    });
                queue.commands.push_back(
                    software::runtime::encode_vxm_stream_nd_command(
                        sequence.instruction, schedule));
            } else if (key.first == QueueKind::SxmTranspose
                || key.first == QueueKind::SxmPermute) {
                const auto schedule = canonicalize_stream_nd_dimensions(
                    IcuSxmTileProgramSchedule {
                        static_cast<std::size_t>(sequence.cycle),
                        rank,
                        {static_cast<std::size_t>(
                             sequence.repeat_count),
                            static_cast<std::size_t>(
                                sequence.outer_count),
                            static_cast<std::size_t>(
                                sequence.depth_count)},
                        {static_cast<std::size_t>(
                             sequence.repeat_interval),
                            static_cast<std::size_t>(
                                sequence.outer_interval),
                            static_cast<std::size_t>(
                                sequence.depth_interval)},
                        {0, 0, 0},
                        IcuInductionTarget::None,
                    });
                queue.commands.push_back(
                    software::runtime::encode_sxm_tile_program_command(
                        sequence.instruction, schedule));
            } else {
                queue.commands.push_back(
                    software::runtime::encode_macro_schedule_command(
                        sequence.instruction,
                        IcuMacroSchedule {
                            static_cast<std::size_t>(sequence.cycle),
                            static_cast<std::size_t>(
                                sequence.repeat_count),
                            static_cast<std::size_t>(
                                sequence.repeat_interval),
                            sequence.address_stride,
                            static_cast<std::size_t>(
                                sequence.outer_count),
                            static_cast<std::size_t>(
                                sequence.outer_interval),
                            sequence.outer_stride,
                            sequence.induction_target,
                        }));
            }
            if (sequence.scale_binding >= 0) {
                scaleRelocations.push_back(BinaryScaleRelocation {
                    static_cast<std::uint32_t>(sequence.scale_binding),
                    0,
                    key.first,
                    static_cast<std::uint16_t>(key.second),
                    static_cast<std::uint32_t>(instructionIndex),
                    VxmImmediateOperand::Rhs,
                });
            }
            if (sequence.address_binding >= 0) {
                addressRelocations.push_back(BinaryAddressRelocation {
                    static_cast<std::uint32_t>(sequence.address_binding),
                    sequence.address_binding_access,
                    key.first,
                    static_cast<std::uint16_t>(key.second),
                    static_cast<std::uint32_t>(instructionIndex),
                    false,
                });
            }
            if (sequence.write_address_binding >= 0) {
                addressRelocations.push_back(BinaryAddressRelocation {
                    static_cast<std::uint32_t>(
                        sequence.write_address_binding),
                    sequence.write_address_binding_access,
                    key.first,
                    static_cast<std::uint16_t>(key.second),
                    static_cast<std::uint32_t>(instructionIndex),
                    true,
                });
            }
            const int64_t finalCycle = sequence_final_cycle(sequence);
            max_cycle = std::max(
                max_cycle, static_cast<std::size_t>(finalCycle));
        }
        return queue;
    }
    legalize_encoded_repeat_limits(sequences);
    std::sort(sequences.begin(), sequences.end(), [](const auto& lhs,
                                                   const auto& rhs) {
        return lhs.cycle < rhs.cycle;
    });
    QueueProgram queue {key.first, static_cast<std::size_t>(key.second), {}};
    int64_t cursor = 0;
    const CommandSequence* previous = nullptr;
    for (const CommandSequence& sequence : sequences) {
        if (sequence.cycle < cursor)
            throw std::runtime_error("overlapping Command IR sequences target ICU queue kind="
                + std::to_string(static_cast<int>(key.first)) + " index="
                + std::to_string(key.second) + " at cycle="
                + std::to_string(sequence.cycle) + " (busy through "
                + std::to_string(cursor - 1) + "; previous start="
                + std::to_string(previous ? previous->cycle : -1) + " count="
                + std::to_string(previous ? previous->repeat_count : -1) + " interval="
                + std::to_string(previous ? previous->repeat_interval : -1)
                + ")");
        if (sequence.cycle > cursor)
            queue.commands.push_back(control_command(
                isa::encode_icu_nop(static_cast<std::size_t>(sequence.cycle - cursor))));
        const std::size_t instructionIndex = queue.commands.size();
        queue.commands.push_back(sequence.instruction);
        if (sequence.scale_binding >= 0) {
            scaleRelocations.push_back(BinaryScaleRelocation {
                static_cast<std::uint32_t>(sequence.scale_binding),
                0,
                key.first,
                static_cast<std::uint16_t>(key.second),
                static_cast<std::uint32_t>(instructionIndex),
                VxmImmediateOperand::Rhs,
            });
        }
        if (sequence.address_binding >= 0) {
            addressRelocations.push_back(BinaryAddressRelocation {
                static_cast<std::uint32_t>(sequence.address_binding),
                sequence.address_binding_access,
                key.first,
                static_cast<std::uint16_t>(key.second),
                static_cast<std::uint32_t>(instructionIndex),
                false,
            });
        }
        if (sequence.write_address_binding >= 0) {
            addressRelocations.push_back(BinaryAddressRelocation {
                static_cast<std::uint32_t>(
                    sequence.write_address_binding),
                sequence.write_address_binding_access,
                key.first,
                static_cast<std::uint16_t>(key.second),
                static_cast<std::uint32_t>(instructionIndex),
                true,
            });
        }
        if (sequence.outer_count > 1) {
            queue.commands.push_back(repeat_2d_command(IcuRepeat2D {
                static_cast<std::size_t>(sequence.repeat_count),
                static_cast<std::size_t>(sequence.repeat_interval),
                sequence.address_stride,
                static_cast<std::size_t>(sequence.outer_count),
                static_cast<std::size_t>(sequence.outer_interval),
                sequence.outer_stride,
                sequence.induction_target,
            }));
        } else if (sequence.repeat_count > 1) {
            queue.commands.push_back(control_command(isa::encode_icu_repeat(
                InstructionControlUnit::Repeat {
                    static_cast<std::size_t>(sequence.repeat_count - 1),
                    static_cast<std::size_t>(sequence.repeat_interval),
                    sequence.address_stride,
                })));
        }
        const int64_t final_cycle = sequence_final_cycle(sequence);
        cursor = final_cycle + 1;
        max_cycle = std::max(max_cycle, static_cast<std::size_t>(final_cycle));
        previous = &sequence;
    }
    return queue;
}

void emit_compiled_mem_write_sync(software::runtime::BinaryProgram& program)
{
    using namespace software::runtime;
    if (program.weight_page_uses.empty()) return;

    auto plans = plan_weight_prefetches(program);
    std::vector<C2cWeightPage> pages;
    pages.reserve(plans.size());
    std::vector<std::uint8_t> geometryBytes;
    for (auto& plan : plans) {
        C2cWeightPage page;
        page.bank = plan.bank;
        plan.bytes = {};
        for (const std::size_t useIndex : plan.use_indices) {
            const auto& use = program.weight_page_uses.at(useIndex);
            const auto binding = std::ranges::find_if(program.bindings,
                [&](const BinaryBinding& candidate) {
                    return candidate.access == BindingAccess::Input
                        && candidate.index == use.binding_index;
                });
            if (binding == program.bindings.end()
                || binding->byte_size >
                    std::numeric_limits<std::size_t>::max())
                throw std::runtime_error(
                    "paged MEM_WRITE_SYNC needs a valid input binding");
            const auto logicalBytes =
                static_cast<std::size_t>(binding->byte_size);
            if (geometryBytes.size() < logicalBytes)
                geometryBytes.resize(logicalBytes);
            // Packing geometry depends on the binding and page placement,
            // never on weight values. Use zeros to get the exact touched-row
            // runs; residency regions may conservatively include holes.
            const auto image = pack_weight_binding_page(*binding,
                use.page_index,
                std::span<const std::uint8_t>(
                    geometryBytes.data(), logicalBytes),
                program.hardware);
            for (const auto& segment : image.segments) {
                if (segment.vector_count
                    > std::numeric_limits<std::uint16_t>::max())
                    throw std::runtime_error(
                        "paged MEM_WRITE_SYNC segment exceeds its count field");
                page.segments.push_back(C2cWeightSegment {
                    static_cast<Hemisphere>(segment.hemisphere),
                    segment.slice, plan.bank, segment.base_row, 0, 0,
                    static_cast<std::uint16_t>(segment.vector_count)});
                plan.bytes[segment.hemisphere] +=
                    static_cast<std::uint64_t>(segment.vector_count)
                    * hw::kPhysicalVectorBytes;
            }
        }
        if (page.segments.empty())
            throw std::runtime_error("paged MEM_WRITE_SYNC has no segments");
        pages.push_back(std::move(page));
    }
    schedule_weight_prefetches(program, plans);

    auto system = std::make_unique<C2cDmaSystem>();
    SystemHardwareConfiguration hardware;
    hardware.sram_depth_rows = program.hardware.sram_depth_rows;
    hardware.mxms_per_hemisphere = program.hardware.mxms_per_hemisphere;
    hardware.mxm_weight_buffers = program.hardware.mxm_weight_buffers;
    hardware.vxm_alus = program.hardware.vxm_alus;
    hardware.c2c_streams_per_direction =
        program.hardware.c2c_streams_per_direction;
    system->chip().configure_hardware(hardware);
    C2cWeightPager pager(*system);
    pager.begin_schedule(program);
    std::vector<std::size_t> launchOrder;
    for (std::size_t index = 0; index < plans.size(); ++index)
        if (!plans[index].pre_execution) launchOrder.push_back(index);
    std::ranges::sort(launchOrder, [&](std::size_t lhs, std::size_t rhs) {
        return std::tie(plans[lhs].start_cycle, plans[lhs].ready_cycle)
            < std::tie(plans[rhs].start_cycle, plans[rhs].ready_cycle);
    });
    for (const auto index : launchOrder) {
        const auto& plan = plans[index];
        static_cast<void>(pager.schedule(program, pages[index],
            static_cast<std::size_t>(plan.start_cycle),
            static_cast<std::size_t>(plan.transfer_end_cycle),
            static_cast<std::size_t>(plan.ready_cycle),
            0x10000u + index));
    }
    pager.finalize_schedule(program);
    // DDR addresses and launch events belong to the model-package link.
    // Only the MEM iMEM words are part of this compiler image.
    std::erase_if(program.queues, [](const QueueProgram& queue) {
        return queue.kind == QueueKind::C2cDma
            || queue.kind == QueueKind::C2cRx;
    });
}

void emit_compiled_mem_read_sync(software::runtime::BinaryProgram& program)
{
    using namespace software::runtime;
    constexpr std::uint32_t kFirstStateReadTag = 0x8000;
    std::uint32_t nextTag = kFirstStateReadTag;
    std::array<std::size_t, hw::kHemispheres> nextLane{};

    for (const BinaryBinding& binding : program.bindings) {
        if (binding.access != BindingAccess::Internal
            || !binding.role.starts_with("state.kv."))
            continue;
        if (binding.byte_size > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error(
                "KV-cache MEM_READ_SYNC binding is too large");
        const std::vector<std::uint8_t> zero(
            static_cast<std::size_t>(binding.byte_size), 0);
        const auto image = pack_binding_image(binding, zero, program.hardware);
        for (const PackedWeightSegment& segment : image.segments) {
            if (segment.vector_count == 0
                || segment.vector_count
                    > std::numeric_limits<std::uint16_t>::max())
                throw std::runtime_error(
                    "KV-cache MEM_READ_SYNC segment count is invalid");
            if (nextTag > std::numeric_limits<std::uint16_t>::max())
                throw std::runtime_error(
                    "KV-cache MEM_READ_SYNC exhausted synchronization tags");
            const auto hemisphere =
                static_cast<Hemisphere>(segment.hemisphere);
            const auto side = static_cast<std::size_t>(segment.hemisphere);
            const std::size_t lane = nextLane.at(side)++
                % program.hardware.c2c_streams_per_direction;
            const std::size_t queueIndex = InstructionControlUnit::mem_queue(
                hemisphere, segment.slice, binding.bank);
            auto queue = std::ranges::find_if(program.queues,
                [&](const QueueProgram& candidate) {
                    return candidate.kind == QueueKind::Mem
                        && candidate.index == queueIndex;
                });
            if (queue == program.queues.end()) {
                program.queues.push_back(
                    QueueProgram {QueueKind::Mem, queueIndex, {}});
                queue = std::prev(program.queues.end());
            }
            const auto packet = InstructionControlUnit::MemIcu::
                encode_synchronized_raw_packet(segment.vector_count,
                    nextTag++, 0, 1,
                    MemInstruction::Read(segment.base_row,
                        StreamId::East(lane)),
                    segment.vector_count);
            const auto commands = encode_mem_synchronized_icu_packet(packet);
            const std::size_t packetStart = queue->commands.size();
            queue->commands.insert(queue->commands.end(),
                commands.begin(), commands.end());
            program.address_relocations.push_back(BinaryAddressRelocation {
                binding.index, BindingAccess::Internal, QueueKind::Mem,
                static_cast<std::uint16_t>(queueIndex),
                static_cast<std::uint32_t>(packetStart), false});
        }
    }
    std::sort(program.queues.begin(), program.queues.end(),
        [](const QueueProgram& lhs, const QueueProgram& rhs) {
            return std::tie(lhs.kind, lhs.index)
                < std::tie(rhs.kind, rhs.index);
        });
}

} // namespace

software::runtime::BinaryProgram translate_command_module(mlir::ModuleOp module)
{
    const auto target = LPUTargetModel::from_operation(module);
    if (mlir::failed(target))
        throw std::runtime_error("Command IR module has an invalid target");
    const auto loweringMode = module->getAttrOfType<mlir::StringAttr>(
        "ftlpu.command_lowering");
    if (loweringMode && loweringMode.getValue() != "direct"
        && loweringMode.getValue() != "legacy")
        throw std::runtime_error(
            "Command IR module has an invalid ftlpu.command_lowering");
    const bool requiresDirectLowering = loweringMode
        && loweringMode.getValue() == "direct";
    IcuCompressionMode compressionMode = IcuCompressionMode::Macro;
    if (const auto compressionAttr =
            module->getAttrOfType<mlir::StringAttr>(
                "ftlpu.icu_compression")) {
        const auto parsed = parse_icu_compression_mode(
            compressionAttr.getValue().str());
        if (!parsed)
            throw std::runtime_error(
                "Command IR module has an invalid ftlpu.icu_compression");
        compressionMode = *parsed;
    } else if (const auto legacyMacroAttr =
                   module->getAttrOfType<mlir::BoolAttr>(
                       "ftlpu.icu_macro_schedule")) {
        compressionMode = legacyMacroAttr.getValue()
            ? IcuCompressionMode::Macro
            : IcuCompressionMode::Control;
    }
    // MEM_SLICE_PROGRAM has no fixed hardware packet and the FU raw-word
    // queues deliberately reject it.  Keep it available only when a module
    // explicitly opts into the legacy software-validation format.
    bool memSliceProgramEnabled = false;
    if (const auto attr = module->getAttrOfType<mlir::BoolAttr>(
            "ftlpu.mem_slice_program"))
        memSliceProgramEnabled = attr.getValue();
    auto streamReleaseSummary = stream_release_cycles(module, *target);
    QueueMap queues;
    Raw3DQueueMap raw3DQueues;
    std::vector<BinaryBinding> bindings;
    std::vector<BinaryTimeline> timelines;
    std::vector<BinaryWeightPageUse> weightPageUses;
    module.walk([&](command::BindingOp op) { bindings.push_back(translate_binding(op)); });
    module.walk([&](command::TimelineOp op) {
        timelines.push_back(BinaryTimeline {
            op.getName().str(),
            static_cast<std::uint64_t>(op.getStart()),
            static_cast<std::uint64_t>(op.getEnd()),
        });
    });
    module.walk([&](command::WeightPageOp op) {
        weightPageUses.push_back(BinaryWeightPageUse {
            static_cast<std::uint32_t>(op.getBindingIndex()),
            static_cast<std::uint32_t>(op.getPageIndex()),
            static_cast<std::uint16_t>(op.getBank()),
            static_cast<std::uint64_t>(op.getReadyCycle()),
            static_cast<std::uint64_t>(op.getReleaseCycle()),
        });
    });
    module.walk([&](command::MemOp op) { collect_mem(op, queues); });
    module.walk([&](command::MemBundleOp op) {
        collect_mem_bundle(op, queues);
    });
    module.walk([&](command::MxmOp op) {
        if (is_native4_decode(op))
            collect_native4_decode(op, raw3DQueues);
        else
            collect_mxm(op, queues);
    });
    module.walk([&](command::MxmDequantOp op) {
        collect_mxm_dequant(op, queues);
    });
    module.walk([&](command::VxmOp op) { collect_vxm(op, queues); });
    module.walk([&](command::SxmOp op) { collect_sxm(op, queues); });
    module.walk([&](command::Mem3DOp op) {
        collect_raw_3d<isa::EncodedMemIcu3DPacket,
            MemIcuInstruction>(op.getOperation(), op.getQueue(),
            op.getWords(), QueueKind::Mem, InstructionKind::Mem,
            isa::decode_mem_icu_3d_instruction, raw3DQueues);
        auto& packet = raw3DQueues[
            {QueueKind::Mem, op.getQueue()}].back();
        packet.address_binding = op.getAddressBinding()
            ? static_cast<int64_t>(*op.getAddressBinding()) : -1;
        packet.address_binding_access = address_binding_access(op);
    });
    module.walk([&](command::MemWriteRead2DOp op) {
        const auto packet = raw_3d_packet<
            isa::EncodedMemIcuWriteRead2DPacket>(
            op.getOperation(), op.getWords());
        const auto instruction =
            isa::decode_mem_icu_write_read_2d_instruction(packet);
        const auto cycle = static_cast<std::size_t>(op.getCycle());
        const auto last =
            ::ftlpu::detail::mem_icu_write_read_2d_last_issue_cycle(
                instruction);
        if (last > std::numeric_limits<std::size_t>::max() - cycle)
            throw std::runtime_error(
                "MEM WRITE_READ_2D cycle range overflows");
        raw3DQueues[{QueueKind::Mem, op.getQueue()}].push_back(
            Raw3DPacketSequence {cycle, cycle + last,
                raw_3d_queue_words(packet, InstructionKind::Mem),
                -1, BindingAccess::Input});
    });
    module.walk([&](command::MxmLoad3DOp op) {
        collect_raw_3d<isa::EncodedMxmLoadIcu3DPacket,
            MxmLoadIcuInstruction>(op.getOperation(), op.getQueue(),
            op.getWords(), QueueKind::MxmLoad, InstructionKind::Mxm,
            isa::decode_mxm_load_icu_3d_instruction, raw3DQueues);
    });
    module.walk([&](command::MxmDequant3DOp op) {
        collect_raw_3d<isa::EncodedMxmDequantIcu3DPacket,
            MxmDequantIcuInstruction>(op.getOperation(), op.getQueue(),
            op.getWords(), QueueKind::MxmDequant,
            InstructionKind::MxmDequant,
            isa::decode_mxm_dequant_icu_3d_instruction, raw3DQueues);
        raw3DQueues[{QueueKind::MxmDequant, op.getQueue()}].back()
            .scale_binding = op.getScaleBinding()
            ? static_cast<int64_t>(*op.getScaleBinding()) : -1;
    });
    module.walk([&](command::MxmCompute3DOp op) {
        collect_raw_3d<isa::EncodedMxmComputeIcu3DPacket,
            MxmComputeIcuInstruction>(op.getOperation(), op.getQueue(),
            op.getWords(), QueueKind::MxmCompute, InstructionKind::Mxm,
            isa::decode_mxm_compute_icu_3d_instruction, raw3DQueues);
    });
    module.walk([&](command::VxmRun2DOp op) {
        collect_raw_3d<isa::EncodedVxmIcuRun2DPacket,
            VxmIcuRun2DInstruction>(op.getOperation(), op.getQueue(),
            op.getWords(), QueueKind::Vxm, InstructionKind::Vxm,
            isa::decode_vxm_icu_run_2d_instruction, raw3DQueues);
        raw3DQueues[{QueueKind::Vxm, op.getQueue()}].back()
            .scale_binding = op.getScaleBinding()
            ? static_cast<int64_t>(*op.getScaleBinding()) : -1;
    });
    module.walk([&](command::SxmRun2DOp op) {
        const auto kind = op.getKind() == "transpose"
            ? QueueKind::SxmTranspose : QueueKind::SxmPermute;
        collect_raw_3d<isa::EncodedSxmIcuRun2DPacket,
            SxmIcuRun2DInstruction>(op.getOperation(), op.getQueue(),
            op.getWords(), kind, InstructionKind::Sxm,
            isa::decode_sxm_icu_run_2d_instruction, raw3DQueues);
    });
    if (queues.empty() && raw3DQueues.empty())
        throw std::runtime_error("Command IR module has no queue commands");
    if (requiresDirectLowering && !queues.empty())
        throw std::runtime_error(
            "direct lowering produced a legacy CommandSequence; FU loop "
            "instructions must be emitted directly from operator domains");

    // A raw FU packet owns the physical queue i-MEM image. Mixing it with
    // legacy CommandSequence output would require an explicit packet-order
    // model and must not silently route the raw instruction through the
    // legacy compression path.
    for (const auto& [key, packets] : raw3DQueues) {
        (void)packets;
        if (queues.contains(key))
            throw std::runtime_error(
                "a physical ICU queue cannot mix raw FU loop packets with legacy commands");
    }

    // A binary program starts its ICU clock at zero. Full programs naturally
    // have an origin of zero; rebasing also makes a standalone scheduled phase
    // (for example the QK trace) runnable without materializing unrelated
    // preceding phases as tens of thousands of ICU NOPs.
    int64_t cycle_origin = std::numeric_limits<int64_t>::max();
    for (const auto& [key, sequences] : queues) {
        (void)key;
        for (const CommandSequence& sequence : sequences)
            cycle_origin = std::min(cycle_origin, sequence.cycle);
    }
    for (const BinaryTimeline& timeline : timelines)
        cycle_origin = std::min(
            cycle_origin, static_cast<int64_t>(timeline.start_cycle));
    // Raw FU operations retain their compiler schedule relative to program
    // cycle zero. CommandBinary materializes that initial delay as an ICU NOP;
    // no absolute cycle is stored in the hardware packet itself.
    if (!raw3DQueues.empty()) cycle_origin = 0;
    if (cycle_origin > 0) {
        for (auto& [key, sequences] : queues) {
            (void)key;
            for (CommandSequence& sequence : sequences) sequence.cycle -= cycle_origin;
        }
        for (BinaryBinding& binding : bindings)
            binding.ready_cycle =
                binding.ready_cycle > static_cast<std::uint64_t>(cycle_origin)
                ? binding.ready_cycle
                    - static_cast<std::uint64_t>(cycle_origin)
                : 0;
        for (BinaryTimeline& timeline : timelines) {
            timeline.start_cycle -= static_cast<std::uint64_t>(cycle_origin);
            timeline.end_cycle -= static_cast<std::uint64_t>(cycle_origin);
        }
        for (BinaryWeightPageUse& use : weightPageUses) {
            use.ready_cycle = use.ready_cycle
                    > static_cast<std::uint64_t>(cycle_origin)
                ? use.ready_cycle
                    - static_cast<std::uint64_t>(cycle_origin)
                : 0;
            use.release_cycle = use.release_cycle
                    > static_cast<std::uint64_t>(cycle_origin)
                ? use.release_cycle
                    - static_cast<std::uint64_t>(cycle_origin)
                : 0;
        }
        for (std::uint64_t& release : streamReleaseSummary.cycles)
            release = release > static_cast<std::uint64_t>(cycle_origin)
                ? release - static_cast<std::uint64_t>(cycle_origin)
                : 0;
    }

    std::sort(bindings.begin(), bindings.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.access, lhs.index) < std::tie(rhs.access, rhs.index);
    });
    software::runtime::BinaryProgram program;
    program.target_name = target->name();
    program.target_abi = target->abi_fingerprint();
    auto& hardware = program.hardware;
    const auto& memory = target->memory();
    const auto& streams = target->streams();
    const auto& throughput = target->throughput();
    const auto& externalMemory = target->external_memory();
    const auto& icuMemory = target->icu_memory();
#define COPY_MEMORY(field) \
    hardware.field = static_cast<std::uint32_t>(memory.field)
    COPY_MEMORY(hemispheres);
    COPY_MEMORY(slices_per_hemisphere);
    COPY_MEMORY(banks_per_slice);
    COPY_MEMORY(words_per_bank);
    COPY_MEMORY(bytes_per_word);
    COPY_MEMORY(sram_depth_rows);
    COPY_MEMORY(sram_read_ports_per_slice);
    COPY_MEMORY(sram_write_ports_per_slice);
#undef COPY_MEMORY
#define COPY_STREAM(field) \
    hardware.field = static_cast<std::uint32_t>(streams.field)
    COPY_STREAM(streams_per_direction);
    COPY_STREAM(encoded_streams);
    COPY_STREAM(c2c_streams_per_direction);
    COPY_STREAM(c2c_bytes_per_stream_per_cycle);
    COPY_STREAM(mem_boundary_register_columns);
    COPY_STREAM(system_register_columns);
    COPY_STREAM(mem_slices_per_register_group);
#undef COPY_STREAM
#define COPY_THROUGHPUT(field) \
    hardware.field = static_cast<std::uint32_t>(throughput.field)
    COPY_THROUGHPUT(tile_rows);
    COPY_THROUGHPUT(lanes_per_tile);
    COPY_THROUGHPUT(mem_read_bytes_per_cycle);
    COPY_THROUGHPUT(mem_write_bytes_per_cycle);
    COPY_THROUGHPUT(mxm_rows);
    COPY_THROUGHPUT(mxm_columns);
    COPY_THROUGHPUT(mxm_load_streams_per_cycle);
    COPY_THROUGHPUT(mxm_int8_load_streams_per_cycle);
    COPY_THROUGHPUT(mxm_load_bytes_per_cycle);
    COPY_THROUGHPUT(mxm_activation_streams);
    COPY_THROUGHPUT(mxm_result_streams);
    COPY_THROUGHPUT(mxm_pipeline_rows);
    COPY_THROUGHPUT(mxm_block_rows);
    COPY_THROUGHPUT(mxm_local_dequant_enabled);
    COPY_THROUGHPUT(mxm_block_compute_enabled);
    COPY_THROUGHPUT(mxm_weight_activation_overlap_enabled);
    COPY_THROUGHPUT(mxm_local_load_to_compute_latency);
    COPY_THROUGHPUT(mxm_block_group_interval);
    COPY_THROUGHPUT(mxm_earliest_iw_cycle);
    COPY_THROUGHPUT(qk_iw_to_compute_latency);
    COPY_THROUGHPUT(mxms_per_hemisphere);
    COPY_THROUGHPUT(mxm_weight_buffers);
    COPY_THROUGHPUT(mxm_accumulator_blocks);
    COPY_THROUGHPUT(vxm_alus);
    COPY_THROUGHPUT(vxm_cross_hemisphere_streams_enabled);
    COPY_THROUGHPUT(vxm_fma_enabled);
    COPY_THROUGHPUT(vxm_weight_to_iw_latency);
    COPY_THROUGHPUT(mem_to_sxm_latency);
    COPY_THROUGHPUT(mem_to_mxm_latency);
    COPY_THROUGHPUT(mxm0_accumulator_latency);
    COPY_THROUGHPUT(mxm1_accumulator_latency);
    COPY_THROUGHPUT(accumulator_to_vxm_latency);
    COPY_THROUGHPUT(accumulator_read_to_vxm_latency);
    COPY_THROUGHPUT(swiglu_write_latency);
#undef COPY_THROUGHPUT
#define COPY_EXTERNAL(field) \
    hardware.field = static_cast<std::uint32_t>(externalMemory.field)
    COPY_EXTERNAL(lpu_clock_mhz);
    COPY_EXTERNAL(ddr_peak_bandwidth_mbytes_per_second);
    COPY_EXTERNAL(ddr_scheduling_efficiency_percent);
    COPY_EXTERNAL(ddr_read_latency_cycles);
    COPY_EXTERNAL(ddr_write_latency_cycles);
    COPY_EXTERNAL(ddr_read_latency_jitter_cycles);
    COPY_EXTERNAL(ddr_write_latency_jitter_cycles);
    COPY_EXTERNAL(ddr_request_queue_depth);
    COPY_EXTERNAL(ddr_latency_random_seed);
#undef COPY_EXTERNAL
#define COPY_ICU_MEMORY(field) \
    hardware.icu_##field = static_cast<std::uint32_t>(icuMemory.field)
    COPY_ICU_MEMORY(mem_instruction_bits);
    COPY_ICU_MEMORY(mem_imem_depth);
    COPY_ICU_MEMORY(mxm_instruction_bits);
    COPY_ICU_MEMORY(mxm_imem_depth);
    COPY_ICU_MEMORY(vxm_instruction_bits);
    COPY_ICU_MEMORY(vxm_imem_depth);
    COPY_ICU_MEMORY(sxm_instruction_bits);
    COPY_ICU_MEMORY(sxm_imem_depth);
    COPY_ICU_MEMORY(macro_encoding_version);
    COPY_ICU_MEMORY(mem_macro_contexts);
    COPY_ICU_MEMORY(mxm_macro_contexts);
    COPY_ICU_MEMORY(mem_macro_context_bits);
    COPY_ICU_MEMORY(mxm_macro_context_bits);
#undef COPY_ICU_MEMORY
    program.memory_floors = static_memory_floors(module,
        target->memory().slices_per_hemisphere,
        target->memory().banks_per_slice,
        target->memory().sram_depth_rows);
    program.bindings = std::move(bindings);
    program.timelines = std::move(timelines);
    std::sort(weightPageUses.begin(), weightPageUses.end(),
        [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.ready_cycle, lhs.binding_index,
                       lhs.page_index)
                < std::tie(rhs.ready_cycle, rhs.binding_index,
                       rhs.page_index);
        });
    program.weight_page_uses = std::move(weightPageUses);
    program.stream_release_cycles =
        std::move(streamReleaseSummary.cycles);
    for (auto& [key, sequences] : queues)
        program.queues.push_back(encode_queue(key, std::move(sequences),
            program.max_cycle, program.scale_relocations,
            program.address_relocations,
            target->throughput().icu_repeat_2d_enabled != 0,
            compressionMode, memSliceProgramEnabled));
    for (auto& [key, packets] : raw3DQueues) {
        std::stable_sort(packets.begin(), packets.end(),
            [](const Raw3DPacketSequence& lhs,
                const Raw3DPacketSequence& rhs) {
                return lhs.start_cycle < rhs.start_cycle;
            });
        validate_raw_3d_frontend(key, packets);
        QueueProgram queue {key.first,
            static_cast<std::size_t>(key.second), {}};
        // The physical ICU owns one active coarse instruction. Encode the
        // queue-local gap in each FU 3-D packet's wait_cycle field.
        std::size_t cursor = 0;
        for (Raw3DPacketSequence& packet : packets) {
            if (packet.start_cycle < cursor)
                throw std::runtime_error(
                    "overlapping FU loop packets target a single-context ICU "
                    "queue: resource="
                    + std::string(raw_loop_queue_name(key.first))
                    + ", queue=" + std::to_string(key.second)
                    + ", start_cycle="
                    + std::to_string(packet.start_cycle)
                    + ", busy_through=" + std::to_string(cursor - 1));
            if (packet.start_cycle > cursor) {
                const auto wait = packet.start_cycle - cursor;
                if (can_encode_packet_wait(packet, wait))
                    set_packet_wait(packet, wait);
                else
                    queue.commands.push_back(control_command(
                        isa::encode_icu_nop(wait)));
            }
            program.max_cycle = std::max(
                program.max_cycle, packet.final_cycle);
            const std::size_t packetStart = queue.commands.size();
            queue.commands.insert(queue.commands.end(),
                std::make_move_iterator(packet.physical_words.begin()),
                std::make_move_iterator(packet.physical_words.end()));
            if (packet.address_binding >= 0)
                program.address_relocations.push_back(
                    BinaryAddressRelocation {
                        static_cast<std::uint32_t>(
                            packet.address_binding),
                        packet.address_binding_access,
                        key.first,
                        static_cast<std::uint16_t>(key.second),
                        static_cast<std::uint32_t>(packetStart),
                        false,
                    });
            if (packet.scale_binding >= 0)
                program.scale_relocations.push_back(BinaryScaleRelocation {
                    static_cast<std::uint32_t>(packet.scale_binding),
                    0, key.first, static_cast<std::uint16_t>(key.second),
                    static_cast<std::uint32_t>(packetStart),
                    VxmImmediateOperand::Rhs,
                });
            if (packet.final_cycle
                == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error("FU loop packet end cycle overflows");
            cursor = packet.final_cycle + 1;
        }
        program.queues.push_back(std::move(queue));
    }
    std::sort(program.queues.begin(), program.queues.end(),
        [](const QueueProgram& lhs, const QueueProgram& rhs) {
            return std::tie(lhs.kind, lhs.index)
                < std::tie(rhs.kind, rhs.index);
        });
    if (requiresDirectLowering) {
        emit_compiled_mem_write_sync(program);
        emit_compiled_mem_read_sync(program);
    }
    return program;
}

} // namespace ftlpu::compiler::target
