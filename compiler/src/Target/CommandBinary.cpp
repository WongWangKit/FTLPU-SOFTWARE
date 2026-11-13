#include "ftlpu/compiler/Target/vxm_chain_lowering.hpp"
// Keep this translation unit rebuilt when BinaryProgram or binding ABI evolves.
#include "ftlpu/compiler/Target/command_binary.hpp"

#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Command/Transforms/macro_command_materializer.hpp"
#include "ftlpu/compiler/Target/icu_compression.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/macro_queue_link.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
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
#include <type_traits>
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

struct MacroCommandSequence {
    std::size_t start_cycle{0};
    std::size_t final_cycle{0};
    std::vector<QueueCommand> physical_words;
    int64_t address_binding{-1};
    BindingAccess address_binding_access{BindingAccess::Input};
    int64_t scale_binding{-1};
};

using MacroQueueMap =
    std::map<QueueKey, std::vector<MacroCommandSequence>>;

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
void validate_sequential_frontend(const QueueKey& key,
    const std::vector<MacroCommandSequence>& packets)
{
    if (std::ranges::all_of(packets,[](const auto& p){return p.physical_words.size()==1
            && software::runtime::is_macro_schedule_command(p.physical_words.front());})) {
        std::optional<std::size_t> end;
        for(const auto& p:packets) {
            if(end && p.start_cycle<=*end) throw std::runtime_error("interleaving Macro domains on one ICU");
            end=p.final_cycle;
        }
        return; // Packed frontend bandwidth is checked by the real raw decoder.
    }
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
    std::size_t previousStart = 0;
    const MacroCommandSequence* previousPacket = nullptr;
    for (const MacroCommandSequence& packet : packets) {
        if (packet.start_cycle < scheduleCursor)
            throw std::runtime_error(
                "overlapping FU loop packets target a single-context ICU "
                "queue: resource="
                + std::string(raw_loop_queue_name(key.first))
                + ", queue=" + std::to_string(key.second)
                + ", start_cycle=" + std::to_string(packet.start_cycle)
                + ", previous_start=" + std::to_string(previousStart)
                + ", previous_words=" + (previousPacket
                    ? std::to_string(previousPacket->physical_words.front().words[0])
                        + "/" + std::to_string(previousPacket->physical_words.front().words[1])
                        + "/" + std::to_string(previousPacket->physical_words.front().words[2])
                    : std::string("none"))
                + ", current_words="
                    + std::to_string(packet.physical_words.front().words[0])
                    + "/" + std::to_string(packet.physical_words.front().words[1])
                    + "/" + std::to_string(packet.physical_words.front().words[2])
                + ", busy_through=" + std::to_string(scheduleCursor - 1));
        if (packet.start_cycle > scheduleCursor)
            totalWords = checkedAdd(totalWords, 1,
                "raw FU loop queue word count overflows");
        totalWords = checkedAdd(totalWords, packet.physical_words.size(),
            "raw FU loop queue word count overflows");
        if (packet.final_cycle == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("FU loop packet end cycle overflows");
        scheduleCursor = packet.final_cycle + 1;
        previousStart = packet.start_cycle;
        previousPacket = &packet;
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
            std::string nearby;
            for (const MacroCommandSequence& candidate : packets)
                if (candidate.start_cycle <= cycle
                    && candidate.start_cycle + 64 >= cycle)
                    nearby += std::to_string(candidate.start_cycle)
                        + ":" + std::to_string(
                            candidate.physical_words.size()) + ",";
            throw std::runtime_error(
                "raw FU loop schedule exceeds ICU frontend bandwidth: "
                "resource=" + std::string(raw_loop_queue_name(key.first))
                + ", queue=" + std::to_string(key.second)
                + ", cycle=" + std::to_string(cycle)
                + ", command=" + commandName
                + ", required_words=" + std::to_string(requiredWords)
                + ", available_words=" + std::to_string(iqWords)
                + ", nearby_packets=" + nearby
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
    for (const MacroCommandSequence& packet : packets) {
        const auto wait = packet.start_cycle - scheduleCursor;
        if (wait != 0)
            consume(scheduleCursor, 1, "NOP");
        consume(packet.start_cycle,
            packet.physical_words.size(), "packet");
        scheduleCursor = packet.final_cycle + 1;
    }
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
    module.walk([&](command::MacroOp op) {
        if(static_cast<QueueKind>(op.getKind())!=QueueKind::Mem || op.getAddressBinding()) return;
        const auto command=command::readMacroCommand(op);
        const auto schedule=software::runtime::decode_macro_schedule_command(command);
        const auto bits=std::uint64_t(command.words[0])|(std::uint64_t(command.words[1])<<32);
        const auto instruction=isa::decode_mem_instruction(bits);
        reserveFloor(op.getQueue(),instruction.address,schedule.inner_count,schedule.inner_stride,
            schedule.outer_count,schedule.outer_stride);
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
    mlir::ModuleOp module, const LPUTargetModel& target,
    std::optional<std::uint64_t> lastPagedConsumer)
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

    // Every dynamic page is complete at its compiler-authored consumer
    // rendezvous. Operations starting after the last such rendezvous cannot
    // overlap weight ingress: a late physical transfer stalls those queues
    // at the rendezvous too. Do not turn their future SR use into a launch
    // floor for earlier pages. Domains starting before it retain their full
    // release time, including the SR drain.
    const auto afterWeightIngress = [&](mlir::Operation* operation) {
        return lastPagedConsumer
            && static_cast<std::uint64_t>(command_cycle(operation))
                >= *lastPagedConsumer;
    };

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
        if (afterWeightIngress(op)) return;
        markPacked(op.getPackedStream(),
            sequenceEnd(op, command_cycle(op)));
    });
    module.walk([&](command::MemBundleOp op) {
        for (std::size_t index = 0; index < op.getCycles().size(); ++index) {
            const int64_t cycle = llvm::cast<mlir::IntegerAttr>(
                op.getCycles()[index]).getInt();
            if (lastPagedConsumer && static_cast<std::uint64_t>(cycle)
                    >= *lastPagedConsumer) continue;
            const int64_t packed = llvm::cast<mlir::IntegerAttr>(
                op.getPackedStreams()[index]).getInt();
            markPacked(packed, sequenceEnd(op, cycle));
        }
    });
    module.walk([&](command::MxmOp op) {
        if (afterWeightIngress(op)) return;
        const int64_t end = sequenceEnd(op, command_cycle(op));
        if (op.getOpcode() == "iw") {
            const bool int8 = op.getWeightInputMode().value_or("direct16")
                != "direct16";
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
        if (afterWeightIngress(op)) return;
        const int64_t end = command_cycle(op) + op.getRepeatCount() - 1;
        for (const auto stage : {static_cast<std::size_t>(op.getQueue()),
                 static_cast<std::size_t>(op.getQueue()) + 8}) {
            const auto markOperand = [&](llvm::StringRef kind, bool rhs) {
                if (!kind.starts_with("stream_")) return;
                const auto operand = kind == "stream_f32" ? VxmLaneOperand::StreamFloat32()
                    : kind == "stream_f16" ? VxmLaneOperand::StreamFloat16()
                    : VxmLaneOperand::StreamBFloat16();
                const auto base = VxmLane::fixed_input_base_for_stage(stage, rhs, operand.kind);
                const auto width = VxmLane::stream_operand_bytes(operand.kind);
                markRange(base, width, end); markRange(streamCount + base, width, end);
            };
            markOperand(op.getLhsKind(), false); markOperand(op.getRhsKind(), true);
            if (op.getOutputStreamAttr().getInt() >= 0) {
                const auto base = VxmLane::fixed_output_stream_for_block(stage / 2);
                const auto width = op.getCastTarget() == "i8" ? 1 : op.getCastTarget() == "fp32" ? 4 : 2;
                markRange(base, width, end); markRange(streamCount + base, width, end);
            }
        }
    });
    module.walk([&](command::SxmOp op) {
        if (afterWeightIngress(op)) return;
        const int64_t end = sequenceEnd(op, command_cycle(op));
        for (mlir::Attribute stream : op.getSourceStreams())
            markPacked(llvm::cast<mlir::IntegerAttr>(stream).getInt(), end);
        for (mlir::Attribute stream : op.getDestinationStreams())
            markPacked(llvm::cast<mlir::IntegerAttr>(stream).getInt(), end);
    });

    module.walk([&](command::MacroOp op) {
        if(afterWeightIngress(op)) return;
        const auto command=command::readMacroCommand(op);
        const auto schedule=software::runtime::decode_macro_schedule_command(command);
        const int64_t end=static_cast<int64_t>(schedule.start_cycle+(schedule.inner_count-1)*schedule.inner_interval
            +(schedule.outer_count-1)*schedule.outer_interval);
        const auto bits=std::uint64_t(command.words[0])|(std::uint64_t(command.words[1])<<32);
        const auto kind=static_cast<QueueKind>(op.getKind());
        if(kind==QueueKind::Mem) markPacked(isa::decode_mem_instruction(bits).stream,end);
        else if(kind==QueueKind::MxmLoad || kind==QueueKind::MxmCompute) {
            const auto ins=isa::decode_mxm_instruction(bits);
            if(kind==QueueKind::MxmLoad) {
                const auto width=ins.weight_input_mode==MxmWeightInputMode::Direct16
                    ?target.throughput().mxm_load_streams_per_cycle:target.throughput().mxm_int8_load_streams_per_cycle;
                markRange(ins.weight_stream_base,width,end);
            } else {
                if(ins.opcode==MxmControlOpcode::Compute || ins.opcode==MxmControlOpcode::IS)
                    markRange(ins.activation_stream_base,target.throughput().mxm_activation_streams,end);
                if(ins.accumulator_destination==MxmAccumulatorDestination::Stream)
                    markRange(streamCount+ins.stream_base%streamCount,target.throughput().mxm_result_streams,end);
            }
        } else if(kind==QueueKind::SxmTranspose || kind==QueueKind::SxmPermute) {
            const auto ins=software::runtime::decode_sxm_command(command);
            for(auto stream:ins.src_streams) markPacked(stream.stream,end);
            for(auto stream:ins.dst_streams) markPacked(stream.stream,end);
        }
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
    if (value == "w8a16_mxm_weight_wave_split_dense")
        return BindingLayout::W8A16MxmWeightWaveSplitDense;
    if (value == "w8a16_mxm_weight_wave_striped")
        return BindingLayout::W8A16MxmWeightWaveStriped;
    if (value == "w8a16_attention_weight_striped")
        return BindingLayout::W8A16AttentionWeightStriped;
    if (value == "w8a16_is_weight")
        return BindingLayout::W8A16Native4Weight;
    if (value == "fp16_pair_planar") return BindingLayout::Fp16PairPlanar;
    if (value == "fp16_hemisphere_pair_planar") return BindingLayout::Fp16HemispherePairPlanar;
    if (value == "fp32_hemisphere_pair_planar") return BindingLayout::Fp32HemispherePairPlanar;
    if (value == "token_partition_east_first_planar") return BindingLayout::TokenPartitionEastFirstPlanar;
    if (value == "token_partition_planar") return BindingLayout::TokenPartitionPlanar;
    if (value == "column_partition_planar") return BindingLayout::ColumnPartitionPlanar;
    if (value == "fp32_quad_planar") return BindingLayout::Fp32QuadPlanar;
    if (value == "fp16_scalar_broadcast_planar")
        return BindingLayout::Fp16ScalarBroadcastPlanar;
    if (value == "fp32_scalar_broadcast_planar")
        return BindingLayout::Fp32ScalarBroadcastPlanar;
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
    if (value == "fp16_depthwise_kernel4")
        return BindingLayout::Fp16DepthwiseKernel4;
    if (value == "fp16_vxm_gamma_broadcast")
        return BindingLayout::Fp16VxmGammaBroadcast;
    if (value == "fp16_mxm_compact_block4_partition_distributed_16")
        return BindingLayout::Fp16MxmCompactBlock4PartitionDistributed16;
    if (value == "fp16_mxm_block4_partition_distributed_16")
        return BindingLayout::Fp16MxmBlock4PartitionDistributed16;
    if (value == "fp16_mxm_distributed_16")
        return BindingLayout::Fp16MxmDistributed16;
    if (value == "fp16_rope_table")
        return BindingLayout::Fp16RopeTable;
    if (value == "fp16_rope_table_decode_compact")
        return BindingLayout::Fp16RopeTableDecodeCompact;
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
    if(auto stripe=op.getPlacement().getAs<mlir::IntegerAttr>("bank_tile_rows")) {
        if(stripe.getInt()!=4 || (binding.layout!=BindingLayout::Fp16MxmBlock4PartitionDistributed16 && binding.layout!=BindingLayout::Fp16MxmDistributed16))
            throw std::runtime_error("unsupported bank-striped binding geometry");
        binding.layout=BindingLayout::Fp16MxmBankStripedBlock4PartitionDistributed16;
    }
    auto hemisphere = op.getPlacement().getAs<mlir::StringAttr>("hemisphere");
    binding.hemisphere_mask = !hemisphere || hemisphere.getValue() == "east" ? 1
        : hemisphere.getValue() == "west" ? 2 : 3;
    if (auto bank = op.getPlacement().getAs<mlir::IntegerAttr>("bank"))
        binding.bank = static_cast<std::uint16_t>(bank.getInt());
    binding.base_row = op.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt();
    binding.instruction_count = op.getPlacement().getAs<mlir::IntegerAttr>("instruction_count").getInt();
    binding.address_stride = op.getPlacement().getAs<mlir::IntegerAttr>("address_stride").getInt();
    const llvm::StringRef initializer = op.getInitializer();
    const auto ropeConfig = op.getInitializerConfig();
    if (const auto dim = ropeConfig.getAs<mlir::IntegerAttr>("rotary_dim")) {
        binding.rope_rotary_dim = static_cast<std::uint32_t>(dim.getInt());
        binding.rope_head_dim = static_cast<std::uint32_t>(ropeConfig.getAs<mlir::IntegerAttr>("head_dim").getInt());
    }
    if (const auto factor = ropeConfig.getAs<mlir::FloatAttr>("attention_factor"))
        binding.rope_attention_factor = static_cast<float>(factor.getValueAsDouble());
    if (const auto frequencies = ropeConfig.getAs<mlir::DenseFPElementsAttr>("inverse_frequencies"))
        for (const auto frequency : frequencies.getValues<llvm::APFloat>())
            binding.rope_inverse_frequencies.push_back(frequency.convertToFloat());
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
    if (auto pieces = placement.getAs<mlir::ArrayAttr>("output_pieces")) {
        // Copy metadata from a piece-free template. Copying the growing
        // parent then clear() retained a nested vector capacity in every
        // piece, making a long strided output consume quadratic memory.
        const BinaryBinding prototype = binding;
        binding.pieces.reserve(pieces.size());
        for (auto attribute : pieces) {
            auto p = llvm::cast<mlir::DictionaryAttr>(attribute);
            BinaryBinding piece = prototype;
            piece.layout = parse_layout(p.getAs<mlir::StringAttr>("kind").getValue());
            const auto hemisphere=p.getAs<mlir::StringAttr>("hemisphere").getValue();
            piece.hemisphere_mask = hemisphere == "both" ? 3 : hemisphere == "west" ? 2 : 1;
            piece.base_row = p.getAs<mlir::IntegerAttr>("base_row").getInt();
            piece.bank = p.getAs<mlir::IntegerAttr>("bank").getInt();
            piece.instruction_count = p.getAs<mlir::IntegerAttr>("instruction_count").getInt();
            piece.address_stride = p.getAs<mlir::IntegerAttr>("address_stride").getInt();
            piece.shape.clear();
            piece.slices.clear();
            piece.byte_size = binding.element_type == BindingElementType::F32 ? 4 : 2;
            for (auto dim : p.getAs<mlir::ArrayAttr>("storage_shape")) {
                auto size = llvm::cast<mlir::IntegerAttr>(dim).getInt();
                piece.shape.push_back(size);
                piece.byte_size *= size;
            }
            for (auto slice : p.getAs<mlir::ArrayAttr>("slices"))
                piece.slices.push_back(llvm::cast<mlir::IntegerAttr>(slice).getInt());
            binding.pieces.push_back(std::move(piece));
        }
    }
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
    const auto inputMode = op.getWeightInputMode().value_or("direct16") == "fp8_dequant_bf16"
        ? MxmWeightInputMode::Fp8DequantBf16
        : op.getWeightInputMode().value_or("direct16")
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
            ? op.getWeightLoadMode().value_or("block") == "vector"
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
    if (value == "fma" || value == "fms")
        throw std::runtime_error(
            "VXM ISA has no FMA/FMS; lower to multiply plus add/subtract");
    if (value == "negate") return VxmAluOpcode::Negate;
    if (value == "max") return VxmAluOpcode::Max;
    if (value == "exp") return VxmSpecialAluOpcode::Exp;
    if (value == "reciprocal" || value == "divide")
        return VxmSpecialAluOpcode::Reciprocal;
    if (value == "rsqrt") return VxmSpecialAluOpcode::Rsqrt;
    if (value == "log") return VxmSpecialAluOpcode::Log;
    throw std::runtime_error(
        "Command IR VXM operation is not implemented by the current CModel");
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
    target::validateVxmChainControls(op);
    if (op.getRepeatInterval() != 1)
        throw std::runtime_error("VXM configurations support only contiguous FU repetition");
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
    instruction.lhs = target::lowerVxmChainOperand(op.getLhsKind(), op.getLhsIndex(),
        static_cast<float>(op.getLhsImmediateAttr().getValueAsDouble()), queue, false,
        lhsSource ? lhsSource.getValue() : llvm::StringRef{});
    instruction.rhs = target::lowerVxmChainOperand(op.getRhsKind(), op.getRhsIndex(),
        static_cast<float>(op.getRhsImmediateAttr().getValueAsDouble()), queue, true,
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

bool is_native4_decode(command::MxmOp op)
{
    return (op.getOpcode() == "is_load_activation"
            || op.getOpcode() == "is_stream_compute")
        && (op.getIsMode().value_or("") == "is-block"
            || op.getIsMode().value_or("") == "is-vector");
}

void collect_native4_decode(
    command::MxmOp op, QueueMap& queues)
{
    const bool load = op.getOpcode() == "is_load_activation";
    const auto format = op.getDataFormat().value_or("bf16") == "bf16"
        ? MxmDataFormat::BFloat16 : MxmDataFormat::Float16;
    const auto destination = op.getAccumulatorDestination() == "stream"
        ? MxmAccumulatorDestination::Stream
        : MxmAccumulatorDestination::Sram;
    const auto mode = op.getIsMode().value_or("") == "is-vector"
        ? MxmIsMode::Vector : MxmIsMode::Block;
    const auto instruction = load
        ? MxmControlInstruction::IsLoadActivation(
            static_cast<std::size_t>(op.getWeightBuffer()),
            static_cast<std::size_t>(op.getActivationStreamBase()), format,
            mode)
        : MxmControlInstruction::IsStreamCompute(
            static_cast<std::size_t>(op.getWeightBuffer()),
            static_cast<std::size_t>(op.getOutputStreamBase()), format,
            static_cast<std::size_t>(op.getAccumulatorAddress()),
            static_cast<std::size_t>(op.getWeightColumn()), destination,
            op.getAccumulatorClear(), mode);
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
    const auto kind = load ? QueueKind::MxmLoad : QueueKind::MxmCompute;
    queues[{kind,static_cast<int64_t>(op.getQueue())}].push_back(CommandSequence{
        command_cycle(op),static_cast<int64_t>(repeatCount),static_cast<int64_t>(repeatInterval),repeatStride,
        mxm_instruction_command(isa::encode_mxm_instruction(instruction)),-1,-1,-1,
        static_cast<int64_t>(waveCount),static_cast<int64_t>(waveInterval),waveStride,induction,
        static_cast<int64_t>(groupCount),static_cast<int64_t>(groupInterval),0});
}

QueueProgram encode_queue(const QueueKey& key, std::vector<CommandSequence> sequences,
    std::size_t& max_cycle,
    std::vector<BinaryScaleRelocation>& scaleRelocations,
    std::vector<BinaryAddressRelocation>& addressRelocations)
{
    // VXM is a sequential configuration stream in every binary mode.
    // Contiguous FU repetition is encoded inside the 96-bit configuration;
    // neither multi-dimensional launch domains nor ICU Repeat are emitted.
    if (key.first == QueueKind::Vxm) {
        std::stable_sort(sequences.begin(), sequences.end(),
            [](const auto& a, const auto& b) { return a.cycle < b.cycle; });
        QueueProgram queue{key.first, static_cast<std::size_t>(key.second), {}};
        std::vector<MacroCommandSequence> frontend;
        std::size_t cursor = 0;
        for (const auto& sequence : sequences) {
            if (sequence.repeat_count != 1 || sequence.outer_count != 1 || sequence.depth_count != 1)
                throw std::runtime_error("VXM requires sequential configurations, not ICU launch loops");
            const auto start = static_cast<std::size_t>(sequence.cycle);
            const auto& words = sequence.instruction.words;
            const auto decoded = isa::decode_vxm_instruction(key.second,
                {std::uint64_t(words[0]) | (std::uint64_t(words[1]) << 32), words[2]});
            if (start < cursor)
                throw std::runtime_error("overlapping VXM configurations in a sequential queue");
            if (start > cursor)
                queue.commands.push_back(control_command(isa::encode_icu_nop(start - cursor)));
            const auto index = queue.commands.size();
            queue.commands.push_back(sequence.instruction);
            frontend.push_back({start, start, {sequence.instruction}, -1, BindingAccess::Input});
            if (sequence.scale_binding >= 0)
                scaleRelocations.push_back({static_cast<std::uint32_t>(sequence.scale_binding), 0,
                    key.first, static_cast<std::uint16_t>(key.second), static_cast<std::uint32_t>(index),
                    VxmImmediateOperand::Rhs});
            cursor = start + 1;
            max_cycle = std::max(max_cycle, start + decoded.instruction.repeat_count - 1);
        }
        validate_sequential_frontend(key, frontend);
        return queue;
    }
    struct PendingMacro { QueueCommand command;const CommandSequence* source; };
    std::vector<PendingMacro> pending;
    for(const auto& sequence:sequences) {
        if(sequence.write_address_binding>=0)
            throw std::runtime_error("compound MEM template must be split into direct read/write domains by lowering");
        const auto target=key.first==QueueKind::Mem?IcuInductionTarget::MemAddress:sequence.induction_target;
        auto commands=command::lowerMacroDomain(sequence.cycle,sequence.instruction,target,
            {static_cast<std::size_t>(sequence.repeat_count),static_cast<std::size_t>(sequence.outer_count),static_cast<std::size_t>(sequence.depth_count)},
            {static_cast<std::size_t>(sequence.repeat_interval),static_cast<std::size_t>(sequence.outer_interval),static_cast<std::size_t>(sequence.depth_interval)},
            {sequence.address_stride,sequence.outer_stride,sequence.depth_stride});
        for(auto& command:commands) pending.push_back({std::move(command),&sequence});
    }
    std::stable_sort(pending.begin(),pending.end(),[](const auto& a,const auto& b){
        return software::runtime::decode_macro_schedule_command(a.command).start_cycle
            <software::runtime::decode_macro_schedule_command(b.command).start_cycle;});
    QueueProgram queue{key.first,static_cast<std::size_t>(key.second),{}};
    std::size_t cursor=0;
    for(auto& item:pending) {
        const auto schedule=software::runtime::decode_macro_schedule_command(item.command);
        if(schedule.start_cycle<cursor) throw std::runtime_error("interleaved command domains require operator scheduling before Macro emission");
        if(schedule.start_cycle>cursor) queue.commands.push_back(control_command(isa::encode_icu_nop(schedule.start_cycle-cursor)));
        const auto index=static_cast<std::uint32_t>(queue.commands.size());
        const auto& sequence=*item.source;
        if(sequence.address_binding>=0) addressRelocations.push_back({static_cast<std::uint32_t>(sequence.address_binding),
            sequence.address_binding_access,key.first,static_cast<std::uint16_t>(key.second),index,false});
        if(sequence.scale_binding>=0) scaleRelocations.push_back({static_cast<std::uint32_t>(sequence.scale_binding),0,
            key.first,static_cast<std::uint16_t>(key.second),index,VxmImmediateOperand::Rhs});
        cursor=schedule.start_cycle+(schedule.inner_count-1)*schedule.inner_interval+(schedule.outer_count-1)*schedule.outer_interval+1;
        max_cycle=std::max(max_cycle,cursor-1);queue.commands.push_back(std::move(item.command));
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
            PackedWeightImage image;
            try {
                image = pack_weight_binding_page(*binding, use.page_index,
                    std::span<const std::uint8_t>(
                        geometryBytes.data(), logicalBytes),
                    program.hardware);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "weight binding " + std::to_string(binding->index)
                    + " page " + std::to_string(use.page_index)
                    + " layout " + std::to_string(
                        static_cast<int>(binding->layout))
                    + " target_rows " + std::to_string(program.hardware.mxm_rows)
                    + " target_bytes " + std::to_string(program.hardware.bytes_per_word)
                    + ": " + error.what());
            }
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
            throw std::runtime_error(
                "paged MEM_WRITE_SYNC has no segments: binding="
                + std::to_string(program.weight_page_uses.at(
                    plan.use_indices.front()).binding_index)
                + " page=" + std::to_string(plan.page_index)
                + " bank=" + std::to_string(plan.bank));
        pages.push_back(std::move(page));
    }
    schedule_weight_prefetches(program, plans);
    for (std::size_t index = 0; index < pages.size(); ++index)
        pages[index].fabric_streams = plans[index].fabric_streams;

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
    const auto beforeSync = program;
    for (const auto index : launchOrder) {
        const auto& plan = plans[index];
        // The compiler owns the consumer rendezvous as well as the MEM
        // page write. Runtime binds the concrete DDR addresses/events.
        const auto epoch = std::ranges::count_if(plans,
            [&](const auto& candidate) {
                return !candidate.pre_execution
                    && candidate.ready_cycle < plan.ready_cycle;
            });
        static_cast<void>(pager.schedule(program, pages[index],
            static_cast<std::size_t>(plan.start_cycle),
            static_cast<std::size_t>(plan.transfer_end_cycle),
            static_cast<std::size_t>(plan.ready_cycle),
            0x10000u + index, 0x30000u + epoch,
            static_cast<std::size_t>(plan.earliest_mem_cycle)));
    }
    static_cast<void>(pager.coalesce_page_ready_events());
    pager.finalize_schedule(program);
    software::runtime::record_compiler_sync_origins(program, beforeSync);
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

void lower_direct_raw_macro(software::runtime::BinaryProgram& program)
{
    using namespace software::runtime;
    using Key = std::pair<QueueKind, std::size_t>;
    std::map<Key, std::vector<std::vector<std::size_t>>> indices;
    for (auto& queue : program.queues) {
        if (queue.kind != QueueKind::Mem && queue.kind != QueueKind::MxmLoad
            && queue.kind != QueueKind::MxmCompute && queue.kind != QueueKind::MxmDequant
            && queue.kind != QueueKind::SxmTranspose && queue.kind != QueueKind::SxmPermute)
            continue;
        if (queue.commands.empty()) continue;
        const bool allMacro = std::ranges::all_of(queue.commands,
            [](const auto& command) { return is_macro_schedule_command(command); });
        if (!allMacro) {
            try {
                auto lowered = link_macro_queue_controls(queue);
                indices.emplace(Key {queue.kind, queue.index}, std::move(lowered.command_indices));
                queue = std::move(lowered.queue);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "raw-macro contains an unmapped synchronization control or an unrepresentable FU loop: "
                    + std::string(queue_kind_name(queue.kind)) + "["
                    + std::to_string(queue.index) + "]: " + error.what());
            }
        }
        queue.requires_raw_macro = !queue.commands.empty();
        queue.packed_macro_imem.reset();
    }
    // One coarse loop can become multiple descriptors. Every descriptor must
    // retain its binding/scale owner, including after chronological sorting.
    const auto remap = [&](auto& relocations) {
        using Container = std::remove_reference_t<decltype(relocations)>;
        Container mapped;
        for (const auto& relocation : relocations) {
            const auto queue = indices.find(Key {relocation.queue_kind, relocation.queue_index});
            if (queue == indices.end()) { mapped.push_back(relocation); continue; }
            if (relocation.command_index >= queue->second.size()
                || queue->second[relocation.command_index].empty())
                throw std::runtime_error("raw-macro relocation does not name a functional packet header");
            for (const auto index : queue->second[relocation.command_index]) {
                if (index > std::numeric_limits<std::uint32_t>::max())
                    throw std::runtime_error("raw-macro relocation index exceeds the binary format");
                auto copy = relocation;
                copy.command_index = static_cast<std::uint32_t>(index);
                mapped.push_back(copy);
            }
        }
        relocations = std::move(mapped);
    };
    remap(program.address_relocations);
    remap(program.scale_relocations);
    const auto physical = analyze_physical_imem(program);
    if (!physical.fits()) {
        try {
            static_cast<void>(plan_compute_imem_segments(program));
        } catch (const std::exception& error) {
            throw std::runtime_error("raw-macro exceeds physical i-MEM or Active Context capacity: "
                + std::string(error.what()));
        }
    }
}

} // namespace

software::runtime::BinaryProgram translate_command_module(mlir::ModuleOp module)
{
    const bool progress=module->hasAttr("ftlpu.report_encoding_progress");
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
    IcuCompressionMode compressionMode = IcuCompressionMode::RawMacro;
    if (const auto compressionAttr =
            module->getAttrOfType<mlir::StringAttr>(
                "ftlpu.icu_compression")) {
        const auto parsed = parse_icu_compression_mode(
            compressionAttr.getValue().str());
        if (!parsed)
            throw std::runtime_error(
                "Command IR module has an invalid ftlpu.icu_compression");
        compressionMode = *parsed;
    }
    QueueMap queues;
    MacroQueueMap macroQueues;
    std::vector<BinaryBinding> bindings;
    std::vector<BinaryTimeline> timelines;
    std::vector<BinaryWeightPageUse> weightPageUses;
    if(progress)llvm::errs()<<"encode: translating bindings\n";
    module.walk([&](command::BindingOp op) {
        try { bindings.push_back(translate_binding(op)); }
        catch(const std::exception& error) {
            throw std::runtime_error("binding " + std::to_string(op.getIndex())
                + " translation failed: " + error.what());
        }
    });
    if(progress)llvm::errs()<<"encode: bindings="<<bindings.size()<<"; collecting timelines\n";
    module.walk([&](command::TimelineOp op) {
        timelines.push_back(BinaryTimeline {
            op.getName().str(),
            static_cast<std::uint64_t>(op.getStart()),
            static_cast<std::uint64_t>(op.getEnd()),
        });
    });
    module.walk([&](command::WeightPageOp op) {
        const auto runtimePrefetch =
            op->getAttrOfType<mlir::BoolAttr>("runtime_prefetch");
        weightPageUses.push_back(BinaryWeightPageUse {
            static_cast<std::uint32_t>(op.getBindingIndex()),
            static_cast<std::uint32_t>(op.getPageIndex()),
            static_cast<std::uint16_t>(op.getBank()),
            static_cast<std::uint64_t>(op.getReadyCycle()),
            static_cast<std::uint64_t>(op.getReleaseCycle()),
            runtimePrefetch && runtimePrefetch.getValue(),
        });
    });
    std::optional<std::uint64_t> lastPagedConsumer;
    if (requiresDirectLowering)
        for (const auto& use : weightPageUses)
            lastPagedConsumer = std::max(
                lastPagedConsumer.value_or(0), use.ready_cycle);
    if(progress)llvm::errs()<<"encode: collecting stream release cycles\n";
    auto streamReleaseSummary = stream_release_cycles(
        module, *target, lastPagedConsumer);
    if(progress)llvm::errs()<<"encode: collecting queue domains\n";
    module.walk([&](command::MemOp op) { collect_mem(op, queues); });
    module.walk([&](command::MemBundleOp op) {
        collect_mem_bundle(op, queues);
    });
    module.walk([&](command::MxmOp op) {
        if (is_native4_decode(op))
            collect_native4_decode(op, queues);
        else
            collect_mxm(op, queues);
    });
    module.walk([&](command::MxmDequantOp op) {
        collect_mxm_dequant(op, queues);
    });
    module.walk([&](command::VxmOp op) { collect_vxm(op, queues); });
    module.walk([&](command::SxmOp op) { collect_sxm(op, queues); });

    module.walk([&](command::MacroOp op) {
        auto encoded=command::readMacroCommand(op);
        const auto schedule=software::runtime::decode_macro_schedule_command(encoded);
        const auto final=schedule.start_cycle+(schedule.inner_count-1)*schedule.inner_interval
            +(schedule.outer_count-1)*schedule.outer_interval;
        macroQueues[{static_cast<QueueKind>(op.getKind()),op.getQueue()}].push_back({schedule.start_cycle,final,
            {std::move(encoded)},static_cast<int64_t>(op.getAddressBinding().value_or(-1)),address_binding_access(op),static_cast<int64_t>(op.getScaleBinding().value_or(-1))});
    });
    if (queues.empty() && macroQueues.empty())
        throw std::runtime_error("Command IR module has no queue commands");
    // A raw FU packet owns the physical queue i-MEM image. Mixing it with
    // legacy CommandSequence output would require an explicit packet-order
    // model and must not silently route the raw instruction through the
    // legacy compression path.
    for (const auto& [key, packets] : macroQueues) {
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
    if (requiresDirectLowering || !macroQueues.empty()) cycle_origin = 0;
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
    hardware.vxm_reserved_feature = 1;
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
    hardware.common_barrier_encoding_version = static_cast<std::uint32_t>(icuMemory.common_barrier_encoding_version);
    hardware.common_barrier_capacity = static_cast<std::uint32_t>(icuMemory.common_barrier_capacity);
    hardware.common_transport_overlap_version = static_cast<std::uint32_t>(icuMemory.common_transport_overlap_version);
    hardware.queue_control_version = static_cast<std::uint32_t>(icuMemory.queue_control_version);
    hardware.queue_control_capacity = static_cast<std::uint32_t>(icuMemory.queue_control_capacity);
    hardware.queue_control_cache_depth = static_cast<std::uint32_t>(icuMemory.queue_control_cache_depth);
    hardware.queue_control_read_latency = static_cast<std::uint32_t>(icuMemory.queue_control_read_latency);
    hardware.queue_control_read_ports = static_cast<std::uint32_t>(icuMemory.queue_control_read_ports);
    hardware.queue_control_storage_version = static_cast<std::uint32_t>(icuMemory.queue_control_storage_version);
    hardware.queue_control_pool_capacity = static_cast<std::uint32_t>(icuMemory.queue_control_pool_capacity);
    hardware.queue_control_pool_read_ii = static_cast<std::uint32_t>(icuMemory.queue_control_pool_read_ii);

#undef COPY_ICU_MEMORY
    if (compressionMode == IcuCompressionMode::RawMacro
        && (hardware.icu_mem_instruction_bits != 96 || hardware.icu_mxm_instruction_bits != 128
            || hardware.icu_macro_encoding_version != 1))
        throw std::runtime_error("raw-macro requires the 96-bit MEM / 128-bit MXM target");

    if(progress)llvm::errs()<<"encode: planning memory floors\n";
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
            program.address_relocations));
    for (auto& [key, packets] : macroQueues) {
        if(progress)llvm::errs()<<"encode: "<<raw_loop_queue_name(key.first)<<"["<<key.second<<"] domains="<<packets.size()<<"\n";
        std::stable_sort(packets.begin(), packets.end(),
            [](const MacroCommandSequence& lhs,
                const MacroCommandSequence& rhs) {
                return lhs.start_cycle < rhs.start_cycle;
            });
        validate_sequential_frontend(key, packets);
        QueueProgram queue {key.first,
            static_cast<std::size_t>(key.second), {}};
        // The physical ICU owns one active coarse instruction. Encode the
        // queue-local gap as a control boundary before final Macro linking.
        std::size_t cursor = 0;
        for (MacroCommandSequence& packet : packets) {
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
                queue.commands.push_back(control_command(isa::encode_icu_nop(wait)));
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
    const auto commonBarrierAttr = module->getAttrOfType<mlir::BoolAttr>(
        "ftlpu.common_barrier_protocol");
    // Paged Macro programs use the common-barrier linker directly. The retired
    // Native loop linker must not be used to manufacture a paging recipe.
    if (compressionMode == IcuCompressionMode::RawMacro && requiresDirectLowering
        && program.weight_page_uses.empty()
        && program.hardware.queue_control_version == 1
        && !(commonBarrierAttr && commonBarrierAttr.getValue())) {
        // Keep the compact unpacked plan and its
        // coarse domains for a checked final paging link after relocation.
        emit_compiled_mem_write_sync(program);
        emit_compiled_mem_read_sync(program);
        software::runtime::BinaryQueueLinkSource source;
        source.required = !program.weight_page_uses.empty()
            || std::ranges::any_of(program.queues, [](const auto& queue) {
                return queue.kind == software::runtime::QueueKind::Mem
                    && std::ranges::any_of(queue.commands, [](const auto& command) {
                        return software::runtime::is_mem_synchronized_raw_packet_header(command);
                    });
            });
        source.queues = program.queues;
        source.address_relocations = program.address_relocations;
        source.scale_relocations = program.scale_relocations;
        program.queue_link_source = std::move(source);
        software::runtime::link_queue_controls(program);
        software::runtime::validate_queue_sync_plan(program);
    } else if (compressionMode == IcuCompressionMode::RawMacro
        || (commonBarrierAttr && commonBarrierAttr.getValue())) {
        // Runtime paging owns concrete transport instances and final common
        // logical stops. Keep the computation image free of queue-local
        // WAIT/SYNC; weight_page_uses and exclusive ingress timelines retain
        // the information needed for the checked final link.
        if (!program.weight_page_uses.empty()
            && (program.hardware.common_barrier_encoding_version != 1
                || program.hardware.common_barrier_capacity == 0))
            throw std::runtime_error("Raw Macro paging requires the common barrier target capability");
        program.common_barrier_protocol = program.hardware.common_barrier_encoding_version == 1;
        const auto overlapPolicy=module->getAttrOfType<mlir::BoolAttr>("ftlpu.common_transport_overlap");
        program.transport_overlap_protocol = program.common_barrier_protocol
            && !program.weight_page_uses.empty() && program.hardware.common_transport_overlap_version == 1
            && (!overlapPolicy || overlapPolicy.getValue());
        if (compressionMode == IcuCompressionMode::RawMacro)
            lower_direct_raw_macro(program);
        if (requiresDirectLowering) {
            software::runtime::BinaryProgram readback;
            readback.hardware = program.hardware;
            readback.bindings = program.bindings;
            emit_compiled_mem_read_sync(readback);
            program.transport_queues = std::move(readback.queues);
            for (auto& relocation : readback.address_relocations) {
                relocation.transport_queue = true;
                program.address_relocations.push_back(relocation);
            }
        }
    } else if (requiresDirectLowering) {
        emit_compiled_mem_write_sync(program);
        emit_compiled_mem_read_sync(program);
    }
    return program;
}

} // namespace ftlpu::compiler::target
