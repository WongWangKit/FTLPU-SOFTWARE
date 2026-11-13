#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/program_loader.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu::software::runtime {

struct BinaryProgram;

enum class QueueKind : std::uint16_t {
    Mem = 0,
    MxmLoad = 1,
    MxmCompute = 2,
    MxmDequant = 3,
    Vxm = 4,
    SxmTranspose = 5,
    SxmPermute = 6,
    C2cDma = 7,
    C2cTx = 8,
    C2cRx = 9,
};

enum class InstructionKind : std::uint16_t {
    None = 0,
    Mem = 1,
    Mxm = 2,
    Vxm = 3,
    Sxm = 4,
    MxmDequant = 5,
    C2cEndpoint = 6,
    C2cDma = 7,
};

struct QueueCommand {
    isa::EncodedIcuCommand command{0};
    InstructionKind instruction_kind{InstructionKind::None};
    std::uint16_t word_count{0};
    std::array<std::uint32_t, 4> words{};
    // MEM/MXM/VXM fit in the fixed header. Extended ICU descriptors and SXM's
    // variable stream/lane maps use this trailing payload.
    std::vector<std::uint32_t> extension_words{};
};

// Word-aligned physical i-MEM bytes retained by the binary reader for an
// all-Macro queue. software_valid_bit_length is file/container metadata: it
// includes word 0 and the meaningful payload bits, but excludes final word
// padding. Runtime writes bytes to CModel in lane/bit-low-first order.
struct PackedMacroImemImage {
    std::uint32_t word_bits{0};
    std::uint64_t software_valid_bit_length{0};
    std::vector<std::uint8_t> bytes{};

    std::size_t word_count() const
    {
        return word_bits == 0 ? 0 : bytes.size() * 8 / word_bits;
    }
};

inline constexpr std::uint32_t kIcuMacroScheduleMagic = 0x4d414352u;
inline bool is_macro_schedule_command(const QueueCommand& command)
{
    return command.instruction_kind != InstructionKind::None
        && isa::decode_icu_command_opcode(command.command)
            == isa::IcuCommandOpcode::Extended
        && (command.extension_words.size() == 9
            || (command.instruction_kind == InstructionKind::Sxm && command.extension_words.size() > 9))
        && command.extension_words[0] == kIcuMacroScheduleMagic;
}

inline IcuMacroSchedule decode_macro_schedule_command(
    const QueueCommand& command)
{
    if (!is_macro_schedule_command(command))
        throw std::logic_error("queue command is not an ICU macro schedule");
    return IcuMacroSchedule {
        command.extension_words[1],
        command.extension_words[2],
        command.extension_words[3],
        static_cast<std::int32_t>(command.extension_words[4]),
        command.extension_words[5],
        command.extension_words[6],
        static_cast<std::int32_t>(command.extension_words[7]),
        static_cast<IcuInductionTarget>(command.extension_words[8]),
    };
}

inline QueueCommand encode_macro_schedule_command(
    QueueCommand instruction, const IcuMacroSchedule& schedule)
{
    const auto fits_u32 = [](std::size_t value) {
        return value <= std::numeric_limits<std::uint32_t>::max();
    };
    const auto fits_i32 = [](std::int64_t value) {
        return value >= std::numeric_limits<std::int32_t>::min()
            && value <= std::numeric_limits<std::int32_t>::max();
    };
    if (instruction.instruction_kind == InstructionKind::None
        || isa::decode_icu_command_opcode(instruction.command)
            != isa::IcuCommandOpcode::Instruction
        || (!instruction.extension_words.empty() && instruction.instruction_kind != InstructionKind::Sxm))
        throw std::invalid_argument(
            "ICU macro requires one native functional instruction");
    if (schedule.inner_count == 0 || schedule.outer_count == 0
        || schedule.inner_interval == 0 || schedule.outer_interval == 0
        || !fits_u32(schedule.start_cycle)
        || !fits_u32(schedule.inner_count)
        || !fits_u32(schedule.inner_interval)
        || !fits_u32(schedule.outer_count)
        || !fits_u32(schedule.outer_interval)
        || !fits_i32(schedule.inner_stride)
        || !fits_i32(schedule.outer_stride))
        throw std::invalid_argument(
            "ICU macro schedule does not fit the binary descriptor");
    if (static_cast<std::uint8_t>(schedule.induction_target)
        > static_cast<std::uint8_t>(
            IcuInductionTarget::MxmAccumulatorAddress))
        throw std::invalid_argument("ICU macro has an invalid induction target");

    instruction.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Extended);
    const auto nativeExtension = std::move(instruction.extension_words);
    instruction.extension_words = {
        kIcuMacroScheduleMagic,
        static_cast<std::uint32_t>(schedule.start_cycle),
        static_cast<std::uint32_t>(schedule.inner_count),
        static_cast<std::uint32_t>(schedule.inner_interval),
        static_cast<std::uint32_t>(schedule.inner_stride),
        static_cast<std::uint32_t>(schedule.outer_count),
        static_cast<std::uint32_t>(schedule.outer_interval),
        static_cast<std::uint32_t>(schedule.outer_stride),
        static_cast<std::uint32_t>(schedule.induction_target),
    };
    instruction.extension_words.insert(instruction.extension_words.end(),nativeExtension.begin(),nativeExtension.end());
    return instruction;
}

QueueCommand encode_sxm_command(const SxmInstruction& instruction);
SxmInstruction decode_sxm_command(const QueueCommand& command);

struct QueueProgram {
    QueueKind kind{QueueKind::Mem};
    std::size_t index{0};
    std::vector<QueueCommand> commands{};
    // Present when this QueueProgram came from a packed-Macro .ftlpu record.
    // Any operation that mutates a command must reset this cached image.
    std::optional<PackedMacroImemImage> packed_macro_imem{};
    // A physical packed queue must not silently fall back to FU packets if a
    // later linker inserts native controls. Persisted by MacroPacked encoding.
    bool requires_raw_macro{false};
};

// Raw control records preserve one physical 96-bit local-iMEM word. They are
// used for tagged WAIT_EVENT/NOTIFY/SYNC commands that do not fit the legacy
// 32-bit QueueCommand::command field.
QueueCommand encode_icu_control_raw_word(
    const IcuControlInstruction& instruction);
bool is_icu_control_raw_word_command(
    const QueueCommand& command) noexcept;
IcuControlInstruction decode_icu_control_raw_word(
    const QueueCommand& command);

// MEM_WRITE_SYNC is one coarse MEM ICU instruction encoded as two consecutive
// physical 96-bit words: a synchronized header and a native MEM Write
// template. Each QueueCommand below corresponds to exactly one iMEM slot.
std::array<QueueCommand,
    InstructionControlUnit::MemIcu::synchronized_packet_word_count>
encode_mem_synchronized_icu_packet(
    const InstructionControlUnit::MemIcu::EncodedSynchronizedPacket& packet);
bool is_mem_synchronized_raw_word_command(
    const QueueCommand& command) noexcept;
bool is_mem_synchronized_raw_packet_header(
    const QueueCommand& command) noexcept;
InstructionControlUnit::MemIcu::EncodedSynchronizedPacket
decode_mem_synchronized_icu_packet(
    const QueueProgram& queue, std::size_t command_index);
// Recover the scheduled MEM idle slots for a direct CModel golden run or a
// transport-plan check. The executable keeps its compiler-authored packets.
BinaryProgram without_compiled_mem_write_sync(const BinaryProgram& compiled);
// Host-only coarse-domain timing, kept separately from hardware i-MEM words,
// lets rebinding recover the pre-SYNC queue phases without expanding FU issues.
void record_compiler_sync_origins(BinaryProgram& compiled,
    const BinaryProgram& before_sync);
// Recover compiler-authored KV-cache page-out packets. Runtime links C2C TX
// and DMA around the extracted MEM_READ_SYNC program after main execution.
BinaryProgram without_compiled_mem_read_sync(const BinaryProgram& compiled);

// C2C queue records preserve one physical 96-bit local-iMEM word per
// QueueCommand.  Endpoint packets contain one record and DMA packets contain
// two consecutive records.  Keeping this boundary visible makes instruction
// memory accounting match the hardware fetch path.
inline QueueCommand encode_c2c_raw_word(
    InstructionKind kind, const C2cEndpointIcuPacket& word)
{
    if (kind != InstructionKind::C2cEndpoint
        && kind != InstructionKind::C2cDma)
        throw std::invalid_argument(
            "C2C raw word requires a C2C instruction kind");
    QueueCommand command;
    command.command = static_cast<isa::EncodedIcuCommand>(
        isa::IcuCommandOpcode::Instruction);
    command.instruction_kind = kind;
    command.word_count = 3;
    command.words[0] = word.lanes[0];
    command.words[1] = word.lanes[1];
    command.words[2] = word.lanes[2];
    return command;
}

inline C2cEndpointIcuPacket decode_c2c_raw_word(
    const QueueCommand& command, InstructionKind expected_kind)
{
    if ((expected_kind != InstructionKind::C2cEndpoint
            && expected_kind != InstructionKind::C2cDma)
        || command.instruction_kind != expected_kind
        || command.word_count != 3
        || !command.extension_words.empty()
        || isa::decode_icu_command_opcode(command.command)
            != isa::IcuCommandOpcode::Instruction)
        throw std::logic_error("malformed C2C 96-bit raw word");
    return C2cEndpointIcuPacket {{
        command.words[0], command.words[1], command.words[2]}};
}

inline QueueCommand encode_c2c_endpoint_icu_packet(
    const C2cEndpointIcuPacket& packet)
{
    return encode_c2c_raw_word(
        InstructionKind::C2cEndpoint, packet);
}

inline std::array<QueueCommand, C2cDmaIcuPacket::kWordCount>
encode_c2c_dma_icu_packet(const C2cDmaIcuPacket& packet)
{
    std::array<QueueCommand, C2cDmaIcuPacket::kWordCount> commands {};
    for (std::size_t word = 0; word < commands.size(); ++word)
        commands[word] = encode_c2c_raw_word(
            InstructionKind::C2cDma, packet.words[word]);
    return commands;
}

inline C2cDmaIcuPacket decode_c2c_dma_icu_packet(
    const QueueProgram& queue, std::size_t command_index)
{
    if (queue.kind != QueueKind::C2cDma
        || command_index > queue.commands.size()
        || C2cDmaIcuPacket::kWordCount
            > queue.commands.size() - command_index)
        throw std::logic_error("truncated C2C DMA raw packet");
    C2cDmaIcuPacket packet {};
    for (std::size_t word = 0; word < packet.words.size(); ++word)
        packet.words[word] = decode_c2c_raw_word(
            queue.commands[command_index + word],
            InstructionKind::C2cDma);
    return packet;
}

class IcuProgram {
public:
    void emit_mem(std::size_t cycle, std::size_t column, MemInstruction instruction);
    void emit_mxm_load(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction);
    void emit_mxm_dequant(std::size_t cycle, std::size_t mxm, MxmDequantInstruction instruction);
    void emit_mxm_compute(std::size_t cycle, std::size_t mxm, MxmControlInstruction instruction);
    void emit_vxm(std::size_t cycle, std::size_t alu,
        VxmChainDepth depth, VxmLaneAluInstruction instruction);
    void emit_vxm(std::size_t cycle, std::size_t alu, VxmLaneAluInstruction instruction);
    void emit_sxm_transpose(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction);
    void emit_sxm_permute(std::size_t cycle, Hemisphere hemisphere, SxmInstruction instruction);

    std::vector<QueueProgram> encode_queues() const;
    void load_into(InstructionControlUnit& icu) const;

    std::size_t last_cycle() const;
    bool empty() const;

private:
    template <typename Instruction>
    struct ScheduledInstruction {
        std::size_t cycle{0};
        Instruction instruction{};
    };

    using MemQueue = std::vector<ScheduledInstruction<MemInstruction>>;
    using MxmQueue = std::vector<ScheduledInstruction<MxmControlInstruction>>;
    using MxmDequantQueue =
        std::vector<ScheduledInstruction<MxmDequantInstruction>>;
    struct VxmInstruction {
        VxmChainDepth depth{VxmChainDepth::Eight};
        VxmLaneAluInstruction instruction{};
    };
    using VxmQueue = std::vector<ScheduledInstruction<VxmInstruction>>;
    using SxmQueue = std::vector<ScheduledInstruction<SxmInstruction>>;

    void check_mem_column(std::size_t column) const;
    void check_mxm(std::size_t mxm) const;
    void check_vxm_alu(std::size_t alu) const;

    template <typename Instruction, typename EncodeFn>
    static std::vector<QueueCommand> encode_scheduled_queue(
        std::vector<ScheduledInstruction<Instruction>> events,
        const std::string& queue_name,
        EncodeFn encode);

    template <typename Instruction, typename NopFn, typename EmitFn>
    static void load_scheduled_queue(
        std::vector<ScheduledInstruction<Instruction>> events,
        const std::string& queue_name,
        NopFn nop,
        EmitFn emit);

    std::array<MemQueue, InstructionControlUnit::kMemQueues> mem_{};
    std::array<MxmQueue, InstructionControlUnit::kMxmQueues> mxm_load_{};
    std::array<MxmDequantQueue, InstructionControlUnit::kMxmQueues>
        mxm_dequant_{};
    std::array<MxmQueue, InstructionControlUnit::kMxmQueues> mxm_compute_{};
    std::array<VxmQueue, InstructionControlUnit::kVxmQueues> vxm_{};
    std::array<SxmQueue, hw::kHemispheres> sxm_transpose_{};
    std::array<SxmQueue, hw::kHemispheres> sxm_permute_{};
    std::size_t last_cycle_{0};
};

const char* queue_kind_name(QueueKind kind);
void load_transport_queue_programs_into_icu(const std::vector<QueueProgram>& queues,
    InstructionControlUnit& icu);


struct ProgramQueueImage {
    ProgramImage descriptor{};
    std::vector<std::uint8_t> bytes{};
};

// No semantic entries are installed here. Unsupported physical codecs fail
// before DDR or any target queue is modified.
std::vector<ProgramQueueImage> prepare_program_images(
    const std::vector<QueueProgram>& queues, std::uint64_t source_base,
    std::size_t logical_mxms_per_hemisphere = hw::kMxmsPerHemisphere);
std::vector<ProgramQueueImage> prepare_program_images(
    const BinaryProgram& program, std::uint64_t source_base, bool include_imem = true);
std::size_t load_queue_programs_into_icu(const std::vector<QueueProgram>& queues,
    InstructionControlUnit& icu,
    std::size_t logical_mxms_per_hemisphere = hw::kMxmsPerHemisphere,
    const BinaryProgram* control_program = nullptr, bool prime_frontends = true);

} // namespace ftlpu::software::runtime
