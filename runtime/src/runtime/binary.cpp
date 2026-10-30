// Keep serialization rebuilt when BinaryProgram or BinaryBinding ABI evolves.
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <streambuf>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>

namespace ftlpu::software::runtime {

namespace {

constexpr std::array<char, 8> kMagic {'F', 'T', 'L', 'P', 'U', 'B', '0', '1'};
constexpr std::uint8_t kCompactInstructionKindMask = 0x07;
constexpr std::uint8_t kCompactWordCountMask = 0x38;
constexpr std::uint8_t kCompactWordCountShift = 3;
constexpr std::uint8_t kCompactHasExtension = 0x40;
constexpr std::uint8_t kCompactMacro = 0x80;

// Internal .ftlpu container discriminant. This is not the two-bit hardware
// QueueMode (00 Native, 01 Macro); values 1 and 2 remain readable for old
// container formats.
enum class QueueEncodingMode : std::uint8_t {
    Native = 0,
    MacroSchedule = 1,
    CompactTagged = 2,
    MacroPacked = 3,
};

constexpr bool has_queue_encoding_mode(std::uint32_t version)
{
    // Version 26 on origin/main added VXM capability fields but retained the
    // v25 compact queue layout. Local experimental v27/v28 binaries introduced
    // queue-level encoding modes; v29 is the first merged format.
    return version == 27 || version == 28 || version >= 29;
}

constexpr bool has_mem_macro_packed(std::uint32_t version)
{
    return version == 28 || version >= 29;
}

constexpr bool has_mxm_macro_packed(std::uint32_t version)
{
    // v33-v35 on main used semantic MXM descriptors. The experimental v33
    // format has conflicting binding layout IDs and is not this ABI.
    return version >= 36;
}

constexpr bool is_mxm_macro_queue(QueueKind kind)
{
    return kind == QueueKind::MxmLoad || kind == QueueKind::MxmCompute ||
           kind == QueueKind::MxmDequant;
}

constexpr bool has_macro_packed(std::uint32_t version, QueueKind kind)
{
    if (kind == QueueKind::Mem)
        return has_mem_macro_packed(version);
    return is_mxm_macro_queue(kind) && has_mxm_macro_packed(version);
}

constexpr std::uint32_t kNativeWordCountShift = 2;
constexpr std::uint32_t kNativeWordCountMask = 0x1cu;
constexpr std::uint32_t kNativeSxmFixedWordCount = 7;
constexpr std::uint32_t kIcuExtendedReservedMask = 0x3fu;
constexpr std::uint32_t kIcuExtendedSubtypeShift = 8;
constexpr std::uint32_t kIcuExtendedRepeat2D = 0;

template <typename T>
void write_scalar(std::ostream& os, T value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error("failed to write FTLPU binary");
    }
}

template <typename T>
T read_scalar(std::istream& is)
{
    T value {};
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!is) {
        throw std::runtime_error("truncated FTLPU binary");
    }
    return value;
}

template <typename T>
void write_le(std::ostream& os, T value)
{
    static_assert(std::is_unsigned_v<T>);
    for (unsigned i = 0; i < sizeof(T); ++i)
        write_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(value >> (8 * i)));
}

InstructionKind instruction_kind_for_queue(QueueKind kind)
{
    switch (kind) {
    case QueueKind::Mem:
        return InstructionKind::Mem;
    case QueueKind::MxmLoad:
    case QueueKind::MxmCompute:
        return InstructionKind::Mxm;
    case QueueKind::MxmDequant:
        return InstructionKind::MxmDequant;
    case QueueKind::Vxm:
        return InstructionKind::Vxm;
    case QueueKind::SxmTranspose:
    case QueueKind::SxmPermute:
        return InstructionKind::Sxm;
    case QueueKind::C2cTx:
    case QueueKind::C2cRx:
        return InstructionKind::C2cEndpoint;
    case QueueKind::C2cDma:
        return InstructionKind::C2cDma;
    }
    throw std::runtime_error("invalid FTLPU queue kind");
}

bool is_sxm_queue(QueueKind kind)
{
    return kind == QueueKind::SxmTranspose
        || kind == QueueKind::SxmPermute;
}

QueueCommand sxm_queue_command(const SxmInstruction& instruction)
{
    QueueCommand command {
        static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction),
        InstructionKind::Sxm, 4, {},
    };
    command.words[0] = static_cast<std::uint32_t>(instruction.opcode);
    command.words[1] =
        static_cast<std::uint32_t>(instruction.shift_source);
    command.words[2] = instruction.shift_distance;
    command.words[3] =
        (instruction.output_row == SxmInstruction::kAllOutputRows
                ? 0xffu
                : static_cast<std::uint32_t>(instruction.output_row))
        | ((instruction.input_row == SxmInstruction::kAllInputRows
                   ? 0xffu
                   : static_cast<std::uint32_t>(instruction.input_row))
            << 8)
        | ((instruction.output_tile == SxmInstruction::kAllOutputTiles
                   ? 0xffu
                   : static_cast<std::uint32_t>(instruction.output_tile))
            << 16);
    command.extension_words.push_back(
        static_cast<std::uint32_t>(instruction.src_streams.size()));
    command.extension_words.push_back(
        static_cast<std::uint32_t>(instruction.dst_streams.size()));
    for (const auto stream : instruction.src_streams)
        command.extension_words.push_back(
            static_cast<std::uint32_t>(stream.stream));
    for (const auto stream : instruction.dst_streams)
        command.extension_words.push_back(
            static_cast<std::uint32_t>(stream.stream));
    for (const auto lane : instruction.permute_map)
        command.extension_words.push_back(
            lane == SxmInstruction::kZeroFill
                ? UINT32_MAX
                : static_cast<std::uint32_t>(lane));
    return command;
}

SxmInstruction decode_sxm_queue_command(const QueueCommand& command)
{
    if (command.instruction_kind != InstructionKind::Sxm
        || command.word_count != 4
        || command.extension_words.size()
            < 2 + SxmInstruction::kTotalLanes)
        throw std::runtime_error(
            "SXM queue command has an invalid variable payload");
    SxmInstruction instruction {};
    instruction.opcode = static_cast<SxmOpcode>(command.words[0]);
    instruction.shift_source =
        static_cast<SxmShiftSource>(command.words[1]);
    instruction.shift_distance = command.words[2];
    const auto outputRow = command.words[3] & 0xffu;
    const auto inputRow = (command.words[3] >> 8) & 0xffu;
    const auto outputTile = (command.words[3] >> 16) & 0xffu;
    instruction.output_row = outputRow == 0xffu
        ? SxmInstruction::kAllOutputRows : outputRow;
    instruction.input_row = inputRow == 0xffu
        ? SxmInstruction::kAllInputRows : inputRow;
    instruction.output_tile = outputTile == 0xffu
        ? SxmInstruction::kAllOutputTiles : outputTile;
    const auto sourceCount = command.extension_words[0];
    const auto destinationCount = command.extension_words[1];
    const std::size_t mapBegin = 2 + sourceCount + destinationCount;
    if (command.extension_words.size()
        != mapBegin + SxmInstruction::kTotalLanes)
        throw std::runtime_error("SXM queue command has malformed streams");
    for (std::size_t index = 0; index < sourceCount; ++index)
        instruction.src_streams.push_back(SxmStreamId {
            command.extension_words[2 + index]});
    for (std::size_t index = 0; index < destinationCount; ++index)
        instruction.dst_streams.push_back(SxmStreamId {
            command.extension_words[2 + sourceCount + index]});
    for (std::size_t lane = 0;
         lane < SxmInstruction::kTotalLanes; ++lane) {
        const auto value = command.extension_words[mapBegin + lane];
        instruction.permute_map[lane] = value == UINT32_MAX
            ? SxmInstruction::kZeroFill : value;
    }
    return instruction;
}

std::array<std::uint32_t, 3> encode_binary_repeat_2d(
    const IcuRepeat2D& repeat)
{
    // Validate the public iteration-space contract with the hardware codec,
    // then repack it so word zero is a self-describing extended-control record.
    (void)isa::encode_icu_repeat_2d(repeat);
    std::array<std::uint32_t, 3> words {};
    const auto write = [&](std::size_t offset, std::size_t width,
                           std::uint64_t value) {
        for (std::size_t bit = 0; bit < width; ++bit)
            if ((value & (std::uint64_t {1} << bit)) != 0)
                words[(offset + bit) / 32]
                    |= std::uint32_t {1} << ((offset + bit) % 32);
    };
    write(0, 2, static_cast<std::uint8_t>(
        isa::IcuCommandOpcode::Extended));
    // Bits 2..7 remain zero to identify the fixed-width extended escape.
    write(kIcuExtendedSubtypeShift, 2, kIcuExtendedRepeat2D);
    write(10, 10, repeat.inner_count);
    write(20, 10, repeat.outer_count);
    write(30, 16, repeat.inner_interval);
    write(46, 16, repeat.outer_interval);
    write(62, 16, static_cast<std::uint16_t>(repeat.inner_stride));
    write(78, 16, static_cast<std::uint16_t>(repeat.outer_stride));
    write(94, 2, static_cast<std::uint8_t>(repeat.induction_target));
    return words;
}

IcuRepeat2D decode_binary_repeat_2d(
    const std::array<std::uint32_t, 3>& words)
{
    const auto read = [&](std::size_t offset, std::size_t width) {
        std::uint64_t value = 0;
        for (std::size_t bit = 0; bit < width; ++bit)
            if ((words[(offset + bit) / 32]
                    & (std::uint32_t {1} << ((offset + bit) % 32))) != 0)
                value |= std::uint64_t {1} << bit;
        return value;
    };
    const auto signExtend16 = [](std::uint64_t value) {
        return static_cast<std::int64_t>(
            static_cast<std::int16_t>(value));
    };
    if (isa::decode_icu_command_opcode(words[0])
            != isa::IcuCommandOpcode::Extended
        || ((words[0] >> 2) & kIcuExtendedReservedMask) != 0
        || read(kIcuExtendedSubtypeShift, 2) != kIcuExtendedRepeat2D)
        throw std::runtime_error("invalid compact Repeat2D record");
    IcuRepeat2D repeat {
        static_cast<std::size_t>(read(10, 10)),
        static_cast<std::size_t>(read(30, 16)),
        signExtend16(read(62, 16)),
        static_cast<std::size_t>(read(20, 10)),
        static_cast<std::size_t>(read(46, 16)),
        signExtend16(read(78, 16)),
        static_cast<IcuInductionTarget>(read(94, 2)),
    };
    (void)isa::encode_icu_repeat_2d(repeat);
    return repeat;
}

QueueCommand repeat_2d_queue_command(const IcuRepeat2D& repeat)
{
    const auto encoded = isa::encode_icu_repeat_2d(repeat);
    return QueueCommand {
        encoded.words[0], InstructionKind::None, 3,
        {encoded.words[0], encoded.words[1], encoded.words[2], 0},
    };
}

struct CompactQueueRecordHeader {
    InstructionKind instruction_kind;
    std::uint16_t word_count;
    bool has_extension;
    bool macro;
};

CompactQueueRecordHeader decode_compact_queue_record_header(
    std::uint8_t flags)
{
    const auto instruction_kind = static_cast<InstructionKind>(
        flags & kCompactInstructionKindMask);
    const auto word_count = static_cast<std::uint16_t>(
        (flags & kCompactWordCountMask) >> kCompactWordCountShift);
    const bool has_extension = (flags & kCompactHasExtension) != 0;
    const bool macro = (flags & kCompactMacro) != 0;
    if (static_cast<std::uint16_t>(instruction_kind)
            > static_cast<std::uint16_t>(InstructionKind::C2cDma)
        || word_count > QueueCommand {}.words.size()
        || (macro && (instruction_kind == InstructionKind::None
                         || has_extension)))
        throw std::runtime_error("invalid compact FTLPU queue record");
    return {instruction_kind, word_count, has_extension, macro};
}

std::uint8_t compact_queue_record_flags(const QueueCommand& command,
    bool macro)
{
    if (static_cast<std::uint16_t>(command.instruction_kind)
            > static_cast<std::uint16_t>(InstructionKind::C2cDma)
        || command.word_count > command.words.size()
        || command.extension_words.size()
            > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error(
            "FTLPU queue command does not fit compact encoding");
    return static_cast<std::uint8_t>(command.instruction_kind)
        | static_cast<std::uint8_t>(command.word_count
            << kCompactWordCountShift)
        | static_cast<std::uint8_t>(
            !macro && !command.extension_words.empty()
                ? kCompactHasExtension : 0)
        | static_cast<std::uint8_t>(macro ? kCompactMacro : 0);
}

void write_compact_queue_command(
    std::ostream& os, const QueueCommand& command)
{
    const bool macro = is_macro_schedule_command(command);
    write_scalar<std::uint8_t>(
        os, compact_queue_record_flags(command, macro));
    if (!macro) write_scalar<std::uint32_t>(os, command.command);
    for (std::uint16_t word = 0; word < command.word_count; ++word)
        write_scalar<std::uint32_t>(os, command.words[word]);
    if (macro) {
        const auto schedule = decode_macro_schedule_command(command);
        write_scalar<std::uint32_t>(os,
            static_cast<std::uint32_t>(schedule.start_cycle));
        write_scalar<std::uint32_t>(os,
            static_cast<std::uint32_t>(schedule.inner_count));
        write_scalar<std::uint32_t>(os,
            static_cast<std::uint32_t>(schedule.inner_interval));
        write_scalar<std::int32_t>(os,
            static_cast<std::int32_t>(schedule.inner_stride));
        write_scalar<std::uint32_t>(os,
            static_cast<std::uint32_t>(schedule.outer_count));
        write_scalar<std::uint32_t>(os,
            static_cast<std::uint32_t>(schedule.outer_interval));
        write_scalar<std::int32_t>(os,
            static_cast<std::int32_t>(schedule.outer_stride));
        write_scalar<std::uint8_t>(os,
            static_cast<std::uint8_t>(schedule.induction_target));
    } else if (!command.extension_words.empty()) {
        write_scalar<std::uint16_t>(os,
            static_cast<std::uint16_t>(command.extension_words.size()));
        for (const auto word : command.extension_words)
            write_scalar<std::uint32_t>(os, word);
    }
}

QueueEncodingMode queue_encoding_mode(const QueueProgram& queue, bool rawMacroSupported)
{
    const auto macroCount = std::count_if(queue.commands.begin(),
        queue.commands.end(), [](const QueueCommand& command) {
            return is_macro_schedule_command(command);
        });
    if (queue.requires_raw_macro && !queue.commands.empty()
        && (macroCount != static_cast<std::ptrdiff_t>(queue.commands.size())
            || !rawMacroSupported
            || (queue.kind != QueueKind::Mem && !is_mxm_macro_queue(queue.kind))))
        throw std::runtime_error(
            "raw-macro queue cannot be serialized as native FU packets or mixed controls");
    if (macroCount == 0) {
        const bool hasExtendedDescriptor = std::any_of(
            queue.commands.begin(), queue.commands.end(),
            [](const QueueCommand& command) {
                return is_icu_control_raw_word_command(command)
                    || isa::decode_icu_command_opcode(command.command)
                        == isa::IcuCommandOpcode::Extended;
            });
        return hasExtendedDescriptor
            ? QueueEncodingMode::CompactTagged
            : QueueEncodingMode::Native;
    }
    if (macroCount == static_cast<std::ptrdiff_t>(queue.commands.size())) {
        if (rawMacroSupported && (queue.kind == QueueKind::Mem || is_mxm_macro_queue(queue.kind)))
            return QueueEncodingMode::MacroPacked;
        return QueueEncodingMode::MacroSchedule;
    }
    return QueueEncodingMode::CompactTagged;
}

void write_native_queue_command(std::ostream& os, QueueKind queueKind,
    const QueueCommand& command)
{
    if (is_macro_schedule_command(command))
        throw std::runtime_error(
            "native FTLPU queue cannot contain a macro record");
    if (is_repeat_2d_command(command)) {
        const auto words = encode_binary_repeat_2d(
            decode_repeat_2d_command(command));
        for (const auto word : words) write_scalar(os, word);
        return;
    }
    if (command.instruction_kind == InstructionKind::None) {
        if (command.word_count != 0 || !command.extension_words.empty()
            || isa::decode_icu_command_opcode(command.command)
                == isa::IcuCommandOpcode::Instruction)
            throw std::runtime_error(
                "invalid native FTLPU control record");
        write_scalar(os, command.command);
        return;
    }
    if (command.instruction_kind != instruction_kind_for_queue(queueKind)
        || isa::decode_icu_command_opcode(command.command)
            != isa::IcuCommandOpcode::Instruction)
        throw std::runtime_error(
            "functional instruction does not match its FTLPU queue");
    if (is_sxm_queue(queueKind)) {
        const auto encoded = isa::encode_sxm_instruction(
            decode_sxm_queue_command(command));
        write_scalar<std::uint32_t>(os,
            kNativeSxmFixedWordCount << kNativeWordCountShift);
        for (const auto word : encoded.words) write_scalar(os, word);
        return;
    }
    if (command.word_count == 0 || command.word_count > 4
        || !command.extension_words.empty())
        throw std::runtime_error(
            "functional instruction does not fit native FTLPU encoding");
    write_scalar<std::uint32_t>(os,
        static_cast<std::uint32_t>(command.word_count)
            << kNativeWordCountShift);
    for (std::uint16_t word = 0; word < command.word_count; ++word)
        write_scalar(os, command.words[word]);
}

void write_macro_schedule_payload(
    std::ostream& os, const IcuMacroSchedule& schedule)
{
    write_scalar<std::uint32_t>(os,
        static_cast<std::uint32_t>(schedule.start_cycle));
    write_scalar<std::uint32_t>(os,
        static_cast<std::uint32_t>(schedule.inner_count));
    write_scalar<std::uint32_t>(os,
        static_cast<std::uint32_t>(schedule.inner_interval));
    write_scalar<std::int32_t>(os,
        static_cast<std::int32_t>(schedule.inner_stride));
    write_scalar<std::uint32_t>(os,
        static_cast<std::uint32_t>(schedule.outer_count));
    write_scalar<std::uint32_t>(os,
        static_cast<std::uint32_t>(schedule.outer_interval));
    write_scalar<std::int32_t>(os,
        static_cast<std::int32_t>(schedule.outer_stride));
    write_scalar<std::uint8_t>(os,
        static_cast<std::uint8_t>(schedule.induction_target));
}

void write_macro_queue_commands(
    std::ostream& os, const QueueProgram& queue)
{
    std::vector<std::uint8_t> wideWords(
        (queue.commands.size() + 7) / 8, 0);
    for (std::size_t index = 0; index < queue.commands.size(); ++index) {
        const auto& command = queue.commands[index];
        if (!is_macro_schedule_command(command)
            || command.instruction_kind
                != instruction_kind_for_queue(queue.kind)
            || command.word_count == 0 || command.word_count > 2)
            throw std::runtime_error(
                "invalid FTLPU macro queue command");
        if (command.word_count == 2)
            wideWords[index / 8]
                |= static_cast<std::uint8_t>(1u << (index % 8));
    }
    for (const auto byte : wideWords) write_scalar(os, byte);
    for (const auto& command : queue.commands) {
        for (std::uint16_t word = 0; word < command.word_count; ++word)
            write_scalar(os, command.words[word]);
        write_macro_schedule_payload(
            os, decode_macro_schedule_command(command));
    }
}

void write_mem_macro_bitstream(std::ostream& os, const QueueProgram& queue)
{
    const auto image = encode_mem_macro_bitstream(queue);
    write_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(image.version));
    write_scalar<std::uint8_t>(os, image.delta_count);
    write_scalar<std::uint64_t>(os, image.bit_count);
    for (std::size_t i = 0; i < image.delta_count; ++i) {
        write_scalar<std::uint32_t>(os, image.deltas[i].start_cycle);
        write_scalar<std::int32_t>(os, image.deltas[i].address);
    }
    os.write(reinterpret_cast<const char*>(image.bytes.data()),
        static_cast<std::streamsize>(image.bytes.size()));
    if (!os) throw std::runtime_error("failed to write MEM Macro bitstream");
}

void write_mxm_macro_bitstream(std::ostream& os, const QueueProgram& queue)
{
    const auto image = encode_mxm_macro_bitstream(queue);
    write_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(image.version));
    write_scalar<std::uint8_t>(os, image.delta_count);
    write_scalar<std::uint64_t>(os, image.bit_count);
    for (std::size_t i = 0; i < image.delta_count; ++i) {
        write_scalar<std::uint32_t>(os, image.deltas[i].start_cycle);
        write_scalar<std::int32_t>(os, image.deltas[i].operand);
    }
    os.write(reinterpret_cast<const char*>(image.bytes.data()),
        static_cast<std::streamsize>(image.bytes.size()));
    if (!os) throw std::runtime_error("failed to write MXM Macro bitstream");
}

QueueCommand read_compact_queue_command(std::istream& is)
{
    const auto header = decode_compact_queue_record_header(
        read_scalar<std::uint8_t>(is));
    QueueCommand command;
    command.instruction_kind = header.instruction_kind;
    command.word_count = header.word_count;
    command.command = header.macro
        ? static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction)
        : read_scalar<std::uint32_t>(is);
    for (std::uint16_t word = 0; word < command.word_count; ++word)
        command.words[word] = read_scalar<std::uint32_t>(is);
    if (header.macro) {
        const IcuMacroSchedule schedule {
            read_scalar<std::uint32_t>(is),
            read_scalar<std::uint32_t>(is),
            read_scalar<std::uint32_t>(is),
            read_scalar<std::int32_t>(is),
            read_scalar<std::uint32_t>(is),
            read_scalar<std::uint32_t>(is),
            read_scalar<std::int32_t>(is),
            static_cast<IcuInductionTarget>(read_scalar<std::uint8_t>(is)),
        };
        return encode_macro_schedule_command(std::move(command), schedule);
    }
    if (header.has_extension) {
        const auto extension_count = read_scalar<std::uint16_t>(is);
        command.extension_words.reserve(extension_count);
        for (std::uint16_t word = 0; word < extension_count; ++word)
            command.extension_words.push_back(read_scalar<std::uint32_t>(is));
    }
    return command;
}

void write_binding(std::ostream& os, const BinaryBinding& binding)
{
    const auto pagePlacementCount = binding.page_banks.size();
    if ((pagePlacementCount != 0
            || !binding.page_slice_group_bases.empty()
            || !binding.page_slice_group_counts.empty()
            || !binding.page_base_rows.empty()
            || !binding.page_row_counts.empty())
        && (binding.page_slice_group_bases.size() != pagePlacementCount
            || binding.page_slice_group_counts.size() != pagePlacementCount
            || binding.page_base_rows.size() != pagePlacementCount
            || binding.page_row_counts.size() != pagePlacementCount))
        throw std::runtime_error(
            "FTLPU binding page placement arrays have different lengths");
    if (pagePlacementCount != 0
        && pagePlacementCount != binding.page_count)
        throw std::runtime_error(
            "FTLPU binding page placement count does not match page_count");
    write_scalar<std::uint32_t>(os, binding.index);
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.access));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.element_type));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.layout));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.shape.size()));
    write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(binding.slices.size()));
    write_scalar<std::uint16_t>(os, binding.hemisphere_mask);
    write_scalar<std::uint64_t>(os, binding.byte_size);
    write_scalar<std::int64_t>(os, binding.base_row);
    write_scalar<std::int64_t>(os, binding.instruction_count);
    write_scalar<std::int64_t>(os, binding.address_stride);
    for (std::uint64_t dimension : binding.shape) write_scalar(os, dimension);
    for (std::uint16_t slice : binding.slices) write_scalar(os, slice);
    if (binding.role.size() > std::numeric_limits<std::uint16_t>::max()
        || binding.name.size() > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("FTLPU binding metadata string is too long");
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(binding.role.size()));
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(binding.name.size()));
    write_scalar<std::uint64_t>(os, binding.ready_cycle);
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(binding.initializer));
    write_scalar<std::uint16_t>(os, binding.bank);
    write_scalar<float>(os, binding.rope_theta);
    write_scalar<std::uint32_t>(os, binding.rope_head_dim);
    write_scalar<std::uint32_t>(os, binding.rope_rotary_dim);
    write_scalar<float>(os, binding.rope_attention_factor);
    write_scalar<std::uint16_t>(os, binding.paged_weight ? 1 : 0);
    write_scalar<std::uint16_t>(os,
        static_cast<std::uint16_t>(binding.page_storage_slices.size()));
    write_scalar<std::uint32_t>(os, binding.page_count);
    write_scalar<std::uint32_t>(os, binding.page_rows);
    write_scalar<std::uint32_t>(os, binding.page_granularity);
    write_scalar<std::uint32_t>(os, binding.page_role_group_base);
    write_scalar<std::uint32_t>(os, binding.page_role_group_count);
    write_scalar<std::uint32_t>(os, binding.page_items_per_slice_group);
    write_scalar<std::uint32_t>(os, binding.page_bank_count);
    for (std::uint16_t slice : binding.page_storage_slices)
        write_scalar<std::uint16_t>(os, slice);
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(pagePlacementCount));
    for (std::size_t page = 0; page < pagePlacementCount; ++page) {
        write_scalar<std::uint16_t>(os, binding.page_banks[page]);
        write_scalar<std::uint16_t>(
            os, binding.page_slice_group_bases[page]);
        write_scalar<std::uint16_t>(
            os, binding.page_slice_group_counts[page]);
        write_scalar<std::uint16_t>(os, 0);
        write_scalar<std::uint32_t>(os, binding.page_base_rows[page]);
        write_scalar<std::uint32_t>(os, binding.page_row_counts[page]);
    }
    os.write(binding.role.data(),
        static_cast<std::streamsize>(binding.role.size()));
    os.write(binding.name.data(),
        static_cast<std::streamsize>(binding.name.size()));
    if (!os) throw std::runtime_error("failed to write binding metadata");
    if (binding.rope_inverse_frequencies.size() > 65536)
        throw std::runtime_error("too many RoPE inverse frequencies");
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(binding.rope_inverse_frequencies.size()));
    for (const float frequency : binding.rope_inverse_frequencies)
        write_scalar<float>(os, frequency);
}

BinaryBinding read_binding(std::istream& is, std::uint32_t version)
{
    BinaryBinding binding;
    binding.index = read_scalar<std::uint32_t>(is);
    binding.access = static_cast<BindingAccess>(read_scalar<std::uint16_t>(is));
    binding.element_type = static_cast<BindingElementType>(read_scalar<std::uint16_t>(is));
    binding.layout = static_cast<BindingLayout>(read_scalar<std::uint16_t>(is));
    const auto rank = read_scalar<std::uint16_t>(is);
    const auto slice_count = read_scalar<std::uint16_t>(is);
    const auto hemisphere_mask = read_scalar<std::uint16_t>(is);
    binding.hemisphere_mask = version >= 3 ? hemisphere_mask : 1;
    binding.byte_size = read_scalar<std::uint64_t>(is);
    binding.base_row = read_scalar<std::int64_t>(is);
    binding.instruction_count = read_scalar<std::int64_t>(is);
    binding.address_stride = read_scalar<std::int64_t>(is);
    binding.shape.reserve(rank);
    binding.slices.reserve(slice_count);
    for (std::uint16_t i = 0; i < rank; ++i)
        binding.shape.push_back(read_scalar<std::uint64_t>(is));
    for (std::uint16_t i = 0; i < slice_count; ++i)
        binding.slices.push_back(read_scalar<std::uint16_t>(is));
    if (version >= 6) {
        const auto role_size = read_scalar<std::uint16_t>(is);
        const auto name_size = read_scalar<std::uint16_t>(is);
        binding.ready_cycle = read_scalar<std::uint64_t>(is);
        if (version >= 7) {
            binding.initializer = static_cast<BindingInitializer>(
                read_scalar<std::uint16_t>(is));
            const auto bank_or_reserved = read_scalar<std::uint16_t>(is);
            binding.bank = version >= 17 ? bank_or_reserved : 0;
            binding.rope_theta = read_scalar<float>(is);
            binding.rope_head_dim = read_scalar<std::uint32_t>(is);
            if (version >= 33) {
                binding.rope_rotary_dim = read_scalar<std::uint32_t>(is);
                binding.rope_attention_factor = read_scalar<float>(is);
            }
            if (version >= 22) {
                binding.paged_weight =
                    read_scalar<std::uint16_t>(is) != 0;
                const auto page_slice_count =
                    read_scalar<std::uint16_t>(is);
                binding.page_count = read_scalar<std::uint32_t>(is);
                binding.page_rows = read_scalar<std::uint32_t>(is);
                binding.page_granularity = read_scalar<std::uint32_t>(is);
                binding.page_role_group_base =
                    read_scalar<std::uint32_t>(is);
                binding.page_role_group_count =
                    read_scalar<std::uint32_t>(is);
                binding.page_items_per_slice_group =
                    read_scalar<std::uint32_t>(is);
                binding.page_bank_count = read_scalar<std::uint32_t>(is);
                binding.page_storage_slices.reserve(page_slice_count);
                for (std::uint16_t index = 0;
                     index < page_slice_count; ++index)
                    binding.page_storage_slices.push_back(
                        read_scalar<std::uint16_t>(is));
                if (version >= 30) {
                    const auto placementCount =
                        read_scalar<std::uint32_t>(is);
                    binding.page_banks.reserve(placementCount);
                    binding.page_slice_group_bases.reserve(placementCount);
                    binding.page_slice_group_counts.reserve(placementCount);
                    binding.page_base_rows.reserve(placementCount);
                    binding.page_row_counts.reserve(placementCount);
                    for (std::uint32_t page = 0;
                         page < placementCount; ++page) {
                        binding.page_banks.push_back(
                            read_scalar<std::uint16_t>(is));
                        binding.page_slice_group_bases.push_back(
                            read_scalar<std::uint16_t>(is));
                        binding.page_slice_group_counts.push_back(
                            read_scalar<std::uint16_t>(is));
                        (void)read_scalar<std::uint16_t>(is);
                        binding.page_base_rows.push_back(
                            read_scalar<std::uint32_t>(is));
                        binding.page_row_counts.push_back(
                            read_scalar<std::uint32_t>(is));
                    }
                }
            }
        } else if (binding.layout == BindingLayout::Fp32CausalMaskTile
                   || binding.layout == BindingLayout::Fp16CausalMaskTile) {
            binding.initializer = BindingInitializer::CausalMask;
        }
        binding.role.resize(role_size);
        binding.name.resize(name_size);
        is.read(binding.role.data(),
            static_cast<std::streamsize>(binding.role.size()));
        is.read(binding.name.data(),
            static_cast<std::streamsize>(binding.name.size()));
        if (!is) throw std::runtime_error("truncated binding metadata");
        if (version >= 34) {
            const auto count = read_scalar<std::uint32_t>(is);
            if (count > 65536) throw std::runtime_error("too many RoPE inverse frequencies");
            for (std::uint32_t i = 0; i < count; ++i)
                binding.rope_inverse_frequencies.push_back(read_scalar<float>(is));
        }
    }
    return binding;
}

class SpanStreamBuffer final : public std::streambuf {
public:
    explicit SpanStreamBuffer(std::span<const std::uint8_t> data)
    {
        auto* begin = const_cast<char*>(
            reinterpret_cast<const char*>(data.data()));
        setg(begin, begin, begin + data.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
        std::ios_base::openmode mode) override
    {
        if ((mode & std::ios_base::in) == 0) return pos_type(off_type(-1));
        char* base = eback();
        char* target = nullptr;
        if (direction == std::ios_base::beg)
            target = base + offset;
        else if (direction == std::ios_base::cur)
            target = gptr() + offset;
        else if (direction == std::ios_base::end)
            target = egptr() + offset;
        if (target < base || target > egptr()) return pos_type(off_type(-1));
        setg(base, target, egptr());
        return pos_type(target - base);
    }

    pos_type seekpos(pos_type position,
        std::ios_base::openmode mode) override
    {
        return seekoff(static_cast<off_type>(position), std::ios_base::beg,
            mode);
    }
};

class ByteReader {
public:
    explicit ByteReader(std::span<const std::uint8_t> data)
        : data_(data)
    {
    }

    template <typename T>
    T read()
    {
        require(sizeof(T));
        T value;
        std::memcpy(&value, data_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    void read_bytes(void* destination, std::size_t size)
    {
        require(size);
        std::memcpy(destination, data_.data() + offset_, size);
        offset_ += size;
    }

    void skip(std::uint64_t size)
    {
        if (size > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("FTLPU binary section is too large");
        require(static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
    }

    bool empty() const { return offset_ == data_.size(); }

private:
    void require(std::size_t size) const
    {
        if (size > data_.size() - offset_)
            throw std::runtime_error("truncated FTLPU binary");
    }

    std::span<const std::uint8_t> data_;
    std::size_t offset_{0};
};

QueueCommand read_compact_queue_command(ByteReader& reader)
{
    const auto header = decode_compact_queue_record_header(
        reader.read<std::uint8_t>());
    QueueCommand command;
    command.instruction_kind = header.instruction_kind;
    command.word_count = header.word_count;
    command.command = header.macro
        ? static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction)
        : reader.read<std::uint32_t>();
    for (std::uint16_t word = 0; word < command.word_count; ++word)
        command.words[word] = reader.read<std::uint32_t>();
    if (header.macro) {
        const IcuMacroSchedule schedule {
            reader.read<std::uint32_t>(),
            reader.read<std::uint32_t>(),
            reader.read<std::uint32_t>(),
            reader.read<std::int32_t>(),
            reader.read<std::uint32_t>(),
            reader.read<std::uint32_t>(),
            reader.read<std::int32_t>(),
            static_cast<IcuInductionTarget>(reader.read<std::uint8_t>()),
        };
        return encode_macro_schedule_command(std::move(command), schedule);
    }
    if (header.has_extension) {
        const auto extension_count = reader.read<std::uint16_t>();
        command.extension_words.reserve(extension_count);
        for (std::uint16_t word = 0; word < extension_count; ++word)
            command.extension_words.push_back(reader.read<std::uint32_t>());
    }
    return command;
}

template <typename T>
T read_value(std::istream& is)
{
    return read_scalar<T>(is);
}

template <typename T>
T read_value(ByteReader& reader)
{
    return reader.read<T>();
}

template <typename T, typename Reader>
T read_le(Reader& reader)
{
    static_assert(std::is_unsigned_v<T>);
    T result = 0;
    for (unsigned i = 0; i < sizeof(T); ++i)
        result |= static_cast<T>(read_value<std::uint8_t>(reader)) << (8 * i);
    return result;
}

template <typename Reader>
void read_barrier_sections(Reader& reader, BinaryProgram& program,
                           std::uint32_t event_count)
{
    if (program.common_barrier_protocol
        && (program.hardware.common_barrier_encoding_version != 1
            || program.hardware.common_barrier_capacity == 0
            || program.hardware.common_barrier_capacity > hw::kCommonBarrierCapacity))
        throw std::runtime_error("unsupported FTLPU common barrier hardware capability");
    if (event_count > program.hardware.common_barrier_capacity)
        throw std::runtime_error("FTLPU barrier event capacity exceeded");
    program.barrier_events.reserve(event_count);
    std::uint32_t completion_count = 0;
    for (std::uint32_t i = 0; i < event_count; ++i) {
        BinaryBarrierEvent event;
        event.event_id = read_le<std::uint32_t>(reader);
        const auto count = read_le<std::uint32_t>(reader);
        if (count == 0 || count > program.weight_page_uses.size()
            || count > kBinaryCommonBarrierMaxCompletionInstances - completion_count)
            throw std::runtime_error("invalid FTLPU barrier dependency count or completion capacity");
        completion_count += count;
        event.page_use_indices.reserve(count);
        for (std::uint32_t j = 0; j < count; ++j)
            event.page_use_indices.push_back(read_le<std::uint32_t>(reader));
        program.barrier_events.push_back(std::move(event));
    }
    for (const auto expected : std::array<std::uint8_t, 4>{'B', 'A', 'R', 'R'})
        if (read_le<std::uint8_t>(reader) != expected)
            throw std::runtime_error("invalid FTLPU BarrierBlock magic");
    const auto encoding = read_le<std::uint16_t>(reader);
    const auto entry_bytes = read_le<std::uint16_t>(reader);
    const auto count = read_le<std::uint32_t>(reader);
    const auto reserved = read_le<std::uint32_t>(reader);
    if (encoding != 1 || entry_bytes != 8 || reserved != 0)
        throw std::runtime_error("invalid FTLPU BarrierBlock header");
    if (count > program.hardware.common_barrier_capacity)
        throw std::runtime_error("FTLPU barrier capacity exceeded");
    program.barriers.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto cycle = read_le<std::uint32_t>(reader);
        const auto event = read_le<std::uint32_t>(reader);
        program.barriers.push_back({cycle, event});
    }
    validate_binary_barriers(program);
}

void write_barrier_sections(std::ostream& os, const BinaryProgram& program)
{
    for (const auto& event : program.barrier_events) {
        write_le<std::uint32_t>(os, event.event_id);
        write_le<std::uint32_t>(os, static_cast<std::uint32_t>(event.page_use_indices.size()));
        for (const auto use : event.page_use_indices)
            write_le<std::uint32_t>(os, use);
    }
    os.write("BARR", 4);
    write_le<std::uint16_t>(os, 1);
    write_le<std::uint16_t>(os, 8);
    write_le<std::uint32_t>(os, static_cast<std::uint32_t>(program.barriers.size()));
    write_le<std::uint32_t>(os, 0);
    for (const auto& barrier : program.barriers) {
        write_le<std::uint32_t>(os, barrier.stop_cycle);
        write_le<std::uint32_t>(os, barrier.event_id);
    }
}

template <typename Reader>
void read_queue_controls(Reader& reader, BinaryProgram& program)
{
    for (const auto byte : std::array<std::uint8_t,4>{'Q','C','T','L'})
        if (read_le<std::uint8_t>(reader) != byte)
            throw std::runtime_error("invalid QCTL magic");
    const auto version = read_le<std::uint16_t>(reader);
    const auto bytes = read_le<std::uint16_t>(reader);
    const auto directories = read_le<std::uint32_t>(reader);
    const auto events = read_le<std::uint32_t>(reader);
    if (version != 1 || bytes != 24 || directories > 4096 || events > 65535)
        throw std::runtime_error("invalid QCTL encoding or count");
    for (std::uint32_t q=0; q<directories; ++q) {
        BinaryQueueControlPartition partition;
        partition.kind = static_cast<QueueKind>(read_le<std::uint16_t>(reader));
        partition.index = read_le<std::uint16_t>(reader);
        const auto count = read_le<std::uint32_t>(reader);
        partition.end_cycle = read_le<std::uint32_t>(reader);
        if (count > program.hardware.queue_control_capacity || count > 256)
            throw std::runtime_error("QCTL partition exceeds physical capacity");
        for (std::uint32_t i=0; i<count; ++i) {
            BinaryQueueControl control;
            control.local_cycle = read_le<std::uint32_t>(reader);
            control.opcode = static_cast<IcuControlOpcode>(read_le<std::uint16_t>(reader));
            const auto flags = read_le<std::uint16_t>(reader);
            if (flags > 3) throw std::runtime_error("invalid QCTL binding flags");
            control.native_binding = (flags & 1) != 0;
            control.activation_producer = (flags & 2) != 0;
            control.operand = read_le<std::uint32_t>(reader);
            control.source_command = read_le<std::uint32_t>(reader);
            control.release_group = read_le<std::uint32_t>(reader);
            control.phase_offset = read_le<std::uint32_t>(reader);
            partition.entries.push_back(control);
        }
        program.queue_sync_plan.partitions.push_back(std::move(partition));
    }
    std::uint32_t instances=0;
    for (std::uint32_t i=0; i<events; ++i) {
        BinaryQueueControlEvent event;
        event.event_id = read_le<std::uint32_t>(reader);
        event.source = static_cast<QueueEventSource>(read_le<std::uint16_t>(reader));
        if (read_le<std::uint16_t>(reader)) throw std::runtime_error("nonzero QCTL reserved field");
        const auto count = read_le<std::uint32_t>(reader);
        if (count > 65535-instances) throw std::runtime_error("QCTL completion capacity exceeded");
        instances += count;
        for (std::uint32_t n=0; n<count; ++n) event.page_use_indices.push_back(read_le<std::uint32_t>(reader));
        program.queue_sync_plan.events.push_back(std::move(event));
    }
}

void write_queue_controls(std::ostream& os, const BinaryProgram& program)
{
    os.write("QCTL",4);
    write_le<std::uint16_t>(os,1); write_le<std::uint16_t>(os,24);
    write_le<std::uint32_t>(os,static_cast<std::uint32_t>(program.queue_sync_plan.partitions.size()));
    write_le<std::uint32_t>(os,static_cast<std::uint32_t>(program.queue_sync_plan.events.size()));
    for (const auto& partition : program.queue_sync_plan.partitions) {
        write_le<std::uint16_t>(os,static_cast<std::uint16_t>(partition.kind));
        write_le<std::uint16_t>(os,partition.index);
        write_le<std::uint32_t>(os,static_cast<std::uint32_t>(partition.entries.size()));
        write_le<std::uint32_t>(os,partition.end_cycle);
        for (const auto& c:partition.entries) {
            write_le<std::uint32_t>(os,c.local_cycle);
            write_le<std::uint16_t>(os,static_cast<std::uint16_t>(c.opcode));
            write_le<std::uint16_t>(os,(c.native_binding ? 1 : 0) | (c.activation_producer ? 2 : 0));
            write_le<std::uint32_t>(os,c.operand);
            write_le<std::uint32_t>(os,c.source_command);
            write_le<std::uint32_t>(os,c.release_group);
            write_le<std::uint32_t>(os,c.phase_offset);
        }
    }
    for (const auto& event:program.queue_sync_plan.events) {
        write_le<std::uint32_t>(os,event.event_id);
        write_le<std::uint16_t>(os,static_cast<std::uint16_t>(event.source));
        write_le<std::uint16_t>(os,0);
        write_le<std::uint32_t>(os,static_cast<std::uint32_t>(event.page_use_indices.size()));
        for (const auto instance:event.page_use_indices) write_le<std::uint32_t>(os,instance);
    }
}

void skip_value(std::istream& is, std::uint64_t bytes)
{
    if (bytes > static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("FTLPU binary section is too large");
    is.seekg(static_cast<std::streamoff>(bytes), std::ios_base::cur);
    if (!is) throw std::runtime_error("truncated FTLPU binary");
}

void skip_value(ByteReader& reader, std::uint64_t bytes)
{
    reader.skip(bytes);
}

template <typename Reader>
IcuMacroSchedule read_macro_schedule_payload(Reader& reader)
{
    return IcuMacroSchedule {
        read_value<std::uint32_t>(reader),
        read_value<std::uint32_t>(reader),
        read_value<std::uint32_t>(reader),
        read_value<std::int32_t>(reader),
        read_value<std::uint32_t>(reader),
        read_value<std::uint32_t>(reader),
        read_value<std::int32_t>(reader),
        static_cast<IcuInductionTarget>(read_value<std::uint8_t>(reader)),
    };
}

template <typename Reader>
QueueCommand read_native_queue_command(Reader& reader, QueueKind queueKind)
{
    const auto prefix = read_value<std::uint32_t>(reader);
    const auto opcode = isa::decode_icu_command_opcode(prefix);
    if (opcode == isa::IcuCommandOpcode::Instruction) {
        if ((prefix & ~(kNativeWordCountMask | 0x3u)) != 0)
            throw std::runtime_error(
                "invalid native FTLPU instruction prefix");
        const auto wordCount = static_cast<std::uint16_t>(
            (prefix & kNativeWordCountMask) >> kNativeWordCountShift);
        if (is_sxm_queue(queueKind)) {
            if (wordCount != kNativeSxmFixedWordCount)
                throw std::runtime_error(
                    "invalid fixed SXM FTLPU instruction");
            isa::EncodedSxmInstruction encoded {};
            for (auto& word : encoded.words)
                word = read_value<std::uint32_t>(reader);
            return sxm_queue_command(isa::decode_sxm_instruction(encoded));
        }
        if (wordCount == 0 || wordCount > 4)
            throw std::runtime_error(
                "invalid native FTLPU instruction width");
        QueueCommand command {
            static_cast<isa::EncodedIcuCommand>(
                isa::IcuCommandOpcode::Instruction),
            instruction_kind_for_queue(queueKind), wordCount, {},
        };
        for (std::uint16_t word = 0; word < wordCount; ++word)
            command.words[word] = read_value<std::uint32_t>(reader);
        return command;
    }
    if (opcode == isa::IcuCommandOpcode::Extended
        && ((prefix >> 2) & kIcuExtendedReservedMask) == 0) {
        std::array<std::uint32_t, 3> words {
            prefix,
            read_value<std::uint32_t>(reader),
            read_value<std::uint32_t>(reader),
        };
        return repeat_2d_queue_command(decode_binary_repeat_2d(words));
    }
    return QueueCommand {prefix};
}

template <typename Reader>
QueueCommand read_macro_queue_command(
    Reader& reader, QueueKind queueKind, bool wide)
{
    const auto wordCount = static_cast<std::uint16_t>(wide ? 2 : 1);
    QueueCommand instruction {
        static_cast<isa::EncodedIcuCommand>(
            isa::IcuCommandOpcode::Instruction),
        instruction_kind_for_queue(queueKind), wordCount, {},
    };
    for (std::uint16_t word = 0; word < wordCount; ++word)
        instruction.words[word] = read_value<std::uint32_t>(reader);
    return encode_macro_schedule_command(
        std::move(instruction), read_macro_schedule_payload(reader));
}

template <typename Reader>
std::vector<std::uint8_t> read_macro_width_bitmap(
    Reader& reader, std::uint32_t commandCount)
{
    std::vector<std::uint8_t> bitmap((commandCount + 7) / 8);
    for (auto& byte : bitmap) byte = read_value<std::uint8_t>(reader);
    return bitmap;
}

template <typename Reader>
void skip_native_queue_command(Reader& reader, QueueKind queueKind)
{
    const auto prefix = read_value<std::uint32_t>(reader);
    const auto opcode = isa::decode_icu_command_opcode(prefix);
    if (opcode == isa::IcuCommandOpcode::Instruction) {
        const auto wordCount = static_cast<std::uint16_t>(
            (prefix & kNativeWordCountMask) >> kNativeWordCountShift);
        if (is_sxm_queue(queueKind)) {
            if (wordCount != kNativeSxmFixedWordCount)
                throw std::runtime_error(
                    "invalid fixed SXM FTLPU instruction");
            skip_value(reader,
                isa::EncodedSxmInstruction {}.words.size()
                    * sizeof(std::uint32_t));
            return;
        }
        if (wordCount == 0 || wordCount > 4)
            throw std::runtime_error(
                "invalid native FTLPU instruction width");
        skip_value(reader,
            static_cast<std::uint64_t>(wordCount)
                * sizeof(std::uint32_t));
        return;
    }
    if (opcode == isa::IcuCommandOpcode::Extended
        && ((prefix >> 2) & kIcuExtendedReservedMask) == 0) {
        if (((prefix >> kIcuExtendedSubtypeShift) & 0x3u)
            != kIcuExtendedRepeat2D)
            throw std::runtime_error(
                "unsupported extended FTLPU control record");
        skip_value(reader, 2 * sizeof(std::uint32_t));
    }
}

template <typename Reader>
void skip_macro_queue_command(Reader& reader, bool wide)
{
    skip_value(reader,
        static_cast<std::uint64_t>(wide ? 2 : 1)
                * sizeof(std::uint32_t)
            + 29);
}

QueueEncodingMode decode_queue_encoding_mode(std::uint8_t value)
{
    if (value > static_cast<std::uint8_t>(
                    QueueEncodingMode::MacroPacked))
        throw std::runtime_error("invalid FTLPU queue encoding mode");
    return static_cast<QueueEncodingMode>(value);
}

template <typename Reader>
QueueProgram read_mem_macro_bitstream(Reader& reader,
    std::uint32_t commandCount, std::size_t queueIndex)
{
    MemMacroBitstream image;
    image.version = read_value<std::uint8_t>(reader);
    image.command_count = commandCount;
    image.delta_count = read_value<std::uint8_t>(reader);
    image.bit_count = read_value<std::uint64_t>(reader);
    if (image.delta_count > image.deltas.size())
        throw std::runtime_error("invalid MEM Macro delta dictionary size");
    for (std::size_t i = 0; i < image.delta_count; ++i) {
        image.deltas[i].start_cycle = read_value<std::uint32_t>(reader);
        image.deltas[i].address = read_value<std::int32_t>(reader);
    }
    const auto byteCount = (image.bit_count + 7) / 8;
    if (byteCount > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("MEM Macro bitstream is too large");
    image.bytes.resize(static_cast<std::size_t>(byteCount));
    if constexpr (std::is_same_v<Reader, std::istream>) {
        reader.read(reinterpret_cast<char*>(image.bytes.data()),
            static_cast<std::streamsize>(image.bytes.size()));
        if (!reader) throw std::runtime_error("truncated MEM Macro bitstream");
    } else {
        reader.read_bytes(image.bytes.data(), image.bytes.size());
    }
    auto queue = decode_mem_macro_bitstream(image, queueIndex);
    queue.packed_macro_imem = pack_mem_macro_imem(image);
    queue.requires_raw_macro = true;
    return queue;
}

template <typename Reader>
QueueProgram read_mxm_macro_bitstream(Reader& reader,
    std::uint32_t commandCount, std::size_t queueIndex, QueueKind queueKind)
{
    if (!is_mxm_macro_queue(queueKind))
        throw std::runtime_error("invalid MXM Macro packed queue kind");
    MxmMacroBitstream image;
    image.kind = queueKind;
    image.version = read_value<std::uint8_t>(reader);
    image.command_count = commandCount;
    image.delta_count = read_value<std::uint8_t>(reader);
    image.bit_count = read_value<std::uint64_t>(reader);
    if (image.delta_count > image.deltas.size())
        throw std::runtime_error("invalid MXM Macro delta dictionary size");
    for (std::size_t i = 0; i < image.delta_count; ++i) {
        image.deltas[i].start_cycle = read_value<std::uint32_t>(reader);
        image.deltas[i].operand = read_value<std::int32_t>(reader);
    }
    const auto byteCount = (image.bit_count + 7) / 8;
    if (byteCount > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("MXM Macro bitstream is too large");
    image.bytes.resize(static_cast<std::size_t>(byteCount));
    if constexpr (std::is_same_v<Reader, std::istream>) {
        reader.read(reinterpret_cast<char*>(image.bytes.data()),
            static_cast<std::streamsize>(image.bytes.size()));
        if (!reader) throw std::runtime_error("truncated MXM Macro bitstream");
    } else {
        reader.read_bytes(image.bytes.data(), image.bytes.size());
    }
    auto queue = decode_mxm_macro_bitstream(image, queueIndex);
    queue.packed_macro_imem = pack_mxm_macro_imem(image);
    queue.requires_raw_macro = true;
    return queue;
}

template <typename Reader>
void skip_macro_bitstream(Reader& reader, QueueKind queueKind)
{
    (void)read_value<std::uint8_t>(reader);
    const auto deltaCount = read_value<std::uint8_t>(reader);
    const auto dictionaryCapacity = queueKind == QueueKind::Mem
        ? kMemMacroDeltaDictionarySize
        : kMxmMacroDeltaDictionarySize;
    if (deltaCount > dictionaryCapacity)
        throw std::runtime_error("invalid Macro delta dictionary size");
    const auto bitCount = read_value<std::uint64_t>(reader);
    skip_value(reader, static_cast<std::uint64_t>(deltaCount) * 8
        + (bitCount + 7) / 8);
}

BinaryBinding read_binding(ByteReader& reader, std::uint32_t version)
{
    BinaryBinding binding;
    binding.index = reader.read<std::uint32_t>();
    binding.access =
        static_cast<BindingAccess>(reader.read<std::uint16_t>());
    binding.element_type =
        static_cast<BindingElementType>(reader.read<std::uint16_t>());
    binding.layout =
        static_cast<BindingLayout>(reader.read<std::uint16_t>());
    const auto rank = reader.read<std::uint16_t>();
    const auto slice_count = reader.read<std::uint16_t>();
    const auto hemisphere_mask = reader.read<std::uint16_t>();
    binding.hemisphere_mask = version >= 3 ? hemisphere_mask : 1;
    binding.byte_size = reader.read<std::uint64_t>();
    binding.base_row = reader.read<std::int64_t>();
    binding.instruction_count = reader.read<std::int64_t>();
    binding.address_stride = reader.read<std::int64_t>();
    binding.shape.reserve(rank);
    binding.slices.reserve(slice_count);
    for (std::uint16_t i = 0; i < rank; ++i)
        binding.shape.push_back(reader.read<std::uint64_t>());
    for (std::uint16_t i = 0; i < slice_count; ++i)
        binding.slices.push_back(reader.read<std::uint16_t>());
    if (version >= 6) {
        const auto role_size = reader.read<std::uint16_t>();
        const auto name_size = reader.read<std::uint16_t>();
        binding.ready_cycle = reader.read<std::uint64_t>();
        if (version >= 7) {
            binding.initializer = static_cast<BindingInitializer>(
                reader.read<std::uint16_t>());
            const auto bank_or_reserved = reader.read<std::uint16_t>();
            binding.bank = version >= 17 ? bank_or_reserved : 0;
            binding.rope_theta = reader.read<float>();
            binding.rope_head_dim = reader.read<std::uint32_t>();
            if (version >= 33) {
                binding.rope_rotary_dim = reader.read<std::uint32_t>();
                binding.rope_attention_factor = reader.read<float>();
            }
            if (version >= 22) {
                binding.paged_weight =
                    reader.read<std::uint16_t>() != 0;
                const auto page_slice_count =
                    reader.read<std::uint16_t>();
                binding.page_count = reader.read<std::uint32_t>();
                binding.page_rows = reader.read<std::uint32_t>();
                binding.page_granularity = reader.read<std::uint32_t>();
                binding.page_role_group_base =
                    reader.read<std::uint32_t>();
                binding.page_role_group_count =
                    reader.read<std::uint32_t>();
                binding.page_items_per_slice_group =
                    reader.read<std::uint32_t>();
                binding.page_bank_count = reader.read<std::uint32_t>();
                binding.page_storage_slices.reserve(page_slice_count);
                for (std::uint16_t index = 0;
                     index < page_slice_count; ++index)
                    binding.page_storage_slices.push_back(
                        reader.read<std::uint16_t>());
                if (version >= 30) {
                    const auto placementCount =
                        reader.read<std::uint32_t>();
                    binding.page_banks.reserve(placementCount);
                    binding.page_slice_group_bases.reserve(placementCount);
                    binding.page_slice_group_counts.reserve(placementCount);
                    binding.page_base_rows.reserve(placementCount);
                    binding.page_row_counts.reserve(placementCount);
                    for (std::uint32_t page = 0;
                         page < placementCount; ++page) {
                        binding.page_banks.push_back(
                            reader.read<std::uint16_t>());
                        binding.page_slice_group_bases.push_back(
                            reader.read<std::uint16_t>());
                        binding.page_slice_group_counts.push_back(
                            reader.read<std::uint16_t>());
                        (void)reader.read<std::uint16_t>();
                        binding.page_base_rows.push_back(
                            reader.read<std::uint32_t>());
                        binding.page_row_counts.push_back(
                            reader.read<std::uint32_t>());
                    }
                }
            }
        } else if (binding.layout == BindingLayout::Fp32CausalMaskTile
                   || binding.layout == BindingLayout::Fp16CausalMaskTile) {
            binding.initializer = BindingInitializer::CausalMask;
        }
        binding.role.resize(role_size);
        binding.name.resize(name_size);
        reader.read_bytes(binding.role.data(), binding.role.size());
        reader.read_bytes(binding.name.data(), binding.name.size());
        if (version >= 34) {
            const auto count = reader.read<std::uint32_t>();
            if (count > 65536) throw std::runtime_error("too many RoPE inverse frequencies");
            for (std::uint32_t i = 0; i < count; ++i)
                binding.rope_inverse_frequencies.push_back(reader.read<float>());
        }
    }
    return binding;
}

void write_timeline(std::ostream& os, const BinaryTimeline& timeline)
{
    if (timeline.name.empty()
        || timeline.name.size() > std::numeric_limits<std::uint16_t>::max()
        || timeline.end_cycle < timeline.start_cycle)
        throw std::runtime_error("invalid FTLPU binary timeline");
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(timeline.name.size()));
    write_scalar<std::uint16_t>(os, 0);
    write_scalar<std::uint64_t>(os, timeline.start_cycle);
    write_scalar<std::uint64_t>(os, timeline.end_cycle);
    os.write(timeline.name.data(),
        static_cast<std::streamsize>(timeline.name.size()));
    if (!os) throw std::runtime_error("failed to write timeline metadata");
}

BinaryTimeline read_timeline(std::istream& is)
{
    BinaryTimeline timeline;
    const auto name_size = read_scalar<std::uint16_t>(is);
    (void)read_scalar<std::uint16_t>(is);
    timeline.start_cycle = read_scalar<std::uint64_t>(is);
    timeline.end_cycle = read_scalar<std::uint64_t>(is);
    timeline.name.resize(name_size);
    is.read(timeline.name.data(),
        static_cast<std::streamsize>(timeline.name.size()));
    if (!is || timeline.name.empty()
        || timeline.end_cycle < timeline.start_cycle)
        throw std::runtime_error("invalid FTLPU binary timeline");
    return timeline;
}

BinaryTimeline read_timeline(ByteReader& reader)
{
    BinaryTimeline timeline;
    const auto name_size = reader.read<std::uint16_t>();
    (void)reader.read<std::uint16_t>();
    timeline.start_cycle = reader.read<std::uint64_t>();
    timeline.end_cycle = reader.read<std::uint64_t>();
    timeline.name.resize(name_size);
    reader.read_bytes(timeline.name.data(), timeline.name.size());
    if (timeline.name.empty()
        || timeline.end_cycle < timeline.start_cycle)
        throw std::runtime_error("invalid FTLPU binary timeline");
    return timeline;
}

template <typename Reader>
void read_queue_link_source(Reader& reader,BinaryProgram& program)
{
    for(const auto byte:std::array<std::uint8_t,4>{'Q','L','N','K'})
        if(read_le<std::uint8_t>(reader)!=byte) throw std::runtime_error("invalid QLNK magic");
    const auto version=read_le<std::uint16_t>(reader);
    const auto flags=read_le<std::uint16_t>(reader);
    const auto queues=read_le<std::uint32_t>(reader);
    const auto addresses=read_le<std::uint32_t>(reader);
    const auto scales=read_le<std::uint32_t>(reader);
    const auto reserved=read_le<std::uint32_t>(reader);
    if(version!=1 || flags>1 || reserved || queues>4096 || addresses>1000000 || scales>1000000
        || (flags && !queues)) throw std::runtime_error("invalid QLNK header or capacity");
    auto& source=program.queue_link_source;source.required=flags!=0;
    std::size_t total=0;
    for(std::uint32_t i=0;i<queues;++i) {
        QueueProgram queue;
        queue.kind=static_cast<QueueKind>(read_le<std::uint16_t>(reader));
        queue.index=read_le<std::uint16_t>(reader);
        const auto count=read_le<std::uint32_t>(reader);
        if(count>1000000-total) throw std::runtime_error("QLNK exceeds compact source budget");
        total+=count;
        for(std::uint32_t pc=0;pc<count;++pc) queue.commands.push_back(read_compact_queue_command(reader));
        source.queues.push_back(std::move(queue));
    }
    for(std::uint32_t i=0;i<addresses;++i) {
        BinaryAddressRelocation a;
        a.binding_index=read_le<std::uint32_t>(reader);
        a.binding_access=static_cast<BindingAccess>(read_le<std::uint16_t>(reader));
        a.queue_kind=static_cast<QueueKind>(read_le<std::uint16_t>(reader));
        a.queue_index=read_le<std::uint16_t>(reader);
        const auto bits=read_le<std::uint16_t>(reader);
        if(bits>1) throw std::runtime_error("invalid QLNK relocation flags");
        a.write_port=bits!=0;a.command_index=read_le<std::uint32_t>(reader);
        source.address_relocations.push_back(a);
    }
    for(std::uint32_t i=0;i<scales;++i) {
        BinaryScaleRelocation a;
        a.binding_index=read_le<std::uint32_t>(reader);
        a.scale_index=read_le<std::uint32_t>(reader);
        a.queue_kind=static_cast<QueueKind>(read_le<std::uint16_t>(reader));
        a.queue_index=read_le<std::uint16_t>(reader);
        a.command_index=read_le<std::uint32_t>(reader);
        a.operand=static_cast<VxmImmediateOperand>(read_le<std::uint16_t>(reader));
        if(static_cast<unsigned>(a.operand)>1) throw std::runtime_error("invalid QLNK scale operand");
        source.scale_relocations.push_back(a);
    }
}
void write_queue_link_source(std::ostream& os,const BinaryProgram& program)
{
    const auto& source=program.queue_link_source;
    os.write("QLNK",4);write_le<std::uint16_t>(os,1);write_le<std::uint16_t>(os,source.required?1:0);
    write_le<std::uint32_t>(os,static_cast<std::uint32_t>(source.queues.size()));
    write_le<std::uint32_t>(os,static_cast<std::uint32_t>(source.address_relocations.size()));
    write_le<std::uint32_t>(os,static_cast<std::uint32_t>(source.scale_relocations.size()));
    write_le<std::uint32_t>(os,0);
    for(const auto& q:source.queues) {
        write_le<std::uint16_t>(os,static_cast<std::uint16_t>(q.kind));
        write_le<std::uint16_t>(os,static_cast<std::uint16_t>(q.index));
        write_le<std::uint32_t>(os,static_cast<std::uint32_t>(q.commands.size()));
        for(const auto& c:q.commands) write_compact_queue_command(os,c);
    }
    for(const auto& a:source.address_relocations) {
        write_le<std::uint32_t>(os,a.binding_index);write_le<std::uint16_t>(os,static_cast<std::uint16_t>(a.binding_access));
        write_le<std::uint16_t>(os,static_cast<std::uint16_t>(a.queue_kind));write_le<std::uint16_t>(os,a.queue_index);
        write_le<std::uint16_t>(os,a.write_port?1:0);write_le<std::uint32_t>(os,a.command_index);
    }
    for(const auto& a:source.scale_relocations) {
        write_le<std::uint32_t>(os,a.binding_index);write_le<std::uint32_t>(os,a.scale_index);
        write_le<std::uint16_t>(os,static_cast<std::uint16_t>(a.queue_kind));write_le<std::uint16_t>(os,a.queue_index);
        write_le<std::uint32_t>(os,a.command_index);write_le<std::uint16_t>(os,static_cast<std::uint16_t>(a.operand));
    }
}

struct BinaryHeader {
    BinaryProgram program;
    std::uint32_t version;
    std::uint32_t queue_count;
    std::uint32_t relocation_count;
    std::uint32_t address_relocation_count;
    std::uint32_t weight_page_use_count;
    std::uint32_t stream_release_count;
    std::uint32_t transport_queue_count;
};

template <typename ReadValue>
void read_hardware_config(BinaryProgram& program, std::uint32_t version,
    ReadValue&& read_value)
{
    auto read = [&](std::uint32_t& value) { value = read_value(); };
    bool upgradeTargetAbi = false;
    if (version >= 39) {
        program.hardware.visit(read);
    } else if (version == 38) {
        program.hardware.visit_pre_v39(read);
    } else if (version == 37) {
        program.hardware.visit_pre_v38(read);
    } else if (version >= 29) {
        program.hardware.visit_pre_v37(read);
    } else if (version == 28) {
        // Local pre-merge format: i-MEM and Macro geometry, no VXM capability
        // fields. Preserve existing Qwen v28 artifacts.
        program.hardware.visit_legacy_v28(read);
        upgradeTargetAbi = true;
    } else if (version == 27) {
        // Local pre-merge format: i-MEM geometry only.
        program.hardware.visit_legacy_v27(read);
        upgradeTargetAbi = true;
    } else if (version >= 26) {
        // Official origin/main v26 format: VXM capability fields, old queues.
        program.hardware.visit_v26(read);
    } else if (version >= 24) {
        program.hardware.visit_pre_v26(read);
        upgradeTargetAbi = true;
    } else if (version >= 23) {
        program.hardware.visit_pre_v24(read);
        upgradeTargetAbi = true;
    } else if (version >= 20) {
        program.hardware.visit_pre_v23(read);
        upgradeTargetAbi = true;
    } else if (version >= 19) {
        program.hardware.visit_pre_v20(read);
        upgradeTargetAbi = true;
    } else {
        program.hardware.visit_pre_v19(read);
        upgradeTargetAbi = true;
    }
    if (version < 39) {
        program.hardware.queue_control_version = 0;
        program.hardware.queue_control_capacity = 0;
        program.hardware.queue_control_cache_depth = 0;
        program.hardware.queue_control_read_latency = 0;
        program.hardware.queue_control_read_ports = 0;
        if (version < 38) program.hardware.common_transport_overlap_version = 0;
        TargetAbiHasher legacy;
        legacy.add(21);
        program.hardware.visit_command_abi([&](std::uint32_t value) { legacy.add(value); });
        if (version >= 37) {
            legacy.add(program.hardware.common_barrier_encoding_version);
            legacy.add(program.hardware.common_barrier_capacity);
        }
        if (version >= 38) legacy.add(program.hardware.common_transport_overlap_version);
        if (program.target_abi == legacy.value())
            upgradeTargetAbi = true;
    }
    if (upgradeTargetAbi)
        program.target_abi = executable_target_abi(program.hardware);
}

BinaryHeader read_header(ByteReader& reader)
{
    std::array<char, 8> magic {};
    reader.read_bytes(magic.data(), magic.size());
    if (magic != kMagic)
        throw std::runtime_error("invalid FTLPU binary magic");
    const auto version = reader.read<std::uint32_t>();
    if (version < 1 || version > kBinaryFormatVersion)
        throw std::runtime_error("unsupported FTLPU binary version");

    BinaryProgram program;
    if (version >= 5) {
        program.target_abi = reader.read<std::uint64_t>();
        const auto target_name_size = reader.read<std::uint16_t>();
        program.target_name.resize(target_name_size);
        reader.read_bytes(
            program.target_name.data(), program.target_name.size());
        if (version >= 16) {
            read_hardware_config(program, version,
                [&]() { return reader.read<std::uint32_t>(); });
        } else {
            if (version >= 9)
                program.hardware.sram_depth_rows =
                    reader.read<std::uint32_t>();
            if (version >= 15)
                program.hardware.mxms_per_hemisphere =
                    reader.read<std::uint32_t>();
        }
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle =
        static_cast<std::size_t>(reader.read<std::uint64_t>());
    const auto queue_count = reader.read<std::uint32_t>();
    const auto binding_count =
        version >= 2 ? reader.read<std::uint32_t>() : 0;
    const auto relocation_count =
        version >= 8 ? reader.read<std::uint32_t>() : 0;
    const auto address_relocation_count =
        version >= 9 ? reader.read<std::uint32_t>() : 0;
    const auto timeline_count =
        version >= 11 ? reader.read<std::uint32_t>() : 0;
    const auto memory_floor_count =
        version >= 12 ? reader.read<std::uint32_t>() : 0;
    const auto weight_page_use_count =
        version >= 22 ? reader.read<std::uint32_t>() : 0;
    const auto stream_release_count =
        version >= 31 ? reader.read<std::uint32_t>() : 0;
    const auto barrier_event_count = version >= 37 ? reader.read<std::uint32_t>() : 0;
    const auto metadata_flags = version >= 37 ? reader.read<std::uint32_t>() : 0;
    const auto transport_queue_count = version >= 37 ? reader.read<std::uint32_t>() : 0;
    if ((metadata_flags & ~(version >= 39 ? 7u : version >= 38 ? 3u : 1u)) != 0 || transport_queue_count > queue_count
        || (transport_queue_count != 0 && (metadata_flags & 5u) == 0))
        throw std::runtime_error("invalid FTLPU common barrier metadata flags or queue count");
    program.common_barrier_protocol = (metadata_flags & 1u) != 0;
    program.transport_overlap_protocol = (metadata_flags & 2u) != 0;
    program.queue_control_protocol = (metadata_flags & 4u) != 0;
    program.bindings.reserve(binding_count);
    for (std::uint32_t index = 0; index < binding_count; ++index)
        program.bindings.push_back(read_binding(reader, version));
    program.timelines.reserve(timeline_count);
    for (std::uint32_t index = 0; index < timeline_count; ++index)
        program.timelines.push_back(read_timeline(reader));
    program.memory_floors.reserve(memory_floor_count);
    for (std::uint32_t index = 0;
         index < memory_floor_count; ++index) {
        const auto hemisphere = reader.read<std::uint16_t>();
        const auto slice = reader.read<std::uint16_t>();
        const auto bank = version >= 17
            ? reader.read<std::uint16_t>() : std::uint16_t {0};
        const auto firstFreeRow = reader.read<std::uint32_t>();
        program.memory_floors.push_back(BinaryMemoryFloor {
            hemisphere, slice, firstFreeRow, bank,
        });
    }
    program.weight_page_uses.reserve(weight_page_use_count);
    for (std::uint32_t index = 0;
         index < weight_page_use_count; ++index) {
        BinaryWeightPageUse use;
        use.binding_index = reader.read<std::uint32_t>();
        use.page_index = reader.read<std::uint32_t>();
        use.bank = reader.read<std::uint16_t>();
        use.runtime_prefetch = reader.read<std::uint16_t>() != 0;
        use.ready_cycle = reader.read<std::uint64_t>();
        use.release_cycle = reader.read<std::uint64_t>();
        if (version >= 37)
            use.transport_launch_event_id = read_le<std::uint32_t>(reader);
        if (version >= 38)
            use.transport_launch_cycle = read_le<std::uint32_t>(reader);
        program.weight_page_uses.push_back(use);
    }
    program.stream_release_cycles.reserve(stream_release_count);
    for (std::uint32_t index = 0; index < stream_release_count; ++index)
        program.stream_release_cycles.push_back(reader.read<std::uint64_t>());
    if (version >= 37)
        read_barrier_sections(reader, program, barrier_event_count);
    if (version >= 39) { read_queue_controls(reader, program);read_queue_link_source(reader,program); validate_queue_sync_plan(program, false); }
    return {std::move(program), version, queue_count, relocation_count,
        address_relocation_count, weight_page_use_count,
        stream_release_count, transport_queue_count};
}

void skip_bytes(std::istream& is, std::uint64_t bytes)
{
    if (bytes > static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("FTLPU binary section is too large");
    is.seekg(static_cast<std::streamoff>(bytes), std::ios_base::cur);
    if (!is) throw std::runtime_error("truncated FTLPU binary");
}

void skip_compact_queue_command(std::istream& is)
{
    const auto header = decode_compact_queue_record_header(
        read_scalar<std::uint8_t>(is));
    skip_bytes(is, static_cast<std::uint64_t>(header.word_count)
            * sizeof(std::uint32_t)
        + (header.macro ? 29 : sizeof(std::uint32_t)));
    if (header.has_extension) {
        const auto extension_count = read_scalar<std::uint16_t>(is);
        skip_bytes(is, static_cast<std::uint64_t>(extension_count)
            * sizeof(std::uint32_t));
    }
}

void skip_compact_queue_command(ByteReader& reader)
{
    const auto header = decode_compact_queue_record_header(
        reader.read<std::uint8_t>());
    reader.skip(static_cast<std::uint64_t>(header.word_count)
            * sizeof(std::uint32_t)
        + (header.macro ? 29 : sizeof(std::uint32_t)));
    if (header.has_extension) {
        const auto extension_count = reader.read<std::uint16_t>();
        reader.skip(static_cast<std::uint64_t>(extension_count)
            * sizeof(std::uint32_t));
    }
}

} // namespace

BinaryWeightPagePlacement resolve_weight_page_placement(
    const BinaryBinding& binding, std::uint32_t pageIndex)
{
    if (!binding.paged_weight || pageIndex >= binding.page_count)
        throw std::out_of_range(
            "weight page placement index is outside its binding");
    const bool hasExactPlacements = !binding.page_banks.empty();
    if (hasExactPlacements) {
        if (binding.page_banks.size() != binding.page_count
            || binding.page_slice_group_bases.size() != binding.page_count
            || binding.page_slice_group_counts.size() != binding.page_count
            || binding.page_base_rows.size() != binding.page_count
            || binding.page_row_counts.size() != binding.page_count)
            throw std::logic_error(
                "weight page placement arrays do not match page_count");
        return BinaryWeightPagePlacement {
            binding.page_banks[pageIndex],
            binding.page_slice_group_bases[pageIndex],
            binding.page_slice_group_counts[pageIndex],
            binding.page_base_rows[pageIndex],
            binding.page_row_counts[pageIndex],
        };
    }
    const std::uint32_t bankCount =
        std::max<std::uint32_t>(1, binding.page_bank_count);
    return BinaryWeightPagePlacement {
        static_cast<std::uint16_t>(
            (binding.bank + pageIndex) % bankCount),
        static_cast<std::uint16_t>(binding.page_role_group_base),
        static_cast<std::uint16_t>(
            std::max<std::uint32_t>(1, binding.page_role_group_count)),
        0,
        static_cast<std::uint32_t>(
            std::max<std::int64_t>(1, binding.instruction_count)),
    };
}

BinaryBarrier make_binary_barrier(std::uint64_t cycle, std::uint32_t event_id)
{
    if (cycle > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("FTLPU barrier cycle exceeds uint32 logical time");
    if (event_id == 0)
        throw std::runtime_error("FTLPU barrier event ID zero is reserved");
    return {static_cast<std::uint32_t>(cycle), event_id};
}

void validate_binary_barriers(const BinaryProgram& program)
{
    const auto& hardware = program.hardware;
    if (program.transport_overlap_protocol && (!program.common_barrier_protocol
        || hardware.common_transport_overlap_version != 1))
        throw std::runtime_error("unsupported FTLPU transport overlap protocol");
    std::unordered_map<std::uint32_t, std::uint32_t> launch_cycles;
    for (const auto& use : program.weight_page_uses) {
        if (!program.transport_overlap_protocol && use.transport_launch_cycle)
            throw std::runtime_error("transport launch cycle requires the overlap protocol flag");
        if (!use.transport_launch_event_id) {
            if (use.transport_launch_cycle) throw std::runtime_error("orphan transport launch cycle");
            continue;
        }
        const auto [entry, inserted] = launch_cycles.emplace(use.transport_launch_event_id, use.transport_launch_cycle);
        if (!inserted && entry->second != use.transport_launch_cycle)
            throw std::runtime_error("inconsistent transport launch cycle");
    }
    if (program.common_barrier_protocol
        && (hardware.common_barrier_encoding_version != 1 || hardware.common_barrier_capacity == 0
            || hardware.common_barrier_capacity > hw::kCommonBarrierCapacity))
        throw std::runtime_error("unsupported FTLPU common barrier hardware capability");
    if ((!program.barriers.empty() || !program.barrier_events.empty()
            || (!program.queue_control_protocol && !program.transport_queues.empty()))
        && !program.common_barrier_protocol)
        throw std::runtime_error("FTLPU common barrier data requires the protocol flag");
    if (program.barriers.size() > hardware.common_barrier_capacity
        || program.barrier_events.size() > hardware.common_barrier_capacity)
        throw std::runtime_error("FTLPU common barrier RAM capacity exceeded");
    if (program.common_barrier_protocol && program.max_cycle > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("FTLPU common barrier program exceeds uint32 logical time");
    if (program.barrier_events.size() != program.barriers.size())
        throw std::runtime_error("FTLPU common barrier events and entries must have a one-to-one association");
    std::unordered_set<std::uint32_t> event_ids;
    std::unordered_set<std::uint32_t> instances;
    std::unordered_map<std::uint32_t, std::uint32_t> launch_events;
    std::size_t completion_count = 0;
    for (const auto& event : program.barrier_events) {
        if (event.event_id == 0 || !event_ids.insert(event.event_id).second || event.page_use_indices.empty())
            throw std::runtime_error("invalid or duplicate FTLPU common barrier event ID");
        if (event.page_use_indices.size() > kBinaryCommonBarrierMaxCompletionInstances - completion_count)
            throw std::runtime_error("FTLPU common barrier completion capacity exceeds 65535 instances");
        completion_count += event.page_use_indices.size();
        for (const auto instance : event.page_use_indices) {
            if (instance >= program.weight_page_uses.size() || !instances.insert(instance).second)
                throw std::runtime_error("invalid or repeated FTLPU barrier transport instance");
            const auto& use = program.weight_page_uses[instance];
            const auto binding = std::find_if(program.bindings.begin(), program.bindings.end(),
                [&](const auto& value) { return value.access == BindingAccess::Input
                    && value.index == use.binding_index; });
            if (use.transport_launch_event_id == 0)
                throw std::runtime_error("FTLPU barrier transport instance has no DMA/RX launch event");
            const auto [launch, inserted] = launch_events.emplace(use.transport_launch_event_id, event.event_id);
            if (!inserted && launch->second != event.event_id)
                throw std::runtime_error("FTLPU transport launch event is reused across common barriers");
            if (binding == program.bindings.end() || !binding->paged_weight
                || use.page_index >= binding->page_count || use.bank >= hardware.banks_per_slice
                || use.release_cycle < use.ready_cycle)
                throw std::runtime_error("FTLPU barrier references an invalid page transport instance");
            if (resolve_weight_page_placement(*binding, use.page_index).bank != use.bank)
                throw std::runtime_error("FTLPU barrier transport bank differs from page placement");
        }
    }
    std::unordered_set<std::uint32_t> consumed;
    std::uint32_t previous = 0;
    bool first = true;
    for (const auto& barrier : program.barriers) {
        if ((!first && barrier.stop_cycle <= previous) || barrier.stop_cycle > program.max_cycle)
            throw std::runtime_error("FTLPU common barriers must have strictly increasing in-range cycles");
        if (!event_ids.contains(barrier.event_id) || !consumed.insert(barrier.event_id).second)
            throw std::runtime_error("FTLPU barrier references an unknown or repeated event");
        const auto event = std::find_if(program.barrier_events.begin(), program.barrier_events.end(),
            [&](const auto& value) { return value.event_id == barrier.event_id; });
        for (const auto instance : event->page_use_indices) {
            const auto& use = program.weight_page_uses[instance];
            if (barrier.stop_cycle > use.ready_cycle)
                throw std::runtime_error("FTLPU barrier is later than its page consumer");
            if (program.transport_overlap_protocol && use.transport_launch_cycle > barrier.stop_cycle)
                throw std::runtime_error("FTLPU transport launch is later than its barrier");
        }
        previous = barrier.stop_cycle;
        first = false;
    }
    for (const auto& queue : program.transport_queues) {
        if (queue.kind != QueueKind::Mem || queue.packed_macro_imem.has_value()
            || queue.requires_raw_macro)
            throw std::runtime_error("FTLPU transport control queues must be Native MEM");
        for (const auto& command : queue.commands)
            if (is_macro_schedule_command(command))
                throw std::runtime_error("FTLPU transport control queue contains a Macro descriptor");
    }
}

void validate_queue_sync_plan(const BinaryProgram& program, bool bindings)
{
    const auto& plan=program.queue_sync_plan;
    const auto& hw=program.hardware;
    if (!program.queue_control_protocol) {
        if (!plan.partitions.empty() || !plan.events.empty() || !program.queue_link_source.queues.empty() || program.queue_link_source.required)
            throw std::runtime_error("QCTL data lacks protocol capability");
        return;
    }
    if (program.common_barrier_protocol || program.transport_overlap_protocol
        || hw.queue_control_version != 1 || !hw.queue_control_capacity
        || hw.queue_control_capacity > ftlpu::hw::kQueueControlCapacity
        || !hw.queue_control_cache_depth || hw.queue_control_cache_depth > hw.queue_control_capacity
        || !hw.queue_control_read_latency || hw.queue_control_read_ports != 1
        || program.max_cycle > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("unsupported QCTL protocol, geometry or timeline");
    const auto& source=program.queue_link_source;
    if(source.queues.size()>4096 || source.address_relocations.size()>1000000 || source.scale_relocations.size()>1000000
        || (source.required && source.queues.empty())) throw std::runtime_error("invalid QLNK source capacity");
    std::unordered_set<std::uint32_t> source_queues;
    std::size_t source_words=0;
    for(const auto& q:source.queues) {
        if(q.requires_raw_macro || q.packed_macro_imem || q.index>65535 || static_cast<unsigned>(q.kind)>9
            || !source_queues.insert((static_cast<std::uint32_t>(q.kind)<<16)|q.index).second
            || q.commands.size()>1000000-source_words)
            throw std::runtime_error("QLNK source is not a bounded Native domain plan");
        source_words+=q.commands.size();
        for(const auto& c:q.commands)
            if(is_macro_schedule_command(c)) throw std::runtime_error("QLNK cannot contain already folded Macro records");
    }
    const auto check_source_relocation=[&](const auto& a) {
        const auto q=std::ranges::find_if(source.queues,[&](const auto& q){return q.kind==a.queue_kind && q.index==a.queue_index;});
        if(q==source.queues.end() || a.command_index>=q->commands.size()) throw std::runtime_error("QLNK relocation has no source boundary");
    };
    for(const auto& a:source.address_relocations) {
        if(a.transport_queue || static_cast<unsigned>(a.binding_access)>2) throw std::runtime_error("invalid QLNK address binding");
        check_source_relocation(a);
    }
    for(const auto& a:source.scale_relocations) check_source_relocation(a);
    if (plan.events.size()>65535) throw std::runtime_error("QCTL event capacity exceeded");
    std::unordered_set<std::uint32_t> events,used,instances,partitions;
    for (const auto& e:plan.events) {
        if ((e.source==QueueEventSource::PageCommit && !e.event_id) || !events.insert(e.event_id).second
            || (e.source != QueueEventSource::External && e.source != QueueEventSource::PageCommit)
            || ((e.source == QueueEventSource::External) != e.page_use_indices.empty()))
            throw std::runtime_error("invalid QCTL event association");
        for (const auto i:e.page_use_indices) {
            if (i>=program.weight_page_uses.size() || !instances.insert(i).second || instances.size()>65535)
                throw std::runtime_error("invalid QCTL transport instance");
            const auto& use=program.weight_page_uses[i];
            const auto binding=std::ranges::find_if(program.bindings,[&](const auto& b) {
                return b.access==BindingAccess::Input && b.index==use.binding_index;
            });
            if (binding==program.bindings.end() || !binding->paged_weight || use.page_index>=binding->page_count
                || use.release_cycle<use.ready_cycle || !use.transport_launch_event_id
                || resolve_weight_page_placement(*binding,use.page_index).bank!=use.bank)
                throw std::runtime_error("QCTL event references invalid page placement or launch");
        }
    }
    if(!source.required)
        for(std::uint32_t i=0;i<program.weight_page_uses.size();++i)
            if(program.weight_page_uses[i].runtime_prefetch && !instances.contains(i))
                throw std::runtime_error("QCTL runtime page has no SRAM completion event");
    std::unordered_set<std::uint32_t> transport_partitions;
    for (const auto& p:plan.partitions) {
        std::size_t limit=0;
        switch(p.kind) {
        case QueueKind::Mem: limit=hw.hemispheres*hw.slices_per_hemisphere*hw.banks_per_slice;break;
        case QueueKind::MxmLoad: case QueueKind::MxmCompute: case QueueKind::MxmDequant: limit=hw.hemispheres*hw.mxms_per_hemisphere;break;
        case QueueKind::Vxm: limit=hw.vxm_alus;break;
        case QueueKind::SxmTranspose: case QueueKind::SxmPermute:
        case QueueKind::C2cDma: case QueueKind::C2cTx: case QueueKind::C2cRx: limit=hw.hemispheres;break;
        default: throw std::runtime_error("unknown QCTL queue kind");
        }
        if (p.index>=limit || !partitions.insert((static_cast<std::uint32_t>(p.kind)<<16)|p.index).second
            || p.entries.size()>hw.queue_control_capacity || static_cast<std::uint64_t>(p.end_cycle)>static_cast<std::uint64_t>(program.max_cycle)+1)
            throw std::runtime_error("invalid QCTL partition or capacity: kind="+std::to_string(static_cast<unsigned>(p.kind))+" queue="+std::to_string(p.index)+" end="+std::to_string(p.end_cycle)+" nominal_last="+std::to_string(program.max_cycle));
        const auto q=std::ranges::find_if(program.queues,[&](const auto& queue) {return queue.kind==p.kind && queue.index==p.index;});
        if (bindings && q==program.queues.end()) throw std::runtime_error("QCTL targets a missing queue");
        if (bindings && p.entries.empty() && !q->requires_raw_macro && !q->packed_macro_imem)
            throw std::runtime_error("empty QCTL extent must name a Raw computation queue");
        std::unordered_set<std::uint32_t> sources;
        std::uint32_t previous=0,transport_ordinal=0;
        std::optional<std::uint32_t> previous_source;
        for (const auto& c:p.entries) {
            if ((previous_source && c.source_command<=*previous_source) || c.local_cycle<previous || c.local_cycle>p.end_cycle || !sources.insert(c.source_command).second
                || (c.opcode!=IcuControlOpcode::Sync && c.opcode!=IcuControlOpcode::WaitEvent && c.opcode!=IcuControlOpcode::Notify && c.opcode!=kQueueControlMemPacket)
                || (c.opcode!=IcuControlOpcode::WaitEvent && c.opcode!=kQueueControlMemPacket && (c.operand || c.release_group || c.phase_offset))
                || (c.opcode==kQueueControlMemPacket && (p.kind!=QueueKind::Mem || !c.operand || c.release_group || c.phase_offset))
                || (c.release_group && c.release_group!=c.operand) || (!c.release_group && (c.phase_offset || c.activation_producer)))
                throw std::runtime_error("invalid QCTL operation, position, order or binding");
            previous=c.local_cycle;
            previous_source=c.source_command;
            if (c.opcode==kQueueControlMemPacket && c.operand!=++transport_ordinal)
                throw std::runtime_error("QCTL transport packet ordinal is not contiguous");
            if(c.opcode==IcuControlOpcode::WaitEvent) {
                if(!events.contains(c.operand)) throw std::runtime_error("QCTL wait has an orphan event");
                const auto event=std::ranges::find_if(plan.events,[&](const auto& e){return e.event_id==c.operand;});
                if ((event->source==QueueEventSource::PageCommit) != (c.release_group!=0))
                    throw std::runtime_error("QCTL page wait lacks a release plan or external event has one");
                used.insert(c.operand);
            }
            if (!bindings) continue;
            const bool raw=q->requires_raw_macro || q->packed_macro_imem.has_value();
            if (raw==c.native_binding) throw std::runtime_error("QCTL binding would duplicate or omit a control");
            if(c.native_binding && c.opcode==kQueueControlMemPacket) {
                if(c.source_command>=q->commands.size() || !is_mem_synchronized_raw_packet_header(q->commands[c.source_command]))
                    throw std::runtime_error("QCTL MEM packet binding is invalid");
            } else if(c.native_binding) {
                if(c.source_command>=q->commands.size() || !is_icu_control_raw_word_command(q->commands[c.source_command]))
                    throw std::runtime_error("QCTL Native binding does not name a control");
                const auto native=decode_icu_control_raw_word(q->commands[c.source_command]);
                if(native.opcode!=c.opcode || native.event_tag!=c.operand)
                    throw std::runtime_error("QCTL Native control differs from binding");
            }
        }
        if(bindings && transport_ordinal && !p.entries.front().native_binding) {
            const auto transport=std::ranges::find_if(program.transport_queues,[&](const auto& t){return t.kind==p.kind && t.index==p.index;});
            if(transport==program.transport_queues.end() || !transport_partitions.insert(p.index).second)
                throw std::runtime_error("QCTL transport has no independent Native store");
            std::size_t packets=0;
            for(std::size_t pc=0;pc<transport->commands.size();) {
                if(!is_mem_synchronized_raw_packet_header(transport->commands[pc]))
                    throw std::runtime_error("QCTL transport store contains an unowned operation");
                (void)decode_mem_synchronized_icu_packet(*transport,pc);
                ++packets;pc+=InstructionControlUnit::MemIcu::synchronized_packet_word_count;
            }
            if(packets!=transport_ordinal) throw std::runtime_error("QCTL transport binding count differs from packet store");
        }
    }
    if(bindings && transport_partitions.size()!=program.transport_queues.size())
        throw std::runtime_error("QCTL contains an orphan transport partition");
    if(used.size()!=events.size()) throw std::runtime_error("QCTL event is unreferenced");
}

void write_binary_program(const BinaryProgram& program, std::ostream& os)
{
    validate_binary_barriers(program);
    validate_queue_sync_plan(program);
    os.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar<std::uint32_t>(os, kBinaryFormatVersion);
    if (program.target_name.empty()
        || program.target_name.size() > std::numeric_limits<std::uint16_t>::max()
        || program.target_abi == 0)
        throw std::runtime_error("FTLPU binary requires a valid target identity");
    write_scalar<std::uint64_t>(os, program.target_abi);
    write_scalar<std::uint16_t>(
        os, static_cast<std::uint16_t>(program.target_name.size()));
    os.write(program.target_name.data(),
        static_cast<std::streamsize>(program.target_name.size()));
    if (!os) throw std::runtime_error("failed to write FTLPU target name");
    if (program.target_abi != executable_target_abi(program.hardware))
        throw std::runtime_error(
            "FTLPU target ABI does not match its hardware configuration");
    if (program.hardware.mxms_per_hemisphere == 0)
        throw std::runtime_error(
            "FTLPU binary requires a nonzero MXM topology");
    if (!program.stream_release_cycles.empty()
        && program.stream_release_cycles.size()
            != program.hardware.encoded_streams)
        throw std::runtime_error(
            "FTLPU stream-release metadata must describe every packed "
            "ordinary stream register");
    program.hardware.visit([&](std::uint32_t value) {
        write_scalar<std::uint32_t>(os, value);
    });
    write_scalar<std::uint64_t>(os, static_cast<std::uint64_t>(program.max_cycle));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.queues.size() + program.transport_queues.size()));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.bindings.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.scale_relocations.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.address_relocations.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.timelines.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.memory_floors.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.weight_page_uses.size()));
    write_scalar<std::uint32_t>(
        os, static_cast<std::uint32_t>(program.stream_release_cycles.size()));
    write_le<std::uint32_t>(os, static_cast<std::uint32_t>(program.barrier_events.size()));
    write_le<std::uint32_t>(os, (program.common_barrier_protocol ? 1u : 0u)
        | (program.transport_overlap_protocol ? 2u : 0u)
        | (program.queue_control_protocol ? 4u : 0u));
    write_le<std::uint32_t>(os, static_cast<std::uint32_t>(program.transport_queues.size()));

    for (const auto& binding : program.bindings) write_binding(os, binding);
    for (const auto& timeline : program.timelines)
        write_timeline(os, timeline);
    for (const BinaryMemoryFloor& floor : program.memory_floors) {
        write_scalar<std::uint16_t>(os, floor.hemisphere);
        write_scalar<std::uint16_t>(os, floor.slice);
        write_scalar<std::uint16_t>(os, floor.bank);
        write_scalar<std::uint32_t>(os, floor.first_free_row);
    }
    for (const BinaryWeightPageUse& use : program.weight_page_uses) {
        write_scalar<std::uint32_t>(os, use.binding_index);
        write_scalar<std::uint32_t>(os, use.page_index);
        write_scalar<std::uint16_t>(os, use.bank);
        write_scalar<std::uint16_t>(os, use.runtime_prefetch ? 1 : 0);
        write_scalar<std::uint64_t>(os, use.ready_cycle);
        write_scalar<std::uint64_t>(os, use.release_cycle);
        write_le<std::uint32_t>(os, use.transport_launch_event_id);
        write_le<std::uint32_t>(os, use.transport_launch_cycle);
    }
    for (const std::uint64_t cycle : program.stream_release_cycles)
        write_scalar<std::uint64_t>(os, cycle);

    write_barrier_sections(os, program);
    write_queue_controls(os, program);
    write_queue_link_source(os,program);
    for (std::size_t queue_id = 0;
         queue_id < program.queues.size() + program.transport_queues.size(); ++queue_id) {
        const auto& queue = queue_id < program.queues.size()
            ? program.queues[queue_id] : program.transport_queues[queue_id - program.queues.size()];
        write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(queue.kind));
        write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(queue.index));
        const auto wordBits = queue.kind == QueueKind::Mem
            ? program.hardware.icu_mem_instruction_bits : program.hardware.icu_mxm_instruction_bits;
        const bool rawSupported = program.hardware.icu_macro_encoding_version == 1
            && wordBits == (queue.kind == QueueKind::Mem ? kMemIcuImemWordBits : kMxmIcuImemWordBits);
        const auto mode = queue_encoding_mode(queue, rawSupported);
        write_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(mode));
        write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(queue.commands.size()));
        if (mode == QueueEncodingMode::MacroPacked) {
            if (queue.kind == QueueKind::Mem)
                write_mem_macro_bitstream(os, queue);
            else
                write_mxm_macro_bitstream(os, queue);
        } else if (mode == QueueEncodingMode::MacroSchedule) {
            write_macro_queue_commands(os, queue);
        } else if (mode == QueueEncodingMode::CompactTagged) {
            for (const auto& command : queue.commands)
                write_compact_queue_command(os, command);
        } else {
            for (const auto& command : queue.commands)
                write_native_queue_command(os, queue.kind, command);
        }
    }
    for (const auto& relocation : program.scale_relocations) {
        write_scalar<std::uint32_t>(os, relocation.binding_index);
        write_scalar<std::uint32_t>(os, relocation.scale_index);
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.queue_kind));
        write_scalar<std::uint16_t>(os, relocation.queue_index);
        write_scalar<std::uint32_t>(os, relocation.command_index);
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.operand));
    }
    for (const auto& relocation : program.address_relocations) {
        write_scalar<std::uint32_t>(os, relocation.binding_index);
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.binding_access));
        write_scalar<std::uint16_t>(
            os, static_cast<std::uint16_t>(relocation.queue_kind));
        write_scalar<std::uint16_t>(os, relocation.queue_index);
        write_scalar<std::uint32_t>(os, relocation.command_index);
        write_scalar<std::uint16_t>(
            os, relocation.write_port ? 1 : 0);
        write_le<std::uint16_t>(os, relocation.transport_queue ? 1 : 0);
    }
}

BinaryProgram read_binary_program(std::istream& is)
{
    std::array<char, 8> magic {};
    is.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!is || magic != kMagic) {
        throw std::runtime_error("invalid FTLPU binary magic");
    }

    const auto version = read_scalar<std::uint32_t>(is);
    if (version < 1 || version > kBinaryFormatVersion) {
        throw std::runtime_error("unsupported FTLPU binary version");
    }

    auto program = BinaryProgram {};
    if (version >= 5) {
        program.target_abi = read_scalar<std::uint64_t>(is);
        const auto target_name_size = read_scalar<std::uint16_t>(is);
        program.target_name.resize(target_name_size);
        is.read(program.target_name.data(),
            static_cast<std::streamsize>(program.target_name.size()));
        if (!is) throw std::runtime_error("truncated FTLPU target name");
        if (version >= 16) {
            read_hardware_config(program, version,
                [&]() { return read_scalar<std::uint32_t>(is); });
        } else {
            if (version >= 9)
                program.hardware.sram_depth_rows =
                    read_scalar<std::uint32_t>(is);
            if (version >= 15)
                program.hardware.mxms_per_hemisphere =
                    read_scalar<std::uint32_t>(is);
        }
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle = static_cast<std::size_t>(read_scalar<std::uint64_t>(is));
    const auto queue_count = read_scalar<std::uint32_t>(is);
    const auto binding_count = version >= 2 ? read_scalar<std::uint32_t>(is) : 0;
    const auto relocation_count =
        version >= 8 ? read_scalar<std::uint32_t>(is) : 0;
    const auto address_relocation_count =
        version >= 9 ? read_scalar<std::uint32_t>(is) : 0;
    const auto timeline_count =
        version >= 11 ? read_scalar<std::uint32_t>(is) : 0;
    const auto memory_floor_count =
        version >= 12 ? read_scalar<std::uint32_t>(is) : 0;
    const auto weight_page_use_count =
        version >= 22 ? read_scalar<std::uint32_t>(is) : 0;
    const auto stream_release_count =
        version >= 31 ? read_scalar<std::uint32_t>(is) : 0;
    const auto barrier_event_count = version >= 37 ? read_scalar<std::uint32_t>(is) : 0;
    const auto metadata_flags = version >= 37 ? read_scalar<std::uint32_t>(is) : 0;
    const auto transport_queue_count = version >= 37 ? read_scalar<std::uint32_t>(is) : 0;
    if ((metadata_flags & ~(version >= 39 ? 7u : version >= 38 ? 3u : 1u)) != 0 || transport_queue_count > queue_count
        || (transport_queue_count != 0 && (metadata_flags & 5u) == 0))
        throw std::runtime_error("invalid FTLPU common barrier metadata flags or queue count");
    program.common_barrier_protocol = (metadata_flags & 1u) != 0;
    program.transport_overlap_protocol = (metadata_flags & 2u) != 0;
    program.queue_control_protocol = (metadata_flags & 4u) != 0;
    program.bindings.reserve(binding_count);
    for (std::uint32_t binding_id = 0; binding_id < binding_count; ++binding_id)
        program.bindings.push_back(read_binding(is, version));
    program.timelines.reserve(timeline_count);
    for (std::uint32_t timeline_id = 0;
         timeline_id < timeline_count; ++timeline_id)
        program.timelines.push_back(read_timeline(is));
    program.memory_floors.reserve(memory_floor_count);
    for (std::uint32_t index = 0;
         index < memory_floor_count; ++index) {
        const auto hemisphere = read_scalar<std::uint16_t>(is);
        const auto slice = read_scalar<std::uint16_t>(is);
        const auto bank = version >= 17
            ? read_scalar<std::uint16_t>(is) : std::uint16_t {0};
        const auto firstFreeRow = read_scalar<std::uint32_t>(is);
        program.memory_floors.push_back(BinaryMemoryFloor {
            hemisphere, slice, firstFreeRow, bank,
        });
    }
    program.weight_page_uses.reserve(weight_page_use_count);
    for (std::uint32_t index = 0;
         index < weight_page_use_count; ++index) {
        BinaryWeightPageUse use;
        use.binding_index = read_scalar<std::uint32_t>(is);
        use.page_index = read_scalar<std::uint32_t>(is);
        use.bank = read_scalar<std::uint16_t>(is);
        use.runtime_prefetch = read_scalar<std::uint16_t>(is) != 0;
        use.ready_cycle = read_scalar<std::uint64_t>(is);
        use.release_cycle = read_scalar<std::uint64_t>(is);
        if (version >= 37)
            use.transport_launch_event_id = read_le<std::uint32_t>(is);
        if (version >= 38)
            use.transport_launch_cycle = read_le<std::uint32_t>(is);
        program.weight_page_uses.push_back(use);
    }
    program.stream_release_cycles.reserve(stream_release_count);
    for (std::uint32_t index = 0; index < stream_release_count; ++index)
        program.stream_release_cycles.push_back(
            read_scalar<std::uint64_t>(is));
    if (version >= 37)
        read_barrier_sections(is, program, barrier_event_count);
    if (version >= 39) { read_queue_controls(is, program);read_queue_link_source(is,program); validate_queue_sync_plan(program, false); }
    program.queues.reserve(queue_count - transport_queue_count);

    for (std::uint32_t queue_id = 0; queue_id < queue_count; ++queue_id) {
        auto queue = QueueProgram {};
        queue.kind = static_cast<QueueKind>(read_scalar<std::uint16_t>(is));
        queue.index = read_scalar<std::uint16_t>(is);
        const auto mode = has_queue_encoding_mode(version)
            ? decode_queue_encoding_mode(read_scalar<std::uint8_t>(is))
            : QueueEncodingMode::CompactTagged;
        const auto command_count = read_scalar<std::uint32_t>(is);
        queue.commands.reserve(command_count);

        std::vector<std::uint8_t> macroWidths;
        if (has_queue_encoding_mode(version)
            && mode == QueueEncodingMode::MacroSchedule)
            macroWidths = read_macro_width_bitmap(is, command_count);

        if (queue_id >= queue_count - transport_queue_count
            && (queue.kind != QueueKind::Mem || mode == QueueEncodingMode::MacroPacked
                || mode == QueueEncodingMode::MacroSchedule))
            throw std::runtime_error("invalid FTLPU transport control queue encoding");
        if (mode == QueueEncodingMode::MacroPacked) {
            if (!has_macro_packed(version, queue.kind))
                throw std::runtime_error(
                    "unsupported FTLPU packed Macro queue kind or version");
            if (queue.kind == QueueKind::Mem)
                program.queues.push_back(read_mem_macro_bitstream(
                    is, command_count, queue.index));
            else
                program.queues.push_back(read_mxm_macro_bitstream(
                    is, command_count, queue.index, queue.kind));
            continue;
        }

        for (std::uint32_t command_id = 0; command_id < command_count; ++command_id) {
            if (has_queue_encoding_mode(version)) {
                if (mode == QueueEncodingMode::Native)
                    queue.commands.push_back(
                        read_native_queue_command(is, queue.kind));
                else if (mode == QueueEncodingMode::MacroSchedule)
                    queue.commands.push_back(read_macro_queue_command(is,
                        queue.kind,
                        (macroWidths[command_id / 8]
                            & (1u << (command_id % 8))) != 0));
                else
                    queue.commands.push_back(read_compact_queue_command(is));
                continue;
            }
            if (version >= 25) {
                queue.commands.push_back(read_compact_queue_command(is));
                continue;
            }
            auto command = QueueCommand {};
            command.command = read_scalar<std::uint32_t>(is);
            command.instruction_kind = static_cast<InstructionKind>(read_scalar<std::uint16_t>(is));
            command.word_count = read_scalar<std::uint16_t>(is);
            const std::size_t serialized_words = version >= 3 ? command.words.size() : 3;
            for (std::size_t word = 0; word < serialized_words; ++word)
                command.words[word] = read_scalar<std::uint32_t>(is);
            if (version >= 4) {
                const auto extension_count = read_scalar<std::uint16_t>(is);
                command.extension_words.reserve(extension_count);
                for (std::uint16_t word = 0; word < extension_count; ++word)
                    command.extension_words.push_back(read_scalar<std::uint32_t>(is));
            }
            queue.commands.push_back(std::move(command));
        }
        if (queue_id < queue_count - transport_queue_count)
            program.queues.push_back(std::move(queue));
        else
            program.transport_queues.push_back(std::move(queue));
    }
    program.scale_relocations.reserve(relocation_count);
    for (std::uint32_t index = 0; index < relocation_count; ++index) {
        BinaryScaleRelocation relocation;
        relocation.binding_index = read_scalar<std::uint32_t>(is);
        relocation.scale_index = read_scalar<std::uint32_t>(is);
        relocation.queue_kind =
            static_cast<QueueKind>(read_scalar<std::uint16_t>(is));
        relocation.queue_index = read_scalar<std::uint16_t>(is);
        relocation.command_index = read_scalar<std::uint32_t>(is);
        relocation.operand = static_cast<VxmImmediateOperand>(
            read_scalar<std::uint16_t>(is));
        program.scale_relocations.push_back(relocation);
    }
    program.address_relocations.reserve(address_relocation_count);
    for (std::uint32_t index = 0;
         index < address_relocation_count; ++index) {
        BinaryAddressRelocation relocation;
        relocation.binding_index = read_scalar<std::uint32_t>(is);
        if (version >= 10)
            relocation.binding_access = static_cast<BindingAccess>(
                read_scalar<std::uint16_t>(is));
        relocation.queue_kind =
            static_cast<QueueKind>(read_scalar<std::uint16_t>(is));
        relocation.queue_index = read_scalar<std::uint16_t>(is);
        relocation.command_index = read_scalar<std::uint32_t>(is);
        if (version >= 14)
            relocation.write_port =
                read_scalar<std::uint16_t>(is) != 0;
        if (version >= 37) {
            const auto flags = read_le<std::uint16_t>(is);
            if (flags > 1) throw std::runtime_error("invalid FTLPU address relocation flags");
            relocation.transport_queue = flags != 0;
        }
        program.address_relocations.push_back(relocation);
    }

    validate_binary_barriers(program);
    validate_queue_sync_plan(program);
    return program;
}

BinaryProgram read_binary_program_metadata(std::istream& is)
{
    const auto start = is.tellg();
    std::array<char, 8> magic {};
    is.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!is || magic != kMagic)
        throw std::runtime_error("invalid FTLPU binary magic");

    const auto version = read_scalar<std::uint32_t>(is);
    if (version >= 39 && version <= kBinaryFormatVersion) {
        is.seekg(start);
        auto metadata=read_binary_program(is);
        metadata.queues.clear(); metadata.transport_queues.clear();
        metadata.address_relocations.clear(); metadata.scale_relocations.clear();
        return metadata;
    }
    if (version < 1 || version > kBinaryFormatVersion)
        throw std::runtime_error("unsupported FTLPU binary version");

    BinaryProgram program;
    if (version >= 5) {
        program.target_abi = read_scalar<std::uint64_t>(is);
        const auto target_name_size = read_scalar<std::uint16_t>(is);
        program.target_name.resize(target_name_size);
        is.read(program.target_name.data(),
            static_cast<std::streamsize>(program.target_name.size()));
        if (!is) throw std::runtime_error("truncated FTLPU target name");
        if (version >= 16) {
            read_hardware_config(program, version,
                [&]() { return read_scalar<std::uint32_t>(is); });
        } else {
            if (version >= 9)
                program.hardware.sram_depth_rows =
                    read_scalar<std::uint32_t>(is);
            if (version >= 15)
                program.hardware.mxms_per_hemisphere =
                    read_scalar<std::uint32_t>(is);
        }
    } else {
        program.target_name = "legacy-unidentified";
        program.target_abi = 0;
    }
    program.max_cycle =
        static_cast<std::size_t>(read_scalar<std::uint64_t>(is));
    const auto queue_count = read_scalar<std::uint32_t>(is);
    const auto binding_count =
        version >= 2 ? read_scalar<std::uint32_t>(is) : 0;
    const auto relocation_count =
        version >= 8 ? read_scalar<std::uint32_t>(is) : 0;
    const auto address_relocation_count =
        version >= 9 ? read_scalar<std::uint32_t>(is) : 0;
    const auto timeline_count =
        version >= 11 ? read_scalar<std::uint32_t>(is) : 0;
    const auto memory_floor_count =
        version >= 12 ? read_scalar<std::uint32_t>(is) : 0;
    const auto weight_page_use_count =
        version >= 22 ? read_scalar<std::uint32_t>(is) : 0;
    const auto stream_release_count =
        version >= 31 ? read_scalar<std::uint32_t>(is) : 0;
    const auto barrier_event_count = version >= 37 ? read_scalar<std::uint32_t>(is) : 0;
    const auto metadata_flags = version >= 37 ? read_scalar<std::uint32_t>(is) : 0;
    const auto transport_queue_count = version >= 37 ? read_scalar<std::uint32_t>(is) : 0;
    if ((metadata_flags & ~(version >= 39 ? 7u : version >= 38 ? 3u : 1u)) != 0 || transport_queue_count > queue_count
        || (transport_queue_count != 0 && (metadata_flags & 5u) == 0))
        throw std::runtime_error("invalid FTLPU common barrier metadata flags or queue count");
    program.common_barrier_protocol = (metadata_flags & 1u) != 0;
    program.transport_overlap_protocol = (metadata_flags & 2u) != 0;
    program.queue_control_protocol = (metadata_flags & 4u) != 0;

    program.bindings.reserve(binding_count);
    for (std::uint32_t index = 0; index < binding_count; ++index)
        program.bindings.push_back(read_binding(is, version));
    program.timelines.reserve(timeline_count);
    for (std::uint32_t index = 0; index < timeline_count; ++index)
        program.timelines.push_back(read_timeline(is));
    program.memory_floors.reserve(memory_floor_count);
    for (std::uint32_t index = 0;
         index < memory_floor_count; ++index) {
        const auto hemisphere = read_scalar<std::uint16_t>(is);
        const auto slice = read_scalar<std::uint16_t>(is);
        const auto bank = version >= 17
            ? read_scalar<std::uint16_t>(is) : std::uint16_t {0};
        const auto firstFreeRow = read_scalar<std::uint32_t>(is);
        program.memory_floors.push_back(BinaryMemoryFloor {
            hemisphere, slice, firstFreeRow, bank,
        });
    }
    program.weight_page_uses.reserve(weight_page_use_count);
    for (std::uint32_t index = 0; index < weight_page_use_count; ++index) {
        BinaryWeightPageUse use;
        use.binding_index = read_scalar<std::uint32_t>(is);
        use.page_index = read_scalar<std::uint32_t>(is);
        use.bank = read_scalar<std::uint16_t>(is);
        use.runtime_prefetch = read_scalar<std::uint16_t>(is) != 0;
        use.ready_cycle = read_scalar<std::uint64_t>(is);
        use.release_cycle = read_scalar<std::uint64_t>(is);
        if (version >= 37)
            use.transport_launch_event_id = read_le<std::uint32_t>(is);
        if (version >= 38)
            use.transport_launch_cycle = read_le<std::uint32_t>(is);
        program.weight_page_uses.push_back(use);
    }
    program.stream_release_cycles.reserve(stream_release_count);
    for (std::uint32_t index = 0; index < stream_release_count; ++index)
        program.stream_release_cycles.push_back(
            read_scalar<std::uint64_t>(is));
    if (version >= 37)
        read_barrier_sections(is, program, barrier_event_count);
    if (version >= 39) { read_queue_controls(is, program);read_queue_link_source(is,program); validate_queue_sync_plan(program, false); }

    for (std::uint32_t queue = 0; queue < queue_count; ++queue) {
        const auto queueKind = static_cast<QueueKind>(
            read_scalar<std::uint16_t>(is));
        (void)read_scalar<std::uint16_t>(is);
        const auto mode = has_queue_encoding_mode(version)
            ? decode_queue_encoding_mode(read_scalar<std::uint8_t>(is))
            : QueueEncodingMode::CompactTagged;
        const auto command_count = read_scalar<std::uint32_t>(is);
        std::vector<std::uint8_t> macroWidths;
        if (has_queue_encoding_mode(version)
            && mode == QueueEncodingMode::MacroSchedule)
            macroWidths = read_macro_width_bitmap(is, command_count);
        if (queue >= queue_count - transport_queue_count
            && (queueKind != QueueKind::Mem || mode == QueueEncodingMode::MacroPacked
                || mode == QueueEncodingMode::MacroSchedule))
            throw std::runtime_error("invalid FTLPU transport control queue encoding");
        if (mode == QueueEncodingMode::MacroPacked) {
            if (!has_macro_packed(version, queueKind))
                throw std::runtime_error(
                    "unsupported FTLPU packed Macro queue kind or version");
            skip_macro_bitstream(is, queueKind);
            continue;
        }
        for (std::uint32_t command = 0;
             command < command_count; ++command) {
            if (has_queue_encoding_mode(version)) {
                if (mode == QueueEncodingMode::Native)
                    skip_native_queue_command(is, queueKind);
                else if (mode == QueueEncodingMode::MacroSchedule)
                    skip_macro_queue_command(is,
                        (macroWidths[command / 8]
                            & (1u << (command % 8))) != 0);
                else
                    skip_compact_queue_command(is);
                continue;
            }
            if (version >= 25) {
                skip_compact_queue_command(is);
                continue;
            }
            (void)read_scalar<std::uint32_t>(is);
            (void)read_scalar<std::uint16_t>(is);
            (void)read_scalar<std::uint16_t>(is);
            skip_bytes(is, (version >= 3 ? 4u : 3u)
                    * sizeof(std::uint32_t));
            if (version >= 4) {
                const auto extension_count =
                    read_scalar<std::uint16_t>(is);
                skip_bytes(is, static_cast<std::uint64_t>(extension_count)
                        * sizeof(std::uint32_t));
            }
        }
    }

    skip_bytes(is, static_cast<std::uint64_t>(relocation_count)
            * (sizeof(std::uint32_t) * 3 + sizeof(std::uint16_t) * 3));
    const std::uint64_t address_relocation_size =
        sizeof(std::uint32_t) * 2 + sizeof(std::uint16_t) * 2
        + (version >= 10 ? sizeof(std::uint16_t) : 0)
        + (version >= 14 ? sizeof(std::uint16_t) : 0)
        + (version >= 37 ? sizeof(std::uint16_t) : 0);
    skip_bytes(is,
        static_cast<std::uint64_t>(address_relocation_count)
            * address_relocation_size);
    return program;
}

BinaryProgram read_binary_program(std::span<const std::uint8_t> data)
{
    ByteReader reader(data);
    BinaryHeader header = read_header(reader);
    BinaryProgram& program = header.program;
    program.queues.reserve(header.queue_count);
    for (std::uint32_t queue_id = 0;
         queue_id < header.queue_count; ++queue_id) {
        QueueProgram queue;
        queue.kind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        queue.index = reader.read<std::uint16_t>();
        const auto mode = has_queue_encoding_mode(header.version)
            ? decode_queue_encoding_mode(reader.read<std::uint8_t>())
            : QueueEncodingMode::CompactTagged;
        const auto command_count = reader.read<std::uint32_t>();
        queue.commands.reserve(command_count);
        std::vector<std::uint8_t> macroWidths;
        if (has_queue_encoding_mode(header.version)
            && mode == QueueEncodingMode::MacroSchedule)
            macroWidths = read_macro_width_bitmap(reader, command_count);
        if (queue_id >= header.queue_count - header.transport_queue_count
            && (queue.kind != QueueKind::Mem || mode == QueueEncodingMode::MacroPacked
                || mode == QueueEncodingMode::MacroSchedule))
            throw std::runtime_error("invalid FTLPU transport control queue encoding");
        if (mode == QueueEncodingMode::MacroPacked) {
            if (!has_macro_packed(header.version, queue.kind))
                throw std::runtime_error(
                    "unsupported FTLPU packed Macro queue kind or version");
            if (queue.kind == QueueKind::Mem)
                program.queues.push_back(read_mem_macro_bitstream(
                    reader, command_count, queue.index));
            else
                program.queues.push_back(read_mxm_macro_bitstream(
                    reader, command_count, queue.index, queue.kind));
            continue;
        }
        for (std::uint32_t command_id = 0;
             command_id < command_count; ++command_id) {
            if (has_queue_encoding_mode(header.version)) {
                if (mode == QueueEncodingMode::Native)
                    queue.commands.push_back(
                        read_native_queue_command(reader, queue.kind));
                else if (mode == QueueEncodingMode::MacroSchedule)
                    queue.commands.push_back(read_macro_queue_command(reader,
                        queue.kind,
                        (macroWidths[command_id / 8]
                            & (1u << (command_id % 8))) != 0));
                else
                    queue.commands.push_back(
                        read_compact_queue_command(reader));
                continue;
            }
            if (header.version >= 25) {
                queue.commands.push_back(read_compact_queue_command(reader));
                continue;
            }
            QueueCommand command;
            command.command = reader.read<std::uint32_t>();
            command.instruction_kind =
                static_cast<InstructionKind>(
                    reader.read<std::uint16_t>());
            command.word_count = reader.read<std::uint16_t>();
            const std::size_t serialized_words =
                header.version >= 3 ? command.words.size() : 3;
            for (std::size_t word = 0;
                 word < serialized_words; ++word)
                command.words[word] = reader.read<std::uint32_t>();
            if (header.version >= 4) {
                const auto extension_count =
                    reader.read<std::uint16_t>();
                command.extension_words.reserve(extension_count);
                for (std::uint16_t word = 0;
                     word < extension_count; ++word)
                    command.extension_words.push_back(
                        reader.read<std::uint32_t>());
            }
            queue.commands.push_back(std::move(command));
        }
        if (queue_id < header.queue_count - header.transport_queue_count)
            program.queues.push_back(std::move(queue));
        else
            program.transport_queues.push_back(std::move(queue));
    }
    program.scale_relocations.reserve(header.relocation_count);
    for (std::uint32_t index = 0;
         index < header.relocation_count; ++index) {
        BinaryScaleRelocation relocation;
        relocation.binding_index = reader.read<std::uint32_t>();
        relocation.scale_index = reader.read<std::uint32_t>();
        relocation.queue_kind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        relocation.queue_index = reader.read<std::uint16_t>();
        relocation.command_index = reader.read<std::uint32_t>();
        relocation.operand = static_cast<VxmImmediateOperand>(
            reader.read<std::uint16_t>());
        program.scale_relocations.push_back(relocation);
    }
    program.address_relocations.reserve(
        header.address_relocation_count);
    for (std::uint32_t index = 0;
         index < header.address_relocation_count; ++index) {
        BinaryAddressRelocation relocation;
        relocation.binding_index = reader.read<std::uint32_t>();
        if (header.version >= 10)
            relocation.binding_access = static_cast<BindingAccess>(
                reader.read<std::uint16_t>());
        relocation.queue_kind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        relocation.queue_index = reader.read<std::uint16_t>();
        relocation.command_index = reader.read<std::uint32_t>();
        if (header.version >= 14)
            relocation.write_port =
                reader.read<std::uint16_t>() != 0;
        if (header.version >= 37) {
            const auto flags = read_le<std::uint16_t>(reader);
            if (flags > 1) throw std::runtime_error("invalid FTLPU address relocation flags");
            relocation.transport_queue = flags != 0;
        }
        program.address_relocations.push_back(relocation);
    }
    if (!reader.empty())
        throw std::runtime_error("FTLPU binary has trailing data");
    validate_binary_barriers(program);
    validate_queue_sync_plan(program);
    return std::move(program);
}

BinaryProgram read_binary_program_metadata(
    std::span<const std::uint8_t> data)
{
    if (data.size() >= 12 && data[8] == 39 && data[9] == 0 && data[10] == 0 && data[11] == 0) {
        auto metadata=read_binary_program(data);
        metadata.queues.clear(); metadata.transport_queues.clear();
        metadata.address_relocations.clear(); metadata.scale_relocations.clear();
        return metadata;
    }
    ByteReader reader(data);
    BinaryHeader header = read_header(reader);
    for (std::uint32_t queue = 0;
         queue < header.queue_count; ++queue) {
        const auto queueKind =
            static_cast<QueueKind>(reader.read<std::uint16_t>());
        (void)reader.read<std::uint16_t>();
        const auto mode = has_queue_encoding_mode(header.version)
            ? decode_queue_encoding_mode(reader.read<std::uint8_t>())
            : QueueEncodingMode::CompactTagged;
        const auto command_count = reader.read<std::uint32_t>();
        std::vector<std::uint8_t> macroWidths;
        if (has_queue_encoding_mode(header.version)
            && mode == QueueEncodingMode::MacroSchedule)
            macroWidths = read_macro_width_bitmap(reader, command_count);
        if (queue >= header.queue_count - header.transport_queue_count
            && (queueKind != QueueKind::Mem || mode == QueueEncodingMode::MacroPacked
                || mode == QueueEncodingMode::MacroSchedule))
            throw std::runtime_error("invalid FTLPU transport control queue encoding");
        if (mode == QueueEncodingMode::MacroPacked) {
            if (!has_macro_packed(header.version, queueKind))
                throw std::runtime_error(
                    "unsupported FTLPU packed Macro queue kind or version");
            skip_macro_bitstream(reader, queueKind);
            continue;
        }
        for (std::uint32_t command = 0;
             command < command_count; ++command) {
            if (has_queue_encoding_mode(header.version)) {
                if (mode == QueueEncodingMode::Native)
                    skip_native_queue_command(reader, queueKind);
                else if (mode == QueueEncodingMode::MacroSchedule)
                    skip_macro_queue_command(reader,
                        (macroWidths[command / 8]
                            & (1u << (command % 8))) != 0);
                else
                    skip_compact_queue_command(reader);
                continue;
            }
            if (header.version >= 25) {
                skip_compact_queue_command(reader);
                continue;
            }
            (void)reader.read<std::uint32_t>();
            (void)reader.read<std::uint16_t>();
            (void)reader.read<std::uint16_t>();
            reader.skip((header.version >= 3 ? 4u : 3u)
                * sizeof(std::uint32_t));
            if (header.version >= 4) {
                const auto extension_count =
                    reader.read<std::uint16_t>();
                reader.skip(
                    static_cast<std::uint64_t>(extension_count)
                    * sizeof(std::uint32_t));
            }
        }
    }
    reader.skip(static_cast<std::uint64_t>(header.relocation_count)
        * (sizeof(std::uint32_t) * 3 + sizeof(std::uint16_t) * 3));
    const std::uint64_t address_relocation_size =
        sizeof(std::uint32_t) * 2 + sizeof(std::uint16_t) * 2
        + (header.version >= 10 ? sizeof(std::uint16_t) : 0)
        + (header.version >= 14 ? sizeof(std::uint16_t) : 0)
        + (header.version >= 37 ? sizeof(std::uint16_t) : 0);
    reader.skip(
        static_cast<std::uint64_t>(header.address_relocation_count)
        * address_relocation_size);
    if (!reader.empty())
        throw std::runtime_error("FTLPU binary has trailing data");
    return std::move(header.program);
}

void write_binary_program(
    const BinaryProgram& program, const std::filesystem::path& path)
{
    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("failed to open FTLPU binary for writing");
    write_binary_program(program, os);
}

BinaryProgram read_binary_program(const std::filesystem::path& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("failed to open FTLPU binary for reading");
    return read_binary_program(is);
}

} // namespace ftlpu::software::runtime
