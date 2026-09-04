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

namespace ftlpu::software::runtime {

namespace {

constexpr std::array<char, 8> kMagic {'F', 'T', 'L', 'P', 'U', 'B', '0', '1'};
constexpr std::uint8_t kCompactInstructionKindMask = 0x07;
constexpr std::uint8_t kCompactWordCountMask = 0x38;
constexpr std::uint8_t kCompactWordCountShift = 3;
constexpr std::uint8_t kCompactHasExtension = 0x40;
constexpr std::uint8_t kCompactMacro = 0x80;

enum class QueueEncodingMode : std::uint8_t {
    Native = 0,
    MacroSchedule = 1,
    CompactTagged = 2,
    MemMacroDeltaRle = 3,
};

constexpr bool has_queue_encoding_mode(std::uint32_t version)
{
    // Version 26 on origin/main added VXM capability fields but retained the
    // v25 compact queue layout. Local experimental v27/v28 binaries introduced
    // queue-level encoding modes; v29 is the first merged format.
    return version == 27 || version == 28 || version >= 29;
}

constexpr bool has_mem_macro_delta_rle(std::uint32_t version)
{
    return version == 28 || version >= 29;
}

constexpr std::uint32_t kNativeWordCountShift = 2;
constexpr std::uint32_t kNativeWordCountMask = 0x1cu;
constexpr std::uint32_t kNativeSxmFixedWordCount = 7;
constexpr std::uint32_t kIcuLoopWindowMask = 0x3fu;
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
    // then repack it so word zero is a self-describing extended Loop record.
    (void)isa::encode_icu_repeat_2d(repeat);
    std::array<std::uint32_t, 3> words {};
    const auto write = [&](std::size_t offset, std::size_t width,
                           std::uint64_t value) {
        for (std::size_t bit = 0; bit < width; ++bit)
            if ((value & (std::uint64_t {1} << bit)) != 0)
                words[(offset + bit) / 32]
                    |= std::uint32_t {1} << ((offset + bit) % 32);
    };
    write(0, 2, static_cast<std::uint8_t>(isa::IcuCommandOpcode::Loop));
    // bits 2..7 are the otherwise-invalid Loop window_size=0 escape.
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
            != isa::IcuCommandOpcode::Loop
        || ((words[0] >> 2) & kIcuLoopWindowMask) != 0
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
            > static_cast<std::uint16_t>(InstructionKind::MxmDequant)
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
            > static_cast<std::uint16_t>(InstructionKind::MxmDequant)
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

QueueEncodingMode queue_encoding_mode(const QueueProgram& queue)
{
    const auto macroCount = std::count_if(queue.commands.begin(),
        queue.commands.end(), [](const QueueCommand& command) {
            return is_macro_schedule_command(command);
        });
    if (macroCount == 0) {
        const bool hasExtendedDescriptor = std::any_of(
            queue.commands.begin(), queue.commands.end(),
            [](const QueueCommand& command) {
                return isa::decode_icu_command_opcode(command.command)
                    == isa::IcuCommandOpcode::Extended;
            });
        return hasExtendedDescriptor
            ? QueueEncodingMode::CompactTagged
            : QueueEncodingMode::Native;
    }
    if (macroCount == static_cast<std::ptrdiff_t>(queue.commands.size())) {
        if (queue.kind == QueueKind::Mem)
            return QueueEncodingMode::MemMacroDeltaRle;
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
    os.write(binding.role.data(),
        static_cast<std::streamsize>(binding.role.size()));
    os.write(binding.name.data(),
        static_cast<std::streamsize>(binding.name.size()));
    if (!os) throw std::runtime_error("failed to write binding metadata");
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
    if (opcode == isa::IcuCommandOpcode::Loop
        && ((prefix >> 2) & kIcuLoopWindowMask) == 0) {
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
    if (opcode == isa::IcuCommandOpcode::Loop
        && ((prefix >> 2) & kIcuLoopWindowMask) == 0) {
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
                    QueueEncodingMode::MemMacroDeltaRle))
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
    return decode_mem_macro_bitstream(image, queueIndex);
}

template <typename Reader>
void skip_mem_macro_bitstream(Reader& reader)
{
    (void)read_value<std::uint8_t>(reader);
    const auto deltaCount = read_value<std::uint8_t>(reader);
    if (deltaCount > kMemMacroDeltaDictionarySize)
        throw std::runtime_error("invalid MEM Macro delta dictionary size");
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
            }
        } else if (binding.layout == BindingLayout::Fp32CausalMaskTile
                   || binding.layout == BindingLayout::Fp16CausalMaskTile) {
            binding.initializer = BindingInitializer::CausalMask;
        }
        binding.role.resize(role_size);
        binding.name.resize(name_size);
        reader.read_bytes(binding.role.data(), binding.role.size());
        reader.read_bytes(binding.name.data(), binding.name.size());
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

struct BinaryHeader {
    BinaryProgram program;
    std::uint32_t version;
    std::uint32_t queue_count;
    std::uint32_t relocation_count;
    std::uint32_t address_relocation_count;
    std::uint32_t weight_page_use_count;
};

template <typename ReadValue>
void read_hardware_config(BinaryProgram& program, std::uint32_t version,
    ReadValue&& read_value)
{
    auto read = [&](std::uint32_t& value) { value = read_value(); };
    bool upgradeTargetAbi = false;
    if (version >= 29) {
        program.hardware.visit(read);
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
        (void)reader.read<std::uint16_t>();
        use.ready_cycle = reader.read<std::uint64_t>();
        use.release_cycle = reader.read<std::uint64_t>();
        program.weight_page_uses.push_back(use);
    }
    return {std::move(program), version, queue_count, relocation_count,
        address_relocation_count, weight_page_use_count};
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

void write_binary_program(const BinaryProgram& program, std::ostream& os)
{
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
    program.hardware.visit([&](std::uint32_t value) {
        write_scalar<std::uint32_t>(os, value);
    });
    write_scalar<std::uint64_t>(os, static_cast<std::uint64_t>(program.max_cycle));
    write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(program.queues.size()));
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
        write_scalar<std::uint16_t>(os, 0);
        write_scalar<std::uint64_t>(os, use.ready_cycle);
        write_scalar<std::uint64_t>(os, use.release_cycle);
    }

    for (const auto& queue : program.queues) {
        write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(queue.kind));
        write_scalar<std::uint16_t>(os, static_cast<std::uint16_t>(queue.index));
        const auto mode = queue_encoding_mode(queue);
        write_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(mode));
        write_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(queue.commands.size()));
        if (mode == QueueEncodingMode::MemMacroDeltaRle) {
            write_mem_macro_bitstream(os, queue);
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
        (void)read_scalar<std::uint16_t>(is);
        use.ready_cycle = read_scalar<std::uint64_t>(is);
        use.release_cycle = read_scalar<std::uint64_t>(is);
        program.weight_page_uses.push_back(use);
    }
    program.queues.reserve(queue_count);

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

        if (has_mem_macro_delta_rle(version)
            && mode == QueueEncodingMode::MemMacroDeltaRle) {
            program.queues.push_back(read_mem_macro_bitstream(
                is, command_count, queue.index));
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
        program.queues.push_back(std::move(queue));
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
        program.address_relocations.push_back(relocation);
    }

    return program;
}

BinaryProgram read_binary_program_metadata(std::istream& is)
{
    std::array<char, 8> magic {};
    is.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!is || magic != kMagic)
        throw std::runtime_error("invalid FTLPU binary magic");

    const auto version = read_scalar<std::uint32_t>(is);
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
    for (std::uint32_t index = 0;
         index < weight_page_use_count; ++index) {
        (void)read_scalar<std::uint32_t>(is);
        (void)read_scalar<std::uint32_t>(is);
        (void)read_scalar<std::uint16_t>(is);
        (void)read_scalar<std::uint16_t>(is);
        (void)read_scalar<std::uint64_t>(is);
        (void)read_scalar<std::uint64_t>(is);
    }

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
        if (has_mem_macro_delta_rle(version)
            && mode == QueueEncodingMode::MemMacroDeltaRle) {
            skip_mem_macro_bitstream(is);
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
        + (version >= 14 ? sizeof(std::uint16_t) : 0);
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
        if (has_mem_macro_delta_rle(header.version)
            && mode == QueueEncodingMode::MemMacroDeltaRle) {
            program.queues.push_back(read_mem_macro_bitstream(
                reader, command_count, queue.index));
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
        program.queues.push_back(std::move(queue));
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
        program.address_relocations.push_back(relocation);
    }
    if (!reader.empty())
        throw std::runtime_error("FTLPU binary has trailing data");
    return std::move(program);
}

BinaryProgram read_binary_program_metadata(
    std::span<const std::uint8_t> data)
{
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
        if (has_mem_macro_delta_rle(header.version)
            && mode == QueueEncodingMode::MemMacroDeltaRle) {
            skip_mem_macro_bitstream(reader);
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
        + (header.version >= 14 ? sizeof(std::uint16_t) : 0);
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
