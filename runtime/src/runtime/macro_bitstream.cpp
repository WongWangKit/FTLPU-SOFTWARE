#include "ftlpu/software/runtime/macro_bitstream.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ftlpu::software::runtime {
namespace {

constexpr unsigned kCompactStartDeltaBits = 22;
constexpr unsigned kCompactAddressDeltaBits = 14;
constexpr unsigned kMemAddressBits = 13;

class BitWriter {
  public:
    void bit(bool value)
    {
        if ((bits_ & 7u) == 0)
            bytes_.push_back(0);
        if (value)
            bytes_.back() |= static_cast<std::uint8_t>(1u << (bits_ & 7u));
        ++bits_;
    }

    void bits(std::uint64_t value, unsigned width)
    {
        for (unsigned i = 0; i < width; ++i)
            bit(((value >> i) & 1u) != 0);
    }

    std::vector<std::uint8_t> take() { return std::move(bytes_); }
    std::uint64_t size() const { return bits_; }

  private:
    std::vector<std::uint8_t> bytes_;
    std::uint64_t bits_{0};
};

class BitReader {
  public:
    BitReader(const std::vector<std::uint8_t>& bytes, std::uint64_t bits)
        : bytes_(bytes), bits_(bits)
    {
        if (bits > static_cast<std::uint64_t>(bytes.size()) * 8)
            throw std::invalid_argument(
                "MEM Macro bit count exceeds byte payload");
    }

    bool bit()
    {
        require(1);
        const bool result =
            ((bytes_[position_ / 8] >> (position_ & 7u)) & 1u) != 0;
        ++position_;
        return result;
    }

    std::uint64_t bits(unsigned width)
    {
        if (width > 64)
            throw std::invalid_argument("bit field is wider than 64 bits");
        require(width);
        std::uint64_t result = 0;
        for (unsigned i = 0; i < width; ++i)
            if (bit())
                result |= std::uint64_t{1} << i;
        return result;
    }

    std::uint64_t remaining() const { return bits_ - position_; }

  private:
    void require(std::uint64_t count) const
    {
        if (count > bits_ - position_)
            throw std::runtime_error("truncated MEM Macro bitstream");
    }

    const std::vector<std::uint8_t>& bytes_;
    std::uint64_t bits_{0};
    std::uint64_t position_{0};
};

std::uint64_t unsigned_limit(unsigned width)
{
    return width == 64 ? std::numeric_limits<std::uint64_t>::max()
                       : ((std::uint64_t{1} << width) - 1);
}

bool fits_unsigned(std::uint64_t value, unsigned width)
{
    return value <= unsigned_limit(width);
}

bool fits_signed(std::int64_t value, unsigned width)
{
    const auto limit = std::int64_t{1} << (width - 1);
    return value >= -limit && value < limit;
}

void write_signed(BitWriter& writer, std::int64_t value, unsigned width)
{
    if (!fits_signed(value, width))
        throw std::invalid_argument("signed MEM Macro field does not fit");
    writer.bits(static_cast<std::uint64_t>(value) & unsigned_limit(width),
                width);
}

std::int64_t read_signed(BitReader& reader, unsigned width)
{
    const auto raw = reader.bits(width);
    const auto sign = std::uint64_t{1} << (width - 1);
    if ((raw & sign) == 0)
        return static_cast<std::int64_t>(raw);
    return static_cast<std::int64_t>(raw | ~unsigned_limit(width));
}

std::uint64_t instruction_word(const QueueCommand& command)
{
    return command.words[0] |
           (static_cast<std::uint64_t>(command.words[1]) << 32);
}

struct MacroRecord {
    MemInstruction instruction{};
    IcuMacroSchedule schedule{};
};

MacroRecord decode_record(const QueueCommand& command)
{
    if (!is_macro_schedule_command(command) ||
        command.instruction_kind != InstructionKind::Mem ||
        command.word_count == 0 || command.word_count > 2)
        throw std::invalid_argument(
            "MEM Macro bitstream requires all-Macro MEM commands");
    return {isa::decode_mem_instruction(instruction_word(command)),
            decode_macro_schedule_command(command)};
}

enum class Shape : std::uint8_t {
    Single = 0,
    Inner1D = 1,
    Outer1D = 2,
    Full2D = 3
};

Shape shape_of(const IcuMacroSchedule& schedule)
{
    if (schedule.inner_count == 1 && schedule.outer_count == 1)
        return Shape::Single;
    if (schedule.outer_count == 1)
        return Shape::Inner1D;
    if (schedule.inner_count == 1)
        return Shape::Outer1D;
    return Shape::Full2D;
}

bool same_template(const MacroRecord& a, const MacroRecord& b)
{
    const auto shape = shape_of(a.schedule);
    if (shape != shape_of(b.schedule) ||
        a.instruction.opcode != b.instruction.opcode ||
        a.instruction.stream != b.instruction.stream ||
        a.instruction.map_stream != b.instruction.map_stream ||
        a.instruction.preserve_stream != b.instruction.preserve_stream ||
        a.schedule.induction_target != b.schedule.induction_target)
        return false;
    if ((shape == Shape::Inner1D || shape == Shape::Full2D) &&
        (a.schedule.inner_count != b.schedule.inner_count ||
         a.schedule.inner_interval != b.schedule.inner_interval ||
         a.schedule.inner_stride != b.schedule.inner_stride))
        return false;
    return (shape != Shape::Outer1D && shape != Shape::Full2D) ||
           (a.schedule.outer_count == b.schedule.outer_count &&
            a.schedule.outer_interval == b.schedule.outer_interval &&
            a.schedule.outer_stride == b.schedule.outer_stride);
}

bool compact_template(const MacroRecord& record)
{
    const auto& instruction = record.instruction;
    const auto& schedule = record.schedule;
    if ((instruction.opcode != MemOpcode::Read &&
         instruction.opcode != MemOpcode::Write) ||
        instruction.stream > 63 || instruction.map_stream != 0 ||
        schedule.induction_target != IcuInductionTarget::MemAddress)
        return false;
    const auto shape = shape_of(schedule);
    if ((shape == Shape::Inner1D || shape == Shape::Full2D) &&
        (!fits_unsigned(schedule.inner_count - 1, 11) ||
         !fits_unsigned(schedule.inner_interval - 1, 8) ||
         !fits_signed(schedule.inner_stride, 10)))
        return false;
    if (shape == Shape::Outer1D || shape == Shape::Full2D) {
        const auto innerSpan =
            (schedule.inner_count - 1) * schedule.inner_interval;
        if (schedule.outer_interval <= innerSpan)
            return false;
        const auto residualMinusOne = schedule.outer_interval - innerSpan - 1;
        if (!fits_unsigned(schedule.outer_count - 1, 9) ||
            !fits_unsigned(residualMinusOne, 15) ||
            !fits_signed(schedule.outer_stride, 12))
            return false;
    }
    return true;
}

void write_template(BitWriter& writer, const MacroRecord& record,
                    MemMacroBitstreamStats& stats)
{
    if (!compact_template(record)) {
        writer.bit(true);
        const auto instruction =
            isa::encode_mem_instruction(record.instruction);
        writer.bits(instruction, 32);
        writer.bits(record.schedule.inner_count, 32);
        writer.bits(record.schedule.inner_interval, 32);
        writer.bits(static_cast<std::uint32_t>(record.schedule.inner_stride),
                    32);
        writer.bits(record.schedule.outer_count, 32);
        writer.bits(record.schedule.outer_interval, 32);
        writer.bits(static_cast<std::uint32_t>(record.schedule.outer_stride),
                    32);
        writer.bits(static_cast<std::uint8_t>(record.schedule.induction_target),
                    2);
        ++stats.extended_template_runs;
        return;
    }

    writer.bit(false);
    const auto shape = shape_of(record.schedule);
    writer.bits(static_cast<std::uint8_t>(shape), 2);
    writer.bit(record.instruction.opcode == MemOpcode::Write);
    writer.bits(record.instruction.stream, 6);
    writer.bit(record.instruction.preserve_stream);
    if (shape == Shape::Inner1D || shape == Shape::Full2D) {
        writer.bits(record.schedule.inner_count - 1, 11);
        writer.bits(record.schedule.inner_interval - 1, 8);
        write_signed(writer, record.schedule.inner_stride, 10);
    }
    if (shape == Shape::Outer1D || shape == Shape::Full2D) {
        const auto innerSpan =
            (record.schedule.inner_count - 1) * record.schedule.inner_interval;
        writer.bits(record.schedule.outer_count - 1, 9);
        writer.bits(record.schedule.outer_interval - innerSpan - 1, 15);
        write_signed(writer, record.schedule.outer_stride, 12);
    }
    ++stats.compact_template_runs;
}

MacroRecord read_template(BitReader& reader)
{
    MacroRecord result;
    if (reader.bit()) {
        result.instruction = isa::decode_mem_instruction(reader.bits(32));
        result.schedule.inner_count = reader.bits(32);
        result.schedule.inner_interval = reader.bits(32);
        result.schedule.inner_stride =
            static_cast<std::int32_t>(reader.bits(32));
        result.schedule.outer_count = reader.bits(32);
        result.schedule.outer_interval = reader.bits(32);
        result.schedule.outer_stride =
            static_cast<std::int32_t>(reader.bits(32));
        result.schedule.induction_target =
            static_cast<IcuInductionTarget>(reader.bits(2));
        return result;
    }

    const auto shape = static_cast<Shape>(reader.bits(2));
    const auto opcode = reader.bit() ? MemOpcode::Write : MemOpcode::Read;
    const auto stream = static_cast<std::size_t>(reader.bits(6));
    const bool preserve = reader.bit();
    result.instruction = opcode == MemOpcode::Write
                             ? (preserve ? MemInstruction::WriteTap(0, stream)
                                         : MemInstruction::Write(0, stream))
                             : MemInstruction::Read(0, stream);
    result.schedule =
        IcuMacroSchedule{0, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress};
    if (shape == Shape::Inner1D || shape == Shape::Full2D) {
        result.schedule.inner_count = reader.bits(11) + 1;
        result.schedule.inner_interval = reader.bits(8) + 1;
        result.schedule.inner_stride = read_signed(reader, 10);
    }
    if (shape == Shape::Outer1D || shape == Shape::Full2D) {
        result.schedule.outer_count = reader.bits(9) + 1;
        const auto residual = reader.bits(15) + 1;
        result.schedule.outer_interval =
            residual +
            (result.schedule.inner_count - 1) * result.schedule.inner_interval;
        result.schedule.outer_stride = read_signed(reader, 12);
    }
    return result;
}

void write_prefix(BitWriter& writer, std::size_t index)
{
    static constexpr std::array<const char*, 7> codes{
        "0", "10", "1100", "1101", "11100", "11101", "11110"};
    for (const char* p = codes[index]; *p != '\0'; ++p)
        writer.bit(*p == '1');
}

std::size_t read_prefix(BitReader& reader)
{
    if (!reader.bit())
        return 0;
    if (!reader.bit())
        return 1;
    if (!reader.bit())
        return reader.bit() ? 3 : 2;
    if (!reader.bit())
        return reader.bit() ? 5 : 4;
    return reader.bit() ? 7 : 6; // 7 is the escape symbol.
}

bool compact_delta(const MemMacroDelta& delta)
{
    return fits_unsigned(delta.start_cycle, kCompactStartDeltaBits) &&
           fits_signed(delta.address, kCompactAddressDeltaBits);
}

void write_delta(BitWriter& writer, const MemMacroDelta& delta,
                 const MemMacroBitstream& image, MemMacroBitstreamStats& stats)
{
    for (std::size_t i = 0; i < image.delta_count; ++i) {
        if (image.deltas[i].start_cycle == delta.start_cycle &&
            image.deltas[i].address == delta.address) {
            write_prefix(writer, i);
            ++stats.dictionary_transitions;
            return;
        }
    }
    writer.bits(0x1f, 5); // 11111 escape
    ++stats.escaped_transitions;
    if (compact_delta(delta)) {
        writer.bit(false);
        writer.bits(delta.start_cycle, kCompactStartDeltaBits);
        write_signed(writer, delta.address, kCompactAddressDeltaBits);
    } else {
        writer.bit(true);
        writer.bits(delta.start_cycle, 32);
        writer.bits(static_cast<std::uint32_t>(delta.address), 32);
        ++stats.wide_escaped_transitions;
    }
}

MemMacroDelta read_delta(BitReader& reader, const MemMacroBitstream& image)
{
    const auto symbol = read_prefix(reader);
    if (symbol < image.delta_count)
        return image.deltas[symbol];
    if (symbol != 7)
        throw std::runtime_error(
            "MEM Macro delta code references a missing dictionary entry");
    if (!reader.bit())
        return {static_cast<std::uint32_t>(reader.bits(kCompactStartDeltaBits)),
                static_cast<std::int32_t>(
                    read_signed(reader, kCompactAddressDeltaBits))};
    return {static_cast<std::uint32_t>(reader.bits(32)),
            static_cast<std::int32_t>(reader.bits(32))};
}

QueueCommand make_command(const MacroRecord& record)
{
    auto instruction = record.instruction;
    const auto encoded = isa::encode_mem_instruction(instruction);
    QueueCommand command{
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mem,
        static_cast<std::uint16_t>((encoded >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(encoded),
         static_cast<std::uint32_t>(encoded >> 32), 0, 0},
    };
    return encode_macro_schedule_command(std::move(command), record.schedule);
}

struct MxmMacroRecord {
    std::uint64_t instruction{0};
    IcuMacroSchedule schedule{};
};

bool is_mxm_macro_kind(QueueKind kind)
{
    return kind == QueueKind::MxmLoad || kind == QueueKind::MxmCompute ||
           kind == QueueKind::MxmDequant;
}

unsigned mxm_instruction_bits(QueueKind kind)
{
    switch (kind) {
    case QueueKind::MxmLoad:
    case QueueKind::MxmDequant:
        return 16;
    case QueueKind::MxmCompute:
        return 49;
    default:
        throw std::invalid_argument(
            "MXM Macro bitstream requires an MXM queue");
    }
}

unsigned mxm_operand_bits(QueueKind kind)
{
    switch (kind) {
    case QueueKind::MxmLoad:
        return 2;
    case QueueKind::MxmCompute:
        return 13;
    case QueueKind::MxmDequant:
        return 0;
    default:
        throw std::invalid_argument(
            "MXM Macro bitstream requires an MXM queue");
    }
}

MxmMacroRecord decode_mxm_record(const QueueCommand& command, QueueKind kind)
{
    if (!is_macro_schedule_command(command))
        throw std::invalid_argument(
            "MXM Macro bitstream requires all-Macro commands");
    if (kind == QueueKind::MxmDequant) {
        if (command.instruction_kind != InstructionKind::MxmDequant ||
            command.word_count != 1)
            throw std::invalid_argument(
                "MXM dequant Macro carries an invalid instruction");
        return {static_cast<std::uint16_t>(command.words[0]),
                decode_macro_schedule_command(command)};
    }
    if ((kind != QueueKind::MxmLoad && kind != QueueKind::MxmCompute) ||
        command.instruction_kind != InstructionKind::Mxm ||
        command.word_count == 0 || command.word_count > 2)
        throw std::invalid_argument(
            "MXM Macro carries an invalid control instruction");
    return {command.words[0] |
                (static_cast<std::uint64_t>(command.words[1]) << 32),
            decode_macro_schedule_command(command)};
}

std::uint32_t mxm_operand(const MxmMacroRecord& record, QueueKind kind)
{
    if (kind == QueueKind::MxmDequant)
        return 0;
    const auto instruction = isa::decode_mxm_instruction(record.instruction);
    return static_cast<std::uint32_t>(kind == QueueKind::MxmLoad
                                          ? instruction.weight_column
                                          : instruction.accumulator_address);
}

std::uint64_t normalized_mxm_instruction(const MxmMacroRecord& record,
                                         QueueKind kind)
{
    if (kind == QueueKind::MxmDequant)
        return record.instruction;
    auto instruction = isa::decode_mxm_instruction(record.instruction);
    if (kind == QueueKind::MxmLoad)
        instruction.weight_column = 0;
    else
        instruction.accumulator_address = 0;
    return isa::encode_mxm_instruction(instruction);
}

void set_mxm_operand(MxmMacroRecord& record, QueueKind kind,
                     std::uint32_t operand)
{
    if (kind == QueueKind::MxmDequant)
        return;
    auto instruction = isa::decode_mxm_instruction(record.instruction);
    if (kind == QueueKind::MxmLoad)
        instruction.weight_column = operand;
    else
        instruction.accumulator_address = operand;
    record.instruction = isa::encode_mxm_instruction(instruction);
}

void validate_mxm_record(const MxmMacroRecord& record, QueueKind kind)
{
    if (kind == QueueKind::MxmDequant) {
        if (record.schedule.induction_target != IcuInductionTarget::None ||
            record.schedule.inner_stride != 0 ||
            record.schedule.outer_stride != 0)
            throw std::invalid_argument(
                "MXM dequant Macro cannot induce an operand");
        return;
    }
    const auto instruction = isa::decode_mxm_instruction(record.instruction);
    if (kind == QueueKind::MxmLoad) {
        if (instruction.opcode != MxmControlOpcode::IW ||
            (record.schedule.induction_target != IcuInductionTarget::None &&
             record.schedule.induction_target !=
                 IcuInductionTarget::MxmWeightColumn))
            throw std::invalid_argument(
                "MXM load Macro has an invalid opcode or induction target");
    } else {
        if (instruction.opcode == MxmControlOpcode::IW ||
            (record.schedule.induction_target != IcuInductionTarget::None &&
             record.schedule.induction_target !=
                 IcuInductionTarget::MxmAccumulatorAddress))
            throw std::invalid_argument(
                "MXM compute Macro has an invalid opcode or induction target");
    }
}

bool same_mxm_template(const MxmMacroRecord& a, const MxmMacroRecord& b,
                       QueueKind kind)
{
    const auto shape = shape_of(a.schedule);
    if (shape != shape_of(b.schedule) ||
        normalized_mxm_instruction(a, kind) !=
            normalized_mxm_instruction(b, kind) ||
        a.schedule.induction_target != b.schedule.induction_target)
        return false;
    if ((shape == Shape::Inner1D || shape == Shape::Full2D) &&
        (a.schedule.inner_count != b.schedule.inner_count ||
         a.schedule.inner_interval != b.schedule.inner_interval ||
         a.schedule.inner_stride != b.schedule.inner_stride))
        return false;
    return (shape != Shape::Outer1D && shape != Shape::Full2D) ||
           (a.schedule.outer_count == b.schedule.outer_count &&
            a.schedule.outer_interval == b.schedule.outer_interval &&
            a.schedule.outer_stride == b.schedule.outer_stride);
}

bool compact_mxm_template(const MxmMacroRecord& record)
{
    const auto shape = shape_of(record.schedule);
    if ((shape == Shape::Inner1D || shape == Shape::Full2D) &&
        (!fits_unsigned(record.schedule.inner_count - 1, 11) ||
         !fits_unsigned(record.schedule.inner_interval - 1, 8) ||
         !fits_signed(record.schedule.inner_stride, 10)))
        return false;
    if (shape == Shape::Outer1D || shape == Shape::Full2D) {
        const auto innerSpan =
            (record.schedule.inner_count - 1) * record.schedule.inner_interval;
        if (record.schedule.outer_interval <= innerSpan)
            return false;
        return fits_unsigned(record.schedule.outer_count - 1, 9) &&
               fits_unsigned(record.schedule.outer_interval - innerSpan - 1,
                             15) &&
               fits_signed(record.schedule.outer_stride, 12);
    }
    return true;
}

void write_mxm_template(BitWriter& writer, const MxmMacroRecord& record,
                        QueueKind kind, MxmMacroBitstreamStats& stats)
{
    const auto native = normalized_mxm_instruction(record, kind);
    if (!compact_mxm_template(record)) {
        writer.bit(true);
        writer.bits(native, mxm_instruction_bits(kind));
        writer.bits(record.schedule.inner_count, 32);
        writer.bits(record.schedule.inner_interval, 32);
        writer.bits(static_cast<std::uint32_t>(record.schedule.inner_stride),
                    32);
        writer.bits(record.schedule.outer_count, 32);
        writer.bits(record.schedule.outer_interval, 32);
        writer.bits(static_cast<std::uint32_t>(record.schedule.outer_stride),
                    32);
        writer.bits(static_cast<std::uint8_t>(record.schedule.induction_target),
                    2);
        ++stats.extended_template_runs;
        return;
    }
    writer.bit(false);
    writer.bits(native, mxm_instruction_bits(kind));
    const auto shape = shape_of(record.schedule);
    writer.bits(static_cast<std::uint8_t>(shape), 2);
    writer.bits(static_cast<std::uint8_t>(record.schedule.induction_target), 2);
    if (shape == Shape::Inner1D || shape == Shape::Full2D) {
        writer.bits(record.schedule.inner_count - 1, 11);
        writer.bits(record.schedule.inner_interval - 1, 8);
        write_signed(writer, record.schedule.inner_stride, 10);
    }
    if (shape == Shape::Outer1D || shape == Shape::Full2D) {
        const auto innerSpan =
            (record.schedule.inner_count - 1) * record.schedule.inner_interval;
        writer.bits(record.schedule.outer_count - 1, 9);
        writer.bits(record.schedule.outer_interval - innerSpan - 1, 15);
        write_signed(writer, record.schedule.outer_stride, 12);
    }
    ++stats.compact_template_runs;
}

MxmMacroRecord read_mxm_template(BitReader& reader, QueueKind kind)
{
    MxmMacroRecord result;
    const bool extended = reader.bit();
    result.instruction = reader.bits(mxm_instruction_bits(kind));
    if (extended) {
        result.schedule.inner_count = reader.bits(32);
        result.schedule.inner_interval = reader.bits(32);
        result.schedule.inner_stride =
            static_cast<std::int32_t>(reader.bits(32));
        result.schedule.outer_count = reader.bits(32);
        result.schedule.outer_interval = reader.bits(32);
        result.schedule.outer_stride =
            static_cast<std::int32_t>(reader.bits(32));
        result.schedule.induction_target =
            static_cast<IcuInductionTarget>(reader.bits(2));
        return result;
    }
    const auto shape = static_cast<Shape>(reader.bits(2));
    result.schedule = IcuMacroSchedule{
        0, 1, 1, 0, 1, 1, 0, static_cast<IcuInductionTarget>(reader.bits(2))};
    if (shape == Shape::Inner1D || shape == Shape::Full2D) {
        result.schedule.inner_count = reader.bits(11) + 1;
        result.schedule.inner_interval = reader.bits(8) + 1;
        result.schedule.inner_stride = read_signed(reader, 10);
    }
    if (shape == Shape::Outer1D || shape == Shape::Full2D) {
        result.schedule.outer_count = reader.bits(9) + 1;
        const auto residual = reader.bits(15) + 1;
        result.schedule.outer_interval =
            residual +
            (result.schedule.inner_count - 1) * result.schedule.inner_interval;
        result.schedule.outer_stride = read_signed(reader, 12);
    }
    return result;
}

bool compact_mxm_delta(const MxmMacroDelta& delta)
{
    return fits_unsigned(delta.start_cycle, kCompactStartDeltaBits) &&
           fits_signed(delta.operand, kCompactAddressDeltaBits);
}

void write_mxm_delta(BitWriter& writer, const MxmMacroDelta& delta,
                     const MxmMacroBitstream& image,
                     MxmMacroBitstreamStats& stats)
{
    for (std::size_t i = 0; i < image.delta_count; ++i) {
        if (image.deltas[i].start_cycle == delta.start_cycle &&
            image.deltas[i].operand == delta.operand) {
            write_prefix(writer, i);
            ++stats.dictionary_transitions;
            return;
        }
    }
    writer.bits(0x1f, 5);
    ++stats.escaped_transitions;
    if (compact_mxm_delta(delta)) {
        writer.bit(false);
        writer.bits(delta.start_cycle, kCompactStartDeltaBits);
        write_signed(writer, delta.operand, kCompactAddressDeltaBits);
    } else {
        writer.bit(true);
        writer.bits(delta.start_cycle, 32);
        writer.bits(static_cast<std::uint32_t>(delta.operand), 32);
        ++stats.wide_escaped_transitions;
    }
}

MxmMacroDelta read_mxm_delta(BitReader& reader, const MxmMacroBitstream& image)
{
    const auto symbol = read_prefix(reader);
    if (symbol < image.delta_count)
        return image.deltas[symbol];
    if (symbol != 7)
        throw std::runtime_error(
            "MXM Macro delta code references a missing dictionary entry");
    if (!reader.bit())
        return {static_cast<std::uint32_t>(reader.bits(kCompactStartDeltaBits)),
                static_cast<std::int32_t>(
                    read_signed(reader, kCompactAddressDeltaBits))};
    return {static_cast<std::uint32_t>(reader.bits(32)),
            static_cast<std::int32_t>(reader.bits(32))};
}

QueueCommand make_mxm_command(const MxmMacroRecord& record, QueueKind kind)
{
    if (kind == QueueKind::MxmDequant) {
        QueueCommand command{
            static_cast<isa::EncodedIcuCommand>(
                isa::IcuCommandOpcode::Instruction),
            InstructionKind::MxmDequant,
            1,
            {static_cast<std::uint32_t>(record.instruction), 0, 0, 0},
        };
        return encode_macro_schedule_command(std::move(command),
                                             record.schedule);
    }
    QueueCommand command{
        static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        InstructionKind::Mxm,
        static_cast<std::uint16_t>((record.instruction >> 32) == 0 ? 1 : 2),
        {static_cast<std::uint32_t>(record.instruction),
         static_cast<std::uint32_t>(record.instruction >> 32), 0, 0},
    };
    return encode_macro_schedule_command(std::move(command), record.schedule);
}

} // namespace

std::uint32_t encode_icu_queue_control(const IcuQueueControl& control)
{
    if (control.command_count > kIcuQueueControlMaxCommandCount)
        throw std::invalid_argument(
            "ICU queue command count does not fit the 24-bit control field");
    const auto mode = static_cast<std::uint32_t>(control.mode);
    if (mode >= (std::uint32_t{1} << kIcuQueueControlModeBits))
        throw std::invalid_argument("ICU queue mode does not fit the control word");
    return (control.valid ? std::uint32_t{1} << kIcuQueueControlValidBit : 0) |
           (control.enable ? std::uint32_t{1} << kIcuQueueControlEnableBit : 0) |
           (mode << kIcuQueueControlModeShift) |
           (control.command_count << kIcuQueueControlCommandCountShift);
}

IcuQueueControl decode_icu_queue_control(std::uint32_t word)
{
    constexpr auto usedBits = kIcuQueueControlCommandCountShift +
                              kIcuQueueControlCommandCountBits;
    constexpr auto reservedMask = ~((std::uint32_t{1} << usedBits) - 1);
    if ((word & reservedMask) != 0)
        throw std::invalid_argument(
            "ICU queue control word has non-zero reserved bits");
    return {
        (word & (std::uint32_t{1} << kIcuQueueControlValidBit)) != 0,
        (word & (std::uint32_t{1} << kIcuQueueControlEnableBit)) != 0,
        static_cast<IcuQueueMode>(
            (word >> kIcuQueueControlModeShift) &
            ((std::uint32_t{1} << kIcuQueueControlModeBits) - 1)),
        (word >> kIcuQueueControlCommandCountShift) &
            kIcuQueueControlMaxCommandCount,
    };
}

MemMacroBitstream encode_mem_macro_bitstream(const QueueProgram& queue)
{
    if (queue.kind != QueueKind::Mem)
        throw std::invalid_argument("MEM Macro bitstream requires a MEM queue");
    if (queue.commands.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("MEM Macro queue is too large");

    std::vector<MacroRecord> records;
    records.reserve(queue.commands.size());
    for (const auto& command : queue.commands)
        records.push_back(decode_record(command));

    MemMacroBitstream image;
    image.command_count = static_cast<std::uint32_t>(records.size());
    image.stats.command_count = records.size();
    if (records.empty())
        return image;

    std::map<std::pair<std::uint32_t, std::int32_t>, std::size_t> frequencies;
    for (std::size_t i = 1; i < records.size(); ++i) {
        if (records[i].schedule.start_cycle <
            records[i - 1].schedule.start_cycle)
            throw std::invalid_argument(
                "MEM Macro start cycles are not monotonic");
        const auto startDelta = records[i].schedule.start_cycle -
                                records[i - 1].schedule.start_cycle;
        const auto addressDelta =
            static_cast<std::int64_t>(records[i].instruction.address) -
            static_cast<std::int64_t>(records[i - 1].instruction.address);
        if (startDelta > std::numeric_limits<std::uint32_t>::max() ||
            !fits_signed(addressDelta, 32))
            throw std::invalid_argument(
                "MEM Macro transition exceeds the wide escape");
        MemMacroDelta delta{static_cast<std::uint32_t>(startDelta),
                            static_cast<std::int32_t>(addressDelta)};
        if (compact_delta(delta))
            ++frequencies[{delta.start_cycle, delta.address}];
    }
    std::vector<std::pair<std::pair<std::uint32_t, std::int32_t>, std::size_t>>
        ranked(frequencies.begin(), frequencies.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
    image.delta_count = static_cast<std::uint8_t>(
        std::min(ranked.size(), kMemMacroDeltaDictionarySize));
    for (std::size_t i = 0; i < image.delta_count; ++i)
        image.deltas[i] = {ranked[i].first.first, ranked[i].first.second};
    // Three bits select how many of the seven queue-local decoder registers
    // are initialized; the entries themselves use the compact 22+14 form.
    image.stats.dictionary_bits =
        3 +
        image.delta_count * (kCompactStartDeltaBits + kCompactAddressDeltaBits);

    BitWriter writer;
    writer.bits(records.front().schedule.start_cycle, 32);
    if (!fits_unsigned(records.front().instruction.address, kMemAddressBits))
        throw std::invalid_argument(
            "first MEM Macro address does not fit physical SRAM address");
    writer.bits(records.front().instruction.address, kMemAddressBits);

    std::size_t begin = 0;
    while (begin < records.size()) {
        std::size_t end = begin + 1;
        while (end < records.size() && end - begin < 65536 &&
               same_template(records[begin], records[end]))
            ++end;
        const auto runLength = end - begin;
        writer.bit(runLength != 1);
        if (runLength != 1)
            writer.bits(runLength - 1, 16);
        write_template(writer, records[begin], image.stats);
        ++image.stats.run_count;
        for (std::size_t i = begin; i < end; ++i) {
            if (i == 0)
                continue;
            const auto startDelta = records[i].schedule.start_cycle -
                                    records[i - 1].schedule.start_cycle;
            const auto addressDelta =
                static_cast<std::int64_t>(records[i].instruction.address) -
                static_cast<std::int64_t>(records[i - 1].instruction.address);
            write_delta(writer,
                        {static_cast<std::uint32_t>(startDelta),
                         static_cast<std::int32_t>(addressDelta)},
                        image, image.stats);
        }
        begin = end;
    }
    image.bit_count = writer.size();
    image.stats.stream_bits = image.bit_count;
    image.bytes = writer.take();
    return image;
}

QueueProgram decode_mem_macro_bitstream(const MemMacroBitstream& image,
                                        std::size_t queue_index)
{
    if (image.version != kMemMacroBitstreamVersion)
        throw std::invalid_argument("unsupported MEM Macro bitstream version");
    if (image.delta_count > image.deltas.size())
        throw std::invalid_argument("invalid MEM Macro delta dictionary size");
    QueueProgram queue{QueueKind::Mem, queue_index, {}};
    queue.commands.reserve(image.command_count);
    if (image.command_count == 0) {
        if (image.bit_count != 0)
            throw std::invalid_argument(
                "empty MEM Macro image has payload bits");
        return queue;
    }

    BitReader reader(image.bytes, image.bit_count);
    std::uint64_t start = reader.bits(32);
    std::int64_t address = reader.bits(kMemAddressBits);
    while (queue.commands.size() < image.command_count) {
        const auto longRun = reader.bit();
        const auto runLength = longRun ? reader.bits(16) + 1 : 1;
        if (runLength > image.command_count - queue.commands.size())
            throw std::runtime_error(
                "MEM Macro run exceeds declared command count");
        const auto templ = read_template(reader);
        for (std::uint64_t offset = 0; offset < runLength; ++offset) {
            if (!queue.commands.empty()) {
                const auto delta = read_delta(reader, image);
                start += delta.start_cycle;
                address += delta.address;
            }
            if (start > std::numeric_limits<std::uint32_t>::max() ||
                address < 0 || !fits_unsigned(address, kMemAddressBits))
                throw std::runtime_error(
                    "decoded MEM Macro state is out of range");
            auto record = templ;
            record.schedule.start_cycle = start;
            record.instruction.address = static_cast<std::size_t>(address);
            queue.commands.push_back(make_command(record));
        }
    }
    if (reader.remaining() != 0)
        throw std::runtime_error("MEM Macro bitstream has trailing bits");
    return queue;
}

MxmMacroBitstream encode_mxm_macro_bitstream(const QueueProgram& queue)
{
    if (!is_mxm_macro_kind(queue.kind))
        throw std::invalid_argument(
            "MXM Macro bitstream requires an MXM queue");
    if (queue.commands.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("MXM Macro queue is too large");

    std::vector<MxmMacroRecord> records;
    records.reserve(queue.commands.size());
    for (const auto& command : queue.commands) {
        records.push_back(decode_mxm_record(command, queue.kind));
        validate_mxm_record(records.back(), queue.kind);
    }

    MxmMacroBitstream image;
    image.kind = queue.kind;
    image.command_count = static_cast<std::uint32_t>(records.size());
    image.stats.command_count = records.size();
    if (records.empty())
        return image;

    std::map<std::pair<std::uint32_t, std::int32_t>, std::size_t> frequencies;
    for (std::size_t i = 1; i < records.size(); ++i) {
        if (records[i].schedule.start_cycle <
            records[i - 1].schedule.start_cycle)
            throw std::invalid_argument(
                "MXM Macro start cycles are not monotonic");
        const auto startDelta = records[i].schedule.start_cycle -
                                records[i - 1].schedule.start_cycle;
        const auto operandDelta =
            static_cast<std::int64_t>(mxm_operand(records[i], queue.kind)) -
            static_cast<std::int64_t>(mxm_operand(records[i - 1], queue.kind));
        if (startDelta > std::numeric_limits<std::uint32_t>::max() ||
            !fits_signed(operandDelta, 32))
            throw std::invalid_argument(
                "MXM Macro transition exceeds the wide escape");
        const MxmMacroDelta delta{static_cast<std::uint32_t>(startDelta),
                                  static_cast<std::int32_t>(operandDelta)};
        if (compact_mxm_delta(delta))
            ++frequencies[{delta.start_cycle, delta.operand}];
    }
    std::vector<std::pair<std::pair<std::uint32_t, std::int32_t>, std::size_t>>
        ranked(frequencies.begin(), frequencies.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
    image.delta_count = static_cast<std::uint8_t>(
        std::min(ranked.size(), kMxmMacroDeltaDictionarySize));
    for (std::size_t i = 0; i < image.delta_count; ++i)
        image.deltas[i] = {ranked[i].first.first, ranked[i].first.second};
    image.stats.dictionary_bits =
        3 +
        image.delta_count * (kCompactStartDeltaBits + kCompactAddressDeltaBits);

    BitWriter writer;
    writer.bits(records.front().schedule.start_cycle, 32);
    const auto operandBits = mxm_operand_bits(queue.kind);
    if (operandBits != 0) {
        const auto operand = mxm_operand(records.front(), queue.kind);
        if (!fits_unsigned(operand, operandBits))
            throw std::invalid_argument(
                "first MXM Macro operand does not fit physical field");
        writer.bits(operand, operandBits);
    }

    std::size_t begin = 0;
    while (begin < records.size()) {
        std::size_t end = begin + 1;
        while (end < records.size() && end - begin < 65536 &&
               same_mxm_template(records[begin], records[end], queue.kind))
            ++end;
        const auto runLength = end - begin;
        writer.bit(runLength != 1);
        if (runLength != 1)
            writer.bits(runLength - 1, 16);
        write_mxm_template(writer, records[begin], queue.kind, image.stats);
        ++image.stats.run_count;
        for (std::size_t i = begin; i < end; ++i) {
            if (i == 0)
                continue;
            const auto operandDelta =
                static_cast<std::int64_t>(mxm_operand(records[i], queue.kind)) -
                static_cast<std::int64_t>(
                    mxm_operand(records[i - 1], queue.kind));
            write_mxm_delta(writer,
                            {static_cast<std::uint32_t>(
                                 records[i].schedule.start_cycle -
                                 records[i - 1].schedule.start_cycle),
                             static_cast<std::int32_t>(operandDelta)},
                            image, image.stats);
        }
        begin = end;
    }
    image.bit_count = writer.size();
    image.stats.stream_bits = image.bit_count;
    image.bytes = writer.take();
    return image;
}

QueueProgram decode_mxm_macro_bitstream(const MxmMacroBitstream& image,
                                        std::size_t queue_index)
{
    if (image.version != kMxmMacroBitstreamVersion)
        throw std::invalid_argument("unsupported MXM Macro bitstream version");
    if (!is_mxm_macro_kind(image.kind))
        throw std::invalid_argument(
            "MXM Macro bitstream has an invalid queue kind");
    if (image.delta_count > image.deltas.size())
        throw std::invalid_argument("invalid MXM Macro delta dictionary size");
    QueueProgram queue{image.kind, queue_index, {}};
    queue.commands.reserve(image.command_count);
    if (image.command_count == 0) {
        if (image.bit_count != 0)
            throw std::invalid_argument(
                "empty MXM Macro image has payload bits");
        return queue;
    }

    BitReader reader(image.bytes, image.bit_count);
    std::uint64_t start = reader.bits(32);
    const auto operandBits = mxm_operand_bits(image.kind);
    std::int64_t operand = operandBits == 0 ? 0 : reader.bits(operandBits);
    while (queue.commands.size() < image.command_count) {
        const auto longRun = reader.bit();
        const auto runLength = longRun ? reader.bits(16) + 1 : 1;
        if (runLength > image.command_count - queue.commands.size())
            throw std::runtime_error(
                "MXM Macro run exceeds declared command count");
        const auto templ = read_mxm_template(reader, image.kind);
        for (std::uint64_t offset = 0; offset < runLength; ++offset) {
            if (!queue.commands.empty()) {
                const auto delta = read_mxm_delta(reader, image);
                start += delta.start_cycle;
                operand += delta.operand;
            }
            if (start > std::numeric_limits<std::uint32_t>::max() ||
                operand < 0 ||
                (operandBits != 0 && !fits_unsigned(operand, operandBits)))
                throw std::runtime_error(
                    "decoded MXM Macro state is out of range");
            auto record = templ;
            record.schedule.start_cycle = start;
            set_mxm_operand(record, image.kind,
                            static_cast<std::uint32_t>(operand));
            validate_mxm_record(record, image.kind);
            queue.commands.push_back(make_mxm_command(record, image.kind));
        }
    }
    if (reader.remaining() != 0)
        throw std::runtime_error("MXM Macro bitstream has trailing bits");
    return queue;
}

namespace {

bool raw_bit(const std::vector<std::uint8_t>& bytes, std::uint64_t position)
{
    return ((bytes.at(static_cast<std::size_t>(position / 8)) >>
             (position & 7u)) &
            1u) != 0;
}

void append_raw_bits(BitWriter& writer,
                     const std::vector<std::uint8_t>& bytes,
                     std::uint64_t bitCount)
{
    if (bitCount > static_cast<std::uint64_t>(bytes.size()) * 8)
        throw std::invalid_argument(
            "Macro body bit count exceeds its software byte payload");
    for (std::uint64_t bit = 0; bit < bitCount; ++bit)
        writer.bit(raw_bit(bytes, bit));
}

void validate_packed_image(const PackedMacroImemImage& image,
                           std::uint32_t expectedWordBits)
{
    if (image.word_bits != expectedWordBits)
        throw std::invalid_argument("Macro i-MEM image has the wrong word width");
    if (image.bytes.size() * 8 % expectedWordBits != 0)
        throw std::invalid_argument(
            "Macro i-MEM image is not padded to a complete physical word");
    const auto storageBits = static_cast<std::uint64_t>(image.bytes.size()) * 8;
    if (image.software_valid_bit_length < expectedWordBits ||
        image.software_valid_bit_length > storageBits)
        throw std::invalid_argument(
            "Macro i-MEM software bit length is outside the physical image");
    for (std::uint64_t bit = image.software_valid_bit_length;
         bit < storageBits; ++bit) {
        if (raw_bit(image.bytes, bit))
            throw std::invalid_argument(
                "Macro i-MEM final word has non-zero padding bits");
    }
}

IcuQueueControl read_control_word(BitReader& reader, std::uint32_t wordBits)
{
    const auto control = decode_icu_queue_control(
        static_cast<std::uint32_t>(reader.bits(32)));
    for (std::uint32_t bit = 32; bit < wordBits; ++bit) {
        if (reader.bit())
            throw std::invalid_argument(
                "Macro i-MEM control word has non-zero upper reserved bits");
    }
    if (!control.valid)
        throw std::invalid_argument("Macro i-MEM queue is not valid");
    if (control.mode != IcuQueueMode::Macro)
        throw std::invalid_argument("Macro i-MEM queue has a non-Macro mode");
    return control;
}

void finish_packed_image(BitWriter& writer,
                         std::uint32_t wordBits,
                         PackedMacroImemImage& result)
{
    result.word_bits = wordBits;
    result.software_valid_bit_length = writer.size();
    while (writer.size() % wordBits != 0)
        writer.bit(false);
    result.bytes = writer.take();
}

} // namespace

PackedMacroImemImage pack_mem_macro_imem(const MemMacroBitstream& image,
                                         bool enable,
                                         bool valid)
{
    if (image.command_count > kIcuQueueControlMaxCommandCount)
        throw std::invalid_argument(
            "MEM Macro command count does not fit the physical control word");
    if (image.delta_count > image.deltas.size())
        throw std::invalid_argument("invalid MEM Macro delta dictionary size");
    if (image.command_count == 0 &&
        (image.delta_count != 0 || image.bit_count != 0))
        throw std::invalid_argument("empty MEM Macro image has a payload");

    BitWriter writer;
    writer.bits(encode_icu_queue_control(
                    {valid, enable, IcuQueueMode::Macro, image.command_count}),
                32);
    writer.bits(0, kMemIcuImemWordBits - 32);
    if (image.command_count != 0) {
        writer.bits(image.delta_count, 3);
        for (std::size_t i = 0; i < image.delta_count; ++i) {
            writer.bits(image.deltas[i].start_cycle,
                        kCompactStartDeltaBits);
            write_signed(writer, image.deltas[i].address,
                         kCompactAddressDeltaBits);
        }
        append_raw_bits(writer, image.bytes, image.bit_count);
    }

    PackedMacroImemImage result;
    finish_packed_image(writer, kMemIcuImemWordBits, result);
    return result;
}

QueueProgram decode_mem_macro_imem(const PackedMacroImemImage& packed,
                                   std::size_t queue_index)
{
    validate_packed_image(packed, kMemIcuImemWordBits);
    BitReader reader(packed.bytes, packed.software_valid_bit_length);
    const auto control = read_control_word(reader, kMemIcuImemWordBits);

    MemMacroBitstream image;
    image.command_count = control.command_count;
    if (image.command_count != 0) {
        image.delta_count = static_cast<std::uint8_t>(reader.bits(3));
        if (image.delta_count > image.deltas.size())
            throw std::invalid_argument(
                "invalid MEM Macro inline dictionary size");
        for (std::size_t i = 0; i < image.delta_count; ++i) {
            image.deltas[i].start_cycle =
                static_cast<std::uint32_t>(reader.bits(kCompactStartDeltaBits));
            image.deltas[i].address = static_cast<std::int32_t>(
                read_signed(reader, kCompactAddressDeltaBits));
        }
        BitWriter body;
        image.bit_count = reader.remaining();
        while (reader.remaining() != 0)
            body.bit(reader.bit());
        image.bytes = body.take();
    } else if (reader.remaining() != 0) {
        throw std::invalid_argument("empty MEM Macro i-MEM has payload bits");
    }
    return decode_mem_macro_bitstream(image, queue_index);
}

PackedMacroImemImage pack_mxm_macro_imem(const MxmMacroBitstream& image,
                                         bool enable,
                                         bool valid)
{
    if (image.command_count > kIcuQueueControlMaxCommandCount)
        throw std::invalid_argument(
            "MXM Macro command count does not fit the physical control word");
    if (!is_mxm_macro_kind(image.kind))
        throw std::invalid_argument("invalid MXM Macro queue kind");
    if (image.delta_count > image.deltas.size())
        throw std::invalid_argument("invalid MXM Macro delta dictionary size");
    if (image.command_count == 0 &&
        (image.delta_count != 0 || image.bit_count != 0))
        throw std::invalid_argument("empty MXM Macro image has a payload");

    BitWriter writer;
    writer.bits(encode_icu_queue_control(
                    {valid, enable, IcuQueueMode::Macro, image.command_count}),
                32);
    writer.bits(0, kMxmIcuImemWordBits - 32);
    if (image.command_count != 0) {
        writer.bits(image.delta_count, 3);
        for (std::size_t i = 0; i < image.delta_count; ++i) {
            writer.bits(image.deltas[i].start_cycle,
                        kCompactStartDeltaBits);
            write_signed(writer, image.deltas[i].operand,
                         kCompactAddressDeltaBits);
        }
        append_raw_bits(writer, image.bytes, image.bit_count);
    }

    PackedMacroImemImage result;
    finish_packed_image(writer, kMxmIcuImemWordBits, result);
    return result;
}

QueueProgram decode_mxm_macro_imem(const PackedMacroImemImage& packed,
                                   QueueKind kind,
                                   std::size_t queue_index)
{
    if (!is_mxm_macro_kind(kind))
        throw std::invalid_argument("invalid MXM Macro queue kind");
    validate_packed_image(packed, kMxmIcuImemWordBits);
    BitReader reader(packed.bytes, packed.software_valid_bit_length);
    const auto control = read_control_word(reader, kMxmIcuImemWordBits);

    MxmMacroBitstream image;
    image.kind = kind;
    image.command_count = control.command_count;
    if (image.command_count != 0) {
        image.delta_count = static_cast<std::uint8_t>(reader.bits(3));
        if (image.delta_count > image.deltas.size())
            throw std::invalid_argument(
                "invalid MXM Macro inline dictionary size");
        for (std::size_t i = 0; i < image.delta_count; ++i) {
            image.deltas[i].start_cycle =
                static_cast<std::uint32_t>(reader.bits(kCompactStartDeltaBits));
            image.deltas[i].operand = static_cast<std::int32_t>(
                read_signed(reader, kCompactAddressDeltaBits));
        }
        BitWriter body;
        image.bit_count = reader.remaining();
        while (reader.remaining() != 0)
            body.bit(reader.bit());
        image.bytes = body.take();
    } else if (reader.remaining() != 0) {
        throw std::invalid_argument("empty MXM Macro i-MEM has payload bits");
    }
    return decode_mxm_macro_bitstream(image, queue_index);
}

} // namespace ftlpu::software::runtime
