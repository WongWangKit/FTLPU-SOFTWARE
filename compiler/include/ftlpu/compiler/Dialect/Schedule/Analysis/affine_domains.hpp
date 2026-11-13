#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/sxm/instruction.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/weight_dequantizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ftlpu::compiler {

// Compiler-only affine iteration domain. This is not an ICU instruction or packet.
// Counter 0 is innermost; materialization emits packed Macro templates directly.
struct AffineLoopDomain {
    static constexpr std::size_t kDimensions = 3;

    // Absolute issue coordinate in the compiler schedule.
    std::size_t start_cycle{0};
    std::array<std::size_t, kDimensions> counts{1, 1, 1};
    std::array<std::size_t, kDimensions> cycle_strides{1, 1, 1};
    // Additional compiler scheduling offset.
    std::size_t wait_cycle{0};
};

struct DomainCoordinate {
    std::array<std::size_t, AffineLoopDomain::kDimensions> index{0, 0, 0};

    std::size_t operator[](std::size_t dimension) const
    {
        return index.at(dimension);
    }
};

// The outer counter may use a power-of-two carry/wrap address generator.
// Hardware takes the group index and within-group index from a shift and a
// mask; it does not implement a general divider or a hidden launch counter.
// For group_size=1 this reduces to an ordinary affine 3-D address with
// outer_group_stride as the counter-2 stride.
struct MemAddressPattern {
    std::size_t base_address{0};
    std::int64_t inner_stride{0};
    std::int64_t middle_stride{0};
    std::size_t outer_group_size{1};
    std::int64_t outer_inner_stride{0};
    std::int64_t outer_group_stride{0};

    static constexpr MemAddressPattern Affine(
        std::size_t base_address,
        std::array<std::int64_t, 3> strides) noexcept
    {
        return {base_address, strides[0], strides[1], 1, 0, strides[2]};
    }

    static constexpr MemAddressPattern BlockedOuter(
        std::size_t base_address,
        std::int64_t inner_stride,
        std::int64_t middle_stride,
        std::size_t outer_group_size,
        std::int64_t outer_inner_stride,
        std::int64_t outer_group_stride) noexcept
    {
        return {base_address, inner_stride, middle_stride,
            outer_group_size, outer_inner_stride, outer_group_stride};
    }
};

enum class MemDomainOperation : std::uint8_t {
    Read = 0,
    Write = 1,
    WriteTap = 2,
};

struct MemAddressDomain {
    MemDomainOperation opcode{MemDomainOperation::Read};
    AffineLoopDomain loop{};
    MemAddressPattern address{};
    std::size_t stream{0};

    static MemAddressDomain Read(
        AffineLoopDomain loop,
        MemAddressPattern address,
        StreamId stream)
    {
        return {MemDomainOperation::Read, loop, address, stream.packed()};
    }

    static MemAddressDomain Write(
        AffineLoopDomain loop,
        MemAddressPattern address,
        StreamId stream)
    {
        return {MemDomainOperation::Write, loop, address, stream.packed()};
    }

    static MemAddressDomain WriteTap(
        AffineLoopDomain loop,
        MemAddressPattern address,
        StreamId stream)
    {
        return {MemDomainOperation::WriteTap, loop, address, stream.packed()};
    }
};

// One physical MEM bank alternates two statically scheduled 2-D event
// streams. Both streams address the same affine rows; there is no data FIFO,
// ready/valid handshake, or run-time empty/full test in the ICU.
struct MemTransferPairDomain {
    std::size_t start_wait{0};
    std::array<std::size_t, 2> counts{1, 1};
    std::array<std::size_t, 2> write_cycle_strides{1, 1};
    std::array<std::size_t, 2> read_cycle_strides{1, 1};
    std::size_t read_start_offset{1};
    std::size_t base_address{0};
    std::array<std::int64_t, 2> address_strides{0, 0};
    std::size_t write_stream{0};
    std::size_t read_stream_base{0};
    std::int64_t read_stream_outer_stride{0};
};

// Two physical MXM buffers make dimension-parity selection a small XOR in
// the ICU instead of a model-specific block-number calculation.
enum class MxmBufferSelection : std::uint8_t {
    Fixed = 0,
    ToggleDimension0 = 1,
    ToggleDimension1 = 2,
    ToggleDimension2 = 3,
};

enum class MxmLoadDomainOperation : std::uint8_t {
    Load = 0,
    LoadActivation = 1,
    // Affine column coordinates address individual columns, not supercells.
    LoadColumn = 2,
};

struct MxmLoadDomain {
    MxmLoadDomainOperation opcode{MxmLoadDomainOperation::Load};
    AffineLoopDomain loop{};
    std::size_t weight_buffer_base{0};
    MxmBufferSelection weight_buffer_mode{MxmBufferSelection::Fixed};
    // Load uses supercell columns. LoadColumn uses physical columns;
    // expansion divides by eight to select supercell and inner column.
    std::size_t weight_column_base{0};
    std::array<std::int64_t, 3> weight_column_strides{1, 0, 0};
    std::size_t weight_stream_base{0};
    MxmWeightInputMode weight_input_mode{
        MxmWeightInputMode::Int8DequantBf16};
    MxmDataFormat data_format{MxmDataFormat::BFloat16};
    MxmIsMode is_mode{MxmIsMode::Block};

    static MxmLoadDomain Load(
        AffineLoopDomain loop,
        std::size_t weight_buffer_base,
        MxmBufferSelection weight_buffer_mode,
        std::size_t weight_column_base,
        std::array<std::int64_t, 3> weight_column_strides,
        std::size_t weight_stream_base,
        MxmWeightInputMode weight_input_mode =
            MxmWeightInputMode::Int8DequantBf16)
    {
        auto instruction = MxmLoadDomain {};
        instruction.opcode = MxmLoadDomainOperation::Load;
        instruction.loop = loop;
        instruction.weight_buffer_base = weight_buffer_base;
        instruction.weight_buffer_mode = weight_buffer_mode;
        instruction.weight_column_base = weight_column_base;
        instruction.weight_column_strides = weight_column_strides;
        instruction.weight_stream_base = weight_stream_base;
        instruction.weight_input_mode = weight_input_mode;
        return instruction;
    }

    static MxmLoadDomain LoadActivation(
        AffineLoopDomain loop,
        std::size_t activation_buffer_base,
        MxmBufferSelection activation_buffer_mode,
        std::size_t activation_stream_base,
        MxmDataFormat data_format,
        MxmIsMode is_mode)
    {
        auto instruction = MxmLoadDomain {};
        instruction.opcode = MxmLoadDomainOperation::LoadActivation;
        instruction.loop = loop;
        instruction.weight_buffer_base = activation_buffer_base;
        instruction.weight_buffer_mode = activation_buffer_mode;
        instruction.weight_stream_base = activation_stream_base;
        instruction.data_format = data_format;
        instruction.is_mode = is_mode;
        return instruction;
    }
};

struct MxmDequantDomain {
    AffineLoopDomain loop{};
    MxmDequantInstruction instruction{};

    static MxmDequantDomain Dequant(
        AffineLoopDomain loop,
        MxmDequantInstruction instruction)
    {
        return {loop, instruction};
    }
};

struct MxmComputeMode {
    MxmAccumulatorDestination accumulator_destination{
        MxmAccumulatorDestination::Sram};
    bool accumulator_clear{false};
    MxmAccumulatorOutputFormat accumulator_output_format{
        MxmAccumulatorOutputFormat::Float32};
};

// FU-local operations decoded by one physical MXM compute ICU.  These values
// are carried by the local-operation field in every word of its 3-D packet;
// they are not global ICU opcodes or functional-unit selectors.
enum class MxmComputeDomainOperation : std::uint8_t {
    Compute = 0,
    AccumulatorRead = 1,
    StreamCompute = 2,
};

struct MxmComputeDomain {
    static constexpr std::size_t kNoTerminalDimension = 3;

    MxmComputeDomainOperation opcode{MxmComputeDomainOperation::Compute};
    AffineLoopDomain loop{};
    std::size_t weight_buffer_base{0};
    MxmBufferSelection weight_buffer_mode{MxmBufferSelection::Fixed};
    std::size_t activation_stream_base{0};
    std::size_t result_stream_base{0};
    std::size_t accumulator_address_base{0};
    std::array<std::int64_t, 3> accumulator_address_strides{0, 0, 0};
    std::size_t accumulator_row_stride{1};
    MxmDataFormat data_format{MxmDataFormat::BFloat16};
    std::size_t accumulator_column{0};
    MxmIsMode is_mode{MxmIsMode::Block};
    MxmComputeMode regular_mode{};
    std::size_t terminal_dimension{kNoTerminalDimension};
    MxmComputeMode terminal_mode{};

    static MxmComputeDomain Compute(
        AffineLoopDomain loop,
        std::size_t weight_buffer_base,
        MxmBufferSelection weight_buffer_mode,
        std::size_t activation_stream_base,
        std::size_t result_stream_base,
        std::size_t accumulator_address_base,
        std::array<std::int64_t, 3> accumulator_address_strides,
        std::size_t accumulator_row_stride,
        MxmDataFormat data_format,
        MxmComputeMode regular_mode,
        std::size_t terminal_dimension = kNoTerminalDimension,
        MxmComputeMode terminal_mode = {})
    {
        auto instruction = MxmComputeDomain {};
        instruction.opcode = MxmComputeDomainOperation::Compute;
        instruction.loop = loop;
        instruction.weight_buffer_base = weight_buffer_base;
        instruction.weight_buffer_mode = weight_buffer_mode;
        instruction.activation_stream_base = activation_stream_base;
        instruction.result_stream_base = result_stream_base;
        instruction.accumulator_address_base = accumulator_address_base;
        instruction.accumulator_address_strides = accumulator_address_strides;
        instruction.accumulator_row_stride = accumulator_row_stride;
        instruction.data_format = data_format;
        instruction.regular_mode = regular_mode;
        instruction.terminal_dimension = terminal_dimension;
        instruction.terminal_mode = terminal_mode;
        return instruction;
    }

    static MxmComputeDomain AccumulatorRead(
        AffineLoopDomain loop,
        std::size_t result_stream_base,
        std::size_t accumulator_address_base,
        std::array<std::int64_t, 3> accumulator_address_strides,
        bool accumulator_clear = true,
        MxmAccumulatorOutputFormat accumulator_output_format =
            MxmAccumulatorOutputFormat::Float32,
        MxmAccumulatorDestination accumulator_destination =
            MxmAccumulatorDestination::Stream)
    {
        auto instruction = MxmComputeDomain {};
        instruction.opcode = MxmComputeDomainOperation::AccumulatorRead;
        instruction.loop = loop;
        instruction.result_stream_base = result_stream_base;
        instruction.accumulator_address_base = accumulator_address_base;
        instruction.accumulator_address_strides = accumulator_address_strides;
        instruction.regular_mode = {
            accumulator_destination,
            accumulator_clear,
            accumulator_output_format,
        };
        return instruction;
    }

    static MxmComputeDomain StreamCompute(
        AffineLoopDomain loop,
        std::size_t activation_buffer_base,
        MxmBufferSelection activation_buffer_mode,
        std::size_t result_stream_base,
        MxmDataFormat data_format,
        std::size_t accumulator_address_base,
        std::array<std::int64_t, 3> accumulator_address_strides,
        std::size_t accumulator_column,
        MxmAccumulatorDestination accumulator_destination,
        bool accumulator_clear,
        MxmIsMode is_mode)
    {
        auto instruction = MxmComputeDomain {};
        instruction.opcode = MxmComputeDomainOperation::StreamCompute;
        instruction.loop = loop;
        instruction.weight_buffer_base = activation_buffer_base;
        instruction.weight_buffer_mode = activation_buffer_mode;
        instruction.result_stream_base = result_stream_base;
        instruction.accumulator_address_base = accumulator_address_base;
        instruction.accumulator_address_strides = accumulator_address_strides;
        instruction.data_format = data_format;
        instruction.accumulator_column = accumulator_column;
        instruction.regular_mode.accumulator_destination =
            accumulator_destination;
        instruction.regular_mode.accumulator_clear = accumulator_clear;
        instruction.is_mode = is_mode;
        return instruction;
    }
};

struct SxmLaunchDomain {
    AffineLoopDomain loop{};
    SxmInstruction instruction{};
    std::size_t permute_map_stride{0};
    static SxmLaunchDomain Launch(std::size_t start,std::array<std::size_t,2> counts,
        std::array<std::size_t,2> strides,SxmInstruction instruction,std::size_t mapStride=0) {
        return {{start,{counts[0],counts[1],1},{strides[0],strides[1],1}},std::move(instruction),mapStride};
    }
};

namespace detail {

inline void validate_affine_loop(const AffineLoopDomain& loop)
{

    if (loop.wait_cycle >= (std::size_t {1} << 24))
        throw std::invalid_argument(
            "ICU 3-D wait_cycle exceeds its 24-bit field");
    std::size_t lower_span = 0;
    for (std::size_t dimension = 0;
         dimension < AffineLoopDomain::kDimensions; ++dimension) {
        const auto count = loop.counts[dimension];
        const auto stride = loop.cycle_strides[dimension];
        if (count == 0 || count > 65536 || stride == 0
            || stride >= (std::size_t {1} << 24))
            throw std::invalid_argument(
                "ICU 3-D loop dimension " + std::to_string(dimension)
                + " exceeds its fixed field width: count="
                + std::to_string(count) + " stride="
                + std::to_string(stride));
        if (dimension != 0 && count > 1 && stride <= lower_span)
            throw std::invalid_argument(
                "ICU 3-D loop dimensions overlap in issue time");
        const auto steps = count - 1;
        if (steps != 0
            && stride > (std::numeric_limits<std::size_t>::max()
                    - lower_span) / steps)
            throw std::overflow_error("ICU 3-D cycle span overflows");
        lower_span += steps * stride;
    }
    constexpr auto cycleLimit = std::size_t {1} << 24;
    if (lower_span >= cycleLimit)
        throw std::invalid_argument(
            "ICU 3-D relative cycle span exceeds the 24-bit schedule domain");
}

inline std::size_t affine_point_count(const AffineLoopDomain& loop)
{
    std::size_t points = 1;
    for (const auto count : loop.counts) {
        if (points > std::numeric_limits<std::size_t>::max() / count)
            throw std::overflow_error("ICU 3-D point count overflows");
        points *= count;
    }
    return points;
}

inline std::size_t affine_issue_offset(
    const AffineLoopDomain& loop,
    const DomainCoordinate& coordinate)
{
    // This is an instruction-local cycle offset. The physical ICU never
    // compares it with a program-global or system-global cycle counter.
    auto cycle = loop.wait_cycle;
    for (std::size_t dimension = 0;
         dimension < AffineLoopDomain::kDimensions; ++dimension)
        cycle += coordinate.index[dimension]
            * loop.cycle_strides[dimension];
    return cycle;
}

inline bool advance_domain_coordinate(
    const AffineLoopDomain& loop,
    DomainCoordinate& coordinate) noexcept
{
    for (std::size_t dimension = 0;
         dimension < AffineLoopDomain::kDimensions; ++dimension) {
        if (coordinate.index[dimension] < loop.counts[dimension] - 1) {
            ++coordinate.index[dimension];
            return true;
        }
        coordinate.index[dimension] = 0;
    }
    return false;
}

inline std::int64_t checked_domain_operand(
    std::size_t base,
    const std::array<std::int64_t, 3>& strides,
    const DomainCoordinate& coordinate,
    const char* field)
{
    if (base > static_cast<std::size_t>(
                   std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error(std::string(field) + " base overflows");
    auto value = static_cast<std::int64_t>(base);
    for (std::size_t dimension = 0;
         dimension < AffineLoopDomain::kDimensions; ++dimension) {
        const auto index = coordinate.index[dimension];
        const auto stride = strides[dimension];
        if (index > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max())
            || (stride > 0
                && index > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max() / stride))
            || (stride < 0 && stride != -1
                && index > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::min() / stride)))
            throw std::overflow_error(
                std::string(field) + " induction multiply overflows");
        const auto delta = static_cast<std::int64_t>(index) * stride;
        if ((delta > 0
                && value > std::numeric_limits<std::int64_t>::max() - delta)
            || (delta < 0
                && value < std::numeric_limits<std::int64_t>::min() - delta))
            throw std::overflow_error(
                std::string(field) + " induction addition overflows");
        value += delta;
    }
    return value;
}

inline std::size_t mem_domain_address(
    const MemAddressDomain& instruction,
    const DomainCoordinate& coordinate)
{
    if (instruction.address.outer_group_size == 0
        || !std::has_single_bit(
            instruction.address.outer_group_size))
        throw std::invalid_argument(
            "MEM ICU 3-D outer group size must be a power of two");
    const auto outer = coordinate.index[2];
    const auto groupShift = std::countr_zero(
        instruction.address.outer_group_size);
    const auto groupMask = instruction.address.outer_group_size - 1;
    const DomainCoordinate transformed {{
        coordinate.index[0],
        coordinate.index[1],
        outer >> groupShift,
    }};
    const std::array<std::int64_t, 3> strides {
        instruction.address.inner_stride,
        instruction.address.middle_stride,
        instruction.address.outer_group_stride,
    };
    auto address = checked_domain_operand(
        instruction.address.base_address,
        strides,
        transformed,
        "MEM ICU 3-D address");
    const auto remainder = outer & groupMask;
    const auto extra = checked_domain_operand(
        0,
        {instruction.address.outer_inner_stride, 0, 0},
        DomainCoordinate {{remainder, 0, 0}},
        "MEM ICU 3-D blocked address");
    if ((extra > 0
            && address > std::numeric_limits<std::int64_t>::max() - extra)
        || (extra < 0
            && address < std::numeric_limits<std::int64_t>::min() - extra))
        throw std::overflow_error("MEM ICU 3-D blocked address overflows");
    address += extra;
    if (address < 0
        || static_cast<std::size_t>(address) >= hw::kSramDepthRows)
        throw std::out_of_range(
            "MEM ICU 3-D address is outside the queue-local SRAM bank");
    return static_cast<std::size_t>(address);
}

inline std::size_t mxm_domain_buffer(
    std::size_t base,
    MxmBufferSelection mode,
    const DomainCoordinate& coordinate)
{
    MxmControlInstruction::check_weight_buffer(base);
    if (mode == MxmBufferSelection::Fixed) return base;
    const auto encoded = static_cast<std::size_t>(mode);
    if (encoded < 1 || encoded > 3)
        throw std::invalid_argument("MXM ICU 3-D buffer mode is invalid");
    const auto dimension = encoded - 1;
    const auto buffer = base ^ (coordinate.index[dimension] & 1U);
    MxmControlInstruction::check_weight_buffer(buffer);
    return buffer;
}

inline void validate_mem_domain(
    const MemAddressDomain& instruction)
{
    validate_affine_loop(instruction.loop);
    static_cast<void>(StreamId::from_packed(instruction.stream));
    switch (instruction.opcode) {
    case MemDomainOperation::Read:
    case MemDomainOperation::Write:
    case MemDomainOperation::WriteTap:
        break;
    default:
        throw std::invalid_argument("MEM ICU 3-D opcode is invalid");
    }
    for (std::size_t outer = 0; outer < instruction.loop.counts[2]; ++outer) {
        for (const auto inner : {std::size_t {0},
                 instruction.loop.counts[0] - 1}) {
            for (const auto middle : {std::size_t {0},
                     instruction.loop.counts[1] - 1})
                static_cast<void>(mem_domain_address(
                    instruction, DomainCoordinate {{inner, middle, outer}}));
        }
    }
}

inline std::size_t mem_transfer_pair_last_offset(
    const MemTransferPairDomain& instruction) noexcept
{
    const auto lastWrite = instruction.start_wait
        + (instruction.counts[0] - 1) * instruction.write_cycle_strides[0]
        + (instruction.counts[1] - 1) * instruction.write_cycle_strides[1];
    const auto lastRead = instruction.start_wait + instruction.read_start_offset
        + (instruction.counts[0] - 1) * instruction.read_cycle_strides[0]
        + (instruction.counts[1] - 1) * instruction.read_cycle_strides[1];
    return std::max(lastWrite, lastRead);
}

inline void validate_mem_transfer_pair(
    const MemTransferPairDomain& instruction)
{
    constexpr auto cycleLimit = std::size_t {1} << 24;
    constexpr auto countLimit = std::size_t {1} << 16;
    if (instruction.start_wait >= cycleLimit
        || instruction.read_start_offset >= cycleLimit
        || instruction.base_address >= hw::kSramDepthRows)
        throw std::invalid_argument(
            "MEM read/write domain pair wait, offset, or base is out of range");
    static_cast<void>(StreamId::from_packed(instruction.write_stream));
    static_cast<void>(StreamId::from_packed(instruction.read_stream_base));
    if (instruction.read_stream_outer_stride < -32
        || instruction.read_stream_outer_stride > 31)
        throw std::invalid_argument(
            "MEM read/write domain pair read stream stride exceeds 6 bits");

    const auto validateSweep = [&](const std::array<std::size_t, 2>& strides) {
        for (std::size_t dim = 0; dim < 2; ++dim)
            if (instruction.counts[dim] == 0
                || instruction.counts[dim] > countLimit
                || strides[dim] == 0 || strides[dim] >= cycleLimit)
                throw std::invalid_argument(
                    "MEM read/write domain pair count or cycle stride is out of range");
        const auto innerSpan = (instruction.counts[0] - 1) * strides[0];
        if (instruction.counts[1] > 1 && strides[1] <= innerSpan)
            throw std::invalid_argument(
                "MEM read/write domain pair sweep overlaps its own issue cycles");
        return innerSpan + (instruction.counts[1] - 1) * strides[1];
    };
    const auto writeSpan = validateSweep(instruction.write_cycle_strides);
    const auto readSpan = validateSweep(instruction.read_cycle_strides);
    if (instruction.start_wait + writeSpan >= cycleLimit
        || instruction.start_wait + instruction.read_start_offset
                + readSpan >= cycleLimit)
        throw std::invalid_argument(
            "MEM read/write domain pair relative cycle span exceeds 24 bits");

    // The same coordinate must have been written before it is read. Because
    // this difference is affine, its minimum occurs at a rectangle corner.
    auto minReadAfterWrite = static_cast<std::int64_t>(
        instruction.read_start_offset);
    for (std::size_t dim = 0; dim < 2; ++dim) {
        const auto difference = static_cast<std::int64_t>(
            instruction.read_cycle_strides[dim])
            - static_cast<std::int64_t>(
                instruction.write_cycle_strides[dim]);
        if (difference < 0)
            minReadAfterWrite +=
                static_cast<std::int64_t>(instruction.counts[dim] - 1)
                * difference;
    }
    if (minReadAfterWrite <= 0)
        throw std::invalid_argument(
            "MEM read/write domain pair may read a coordinate before its write");

    const auto lastReadStream = static_cast<std::int64_t>(
        instruction.read_stream_base)
        + static_cast<std::int64_t>(instruction.counts[1] - 1)
            * instruction.read_stream_outer_stride;
    if (lastReadStream < 0
        || lastReadStream >= static_cast<std::int64_t>(hw::kStreams))
        throw std::invalid_argument(
            "MEM read/write domain pair read stream induction is out of range");

    for (std::size_t dim = 0; dim < 2; ++dim)
        if (instruction.address_strides[dim] < -(std::int64_t {1} << 19)
            || instruction.address_strides[dim]
                >= (std::int64_t {1} << 19))
            throw std::invalid_argument(
                "MEM read/write domain pair address stride exceeds 20 bits");
    for (std::size_t i : {std::size_t {0}, instruction.counts[0] - 1})
        for (std::size_t j : {std::size_t {0}, instruction.counts[1] - 1}) {
            const auto address = static_cast<std::int64_t>(
                instruction.base_address)
                + static_cast<std::int64_t>(i)
                    * instruction.address_strides[0]
                + static_cast<std::int64_t>(j)
                    * instruction.address_strides[1];
            if (address < 0
                || address >= static_cast<std::int64_t>(hw::kSramDepthRows))
                throw std::invalid_argument(
                    "MEM read/write domain pair affine address exceeds bank rows");
        }

    // A compact conservative proof of no row overwrite: addresses are
    // injective over the rectangle. A future ISA may permit safe row reuse.
    const auto a = instruction.address_strides[0] < 0
        ? -instruction.address_strides[0]
        : instruction.address_strides[0];
    const auto b = instruction.address_strides[1] < 0
        ? -instruction.address_strides[1]
        : instruction.address_strides[1];
    if ((instruction.counts[0] > 1 && a == 0)
        || (instruction.counts[1] > 1 && b == 0))
        throw std::invalid_argument(
            "MEM read/write domain pair row reuse is not statically proven safe");
    if (a != 0 && b != 0) {
        const auto divisor = std::gcd(a, b);
        if (b / divisor <= static_cast<std::int64_t>(
                instruction.counts[0] - 1)
            && a / divisor <= static_cast<std::int64_t>(
                instruction.counts[1] - 1))
            throw std::invalid_argument(
                "MEM read/write domain pair row reuse is not statically proven safe");
    }

    // Either all writes finish before reads begin, or the two periodic
    // sweeps occupy disjoint residue classes. Both proofs avoid enumerating
    // a potentially 65536 x 65536 domain during packet decode.
    if (instruction.read_start_offset <= writeSpan) {
        std::size_t period = 0;
        for (std::size_t dim = 0; dim < 2; ++dim) {
            if (instruction.counts[dim] <= 1) continue;
            period = std::gcd(period,
                instruction.write_cycle_strides[dim]);
            period = std::gcd(period,
                instruction.read_cycle_strides[dim]);
        }
        if (period == 0
            || instruction.read_start_offset % period == 0)
            throw std::invalid_argument(
                "MEM read/write domain pair single-port conflict is not statically excluded");
    }
}

inline MemInstruction mem_transfer_pair_instruction(
    const MemTransferPairDomain& instruction,
    std::size_t inner, std::size_t outer, bool read)
{
    const auto address = static_cast<std::size_t>(
        static_cast<std::int64_t>(instruction.base_address)
        + static_cast<std::int64_t>(inner)
            * instruction.address_strides[0]
        + static_cast<std::int64_t>(outer)
            * instruction.address_strides[1]);
    if (read)
        return MemInstruction::Read(address, static_cast<std::size_t>(
            static_cast<std::int64_t>(instruction.read_stream_base)
            + static_cast<std::int64_t>(outer)
                * instruction.read_stream_outer_stride));
    return MemInstruction::Write(address, instruction.write_stream);
}

inline MemInstruction mem_domain_instruction(
    const MemAddressDomain& instruction,
    const DomainCoordinate& coordinate)
{
    const auto address = mem_domain_address(instruction, coordinate);
    switch (instruction.opcode) {
    case MemDomainOperation::Read:
        return MemInstruction::Read(address, instruction.stream);
    case MemDomainOperation::Write:
        return MemInstruction::Write(address, instruction.stream);
    case MemDomainOperation::WriteTap:
        return MemInstruction::WriteTap(address, instruction.stream);
    }
    throw std::logic_error("MEM ICU 3-D opcode is invalid");
}

inline void validate_mxm_load_domain(
    const MxmLoadDomain& instruction)
{
    validate_affine_loop(instruction.loop);
    if (instruction.opcode
        == MxmLoadDomainOperation::LoadActivation) {
        MxmControlInstruction::check_data_format(instruction.data_format);
        MxmControlInstruction::check_is_mode(
            instruction.is_mode);
        for (std::size_t mask = 0; mask < 8; ++mask) {
            const DomainCoordinate coordinate {{
                (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
                (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
                (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
            }};
            static_cast<void>(mxm_domain_buffer(
                instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate));
        }
        static_cast<void>(MxmControlInstruction::IsLoadActivation(
            instruction.weight_buffer_base,
            instruction.weight_stream_base,
            instruction.data_format,
            instruction.is_mode));
        return;
    }
    if (instruction.opcode != MxmLoadDomainOperation::Load
        && instruction.opcode != MxmLoadDomainOperation::LoadColumn)
        throw std::invalid_argument("MXM load ICU 3-D opcode is invalid");
    for (std::size_t mask = 0; mask < 8; ++mask) {
        DomainCoordinate coordinate {{
            (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
            (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
            (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
        }};
        const auto column = checked_domain_operand(
            instruction.weight_column_base,
            instruction.weight_column_strides,
            coordinate,
            "MXM load domain weight column");
        if (column < 0)
            throw std::out_of_range(
                "MXM load domain weight column underflows");
        MxmControlInstruction::check_column(static_cast<std::size_t>(column)
            / (instruction.opcode == MxmLoadDomainOperation::LoadColumn ? 8 : 1));
        static_cast<void>(mxm_domain_buffer(
            instruction.weight_buffer_base,
            instruction.weight_buffer_mode,
            coordinate));
    }
    if (instruction.opcode == MxmLoadDomainOperation::LoadColumn) {
        static_cast<void>(MxmControlInstruction::IWColumn(
            instruction.weight_buffer_base, instruction.weight_column_base / 8,
            instruction.weight_column_base % 8, instruction.weight_input_mode,
            instruction.weight_stream_base));
        return;
    }
    static_cast<void>(MxmControlInstruction::IW(
        instruction.weight_buffer_base,
        instruction.weight_column_base,
        instruction.weight_input_mode,
        instruction.weight_stream_base));
}

inline MxmControlInstruction mxm_load_domain_instruction(
    const MxmLoadDomain& instruction,
    const DomainCoordinate& coordinate)
{
    if (instruction.opcode
        == MxmLoadDomainOperation::LoadActivation)
        return MxmControlInstruction::IsLoadActivation(
            mxm_domain_buffer(instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate),
            instruction.weight_stream_base,
            instruction.data_format,
            instruction.is_mode);
    if (instruction.opcode != MxmLoadDomainOperation::Load
        && instruction.opcode != MxmLoadDomainOperation::LoadColumn)
        throw std::logic_error("MXM load ICU 3-D opcode is invalid");
    const auto column = checked_domain_operand(
        instruction.weight_column_base,
        instruction.weight_column_strides,
        coordinate,
        "MXM load domain weight column");
    if (column < 0)
        throw std::out_of_range("MXM load domain weight column underflows");
    if (instruction.opcode == MxmLoadDomainOperation::LoadColumn)
        return MxmControlInstruction::IWColumn(
            mxm_domain_buffer(instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate),
            static_cast<std::size_t>(column) / 8,
            static_cast<std::size_t>(column) % 8,
            instruction.weight_input_mode, instruction.weight_stream_base);
    return MxmControlInstruction::IW(
        mxm_domain_buffer(instruction.weight_buffer_base,
            instruction.weight_buffer_mode, coordinate),
        static_cast<std::size_t>(column),
        instruction.weight_input_mode,
        instruction.weight_stream_base);
}

inline void validate_mxm_dequant_domain(
    const MxmDequantDomain& instruction)
{
    validate_affine_loop(instruction.loop);
}

inline void validate_mxm_compute_mode(const MxmComputeMode& mode)
{
    switch (mode.accumulator_destination) {
    case MxmAccumulatorDestination::Sram:
    case MxmAccumulatorDestination::Stream:
        break;
    default:
        throw std::invalid_argument("MXM compute domain destination is invalid");
    }
    switch (mode.accumulator_output_format) {
    case MxmAccumulatorOutputFormat::Float32:
    case MxmAccumulatorOutputFormat::BFloat16:
        break;
    default:
        throw std::invalid_argument("MXM compute domain output format is invalid");
    }
}

inline MxmComputeMode mxm_domain_compute_mode(
    const MxmComputeDomain& instruction,
    const DomainCoordinate& coordinate)
{
    if (instruction.terminal_dimension
            < AffineLoopDomain::kDimensions
        && coordinate.index[instruction.terminal_dimension] + 1
            == instruction.loop.counts[instruction.terminal_dimension])
        return instruction.terminal_mode;
    return instruction.regular_mode;
}

inline void validate_mxm_compute_domain(
    const MxmComputeDomain& instruction)
{
    validate_affine_loop(instruction.loop);
    if (instruction.opcode == MxmComputeDomainOperation::AccumulatorRead) {
        validate_mxm_compute_mode(instruction.regular_mode);
        for (std::size_t mask = 0; mask < 8; ++mask) {
            const DomainCoordinate coordinate {{
                (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
                (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
                (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
            }};
            const auto address = checked_domain_operand(
                instruction.accumulator_address_base,
                instruction.accumulator_address_strides,
                coordinate,
                "MXM accumulator-read domain accumulator address");
            if (address < 0)
                throw std::out_of_range(
                    "MXM accumulator-read domain accumulator address underflows");
            MxmControlInstruction::check_accumulator_address(
                static_cast<std::size_t>(address));
        }
        static_cast<void>(MxmControlInstruction::AccumulatorRead(
            instruction.accumulator_address_base,
            instruction.result_stream_base,
            instruction.regular_mode.accumulator_clear,
            instruction.regular_mode.accumulator_output_format,
            instruction.regular_mode.accumulator_destination));
        return;
    }
    if (instruction.opcode
        == MxmComputeDomainOperation::StreamCompute) {
        validate_mxm_compute_mode(instruction.regular_mode);
        MxmControlInstruction::check_data_format(instruction.data_format);
        MxmControlInstruction::check_is_mode(
            instruction.is_mode);
        MxmControlInstruction::check_column(
            instruction.accumulator_column);
        for (std::size_t mask = 0; mask < 8; ++mask) {
            const DomainCoordinate coordinate {{
                (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
                (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
                (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
            }};
            static_cast<void>(mxm_domain_buffer(
                instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate));
            const auto address = checked_domain_operand(
                instruction.accumulator_address_base,
                instruction.accumulator_address_strides,
                coordinate,
                "MXM decode stream-compute domain accumulator address");
            if (address < 0)
                throw std::out_of_range(
                    "MXM decode stream-compute domain accumulator address underflows");
            MxmControlInstruction::check_accumulator_address(
                static_cast<std::size_t>(address));
        }
        static_cast<void>(MxmControlInstruction::IsStreamCompute(
            instruction.weight_buffer_base,
            instruction.result_stream_base,
            instruction.data_format,
            instruction.accumulator_address_base,
            instruction.accumulator_column,
            instruction.regular_mode.accumulator_destination,
            instruction.regular_mode.accumulator_clear,
            instruction.is_mode));
        return;
    }
    if (instruction.opcode != MxmComputeDomainOperation::Compute)
        throw std::invalid_argument("MXM compute ICU 3-D opcode is invalid");
    if (instruction.loop.counts[0] > hw::kMxmRows)
        throw std::out_of_range(
            "MXM compute domain inner count exceeds one row wave");
    if (instruction.accumulator_row_stride == 0)
        throw std::invalid_argument(
            "MXM compute domain accumulator row stride must be non-zero");
    if (instruction.accumulator_address_strides[0] != 0)
        throw std::invalid_argument(
            "MXM compute domain counter 0 cannot induce the accumulator base because the MXM row wave applies row_stride");
    if (instruction.terminal_dimension
            > MxmComputeDomain::kNoTerminalDimension)
        throw std::invalid_argument(
            "MXM compute domain terminal dimension is invalid");
    validate_mxm_compute_mode(instruction.regular_mode);
    validate_mxm_compute_mode(instruction.terminal_mode);
    MxmControlInstruction::check_data_format(instruction.data_format);
    const auto rowCount = instruction.loop.counts[0];
    if (rowCount > 1 && instruction.accumulator_row_stride
        > std::numeric_limits<std::size_t>::max() / (rowCount - 1))
        throw std::overflow_error(
            "MXM compute domain accumulator row span overflows");
    const auto rowSpan = (rowCount - 1)
        * instruction.accumulator_row_stride;
    for (std::size_t mask = 0; mask < 8; ++mask) {
        const DomainCoordinate coordinate {{
            (mask & 1U) != 0 ? instruction.loop.counts[0] - 1 : 0,
            (mask & 2U) != 0 ? instruction.loop.counts[1] - 1 : 0,
            (mask & 4U) != 0 ? instruction.loop.counts[2] - 1 : 0,
        }};
        static_cast<void>(mxm_domain_buffer(
            instruction.weight_buffer_base,
            instruction.weight_buffer_mode,
            coordinate));
        const auto address = checked_domain_operand(
            instruction.accumulator_address_base,
            instruction.accumulator_address_strides,
            coordinate,
            "MXM compute domain accumulator address");
        if (address < 0)
            throw std::out_of_range(
                "MXM compute domain accumulator address underflows");
        const auto base = static_cast<std::size_t>(address);
        if (base > std::numeric_limits<std::size_t>::max() - rowSpan)
            throw std::overflow_error(
                "MXM compute domain accumulator row address overflows");
        MxmControlInstruction::check_accumulator_address(base + rowSpan);
    }
    for (const auto mode : {instruction.regular_mode,
             instruction.terminal_mode})
        static_cast<void>(MxmControlInstruction::Compute(
            instruction.weight_buffer_base,
            instruction.activation_stream_base,
            instruction.result_stream_base,
            instruction.accumulator_address_base,
            instruction.accumulator_row_stride,
            mode.accumulator_destination,
            instruction.data_format,
            mode.accumulator_clear,
            mode.accumulator_output_format));
}

inline MxmControlInstruction mxm_compute_domain_instruction(
    const MxmComputeDomain& instruction,
    const DomainCoordinate& coordinate)
{
    const auto address = checked_domain_operand(
        instruction.accumulator_address_base,
        instruction.accumulator_address_strides,
        coordinate,
        "MXM compute domain accumulator address");
    if (address < 0)
        throw std::out_of_range(
            "MXM compute domain accumulator address underflows");
    if (instruction.opcode == MxmComputeDomainOperation::AccumulatorRead)
        return MxmControlInstruction::AccumulatorRead(
            static_cast<std::size_t>(address),
            instruction.result_stream_base,
            instruction.regular_mode.accumulator_clear,
            instruction.regular_mode.accumulator_output_format,
            instruction.regular_mode.accumulator_destination);
    if (instruction.opcode
        == MxmComputeDomainOperation::StreamCompute)
        return MxmControlInstruction::IsStreamCompute(
            mxm_domain_buffer(instruction.weight_buffer_base,
                instruction.weight_buffer_mode, coordinate),
            instruction.result_stream_base,
            instruction.data_format,
            static_cast<std::size_t>(address),
            instruction.accumulator_column,
            instruction.regular_mode.accumulator_destination,
            instruction.regular_mode.accumulator_clear,
            instruction.is_mode);
    if (instruction.opcode != MxmComputeDomainOperation::Compute)
        throw std::logic_error("MXM compute ICU 3-D opcode is invalid");
    const auto mode = mxm_domain_compute_mode(instruction, coordinate);
    return MxmControlInstruction::Compute(
        mxm_domain_buffer(instruction.weight_buffer_base,
            instruction.weight_buffer_mode, coordinate),
        instruction.activation_stream_base,
        instruction.result_stream_base,
        static_cast<std::size_t>(address),
        instruction.accumulator_row_stride,
        mode.accumulator_destination,
        instruction.data_format,
        mode.accumulator_clear,
        mode.accumulator_output_format);
}

} // namespace detail

} // namespace ftlpu::compiler
