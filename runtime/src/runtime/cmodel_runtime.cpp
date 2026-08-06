#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

// Keep this translation unit rebuilt when BinaryProgram ABI evolves.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ftlpu::software::runtime {
namespace {

void write_sram_byte(TspSliceSystem& system, Hemisphere hemisphere, std::size_t slice,
    std::size_t address, std::size_t column, std::uint8_t value)
{
    system.initialize_mem_sram_lane_byte(hemisphere, slice,
        column / hw::kLanesPerTile, address, column % hw::kLanesPerTile, value);
}

std::uint8_t read_sram_byte(const TspSliceSystem& system, Hemisphere hemisphere,
    std::size_t slice,
    std::size_t address, std::size_t column)
{
    return system.read_mem_sram_lane_byte(hemisphere, slice,
        column / hw::kLanesPerTile, address, column % hw::kLanesPerTile);
}

template <typename Fn>
void for_each_binding_hemisphere(const BinaryBinding& binding, Fn&& fn)
{
    if ((binding.hemisphere_mask & 1) != 0) fn(Hemisphere::East);
    if ((binding.hemisphere_mask & 2) != 0) fn(Hemisphere::West);
}

void require_matrix_binding(const BinaryBinding& binding)
{
    if (binding.shape.size() != 2 || binding.base_row < 0
        || binding.address_stride == 0 || binding.slices.empty())
        throw std::logic_error("runtime currently requires a valid rank-2 physical binding");
}

bool is_16bit_float(BindingElementType type)
{
    return type == BindingElementType::F16
        || type == BindingElementType::BF16;
}

std::uint16_t encode_16bit_float(float value, BindingElementType type)
{
    if (type == BindingElementType::BF16)
        return Bf16::from_float(value).bits();
    if (type == BindingElementType::F16)
        return Fp16::from_float(value).bits();
    throw std::logic_error("binding is not a supported 16-bit float");
}

std::uint16_t read_16bit_element(const TspSliceSystem& system,
    const BinaryBinding& binding, std::size_t row, std::size_t column,
    std::size_t mxms_per_hemisphere)
{
    const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && binding.slices.size() == 4) {
        const std::size_t local_mxm = (column % 64) / 32;
        const bool dual_hemisphere = (binding.hemisphere_mask & 3) == 3;
        const auto hemisphere = dual_hemisphere
            ? static_cast<Hemisphere>((column / 64) % 2)
            : Hemisphere::East;
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (dual_hemisphere ? column / 128 : column / 64) * rows + row;
        return static_cast<std::uint16_t>(read_sram_byte(system, hemisphere,
                   binding.slices[local_mxm * 2], address, column % 32))
            | (static_cast<std::uint16_t>(read_sram_byte(system, hemisphere,
                   binding.slices[local_mxm * 2 + 1], address, column % 32))
                << 8);
    }
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && binding.slices.size() == 2) {
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (column / 32) * rows + row;
        return static_cast<std::uint16_t>(read_sram_byte(system,
                   Hemisphere::East, binding.slices[0], address, column % 32))
            | (static_cast<std::uint16_t>(read_sram_byte(system,
                   Hemisphere::East, binding.slices[1], address, column % 32))
                << 8);
    }
    if ((binding.layout == BindingLayout::Fp16MxmDistributed16
            || binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16)
        && binding.slices.size() == 16) {
        const std::size_t hidden_blocks = columns / 32;
        const std::size_t token_block = row / 32;
        const std::size_t token_wave = (row % 32) / 8;
        const std::size_t token_lane = row % 8;
        const std::size_t hidden_block = column / 32;
        const std::size_t feature_wave = (column % 32) / 8;
        const std::size_t feature_lane = column % 8;
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (token_block * hidden_blocks + hidden_block) * 4 + token_wave;
        const auto hemisphere = binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16
            ? static_cast<Hemisphere>(
                (hidden_block
                    % (hw::kHemispheres * mxms_per_hemisphere))
                / mxms_per_hemisphere)
            : Hemisphere::East;
        return static_cast<std::uint16_t>(read_sram_byte(system,
                   hemisphere, binding.slices[2 * token_lane], address,
                   feature_wave * 8 + feature_lane))
            | (static_cast<std::uint16_t>(read_sram_byte(system,
                   hemisphere, binding.slices[2 * token_lane + 1],
                   address, feature_wave * 8 + feature_lane))
                << 8);
    }
    throw std::logic_error(
        "device copy does not support the source 16-bit float layout");
}

void write_16bit_element(TspSliceSystem& system,
    const BinaryBinding& binding, std::size_t row, std::size_t column,
    std::uint16_t value)
{
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
    if (binding.layout == BindingLayout::Fp16MxmDistributed16
        && binding.slices.size() == 16) {
        const std::size_t hidden_blocks = columns / 32;
        const std::size_t token_block = row / 32;
        const std::size_t token_wave = (row % 32) / 8;
        const std::size_t token_lane = row % 8;
        const std::size_t hidden_block = column / 32;
        const std::size_t feature_wave = (column % 32) / 8;
        const std::size_t feature_lane = column % 8;
        const std::size_t address = static_cast<std::size_t>(binding.base_row)
            + (token_block * hidden_blocks + hidden_block) * 4 + token_wave;
        for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
            write_sram_byte(system, hemisphere,
                binding.slices[2 * token_lane], address,
                feature_wave * 8 + feature_lane,
                static_cast<std::uint8_t>(value));
            write_sram_byte(system, hemisphere,
                binding.slices[2 * token_lane + 1], address,
                feature_wave * 8 + feature_lane,
                static_cast<std::uint8_t>(value >> 8));
        });
        return;
    }
    throw std::logic_error(
        "device copy does not support the destination 16-bit float layout");
}

} // namespace

CModelRuntime::CModelRuntime(TspSliceSystem& system)
    : system_(system)
{
}

void CModelRuntime::load(const BinaryProgram& program)
{
    const bool cmodelLargeSram =
        program.target_name == "lpu32-cmodel-large-sram";
    const std::uint64_t compatibleAbi =
        lpu_32stream_target_abi(program.mxms_per_hemisphere);
    if (program.target_abi != compatibleAbi
        && !cmodelLargeSram) {
        std::ostringstream message;
        message << "CModel target ABI mismatch: binary target '"
                << program.target_name << "' has 0x" << std::hex
                << program.target_abi << ", runtime requires '"
                << kLpu32StreamTargetName << "' ABI 0x"
                << compatibleAbi;
        throw std::invalid_argument(message.str());
    }
    system_.reset_execution_state();
    load_queue_programs_into_icu(program.queues, system_.icu(),
        program.mxms_per_hemisphere);
    loaded_max_cycle_ = program.max_cycle;
    loaded_mxms_per_hemisphere_ = program.mxms_per_hemisphere;
    bindings_ = program.bindings;
    for (const BinaryBinding& binding : bindings_) {
        if (binding.access != BindingAccess::Internal) continue;
        if (binding.initializer == BindingInitializer::None) continue;
        if (binding.initializer == BindingInitializer::Zero) {
            if (binding.base_row < 0 || binding.instruction_count <= 0)
                throw std::logic_error("invalid zero-initialized internal binding");
            for (std::int64_t row = 0; row < binding.instruction_count; ++row)
                for (std::uint16_t slice : binding.slices)
                    for (std::size_t column = 0; column < 32; ++column)
                        for_each_binding_hemisphere(binding,
                            [&](Hemisphere hemisphere) {
                                write_sram_byte(system_, hemisphere, slice,
                                    static_cast<std::size_t>(binding.base_row + row),
                                    column, 0);
                            });
            continue;
        }
        if (binding.initializer == BindingInitializer::CausalMask) {
            require_matrix_binding(binding);
            if (binding.layout != BindingLayout::Fp32CausalMaskTile
                || binding.element_type != BindingElementType::F32
                || binding.slices.size() != sizeof(float)
                || binding.shape[0] + 1 != binding.shape[1])
                throw std::logic_error("invalid internal causal-mask binding");
            const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
            const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
            const std::size_t stride =
                static_cast<std::size_t>(std::abs(binding.address_stride));
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t local_key = row + 1;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row) + row * stride;
                for (std::size_t query_lane = 0; query_lane < columns; ++query_lane) {
                    const float mask = local_key <= query_lane ? 0.0f : -1.0e9f;
                    const std::uint32_t bits = std::bit_cast<std::uint32_t>(mask);
                    for (std::size_t byte = 0; byte < binding.slices.size(); ++byte)
                        for_each_binding_hemisphere(binding,
                            [&](Hemisphere hemisphere) {
                                write_sram_byte(system_, hemisphere,
                                    binding.slices[byte], address, query_lane,
                                    static_cast<std::uint8_t>(bits >> (8 * byte)));
                            });
                }
            }
            continue;
        }
        if (binding.initializer == BindingInitializer::RopeTable) {
            if (binding.layout != BindingLayout::Fp16RopeTable
                || !is_16bit_float(binding.element_type)
                || binding.shape.size() != 3 || binding.shape[2] != 2
                || binding.slices.size() != 4 || binding.base_row < 0
                || binding.address_stride == 0
                || binding.rope_head_dim != 2 * binding.shape[1]
                || !std::isfinite(binding.rope_theta)
                || binding.rope_theta <= 1.0f)
                throw std::logic_error("invalid internal RoPE-table binding");
            const std::size_t sequence =
                static_cast<std::size_t>(binding.shape[0]);
            const std::size_t dimensions =
                static_cast<std::size_t>(binding.shape[1]);
            const std::size_t stride =
                static_cast<std::size_t>(std::abs(binding.address_stride));
            for (std::size_t token = 0; token < sequence; ++token) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + token * stride;
                for (std::size_t dimension = 0;
                     dimension < dimensions; ++dimension) {
                    const float inverse = 1.0f / std::pow(
                        binding.rope_theta,
                        static_cast<float>(2 * dimension)
                            / binding.rope_head_dim);
                    const float angle = static_cast<float>(token) * inverse;
                    const auto cosine = encode_16bit_float(
                        std::cos(angle), binding.element_type);
                    const auto sine = encode_16bit_float(
                        std::sin(angle), binding.element_type);
                    for_each_binding_hemisphere(binding,
                        [&](Hemisphere hemisphere) {
                            write_sram_byte(system_, hemisphere,
                                binding.slices[0], address, dimension,
                                static_cast<std::uint8_t>(cosine));
                            write_sram_byte(system_, hemisphere,
                                binding.slices[1], address, dimension,
                                static_cast<std::uint8_t>(cosine >> 8));
                            write_sram_byte(system_, hemisphere,
                                binding.slices[2], address, dimension,
                                static_cast<std::uint8_t>(sine));
                            write_sram_byte(system_, hemisphere,
                                binding.slices[3], address, dimension,
                                static_cast<std::uint8_t>(sine >> 8));
                        });
                }
            }
            continue;
        }
        throw std::logic_error("unsupported internal binding initializer");
    }
}

const BinaryBinding& CModelRuntime::find_binding(BindingAccess access, std::size_t index) const
{
    for (const auto& binding : bindings_) {
        if (binding.access == access && binding.index == index) return binding;
    }
    throw std::out_of_range("FTLPU binary does not contain the requested runtime binding");
}

void CModelRuntime::upload_input(std::size_t index, std::span<const std::uint8_t> data)
{
    upload_binding(find_binding(BindingAccess::Input, index), data);
}

void CModelRuntime::upload_binding(
    const BinaryBinding& binding, std::span<const std::uint8_t> data)
{
    if ((binding.shape.empty() || binding.shape.size() > 3)
        || binding.base_row < 0 || binding.address_stride == 0
        || binding.slices.empty())
        throw std::logic_error(
            "runtime requires a valid rank-1 through rank-3 physical binding");
    if (data.size() != binding.byte_size)
        throw std::invalid_argument("input byte size or element type does not match binding");
    const bool vector = binding.shape.size() == 1;
    const std::size_t rows =
        vector ? 1 : static_cast<std::size_t>(binding.shape[0]);
    std::size_t columns = static_cast<std::size_t>(
        vector ? binding.shape[0] : binding.shape[1]);
    if (binding.shape.size() == 3)
        columns *= static_cast<std::size_t>(binding.shape[2]);
    const std::size_t row_stride = static_cast<std::size_t>(std::abs(binding.address_stride));

    if (binding.layout == BindingLayout::Vector) {
        if (binding.slices.size() != 1)
            throw std::logic_error("vector binding requires exactly one MEM slice");
        for (std::size_t row = 0; row < rows; ++row) {
            const auto address = static_cast<std::size_t>(binding.base_row) + row * row_stride;
            for (std::size_t column = 0; column < columns; ++column)
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_sram_byte(system_, hemisphere, binding.slices[0], address,
                        column, data[row * columns + column]);
                });
        }
        return;
    }

    if (binding.layout == BindingLayout::MxmWeightStriped) {
        constexpr std::size_t mxm_k = 320;
        const std::size_t column_blocks =
            (columns + binding.slices.size() - 1) / binding.slices.size();
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto slice = binding.slices[column % binding.slices.size()];
                const auto address = static_cast<std::size_t>(binding.base_row)
                    + ((k / mxm_k) * column_blocks
                        + column / binding.slices.size()) * row_stride;
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_sram_byte(system_, hemisphere, slice, address, k % mxm_k,
                        data[k * columns + column]);
                });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16MxmActivationPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 4) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t k = 0; k < columns; ++k) {
                const std::size_t address = static_cast<std::size_t>(binding.base_row)
                    + (k / 32) * rows + row;
                const std::size_t offset = (row * columns + k) * 2;
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_sram_byte(system_, hemisphere, binding.slices[0], address,
                        k % 32, data[offset]);
                    write_sram_byte(system_, hemisphere, binding.slices[1], address,
                        k % 32, data[offset + 1]);
                    write_sram_byte(system_, hemisphere, binding.slices[2], address,
                        k % 32, data[offset]);
                    write_sram_byte(system_, hemisphere, binding.slices[3], address,
                        k % 32, data[offset + 1]);
                });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 2) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (column / 32) * rows + row;
                const std::size_t offset = (row * columns + column) * 2;
                for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                    write_sram_byte(system_, hemisphere, binding.slices[0],
                        address, column % 32, data[offset]);
                    write_sram_byte(system_, hemisphere, binding.slices[1],
                        address, column % 32, data[offset + 1]);
                });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16SxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (vector) {
            if (columns % 32 != 0)
                throw std::logic_error(
                    "distributed16 16-bit float vector requires "
                    "32-aligned length");
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hiddenBlock = column / 32;
                const std::size_t featureWave = (column % 32) / 8;
                const std::size_t featureLane = column % 8;
                const std::size_t offset = column * 2;
                for (std::size_t tokenWave = 0;
                     tokenWave < 4; ++tokenWave) {
                    const std::size_t destinationTile =
                        (tokenWave + 4 - featureWave) % 4;
                    const std::size_t address =
                        static_cast<std::size_t>(binding.base_row)
                        + hiddenBlock * 4 + tokenWave;
                    for_each_binding_hemisphere(
                        binding, [&](Hemisphere hemisphere) {
                            for (std::size_t lane = 0; lane < 8; ++lane) {
                                write_sram_byte(system_, hemisphere,
                                    binding.slices[2 * featureLane],
                                    address, destinationTile * 8 + lane,
                                    data[offset]);
                                write_sram_byte(system_, hemisphere,
                                    binding.slices[2 * featureLane + 1],
                                    address, destinationTile * 8 + lane,
                                    data[offset + 1]);
                            }
                        });
                }
            }
            return;
        }
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "distributed16 16-bit float input requires "
                "32-aligned dimensions");
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t token_wave = (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t destination_tile =
                    (token_wave + 4 - feature_wave) % 4;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                for_each_binding_hemisphere(
                    binding, [&](Hemisphere hemisphere) {
                        write_sram_byte(system_, hemisphere,
                            binding.slices[2 * feature_lane], address,
                            destination_tile * 8 + token_lane,
                            data[offset]);
                        write_sram_byte(system_, hemisphere,
                            binding.slices[2 * feature_lane + 1], address,
                            destination_tile * 8 + token_lane,
                            data[offset + 1]);
                    });
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::Fp16VxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (columns % 32 != 0 || (!vector && rows % 32 != 0))
            throw std::logic_error(
                "VXM distributed16 16-bit float input requires "
                "32-aligned dimensions");
        const std::size_t hidden_blocks = columns / 32;
        const std::size_t stored_rows = vector ? 32 : rows;
        for (std::size_t row = 0; row < stored_rows; ++row) {
            const std::size_t logical_row = vector ? 0 : row;
            const std::size_t token_block = row / 32;
            const std::size_t lane = row % 32;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + feature_wave;
                const std::size_t offset =
                    (logical_row * columns + column) * 2;
                for_each_binding_hemisphere(
                    binding, [&](Hemisphere hemisphere) {
                        write_sram_byte(system_, hemisphere,
                            binding.slices[2 * feature_lane], address,
                            lane, data[offset]);
                        write_sram_byte(system_, hemisphere,
                            binding.slices[2 * feature_lane + 1], address,
                            lane, data[offset + 1]);
                    });
            }
        }
        return;
    }
    if ((binding.layout == BindingLayout::Fp16MxmDistributed16
            || binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16)
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "MXM distributed16 16-bit float input requires "
                "32-aligned dimensions");
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t token_wave = (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                for_each_binding_hemisphere(
                    binding, [&](Hemisphere hemisphere) {
                        write_sram_byte(system_, hemisphere,
                            binding.slices[2 * token_lane], address,
                            feature_wave * 8 + feature_lane, data[offset]);
                        write_sram_byte(system_, hemisphere,
                            binding.slices[2 * token_lane + 1], address,
                            feature_wave * 8 + feature_lane, data[offset + 1]);
                    });
            }
        }
        return;
    }
    if ((binding.layout == BindingLayout::W8A16MxmWeightStriped
            || binding.layout == BindingLayout::W8A16MxmWeightWaveStriped)
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t pulse = 3 - local / 8;
                const std::size_t stream = local % 8;
                std::size_t address = static_cast<std::size_t>(binding.base_row);
                Hemisphere hemisphere = Hemisphere::East;
                if (binding.layout == BindingLayout::W8A16MxmWeightWaveStriped) {
                    hemisphere = static_cast<Hemisphere>((n / 64) % 2);
                    address += ((n / 128) * (rows / 32) + k / 32) * 8
                        + ((n % 64) / 32) * 4 + pulse;
                } else {
                    hemisphere = ((n / 32) % 2) == 0
                        ? Hemisphere::East : Hemisphere::West;
                    address += ((n / 64) * (rows / 32) + k / 32) * 4 + pulse;
                }
                write_sram_byte(system_, hemisphere, binding.slices[stream], address,
                    k % 32, data[k * columns + n]);
            }
        }
        return;
    }
    if (binding.layout
            == BindingLayout::W8A16Block8WeightWaveStriped
        && binding.element_type == BindingElementType::I8
        && binding.slices.size()
            == hw::kMxmsPerHemisphere * 8) {
        const std::size_t columnsPerHemisphere =
            hw::kMxmsPerHemisphere * 32;
        const std::size_t columnsPerWave =
            hw::kHemispheres * columnsPerHemisphere;
        if (rows % 32 != 0 || columns % columnsPerWave != 0)
            throw std::logic_error(
                "Block8 weight binding must align to the physical MXM wave");
        const std::size_t reductionBlocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t localColumn = n % 32;
                const std::size_t pulse = 3 - localColumn / 8;
                const std::size_t localMxm =
                    (n % columnsPerHemisphere) / 32;
                const std::size_t stream = localColumn % 8;
                const auto hemisphere =
                    static_cast<Hemisphere>(
                        (n / columnsPerHemisphere) % hw::kHemispheres);
                const std::size_t wave = n / columnsPerWave;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (wave * reductionBlocks + k / 32) * 4
                    + pulse;
                write_sram_byte(system_, hemisphere,
                    binding.slices[localMxm * 8 + stream],
                    address, k % 32, data[k * columns + n]);
            }
        }
        return;
    }
    if (binding.layout == BindingLayout::W8A16AttentionWeightStriped
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        if (rows % 32 || columns % 64)
            throw std::logic_error("attention weight binding must be K32/N64 aligned");
        const std::size_t reduction_blocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local_column = n % 32;
                const std::size_t pulse = 3 - local_column / 8;
                const std::size_t stream = local_column % 8;
                const auto hemisphere = static_cast<Hemisphere>((n / 64) % 2);
                const std::size_t head_group = n / 128;
                const std::size_t local_mxm = (n % 64) / 32;
                const std::size_t address = static_cast<std::size_t>(binding.base_row)
                    + (head_group * reduction_blocks + k / 32) * 8
                    + local_mxm * 4 + pulse;
                write_sram_byte(system_, hemisphere, binding.slices[stream], address,
                    k % 32, data[k * columns + n]);
            }
        }
        return;
    }
    throw std::logic_error("unsupported input binding layout");
}

std::vector<std::uint8_t> CModelRuntime::download_output(std::size_t index) const
{
    const auto& binding = find_binding(BindingAccess::Output, index);
    require_matrix_binding(binding);
    const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
    const std::size_t row_stride = static_cast<std::size_t>(std::abs(binding.address_stride));
    if (binding.layout == BindingLayout::Fp16MxmActivationPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 4) {
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (column / 32) * rows + row;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_sram_byte(system_, Hemisphere::East,
                    binding.slices[0], address, column % 32);
                result[offset + 1] = read_sram_byte(system_, Hemisphere::East,
                    binding.slices[1], address, column % 32);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Vector
        && binding.element_type == BindingElementType::I8 && binding.slices.size() == 1) {
        std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            const auto address = static_cast<std::size_t>(binding.base_row) + row * row_stride;
            for (std::size_t column = 0; column < columns; ++column)
                result[row * columns + column] =
                    read_sram_byte(system_, Hemisphere::East,
                        binding.slices[0], address, column);
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16PairPlanar
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 4) {
        std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t local_mxm = (column % 64) / 32;
                const bool dual_hemisphere = (binding.hemisphere_mask & 3) == 3;
                const auto hemisphere = dual_hemisphere
                    ? static_cast<Hemisphere>((column / 64) % 2)
                    : Hemisphere::East;
                const std::size_t address = static_cast<std::size_t>(binding.base_row)
                    + (dual_hemisphere ? column / 128 : column / 64) * rows + row;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_sram_byte(system_, hemisphere,
                    binding.slices[local_mxm * 2], address, column % 32);
                result[offset + 1] = read_sram_byte(system_, hemisphere,
                    binding.slices[local_mxm * 2 + 1], address, column % 32);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16SxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "distributed16 16-bit float output requires "
                "32-aligned dimensions");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t token_wave = (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t source_tile =
                    (token_wave + 4 - feature_wave) % 4;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_sram_byte(system_,
                    Hemisphere::East,
                    binding.slices[2 * feature_lane], address,
                    source_tile * 8 + token_lane);
                result[offset + 1] = read_sram_byte(system_,
                    Hemisphere::East,
                    binding.slices[2 * feature_lane + 1], address,
                    source_tile * 8 + token_lane);
            }
        }
        return result;
    }
    if (binding.layout == BindingLayout::Fp16VxmDistributed16
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "VXM distributed16 16-bit float output requires "
                "32-aligned dimensions");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t lane = row % 32;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + feature_wave;
                const std::size_t offset = (row * columns + column) * 2;
                result[offset] = read_sram_byte(system_, Hemisphere::East,
                    binding.slices[2 * feature_lane], address, lane);
                result[offset + 1] = read_sram_byte(system_, Hemisphere::East,
                    binding.slices[2 * feature_lane + 1], address, lane);
            }
        }
        return result;
    }
    if ((binding.layout == BindingLayout::Fp16MxmDistributed16
            || binding.layout
                == BindingLayout::Fp16MxmBlock8Distributed16)
        && is_16bit_float(binding.element_type)
        && binding.slices.size() == 16) {
        if (rows % 32 != 0 || columns % 32 != 0)
            throw std::logic_error(
                "MXM distributed16 16-bit float output requires "
                "32-aligned dimensions");
        std::vector<std::uint8_t> result(
            static_cast<std::size_t>(binding.byte_size));
        const std::size_t hidden_blocks = columns / 32;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t token_block = row / 32;
            const std::size_t token_wave = (row % 32) / 8;
            const std::size_t token_lane = row % 8;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::size_t address =
                    static_cast<std::size_t>(binding.base_row)
                    + (token_block * hidden_blocks + hidden_block) * 4
                    + token_wave;
                const std::size_t offset = (row * columns + column) * 2;
                const auto hemisphere = binding.layout
                        == BindingLayout::Fp16MxmBlock8Distributed16
                    ? static_cast<Hemisphere>(
                        (hidden_block
                            % (hw::kHemispheres
                                * loaded_mxms_per_hemisphere_))
                        / loaded_mxms_per_hemisphere_)
                    : Hemisphere::East;
                result[offset] = read_sram_byte(system_, hemisphere,
                    binding.slices[2 * token_lane], address,
                    feature_wave * 8 + feature_lane);
                result[offset + 1] = read_sram_byte(system_, hemisphere,
                    binding.slices[2 * token_lane + 1], address,
                    feature_wave * 8 + feature_lane);
            }
        }
        return result;
    }
    if (binding.layout != BindingLayout::Int32BytePlanar
        || binding.element_type != BindingElementType::I32 || binding.slices.size() != 4)
        throw std::logic_error("unsupported output binding layout");
    std::vector<std::uint8_t> result(static_cast<std::size_t>(binding.byte_size));
    for (std::size_t row = 0; row < rows; ++row) {
        const auto address = static_cast<std::size_t>(binding.base_row) + row * row_stride;
        for (std::size_t column = 0; column < columns; ++column) {
            const auto offset = (row * columns + column) * 4;
            for (std::size_t byte = 0; byte < 4; ++byte)
                result[offset + byte] = read_sram_byte(
                    system_, Hemisphere::East, binding.slices[byte], address, column);
        }
    }
    return result;
}

void CModelRuntime::copy_binding(
    const BinaryBinding& source, const BinaryBinding& destination)
{
    require_matrix_binding(source);
    require_matrix_binding(destination);
    if (!is_16bit_float(source.element_type)
        || source.element_type != destination.element_type
        || source.shape != destination.shape
        || source.byte_size != destination.byte_size)
        throw std::invalid_argument(
            "device binding copy requires matching 16-bit float tensors");
    if (source.layout == destination.layout
        && source.base_row == destination.base_row
        && source.slices == destination.slices
        && source.hemisphere_mask == destination.hemisphere_mask)
        return;

    const std::size_t rows = static_cast<std::size_t>(source.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(source.shape[1]);
    std::vector<std::uint16_t> staging(rows * columns);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            staging[row * columns + column] =
                read_16bit_element(system_, source, row, column,
                    loaded_mxms_per_hemisphere_);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            write_16bit_element(system_, destination, row, column,
                staging[row * columns + column]);
}

void CModelRuntime::load_file(const std::filesystem::path& path)
{
    load(read_binary_program(path));
}

void CModelRuntime::dispatch_icu_cycles(std::size_t cycles, std::ostream* log)
{
    const auto count = cycles == 0 ? loaded_max_cycle_ + 1 : cycles;
    for (std::size_t cycle = 0; cycle < count; ++cycle) {
        if (log != nullptr) system_.tick(*log);
        else system_.tick({});
    }
}

void CModelRuntime::run_cycles(std::size_t cycles, std::ostream* log)
{
    const auto count = cycles == 0 ? loaded_max_cycle_ + 1 : cycles;
    auto sinks = TspSliceSystem::LogSinks {};
    if (log != nullptr) sinks = TspSliceSystem::LogSinks {log, log, log, log, log};
    for (std::size_t cycle = 0; cycle < count; ++cycle) {
        try {
            system_.tick(sinks);
        } catch (const std::exception& ex) {
            std::ostringstream message;
            message << "CModel cycle " << cycle << ": " << ex.what();
            throw std::logic_error(message.str());
        }
    }
}

} // namespace ftlpu::software::runtime
