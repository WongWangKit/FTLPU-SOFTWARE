#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/system/tsp_slice_system.hpp"

// Keep this translation unit rebuilt when BinaryProgram ABI evolves.
#include <bit>
#include <cstdlib>
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

} // namespace

CModelRuntime::CModelRuntime(TspSliceSystem& system)
    : system_(system)
{
}

void CModelRuntime::load(const BinaryProgram& program)
{
    load_queue_programs_into_icu(program.queues, system_.icu());
    loaded_max_cycle_ = program.max_cycle;
    bindings_ = program.bindings;
    for (const BinaryBinding& binding : bindings_) {
        if (binding.access != BindingAccess::Internal
            || binding.layout != BindingLayout::Fp32CausalMaskTile)
            continue;
        require_matrix_binding(binding);
        if (binding.element_type != BindingElementType::F32
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
                for (std::size_t byte = 0; byte < binding.slices.size(); ++byte) {
                    for_each_binding_hemisphere(binding, [&](Hemisphere hemisphere) {
                        write_sram_byte(system_, hemisphere, binding.slices[byte],
                            address, query_lane,
                            static_cast<std::uint8_t>(bits >> (8 * byte)));
                    });
                }
            }
        }
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
    const auto& binding = find_binding(BindingAccess::Input, index);
    require_matrix_binding(binding);
    if (data.size() != binding.byte_size)
        throw std::invalid_argument("input byte size or element type does not match binding");
    const std::size_t rows = static_cast<std::size_t>(binding.shape[0]);
    const std::size_t columns = static_cast<std::size_t>(binding.shape[1]);
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
        && binding.element_type == BindingElementType::F16
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
        && binding.element_type == BindingElementType::F16
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
