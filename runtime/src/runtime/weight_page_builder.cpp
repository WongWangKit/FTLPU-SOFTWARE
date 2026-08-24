#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include "ftlpu/core/hardware_params.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace ftlpu::software::runtime {
namespace {

using PhysicalRow = std::tuple<std::uint16_t, std::uint16_t, std::uint32_t>;

class ImageWriter {
public:
    explicit ImageWriter(const ExecutableHardwareConfig& hardware)
        : hardware_(hardware)
        , planes_(static_cast<std::size_t>(hardware.hemispheres)
              * hardware.slices_per_hemisphere)
    {
    }

    void write(std::uint16_t hemisphere, std::uint16_t slice,
        std::uint32_t row, std::uint32_t column, std::uint8_t value)
    {
        if (hemisphere >= hardware_.hemispheres
            || slice >= hardware_.slices_per_hemisphere
            || row >= hardware_.sram_depth_rows
            || column >= hardware_.bytes_per_word)
            throw std::out_of_range(
                "packed weight address is outside the executable target");
        Plane& plane = planes_[static_cast<std::size_t>(hemisphere)
                * hardware_.slices_per_hemisphere + slice];
        if (plane.rows.empty()) {
            plane.rows.resize(hardware_.sram_depth_rows);
            plane.touched.resize(hardware_.sram_depth_rows);
        }
        plane.rows[row][column] = value;
        plane.touched[row] = true;
    }

    PackedWeightImage finish() const
    {
        PackedWeightImage result;
        for (std::uint16_t hemisphere = 0;
             hemisphere < hardware_.hemispheres; ++hemisphere) {
            for (std::uint16_t slice = 0;
                 slice < hardware_.slices_per_hemisphere; ++slice) {
                const Plane& plane = planes_[
                    static_cast<std::size_t>(hemisphere)
                        * hardware_.slices_per_hemisphere + slice];
                if (plane.rows.empty()) continue;
            std::uint32_t row = 0;
            while (row < plane.rows.size()) {
                while (row < plane.rows.size() && !plane.touched[row])
                    ++row;
                if (row == plane.rows.size()) break;
                const std::uint32_t base = row;
                std::uint32_t count = 0;
                const std::uint64_t offset = result.data.size();
                while (row < plane.rows.size() && plane.touched[row]) {
                    result.data.insert(result.data.end(),
                        plane.rows[row].begin(), plane.rows[row].end());
                    ++count;
                    ++row;
                }
                result.segments.push_back(PackedWeightSegment {
                    offset, hemisphere, slice, base, count});
            }
            }
        }
        return result;
    }

private:
    struct Plane {
        std::vector<std::array<std::uint8_t, 32>> rows{};
        std::vector<bool> touched{};
    };
    const ExecutableHardwareConfig& hardware_;
    std::vector<Plane> planes_{};
};

std::uint32_t invocation_layer(
    const ModelInvocation& invocation, std::size_t fallback)
{
    constexpr std::string_view prefix = "layers.";
    const std::string_view name = invocation.name;
    if (!name.starts_with(prefix)) return static_cast<std::uint32_t>(fallback);
    const char* begin = name.data() + prefix.size();
    const char* end = begin;
    while (end != name.data() + name.size()
        && *end >= '0' && *end <= '9')
        ++end;
    std::uint32_t layer = 0;
    const auto parsed = std::from_chars(begin, end, layer);
    return parsed.ec == std::errc {} && parsed.ptr == end && begin != end
        ? layer : static_cast<std::uint32_t>(fallback);
}

std::size_t checked_dimension(
    const BinaryBinding& binding, std::size_t index)
{
    if (index >= binding.shape.size()
        || binding.shape[index] > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("weight binding has an invalid shape");
    return static_cast<std::size_t>(binding.shape[index]);
}

template <typename Fn>
void for_each_hemisphere(const BinaryBinding& binding, Fn&& fn)
{
    for (std::uint16_t hemisphere = 0; hemisphere < 2; ++hemisphere)
        if ((binding.hemisphere_mask & (1u << hemisphere)) != 0)
            fn(hemisphere);
}

const ModelTensor& find_tensor(
    const ModelPackage& package, const std::string& name)
{
    const auto tensor = std::find_if(package.tensors.begin(),
        package.tensors.end(), [&](const ModelTensor& candidate) {
            return candidate.name == name;
        });
    if (tensor == package.tensors.end())
        throw std::invalid_argument(
            "weight-page input is not a model tensor: " + name);
    return *tensor;
}

const BinaryBinding& find_input_binding(
    const BinaryProgram& program, std::uint32_t index)
{
    const auto binding = std::find_if(program.bindings.begin(),
        program.bindings.end(), [&](const BinaryBinding& candidate) {
            return candidate.access == BindingAccess::Input
                && candidate.index == index;
        });
    if (binding == program.bindings.end())
        throw std::invalid_argument(
            "invocation references a missing executable input binding");
    return *binding;
}

} // namespace

PackedWeightImage pack_weight_binding(const BinaryBinding& binding,
    std::span<const std::uint8_t> data,
    const ExecutableHardwareConfig& hardware)
{
    if (binding.role != "weight" || binding.base_row < 0
        || binding.slices.empty() || binding.address_stride <= 0
        || data.size() != binding.byte_size)
        throw std::invalid_argument(
            "weight packing requires a valid logical weight binding");
    if (hardware.bytes_per_word != hw::kPhysicalVectorBytes)
        throw std::invalid_argument(
            "weight packing currently requires 32-byte physical SRAM rows");

    ImageWriter image(hardware);
    const std::size_t rows = binding.shape.size() == 1
        ? 1 : checked_dimension(binding, 0);
    const std::size_t columns = binding.shape.size() == 1
        ? checked_dimension(binding, 0) : checked_dimension(binding, 1);
    const std::uint32_t base = static_cast<std::uint32_t>(binding.base_row);
    const std::uint32_t stride =
        static_cast<std::uint32_t>(binding.address_stride);

    if (binding.layout == BindingLayout::Vector) {
        if (binding.slices.size() != 1 || columns > hardware.bytes_per_word)
            throw std::invalid_argument("unsupported vector weight binding");
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                for_each_hemisphere(binding, [&](std::uint16_t hemisphere) {
                    image.write(hemisphere, binding.slices[0],
                        base + static_cast<std::uint32_t>(row) * stride,
                        static_cast<std::uint32_t>(column),
                        data[row * columns + column]);
                });
        return image.finish();
    }

    if (binding.layout == BindingLayout::Fp16VxmDistributed16
        && (binding.element_type == BindingElementType::F16
            || binding.element_type == BindingElementType::BF16)
        && binding.slices.size() == 16) {
        if (columns % 32 != 0 || (binding.shape.size() != 1 && rows % 32))
            throw std::invalid_argument(
                "VXM distributed16 weight must be 32-aligned");
        const std::size_t hidden_blocks = columns / 32;
        const std::size_t stored_rows = binding.shape.size() == 1 ? 32 : rows;
        for (std::size_t row = 0; row < stored_rows; ++row) {
            const std::size_t logical_row = binding.shape.size() == 1 ? 0 : row;
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t hidden_block = column / 32;
                const std::size_t feature_wave = (column % 32) / 8;
                const std::size_t feature_lane = column % 8;
                const std::uint32_t address = base
                    + static_cast<std::uint32_t>(
                        ((row / 32) * hidden_blocks + hidden_block) * 4
                        + feature_wave) * stride;
                const std::size_t offset =
                    (logical_row * columns + column) * 2;
                for_each_hemisphere(binding,
                    [&](std::uint16_t hemisphere) {
                        image.write(hemisphere,
                            binding.slices[2 * feature_lane], address,
                            static_cast<std::uint32_t>(row % 32), data[offset]);
                        image.write(hemisphere,
                            binding.slices[2 * feature_lane + 1], address,
                            static_cast<std::uint32_t>(row % 32),
                            data[offset + 1]);
                    });
            }
        }
        return image.finish();
    }
    if (binding.layout == BindingLayout::Fp16VxmRowParallel8
        && (binding.element_type == BindingElementType::F16
            || binding.element_type == BindingElementType::BF16)
        && binding.slices.size() == 16) {
        if (binding.shape.size() != 1)
            throw std::invalid_argument(
                "resident VXM row-parallel binding must be a vector");
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t offset = column * 2;
            const std::uint32_t address = base
                + static_cast<std::uint32_t>(column) * stride;
            for_each_hemisphere(binding,
                [&](std::uint16_t hemisphere) {
                    for (std::size_t pair = 0; pair < 8; ++pair) {
                        for (std::size_t lane = 0; lane < 32; ++lane) {
                            image.write(hemisphere,
                                binding.slices[2 * pair], address,
                                static_cast<std::uint32_t>(lane),
                                data[offset]);
                            image.write(hemisphere,
                                binding.slices[2 * pair + 1], address,
                                static_cast<std::uint32_t>(lane),
                                data[offset + 1]);
                        }
                    }
                });
        }
        return image.finish();
    }
    if (binding.layout == BindingLayout::Fp16VxmGammaBroadcast
        && (binding.element_type == BindingElementType::F16
            || binding.element_type == BindingElementType::BF16)
        && binding.slices.size() == 2) {
        if (binding.shape.size() != 1)
            throw std::invalid_argument(
                "VXM gamma broadcast weight must be a vector");
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t offset = column * 2;
            const std::uint32_t address = base
                + static_cast<std::uint32_t>(column) * stride;
            for_each_hemisphere(binding,
                [&](std::uint16_t hemisphere) {
                    for (std::size_t lane = 0; lane < 32; ++lane) {
                        image.write(hemisphere, binding.slices[0], address,
                            static_cast<std::uint32_t>(lane), data[offset]);
                        image.write(hemisphere, binding.slices[1], address,
                            static_cast<std::uint32_t>(lane),
                            data[offset + 1]);
                    }
                });
        }
        return image.finish();
    }

    const auto write_i8 = [&](std::size_t k, std::size_t n,
                              std::uint16_t hemisphere,
                              std::size_t slice_index,
                              std::uint32_t address) {
        image.write(hemisphere, binding.slices.at(slice_index), address,
            static_cast<std::uint32_t>(k % 32), data[k * columns + n]);
    };
    if ((binding.layout == BindingLayout::W8A16MxmWeightStriped
            || binding.layout == BindingLayout::W8A16MxmWeightWaveStriped)
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        for (std::size_t k = 0; k < rows; ++k)
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t pulse = 3 - local / 8;
                const std::size_t stream = local % 8;
                std::uint16_t hemisphere = 0;
                std::uint32_t address = base;
                if (binding.layout
                    == BindingLayout::W8A16MxmWeightWaveStriped) {
                    hemisphere = static_cast<std::uint16_t>((n / 64) % 2);
                    address += static_cast<std::uint32_t>(
                        ((n / 128) * (rows / 32) + k / 32) * 8
                        + ((n % 64) / 32) * 4 + pulse) * stride;
                } else {
                    hemisphere = static_cast<std::uint16_t>((n / 32) % 2);
                    address += static_cast<std::uint32_t>(
                        ((n / 64) * (rows / 32) + k / 32) * 4
                        + pulse) * stride;
                }
                write_i8(k, n, hemisphere, stream, address);
            }
        return image.finish();
    }

    if (binding.layout == BindingLayout::W8A16MxmWeightReplicated
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        if (rows % 32 || columns % 32)
            throw std::invalid_argument(
                "replicated W8A16 weight must be K32/N32 aligned");
        const std::size_t reduction_blocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k)
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t pulse = 3 - local / 8;
                const std::uint32_t address = base
                    + static_cast<std::uint32_t>(
                        ((n / 32) * reduction_blocks + k / 32) * 4
                        + pulse) * stride;
                for (std::uint16_t hemisphere = 0;
                     hemisphere < hardware.hemispheres; ++hemisphere)
                    write_i8(k, n, hemisphere, local % 8, address);
            }
        return image.finish();
    }

    if (binding.layout == BindingLayout::W8A16AttentionWeightStriped
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == 8) {
        if (rows % 32 || columns % 64)
            throw std::invalid_argument(
                "attention weight must be K32/N64 aligned");
        const std::size_t reduction_blocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k)
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t pulse = 3 - local / 8;
                const std::uint16_t hemisphere =
                    static_cast<std::uint16_t>((n / 64) % 2);
                const std::uint32_t address = base
                    + static_cast<std::uint32_t>(
                        ((n / 128) * reduction_blocks + k / 32) * 8
                        + ((n % 64) / 32) * 4 + pulse) * stride;
                write_i8(k, n, hemisphere, local % 8, address);
            }
        return image.finish();
    }

    if (binding.layout == BindingLayout::W8A16Block8WeightWaveStriped
        && binding.element_type == BindingElementType::I8
        && binding.slices.size() == hardware.mxms_per_hemisphere * 8) {
        const std::size_t columns_per_hemisphere =
            hardware.mxms_per_hemisphere * 32;
        const std::size_t columns_per_wave =
            hardware.hemispheres * columns_per_hemisphere;
        if (rows % 32 || columns % columns_per_wave)
            throw std::invalid_argument(
                "Block8 weight must align to the physical MXM wave");
        const std::size_t reduction_blocks = rows / 32;
        for (std::size_t k = 0; k < rows; ++k)
            for (std::size_t n = 0; n < columns; ++n) {
                const std::size_t local = n % 32;
                const std::size_t local_mxm =
                    (n % columns_per_hemisphere) / 32;
                const std::uint16_t hemisphere =
                    static_cast<std::uint16_t>(
                        (n / columns_per_hemisphere) % hardware.hemispheres);
                const std::uint32_t address = base
                    + static_cast<std::uint32_t>(
                        ((n / columns_per_wave) * reduction_blocks + k / 32)
                            * 4
                        + 3 - local / 8) * stride;
                write_i8(k, n, hemisphere,
                    local_mxm * 8 + local % 8, address);
            }
        return image.finish();
    }

    throw std::invalid_argument(
        "offline C2C packing does not support this weight layout");
}

PackedWeightImage pack_weight_binding_page(const BinaryBinding& binding,
    std::uint32_t page_index,
    std::span<const std::uint8_t> data,
    const ExecutableHardwareConfig& hardware)
{
    if (!binding.paged_weight || binding.role != "weight"
        || binding.element_type != BindingElementType::I8
        || binding.layout != BindingLayout::W8A16MxmWeightWaveStriped
        || binding.shape.size() != 2
        || data.size() != binding.byte_size
        || page_index >= binding.page_count
        || binding.page_granularity == 0
        || binding.page_items_per_slice_group == 0
        || binding.page_bank_count < 2
        || binding.page_storage_slices.empty()
        || binding.slices.size() != 8
        || binding.page_storage_slices.size() % binding.slices.size() != 0)
        throw std::invalid_argument(
            "invalid paged Vector-MXM weight binding");
    if (hardware.bytes_per_word != hw::kPhysicalVectorBytes
        || binding.page_rows > hardware.sram_depth_rows)
        throw std::invalid_argument(
            "paged weight does not fit the executable SRAM geometry");

    const std::size_t rows = checked_dimension(binding, 0);
    const std::size_t columns = checked_dimension(binding, 1);
    if (rows % 32 || columns % 128)
        throw std::invalid_argument(
            "paged Vector-MXM weight must be K32/N128 aligned");
    constexpr std::size_t loadSlices = 8;
    constexpr std::size_t logicalSlots = 2;
    constexpr std::size_t rowsPerLoad = 4;
    const bool projection = rows < columns;
    const std::size_t reductionBlocks = rows / 32;
    const std::size_t outputWaves = columns / 128;
    const std::size_t pagesPerOutputWave = projection ? 0
        : (reductionBlocks + binding.page_granularity - 1)
            / binding.page_granularity;
    ImageWriter image(hardware);

    for (std::size_t k = 0; k < rows; ++k) {
        const std::size_t reduction = k / 32;
        for (std::size_t n = 0; n < columns; ++n) {
            const std::size_t local = n % 32;
            const std::size_t pulse = 3 - local / 8;
            const std::size_t stream = local % 8;
            const std::uint16_t hemisphere =
                static_cast<std::uint16_t>((n / 64) % 2);
            std::size_t logicalPage = 0;
            std::size_t itemInPage = 0;
            std::size_t localItem = 0;
            std::size_t slot = 0;
            if (projection) {
                const std::size_t pair =
                    (n / 128) * 2 + ((n % 64) / 32);
                logicalPage = pair / binding.page_granularity;
                itemInPage = pair % binding.page_granularity;
                localItem = itemInPage
                    % binding.page_items_per_slice_group;
                slot = localItem % logicalSlots;
            } else {
                const std::size_t outputWave = n / 128;
                logicalPage = outputWave * pagesPerOutputWave
                    + reduction / binding.page_granularity;
                itemInPage = reduction % binding.page_granularity;
                localItem = itemInPage
                    % binding.page_items_per_slice_group;
                slot = (n % 64) / 32;
            }
            if (logicalPage != page_index) continue;
            const std::size_t sliceGroup =
                binding.page_role_group_base
                + itemInPage / binding.page_items_per_slice_group;
            if (sliceGroup >= binding.page_role_group_base
                    + binding.page_role_group_count
                || sliceGroup * loadSlices + stream
                    >= binding.page_storage_slices.size())
                throw std::out_of_range(
                    "paged weight slice group is outside its binding");
            const std::uint16_t slice = binding.page_storage_slices[
                sliceGroup * loadSlices + stream];
            const std::size_t address = projection
                ? ((localItem / logicalSlots) * reductionBlocks
                      + reduction)
                        * logicalSlots * rowsPerLoad
                    + slot * rowsPerLoad + pulse
                : localItem * logicalSlots * rowsPerLoad
                    + slot * rowsPerLoad + pulse;
            if (address >= binding.page_rows)
                throw std::out_of_range(
                    "paged weight row is outside its SRAM page");
            image.write(hemisphere, slice,
                static_cast<std::uint32_t>(binding.base_row
                    + address * binding.address_stride),
                static_cast<std::uint32_t>(k % 32),
                data[k * columns + n]);
        }
    }
    return image.finish();
}

void build_weight_pages(
    ModelPackage& package, const WeightPageBuildOptions& options)
{
    if (!package.weight_pages.empty())
        throw std::invalid_argument(
            "model package already contains weight pages");
    std::vector<ModelTensor> packed_tensors;
    std::unordered_set<std::string> replaced_tensors;

    for (std::size_t invocation_index = 0;
         invocation_index < package.invocations.size(); ++invocation_index) {
        ModelInvocation& invocation = package.invocations[invocation_index];
        const BinaryProgram program = materialize_model_executable(
            package.executables.at(invocation.executable_index));
        const std::uint16_t bank = static_cast<std::uint16_t>(
            (options.first_bank + package.weight_pages.size())
            % program.hardware.banks_per_slice);
        ModelWeightPage page;
        page.layer = invocation_layer(invocation, invocation_index);
        page.bank = bank;
        std::map<PhysicalRow, std::string> owners;

        for (ModelBindingRef& input : invocation.inputs) {
            const BinaryBinding& binding =
                find_input_binding(program, input.binding_index);
            if (binding.role != "weight") continue;
            if (binding.bank != bank)
                throw std::invalid_argument(
                    "executable weight bank does not match alternating page bank: "
                    + invocation.name + " binding "
                    + std::to_string(binding.index));
            const ModelTensor& logical = find_tensor(package, input.value);
            const PackedWeightImage image = pack_weight_binding(
                binding, logical.data, program.hardware);
            const std::string packed_name = logical.name + ".sram_page."
                + std::to_string(invocation_index);
            ModelTensor packed = logical;
            packed.name = packed_name;
            packed.element_type = BindingElementType::I8;
            packed.shape = {image.data.size()};
            packed.data = image.data;
            packed.encoding = ModelTensorEncoding::TargetPackedSramVectors;

            for (const PackedWeightSegment& source : image.segments) {
                for (std::uint32_t row = 0; row < source.vector_count; ++row) {
                    const PhysicalRow address {source.hemisphere,
                        source.slice, source.base_row + row};
                    const auto [owner, inserted] =
                        owners.emplace(address, logical.name);
                    if (!inserted && owner->second != logical.name)
                        throw std::invalid_argument(
                            "page-resident weights overlap at hemisphere="
                            + std::to_string(source.hemisphere) + " slice="
                            + std::to_string(source.slice) + " bank="
                            + std::to_string(bank) + " row="
                            + std::to_string(source.base_row + row) + ": "
                            + owner->second + " and " + logical.name);
                }
                page.segments.push_back(ModelWeightPage::Segment {
                    packed_name, source.byte_offset, source.hemisphere,
                    source.slice, source.base_row, source.vector_count,
                    static_cast<std::uint16_t>(page.segments.size()
                        % hw::kStreamsPerDirection)});
            }
            page.tensors.push_back(packed_name);
            input.value = packed_name;
            replaced_tensors.insert(logical.name);
            packed_tensors.push_back(std::move(packed));
        }
        if (!page.tensors.empty()) {
            invocation.weight_page =
                static_cast<std::uint32_t>(package.weight_pages.size());
            package.weight_pages.push_back(std::move(page));
        }
    }

    if (options.remove_logical_weights) {
        std::erase_if(package.tensors, [&](const ModelTensor& tensor) {
            return replaced_tensors.contains(tensor.name);
        });
    }
    package.tensors.insert(package.tensors.end(),
        std::make_move_iterator(packed_tensors.begin()),
        std::make_move_iterator(packed_tensors.end()));
    validate_model_package(package);
}

} // namespace ftlpu::software::runtime
