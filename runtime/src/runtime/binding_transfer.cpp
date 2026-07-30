#include "ftlpu/software/runtime/binding_transfer.hpp"

#include "ftlpu/mem/address.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ftlpu::software::runtime {
namespace {

std::string_view layout_name(BindingLayout layout)
{
    switch (layout) {
    case BindingLayout::Vector: return "Vector";
    case BindingLayout::MxmWeightStriped: return "MxmWeightStriped";
    case BindingLayout::Int32BytePlanar: return "Int32BytePlanar";
    case BindingLayout::Fp16BytePlanar: return "Fp16BytePlanar";
    case BindingLayout::Fp16MxmActivationPlanar:
        return "Fp16MxmActivationPlanar";
    case BindingLayout::W8A16MxmWeightStriped:
        return "W8A16MxmWeightStriped";
    case BindingLayout::Fp16PairPlanar: return "Fp16PairPlanar";
    case BindingLayout::W8A16AttentionWeightStriped:
        return "W8A16AttentionWeightStriped";
    case BindingLayout::W8A16MxmWeightWaveStriped:
        return "W8A16MxmWeightWaveStriped";
    case BindingLayout::Fp32CausalMaskTile: return "Fp32CausalMaskTile";
    case BindingLayout::Fp16SxmDistributed16:
        return "Fp16SxmDistributed16";
    case BindingLayout::Fp16VxmDistributed16:
        return "Fp16VxmDistributed16";
    case BindingLayout::Fp16MxmDistributed16:
        return "Fp16MxmDistributed16";
    case BindingLayout::Fp16RopeTable: return "Fp16RopeTable";
    }
    return "Unknown";
}

DmaPurpose binding_purpose(const BinaryBinding& binding)
{
    switch (binding.access) {
    case BindingAccess::Input: return DmaPurpose::InputTensor;
    case BindingAccess::Output: return DmaPurpose::OutputTensor;
    case BindingAccess::Internal: return DmaPurpose::Model;
    }
    return DmaPurpose::General;
}

std::size_t checked_add(
    std::size_t lhs, std::size_t rhs, std::string_view context)
{
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(std::string(context));
    }
    return lhs + rhs;
}

std::size_t checked_multiply(
    std::size_t lhs, std::size_t rhs, std::string_view context)
{
    if (lhs != 0
        && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(context));
    }
    return lhs * rhs;
}

std::size_t element_bytes(BindingElementType type)
{
    switch (type) {
    case BindingElementType::I8: return 1;
    case BindingElementType::I32: return 4;
    case BindingElementType::F16: return 2;
    case BindingElementType::F32: return 4;
    }
    throw std::invalid_argument("BinaryBinding has an unknown element type");
}

struct ValidatedBinding {
    std::size_t rows{0};
    std::size_t columns{0};
    std::size_t byte_size{0};
    std::size_t base_row{0};
    std::size_t slice_capacity{0};
    std::uint16_t valid_hemisphere_mask{0};
    std::uint16_t hemisphere_mask{0};
    std::vector<std::size_t> hemispheres{};
};

void require_layout(
    const BinaryBinding& binding,
    BindingElementType element_type,
    std::size_t rank)
{
    if (binding.element_type != element_type) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " has an incompatible element type");
    }
    if (binding.shape.size() != rank) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " requires a rank-" + std::to_string(rank) + " shape");
    }
}

ValidatedBinding validate_common(
    const BinaryBinding& binding,
    const TargetDescription& target)
{
    if (target.vector_bytes == 0
        || target.hemisphere_count == 0
        || target.mem_slices_per_hemisphere == 0
        || target.sram_banks_per_tile == 0
        || target.sram_rows_per_bank == 0) {
        throw std::invalid_argument(
            "target has incomplete binding DMA geometry");
    }
    if (target.hemisphere_count
        > std::numeric_limits<std::uint16_t>::digits) {
        throw std::invalid_argument(
            "target hemisphere mask does not fit BinaryBinding");
    }
    if (binding.address_stride != 1) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " requires address_stride == 1");
    }
    if (binding.base_row < 0) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " has a negative base_row");
    }
    if (binding.byte_size == 0
        || binding.byte_size
            > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " has an invalid byte_size");
    }

    const auto valid_mask = static_cast<std::uint16_t>(
        (std::uint32_t {1} << target.hemisphere_count) - 1);
    const auto mask = static_cast<std::uint16_t>(
        binding.hemisphere_mask & valid_mask);
    if (mask == 0 || mask != binding.hemisphere_mask) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " has an invalid hemisphere_mask");
    }

    std::vector<std::size_t> hemispheres;
    for (std::size_t hemisphere = 0;
         hemisphere < target.hemisphere_count;
         ++hemisphere) {
        if ((mask & (std::uint16_t {1} << hemisphere)) != 0) {
            hemispheres.push_back(hemisphere);
        }
    }

    if (binding.slices.empty()) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " has no MEM slices");
    }
    std::vector<std::uint16_t> sorted_slices = binding.slices;
    std::sort(sorted_slices.begin(), sorted_slices.end());
    if (std::adjacent_find(
            sorted_slices.begin(), sorted_slices.end())
        != sorted_slices.end()) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " contains duplicate MEM slices");
    }
    if (sorted_slices.back()
        >= target.mem_slices_per_hemisphere) {
        throw std::out_of_range(
            std::string(layout_name(binding.layout))
            + " MEM slice is outside the target");
    }

    std::size_t shape_elements = 1;
    for (const auto dimension : binding.shape) {
        if (dimension == 0
            || dimension > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument(
                std::string(layout_name(binding.layout))
                + " has an invalid shape");
        }
        shape_elements = checked_multiply(
            shape_elements,
            static_cast<std::size_t>(dimension),
            "BinaryBinding shape element count overflows size_t");
    }
    const auto expected_bytes = checked_multiply(
        shape_elements, element_bytes(binding.element_type),
        "BinaryBinding logical byte count overflows size_t");
    if (expected_bytes != binding.byte_size) {
        throw std::invalid_argument(
            std::string(layout_name(binding.layout))
            + " shape does not match byte_size");
    }

    const auto capacity = checked_multiply(
        target.sram_banks_per_tile,
        target.sram_rows_per_bank,
        "target SRAM slice capacity overflows size_t");
    return {
        binding.shape.size() == 2
            ? static_cast<std::size_t>(binding.shape[0])
            : 0,
        binding.shape.size() == 2
            ? static_cast<std::size_t>(binding.shape[1])
            : 0,
        static_cast<std::size_t>(binding.byte_size),
        static_cast<std::size_t>(binding.base_row),
        capacity,
        valid_mask,
        mask,
        std::move(hemispheres),
    };
}

struct PhysicalByte {
    std::size_t hemisphere{0};
    std::size_t slice{0};
    std::size_t row{0};
    std::size_t offset{0};
};

template <typename Emit>
void for_each_physical_byte(
    const BinaryBinding& binding,
    const TargetDescription& target,
    const ValidatedBinding& validated,
    Emit&& emit)
{
    const auto emit_checked = [&](std::size_t logical_offset,
                                  PhysicalByte physical) {
        if (logical_offset >= validated.byte_size
            || physical.hemisphere >= target.hemisphere_count
            || physical.slice
                >= target.mem_slices_per_hemisphere
            || physical.row >= validated.slice_capacity
            || physical.offset >= target.vector_bytes) {
            throw std::out_of_range(
                std::string(layout_name(binding.layout))
                + " physical address exceeds the target");
        }
        emit(logical_offset, physical);
    };

    if (binding.layout == BindingLayout::Vector) {
        require_layout(binding, BindingElementType::I8, 1);
        if (binding.slices.size() != 1
            || validated.byte_size % target.vector_bytes != 0) {
            throw std::invalid_argument(
                "Vector requires one slice and whole physical vectors");
        }
        for (std::size_t logical = 0;
             logical < validated.byte_size;
             ++logical) {
            for (const auto hemisphere : validated.hemispheres) {
                emit_checked(logical, {
                    hemisphere,
                    binding.slices[0],
                    checked_add(
                        validated.base_row,
                        logical / target.vector_bytes,
                        "Vector row address overflows size_t"),
                    logical % target.vector_bytes,
                });
            }
        }
        return;
    }

    if (binding.layout
        == BindingLayout::Fp16MxmActivationPlanar) {
        require_layout(binding, BindingElementType::F16, 2);
        if (target.mxm_count == 0
            || target.mxm_count % target.hemisphere_count != 0
            || target.mxm_activation_bytes_per_value != 2
            || target.mxm_reduction != target.vector_bytes) {
            throw std::invalid_argument(
                "target has incompatible FP16 MXM activation geometry");
        }
        const auto local_mxm_count =
            target.mxm_count / target.hemisphere_count;
        const auto expected_slices = checked_multiply(
            local_mxm_count,
            target.mxm_activation_bytes_per_value,
            "FP16 activation slice count overflows size_t");
        if (binding.slices.size() != expected_slices
            || validated.columns % target.mxm_reduction != 0) {
            throw std::invalid_argument(
                "Fp16MxmActivationPlanar has incompatible slices or shape");
        }
        for (std::size_t row = 0; row < validated.rows; ++row) {
            for (std::size_t column = 0;
                 column < validated.columns;
                 ++column) {
                const auto physical_row = checked_add(
                    validated.base_row,
                    checked_add(
                        checked_multiply(
                            column / target.mxm_reduction,
                            validated.rows,
                            "FP16 activation row address overflows size_t"),
                        row,
                        "FP16 activation row address overflows size_t"),
                    "FP16 activation row address overflows size_t");
                const auto logical =
                    (row * validated.columns + column) * 2;
                for (const auto hemisphere : validated.hemispheres) {
                    for (std::size_t local_mxm = 0;
                         local_mxm < local_mxm_count;
                         ++local_mxm) {
                        for (std::size_t byte = 0; byte < 2; ++byte) {
                            emit_checked(logical + byte, {
                                hemisphere,
                                binding.slices[local_mxm * 2 + byte],
                                physical_row,
                                column % target.mxm_reduction,
                            });
                        }
                    }
                }
            }
        }
        return;
    }

    if (binding.layout == BindingLayout::Fp16PairPlanar) {
        require_layout(binding, BindingElementType::F16, 2);
        if (target.mxm_count == 0
            || target.mxm_count % target.hemisphere_count != 0
            || target.mxm_columns != target.vector_bytes) {
            throw std::invalid_argument(
                "target has incompatible FP16 pair geometry");
        }
        const auto local_mxm_count =
            target.mxm_count / target.hemisphere_count;
        const bool replicated_pair =
            binding.slices.size() == 2 && local_mxm_count != 1;
        const auto expected_slices =
            checked_multiply(local_mxm_count, std::size_t {2},
                "FP16 pair slice count overflows size_t");
        if ((!replicated_pair
                && binding.slices.size() != expected_slices)
            || validated.columns % target.vector_bytes != 0) {
            throw std::invalid_argument(
                "Fp16PairPlanar has incompatible slices or shape");
        }

        if (replicated_pair) {
            for (std::size_t row = 0; row < validated.rows; ++row) {
                for (std::size_t column = 0;
                     column < validated.columns;
                     ++column) {
                    const auto physical_row = checked_add(
                        validated.base_row,
                        checked_add(
                            checked_multiply(
                                column / target.vector_bytes,
                                validated.rows,
                                "FP16 pair row address overflows size_t"),
                            row,
                            "FP16 pair row address overflows size_t"),
                        "FP16 pair row address overflows size_t");
                    const auto logical =
                        (row * validated.columns + column) * 2;
                    for (const auto hemisphere : validated.hemispheres) {
                        for (std::size_t byte = 0; byte < 2; ++byte) {
                            emit_checked(logical + byte, {
                                hemisphere,
                                binding.slices[byte],
                                physical_row,
                                column % target.vector_bytes,
                            });
                        }
                    }
                }
            }
            return;
        }

        const auto hemisphere_width = checked_multiply(
            target.vector_bytes, local_mxm_count,
            "FP16 pair hemisphere width overflows size_t");
        if (validated.columns % hemisphere_width != 0) {
            throw std::invalid_argument(
                "Fp16PairPlanar columns do not fill local MXM pairs");
        }
        for (std::size_t row = 0; row < validated.rows; ++row) {
            for (std::size_t column = 0;
                 column < validated.columns;
                 ++column) {
                const auto hemisphere_group = column / hemisphere_width;
                const auto hemisphere = validated.hemispheres[
                    hemisphere_group % validated.hemispheres.size()];
                const auto address_group =
                    hemisphere_group / validated.hemispheres.size();
                const auto local_mxm =
                    (column % hemisphere_width) / target.vector_bytes;
                const auto physical_row = checked_add(
                    validated.base_row,
                    checked_add(
                        checked_multiply(
                            address_group,
                            validated.rows,
                            "FP16 pair row address overflows size_t"),
                        row,
                        "FP16 pair row address overflows size_t"),
                    "FP16 pair row address overflows size_t");
                const auto logical =
                    (row * validated.columns + column) * 2;
                for (std::size_t byte = 0; byte < 2; ++byte) {
                    emit_checked(logical + byte, {
                        hemisphere,
                        binding.slices[local_mxm * 2 + byte],
                        physical_row,
                        column % target.vector_bytes,
                    });
                }
            }
        }
        return;
    }

    if (binding.layout == BindingLayout::W8A16MxmWeightStriped
        || binding.layout
            == BindingLayout::W8A16MxmWeightWaveStriped) {
        require_layout(binding, BindingElementType::I8, 2);
        if (target.mxm_count == 0
            || target.mxm_count % target.hemisphere_count != 0
            || target.mxm_reduction != target.vector_bytes
            || target.mxm_columns == 0
            || target.lanes_per_tile == 0
            || target.mxm_weight_bytes_per_value == 0
            || target.mxm_load_streams_per_cycle
                % target.mxm_weight_bytes_per_value != 0
            || target.mxm_load_streams_per_cycle
                    / target.mxm_weight_bytes_per_value
                != target.lanes_per_tile
            || target.mxm_columns % target.lanes_per_tile != 0) {
            throw std::invalid_argument(
                "target has incompatible W8A16 MXM weight geometry");
        }
        if (binding.slices.size() != target.lanes_per_tile
            || validated.rows % target.mxm_reduction != 0
            || validated.hemisphere_mask
                != validated.valid_hemisphere_mask) {
            throw std::invalid_argument(
                "W8A16 MXM weight has incompatible slices, shape, or mask");
        }
        const auto pulses =
            target.mxm_columns / target.lanes_per_tile;
        const auto reduction_blocks =
            validated.rows / target.mxm_reduction;
        const auto local_mxm_count =
            target.mxm_count / target.hemisphere_count;
        const auto striped_width = checked_multiply(
            target.mxm_columns, target.hemisphere_count,
            "striped weight width overflows size_t");
        const auto hemisphere_width = checked_multiply(
            target.mxm_columns, local_mxm_count,
            "wave-striped weight hemisphere width overflows size_t");
        if (binding.layout == BindingLayout::W8A16MxmWeightStriped
            && validated.columns % striped_width != 0) {
            throw std::invalid_argument(
                "W8A16MxmWeightStriped columns do not fill hemispheres");
        }
        if (binding.layout
                == BindingLayout::W8A16MxmWeightWaveStriped
            && validated.columns % hemisphere_width != 0) {
            throw std::invalid_argument(
                "W8A16MxmWeightWaveStriped columns do not fill local MXMs");
        }

        for (std::size_t reduction = 0;
             reduction < validated.rows;
             ++reduction) {
            for (std::size_t column = 0;
                 column < validated.columns;
                 ++column) {
                const auto local_column =
                    column % target.mxm_columns;
                const auto pulse =
                    pulses - 1 - local_column / target.lanes_per_tile;
                const auto stream =
                    local_column % target.lanes_per_tile;
                std::size_t hemisphere = 0;
                std::size_t physical_row = 0;
                if (binding.layout
                    == BindingLayout::W8A16MxmWeightStriped) {
                    hemisphere =
                        (column / target.mxm_columns)
                        % target.hemisphere_count;
                    const auto group = column / striped_width;
                    const auto block = checked_add(
                        checked_multiply(
                            group,
                            reduction_blocks,
                            "striped weight row address overflows size_t"),
                        reduction / target.mxm_reduction,
                        "striped weight row address overflows size_t");
                    physical_row = checked_add(
                        validated.base_row,
                        checked_add(
                            checked_multiply(
                                block, pulses,
                                "striped weight row address overflows size_t"),
                            pulse,
                            "striped weight row address overflows size_t"),
                        "striped weight row address overflows size_t");
                } else {
                    const auto full_width = checked_multiply(
                        hemisphere_width,
                        target.hemisphere_count,
                        "wave-striped weight width overflows size_t");
                    const auto within_group = column % full_width;
                    hemisphere = within_group / hemisphere_width;
                    const auto local_mxm =
                        (within_group % hemisphere_width)
                        / target.mxm_columns;
                    const auto group = column / full_width;
                    const auto block = checked_add(
                        checked_multiply(
                            group,
                            reduction_blocks,
                            "wave-striped weight row address overflows size_t"),
                        reduction / target.mxm_reduction,
                        "wave-striped weight row address overflows size_t");
                    const auto rows_per_block = checked_multiply(
                        pulses, local_mxm_count,
                        "wave-striped rows per block overflows size_t");
                    physical_row = checked_add(
                        validated.base_row,
                        checked_add(
                            checked_multiply(
                                block,
                                rows_per_block,
                                "wave-striped weight row address overflows size_t"),
                            checked_add(
                                checked_multiply(
                                    local_mxm,
                                    pulses,
                                    "wave-striped weight row address overflows size_t"),
                                pulse,
                                "wave-striped weight row address overflows size_t"),
                            "wave-striped weight row address overflows size_t"),
                        "wave-striped weight row address overflows size_t");
                }
                emit_checked(
                    reduction * validated.columns + column,
                    {
                        hemisphere,
                        binding.slices[stream],
                        physical_row,
                        reduction % target.mxm_reduction,
                    });
            }
        }
        return;
    }

    throw std::invalid_argument(
        "unsupported binding layout "
        + std::string(layout_name(binding.layout)));
}

struct PhysicalPlane {
    std::size_t hemisphere{0};
    std::size_t slice{0};

    friend bool operator<(
        const PhysicalPlane& lhs,
        const PhysicalPlane& rhs) noexcept
    {
        return std::pair(lhs.hemisphere, lhs.slice)
            < std::pair(rhs.hemisphere, rhs.slice);
    }
};

struct PlannedRegion {
    BindingRegion region{};
    PhysicalPlane plane{};
    std::size_t first_row{0};
    std::size_t row_count{0};
};

MemGlobalAddress24 region_address(
    const PhysicalPlane& plane,
    std::size_t linear_row,
    const TargetDescription& target)
{
    const auto bank = linear_row / target.sram_rows_per_bank;
    const auto row = linear_row % target.sram_rows_per_bank;
    return MemGlobalAddress24::FromFields(
        plane.hemisphere,
        plane.slice,
        MemSliceByteAddress17::FromFields(bank, row, 0));
}

std::vector<PlannedRegion> plan_regions(
    const BinaryBinding& binding,
    const TargetDescription& target)
{
    const auto validated = validate_common(binding, target);
    std::map<PhysicalPlane, std::vector<bool>> touched;
    for_each_physical_byte(
        binding, target, validated,
        [&](std::size_t, const PhysicalByte& physical) {
            const auto plane =
                PhysicalPlane {physical.hemisphere, physical.slice};
            auto [it, inserted] = touched.try_emplace(plane);
            if (inserted) {
                it->second.resize(validated.slice_capacity);
            }
            it->second[physical.row] = true;
        });

    std::vector<PlannedRegion> result;
    for (const auto& [plane, rows] : touched) {
        std::size_t row = 0;
        while (row < rows.size()) {
            row = static_cast<std::size_t>(
                std::find(rows.begin() + row, rows.end(), true)
                - rows.begin());
            if (row == rows.size()) {
                break;
            }
            const auto first = row;
            row = static_cast<std::size_t>(
                std::find(rows.begin() + row, rows.end(), false)
                - rows.begin());
            const auto count = row - first;
            result.push_back({
                {
                    region_address(plane, first, target),
                    binding_purpose(binding),
                    checked_multiply(
                        count,
                        target.vector_bytes,
                        "binding region byte count overflows size_t"),
                },
                plane,
                first,
                count,
            });
        }
    }
    if (result.empty()) {
        throw std::logic_error("binding layout produced no physical regions");
    }
    return result;
}

std::size_t find_region(
    const std::vector<PlannedRegion>& regions,
    const PhysicalByte& physical)
{
    const auto found = std::find_if(
        regions.begin(), regions.end(),
        [&](const PlannedRegion& region) {
            return region.plane.hemisphere == physical.hemisphere
                && region.plane.slice == physical.slice
                && physical.row >= region.first_row
                && physical.row - region.first_row
                    < region.row_count;
        });
    if (found == regions.end()) {
        throw std::logic_error(
            "physical byte is outside the planned binding regions");
    }
    return static_cast<std::size_t>(found - regions.begin());
}

} // namespace

std::vector<BindingRegion> plan_binding_regions(
    const BinaryBinding& binding,
    const TargetDescription& target)
{
    const auto planned = plan_regions(binding, target);
    std::vector<BindingRegion> result;
    result.reserve(planned.size());
    for (const auto& region : planned) {
        result.push_back(region.region);
    }
    return result;
}

std::vector<BindingTransfer> pack_binding(
    const BinaryBinding& binding,
    std::span<const std::uint8_t> logical_bytes,
    const TargetDescription& target)
{
    if (logical_bytes.size() != binding.byte_size) {
        throw std::invalid_argument(
            "logical binding byte count does not match BinaryBinding::byte_size");
    }
    const auto validated = validate_common(binding, target);
    const auto planned = plan_regions(binding, target);
    std::vector<BindingTransfer> result;
    std::vector<std::vector<bool>> written;
    result.reserve(planned.size());
    written.reserve(planned.size());
    for (const auto& region : planned) {
        result.push_back({
            region.region.address,
            region.region.purpose,
            std::vector<std::uint8_t>(region.region.byte_size),
        });
        written.emplace_back(region.region.byte_size);
    }

    for_each_physical_byte(
        binding, target, validated,
        [&](std::size_t logical, const PhysicalByte& physical) {
            const auto region_index =
                find_region(planned, physical);
            const auto physical_offset =
                (physical.row - planned[region_index].first_row)
                    * target.vector_bytes
                + physical.offset;
            result[region_index].bytes[physical_offset] =
                logical_bytes[logical];
            written[region_index][physical_offset] = true;
        });
    for (const auto& bytes : written) {
        if (std::find(bytes.begin(), bytes.end(), false)
            != bytes.end()) {
            throw std::invalid_argument(
                std::string(layout_name(binding.layout))
                + " would require a partial-vector DMA write");
        }
    }
    return result;
}

std::vector<std::uint8_t> unpack_binding(
    const BinaryBinding& binding,
    const std::vector<BindingTransfer>& transfers,
    const TargetDescription& target)
{
    const auto validated = validate_common(binding, target);
    const auto planned = plan_regions(binding, target);
    if (transfers.size() != planned.size()) {
        throw std::invalid_argument(
            "binding transfer plan contains an unexpected transfer count");
    }

    std::vector<const BindingTransfer*> sources(
        planned.size(), nullptr);
    std::vector<bool> used(transfers.size());
    for (std::size_t region_index = 0;
         region_index < planned.size();
         ++region_index) {
        for (std::size_t transfer_index = 0;
             transfer_index < transfers.size();
             ++transfer_index) {
            if (!used[transfer_index]
                && transfers[transfer_index].address
                    == planned[region_index].region.address
                && transfers[transfer_index].bytes.size()
                    == planned[region_index].region.byte_size) {
                sources[region_index] = &transfers[transfer_index];
                used[transfer_index] = true;
                break;
            }
        }
        if (sources[region_index] == nullptr) {
            throw std::invalid_argument(
                "binding transfer plan is missing a physical region");
        }
    }

    std::vector<std::uint8_t> result(validated.byte_size);
    std::vector<bool> seen(validated.byte_size);
    for_each_physical_byte(
        binding, target, validated,
        [&](std::size_t logical, const PhysicalByte& physical) {
            const auto region_index =
                find_region(planned, physical);
            const auto physical_offset =
                (physical.row - planned[region_index].first_row)
                    * target.vector_bytes
                + physical.offset;
            const auto value =
                sources[region_index]->bytes[physical_offset];
            if (!seen[logical]) {
                result[logical] = value;
                seen[logical] = true;
            } else if (result[logical] != value) {
                throw std::invalid_argument(
                    "replicated binding downloads disagree");
            }
        });
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::logic_error(
            "binding layout did not reconstruct every logical byte");
    }
    return result;
}

} // namespace ftlpu::software::runtime
