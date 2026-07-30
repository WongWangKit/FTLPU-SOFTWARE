#include "ftlpu/software/runtime/binding_transfer.hpp"

#include "ftlpu/mem/address.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace ftlpu::software::runtime {
namespace {

std::string_view layout_name(BindingLayout layout)
{
    switch (layout) {
    case BindingLayout::Vector: return "Vector";
    case BindingLayout::MxmWeightStriped: return "MxmWeightStriped";
    case BindingLayout::Int32BytePlanar: return "Int32BytePlanar";
    case BindingLayout::Fp16BytePlanar: return "Fp16BytePlanar";
    case BindingLayout::Fp16MxmActivationPlanar: return "Fp16MxmActivationPlanar";
    case BindingLayout::W8A16MxmWeightStriped: return "W8A16MxmWeightStriped";
    case BindingLayout::Fp16PairPlanar: return "Fp16PairPlanar";
    case BindingLayout::W8A16AttentionWeightStriped:
        return "W8A16AttentionWeightStriped";
    case BindingLayout::W8A16MxmWeightWaveStriped:
        return "W8A16MxmWeightWaveStriped";
    case BindingLayout::Fp32CausalMaskTile: return "Fp32CausalMaskTile";
    case BindingLayout::Fp16SxmDistributed16: return "Fp16SxmDistributed16";
    case BindingLayout::Fp16VxmDistributed16: return "Fp16VxmDistributed16";
    case BindingLayout::Fp16MxmDistributed16: return "Fp16MxmDistributed16";
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

struct ValidatedVectorBinding {
    std::size_t vector_count{0};
    std::size_t slice{0};
    std::size_t linear_base_row{0};
    std::uint16_t hemisphere_mask{0};
};

ValidatedVectorBinding validate_vector_binding(
    const BinaryBinding& binding,
    const TargetDescription& target)
{
    if (binding.layout != BindingLayout::Vector) {
        throw std::invalid_argument(
            "unsupported binding layout " + std::string(layout_name(binding.layout)));
    }
    if (binding.element_type != BindingElementType::I8) {
        throw std::invalid_argument(
            "binding layout Vector currently supports I8 elements only");
    }
    if (binding.slices.size() != 1) {
        throw std::invalid_argument(
            "binding layout Vector requires exactly one MEM slice");
    }
    if (binding.address_stride != 1) {
        throw std::invalid_argument(
            "binding layout Vector requires address_stride == 1");
    }
    if (target.vector_bytes == 0 || target.sram_rows_per_bank == 0
        || target.sram_banks_per_tile == 0
        || target.mem_slices_per_hemisphere == 0
        || target.hemisphere_count == 0) {
        throw std::invalid_argument("target has incomplete Vector DMA geometry");
    }
    if (binding.byte_size == 0
        || binding.byte_size % target.vector_bytes != 0) {
        throw std::invalid_argument(
            "binding layout Vector requires whole physical-vector bytes");
    }
    if (binding.base_row < 0) {
        throw std::invalid_argument(
            "binding layout Vector has a negative base_row");
    }
    const auto slice = static_cast<std::size_t>(binding.slices.front());
    if (slice >= target.mem_slices_per_hemisphere) {
        throw std::out_of_range(
            "binding layout Vector MEM slice is outside the target");
    }
    if (target.hemisphere_count > std::numeric_limits<std::uint16_t>::digits) {
        throw std::invalid_argument("target hemisphere mask does not fit BinaryBinding");
    }
    const auto valid_mask = static_cast<std::uint16_t>(
        (std::uint32_t {1} << target.hemisphere_count) - 1);
    const auto mask = static_cast<std::uint16_t>(
        binding.hemisphere_mask & valid_mask);
    if (mask == 0 || mask != binding.hemisphere_mask) {
        throw std::invalid_argument(
            "binding layout Vector has an invalid hemisphere_mask");
    }

    const auto vector_count =
        static_cast<std::size_t>(binding.byte_size / target.vector_bytes);
    const auto base = static_cast<std::size_t>(binding.base_row);
    const auto capacity =
        target.sram_banks_per_tile * target.sram_rows_per_bank;
    if (base >= capacity || vector_count > capacity - base) {
        throw std::out_of_range(
            "binding layout Vector transfer exceeds SRAM slice capacity");
    }
    if (!binding.shape.empty()) {
        std::size_t elements = 1;
        for (const auto dimension : binding.shape) {
            if (dimension == 0
                || dimension > std::numeric_limits<std::size_t>::max() / elements) {
                throw std::invalid_argument(
                    "binding layout Vector has an invalid shape");
            }
            elements *= static_cast<std::size_t>(dimension);
        }
        if (elements != binding.byte_size
            || binding.shape.back() != target.vector_bytes) {
            throw std::invalid_argument(
                "binding layout Vector shape is not full physical vectors");
        }
    }
    return {vector_count, slice, base, mask};
}

MemGlobalAddress24 binding_address(
    std::size_t hemisphere,
    const ValidatedVectorBinding& binding,
    const TargetDescription& target)
{
    const auto bank = binding.linear_base_row / target.sram_rows_per_bank;
    const auto row = binding.linear_base_row % target.sram_rows_per_bank;
    return MemGlobalAddress24::FromFields(
        hemisphere,
        binding.slice,
        MemSliceByteAddress17::FromFields(bank, row, 0));
}

} // namespace

std::vector<BindingTransfer> pack_binding(
    const BinaryBinding& binding,
    std::span<const std::uint8_t> logical_bytes,
    const TargetDescription& target)
{
    const auto validated = validate_vector_binding(binding, target);
    if (logical_bytes.size() != binding.byte_size) {
        throw std::invalid_argument(
            "logical binding byte count does not match BinaryBinding::byte_size");
    }

    std::vector<BindingTransfer> result;
    for (std::size_t hemisphere = 0;
         hemisphere < target.hemisphere_count;
         ++hemisphere) {
        if ((validated.hemisphere_mask
                & (std::uint16_t {1} << hemisphere))
            == 0) {
            continue;
        }
        result.push_back(BindingTransfer {
            binding_address(hemisphere, validated, target),
            binding_purpose(binding),
            std::vector<std::uint8_t>(
                logical_bytes.begin(), logical_bytes.end()),
        });
    }
    return result;
}

std::vector<std::uint8_t> unpack_binding(
    const BinaryBinding& binding,
    const std::vector<BindingTransfer>& transfers,
    const TargetDescription& target)
{
    const auto validated = validate_vector_binding(binding, target);
    std::vector<std::uint8_t> result;
    std::size_t expected_count = 0;
    for (std::size_t hemisphere = 0;
         hemisphere < target.hemisphere_count;
         ++hemisphere) {
        if ((validated.hemisphere_mask
                & (std::uint16_t {1} << hemisphere))
            == 0) {
            continue;
        }
        ++expected_count;
        const auto address = binding_address(hemisphere, validated, target);
        const auto it = std::find_if(
            transfers.begin(), transfers.end(),
            [&](const BindingTransfer& transfer) {
                return transfer.address == address;
            });
        if (it == transfers.end()) {
            throw std::invalid_argument(
                "binding transfer plan is missing a target hemisphere");
        }
        if (it->bytes.size() != binding.byte_size) {
            throw std::invalid_argument(
                "binding transfer byte count does not match BinaryBinding::byte_size");
        }
        if (result.empty()) {
            result = it->bytes;
        } else if (result != it->bytes) {
            throw std::invalid_argument(
                "replicated binding downloads disagree between hemispheres");
        }
    }
    if (transfers.size() != expected_count) {
        throw std::invalid_argument(
            "binding transfer plan contains unexpected transfers");
    }
    return result;
}

} // namespace ftlpu::software::runtime
