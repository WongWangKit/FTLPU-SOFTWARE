#include "ftlpu/software/runtime/binding_transfer.hpp"
#include "ftlpu/software/runtime/cmodel_device_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu::software::runtime;
using ftlpu::MemGlobalAddress24;
using ftlpu::MemSliceByteAddress17;

constexpr std::size_t kM = 32;
constexpr std::size_t kK = 576;
constexpr std::size_t kHidden = 1536;
constexpr std::size_t kN = 576;

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

BinaryBinding binding(
    BindingAccess access,
    BindingElementType element_type,
    BindingLayout layout,
    std::size_t rows,
    std::size_t columns,
    std::vector<std::uint16_t> slices,
    std::size_t base_row)
{
    BinaryBinding result;
    result.access = access;
    result.element_type = element_type;
    result.layout = layout;
    result.byte_size = rows * columns
        * (element_type == BindingElementType::F16 ? 2 : 1);
    result.base_row = static_cast<std::int64_t>(base_row);
    result.address_stride = 1;
    result.shape = {rows, columns};
    result.slices = std::move(slices);
    result.hemisphere_mask = 3;
    return result;
}

std::vector<std::uint8_t> logical_bytes(std::size_t count, unsigned seed)
{
    std::vector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = static_cast<std::uint8_t>(
            (index * 131 + index / 17 + seed * 29) & 0xffu);
    }
    return result;
}

std::size_t transfer_bytes(
    const std::vector<BindingTransfer>& transfers)
{
    return std::accumulate(
        transfers.begin(), transfers.end(), std::size_t {0},
        [](std::size_t total, const BindingTransfer& transfer) {
            return total + transfer.bytes.size();
        });
}

MemGlobalAddress24 physical_address(
    std::size_t hemisphere,
    std::size_t slice,
    std::size_t row,
    const TargetDescription& target)
{
    return MemGlobalAddress24::FromFields(
        hemisphere,
        slice,
        MemSliceByteAddress17::FromFields(
            row / target.sram_rows_per_bank,
            row % target.sram_rows_per_bank,
            0));
}

std::uint8_t physical_byte(
    const std::vector<BindingTransfer>& transfers,
    std::size_t hemisphere,
    std::size_t slice,
    std::size_t region_first_row,
    std::size_t row,
    std::size_t offset,
    const TargetDescription& target)
{
    const auto address = physical_address(
        hemisphere, slice, region_first_row, target);
    const auto found = std::find_if(
        transfers.begin(), transfers.end(),
        [&](const BindingTransfer& transfer) {
            return transfer.address == address;
        });
    require(found != transfers.end(),
        "expected physical binding region was not planned");
    return found->bytes[
        (row - region_first_row) * target.vector_bytes + offset];
}

void dma_round_trip(
    CModelDeviceBackend& backend,
    const BinaryBinding& binding,
    const std::vector<std::uint8_t>& logical)
{
    const auto packed =
        pack_binding(binding, logical, backend.target());
    for (const auto& transfer : packed) {
        backend.upload(
            transfer.address, transfer.bytes, transfer.purpose);
    }

    auto downloaded = packed;
    for (auto& transfer : downloaded) {
        transfer.bytes = backend.download(
            transfer.address,
            transfer.bytes.size(),
            transfer.purpose);
    }
    require(
        unpack_binding(binding, downloaded, backend.target())
            == logical,
        "binding DMA pack/upload/download/unpack did not round trip");
}

} // namespace

int main()
try {
    CModelDeviceBackend backend;
    const auto& target = backend.target();
    if (target.mxm_reduction != 32
        || target.mxm_columns != 32
        || target.mxm_count != 4
        || target.hemisphere_count != 2
        || target.lanes_per_tile != 8
        || target.vector_bytes != 32) {
        // These are the real Compiler SmolLM2 FFN bindings and are only
        // executable on the transformer-eval target.
        return 0;
    }

    const std::vector<std::uint16_t> activation_slices {
        32, 33, 34, 35};
    const std::vector<std::uint16_t> weight_slices {
        0, 4, 8, 12, 16, 20, 24, 28};
    const std::vector<std::uint16_t> result_slices {
        24, 25, 26, 27};

    const auto activation = binding(
        BindingAccess::Input,
        BindingElementType::F16,
        BindingLayout::Fp16MxmActivationPlanar,
        kM, kK, activation_slices, 0);
    const auto gate = binding(
        BindingAccess::Input,
        BindingElementType::I8,
        BindingLayout::W8A16MxmWeightStriped,
        kK, kHidden, weight_slices, 10000);
    const auto up = binding(
        BindingAccess::Input,
        BindingElementType::I8,
        BindingLayout::W8A16MxmWeightStriped,
        kK, kHidden, weight_slices, 11728);
    const auto down = binding(
        BindingAccess::Input,
        BindingElementType::I8,
        BindingLayout::W8A16MxmWeightWaveStriped,
        kHidden, kN, weight_slices, 13456);
    const auto result = binding(
        BindingAccess::Output,
        BindingElementType::F16,
        BindingLayout::Fp16PairPlanar,
        kM, kN, result_slices, 0);

    const auto activation_bytes =
        logical_bytes(activation.byte_size, 1);
    const auto gate_bytes = logical_bytes(gate.byte_size, 2);
    const auto up_bytes = logical_bytes(up.byte_size, 3);
    const auto down_bytes = logical_bytes(down.byte_size, 4);
    const auto result_bytes = logical_bytes(result.byte_size, 5);

    const auto activation_transfers =
        pack_binding(activation, activation_bytes, target);
    require(activation_transfers.size() == 8,
        "activation did not plan one region per hemisphere/slice");
    require(
        transfer_bytes(activation_transfers)
            == activation.byte_size * 4,
        "activation replicas have the wrong physical byte count");
    require(
        physical_byte(
            activation_transfers, 0, 32, 0, 0, 0, target)
            == activation_bytes[0]
        && physical_byte(
            activation_transfers, 1, 34, 0, 0, 0, target)
            == activation_bytes[0],
        "activation low byte was not replicated by hemisphere/local MXM");

    const auto gate_transfers =
        pack_binding(gate, gate_bytes, target);
    require(gate_transfers.size() == 16,
        "striped weight did not plan one region per hemisphere/slice");
    require(transfer_bytes(gate_transfers) == gate.byte_size,
        "striped weight physical byte count changed");
    require(
        physical_byte(
            gate_transfers, 0, 0, 10000, 10003, 31, target)
            == gate_bytes[31 * kHidden]
        && physical_byte(
            gate_transfers, 1, 0, 10000, 10003, 31, target)
            == gate_bytes[31 * kHidden + 32],
        "striped weight hemisphere/pulse mapping is wrong");

    const auto down_transfers =
        pack_binding(down, down_bytes, target);
    require(down_transfers.size() == 16,
        "wave-striped weight did not plan one region per hemisphere/slice");
    require(transfer_bytes(down_transfers) == down.byte_size,
        "wave-striped weight physical byte count changed");
    require(
        physical_byte(
            down_transfers, 0, 0, 13456, 13463, 31, target)
            == down_bytes[31 * kN + 32]
        && physical_byte(
            down_transfers, 1, 0, 13456, 13459, 31, target)
            == down_bytes[31 * kN + 64],
        "wave-striped weight hemisphere/local-MXM mapping is wrong");

    const auto result_transfers =
        pack_binding(result, result_bytes, target);
    require(result_transfers.size() == 8,
        "pair-planar output did not plan physical regions");
    require(transfer_bytes(result_transfers) == result.byte_size,
        "pair-planar output physical byte count changed");
    require(
        physical_byte(
            result_transfers, 0, 26, 0, 0, 0, target)
            == result_bytes[32 * 2]
        && physical_byte(
            result_transfers, 1, 24, 0, 0, 0, target)
            == result_bytes[64 * 2],
        "pair-planar hemisphere/local-MXM mapping is wrong");

    require(
        unpack_binding(activation, activation_transfers, target)
            == activation_bytes,
        "activation pack/unpack did not round trip");
    require(unpack_binding(gate, gate_transfers, target) == gate_bytes,
        "striped weight pack/unpack did not round trip");
    require(unpack_binding(down, down_transfers, target) == down_bytes,
        "wave-striped weight pack/unpack did not round trip");
    require(
        unpack_binding(result, result_transfers, target)
            == result_bytes,
        "pair-planar pack/unpack did not round trip");

    auto inconsistent = activation_transfers;
    inconsistent.back().bytes.front() ^= 1;
    bool rejected_replica = false;
    try {
        (void)unpack_binding(activation, inconsistent, target);
    } catch (const std::invalid_argument&) {
        rejected_replica = true;
    }
    require(rejected_replica,
        "activation unpack accepted inconsistent MXM replicas");

    dma_round_trip(backend, activation, activation_bytes);
    dma_round_trip(backend, gate, gate_bytes);
    dma_round_trip(backend, up, up_bytes);
    dma_round_trip(backend, down, down_bytes);
    dma_round_trip(backend, result, result_bytes);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ffn_binding_transfer_test: "
              << error.what() << '\n';
    return 1;
}
