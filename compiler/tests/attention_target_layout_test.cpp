#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/software/runtime/target_abi.hpp"

#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main()
{
    try {
    using ftlpu::compiler::target::LPUTargetModel;
    const LPUTargetModel target;
    if (target.name() != ftlpu::software::runtime::kLpu32StreamTargetName
        || target.abi_fingerprint()
            != ftlpu::software::runtime::lpu_32stream_target_abi(
                target.throughput().mxms_per_hemisphere))
        throw std::logic_error("default compiler target ABI diverges from runtime");
    constexpr std::array<int64_t, 16> kFirstReduction {
        0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33};
    constexpr std::array<int64_t, 16> kSecondReduction {
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35};
    constexpr std::array<int64_t, 16> kDedicatedReduction {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    constexpr std::array<int64_t, 16> kDualFirstReduction {
        44, 45, 46, 47, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33};
    constexpr std::array<int64_t, 16> kDualSecondReduction {
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35};
    const auto& expectedFirst = target.uses_dedicated_slice_roles()
        ? kDedicatedReduction
        : target.throughput().mxms_per_hemisphere == 1
        ? kFirstReduction : kDualFirstReduction;
    const auto& expectedSecond = target.uses_dedicated_slice_roles()
        ? kDedicatedReduction
        : target.throughput().mxms_per_hemisphere == 1
        ? kSecondReduction : kDualSecondReduction;
    if (target.attention_query_iw_slices(0) != expectedFirst
        || target.attention_query_iw_slices(1) != expectedSecond)
        throw std::logic_error("query IW physical slice map diverges from CModel");
    if (target.attention_query_iw_slices(2) != expectedFirst
        || target.attention_query_iw_slices(3) != expectedSecond)
        throw std::logic_error(
            "query IW bank reuse does not support a 128-wide head");
    if (target.attention_query_iw_base_row() != 7600
        || target.attention_score_base_row() != 3000)
        throw std::logic_error("attention scratch-row map diverges from CModel");
    try {
        static_cast<void>(target.attention_query_iw_slices(-1));
        throw std::logic_error("invalid reduction block was accepted");
    } catch (const std::out_of_range&) {
    }

    auto memory = target.memory();
    memory.dedicated_slice_roles = 1;
    memory.w8a16_weight_slice_base = 20;
    memory.w8a16_weight_slice_count = 8;
    memory.w8a16_weight_slice_stride = 4;
    const LPUTargetModel partitioned(
        memory, target.streams(), target.throughput());
    const auto activations = partitioned.activation_storage_slices();
    const auto weights = partitioned.weight_storage_slices();
    if (activations.size() != 20 || activations.front() != 0
        || activations.back() != 19
        || weights.size() != 32 || weights.front() != 20
        || weights.back() != 51)
        throw std::logic_error("dedicated MEM slice partition is incorrect");
    constexpr std::array<int64_t, 8> kGateTemps {
        0, 1, 2, 3, 4, 5, 6, 7};
    constexpr std::array<int64_t, 8> kUpTemps {
        8, 9, 10, 11, 12, 13, 14, 15};
    const auto gateTemps = partitioned.ffn_gate_temp_slices();
    const auto upTemps = partitioned.ffn_up_temp_slices();
    if (gateTemps.size() != kGateTemps.size()
        || !std::equal(gateTemps.begin(), gateTemps.end(), kGateTemps.begin())
        || upTemps.size() != kUpTemps.size()
        || !std::equal(upTemps.begin(), upTemps.end(), kUpTemps.begin()))
        throw std::logic_error(
            "FFN temporaries do not reserve a distributed-16 activation plane");
    auto undersizedMemory = memory;
    undersizedMemory.w8a16_weight_slice_base = 12;
    const LPUTargetModel undersized(
        undersizedMemory, target.streams(), target.throughput());
    std::string validationError;
    if (mlir::succeeded(undersized.validate(&validationError))
        || validationError.find("at least 16 MEM slices")
            == std::string::npos)
        throw std::logic_error(
            "an undersized SXM activation plane was accepted");
    const auto allWeightSlicesAreLocalToMxm =
        [&](llvm::ArrayRef<int64_t> slices) {
            return std::all_of(slices.begin(), slices.end(),
                [&](int64_t slice) {
                    return partitioned.is_weight_storage_slice(slice);
                });
        };
    if (!allWeightSlicesAreLocalToMxm(
            partitioned.ffn_projection_weight_slices(
                ftlpu::compiler::target::FfnProjectionKind::Gate))
        || !allWeightSlicesAreLocalToMxm(
            partitioned.ffn_projection_weight_slices(
                ftlpu::compiler::target::FfnProjectionKind::Up))
        || !allWeightSlicesAreLocalToMxm(
            partitioned.ffn_down_projection_weight_slices())
        || !allWeightSlicesAreLocalToMxm(
            partitioned.page_resident_attention_weight_slices()))
        throw std::logic_error("weight plane escaped the MXM-local slice pool");
    for (int64_t slice : partitioned.attention_query_iw_slices(0))
        if (!partitioned.is_activation_storage_slice(slice))
            throw std::logic_error(
                "attention activation escaped the VXM-local slice pool");
    std::cout << "attention_target_layout_test passed\n";
    return 0;
    } catch (const std::exception& error) {
        std::cerr << "attention_target_layout_test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
