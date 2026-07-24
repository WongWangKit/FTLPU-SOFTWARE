#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/software/runtime/target_abi.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

int main()
{
    using ftlpu::compiler::target::LPUTargetModel;
    const LPUTargetModel target;
    if (target.name() != ftlpu::software::runtime::kLpu32StreamTargetName
        || target.abi_fingerprint()
            != ftlpu::software::runtime::kLpu32StreamTargetAbi)
        throw std::logic_error("default compiler target ABI diverges from runtime");
    constexpr std::array<int64_t, 16> kFirstReduction {
        0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 33};
    constexpr std::array<int64_t, 16> kSecondReduction {
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35};
    if (target.attention_query_iw_slices(0) != kFirstReduction
        || target.attention_query_iw_slices(1) != kSecondReduction)
        throw std::logic_error("query IW physical slice map diverges from CModel");
    if (target.attention_query_iw_base_row() != 7600
        || target.attention_score_base_row() != 3000)
        throw std::logic_error("attention scratch-row map diverges from CModel");
    try {
        static_cast<void>(target.attention_query_iw_slices(2));
        throw std::logic_error("invalid reduction block was accepted");
    } catch (const std::out_of_range&) {
    }
    std::cout << "attention_target_layout_test passed\n";
    return 0;
}
