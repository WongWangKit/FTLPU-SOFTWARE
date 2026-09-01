#include "ftlpu/compiler/Dialect/Tensor/Analysis/attention_weight_tile_plan.hpp"

#include <iostream>
#include <stdexcept>

using namespace ftlpu::compiler;

int main()
try {
    target::LPUTargetModel defaults;
    auto memory = defaults.memory();
    memory.slices_per_hemisphere = 52;
    memory.sram_depth_rows = 8192;
    memory.banks_per_slice = 2;
    memory.dedicated_slice_roles = 1;
    memory.w8a16_weight_slice_base = 20;
    memory.w8a16_weight_slice_count = 8;
    memory.w8a16_weight_slice_stride = 4;
    auto throughput = defaults.throughput();
    throughput.mxms_per_hemisphere = 1;
    target::LPUTargetModel target(
        memory, defaults.streams(), throughput);

    auto plan = tensor::planAttentionWeightTiles(
        1536, 12, 2, 128, 1, target);
    if (mlir::failed(plan))
        throw std::logic_error("Qwen attention weight tiling failed");
    const auto& query = plan->get(
        tensor::AttentionWeightTileKind::Query);
    const auto& key = plan->get(
        tensor::AttentionWeightTileKind::Key);
    const auto& value = plan->get(
        tensor::AttentionWeightTileKind::Value);
    const auto& output = plan->get(
        tensor::AttentionWeightTileKind::Output);
    if (query.base_row != 0 || query.rows != 4608
        || key.base_row != 4608 || key.rows != 768
        || value.base_row != 0 || value.rows != 768
        || output.base_row != 0 || output.rows != 4608)
        throw std::logic_error(
            "residency metadata describes group capacity instead of "
            "occupied rows");
    if (query.slice_group_begin != key.slice_group_begin
        || value.slice_group_begin == query.slice_group_begin
        || output.bank == query.bank)
        throw std::logic_error("unexpected attention weight placement");

    std::cout << "attention_weight_tile_planner_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "attention_weight_tile_planner_test failed: "
              << error.what() << '\n';
    return 1;
}
