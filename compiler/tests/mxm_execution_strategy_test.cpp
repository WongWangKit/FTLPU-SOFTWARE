#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
try {
    using namespace ftlpu::compiler::target;
    const LPUTargetModel target;

    auto block8 = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target);
    require(mlir::succeeded(block8),
        "BF16 W8A16 strategy selection failed");
    require(block8->uses_local_dequant(),
        "BF16 W8A16 should select MXM-local dequant");
    require(block8->uses_block8(),
        "aligned BF16 W8A16 should select Block8");
    if (block8->weight_stream_count != 8
        || block8->activation_stream_count != 16
        || block8->rows_per_compute_issue != 8) {
        std::cerr << "Block8 geometry: weight_streams="
                  << block8->weight_stream_count
                  << ", activation_streams="
                  << block8->activation_stream_count
                  << ", rows_per_issue="
                  << block8->rows_per_compute_issue << '\n';
    }
    require(block8->weight_stream_count == 8
            && block8->activation_stream_count == 16
            && block8->rows_per_compute_issue == 8,
        "Block8 stream geometry is incorrect");

    auto fp16 = plan_mxm_execution_strategy(
        {32, 64, 64, false, true, true, true}, target);
    require(mlir::succeeded(fp16)
            && !fp16->uses_local_dequant()
            && !fp16->uses_block8(),
        "FP16 activation must use the compatibility strategy");

    auto streamResult = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, false}, target);
    require(mlir::succeeded(streamResult)
            && streamResult->uses_local_dequant()
            && !streamResult->uses_block8(),
        "stream-result request must retain local-dequant Vector compute");

    auto throughput = target.throughput();
    throughput.mxm_block_compute_enabled = 0;
    LPUTargetModel noBlock8(
        target.memory(), target.streams(), throughput);
    auto fallback = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, noBlock8);
    require(mlir::succeeded(fallback)
            && fallback->uses_local_dequant()
            && !fallback->uses_block8(),
        "target without Block8 must retain MXM-local dequant for Vector compute");
    auto unavailableForcedBlock8 = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, noBlock8,
        MxmExecutionPolicy::Block8);
    require(mlir::failed(unavailableForcedBlock8),
        "Block8 policy must not enable a missing hardware capability");

    auto forcedLegacy = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target,
        MxmExecutionPolicy::Legacy);
    require(mlir::succeeded(forcedLegacy)
            && forcedLegacy->uses_local_dequant()
            && !forcedLegacy->uses_block8(),
        "legacy policy must select local-dequant Vector compute");

    auto forcedVector = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target,
        MxmExecutionPolicy::Vector);
    require(mlir::succeeded(forcedVector)
            && forcedVector->uses_local_dequant()
            && !forcedVector->uses_block8()
            && forcedVector->compute_mode() == "vector",
        "vector policy must select local-dequant Vector compute");

    auto forcedBlock8 = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target,
        MxmExecutionPolicy::Block8);
    require(mlir::succeeded(forcedBlock8)
            && forcedBlock8->uses_block8(),
        "Block8 policy did not require Block8");

    auto illegalForcedBlock8 = plan_mxm_execution_strategy(
        {32, 64, 64, false, true, true, true}, target,
        MxmExecutionPolicy::Block8);
    require(mlir::failed(illegalForcedBlock8),
        "Block8 policy accepted an illegal projection");

    std::cout << "mxm_execution_strategy_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "mxm_execution_strategy_test failed: "
              << ex.what() << '\n';
    return 1;
}
