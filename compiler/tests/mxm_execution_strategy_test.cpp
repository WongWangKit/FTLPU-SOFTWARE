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

    auto automatic = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target);
    require(mlir::succeeded(automatic),
        "BF16 W8A16 strategy selection failed");
    require(automatic->uses_local_dequant(),
        "BF16 W8A16 should select MXM-local dequant");
    require(automatic->weight_stream_count == 8
            && automatic->activation_stream_count
                == target.throughput().mxm_activation_streams
            && automatic->rows_per_compute_issue == 1,
        "Vector stream geometry is incorrect");

    auto fp16 = plan_mxm_execution_strategy(
        {32, 64, 64, false, true, true, true}, target);
    require(mlir::succeeded(fp16)
            && !fp16->uses_local_dequant(),
        "FP16 activation must use the compatibility strategy");

    auto streamResult = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, false}, target);
    require(mlir::succeeded(streamResult)
            && streamResult->uses_local_dequant(),
        "stream-result request must retain local-dequant Vector compute");

    auto forcedLegacy = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target,
        MxmExecutionPolicy::Legacy);
    require(mlir::succeeded(forcedLegacy)
            && forcedLegacy->uses_local_dequant(),
        "legacy policy must select local-dequant Vector compute");

    auto forcedVector = plan_mxm_execution_strategy(
        {32, 64, 64, true, true, true, true}, target,
        MxmExecutionPolicy::Vector);
    require(mlir::succeeded(forcedVector)
            && forcedVector->uses_local_dequant(),
        "vector policy must select local-dequant Vector compute");

    std::cout << "mxm_execution_strategy_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "mxm_execution_strategy_test failed: "
              << ex.what() << '\n';
    return 1;
}
