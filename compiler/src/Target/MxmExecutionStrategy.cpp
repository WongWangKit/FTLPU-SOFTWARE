#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"

#include "mlir/IR/BuiltinOps.h"

namespace ftlpu::compiler::target {

llvm::StringRef mxm_execution_policy_name(MxmExecutionPolicy policy)
{
    switch (policy) {
    case MxmExecutionPolicy::Auto: return "auto";
    case MxmExecutionPolicy::Legacy: return "legacy";
    case MxmExecutionPolicy::Block8: return "block8";
    }
    return "auto";
}

mlir::FailureOr<MxmExecutionPolicy> parse_mxm_execution_policy(
    llvm::StringRef value)
{
    if (value == "auto") return MxmExecutionPolicy::Auto;
    if (value == "legacy") return MxmExecutionPolicy::Legacy;
    if (value == "block8") return MxmExecutionPolicy::Block8;
    return mlir::failure();
}

mlir::FailureOr<MxmExecutionPolicy> mxm_execution_policy_from_operation(
    mlir::Operation* operation)
{
    if (!operation) return mlir::failure();
    mlir::Operation* cursor = operation;
    while (cursor->getParentOp()) cursor = cursor->getParentOp();
    auto module = llvm::dyn_cast<mlir::ModuleOp>(cursor);
    if (!module) return mlir::failure();
    const auto value =
        module->getAttrOfType<mlir::StringAttr>(
            "ftlpu.mxm_execution_policy");
    if (!value) return MxmExecutionPolicy::Auto;
    return parse_mxm_execution_policy(value.getValue());
}

llvm::StringRef MxmExecutionStrategy::weight_input_mode() const
{
    return uses_local_dequant()
        ? "int8_dequant_bf16"
        : "direct16";
}

llvm::StringRef MxmExecutionStrategy::compute_mode() const
{
    return uses_block8() ? "block8" : "vector";
}

mlir::FailureOr<MxmExecutionStrategy> plan_mxm_execution_strategy(
    const MxmExecutionRequest& request,
    const LPUTargetModel& target)
{
    return plan_mxm_execution_strategy(
        request, target, MxmExecutionPolicy::Auto);
}

mlir::FailureOr<MxmExecutionStrategy> plan_mxm_execution_strategy(
    const MxmExecutionRequest& request,
    const LPUTargetModel& target,
    MxmExecutionPolicy policy)
{
    const auto& throughput = target.throughput();
    if (request.m <= 0 || request.n <= 0 || request.k <= 0
        || throughput.mxm_rows <= 0
        || throughput.mxm_columns <= 0)
        return mlir::failure();

    MxmExecutionStrategy strategy;
    strategy.weight_stream_count =
        throughput.mxm_load_streams_per_cycle;
    strategy.activation_stream_count =
        throughput.mxm_activation_streams;

    const bool localDequantLegal =
        target.supports_mxm_local_dequant()
        && request.weight_is_i8
        && request.activation_is_bf16
        && request.k % throughput.mxm_rows == 0
        && request.n % throughput.mxm_columns == 0
        && throughput.mxm_int8_load_streams_per_cycle > 0
        && throughput.mxm_int8_load_streams_per_cycle
            <= target.streams().streams_per_direction;
    const bool block8Legal =
        localDequantLegal
        && target.supports_mxm_block8_compute()
        && request.accumulator_result_allowed
        && request.result_is_16bit_float
        && throughput.mxm_block_rows > 0
        && request.m % throughput.mxm_rows == 0
        && throughput.mxm_rows % throughput.mxm_block_rows == 0
        && 2 * throughput.mxm_block_rows
            <= target.streams().streams_per_direction;
    if (policy == MxmExecutionPolicy::Block8 && !block8Legal)
        return mlir::failure();
    if (policy != MxmExecutionPolicy::Legacy && block8Legal) {
        strategy.weight_preparation =
            MxmWeightPreparation::LocalInt8DequantBf16;
        strategy.compute = MxmComputeStrategy::Block8;
        strategy.weight_stream_count =
            throughput.mxm_int8_load_streams_per_cycle;
        strategy.activation_stream_count =
            2 * throughput.mxm_block_rows;
        strategy.rows_per_compute_issue =
            throughput.mxm_block_rows;
    }
    return strategy;
}

} // namespace ftlpu::compiler::target
