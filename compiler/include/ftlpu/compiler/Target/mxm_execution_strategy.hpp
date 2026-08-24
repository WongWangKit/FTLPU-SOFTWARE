#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace ftlpu::compiler::target {

enum class MxmExecutionPolicy {
    Auto,
    Vector,
    Legacy,
    Block8,
};

enum class MxmWeightPreparation {
    VxmDequantDirect16,
    LocalInt8DequantBf16,
};

enum class MxmComputeStrategy {
    Vector,
    Block8,
};

struct MxmExecutionRequest {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    bool activation_is_bf16 = false;
    bool weight_is_i8 = false;
    bool result_is_16bit_float = false;
    bool accumulator_result_allowed = true;
};

struct MxmExecutionStrategy {
    MxmWeightPreparation weight_preparation =
        MxmWeightPreparation::VxmDequantDirect16;
    MxmComputeStrategy compute = MxmComputeStrategy::Vector;
    int64_t weight_stream_count = 16;
    int64_t activation_stream_count = 2;
    int64_t rows_per_compute_issue = 1;

    bool uses_local_dequant() const
    {
        return weight_preparation
            == MxmWeightPreparation::LocalInt8DequantBf16;
    }
    bool uses_block8() const
    {
        return compute == MxmComputeStrategy::Block8;
    }
    llvm::StringRef weight_input_mode() const;
    llvm::StringRef compute_mode() const;
};

mlir::FailureOr<MxmExecutionStrategy> plan_mxm_execution_strategy(
    const MxmExecutionRequest& request,
    const LPUTargetModel& target);
mlir::FailureOr<MxmExecutionStrategy> plan_mxm_execution_strategy(
    const MxmExecutionRequest& request,
    const LPUTargetModel& target, MxmExecutionPolicy policy);

llvm::StringRef mxm_execution_policy_name(MxmExecutionPolicy policy);
mlir::FailureOr<MxmExecutionPolicy> parse_mxm_execution_policy(
    llvm::StringRef value);
mlir::FailureOr<MxmExecutionPolicy> mxm_execution_policy_from_operation(
    mlir::Operation* operation);

} // namespace ftlpu::compiler::target
