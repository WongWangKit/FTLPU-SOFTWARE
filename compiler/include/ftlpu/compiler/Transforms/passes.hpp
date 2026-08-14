#pragma once

#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace ftlpu::compiler {
enum class FfnScheduleStrategy {
    Tail,
    Fused,
};

enum class AttentionScheduleStrategy {
    Tail,
    Fused,
};

enum class RmsNormLoweringStrategy {
    VxmSquareMxmReduce,
    VxmFeedback,
};

std::unique_ptr<mlir::Pass> create_lower_stablehlo_to_kernel_pass();
std::unique_ptr<mlir::Pass> create_lower_kernel_to_tensor_pass(
    RmsNormLoweringStrategy rmsnorm_strategy =
        RmsNormLoweringStrategy::VxmSquareMxmReduce,
    std::int64_t weight_bank = -1);
std::unique_ptr<mlir::Pass> create_lower_tensor_to_stream_pass();
std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass(
    FfnScheduleStrategy ffn_strategy = FfnScheduleStrategy::Tail,
    AttentionScheduleStrategy attention_strategy =
        AttentionScheduleStrategy::Tail,
    bool stage_timing = false);
std::unique_ptr<mlir::Pass> create_assign_weight_bank_pass(
    std::int64_t bank);
std::unique_ptr<mlir::Pass> create_compress_schedule_pass();
std::unique_ptr<mlir::Pass> create_verify_schedule_pass();
std::unique_ptr<mlir::Pass> create_lower_schedule_to_command_pass();
void register_ftlpu_passes();
} // namespace ftlpu::compiler
