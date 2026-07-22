#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace ftlpu::compiler {
std::unique_ptr<mlir::Pass> create_lower_stablehlo_to_kernel_pass();
std::unique_ptr<mlir::Pass> create_lower_kernel_to_tensor_pass();
std::unique_ptr<mlir::Pass> create_lower_tensor_to_stream_pass();
std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass();
std::unique_ptr<mlir::Pass> create_lower_schedule_to_command_pass();
void register_ftlpu_passes();
} // namespace ftlpu::compiler
