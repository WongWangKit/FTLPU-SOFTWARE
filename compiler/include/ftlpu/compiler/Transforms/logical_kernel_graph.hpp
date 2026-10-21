#pragma once

#include <filesystem>
#include "mlir/IR/BuiltinOps.h"

namespace ftlpu::compiler {
// Compile a model-independent graph of Kernel IR regions and logical tensor
// views to one resident, paged Schedule module. Checkpoint input tiles are
// materialized in inputsDirectory alongside the generated inputs.txt.
void compile_logical_kernel_graph(const std::filesystem::path &input,
                                  const std::filesystem::path &output,
                                  const std::filesystem::path &targetConfig,
                                  const std::filesystem::path &inputsDirectory);

// Import semantic regions/views as a single SSA function. No hardware lowering
// is performed by this entry; the result can be inspected and compiled through
// the ordinary --input-stage kernel CLI.
void import_logical_kernel_graph(const std::filesystem::path &input,
                                const std::filesystem::path &output);
mlir::OwningOpRef<mlir::ModuleOp> lower_connected_decoder_graph(
    mlir::ModuleOp module, const std::filesystem::path &inputsDirectory);
} // namespace ftlpu::compiler
