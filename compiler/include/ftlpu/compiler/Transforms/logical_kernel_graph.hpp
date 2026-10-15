#pragma once

#include <filesystem>

namespace ftlpu::compiler {
// Compile a model-independent graph of Kernel IR regions and logical tensor
// views to one resident, paged Schedule module. Checkpoint input tiles are
// materialized in inputsDirectory alongside the generated inputs.txt.
void compile_logical_kernel_graph(const std::filesystem::path &input,
                                  const std::filesystem::path &output,
                                  const std::filesystem::path &targetConfig,
                                  const std::filesystem::path &inputsDirectory);
} // namespace ftlpu::compiler
