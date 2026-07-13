#pragma once

#include "ftlpu/compiler/ir.hpp"

#include <string>
#include <vector>

namespace ftlpu::compiler {

Module lower_stablehlo_to_tensor(const Module& module);
Module lower_tensor_to_stream(const Module& module, std::size_t tile_size = 20);
Module lower_stream_to_schedule(const Module& module, std::size_t tile_size = 20);

Module run_pipeline(Module module, const std::vector<std::string>& pass_names);
std::vector<std::string> split_pipeline(const std::string& pipeline);

} // namespace ftlpu::compiler
