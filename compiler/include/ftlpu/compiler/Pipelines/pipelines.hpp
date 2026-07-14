#pragma once

#include "ftlpu/compiler/Pipelines/phases.hpp"
#include "ftlpu/compiler/Target/target_backend.hpp"
#include "ftlpu/compiler/ir.hpp"

#include <string>
#include <vector>

namespace ftlpu::compiler::pipeline {

struct Options {
    std::size_t tile_size{20};
    std::size_t south_to_north_tiles{20};
    Phase compile_from{Phase::Start};
    Phase compile_to{Phase::End};
};

Module lower_stablehlo_to_kernel(const Module& module, const target::TargetBackend& target);
Module lower_kernel_to_tensor(const Module& module, const target::TargetBackend& target, std::size_t tile_size = 20);
Module lower_tensor_to_stream(const Module& module, const target::TargetBackend& target, std::size_t south_to_north_tiles = 20);
Module lower_stream_to_schedule(const Module& module, const target::TargetBackend& target, std::size_t south_to_north_tiles = 20);

Module run_ftlpu_pipeline(Module module, const target::TargetBackend& target, const Options& options = {});
Module run_named_pipeline(Module module, const target::TargetBackend& target, const std::string& pipeline, const Options& options = {});
Module run_legacy_pipeline(Module module, const target::TargetBackend& target, const std::vector<std::string>& pass_names, const Options& options = {});

std::vector<std::string> split_pipeline(const std::string& pipeline);

} // namespace ftlpu::compiler::pipeline
