#include "ftlpu/compiler/Pipelines/pipelines.hpp"

#include <stdexcept>

namespace ftlpu::compiler::pipeline {

std::vector<std::string> split_pipeline(const std::string& pipeline)
{
    std::vector<std::string> passes;
    std::string current;
    for (const auto ch : pipeline) {
        if (ch == ',') {
            if (!current.empty()) {
                passes.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        passes.push_back(current);
    }
    return passes;
}

Module run_ftlpu_pipeline(Module module, const target::TargetBackend& target, const Options& options)
{
    if (options.compile_from < Phase::Kernel && options.compile_to >= Phase::Kernel) {
        module = lower_stablehlo_to_kernel(module, target);
    }
    if (options.compile_from < Phase::Tensor && options.compile_to >= Phase::Tensor) {
        module = lower_kernel_to_tensor(module, target, options.tile_size);
    }
    if (options.compile_from < Phase::Stream && options.compile_to >= Phase::Stream) {
        module = lower_tensor_to_stream(module, target, options.south_to_north_tiles);
    }
    if (options.compile_from < Phase::Schedule && options.compile_to >= Phase::Schedule) {
        module = lower_stream_to_schedule(module, target, options.south_to_north_tiles);
    }
    if (options.compile_to >= Phase::ExecutableBinary) {
        throw std::runtime_error("executable-binary phase is not implemented yet");
    }
    return module;
}

Module run_named_pipeline(Module module, const target::TargetBackend& target, const std::string& pipeline, const Options& options)
{
    if (pipeline == "ftlpu-lpu-pipeline" || pipeline == "default") {
        return run_ftlpu_pipeline(module, target, options);
    }
    return run_legacy_pipeline(module, target, split_pipeline(pipeline), options);
}

Module run_legacy_pipeline(Module module, const target::TargetBackend& target, const std::vector<std::string>& pass_names, const Options& options)
{
    for (const auto& pass : pass_names) {
        if (pass == "stablehlo-to-kernel") {
            module = lower_stablehlo_to_kernel(module, target);
        } else if (pass == "kernel-to-tensor") {
            module = lower_kernel_to_tensor(module, target, options.tile_size);
        } else if (pass == "stablehlo-to-tensor") {
            module = lower_kernel_to_tensor(lower_stablehlo_to_kernel(module, target), target, options.tile_size);
        } else if (pass == "tensor-to-stream") {
            module = lower_tensor_to_stream(module, target, options.south_to_north_tiles);
        } else if (pass == "stream-to-schedule") {
            module = lower_stream_to_schedule(module, target, options.south_to_north_tiles);
        } else if (pass == "ftlpu-lpu-pipeline" || pass == "default") {
            module = run_ftlpu_pipeline(module, target, options);
        } else {
            throw std::runtime_error("unknown compiler pass or pipeline: " + pass);
        }
    }
    return module;
}

} // namespace ftlpu::compiler::pipeline
