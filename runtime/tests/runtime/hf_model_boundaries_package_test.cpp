#include "ftlpu/software/runtime/model_package.hpp"
#include "ftlpu/software/runtime/session_memory_planner.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: hf_model_boundaries_package_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    const auto start = std::chrono::steady_clock::now();
    const ModelPackage package = read_model_package(
        std::filesystem::path(argv[1]),
        ModelPackageLoadMode::LazyExecutables);
    const auto loaded = std::chrono::steady_clock::now();
    std::cout << "package metadata loaded"
              << " tensors=" << package.tensors.size()
              << " executables=" << package.executables.size()
              << " invocations=" << package.invocations.size()
              << " load_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     loaded - start).count()
              << std::endl;
    if (package.embedding_lookups.size() != 1
        || package.host_lm_heads.size() != 1
        || package.host_lm_heads[0].weight
            != package.embedding_lookups[0].table
        || !package.host_lm_heads[0].last_token_only)
        throw std::logic_error(
            "model package does not share the embedding and LM-head table");
    const auto embedding = std::find_if(package.tensors.begin(),
        package.tensors.end(), [&](const ModelTensor& tensor) {
            return tensor.name == package.embedding_lookups[0].table;
        });
    if (embedding == package.tensors.end()
        || embedding->element_type != BindingElementType::BF16)
        throw std::logic_error(
            "model package embedding/LM-head table is not BF16");
    if (package.invocations.empty()
        || package.invocations.back().name != "final_norm"
        || package.host_lm_heads[0].hidden != "final_hidden"
        || package.host_lm_heads[0].output != "logits")
        throw std::logic_error(
            "model package does not connect final RMSNorm to host LM head");
    for (const ModelExecutable& executable : package.executables)
        if (executable.serialized_program.empty()
            || executable.program.bindings.empty()
            || !executable.program.queues.empty())
            throw std::logic_error(
                "model package executable was not loaded lazily");
    const SessionMemoryPlan memory_plan =
        SessionMemoryPlanner::plan(package);
    if (memory_plan.invocations.size() != package.invocations.size())
        throw std::logic_error(
            "model package global memory plan is incomplete");
    const auto planned = std::chrono::steady_clock::now();
    std::cout << "global memory plan completed"
              << " residents=" << memory_plan.resident_tensors.size()
              << " plan_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     planned - loaded).count()
              << std::endl;

    std::cout << "hf_model_boundaries_package_test passed"
              << " tensors=" << package.tensors.size()
              << " executables=" << package.executables.size()
              << " invocations=" << package.invocations.size()
              << " residents=" << memory_plan.resident_tensors.size()
              << " load_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     loaded - start).count()
              << " plan_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     planned - loaded).count()
              << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_model_boundaries_package_test failed: "
              << exception.what() << '\n';
    return 1;
}
