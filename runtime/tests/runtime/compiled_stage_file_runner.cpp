#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

// Generic harness: no model arithmetic, shape transforms or reference outputs.
// Each compiled stage executes its actual ICU program against uploaded bindings.
int main(int argc, char** argv)
try {
    using namespace ftlpu::software::runtime;
    if (argc != 4 && argc != 5) throw std::runtime_error("usage: compiled_stage_file_runner program.ftlpu inputs.txt output-prefix [runtime-pipeline.csv]");
#ifdef FTLPU_FILE_RUNNER_LINK_CHECK
#ifdef _WIN32
    _putenv_s("FTLPU_SESSION_LINK_ONLY", "1");
#else
    setenv("FTLPU_SESSION_LINK_ONLY", "1", 1);
#endif
#endif
    const auto program = read_binary_program(argv[1]);
    const auto capacity = analyze_physical_imem(program);
    for (const auto& queue : capacity.queues)
        if (queue.physical_slots > queue.depth)
            std::cerr << "iMEM overflow kind=" << static_cast<int>(queue.kind)
                      << " queue=" << queue.index << " slots=" << queue.physical_slots
                      << " depth=" << queue.depth << '\n';
    if (!program.weight_page_uses.empty()
        || std::getenv("FTLPU_FILE_RUNNER_MODEL_SESSION") != nullptr) {
        // Use the generic executable session for paging, or explicitly for
        // C2C I/O of a resident program. Paged compiler images already carry
        // their MEM_WRITE_SYNC and page SYNC contracts.
        ModelPackage package;
        package.model_name = "compiled_program";
        package.architecture = "native_icu";
        package.executables.push_back({"layer", program});
        ModelInvocation invocation;
        invocation.name = "layer";
        std::vector<std::pair<std::string,std::vector<std::uint8_t>>> initialInputs;
        std::ifstream listing(argv[2]);
        if (!listing) throw std::runtime_error("cannot read input list");
        std::string path;
        std::uint32_t index = 0;
        while (std::getline(listing, path)) {
            if (!path.empty() && path.back() == '\r') path.pop_back();
            if (path.empty()) continue;
            const auto binding = std::find_if(program.bindings.begin(), program.bindings.end(),
                [&](const auto& b) { return b.access == BindingAccess::Input && b.index == index; });
            if (binding == program.bindings.end()) throw std::runtime_error("missing input binding");
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("cannot read input: " + path);
            std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
            const auto name = "input" + std::to_string(index);
            if (binding->paged_weight) {
                package.tensors.push_back({name, binding->element_type, binding->shape, std::move(bytes)});
            } else {
                // The compiler has already allocated every immutable input
                // and intermediate. Preserve those addresses; a second
                // resident allocator must not reserve all lifetimes at once.
                package.values.push_back({name,binding->element_type,binding->shape,true,false});
                initialInputs.emplace_back(name,std::move(bytes));
            }
            invocation.inputs.push_back({index++, name});
        }
        for (const auto& b : program.bindings) {
            if (b.access != BindingAccess::Output) continue;
            const auto name = "output" + std::to_string(b.index);
            package.values.push_back({name, b.element_type, b.shape, false, true});
            invocation.outputs.push_back({b.index, name});
        }
        package.invocations.push_back(invocation);
        auto dma = std::make_unique<ftlpu::C2cDmaSystem>();
        ModelSession session(*dma);
        if (argc == 5) session.enable_execution_trace();
        session.load(std::move(package));
        for (const auto& [name,bytes] : initialInputs) session.set_input(name,bytes);
        try { session.run(128); }
        catch (...) {
            if (std::getenv("FTLPU_FILE_RUNNER_DEBUG_ICU") != nullptr) {
                for (std::size_t queue = 0; queue < ftlpu::InstructionControlUnit::kMemQueues; ++queue) {
                    const auto& state = dma->chip().icu().mem_iq(queue);
                    const auto& trace = state.last_trace();
                    if (trace.issue_pc)
                        std::cerr << "debug mem queue=" << queue << " local_cycle=" << trace.cycle
                                  << " pc=" << *trace.issue_pc << '\n';
                }
            }
            try {
                session.write_last_linked_program(std::string(argv[3]) + ".runtime.ftlpu");
                session.write_pre_execution_icu_programs(std::string(argv[3]) + ".pre_execution");
                if (argc == 5) session.write_execution_trace_csv(argv[4]);
                if (std::getenv("FTLPU_FILE_RUNNER_FAILURE_SNAPSHOT") != nullptr) {
                    // Debug only: preserve raw final SRAM values if external
                    // transport fails. These are never validation outputs.
                    CModelRuntime snapshot(dma->chip());
                    snapshot.load(read_binary_program(std::string(argv[3]) + ".runtime.ftlpu"), false);
                    for (const auto& b : program.bindings) {
                        if (b.access != BindingAccess::Output) continue;
                        const auto bytes = snapshot.download_output(b.index);
                        std::ofstream file(std::string(argv[3]) + ".diagnostic." + std::to_string(b.index) + ".bin", std::ios::binary);
                        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    }
                }
            } catch (...) {}
            throw;
        }
        session.write_last_linked_program(std::string(argv[3]) + ".runtime.ftlpu");
        session.write_pre_execution_icu_programs(std::string(argv[3]) + ".pre_execution");
        if (argc == 5) session.write_execution_trace_csv(argv[4]);
        if (std::getenv("FTLPU_SESSION_LINK_ONLY") != nullptr) {
            std::cout << "linked_only=1 arithmetic_executed=0\n";
            return 0;
        }
        for (const auto& b : program.bindings) {
            if (b.access != BindingAccess::Output) continue;
            const auto& bytes = session.value("output" + std::to_string(b.index));
            std::ofstream file(std::string(argv[3]) + "." + std::to_string(b.index) + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        const auto& stats = session.stats();
        std::cout << "cycles=" << program.max_cycle + 128
                  << " outputs=" << invocation.outputs.size()
                  << " execution_epochs=" << stats.execution_epochs
                  << " weight_page_prefetches=" << stats.weight_page_prefetches
                  << " weight_page_prefetch_bytes=" << stats.weight_page_prefetch_bytes
                  << " weight_page_runtime_wait_cycles=" << stats.weight_page_runtime_wait_cycles
                  << " synchronized_writes=" << stats.weight_page_synchronized_writes
                  << " host_operations=" << stats.host_operations << '\n';
        return 0;
    }
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    CModelRuntime runtime(*system);
    if (argc == 5) runtime.enable_execution_trace();
    runtime.load(program);
    if (std::getenv("FTLPU_SESSION_LINK_ONLY") != nullptr) {
        write_binary_program(program, std::string(argv[3]) + ".runtime.ftlpu");
        std::cout << "linked_only=1 arithmetic_executed=0\n";
        return 0;
    }
    std::ifstream inputList(argv[2]);
    if (!inputList) throw std::runtime_error("cannot read input list");
    std::string path;
    std::size_t index = 0;
    while (std::getline(inputList, path)) {
        if (!path.empty() && path.back() == '\r') path.pop_back();
        if (path.empty()) continue;
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot read input: " + path);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
        runtime.upload_input(index++, bytes);
    }
    runtime.run_cycles(program.max_cycle + 128);
    if (argc == 5) runtime.write_execution_trace_csv(argv[4]);
    std::size_t outputs = 0;
    for (const auto& binding : program.bindings) {
        if (binding.access != BindingAccess::Output) continue;
        const auto bytes = runtime.download_output(binding.index);
        std::ofstream file(std::string(argv[3]) + "." + std::to_string(binding.index) + ".bin", std::ios::binary);
        if (!file) throw std::runtime_error("cannot write output");
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ++outputs;
    }
    if (!outputs) throw std::runtime_error("program has no output bindings");
    std::cout << "cycles=" << program.max_cycle + 128 << " outputs=" << outputs << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
