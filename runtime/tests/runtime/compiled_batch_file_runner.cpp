#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>

// Execute independent requests with one shared executable. No tensor arithmetic
// or instruction synthesis: ModelSession owns C2C ingress, execution, egress,
// and compiler-declared next-invocation input prefetches.
int main(int argc, char** argv) try {
    using namespace ftlpu::software::runtime;
    if (argc != 4 && argc != 7) throw std::runtime_error(
        "usage: compiled_batch_file_runner program.ftlpu fixture-directory batch [bank1-terminal bank0-prefetch bank1-prefetch]");
    const auto program = read_binary_program(argv[1]);
    const std::filesystem::path directory(argv[2]);
    const auto batch = std::stoul(argv[3]);
    if (!batch) throw std::runtime_error("batch must be positive");
    ModelPackage package;
    package.model_name = "independent_requests";
    package.architecture = "native_icu";
    package.executables.push_back({"step", program});
    const bool prefetch = argc == 7;
    if (prefetch)
        for (int i=4;i<7;++i) package.executables.push_back({"variant"+std::to_string(i),read_binary_program(argv[i])});
    for (std::size_t request = 0; request < batch; ++request) {
        ModelInvocation invocation;
        invocation.name = "request" + std::to_string(request);
        invocation.executable_index = prefetch ? (request+1 < batch ? 2 : 0) + request%2 : 0;
        for (const auto& binding : package.executables[invocation.executable_index].program.bindings) {
            if (binding.access != BindingAccess::Input &&
                binding.access != BindingAccess::Output) continue;
            const bool input = binding.access == BindingAccess::Input;
            if (input && binding.role == "prefetch.input") {
                invocation.inputs.push_back({binding.index,"request"+std::to_string(request+1)+".input5"});
                continue;
            }
            const auto name = invocation.name + (input ? ".input" : ".output")
                + std::to_string(binding.index);
            package.values.push_back({name, binding.element_type, binding.shape, input, !input});
            (input ? invocation.inputs : invocation.outputs).push_back({binding.index, name});
        }
        package.invocations.push_back(std::move(invocation));
    }
    auto dma = std::make_unique<ftlpu::C2cDmaSystem>();
    ModelSession session(*dma);
    session.enable_execution_trace();
    session.load(std::move(package));
    // All external values are fixed before prefetch starts. Updating an input
    // later invalidates its resident image through ModelSession::set_input.
    for (std::size_t request = 0; request < batch; ++request) {
        const auto requestDir = directory / ("request" + std::to_string(request));
        for (const auto& input : session.package().invocations[request].inputs) {
            if (input.value != "request"+std::to_string(request)+".input"+std::to_string(input.binding_index)) continue;
            const auto path = requestDir / ("input" + std::to_string(input.binding_index) + ".bin");
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("cannot read " + path.string());
            const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
            session.set_input(input.value, bytes);
        }
    }
    std::filesystem::create_directories(directory / "invocation_programs");
    for (std::size_t request = 0; request < batch; ++request) {
        const auto requestDir = directory / ("request" + std::to_string(request));
        session.run_invocation(request, 128);
        session.write_last_linked_program(directory / "invocation_programs" / ("invocation"+std::to_string(request)+".ftlpu"));
        for (const auto& output : session.package().invocations[request].outputs) {
            const auto& bytes = session.value(output.value);
            std::ofstream file(requestDir / ("actual." + std::to_string(output.binding_index) + ".bin"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!file) throw std::runtime_error("cannot write request output");
        }
        std::cout << "completed_request=" << request << std::endl;
    }
    session.write_execution_trace_csv(directory / "runtime.pipeline.csv");
    session.write_last_linked_program(directory / "actual.runtime.ftlpu");
    session.write_pre_execution_icu_programs(directory / "actual.pre_execution");
    const auto& stats = session.stats();
    std::ofstream report(directory / "execution.json");
    report << "{\"requests\":" << batch
        << ",\"host_operations\":" << stats.host_operations
        << ",\"prefetched_input_hits\":" << stats.prefetched_input_hits
        << ",\"prefetched_input_bytes\":" << stats.weight_page_prefetch_bytes
        << ",\"c2c_ingress_bytes\":" << stats.c2c_ingress_bytes
        << ",\"c2c_egress_bytes\":" << stats.c2c_egress_bytes
        << ",\"c2c_ingress_cycles\":" << stats.c2c_ingress_cycles
        << ",\"c2c_egress_cycles\":" << stats.c2c_egress_cycles << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
