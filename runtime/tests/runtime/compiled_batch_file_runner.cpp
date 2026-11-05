#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <map>
#include <cmath>
#include <cstdlib>

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
    // Optional ordered input lists allow real paged checkpoint tensors. Weight
    // files are shared across requests; activations and histories stay private.
    const bool lists = std::filesystem::exists(directory / "request0" / "inputs.txt");
    std::map<std::string, std::vector<std::uint8_t>> externalInputs;
    std::map<std::string, std::string> sharedTensors;
    if (prefetch)
        for (int i=4;i<7;++i) package.executables.push_back({"variant"+std::to_string(i),read_binary_program(argv[i])});
    for (std::size_t request = 0; request < batch; ++request) {
        ModelInvocation invocation;
        invocation.name = "request" + std::to_string(request);
        invocation.executable_index = prefetch ? (request+1 < batch ? 2 : 0) + request%2 : 0;
        std::vector<std::string> paths;
        if (lists) {
            std::ifstream listing(directory / invocation.name / "inputs.txt");
            if (!listing) throw std::runtime_error("missing request input list");
            std::string line;
            while (std::getline(listing, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) paths.push_back(line);
            }
        }
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
            if (input && lists) {
                auto path = paths.at(binding.index);
                const auto separator = path.find('\t');
                const bool scaled = separator != std::string::npos;
                const float scale = scaled ? std::stof(path.substr(separator + 1)) : 1.f;
                if (scaled && (!std::isfinite(scale) || scale <= 0 || binding.element_type != BindingElementType::I8))
                    throw std::runtime_error("invalid INT8 input scale");
                const auto key = path;
                if (scaled) path.resize(separator);
                std::ifstream file(path, std::ios::binary);
                if (!file) throw std::runtime_error("cannot read " + path);
                std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
                if (binding.paged_weight || scaled) {
                    auto [it, inserted] = sharedTensors.emplace(key, name);
                    if (inserted) {
                        package.tensors.push_back({name, binding.element_type, binding.shape, std::move(bytes)});
                        if (scaled) {
                            package.tensors.back().encoding = ModelTensorEncoding::SymmetricPerTensorI8;
                            package.tensors.back().scales = {scale};
                        }
                    }
                    invocation.inputs.push_back({binding.index, it->second});
                    continue;
                }
                externalInputs.emplace(name, std::move(bytes));
            }
            package.values.push_back({name, binding.element_type, binding.shape, input, !input});
            (input ? invocation.inputs : invocation.outputs).push_back({binding.index, name});
        }
        package.invocations.push_back(std::move(invocation));
    }
    const char* sxmChoice = std::getenv("FTLPU_FILE_RUNNER_SXM_ARCHITECTURE");
    if (sxmChoice && std::string(sxmChoice) != "wide" && std::string(sxmChoice) != "local")
        throw std::runtime_error("SXM architecture must be wide or local");
    const auto sxmArchitecture = sxmChoice && std::string(sxmChoice) == "wide"
        ? ftlpu::SxmTransposeArchitecture::WideStreaming
        : ftlpu::SxmTransposeArchitecture::LocalBlocks;
    std::cout << "sxm_architecture=" << (sxmArchitecture == ftlpu::SxmTransposeArchitecture::WideStreaming ? "wide" : "local") << '\n';
    auto dma = std::make_unique<ftlpu::C2cDmaSystem>(ftlpu::Ddr4Config{},
        ftlpu::hw::config::kDmaFifoDepthVectors, ftlpu::SystemHardwareConfiguration{}, sxmArchitecture);
    ModelSession session(*dma);
    session.enable_execution_trace();
    session.load(std::move(package));
    for (const auto& [name, bytes] : externalInputs) session.set_input(name, bytes);
    // All external values are fixed before prefetch starts. Updating an input
    // later invalidates its resident image through ModelSession::set_input.
    for (std::size_t request = 0; request < batch; ++request) {
        const auto requestDir = directory / ("request" + std::to_string(request));
        for (const auto& input : session.package().invocations[request].inputs) {
            if (lists) continue;
            if (input.value != "request"+std::to_string(request)+".input"+std::to_string(input.binding_index)) continue;
            const auto path = requestDir / ("input" + std::to_string(input.binding_index) + ".bin");
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("cannot read " + path.string());
            const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
            session.set_input(input.value, bytes);
        }
    }
    std::filesystem::create_directories(directory / "invocation_programs");
    std::size_t computeSegments = 0;
    for (std::size_t request = 0; request < batch; ++request) {
        const auto requestDir = directory / ("request" + std::to_string(request));
        session.run_invocation(request, 128);
        session.write_last_linked_program(directory / "invocation_programs" / ("invocation"+std::to_string(request)+".ftlpu"));
        const auto segments = plan_compute_imem_segments(session.last_linked_program());
        computeSegments += segments.size();
        for (std::size_t segment = 0; segment < segments.size(); ++segment)
            write_binary_program(segments[segment], directory / "invocation_programs" /
                ("invocation"+std::to_string(request)+".compute_segment"+std::to_string(segment)+".ftlpu"));
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
        << ",\"compute_imem_segments\":" << computeSegments
        << ",\"weight_page_prefetches\":" << stats.weight_page_prefetches
        << ",\"weight_page_runtime_wait_cycles\":" << stats.weight_page_runtime_wait_cycles
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
