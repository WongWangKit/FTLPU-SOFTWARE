#include "ftlpu/compiler/ir.hpp"
#include "ftlpu/compiler/Pipelines/pipelines.hpp"
#include "ftlpu/compiler/Target/ftlpu_cmodel_target.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::string pipeline{"ftlpu-lpu-pipeline"};
    ftlpu::compiler::pipeline::Options pipeline_options;
};

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + arg);
            }
            return argv[i];
        };
        if (arg == "--input") {
            args.input = next();
        } else if (arg == "--output") {
            args.output = next();
        } else if (arg == "--pipeline") {
            args.pipeline = next();
        } else if (arg == "--compile-from") {
            args.pipeline_options.compile_from = ftlpu::compiler::pipeline::parse_phase(next().c_str());
        } else if (arg == "--compile-to") {
            args.pipeline_options.compile_to = ftlpu::compiler::pipeline::parse_phase(next().c_str());
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.input.empty() || args.output.empty()) {
        throw std::runtime_error(
            "usage: ftlpu_opt --input in.mlir --output out.mlir "
            "[--pipeline ftlpu-lpu-pipeline] [--compile-to schedule]");
    }
    return args;
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file");
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
}

void write_file(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output file");
    }
    output << text;
}

} // namespace

int main(int argc, char** argv)
try {
    const auto args = parse_args(argc, argv);
    const auto target = ftlpu::compiler::target::FtlpuCModelTarget {};
    auto module = ftlpu::compiler::parse_stablehlo_module(read_file(args.input));
    module = ftlpu::compiler::pipeline::run_named_pipeline(module, target, args.pipeline, args.pipeline_options);
    write_file(args.output, ftlpu::compiler::print_module(module));
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu-opt failed: " << ex.what() << '\n';
    return 1;
}
