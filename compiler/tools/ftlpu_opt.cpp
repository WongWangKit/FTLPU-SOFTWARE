#include "ftlpu/compiler/ir.hpp"
#include "ftlpu/compiler/passes.hpp"

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
    std::string pipeline;
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
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.input.empty() || args.output.empty() || args.pipeline.empty()) {
        throw std::runtime_error(
            "usage: ftlpu_opt --input in.mlir --output out.mlir "
            "--pipeline stablehlo-to-tensor,tensor-to-stream,stream-to-schedule");
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
    auto module = ftlpu::compiler::parse_stablehlo_module(read_file(args.input));
    module = ftlpu::compiler::run_pipeline(module, ftlpu::compiler::split_pipeline(args.pipeline));
    write_file(args.output, ftlpu::compiler::print_module(module));
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu-opt failed: " << ex.what() << '\n';
    return 1;
}
