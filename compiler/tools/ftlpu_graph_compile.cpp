#include "ftlpu/compiler/Transforms/logical_kernel_graph.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
namespace fs = std::filesystem;
int main(int argc, char **argv) try {
  fs::path input, output, config, directory;
  bool emitKernel = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--emit-kernel") {emitKernel = true; continue;}
    if (i + 1 == argc)
      throw std::runtime_error("missing option value");
    auto value = argv[++i];
    if (arg == "--input")
      input = value;
    else if (arg == "--output")
      output = value;
    else if (arg == "--target-config")
      config = value;
    else if (arg == "--inputs-dir")
      directory = value;
    else
      throw std::runtime_error("unknown option " + arg);
  }
  if (input.empty() || output.empty() || (!emitKernel && directory.empty()))
    throw std::runtime_error(
        "usage: ftlpu-graph-compile --input graph.json --output schedule.mlir "
        "--inputs-dir dir [--target-config target.json]");
  if (emitKernel) ftlpu::compiler::import_logical_kernel_graph(input, output);
  else ftlpu::compiler::compile_logical_kernel_graph(input, output, config, directory);
  return 0;
} catch (const std::exception &e) {
  std::cerr << "ftlpu-graph-compile: " << e.what() << "\n";
  return 1;
}
