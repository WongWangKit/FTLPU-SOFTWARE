// Keep the tool rebuilt with the BinaryProgram return ABI.
#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/software/runtime/binary.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
try {
    std::filesystem::path input;
    std::filesystem::path output;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (++i >= argc) throw std::runtime_error("missing value for " + argument);
        if (argument == "--input") input = argv[i];
        else if (argument == "--output") output = argv[i];
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (input.empty() || output.empty())
        throw std::runtime_error("usage: ftlpu-translate --input command.mlir --output program.ftlpu");

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, ftlpu::compiler::command::CommandDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(input.string(), &context);
    if (!module) return 1;

    std::error_code error;
    std::filesystem::create_directories(output.parent_path(), error);
    auto program = ftlpu::compiler::target::translate_command_module(*module);
    ftlpu::software::runtime::write_binary_program(program, output);
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu-translate failed: " << ex.what() << '\n';
    return 1;
}
