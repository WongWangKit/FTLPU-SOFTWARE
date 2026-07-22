#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::string pipeline{"ftlpu-stablehlo-to-kernel"};
};

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--input") args.input = next();
        else if (arg == "--output") args.output = next();
        else if (arg == "--pipeline") args.pipeline = next();
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (args.input.empty() || args.output.empty()) {
        throw std::runtime_error("usage: ftlpu-opt --input in.mlir --output out.mlir "
                                 "[--pipeline ftlpu-stablehlo-to-kernel|ftlpu-stablehlo-to-tensor|"
                                 "ftlpu-stablehlo-to-stream|ftlpu-stablehlo-to-schedule|"
                                 "ftlpu-stablehlo-to-commands]");
    }
    return args;
}

} // namespace

int main(int argc, char** argv)
try {
    const auto args = parse_args(argc, argv);
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
        ftlpu::compiler::kernel::KernelDialect, ftlpu::compiler::tensor::TensorDialect>();
    registry.insert<ftlpu::compiler::stream::StreamDialect,
        ftlpu::compiler::schedule::ScheduleDialect,
        ftlpu::compiler::command::CommandDialect>();
    mlir::MLIRContext context(registry);
    context.disableMultithreading();
    context.getOrLoadDialect<ftlpu::compiler::kernel::KernelDialect>();
    context.getOrLoadDialect<ftlpu::compiler::tensor::TensorDialect>();
    context.getOrLoadDialect<ftlpu::compiler::stream::StreamDialect>();
    context.getOrLoadDialect<ftlpu::compiler::schedule::ScheduleDialect>();
    context.getOrLoadDialect<ftlpu::compiler::command::CommandDialect>();

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(args.input.string(), &context);
    if (!module) return 1;

    mlir::PassManager pass_manager(&context);
    if (args.pipeline != "ftlpu-stablehlo-to-kernel"
        && args.pipeline != "ftlpu-stablehlo-to-tensor"
        && args.pipeline != "ftlpu-stablehlo-to-stream"
        && args.pipeline != "ftlpu-stablehlo-to-schedule"
        && args.pipeline != "ftlpu-stablehlo-to-commands") {
        throw std::runtime_error("unknown MLIR pipeline: " + args.pipeline);
    }
    pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_stablehlo_to_kernel_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-tensor" || args.pipeline == "ftlpu-stablehlo-to-stream"
        || args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_kernel_to_tensor_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-stream" || args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_tensor_to_stream_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-schedule" || args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_stream_to_schedule_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_schedule_to_command_pass());
    if (mlir::failed(pass_manager.run(*module))) return 1;

    std::error_code error;
    std::filesystem::create_directories(args.output.parent_path(), error);
    std::string output_error;
    auto output = mlir::openOutputFile(args.output.string(), &output_error);
    if (!output) throw std::runtime_error(output_error);
    module->print(output->os());
    output->os() << '\n';
    output->keep();
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu-opt failed: " << ex.what() << '\n';
    return 1;
}
