#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"
#include "ftlpu/software/runtime/binary.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

enum class InputStage {
    StableHlo,
    Stream,
    Schedule,
    VerifiedSchedule,
    Command,
};

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path target_config;
    InputStage input_stage{InputStage::StableHlo};
    ftlpu::compiler::FfnScheduleStrategy ffn_schedule{
        ftlpu::compiler::FfnScheduleStrategy::Tail};
    ftlpu::compiler::AttentionScheduleStrategy attention_schedule{
        ftlpu::compiler::AttentionScheduleStrategy::Tail};
    ftlpu::compiler::RmsNormLoweringStrategy rmsnorm_strategy{
        ftlpu::compiler::RmsNormLoweringStrategy::VxmSquareMxmReduce};
    ftlpu::compiler::target::MxmExecutionPolicy mxm_execution_policy{
        ftlpu::compiler::target::MxmExecutionPolicy::Auto};
    std::int64_t weight_bank{-1};
    bool pass_timing{false};
};

InputStage parse_input_stage(const std::string& value)
{
    if (value == "stablehlo") return InputStage::StableHlo;
    if (value == "stream") return InputStage::Stream;
    if (value == "schedule") return InputStage::Schedule;
    if (value == "verified-schedule") return InputStage::VerifiedSchedule;
    if (value == "command") return InputStage::Command;
    throw std::runtime_error("unknown input stage: " + value);
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc)
                throw std::runtime_error("missing value for " + argument);
            return argv[i];
        };
        if (argument == "--input") args.input = next();
        else if (argument == "--output") args.output = next();
        else if (argument == "--input-stage")
            args.input_stage = parse_input_stage(next());
        else if (argument == "--target-config") args.target_config = next();
        else if (argument == "--weight-bank")
            args.weight_bank = std::stoll(next());
        else if (argument == "--pass-timing")
            args.pass_timing = true;
        else if (argument == "--ffn-schedule") {
            const std::string value = next();
            if (value == "tail")
                args.ffn_schedule =
                    ftlpu::compiler::FfnScheduleStrategy::Tail;
            else if (value == "fused")
                args.ffn_schedule =
                    ftlpu::compiler::FfnScheduleStrategy::Fused;
            else
                throw std::runtime_error(
                    "unknown FFN schedule strategy: " + value);
        } else if (argument == "--attention-schedule") {
            const std::string value = next();
            if (value == "tail")
                args.attention_schedule =
                    ftlpu::compiler::AttentionScheduleStrategy::Tail;
            else if (value == "fused")
                args.attention_schedule =
                    ftlpu::compiler::AttentionScheduleStrategy::Fused;
            else
                throw std::runtime_error(
                    "unknown Attention schedule strategy: " + value);
        } else if (argument == "--rmsnorm-strategy") {
            const std::string value = next();
            if (value == "vxm-square-mxm-reduce")
                args.rmsnorm_strategy = ftlpu::compiler::
                    RmsNormLoweringStrategy::VxmSquareMxmReduce;
            else if (value == "vxm-feedback")
                args.rmsnorm_strategy = ftlpu::compiler::
                    RmsNormLoweringStrategy::VxmFeedback;
            else
                throw std::runtime_error(
                    "unknown RMSNorm lowering strategy: " + value);
        } else if (argument == "--mxm-execution") {
            const std::string value = next();
            auto parsed = ftlpu::compiler::target::
                parse_mxm_execution_policy(value);
            if (mlir::failed(parsed))
                throw std::runtime_error(
                    "unknown MXM execution policy: " + value);
            args.mxm_execution_policy = *parsed;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (args.input.empty() || args.output.empty())
        throw std::runtime_error(
            "usage: ftlpu-compile --input in.mlir --output program.ftlpu "
            "[--input-stage stablehlo|stream|schedule|verified-schedule|command] "
            "[--target-config target.json] [--weight-bank 0|1] "
            "[--mxm-execution auto|legacy|block8] "
            "[--ffn-schedule tail|fused] "
            "[--attention-schedule tail|fused] "
            "[--rmsnorm-strategy vxm-square-mxm-reduce|vxm-feedback]");
    return args;
}

ftlpu::compiler::target::LPUTargetModel load_target(
    const std::filesystem::path& path)
{
    ftlpu::compiler::target::LPUTargetModel target;
    if (path.empty()) return target;
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(
            "cannot read target configuration: " + path.string());
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string error;
    auto parsed = ftlpu::compiler::target::LPUTargetModel::from_json(
        buffer.str(), error);
    if (mlir::failed(parsed))
        throw std::runtime_error("invalid target configuration: " + error);
    return *parsed;
}

} // namespace

int main(int argc, char** argv)
try {
    const Args args = parse_args(argc, argv);
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect,
        mlir::stablehlo::StablehloDialect,
        ftlpu::compiler::kernel::KernelDialect,
        ftlpu::compiler::tensor::TensorDialect,
        ftlpu::compiler::stream::StreamDialect,
        ftlpu::compiler::schedule::ScheduleDialect,
        ftlpu::compiler::command::CommandDialect>();
    mlir::MLIRContext context(registry);
    context.disableMultithreading();
    context.loadAllAvailableDialects();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        args.input.string(), &context);
    if (!module) return 1;

    const auto target = load_target(args.target_config);
    if (!args.target_config.empty()
        || args.input_stage == InputStage::StableHlo
        || args.input_stage == InputStage::Stream)
        (*module)->setAttr("ftlpu.target", target.to_attribute(&context));
    if (args.weight_bank >= target.memory().banks_per_slice)
        throw std::runtime_error(
            "weight bank is outside the target memory");
    (*module)->setAttr("ftlpu.mxm_execution_policy",
        mlir::StringAttr::get(&context,
            ftlpu::compiler::target::mxm_execution_policy_name(
                args.mxm_execution_policy)));

    mlir::PassManager passes(&context);
    if (args.pass_timing) passes.enableTiming();
    if (args.input_stage == InputStage::StableHlo) {
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_stablehlo_to_kernel_pass());
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_kernel_to_tensor_pass(
                args.rmsnorm_strategy, args.weight_bank));
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_tensor_to_stream_pass());
    }
    if (args.input_stage == InputStage::StableHlo
        || args.input_stage == InputStage::Stream)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_stream_to_schedule_pass(
                args.ffn_schedule, args.attention_schedule,
                args.pass_timing));
    if (args.weight_bank >= 0
        && args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_assign_weight_bank_pass(
                args.weight_bank));
    if (args.input_stage != InputStage::VerifiedSchedule
        && args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_compress_schedule_pass());
    if (args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_verify_schedule_pass());
    if (args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_schedule_to_command_pass());
    if (mlir::failed(passes.run(*module))) return 1;

    std::error_code error;
    std::filesystem::create_directories(args.output.parent_path(), error);
    auto program =
        ftlpu::compiler::target::translate_command_module(*module);
    ftlpu::software::runtime::write_binary_program(program, args.output);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ftlpu-compile failed: " << error.what() << '\n';
    return 1;
}
