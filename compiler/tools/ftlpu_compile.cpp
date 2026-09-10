// BinaryProgram crosses this tool/library ABI boundary by value.
#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/compiler/Target/icu_compression.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/InitLLVM.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
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
    ftlpu::compiler::target::MxmExecutionPolicy mxm_execution_policy{
        ftlpu::compiler::target::MxmExecutionPolicy::Auto};
    std::int64_t weight_bank{-1};
    std::int64_t kv_cache_capacity{0};
    bool pass_timing{false};
    ftlpu::compiler::target::IcuCompressionMode icu_compression{
        ftlpu::compiler::target::IcuCompressionMode::Macro};
    std::optional<bool> mem_slice_program_override{};
    bool verify_icu_issues{false};
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
        else if (argument == "--kv-cache-capacity")
            args.kv_cache_capacity = std::stoll(next());
        else if (argument == "--pass-timing")
            args.pass_timing = true;
        else if (argument == "--verify-icu-issues")
            args.verify_icu_issues = true;
        else if (argument == "--icu-macro-schedule")
            args.icu_compression =
                ftlpu::compiler::target::IcuCompressionMode::Macro;
        else if (argument == "--icu-compression") {
            const std::string value = next();
            const auto parsed = ftlpu::compiler::target::
                parse_icu_compression_mode(value);
            if (!parsed)
                throw std::runtime_error(
                    "unknown ICU compression mode: " + value);
            args.icu_compression = *parsed;
        }
        else if (argument == "--mem-slice-program") {
            const std::string value = next();
            if (value == "on") args.mem_slice_program_override = true;
            else if (value == "off") args.mem_slice_program_override = false;
            else throw std::runtime_error(
                "expected on or off for --mem-slice-program");
        }
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
            "[--kv-cache-capacity tokens] "
            "[--mxm-execution auto|vector|legacy] "
            "[--icu-compression none|repeat|macro|macro-slice] "
            "[--mem-slice-program on|off] "
            "[--verify-icu-issues] "
            "[--ffn-schedule tail|fused] "
            "[--attention-schedule tail|fused]");
    if (args.mem_slice_program_override)
        args.icu_compression =
            ftlpu::compiler::target::set_mem_slice_program(
                args.icu_compression, *args.mem_slice_program_override);
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
    llvm::InitLLVM initLLVM(argc, argv);
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
    if (args.kv_cache_capacity < 0)
        throw std::runtime_error(
            "KV cache capacity must be non-negative");
    if (args.kv_cache_capacity > 0)
        (*module)->setAttr("ftlpu.kv_cache_capacity",
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(&context, 64),
                args.kv_cache_capacity));
    (*module)->setAttr("ftlpu.mxm_execution_policy",
        mlir::StringAttr::get(&context,
            ftlpu::compiler::target::mxm_execution_policy_name(
                args.mxm_execution_policy)));
    (*module)->setAttr("ftlpu.icu_compression",
        mlir::StringAttr::get(&context,
            ftlpu::compiler::target::icu_compression_mode_name(
                args.icu_compression)));
    (*module)->setAttr("ftlpu.mem_slice_program",
        mlir::BoolAttr::get(&context,
            ftlpu::compiler::target::
                icu_compression_uses_mem_slice_program(
                    args.icu_compression)));
    // Keep the old attribute during the command-IR compatibility window.
    (*module)->setAttr("ftlpu.icu_macro_schedule",
        mlir::BoolAttr::get(&context,
            ftlpu::compiler::target::icu_compression_uses_macro(
                args.icu_compression)));

    mlir::PassManager passes(&context);
    // Model-scale schedules contain hundreds of thousands of primitive ops.
    // Verify once after schedule compression and once after command lowering
    // instead of rescanning the uncompressed IR after every pass.
    passes.enableVerifier(false);
    if (args.pass_timing) passes.enableTiming();
    if (args.input_stage == InputStage::StableHlo) {
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_stablehlo_to_kernel_pass());
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_kernel_to_tensor_pass(
                args.weight_bank));
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
    if (args.icu_compression
            != ftlpu::compiler::target::IcuCompressionMode::None
        && args.input_stage != InputStage::VerifiedSchedule
        && args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_compress_schedule_pass());
    if (args.input_stage != InputStage::VerifiedSchedule
        && args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_verify_schedule_pass());
    if (args.input_stage != InputStage::Command)
        passes.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_schedule_to_command_pass());
    if (mlir::failed(passes.run(*module))) return 1;
    if (mlir::failed(mlir::verify(*module))) return 1;

    std::error_code error;
    std::filesystem::create_directories(args.output.parent_path(), error);
    auto program =
        ftlpu::compiler::target::translate_command_module(*module);
    if (args.verify_icu_issues) {
        (*module)->setAttr("ftlpu.icu_compression",
            mlir::StringAttr::get(&context, "none"));
        (*module)->setAttr("ftlpu.mem_slice_program",
            mlir::BoolAttr::get(&context, false));
        const auto baseline =
            ftlpu::compiler::target::translate_command_module(*module);
        const auto comparison =
            ftlpu::software::runtime::compare_logical_issues(
                program, baseline);
        if (!comparison.equivalent) {
            std::string detail = comparison.first_mismatch
                ? comparison.first_mismatch->reason : "unknown mismatch";
            if (comparison.first_mismatch)
                detail += " at cycle="
                    + std::to_string(
                        comparison.first_mismatch->cycle);
            throw std::runtime_error(
                "ICU compression changed the logical issue stream: "
                + detail);
        }
        std::cout << "verified ICU logical issue equivalence: functional_issues="
                  << ftlpu::software::runtime::inspect_logical_issues(
                         program).functional_issues
                  << " max_cycle=" << program.max_cycle << '\n';
    }
    ftlpu::software::runtime::write_binary_program(program, args.output);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ftlpu-compile failed: " << error.what() << '\n';
    return 1;
}
