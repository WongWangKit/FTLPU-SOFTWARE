// Keep the tool rebuilt with the BinaryProgram timeline and binding ABI.
#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/compiler/Target/icu_compression.hpp"
#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/issue_inspector.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
try {
    std::filesystem::path input;
    std::filesystem::path output;
    std::optional<ftlpu::compiler::target::IcuCompressionMode>
        icuCompression;
    std::optional<bool> memSliceProgram;
    bool verifyIcuIssues = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--verify-icu-issues") {
            verifyIcuIssues = true;
            continue;
        }
        if (++i >= argc) throw std::runtime_error("missing value for " + argument);
        if (argument == "--input") input = argv[i];
        else if (argument == "--output") output = argv[i];
        else if (argument == "--icu-compression") {
            const std::string value = argv[i];
            icuCompression = ftlpu::compiler::target::
                parse_icu_compression_mode(value);
            if (!icuCompression)
                throw std::runtime_error(
                    "unknown ICU compression mode: " + value);
        }
        else if (argument == "--mem-slice-program") {
            const std::string value = argv[i];
            if (value == "on") memSliceProgram = true;
            else if (value == "off") memSliceProgram = false;
            else throw std::runtime_error(
                "expected on or off for --mem-slice-program");
        }
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (input.empty() || output.empty())
        throw std::runtime_error(
            "usage: ftlpu-translate --input command.mlir --output program.ftlpu "
            "[--icu-compression none|repeat|macro|macro-slice] "
            "[--mem-slice-program on|off] [--verify-icu-issues]");

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, ftlpu::compiler::command::CommandDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(input.string(), &context);
    if (!module) return 1;
    using ftlpu::compiler::target::IcuCompressionMode;
    auto effectiveCompression = IcuCompressionMode::Macro;
    if (icuCompression) {
        effectiveCompression = *icuCompression;
    } else if (const auto attr = (*module)->getAttrOfType<mlir::StringAttr>(
                   "ftlpu.icu_compression")) {
        const auto parsed = ftlpu::compiler::target::
            parse_icu_compression_mode(attr.getValue().str());
        if (!parsed)
            throw std::runtime_error(
                "Command IR module has an invalid ftlpu.icu_compression");
        effectiveCompression = *parsed;
    } else if (const auto attr = (*module)->getAttrOfType<mlir::BoolAttr>(
                   "ftlpu.icu_macro_schedule")) {
        effectiveCompression = attr.getValue()
            ? IcuCompressionMode::Macro : IcuCompressionMode::Repeat;
    }
    if (!icuCompression && effectiveCompression == IcuCompressionMode::Macro)
        if (const auto attr = (*module)->getAttrOfType<mlir::BoolAttr>(
                "ftlpu.mem_slice_program"); attr && attr.getValue())
            effectiveCompression = IcuCompressionMode::MacroSlice;
    if (memSliceProgram)
        effectiveCompression = ftlpu::compiler::target::
            set_mem_slice_program(effectiveCompression, *memSliceProgram);
    (*module)->setAttr("ftlpu.icu_compression",
        mlir::StringAttr::get(&context,
            ftlpu::compiler::target::icu_compression_mode_name(
                effectiveCompression)));
    (*module)->setAttr("ftlpu.mem_slice_program",
        mlir::BoolAttr::get(&context,
            ftlpu::compiler::target::
                icu_compression_uses_mem_slice_program(
                    effectiveCompression)));
    (*module)->setAttr("ftlpu.icu_macro_schedule",
        mlir::BoolAttr::get(&context,
            ftlpu::compiler::target::icu_compression_uses_macro(
                effectiveCompression)));

    std::error_code error;
    std::filesystem::create_directories(output.parent_path(), error);
    auto program = ftlpu::compiler::target::translate_command_module(*module);
    if (verifyIcuIssues) {
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
    ftlpu::software::runtime::write_binary_program(program, output);
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "ftlpu-translate failed: " << ex.what() << '\n';
    return 1;
}
