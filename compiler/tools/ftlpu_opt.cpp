#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Target/icu_compression.hpp"
#include "ftlpu/compiler/Target/mxm_execution_strategy.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/VhloOps.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/APFloat.h"

#include <filesystem>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>

namespace {

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path target_config;
    std::filesystem::path diagnostic_dir;
    std::filesystem::path constant_bindings;
    std::filesystem::path backend_profile;
    std::filesystem::path portable_output;
    std::string portable_version{"1.13.0"};
    std::string pipeline{"ftlpu-stablehlo-to-kernel"};
    ftlpu::compiler::FfnScheduleStrategy ffn_schedule{
        ftlpu::compiler::FfnScheduleStrategy::Tail};
    ftlpu::compiler::AttentionScheduleStrategy attention_schedule{
        ftlpu::compiler::AttentionScheduleStrategy::Tail};
    ftlpu::compiler::target::MxmExecutionPolicy mxm_execution_policy{
        ftlpu::compiler::target::MxmExecutionPolicy::Auto};
    std::int64_t weight_bank{-1};
    std::int64_t kv_cache_capacity{0};
    bool pass_timing{false};
    bool kernel_partial{false};
    bool stablehlo_normalize{true};
    bool fold_static_mask{false};
    bool prune_unused_inputs{false};
    ftlpu::compiler::target::IcuCompressionMode icu_compression{
        ftlpu::compiler::target::IcuCompressionMode::Macro};
    std::optional<bool> mem_slice_program_override{};
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
        else if (arg == "--target-config") args.target_config = next();
        else if (arg == "--pipeline") args.pipeline = next();
        else if (arg == "--diagnostic-dir") args.diagnostic_dir = next();
        else if (arg == "--constant-bindings") args.constant_bindings = next();
        else if (arg == "--backend-profile") args.backend_profile = next();
        else if (arg == "--portable-output") args.portable_output = next();
        else if (arg == "--portable-version") args.portable_version = next();
        else if (arg == "--prune-unused-inputs") args.prune_unused_inputs = true;
        else if (arg == "--kernel-mode") {
            const auto value = next();
            if (value == "partial") args.kernel_partial = true;
            else if (value == "complete") args.kernel_partial = false;
            else throw std::runtime_error("expected complete or partial for --kernel-mode");
        }
        else if (arg == "--stablehlo-normalize") {
            const auto value = next();
            if (value == "on") args.stablehlo_normalize = true;
            else if (value == "off") args.stablehlo_normalize = false;
            else throw std::runtime_error("expected on or off for --stablehlo-normalize");
        }
        else if (arg == "--fold-static-mask") {
            const auto value = next();
            if (value == "on") args.fold_static_mask = true;
            else if (value == "off") args.fold_static_mask = false;
            else throw std::runtime_error("expected on or off for --fold-static-mask");
        }
        else if (arg == "--ffn-schedule") {
            const std::string strategy = next();
            if (strategy == "tail")
                args.ffn_schedule = ftlpu::compiler::FfnScheduleStrategy::Tail;
            else if (strategy == "fused")
                args.ffn_schedule = ftlpu::compiler::FfnScheduleStrategy::Fused;
            else
                throw std::runtime_error(
                    "unknown FFN schedule strategy: " + strategy);
        }
        else if (arg == "--attention-schedule") {
            const std::string strategy = next();
            if (strategy == "tail")
                args.attention_schedule =
                    ftlpu::compiler::AttentionScheduleStrategy::Tail;
            else if (strategy == "fused")
                args.attention_schedule =
                    ftlpu::compiler::AttentionScheduleStrategy::Fused;
            else
                throw std::runtime_error(
                    "unknown Attention schedule strategy: " + strategy);
        }
        else if (arg == "--mxm-execution") {
            const std::string policy = next();
            auto parsed =
                ftlpu::compiler::target::parse_mxm_execution_policy(
                    policy);
            if (mlir::failed(parsed))
                throw std::runtime_error(
                    "unknown MXM execution policy: " + policy);
            args.mxm_execution_policy = *parsed;
        }
        else if (arg == "--weight-bank")
            args.weight_bank = std::stoll(next());
        else if (arg == "--kv-cache-capacity")
            args.kv_cache_capacity = std::stoll(next());
        else if (arg == "--pass-timing")
            args.pass_timing = true;
        else if (arg == "--icu-macro-schedule")
            args.icu_compression =
                ftlpu::compiler::target::IcuCompressionMode::Macro;
        else if (arg == "--icu-compression") {
            const std::string value = next();
            const auto parsed = ftlpu::compiler::target::
                parse_icu_compression_mode(value);
            if (!parsed)
                throw std::runtime_error(
                    "unknown ICU compression mode: " + value);
            args.icu_compression = *parsed;
        }
        else if (arg == "--mem-slice-program") {
            const std::string value = next();
            if (value == "on") args.mem_slice_program_override = true;
            else if (value == "off") args.mem_slice_program_override = false;
            else throw std::runtime_error(
                "expected on or off for --mem-slice-program");
        }
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (args.input.empty() || args.output.empty()) {
        throw std::runtime_error("usage: ftlpu-opt --input in.mlir --output out.mlir "
                                 "[--pipeline ftlpu-stablehlo-to-kernel|ftlpu-stablehlo-to-tensor|"
                                 "ftlpu-normalize-stablehlo|"
                                 "ftlpu-stablehlo-to-stream|ftlpu-stablehlo-to-schedule|"
                                 "ftlpu-stream-to-uncompressed-schedule|"
                                 "ftlpu-stream-to-compressed-schedule|"
                                 "ftlpu-stream-to-schedule|"
                                 "ftlpu-stablehlo-to-commands|ftlpu-schedule-to-commands|"
                                 "ftlpu-verified-schedule-to-commands|"
                                 "ftlpu-compress-schedule|ftlpu-verify-schedule] "
                                 "[--ffn-schedule tail|fused] "
                                 "[--attention-schedule tail|fused] "
                                 "[--mxm-execution auto|vector|legacy] "
                                 "[--icu-compression none|repeat|macro|macro-slice] "
                                 "[--mem-slice-program on|off] "
                                 "[--weight-bank 0|1] "
                                 "[--kv-cache-capacity tokens] "
                                 "[--target-config target.json] "
                                 "[--kernel-mode complete|partial] "
                                 "[--stablehlo-normalize on|off] "
                                 "[--fold-static-mask on|off] "
                                 "[--diagnostic-dir fresh-directory] "
                                 "[--backend-profile profile.json] "
                                 "[--constant-bindings constants.json] "
                                 "[--prune-unused-inputs] "
                                 "[--portable-output normalized.bytecode] "
                                 "[--portable-version 1.13.0]");
    }
    if (args.kernel_partial && args.pipeline != "ftlpu-stablehlo-to-kernel")
        throw std::runtime_error("--kernel-mode partial is restricted to the Kernel debug pipeline");
    if (args.prune_unused_inputs && args.diagnostic_dir.empty())
        throw std::runtime_error("--prune-unused-inputs requires --diagnostic-dir to record the ABI map");
    if (!args.portable_output.empty() && args.pipeline != "ftlpu-normalize-stablehlo")
        throw std::runtime_error("--portable-output is restricted to the StableHLO normalization pipeline");
    if (args.mem_slice_program_override)
        args.icu_compression =
            ftlpu::compiler::target::set_mem_slice_program(
                args.icu_compression, *args.mem_slice_program_override);
    return args;
}

llvm::json::Value read_json(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read JSON: " + path.string());
    std::ostringstream buffer;
    buffer << file.rdbuf();
    auto parsed = llvm::json::parse(buffer.str());
    if (!parsed) throw std::runtime_error("invalid JSON in " + path.string()
        + ": " + llvm::toString(parsed.takeError()));
    return std::move(*parsed);
}

void write_json(const std::filesystem::path& path, const llvm::json::Value& value)
{
    std::error_code error;
    auto file = llvm::raw_fd_ostream(path.string(), error);
    if (error) throw std::runtime_error("cannot write report: " + path.string()
        + ": " + error.message());
    file << llvm::formatv("{0:2}", value) << '\n';
}

struct PipelineDiagnostics {
    const Args& args;
    llvm::json::Array stages;
    llvm::json::Array diagnostics;
    llvm::json::Object kernel_coverage;
    std::string status{"failed"};
    std::string failed_stage{"configuration_or_parse"};
    std::map<mlir::Pass*, std::chrono::steady_clock::time_point> starts;
    int snapshot = 0;
    explicit PipelineDiagnostics(const Args& args) : args(args)
    {
        if (!args.diagnostic_dir.empty()) {
            std::filesystem::create_directories(args.diagnostic_dir);
            if (std::filesystem::exists(args.diagnostic_dir / "report.json"))
                throw std::runtime_error("diagnostic directory already contains report.json; "
                    "choose a fresh directory to preserve prior evidence");
        }
    }
    void save_ir(mlir::Operation* op, llvm::StringRef name)
    {
        if (args.diagnostic_dir.empty()) return;
        while (op->getParentOp()) op = op->getParentOp();
        const auto path = args.diagnostic_dir / name.str();
        std::error_code error;
        llvm::raw_fd_ostream file(path.string(), error);
        if (error) throw std::runtime_error("cannot write diagnostic IR: " + path.string());
        op->print(file, mlir::OpPrintingFlags().enableDebugInfo());
        file << '\n';
    }
    void flush()
    {
        if (args.diagnostic_dir.empty()) return;
        write_json(args.diagnostic_dir / "report.json", llvm::json::Object{
            {"schema_version", 1}, {"status", status},
            {"compiler_llvm_version", LLVM_VERSION_STRING},
            {"failed_stage", status == "failed" ? failed_stage : ""},
            {"input", args.input.string()}, {"output", args.output.string()},
            {"pipeline", args.pipeline},
            {"kernel_mode", args.kernel_partial ? "partial" : "complete"},
            {"stablehlo_normalize", args.stablehlo_normalize},
            {"fold_static_mask", args.fold_static_mask},
            {"target_config", args.target_config.string()},
            {"backend_profile", args.backend_profile.string()},
            {"constant_bindings", args.constant_bindings.string()},
            {"portable_output", args.portable_output.string()},
            {"portable_version", args.portable_version},
            {"prune_unused_inputs", args.prune_unused_inputs},
            {"stages", llvm::json::Array(stages)},
            {"diagnostics", llvm::json::Array(diagnostics)},
            {"kernel_coverage", llvm::json::Object(kernel_coverage)}});
    }
    ~PipelineDiagnostics()
    {
        try { flush(); }
        catch (const std::exception& error) {
            std::cerr << "cannot save diagnostic report: " << error.what() << '\n';
        }
    }
};

class DiagnosticInstrumentation final : public mlir::PassInstrumentation {
public:
    explicit DiagnosticInstrumentation(PipelineDiagnostics& diagnostics)
        : diagnostics_(diagnostics) {}
    void runBeforePass(mlir::Pass* pass, mlir::Operation*) override
    {
        diagnostics_.starts[pass] = std::chrono::steady_clock::now();
        diagnostics_.failed_stage = pass->getArgument().str();
    }
    void finish(mlir::Pass* pass, mlir::Operation* op, bool succeeded)
    {
        auto stage = pass->getArgument().str();
        if (stage.empty()) stage = pass->getName().str();
        if (stage.find("Adaptor") != std::string::npos) return;
        if (stage == "ftlpu-verify-kernel-coverage") {
            op->walk([&](mlir::func::FuncOp function) {
                auto complete = function->getAttrOfType<mlir::BoolAttr>("ftlpu.kernel_complete");
                auto groups = function->getAttrOfType<mlir::ArrayAttr>("ftlpu.kernel_unconverted");
                llvm::json::Array rejected;
                if (groups) for (auto attribute : groups) {
                    auto group = llvm::cast<mlir::DictionaryAttr>(attribute);
                    llvm::json::Array locations;
                    for (auto location : llvm::cast<mlir::ArrayAttr>(group.get("locations")))
                        locations.push_back(llvm::cast<mlir::StringAttr>(location).getValue().str());
                    rejected.push_back(llvm::json::Object{
                        {"operation", llvm::cast<mlir::StringAttr>(group.get("operation")).getValue().str()},
                        {"count", llvm::cast<mlir::IntegerAttr>(group.get("count")).getInt()},
                        {"reason", llvm::cast<mlir::StringAttr>(group.get("reason")).getValue().str()},
                        {"locations", std::move(locations)}});
                }
                // The report outlives MLIRContext. JSON StringRef keys borrow
                // storage, so own the symbol text before the IR is destroyed.
                diagnostics_.kernel_coverage[function.getName().str()] = llvm::json::Object{
                    {"complete", complete && complete.getValue()},
                    {"unconverted", std::move(rejected)}};
            });
        }
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - diagnostics_.starts[pass]).count();
        const auto path = std::to_string(++diagnostics_.snapshot) + "-"
            + stage + (succeeded ? ".mlir" : ".failed.mlir");
        diagnostics_.save_ir(op, path);
        diagnostics_.stages.push_back(llvm::json::Object{
            {"name", stage}, {"status", succeeded ? "passed" : "failed"},
            {"seconds", seconds}, {"ir", path}});
        std::cerr << "ftlpu-opt stage " << stage << ": "
            << (succeeded ? "passed" : "failed") << " (" << seconds << " s)\n";
        if (!succeeded) diagnostics_.failed_stage = stage;
        diagnostics_.flush();
    }
    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override
    { finish(pass, op, true); }
    void runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) override
    { finish(pass, op, false); }
private:
    PipelineDiagnostics& diagnostics_;
};

void bind_constants_and_profile(mlir::ModuleOp module, const Args& args)
{
    mlir::Builder builder(module.getContext());
    if (!args.backend_profile.empty()) {
        auto json = read_json(args.backend_profile);
        auto* profile = json.getAsObject();
        if (!profile || profile->getInteger("schema_version") != 1)
            throw std::runtime_error("backend profile requires schema_version 1");
        const auto name = profile->getString("profile");
        const auto theta = profile->getNumber("rope_theta");
        if (!name || !theta || !std::isfinite(*theta) || *theta <= 0
            || !std::isfinite(static_cast<float>(*theta)) || static_cast<float>(*theta) <= 0)
            throw std::runtime_error("backend profile requires profile and positive finite rope_theta");
        const auto name_attr = builder.getStringAttr(*name);
        const auto theta_attr = builder.getF32FloatAttr(static_cast<float>(*theta));
        module->setAttr("ftlpu.backend_numeric_profile", name_attr);
        module->setAttr("ftlpu.rope_theta", theta_attr);
        module.walk([&](mlir::func::FuncOp function) {
            function->setAttr("ftlpu.backend_numeric_profile", name_attr);
            function->setAttr("ftlpu.rope_theta", theta_attr);
        });
    }
    llvm::json::Value bindings(nullptr);
    llvm::json::Object* functions = nullptr;
    if (!args.constant_bindings.empty()) {
        bindings = read_json(args.constant_bindings);
        auto* object = bindings.getAsObject();
        if (!object || object->getInteger("schema_version") != 1
            || !(functions = object->getObject("functions")))
            throw std::runtime_error("constant bindings require schema_version 1 and functions object");
        for (const auto& function : *functions)
            if (!module.lookupSymbol<mlir::func::FuncOp>(function.first))
                throw std::runtime_error("constant bindings name an absent function: " + function.first.str());
    }
    llvm::json::Object maps;
    module.walk([&](mlir::func::FuncOp function) {
        if (function.isExternal()) return;
        llvm::BitVector specialized(function.getNumArguments());
        if (functions) {
            if (auto* entries = functions->getArray(function.getName())) {
                mlir::OpBuilder insertion(&function.getBody().front(),
                    function.getBody().front().begin());
                for (const auto& entry : *entries) {
                    auto* object = entry.getAsObject();
                    if (!object) throw std::runtime_error("constant binding must be an object");
                    const auto index = object->getInteger("argument_index");
                    auto* shape = object->getArray("shape");
                    auto* values = object->getArray("values");
                    const auto dtype = object->getString("dtype");
                    const bool boolean = dtype && (*dtype == "bool" || *dtype == "i1");
                    const bool bfloat = dtype && (*dtype == "bfloat16" || *dtype == "bf16");
                    const bool float32 = dtype && (*dtype == "float32" || *dtype == "f32");
                    if (!index || *index < 0 || *index >= function.getNumArguments()
                        || !shape || !values || !dtype
                        || (!boolean && !bfloat && !float32)
                        || specialized[*index])
                        throw std::runtime_error("invalid/duplicate constant binding argument index, shape or dtype");
                    auto argument = function.getArgument(*index);
                    auto type = llvm::dyn_cast<mlir::RankedTensorType>(argument.getType());
                    if (!type || !type.hasStaticShape()
                        || (float32 && !type.getElementType().isF32())
                        || (bfloat && !type.getElementType().isBF16())
                        || (boolean && !type.getElementType().isInteger(1))
                        || type.getRank() != shape->size() || type.getNumElements() > 4096
                        || type.getNumElements() != values->size())
                        throw std::runtime_error("constant binding disagrees with the actual function signature");
                    for (auto [axis, extent] : llvm::enumerate(*shape))
                        if (extent.getAsInteger() != type.getDimSize(axis))
                            throw std::runtime_error("constant binding shape disagrees with function signature");
                    mlir::DenseElementsAttr data;
                    if (boolean) {
                        llvm::SmallVector<llvm::APInt> bits;
                        for (const auto& value : *values) {
                            auto bit = value.getAsBoolean();
                            if (!bit) throw std::runtime_error("boolean constant binding requires JSON boolean values");
                            bits.emplace_back(1, *bit);
                        }
                        data = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<llvm::APInt>(bits));
                    } else {
                        llvm::SmallVector<llvm::APFloat> floats;
                        for (const auto& value : *values) {
                            auto number = value.getAsNumber();
                            if (!number || !std::isfinite(*number))
                                throw std::runtime_error("constant binding requires finite floating values");
                            llvm::APFloat converted(*number);
                            bool loses_info;
                            converted.convert(llvm::cast<mlir::FloatType>(type.getElementType()).getFloatSemantics(),
                                llvm::APFloat::rmNearestTiesToEven, &loses_info);
                            if (!converted.isFinite())
                                throw std::runtime_error("constant binding value is not representable in the declared dtype");
                            floats.push_back(converted);
                        }
                        data = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<llvm::APFloat>(floats));
                    }
                    auto constant = insertion.create<mlir::stablehlo::ConstantOp>(
                        function.getLoc(), data);
                    argument.replaceAllUsesWith(constant.getResult());
                    specialized.set(*index);
                }
            } else if (functions->get(function.getName())) {
                throw std::runtime_error("constant bindings function entries must be an array");
            }
        }
        llvm::BitVector remove(function.getNumArguments());
        llvm::json::Array entries;
        int64_t next = 0;
        for (auto argument : function.getArguments()) {
            const auto index = argument.getArgNumber();
            const bool prune = args.prune_unused_inputs && argument.use_empty();
            if (prune) remove.set(index);
            std::string type;
            llvm::raw_string_ostream stream(type);
            argument.getType().print(stream);
            entries.push_back(llvm::json::Object{
                {"original_index", static_cast<int64_t>(index)},
                {"final_index", prune ? llvm::json::Value(nullptr) : llvm::json::Value(next++)},
                {"specialized_constant", static_cast<bool>(specialized[index])}, {"original_type", type}});
        }
        if (remove.any()) {
            bool called = false;
            module.walk([&](mlir::func::CallOp call) {
                called |= call.getCallee() == function.getName();
            });
            if (called)
                throw std::runtime_error("cannot prune a called function without updating its call ABI");
            if (mlir::failed(function.eraseArguments(remove)))
                throw std::runtime_error("failed to prune unused function arguments");
        }
        maps[function.getName()] = std::move(entries);
    });
    if (!args.diagnostic_dir.empty())
        write_json(args.diagnostic_dir / "abi_map.json", llvm::json::Object{
            {"schema_version", 1}, {"functions", std::move(maps)}});
}

} // namespace

int main(int argc, char** argv)
try {
    llvm::InitLLVM initLLVM(argc, argv);
    const auto args = parse_args(argc, argv);
    PipelineDiagnostics diagnostics(args);
    try {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
        ftlpu::compiler::kernel::KernelDialect, ftlpu::compiler::tensor::TensorDialect>();
    registry.insert<ftlpu::compiler::stream::StreamDialect,
        ftlpu::compiler::schedule::ScheduleDialect,
        ftlpu::compiler::command::CommandDialect>();
    mlir::MLIRContext context(registry);
    mlir::ScopedDiagnosticHandler diagnostic_handler(&context,
        [&](mlir::Diagnostic& diagnostic) {
            std::string message;
            llvm::raw_string_ostream stream(message);
            diagnostic.print(stream);
            std::string location;
            llvm::raw_string_ostream location_stream(location);
            diagnostic.getLocation().print(location_stream);
            diagnostics.diagnostics.push_back(llvm::json::Object{
                {"location", location}, {"message", message}});
            // The default handler does not print warnings in this CLI setup.
            // Partial conversion must be visible even without reading JSON.
            if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Warning)
                llvm::errs() << location << ": warning: " << message << '\n';
            return mlir::failure(); // Preserve MLIR's original stderr diagnostic.
        });
    context.disableMultithreading();
    context.getOrLoadDialect<ftlpu::compiler::kernel::KernelDialect>();
    context.getOrLoadDialect<ftlpu::compiler::tensor::TensorDialect>();
    context.getOrLoadDialect<ftlpu::compiler::stream::StreamDialect>();
    context.getOrLoadDialect<ftlpu::compiler::schedule::ScheduleDialect>();
    context.getOrLoadDialect<ftlpu::compiler::command::CommandDialect>();

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(args.input.string(), &context);
    if (!module) return 1;
    diagnostics.save_ir(*module, "0-input.mlir");
    bind_constants_and_profile(*module, args);
    diagnostics.save_ir(*module, "0-bound-input.mlir");
    ftlpu::compiler::target::LPUTargetModel target;
    if (!args.target_config.empty()) {
        std::ifstream file(args.target_config, std::ios::binary);
        if (!file)
            throw std::runtime_error(
                "cannot read target configuration: "
                + args.target_config.string());
        std::ostringstream buffer;
        buffer << file.rdbuf();
        std::string target_error;
        auto parsed_target =
            ftlpu::compiler::target::LPUTargetModel::from_json(
                buffer.str(), target_error);
        if (mlir::failed(parsed_target))
            throw std::runtime_error(
                "invalid target configuration: " + target_error);
        target = *parsed_target;
    }
    if (args.weight_bank >= target.memory().banks_per_slice)
        throw std::runtime_error(
            "weight bank is outside the target memory");
    if (args.kv_cache_capacity < 0)
        throw std::runtime_error(
            "KV cache capacity must be non-negative");
    (*module)->setAttr("ftlpu.target", target.to_attribute(&context));
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

    mlir::PassManager pass_manager(&context);
    pass_manager.addInstrumentation(std::make_unique<DiagnosticInstrumentation>(diagnostics));
    const bool deferVerificationUntilCompressed =
        args.pipeline == "ftlpu-stream-to-compressed-schedule";
    if (deferVerificationUntilCompressed)
        pass_manager.enableVerifier(false);
    if (args.pass_timing) pass_manager.enableTiming();
    if (args.pipeline != "ftlpu-stablehlo-to-kernel"
        && args.pipeline != "ftlpu-normalize-stablehlo"
        && args.pipeline != "ftlpu-stablehlo-to-tensor"
        && args.pipeline != "ftlpu-stablehlo-to-stream"
        && args.pipeline != "ftlpu-stablehlo-to-schedule"
        && args.pipeline != "ftlpu-stream-to-uncompressed-schedule"
        && args.pipeline != "ftlpu-stream-to-compressed-schedule"
        && args.pipeline != "ftlpu-stream-to-schedule"
        && args.pipeline != "ftlpu-stablehlo-to-commands"
        && args.pipeline != "ftlpu-schedule-to-commands"
        && args.pipeline != "ftlpu-verified-schedule-to-commands"
        && args.pipeline != "ftlpu-compress-schedule"
        && args.pipeline != "ftlpu-verify-schedule") {
        throw std::runtime_error("unknown MLIR pipeline: " + args.pipeline);
    }
    const bool frontend_pipeline = args.pipeline.starts_with("ftlpu-stablehlo-to-")
        || args.pipeline == "ftlpu-normalize-stablehlo";
    if (frontend_pipeline && args.stablehlo_normalize) {
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_normalize_stablehlo_pass(args.fold_static_mask));
        // Upstream CSE is sufficient here; unrestricted arithmetic folding and
        // reassociation would change numerical requirements or matcher forms.
        pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    }
    if (args.pipeline != "ftlpu-normalize-stablehlo"
        && args.pipeline != "ftlpu-verify-schedule"
        && args.pipeline != "ftlpu-stream-to-uncompressed-schedule"
        && args.pipeline != "ftlpu-stream-to-compressed-schedule"
        && args.pipeline != "ftlpu-stream-to-schedule"
        && args.pipeline != "ftlpu-schedule-to-commands"
        && args.pipeline != "ftlpu-verified-schedule-to-commands"
        && args.pipeline != "ftlpu-compress-schedule")
    {
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_stablehlo_to_kernel_pass());
        pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_verify_kernel_coverage_pass(args.kernel_partial));
    }
    if (args.pipeline == "ftlpu-stablehlo-to-tensor" || args.pipeline == "ftlpu-stablehlo-to-stream"
        || args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_kernel_to_tensor_pass(
                args.weight_bank));
    if (args.pipeline == "ftlpu-stablehlo-to-stream" || args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_tensor_to_stream_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stream-to-uncompressed-schedule"
        || args.pipeline == "ftlpu-stream-to-compressed-schedule"
        || args.pipeline == "ftlpu-stream-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_stream_to_schedule_pass(
                args.ffn_schedule, args.attention_schedule,
                args.pass_timing));
    if (args.weight_bank >= 0
        && (args.pipeline == "ftlpu-stablehlo-to-schedule"
            || args.pipeline == "ftlpu-stablehlo-to-commands"
            || args.pipeline == "ftlpu-stream-to-uncompressed-schedule"
            || args.pipeline == "ftlpu-stream-to-compressed-schedule"
            || args.pipeline == "ftlpu-stream-to-schedule"
            || args.pipeline == "ftlpu-schedule-to-commands"
            || args.pipeline == "ftlpu-verified-schedule-to-commands"))
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_assign_weight_bank_pass(
                args.weight_bank));
    if (args.icu_compression
            != ftlpu::compiler::target::IcuCompressionMode::None
        && (args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands"
        || args.pipeline == "ftlpu-stream-to-compressed-schedule"
        || args.pipeline == "ftlpu-schedule-to-commands"
        || args.pipeline == "ftlpu-compress-schedule"))
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_compress_schedule_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-schedule"
        || args.pipeline == "ftlpu-stablehlo-to-commands"
        || args.pipeline == "ftlpu-schedule-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_verify_schedule_pass());
    if (args.pipeline == "ftlpu-verify-schedule")
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_verify_schedule_pass());
    if (args.pipeline == "ftlpu-stablehlo-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_schedule_to_command_pass());
    if (args.pipeline == "ftlpu-schedule-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(ftlpu::compiler::create_lower_schedule_to_command_pass());
    if (args.pipeline == "ftlpu-verified-schedule-to-commands")
        pass_manager.addNestedPass<mlir::func::FuncOp>(
            ftlpu::compiler::create_lower_schedule_to_command_pass());
    if (mlir::failed(pass_manager.run(*module))) return 1;
    if (deferVerificationUntilCompressed
        && mlir::failed(mlir::verify(*module)))
        return 1;

    std::error_code error;
    std::filesystem::create_directories(args.output.parent_path(), error);
    std::string output_error;
    auto output = mlir::openOutputFile(args.output.string(), &output_error);
    if (!output) throw std::runtime_error(output_error);
    module->print(output->os());
    output->os() << '\n';
    output->keep();
    if (!args.portable_output.empty()) {
        diagnostics.failed_stage = "stablehlo_portable_serialization";
        mlir::OwningOpRef<mlir::ModuleOp> portable = module->clone();
        // Backend diagnostics are separate metadata, not StableHLO semantics.
        portable->walk([&](mlir::Operation* operation) {
            llvm::SmallVector<mlir::StringAttr> remove;
            for (const auto& attribute : operation->getAttrs())
                if (attribute.getName().getValue().starts_with("ftlpu."))
                    remove.push_back(attribute.getName());
            for (const auto& name : remove) operation->removeAttr(name);
        });
        context.getOrLoadDialect<mlir::vhlo::VhloDialect>();
        if (!args.portable_output.parent_path().empty())
            std::filesystem::create_directories(args.portable_output.parent_path());
        auto artifact = mlir::openOutputFile(args.portable_output.string(), &output_error);
        if (!artifact) throw std::runtime_error(output_error);
        if (mlir::failed(mlir::stablehlo::serializePortableArtifact(
                *portable, args.portable_version, artifact->os())))
            return 1;
        artifact->keep();
    }
    diagnostics.status = args.kernel_partial ? "partial_debug_output" : "passed";
    diagnostics.flush();
    return 0;
    } catch (const std::exception& error) {
        diagnostics.diagnostics.push_back(llvm::json::Object{
            {"location", "configuration_or_binding"}, {"message", error.what()}});
        throw;
    }
} catch (const std::exception& ex) {
    std::cerr << "ftlpu-opt failed: " << ex.what() << '\n';
    return 1;
}
