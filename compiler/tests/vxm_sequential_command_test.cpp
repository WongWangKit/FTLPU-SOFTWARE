#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Command/Transforms/macro_command_materializer.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Target/vxm_chain_lowering.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

ftlpu::SxmInstruction::StreamList stream_range(
    std::size_t base, std::size_t count)
{
    auto streams = ftlpu::SxmInstruction::StreamList{};
    for (std::size_t index = 0; index < count; ++index)
        streams.push_back(ftlpu::SxmStreamId {base + index});
    return streams;
}

} // namespace

int main() try
{
    using namespace ftlpu;
    using namespace ftlpu::compiler;
    using namespace ftlpu::software::runtime;

    require(ftlpu::compiler::target::lowerVxmChainOperand("stream_bf16", 32, 0, 0, false, "local").kind
            == VxmLaneOperandKind::StreamBFloat16,
        "chain head fixed lhs input did not lower");
    for (const auto rhs : {false, true}) {
        bool rejected = false;
        try {
            ftlpu::compiler::target::lowerVxmChainOperand("stream_bf16", rhs ? 32 : 34,
                0, 0, rhs, "local");
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "chain lowering silently discarded an incompatible stream index");
    }
    bool rejectedMirrorQueue = false;
    try {
        ftlpu::compiler::target::lowerVxmChainOperand("immediate", 0, 1, 15, false, "local");
    } catch (const std::runtime_error&) { rejectedMirrorQueue = true; }
    require(rejectedMirrorQueue, "chain lowering accepted an independent mirror queue");

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, command::CommandDialect, schedule::ScheduleDialect>();
    mlir::MLIRContext context(registry); context.loadAllAvailableDialects();
    const auto location=mlir::UnknownLoc::get(&context);
    const ftlpu::compiler::target::LPUTargetModel targetModel;
    auto module=mlir::ModuleOp::create(location);
    module->setAttr("ftlpu.target",targetModel.to_attribute(&context));
    module->setAttr("ftlpu.command_lowering",mlir::StringAttr::get(&context,"direct"));
    auto function=mlir::func::FuncOp::create(location,"vxm_sequential",mlir::FunctionType::get(&context,{},{}));
    auto* entry=function.addEntryBlock();module.push_back(function);
    mlir::OpBuilder builder(entry,entry->begin());
    const auto emit=[&](int64_t cycle,int64_t count) {
        mlir::OperationState state(location,command::VxmOp::getOperationName());
        for(const auto& [key,value]:std::vector<std::pair<const char*,int64_t>>{
            {"cycle",cycle},{"queue",0},{"chain_depth",2},{"lhs_index",0},{"rhs_index",0},
            {"output_stream",-1},{"repeat_count",count},{"repeat_interval",1},{"scale_binding",3}})
            state.addAttribute(key,builder.getI64IntegerAttr(value));
        for(const auto& [key,value]:std::vector<std::pair<const char*,const char*>>{
            {"opcode","multiply"},{"lhs_kind","stream_bf16"},{"rhs_kind","immediate"},
            {"cast_target","bf16"},{"input_hemisphere","east"},{"output_hemisphere","west"}})
            state.addAttribute(key,builder.getStringAttr(value));
        state.addAttribute("lhs_immediate",builder.getF32FloatAttr(0));
        state.addAttribute("rhs_immediate",builder.getF32FloatAttr(0.5));
        builder.create(state);
    };
    emit(2,32);emit(40,32);builder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(module)),"sequential VXM command failed verification");
    const auto binary=ftlpu::compiler::target::translate_command_module(module);
    require(binary.scale_relocations.size()==2 && binary.scale_relocations[0].command_index==1
        && binary.scale_relocations[1].command_index==3,"sequential scale relocation lost");
    require(binary.queues.size()==1 && binary.queues[0].commands.size()==4,"expected two configs and two waits");
    const auto capacity=analyze_physical_imem(binary);
    require(capacity.used_slots==4 && capacity.queues[0].peak_macro_contexts==0,"sequential VXM capacity incorrect");
    InstructionControlUnit icu;load_queue_programs_into_icu(binary.queues,icu);
    std::vector<std::size_t> issueCycles;
    for(std::size_t cycle=0;cycle<75;++cycle)if(auto issued=icu.vxm_iq(0).tick()) {
        require(isa::decode_vxm_instruction(0,*issued).instruction.repeat_count==32,"FU repeat was expanded");
        issueCycles.push_back(cycle);
    }
    require(issueCycles==std::vector<std::size_t>{2,40},"sequential issue timing changed");
    std::string error;

    auto sxmMap = SxmInstruction::PermuteMap {};
    for (std::size_t laneIndex = 0; laneIndex < sxmMap.size(); ++laneIndex)
        sxmMap[laneIndex] =
            (laneIndex + hw::kLanesPerTile) % sxmMap.size();
    const auto sxmTile = SxmInstruction::Permute(
        stream_range(0, 16), stream_range(16, 16), sxmMap);
    const auto sxmRun = SxmLaunchDomain::Launch(
        7, {3, 2}, {2, 13}, sxmTile);
    auto sxmModule = mlir::ModuleOp::create(location);
    sxmModule->setAttr("ftlpu.target", targetModel.to_attribute(&context));
    sxmModule->setAttr("ftlpu.command_lowering",
        mlir::StringAttr::get(&context, "direct"));
    auto sxmFunction = mlir::func::FuncOp::create(location, "sxm_macro",
        mlir::FunctionType::get(&context, {}, {}));
    auto* sxmEntry = sxmFunction.addEntryBlock();
    sxmModule.push_back(sxmFunction);
    mlir::OpBuilder sxmBuilder(sxmEntry, sxmEntry->begin());
    require(mlir::succeeded(command::materializeSxmMacroCommand(
                sxmBuilder, location, false, 0, sxmRun, &error)),
        "direct SXM Macro materialization failed");
    sxmBuilder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(sxmModule)),
        "command.sxm_macro did not verify");
    const auto sxmBinary =
        ftlpu::compiler::target::translate_command_module(sxmModule);
    require(sxmBinary.queues.size()==1 && sxmBinary.queues[0].kind==QueueKind::SxmPermute
        && sxmBinary.queues[0].commands.size()==1
        && is_macro_schedule_command(sxmBinary.queues[0].commands[0]),"SXM did not lower directly to Macro");
    const auto sxmCapacity=analyze_physical_imem(sxmBinary);
    require(sxmCapacity.queues[0].macro_entries==1 && sxmCapacity.queues[0].expanded_work==6
        && sxmCapacity.queues[0].macro_context_bits==384,"SXM Macro context accounting changed");
    InstructionControlUnit sxmIcu;
    load_queue_programs_into_icu(sxmBinary.queues, sxmIcu);
    issueCycles.clear();
    for (std::size_t cycle = 0; cycle <= 24; ++cycle) {
        if (const auto issued =
                sxmIcu.sxm_permute_iq(Hemisphere::East).tick()) {
            require(issued->opcode == SxmOpcode::Permute
                    && issued->permute_map == sxmMap,
                "runtime changed the SXM tile-local instruction");
            issueCycles.push_back(cycle);
        }
    }
    require(issueCycles
            == std::vector<std::size_t> {7, 9, 11, 20, 22, 24},
        "SXM Macro ICU issued at incorrect absolute cycles");

    const auto transposeTile = SxmInstruction::Transpose(
        stream_range(0, 16), stream_range(16, 16));
    const auto transposeRun = SxmLaunchDomain::Launch(
        3, {1, 1}, {1, 1}, transposeTile);
    auto transposeModule = mlir::ModuleOp::create(location);
    transposeModule->setAttr(
        "ftlpu.target", targetModel.to_attribute(&context));
    transposeModule->setAttr("ftlpu.command_lowering",
        mlir::StringAttr::get(&context, "direct"));
    auto transposeFunction = mlir::func::FuncOp::create(location,
        "sxm_transpose_run_2d", mlir::FunctionType::get(&context, {}, {}));
    auto* transposeEntry = transposeFunction.addEntryBlock();
    transposeModule.push_back(transposeFunction);
    mlir::OpBuilder transposeBuilder(
        transposeEntry, transposeEntry->begin());
    require(mlir::succeeded(command::materializeSxmMacroCommand(
                transposeBuilder, location, true, 0, transposeRun, &error)),
        "direct SXM transpose RUN_2D materialization failed");
    transposeBuilder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(transposeModule)),
        "command.sxm_macro transpose did not verify");
    const auto transposeBinary =
        ftlpu::compiler::target::translate_command_module(transposeModule);
    require(transposeBinary.queues.size() == 1
            && transposeBinary.queues[0].kind == QueueKind::SxmTranspose
            && transposeBinary.queues[0].commands.size()
                == 1,
        "SXM transpose did not become one Macro descriptor");
    InstructionControlUnit transposeIcu;
    load_queue_programs_into_icu(transposeBinary.queues, transposeIcu);
    issueCycles.clear();
    for (std::size_t cycle = 0; cycle <= 3; ++cycle)
        if (transposeIcu.sxm_transpose_iq(Hemisphere::East).tick())
            issueCycles.push_back(cycle);
    require(issueCycles == std::vector<std::size_t> {3},
        "SXM transpose compact packet changed its issue cycle");

    const auto makeSxmFrontendModule =
        [&](const std::vector<std::size_t>& startCycles,
            const char* functionName) {
            auto frontendModule = mlir::ModuleOp::create(location);
            frontendModule->setAttr(
                "ftlpu.target", targetModel.to_attribute(&context));
            frontendModule->setAttr("ftlpu.command_lowering",
                mlir::StringAttr::get(&context, "direct"));
            auto frontendFunction = mlir::func::FuncOp::create(location,
                functionName, mlir::FunctionType::get(&context, {}, {}));
            auto* frontendEntry = frontendFunction.addEntryBlock();
            frontendModule.push_back(frontendFunction);
            mlir::OpBuilder frontendBuilder(
                frontendEntry, frontendEntry->begin());
            std::size_t ordinal = 0;
            for (const std::size_t startCycle : startCycles) {
                auto varied = sxmTile;
                for (std::size_t lane = 0; lane < varied.permute_map.size(); ++lane)
                    varied.permute_map[lane] = (lane + ordinal) % varied.permute_map.size();
                ++ordinal;
                const auto onePoint = SxmLaunchDomain::Launch(
                    startCycle, {1, 1}, {1, 1}, varied);
                std::string frontendError;
                require(mlir::succeeded(
                            command::materializeSxmMacroCommand(
                                frontendBuilder, location, false, 0,
                                onePoint, &frontendError)),
                    "failed to materialize frontend bandwidth fixture");
            }
            frontendBuilder.create<mlir::func::ReturnOp>(location);
            require(mlir::succeeded(mlir::verify(frontendModule)),
                "frontend bandwidth fixture did not verify");
            return frontendModule;
        };

    // Distinct permutation maps cannot be folded into one template run.
    // Check actual bounded Macro fetch/parser throughput, not the retired
    // four-word SXM packet model.
    std::vector<std::size_t> denseCycles(32), spacedCycles(32);
    for (std::size_t i = 0; i < denseCycles.size(); ++i) {
        denseCycles[i] = i; spacedCycles[i] = i * 16;
    }
    const auto replayFrontend = [&](const std::vector<std::size_t>& starts) {
        const auto binary = ftlpu::compiler::target::translate_command_module(
            makeSxmFrontendModule(starts, "sxm_macro_frontend"));
        require(binary.queues.size() == 1 && binary.queues[0].commands.size() == starts.size(),
            "distinct SXM maps lost their Macro templates");
        InstructionControlUnit icu;
        load_queue_programs_into_icu(binary.queues, icu);
        std::vector<std::size_t> issued;
        for (std::size_t cycle = 0; cycle <= starts.back(); ++cycle)
            if (icu.sxm_permute_iq(Hemisphere::East).tick()) issued.push_back(cycle);
        require(issued == starts, "SXM Macro frontend changed exact issue cycles");
    };
    bool bandwidthRejected = false;
    try { replayFrontend(denseCycles); }
    catch (const StaticScheduleError&) { bandwidthRejected = true; }
    require(bandwidthRejected, "SXM Macro decoder accepted a sustained oversubscribed stream");
    replayFrontend(spacedCycles);

    auto dequantModule = mlir::ModuleOp::create(location);
    dequantModule->setAttr("ftlpu.target",
        targetModel.to_attribute(&context));
    dequantModule->setAttr("ftlpu.command_lowering",
        mlir::StringAttr::get(&context, "direct"));
    auto dequantFunction = mlir::func::FuncOp::create(location,
        "mxm_dequant_domain", mlir::FunctionType::get(&context, {}, {}));
    auto* dequantEntry = dequantFunction.addEntryBlock();
    dequantModule.push_back(dequantFunction);
    mlir::OpBuilder dequantBuilder(dequantEntry, dequantEntry->begin());
    const auto dequant = MxmDequantDomain::Dequant(
        AffineLoopDomain {3, {4, 2, 1}, {1, 9, 1}},
        MxmDequantInstruction::Scale(1.0f));
    require(mlir::succeeded(command::materializeMxmDequantMacroCommand(
                dequantBuilder, location, 0, dequant, 7, &error)),
        "direct MXM Macro dequant materialization failed");
    dequantBuilder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(dequantModule)),
        "command.mxm_dequant_3d did not verify");
    const auto dequantBinary =
        ftlpu::compiler::target::translate_command_module(dequantModule);
    require(dequantBinary.scale_relocations.size() == 1
            && dequantBinary.scale_relocations[0].binding_index == 7
            && dequantBinary.scale_relocations[0].queue_kind
                == QueueKind::MxmDequant
            && dequantBinary.scale_relocations[0].command_index == 0,
        "raw MXM dequant scale relocation was not preserved");

    auto scheduleModule = mlir::ModuleOp::create(location);
    scheduleModule->setAttr("ftlpu.target",
        targetModel.to_attribute(&context));
    const auto vxmType = mlir::RankedTensorType::get(
        {1}, mlir::BFloat16Type::get(&context));
    auto scheduleFunction = mlir::func::FuncOp::create(location,
        "vxm_wave_interval_boundary",
        mlir::FunctionType::get(&context, {vxmType, vxmType}, {}));
    auto* scheduleEntry = scheduleFunction.addEntryBlock();
    scheduleModule.push_back(scheduleFunction);
    mlir::OpBuilder scheduleBuilder(scheduleEntry, scheduleEntry->begin());
    mlir::OperationState vxmState(
        location, schedule::VxmOp::getOperationName());
    vxmState.addOperands(
        {scheduleEntry->getArgument(0), scheduleEntry->getArgument(1)});
    vxmState.addTypes(vxmType);
    vxmState.addAttributes({
        scheduleBuilder.getNamedAttr("cycle",
            scheduleBuilder.getI64IntegerAttr(0)),
        scheduleBuilder.getNamedAttr("queue",
            scheduleBuilder.getI64IntegerAttr(0)),
        scheduleBuilder.getNamedAttr("opcode",
            scheduleBuilder.getStringAttr("multiply")),
        scheduleBuilder.getNamedAttr("chain_depth",
            scheduleBuilder.getI64IntegerAttr(2)),
        scheduleBuilder.getNamedAttr("lhs_kind",
            scheduleBuilder.getStringAttr("stream_bf16")),
        scheduleBuilder.getNamedAttr("lhs_index",
            scheduleBuilder.getI64IntegerAttr(0)),
        scheduleBuilder.getNamedAttr("lhs_immediate",
            scheduleBuilder.getF32FloatAttr(0.0)),
        scheduleBuilder.getNamedAttr("rhs_kind",
            scheduleBuilder.getStringAttr("immediate")),
        scheduleBuilder.getNamedAttr("rhs_index",
            scheduleBuilder.getI64IntegerAttr(0)),
        scheduleBuilder.getNamedAttr("rhs_immediate",
            scheduleBuilder.getF32FloatAttr(1.0)),
        scheduleBuilder.getNamedAttr("cast_target",
            scheduleBuilder.getStringAttr("bf16")),
        scheduleBuilder.getNamedAttr("output_stream",
            scheduleBuilder.getI64IntegerAttr(-1)),
        scheduleBuilder.getNamedAttr("repeat_count",
            scheduleBuilder.getI64IntegerAttr(4)),
        scheduleBuilder.getNamedAttr("repeat_interval",
            scheduleBuilder.getI64IntegerAttr(1)),
        scheduleBuilder.getNamedAttr("wave_count",
            scheduleBuilder.getI64IntegerAttr(2)),
        scheduleBuilder.getNamedAttr("wave_interval",
            scheduleBuilder.getI64IntegerAttr(4)),
        scheduleBuilder.getNamedAttr("input_hemisphere",
            scheduleBuilder.getStringAttr("east")),
        scheduleBuilder.getNamedAttr("output_hemisphere",
            scheduleBuilder.getStringAttr("east")),
    });
    auto vxmSchedule = llvm::cast<schedule::VxmOp>(
        scheduleBuilder.create(vxmState));
    scheduleBuilder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(scheduleModule)),
        "VXM verifier rejected a wave starting after the contiguous run");
    vxmSchedule->setAttr("wave_interval",
        scheduleBuilder.getI64IntegerAttr(3));
    require(mlir::failed(mlir::verify(scheduleModule)),
        "VXM verifier accepted overlapping contiguous runs");

    std::cout << "vxm_sequential_command_test passed: "
              << "vxm_words=1 vxm_launches=2 run_length=32 "
              << "sxm_transpose_context_bits=256 sxm_permute_context_bits=384 "
              << "sxm_launches=6\n";
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "vxm_sequential_command_test failed: "
              << exception.what() << '\n';
    return 1;
}
