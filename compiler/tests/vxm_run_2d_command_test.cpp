#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Command/Transforms/fu_3d_command_materializer.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
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

} // namespace

int main() try
{
    using namespace ftlpu;
    using namespace ftlpu::compiler;
    using namespace ftlpu::software::runtime;

    VxmLaneAluInstruction lane {};
    lane.operation = VxmAluOpcode::Multiply;
    lane.lhs = VxmLaneOperand::StreamBFloat16(1.0f, 0);
    lane.rhs = VxmLaneOperand::Imm(0.5f);
    lane.output_type = VxmCastTarget::BFloat16;
    lane.repeat_count = 32;
    const auto compact = VxmCompactInstructionCodec::encode(
        0, VxmChainDepth::Two, lane);
    const auto instruction = VxmIcuRun2DInstruction::Run2D(
        2, {2, 2}, {5, 17}, compact);
    auto hardwareInstruction = instruction;
    hardwareInstruction.loop.start_cycle = 0;
    const auto expectedPacket =
        isa::encode_vxm_icu_run_2d_instruction(hardwareInstruction);

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect,
        command::CommandDialect, schedule::ScheduleDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    const auto location = mlir::UnknownLoc::get(&context);
    auto module = mlir::ModuleOp::create(location);
    const ftlpu::compiler::target::LPUTargetModel targetModel;
    module->setAttr("ftlpu.target", targetModel.to_attribute(&context));
    module->setAttr("ftlpu.command_lowering",
        mlir::StringAttr::get(&context, "direct"));
    auto function = mlir::func::FuncOp::create(location, "vxm_run_2d",
        mlir::FunctionType::get(&context, {}, {}));
    auto* entry = function.addEntryBlock();
    module.push_back(function);
    mlir::OpBuilder builder(entry, entry->begin());
    std::string error;
    require(mlir::succeeded(command::materializeVxmRun2DCommand(
                builder, location, 0, instruction, &error, 3)),
        "direct VXM RUN_2D materialization failed");
    builder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(module)),
        "command.vxm_run_2d did not verify");

    const auto binary =
        ftlpu::compiler::target::translate_command_module(module);
    require(binary.scale_relocations.size() == 1
            && binary.scale_relocations[0].binding_index == 3
            && binary.scale_relocations[0].queue_kind == QueueKind::Vxm
            && binary.scale_relocations[0].command_index == 1,
        "raw VXM scale relocation was not preserved");
    require(binary.queues.size() == 1
            && binary.queues[0].kind == QueueKind::Vxm
            && binary.queues[0].index == 0
            && binary.queues[0].commands.size()
                == isa::EncodedVxmIcuRun2DPacket::kWordCount + 1,
        "VXM RUN_2D did not translate to NOP plus one three-word queue packet");
    require(isa::decode_icu_nop_cycles(
                binary.queues[0].commands[0].command) == 2,
        "VXM RUN_2D initial schedule gap was not encoded as NOP(2)");
    for (std::size_t word = 0; word < expectedPacket.words.size(); ++word) {
        const auto& actual = binary.queues[0].commands[word + 1];
        require(actual.word_count
                    == isa::EncodedVxmIcuRun2DPacket::kLanesPerWord,
            "CommandBinary changed the VXM RUN_2D word width");
        for (std::size_t laneIndex = 0;
             laneIndex < expectedPacket.words[word].lanes.size(); ++laneIndex)
            require(actual.words[laneIndex]
                        == expectedPacket.words[word].lanes[laneIndex],
                "CommandBinary changed a VXM RUN_2D physical word");
    }

    const auto capacity = analyze_physical_imem(binary);
    require(capacity.used_slots == 4 && capacity.queues.size() == 1
            && capacity.queues[0].coarse_program_entries == 1
            && capacity.queues[0].expanded_work == 4
            && capacity.queues[0].peak_fu_3d_contexts == 1,
        "VXM RUN_2D i-MEM or context accounting is incorrect");

    InstructionControlUnit icu;
    load_queue_programs_into_icu(binary.queues, icu);
    std::vector<std::size_t> issueCycles;
    for (std::size_t cycle = 0; cycle <= 24; ++cycle) {
        if (const auto issued = icu.vxm_iq(0).tick()) {
            require(*issued == compact,
                "runtime changed the VXM compact instruction");
            require(VxmCompactInstructionCodec::decode(0, *issued)
                        .instruction.repeat_count == 32,
                "runtime expanded the VXM contiguous datapath run");
            issueCycles.push_back(cycle);
        }
    }
    require(issueCycles == std::vector<std::size_t> {2, 7, 19, 24},
        "VXM RUN_2D ICU issued at incorrect absolute cycles");

    auto sxmMap = SxmInstruction::PermuteMap {};
    for (std::size_t laneIndex = 0; laneIndex < sxmMap.size(); ++laneIndex)
        sxmMap[laneIndex] =
            (laneIndex + hw::kLanesPerTile) % sxmMap.size();
    const auto sxmTile = SxmInstruction::Permute(
        {{0}, {1}}, {{16}, {17}}, sxmMap);
    const auto sxmRun = SxmIcuRun2DInstruction::Run2D(
        7, {3, 2}, {2, 13}, sxmTile);
    auto sxmHardwareRun = sxmRun;
    sxmHardwareRun.loop.start_cycle = 0;
    const auto sxmExpectedPacket =
        isa::encode_sxm_icu_run_2d_instruction(sxmHardwareRun);
    auto sxmModule = mlir::ModuleOp::create(location);
    sxmModule->setAttr("ftlpu.target", targetModel.to_attribute(&context));
    sxmModule->setAttr("ftlpu.command_lowering",
        mlir::StringAttr::get(&context, "direct"));
    auto sxmFunction = mlir::func::FuncOp::create(location, "sxm_run_2d",
        mlir::FunctionType::get(&context, {}, {}));
    auto* sxmEntry = sxmFunction.addEntryBlock();
    sxmModule.push_back(sxmFunction);
    mlir::OpBuilder sxmBuilder(sxmEntry, sxmEntry->begin());
    require(mlir::succeeded(command::materializeSxmRun2DCommand(
                sxmBuilder, location, false, 0, sxmRun, &error)),
        "direct SXM RUN_2D materialization failed");
    sxmBuilder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(sxmModule)),
        "command.sxm_run_2d did not verify");
    const auto sxmBinary =
        ftlpu::compiler::target::translate_command_module(sxmModule);
    require(sxmBinary.queues.size() == 1
            && sxmBinary.queues[0].kind == QueueKind::SxmPermute
            && sxmBinary.queues[0].commands.size()
                == isa::EncodedSxmIcuRun2DPacket::kWordCount + 1,
        "SXM RUN_2D did not translate to NOP plus one six-word queue packet");
    require(isa::decode_icu_nop_cycles(
                sxmBinary.queues[0].commands[0].command) == 7,
        "SXM RUN_2D initial schedule gap was not encoded as NOP(7)");
    for (std::size_t word = 0; word < sxmExpectedPacket.words.size(); ++word)
        for (std::size_t laneIndex = 0;
             laneIndex < sxmExpectedPacket.words[word].lanes.size();
             ++laneIndex)
            require(sxmBinary.queues[0].commands[word + 1].words[laneIndex]
                        == sxmExpectedPacket.words[word].lanes[laneIndex],
                "CommandBinary changed an SXM RUN_2D physical word");
    const auto sxmCapacity = analyze_physical_imem(sxmBinary);
    require(sxmCapacity.used_slots == 7
            && sxmCapacity.queues[0].coarse_program_entries == 1
            && sxmCapacity.queues[0].expanded_work == 6
            && sxmCapacity.queues[0].fu_3d_context_bits
                == hw::kIcuSxmRun2DContextBits,
        "SXM RUN_2D i-MEM or context accounting is incorrect");
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
        "SXM RUN_2D ICU issued at incorrect absolute cycles");

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
            for (const std::size_t startCycle : startCycles) {
                const auto onePoint = SxmIcuRun2DInstruction::Run2D(
                    startCycle, {1, 1}, {1, 1}, sxmTile);
                std::string frontendError;
                require(mlir::succeeded(
                            command::materializeSxmRun2DCommand(
                                frontendBuilder, location, false, 0,
                                onePoint, &frontendError)),
                    "failed to materialize frontend bandwidth fixture");
            }
            frontendBuilder.create<mlir::func::ReturnOp>(location);
            require(mlir::succeeded(mlir::verify(frontendModule)),
                "frontend bandwidth fixture did not verify");
            return frontendModule;
        };

    // Four adjacent six-word packets fit the static one-context schedule, but
    // not the physical frontend: the 16-word initial IQ and one-word/cycle
    // refill leave only one word resident for the packet due at cycle 3.
    auto impossibleFrontendModule = makeSxmFrontendModule(
        {0, 1, 2, 3}, "sxm_impossible_frontend");
    std::string frontendDiagnostic;
    try {
        static_cast<void>(ftlpu::compiler::target::translate_command_module(
            impossibleFrontendModule));
    } catch (const std::runtime_error& exception) {
        frontendDiagnostic = exception.what();
    }
    require(frontendDiagnostic.find(
                "raw FU loop schedule exceeds ICU frontend bandwidth")
                != std::string::npos
            && frontendDiagnostic.find("resource=sxm_permute, queue=0")
                != std::string::npos
            && frontendDiagnostic.find("cycle=3") != std::string::npos
            && frontendDiagnostic.find("required_words=6")
                != std::string::npos
            && frontendDiagnostic.find("available_words=1")
                != std::string::npos,
        "CommandBinary accepted an SXM packet that the ICU cannot fetch on time");

    // Moving only the fourth packet to cycle 9 gives the same frontend enough
    // time to fetch all six words, including the intervening NOP descriptor.
    auto feasibleFrontendModule = makeSxmFrontendModule(
        {0, 1, 2, 9}, "sxm_feasible_frontend");
    const auto feasibleFrontendBinary =
        ftlpu::compiler::target::translate_command_module(
            feasibleFrontendModule);
    require(feasibleFrontendBinary.queues.size() == 1
            && feasibleFrontendBinary.queues[0].commands.size() == 25,
        "feasible SXM frontend fixture produced an unexpected i-MEM image");
    InstructionControlUnit feasibleFrontendIcu;
    load_queue_programs_into_icu(
        feasibleFrontendBinary.queues, feasibleFrontendIcu);
    issueCycles.clear();
    for (std::size_t cycle = 0; cycle <= 9; ++cycle) {
        if (feasibleFrontendIcu.sxm_permute_iq(Hemisphere::East).tick())
            issueCycles.push_back(cycle);
    }
    require(issueCycles == std::vector<std::size_t> {0, 1, 2, 9},
        "frontend-feasible SXM packets did not retain their exact schedule");

    auto dequantModule = mlir::ModuleOp::create(location);
    dequantModule->setAttr("ftlpu.target",
        targetModel.to_attribute(&context));
    dequantModule->setAttr("ftlpu.command_lowering",
        mlir::StringAttr::get(&context, "direct"));
    auto dequantFunction = mlir::func::FuncOp::create(location,
        "mxm_dequant_3d", mlir::FunctionType::get(&context, {}, {}));
    auto* dequantEntry = dequantFunction.addEntryBlock();
    dequantModule.push_back(dequantFunction);
    mlir::OpBuilder dequantBuilder(dequantEntry, dequantEntry->begin());
    const auto dequant = MxmDequantIcuInstruction::Dequant3D(
        IcuLoop3D {3, {4, 2, 1}, {1, 9, 1}},
        MxmDequantInstruction::Scale(1.0f));
    require(mlir::succeeded(command::materializeMxmDequant3DCommand(
                dequantBuilder, location, 0, dequant, 7, &error)),
        "direct MXM DEQUANT_3D materialization failed");
    dequantBuilder.create<mlir::func::ReturnOp>(location);
    require(mlir::succeeded(mlir::verify(dequantModule)),
        "command.mxm_dequant_3d did not verify");
    const auto dequantBinary =
        ftlpu::compiler::target::translate_command_module(dequantModule);
    require(dequantBinary.scale_relocations.size() == 1
            && dequantBinary.scale_relocations[0].binding_index == 7
            && dequantBinary.scale_relocations[0].queue_kind
                == QueueKind::MxmDequant
            && dequantBinary.scale_relocations[0].command_index == 1,
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

    std::cout << "vxm_run_2d_command_test passed: "
              << "vxm_words=3 vxm_launches=4 run_length=32 "
              << "sxm_words=6 sxm_launches=6\n";
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "vxm_run_2d_command_test failed: "
              << exception.what() << '\n';
    return 1;
}
