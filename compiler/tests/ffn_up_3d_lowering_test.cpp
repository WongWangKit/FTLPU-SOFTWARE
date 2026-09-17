#include "ftlpu/compiler/Dialect/Command/IR/command_dialect.hpp"
#include "ftlpu/compiler/Dialect/Command/Transforms/fu_3d_command_materializer.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_up_3d_lowering.hpp"
#include "ftlpu/compiler/Target/command_binary.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/software/runtime/imem_capacity.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::logic_error(message);
}

ftlpu::compiler::schedule::FfnUp3DPlacement qwenPlacement()
{
    using namespace ftlpu::compiler::schedule;
    FfnUp3DPlacement placement;
    placement.mem_slices_per_hemisphere = 52;
    placement.mem_banks_per_slice = 2;
    placement.mxms_per_hemisphere = 1;
    placement.weight_stream_base = 8;
    placement.activation_stream_base = 16;
    placement.activation_bank = 1;
    placement.result_bank = 0;
    placement.accumulator_address_base = 32;
    placement.weight_outer_group_size = 2;
    placement.hemispheres.resize(2);
    for (std::size_t h = 0; h < placement.hemispheres.size(); ++h) {
        auto& hemisphere = placement.hemispheres[h];
        hemisphere.weight_regions = {
            {0, 42, 1, 36, 0},
            {42, 42, 1, 44, 0},
            {84, 42, 0, 36, 0},
            {126, 14, 0, 44, 0},
        };
        for (std::size_t slice = 0;
             slice < hemisphere.activation_slices.size(); ++slice)
            hemisphere.activation_slices[slice] =
                static_cast<std::int64_t>(slice);
        hemisphere.result_slices = {8, 9};
        hemisphere.mxm_queue = static_cast<std::int64_t>(h);
        hemisphere.result_stream_base =
            static_cast<std::int64_t>(12 + 8 * h);
    }
    return placement;
}

ftlpu::compiler::schedule::FfnUp3DTimeline qwenTimeline()
{
    using namespace ftlpu::compiler::schedule;
    FfnUp3DTimeline timeline;
    timeline.mxm_load_start_cycle = 32;
    timeline.mxm_dequant_start_cycle = 32;
    timeline.mxm_compute_start_cycle = 64;
    timeline.mxm_result_start_cycle = 1584;
    timeline.reduction_cycle_stride = 32;
    timeline.pair_cycle_stride = 1536;
    return timeline;
}

ftlpu::compiler::schedule::FfnUp3DRouteLatencies qwenRouteLatencies()
{
    using namespace ftlpu::compiler;
    schedule::FfnUp3DRouteLatencies result;
    const target::LPUTargetModel model;
    const auto latency = [&](target::StreamEndpoint source,
                             target::StreamEndpoint destination,
                             target::StreamDirection direction,
                             std::size_t slice) {
        const auto value = model.transport_latency(source, destination,
            direction, static_cast<std::int64_t>(slice));
        if (!value)
            throw std::logic_error(
                "default target is missing a Qwen Up physical route");
        return *value;
    };
    for (std::size_t slice = 0;
         slice < ftlpu::hw::kMemSliceColumns; ++slice) {
        result.mem_to_mxm_weight_cycles[slice] = latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmWeight,
            target::StreamDirection::East, slice);
        result.mem_to_mxm_activation_cycles[slice] = latency(
            target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmActivation,
            target::StreamDirection::East, slice);
        result.mxm_result_to_mem_cycles[slice] = latency(
            target::StreamEndpoint::MxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::West, slice);
    }
    return result;
}

std::size_t points(const ftlpu::IcuLoop3D& loop)
{
    return ftlpu::detail::icu_loop_3d_point_count(loop);
}

std::size_t physicalMxmQueue(
    std::size_t logicalQueue, std::size_t logicalMxmsPerHemisphere)
{
    const auto hemisphere = logicalQueue / logicalMxmsPerHemisphere;
    const auto localMxm = logicalQueue % logicalMxmsPerHemisphere;
    return hemisphere * ftlpu::hw::kMxmsPerHemisphere + localMxm;
}

} // namespace

int main() try
{
    using namespace ftlpu;
    using namespace ftlpu::compiler::schedule;

    // Qwen2.5-1.5B FFN Up: [32,1536] x [1536,8960]. The
    // lowering is invoked directly from shape/placement/timeline/topology; no
    // Schedule dialect events or schedule-compression API exist in this path.
    const FfnUp3DShape shape {32, 1536, 8960};
    auto placement = qwenPlacement();
    const auto timeline = qwenTimeline();
    const auto routeLatencies = qwenRouteLatencies();
    std::string error;
    auto lowered = lowerFfnUpToFu3D(
        shape, placement, timeline, routeLatencies, 0x3c00, &error);
    require(mlir::succeeded(lowered),
        "direct Qwen Up 3-D lowering failed: " + error);

    const auto& program = *lowered;
    const auto memReads = std::count_if(program.mem_commands.begin(),
        program.mem_commands.end(), [](const auto& command) {
            return command.instruction.opcode == MemIcuOpcode::Read3D;
        });
    const auto memWrites = std::count_if(program.mem_commands.begin(),
        program.mem_commands.end(), [](const auto& command) {
            return command.instruction.opcode == MemIcuOpcode::Write3D;
        });
    require(program.mem_commands.size() == 100
            && memReads == 96 && memWrites == 4,
        "Qwen Up must lower to 64 weight reads, 32 activation reads, and 4 writes");
    require(program.mxm_load_commands.size() == 2
            && program.mxm_dequant_commands.size() == 2
            && program.mxm_compute_commands.size() == 2
            && program.hardware_command_count() == 106,
        "Qwen Up must lower to 106 FU-specific coarse commands");
    const auto& firstWeight = program.mem_commands[0];
    const auto& firstActivation = program.mem_commands[64];
    const auto& firstResult = program.mem_commands[96];
    std::unordered_set<std::size_t> memQueues;
    for (const auto& command : program.mem_commands)
        require(memQueues.insert(command.queue).second,
            "Qwen Up emitted more than one coarse command to a MEM queue");
    const auto requireZeroHardwareOrigins = [](const auto& commands,
                                                const char* family) {
        for (const auto& command : commands)
            require(command.instruction.loop.start_cycle == 0,
                std::string(family)
                    + " hardware instruction retained an absolute cycle");
    };
    requireZeroHardwareOrigins(program.mem_commands, "MEM");
    requireZeroHardwareOrigins(program.mxm_load_commands, "MXM load");
    requireZeroHardwareOrigins(program.mxm_dequant_commands, "MXM dequant");
    requireZeroHardwareOrigins(program.mxm_compute_commands, "MXM compute");

    std::size_t physicalWords = 0;
    for (const auto& command : program.mem_commands) {
        static_cast<void>(
            isa::encode_mem_icu_3d_instruction(command.instruction));
        physicalWords += isa::EncodedMemIcu3DPacket::kWordCount;
    }
    for (const auto& command : program.mxm_load_commands) {
        static_cast<void>(
            isa::encode_mxm_load_icu_3d_instruction(command.instruction));
        physicalWords += isa::EncodedMxmLoadIcu3DPacket::kWordCount;
    }
    for (const auto& command : program.mxm_dequant_commands) {
        static_cast<void>(isa::encode_mxm_dequant_icu_3d_instruction(
            command.instruction));
        physicalWords += isa::EncodedMxmDequantIcu3DPacket::kWordCount;
    }
    for (const auto& command : program.mxm_compute_commands) {
        static_cast<void>(isa::encode_mxm_compute_icu_3d_instruction(
            command.instruction));
        physicalWords += isa::EncodedMxmComputeIcu3DPacket::kWordCount;
    }
    require(physicalWords == 312,
        "106 coarse commands must encode to 312 physical ICU words");

    // Prove the compiler boundary end to end: the direct lower result is
    // materialized as FU-specific raw Command ops, translated without a
    // Schedule pass, serialized bit-for-bit, and loaded as physical i-MEM
    // words. CommandBinary sees no fine issue list to discover or compress.
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect,
        ftlpu::compiler::command::CommandDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    const auto location = mlir::UnknownLoc::get(&context);
    auto module = mlir::ModuleOp::create(location);
    auto oneMxmThroughput =
        ftlpu::compiler::target::ThroughputModel {};
    oneMxmThroughput.mxms_per_hemisphere = 1;
    const ftlpu::compiler::target::LPUTargetModel oneMxmTarget(
        ftlpu::compiler::target::MemoryTopology {},
        ftlpu::compiler::target::StreamTopology {},
        oneMxmThroughput);
    module->setAttr("ftlpu.target", oneMxmTarget.to_attribute(&context));
    auto function = mlir::func::FuncOp::create(location, "qwen_up_3d",
        mlir::FunctionType::get(&context, {}, {}));
    auto* entry = function.addEntryBlock();
    module.push_back(function);
    mlir::OpBuilder builder(entry, entry->begin());
    require(mlir::succeeded(
                ftlpu::compiler::command::materializeFfnUp3DCommands(
                    builder, location, program, &error)),
        "failed to materialize direct FU 3-D Command IR: " + error);
    builder.create<mlir::func::ReturnOp>(location);
    ftlpu::compiler::command::Mem3DOp relocatedMem;
    function.walk([&](ftlpu::compiler::command::Mem3DOp op) {
        if (!relocatedMem) relocatedMem = op;
    });
    require(static_cast<bool>(relocatedMem),
        "direct lowering did not materialize a MEM 3-D command");
    relocatedMem->setAttr("address_binding",
        builder.getI64IntegerAttr(7));
    relocatedMem->setAttr("address_binding_access",
        builder.getStringAttr("input"));
    require(mlir::succeeded(mlir::verify(module)),
        "direct FU 3-D Command IR did not verify");

    auto binaryProgram =
        ftlpu::compiler::target::translate_command_module(module);
    require(binaryProgram.hardware.mxms_per_hemisphere == 1,
        "direct Up binary lost its one-MXM-per-hemisphere topology");
    require(binaryProgram.address_relocations.size() == 1
            && binaryProgram.address_relocations.front().binding_index == 7
            && binaryProgram.address_relocations.front().queue_kind
                == ftlpu::software::runtime::QueueKind::Mem
            && binaryProgram.address_relocations.front().queue_index
                == relocatedMem.getQueue()
            && binaryProgram.address_relocations.front().command_index == 1,
        "raw MEM 3-D command did not preserve its direct relocation");
    std::size_t binaryPhysicalWords = 0;
    std::size_t binaryPacketHeaders = 0;
    std::size_t binaryNops = 0;
    for (const auto& queue : binaryProgram.queues) {
        binaryPhysicalWords += queue.commands.size();
        binaryPacketHeaders += static_cast<std::size_t>(std::count_if(
            queue.commands.begin(), queue.commands.end(),
            ftlpu::software::runtime::is_fu_3d_raw_packet_header));
        binaryNops += static_cast<std::size_t>(std::count_if(
            queue.commands.begin(), queue.commands.end(),
            [](const auto& command) {
                return command.instruction_kind
                        == ftlpu::software::runtime::InstructionKind::None
                    && isa::decode_icu_command_opcode(command.command)
                        == isa::IcuCommandOpcode::Nop;
            }));
        require(std::all_of(queue.commands.begin(), queue.commands.end(),
                    [](const auto& command) {
                        return ftlpu::software::runtime::
                                   is_fu_3d_raw_word_command(command)
                            || (command.instruction_kind
                                    == ftlpu::software::runtime::
                                        InstructionKind::None
                                && isa::decode_icu_command_opcode(
                                       command.command)
                                    == isa::IcuCommandOpcode::Nop);
                    }),
            "CommandBinary emitted something other than FU 3-D words and timing NOPs");
    }
    require(binaryPhysicalWords == 418 && binaryPacketHeaders == 106
            && binaryNops == 106,
        "CommandBinary did not encode 106 coarse packets behind 106 timing NOPs");

    const auto capacity =
        ftlpu::software::runtime::analyze_physical_imem(binaryProgram);
    std::size_t capacityCoarsePackets = 0;
    for (const auto& queue : capacity.queues)
        capacityCoarsePackets += queue.coarse_program_entries;
    require(capacity.used_slots == 418
            && capacityCoarsePackets == 106,
        "FU 3-D capacity accounting confused coarse packets with physical words");
    require(InstructionControlUnit::MemIcu::three_d_context_depth
                == hw::kIcuMem3DContextDepth
            && InstructionControlUnit::MxmLoadIcu::three_d_context_depth
                == hw::kIcuMxmLoad3DContextDepth
            && InstructionControlUnit::MxmDequantIcu::three_d_context_depth
                == hw::kIcuMxmDequant3DContextDepth
            && InstructionControlUnit::MxmComputeIcu::three_d_context_depth
                == hw::kIcuMxmCompute3DContextDepth
            && hw::kIcuMem3DContextDepth == 1
            && hw::kIcuMxmLoad3DContextDepth == 1
            && hw::kIcuMxmDequant3DContextDepth == 1
            && hw::kIcuMxmCompute3DContextDepth == 1,
        "a physical ICU exposed more than one active coarse-instruction context");

    // Exceeding the FU-specific context file must fail at the compiler
    // boundary instead of failing late in the ICU.
    auto overlapModule = mlir::ModuleOp::create(location);
    auto overlapFunction = mlir::func::FuncOp::create(location,
        "overlapping_3d_contexts",
        mlir::FunctionType::get(&context, {}, {}));
    auto* overlapEntry = overlapFunction.addEntryBlock();
    overlapModule.push_back(overlapFunction);
    mlir::OpBuilder overlapBuilder(overlapEntry, overlapEntry->begin());
    FfnUp3DLoweringResult overlapProgram;
    for (std::size_t context = 0;
         context <= hw::kIcuMem3DContextDepth; ++context) {
        auto interleavedWeight = firstWeight;
        interleavedWeight.cycle += context;
        overlapProgram.mem_commands.push_back(interleavedWeight);
    }
    require(mlir::succeeded(
                ftlpu::compiler::command::materializeFfnUp3DCommands(
                    overlapBuilder, location, overlapProgram, &error)),
        "failed to materialize the context-overflow fixture: " + error);
    overlapBuilder.create<mlir::func::ReturnOp>(location);
    bool overlapRejected = false;
    try {
        static_cast<void>(
            ftlpu::compiler::target::translate_command_module(overlapModule));
    } catch (const std::runtime_error& exception) {
        overlapRejected = std::string(exception.what())
            .find("single-context ICU") != std::string::npos;
    }
    require(overlapRejected,
        "compiler accepted more overlapping descriptors than the MEM context file");

    std::stringstream image(
        std::ios::in | std::ios::out | std::ios::binary);
    ftlpu::software::runtime::write_binary_program(binaryProgram, image);
    image.seekg(0);
    auto roundTrip =
        ftlpu::software::runtime::read_binary_program(image);
    require(roundTrip.queues.size() == binaryProgram.queues.size(),
        "FU 3-D binary round trip changed the executable queue count");
    for (std::size_t queue = 0; queue < roundTrip.queues.size(); ++queue) {
        const auto& expected = binaryProgram.queues[queue];
        const auto& actual = roundTrip.queues[queue];
        require(actual.kind == expected.kind
                && actual.index == expected.index
                && actual.commands.size() == expected.commands.size(),
            "FU 3-D binary round trip changed a physical queue");
        for (std::size_t word = 0;
             word < actual.commands.size(); ++word)
            require(actual.commands[word].command
                        == expected.commands[word].command
                    && actual.commands[word].instruction_kind
                        == expected.commands[word].instruction_kind
                    && actual.commands[word].word_count
                        == expected.commands[word].word_count
                    && actual.commands[word].words
                        == expected.commands[word].words
                    && actual.commands[word].extension_words.empty(),
                "FU 3-D binary round trip changed a raw i-MEM bit");
    }

    InstructionControlUnit rawIcu;
    ftlpu::software::runtime::load_queue_programs_into_icu(
        roundTrip.queues, rawIcu,
        roundTrip.hardware.mxms_per_hemisphere);
    std::size_t loadedWords = 0;
    for (const auto& queue : roundTrip.queues) {
        switch (queue.kind) {
        case ftlpu::software::runtime::QueueKind::Mem:
            loadedWords += rawIcu.mem_iq(queue.index).imem_occupancy();
            break;
        case ftlpu::software::runtime::QueueKind::MxmLoad:
            loadedWords += rawIcu.mxm_load_iq(physicalMxmQueue(queue.index,
                roundTrip.hardware.mxms_per_hemisphere)).imem_occupancy();
            break;
        case ftlpu::software::runtime::QueueKind::MxmDequant:
            loadedWords += rawIcu.mxm_dequant_iq(physicalMxmQueue(queue.index,
                roundTrip.hardware.mxms_per_hemisphere))
                .imem_occupancy();
            break;
        case ftlpu::software::runtime::QueueKind::MxmCompute:
            loadedWords += rawIcu.mxm_compute_iq(physicalMxmQueue(queue.index,
                roundTrip.hardware.mxms_per_hemisphere))
                .imem_occupancy();
            break;
        default:
            throw std::logic_error(
                "direct Qwen Up emitted a non-MEM/MXM queue");
        }
    }
    require(loadedWords == binaryPhysicalWords,
        "runtime did not load the exact FU 3-D physical i-MEM image");

    bool sawFirstWeightIssue = false;
    for (std::size_t cycle = 0; cycle <= 26; ++cycle) {
        const auto issue = rawIcu.mem_iq(firstWeight.queue).tick();
        if (issue) {
            require(cycle == 26 && issue->opcode == MemOpcode::Read
                    && issue->address == 0
                    && issue->stream == StreamId::East(8).packed(),
                "raw MEM packet expanded to the wrong first native issue");
            sawFirstWeightIssue = true;
        }
    }
    require(sawFirstWeightIssue,
        "raw MEM packet did not reach the ICU decode/issue path");

    bool sawFirstLoadIssue = false;
    bool sawFirstDequantIssue = false;
    for (std::size_t cycle = 0; cycle <= 32; ++cycle) {
        if (const auto issue = rawIcu.mxm_load_iq(0).tick()) {
            require(cycle == 32
                    && issue->opcode == MxmControlOpcode::IW
                    && issue->weight_buffer == 0
                    && issue->weight_column == 0,
                "raw MXM LOAD_3D packet expanded to the wrong first issue");
            sawFirstLoadIssue = true;
        }
        if (const auto issue = rawIcu.mxm_dequant_iq(0).tick()) {
            require(cycle == 32 && issue->scale_bf16 == 0x3c00,
                "raw MXM DEQUANT_3D packet expanded to the wrong first issue");
            sawFirstDequantIssue = true;
        }
    }
    require(sawFirstLoadIssue && sawFirstDequantIssue,
        "raw MXM load/dequant packets did not reach their ICU decoders");

    bool sawFirstComputeIssue = false;
    for (std::size_t cycle = 0; cycle <= 64; ++cycle) {
        if (const auto issue = rawIcu.mxm_compute_iq(0).tick()) {
            require(cycle == 64
                    && issue->opcode == MxmControlOpcode::Compute
                    && issue->weight_buffer == 0
                    && issue->accumulator_address == 32,
                "raw MXM COMPUTE_3D packet expanded to the wrong first issue");
            sawFirstComputeIssue = true;
        }
    }
    require(sawFirstComputeIssue,
        "raw MXM compute packet did not reach its ICU decoder");

    // Exercise representative queues all the way to their final loop point.
    // This consumes the compiler-produced binary through the runtime loader;
    // the expected native fields are therefore checked after the CModel ICU
    // has decoded and advanced its own hardware counters.
    InstructionControlUnit coverageIcu;
    ftlpu::software::runtime::load_queue_programs_into_icu(
        roundTrip.queues, coverageIcu,
        roundTrip.hardware.mxms_per_hemisphere);
    constexpr std::size_t kFinalCycle = 215132;
    std::size_t weightIssueCount = 0;
    std::size_t activationIssueCount = 0;
    std::size_t resultIssueCount = 0;
    std::size_t loadIssueCount = 0;
    std::size_t dequantIssueCount = 0;
    std::size_t computeIssueCount = 0;
    std::size_t westLoadIssueCount = 0;
    std::size_t westDequantIssueCount = 0;
    std::size_t westComputeIssueCount = 0;
    bool sawWeightMiddle = false;
    bool sawWeightFinal = false;
    bool sawActivationMiddle = false;
    bool sawActivationFinal = false;
    bool sawResultMiddle = false;
    bool sawResultFinal = false;
    bool sawLoadMiddle = false;
    bool sawLoadFinal = false;
    bool sawDequantMiddle = false;
    bool sawDequantFinal = false;
    bool sawComputeMiddle = false;
    bool sawComputeFinal = false;
    for (std::size_t cycle = 0; cycle <= kFinalCycle; ++cycle) {
        if (const auto issue =
                coverageIcu.mem_iq(firstWeight.queue).tick()) {
            ++weightIssueCount;
            if (cycle == 31484) {
                require(issue->address == 4026,
                    "middle weight counter decoded the wrong MEM address");
                sawWeightMiddle = true;
            }
            if (cycle == 64509) {
                require(issue->address == 8063,
                    "final weight counter decoded the wrong MEM address");
                sawWeightFinal = true;
            }
        }
        if (const auto issue =
                coverageIcu.mem_iq(firstActivation.queue).tick()) {
            ++activationIssueCount;
            if (cycle == 108321) {
                require(issue->address == 94,
                    "middle activation counter decoded the wrong MEM address");
                sawActivationMiddle = true;
            }
            if (cycle == 215081) {
                require(issue->address == 191,
                    "final activation counter decoded the wrong MEM address");
                sawActivationFinal = true;
            }
        }
        if (const auto issue =
                coverageIcu.mem_iq(firstResult.queue).tick()) {
            ++resultIssueCount;
            if (cycle == 109133) {
                require(issue->opcode == MemOpcode::Write
                        && issue->address == 2256,
                    "middle result counter decoded the wrong MEM write");
                sawResultMiddle = true;
            }
            if (cycle == 215132) {
                require(issue->opcode == MemOpcode::Write
                        && issue->address == 4479,
                    "final result counter decoded the wrong MEM write");
                sawResultFinal = true;
            }
        }
        if (const auto issue = coverageIcu.mxm_load_iq(0).tick()) {
            ++loadIssueCount;
            if (cycle == 108290) {
                require(issue->weight_column == 2
                        && issue->weight_buffer == 1,
                    "middle LOAD_3D counter decoded the wrong native issue");
                sawLoadMiddle = true;
            }
            if (cycle == 215043) {
                require(issue->weight_column == 3
                        && issue->weight_buffer == 1,
                    "final LOAD_3D counter decoded the wrong native issue");
                sawLoadFinal = true;
            }
        }
        if (coverageIcu.mxm_load_iq(2).tick()) ++westLoadIssueCount;
        if (const auto issue = coverageIcu.mxm_dequant_iq(0).tick()) {
            ++dequantIssueCount;
            if (cycle == 108290) {
                require(issue->scale_bf16 == 0x3c00,
                    "middle DEQUANT_3D counter changed its scale");
                sawDequantMiddle = true;
            }
            if (cycle == 215043) {
                require(issue->scale_bf16 == 0x3c00,
                    "final DEQUANT_3D counter changed its scale");
                sawDequantFinal = true;
            }
        }
        if (coverageIcu.mxm_dequant_iq(2).tick()) ++westDequantIssueCount;
        if (const auto issue = coverageIcu.mxm_compute_iq(0).tick()) {
            ++computeIssueCount;
            if (cycle == 108336) {
                require(issue->weight_buffer == 1
                        && issue->accumulator_destination
                            == MxmAccumulatorDestination::Sram
                        && !issue->accumulator_clear
                        && issue->accumulator_output_format
                            == MxmAccumulatorOutputFormat::Float32,
                    "middle COMPUTE_3D counter selected the wrong mode");
                sawComputeMiddle = true;
            }
            if (cycle == 215103) {
                require(issue->weight_buffer == 1
                        && issue->accumulator_destination
                            == MxmAccumulatorDestination::Stream
                        && issue->accumulator_clear
                        && issue->accumulator_output_format
                            == MxmAccumulatorOutputFormat::BFloat16,
                    "terminal COMPUTE_3D counter did not drain and clear");
                sawComputeFinal = true;
            }
        }
        if (coverageIcu.mxm_compute_iq(2).tick()) ++westComputeIssueCount;
    }
    std::ostringstream coverageCounts;
    coverageCounts << "runtime/CModel expansion counts: weight="
                   << weightIssueCount << " activation="
                   << activationIssueCount << " result="
                   << resultIssueCount << " load=" << loadIssueCount
                   << " dequant=" << dequantIssueCount << " compute="
                   << computeIssueCount << " west-load="
                   << westLoadIssueCount << " west-dequant="
                   << westDequantIssueCount << " west-compute="
                   << westComputeIssueCount;
    require(weightIssueCount == 8064
            && activationIssueCount == 26880
            && resultIssueCount == 4480
            && loadIssueCount == 26880
            && dequantIssueCount == 26880
            && computeIssueCount == 215040
            && westLoadIssueCount == 26880
            && westDequantIssueCount == 26880
            && westComputeIssueCount == 215040,
        coverageCounts.str());
    std::ostringstream checkpointState;
    checkpointState << "runtime/CModel 3-D checkpoints: weight="
                    << sawWeightMiddle << '/' << sawWeightFinal
                    << " activation=" << sawActivationMiddle << '/'
                    << sawActivationFinal << " result=" << sawResultMiddle
                    << '/' << sawResultFinal << " load=" << sawLoadMiddle
                    << '/' << sawLoadFinal << " dequant="
                    << sawDequantMiddle << '/' << sawDequantFinal
                    << " compute=" << sawComputeMiddle << '/'
                    << sawComputeFinal;
    require(sawWeightMiddle && sawWeightFinal
            && sawActivationMiddle && sawActivationFinal
            && sawResultMiddle && sawResultFinal
            && sawLoadMiddle && sawLoadFinal
            && sawDequantMiddle && sawDequantFinal
            && sawComputeMiddle && sawComputeFinal,
        checkpointState.str());

    auto corrupted = roundTrip;
    auto corruptQueue = std::find_if(corrupted.queues.begin(),
        corrupted.queues.end(), [&](const auto& queue) {
            return queue.kind
                    == ftlpu::software::runtime::QueueKind::Mem
                && queue.index == firstWeight.queue;
        });
    require(corruptQueue != corrupted.queues.end(),
        "missing raw MEM queue for corruption test");
    // Flip the packet-version header bit while keeping the tagged container's
    // duplicate lane zero consistent. Raw-packet validation may reject this
    // while calculating context capacity at load time or in the ICU decoder;
    // neither boundary may silently accept or re-encode the malformed bits.
    auto corruptHeader = std::find_if(corruptQueue->commands.begin(),
        corruptQueue->commands.end(),
        ftlpu::software::runtime::is_fu_3d_raw_packet_header);
    require(corruptHeader != corruptQueue->commands.end(),
        "missing raw MEM packet header for corruption test");
    corruptHeader->words[0] ^= std::uint32_t {1} << 7;
    corruptHeader->command = corruptHeader->words[0];
    bool decoderRejected = false;
    try {
        InstructionControlUnit corruptIcu;
        ftlpu::software::runtime::load_queue_programs_into_icu(
            corrupted.queues, corruptIcu,
            corrupted.hardware.mxms_per_hemisphere);
        for (std::size_t cycle = 0; cycle < 8; ++cycle)
            static_cast<void>(
                corruptIcu.mem_iq(firstWeight.queue).tick());
    } catch (const std::exception& exception) {
        decoderRejected = std::string(exception.what())
            .find("invalid word-0 header") != std::string::npos;
    }
    require(decoderRejected,
        "malformed raw packet was not rejected by load-time or ICU validation");

    // These are physical queue issue cycles. Adding each source slice's route
    // latency must recover the matching LOAD/COMPUTE consumer anchor without
    // enumerating the native loop points.
    std::size_t commandIndex = 0;
    for (const auto& hemisphere : placement.hemispheres) {
        for (const auto& region : hemisphere.weight_regions) {
            const auto loadAnchor = static_cast<std::size_t>(
                timeline.mxm_load_start_cycle
                + region.first_pair * timeline.pair_cycle_stride);
            for (std::size_t lane = 0;
                 lane < hw::kMxmInt8WeightStreamsPerCycle; ++lane) {
                const auto slice = static_cast<std::size_t>(
                    region.first_slice + static_cast<std::int64_t>(lane));
                const auto& command = program.mem_commands[commandIndex++];
                require(command.cycle
                            + static_cast<std::size_t>(
                                routeLatencies.mem_to_mxm_weight_cycles[slice])
                        == loadAnchor,
                    "weight MEM issue did not align at its LOAD_3D consumer");
            }
        }
    }
    for (const auto& hemisphere : placement.hemispheres) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t byte = 0; byte < sizeof(std::uint16_t); ++byte) {
                const auto slice = static_cast<std::size_t>(
                    hemisphere.activation_slices[2 * lane + byte]);
                const auto& command = program.mem_commands[commandIndex++];
                require(command.cycle
                            + static_cast<std::size_t>(routeLatencies
                                    .mem_to_mxm_activation_cycles[slice])
                        == static_cast<std::size_t>(
                            timeline.mxm_compute_start_cycle)
                            + lane,
                    "activation MEM issue did not align at its COMPUTE_3D row");
            }
        }
    }
    for (const auto& hemisphere : placement.hemispheres) {
        for (std::size_t byte = 0; byte < sizeof(std::uint16_t); ++byte) {
            const auto slice = static_cast<std::size_t>(
                hemisphere.result_slices[byte]);
            const auto& command = program.mem_commands[commandIndex++];
            require(command.cycle
                        == static_cast<std::size_t>(
                            timeline.mxm_result_start_cycle
                            + routeLatencies.mxm_result_to_mem_cycles[slice]),
                "result MEM issue did not include its MXM-to-MEM latency");
        }
    }
    require(commandIndex == program.mem_commands.size(),
        "route-alignment checks did not cover every MEM coarse command");

    require(firstWeight.queue == 73
            && firstWeight.cycle == 26
            && firstWeight.instruction.stream == StreamId::East(8).packed()
            && firstWeight.instruction.loop.start_cycle == 0
            && firstWeight.instruction.loop.counts
                == std::array<std::size_t, 3> {4, 48, 42}
            && firstWeight.instruction.loop.cycle_strides
                == std::array<std::size_t, 3> {1, 32, 1536},
        "first Qwen weight Read3D has the wrong physical descriptor");
    require(detail::mem_icu_address_3d(firstWeight.instruction,
                IcuCoordinate3D {{3, 47, 41}})
            == 8063,
        "blocked Qwen weight address does not reach its high boundary");

    const auto& finalEastWeightRegion = program.mem_commands[31];
    require(finalEastWeightRegion.queue == 102
            && finalEastWeightRegion.cycle == 32 + 126 * 1536 - 3
            && finalEastWeightRegion.instruction.loop.start_cycle == 0
            && finalEastWeightRegion.instruction.loop.counts[2] == 14
            && detail::mem_icu_address_3d(
                finalEastWeightRegion.instruction,
                IcuCoordinate3D {{3, 47, 13}}) == 2687,
        "tail Qwen weight region was not lowered directly");
    require(program.mem_commands[32].queue == 177,
        "west-hemisphere MEM queue mapping is wrong");

    require(firstActivation.queue == 1
            && firstActivation.cycle == 49
            && firstActivation.instruction.stream
                == StreamId::East(16).packed()
            && firstActivation.instruction.loop.start_cycle == 0
            && firstActivation.instruction.loop.counts
                == std::array<std::size_t, 3> {4, 48, 140}
            && firstActivation.instruction.loop.cycle_strides
                == std::array<std::size_t, 3> {8, 32, 1536}
            && detail::mem_icu_address_3d(firstActivation.instruction,
                IcuCoordinate3D {{3, 47, 139}}) == 191,
        "distributed activation Read3D has the wrong closed-form domain");

    require(firstResult.queue == 16
            && firstResult.cycle == 1597
            && firstResult.instruction.stream
                == StreamId::West(12).packed()
            && firstResult.instruction.loop.start_cycle == 0
            && firstResult.instruction.loop.counts
                == std::array<std::size_t, 3> {32, 140, 1}
            && detail::mem_icu_address_3d(firstResult.instruction,
                IcuCoordinate3D {{31, 139, 0}}) == 4479,
        "Qwen result Write3D has the wrong physical descriptor");

    const auto& load = program.mxm_load_commands.front().instruction;
    require(program.mxm_load_commands.front().queue == 0
            && program.mxm_load_commands.back().queue == 1
            && program.mxm_load_commands.front().cycle
                == static_cast<std::size_t>(timeline.mxm_load_start_cycle)
            && program.mxm_dequant_commands.front().queue == 0
            && program.mxm_dequant_commands.back().queue == 1
            && program.mxm_dequant_commands.front().cycle
                == static_cast<std::size_t>(timeline.mxm_dequant_start_cycle)
            && program.mxm_compute_commands.front().queue == 0
            && program.mxm_compute_commands.back().queue == 1
            && program.mxm_compute_commands.front().cycle
                == static_cast<std::size_t>(timeline.mxm_compute_start_cycle)
            && load.loop.start_cycle == 0
            && load.loop.counts
                == std::array<std::size_t, 3> {4, 48, 140}
            && load.weight_buffer_mode
                == MxmIcuBufferMode::ToggleDimension1
            && load.weight_stream_base == 8,
        "MXM Load3D descriptor is wrong");
    const auto loadK0 = detail::expand_mxm_load_icu_instruction(
        load, IcuCoordinate3D {{3, 0, 0}});
    const auto loadK1 = detail::expand_mxm_load_icu_instruction(
        load, IcuCoordinate3D {{3, 1, 0}});
    require(loadK0.weight_column == 3 && loadK0.weight_buffer == 0
            && loadK1.weight_column == 3 && loadK1.weight_buffer == 1,
        "Load3D does not ping-pong the K-reduction weight buffer");
    require(program.mxm_dequant_commands.front()
                .instruction.instruction.scale_bf16 == 0x3c00,
        "Dequant3D did not preserve the Up scale bits");

    const auto& compute = program.mxm_compute_commands.front().instruction;
    require(compute.loop.counts
                == std::array<std::size_t, 3> {32, 48, 140}
            && compute.activation_stream_base == 16
            && compute.result_stream_base == 12
            && compute.accumulator_address_base == 32
            && compute.terminal_dimension == 1,
        "MXM Compute3D descriptor is wrong");
    const auto intermediate = detail::expand_mxm_compute_icu_instruction(
        compute, IcuCoordinate3D {{0, 46, 0}});
    const auto terminal = detail::expand_mxm_compute_icu_instruction(
        compute, IcuCoordinate3D {{0, 47, 0}});
    require(intermediate.accumulator_destination
                == MxmAccumulatorDestination::Sram
            && !intermediate.accumulator_clear
            && intermediate.accumulator_output_format
                == MxmAccumulatorOutputFormat::Float32
            && terminal.accumulator_destination
                == MxmAccumulatorDestination::Stream
            && terminal.accumulator_clear
            && terminal.accumulator_output_format
                == MxmAccumulatorOutputFormat::BFloat16,
        "Compute3D terminal K iteration does not drain and clear");

    std::size_t weightPoints = 0;
    std::size_t activationPoints = 0;
    std::size_t resultPoints = 0;
    for (std::size_t index = 0; index < program.mem_commands.size(); ++index) {
        const auto count = points(program.mem_commands[index].instruction.loop);
        if (index < 64) weightPoints += count;
        else if (index < 96) activationPoints += count;
        else resultPoints += count;
    }
    std::size_t loadPoints = 0;
    for (const auto& command : program.mxm_load_commands)
        loadPoints += points(command.instruction.loop);
    std::size_t dequantPoints = 0;
    for (const auto& command : program.mxm_dequant_commands)
        dequantPoints += points(command.instruction.loop);
    std::size_t computePoints = 0;
    for (const auto& command : program.mxm_compute_commands)
        computePoints += points(command.instruction.loop);
    require(weightPoints == 430080 && activationPoints == 860160
            && resultPoints == 17920 && loadPoints == 53760
            && dequantPoints == 53760 && computePoints == 430080,
        "coarse commands do not cover the complete Qwen Up issue domain");

    // Put the two bytes of one logical activation lane in different route
    // groups. Their independently addressed MEM ICUs must receive different
    // starts while both values still reach the same COMPUTE_3D row.
    auto byteSkewPlacement = placement;
    std::swap(byteSkewPlacement.hemispheres[0].activation_slices[1],
        byteSkewPlacement.hemispheres[0].activation_slices[4]);
    auto byteSkew = lowerFfnUpToFu3D(shape, byteSkewPlacement, timeline,
        routeLatencies, 0x3c00, &error);
    require(mlir::succeeded(byteSkew),
        "direct lowering rejected a valid per-byte route skew: " + error);
    const auto& byte0 = byteSkew->mem_commands[64];
    const auto& byte1 = byteSkew->mem_commands[65];
    const auto byte0Slice = static_cast<std::size_t>(
        byteSkewPlacement.hemispheres[0].activation_slices[0]);
    const auto byte1Slice = static_cast<std::size_t>(
        byteSkewPlacement.hemispheres[0].activation_slices[1]);
    require(byte0.cycle != byte1.cycle
            && byte0.instruction.loop.start_cycle == 0
            && byte1.instruction.loop.start_cycle == 0
            && byte0.cycle
                    + static_cast<std::size_t>(routeLatencies
                            .mem_to_mxm_activation_cycles[byte0Slice])
                == static_cast<std::size_t>(
                    timeline.mxm_compute_start_cycle)
            && byte1.cycle
                    + static_cast<std::size_t>(routeLatencies
                            .mem_to_mxm_activation_cycles[byte1Slice])
                == static_cast<std::size_t>(
                    timeline.mxm_compute_start_cycle),
        "two activation bytes did not independently compensate route skew");

    auto invalidPlacement = placement;
    ++invalidPlacement.hemispheres[1].weight_regions[1].first_pair;
    auto invalid = lowerFfnUpToFu3D(
        shape, invalidPlacement, timeline, routeLatencies, 0x3c00, &error);
    require(mlir::failed(invalid)
            && error.find("contiguously") != std::string::npos,
        "direct lowering accepted a gap in physical weight placement");

    auto unpairedTimeline = timeline;
    ++unpairedTimeline.mxm_dequant_start_cycle;
    invalid = lowerFfnUpToFu3D(
        shape, placement, unpairedTimeline, routeLatencies, 0x3c00, &error);
    require(mlir::failed(invalid)
            && error.find("paired") != std::string::npos,
        "direct lowering accepted unpaired MXM load/dequant loops");

    const auto requireWeightStrideOverflow = [&](std::int64_t groupSize,
                                                 const char* expectedError) {
        auto overflowPlacement = placement;
        overflowPlacement.weight_outer_group_size = groupSize;
        const auto overflowed = lowerFfnUpToFu3D(shape, overflowPlacement,
            timeline, routeLatencies, 0x3c00, &error);
        require(mlir::failed(overflowed) && error == expectedError,
            std::string("direct lowering did not reject ") + expectedError
                + ": " + error);
    };
    requireWeightStrideOverflow(std::int64_t {1} << 62,
        "weight middle address stride overflows size_t");
    requireWeightStrideOverflow(std::int64_t {1} << 61,
        "weight middle address stride overflows int64");
    requireWeightStrideOverflow(std::int64_t {1} << 58,
        "weight outer-group address stride overflows size_t");
    requireWeightStrideOverflow(std::int64_t {1} << 56,
        "weight outer-group address stride overflows int64");

    std::cout << "ffn_up_3d_lowering_test passed: coarse=106"
              << " physical_words=" << physicalWords
              << " expanded_native_issues="
              << weightPoints + activationPoints + resultPoints
                    + loadPoints + dequantPoints + computePoints
              << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "ffn_up_3d_lowering_test failed: "
              << exception.what() << '\n';
    return 1;
}
