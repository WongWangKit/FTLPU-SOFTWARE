#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"
#include "ftlpu/core/bf16.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;
namespace {
void require(bool value, const char* message)
{ if (!value) throw std::runtime_error(message); }
template <typename F> void rejects(F action)
{
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("unsupported format was accepted");
}
rt::QueueCommand macro(MemInstruction instruction, std::size_t start,
    std::size_t count = 4)
{
    const auto encoded = isa::encode_mem_instruction(instruction);
    rt::QueueCommand command{static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mem, 2,
        {static_cast<std::uint32_t>(encoded),static_cast<std::uint32_t>(encoded >> 32),0,0}};
    return rt::encode_macro_schedule_command(std::move(command),
        IcuMacroSchedule{start,count,1,1,1,1,0,IcuInductionTarget::MemAddress});
}
rt::BinaryProgram copy_program()
{
    // Slice 0's westward read reaches the passive bridge after one cycle;
    // the eastward destination write follows one further cycle later.
    rt::BinaryProgram program;
    const auto src = InstructionControlUnit::mem_queue(Hemisphere::East, 0, 0);
    const auto dst = InstructionControlUnit::mem_queue(Hemisphere::West, 0, 0);
    program.queues.push_back({rt::QueueKind::Mem,src,
        {macro(MemInstruction::Read(100,StreamId::West(0)),4)}});
    program.queues.push_back({rt::QueueKind::Mem,dst,
        {macro(MemInstruction::Write(200,StreamId::East(0)),6)}});
    program.max_cycle = 16;
    rt::BinaryBinding input;
    input.index = 0; input.access = rt::BindingAccess::Input;
    input.element_type = rt::BindingElementType::I8;
    input.layout = rt::BindingLayout::Vector;
    input.shape = {4,32}; input.byte_size = 128;
    input.base_row = 100; input.address_stride = 1;
    input.slices = {0}; input.hemisphere_mask = 1;
    program.bindings.push_back(input);
    return program;
}
std::vector<std::uint8_t> output(TspSliceSystem& system)
{
    std::vector<std::uint8_t> bytes;
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t b = 0; b < 32; ++b)
            bytes.push_back(system.read_mem_sram_lane_byte(Hemisphere::West,
                0,0,b/hw::kLanesPerTile,200+row,b%hw::kLanesPerTile));
    return bytes;
}

rt::QueueCommand mxm_macro(MxmControlInstruction instruction, std::size_t start,
    std::size_t count, std::int64_t stride, IcuInductionTarget target)
{
    const auto encoded = isa::encode_mxm_instruction(instruction);
    rt::QueueCommand command{static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::Mxm,2,
        {static_cast<std::uint32_t>(encoded),static_cast<std::uint32_t>(encoded >> 32),0,0}};
    return rt::encode_macro_schedule_command(std::move(command),
        IcuMacroSchedule{start,count,1,stride,1,1,0,target});
}

void numerical_mxm_regression()
{
    rt::BinaryProgram program;
    const auto latency = [](std::size_t slice) {
        return hw::kMemGroups + hw::kC2cToSxmStreamRegisterColumns + 1
            - slice / hw::kMemSlicesPerGroup;
    };
    for (std::size_t slice = 0; slice < 16; ++slice)
        program.queues.push_back({rt::QueueKind::Mem,
            InstructionControlUnit::mem_queue(Hemisphere::East,slice,0),
            {macro(MemInstruction::Read(64,StreamId::East(slice)),20-latency(slice),4)}});
    for (std::size_t slice = 0; slice < hw::kMxmInt8LoadStreamsPerCycle; ++slice)
        program.queues[slice].commands.push_back(
            macro(MemInstruction::Read(90,StreamId::East(slice)),35-latency(slice),1));
    for (std::size_t byte = 0; byte < 2; ++byte)
        program.queues.push_back({rt::QueueKind::Mem,
            InstructionControlUnit::mem_queue(Hemisphere::East,50+byte,0),
            {macro(MemInstruction::Read(80,StreamId::East(byte)),30-latency(50+byte),1)}});
    program.queues.push_back({rt::QueueKind::MxmLoad,0,
        {mxm_macro(MxmControlInstruction::IWDirect16(0,0),20,4,1,
            IcuInductionTarget::MxmWeightColumn),
         mxm_macro(MxmControlInstruction::IW(1,0),35,1,0,IcuInductionTarget::None)}});
    program.queues.push_back({rt::QueueKind::MxmCompute,0,
        {mxm_macro(MxmControlInstruction::Compute(0,0,0,128,1,
            MxmAccumulatorDestination::Sram,MxmDataFormat::BFloat16),30,1,0,
            IcuInductionTarget::None)}});
    rt::QueueCommand scale{static_cast<isa::EncodedIcuCommand>(isa::IcuCommandOpcode::Instruction),
        rt::InstructionKind::MxmDequant,1,
        {isa::encode_mxm_dequant_instruction(MxmDequantInstruction::Scale(0.125f)),0,0,0}};
    program.queues.push_back({rt::QueueKind::MxmDequant,0,
        {rt::encode_macro_schedule_command(std::move(scale),IcuMacroSchedule{35})}});
    program.max_cycle = 48;
    const auto weight = [](std::size_t row, std::size_t column) {
        return static_cast<float>(static_cast<int>((row*5+column*3)%23)-11)/16.0f;
    };
    const auto activation = [](std::size_t row) {
        return static_cast<float>(static_cast<int>((row*7)%17)-8)/8.0f;
    };
    const auto initialize = [&](TspSliceSystem& system) {
        for (std::size_t row = 0; row < hw::kMxmRows; ++row) {
            for (std::size_t col = 0; col < hw::kMxmColumns; ++col) {
                const auto bits = Bf16::from_float(weight(row,col)).bits();
                for (std::size_t byte = 0; byte < 2; ++byte)
                    system.initialize_mem_sram_lane_byte(Hemisphere::East,
                        (col%hw::kMxmSupercellColumns)*2+byte,0,
                        row/hw::kLanesPerTile,64+col/hw::kMxmSupercellColumns,
                        row%hw::kLanesPerTile,static_cast<std::uint8_t>(bits>>(byte*8)));
            }
            const auto bits = Bf16::from_float(activation(row)).bits();
            for (std::size_t byte = 0; byte < 2; ++byte)
                system.initialize_mem_sram_lane_byte(Hemisphere::East,50+byte,0,
                    row/hw::kLanesPerTile,80,row%hw::kLanesPerTile,
                    static_cast<std::uint8_t>(bits>>(byte*8)));
            for (std::size_t col = 0; col < hw::kMxmSupercellColumns; ++col)
                system.initialize_mem_sram_lane_byte(Hemisphere::East,col,0,
                    row/hw::kLanesPerTile,90,row%hw::kLanesPerTile,
                    static_cast<std::uint8_t>(static_cast<int>((row*13+col*7)%31)-15));
        }
    };
    auto direct = std::make_unique<TspSliceSystem>();
    rt::CModelRuntime reference(*direct);
    reference.load_direct(program); initialize(*direct); reference.run_cycles(49);
    auto system = std::make_unique<C2cDmaSystem>();
    rt::CModelRuntime timed(*system);
    timed.load_timed(program); initialize(system->chip()); timed.run_cycles(49);
    for (std::size_t col = 0; col < hw::kMxmColumns; ++col) {
        float expected = 0;
        for (std::size_t row = 0; row < hw::kMxmRows; ++row)
            expected += Bf16::from_float(weight(row,col)).to_float()
                * Bf16::from_float(activation(row)).to_float();
        const auto actual = system->chip().mxm_unit(0).accumulator().value(128,col);
        require(std::fabs(actual-expected) <= 1.0e-5f
            && actual == direct->mxm_unit(0).accumulator().value(128,col),
            "timed MEM/MXM BF16 matrix-vector numerical regression failed");
    }
    require(timed.program_load_statistics().images.size() == 21,
        "numerical program did not load all MEM/MXM/dequant queues");
    for (std::size_t row = 0; row < hw::kMxmRows; ++row)
        for (std::size_t col = 0; col < hw::kMxmSupercellColumns; ++col) {
            const auto expected = Bf16::from_float(
                static_cast<float>(static_cast<int>((row*13+col*7)%31)-15)*0.125f).bits();
            require(system->chip().mxm_unit(0).array().cell(row/hw::kLanesPerTile,0)
                    .weight_bits(1,row%hw::kLanesPerTile,col) == expected,
                "timed MXM INT8 dequant numerical regression failed");
        }
}
}

int main() try
{
    numerical_mxm_regression();
    auto program = copy_program();
    const auto images = rt::prepare_program_images(program.queues,0x100000);
    require(images.size() == 2 && images[0].bytes.size() % 12 == 0,
        "runtime image preparation lost physical word boundaries");
    std::vector<std::uint8_t> input(128);
    for (std::size_t b = 0; b < input.size(); ++b) input[b] = static_cast<std::uint8_t>(b * 37 + 11);

    auto direct = std::make_unique<TspSliceSystem>();
    rt::CModelRuntime legacy(*direct);
    legacy.load_direct(program); legacy.upload_input(0,input); legacy.run_cycles(17);
    require(output(*direct) == input, "reference copy program produced incorrect values");

    auto system = std::make_unique<C2cDmaSystem>();
    rt::CModelRuntime runtime(*system,[&](TspSliceSystem::LogSinks sinks) { system->tick(sinks); });
    runtime.load_timed(program);
    require(system->cycle() == 0 && !runtime.all_images_committed(),
        "submission preloaded i-MEM or advanced time synchronously");
    // Preparation and prefetch may finish while required input is still absent.
    for (std::size_t timeout = 0; !system->program_loader().ready_to_start(); ++timeout) {
        require(timeout < 1000,"runtime loader did not prime"); system->tick();
    }
    require(!runtime.ready_to_start(), "missing input did not gate start");
    runtime.upload_input(0,input);
    std::size_t weightWaitTicks = 0;
    // A first-batch weight dependency must also gate logical cycle zero.
    auto weightProgram = program;
    // Exercise the existing page checker independently of input upload: its
    // ready-cycle zero use references the physical input binding.
    weightProgram.weight_page_uses.push_back({0,0,0,0,1});
    runtime.run_cycles(17);
    require(output(system->chip()) == input, "timed copy program differs from reference");
    const auto first = runtime.program_load_statistics();
    require(first.program_start && *first.program_start > *first.frontend_ready,
        "program started before frontend was ready");
    require(runtime.logical_cycles() == 17 && runtime.physical_cycles() == system->cycle(),
        "runtime lost physical load/prefetch cycles");
    const auto physical = system->cycle();
    runtime.load_timed(weightProgram,0x200000);
    runtime.upload_input(0,input);
    runtime.set_weight_page_residency_checker([&](const auto&) { return ++weightWaitTicks > 100; });
    runtime.run_cycles(17);
    require(runtime.program_load_statistics().request_accepted == physical
        && *runtime.program_load_statistics().program_start >= physical+100,
        "kernel switch reset physical time or bypassed initial weights");
    require(output(system->chip()) == input, "second timed kernel changed numerical output");

    auto invalid = program;
    invalid.queues.push_back({rt::QueueKind::Vxm,0,{rt::QueueCommand{}}});
    const auto before = system->cycle();
    rejects([&] { runtime.load_timed(invalid); });
    require(system->cycle() == before && system->program_loader().started(),
        "unsupported queue mutated active program");
    auto shortProgram = program;
    shortProgram.queues.resize(1);
    runtime.load_timed(shortProgram,0x300000);
    runtime.upload_input(0,input); runtime.run_cycles(17);
    require(runtime.program_load_statistics().images.size() == 1,
        "previous queue notifications leaked into new kernel");
    std::cout << "program_load_runtime_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "program_load_runtime_test failed: " << ex.what() << '\n'; return 1;
}
