#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

BinaryBinding make_weight_binding()
{
    BinaryBinding binding;
    binding.index = 0;
    binding.access = BindingAccess::Input;
    binding.element_type = BindingElementType::I8;
    binding.layout = BindingLayout::W8A16AttentionWeightStriped;
    binding.byte_size = 32 * 128;
    binding.base_row = 0;
    binding.instruction_count = 8;
    binding.address_stride = 1;
    binding.shape = {32, 128};
    binding.slices = {20, 21, 22, 23, 24, 25, 26, 27};
    binding.role = "weight";
    binding.name = "q_proj.weight";
    binding.hemisphere_mask = 3;
    binding.bank = 0;
    binding.paged_weight = true;
    binding.page_count = 1;
    binding.page_rows = 8;
    binding.page_granularity = 1;
    binding.page_role_group_count = 1;
    binding.page_items_per_slice_group = 1;
    binding.page_bank_count = 2;
    binding.page_storage_slices = binding.slices;
    return binding;
}

} // namespace

int main()
try {
    BinaryProgram program;
    program.bindings.push_back(make_weight_binding());
    program.weight_page_uses.push_back({0, 0, 0, 2, 3});
    program.max_cycle = 3;

    TspSliceSystem system;
    std::size_t physicalTicks = 0;
    CModelRuntime runtime(system,
        [&](TspSliceSystem::LogSinks sinks) {
            system.tick(sinks);
            ++physicalTicks;
        });
    runtime.enable_execution_trace();
    runtime.load(program);
    runtime.set_weight_page_residency_checker(
        [&](const BinaryWeightPageUse&) { return physicalTicks >= 7; });
    runtime.run_cycles(4);

    if (runtime.logical_cycles() != 4 || runtime.physical_cycles() != 9
        || physicalTicks != 9)
        throw std::runtime_error(
            "page-ready wait did not separate logical and physical cycles");
    if (!system.icu().program_issue_enabled())
        throw std::runtime_error(
            "runtime left the compute ICUs held after page release");

    const auto path = std::filesystem::temp_directory_path()
        / "ftlpu_runtime_page_ready_sync.csv";
    runtime.write_execution_trace_csv(path);
    std::ifstream input(path);
    const std::string trace {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (trace.find("2,7,\"ICU.PageReadyWait\"") == std::string::npos
        || trace.find("source=runtime issues=5") == std::string::npos)
        throw std::runtime_error(
            "runtime trace did not contain the actual page-ready stall");
    input.close();

    RuntimeExecutionTrace segmented;
    segmented.begin_segment(program, 100, false);
    segmented.record_interval(0, 1, "Session.Invocation", "index=0");
    segmented.begin_segment(program, 200, true);
    segmented.record_interval(0, 1, "Session.Invocation", "index=1");
    segmented.write_csv(path);
    std::ifstream segmentedInput(path);
    const std::string segmentedTrace {
        std::istreambuf_iterator<char>(segmentedInput),
        std::istreambuf_iterator<char>()};
    if (segmentedTrace.find("100,101,\"Session.Invocation\"")
            == std::string::npos
        || segmentedTrace.find("200,201,\"Session.Invocation\"")
            == std::string::npos)
        throw std::runtime_error(
            "runtime trace did not append cycle-offset segments");
    segmentedInput.close();

    RuntimeExecutionTrace repeated;
    repeated.record_interval(10, 11, "MEM.E.Read", "slice=0 bank=0");
    repeated.record_interval(18, 19, "MEM.E.Read", "slice=0 bank=0");
    repeated.record_interval(26, 27, "MEM.E.Read", "slice=0 bank=0");
    repeated.record_interval(50, 51, "MEM.E.Read", "slice=0 bank=0");
    repeated.record_interval(58, 59, "MEM.E.Read", "slice=0 bank=0");
    repeated.record_interval(66, 67, "MEM.E.Read", "slice=0 bank=0");
    repeated.write_csv(path);
    std::ifstream repeatedInput(path);
    const std::string repeatedTrace {
        std::istreambuf_iterator<char>(repeatedInput),
        std::istreambuf_iterator<char>()};
    if (repeatedTrace.find("\"repeat2d\",3,8,0,2,40")
            == std::string::npos
        || repeatedTrace.find("source=runtime issues=6")
            == std::string::npos)
        throw std::runtime_error(
            "runtime trace did not compact observed periodic issues");
    repeatedInput.close();
    std::filesystem::remove(path);

    std::cout << "runtime_page_ready_sync_test passed logical=4 physical=9 "
                 "wait=5\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "runtime_page_ready_sync_test failed: "
              << error.what() << '\n';
    return 1;
}
