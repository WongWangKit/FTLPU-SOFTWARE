#include "ftlpu/software/runtime/model_session.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    binding.page_role_group_base = 0;
    binding.page_role_group_count = 1;
    binding.page_items_per_slice_group = 1;
    binding.page_bank_count = 2;
    binding.page_storage_slices = binding.slices;
    return binding;
}

} // namespace

int main()
try {
    BinaryProgram preloadProgram;
    auto firstPreload = make_weight_binding();
    firstPreload.bank = 1;
    auto alternateBankPreload = make_weight_binding();
    alternateBankPreload.index = 1;
    alternateBankPreload.bank = 0;
    alternateBankPreload.slices = {28, 29, 30, 31, 32, 33, 34, 35};
    alternateBankPreload.page_storage_slices =
        alternateBankPreload.slices;
    preloadProgram.bindings = {firstPreload, alternateBankPreload};
    preloadProgram.weight_page_uses = {
        {0, 0, 1, 200, 220},
        {1, 0, 0, 400, 420},
    };
    const auto preloadPlans = plan_weight_prefetches(preloadProgram);
    if (preloadPlans.size() != 2 || !preloadPlans[0].pre_execution
        || !preloadPlans[1].pre_execution)
        throw std::runtime_error(
            "disjoint alternate-bank weight page was not preloaded");
    auto lowBandwidthHardware = preloadProgram.hardware;
    lowBandwidthHardware.ddr_peak_bandwidth_mbytes_per_second = 12800;
    const auto lowBandwidthPlans =
        plan_weight_prefetches(preloadProgram, lowBandwidthHardware);
    if (lowBandwidthPlans.size() != preloadPlans.size()
        || lowBandwidthPlans[0].transfer_end_cycle
            <= preloadPlans[0].transfer_end_cycle)
        throw std::runtime_error(
            "runtime DDR bandwidth did not retime weight prefetches");

    BinaryProgram overlappingProgram;
    auto resident = make_weight_binding();
    auto replacement = make_weight_binding();
    replacement.index = 1;
    overlappingProgram.bindings = {resident, replacement};
    overlappingProgram.weight_page_uses = {
        {0, 0, 0, 100, 320},
        {1, 0, 0, 1000, 1100},
    };
    const auto overlappingPlans =
        plan_weight_prefetches(overlappingProgram);
    if (overlappingPlans.size() != 2
        || !overlappingPlans[0].pre_execution
        || overlappingPlans[1].pre_execution
        || overlappingPlans[1].start_cycle != 320)
        throw std::runtime_error(
            "overlapping weight replacement did not launch at release");

    BinaryProgram program;
    program.bindings.push_back(make_weight_binding());
    program.weight_page_uses.push_back({0, 0, 0, 200, 220});
    program.max_cycle = 228;

    std::vector<std::uint8_t> logical(program.bindings[0].byte_size);
    for (std::size_t index = 0; index < logical.size(); ++index)
        logical[index] = static_cast<std::uint8_t>(index * 17 + 3);
    const PackedWeightImage expected = pack_weight_binding_page(
        program.bindings[0], 0, logical, program.hardware);

    ModelPackage package;
    package.model_name = "executable-c2c-weight";
    package.architecture = "test";
    package.tensors.push_back(ModelTensor {
        "q_proj.weight", BindingElementType::I8, {32, 128}, logical});
    package.executables.push_back({"projection", program, {}});
    package.invocations.push_back(ModelInvocation {
        "projection", 0, {{0, "q_proj.weight"}}, {}, {}});

    C2cDmaSystem system;
    ModelSession session(system);
    session.set_ddr_peak_bandwidth_mbytes_per_second(12800);
    session.load(std::move(package));
    if (system.ddr4().config().peak_bandwidth_bytes_per_second
        != 12'800'000'000ULL)
        throw std::runtime_error(
            "ModelSession did not apply the runtime DDR bandwidth");
    session.run_invocation(0, 0);

    if (session.stats().weight_page_prefetches != 1
        || session.stats().weight_page_prefetch_bytes != expected.data.size()
        || session.stats().weight_page_initial_wait_cycles == 0)
        throw std::runtime_error(
            "executable-local page did not run through the C2C pager");

    for (const PackedWeightSegment& segment : expected.segments) {
        for (std::uint32_t row = 0; row < segment.vector_count; ++row) {
            const std::size_t offset =
                static_cast<std::size_t>(segment.byte_offset)
                + static_cast<std::size_t>(row) * hw::kPhysicalVectorBytes;
            for (std::size_t column = 0;
                 column < hw::kPhysicalVectorBytes; ++column) {
                const auto actual = system.chip().read_mem_sram_lane_byte(
                    static_cast<Hemisphere>(segment.hemisphere),
                    segment.slice, 0, column / hw::kLanesPerTile,
                    segment.base_row + row, column % hw::kLanesPerTile);
                if (actual != expected.data[offset + column])
                    throw std::runtime_error(
                        "C2C page payload does not match packed weight SRAM");
            }
        }
    }

    std::cout << "model_session_executable_c2c_weight_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "model_session_executable_c2c_weight_test failed: "
              << error.what() << '\n';
    return 1;
}
