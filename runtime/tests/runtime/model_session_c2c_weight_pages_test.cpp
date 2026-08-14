#include "ftlpu/software/runtime/model_session.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

BinaryProgram make_program(std::uint16_t bank)
{
    BinaryProgram program;
    BinaryBinding weight;
    weight.index = 0;
    weight.access = BindingAccess::Input;
    weight.element_type = BindingElementType::I8;
    weight.layout = BindingLayout::Vector;
    weight.byte_size = hw::kPhysicalVectorBytes;
    weight.base_row = 10;
    weight.instruction_count = 1;
    weight.address_stride = 1;
    weight.shape = {hw::kPhysicalVectorBytes};
    weight.slices = {16};
    weight.role = "weight";
    weight.name = "weight.packed";
    weight.hemisphere_mask = 2;
    weight.bank = bank;
    program.bindings.push_back(std::move(weight));
    program.max_cycle = 96;
    return program;
}

std::vector<std::uint8_t> page_bytes(std::uint8_t base)
{
    std::vector<std::uint8_t> bytes(hw::kPhysicalVectorBytes);
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::uint8_t>(base + index);
    return bytes;
}

} // namespace

int main()
try {
    ModelPackage package;
    package.model_name = "two-layer-c2c-page-session";
    package.architecture = "Qwen2ForCausalLM";
    package.executables = {
        {"layer.bank0", make_program(0), {}},
        {"layer.bank1", make_program(1), {}},
    };
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        const std::string name = "layers." + std::to_string(layer)
            + ".weights.packed";
        package.tensors.push_back(ModelTensor {
            name, BindingElementType::I8, {hw::kPhysicalVectorBytes},
            page_bytes(static_cast<std::uint8_t>(0x20 + layer * 0x40)),
            ModelTensorEncoding::TargetPackedSramVectors});
        ModelWeightPage page;
        page.layer = layer;
        page.bank = static_cast<std::uint16_t>(layer);
        page.tensors = {name};
        page.segments.push_back(ModelWeightPage::Segment {
            name, 0, 1, 16, 10, 1, 5});
        package.weight_pages.push_back(std::move(page));
        package.invocations.push_back(ModelInvocation {
            "layers." + std::to_string(layer), layer, {{0, name}}, {}, {},
            layer});
    }

    C2cDmaSystem system(Ddr4Config {8, 2, 2, 4});
    ModelSession session(system);
    session.load(std::move(package));
    if (session.memory_plan().invocations[0].inputs[0].transfer
            != SessionTransferKind::WeightPage
        || session.memory_plan().invocations[1].inputs[0].transfer
            != SessionTransferKind::WeightPage)
        throw std::runtime_error(
            "session planner did not classify packed inputs as weight pages: "
            + std::to_string(static_cast<int>(session.memory_plan()
                  .invocations[0].inputs[0].transfer))
            + "," + std::to_string(static_cast<int>(session.memory_plan()
                  .invocations[1].inputs[0].transfer))
            + " page=" + std::to_string(
                  session.package().invocations[0].weight_page)
            + "," + std::to_string(
                  session.package().invocations[1].weight_page)
            + " input=" + session.package().invocations[0].inputs[0].value
            + " page_tensor="
            + session.package().weight_pages[0].tensors[0]
            + "," + session.package().weight_pages[1].tensors[0]);
    session.run_invocation(0, 0);

    for (std::size_t bank = 0; bank < 2; ++bank) {
        const auto expected = page_bytes(
            static_cast<std::uint8_t>(0x20 + bank * 0x40));
        for (std::size_t byte = 0; byte < expected.size(); ++byte) {
            const auto actual = system.chip().read_mem_sram_lane_byte(
                Hemisphere::West, 16, bank,
                byte / hw::kLanesPerTile, 10,
                byte % hw::kLanesPerTile);
            if (actual != expected[byte])
                throw std::runtime_error(
                    "session C2C page data mismatch");
        }
    }
    session.run_invocation(1, 0);
    if (session.stats().weight_page_prefetches != 2
        || session.stats().weight_page_prefetch_bytes
            != 2 * hw::kPhysicalVectorBytes
        || session.stats().weight_page_wait_cycles == 0)
        throw std::runtime_error(
            "session did not report C2C page prefetch statistics");
    if (!session.memory_plan().resident_tensors.empty())
        throw std::runtime_error(
            "C2C pages were also allocated as resident weights");
    std::cout << "model_session_c2c_weight_pages_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "model_session_c2c_weight_pages_test failed: "
              << error.what() << '\n';
    return 1;
}
