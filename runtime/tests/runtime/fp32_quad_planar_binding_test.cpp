#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/target_abi.hpp"

#include "ftlpu/system/tsp_slice_system.hpp"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint8_t> makeValues(std::size_t count)
{
    std::vector<std::uint8_t> bytes(count * sizeof(float));
    for (std::size_t index = 0; index < count; ++index) {
        const float value = static_cast<float>(static_cast<int>(index % 29) - 14)
            * 0.125f + static_cast<float>(index / 29) * 0.03125f;
        const auto bits = std::bit_cast<std::uint32_t>(value);
        for (std::size_t byte = 0; byte < sizeof(float); ++byte)
            bytes[index * sizeof(float) + byte] =
                static_cast<std::uint8_t>(bits >> (8 * byte));
    }
    return bytes;
}

} // namespace

int main()
try {
    using namespace ftlpu::software::runtime;
    constexpr std::uint64_t rows = 3;
    constexpr std::uint64_t columns = 65;

    BinaryProgram program;
    program.target_abi = executable_target_abi(program.hardware);
    BinaryBinding input;
    input.index = 0;
    input.access = BindingAccess::Input;
    input.element_type = BindingElementType::F32;
    input.layout = BindingLayout::Fp32QuadPlanar;
    input.byte_size = rows * columns * sizeof(float);
    input.base_row = 17;
    input.instruction_count = 9;
    input.address_stride = 1;
    input.shape = {rows, columns};
    input.slices = {10, 11, 12, 13};
    input.hemisphere_mask = 2;
    input.role = "state";
    input.name = "recurrent_state.input";
    program.bindings.push_back(input);

    BinaryBinding output = input;
    output.access = BindingAccess::Output;
    output.name = "recurrent_state.output";
    program.bindings.push_back(output);

    auto* system = new ftlpu::TspSliceSystem();
    CModelRuntime runtime(*system);
    runtime.load(program);
    const auto expected = makeValues(rows * columns);
    runtime.upload_input(0, expected);
    const auto actual = runtime.download_output(0);
    if (actual != expected)
        throw std::logic_error(
            "FP32 quad-planar binding did not round-trip all four bytes");

    std::cout << "fp32_quad_planar_binding_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "fp32_quad_planar_binding_test failed: "
              << error.what() << '\n';
    return 1;
}
