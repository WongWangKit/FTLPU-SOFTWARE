#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/target_abi.hpp"

#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    input.element_type = BindingElementType::BF16;
    input.layout = BindingLayout::Fp16PairPlanar;
    input.byte_size = rows * columns * 2;
    input.base_row = 17;
    input.instruction_count = 9;
    input.address_stride = 1;
    input.shape = {rows, columns};
    input.slices = {10, 11};
    input.hemisphere_mask = 1;
    input.role = "activation";
    input.name = "pair_planar.input";
    program.bindings.push_back(input);

    BinaryBinding output = input;
    output.access = BindingAccess::Output;
    output.name = "pair_planar.output";
    program.bindings.push_back(output);

    std::vector<std::uint8_t> expected(input.byte_size);
    for (std::size_t index = 0; index < expected.size(); ++index)
        expected[index] = static_cast<std::uint8_t>((index * 37 + 11) & 0xff);

    auto* system = new ftlpu::TspSliceSystem();
    CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, expected);
    const auto actual = runtime.download_output(0);
    if (actual != expected)
        throw std::logic_error(
            "two-slice FP16 pair-planar binding did not round-trip 32-column blocks");

    std::cout << "fp16_pair_planar_binding_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "fp16_pair_planar_binding_test failed: "
              << error.what() << '\n';
    return 1;
}
