#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

constexpr std::size_t kRows = 32;
constexpr std::size_t kColumns = 1536;

BinaryBinding make_binding(BindingAccess access)
{
    BinaryBinding binding;
    binding.index = 0;
    binding.access = access;
    binding.element_type = BindingElementType::BF16;
    binding.layout = BindingLayout::Fp16MxmDistributed16;
    binding.byte_size = kRows * kColumns * sizeof(std::uint16_t);
    binding.base_row = 100;
    binding.instruction_count = 192;
    binding.address_stride = 1;
    binding.shape = {kRows, kColumns};
    for (std::uint16_t slice = 0; slice < 16; ++slice)
        binding.slices.push_back(slice);
    binding.role = access == BindingAccess::Input ? "activation" : "result";
    binding.name = access == BindingAccess::Input ? "input" : "output";
    binding.hemisphere_mask = 3;
    binding.bank = 0;
    return binding;
}

std::vector<std::uint8_t> make_input()
{
    std::vector<std::uint8_t> result(
        kRows * kColumns * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < kRows * kColumns; ++index) {
        const auto bits = Bf16::from_float(
            static_cast<float>(static_cast<int>(index % 37) - 18) / 8.0f)
                              .bits();
        result[index * 2] = static_cast<std::uint8_t>(bits);
        result[index * 2 + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    return result;
}

void test_planar_output(BindingElementType type, BindingLayout layout,
                       std::vector<std::uint64_t> shape,
                       std::vector<std::uint16_t> slices,
                       std::uint16_t hemisphereMask, std::int64_t baseRow)
{
    auto inputBinding = make_binding(BindingAccess::Input);
    inputBinding.element_type = type;
    inputBinding.layout = layout;
    inputBinding.shape = shape;
    inputBinding.slices = slices;
    inputBinding.hemisphere_mask = hemisphereMask;
    inputBinding.bank = 1;
    inputBinding.base_row = baseRow;
    inputBinding.instruction_count = shape[0] * (shape[1] / 32);
    const auto bytesPerElement = type == BindingElementType::F32 ? 4u : 2u;
    inputBinding.byte_size = shape[0] * shape[1] * bytesPerElement;
    auto outputBinding = inputBinding;
    outputBinding.access = BindingAccess::Output;
    outputBinding.name = "output";
    outputBinding.role = "result";
    BinaryProgram program;
    program.hardware.ddr_peak_bandwidth_mbytes_per_second = 51200;
    program.bindings = {inputBinding, outputBinding};
    ModelPackage package;
    package.model_name = "planar-output-roundtrip";
    package.architecture = "test";
    package.values = {{"input", type, shape, true, false},
                      {"output", type, shape, false, true}};
    package.executables.push_back({"identity", std::move(program), {}});
    package.invocations.push_back({"identity", 0, {{0,"input"}}, {{0,"output"}}, {}});
    std::vector<std::uint8_t> input(inputBinding.byte_size);
    for (std::size_t element = 0; element < input.size() / bytesPerElement; ++element) {
        const float value = static_cast<float>(static_cast<int>(element % 37) - 18) / 8.0f;
        if (type == BindingElementType::F32)
            std::memcpy(input.data() + element * 4, &value, 4);
        else {
            const auto bits = Bf16::from_float(value).bits();
            std::memcpy(input.data() + element * 2, &bits, 2);
        }
    }
    auto system = std::make_unique<C2cDmaSystem>();
    ModelSession session(*system);
    session.load(std::move(package));
    session.set_input("input", input);
    session.run();
    if (session.value("output") != input)
        throw std::runtime_error("multi-lane planar C2C output changed bytes");
}

} // namespace

int main()
try {
    BinaryProgram program;
    program.hardware.ddr_peak_bandwidth_mbytes_per_second = 51200;
    program.bindings = {
        make_binding(BindingAccess::Input),
        make_binding(BindingAccess::Output),
    };

    ModelPackage package;
    package.model_name = "c2c-io-roundtrip";
    package.architecture = "test";
    package.values = {
        {"input", BindingElementType::BF16, {kRows, kColumns}, true, false},
        {"output", BindingElementType::BF16, {kRows, kColumns}, false, true},
    };
    package.executables.push_back({"identity", std::move(program), {}});
    package.invocations.push_back(
        {"identity", 0, {{0, "input"}}, {{0, "output"}}, {}});

    C2cDmaSystem system;
    ModelSession session(system);
    session.set_ddr_peak_bandwidth_mbytes_per_second(25600);
    session.load(std::move(package));
    if (system.ddr4().config().peak_bandwidth_bytes_per_second
        != 25'600'000'000ULL)
        throw std::runtime_error(
            "ModelSession did not apply the runtime DDR bandwidth");
    const auto input = make_input();
    session.set_input("input", input);
    session.run();

    if (session.value("output") != input)
        throw std::runtime_error(
            "C2C input/output roundtrip changed the logical tensor");
    const auto& stats = session.stats();
    if (stats.host_uploads != 1 || stats.host_downloads != 1
        || stats.c2c_ingress_bytes == 0 || stats.c2c_ingress_cycles == 0
        || stats.c2c_egress_bytes == 0 || stats.c2c_egress_cycles == 0)
        throw std::runtime_error(
            "ModelSession did not account for physical C2C I/O");

    test_planar_output(BindingElementType::BF16, BindingLayout::Fp16PairPlanar,
                       {32,1024}, {32,33}, 2, 6432);
    test_planar_output(BindingElementType::F32, BindingLayout::Fp32QuadPlanar,
                       {2048,128}, {8,9,10,11}, 1, 0);
    std::cout << "model_session_c2c_io_test passed"
              << " ingress_bytes=" << stats.c2c_ingress_bytes
              << " ingress_cycles=" << stats.c2c_ingress_cycles
              << " egress_bytes=" << stats.c2c_egress_bytes
              << " egress_cycles=" << stats.c2c_egress_cycles << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "model_session_c2c_io_test failed: "
              << error.what() << '\n';
    return 1;
}
