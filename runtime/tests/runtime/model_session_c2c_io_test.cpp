#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
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

} // namespace

int main()
try {
    BinaryProgram program;
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
    session.load(std::move(package));
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
