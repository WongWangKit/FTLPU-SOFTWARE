#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

constexpr std::size_t kRows = 32;
constexpr std::size_t kColumns = 32;

BinaryBinding makeBinding(BindingAccess access)
{
    BinaryBinding binding;
    binding.index = 0;
    binding.access = access;
    binding.element_type = BindingElementType::BF16;
    binding.layout = BindingLayout::Fp16MxmDistributed16;
    binding.byte_size = kRows * kColumns * sizeof(std::uint16_t);
    binding.base_row = 64;
    binding.instruction_count = 4;
    binding.address_stride = 1;
    binding.shape = { kRows, kColumns };
    for (std::uint16_t slice = 0; slice < 16; ++slice)
        binding.slices.push_back(slice);
    binding.role = access == BindingAccess::Input ? "activation" : "result";
    binding.name = access == BindingAccess::Input ? "input" : "output";
    binding.hemisphere_mask = 3;
    binding.bank = 0;
    return binding;
}

std::vector<std::uint8_t> makeInput(int seed)
{
    std::vector<std::uint8_t> result(kRows * kColumns * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < kRows * kColumns; ++index) {
        const auto bits = Bf16::from_float(
            static_cast<float>((static_cast<int>(index) + seed) % 31 - 15)
            / 8.0f)
                              .bits();
        result[2 * index] = static_cast<std::uint8_t>(bits);
        result[2 * index + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    return result;
}

ModelPackage makePackage()
{
    BinaryProgram program;
    program.bindings = {
        makeBinding(BindingAccess::Input),
        makeBinding(BindingAccess::Output),
    };

    ModelPackage package;
    package.model_name = "host-override";
    package.architecture = "test";
    package.values = {
        { "input", BindingElementType::BF16, { kRows, kColumns }, true, false },
        { "middle", BindingElementType::BF16, { kRows, kColumns }, true,
            false },
        { "output", BindingElementType::BF16, { kRows, kColumns }, false,
            true },
    };
    package.executables.push_back({ "identity", std::move(program), { } });
    package.invocations = {
        { "first", 0, { { 0, "input" } }, { { 0, "middle" } }, { } },
        { "second", 0, { { 0, "middle" } }, { { 0, "output" } }, { } },
    };
    return package;
}

} // namespace

int main()
try {
    C2cDmaSystem system;
    ModelSession session(system);
    session.load(makePackage());
    if (session.memory_plan().invocations[1].inputs[0].transfer
        != SessionTransferKind::DeviceAlias)
        throw std::logic_error("test package did not plan a device alias");

    const auto first = makeInput(0);
    const auto override = makeInput(7);
    session.set_input("input", first);
    session.set_input("middle", override);
    session.run(0);

    if (session.value("output") != override)
        throw std::logic_error(
            "explicit intermediate input did not override its device alias");
    if (session.stats().host_uploads != 2
        || session.stats().device_aliases != 0)
        throw std::logic_error(
            "host override used an unexpected transfer plan");

    std::cout << "model_session_host_override_test passed"
              << " host_uploads=" << session.stats().host_uploads
              << " device_aliases=" << session.stats().device_aliases << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "model_session_host_override_test failed: " << error.what()
              << '\n';
    return 1;
}
