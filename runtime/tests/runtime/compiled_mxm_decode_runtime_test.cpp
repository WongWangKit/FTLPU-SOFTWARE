#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"

#include "ftlpu/core/bf16.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

float read_bf16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    const std::size_t offset = index * 2;
    return ftlpu::Bf16::from_bits(
        static_cast<std::uint16_t>(data[offset])
        | (static_cast<std::uint16_t>(data[offset + 1]) << 8)).to_float();
}

} // namespace

int main(int argc, char** argv)
try {
    using namespace ftlpu::software::runtime;
    if (argc != 2)
        throw std::runtime_error(
            "usage: compiled_mxm_decode_runtime_test program.ftlpu");
    const auto program = read_binary_program(std::filesystem::path(argv[1]));
    if (program.bindings.size() != 3
        || program.bindings[0].layout != BindingLayout::Fp16PairPlanar
        || program.bindings[1].layout
            != BindingLayout::W8A16MxmDecodeStreamed
        || program.bindings[2].layout
            != BindingLayout::Fp16MxmDecodePairPlanar)
        throw std::logic_error(
            "compiler did not select the MXM decode physical layouts");
    bool packedLoad = false;
    bool packedCompute = false;
    for (const auto& queue : program.queues) {
        if (!queue.packed_macro_imem) continue;
        packedLoad |= queue.kind == QueueKind::MxmLoad;
        packedCompute |= queue.kind == QueueKind::MxmCompute;
    }
    if (!packedLoad || !packedCompute)
        throw std::logic_error(
            "compiler decode workload did not use packed Macro queues");

    std::vector<std::uint8_t> activation(32 * 2);
    const std::uint16_t one = ftlpu::Bf16::from_float(1.0f).bits();
    for (std::size_t k = 0; k < 32; ++k) {
        activation[2 * k] = static_cast<std::uint8_t>(one);
        activation[2 * k + 1] = static_cast<std::uint8_t>(one >> 8);
    }
    std::vector<std::uint8_t> weight(32 * 64);
    for (std::size_t k = 0; k < 32; ++k)
        for (std::size_t n = 0; n < 64; ++n)
            weight[k * 64 + n] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>(static_cast<int>(n % 5) - 2));

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    CModelRuntime runtime(*system);
    runtime.load(program);
    runtime.upload_input(0, activation);
    runtime.upload_input(1, weight);
    runtime.run_cycles(program.max_cycle + 128);
    const auto output = runtime.download_output(0);
    if (output.size() != 64 * 2)
        throw std::logic_error("decode output has an unexpected size");
    for (std::size_t n = 0; n < 64; ++n) {
        const float expected =
            4.0f * static_cast<float>(static_cast<int>(n % 5) - 2);
        const float actual = read_bf16(output, n);
        if (std::fabs(actual - expected) > 0.01f)
            throw std::logic_error(
                "Decode.StreamCompute produced an incorrect output at "
                + std::to_string(n) + ": expected="
                + std::to_string(expected) + " actual="
                + std::to_string(actual));
    }
    std::cout << "compiled_mxm_decode_runtime_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "compiled_mxm_decode_runtime_test failed: "
              << error.what() << '\n';
    return 1;
}
