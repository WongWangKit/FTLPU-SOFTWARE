#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/fp16.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

float readFp16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Fp16::from_bits(
        static_cast<std::uint16_t>(data[2 * index])
        | (static_cast<std::uint16_t>(data[2 * index + 1]) << 8))
        .to_float();
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 2)
        throw std::runtime_error(
            "usage: hf_two_decoder_layers_model_session_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ModelSession session(*system);
    session.load_file(std::filesystem::path(argv[1]));
    if (session.package().executables.size() != 1
        || session.package().executables[0]
               .program.scale_relocations.empty())
        throw std::logic_error(
            "two-layer package did not reuse a parameterized executable");

    const SessionMemoryPlan& plan = session.memory_plan();
    if (plan.invocations.size() != 2 || plan.lifetimes.size() != 1
        || plan.lifetimes[0].value != "hidden.1"
        || plan.invocations[1].inputs.empty()
        || plan.invocations[1].inputs[0].transfer
            != SessionTransferKind::DeviceAlias)
        throw std::logic_error(
            "two-layer package did not receive a device-resident memory plan");

    session.set_input("hidden.0", session.value("golden.input"));
    session.run();

    const ModelSessionStats& stats = session.stats();
    if (stats.host_uploads != 19 || stats.host_downloads != 1
        || stats.device_aliases != 1 || stats.device_copies != 0
        || stats.device_copy_bytes != 0)
        throw std::logic_error(
            "two-layer session used an unexpected host/device transfer plan");

    const auto& actual = session.value("hidden.2");
    const auto& expected = session.value("golden.output");
    if (actual.size() != expected.size())
        throw std::logic_error("two-layer result size mismatch");
    float maximumError = 0.0f;
    double meanError = 0.0;
    std::size_t maximumIndex = 0;
    for (std::size_t index = 0; index < actual.size() / 2; ++index) {
        const float error = std::fabs(
            readFp16(actual, index) - readFp16(expected, index));
        meanError += error;
        if (error > maximumError) {
            maximumError = error;
            maximumIndex = index;
        }
    }
    meanError /= actual.size() / 2;
    if (!std::isfinite(maximumError) || maximumError > 0.5f
        || meanError > 0.05)
        throw std::logic_error(
            "HF two-layer golden mismatch max_error="
            + std::to_string(maximumError)
            + " mean_error=" + std::to_string(meanError)
            + " index=" + std::to_string(maximumIndex));

    std::cout
        << "hf_two_decoder_layers_model_session_test passed max_error="
        << maximumError << " mean_error=" << meanError
        << " host_uploads=" << stats.host_uploads
        << " host_downloads=" << stats.host_downloads
        << " device_aliases=" << stats.device_aliases
        << " device_copies=" << stats.device_copies
        << " device_copy_bytes=" << stats.device_copy_bytes << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_two_decoder_layers_model_session_test failed: "
              << exception.what() << '\n';
    return 1;
}
