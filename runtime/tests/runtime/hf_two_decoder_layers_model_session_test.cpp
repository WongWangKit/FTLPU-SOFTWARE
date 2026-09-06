#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

float readBf16(const std::vector<std::uint8_t>& data, std::size_t index)
{
    return ftlpu::Bf16::from_bits(
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
    auto system = std::make_unique<ftlpu::C2cDmaSystem>();
    ModelSession session(*system);
    session.load_file(std::filesystem::path(argv[1]));
    const char* tracePath = std::getenv("FTLPU_QWEN_PIPELINE_CSV");
    if (tracePath != nullptr)
        session.enable_execution_trace();
    const bool invocationPaged = !session.package().weight_pages.empty();
    const bool executablePaged = std::any_of(
        session.package().executables.begin(),
        session.package().executables.end(), [](const auto& executable) {
            return !executable.program.weight_page_uses.empty();
        });
    const bool paged = invocationPaged || executablePaged;
    if (session.package().executables.size() != 2
        || session.package().executables[0].serialized_program.empty()
        || session.package().executables[0].program.bindings.empty())
        throw std::logic_error(
            "two-layer package has an unexpected executable layout");
    if (invocationPaged
        && (session.package().weight_pages.size() != 2
            || session.package().weight_pages[0].bank != 0
            || session.package().weight_pages[1].bank != 1
            || session.package().invocations[0].weight_page != 0
            || session.package().invocations[1].weight_page != 1))
        throw std::logic_error(
            "two-layer package does not alternate C2C weight banks");

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
    if (tracePath != nullptr)
        session.write_execution_trace_csv(tracePath);

    const ModelSessionStats& stats = session.stats();
    const auto expectedHostDownloads = static_cast<std::size_t>(std::count_if(
        session.package().values.begin(), session.package().values.end(),
        [](const ModelValue& value) { return value.external_output; }));
    if (stats.host_uploads != 1
        || stats.host_downloads != expectedHostDownloads
        || stats.device_aliases != 1 || stats.device_copies != 0
        || stats.device_copy_bytes != 0 || stats.c2c_ingress_bytes == 0
        || stats.c2c_egress_bytes == 0)
        throw std::logic_error(
            "two-layer session used an unexpected host/device transfer plan: "
            "host_uploads=" + std::to_string(stats.host_uploads)
            + " host_downloads=" + std::to_string(stats.host_downloads)
            + " device_aliases=" + std::to_string(stats.device_aliases)
            + " device_copies=" + std::to_string(stats.device_copies)
            + " device_copy_bytes=" +
                std::to_string(stats.device_copy_bytes)
            + " c2c_ingress_bytes=" +
                std::to_string(stats.c2c_ingress_bytes)
            + " c2c_egress_bytes=" +
                std::to_string(stats.c2c_egress_bytes));
    if (paged && stats.weight_page_prefetches == 0)
        throw std::logic_error(
            "two-layer session did not execute C2C page ping-pong");

    const auto& actual = session.value("hidden.2");
    const auto& expected = session.value("golden.output");
    if (actual.size() != expected.size())
        throw std::logic_error("two-layer result size mismatch");
    float maximumError = 0.0f;
    double meanError = 0.0;
    std::size_t maximumIndex = 0;
    std::size_t mismatches = 0;
    std::vector<float> errors;
    errors.reserve(actual.size() / 2);
    for (std::size_t index = 0; index < actual.size() / 2; ++index) {
        const float actualValue = readBf16(actual, index);
        const float expectedValue = readBf16(expected, index);
        if (!std::isfinite(actualValue)
            || !std::isfinite(expectedValue))
            throw std::logic_error(
                "HF two-layer output contains a non-finite value index="
                + std::to_string(index)
                + " actual=" + std::to_string(actualValue)
                + " expected=" + std::to_string(expectedValue));
        const float error = std::fabs(actualValue - expectedValue);
        errors.push_back(error);
        meanError += error;
        if (error > 0.25f + 0.05f * std::fabs(expectedValue))
            ++mismatches;
        if (error > maximumError) {
            maximumError = error;
            maximumIndex = index;
        }
    }
    const std::size_t values = actual.size() / 2;
    meanError /= values;
    std::sort(errors.begin(), errors.end());
    const float p99 = errors[static_cast<std::size_t>(
        0.99 * static_cast<double>(errors.size() - 1))];
    const double mismatchFraction =
        static_cast<double>(mismatches) / static_cast<double>(values);
    if (!std::isfinite(maximumError) || !std::isfinite(meanError)
        || mismatchFraction > 0.001 || p99 > 0.5f
        || maximumError > 32.0f || meanError > 0.075)
        throw std::logic_error(
            "HF two-layer golden mismatch max_error="
            + std::to_string(maximumError)
            + " mean_error=" + std::to_string(meanError)
            + " p99=" + std::to_string(p99)
            + " mismatches=" + std::to_string(mismatches)
            + " mismatch_fraction=" + std::to_string(mismatchFraction)
            + " index=" + std::to_string(maximumIndex));

    std::cout
        << "hf_two_decoder_layers_model_session_test passed max_error="
        << maximumError << " mean_error=" << meanError
        << " p99=" << p99
        << " mismatches=" << mismatches
        << " mismatch_fraction=" << mismatchFraction
        << " resident_uploads=" << stats.resident_uploads
        << " host_uploads=" << stats.host_uploads
        << " host_downloads=" << stats.host_downloads
        << " device_aliases=" << stats.device_aliases
        << " device_copies=" << stats.device_copies
        << " device_copy_bytes=" << stats.device_copy_bytes
        << " paged=" << paged
        << " executable_paged=" << executablePaged
        << " c2c_ingress_bytes=" << stats.c2c_ingress_bytes
        << " c2c_ingress_cycles=" << stats.c2c_ingress_cycles
        << " c2c_egress_bytes=" << stats.c2c_egress_bytes
        << " c2c_egress_cycles=" << stats.c2c_egress_cycles
        << " initial_page_wait_cycles="
        << stats.weight_page_initial_wait_cycles
        << " boundary_page_wait_cycles="
        << stats.weight_page_boundary_wait_cycles
        << " deferred_page_prefetches="
        << stats.weight_page_deferred_prefetches << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_two_decoder_layers_model_session_test failed: "
              << exception.what() << '\n';
    return 1;
}
