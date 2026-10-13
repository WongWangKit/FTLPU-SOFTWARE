#include "ftlpu/software/runtime/model_session.hpp"

#include "ftlpu/core/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

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
            "usage: hf_decoder_layer_model_session_test model.ftlpum");
    using namespace ftlpu::software::runtime;
    auto system = std::make_unique<ftlpu::C2cDmaSystem>();
    ModelSession session(*system);
    const char* pipeline = std::getenv("FTLPU_DENSE_PIPELINE_CSV");
    if (pipeline)
        session.enable_execution_trace();
    if (const char* bandwidth = std::getenv("FTLPU_DDR_BANDWIDTH_MBPS")) {
        const auto value = std::stoul(bandwidth);
        if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "invalid FTLPU_DDR_BANDWIDTH_MBPS");
        session.set_ddr_peak_bandwidth_mbytes_per_second(
            static_cast<std::uint32_t>(value));
    }
    session.load_file(std::filesystem::path(argv[1]));
    session.set_input("hidden.0", session.value("golden.input"));
    try {
        session.run();
        session.write_last_linked_program(
            std::filesystem::path(std::string(argv[1]) + ".linked.ftlpu"));
        if (pipeline)
            session.write_execution_trace_csv(std::filesystem::path(pipeline));
    } catch (...) {
        // Preserve the exact post-link ICU image on datapath failures.  This
        // includes runtime C2C queues and page-ready barriers, unlike the
        // compiler image embedded in the package.
        try {
            session.write_last_linked_program(
                std::filesystem::path(std::string(argv[1]) + ".linked.ftlpu"));
            if (pipeline)
                session.write_execution_trace_csv(std::filesystem::path(pipeline));
        } catch (...) {
        }
        throw;
    }

    const auto& actual = session.value("hidden.1");
    const auto& expected = session.value("golden.output");
    if (actual.size() != expected.size())
        throw std::logic_error("decoder layer result size mismatch");
    float maximumError = 0.0f;
    double meanError = 0.0;
    std::size_t maximumIndex = 0;
    for (std::size_t index = 0; index < actual.size() / 2; ++index) {
        const float actualValue = readBf16(actual, index);
        const float expectedValue = readBf16(expected, index);
        if (!std::isfinite(actualValue)
            || !std::isfinite(expectedValue))
            throw std::logic_error(
                "HF decoder layer produced a non-finite value index="
                + std::to_string(index)
                + " actual=" + std::to_string(actualValue)
                + " expected=" + std::to_string(expectedValue));
        const float error = std::fabs(actualValue - expectedValue);
        meanError += error;
        if (error > maximumError) {
            maximumError = error;
            maximumIndex = index;
        }
    }
    meanError /= actual.size() / 2;
    if (!std::isfinite(maximumError) || !std::isfinite(meanError)
        || maximumError > 0.35f
        || meanError > 0.035) {
        const auto modelPath = std::filesystem::path(argv[1]);
        const auto actualPath = modelPath.string() + ".actual.bf16.bin";
        const auto expectedPath = modelPath.string() + ".expected.bf16.bin";
        std::ofstream actualStream(actualPath, std::ios::binary);
        actualStream.write(
            reinterpret_cast<const char*>(actual.data()),
            static_cast<std::streamsize>(actual.size()));
        std::ofstream expectedStream(expectedPath, std::ios::binary);
        expectedStream.write(
            reinterpret_cast<const char*>(expected.data()),
            static_cast<std::streamsize>(expected.size()));
        throw std::logic_error(
            "HF decoder layer golden mismatch max_error="
            + std::to_string(maximumError)
            + " mean_error=" + std::to_string(meanError)
            + " index=" + std::to_string(maximumIndex));
    }
    std::cout << "hf_decoder_layer_model_session_test passed max_error="
              << maximumError << " mean_error=" << meanError << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "hf_decoder_layer_model_session_test failed: "
              << exception.what() << '\n';
    return 1;
}
