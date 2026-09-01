#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/model_package.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::software::runtime;

const ModelTensor& findTensor(
    const ModelPackage& package, const std::string& name)
{
    for (const ModelTensor& tensor : package.tensors)
        if (tensor.name == name) return tensor;
    throw std::out_of_range("weight-page tensor is unavailable: " + name);
}

void configureSystem(
    C2cDmaSystem& system, const ExecutableHardwareConfig& hardware)
{
    SystemHardwareConfiguration cmodelHardware;
    cmodelHardware.sram_depth_rows = hardware.sram_depth_rows;
    cmodelHardware.mxms_per_hemisphere = hardware.mxms_per_hemisphere;
    cmodelHardware.mxm_weight_buffers = hardware.mxm_weight_buffers;
    cmodelHardware.vxm_alus = hardware.vxm_alus;
    cmodelHardware.c2c_streams_per_direction =
        hardware.c2c_streams_per_direction;
    cmodelHardware.mxm_local_dequant_enabled =
        hardware.mxm_local_dequant_enabled != 0;
    cmodelHardware.mxm_weight_activation_overlap_enabled =
        hardware.mxm_weight_activation_overlap_enabled != 0;
    system.chip().configure_hardware(cmodelHardware);

    Ddr4Config ddr;
    ddr.beat_bytes = hardware.c2c_bytes_per_stream_per_cycle;
    ddr.read_latency_cycles = hardware.ddr_read_latency_cycles;
    ddr.write_latency_cycles = hardware.ddr_write_latency_cycles;
    ddr.read_latency_jitter_cycles =
        hardware.ddr_read_latency_jitter_cycles;
    ddr.write_latency_jitter_cycles =
        hardware.ddr_write_latency_jitter_cycles;
    ddr.request_queue_depth = hardware.ddr_request_queue_depth;
    ddr.transfer_channels = static_cast<std::size_t>(hardware.hemispheres)
        * hardware.c2c_streams_per_direction;
    ddr.lpu_clock_hz =
        static_cast<std::uint64_t>(hardware.lpu_clock_mhz) * 1'000'000;
    ddr.peak_bandwidth_bytes_per_second =
        static_cast<std::uint64_t>(
            hardware.ddr_peak_bandwidth_mbytes_per_second)
        * 1'000'000;
    ddr.latency_random_seed = hardware.ddr_latency_random_seed;
    system.ddr4().configure(ddr);
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc < 2 || argc > 3)
        throw std::runtime_error(
            "usage: c2c_weight_page_package_roundtrip_test model.ftlpum "
            "[page-index]");
    const ModelPackage package = read_model_package(
        std::filesystem::path(argv[1]), ModelPackageLoadMode::LazyExecutables);
    const std::size_t selected = argc == 3
        ? static_cast<std::size_t>(std::stoul(argv[2])) : 0;
    if (selected >= package.weight_pages.size())
        throw std::out_of_range("weight-page index is out of range");

    if (package.executables.empty())
        throw std::runtime_error("weight-page package has no executable target");
    C2cDmaSystem system;
    configureSystem(system, package.executables.front().program.hardware);
    std::vector<C2cWeightPage> pages;
    std::uint64_t nextDdrAddress = 0;
    for (const ModelWeightPage& source : package.weight_pages) {
        C2cWeightPage page;
        page.layer = source.layer;
        page.bank = source.bank;
        for (const ModelWeightPage::Segment& segment : source.segments) {
            const ModelTensor& tensor = findTensor(package, segment.tensor);
            const std::uint64_t ddrAddress = nextDdrAddress;
            for (std::uint32_t vector = 0; vector < segment.vector_count;
                 ++vector) {
                C2cVector payload;
                const std::size_t sourceOffset = static_cast<std::size_t>(
                    segment.byte_offset
                    + static_cast<std::uint64_t>(vector)
                        * hw::kPhysicalVectorBytes);
                for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile; ++lane)
                        payload.payload[tile][lane] = tensor.data[
                            sourceOffset + tile * hw::kLanesPerTile + lane];
                system.ddr4().initialize_vector(
                    ddrAddress
                        + static_cast<std::uint64_t>(vector)
                            * hw::kPhysicalVectorBytes,
                    payload);
            }
            page.segments.push_back(C2cWeightSegment {
                static_cast<Hemisphere>(segment.hemisphere), segment.slice,
                source.bank, segment.base_row, segment.stream, ddrAddress,
                segment.vector_count});
            nextDdrAddress += static_cast<std::uint64_t>(
                segment.vector_count) * hw::kPhysicalVectorBytes;
        }
        pages.push_back(std::move(page));
    }

    C2cWeightPager pager(system);
    pager.enqueue(pages[selected]);
    pager.wait(pager.stats().vectors * 64 + 1024);

    const ModelWeightPage& sourcePage = package.weight_pages[selected];
    std::uint64_t comparedBytes = 0;
    for (const ModelWeightPage::Segment& segment : sourcePage.segments) {
        const ModelTensor& tensor = findTensor(package, segment.tensor);
        for (std::uint32_t vector = 0; vector < segment.vector_count;
             ++vector) {
            const std::size_t sourceOffset = static_cast<std::size_t>(
                segment.byte_offset
                + static_cast<std::uint64_t>(vector)
                    * hw::kPhysicalVectorBytes);
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile; ++lane) {
                    const std::uint8_t expected = tensor.data[
                        sourceOffset + tile * hw::kLanesPerTile + lane];
                    const std::uint8_t actual =
                        system.chip().read_mem_sram_lane_byte(
                            static_cast<Hemisphere>(segment.hemisphere),
                            segment.slice, sourcePage.bank, tile,
                            segment.base_row + vector, lane);
                    if (actual != expected)
                        throw std::runtime_error(
                            "C2C package page mismatch page="
                            + std::to_string(selected) + " tensor="
                            + segment.tensor + " vector="
                            + std::to_string(vector) + " tile="
                            + std::to_string(tile) + " lane="
                            + std::to_string(lane) + " expected="
                            + std::to_string(expected) + " actual="
                            + std::to_string(actual));
                    ++comparedBytes;
                }
        }
    }
    std::cout << "c2c_weight_page_package_roundtrip_test passed page="
              << selected << " bytes=" << comparedBytes
              << " cycles=" << system.cycle() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "c2c_weight_page_package_roundtrip_test failed: "
              << error.what() << '\n';
    return 1;
}
