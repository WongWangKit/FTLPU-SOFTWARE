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

    C2cDmaSystem system;
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
