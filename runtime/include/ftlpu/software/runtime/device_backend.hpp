#pragma once

#include "ftlpu/dma/descriptor.hpp"
#include "ftlpu/program/program_image.hpp"
#include "ftlpu/software/runtime/target_description.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <vector>

namespace ftlpu::software::runtime {

// ProgramImage already contains packed instruction sections and optional
// device-resident data. execution_cycles is the deterministic schedule budget
// after the autonomous bootstrap reaches schedule epoch zero.
struct DeviceProgram {
    ProgramImage image{};
    std::size_t execution_cycles{0};
};

enum class DeviceBackendState : std::uint8_t {
    Empty,
    Loaded,
    Running,
    Complete,
};

struct DeviceBackendStatistics {
    std::size_t dma_upload_bytes{0};
    std::size_t dma_download_bytes{0};
    std::size_t dma_transfer_count{0};
    std::size_t system_cycles{0};
};

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    virtual const TargetDescription& target() const noexcept = 0;
    virtual DeviceBackendState state() const noexcept = 0;
    virtual DeviceBackendStatistics statistics() const noexcept = 0;

    // Clears queues and functional pipelines while retaining MEM SRAM.
    virtual void reset_execution_state() = 0;

    // Program and embedded data are transferred through the device DMA path.
    virtual void load(const DeviceProgram& program) = 0;

    // Transfers operate on whole physical vectors. Layout packing belongs to
    // the BinaryBinding/ModelSession adapter above this interface.
    virtual void upload(
        MemGlobalAddress24 destination,
        std::span<const std::uint8_t> bytes,
        DmaPurpose purpose = DmaPurpose::InputTensor) = 0;

    virtual void launch() = 0;
    virtual void wait(std::ostream* log = nullptr) = 0;

    virtual std::vector<std::uint8_t> download(
        MemGlobalAddress24 source,
        std::size_t byte_count,
        DmaPurpose purpose = DmaPurpose::OutputTensor) = 0;
};

} // namespace ftlpu::software::runtime