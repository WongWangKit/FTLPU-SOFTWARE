#pragma once

#include "ftlpu/software/runtime/device_backend.hpp"

#include <memory>

namespace ftlpu {
class TspSliceSystem;
}

namespace ftlpu::software::runtime {

TargetDescription make_cmodel_target_description();

class CModelDeviceBackend final : public DeviceBackend {
public:
    CModelDeviceBackend();
    ~CModelDeviceBackend() override;

    CModelDeviceBackend(const CModelDeviceBackend&) = delete;
    CModelDeviceBackend& operator=(const CModelDeviceBackend&) = delete;
    CModelDeviceBackend(CModelDeviceBackend&&) noexcept;
    CModelDeviceBackend& operator=(CModelDeviceBackend&&) noexcept;

    const TargetDescription& target() const noexcept override;
    DeviceBackendState state() const noexcept override;
    DeviceBackendStatistics statistics() const noexcept override;

    void reset_execution_state() override;
    void load(const DeviceProgram& program) override;
    void upload(
        MemGlobalAddress24 destination,
        std::span<const std::uint8_t> bytes,
        DmaPurpose purpose = DmaPurpose::InputTensor) override;
    void launch() override;
    void wait(std::ostream* log = nullptr) override;
    std::vector<std::uint8_t> download(
        MemGlobalAddress24 source,
        std::size_t byte_count,
        DmaPurpose purpose = DmaPurpose::OutputTensor) override;

    // CModel-only diagnostic access. Runtime code should use DeviceBackend.
    TspSliceSystem& system() noexcept;
    const TspSliceSystem& system() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ftlpu::software::runtime
