#pragma once

#include "ftlpu/compiler/Target/target_backend.hpp"

namespace ftlpu::compiler::target {

class FtlpuCModelTarget final : public TargetBackend {
public:
    std::string name() const override;
    const MemoryTopology& memory_topology() const override;
    std::size_t mem_to_mxm_latency(std::size_t mem_slice) const override;
    std::size_t mxm_to_fu_latency() const override;
    std::size_t vxm_pipeline_latency(std::size_t stages) const override;
    std::size_t vxm_to_mem_latency(std::size_t mem_slice) const override;

private:
    MemoryTopology topology_{};
};

} // namespace ftlpu::compiler::target
