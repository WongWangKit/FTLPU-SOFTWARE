#pragma once

#include <cstddef>
#include <string>

namespace ftlpu::compiler::target {

struct MemoryTopology {
    std::size_t hemispheres{2};
    std::size_t slices_per_hemisphere{44};
    std::size_t banks_per_slice{2};
    std::size_t words_per_bank{4096};
    std::size_t bytes_per_word{16};
};

class TargetBackend {
public:
    virtual ~TargetBackend() = default;

    virtual std::string name() const = 0;
    virtual const MemoryTopology& memory_topology() const = 0;
    virtual std::size_t mem_to_mxm_latency(std::size_t mem_slice) const = 0;
    virtual std::size_t mxm_to_fu_latency() const = 0;
    virtual std::size_t vxm_pipeline_latency(std::size_t stages) const = 0;
    virtual std::size_t vxm_to_mem_latency(std::size_t mem_slice) const = 0;
};

} // namespace ftlpu::compiler::target
