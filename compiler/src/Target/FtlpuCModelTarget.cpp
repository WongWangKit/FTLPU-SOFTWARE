#include "ftlpu/compiler/Target/ftlpu_cmodel_target.hpp"

namespace ftlpu::compiler::target {
namespace {

constexpr std::size_t kStreamRegisterColumns = 12;
constexpr std::size_t kSlicesPerStreamRegisterGroup = 4;

} // namespace

std::string FtlpuCModelTarget::name() const
{
    return "ftlpu-cmodel";
}

const MemoryTopology& FtlpuCModelTarget::memory_topology() const
{
    return topology_;
}

std::size_t FtlpuCModelTarget::mem_to_mxm_latency(std::size_t mem_slice) const
{
    return (kStreamRegisterColumns - 1) - mem_slice / kSlicesPerStreamRegisterGroup + 1;
}

std::size_t FtlpuCModelTarget::mxm_to_fu_latency() const
{
    return kStreamRegisterColumns;
}

std::size_t FtlpuCModelTarget::vxm_pipeline_latency(std::size_t stages) const
{
    return stages == 0 ? 0 : stages - 1;
}

std::size_t FtlpuCModelTarget::vxm_to_mem_latency(std::size_t mem_slice) const
{
    return mem_slice / kSlicesPerStreamRegisterGroup + 2;
}

} // namespace ftlpu::compiler::target
