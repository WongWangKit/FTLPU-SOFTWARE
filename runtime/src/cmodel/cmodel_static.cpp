#include "ftlpu/software/runtime/cmodel_link.hpp"

#include "ftlpu/system/tsp_slice_system.hpp"

namespace ftlpu::software::runtime {

std::size_t linked_cmodel_mxm_count()
{
    return TspSliceSystem::kMxmCount;
}

} // namespace ftlpu::software::runtime
