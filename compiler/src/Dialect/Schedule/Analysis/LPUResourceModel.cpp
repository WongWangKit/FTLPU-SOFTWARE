#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"

#include <stdexcept>

namespace ftlpu::compiler::schedule {

std::string LPUResourceModel::mem_slice(int64_t hemisphere, int64_t slice) const
{
    if (hemisphere < 0 || hemisphere >= target_.memory().hemispheres
        || slice < 0 || slice >= target_.memory().slices_per_hemisphere)
        throw std::out_of_range("invalid MEM resource");
    return "mem.h" + std::to_string(hemisphere) + ".s" + std::to_string(slice);
}

std::string LPUResourceModel::mxm(int64_t unit) const
{
    if (!target_.is_valid_mxm_unit(unit)) throw std::out_of_range("invalid MXM resource");
    return "mxm." + std::to_string(unit);
}

std::string LPUResourceModel::mxm_weight_buffer(int64_t unit, int64_t buffer) const
{
    if (!target_.is_valid_mxm_unit(unit) || !target_.is_valid_weight_buffer(buffer))
        throw std::out_of_range("invalid MXM weight-buffer resource");
    return mxm(unit) + ".wb" + std::to_string(buffer);
}

std::string LPUResourceModel::vxm_alu(int64_t alu) const
{
    if (!target_.is_valid_vxm_alu(alu)) throw std::out_of_range("invalid VXM resource");
    return "vxm.alu" + std::to_string(alu);
}

std::string LPUResourceModel::sxm(int64_t hemisphere) const
{
    if (hemisphere < 0 || hemisphere >= target_.memory().hemispheres)
        throw std::out_of_range("invalid SXM resource");
    return "sxm.h" + std::to_string(hemisphere);
}

std::string LPUResourceModel::stream(
    target::StreamDirection direction, int64_t index) const
{
    if (index < 0 || index >= target_.streams().streams_per_direction)
        throw std::out_of_range("invalid stream resource");
    return std::string(direction == target::StreamDirection::East ? "stream.e" : "stream.w")
        + std::to_string(index);
}

} // namespace ftlpu::compiler::schedule
