#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"

#include <stdexcept>

namespace ftlpu::compiler::schedule {

namespace {
void validate_mem(const target::LPUTargetModel& target,
    int64_t hemisphere, int64_t slice, int64_t bank)
{
    if (hemisphere < 0 || hemisphere >= target.memory().hemispheres
        || slice < 0 || slice >= target.memory().slices_per_hemisphere
        || bank < 0 || bank >= target.memory().banks_per_slice)
        throw std::out_of_range("invalid MEM resource");
}
} // namespace

PhysicalQueueId LPUResourceModel::mem_queue(
    int64_t hemisphere, int64_t slice, int64_t bank) const
{
    validate_mem(target_, hemisphere, slice, bank);
    return {PhysicalQueueKind::Mem,
        hemisphere * target_.memory().slices_per_hemisphere
                * target_.memory().banks_per_slice
            + slice * target_.memory().banks_per_slice + bank};
}

PhysicalQueueId LPUResourceModel::mxm_load_queue(int64_t unit) const
{
    if (!target_.is_valid_mxm_unit(unit))
        throw std::out_of_range("invalid MXM load queue");
    return {PhysicalQueueKind::MxmLoad, unit};
}

PhysicalQueueId LPUResourceModel::mxm_compute_queue(int64_t unit) const
{
    if (!target_.is_valid_mxm_unit(unit))
        throw std::out_of_range("invalid MXM compute queue");
    return {PhysicalQueueKind::MxmCompute, unit};
}

PhysicalQueueId LPUResourceModel::mxm_dequant_queue(int64_t unit) const
{
    if (!target_.is_valid_mxm_unit(unit))
        throw std::out_of_range("invalid MXM dequant queue");
    return {PhysicalQueueKind::MxmDequant, unit};
}

PhysicalQueueId LPUResourceModel::vxm_queue(int64_t queue) const
{
    if (queue < 0 || queue >= target_.throughput().vxm_alus)
        throw std::out_of_range("invalid VXM queue");
    return {PhysicalQueueKind::Vxm, queue};
}

PhysicalQueueId LPUResourceModel::sxm_queue(
    bool transpose, int64_t hemisphere) const
{
    if (hemisphere < 0 || hemisphere >= target_.memory().hemispheres)
        throw std::out_of_range("invalid SXM queue");
    return {transpose ? PhysicalQueueKind::SxmTranspose
                      : PhysicalQueueKind::SxmPermute,
        hemisphere};
}

PhysicalResourceId LPUResourceModel::queue_resource(
    PhysicalQueueId queue) const
{
    switch (queue.kind) {
    case PhysicalQueueKind::Mem:
        return {PhysicalResourceKind::MemIcuQueue, queue.index};
    case PhysicalQueueKind::MxmLoad:
        return {PhysicalResourceKind::MxmLoadQueue, queue.index};
    case PhysicalQueueKind::MxmCompute:
        return {PhysicalResourceKind::MxmComputeQueue, queue.index};
    case PhysicalQueueKind::MxmDequant:
        return {PhysicalResourceKind::MxmDequantQueue, queue.index};
    case PhysicalQueueKind::Vxm:
        return {PhysicalResourceKind::VxmQueue, queue.index};
    case PhysicalQueueKind::SxmTranspose:
        return {PhysicalResourceKind::SxmTransposeQueue, queue.index};
    case PhysicalQueueKind::SxmPermute:
        return {PhysicalResourceKind::SxmPermuteQueue, queue.index};
    }
    throw std::logic_error("unknown physical queue kind");
}

PhysicalResourceId LPUResourceModel::mem_read_port(
    int64_t hemisphere, int64_t slice) const
{
    return mem_read_port(hemisphere, slice, 0);
}

PhysicalResourceId LPUResourceModel::mem_read_port(
    int64_t hemisphere, int64_t slice, int64_t bank) const
{
    validate_mem(target_, hemisphere, slice, bank);
    return {PhysicalResourceKind::MemReadPort, hemisphere, slice, bank};
}

PhysicalResourceId LPUResourceModel::mem_write_port(
    int64_t hemisphere, int64_t slice) const
{
    return mem_write_port(hemisphere, slice, 0);
}

PhysicalResourceId LPUResourceModel::mem_write_port(
    int64_t hemisphere, int64_t slice, int64_t bank) const
{
    validate_mem(target_, hemisphere, slice, bank);
    return {PhysicalResourceKind::MemWritePort, hemisphere, slice, bank};
}

PhysicalResourceId LPUResourceModel::mem_icu(
    int64_t hemisphere, int64_t slice, int64_t bank) const
{
    return queue_resource(mem_queue(hemisphere, slice, bank));
}

PhysicalResourceId LPUResourceModel::mxm(int64_t unit) const
{
    return queue_resource(mxm_compute_queue(unit));
}

PhysicalResourceId LPUResourceModel::mxm_weight_buffer(
    int64_t unit, int64_t buffer) const
{
    if (!target_.is_valid_mxm_unit(unit)
        || !target_.is_valid_weight_buffer(buffer))
        throw std::out_of_range("invalid MXM weight-buffer resource");
    return {PhysicalResourceKind::MxmWeightBuffer, unit, buffer};
}

PhysicalResourceId LPUResourceModel::vxm_alu(int64_t alu) const
{
    if (!target_.is_valid_vxm_alu(alu))
        throw std::out_of_range("invalid VXM resource");
    return {PhysicalResourceKind::VxmAlu, alu};
}

PhysicalResourceId LPUResourceModel::sxm(int64_t hemisphere) const
{
    if (hemisphere < 0 || hemisphere >= target_.memory().hemispheres)
        throw std::out_of_range("invalid SXM resource");
    return {PhysicalResourceKind::SxmUnit, hemisphere};
}

PhysicalResourceId LPUResourceModel::stream(
    target::StreamDirection direction, int64_t index) const
{
    if (index < 0 || index >= target_.streams().streams_per_direction)
        throw std::out_of_range("invalid stream resource");
    return {PhysicalResourceKind::StreamChannel,
        direction == target::StreamDirection::East ? 0 : 1, index};
}

} // namespace ftlpu::compiler::schedule
