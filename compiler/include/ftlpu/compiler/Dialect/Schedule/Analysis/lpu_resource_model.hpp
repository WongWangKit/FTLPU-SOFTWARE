#pragma once

#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include <cstdint>
#include <string>

namespace ftlpu::compiler::schedule {

enum class PhysicalQueueKind : std::uint8_t {
    Mem,
    MxmLoad,
    MxmCompute,
    MxmDequant,
    Vxm,
    SxmTranspose,
    SxmPermute,
};

struct PhysicalQueueId {
    PhysicalQueueKind kind;
    int64_t index;

    bool operator==(const PhysicalQueueId&) const = default;
};

class LPUResourceModel {
public:
    explicit LPUResourceModel(const target::LPUTargetModel& target)
        : target_(target) {}

    PhysicalQueueId mem_queue(
        int64_t hemisphere, int64_t slice, int64_t bank = 0) const;
    PhysicalQueueId mxm_load_queue(int64_t unit) const;
    PhysicalQueueId mxm_compute_queue(int64_t unit) const;
    PhysicalQueueId mxm_dequant_queue(int64_t unit) const;
    PhysicalQueueId vxm_queue(int64_t queue) const;
    PhysicalQueueId sxm_queue(bool transpose, int64_t hemisphere) const;
    PhysicalResourceId queue_resource(PhysicalQueueId queue) const;

    PhysicalResourceId mem_read_port(int64_t hemisphere, int64_t slice) const;
    PhysicalResourceId mem_read_port(
        int64_t hemisphere, int64_t slice, int64_t bank) const;
    PhysicalResourceId mem_write_port(int64_t hemisphere, int64_t slice) const;
    PhysicalResourceId mem_write_port(
        int64_t hemisphere, int64_t slice, int64_t bank) const;
    PhysicalResourceId mem_icu(
        int64_t hemisphere, int64_t slice, int64_t bank) const;
    PhysicalResourceId mxm(int64_t unit) const;
    PhysicalResourceId mxm_weight_buffer(int64_t unit, int64_t buffer) const;
    PhysicalResourceId vxm_alu(int64_t alu) const;
    PhysicalResourceId sxm(int64_t hemisphere) const;
    PhysicalResourceId stream(
        target::StreamDirection direction, int64_t index) const;

private:
    const target::LPUTargetModel& target_;
};

} // namespace ftlpu::compiler::schedule
