#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include <cstdint>
#include <string>

namespace ftlpu::compiler::schedule {

class LPUResourceModel {
public:
    explicit LPUResourceModel(const target::LPUTargetModel& target)
        : target_(target) {}

    std::string mem_slice(int64_t hemisphere, int64_t slice) const;
    std::string mxm(int64_t unit) const;
    std::string mxm_weight_buffer(int64_t unit, int64_t buffer) const;
    std::string vxm_alu(int64_t alu) const;
    std::string sxm(int64_t hemisphere) const;
    std::string stream(target::StreamDirection direction, int64_t index) const;

private:
    const target::LPUTargetModel& target_;
};

} // namespace ftlpu::compiler::schedule
