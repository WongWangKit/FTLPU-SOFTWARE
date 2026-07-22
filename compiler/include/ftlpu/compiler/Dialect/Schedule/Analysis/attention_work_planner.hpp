#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace ftlpu::compiler::schedule {

struct AttentionShape {
    int64_t sequence_length;
    int64_t query_heads;
    int64_t kv_heads;
    int64_t head_dim;
};

struct AttentionWorkItem {
    int64_t query_head;
    int64_t kv_head;
    int64_t query_block;
    int64_t hemisphere;
    int64_t local_mxm;
};

struct AttentionWorkWave {
    std::vector<std::optional<AttentionWorkItem>> slots;
};

// Maps logical GQA work onto the target's two hemispheres and MXM slots. The
// mapping keeps query heads beside their shared KV head, while preserving a
// target-defined number of independent MXM work slots per wave.
class AttentionWorkPlanner {
public:
    AttentionWorkPlanner(AttentionShape shape, const target::LPUTargetModel& target);

    const std::vector<AttentionWorkWave>& qk_waves() const { return qk_waves_; }
    const std::vector<AttentionWorkWave>& pv_waves() const { return pv_waves_; }
    int64_t query_block_count() const { return query_block_count_; }

private:
    AttentionShape shape_;
    int64_t query_block_count_{0};
    std::vector<AttentionWorkWave> qk_waves_;
    std::vector<AttentionWorkWave> pv_waves_;
};

} // namespace ftlpu::compiler::schedule
