#pragma once

#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include <cstdint>
#include <vector>

namespace ftlpu::compiler::schedule {

enum class AttentionProjection { Query, Key, Value };

struct AttentionProjectionShape {
    int64_t sequence_length;
    int64_t hidden_size;
    int64_t query_heads;
    int64_t kv_heads;
    int64_t head_dim;
};

struct AttentionProjectionWork {
    AttentionProjection projection;
    int64_t head_group;
    int64_t reduction_block;
    int64_t token_block;
    int64_t hemisphere;
    bool final_reduction;
};

// Models the CModel's W8A16 projection traversal. It describes work only;
// Schedule lowering owns the concrete MEM/MXM/VXM instruction emission.
class AttentionProjectionPlanner {
public:
    AttentionProjectionPlanner(AttentionProjectionShape shape,
        const target::LPUTargetModel& target);

    const std::vector<AttentionProjectionWork>& work() const { return work_; }
    int64_t projection_cycle_count() const { return projection_cycle_count_; }

private:
    std::vector<AttentionProjectionWork> work_;
    int64_t projection_cycle_count_{0};
};

} // namespace ftlpu::compiler::schedule
