#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"

#include <stdexcept>

namespace ftlpu::compiler::schedule {

AttentionProjectionPlanner::AttentionProjectionPlanner(AttentionProjectionShape shape,
    const target::LPUTargetModel& target)
{
    const int64_t tile = target.throughput().mxm_rows;
    if (shape.sequence_length <= 0 || shape.hidden_size <= 0 || shape.head_dim <= 0
        || shape.sequence_length % tile || shape.hidden_size % tile || shape.head_dim % tile)
        throw std::invalid_argument("attention projection shape must be MXM-tile aligned");
    const int64_t reductions = shape.hidden_size / tile;
    const int64_t token_blocks = shape.sequence_length / tile;
    const int64_t projection_heads[] = {shape.query_heads, shape.kv_heads, shape.kv_heads};
    for (int64_t projection = 0; projection < 3; ++projection) {
        for (int64_t group = 0; group < projection_heads[projection]; group += 2) {
            for (int64_t reduction = 0; reduction < reductions; ++reduction) {
                for (int64_t token = 0; token < token_blocks; ++token) {
                    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                        if (group + hemisphere >= projection_heads[projection]) continue;
                        work_.push_back({static_cast<AttentionProjection>(projection), group,
                            reduction, token, hemisphere, reduction + 1 == reductions});
                    }
                }
                projection_cycle_count_ += 32 + token_blocks
                    * (reduction + 1 == reductions ? 64 : target.mxm_block_issue_interval()) + 16;
            }
        }
    }
}

} // namespace ftlpu::compiler::schedule
