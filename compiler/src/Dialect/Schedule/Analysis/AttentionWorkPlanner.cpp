#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"

#include <stdexcept>

namespace ftlpu::compiler::schedule {

AttentionWorkPlanner::AttentionWorkPlanner(
    AttentionShape shape, const target::LPUTargetModel& target)
    : shape_(shape)
{
    const auto& throughput = target.throughput();
    const auto& memory = target.memory();
    if (shape_.sequence_length <= 0 || shape_.query_heads <= 0 || shape_.kv_heads <= 0
        || shape_.head_dim <= 0 || shape_.query_heads % shape_.kv_heads != 0
        || shape_.sequence_length % throughput.mxm_rows != 0
        || shape_.head_dim % throughput.mxm_rows != 0)
        throw std::invalid_argument("attention shape is not tile-aligned GQA");

    query_block_count_ = shape_.sequence_length / throughput.mxm_rows;
    const int64_t groups = shape_.query_heads / shape_.kv_heads;
    const int64_t slots_per_wave = memory.hemispheres * throughput.mxms_per_hemisphere;
    if (slots_per_wave <= 0) throw std::invalid_argument("target has no MXM slots");

    // Generate one work item per Q head and Q tile. A wave owns each physical
    // (hemisphere, MXM) slot at most once, which is the queue exclusivity rule
    // needed before the plan can be expanded into ICU commands.
    for (int64_t query_block = 0; query_block < query_block_count_; ++query_block) {
        std::vector<AttentionWorkWave> block_waves;
        std::vector<int64_t> next_local_mxm(static_cast<std::size_t>(memory.hemispheres), 0);
        for (int64_t query_head = 0; query_head < shape_.query_heads; ++query_head) {
            const int64_t kv_head = query_head / groups;
            const int64_t hemisphere = kv_head % memory.hemispheres;
            const int64_t local_mxm = next_local_mxm[static_cast<std::size_t>(hemisphere)]
                % throughput.mxms_per_hemisphere;
            ++next_local_mxm[static_cast<std::size_t>(hemisphere)];
            const int64_t slot = hemisphere * throughput.mxms_per_hemisphere + local_mxm;
            auto work = AttentionWorkItem {query_head, kv_head, query_block, hemisphere, local_mxm};
            bool placed = false;
            for (AttentionWorkWave& wave : block_waves) {
                if (!wave.slots[static_cast<std::size_t>(slot)]) {
                    wave.slots[static_cast<std::size_t>(slot)] = work;
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                AttentionWorkWave wave;
                wave.slots.resize(static_cast<std::size_t>(slots_per_wave));
                wave.slots[static_cast<std::size_t>(slot)] = work;
                block_waves.push_back(std::move(wave));
            }
        }
        qk_waves_.insert(qk_waves_.end(), block_waves.begin(), block_waves.end());
    }
    // PV consumes exactly the QK score tiles, so it shares the same logical work plan.
    pv_waves_ = qk_waves_;
}

} // namespace ftlpu::compiler::schedule
