#pragma once

#include <cstdint>

namespace ftlpu::compiler::schedule {

// Query blocks reuse one resident Value weight block, but each block still
// occupies a complete MXM issue window. Keep this shape rule outside the
// emitter so every paged-PV path uses the same non-overlapping timeline.
inline constexpr int64_t attention_pv_query_compute_cycle(
    int64_t first_compute_cycle, int64_t query_block, int64_t tile_rows)
{
    return first_compute_cycle + query_block * tile_rows;
}

} // namespace ftlpu::compiler::schedule
