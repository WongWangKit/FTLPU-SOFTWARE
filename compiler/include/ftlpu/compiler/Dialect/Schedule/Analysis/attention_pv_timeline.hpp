#pragma once

#include <cstdint>

namespace ftlpu::compiler::schedule {

// Reusing a resident Value block does not allow query blocks to share the
// same physical MXM issue window.
inline constexpr int64_t attention_pv_query_compute_cycle(
    int64_t first_compute_cycle, int64_t query_block, int64_t tile_rows)
{
    return first_compute_cycle + query_block * tile_rows;
}

} // namespace ftlpu::compiler::schedule
