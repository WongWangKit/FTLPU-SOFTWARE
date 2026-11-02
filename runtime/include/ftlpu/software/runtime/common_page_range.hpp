#pragma once

#include <cstddef>
#include <cstdint>

namespace ftlpu::software::runtime {

// One actual packed MEM segment for one final page-in instance. Ranges from
// different instances may intentionally reuse SRAM; instance and stop retain
// the loading order needed by the storage proof.
struct CommonPageLoadRange {
    std::size_t queue{0};
    std::uint32_t row_begin{0};
    std::uint32_t row_count{0};
    std::uint32_t binding_index{0};
    std::uint32_t instance{0};
    std::uint32_t stop{0};
    std::uint64_t ready{0};
    std::uint64_t release{0};
    std::uint32_t launch{0};
};

} // namespace ftlpu::software::runtime
