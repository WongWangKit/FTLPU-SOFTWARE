#pragma once

#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/weight_prefetch_plan.hpp"

#include <filesystem>
#include <span>

namespace ftlpu::software::runtime {

// Serializes ICU queue timelines as pipeline-viewer CSV. Repeat, Repeat2D,
// macro-schedule and Repeat2D commands remain compact patterns that the viewer
// expands only for its visible cycle window.
void write_schedule_trace_csv(
    const BinaryProgram& program, const std::filesystem::path& path);
void write_schedule_trace_csv(const BinaryProgram& program,
    const std::filesystem::path& path,
    std::span<const WeightPrefetchPlan> physical_prefetches);

} // namespace ftlpu::software::runtime
