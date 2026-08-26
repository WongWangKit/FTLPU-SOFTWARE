#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <filesystem>

namespace ftlpu::software::runtime {

// Serializes ICU queue timelines as pipeline-viewer CSV. Repeat, Repeat2D,
// macro-schedule, and Loop commands remain compact patterns that the viewer
// expands only for its visible cycle window.
void write_schedule_trace_csv(
    const BinaryProgram& program, const std::filesystem::path& path);

} // namespace ftlpu::software::runtime
