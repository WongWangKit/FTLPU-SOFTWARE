#pragma once

#include "ftlpu/software/runtime/binary.hpp"

#include <filesystem>

namespace ftlpu::software::runtime {

// Expands the serialized ICU queue timelines into CModel-compatible trace
// events with columns: start,end,resource,detail.
void write_schedule_trace_csv(
    const BinaryProgram& program, const std::filesystem::path& path);

} // namespace ftlpu::software::runtime
