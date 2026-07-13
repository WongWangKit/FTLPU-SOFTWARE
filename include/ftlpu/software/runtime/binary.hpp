#pragma once

#include "ftlpu/software/runtime/icu_program.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace ftlpu::software::runtime {

struct BinaryProgram {
    std::size_t max_cycle{0};
    std::vector<QueueProgram> queues{};
};

void write_binary_program(const BinaryProgram& program, const std::filesystem::path& path);
BinaryProgram read_binary_program(const std::filesystem::path& path);

} // namespace ftlpu::software::runtime
