#pragma once

#include "ftlpu/software/runtime/icu_program.hpp"

#include <string>

namespace ftlpu::software::runtime {

IcuProgram parse_lpu_ir(const std::string& text);

} // namespace ftlpu::software::runtime
