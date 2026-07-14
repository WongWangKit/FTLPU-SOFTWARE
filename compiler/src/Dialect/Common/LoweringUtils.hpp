#pragma once

#include "ftlpu/compiler/ir.hpp"

#include <cstddef>
#include <string>

namespace ftlpu::compiler::detail {

void require_dialect(const Module& module, Dialect expected, const char* pass_name);
std::size_t ceil_div(std::size_t lhs, std::size_t rhs);
bool is_consumed_by_swiglu(const Module& module, std::size_t matmul_index);
const SwigluOp* swiglu_consumer(const Module& module, std::size_t matmul_index, std::size_t& swiglu_index, std::string& port);

} // namespace ftlpu::compiler::detail
