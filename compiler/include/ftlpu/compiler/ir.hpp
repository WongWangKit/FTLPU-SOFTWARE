#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ftlpu::compiler {

enum class Dialect {
    StableHlo,
    Kernel,
    Tensor,
    Stream,
    Schedule,
};

struct MatmulOp {
    std::size_t m{0};
    std::size_t n{0};
    std::size_t k{0};
    std::string lhs_type;
    std::string rhs_type;
    std::string acc_type;
};

struct Module {
    Dialect dialect{Dialect::StableHlo};
    std::vector<MatmulOp> matmuls;
    std::string text;
};

const char* dialect_name(Dialect dialect);
Module parse_stablehlo_module(const std::string& text);
std::string print_module(const Module& module);

} // namespace ftlpu::compiler
