#include "ftlpu/compiler/ir.hpp"

#include <regex>
#include <sstream>
#include <stdexcept>

namespace ftlpu::compiler {
namespace {

const std::regex kStableHloMatmulPattern(
    R"(stablehlo\.dot_general[\s\S]*?:\s*\(tensor<([0-9]+)x([0-9]+)x([a-z0-9]+)>,\s*tensor<([0-9]+)x([0-9]+)x([a-z0-9]+)>\)\s*->\s*tensor<([0-9]+)x([0-9]+)x([a-z0-9]+)>)",
    std::regex::ECMAScript);

} // namespace

const char* dialect_name(Dialect dialect)
{
    switch (dialect) {
    case Dialect::StableHlo:
        return "stablehlo";
    case Dialect::Kernel:
        return "ftlpu.kernel";
    case Dialect::Tensor:
        return "ftlpu.tensor";
    case Dialect::Stream:
        return "ftlpu.stream";
    case Dialect::Schedule:
        return "ftlpu.schedule";
    }
    return "unknown";
}

Module parse_stablehlo_module(const std::string& text)
{
    Module module;
    module.dialect = Dialect::StableHlo;
    module.text = text;

    auto begin = std::sregex_iterator(text.begin(), text.end(), kStableHloMatmulPattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const auto& match = *it;
        const auto m = static_cast<std::size_t>(std::stoull(match[1].str()));
        const auto k = static_cast<std::size_t>(std::stoull(match[2].str()));
        const auto k2 = static_cast<std::size_t>(std::stoull(match[4].str()));
        const auto n = static_cast<std::size_t>(std::stoull(match[5].str()));
        const auto m2 = static_cast<std::size_t>(std::stoull(match[7].str()));
        const auto n2 = static_cast<std::size_t>(std::stoull(match[8].str()));
        if (m != m2 || n != n2 || k != k2) {
            throw std::runtime_error("inconsistent stablehlo.dot_general matmul shape");
        }
        module.matmuls.push_back(MatmulOp {
            m,
            n,
            k,
            match[3].str(),
            match[6].str(),
            match[9].str(),
        });
    }
    if (module.matmuls.empty()) {
        throw std::runtime_error("expected at least one stablehlo.dot_general matmul");
    }
    if (text.find("stablehlo.logistic") != std::string::npos
        && text.find("stablehlo.multiply") != std::string::npos
        && module.matmuls.size() >= 2) {
        const auto& gate = module.matmuls[0];
        const auto& up = module.matmuls[1];
        if (gate.m == up.m && gate.n == up.n && gate.k == up.k) {
            module.swiglus.push_back(SwigluOp {0, 1, gate.m, gate.n, gate.acc_type, "i8"});
        }
    }
    return module;
}

std::string print_module(const Module& module)
{
    return module.text;
}

} // namespace ftlpu::compiler
