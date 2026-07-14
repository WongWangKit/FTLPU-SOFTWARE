#include "Dialect/Common/LoweringUtils.hpp"

#include <sstream>
#include <stdexcept>

namespace ftlpu::compiler::detail {

void require_dialect(const Module& module, Dialect expected, const char* pass_name)
{
    if (module.dialect != expected) {
        std::ostringstream os;
        os << pass_name << " expected " << dialect_name(expected)
           << " but got " << dialect_name(module.dialect);
        throw std::runtime_error(os.str());
    }
}

std::size_t ceil_div(std::size_t lhs, std::size_t rhs)
{
    return (lhs + rhs - 1) / rhs;
}

bool is_consumed_by_swiglu(const Module& module, std::size_t matmul_index)
{
    for (const auto& swiglu : module.swiglus) {
        if (swiglu.gate_matmul == matmul_index || swiglu.up_matmul == matmul_index) {
            return true;
        }
    }
    return false;
}

const SwigluOp* swiglu_consumer(const Module& module, std::size_t matmul_index, std::size_t& swiglu_index, std::string& port)
{
    for (std::size_t i = 0; i < module.swiglus.size(); ++i) {
        const auto& op = module.swiglus[i];
        if (op.gate_matmul == matmul_index) {
            swiglu_index = i;
            port = "gate";
            return &op;
        }
        if (op.up_matmul == matmul_index) {
            swiglu_index = i;
            port = "up";
            return &op;
        }
    }
    return nullptr;
}

} // namespace ftlpu::compiler::detail
