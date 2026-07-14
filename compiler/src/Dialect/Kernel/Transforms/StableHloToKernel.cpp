#include "ftlpu/compiler/Pipelines/pipelines.hpp"

#include "Dialect/Common/LoweringUtils.hpp"

#include <sstream>

namespace ftlpu::compiler::pipeline {

Module lower_stablehlo_to_kernel(const Module& module, const target::TargetBackend&)
{
    detail::require_dialect(module, Dialect::StableHlo, "stablehlo-to-kernel");
    Module out;
    out.dialect = Dialect::Kernel;
    out.matmuls = module.matmuls;
    out.swiglus = module.swiglus;

    std::ostringstream os;
    os << "// FTLPU kernel IR lowered from StableHLO.\n";
    os << "// This layer maps frontend ops to concrete LPU functional units.\n";
    os << "module {\n";
    os << "  ftlpu.kernel.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        os << "    %c" << i << " = ftlpu.kernel.mxm_matmul %a" << i << ", %b" << i << " {\n";
        os << "      unit = \"MXM\",\n";
        os << "      m = " << op.m << ",\n";
        os << "      n = " << op.n << ",\n";
        os << "      k = " << op.k << ",\n";
        os << "      mxm_count = 2,\n";
        os << "      tile_m = 20,\n";
        os << "      tile_n = 20,\n";
        os << "      tile_k = 20,\n";
        os << "      lhs_layout = \"row_major\",\n";
        os << "      rhs_layout = \"row_major\",\n";
        os << "      out_layout = \"row_major\",\n";
        os << "      lhs_dtype = \"" << op.lhs_type << "\",\n";
        os << "      rhs_dtype = \"" << op.rhs_type << "\",\n";
        os << "      acc_dtype = \"" << op.acc_type << "\"\n";
        os << "    } : (tensor<" << op.m << "x" << op.k << "x" << op.lhs_type << ">, tensor<"
           << op.k << "x" << op.n << "x" << op.rhs_type << ">) -> tensor<"
           << op.m << "x" << op.n << "x" << op.acc_type << ">\n";
    }
    for (std::size_t i = 0; i < module.swiglus.size(); ++i) {
        const auto& op = module.swiglus[i];
        os << "    %swiglu" << i << " = ftlpu.kernel.swiglu %c" << op.gate_matmul
           << ", %c" << op.up_matmul << " {\n";
        os << "      units = [\"MXM0\", \"MXM1\", \"VXM\"],\n";
        os << "      rows = " << op.rows << ",\n";
        os << "      columns = " << op.columns << ",\n";
        os << "      gate_scale = 0.00048828125,\n";
        os << "      up_scale = 0.00048828125,\n";
        os << "      output_scale = 0.0078125,\n";
        os << "      output_dtype = \"" << op.output_type << "\"\n";
        os << "    } : (tensor<" << op.rows << "x" << op.columns << "x" << op.input_type
           << ">, tensor<" << op.rows << "x" << op.columns << "x" << op.input_type
           << ">) -> tensor<" << op.rows << "x" << op.columns << "x" << op.output_type << ">\n";
    }
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

} // namespace ftlpu::compiler::pipeline
