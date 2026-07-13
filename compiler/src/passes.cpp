#include "ftlpu/compiler/passes.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ftlpu::compiler {
namespace {

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

} // namespace

Module lower_stablehlo_to_tensor(const Module& module)
{
    require_dialect(module, Dialect::StableHlo, "stablehlo-to-tensor");
    Module out;
    out.dialect = Dialect::Tensor;
    out.matmuls = module.matmuls;

    std::ostringstream os;
    os << "// FTLPU tensor IR lowered from StableHLO.\n";
    os << "module {\n";
    os << "  ftlpu.tensor.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        os << "    %c" << i << " = ftlpu.tensor.matmul %a" << i << ", %b" << i << " {\n";
        os << "      m = " << op.m << ",\n";
        os << "      n = " << op.n << ",\n";
        os << "      k = " << op.k << ",\n";
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
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

Module lower_tensor_to_stream(const Module& module, std::size_t tile_size)
{
    require_dialect(module, Dialect::Tensor, "tensor-to-stream");
    Module out;
    out.dialect = Dialect::Stream;
    out.matmuls = module.matmuls;

    std::ostringstream os;
    os << "// FTLPU stream IR lowered from ftlpu.tensor.\n";
    os << "module {\n";
    os << "  ftlpu.stream.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        const auto m_tiles = ceil_div(op.m, tile_size);
        const auto n_tiles = ceil_div(op.n, tile_size);
        const auto k_tiles = ceil_div(op.k, tile_size);
        os << "    %a" << i << " = ftlpu.stream.memref @A" << i << " {addr = 0, layout = \"row_major\"}\n";
        os << "    %b" << i << " = ftlpu.stream.memref @B" << i << " {addr = " << (op.m * op.k)
           << ", layout = \"row_major\"}\n";
        os << "    %c" << i << " = ftlpu.stream.memref @C" << i << " {addr = " << (op.m * op.k + op.k * op.n)
           << ", layout = \"row_major\"}\n";
        os << "    ftlpu.stream.matmul_grid %a" << i << ", %b" << i << ", %c" << i
           << " {m = " << op.m << ", n = " << op.n << ", k = " << op.k
           << ", tile_m = " << tile_size << ", tile_n = " << tile_size
           << ", tile_k = " << tile_size << ", m_tiles = " << m_tiles
           << ", n_tiles = " << n_tiles << ", k_tiles = " << k_tiles << "}\n";
        for (std::size_t mt = 0; mt < m_tiles; ++mt) {
            for (std::size_t nt = 0; nt < n_tiles; ++nt) {
                for (std::size_t kt = 0; kt < k_tiles; ++kt) {
                    os << "    ftlpu.stream.matmul_tile @op" << i << "_m" << mt << "_n" << nt << "_k" << kt
                       << " {m_tile = " << mt << ", n_tile = " << nt << ", k_tile = " << kt
                       << ", mxm = " << ((mt + nt) % 2)
                       << ", lhs_stream = " << ((mt * k_tiles + kt) % 64)
                       << ", rhs_stream = " << ((nt * k_tiles + kt + 32) % 64)
                       << ", out_stream = " << ((mt * n_tiles + nt) % 64) << "}\n";
                }
            }
        }
    }
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

Module lower_stream_to_schedule(const Module& module, std::size_t tile_size)
{
    require_dialect(module, Dialect::Stream, "stream-to-schedule");
    Module out;
    out.dialect = Dialect::Schedule;
    out.matmuls = module.matmuls;

    std::ostringstream os;
    os << "// FTLPU schedule IR lowered from ftlpu.stream.\n";
    os << "// This is the low-level queue/timeline layer before .ftlpu emission.\n";
    os << "module {\n";
    os << "  ftlpu.schedule.program @main {\n";
    std::size_t cycle = 0;
    for (const auto& op : module.matmuls) {
        const auto m_tiles = ceil_div(op.m, tile_size);
        const auto n_tiles = ceil_div(op.n, tile_size);
        const auto k_tiles = ceil_div(op.k, tile_size);
        for (std::size_t mt = 0; mt < m_tiles; ++mt) {
            for (std::size_t nt = 0; nt < n_tiles; ++nt) {
                for (std::size_t kt = 0; kt < k_tiles; ++kt) {
                    const auto mxm = (mt + nt) % 2;
                    const auto lhs_addr = mt * tile_size * op.k + kt * tile_size;
                    const auto rhs_addr = kt * tile_size * op.n + nt * tile_size;
                    const auto out_addr = op.m * op.k + op.k * op.n + mt * tile_size * op.n + nt * tile_size;
                    os << "    ftlpu.schedule.mem_read @mem0 cycle " << cycle
                       << " addr " << lhs_addr << " bytes " << (tile_size * tile_size) << " stream 0\n";
                    os << "    ftlpu.schedule.mem_read @mem1 cycle " << (cycle + 1)
                       << " addr " << rhs_addr << " bytes " << (tile_size * tile_size) << " stream 32\n";
                    os << "    ftlpu.schedule.mxm_load @mxm" << mxm << " cycle " << (cycle + 8)
                       << " lhs_stream 0 rhs_stream 32\n";
                    os << "    ftlpu.schedule.mxm_compute @mxm" << mxm << " cycle " << (cycle + 16)
                       << " m " << tile_size << " n " << tile_size << " k " << tile_size
                       << " accumulate " << (kt != 0 ? "true" : "false") << "\n";
                    if (kt + 1 == k_tiles) {
                        os << "    ftlpu.schedule.mxm_output @mxm" << mxm << " cycle " << (cycle + 28)
                           << " stream 48\n";
                        os << "    ftlpu.schedule.mem_write @mem2 cycle " << (cycle + 36)
                           << " addr " << out_addr << " bytes " << (tile_size * tile_size * 4)
                           << " stream 48\n";
                    }
                    cycle += 48;
                }
            }
        }
    }
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

std::vector<std::string> split_pipeline(const std::string& pipeline)
{
    std::vector<std::string> passes;
    std::string current;
    for (const auto ch : pipeline) {
        if (ch == ',') {
            if (!current.empty()) {
                passes.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        passes.push_back(current);
    }
    return passes;
}

Module run_pipeline(Module module, const std::vector<std::string>& pass_names)
{
    for (const auto& pass : pass_names) {
        if (pass == "stablehlo-to-tensor") {
            module = lower_stablehlo_to_tensor(module);
        } else if (pass == "tensor-to-stream") {
            module = lower_tensor_to_stream(module);
        } else if (pass == "stream-to-schedule") {
            module = lower_stream_to_schedule(module);
        } else {
            throw std::runtime_error("unknown compiler pass: " + pass);
        }
    }
    return module;
}

} // namespace ftlpu::compiler
