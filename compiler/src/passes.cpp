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

Module lower_stablehlo_to_kernel(const Module& module)
{
    require_dialect(module, Dialect::StableHlo, "stablehlo-to-kernel");
    Module out;
    out.dialect = Dialect::Kernel;
    out.matmuls = module.matmuls;

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
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

Module lower_kernel_to_tensor(const Module& module, std::size_t tile_size)
{
    require_dialect(module, Dialect::Kernel, "kernel-to-tensor");
    Module out;
    out.dialect = Dialect::Tensor;
    out.matmuls = module.matmuls;

    std::ostringstream os;
    os << "// FTLPU tensor IR lowered from ftlpu.kernel.\n";
    os << "// This layer assigns tensors and tiles to MEM address ranges.\n";
    os << "module {\n";
    os << "  ftlpu.tensor.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        const auto m_tiles = ceil_div(op.m, tile_size);
        const auto n_tiles = ceil_div(op.n, tile_size);
        const auto k_tiles = ceil_div(op.k, tile_size);
        const auto a_bytes = op.m * op.k;
        const auto b_bytes = op.k * op.n;
        const auto c_bytes = op.m * op.n * 4;
        const auto a_addr = std::size_t {0};
        const auto b_addr = a_addr + a_bytes;
        const auto c_addr = b_addr + b_bytes;
        os << "    %a" << i << " = ftlpu.tensor.mem_buffer @A" << i
           << " {mem_space = \"MEM\", base = " << a_addr << ", bytes = " << a_bytes
           << ", shape = [" << op.m << ", " << op.k << "], dtype = \"" << op.lhs_type
           << "\", layout = \"row_major\"}\n";
        os << "    %b" << i << " = ftlpu.tensor.mem_buffer @B" << i
           << " {mem_space = \"MEM\", base = " << b_addr << ", bytes = " << b_bytes
           << ", shape = [" << op.k << ", " << op.n << "], dtype = \"" << op.rhs_type
           << "\", layout = \"row_major\"}\n";
        os << "    %c" << i << " = ftlpu.tensor.mem_buffer @C" << i
           << " {mem_space = \"MEM\", base = " << c_addr << ", bytes = " << c_bytes
           << ", shape = [" << op.m << ", " << op.n << "], dtype = \"" << op.acc_type
           << "\", layout = \"row_major\"}\n";
        os << "    ftlpu.tensor.tile_plan @matmul" << i
           << " {kernel = @mxm_matmul" << i << ", tile_m = " << tile_size
           << ", tile_n = " << tile_size << ", tile_k = " << tile_size
           << ", m_tiles = " << m_tiles << ", n_tiles = " << n_tiles
           << ", k_tiles = " << k_tiles << "}\n";
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
    os << "// This layer binds each logical stream to source, sink, and stream register id.\n";
    os << "module {\n";
    os << "  ftlpu.stream.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        const auto m_tiles = ceil_div(op.m, tile_size);
        const auto n_tiles = ceil_div(op.n, tile_size);
        const auto k_tiles = ceil_div(op.k, tile_size);
        os << "    ftlpu.stream.matmul_grid @matmul" << i
           << " {m = " << op.m << ", n = " << op.n << ", k = " << op.k
           << ", tile_m = " << tile_size << ", tile_n = " << tile_size
           << ", tile_k = " << tile_size << ", m_tiles = " << m_tiles
           << ", n_tiles = " << n_tiles << ", k_tiles = " << k_tiles << "}\n";
        for (std::size_t mt = 0; mt < m_tiles; ++mt) {
            for (std::size_t nt = 0; nt < n_tiles; ++nt) {
                for (std::size_t kt = 0; kt < k_tiles; ++kt) {
                    const auto mxm = (mt + nt) % 2;
                    const auto lhs_stream = (mt * k_tiles + kt) % 64;
                    const auto rhs_stream = (nt * k_tiles + kt + 32) % 64;
                    const auto out_stream = (mt * n_tiles + nt) % 64;
                    os << "    ftlpu.stream.channel @op" << i << "_m" << mt << "_n" << nt << "_k" << kt << "_lhs"
                       << " {stream_id = " << lhs_stream << ", sreg = " << (lhs_stream % 12)
                       << ", source = \"MEM:A" << i << "\", sink = \"MXM" << mxm << ":lhs\", "
                       << "start_addr = " << (mt * tile_size * op.k + kt * tile_size)
                       << ", bytes = " << (tile_size * tile_size) << "}\n";
                    os << "    ftlpu.stream.channel @op" << i << "_m" << mt << "_n" << nt << "_k" << kt << "_rhs"
                       << " {stream_id = " << rhs_stream << ", sreg = " << (rhs_stream % 12)
                       << ", source = \"MEM:B" << i << "\", sink = \"MXM" << mxm << ":rhs\", "
                       << "start_addr = " << (op.m * op.k + kt * tile_size * op.n + nt * tile_size)
                       << ", bytes = " << (tile_size * tile_size) << "}\n";
                    if (kt + 1 == k_tiles) {
                        os << "    ftlpu.stream.channel @op" << i << "_m" << mt << "_n" << nt << "_out"
                           << " {stream_id = " << out_stream << ", sreg = " << (out_stream % 12)
                           << ", source = \"MXM" << mxm << ":output\", sink = \"MEM:C" << i << "\", "
                           << "start_addr = " << (op.m * op.k + op.k * op.n + mt * tile_size * op.n + nt * tile_size)
                           << ", bytes = " << (tile_size * tile_size * 4) << "}\n";
                    }
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
        if (pass == "stablehlo-to-kernel") {
            module = lower_stablehlo_to_kernel(module);
        } else if (pass == "kernel-to-tensor") {
            module = lower_kernel_to_tensor(module);
        } else if (pass == "stablehlo-to-tensor") {
            module = lower_kernel_to_tensor(lower_stablehlo_to_kernel(module));
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
