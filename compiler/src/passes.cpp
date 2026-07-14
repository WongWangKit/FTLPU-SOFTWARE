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

Module lower_tensor_to_stream(const Module& module, std::size_t south_to_north_tiles)
{
    require_dialect(module, Dialect::Tensor, "tensor-to-stream");
    Module out;
    out.dialect = Dialect::Stream;
    out.matmuls = module.matmuls;

    std::ostringstream os;
    os << "// FTLPU stream IR lowered from ftlpu.tensor.\n";
    os << "// This layer binds each long logical stream to source, sink, and stream register ids.\n";
    os << "module {\n";
    os << "  ftlpu.stream.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        os << "    ftlpu.stream.matmul_grid @matmul" << i
           << " {m = " << op.m << ", n = " << op.n << ", k = " << op.k
           << ", vector_lanes = 16, south_to_north_tiles = " << south_to_north_tiles
           << ", mxm_count = 2}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_lhs"
           << " {stream_ids = [0..15], sregs = [0..11], source = \"MEM:A" << i
           << "\", sink = \"MXM*:lhs\", start_addr = 0, bytes = " << (op.m * op.k)
           << ", vector = \"south_to_north\"}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_rhs"
           << " {stream_ids = [32..47], sregs = [0..11], source = \"MEM:B" << i
           << "\", sink = \"MXM*:rhs\", start_addr = " << (op.m * op.k)
           << ", bytes = " << (op.k * op.n) << ", vector = \"south_to_north\"}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_out"
           << " {stream_ids = [48..63], sregs = [0..11], source = \"MXM*:output\", sink = \"MEM:C" << i
           << "\", start_addr = " << (op.m * op.k + op.k * op.n)
           << ", bytes = " << (op.m * op.n * 4) << ", vector = \"south_to_north\"}\n";
    }
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

Module lower_stream_to_schedule(const Module& module, std::size_t south_to_north_tiles)
{
    require_dialect(module, Dialect::Stream, "stream-to-schedule");
    Module out;
    out.dialect = Dialect::Schedule;
    out.matmuls = module.matmuls;

    std::ostringstream os;
    os << "// FTLPU schedule IR lowered from ftlpu.stream.\n";
    os << "// This is a stage-level queue/timeline layer before .ftlpu emission.\n";
    os << "module {\n";
    os << "  ftlpu.schedule.program @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        os << "    ftlpu.schedule.mem_read @matmul" << i << "_read"
           << " {cycle = 0, sources = [@A" << i << ", @B" << i
           << "], streams = [0..15, 32..47], sregs = [0..11], bytes = ["
           << (op.m * op.k) << ", " << (op.k * op.n)
           << "], vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
        os << "    ftlpu.schedule.mxm_load @matmul" << i << "_load"
           << " {cycle = 8, mxms = [0, 1], lhs_streams = [0..15], rhs_streams = [32..47]}\n";
        os << "    ftlpu.schedule.mxm_compute @matmul" << i << "_compute"
           << " {cycle = 16, mxms = [0, 1], m = " << op.m << ", n = " << op.n
           << ", k = " << op.k << ", vector_lanes = 16, south_to_north_tiles = "
           << south_to_north_tiles << ", accumulate = true}\n";
        os << "    ftlpu.schedule.mem_write @matmul" << i << "_write"
           << " {cycle = 32, dest = @C" << i
           << ", streams = [48..63], sregs = [0..11], bytes = " << (op.m * op.n * 4)
           << ", vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
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
