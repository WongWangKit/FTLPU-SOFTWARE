#include "ftlpu/compiler/Pipelines/pipelines.hpp"

#include "Dialect/Common/LoweringUtils.hpp"
#include "ftlpu/compiler/Dialect/Tensor/Analysis/memory_layout.hpp"

#include <map>
#include <sstream>
#include <stdexcept>

namespace ftlpu::compiler::pipeline {
namespace {

struct AllocatedTensorSet {
    std::map<std::string, tensor::MemoryAllocation> tensors;
};

AllocatedTensorSet allocate_tensors(const Module& module)
{
    auto allocator = tensor::MemoryLayout(tensor::Hemisphere::East);
    auto allocated = AllocatedTensorSet {};
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        allocated.tensors.emplace("A" + std::to_string(i), allocator.allocate("A" + std::to_string(i), op.m * op.k));
        allocated.tensors.emplace("B" + std::to_string(i), allocator.allocate("B" + std::to_string(i), op.k * op.n));
        allocated.tensors.emplace("C" + std::to_string(i), allocator.allocate("C" + std::to_string(i), op.m * op.n * 4));
    }
    for (std::size_t i = 0; i < module.swiglus.size(); ++i) {
        const auto& op = module.swiglus[i];
        allocated.tensors.emplace("S" + std::to_string(i), allocator.allocate("S" + std::to_string(i), op.rows * op.columns));
    }
    return allocated;
}

const tensor::MemoryAllocation& tensor_alloc(const AllocatedTensorSet& allocated, const std::string& name)
{
    const auto it = allocated.tensors.find(name);
    if (it == allocated.tensors.end()) {
        throw std::runtime_error("missing allocated tensor: " + name);
    }
    return it->second;
}

} // namespace

Module lower_kernel_to_tensor(const Module& module, const target::TargetBackend&, std::size_t tile_size)
{
    detail::require_dialect(module, Dialect::Kernel, "kernel-to-tensor");
    Module out;
    out.dialect = Dialect::Tensor;
    out.matmuls = module.matmuls;
    out.swiglus = module.swiglus;
    const auto allocated = allocate_tensors(module);

    std::ostringstream os;
    os << "// FTLPU tensor IR lowered from ftlpu.kernel.\n";
    os << "// This layer assigns tensors and tiles to MEM address ranges.\n";
    os << "module {\n";
    os << "  ftlpu.tensor.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        const auto m_tiles = detail::ceil_div(op.m, tile_size);
        const auto n_tiles = detail::ceil_div(op.n, tile_size);
        const auto k_tiles = detail::ceil_div(op.k, tile_size);
        const auto a_bytes = op.m * op.k;
        const auto b_bytes = op.k * op.n;
        const auto c_bytes = op.m * op.n * 4;
        const auto& a_alloc = tensor_alloc(allocated, "A" + std::to_string(i));
        const auto& b_alloc = tensor_alloc(allocated, "B" + std::to_string(i));
        const auto& c_alloc = tensor_alloc(allocated, "C" + std::to_string(i));
        os << "    %a" << i << " = ftlpu.tensor.mem_buffer @A" << i
           << " {mem_space = \"MEM\", allocation = " << tensor::format_allocation(a_alloc) << ", bytes = " << a_bytes
           << ", shape = [" << op.m << ", " << op.k << "], dtype = \"" << op.lhs_type
           << "\", layout = \"row_major\"}\n";
        os << "    %b" << i << " = ftlpu.tensor.mem_buffer @B" << i
           << " {mem_space = \"MEM\", allocation = " << tensor::format_allocation(b_alloc) << ", bytes = " << b_bytes
           << ", shape = [" << op.k << ", " << op.n << "], dtype = \"" << op.rhs_type
           << "\", layout = \"row_major\"}\n";
        os << "    %c" << i << " = ftlpu.tensor.mem_buffer @C" << i
           << " {mem_space = \"MEM\", allocation = " << tensor::format_allocation(c_alloc) << ", bytes = " << c_bytes
           << ", shape = [" << op.m << ", " << op.n << "], dtype = \"" << op.acc_type
           << "\", layout = \"row_major\"}\n";
        os << "    ftlpu.tensor.tile_plan @matmul" << i
           << " {kernel = @mxm_matmul" << i << ", tile_m = " << tile_size
           << ", tile_n = " << tile_size << ", tile_k = " << tile_size
           << ", m_tiles = " << m_tiles << ", n_tiles = " << n_tiles
           << ", k_tiles = " << k_tiles << "}\n";
    }
    for (std::size_t i = 0; i < module.swiglus.size(); ++i) {
        const auto& op = module.swiglus[i];
        const auto& s_alloc = tensor_alloc(allocated, "S" + std::to_string(i));
        os << "    %s" << i << " = ftlpu.tensor.mem_buffer @S" << i
           << " {mem_space = \"MEM\", allocation = " << tensor::format_allocation(s_alloc)
           << ", bytes = " << (op.rows * op.columns)
           << ", shape = [" << op.rows << ", " << op.columns
           << "], dtype = \"" << op.output_type << "\", layout = \"row_major\"}\n";
        os << "    ftlpu.tensor.post_op @swiglu" << i
           << " {gate = @matmul" << op.gate_matmul
           << ", up = @matmul" << op.up_matmul
           << ", output = @S" << i << "}\n";
    }
    os << "  }\n";
    os << "}\n";
    out.text = os.str();
    return out;
}

} // namespace ftlpu::compiler::pipeline
