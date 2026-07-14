#include "ftlpu/compiler/passes.hpp"
#include "ftlpu/compiler/target_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
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

struct AllocatedTensorSet {
    std::map<std::string, MemoryAllocation> tensors;
};

AllocatedTensorSet allocate_tensors(const Module& module)
{
    auto allocator = MemoryAllocator(Hemisphere::East);
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

const MemoryAllocation& tensor_alloc(const AllocatedTensorSet& allocated, const std::string& name)
{
    const auto it = allocated.tensors.find(name);
    if (it == allocated.tensors.end()) {
        throw std::runtime_error("missing allocated tensor: " + name);
    }
    return it->second;
}

} // namespace

Module lower_stablehlo_to_kernel(const Module& module)
{
    require_dialect(module, Dialect::StableHlo, "stablehlo-to-kernel");
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

Module lower_kernel_to_tensor(const Module& module, std::size_t tile_size)
{
    require_dialect(module, Dialect::Kernel, "kernel-to-tensor");
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
        const auto m_tiles = ceil_div(op.m, tile_size);
        const auto n_tiles = ceil_div(op.n, tile_size);
        const auto k_tiles = ceil_div(op.k, tile_size);
        const auto a_bytes = op.m * op.k;
        const auto b_bytes = op.k * op.n;
        const auto c_bytes = op.m * op.n * 4;
        const auto& a_alloc = tensor_alloc(allocated, "A" + std::to_string(i));
        const auto& b_alloc = tensor_alloc(allocated, "B" + std::to_string(i));
        const auto& c_alloc = tensor_alloc(allocated, "C" + std::to_string(i));
        os << "    %a" << i << " = ftlpu.tensor.mem_buffer @A" << i
           << " {mem_space = \"MEM\", allocation = " << format_allocation(a_alloc) << ", bytes = " << a_bytes
           << ", shape = [" << op.m << ", " << op.k << "], dtype = \"" << op.lhs_type
           << "\", layout = \"row_major\"}\n";
        os << "    %b" << i << " = ftlpu.tensor.mem_buffer @B" << i
           << " {mem_space = \"MEM\", allocation = " << format_allocation(b_alloc) << ", bytes = " << b_bytes
           << ", shape = [" << op.k << ", " << op.n << "], dtype = \"" << op.rhs_type
           << "\", layout = \"row_major\"}\n";
        os << "    %c" << i << " = ftlpu.tensor.mem_buffer @C" << i
           << " {mem_space = \"MEM\", allocation = " << format_allocation(c_alloc) << ", bytes = " << c_bytes
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
           << " {mem_space = \"MEM\", allocation = " << format_allocation(s_alloc)
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

Module lower_tensor_to_stream(const Module& module, std::size_t south_to_north_tiles)
{
    require_dialect(module, Dialect::Tensor, "tensor-to-stream");
    Module out;
    out.dialect = Dialect::Stream;
    out.matmuls = module.matmuls;
    out.swiglus = module.swiglus;
    auto scheduler = ResourceScheduler {};
    auto streams = StreamManager {};
    std::map<std::size_t, StreamBinding> matmul_output_streams;

    std::ostringstream os;
    os << "// FTLPU stream IR lowered from ftlpu.tensor.\n";
    os << "// This layer binds each long logical stream to source, sink, and stream register ids.\n";
    os << "module {\n";
    os << "  ftlpu.stream.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        const auto weight_cycle = scheduler.reserve("MEM.east.read", 0, south_to_north_tiles);
        const auto load_cycle = scheduler.reserve("MXM.load", weight_cycle + 1, south_to_north_tiles);
        const auto activation_cycle = scheduler.reserve("MEM.east.read", load_cycle + south_to_north_tiles, op.m);
        const auto compute_cycle = scheduler.reserve("MXM.compute", activation_cycle + mem_to_mxm_latency(32), op.m);
        const auto output_arrival_cycle = compute_cycle + mxm_to_vxm_latency();
        auto swiglu_index = std::size_t {0};
        auto swiglu_port = std::string {};
        const auto* const swiglu = swiglu_consumer(module, i, swiglu_index, swiglu_port);
        const auto consumer_cycle = swiglu == nullptr
            ? scheduler.reserve("MEM.east.write", output_arrival_cycle, op.m)
            : scheduler.reserve("VXM.input", output_arrival_cycle, op.m);
        const auto output_preferred_stream = swiglu == nullptr
            ? std::size_t {48}
            : (swiglu_port == "gate" ? std::size_t {32} : std::size_t {36});
        const auto output_consumer = swiglu == nullptr
            ? StreamEndpoint {"MEM", "C" + std::to_string(i)}
            : StreamEndpoint {"VXM", "swiglu" + std::to_string(swiglu_index), swiglu_port};
        const auto weight_stream = streams.allocate(
            "matmul" + std::to_string(i) + "_weight",
            StreamDirection::East,
            {"MEM", "B" + std::to_string(i)},
            {"MXM", "*", "weight"},
            weight_cycle,
            load_cycle,
            0);
        const auto activation_stream = streams.allocate(
            "matmul" + std::to_string(i) + "_activation",
            StreamDirection::East,
            {"MEM", "A" + std::to_string(i)},
            {"MXM", "*", "activation"},
            activation_cycle,
            compute_cycle,
            16);
        const auto output_stream = streams.allocate(
            "matmul" + std::to_string(i) + "_output",
            swiglu == nullptr ? StreamDirection::West : StreamDirection::East,
            {"MXM", "*", "output"},
            output_consumer,
            compute_cycle,
            consumer_cycle,
            output_preferred_stream);
        matmul_output_streams.emplace(i, output_stream);
        os << "    ftlpu.stream.matmul_grid @matmul" << i
           << " {m = " << op.m << ", n = " << op.n << ", k = " << op.k
           << ", vector_lanes = 16, south_to_north_tiles = " << south_to_north_tiles
           << ", mxm_count = 2}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_lhs"
           << " {stream_id = " << activation_stream.stream_id
           << ", direction = \"" << stream_direction_name(activation_stream.direction)
           << "\", producer = \"" << format_endpoint(activation_stream.producer)
           << "\", consumer = \"" << format_endpoint(activation_stream.consumer)
           << "\", produce_cycle = " << activation_stream.produce_cycle
           << ", consume_cycle = " << activation_stream.consume_cycle
           << ", latency = " << activation_stream.latency
           << ", bytes = " << (op.m * op.k) << ", vector = \"south_to_north\"}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_rhs"
           << " {stream_id = " << weight_stream.stream_id
           << ", direction = \"" << stream_direction_name(weight_stream.direction)
           << "\", producer = \"" << format_endpoint(weight_stream.producer)
           << "\", consumer = \"" << format_endpoint(weight_stream.consumer)
           << "\", produce_cycle = " << weight_stream.produce_cycle
           << ", consume_cycle = " << weight_stream.consume_cycle
           << ", latency = " << weight_stream.latency
           << ", bytes = " << (op.k * op.n) << ", vector = \"south_to_north\"}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_out"
           << " {stream_id = " << output_stream.stream_id
           << ", direction = \"" << stream_direction_name(output_stream.direction)
           << "\", producer = \"" << format_endpoint(output_stream.producer)
           << "\", consumer = \"" << format_endpoint(output_stream.consumer)
           << "\", produce_cycle = " << output_stream.produce_cycle
           << ", consume_cycle = " << output_stream.consume_cycle
           << ", latency = " << output_stream.latency
           << ", bytes = " << (op.m * op.n * 4) << ", vector = \"south_to_north\"}\n";
    }
    for (std::size_t i = 0; i < module.swiglus.size(); ++i) {
        const auto& op = module.swiglus[i];
        const auto gate_stream = matmul_output_streams.at(op.gate_matmul);
        const auto up_stream = matmul_output_streams.at(op.up_matmul);
        const auto vxm_cycle = std::max(gate_stream.consume_cycle, up_stream.consume_cycle);
        const auto write_cycle = scheduler.reserve("MEM.east.write", vxm_cycle + vxm_pipeline_latency(10) + vxm_to_mem_latency(40), op.rows);
        const auto output_stream = streams.allocate(
            "swiglu" + std::to_string(i) + "_output",
            StreamDirection::West,
            {"VXM", "swiglu" + std::to_string(i), "output"},
            {"MEM", "S" + std::to_string(i)},
            vxm_cycle + vxm_pipeline_latency(10),
            write_cycle,
            31);
        os << "    ftlpu.stream.vxm_swiglu @swiglu" << i
           << " {rows = " << op.rows << ", columns = " << op.columns
           << ", gate_stream = " << gate_stream.stream_id
           << ", up_stream = " << up_stream.stream_id
           << ", output_stream = " << output_stream.stream_id
           << ", input_consume_cycle = " << vxm_cycle
           << ", stages = 10, unit = \"VXM\"}\n";
        os << "    ftlpu.stream.channel @swiglu" << i << "_out"
           << " {stream_id = " << output_stream.stream_id
           << ", direction = \"" << stream_direction_name(output_stream.direction)
           << "\", producer = \"" << format_endpoint(output_stream.producer)
           << "\", consumer = \"" << format_endpoint(output_stream.consumer)
           << "\", produce_cycle = " << output_stream.produce_cycle
           << ", consume_cycle = " << output_stream.consume_cycle
           << ", latency = " << output_stream.latency
           << ", bytes = " << (op.rows * op.columns)
           << ", vector = \"south_to_north\"}\n";
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
    out.swiglus = module.swiglus;

    std::ostringstream os;
    os << "// FTLPU schedule IR lowered from ftlpu.stream.\n";
    os << "// This is a stage-level queue/timeline layer before .ftlpu emission.\n";
    os << "module {\n";
    os << "  ftlpu.schedule.program @main {\n";
    auto scheduler = ResourceScheduler {};
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        if (is_consumed_by_swiglu(module, i)) {
            continue;
        }
        const auto& op = module.matmuls[i];
        const auto weight_cycle = scheduler.reserve("MEM.east.read", 0, south_to_north_tiles);
        const auto load_cycle = scheduler.reserve("MXM.load", weight_cycle + 1, south_to_north_tiles);
        const auto activation_cycle = scheduler.reserve("MEM.east.read", load_cycle + south_to_north_tiles, op.m);
        const auto compute_cycle = scheduler.reserve("MXM.compute", activation_cycle + mem_to_mxm_latency(32), op.m);
        const auto write_cycle = scheduler.reserve("MEM.east.write", compute_cycle + mxm_to_vxm_latency(), op.m);
        os << "    ftlpu.schedule.mem_read_weight @matmul" << i << "_read_weight"
           << " {cycle = " << weight_cycle << ", source = @B" << i
           << ", streams = [0..15], sregs = [0..11], bytes = " << (op.k * op.n)
           << ", vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
        os << "    ftlpu.schedule.mxm_load @matmul" << i << "_load"
           << " {cycle = " << load_cycle << ", mxms = [0, 1], weight_streams = [0..15]}\n";
        os << "    ftlpu.schedule.mem_read_activation @matmul" << i << "_read_activation"
           << " {cycle = " << activation_cycle << ", source = @A" << i
           << ", streams = [16..31], sregs = [0..11], bytes = " << (op.m * op.k)
           << ", vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
        os << "    ftlpu.schedule.mxm_compute @matmul" << i << "_compute"
           << " {cycle = " << compute_cycle << ", mxms = [0, 1], m = " << op.m << ", n = " << op.n
           << ", k = " << op.k << ", activation_streams = [16..31], output_streams = [48..63], vector_lanes = 16, south_to_north_tiles = "
           << south_to_north_tiles << ", accumulate = true}\n";
        os << "    ftlpu.schedule.mem_write @matmul" << i << "_write"
           << " {cycle = " << write_cycle << ", dest = @C" << i
           << ", streams = [48..63], sregs = [0..11], bytes = " << (op.m * op.n * 4)
           << ", vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
    }
    for (std::size_t i = 0; i < module.swiglus.size(); ++i) {
        const auto& op = module.swiglus[i];
        const auto weight_cycle = scheduler.reserve("MEM.east.read", 0, south_to_north_tiles);
        const auto load_cycle = scheduler.reserve("MXM.load", weight_cycle + 1, south_to_north_tiles);
        const auto activation_cycle = scheduler.reserve("MEM.east.read", load_cycle + south_to_north_tiles, op.rows);
        const auto compute_cycle = scheduler.reserve("MXM.compute", activation_cycle + mem_to_mxm_latency(32), op.rows);
        const auto vxm_cycle = scheduler.reserve("VXM", compute_cycle + mxm_to_vxm_latency(), op.rows + vxm_pipeline_latency(10));
        const auto write_cycle = scheduler.reserve("MEM.east.write", vxm_cycle + vxm_pipeline_latency(10) + vxm_to_mem_latency(40), op.rows);
        os << "    ftlpu.schedule.mem_read_weight_gate_up @swiglu" << i << "_read_weight"
           << " {cycle = " << weight_cycle << ", sources = [@gate_w" << i << ", @up_w" << i
           << "], mxms = [0, 1], streams = [0..15, 16..31], bytes = ["
           << (op.columns * op.columns) << ", " << (op.columns * op.columns)
           << "], vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
        os << "    ftlpu.schedule.mxm_load_gate_up @swiglu" << i << "_load"
           << " {cycle = " << load_cycle << ", mxms = [0, 1], gate_weight_streams = [0..15], up_weight_streams = [16..31]}\n";
        os << "    ftlpu.schedule.mem_read_activation @swiglu" << i << "_read_activation"
           << " {cycle = " << activation_cycle << ", source = @activation" << i
           << ", streams = [16], bytes = " << (op.rows * op.columns)
           << ", vector = \"south_to_north\", south_to_north_tiles = " << south_to_north_tiles << "}\n";
        os << "    ftlpu.schedule.mxm_compute_gate_up @swiglu" << i << "_gate_up"
           << " {cycle = " << compute_cycle << ", mxms = [0, 1], rows = " << op.rows
           << ", columns = " << op.columns << ", activation_stream = 16"
           << ", gate_output_streams = [32..35], up_output_streams = [36..39]}\n";
        os << "    ftlpu.schedule.vxm_swiglu @swiglu" << i << "_vxm"
           << " {cycle = " << vxm_cycle << ", gate_streams = [32..35], up_streams = [36..39], output_stream = 31, "
           << "stages = 10, output_scale = 0.0078125}\n";
        os << "    ftlpu.schedule.mem_write @swiglu" << i << "_write"
           << " {cycle = " << write_cycle << ", dest = @S" << i
           << ", streams = [31], sregs = [0..11], bytes = " << (op.rows * op.columns)
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
