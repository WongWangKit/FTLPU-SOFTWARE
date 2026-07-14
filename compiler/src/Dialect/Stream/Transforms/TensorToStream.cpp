#include "ftlpu/compiler/Pipelines/pipelines.hpp"

#include "Dialect/Common/LoweringUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_allocator.hpp"

#include <algorithm>
#include <map>
#include <sstream>

namespace ftlpu::compiler::pipeline {

Module lower_tensor_to_stream(const Module& module, const target::TargetBackend& target, std::size_t south_to_north_tiles)
{
    detail::require_dialect(module, Dialect::Tensor, "tensor-to-stream");
    Module out;
    out.dialect = Dialect::Stream;
    out.matmuls = module.matmuls;
    out.swiglus = module.swiglus;
    auto scheduler = schedule::ResourceScheduler {};
    auto streams = stream::StreamAllocator {};
    std::map<std::size_t, stream::Binding> matmul_output_streams;

    std::ostringstream os;
    os << "// FTLPU stream IR lowered from ftlpu.tensor.\n";
    os << "// This layer binds logical data movement to endpoints and stream lifetimes.\n";
    os << "module {\n";
    os << "  ftlpu.stream.func @main {\n";
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        const auto& op = module.matmuls[i];
        const auto weight_cycle = scheduler.reserve("MEM.east.read", 0, south_to_north_tiles);
        const auto load_cycle = scheduler.reserve("MXM.load", weight_cycle + 1, south_to_north_tiles);
        const auto activation_cycle = scheduler.reserve("MEM.east.read", load_cycle + south_to_north_tiles, op.m);
        const auto compute_cycle = scheduler.reserve("MXM.compute", activation_cycle + target.mem_to_mxm_latency(32), op.m);
        const auto output_arrival_cycle = compute_cycle + target.mxm_to_fu_latency();
        auto swiglu_index = std::size_t {0};
        auto swiglu_port = std::string {};
        const auto* const swiglu = detail::swiglu_consumer(module, i, swiglu_index, swiglu_port);
        const auto consumer_cycle = swiglu == nullptr
            ? scheduler.reserve("MEM.east.write", output_arrival_cycle, op.m)
            : scheduler.reserve("VXM.input", output_arrival_cycle, op.m);
        const auto output_preferred_stream = swiglu == nullptr
            ? std::size_t {48}
            : (swiglu_port == "gate" ? std::size_t {32} : std::size_t {36});
        const auto output_consumer = swiglu == nullptr
            ? stream::Endpoint {"MEM", "C" + std::to_string(i)}
            : stream::Endpoint {"VXM", "swiglu" + std::to_string(swiglu_index), swiglu_port};
        const auto weight_stream = streams.allocate(
            "matmul" + std::to_string(i) + "_weight",
            stream::Direction::East,
            {"MEM", "B" + std::to_string(i)},
            {"MXM", "*", "weight"},
            weight_cycle,
            load_cycle,
            0);
        const auto activation_stream = streams.allocate(
            "matmul" + std::to_string(i) + "_activation",
            stream::Direction::East,
            {"MEM", "A" + std::to_string(i)},
            {"MXM", "*", "activation"},
            activation_cycle,
            compute_cycle,
            16);
        const auto output_stream = streams.allocate(
            "matmul" + std::to_string(i) + "_output",
            swiglu == nullptr ? stream::Direction::West : stream::Direction::East,
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
           << ", direction = \"" << stream::direction_name(activation_stream.direction)
           << "\", producer = \"" << stream::format_endpoint(activation_stream.producer)
           << "\", consumer = \"" << stream::format_endpoint(activation_stream.consumer)
           << "\", produce_cycle = " << activation_stream.produce_cycle
           << ", consume_cycle = " << activation_stream.consume_cycle
           << ", latency = " << activation_stream.latency
           << ", bytes = " << (op.m * op.k) << ", vector = \"south_to_north\"}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_rhs"
           << " {stream_id = " << weight_stream.stream_id
           << ", direction = \"" << stream::direction_name(weight_stream.direction)
           << "\", producer = \"" << stream::format_endpoint(weight_stream.producer)
           << "\", consumer = \"" << stream::format_endpoint(weight_stream.consumer)
           << "\", produce_cycle = " << weight_stream.produce_cycle
           << ", consume_cycle = " << weight_stream.consume_cycle
           << ", latency = " << weight_stream.latency
           << ", bytes = " << (op.k * op.n) << ", vector = \"south_to_north\"}\n";
        os << "    ftlpu.stream.channel @matmul" << i << "_out"
           << " {stream_id = " << output_stream.stream_id
           << ", direction = \"" << stream::direction_name(output_stream.direction)
           << "\", producer = \"" << stream::format_endpoint(output_stream.producer)
           << "\", consumer = \"" << stream::format_endpoint(output_stream.consumer)
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
        const auto write_cycle = scheduler.reserve("MEM.east.write", vxm_cycle + target.vxm_pipeline_latency(10) + target.vxm_to_mem_latency(40), op.rows);
        const auto output_stream = streams.allocate(
            "swiglu" + std::to_string(i) + "_output",
            stream::Direction::West,
            {"VXM", "swiglu" + std::to_string(i), "output"},
            {"MEM", "S" + std::to_string(i)},
            vxm_cycle + target.vxm_pipeline_latency(10),
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
           << ", direction = \"" << stream::direction_name(output_stream.direction)
           << "\", producer = \"" << stream::format_endpoint(output_stream.producer)
           << "\", consumer = \"" << stream::format_endpoint(output_stream.consumer)
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

} // namespace ftlpu::compiler::pipeline
