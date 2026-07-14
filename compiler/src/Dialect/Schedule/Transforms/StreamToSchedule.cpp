#include "ftlpu/compiler/Pipelines/pipelines.hpp"

#include "Dialect/Common/LoweringUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"

#include <sstream>

namespace ftlpu::compiler::pipeline {

Module lower_stream_to_schedule(const Module& module, const target::TargetBackend& target, std::size_t south_to_north_tiles)
{
    detail::require_dialect(module, Dialect::Stream, "stream-to-schedule");
    Module out;
    out.dialect = Dialect::Schedule;
    out.matmuls = module.matmuls;
    out.swiglus = module.swiglus;

    std::ostringstream os;
    os << "// FTLPU schedule IR lowered from ftlpu.stream.\n";
    os << "// This is a stage-level queue/timeline layer before .ftlpu emission.\n";
    os << "module {\n";
    os << "  ftlpu.schedule.program @main {\n";
    auto scheduler = schedule::ResourceScheduler {};
    for (std::size_t i = 0; i < module.matmuls.size(); ++i) {
        if (detail::is_consumed_by_swiglu(module, i)) {
            continue;
        }
        const auto& op = module.matmuls[i];
        const auto weight_cycle = scheduler.reserve("MEM.east.read", 0, south_to_north_tiles);
        const auto load_cycle = scheduler.reserve("MXM.load", weight_cycle + 1, south_to_north_tiles);
        const auto activation_cycle = scheduler.reserve("MEM.east.read", load_cycle + south_to_north_tiles, op.m);
        const auto compute_cycle = scheduler.reserve("MXM.compute", activation_cycle + target.mem_to_mxm_latency(32), op.m);
        const auto write_cycle = scheduler.reserve("MEM.east.write", compute_cycle + target.mxm_to_fu_latency(), op.m);
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
        const auto compute_cycle = scheduler.reserve("MXM.compute", activation_cycle + target.mem_to_mxm_latency(32), op.rows);
        const auto vxm_cycle = scheduler.reserve("VXM", compute_cycle + target.mxm_to_fu_latency(), op.rows + target.vxm_pipeline_latency(10));
        const auto write_cycle = scheduler.reserve("MEM.east.write", vxm_cycle + target.vxm_pipeline_latency(10) + target.vxm_to_mem_latency(40), op.rows);
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

} // namespace ftlpu::compiler::pipeline
