#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "ScheduleEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "llvm/Support/FormatVariadic.h"

#include <algorithm>

namespace ftlpu::compiler::schedule {

mlir::LogicalResult lowerSwigluSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target,
    ResourceScheduler& scheduler)
{
    llvm::SmallVector<stream::SwigluOp> swiglus;
    function.walk([&](stream::SwigluOp op) { swiglus.push_back(op); });
    using namespace detail;

        for (stream::SwigluOp swiglu : swiglus) {
            auto activation_route = swiglu.getActivation().getDefiningOp<stream::RouteOp>();
            auto gate_route = swiglu.getGateWeight().getDefiningOp<stream::RouteOp>();
            auto up_route = swiglu.getUpWeight().getDefiningOp<stream::RouteOp>();
            if (!activation_route || !gate_route || !up_route) {
                swiglu.emitError("requires canonical dual-MXM input routes");
                return mlir::failure();
            }
            struct WeightSchedule { int64_t read_cycle; int64_t load_cycle; int64_t duration; };
            auto schedule_weight = [&](stream::RouteOp route, int64_t unit,
                                       int64_t buffer) -> mlir::FailureOr<WeightSchedule> {
                const auto duration = target.route_issue_cycles(target::StreamEndpoint::Mem,
                    target::StreamEndpoint::MxmWeight, route.getBytes());
                if (!duration) return mlir::failure();
                const auto slices = get_slices(route.getPlacement());
                int64_t latency = 0;
                for (int64_t slice : slices)
                    latency = std::max(latency, *target.transport_latency(
                        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                        target::StreamDirection::East, slice));
                llvm::SmallVector<schedule::ResourceWindow> windows;
                for (int64_t slice : slices) {
                    const int64_t slice_latency = *target.transport_latency(
                        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                        target::StreamDirection::East, slice);
                    windows.push_back({mem_resource(slice), latency - slice_latency, *duration});
                }
                windows.push_back({llvm::formatv("MXM.{0}.load", unit).str(), latency, *duration});
                windows.push_back({llvm::formatv("MXM.{0}.weight_buffer.{1}", unit, buffer).str(),
                    latency, *duration});
                add_stream_windows(windows, "east", route.getStreamBase(), route.getStreamCount(),
                    0, latency + *duration);
                const int64_t read_cycle = scheduler.reserve(
                    std::max<int64_t>(0, target.mxm_earliest_iw_cycle() - latency), windows);
                return WeightSchedule {read_cycle, read_cycle + latency, *duration};
            };
            const auto gate_schedule = schedule_weight(gate_route,
                swiglu.getGateUnitId(), swiglu.getGateWeightBuffer());
            const auto up_schedule = schedule_weight(up_route,
                swiglu.getUpUnitId(), swiglu.getUpWeightBuffer());
            const auto activation_duration = target.route_issue_cycles(
                target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
                activation_route.getBytes());
            const auto output_duration = target.route_issue_cycles(
                target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
                swiglu.getResultBytes());
            if (mlir::failed(gate_schedule) || mlir::failed(up_schedule)
                || !activation_duration || !output_duration) {
                swiglu.emitError("cannot derive SwiGLU issue durations");
                return mlir::failure();
            }
            const int64_t activation_latency = activation_route.getTransportLatency();
            const int64_t compute_duration = target.mxm_compute_issue_cycles(swiglu.getM());
            const int64_t result_offset = activation_latency + target.mxm_first_result_latency();
            constexpr int64_t vxm_latency = 9;
            const int64_t write_offset = result_offset + vxm_latency
                + swiglu.getOutputTransportLatency();
            llvm::SmallVector<schedule::ResourceWindow> windows;
            for (int64_t slice : get_slices(activation_route.getPlacement()))
                windows.push_back({mem_resource(slice), 0, *activation_duration});
            for (int64_t unit : {static_cast<int64_t>(swiglu.getGateUnitId()),
                     static_cast<int64_t>(swiglu.getUpUnitId())})
                windows.push_back({llvm::formatv("MXM.{0}.compute", unit).str(),
                    activation_latency, compute_duration});
            windows.push_back({llvm::formatv("MXM.{0}.weight_buffer.{1}",
                swiglu.getGateUnitId(), swiglu.getGateWeightBuffer()).str(),
                activation_latency, compute_duration});
            windows.push_back({llvm::formatv("MXM.{0}.weight_buffer.{1}",
                swiglu.getUpUnitId(), swiglu.getUpWeightBuffer()).str(),
                activation_latency, compute_duration});
            add_stream_windows(windows, "east", activation_route.getStreamBase(), 1,
                0, activation_latency + compute_duration);
            add_stream_windows(windows, "west", swiglu.getGateOutputStreamBase(), 4,
                result_offset, target.mxm_result_window_cycles(swiglu.getM()));
            add_stream_windows(windows, "west", swiglu.getUpOutputStreamBase(), 4,
                result_offset, target.mxm_result_window_cycles(swiglu.getM()));
            for (int64_t alu = 0; alu < 16; ++alu)
                windows.push_back({llvm::formatv("VXM.alu.{0}", alu).str(),
                    result_offset, compute_duration + vxm_latency});
            add_stream_windows(windows, "east", swiglu.getVxmOutputStream(), 1,
                result_offset + vxm_latency,
                compute_duration + swiglu.getOutputTransportLatency());
            const int64_t result_slice = get_slice(swiglu.getResultAddress());
            windows.push_back({mem_resource(result_slice), write_offset, *output_duration});
            const int64_t loads_done = std::max(
                gate_schedule->load_cycle + gate_schedule->duration,
                up_schedule->load_cycle + up_schedule->duration);
            const int64_t activation_read_cycle = scheduler.reserve(
                std::max<int64_t>(0, loads_done - activation_latency), windows);
            const int64_t compute_cycle = activation_read_cycle + activation_latency;
            const int64_t result_cycle = activation_read_cycle + result_offset;

            rewriter.setInsertionPoint(swiglu);
            auto make_read_load = [&](stream::RouteOp route, const WeightSchedule& timing,
                                      int64_t unit, int64_t buffer) {
                auto read = rewriter.create<schedule::MemReadOp>(swiglu.getLoc(), route.getInput(),
                    timing.read_cycle, timing.duration, route.getStreamBase(), route.getStreamCount(),
                    route.getRegisterId(), route.getDirectionAttr(), rewriter.getStringAttr("weight"),
                    route.getAddress(), route.getPlacement(), route.getBytes());
                return rewriter.create<schedule::MxmLoadOp>(swiglu.getLoc(), read.getOutput(),
                    timing.load_cycle, timing.duration, route.getStreamBase(), route.getStreamCount(),
                    unit, buffer);
            };
            auto gate_load = make_read_load(gate_route, *gate_schedule,
                swiglu.getGateUnitId(), swiglu.getGateWeightBuffer());
            auto up_load = make_read_load(up_route, *up_schedule,
                swiglu.getUpUnitId(), swiglu.getUpWeightBuffer());
            auto projection_type = mlir::RankedTensorType::get(
                {swiglu.getMAttr().getInt(), swiglu.getNAttr().getInt()}, rewriter.getI32Type());
            schedule::MxmComputeOp gate_compute;
            schedule::MxmComputeOp up_compute;
            const int64_t segment_rows[] = {15, 4, swiglu.getMAttr().getInt() - 19};
            const int64_t segment_streams[] = {16, 30, 0};
            int64_t row_offset = 0;
            for (int segment = 0; segment < 3; ++segment) {
                const int64_t rows = segment_rows[segment];
                if (rows <= 0) continue;
                const int64_t stream = segment_streams[segment];
                const int64_t bytes = rows * target.throughput().mxm_rows;
                auto placement = subrange_placement(rewriter,
                    activation_route.getPlacement(), row_offset, rows);
                auto read = rewriter.create<schedule::MemReadOp>(swiglu.getLoc(),
                    activation_route.getInput(), activation_read_cycle + row_offset, rows,
                    stream, 1, activation_route.getRegisterId(), activation_route.getDirectionAttr(),
                    rewriter.getStringAttr("activation"), activation_route.getAddress(),
                    placement, bytes);
                gate_compute = rewriter.create<schedule::MxmComputeOp>(swiglu.getLoc(),
                    read.getOutput(), gate_load.getOutput(), projection_type,
                    compute_cycle + row_offset, rows, result_cycle + row_offset,
                    target.mxm_result_window_cycles(rows), stream,
                    swiglu.getGateOutputStreamBase(), swiglu.getGateWeightBuffer(),
                    swiglu.getGateUnitId(), rows, swiglu.getN(), swiglu.getK());
                up_compute = rewriter.create<schedule::MxmComputeOp>(swiglu.getLoc(),
                    read.getOutput(), up_load.getOutput(), projection_type,
                    compute_cycle + row_offset, rows, result_cycle + row_offset,
                    target.mxm_result_window_cycles(rows), stream,
                    swiglu.getUpOutputStreamBase(), swiglu.getUpWeightBuffer(),
                    swiglu.getUpUnitId(), rows, swiglu.getN(), swiglu.getK());
                row_offset += rows;
            }

            mlir::Value vxm_result;
            auto emit_vxm = [&](int64_t stage, int64_t alu, llvm::StringRef opcode,
                                llvm::StringRef lhs_kind, int64_t lhs_index, float lhs_imm,
                                llvm::StringRef rhs_kind, int64_t rhs_index, float rhs_imm,
                                llvm::StringRef cast_target, int64_t output_stream) {
                mlir::OperationState state(swiglu.getLoc(), schedule::VxmOp::getOperationName());
                state.addOperands({gate_compute.getResult(), up_compute.getResult()});
                state.addTypes(swiglu.getResult().getType());
                state.addAttributes({
                    rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(result_cycle + stage)),
                    rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(alu)),
                    rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
                    rewriter.getNamedAttr("lhs_kind", rewriter.getStringAttr(lhs_kind)),
                    rewriter.getNamedAttr("lhs_index", rewriter.getI64IntegerAttr(lhs_index)),
                    rewriter.getNamedAttr("lhs_immediate", rewriter.getF32FloatAttr(lhs_imm)),
                    rewriter.getNamedAttr("rhs_kind", rewriter.getStringAttr(rhs_kind)),
                    rewriter.getNamedAttr("rhs_index", rewriter.getI64IntegerAttr(rhs_index)),
                    rewriter.getNamedAttr("rhs_immediate", rewriter.getF32FloatAttr(rhs_imm)),
                    rewriter.getNamedAttr("cast_target", rewriter.getStringAttr(cast_target)),
                    rewriter.getNamedAttr("output_stream", rewriter.getI64IntegerAttr(output_stream)),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(swiglu.getM())),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("input_hemisphere", rewriter.getStringAttr("east")),
                    rewriter.getNamedAttr("output_hemisphere", rewriter.getStringAttr("east")),
                });
                vxm_result = llvm::cast<schedule::VxmOp>(rewriter.create(state)).getResult();
            };
            const int64_t gate_stream = 32 + swiglu.getGateOutputStreamBase();
            const int64_t up_stream = 32 + swiglu.getUpOutputStreamBase();
            emit_vxm(0, 0, "cast", "stream_i32", gate_stream, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(0, 1, "cast", "stream_i32", up_stream, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(1, 2, "multiply", "alu", 0, 0, "immediate", 0, swiglu.getGateScale().convertToFloat(), "fp32", -1);
            emit_vxm(1, 3, "multiply", "alu", 1, 0, "immediate", 0, swiglu.getUpScale().convertToFloat(), "fp32", -1);
            emit_vxm(2, 4, "multiply", "alu", 2, 0, "alu", 3, 0, "fp32", -1);
            emit_vxm(2, 5, "negate", "alu", 2, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(3, 6, "exp", "alu", 5, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(3, 9, "pass", "alu", 4, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(4, 7, "add", "alu", 6, 0, "immediate", 0, 1, "fp32", -1);
            emit_vxm(4, 10, "pass", "alu", 9, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(5, 8, "divide", "immediate", 0, 1, "alu", 7, 0, "fp32", -1);
            emit_vxm(5, 11, "pass", "alu", 10, 0, "immediate", 0, 0, "fp32", -1);
            emit_vxm(6, 12, "multiply", "alu", 11, 0, "alu", 8, 0, "fp32", -1);
            emit_vxm(7, 13, "multiply", "alu", 12, 0, "immediate", 0,
                1.0f / swiglu.getOutputScale().convertToFloat(), "fp32", -1);
            emit_vxm(8, 14, "add", "alu", 13, 0, "immediate", 0,
                static_cast<float>(swiglu.getOutputZeroPoint()), "fp32", -1);
            emit_vxm(9, 15, "cast", "alu", 14, 0, "immediate", 0, 0, "i8",
                swiglu.getVxmOutputStream());
            auto write = rewriter.create<schedule::MemWriteOp>(swiglu.getLoc(), vxm_result,
                activation_read_cycle + write_offset, *output_duration,
                swiglu.getVxmOutputStream(), 1, swiglu.getOutputRegisterId(),
                rewriter.getStringAttr("east"), swiglu.getResultAddress(),
                swiglu.getResultPlacement(), swiglu.getResultBytes());
            rewriter.replaceOp(swiglu, write.getOutput());
            if (activation_route->use_empty()) rewriter.eraseOp(activation_route);
            if (gate_route->use_empty()) rewriter.eraseOp(gate_route);
            if (up_route->use_empty()) rewriter.eraseOp(up_route);
        }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
