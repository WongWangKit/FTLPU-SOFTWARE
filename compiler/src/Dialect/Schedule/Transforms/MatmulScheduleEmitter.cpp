#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "ScheduleEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <optional>

namespace ftlpu::compiler::schedule {

mlir::LogicalResult lowerMatmulSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target,
    ResourceScheduler& scheduler)
{
    llvm::SmallVector<stream::MatmulOp> matmuls;
    function.walk([&](stream::MatmulOp op) { matmuls.push_back(op); });
    using namespace detail;

        for (stream::MatmulOp matmul : matmuls) {
            auto activation_route = matmul.getLhs().getDefiningOp<stream::RouteOp>();
            auto weight_route = matmul.getRhs().getDefiningOp<stream::RouteOp>();
            if (!activation_route || !weight_route
                || activation_route.getDestination() != "MXM.activation"
                || weight_route.getDestination() != "MXM.weight"
                || activation_route.getDestinationUnitId() != matmul.getUnitId()
                || weight_route.getDestinationUnitId() != matmul.getUnitId()
                || !matmul.getResult().hasOneUse()) {
                matmul.emitError("requires canonical activation/weight/result stream routes");
                return mlir::failure();
            }
            auto result_route = mlir::dyn_cast<stream::RouteOp>(*matmul.getResult().getUsers().begin());
            if (!result_route || result_route.getSource() != "MXM.result"
                || result_route.getDestination() != "MEM"
                || result_route.getSourceUnitId() != matmul.getUnitId()) {
                matmul.emitError("requires a canonical MXM.result to MEM route");
                return mlir::failure();
            }

            auto placement_count = [](stream::RouteOp route) -> std::optional<int64_t> {
                if (auto count = route.getPlacement().getAs<mlir::IntegerAttr>(
                        "instruction_count"))
                    return count.getInt();
                return std::nullopt;
            };
            const auto weight_read_duration = placement_count(weight_route);
            const auto activation_read_duration = placement_count(activation_route);
            const auto write_duration = placement_count(result_route);
            if (!weight_read_duration || !activation_read_duration || !write_duration) {
                matmul.emitError("stream placement requires a physical instruction count");
                return mlir::failure();
            }

            const auto weight_slices = get_slices(weight_route.getPlacement());
            const auto activation_slices = get_slices(activation_route.getPlacement());
            const auto result_slices = get_slices(result_route.getPlacement());
            int64_t weight_latency = 0;
            for (int64_t slice : weight_slices)
                weight_latency = std::max(weight_latency, *target.transport_latency(
                    target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                    target::StreamDirection::East, slice));
            const int64_t activation_latency = activation_route.getTransportLatency();
            const int64_t result_latency = result_route.getTransportLatency();
            const int64_t load_offset = weight_latency;
            const int64_t compute_duration = target.mxm_compute_issue_cycles(matmul.getM());
            const int64_t result_offset = activation_latency + target.mxm_first_result_latency();
            const int64_t result_duration = target.mxm_result_window_cycles(matmul.getM());
            const int64_t write_offset = result_offset + result_latency;

            llvm::SmallVector<schedule::ResourceWindow> weight_windows;
            for (int64_t slice : weight_slices) {
                const int64_t latency = *target.transport_latency(target::StreamEndpoint::Mem,
                    target::StreamEndpoint::MxmWeight, target::StreamDirection::East, slice);
                weight_windows.push_back(
                    {mem_resource(slice), weight_latency - latency, *weight_read_duration});
            }
            weight_windows.push_back({llvm::formatv("MXM.{0}.load", matmul.getUnitId()).str(),
                load_offset, *weight_read_duration});
            weight_windows.push_back({llvm::formatv("MXM.{0}.weight_buffer.{1}",
                matmul.getUnitId(), matmul.getWeightBuffer()).str(),
                load_offset, *weight_read_duration});
            add_stream_windows(weight_windows, "east", weight_route.getStreamBase(),
                weight_route.getStreamCount(), 0, load_offset + *weight_read_duration);
            const int64_t earliest_weight_start = std::max(
                value_ready_cycle(weight_route.getInput()),
                target.mxm_earliest_iw_cycle() - load_offset);
            const int64_t weight_read_cycle = scheduler.reserve(
                earliest_weight_start, weight_windows);
            const int64_t load_cycle = weight_read_cycle + load_offset;

            llvm::SmallVector<schedule::ResourceWindow> compute_windows;
            for (int64_t slice : activation_slices)
                compute_windows.push_back({mem_resource(slice), 0, *activation_read_duration});
            compute_windows.push_back({llvm::formatv("MXM.{0}.compute", matmul.getUnitId()).str(),
                activation_latency, compute_duration});
            compute_windows.push_back({llvm::formatv("MXM.{0}.weight_buffer.{1}",
                matmul.getUnitId(), matmul.getWeightBuffer()).str(),
                activation_latency, compute_duration});
            for (int64_t slice : result_slices) {
                const int64_t latency = *target.transport_latency(target::StreamEndpoint::MxmResult,
                    target::StreamEndpoint::Mem, target::StreamDirection::West, slice);
                compute_windows.push_back(
                    {mem_resource(slice), result_offset + latency, *write_duration});
            }
            add_stream_windows(compute_windows, "east", activation_route.getStreamBase(),
                activation_route.getStreamCount(), 0, activation_latency + compute_duration);
            add_stream_windows(compute_windows, "west", result_route.getStreamBase(),
                result_route.getStreamCount(), result_offset,
                std::max(result_duration, result_latency + *write_duration));
            const int64_t activation_read_cycle = scheduler.reserve(
                value_ready_cycle(activation_route.getInput()), compute_windows);
            const int64_t compute_cycle = activation_read_cycle + activation_latency;
            const int64_t result_cycle = activation_read_cycle + result_offset;
            const int64_t write_cycle = activation_read_cycle + write_offset;

            rewriter.setInsertionPoint(matmul);
            auto weight_read = rewriter.create<schedule::MemReadOp>(matmul.getLoc(),
                weight_route.getInput(), weight_read_cycle, *weight_read_duration,
                weight_route.getStreamBase(), weight_route.getStreamCount(),
                weight_route.getRegisterId(), weight_route.getDirectionAttr(),
                rewriter.getStringAttr("weight"), weight_route.getAddress(),
                weight_route.getPlacement(), weight_route.getBytes());
            auto load = rewriter.create<schedule::MxmLoadOp>(matmul.getLoc(),
                weight_read.getOutput(), load_cycle, *weight_read_duration,
                weight_route.getStreamBase(), weight_route.getStreamCount(),
                matmul.getUnitId(), matmul.getWeightBuffer());
            auto activation_read = rewriter.create<schedule::MemReadOp>(matmul.getLoc(),
                activation_route.getInput(), activation_read_cycle, *activation_read_duration,
                activation_route.getStreamBase(), activation_route.getStreamCount(),
                activation_route.getRegisterId(), activation_route.getDirectionAttr(),
                rewriter.getStringAttr("activation"), activation_route.getAddress(),
                activation_route.getPlacement(), activation_route.getBytes());
            auto compute = rewriter.create<schedule::MxmComputeOp>(matmul.getLoc(),
                activation_read.getOutput(), load.getOutput(), matmul.getResult().getType(),
                compute_cycle, compute_duration, result_cycle, result_duration,
                activation_route.getStreamBase(), result_route.getStreamBase(),
                matmul.getWeightBuffer(),
                matmul.getUnitId(), matmul.getM(), matmul.getN(), matmul.getK());
            auto write = rewriter.create<schedule::MemWriteOp>(matmul.getLoc(),
                compute.getResult(), write_cycle, *write_duration,
                result_route.getStreamBase(), result_route.getStreamCount(),
                result_route.getRegisterId(), result_route.getDirectionAttr(),
                result_route.getAddress(), result_route.getPlacement(), result_route.getBytes());

            rewriter.replaceOp(result_route, write.getOutput());
            rewriter.eraseOp(matmul);
            if (activation_route->use_empty()) rewriter.eraseOp(activation_route);
            if (weight_route->use_empty()) rewriter.eraseOp(weight_route);
        }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
