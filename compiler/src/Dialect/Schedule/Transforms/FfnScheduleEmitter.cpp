#include "ftlpu/compiler/Dialect/Schedule/Transforms/ffn_schedule_emitter.hpp"

#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_projection_timeline.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_swish_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ftlpu::compiler {
namespace {

using namespace schedule::ffn_detail;

template <typename FfnPlan>
mlir::FailureOr<mlir::Value> lower_w8a16_ffn_schedule(
    mlir::IRRewriter& rewriter, FfnPlan& ffn,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target)
{
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = throughput.mxm_rows;
    const int64_t m = ffn.getM();
    const int64_t k = ffn.getK();
    const int64_t intermediate = ffn.getHidden();
    const int64_t n = ffn.getN();
    if (mlir::failed(ffn.task_plan.tasks.validate())) return mlir::failure();
    const int64_t gate_acc_latency = throughput.mxm0_accumulator_latency;
    const int64_t up_acc_latency = throughput.mxm1_accumulator_latency;
    const int64_t projection_accumulator_rows = m * (intermediate / tile);
    const int64_t down_accumulator_base = std::max(
        memory.accumulator_scratch_base_row,
        ((projection_accumulator_rows + tile - 1) / tile) * tile);

    if (!target.supports_w8a16_ffn_shape(ffn.getM(), k, intermediate, n))
        return mlir::failure();
    auto activation_route = ffn.getActivation().getDefiningOp<stream::RouteOp>();
    auto gate_route = ffn.getGateWeight().getDefiningOp<stream::RouteOp>();
    auto up_route = ffn.getUpWeight().getDefiningOp<stream::RouteOp>();
    auto down_route = ffn.getDownWeight0().getDefiningOp<stream::RouteOp>();
    if (!activation_route || !gate_route || !up_route || !down_route) return mlir::failure();

    auto raw_route = [](stream::RouteOp route) -> stream::RouteOp {
        auto dequant = route.getInput().getDefiningOp<stream::DequantizeOp>();
        return dequant ? dequant.getInput().getDefiningOp<stream::RouteOp>() : stream::RouteOp{};
    };
    auto gate_raw = raw_route(gate_route);
    auto up_raw = raw_route(up_route);
    auto down_raw = raw_route(down_route);
    if (!gate_raw || !up_raw || !down_raw) return mlir::failure();

    const auto weight_slices = get_slices(gate_raw.getPlacement());
    const auto activation_slices = get_slices(activation_route.getPlacement());
    const auto hidden_slices = get_slices(ffn.getHidden0Placement());
    const auto result_slices = get_slices(ffn.getResultPlacement());
    if (weight_slices.size() != static_cast<size_t>(memory.w8a16_weight_slice_count)
        || activation_slices.size() != static_cast<size_t>(throughput.mxm_activation_streams)
        || hidden_slices.size() != static_cast<size_t>(throughput.mxm_activation_streams)
        || result_slices.size() != static_cast<size_t>(throughput.mxm_result_streams))
        return mlir::failure();
    llvm::SmallVector<int64_t> gate_acc_slices;
    llvm::SmallVector<int64_t> up_acc_slices;
    for (int64_t index = 0; index < memory.accumulator_slices_per_mxm; ++index) {
        gate_acc_slices.push_back(memory.accumulator_slice_base + index);
        up_acc_slices.push_back(
            memory.accumulator_slice_base + memory.accumulator_slices_per_mxm + index);
    }
    const int64_t activation_latency = *target.transport_latency(
        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, activation_slices.front());
    auto projectionTimeline = schedule::planFfnProjectionTimeline(
        {m, k, intermediate, n}, weight_slices, target);
    if (mlir::failed(projectionTimeline)) return mlir::failure();
    const int64_t weight_load_cycles =
        projectionTimeline->weight_load_cycles;
    const int64_t projection_pair_count = projectionTimeline->pair_count;
    const int64_t m_tile_count = projectionTimeline->m_tile_count;

    const auto projection_type = mlir::RankedTensorType::get({tile, tile}, rewriter.getF32Type());
    const auto one_slice_read = [&](mlir::Value value, stream::RouteOp route,
                                    int64_t cycle, int64_t slice, int64_t base,
                                    int64_t count, int64_t stride, int64_t stream,
                                    llvm::StringRef direction, llvm::StringRef role,
                                    llvm::StringRef hemisphere) {
        auto placement = schedule_placement(rewriter, {slice}, base, count, stride,
            hemisphere, "schedule_slice");
        mlir::NamedAttrList placement_attrs(placement);
        placement_attrs.set("binding_placement", route.getPlacement());
        return rewriter.create<schedule::MemReadOp>(ffn.getLoc(), value, cycle, count,
            stream, 1, slice / target.streams().mem_slices_per_register_group + 1,
            rewriter.getStringAttr(direction), rewriter.getStringAttr(role),
            route.getAddress(), placement_attrs.getDictionary(rewriter.getContext()),
            count * tile);
    };
    const auto west_latency = [&](int64_t slice) {
        return slice / target.streams().mem_slices_per_register_group + 2;
    };
    const auto east_mxm_latency = [&](int64_t slice) {
        return target.streams().system_register_columns
            - slice / target.streams().mem_slices_per_register_group;
    };
    const auto hemi_name = [](int64_t hemisphere) -> llvm::StringRef {
        return hemisphere == 0 ? "east" : "west";
    };
    struct CompletedProjectionTile {
        int64_t pair;
        int64_t m_tile;
        int64_t hemisphere;
        int64_t compute_cycle;
        int64_t deferred_ready_cycle;
        schedule::MemAccumulateOp gate;
        schedule::MemAccumulateOp up;
        mlir::Value gate_temp;
        mlir::Value up_temp;
    };
    llvm::SmallVector<CompletedProjectionTile> completed_tiles;
    std::array<std::vector<schedule::FfnCycleWindow>, 2>
        temp_mem_busy_windows;
    rewriter.setInsertionPoint(ffn.getOperation());
    for (const schedule::FfnProjectionBlockSchedule& block :
         projectionTimeline->blocks) {
            const int64_t pair = block.pair;
            const int64_t kb = block.reduction_block;
            const int64_t dequant_start = block.dequant_start;
            const int64_t weight_buffer = block.weight_buffer;
            for (int64_t hemisphere = 0; hemisphere < memory.hemispheres; ++hemisphere) {
                for (int64_t local_mxm = 0;
                     local_mxm < throughput.mxms_per_hemisphere; ++local_mxm) {
                    stream::RouteOp raw = local_mxm == 0 ? gate_raw : up_raw;
                    stream::RouteOp cooked = local_mxm == 0 ? gate_route : up_route;
                    const int64_t start = dequant_start
                        + (hemisphere * throughput.mxms_per_hemisphere + local_mxm)
                            * weight_load_cycles;
                    const int64_t base = cooked.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt()
                        + (pair * (k / tile) + kb) * weight_load_cycles;
                    emitFfnWeightTile(rewriter, ffn.getLoc(), raw,
                        cooked.getInput().getType(), weight_slices, target,
                        local_mxm == 0
                            ? ffn.getGateScale().convertToFloat()
                            : ffn.getUpScale().convertToFloat(),
                        start, base, hemisphere, local_mxm,
                        hemisphere * throughput.mxms_per_hemisphere + local_mxm,
                        weight_buffer);
                }
            }

            for (const schedule::FfnProjectionTileSchedule& tileSchedule :
                block.tiles) {
                const int64_t m_tile = tileSchedule.m_tile;
                const int64_t compute_cycle = tileSchedule.compute_cycle;
                for (int64_t hemisphere = 0; hemisphere < memory.hemispheres; ++hemisphere) {
                // Activation is laid out as [K-tile][full M rows]. Select the
                // current K block first, then the M=32 subrange within it.
                const int64_t activation_base = kb * m + m_tile * tile;
                llvm::SmallVector<int64_t> segment_rows;
                llvm::SmallVector<int64_t> segment_streams;
                for (const schedule::FfnStreamSegment& segment :
                     tileSchedule.hemisphere_segments[
                         static_cast<std::size_t>(hemisphere)]) {
                    segment_rows.push_back(segment.rows);
                    segment_streams.push_back(segment.stream_base);
                }
                const bool final_k_block = block.final_reduction;
                const int64_t projection_result_stream_base =
                    strategy == FfnScheduleStrategy::Fused && final_k_block
                    ? 8 + hemisphere * 8
                    : 0;
                schedule::MxmComputeOp gate_compute;
                schedule::MxmComputeOp up_compute;
                int64_t row_offset = 0;
                for (size_t segment = 0; segment < segment_rows.size(); ++segment) {
                    const int64_t rows = segment_rows[segment];
                    const int64_t stream_base = segment_streams[segment];
                    mlir::Value activation_value;
                    const int64_t segment_cycle = compute_cycle + row_offset;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        auto read = one_slice_read(activation_route.getInput(), activation_route,
                            segment_cycle - activation_latency, activation_slices[byte],
                            activation_base + row_offset, rows, 1, stream_base + byte,
                            "east", "activation", hemi_name(hemisphere));
                        activation_value = read.getOutput();
                    }
                    gate_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        activation_value, ffn.getGateWeight(), projection_type,
                        segment_cycle, rows,
                        segment_cycle + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(rows),
                        stream_base, projection_result_stream_base,
                        weight_buffer,
                        hemisphere * throughput.mxms_per_hemisphere,
                        rows, tile, tile);
                    up_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        activation_value, ffn.getUpWeight(), projection_type,
                        segment_cycle, rows,
                        segment_cycle + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(rows),
                        stream_base,
                        projection_result_stream_base
                            + throughput.mxm_result_streams,
                        weight_buffer,
                        hemisphere * throughput.mxms_per_hemisphere + 1,
                        rows, tile, tile);
                    row_offset += rows;
                }
                const int64_t nblock = pair * memory.hemispheres + hemisphere;
                const int64_t accumulator_base = m_tile * tile * (intermediate / tile) + nblock;
                auto gate_acc_placement = schedule_placement(rewriter, gate_acc_slices,
                    accumulator_base, tile, intermediate / tile, hemi_name(hemisphere), "fp32_accumulator");
                auto up_acc_placement = schedule_placement(rewriter, up_acc_slices,
                    accumulator_base, tile, intermediate / tile, hemi_name(hemisphere), "fp32_accumulator");
                mlir::OperationState gate_acc(ffn.getLoc(), schedule::MemAccumulateOp::getOperationName());
                gate_acc.addOperands(gate_compute.getResult()); gate_acc.addTypes(projection_type);
                gate_acc.addAttributes({
                    rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(compute_cycle + gate_acc_latency)),
                    rewriter.getNamedAttr("stream_base",
                        rewriter.getI64IntegerAttr(
                            projection_result_stream_base)),
                    rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("address", ffn.getHidden0AddressAttr()),
                    rewriter.getNamedAttr("placement", gate_acc_placement),
                    rewriter.getNamedAttr("hemisphere", rewriter.getStringAttr(hemi_name(hemisphere))),
                    rewriter.getNamedAttr("destination",
                        rewriter.getStringAttr(
                            strategy == FfnScheduleStrategy::Fused
                                    && final_k_block
                                ? "stream"
                                : "sram")),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(tile)),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(intermediate / tile)),
                });
                auto gate_acc_op =
                    llvm::cast<schedule::MemAccumulateOp>(rewriter.create(gate_acc));
                mlir::OperationState up_acc(ffn.getLoc(), schedule::MemAccumulateOp::getOperationName());
                up_acc.addOperands(up_compute.getResult()); up_acc.addTypes(projection_type);
                up_acc.addAttributes({
                    rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(compute_cycle + up_acc_latency)),
                    rewriter.getNamedAttr("stream_base",
                        rewriter.getI64IntegerAttr(
                            projection_result_stream_base
                                + throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("address", ffn.getHidden1AddressAttr()),
                    rewriter.getNamedAttr("placement", up_acc_placement),
                    rewriter.getNamedAttr("hemisphere", rewriter.getStringAttr(hemi_name(hemisphere))),
                    rewriter.getNamedAttr("destination",
                        rewriter.getStringAttr(
                            strategy == FfnScheduleStrategy::Fused
                                    && final_k_block
                                ? "stream"
                                : "sram")),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(tile)),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(intermediate / tile)),
                });
                auto up_acc_op =
                    llvm::cast<schedule::MemAccumulateOp>(rewriter.create(up_acc));
                if (final_k_block) {
                    mlir::Value gate_temp;
                    mlir::Value up_temp;
                    int64_t deferred_ready_cycle =
                        compute_cycle
                        + std::max(
                            gate_acc_latency + tile
                                + west_latency(gate_acc_slices.front()),
                            up_acc_latency + tile
                                + west_latency(up_acc_slices.front()));
                    // A direct ACC-to-VXM bypass conflicts with later
                    // projection traffic on the physical fabric. The fused
                    // policy therefore clears ACC into isolated MEM planes.
                    if (strategy == FfnScheduleStrategy::Fused) {
                        const int64_t temp_base =
                            (pair * m_tile_count + m_tile) * tile;
                        const auto emit_temp_write =
                            [&](schedule::MemAccumulateOp source,
                                llvm::ArrayRef<int64_t> temp_slices,
                                llvm::ArrayRef<int64_t> acc_slices,
                                int64_t acc_cycle, int64_t stream_base,
                                mlir::Value& last_write) {
                                for (int64_t byte = 0;
                                     byte < throughput.mxm_result_streams;
                                     ++byte) {
                                    const int64_t target_slice =
                                        temp_slices[byte];
                                    const int64_t source_boundary =
                                        acc_slices.front()
                                        / target.streams()
                                              .mem_slices_per_register_group;
                                    const int64_t target_input_boundary =
                                        target_slice
                                            / target.streams()
                                                  .mem_slices_per_register_group
                                        + 1;
                                    const int64_t transport =
                                        source_boundary
                                        - target_input_boundary;
                                    const int64_t write_cycle =
                                        acc_cycle + transport + 1;
                                    auto placement = schedule_placement(
                                        rewriter, {target_slice}, temp_base,
                                        tile, 1, hemi_name(hemisphere),
                                        "fp32_swiglu_temp_byte");
                                    auto write =
                                        rewriter.create<schedule::MemWriteOp>(
                                            ffn.getLoc(), source.getOutput(),
                                            write_cycle, tile,
                                            stream_base + byte, 1,
                                            target_input_boundary,
                                            rewriter.getStringAttr("west"),
                                            ffn.getHidden1Address(), placement,
                                            tile
                                                * throughput.lanes_per_tile);
                                    last_write = write.getOutput();
                                    // A repeated MEM write owns its ICU queue
                                    // until all 32 rows have issued. Express
                                    // that interval in VXM-consume cycles so
                                    // the deferred allocator cannot place a
                                    // read from the same byte-plane queue over
                                    // any projection tile's temporary write.
                                    temp_mem_busy_windows[hemisphere].push_back({
                                        write_cycle
                                            + west_latency(target_slice),
                                        write_cycle + tile
                                            + west_latency(target_slice)});
                                    deferred_ready_cycle = std::max(
                                        deferred_ready_cycle,
                                        write_cycle + tile
                                            + west_latency(target_slice));
                                }
                            };
                        emit_temp_write(gate_acc_op,
                            memory.w8a16_fused_gate_temp_slices,
                            gate_acc_slices,
                            compute_cycle + gate_acc_latency,
                            projection_result_stream_base, gate_temp);
                        emit_temp_write(up_acc_op,
                            memory.w8a16_fused_up_temp_slices, up_acc_slices,
                            compute_cycle + up_acc_latency,
                            projection_result_stream_base
                                + throughput.mxm_result_streams,
                            up_temp);
                    }
                    completed_tiles.push_back({
                        pair, m_tile, hemisphere, compute_cycle,
                        deferred_ready_cycle, gate_acc_op, up_acc_op,
                        gate_temp, up_temp});
                }
            }
            }
    }

    const int64_t final_projection_cycle =
        projectionTimeline->final_projection_cycle;
    const int64_t accumulator_queue_release =
        projectionTimeline->accumulator_queue_release;
    mlir::Value last_hidden;
    int64_t last_swish_cycle = 0;
    const auto emit_swiglu_row = [&](mlir::Value gate_value, mlir::Value up_value,
                                     int64_t cycle, int64_t m_tile, int64_t pair,
                                     int64_t row, int64_t hemisphere) {
        last_swish_cycle = std::max(last_swish_cycle, cycle);
        last_hidden = emitFfnSwishRow(rewriter, ffn, target, strategy,
            hidden_slices, gate_value, up_value, cycle, m_tile, pair, row,
            hemisphere);
    };

    if (strategy == FfnScheduleStrategy::Tail) {
        // Preserve the original conservative schedule as the default policy.
        int64_t tail_swish_cycle =
            final_projection_cycle + accumulator_queue_release;
        for (int64_t m_tile = 0; m_tile < m_tile_count; ++m_tile) {
            for (int64_t pair = 0; pair < projection_pair_count; ++pair) {
                for (int64_t row = 0; row < tile; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres; ++hemisphere) {
                        const int64_t nblock =
                            pair * memory.hemispheres + hemisphere;
                        const int64_t address =
                            m_tile * tile * (intermediate / tile)
                            + row * (intermediate / tile) + nblock;
                        const int64_t cycle = tail_swish_cycle++;
                        mlir::Value gate_value, up_value;
                        for (int64_t byte = 0;
                             byte < throughput.mxm_result_streams; ++byte) {
                            gate_value = one_slice_read(ffn.getActivation(),
                                activation_route,
                                cycle - west_latency(gate_acc_slices[byte]),
                                gate_acc_slices[byte], address, 1, 1, byte,
                                "west", "vxm_fp32",
                                hemi_name(hemisphere)).getOutput();
                            up_value = one_slice_read(ffn.getActivation(),
                                activation_route,
                                cycle - west_latency(up_acc_slices[byte]),
                                up_acc_slices[byte], address, 1, 1,
                                throughput.mxm_result_streams + byte,
                                "west", "vxm_fp32",
                                hemi_name(hemisphere)).getOutput();
                        }
                        emit_swiglu_row(gate_value, up_value, cycle, m_tile,
                            pair, row, hemisphere);
                    }
                }
            }
        }
    } else {
        std::vector<schedule::FfnCycleWindow> dequant_windows;
        const int64_t dequant_window_cycles =
            memory.hemispheres * throughput.mxms_per_hemisphere
                * weight_load_cycles
            + 1;
        for (const schedule::FfnProjectionBlockSchedule& block :
             projectionTimeline->blocks) {
            const int64_t start = block.dequant_start;
            dequant_windows.push_back({start, start + dequant_window_cycles});
        }

        llvm::SmallVector<const CompletedProjectionTile*> deferred;
        for (const CompletedProjectionTile& completed : completed_tiles)
            deferred.push_back(&completed);
        llvm::sort(deferred,
            [&](const CompletedProjectionTile* lhs,
                const CompletedProjectionTile* rhs) {
                return lhs->compute_cycle < rhs->compute_cycle;
            });
        schedule::FfnSwishScheduleRequest swish_request;
        swish_request.dequant_windows = std::move(dequant_windows);
        swish_request.tile_rows = tile;
        for (int64_t hemisphere = 0;
             hemisphere < memory.hemispheres; ++hemisphere)
            swish_request.temp_mem_windows[hemisphere] =
                std::move(temp_mem_busy_windows[hemisphere]);
        for (const CompletedProjectionTile* completed : deferred)
            swish_request.tasks.push_back(
                {completed->deferred_ready_cycle, completed->hemisphere});
        auto swish_schedule =
            schedule::planFfnSwishCycles(swish_request, target);
        if (mlir::failed(swish_schedule)) return mlir::failure();

        for (std::size_t index = 0; index < deferred.size(); ++index) {
            const CompletedProjectionTile* completed = deferred[index];
            const int64_t start = (*swish_schedule)[index];
            for (int64_t row = 0; row < tile; ++row) {
                const int64_t cycle = start + row;
                const int64_t temp_base =
                    (completed->pair * m_tile_count + completed->m_tile)
                    * tile;
                mlir::Value gate_value, up_value;
                const int64_t temp_stream_base =
                    8 + completed->hemisphere * 8;
                for (int64_t byte = 0;
                     byte < throughput.mxm_result_streams; ++byte) {
                    const int64_t gate_slice =
                        memory.w8a16_fused_gate_temp_slices[byte];
                    const int64_t up_slice =
                        memory.w8a16_fused_up_temp_slices[byte];
                    gate_value = one_slice_read(completed->gate_temp,
                        activation_route,
                        cycle - west_latency(gate_slice), gate_slice,
                        temp_base + row, 1, 1,
                        temp_stream_base + byte,
                        "west", "vxm_fp32",
                        hemi_name(completed->hemisphere)).getOutput();
                    up_value = one_slice_read(completed->up_temp,
                        activation_route,
                        cycle - west_latency(up_slice), up_slice,
                        temp_base + row, 1, 1,
                        temp_stream_base
                            + throughput.mxm_result_streams + byte,
                        "west", "vxm_fp32",
                        hemi_name(completed->hemisphere)).getOutput();
                }
                emit_swiglu_row(gate_value, up_value, cycle,
                    completed->m_tile, completed->pair, row,
                    completed->hemisphere);
            }
        }
    }

    auto downTimeline = schedule::planFfnDownProjectionTimeline(
        {m, k, intermediate, n}, *projectionTimeline, last_swish_cycle,
        weight_slices, hidden_slices, result_slices, target);
    if (mlir::failed(downTimeline)) return mlir::failure();
    const int64_t down_wave_count = downTimeline->wave_count;
    mlir::Value final_value;
    for (const schedule::FfnDownBlockSchedule& block :
         downTimeline->blocks) {
            const int64_t output_wave = block.output_wave;
            const int64_t rb = block.reduction_block;
            const int64_t active_hemispheres = block.active_hemispheres;
            const int64_t dequant_start = block.dequant_start;
            const int64_t weight_buffer = block.weight_buffer;
            for (int64_t hemisphere = 0; hemisphere < active_hemispheres; ++hemisphere) {
            for (int64_t local_mxm = 0;
                 local_mxm < throughput.mxms_per_hemisphere; ++local_mxm) {
                const int64_t unit = hemisphere * throughput.mxms_per_hemisphere + local_mxm;
                const int64_t start = dequant_start + unit * weight_load_cycles;
                const int64_t base = down_route.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt()
                    + (output_wave * (intermediate / tile) + rb)
                        * throughput.mxms_per_hemisphere * weight_load_cycles
                    + local_mxm * weight_load_cycles;
                emitFfnWeightTile(rewriter, ffn.getLoc(), down_raw,
                    down_route.getInput().getType(), weight_slices, target,
                    ffn.getDownRhsScale().convertToFloat(), start, base,
                    hemisphere, local_mxm, unit, weight_buffer);
            }
            }
            const bool last = block.final_reduction;
            for (const schedule::FfnDownTileSchedule& tileSchedule :
                 block.tiles) {
                const int64_t m_tile = tileSchedule.m_tile;
                const int64_t compute_cycle = tileSchedule.compute_cycle;
                llvm::SmallVector<int64_t> segment_rows;
                llvm::SmallVector<int64_t> segment_streams;
                for (const schedule::FfnStreamSegment& segment :
                     tileSchedule.segments) {
                    segment_rows.push_back(segment.rows);
                    segment_streams.push_back(segment.stream_base);
                }
                for (int64_t hemisphere = 0;
                     hemisphere < active_hemispheres; ++hemisphere) {
                schedule::MxmComputeOp down0;
                schedule::MxmComputeOp down1;
                int64_t row_offset = 0;
                for (size_t segment = 0; segment < segment_rows.size(); ++segment) {
                    const int64_t rows = segment_rows[segment];
                    const int64_t stream_base = segment_streams[segment];
                    const int64_t segment_cycle = compute_cycle + row_offset;
                    mlir::Value hidden_value;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        hidden_value = one_slice_read(last_hidden, activation_route,
                            segment_cycle - east_mxm_latency(hidden_slices[byte]),
                            hidden_slices[byte],
                            rb * m + m_tile * tile + row_offset,
                            rows, 1, stream_base + byte,
                            "east", "activation", hemi_name(hemisphere)).getOutput();
                    }
                    const int64_t unit_base =
                        hemisphere * throughput.mxms_per_hemisphere;
                    down0 = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(), hidden_value,
                        ffn.getDownWeight0(), projection_type, segment_cycle, rows,
                        segment_cycle + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(rows), stream_base, 0,
                        weight_buffer, unit_base, rows, tile, tile);
                    down1 = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(), hidden_value,
                        ffn.getDownWeight1(), projection_type, segment_cycle, rows,
                        segment_cycle + target.mxm_first_result_latency(),
                        target.mxm_result_window_cycles(rows), stream_base,
                        throughput.mxm_result_streams,
                        weight_buffer, unit_base + 1, rows, tile, tile);
                    row_offset += rows;
                }
                const int64_t accumulator_base = down_accumulator_base + m_tile * tile;
                auto acc0_place = schedule_placement(rewriter, gate_acc_slices,
                    accumulator_base,
                    tile, 1, hemi_name(hemisphere), "fp32_accumulator");
                auto acc1_place = schedule_placement(rewriter, up_acc_slices,
                    accumulator_base,
                    tile, 1, hemi_name(hemisphere), "fp32_accumulator");
                auto make_acc = [&](mlir::Value input, mlir::DictionaryAttr placement,
                                    int64_t cycle, int64_t stream_base) {
                    mlir::OperationState state(
                        ffn.getLoc(), schedule::MemAccumulateOp::getOperationName());
                    state.addOperands(input);
                    state.addTypes(projection_type);
                    state.addAttributes({
                        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
                        rewriter.getNamedAttr("stream_base", rewriter.getI64IntegerAttr(stream_base)),
                        rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(
                            throughput.mxm_result_streams)),
                        rewriter.getNamedAttr("address", ffn.getResultAddressAttr()),
                        rewriter.getNamedAttr("placement", placement),
                        rewriter.getNamedAttr("hemisphere",
                            rewriter.getStringAttr(hemi_name(hemisphere))),
                        rewriter.getNamedAttr("destination", rewriter.getStringAttr(
                            last ? "stream" : "sram")),
                        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(tile)),
                        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                        rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(1)),
                    });
                    return llvm::cast<schedule::MemAccumulateOp>(rewriter.create(state));
                };
                auto acc0 = make_acc(down0.getResult(), acc0_place,
                    compute_cycle + gate_acc_latency, 0);
                auto acc1 = make_acc(down1.getResult(), acc1_place,
                    compute_cycle + up_acc_latency, throughput.mxm_result_streams);
                if (last) {
                    const int64_t output_to_vxm_latency = throughput.accumulator_to_vxm_latency;
                    // Keep the final VXM result off the down-activation pair.
                    // The last activation rows can still be in flight through
                    // MEM when the accumulator starts producing results.
                    const int64_t result_stream_base =
                        downTimeline->output_stream_base;
                    const int64_t output_cast_alu_base =
                        hemisphere * downTimeline->vxm_queues_per_hemisphere;
                    for (int64_t row = 0; row < tile; ++row) {
                        const int64_t vxm_cycle = compute_cycle + output_to_vxm_latency + row;
                        auto cast0 = create_vxm(rewriter, ffn.getLoc(), acc0.getOutput(), acc1.getOutput(),
                            ffn.getResult().getType(), vxm_cycle, output_cast_alu_base, "pass",
                            "stream_f32",
                            downTimeline->first_accumulator_stream, 0,
                            "immediate", 0, 0, "fp16",
                            result_stream_base, 1, 1,
                            hemi_name(hemisphere), hemi_name(hemisphere));
                        auto cast1 = create_vxm(rewriter, ffn.getLoc(), acc0.getOutput(), acc1.getOutput(),
                            ffn.getResult().getType(), vxm_cycle, output_cast_alu_base + 1,
                            "pass", "stream_f32",
                            downTimeline->second_accumulator_stream, 0,
                            "immediate", 0, 0, "fp16", result_stream_base + 2,
                            1, 1, hemi_name(hemisphere), hemi_name(hemisphere));
                        for (int64_t byte = 0; byte < throughput.mxm_result_streams; ++byte) {
                            auto placement = schedule_placement(rewriter, {result_slices[byte]},
                                output_wave * m + m_tile * tile + row,
                                1, 1, hemi_name(hemisphere), "fp16_pair_planar");
                            mlir::NamedAttrList attrs(placement);
                            llvm::SmallVector<mlir::Attribute> all_slices;
                            for (int64_t slice : result_slices)
                                all_slices.push_back(rewriter.getI64IntegerAttr(slice));
                            attrs.set("binding_slices", rewriter.getArrayAttr(all_slices));
                            attrs.set("binding_instruction_count", rewriter.getI64IntegerAttr(
                                ffn.getM() * down_wave_count));
                            auto binding_placement = schedule_placement(rewriter,
                                result_slices, 0, ffn.getM() * down_wave_count, 1,
                                "both", "fp16_pair_planar");
                            attrs.set("binding_placement", binding_placement);
                            auto write = rewriter.create<schedule::MemWriteOp>(ffn.getLoc(),
                                byte < 2 ? cast0.getResult() : cast1.getResult(),
                                vxm_cycle + 1 + result_slices[byte]
                                    / target.streams().mem_slices_per_register_group,
                                1,
                                result_stream_base + byte, 1, 0,
                                rewriter.getStringAttr("east"), ffn.getResultAddress(),
                                attrs.getDictionary(rewriter.getContext()), tile);
                            final_value = write.getOutput();
                        }
                    }
                }
            }
            }
    }
    return final_value;
}
} // namespace

mlir::FailureOr<mlir::Value> schedule::lowerFfnSchedule(
    mlir::IRRewriter& rewriter,
    schedule::PrimitiveFfnSchedulePlan& plan,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target)
{
    return lower_w8a16_ffn_schedule(rewriter, plan, strategy, target);
}

} // namespace ftlpu::compiler
