// Keep this translation unit rebuilt with target topology ABI changes.
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <string>

namespace ftlpu::compiler {
namespace {

int64_t get_slice(mlir::DictionaryAttr address)
{
    return address.getAs<mlir::IntegerAttr>("slice").getInt();
}

llvm::SmallVector<int64_t> get_slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute attribute : placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(attribute).getInt());
    return result;
}

mlir::DictionaryAttr subrange_placement(mlir::OpBuilder& builder,
    mlir::DictionaryAttr placement, int64_t row_offset, int64_t row_count)
{
    mlir::NamedAttrList attributes(placement);
    const int64_t base = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const int64_t stride = placement.getAs<mlir::IntegerAttr>("address_stride").getInt();
    attributes.set("base_row", builder.getI64IntegerAttr(base + row_offset * stride));
    attributes.set("instruction_count", builder.getI64IntegerAttr(row_count));
    return attributes.getDictionary(builder.getContext());
}

mlir::DictionaryAttr weight_pass_placement(mlir::OpBuilder& builder,
    mlir::DictionaryAttr placement, int64_t pass)
{
    mlir::NamedAttrList attributes(placement);
    const int64_t base = placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    attributes.set("base_row", builder.getI64IntegerAttr(base + pass * 320));
    attributes.set("instruction_count", builder.getI64IntegerAttr(20));
    return attributes.getDictionary(builder.getContext());
}

std::string mem_resource(int64_t slice)
{
    return llvm::formatv("MEM.slice.{0}", slice).str();
}

int64_t value_ready_cycle(mlir::Value value)
{
    if (auto write = value.getDefiningOp<schedule::MemWriteOp>())
        return write.getCycle() + write.getDuration();
    return 0;
}

void add_stream_windows(llvm::SmallVectorImpl<schedule::ResourceWindow>& windows,
    llvm::StringRef direction, int64_t base, int64_t count,
    int64_t offset, int64_t duration)
{
    for (int64_t stream = base; stream < base + count; ++stream) {
        windows.push_back({llvm::formatv("stream.{0}.{1}", direction, stream).str(),
            offset, duration});
    }
}

mlir::DictionaryAttr schedule_placement(mlir::OpBuilder& builder,
    llvm::ArrayRef<int64_t> slices, int64_t base_row, int64_t count,
    int64_t stride, llvm::StringRef hemisphere, llvm::StringRef kind)
{
    llvm::SmallVector<mlir::Attribute> slice_attrs;
    for (int64_t slice : slices) slice_attrs.push_back(builder.getI64IntegerAttr(slice));
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("hemisphere", builder.getStringAttr(hemisphere)),
        builder.getNamedAttr("slices", builder.getArrayAttr(slice_attrs)),
        builder.getNamedAttr("base_row", builder.getI64IntegerAttr(base_row)),
        builder.getNamedAttr("instruction_count", builder.getI64IntegerAttr(count)),
        builder.getNamedAttr("address_stride", builder.getI64IntegerAttr(stride)),
    });
}

void create_mem_queue_issue(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t address,
    int64_t packed_stream, int64_t repeat_count, int64_t repeat_interval,
    int64_t address_stride, llvm::StringRef accumulator_destination = "sram")
{
    mlir::OperationState state(location, schedule::MemQueueOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("address", rewriter.getI64IntegerAttr(address)),
        rewriter.getNamedAttr("packed_stream", rewriter.getI64IntegerAttr(packed_stream)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeat_count)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeat_interval)),
        rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(address_stride)),
        rewriter.getNamedAttr("accumulator_destination", rewriter.getStringAttr(accumulator_destination)),
    });
    rewriter.create(state);
}

void create_mxm_queue_issue(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t queue, llvm::StringRef opcode, int64_t weight_buffer,
    int64_t weight_column, int64_t activation_stream_base, int64_t output_stream_base,
    int64_t repeat_count, int64_t repeat_interval)
{
    mlir::OperationState state(location, schedule::MxmQueueOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("weight_buffer", rewriter.getI64IntegerAttr(weight_buffer)),
        rewriter.getNamedAttr("weight_column", rewriter.getI64IntegerAttr(weight_column)),
        rewriter.getNamedAttr("activation_stream_base", rewriter.getI64IntegerAttr(activation_stream_base)),
        rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(output_stream_base)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeat_count)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeat_interval)),
    });
    rewriter.create(state);
}

schedule::VxmOp create_vxm(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Value lhs_value, mlir::Value rhs_value, mlir::Type result_type,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    llvm::StringRef lhs_kind, int64_t lhs_index, float lhs_immediate,
    llvm::StringRef rhs_kind, int64_t rhs_index, float rhs_immediate,
    llvm::StringRef cast_target, int64_t output_stream,
    int64_t repeat_count, int64_t repeat_interval,
    llvm::StringRef input_hemisphere, llvm::StringRef output_hemisphere)
{
    mlir::OperationState state(location, schedule::VxmOp::getOperationName());
    state.addOperands({lhs_value, rhs_value});
    state.addTypes(result_type);
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("lhs_kind", rewriter.getStringAttr(lhs_kind)),
        rewriter.getNamedAttr("lhs_index", rewriter.getI64IntegerAttr(lhs_index)),
        rewriter.getNamedAttr("lhs_immediate", rewriter.getF32FloatAttr(lhs_immediate)),
        rewriter.getNamedAttr("rhs_kind", rewriter.getStringAttr(rhs_kind)),
        rewriter.getNamedAttr("rhs_index", rewriter.getI64IntegerAttr(rhs_index)),
        rewriter.getNamedAttr("rhs_immediate", rewriter.getF32FloatAttr(rhs_immediate)),
        rewriter.getNamedAttr("cast_target", rewriter.getStringAttr(cast_target)),
        rewriter.getNamedAttr("output_stream", rewriter.getI64IntegerAttr(output_stream)),
        rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(repeat_count)),
        rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(repeat_interval)),
        rewriter.getNamedAttr("input_hemisphere", rewriter.getStringAttr(input_hemisphere)),
        rewriter.getNamedAttr("output_hemisphere", rewriter.getStringAttr(output_hemisphere)),
    });
    return llvm::cast<schedule::VxmOp>(rewriter.create(state));
}

mlir::FailureOr<mlir::Value> lower_w8a16_ffn_schedule(
    mlir::IRRewriter& rewriter, stream::FfnOp ffn)
{
    const target::LPUTargetModel target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = throughput.mxm_rows;
    const int64_t m = ffn.getM();
    const int64_t k = ffn.getK();
    const int64_t intermediate = ffn.getHidden();
    const int64_t n = ffn.getN();
    const int64_t weight_to_iw = throughput.vxm_weight_to_iw_latency;
    const int64_t compute_block_cycles = tile + 4 * throughput.mxm_pipeline_rows;
    const int64_t gate_acc_latency = throughput.mxm0_accumulator_latency;
    const int64_t up_acc_latency = throughput.mxm1_accumulator_latency;
    const int64_t swish_write_latency = throughput.swiglu_write_latency;
    const int64_t down_accumulator_base = memory.accumulator_scratch_base_row;

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
    // One 32x32 weight tile is dequantized eight columns at a time, so each
    // single MXM consumes four VXM issue cycles. Gate/up on east/west are four
    // independent tiles and are staggered because they share the VXM ALU ICU
    // queues; that aggregate window must not be confused with one dequant.
    const int64_t weight_load_cycles = tile / throughput.lanes_per_tile;
    const int64_t activation_latency = *target.transport_latency(
        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East, activation_slices.front());

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
    int64_t max_weight_west_latency = 0;
    for (int64_t slice : weight_slices)
        max_weight_west_latency = std::max(max_weight_west_latency, west_latency(slice));
    const int64_t pipelined_block_interval = target.mxm_block_issue_interval();
    const int64_t initial_compute_cycle = max_weight_west_latency + 1 + tile;
    const int64_t projection_pair_count =
        intermediate / (memory.hemispheres * tile);
    const int64_t m_tile_count = m / tile;
    // A loaded KxN weight tile stays resident while every M=32 activation tile
    // of the sequence consumes it. Consecutive blocks on the same MXM need the
    // target's complete issue interval, including its pipeline reset cycles.
    const int64_t weight_block_interval = m_tile_count * pipelined_block_interval;
    rewriter.setInsertionPoint(ffn);
    int64_t projection_block = 0;
    for (int64_t pair = 0; pair < intermediate / (memory.hemispheres * tile); ++pair) {
        for (int64_t kb = 0; kb < k / tile; ++kb) {
            const int64_t weight_compute_cycle = initial_compute_cycle
                + projection_block * weight_block_interval;
            const int64_t dequant_lead = tile;
            const int64_t dequant_start = weight_compute_cycle - dequant_lead;
            const int64_t weight_buffer = projection_block % 2;
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
                    mlir::Value read_value;
                    for (int64_t stream = 0; stream < gate_raw.getStreamCount(); ++stream) {
                        auto read = one_slice_read(raw.getInput(), raw,
                            start - west_latency(weight_slices[stream]), weight_slices[stream],
                            base, weight_load_cycles, 1, stream,
                            "west", "weight_i8", hemi_name(hemisphere));
                        read_value = read.getOutput();
                    }
                    mlir::Value vxm_value = read_value;
                    for (int64_t stream = 0; stream < gate_raw.getStreamCount(); ++stream) {
                        vxm_value = create_vxm(rewriter, ffn.getLoc(), read_value, read_value,
                            cooked.getInput().getType(), start, stream, "multiply",
                            "stream_i8", 32 + stream, 0.0f, "immediate", 0,
                            local_mxm == 0 ? ffn.getGateScale().convertToFloat()
                                           : ffn.getUpScale().convertToFloat(),
                            "fp32", -1, weight_load_cycles, 1,
                            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                        vxm_value = create_vxm(rewriter, ffn.getLoc(), vxm_value, read_value,
                            cooked.getInput().getType(), start + 1, 8 + stream, "cast",
                            "alu", stream, 0.0f, "immediate", 0, 0.0f, "fp16",
                            local_mxm * throughput.mxm_load_streams_per_cycle + stream * 2,
                            weight_load_cycles, 1,
                            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                    }
                    rewriter.create<schedule::MxmLoadOp>(ffn.getLoc(), vxm_value,
                        start + weight_to_iw, weight_load_cycles, 0,
                        throughput.mxm_load_streams_per_cycle,
                        hemisphere * throughput.mxms_per_hemisphere + local_mxm,
                        weight_buffer);
                }
            }

            const bool has_next_weight_block = projection_block + 1
                < (intermediate / (memory.hemispheres * tile)) * (k / tile);
            for (int64_t m_tile = 0; m_tile < m_tile_count; ++m_tile) {
                const int64_t compute_cycle = weight_compute_cycle
                    + m_tile * pipelined_block_interval;
                const bool prefetch_next_weight = has_next_weight_block
                    && m_tile + 1 == m_tile_count;
                for (int64_t hemisphere = 0; hemisphere < memory.hemispheres; ++hemisphere) {
                // Activation is laid out as [K-tile][full M rows]. Select the
                // current K block first, then the M=32 subrange within it.
                const int64_t activation_base = kb * m + m_tile * tile;
                llvm::SmallVector<int64_t> segment_rows;
                llvm::SmallVector<int64_t> segment_streams;
                const int64_t next_dequant_lead = tile;
                const int64_t next_weight_distance = prefetch_next_weight
                    ? weight_block_interval - m_tile * pipelined_block_interval
                    : tile + next_dequant_lead;
                const int64_t switch_row = next_weight_distance - next_dequant_lead
                    + weight_to_iw
                    + hemisphere * throughput.mxms_per_hemisphere * weight_load_cycles;
                if (!prefetch_next_weight || switch_row >= tile) {
                    segment_rows = {tile};
                    segment_streams = {0};
                } else {
                    if (switch_row > 0) {
                        segment_rows.push_back(switch_row);
                        segment_streams.push_back(0);
                    }
                    const int64_t switched_rows = std::min(weight_load_cycles,
                        tile - switch_row);
                    segment_rows.push_back(switched_rows);
                    segment_streams.push_back(throughput.mxm_load_streams_per_cycle);
                    if (switch_row + switched_rows < tile) {
                        segment_rows.push_back(tile - switch_row - switched_rows);
                        segment_streams.push_back(0);
                    }
                }
                schedule::MxmComputeOp gate_compute;
                schedule::MxmComputeOp up_compute;
                int64_t row_offset = 0;
                for (size_t segment = 0; segment < segment_rows.size(); ++segment) {
                    const int64_t rows = segment_rows[segment];
                    const int64_t stream_base = segment_streams[segment];
                    mlir::Value activation_value;
                    const int64_t segment_cycle = compute_cycle + row_offset;
                    for (int64_t byte = 0;
                         byte < throughput.mxm_activation_streams; ++byte) {
                        auto read = one_slice_read(activation_route.getInput(), activation_route,
                            segment_cycle - activation_latency, activation_slices[byte],
                            activation_base + row_offset, rows, 1, stream_base + byte,
                            "east", "activation", hemi_name(hemisphere));
                        activation_value = read.getOutput();
                    }
                    gate_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        activation_value, ffn.getGateWeight(), projection_type,
                        segment_cycle, rows, segment_cycle + 3, rows + 3,
                        stream_base, 0, weight_buffer,
                        hemisphere * throughput.mxms_per_hemisphere,
                        rows, tile, tile);
                    up_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        activation_value, ffn.getUpWeight(), projection_type,
                        segment_cycle, rows, segment_cycle + 3, rows + 3,
                        stream_base + 2, throughput.mxm_result_streams,
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
                    rewriter.getNamedAttr("stream_base", rewriter.getI64IntegerAttr(0)),
                    rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("address", ffn.getHidden0AddressAttr()),
                    rewriter.getNamedAttr("placement", gate_acc_placement),
                    rewriter.getNamedAttr("hemisphere", rewriter.getStringAttr(hemi_name(hemisphere))),
                    rewriter.getNamedAttr("destination", rewriter.getStringAttr("sram")),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(tile)),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(intermediate / tile)),
                });
                rewriter.create(gate_acc);
                mlir::OperationState up_acc(ffn.getLoc(), schedule::MemAccumulateOp::getOperationName());
                up_acc.addOperands(up_compute.getResult()); up_acc.addTypes(projection_type);
                up_acc.addAttributes({
                    rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(compute_cycle + up_acc_latency)),
                    rewriter.getNamedAttr("stream_base", rewriter.getI64IntegerAttr(throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("address", ffn.getHidden1AddressAttr()),
                    rewriter.getNamedAttr("placement", up_acc_placement),
                    rewriter.getNamedAttr("hemisphere", rewriter.getStringAttr(hemi_name(hemisphere))),
                    rewriter.getNamedAttr("destination", rewriter.getStringAttr("sram")),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(tile)),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(intermediate / tile)),
                });
                rewriter.create(up_acc);
            }
            }
            ++projection_block;
        }
    }

    const int64_t final_projection_cycle = initial_compute_cycle
        + (projection_block - 1) * weight_block_interval
        + (m_tile_count - 1) * pipelined_block_interval;
    const int64_t accumulator_queue_release = std::max(
        gate_acc_latency + tile + west_latency(gate_acc_slices.front()),
        up_acc_latency + tile + west_latency(up_acc_slices.front()));
    // There is no fixed overlap batch. Until all projection resources are
    // modeled by the exact scheduler, conservatively start SwiGLU at the tail.
    int64_t tail_swish_cycle = final_projection_cycle + accumulator_queue_release;
    mlir::Value last_hidden;
    int64_t last_swish_cycle = 0;
    for (int64_t m_tile = 0; m_tile < m_tile_count; ++m_tile) {
        for (int64_t pair = 0; pair < projection_pair_count; ++pair) {
            for (int64_t row = 0; row < tile; ++row) {
                for (int64_t hemisphere = 0; hemisphere < memory.hemispheres; ++hemisphere) {
                const int64_t nblock = pair * memory.hemispheres + hemisphere;
                const int64_t address = m_tile * tile * (intermediate / tile)
                    + row * (intermediate / tile) + nblock;
                const int64_t swish_input_stream = 0;
                const int64_t swish_output_stream = 0;
                const int64_t cycle = tail_swish_cycle++;
                last_swish_cycle = std::max(last_swish_cycle, cycle);
                mlir::Value gate_value, up_value;
                for (int64_t byte = 0; byte < throughput.mxm_result_streams; ++byte) {
                    gate_value = one_slice_read(ffn.getActivation(), activation_route,
                        cycle - west_latency(gate_acc_slices[byte]), gate_acc_slices[byte], address, 1, 1,
                        swish_input_stream + byte,
                        "west", "vxm_fp32", hemi_name(hemisphere)).getOutput();
                    up_value = one_slice_read(ffn.getActivation(), activation_route,
                        cycle - west_latency(up_acc_slices[byte]), up_acc_slices[byte], address, 1, 1,
                        swish_input_stream + throughput.mxm_result_streams + byte,
                        "west", "vxm_fp32", hemi_name(hemisphere)).getOutput();
                }
                mlir::Value value;
                value = create_vxm(rewriter, ffn.getLoc(), gate_value, up_value, ffn.getResult().getType(),
                    cycle, 0, "negate", "stream_f32", 32 + swish_input_stream, 0,
                    "immediate", 0, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), gate_value, up_value, ffn.getResult().getType(),
                    cycle, 1, "multiply", "stream_f32", 32 + swish_input_stream, 0,
                    "stream_f32", 32 + swish_input_stream + throughput.mxm_result_streams,
                    0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 1, 2, "exp", "alu", 0, 0, "immediate", 0, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 1, 5, "pass", "alu", 1, 0, "immediate", 0, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 2, 3, "add", "alu", 2, 0, "immediate", 0, 1, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 2, 6, "pass", "alu", 5, 0, "immediate", 0, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 3, 4, "divide", "immediate", 0, 1, "alu", 3, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 3, 7, "pass", "alu", 6, 0, "immediate", 0, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                value = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 4, 8, "multiply", "alu", 7, 0, "alu", 4, 0, "fp32", -1, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                auto cast0 = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 5, 9, "cast", "alu", 8, 0, "immediate", 0, 0, "fp16",
                    swish_output_stream, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere));
                auto cast1 = create_vxm(rewriter, ffn.getLoc(), value, up_value, ffn.getResult().getType(),
                    cycle + 5, 10, "cast", "alu", 8, 0, "immediate", 0, 0, "fp16",
                    swish_output_stream + 2, 1, 1,
                    hemi_name(hemisphere), hemi_name(hemisphere));
                for (int64_t byte = 0; byte < throughput.mxm_activation_streams; ++byte) {
                    auto placement = schedule_placement(rewriter, {hidden_slices[byte]},
                        nblock * m + m_tile * tile + row, 1, 1, hemi_name(hemisphere),
                        "fp16_mxm_activation_planar");
                    auto write = rewriter.create<schedule::MemWriteOp>(ffn.getLoc(),
                        byte < 2 ? cast0.getResult() : cast1.getResult(),
                        cycle + 6 + hidden_slices[byte]
                            / target.streams().mem_slices_per_register_group,
                        1, swish_output_stream + byte, 1, 0,
                        rewriter.getStringAttr("east"), ffn.getHidden0Address(),
                        placement, tile);
                    last_hidden = write.getOutput();
                }
                }
            }
        }
    }

    const int64_t slowest_hidden_west_latency = west_latency(hidden_slices.back());
    int64_t copy_cycle = last_swish_cycle + 1
        + swish_write_latency + slowest_hidden_west_latency + 1;
    for (int64_t m_tile = 0; m_tile < m_tile_count; ++m_tile) {
        for (int64_t nblock = 1; nblock < intermediate / tile;
             nblock += memory.hemispheres) {
            for (int64_t row = 0; row < tile; ++row, ++copy_cycle) {
            mlir::Value read_value;
            for (int64_t byte = 0; byte < throughput.mxm_activation_streams; ++byte) {
                read_value = one_slice_read(last_hidden, activation_route,
                    copy_cycle - west_latency(hidden_slices[byte]), hidden_slices[byte],
                    nblock * m + m_tile * tile + row, 1, 1, byte,
                    "west", "vxm_fp16", "west").getOutput();
            }
            auto copy0 = create_vxm(rewriter, ffn.getLoc(), read_value, read_value,
                ffn.getResult().getType(), copy_cycle, 0, "pass", "stream_f16", 32, 0,
                "immediate", 0, 0, "fp16", 0, 1, 1, "west", "east");
            auto copy1 = create_vxm(rewriter, ffn.getLoc(), read_value, read_value,
                ffn.getResult().getType(), copy_cycle, 1, "pass", "stream_f16", 34, 0,
                "immediate", 0, 0, "fp16", 2, 1, 1, "west", "east");
            for (int64_t byte = 0; byte < throughput.mxm_activation_streams; ++byte) {
                auto placement = schedule_placement(rewriter, {hidden_slices[byte]},
                    nblock * m + m_tile * tile + row, 1, 1, "east",
                    "fp16_mxm_activation_planar");
                rewriter.create<schedule::MemWriteOp>(ffn.getLoc(),
                    byte < 2 ? copy0.getResult() : copy1.getResult(),
                    copy_cycle + 1 + hidden_slices[byte]
                        / target.streams().mem_slices_per_register_group,
                    1, byte, 1, 0,
                    rewriter.getStringAttr("east"), ffn.getHidden1Address(), placement, tile);
            }
            }
        }
    }

    int64_t phase_start = copy_cycle + throughput.accumulator_to_vxm_latency;
    int64_t down_pair_transition_interval =
        2 * tile + throughput.accumulator_to_vxm_latency;
    for (int64_t result_slice : result_slices) {
        if (std::find(weight_slices.begin(), weight_slices.end(), result_slice)
            == weight_slices.end())
            continue;
        const int64_t last_write_end = throughput.accumulator_to_vxm_latency
            + tile + result_slice / target.streams().mem_slices_per_register_group + 1;
        down_pair_transition_interval = std::max(down_pair_transition_interval,
            last_write_end + tile + west_latency(result_slice));
    }
    int64_t down_compute_cycle = phase_start + initial_compute_cycle;
    int64_t down_block = 0;
    mlir::Value final_value;
    for (int64_t output_pair = 0;
         output_pair < n / (throughput.mxms_per_hemisphere * tile); ++output_pair) {
        for (int64_t rb = 0; rb < intermediate / tile; ++rb) {
            const int64_t compute_cycle = down_compute_cycle;
            const int64_t dequant_start = compute_cycle - tile;
            const int64_t weight_buffer = down_block % 2;
            for (int64_t local_mxm = 0;
                 local_mxm < throughput.mxms_per_hemisphere; ++local_mxm) {
                const int64_t start = dequant_start + local_mxm * weight_load_cycles;
                const int64_t base = down_route.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt()
                    + (output_pair * (intermediate / tile) + rb)
                        * throughput.mxms_per_hemisphere * weight_load_cycles
                    + local_mxm * weight_load_cycles;
                mlir::Value read_value;
                for (int64_t stream = 0; stream < down_raw.getStreamCount(); ++stream) {
                    read_value = one_slice_read(down_raw.getInput(), down_raw,
                        start - west_latency(weight_slices[stream]), weight_slices[stream],
                        base, weight_load_cycles, 1, stream,
                        "west", "weight_i8", "east").getOutput();
                }
                mlir::Value value = read_value;
                for (int64_t stream = 0; stream < down_raw.getStreamCount(); ++stream) {
                    value = create_vxm(rewriter, ffn.getLoc(), read_value, read_value,
                        down_route.getInput().getType(), start, stream, "multiply",
                        "stream_i8", 32 + stream, 0, "immediate", 0,
                        ffn.getDownRhsScale().convertToFloat(), "fp32", -1,
                        weight_load_cycles, 1,
                        "east", "east").getResult();
                    value = create_vxm(rewriter, ffn.getLoc(), value, read_value,
                        down_route.getInput().getType(), start + 1, 8 + stream, "cast",
                        "alu", stream, 0, "immediate", 0, 0, "fp16",
                        local_mxm * throughput.mxm_load_streams_per_cycle + stream * 2,
                        weight_load_cycles, 1, "east", "east").getResult();
                }
                rewriter.create<schedule::MxmLoadOp>(ffn.getLoc(), value,
                    start + weight_to_iw, weight_load_cycles, 0,
                    throughput.mxm_load_streams_per_cycle,
                    local_mxm, weight_buffer);
            }
            const bool last = rb + 1 == intermediate / tile;
            const int64_t switch_row = pipelined_block_interval - tile + weight_to_iw;
            llvm::SmallVector<int64_t> segment_rows;
            llvm::SmallVector<int64_t> segment_streams;
            if (last || switch_row >= tile) {
                segment_rows = {tile};
                segment_streams = {0};
            } else {
                if (switch_row > 0) {
                    segment_rows.push_back(switch_row);
                    segment_streams.push_back(0);
                }
                const int64_t switched_rows = std::min(weight_load_cycles, tile - switch_row);
                segment_rows.push_back(switched_rows);
                segment_streams.push_back(throughput.mxm_load_streams_per_cycle);
                if (switch_row + switched_rows < tile) {
                    segment_rows.push_back(tile - switch_row - switched_rows);
                    segment_streams.push_back(0);
                }
            }
            schedule::MxmComputeOp down0;
            schedule::MxmComputeOp down1;
            int64_t row_offset = 0;
            for (size_t segment = 0; segment < segment_rows.size(); ++segment) {
                const int64_t rows = segment_rows[segment];
                const int64_t stream_base = segment_streams[segment];
                const int64_t segment_cycle = compute_cycle + row_offset;
                mlir::Value hidden_value;
                for (int64_t byte = 0; byte < throughput.mxm_activation_streams; ++byte) {
                    hidden_value = one_slice_read(last_hidden, activation_route,
                        segment_cycle - east_mxm_latency(hidden_slices[byte]),
                        hidden_slices[byte], rb * tile + row_offset,
                        rows, 1, stream_base + byte,
                        "east", "activation", "east").getOutput();
                }
                down0 = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(), hidden_value,
                    ffn.getDownWeight0(), projection_type, segment_cycle, rows,
                    segment_cycle + 3, rows + 3, stream_base, 0,
                    weight_buffer, 0, rows, tile, tile);
                down1 = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(), hidden_value,
                    ffn.getDownWeight1(), projection_type, segment_cycle, rows,
                    segment_cycle + 3, rows + 3, stream_base + 2,
                    throughput.mxm_result_streams,
                    weight_buffer, 1, rows, tile, tile);
                row_offset += rows;
            }
            auto acc0_place = schedule_placement(rewriter, gate_acc_slices,
                down_accumulator_base,
                tile, 1, "east", "fp32_accumulator");
            auto acc1_place = schedule_placement(rewriter, up_acc_slices,
                down_accumulator_base,
                tile, 1, "east", "fp32_accumulator");
            auto make_acc = [&](mlir::Value input, mlir::DictionaryAttr placement,
                                int64_t cycle, int64_t stream_base) {
                mlir::OperationState state(ffn.getLoc(), schedule::MemAccumulateOp::getOperationName());
                state.addOperands(input); state.addTypes(projection_type);
                state.addAttributes({
                    rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
                    rewriter.getNamedAttr("stream_base", rewriter.getI64IntegerAttr(stream_base)),
                    rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(
                        throughput.mxm_result_streams)),
                    rewriter.getNamedAttr("address", ffn.getResultAddressAttr()),
                    rewriter.getNamedAttr("placement", placement),
                    rewriter.getNamedAttr("hemisphere", rewriter.getStringAttr("east")),
                    rewriter.getNamedAttr("destination", rewriter.getStringAttr(last ? "stream" : "sram")),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(tile)),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("address_stride", rewriter.getI64IntegerAttr(1)),
                });
                return llvm::cast<schedule::MemAccumulateOp>(rewriter.create(state));
            };
            auto acc0 = make_acc(down0.getResult(), acc0_place, compute_cycle + gate_acc_latency, 0);
            auto acc1 = make_acc(down1.getResult(), acc1_place,
                compute_cycle + up_acc_latency, throughput.mxm_result_streams);
            if (last) {
                const int64_t output_to_vxm_latency = throughput.accumulator_to_vxm_latency;
                const int64_t result_stream_base = 2 * throughput.mxm_result_streams;
                for (int64_t row = 0; row < tile; ++row) {
                    const int64_t vxm_cycle = compute_cycle + output_to_vxm_latency + row;
                    auto cast0 = create_vxm(rewriter, ffn.getLoc(), acc0.getOutput(), acc1.getOutput(),
                        ffn.getResult().getType(), vxm_cycle, 0, "pass", "stream_f32", 32, 0,
                        "immediate", 0, 0, "fp16", result_stream_base, 1, 1, "east", "east");
                    auto cast1 = create_vxm(rewriter, ffn.getLoc(), acc0.getOutput(), acc1.getOutput(),
                        ffn.getResult().getType(), vxm_cycle, 1, "pass", "stream_f32", 36, 0,
                        "immediate", 0, 0, "fp16", result_stream_base + 2,
                        1, 1, "east", "east");
                    for (int64_t byte = 0; byte < throughput.mxm_result_streams; ++byte) {
                        auto placement = schedule_placement(rewriter, {result_slices[byte]},
                            output_pair * tile + row, 1, 1, "east", "fp16_pair_planar");
                        mlir::NamedAttrList attrs(placement);
                        llvm::SmallVector<mlir::Attribute> all_slices;
                        for (int64_t slice : result_slices)
                            all_slices.push_back(rewriter.getI64IntegerAttr(slice));
                        attrs.set("binding_slices", rewriter.getArrayAttr(all_slices));
                        attrs.set("binding_instruction_count", rewriter.getI64IntegerAttr(
                            ffn.getM() * n
                                / (tile * throughput.mxms_per_hemisphere)));
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
            ++down_block;
            if (last) {
                if (output_pair + 1
                    < n / (throughput.mxms_per_hemisphere * tile))
                    down_compute_cycle += down_pair_transition_interval;
            } else {
                down_compute_cycle += pipelined_block_interval;
            }
        }
    }
    return final_value;
}

class LowerStreamToSchedulePass final
    : public mlir::PassWrapper<LowerStreamToSchedulePass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStreamToSchedulePass)

    llvm::StringRef getArgument() const final { return "ftlpu-stream-to-schedule"; }
    llvm::StringRef getDescription() const final
    {
        return "Schedules LPU stream routes at exact CModel issue cycles";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError("cycle scheduling currently requires a single-block function");
            signalPassFailure();
            return;
        }

        llvm::SmallVector<stream::MatmulOp> matmuls;
        llvm::SmallVector<stream::SwigluOp> swiglus;
        llvm::SmallVector<stream::FfnOp> ffns;
        llvm::SmallVector<stream::AttentionOp> attentions;
        function.walk([&](stream::MatmulOp op) { matmuls.push_back(op); });
        function.walk([&](stream::SwigluOp op) { swiglus.push_back(op); });
        function.walk([&](stream::FfnOp op) { ffns.push_back(op); });
        // Attention is a function-body operation. Use its registered name here so this
        // boundary remains robust while preceding passes construct it generically.
        for (mlir::Operation& operation : function.getBody().front()) {
            if (operation.getName().getStringRef() == stream::AttentionOp::getOperationName())
                attentions.emplace_back(&operation);
        }
        const target::LPUTargetModel target;
        schedule::ResourceScheduler scheduler;
        mlir::IRRewriter rewriter(&getContext());

        for (stream::AttentionOp op : attentions) {
            schedule::AttentionScheduleEmitter emitter(rewriter, op, target);
            auto lowered = emitter.emit();
            if (mlir::failed(lowered)) {
                signalPassFailure();
                return;
            }
            rewriter.replaceOp(op, lowered->getResult());
        }

        for (stream::FfnOp ffn : ffns) {
            const bool w8a16 = ffn.getActivation().getType().getElementType().isF16()
                && ffn.getGateWeight().getType().getElementType().isF16()
                && ffn.getResult().getType().getElementType().isF16()
                && target.supports_w8a16_ffn_shape(
                    ffn.getM(), ffn.getK(), ffn.getHidden(), ffn.getN());
            if (w8a16) {
                auto result = lower_w8a16_ffn_schedule(rewriter, ffn);
                if (mlir::failed(result)) {
                    ffn.emitError("failed to schedule a tile-aligned W8A16 FFN");
                    signalPassFailure(); return;
                }
                rewriter.replaceOp(ffn, *result);
                continue;
            }
            auto activation_route = ffn.getActivation().getDefiningOp<stream::RouteOp>();
            auto gate_route = ffn.getGateWeight().getDefiningOp<stream::RouteOp>();
            auto up_route = ffn.getUpWeight().getDefiningOp<stream::RouteOp>();
            auto down0_route = ffn.getDownWeight0().getDefiningOp<stream::RouteOp>();
            auto down1_route = ffn.getDownWeight1().getDefiningOp<stream::RouteOp>();
            if (!activation_route || !gate_route || !up_route || !down0_route || !down1_route) {
                ffn.emitError("requires canonical complete-FFN routes");
                signalPassFailure(); return;
            }
            rewriter.setInsertionPoint(ffn);
            auto emit_weight_load = [&](stream::RouteOp route, int64_t pass,
                                        int64_t unit, int64_t buffer, int64_t load_cycle) {
                int64_t latency = 0;
                for (int64_t slice : get_slices(route.getPlacement()))
                    latency = std::max(latency, *target.transport_latency(
                        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                        target::StreamDirection::East, slice));
                auto placement = weight_pass_placement(rewriter, route.getPlacement(), pass);
                auto read = rewriter.create<schedule::MemReadOp>(ffn.getLoc(), route.getInput(),
                    load_cycle - latency, 20, route.getStreamBase(), route.getStreamCount(),
                    route.getRegisterId(), route.getDirectionAttr(), rewriter.getStringAttr("weight"),
                    route.getAddress(), placement, 320 * 320);
                return rewriter.create<schedule::MxmLoadOp>(ffn.getLoc(), read.getOutput(),
                    load_cycle, 20, route.getStreamBase(), route.getStreamCount(), unit, buffer);
            };

            auto gate0_load = emit_weight_load(gate_route, 0, 0, 0, 18);
            auto up0_load = emit_weight_load(up_route, 0, 1, 0, 38);
            auto gate1_load = emit_weight_load(gate_route, 1, 0, 0, 278);
            auto up1_load = emit_weight_load(up_route, 1, 1, 0, 298);
            auto down0_load = emit_weight_load(down0_route, 0, 0, 0, 538);
            auto down1_load = emit_weight_load(down1_route, 1, 1, 0, 558);

            const auto projection_type = mlir::RankedTensorType::get(
                {ffn.getMAttr().getInt(), 320}, rewriter.getI32Type());
            auto emit_vxm = [&](mlir::Value lhs, mlir::Value rhs, mlir::Type result_type,
                                int64_t cycle, int64_t queue, llvm::StringRef opcode,
                                llvm::StringRef lhs_kind, int64_t lhs_index, float lhs_imm,
                                llvm::StringRef rhs_kind, int64_t rhs_index, float rhs_imm,
                                llvm::StringRef cast_target, int64_t output_stream) {
                mlir::OperationState state(ffn.getLoc(), schedule::VxmOp::getOperationName());
                state.addOperands({lhs, rhs});
                state.addTypes(result_type);
                state.addAttributes({
                    rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
                    rewriter.getNamedAttr("queue", rewriter.getI64IntegerAttr(queue)),
                    rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
                    rewriter.getNamedAttr("lhs_kind", rewriter.getStringAttr(lhs_kind)),
                    rewriter.getNamedAttr("lhs_index", rewriter.getI64IntegerAttr(lhs_index)),
                    rewriter.getNamedAttr("lhs_immediate", rewriter.getF32FloatAttr(lhs_imm)),
                    rewriter.getNamedAttr("rhs_kind", rewriter.getStringAttr(rhs_kind)),
                    rewriter.getNamedAttr("rhs_index", rewriter.getI64IntegerAttr(rhs_index)),
                    rewriter.getNamedAttr("rhs_immediate", rewriter.getF32FloatAttr(rhs_imm)),
                    rewriter.getNamedAttr("cast_target", rewriter.getStringAttr(cast_target)),
                    rewriter.getNamedAttr("output_stream", rewriter.getI64IntegerAttr(output_stream)),
                    rewriter.getNamedAttr("repeat_count", rewriter.getI64IntegerAttr(ffn.getM())),
                    rewriter.getNamedAttr("repeat_interval", rewriter.getI64IntegerAttr(1)),
                    rewriter.getNamedAttr("input_hemisphere", rewriter.getStringAttr("east")),
                    rewriter.getNamedAttr("output_hemisphere", rewriter.getStringAttr("east")),
                });
                return llvm::cast<schedule::VxmOp>(rewriter.create(state)).getResult();
            };

            struct ProjectionResult { mlir::Value vxm; mlir::Value stored; };
            auto emit_projection = [&](schedule::MxmLoadOp gate_load,
                                       schedule::MxmLoadOp up_load, int64_t buffer,
                                       int64_t compute_cycle, mlir::DictionaryAttr hidden_address,
                                       mlir::DictionaryAttr hidden_placement) {
                schedule::MxmComputeOp gate_compute;
                schedule::MxmComputeOp up_compute;
                const int64_t rows[] = {15, 4, ffn.getMAttr().getInt() - 19};
                const int64_t streams[] = {16, 30, 0};
                int64_t row_offset = 0;
                for (int segment = 0; segment < 3; ++segment) {
                    if (rows[segment] <= 0) continue;
                    auto placement = subrange_placement(rewriter,
                        activation_route.getPlacement(), row_offset, rows[segment]);
                    auto read = rewriter.create<schedule::MemReadOp>(ffn.getLoc(),
                        activation_route.getInput(), compute_cycle - activation_route.getTransportLatency()
                            + row_offset, rows[segment], streams[segment], 1,
                        activation_route.getRegisterId(), activation_route.getDirectionAttr(),
                        rewriter.getStringAttr("activation"), activation_route.getAddress(),
                        placement, rows[segment] * 320);
                    gate_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        read.getOutput(), gate_load.getOutput(), projection_type,
                        compute_cycle + row_offset, rows[segment], compute_cycle + 19 + row_offset,
                        target.mxm_result_window_cycles(rows[segment]), streams[segment],
                        ffn.getGateOutputStreamBase(), buffer, 0, rows[segment], 320, 320);
                    up_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        read.getOutput(), up_load.getOutput(), projection_type,
                        compute_cycle + row_offset, rows[segment], compute_cycle + 19 + row_offset,
                        target.mxm_result_window_cycles(rows[segment]), streams[segment],
                        ffn.getUpOutputStreamBase(), buffer, 1, rows[segment], 320, 320);
                    row_offset += rows[segment];
                }
                const int64_t start = compute_cycle + 19
                    + target.streams().mem_boundary_register_columns;
                mlir::Value value;
                const int64_t gate_stream = 32 + ffn.getGateOutputStreamBase();
                const int64_t up_stream = 32 + ffn.getUpOutputStreamBase();
                auto issue = [&](int64_t stage, int64_t alu, llvm::StringRef opcode,
                                 llvm::StringRef lhs_kind, int64_t lhs_index, float lhs_imm,
                                 llvm::StringRef rhs_kind, int64_t rhs_index, float rhs_imm,
                                 llvm::StringRef cast_target, int64_t output_stream) {
                    value = emit_vxm(gate_compute.getResult(), up_compute.getResult(),
                        ffn.getResult().getType(), start + stage, alu, opcode,
                        lhs_kind, lhs_index, lhs_imm, rhs_kind, rhs_index, rhs_imm,
                        cast_target, output_stream);
                };
                issue(0, 0, "cast", "stream_i32", gate_stream, 0, "immediate", 0, 0, "fp32", -1);
                issue(0, 1, "cast", "stream_i32", up_stream, 0, "immediate", 0, 0, "fp32", -1);
                issue(1, 2, "multiply", "alu", 0, 0, "immediate", 0, ffn.getGateScale().convertToFloat(), "fp32", -1);
                issue(1, 3, "multiply", "alu", 1, 0, "immediate", 0, ffn.getUpScale().convertToFloat(), "fp32", -1);
                issue(2, 4, "multiply", "alu", 2, 0, "alu", 3, 0, "fp32", -1);
                issue(2, 5, "negate", "alu", 2, 0, "immediate", 0, 0, "fp32", -1);
                issue(3, 6, "exp", "alu", 5, 0, "immediate", 0, 0, "fp32", -1);
                issue(3, 9, "pass", "alu", 4, 0, "immediate", 0, 0, "fp32", -1);
                issue(4, 7, "add", "alu", 6, 0, "immediate", 0, 1, "fp32", -1);
                issue(4, 10, "pass", "alu", 9, 0, "immediate", 0, 0, "fp32", -1);
                issue(5, 8, "divide", "immediate", 0, 1, "alu", 7, 0, "fp32", -1);
                issue(5, 11, "pass", "alu", 10, 0, "immediate", 0, 0, "fp32", -1);
                issue(6, 12, "multiply", "alu", 11, 0, "alu", 8, 0, "fp32", -1);
                issue(7, 13, "multiply", "alu", 12, 0, "immediate", 0,
                    1.0f / ffn.getHiddenScale().convertToFloat(), "fp32", -1);
                issue(8, 14, "add", "alu", 13, 0, "immediate", 0,
                    static_cast<float>(ffn.getHiddenZeroPoint()), "fp32", -1);
                issue(9, 15, "cast", "alu", 14, 0, "immediate", 0, 0, "i8", ffn.getVxmOutputStream());
                const int64_t hidden_slice = get_slice(hidden_address);
                const int64_t output_latency = *target.transport_latency(
                    target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
                    target::StreamDirection::East, hidden_slice);
                const int64_t output_register = *target.stream_register_id(
                    target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
                    target::StreamDirection::East, hidden_slice);
                auto write = rewriter.create<schedule::MemWriteOp>(ffn.getLoc(), value,
                    start + 9 + output_latency, ffn.getM(), ffn.getVxmOutputStream(), 1,
                    output_register, rewriter.getStringAttr("east"), hidden_address,
                    hidden_placement, ffn.getHiddenPassBytes());
                return ProjectionResult {value, write.getOutput()};
            };

            auto hidden0 = emit_projection(gate0_load, up0_load, 0, 58,
                ffn.getHidden0Address(), ffn.getHidden0Placement());
            auto hidden1 = emit_projection(gate1_load, up1_load, 0, 318,
                ffn.getHidden1Address(), ffn.getHidden1Placement());

            const int64_t down_compute_cycle = 590;
            auto emit_down_read_compute = [&](mlir::Value hidden, mlir::DictionaryAttr address,
                                              mlir::DictionaryAttr placement,
                                              schedule::MxmLoadOp load, int64_t unit,
                                              int64_t activation_stream, int64_t output_stream) {
                const int64_t slice = get_slice(address);
                const int64_t latency = *target.transport_latency(
                    target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
                    target::StreamDirection::East, slice);
                const int64_t reg = *target.stream_register_id(
                    target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
                    target::StreamDirection::East, slice);
                auto read = rewriter.create<schedule::MemReadOp>(ffn.getLoc(), hidden,
                    down_compute_cycle - latency, ffn.getM(), activation_stream, 1, reg,
                    rewriter.getStringAttr("east"), rewriter.getStringAttr("activation"),
                    address, placement, ffn.getHiddenPassBytes());
                return rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(), read.getOutput(),
                    load.getOutput(), projection_type, down_compute_cycle, ffn.getM(),
                    down_compute_cycle + 19, target.mxm_result_window_cycles(ffn.getM()),
                    activation_stream, output_stream, 0, unit, ffn.getM(), 320, 320);
            };
            auto down0 = emit_down_read_compute(hidden0.stored, ffn.getHidden0Address(),
                ffn.getHidden0Placement(), down0_load, 0, 0, ffn.getGateOutputStreamBase());
            auto down1 = emit_down_read_compute(hidden1.stored, ffn.getHidden1Address(),
                ffn.getHidden1Placement(), down1_load, 1, 16, ffn.getUpOutputStreamBase());

            const int64_t add_start = down_compute_cycle + 19
                + target.streams().mem_boundary_register_columns;
            mlir::Value final_value;
            auto add_issue = [&](int64_t stage, int64_t alu, llvm::StringRef opcode,
                                 llvm::StringRef lhs_kind, int64_t lhs_index, float lhs_imm,
                                 llvm::StringRef rhs_kind, int64_t rhs_index, float rhs_imm,
                                 llvm::StringRef cast_target, int64_t output_stream) {
                final_value = emit_vxm(down0.getResult(), down1.getResult(),
                    ffn.getResult().getType(), add_start + stage, alu, opcode,
                    lhs_kind, lhs_index, lhs_imm, rhs_kind, rhs_index, rhs_imm,
                    cast_target, output_stream);
            };
            add_issue(0, 0, "cast", "stream_i32", 32 + ffn.getGateOutputStreamBase(), 0, "immediate", 0, 0, "fp32", -1);
            add_issue(0, 1, "cast", "stream_i32", 32 + ffn.getUpOutputStreamBase(), 0, "immediate", 0, 0, "fp32", -1);
            add_issue(1, 2, "multiply", "alu", 0, 0, "immediate", 0, ffn.getDownLhsScale().convertToFloat(), "fp32", -1);
            add_issue(1, 3, "multiply", "alu", 1, 0, "immediate", 0, ffn.getDownRhsScale().convertToFloat(), "fp32", -1);
            add_issue(2, 4, "add", "alu", 2, 0, "alu", 3, 0, "fp32", -1);
            add_issue(3, 5, "multiply", "alu", 4, 0, "immediate", 0,
                1.0f / ffn.getOutputScale().convertToFloat(), "fp32", -1);
            add_issue(4, 6, "add", "alu", 5, 0, "immediate", 0,
                static_cast<float>(ffn.getOutputZeroPoint()), "fp32", -1);
            add_issue(5, 7, "cast", "alu", 6, 0, "immediate", 0, 0, "i8", ffn.getVxmOutputStream());
            const int64_t result_slice = get_slice(ffn.getResultAddress());
            const int64_t result_latency = *target.transport_latency(
                target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
                target::StreamDirection::East, result_slice);
            const int64_t result_register = *target.stream_register_id(
                target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
                target::StreamDirection::East, result_slice);
            auto final_write = rewriter.create<schedule::MemWriteOp>(ffn.getLoc(), final_value,
                add_start + 5 + result_latency, ffn.getM(), ffn.getVxmOutputStream(), 1,
                result_register, rewriter.getStringAttr("east"), ffn.getResultAddress(),
                ffn.getResultPlacement(), ffn.getResultBytes());
            rewriter.replaceOp(ffn, final_write.getOutput());
            for (stream::RouteOp route : {activation_route, gate_route, up_route,
                     down0_route, down1_route})
                if (route->use_empty()) rewriter.eraseOp(route);
        }

        for (stream::SwigluOp swiglu : swiglus) {
            auto activation_route = swiglu.getActivation().getDefiningOp<stream::RouteOp>();
            auto gate_route = swiglu.getGateWeight().getDefiningOp<stream::RouteOp>();
            auto up_route = swiglu.getUpWeight().getDefiningOp<stream::RouteOp>();
            if (!activation_route || !gate_route || !up_route) {
                swiglu.emitError("requires canonical dual-MXM input routes");
                signalPassFailure(); return;
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
                signalPassFailure(); return;
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
                signalPassFailure();
                return;
            }
            auto result_route = mlir::dyn_cast<stream::RouteOp>(*matmul.getResult().getUsers().begin());
            if (!result_route || result_route.getSource() != "MXM.result"
                || result_route.getDestination() != "MEM"
                || result_route.getSourceUnitId() != matmul.getUnitId()) {
                matmul.emitError("requires a canonical MXM.result to MEM route");
                signalPassFailure();
                return;
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
                signalPassFailure();
                return;
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
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass()
{
    return std::make_unique<LowerStreamToSchedulePass>();
}

} // namespace ftlpu::compiler
