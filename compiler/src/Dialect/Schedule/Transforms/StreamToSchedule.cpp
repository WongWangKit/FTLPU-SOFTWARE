// Keep this translation unit rebuilt with target topology ABI changes.
#include "ftlpu/compiler/Dialect/Schedule/Analysis/resource_scheduler.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/lpu_resource_model.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/schedule_graph.hpp"
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

struct TaskAllocation {
    mlir::DictionaryAttr address;
    mlir::DictionaryAttr placement;
    int64_t bytes;
};

mlir::FailureOr<TaskAllocation> get_task_allocation(
    mlir::ArrayAttr allocations, size_t index)
{
    if (index >= allocations.size()) return mlir::failure();
    const auto dictionary =
        llvm::dyn_cast<mlir::DictionaryAttr>(allocations[index]);
    if (!dictionary) return mlir::failure();
    const auto address =
        dictionary.getAs<mlir::DictionaryAttr>("address");
    const auto placement =
        dictionary.getAs<mlir::DictionaryAttr>("placement");
    const auto bytes = dictionary.getAs<mlir::IntegerAttr>("bytes");
    if (!address || !placement || !bytes) return mlir::failure();
    return TaskAllocation{address, placement, bytes.getInt()};
}

struct PrimitiveFfnSchedulePlan {
    stream::ElementwiseTaskOp add;
    stream::MatmulTaskOp down0;
    stream::MatmulTaskOp down1;
    stream::RouteOp hidden0_route;
    stream::RouteOp hidden1_route;
    stream::ElementwiseTaskOp multiply;
    stream::SwishTaskOp swish;
    stream::MatmulTaskOp gate;
    stream::MatmulTaskOp up;
    stream::RouteOp activation_route;
    stream::RouteOp gate_route;
    stream::RouteOp up_route;
    stream::RouteOp down0_route;
    stream::RouteOp down1_route;
    TaskAllocation hidden0;
    TaskAllocation hidden1;
    TaskAllocation result;
    mlir::FloatAttr gate_scale;
    mlir::FloatAttr up_scale;
    mlir::FloatAttr down_rhs_scale;

    mlir::Value getActivation() { return activation_route.getOutput(); }
    mlir::Value getGateWeight() { return gate_route.getOutput(); }
    mlir::Value getUpWeight() { return up_route.getOutput(); }
    mlir::Value getDownWeight0() { return down0_route.getOutput(); }
    mlir::Value getDownWeight1() { return down1_route.getOutput(); }
    mlir::Value getResult() { return add.getResult(); }
    mlir::Operation* getOperation() { return add.getOperation(); }
    mlir::Location getLoc() { return add.getLoc(); }
    uint64_t getM() { return down0.getM(); }
    uint64_t getK() { return gate.getK(); }
    uint64_t getHidden() { return gate.getN(); }
    uint64_t getN() { return down0.getN(); }
    llvm::APFloat getGateScale() { return gate_scale.getValue(); }
    llvm::APFloat getUpScale() { return up_scale.getValue(); }
    llvm::APFloat getDownRhsScale()
    {
        return down_rhs_scale.getValue();
    }
    mlir::DictionaryAttr getHidden0Address() { return hidden0.address; }
    mlir::DictionaryAttr getHidden0AddressAttr() { return hidden0.address; }
    mlir::DictionaryAttr getHidden0Placement() { return hidden0.placement; }
    mlir::DictionaryAttr getHidden1Address() { return hidden1.address; }
    mlir::DictionaryAttr getHidden1AddressAttr() { return hidden1.address; }
    mlir::DictionaryAttr getResultAddress() { return result.address; }
    mlir::DictionaryAttr getResultAddressAttr() { return result.address; }
    mlir::DictionaryAttr getResultPlacement() { return result.placement; }
};

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

template <typename FfnPlan>
mlir::FailureOr<mlir::Value> lower_w8a16_ffn_schedule(
    mlir::IRRewriter& rewriter, FfnPlan& ffn,
    FfnScheduleStrategy strategy, const target::LPUTargetModel& target)
{
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t stream_encoding_offset =
        target.streams().streams_per_direction;
    const int64_t tile = throughput.mxm_rows;
    const int64_t m = ffn.getM();
    const int64_t k = ffn.getK();
    const int64_t intermediate = ffn.getHidden();
    const int64_t n = ffn.getN();
    const int64_t weight_to_iw = throughput.vxm_weight_to_iw_latency;
    const int64_t gate_acc_latency = throughput.mxm0_accumulator_latency;
    const int64_t up_acc_latency = throughput.mxm1_accumulator_latency;
    const int64_t swish_write_latency = throughput.swiglu_write_latency;
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
    struct CycleWindow {
        int64_t start;
        int64_t end;
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
    std::array<llvm::SmallVector<CycleWindow>, 2> temp_mem_busy_windows;
    // A loaded KxN weight tile stays resident while every M=32 activation tile
    // of the sequence consumes it. Consecutive blocks follow the target's MXM
    // issue interval.
    const int64_t weight_block_interval = m_tile_count * pipelined_block_interval;
    rewriter.setInsertionPoint(ffn.getOperation());
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
                            "stream_i8", stream_encoding_offset + stream, 0.0f,
                            "immediate", 0,
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
                const bool final_k_block = kb + 1 == k / tile;
                const int64_t projection_result_stream_base =
                    strategy == FfnScheduleStrategy::Fused && final_k_block
                    ? 8 + hemisphere * 8
                    : 0;
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
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        auto read = one_slice_read(activation_route.getInput(), activation_route,
                            segment_cycle - activation_latency, activation_slices[byte],
                            activation_base + row_offset, rows, 1, stream_base + byte,
                            "east", "activation", hemi_name(hemisphere));
                        activation_value = read.getOutput();
                    }
                    gate_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        activation_value, ffn.getGateWeight(), projection_type,
                        segment_cycle, rows, segment_cycle + 3, rows + 3,
                        stream_base, projection_result_stream_base,
                        weight_buffer,
                        hemisphere * throughput.mxms_per_hemisphere,
                        rows, tile, tile);
                    up_compute = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(),
                        activation_value, ffn.getUpWeight(), projection_type,
                        segment_cycle, rows, segment_cycle + 3, rows + 3,
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
            ++projection_block;
        }
    }

    const int64_t final_projection_cycle = initial_compute_cycle
        + (projection_block - 1) * weight_block_interval
        + (m_tile_count - 1) * pipelined_block_interval;
    const int64_t accumulator_queue_release = std::max(
        gate_acc_latency + tile + west_latency(gate_acc_slices.front()),
        up_acc_latency + tile + west_latency(up_acc_slices.front()));
    mlir::Value last_hidden;
    int64_t last_swish_cycle = 0;
    const auto emit_swiglu_row = [&](mlir::Value gate_value, mlir::Value up_value,
                                     int64_t cycle, int64_t m_tile, int64_t pair,
                                     int64_t row, int64_t hemisphere) {
        const int64_t nblock = pair * memory.hemispheres + hemisphere;
        const int64_t swish_input_stream =
            strategy == FfnScheduleStrategy::Fused
            ? 8 + hemisphere * 8
            : 0;
        // Keep the FP16 result off projection activation streams E0/E1 and
        // their prefetch alternate E16/E17. Dequant windows are excluded by
        // the fused allocator, leaving the top stream pair available here.
        const int64_t swish_output_stream =
            strategy == FfnScheduleStrategy::Fused
            ? target.streams().streams_per_direction - 2
            : 0;
        last_swish_cycle = std::max(last_swish_cycle, cycle);
        mlir::Value value;
        value = create_vxm(rewriter, ffn.getLoc(), gate_value, up_value,
            ffn.getResult().getType(), cycle, 0, "negate", "stream_f32",
            stream_encoding_offset + swish_input_stream, 0,
            "immediate", 0, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), gate_value, up_value,
            ffn.getResult().getType(), cycle, 1, "multiply", "stream_f32",
            stream_encoding_offset + swish_input_stream, 0, "stream_f32",
            stream_encoding_offset + swish_input_stream
                + throughput.mxm_result_streams,
            0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 1, 2, "exp", "alu", 0, 0,
            "immediate", 0, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 1, 5, "pass", "alu", 1, 0,
            "immediate", 0, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 2, 3, "add", "alu", 2, 0,
            "immediate", 0, 1, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 2, 6, "pass", "alu", 5, 0,
            "immediate", 0, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 3, 4, "divide", "immediate", 0,
            1, "alu", 3, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 3, 7, "pass", "alu", 6, 0,
            "immediate", 0, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        value = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 4, 8, "multiply", "alu", 7, 0,
            "alu", 4, 0, "fp32", -1, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
        auto cast0 = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 5, 9, "cast", "alu", 8, 0,
            "immediate", 0, 0, "fp16", swish_output_stream, 1, 1,
            hemi_name(hemisphere), hemi_name(hemisphere));
        const int64_t peer = 1 - hemisphere;
        auto peer_cast0 = create_vxm(rewriter, ffn.getLoc(), value, up_value,
            ffn.getResult().getType(), cycle + 5, 11, "cast", "alu", 8, 0,
            "immediate", 0, 0, "fp16", swish_output_stream, 1, 1,
            hemi_name(hemisphere), hemi_name(peer));
        for (int64_t destination = 0; destination < memory.hemispheres;
             ++destination) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                auto placement = schedule_placement(rewriter, {hidden_slices[byte]},
                    nblock * m + m_tile * tile + row, 1, 1,
                    hemi_name(destination), "fp16_mxm_activation_planar");
                const bool local = destination == hemisphere;
                mlir::Value output =
                    local ? cast0.getResult() : peer_cast0.getResult();
                auto write = rewriter.create<schedule::MemWriteOp>(ffn.getLoc(),
                    output,
                    cycle + 6 + hidden_slices[byte]
                        / target.streams().mem_slices_per_register_group,
                    1, swish_output_stream + byte, 1, 0,
                    rewriter.getStringAttr("east"), ffn.getHidden0Address(),
                    placement, tile);
                last_hidden = write.getOutput();
            }
        }
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
        llvm::SmallVector<CycleWindow> dequant_windows;
        const int64_t dequant_window_cycles =
            memory.hemispheres * throughput.mxms_per_hemisphere
                * weight_load_cycles
            + 1;
        for (int64_t block = 0; block < projection_block; ++block) {
            const int64_t start =
                initial_compute_cycle + block * weight_block_interval - tile;
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
        schedule::ResourceScheduler ffn_resources;
        schedule::LPUResourceModel resource_model(target);
        llvm::SmallVector<schedule::ResourceWindow, 16> all_vxm_alus;
        for (int64_t alu = 0; alu < 16; ++alu)
            all_vxm_alus.push_back({resource_model.vxm_alu(alu), 0, 1});
        for (CycleWindow dequant : dequant_windows) {
            llvm::SmallVector<schedule::ResourceWindow, 16> windows;
            for (const auto& resource : all_vxm_alus)
                windows.push_back(
                    {resource.resource, 0, dequant.end - dequant.start});
            ffn_resources.reserve_at(dequant.start, windows);
        }
        for (int64_t hemisphere = 0; hemisphere < memory.hemispheres; ++hemisphere) {
            llvm::SmallVector<int64_t, 8> temp_slices;
            temp_slices.append(memory.w8a16_fused_gate_temp_slices.begin(),
                memory.w8a16_fused_gate_temp_slices.end());
            temp_slices.append(memory.w8a16_fused_up_temp_slices.begin(),
                memory.w8a16_fused_up_temp_slices.end());
            for (CycleWindow busy : temp_mem_busy_windows[hemisphere]) {
                llvm::SmallVector<schedule::ResourceWindow, 8> windows;
                for (int64_t slice : temp_slices)
                    windows.push_back({resource_model.mem_slice(hemisphere, slice),
                        0, busy.end - busy.start});
                ffn_resources.reserve_at(busy.start, windows);
            }
        }

        schedule::ScheduleGraph swish_graph;
        llvm::SmallVector<schedule::ScheduleNodeId> swish_nodes;
        for (const CompletedProjectionTile* completed : deferred) {
            llvm::SmallVector<schedule::ResourceWindow, 24> windows;
            for (const auto& resource : all_vxm_alus)
                windows.push_back({resource.resource, 0, tile + 5});
            for (int64_t slice : memory.w8a16_fused_gate_temp_slices)
                windows.push_back({resource_model.mem_slice(
                    completed->hemisphere, slice), 0, tile});
            for (int64_t slice : memory.w8a16_fused_up_temp_slices)
                windows.push_back({resource_model.mem_slice(
                    completed->hemisphere, slice), 0, tile});
            swish_nodes.push_back(swish_graph.add_node("ffn.swiglu",
                completed->deferred_ready_cycle, tile, windows));
        }
        auto swish_schedule = swish_graph.schedule(ffn_resources);
        if (mlir::failed(swish_schedule)) return mlir::failure();

        for (std::size_t index = 0; index < deferred.size(); ++index) {
            const CompletedProjectionTile* completed = deferred[index];
            const int64_t start = (*swish_schedule)[swish_nodes[index]].cycle;
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

    const int64_t slowest_hidden_west_latency = west_latency(hidden_slices.back());
    const int64_t phase_start = last_swish_cycle + 1
        + swish_write_latency + slowest_hidden_west_latency + 1
        + throughput.accumulator_to_vxm_latency;
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
    const int64_t down_reduction_blocks = intermediate / tile;
    const int64_t down_columns_per_hemisphere =
        throughput.mxms_per_hemisphere * tile;
    const int64_t down_columns_per_wave =
        memory.hemispheres * down_columns_per_hemisphere;
    const int64_t down_wave_count =
        (n + down_columns_per_wave - 1) / down_columns_per_wave;
    mlir::Value final_value;
    for (int64_t output_wave = 0; output_wave < down_wave_count; ++output_wave) {
        const int64_t active_hemispheres = std::min<int64_t>(memory.hemispheres,
            (n - output_wave * down_columns_per_wave
                + down_columns_per_hemisphere - 1) / down_columns_per_hemisphere);
        for (int64_t rb = 0; rb < down_reduction_blocks; ++rb) {
            const int64_t weight_compute_cycle = down_compute_cycle;
            const int64_t dequant_start = weight_compute_cycle - tile;
            const int64_t weight_buffer = down_block % 2;
            for (int64_t hemisphere = 0; hemisphere < active_hemispheres; ++hemisphere) {
            for (int64_t local_mxm = 0;
                 local_mxm < throughput.mxms_per_hemisphere; ++local_mxm) {
                const int64_t unit = hemisphere * throughput.mxms_per_hemisphere + local_mxm;
                const int64_t start = dequant_start + unit * weight_load_cycles;
                const int64_t base = down_route.getPlacement().getAs<mlir::IntegerAttr>("base_row").getInt()
                    + (output_wave * (intermediate / tile) + rb)
                        * throughput.mxms_per_hemisphere * weight_load_cycles
                    + local_mxm * weight_load_cycles;
                mlir::Value read_value;
                for (int64_t stream = 0; stream < down_raw.getStreamCount(); ++stream) {
                    read_value = one_slice_read(down_raw.getInput(), down_raw,
                        start - west_latency(weight_slices[stream]), weight_slices[stream],
                        base, weight_load_cycles, 1, stream,
                        "west", "weight_i8", hemi_name(hemisphere)).getOutput();
                }
                mlir::Value value = read_value;
                for (int64_t stream = 0; stream < down_raw.getStreamCount(); ++stream) {
                    const int64_t multiply_alu = stream;
                    const int64_t cast_alu = 8 + stream;
                    value = create_vxm(rewriter, ffn.getLoc(), read_value, read_value,
                        down_route.getInput().getType(), start, multiply_alu, "multiply",
                        "stream_i8", stream_encoding_offset + stream, 0,
                        "immediate", 0,
                        ffn.getDownRhsScale().convertToFloat(), "fp32", -1,
                        weight_load_cycles, 1,
                        hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                    value = create_vxm(rewriter, ffn.getLoc(), value, read_value,
                        down_route.getInput().getType(), start + 1, cast_alu, "cast",
                        "alu", multiply_alu, 0, "immediate", 0, 0, "fp16",
                        local_mxm * throughput.mxm_load_streams_per_cycle + stream * 2,
                        weight_load_cycles, 1,
                        hemi_name(hemisphere), hemi_name(hemisphere)).getResult();
                }
                rewriter.create<schedule::MxmLoadOp>(ffn.getLoc(), value,
                    start + weight_to_iw, weight_load_cycles, 0,
                    throughput.mxm_load_streams_per_cycle,
                    unit, weight_buffer);
            }
            }
            const bool last = rb + 1 == down_reduction_blocks;
            for (int64_t m_tile = 0; m_tile < m_tile_count; ++m_tile) {
                const int64_t compute_cycle = weight_compute_cycle
                    + m_tile * pipelined_block_interval;
                const bool prefetch_next_weight = !last
                    && m_tile_count > 1 && m_tile + 1 == m_tile_count;
                llvm::SmallVector<int64_t> segment_rows;
                llvm::SmallVector<int64_t> segment_streams;
                if (!prefetch_next_weight) {
                    segment_rows = {tile};
                    segment_streams = {0};
                } else {
                    // The next reduction uses the same continuous four-cycle
                    // dequant/IW load as Gate/Up. Route activation around each
                    // target MXM's IW stream window while the weight is
                    // prefetched under this final M tile.
                    segment_rows.push_back(weight_to_iw);
                    segment_streams.push_back(0);
                    for (int64_t unit = 0;
                         unit < active_hemispheres * throughput.mxms_per_hemisphere;
                         ++unit) {
                        segment_rows.push_back(weight_load_cycles);
                        segment_streams.push_back(
                            unit % throughput.mxms_per_hemisphere == 0 ? 16 : 0);
                    }
                    const int64_t routed_rows = weight_to_iw
                        + active_hemispheres * throughput.mxms_per_hemisphere
                            * weight_load_cycles;
                    if (routed_rows < tile) {
                        segment_rows.push_back(tile - routed_rows);
                        segment_streams.push_back(0);
                    }
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
                        segment_cycle + 3, rows + 3, stream_base, 0,
                        weight_buffer, unit_base, rows, tile, tile);
                    down1 = rewriter.create<schedule::MxmComputeOp>(ffn.getLoc(), hidden_value,
                        ffn.getDownWeight1(), projection_type, segment_cycle, rows,
                        segment_cycle + 3, rows + 3, stream_base,
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
                    const int64_t result_stream_base = 24;
                    const int64_t output_cast_alu_base = hemisphere == 0 ? 0 : 8;
                    for (int64_t row = 0; row < tile; ++row) {
                        const int64_t vxm_cycle = compute_cycle + output_to_vxm_latency + row;
                        auto cast0 = create_vxm(rewriter, ffn.getLoc(), acc0.getOutput(), acc1.getOutput(),
                            ffn.getResult().getType(), vxm_cycle, output_cast_alu_base, "pass",
                            "stream_f32", stream_encoding_offset, 0,
                            "immediate", 0, 0, "fp16",
                            result_stream_base, 1, 1,
                            hemi_name(hemisphere), hemi_name(hemisphere));
                        auto cast1 = create_vxm(rewriter, ffn.getLoc(), acc0.getOutput(), acc1.getOutput(),
                            ffn.getResult().getType(), vxm_cycle, output_cast_alu_base + 1,
                            "pass", "stream_f32", 36, 0,
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
            ++down_block;
            if (last) {
                if (output_wave + 1 < down_wave_count)
                    down_compute_cycle += (m_tile_count - 1)
                        * pipelined_block_interval + down_pair_transition_interval;
            } else {
                down_compute_cycle += weight_block_interval
                    + (m_tile_count > 1 ? 0 : tile);
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

    LowerStreamToSchedulePass() = default;
    explicit LowerStreamToSchedulePass(FfnScheduleStrategy strategy)
        : ffn_strategy_(strategy)
    {
    }

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

        mlir::IRRewriter rewriter(&getContext());
        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        llvm::SmallVector<PrimitiveFfnSchedulePlan, 2> primitive_ffns;
        llvm::SmallVector<stream::ElementwiseTaskOp> stream_elements;
        function.walk([&](stream::ElementwiseTaskOp op) {
            if (op.getKind() == "add_quant") stream_elements.push_back(op);
        });
        for (stream::ElementwiseTaskOp add : stream_elements) {
            auto down0 =
                add.getLhs().getDefiningOp<stream::MatmulTaskOp>();
            auto down1 =
                add.getRhs().getDefiningOp<stream::MatmulTaskOp>();
            if (!down0 || !down1 || down0.getLhs().size() != 1
                || down1.getLhs().size() != 1
                || down0.getRhs().size() != 1
                || down1.getRhs().size() != 1)
                continue;
            auto hidden0_route =
                down0.getLhs()[0].getDefiningOp<stream::RouteOp>();
            auto hidden1_route =
                down1.getLhs()[0].getDefiningOp<stream::RouteOp>();
            auto multiply = hidden0_route
                ? hidden0_route.getInput()
                      .getDefiningOp<stream::ElementwiseTaskOp>()
                : stream::ElementwiseTaskOp{};
            if (!hidden0_route || !hidden1_route || !multiply
                || multiply.getKind() != "multiply"
                || hidden1_route.getInput() != multiply.getResult())
                continue;
            auto swish =
                multiply.getLhs().getDefiningOp<stream::SwishTaskOp>();
            auto gate = swish
                ? swish.getInput().getDefiningOp<stream::MatmulTaskOp>()
                : stream::MatmulTaskOp{};
            auto up =
                multiply.getRhs().getDefiningOp<stream::MatmulTaskOp>();
            if (!swish || !gate || !up || gate.getLhs().size() != 1
                || up.getLhs().size() != 1 || gate.getRhs().size() != 1
                || up.getRhs().size() != 1
                || gate.getLhs()[0] != up.getLhs()[0])
                continue;

            auto activation_route =
                gate.getLhs()[0].getDefiningOp<stream::RouteOp>();
            auto gate_route =
                gate.getRhs()[0].getDefiningOp<stream::RouteOp>();
            auto up_route =
                up.getRhs()[0].getDefiningOp<stream::RouteOp>();
            auto down0_route =
                down0.getRhs()[0].getDefiningOp<stream::RouteOp>();
            auto down1_route =
                down1.getRhs()[0].getDefiningOp<stream::RouteOp>();
            const auto hidden0 =
                get_task_allocation(multiply.getResultAllocations(), 0);
            const auto hidden1 =
                get_task_allocation(multiply.getResultAllocations(), 1);
            const auto result =
                get_task_allocation(add.getResultAllocations(), 0);
            const auto gate_scale =
                gate.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
            const auto up_scale =
                up.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
            const auto hidden_scale =
                multiply.getConfig().getAs<mlir::FloatAttr>("output_scale");
            const auto hidden_zero_point =
                multiply.getConfig().getAs<mlir::IntegerAttr>(
                    "output_zero_point");
            const auto down_lhs_scale =
                down0.getConfig().getAs<mlir::FloatAttr>("lhs_scale");
            const auto down_rhs_scale =
                down0.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
            const auto output_scale =
                add.getConfig().getAs<mlir::FloatAttr>("output_scale");
            const auto output_zero_point =
                add.getConfig().getAs<mlir::IntegerAttr>(
                    "output_zero_point");
            if (!activation_route || !gate_route || !up_route
                || !down0_route || !down1_route || mlir::failed(hidden0)
                || mlir::failed(hidden1) || mlir::failed(result)
                || !gate_scale || !up_scale || !hidden_scale
                || !hidden_zero_point || !down_lhs_scale
                || !down_rhs_scale || !output_scale || !output_zero_point) {
                add.emitError("incomplete primitive FFN stream graph");
                signalPassFailure();
                return;
            }

            primitive_ffns.push_back(PrimitiveFfnSchedulePlan{
                add, down0, down1, hidden0_route, hidden1_route, multiply,
                swish, gate, up, activation_route, gate_route, up_route,
                down0_route, down1_route, *hidden0, *hidden1, *result,
                gate_scale, up_scale, down_rhs_scale,
            });
        }
        for (PrimitiveFfnSchedulePlan& ffn : primitive_ffns) {
            rewriter.setInsertionPoint(ffn.add);
            auto result = lower_w8a16_ffn_schedule(
                rewriter, ffn, ffn_strategy_, target);
            if (mlir::failed(result)) {
                ffn.add.emitError(
                    "failed to schedule a primitive W8A16 FFN graph");
                signalPassFailure();
                return;
            }
            rewriter.replaceOp(ffn.add, *result);
            rewriter.eraseOp(ffn.down1);
            rewriter.eraseOp(ffn.down0);
            rewriter.eraseOp(ffn.hidden1_route);
            rewriter.eraseOp(ffn.hidden0_route);
            rewriter.eraseOp(ffn.multiply);
            rewriter.eraseOp(ffn.swish);
            rewriter.eraseOp(ffn.up);
            rewriter.eraseOp(ffn.gate);
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
        schedule::ResourceScheduler scheduler;

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
                auto result =
                    lower_w8a16_ffn_schedule(
                        rewriter, ffn, ffn_strategy_, target);
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

private:
    FfnScheduleStrategy ffn_strategy_ = FfnScheduleStrategy::Tail;
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_stream_to_schedule_pass(
    FfnScheduleStrategy ffn_strategy)
{
    return std::make_unique<LowerStreamToSchedulePass>(ffn_strategy);
}

} // namespace ftlpu::compiler
