#include "FfnStageEmitter.hpp"

#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule::ffn_detail {

mlir::FailureOr<FfnSwishEmission> emitFfnSwish(
    FfnEmissionContext& context, FfnProjectionEmission emission)
{
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t mTileCount =
        context.projection_timeline.m_tile_count;
    const int64_t pairCount =
        context.projection_timeline.pair_count;
    const int64_t weightLoadCycles =
        context.projection_timeline.weight_load_cycles;

    FfnSwishEmission result;
    result.last_cycle = 0;
    // Queue 7 is the tail of the physical 8-stage VXM chain. Its output
    // register is permanently bound to stream group 3 (byte streams 6/7).
    constexpr int64_t outputStream = 6;
    if (context.strategy == FfnScheduleStrategy::Tail) {
        int64_t inputCycle =
            context.projection_timeline.final_projection_cycle
            + context.projection_timeline.accumulator_queue_release;
        for (const CompletedProjectionTile& completed :
            emission.completed_tiles)
            inputCycle = std::max(
                inputCycle, completed.deferred_ready_cycle);

        struct PendingOutput {
            int64_t input_cycle;
            int64_t m_tile;
            int64_t pair;
            int64_t row;
            int64_t source_hemisphere;
        };
        llvm::SmallVector<PendingOutput> pendingOutputs;
        mlir::Value gateValue;
        mlir::Value upValue;
        const int64_t firstInputCycle = inputCycle;
        for (int64_t mTile = 0; mTile < mTileCount; ++mTile) {
            for (int64_t pair = 0; pair < pairCount; ++pair) {
                for (int64_t row = 0; row < tile; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < memory.hemispheres;
                         ++hemisphere) {
                        const auto completed = llvm::find_if(
                            emission.completed_tiles,
                            [&](const CompletedProjectionTile& tile) {
                                return tile.m_tile == mTile
                                    && tile.pair == pair
                                    && tile.hemisphere == hemisphere;
                            });
                        if (completed
                            == emission.completed_tiles.end()) {
                            return mlir::failure();
                        }
                        const int64_t tempBase =
                            (pair * mTileCount + mTile) * tile;
                        const int64_t streamBase = hemisphere * 16;
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t gateSlice =
                                memory.w8a16_fused_gate_temp_slices[byte];
                            const int64_t upSlice =
                                memory.w8a16_fused_up_temp_slices[byte];
                            gateValue =
                                context
                                    .emitSliceRead(completed->gate_temp,
                                        context.activation_route,
                                        inputCycle
                                            - context.westLatency(
                                                gateSlice),
                                        gateSlice, tempBase + row, 1, 1,
                                        streamBase + byte, "west",
                                        "vxm_bf16",
                                        context.hemisphereName(
                                            hemisphere))
                                    .getOutput();
                            upValue =
                                context
                                    .emitSliceRead(completed->up_temp,
                                        context.activation_route,
                                        inputCycle
                                            - context.westLatency(
                                                upSlice),
                                        upSlice, tempBase + row, 1, 1,
                                        streamBase + 2 + byte,
                                        "west", "vxm_bf16",
                                        context.hemisphereName(
                                            hemisphere))
                                    .getOutput();
                        }
                        pendingOutputs.push_back({inputCycle, mTile,
                            pair, row, hemisphere});
                    }
                    ++inputCycle;
                }
            }
        }
        const int64_t repeatCount = inputCycle - firstInputCycle;
        if (!gateValue || !upValue || repeatCount <= 0)
            return mlir::failure();
        auto [output, mirroredOutput] = emitFfnSwishAlu(
            context.rewriter, ffn.getLoc(), ffn.getResult().getType(),
            gateValue, upValue, target, context.strategy,
            firstInputCycle - 1, 0, outputStream, repeatCount, 1);
        (void)mirroredOutput;
        for (const PendingOutput& pending : pendingOutputs) {
            result.hidden = emitFfnSwishResultRow(context.rewriter, ffn,
                target, context.hidden_slices, output.getResult(),
                pending.input_cycle, pending.m_tile, pending.pair,
                pending.row, pending.source_hemisphere);
        }
        result.last_cycle = inputCycle - 1;
        return result;
    }

    FfnSwishScheduleRequest request;
    request.tile_rows = tile;
    const int64_t dequantWindowCycles =
        memory.hemispheres * throughput.mxms_per_hemisphere
            * weightLoadCycles
        + 1;
    for (const FfnProjectionBlockSchedule& block :
        context.projection_timeline.blocks) {
        request.dequant_windows.push_back(
            {block.dequant_start,
                block.dequant_start + dequantWindowCycles});
    }
    for (int64_t hemisphere = 0;
         hemisphere < memory.hemispheres; ++hemisphere) {
        request.temp_mem_windows[hemisphere] =
            std::move(emission.temp_mem_busy_windows[hemisphere]);
    }

    llvm::SmallVector<const CompletedProjectionTile*> deferred;
    for (const CompletedProjectionTile& completed :
        emission.completed_tiles)
        deferred.push_back(&completed);
    llvm::sort(deferred,
        [](const CompletedProjectionTile* lhs,
            const CompletedProjectionTile* rhs) {
            return lhs->compute_cycle < rhs->compute_cycle;
        });
    for (const CompletedProjectionTile* completed : deferred)
        request.tasks.push_back(
            {completed->deferred_ready_cycle, completed->hemisphere});

    auto cycles = planFfnSwishCycles(request, target);
    if (mlir::failed(cycles)) return mlir::failure();
    for (std::size_t index = 0; index < deferred.size(); ++index) {
        const CompletedProjectionTile& completed = *deferred[index];
        const int64_t start = (*cycles)[index];
        for (int64_t row = 0; row < tile; ++row) {
            const int64_t cycle = start + row;
            const int64_t tempBase =
                (completed.pair * mTileCount + completed.m_tile)
                * tile;
            const int64_t tempStreamBase =
                8 + completed.hemisphere * 8;
            mlir::Value gateValue;
            mlir::Value upValue;
            for (int64_t byte = 0;
                 byte < throughput.mxm_result_streams; ++byte) {
                const int64_t gateSlice =
                    memory.w8a16_fused_gate_temp_slices[byte];
                const int64_t upSlice =
                    memory.w8a16_fused_up_temp_slices[byte];
                gateValue =
                    context
                        .emitSliceRead(completed.gate_temp,
                            context.activation_route,
                            cycle - context.westLatency(gateSlice),
                            gateSlice, tempBase + row, 1, 1,
                            tempStreamBase + byte, "west",
                            "vxm_fp32",
                            context.hemisphereName(
                                completed.hemisphere))
                        .getOutput();
                upValue =
                    context
                        .emitSliceRead(completed.up_temp,
                            context.activation_route,
                            cycle - context.westLatency(upSlice),
                            upSlice, tempBase + row, 1, 1,
                            tempStreamBase
                                + throughput.mxm_result_streams + byte,
                            "west", "vxm_fp32",
                            context.hemisphereName(
                                completed.hemisphere))
                        .getOutput();
            }
            result.last_cycle = std::max(result.last_cycle, cycle);
            result.hidden = emitFfnSwishRow(context.rewriter, ffn,
                target, context.strategy, context.hidden_slices,
                gateValue, upValue, cycle, completed.m_tile,
                completed.pair, row, completed.hemisphere,
                outputStream);
        }
    }
    return result;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
