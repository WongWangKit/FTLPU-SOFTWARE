#include "FfnStageEmitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_block8_projection_planner.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule::ffn_detail {
namespace {

int64_t functionArgumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

BindingOp createBinding(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::ValueRange source, int64_t index,
    llvm::StringRef access, llvm::StringRef role,
    mlir::RankedTensorType type, mlir::DictionaryAttr placement)
{
    const int64_t bytesPerElement =
        type.getElementType().isInteger(8) ? 1 : 2;
    mlir::OperationState state(location, BindingOp::getOperationName());
    state.addOperands(source);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr(
            "index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr(
            "access", rewriter.getStringAttr(access)),
        rewriter.getNamedAttr(
            "role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("bytes",
            rewriter.getI64IntegerAttr(
                type.getNumElements() * bytesPerElement)),
        rewriter.getNamedAttr("placement", placement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

void createTimeline(mlir::IRRewriter& rewriter,
    mlir::Location location, int64_t start, int64_t end)
{
    mlir::OperationState state(
        location, TimelineOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("ffn.down.block8")),
        rewriter.getNamedAttr(
            "start", rewriter.getI64IntegerAttr(start)),
        rewriter.getNamedAttr(
            "end", rewriter.getI64IntegerAttr(end)),
    });
    rewriter.create(state);
}

} // namespace

mlir::FailureOr<mlir::Value> emitFfnBlock8DownProjection(
    FfnEmissionContext& context, const FfnSwishEmission& swish)
{
    using attention_detail::emitMem;
    using attention_detail::emitMxm;
    using attention_detail::emitMxmDequant;

    auto& rewriter = context.rewriter;
    auto& ffn = context.ffn;
    const auto& target = context.target;
    const auto& memory = target.memory();
    const auto& throughput = target.throughput();
    const int64_t tile = context.tile();
    const int64_t blockRows = throughput.mxm_block_rows;
    const int64_t rowBlocks = context.m() / blockRows;
    const int64_t tokenBlocks = context.m() / tile;
    const int64_t reductionBlocks = context.hidden() / tile;
    const int64_t outputBlocks =
        (context.n() + tile - 1) / tile;
    const int64_t columnsPerWave =
        memory.hemispheres * throughput.mxms_per_hemisphere * tile;
    const int64_t waveCount =
        (context.n() + columnsPerWave - 1) / columnsPerWave;
    const int64_t blockIssues = tile / blockRows;
    if (blockRows != 8 || context.hidden_slices.size() != 16
        || context.result_slices.size() != 16
        || context.down_weight_slices.size()
            != static_cast<std::size_t>(
                throughput.mxms_per_hemisphere
                * memory.w8a16_weight_slice_count)
        || context.m() % tile)
        return mlir::failure();

    rewriter.setInsertionPoint(ffn.getOperation());
    const mlir::Value downWeight = context.down_raw.getInput();
    const int64_t downBinding = functionArgumentIndex(downWeight);
    if (downBinding < 0) {
        ffn.add.emitError(
            "Block8 FFN down weight must be a function argument");
        return mlir::failure();
    }
    createBinding(rewriter, ffn.getLoc(), downWeight, downBinding,
        "input", "weight",
        llvm::cast<mlir::RankedTensorType>(downWeight.getType()),
        context.down_raw.getPlacement());

    const int64_t hiddenBase =
        get_base_row(ffn.getHidden0Placement());
    const int64_t resultBase =
        get_base_row(ffn.getResultPlacement());
    const int64_t weightBase =
        get_base_row(context.down_raw.getPlacement());
    const auto resultType =
        llvm::cast<mlir::RankedTensorType>(
            ffn.getResult().getType());
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(resultType.getElementType());
    const float scale =
        ffn.getDownRhsScale().convertToFloat();
    const int64_t startCycle =
        swish.last_cycle + throughput.swiglu_write_latency + 16;
    int64_t phaseStart = startCycle;
    int64_t lastResultWriteEnd = startCycle;

    for (int64_t wave = 0; wave < waveCount; ++wave) {
        auto reductionSchedule = planFfnBlock8ProjectionSchedule(
            reductionBlocks, tokenBlocks, phaseStart,
            llvm::ArrayRef(context.down_weight_slices).take_front(8),
            llvm::ArrayRef(context.down_weight_slices).drop_front(8),
            context.hidden_slices, target);
        if (mlir::failed(reductionSchedule)) return mlir::failure();

        // All ordinary reduction pairs use the planner's 4-cycle
        // interleave. The final reduction has two BF16 result producers per
        // hemisphere sharing one set of MEM write slices, so schedule those
        // producer beats into the earliest free 4-cycle slots after their
        // penultimate partial dependency.
        const auto& finalSchedule = reductionSchedule->reductions.back();
        const FfnBlock8ReductionSchedule* penultimateSchedule =
            reductionSchedule->reductions.size() > 1
            ? &reductionSchedule->reductions[
                  reductionSchedule->reductions.size() - 2]
            : nullptr;
        llvm::SmallVector<llvm::SmallVector<int64_t>> finalComputeCycles(
            tokenBlocks,
            llvm::SmallVector<int64_t>(
                throughput.mxms_per_hemisphere, -1));
        int64_t finalCandidate =
            finalSchedule.compute_cycles.front();
        const auto overlapsPenultimate = [&](int64_t cycle) {
            if (!penultimateSchedule) return false;
            return llvm::any_of(
                penultimateSchedule->compute_cycles,
                [&](int64_t occupied) {
                    return cycle < occupied + blockIssues
                        && cycle + blockIssues > occupied;
                });
        };
        for (int64_t tokenBlock = 0;
             tokenBlock < tokenBlocks; ++tokenBlock) {
            const int64_t dependencyEnd = penultimateSchedule
                ? penultimateSchedule->compute_cycles[tokenBlock]
                    + blockIssues
                : finalSchedule.compute_cycles[tokenBlock];
            for (int64_t localMxm = 0;
                 localMxm < throughput.mxms_per_hemisphere;
                 ++localMxm) {
                finalCandidate = std::max(
                    finalCandidate, dependencyEnd);
                while (overlapsPenultimate(finalCandidate))
                    finalCandidate += blockIssues;
                finalComputeCycles[tokenBlock][localMxm] =
                    finalCandidate;
                finalCandidate += blockIssues;
            }
        }
        const int64_t finalComputeEnd = finalCandidate;

        for (int64_t reduction = 0;
             reduction < reductionBlocks; ++reduction) {
            const bool finalReduction =
                reduction + 1 == reductionBlocks;
            const auto& scheduled =
                reductionSchedule->reductions[reduction];
            const int64_t loadStart = scheduled.load_cycle;
            const int64_t weightBuffer =
                (wave * reductionBlocks + reduction)
                % throughput.mxm_weight_buffers;
            const int64_t weightAddress = weightBase
                + (wave * reductionBlocks + reduction)
                    * throughput.tile_rows;
            for (int64_t hemisphere = 0;
                 hemisphere < memory.hemispheres; ++hemisphere) {
                for (int64_t localMxm = 0;
                     localMxm < throughput.mxms_per_hemisphere;
                     ++localMxm) {
                    const int64_t outputBlock = wave
                            * memory.hemispheres
                            * throughput.mxms_per_hemisphere
                        + hemisphere
                            * throughput.mxms_per_hemisphere
                        + localMxm;
                    if (outputBlock >= outputBlocks) continue;
                    const int64_t unit = hemisphere
                            * throughput.mxms_per_hemisphere
                        + localMxm;
                    for (int64_t stream = 0; stream < 8;
                         ++stream) {
                        const int64_t slice =
                            context.down_weight_slices[
                                localMxm * 8 + stream];
                        const auto latency =
                            target.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::MxmWeight,
                                target::StreamDirection::East,
                                slice);
                        if (!latency) return mlir::failure();
                        emitMem(rewriter, ffn.getLoc(),
                            loadStart - *latency,
                            hemisphere
                                    * memory.slices_per_hemisphere
                                + slice,
                            "read",
                            weightAddress + throughput.tile_rows - 1,
                            localMxm
                                    * throughput
                                          .mxm_int8_load_streams_per_cycle
                                + stream,
                            throughput.tile_rows, 1, -1,
                            "sram", downBinding);
                    }
                    emitMxmDequant(rewriter, ffn.getLoc(),
                        loadStart, unit, scale,
                        throughput.tile_rows, 1);
                    for (int64_t pulse = 0;
                         pulse < throughput.tile_rows; ++pulse) {
                        emitMxm(rewriter, ffn.getLoc(),
                            loadStart + pulse, unit, "iw", weightBuffer,
                            pulse,
                            0, 0, 1, 1, 0, 1, "sram", true,
                            "supercell", 0, dataFormat,
                            "int8_dequant_bf16");
                    }
                }
            }

            for (int64_t tokenBlock = 0;
                 tokenBlock < tokenBlocks; ++tokenBlock) {
                const int64_t computeCycle =
                    scheduled.compute_cycles[tokenBlock];
                const int64_t activationStreamBase =
                    finalReduction ? 16
                                   : scheduled
                                         .activation_stream_bases[
                                             tokenBlock];
                const int64_t hiddenAddress = hiddenBase
                    + (tokenBlock * reductionBlocks + reduction)
                        * throughput.tile_rows;
                for (int64_t hemisphere = 0;
                     hemisphere < memory.hemispheres; ++hemisphere) {
                    const auto emitActivationAt =
                        [&](int64_t issueCycle) -> mlir::LogicalResult {
                        for (int64_t stream = 0; stream < 16;
                             ++stream) {
                            const int64_t slice =
                                context.hidden_slices[stream];
                            const auto latency =
                                target.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::MxmActivation,
                                    target::StreamDirection::East,
                                    slice);
                            if (!latency) return mlir::failure();
                            emitMem(rewriter, ffn.getLoc(),
                                issueCycle - *latency,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "read", hiddenAddress,
                                activationStreamBase + stream,
                                blockIssues, 1, 1);
                        }
                        return mlir::success();
                    };
                    if (!finalReduction
                        && mlir::failed(
                            emitActivationAt(computeCycle)))
                        return mlir::failure();
                    for (int64_t localMxm = 0;
                         localMxm < throughput.mxms_per_hemisphere;
                         ++localMxm) {
                        const int64_t localComputeCycle = computeCycle
                            + (finalReduction ?
                                finalComputeCycles[tokenBlock][localMxm]
                                    - computeCycle
                                : 0);
                        if (finalReduction
                            && mlir::failed(
                                emitActivationAt(localComputeCycle)))
                            return mlir::failure();
                        const int64_t outputBlock = wave
                                * memory.hemispheres
                                * throughput.mxms_per_hemisphere
                            + hemisphere
                                * throughput.mxms_per_hemisphere
                            + localMxm;
                        if (outputBlock >= outputBlocks) continue;
                        const int64_t unit = hemisphere
                                * throughput.mxms_per_hemisphere
                            + localMxm;
                        const int64_t accumulatorBase =
                            wave * rowBlocks
                            + tokenBlock * blockIssues;
                        const int64_t outputStreamBase =
                            localMxm * blockRows * 2;
                        emitMxm(rewriter, ffn.getLoc(),
                            localComputeCycle, unit, "compute",
                            weightBuffer, 0, activationStreamBase,
                            outputStreamBase,
                            blockIssues, 1,
                            accumulatorBase, 1,
                            finalReduction ? "stream" : "sram", true,
                            "supercell", 0, dataFormat, {},
                            "block8");
                        if (!finalReduction) continue;

                        // The final partial converts the completed FP32
                        // Block8 accumulator to BF16 at the MXM boundary.
                        // Its four result beats line up with a four-cycle MEM
                        // repeat, so no accumulator-read/VXM-cast drain is
                        // needed.
                        const int64_t resultAddress = resultBase
                            + (tokenBlock * outputBlocks + outputBlock)
                                * throughput.tile_rows;
                        for (int64_t stream = 0; stream < 16; ++stream) {
                            const int64_t slice =
                                context.result_slices[stream];
                            const auto latency = target.transport_latency(
                                target::StreamEndpoint::MxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::West,
                                slice);
                            if (!latency) return mlir::failure();
                            const int64_t writeCycle = localComputeCycle
                                + target.mxm_first_result_latency()
                                + *latency;
                            emitMem(rewriter, ffn.getLoc(), writeCycle,
                                hemisphere
                                        * memory.slices_per_hemisphere
                                    + slice,
                                "write", resultAddress,
                                target.streams().streams_per_direction
                                    + outputStreamBase + stream,
                                blockIssues, 1, 1);
                            lastResultWriteEnd = std::max(
                                lastResultWriteEnd,
                                writeCycle + blockIssues);
                        }
                    }
                }
            }
        }
        // Preload the next wave's buffer while the current wave emits its
        // final result beat. It uses E0..15 while final activation uses
        // E16..31 and result traffic travels west.
        phaseStart = finalComputeEnd
            - throughput.mxm_local_load_to_compute_latency;
    }

    phaseStart = std::max(phaseStart, lastResultWriteEnd);

    createTimeline(
        rewriter, ffn.getLoc(), startCycle, phaseStart);
    auto output = createBinding(rewriter, ffn.getLoc(), {},
        0, "output", "result", resultType,
        ffn.getResultPlacement());
    return output.getValue();
}

} // namespace ftlpu::compiler::schedule::ffn_detail
