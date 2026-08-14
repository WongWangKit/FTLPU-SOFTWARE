#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_memory_layout.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include <algorithm>

namespace ftlpu::compiler::schedule {
using namespace attention_detail;
namespace {

int64_t functionArgumentIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    return -1;
}

} // namespace

int64_t AttentionScheduleEmitter::emitOutputProjection(int64_t pvEnd)
{
    const auto resultPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("result");
    const auto resultKind = resultPlacement
        ? resultPlacement.getAs<mlir::StringAttr>("kind")
        : mlir::StringAttr {};
    if (resultKind
        && resultKind.getValue()
            == "fp16_mxm_block8_distributed_16")
        return emitBlock8OutputProjection(pvEnd);
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t reductionBlocks = op_.getQueryHeads() * op_.getHeadDim() / tile;
    const int64_t outputGroups = op_.getHidden()
        / (tile * target_.memory().hemispheres);
    const int64_t localMxm = 0;
    const int64_t accumulatorSlice = target_.memory().accumulator_slice_base
        + localMxm * target_.memory().accumulator_slices_per_mxm;
    const int64_t accumulatorLatency =
        target_.throughput().mxm0_accumulator_latency;
    const int64_t weightToIw = target_.throughput().vxm_weight_to_iw_latency;
    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    const int64_t weightLoadLead =
        (target_.memory().hemispheres - 1) * 8 + 3 + weightToIw + 1;
    // MXM accumulator SRAM is a separate fixed-depth resource from MEM.
    const int64_t accumulatorCapacity =
        32 * target_.throughput().mxm_rows;
    const int64_t outputAccumulatorBase =
        accumulatorCapacity - op_.getSeqLen();
    const auto outputWeightScaleAttr = op_.output.getConfig()
        .getAs<mlir::FloatAttr>("output_weight_scale");
    const float outputWeightScale = outputWeightScaleAttr
        ? static_cast<float>(outputWeightScaleAttr.getValueAsDouble())
        : 1.0f;
    const auto accumulatorAddress = [&](int64_t token) {
        return outputAccumulatorBase + token;
    };

    int64_t phaseStart = pvEnd;
    for (int64_t outputGroup = 0; outputGroup < outputGroups; ++outputGroup) {
        for (int64_t reductionBlock = 0; reductionBlock < reductionBlocks;
             ++reductionBlock) {
            const int64_t firstCompute =
                phaseStart + readLatency(layout.outputWeightSlices().back())
                + weightLoadLead;
            const int64_t dequantStart = firstCompute - weightLoadLead;
            const int64_t weightBuffer =
                (outputGroup * reductionBlocks + reductionBlock)
                % target_.throughput().mxm_weight_buffers;
            for (int64_t hemisphere = 0; hemisphere < target_.memory().hemispheres;
                 ++hemisphere) {
                const char* hemi = hemisphere == 0 ? "east" : "west";
                for (int64_t pulse = 0; pulse < 4; ++pulse) {
                    const int64_t cycle = dequantStart + hemisphere * 8 + pulse;
                    for (int64_t stream = 0; stream < 8; ++stream) {
                        const int64_t slice = layout.outputWeightSlices()[stream];
                        emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", layout.outputWeightAddress(
                                outputGroup, reductionBlock, pulse),
                            32 + stream, 1, 1, 0, "sram",
                            functionArgumentIndex(op_.getOutputWeight()));
                    }
                    for (int64_t lane = 0;
                         lane < target_.throughput().lanes_per_tile; ++lane) {
                        emitVxm(rewriter_, op_.getLoc(), op_.getOutputWeight(), cycle, lane,
                            "multiply", "stream_i8", 32 + lane, 0.0f,
                            "immediate", 0, outputWeightScale, "fp32", -1, hemi, hemi,
                            functionArgumentIndex(op_.getOutputWeight()));
                        emitVxm(rewriter_, op_.getLoc(), op_.getOutputWeight(), cycle + 1,
                            8 + lane, "cast", "alu", lane, 0.0f,
                            "immediate", 0, 0.0f, dataFormat,
                            localMxm * target_.throughput().mxm_load_streams_per_cycle
                                + lane * 2,
                            hemi, hemi);
                    }
                    emitMxm(rewriter_, op_.getLoc(), cycle + weightToIw,
                        hemisphere * target_.throughput().mxms_per_hemisphere + localMxm,
                        "iw", weightBuffer, 3 - pulse, 0, 0, 1, 1,
                        0, 1, "stream", true, "supercell", 0,
                        dataFormat);
                }
            }

            const bool finalReduction = reductionBlock + 1 == reductionBlocks;
            const int64_t computeInterval = tile;
            const int64_t queryHead = reductionBlock / (op_.getHeadDim() / tile);
            const int64_t headBlock = reductionBlock % (op_.getHeadDim() / tile);
            for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
                const int64_t computeBase = firstCompute + tokenBlock * computeInterval;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres; ++hemisphere) {
                    const int64_t computeCycle = computeBase;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice =
                            layout.contextSlices()[headBlock * 2 + byte];
                        const int64_t latency = *target_.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::MxmActivation,
                            target::StreamDirection::East, slice);
                        emitMem(rewriter_, op_.getLoc(), computeCycle - latency,
                            hemisphere * target_.memory().slices_per_hemisphere
                                + slice,
                            "read",
                            layout.contextAddress(
                                queryHead, tokenBlock * tile),
                            hemisphere * 2 + byte, tile, 1, 1);
                    }
                    emitMxm(rewriter_, op_.getLoc(), computeCycle,
                        hemisphere
                            * target_.throughput().mxms_per_hemisphere
                            + localMxm,
                        "compute", weightBuffer, 0, hemisphere * 2,
                        0, tile, 1,
                        accumulatorAddress(tokenBlock * tile),
                        1, "sram", true, "supercell", 0, dataFormat);
                }
            }
            phaseStart = firstCompute + tokenBlocks * computeInterval
                + (finalReduction
                        ? accumulatorLatency + readLatency(accumulatorSlice)
                        : 0);
            if (finalReduction) {
                const int64_t castStart = phaseStart;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres; ++hemisphere) {
                    const char* inputHemisphere =
                        hemisphere == 0 ? "east" : "west";
                    const int64_t mxmOutputStream = 0;
                    const int64_t inputStream =
                        target_.streams().streams_per_direction
                        + mxmOutputStream;
                    const int64_t outputStream = hemisphere * 2;
                    for (int64_t token = 0; token < op_.getSeqLen(); ++token) {
                        const int64_t vxmCycle = castStart + token;
                        emitMxm(rewriter_, op_.getLoc(),
                            vxmCycle
                                - target_.throughput()
                                      .accumulator_read_to_vxm_latency,
                            hemisphere
                                    * target_.throughput()
                                          .mxms_per_hemisphere
                                + localMxm,
                            "accumulator_read", 0, 0, 0,
                            mxmOutputStream, 1, 1,
                            accumulatorAddress(token),
                            1, "sram", true, "supercell", 0,
                            dataFormat);
                        emitVxm(rewriter_, op_.getLoc(), op_.getOutputWeight(), vxmCycle,
                            hemisphere, "pass", "stream_f32", inputStream, 0.0f,
                            "immediate", 0, 0.0f, dataFormat, outputStream,
                            inputHemisphere, "east");
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = 28 + hemisphere * 2 + byte;
                            const int64_t latency = *target_.transport_latency(
                                target::StreamEndpoint::VxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::East, slice);
                            emitMem(rewriter_, op_.getLoc(), vxmCycle + latency,
                                slice, "write",
                                layout.resultAddress(outputGroup, token),
                                outputStream + byte, 1, 1, 0);
                        }
                    }
                }
                const int64_t finalWriteLatency = *target_.transport_latency(
                    target::StreamEndpoint::VxmResult,
                    target::StreamEndpoint::Mem,
                    target::StreamDirection::East, 31);
                phaseStart = castStart
                    + op_.getSeqLen() + finalWriteLatency;
            }
        }
    }
    return phaseStart;
}

} // namespace ftlpu::compiler::schedule
