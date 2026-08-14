#include "ftlpu/compiler/Dialect/Schedule/Transforms/attention_schedule_emitter.hpp"

#include "AttentionEmitterUtils.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_projection_planner.hpp"
#include "ftlpu/compiler/Dialect/Schedule/Analysis/attention_work_planner.hpp"
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

int64_t AttentionScheduleEmitter::emitProjections()
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto resultPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("result");
    const auto resultKind = resultPlacement
        ? resultPlacement.getAs<mlir::StringAttr>("kind")
        : mlir::StringAttr {};
    if (resultKind
        && resultKind.getValue()
            == "fp16_mxm_block8_distributed_16")
        return emitBlock8Projections();
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(elementType);
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t hiddenBlocks = op_.getHidden() / tile;
    const int64_t projectionHeadBlocks = op_.getHeadDim() / tile;
    const int64_t weightToIw = target_.throughput().vxm_weight_to_iw_latency;
    const auto inputPlacement =
        op_.getMemoryPlan().getAs<mlir::DictionaryAttr>("input");
    const auto inputKind =
        inputPlacement.getAs<mlir::StringAttr>("kind");
    const bool inputDistributed16 = inputKind
        && inputKind.getValue() == "fp16_mxm_distributed_16";
    const int64_t inputBase =
        inputPlacement.getAs<mlir::IntegerAttr>("base_row").getInt();
    const auto inputStagingPlacement = inputDistributed16
        ? op_.getMemoryPlan().getAs<mlir::DictionaryAttr>(
              "input_staging")
        : mlir::DictionaryAttr {};
    llvm::SmallVector<int64_t, 16> inputSlices;
    for (mlir::Attribute value :
        inputPlacement.getAs<mlir::ArrayAttr>("slices"))
        inputSlices.push_back(
            llvm::cast<mlir::IntegerAttr>(value).getInt());
    llvm::SmallVector<int64_t, 2> inputStagingSlices;
    if (inputStagingPlacement)
        for (mlir::Attribute value :
             inputStagingPlacement.getAs<mlir::ArrayAttr>("slices"))
            inputStagingSlices.push_back(
                llvm::cast<mlir::IntegerAttr>(value).getInt());
    const auto projectionActivationSlices = inputDistributed16
        ? inputStagingSlices
        : llvm::SmallVector<int64_t>(
              layout.activationSlices().begin(),
              layout.activationSlices().end());
    const int64_t activationLatency = *target_.transport_latency(
        target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
        target::StreamDirection::East,
        projectionActivationSlices.front());
    const int64_t projectionHeads[] = {
        op_.getQueryHeads(), op_.getKvHeads(), op_.getKvHeads()};
    const mlir::Value projectionValues[] = {
        op_.getQueryWeight(), op_.getKeyWeight(), op_.getValueWeight()};
    const auto weightScale = [&](llvm::StringRef name) {
        const auto value = op_.query.getConfig().getAs<mlir::FloatAttr>(name);
        return value ? static_cast<float>(value.getValueAsDouble()) : 1.0f;
    };
    const float projectionScales[] = {
        weightScale("query_weight_scale"),
        weightScale("key_weight_scale"),
        weightScale("value_weight_scale"),
    };
    int64_t phaseStart = 0;

    const auto readLatency = [&](int64_t slice) {
        return slice / target_.streams().mem_slices_per_register_group + 2;
    };
    const auto emitDequant = [&](int64_t cycle, int64_t hemisphere,
                                 int64_t localMxm, mlir::Value weight,
                                 float scale) {
        const char* hemi = hemisphere == 0 ? "east" : "west";
        for (int64_t lane = 0; lane < target_.throughput().lanes_per_tile; ++lane) {
            emitVxm(rewriter_, op_.getLoc(), weight, cycle, lane, "multiply",
                "stream_i8", 32 + lane, 0.0f, "immediate", 0, scale,
                "fp32", -1, hemi, hemi,
                functionArgumentIndex(weight));
            emitVxm(rewriter_, op_.getLoc(), weight, cycle + 1, 8 + lane, "cast",
                "alu", lane, 0.0f, "immediate", 0, 0.0f,
                dataFormat, localMxm * 16 + lane * 2, hemi, hemi);
        }
    };
    const auto emitRopeOrCast = [&](int64_t cycle, int64_t hemisphere,
                                    bool rope, mlir::Value value) {
        attention_detail::emitRopeOrCast(
            rewriter_, op_.getLoc(), target_, cycle, hemisphere, rope, value,
            elementType);
    };

    if (inputDistributed16) {
        const auto& stagingSlices = inputStagingSlices;
        if (inputSlices.size() != 16 || stagingSlices.size() < 2) {
            op_.emitError(
                "distributed attention input requires 16 source slices "
                "and one FP16 staging pair");
            return -1;
        }
        int64_t stagingCycle = 16;
        int64_t lastWriteCycle = stagingCycle;
        const int64_t stagingBase =
            inputStagingPlacement
                .getAs<mlir::IntegerAttr>("base_row")
                .getInt();
        for (int64_t reductionBlock = 0;
             reductionBlock < hiddenBlocks; ++reductionBlock) {
            for (int64_t token = 0; token < op_.getSeqLen();
                 ++token, ++stagingCycle) {
                const int64_t tokenBlock = token / tile;
                const int64_t tokenWithinBlock = token % tile;
                const int64_t tokenWave = tokenWithinBlock / 8;
                const int64_t tokenLane = tokenWithinBlock % 8;
                const int64_t sourceAddress = inputBase
                    + (tokenBlock * hiddenBlocks + reductionBlock) * 4
                    + tokenWave;
                const int64_t stagingAddress = stagingBase
                    + reductionBlock * op_.getSeqLen() + token;
                for (int64_t hemisphere = 0;
                     hemisphere < target_.memory().hemispheres;
                     ++hemisphere) {
                    const char* hemi =
                        hemisphere == 0 ? "east" : "west";
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t sourceSlice =
                            inputSlices[2 * tokenLane + byte];
                        const int64_t sourceLatency =
                            *target_.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::VxmInput,
                                target::StreamDirection::West,
                                sourceSlice);
                        emitMem(rewriter_, op_.getLoc(),
                            stagingCycle - sourceLatency,
                            hemisphere
                                    * target_.memory()
                                          .slices_per_hemisphere
                                + sourceSlice,
                            "read", sourceAddress, 32 + byte,
                            1, 1, 0);
                    }
                    emitVxm(rewriter_, op_.getLoc(), op_.getInput(),
                        stagingCycle, hemisphere, "pass",
                        streamKind, 32, 0.0f,
                        "immediate", 0, 0.0f, dataFormat, 0,
                        hemi, hemi);
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t destinationSlice =
                            stagingSlices[byte];
                        const int64_t destinationLatency =
                            *target_.transport_latency(
                                target::StreamEndpoint::VxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::East,
                                destinationSlice);
                        const int64_t writeCycle =
                            stagingCycle + destinationLatency;
                        emitMem(rewriter_, op_.getLoc(), writeCycle,
                            hemisphere
                                    * target_.memory()
                                          .slices_per_hemisphere
                                + destinationSlice,
                            "write", stagingAddress, byte,
                            1, 1, 0);
                        lastWriteCycle =
                            std::max(lastWriteCycle, writeCycle);
                    }
                }
            }
        }
        phaseStart = lastWriteCycle + 2;
    }

    if (target_.throughput().mxms_per_hemisphere == 1) {
        const int64_t accumulatorHalfStride = 4096;
        const int64_t conservativeComputeSpacing = tile;
        int64_t projectionBlock = 0;

        const auto emitHalfCast = [&](int64_t cycle, int64_t hemisphere,
                                      int64_t half, mlir::Value value) {
            const int64_t alu = hemisphere * 8 + half;
            const char* hemi = hemisphere == 0 ? "east" : "west";
            emitVxm(rewriter_, op_.getLoc(), value, cycle, alu, "pass",
                "stream_f32", 32, 0.0f, "immediate", 0, 0.0f,
                dataFormat, half * 2, hemi, hemi);
        };
        const auto emitStagedRope = [&](int64_t cycle, int64_t hemisphere,
                                        mlir::Value value) {
            const int64_t alu = hemisphere * 8;
            const char* hemi = hemisphere == 0 ? "east" : "west";
            emitVxm(rewriter_, op_.getLoc(), value, cycle, alu, "multiply",
                "stream_f32", 32, 0.0f, streamKind, 40, 0.0f,
                "fp32", -1, hemi, hemi);
            emitVxm(rewriter_, op_.getLoc(), value, cycle, alu + 1,
                "multiply", "stream_f32", 36, 0.0f, streamKind, 42, 0.0f,
                "fp32", -1, hemi, hemi);
            emitVxm(rewriter_, op_.getLoc(), value, cycle, alu + 3,
                "multiply", "stream_f32", 36, 0.0f, streamKind, 40, 0.0f,
                "fp32", -1, hemi, hemi);
            emitVxm(rewriter_, op_.getLoc(), value, cycle, alu + 4,
                "multiply", "stream_f32", 32, 0.0f, streamKind, 42, 0.0f,
                "fp32", -1, hemi, hemi);
            emitVxm(rewriter_, op_.getLoc(), value, cycle + 1, alu + 2,
                "subtract", "alu", alu, 0.0f, "alu", alu + 1, 0.0f,
                dataFormat, 0, hemi, hemi);
            emitVxm(rewriter_, op_.getLoc(), value, cycle + 1, alu + 5,
                "add", "alu", alu + 3, 0.0f, "alu", alu + 4, 0.0f,
                dataFormat, 2, hemi, hemi);
        };

        for (int64_t projection = 0; projection < 3; ++projection) {
            const auto kind = projectionKind(projection);
            for (int64_t headBase = 0;
                 headBase < projectionHeads[projection]; headBase += 2) {
                int64_t rawWriteEnd = phaseStart;
                for (int64_t half = 0; half < 2; ++half) {
                    for (int64_t reductionBlock = 0;
                         reductionBlock < hiddenBlocks; ++reductionBlock) {
                        const int64_t weightBuffer =
                            projectionBlock
                            % target_.throughput().mxm_weight_buffers;
                        const int64_t dequantStart = phaseStart;
                        for (int64_t hemisphere = 0;
                             hemisphere < target_.memory().hemispheres;
                             ++hemisphere) {
                            const int64_t head = headBase + hemisphere;
                            if (head >= projectionHeads[projection]) continue;
                            for (int64_t pulse = 0; pulse < 4; ++pulse) {
                                const int64_t cycle = dequantStart
                                    + hemisphere * 8 + pulse;
                                const int64_t address = layout.weightAddress(
                                    kind, head * projectionHeadBlocks + half,
                                    reductionBlock, half,
                                    pulse);
                                for (int64_t stream = 0; stream < 8;
                                     ++stream) {
                                    const int64_t slice =
                                        layout.weightSlices()[stream];
                                    emitMem(rewriter_, op_.getLoc(),
                                        cycle - readLatency(slice),
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "read", address, 32 + stream,
                                        1, 1, 0, "sram",
                                        functionArgumentIndex(
                                            projectionValues[projection]));
                                }
                                emitDequant(cycle, hemisphere, 0,
                                    projectionValues[projection],
                                    projectionScales[projection]);
                                emitMxm(rewriter_, op_.getLoc(),
                                    cycle + weightToIw, hemisphere, "iw",
                                    weightBuffer, 3 - pulse, 0, 0, 1, 1,
                                    0, 1, "stream", true, "supercell", 0,
                                    dataFormat);
                            }
                        }

                        const int64_t firstCompute = dequantStart
                            + (target_.memory().hemispheres - 1) * 8
                            + 4 + weightToIw;
                        const bool finalReduction =
                            reductionBlock + 1 == hiddenBlocks;
                        const int64_t computeSpacing = finalReduction
                            ? tile
                                + target_.throughput()
                                      .accumulator_to_vxm_latency
                                + activationLatency + 16
                            : conservativeComputeSpacing;
                        for (int64_t tokenBlock = 0;
                             tokenBlock < tokenBlocks; ++tokenBlock) {
                            const int64_t computeCycle = firstCompute
                                + tokenBlock * computeSpacing;
                            for (int64_t hemisphere = 0;
                                 hemisphere < target_.memory().hemispheres;
                                 ++hemisphere) {
                                const int64_t head = headBase + hemisphere;
                                if (head >= projectionHeads[projection])
                                    continue;
                                const int64_t inputAddress =
                                    layout.activationAddress(
                                        reductionBlock, tokenBlock)
                                    + (inputDistributed16
                                            ? inputStagingPlacement
                                                  .getAs<mlir::IntegerAttr>(
                                                      "base_row")
                                                  .getInt()
                                            : 0);
                                for (int64_t byte = 0;
                                     byte < (inputDistributed16 ? 2 : 4);
                                     ++byte) {
                                    const int64_t slice =
                                        projectionActivationSlices[byte];
                                    emitMem(rewriter_, op_.getLoc(),
                                        computeCycle - activationLatency,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "read", inputAddress, byte,
                                        tile, 1, 1);
                                }
                                const int64_t outputAddress =
                                    layout.projectionAddress(
                                        kind, head, tokenBlock);
                                emitMxm(rewriter_, op_.getLoc(),
                                    computeCycle, hemisphere, "compute",
                                    weightBuffer, 0, 0, 0, tile, 1,
                                    outputAddress
                                        + half * accumulatorHalfStride,
                                    1,
                                    finalReduction ? "stream" : "sram",
                                    finalReduction, "supercell", 0,
                                    dataFormat);
                                if (!finalReduction) continue;

                                for (int64_t offset = 0; offset < tile;
                                     ++offset) {
                                    const int64_t token =
                                        tokenBlock * tile + offset;
                                    const int64_t vxmCycle = computeCycle
                                        + target_.throughput()
                                              .accumulator_to_vxm_latency
                                        + offset;
                                    if (kind
                                        != AttentionProjectionKind::Value) {
                                        for (int64_t byte = 0; byte < 4;
                                             ++byte) {
                                            const int64_t slice =
                                                44 + half * 4 + byte;
                                            const int64_t latency =
                                                *target_.transport_latency(
                                                    target::StreamEndpoint::MxmResult,
                                                    target::StreamEndpoint::Mem,
                                                    target::StreamDirection::West,
                                                    slice);
                                            const int64_t cycle =
                                                computeCycle
                                                + target_
                                                      .mxm_first_result_latency()
                                                + offset + latency;
                                            emitMem(rewriter_, op_.getLoc(),
                                                cycle,
                                                hemisphere
                                                        * target_.memory()
                                                              .slices_per_hemisphere
                                                    + slice,
                                                "write",
                                                head * op_.getSeqLen()
                                                    + token,
                                                32 + byte, 1, 1, 0);
                                            rawWriteEnd = std::max(
                                                rawWriteEnd, cycle + 1);
                                        }
                                    } else {
                                        emitHalfCast(vxmCycle, hemisphere,
                                            half,
                                            projectionValues[projection]);
                                        const int64_t writeCycle =
                                            vxmCycle + 1;
                                        const int64_t packedStream =
                                            (token % 8) * 2;
                                        const int64_t row =
                                            (token % tile) / 8;
                                        const auto slices =
                                            layout.valuePackSlices(half);
                                        for (int64_t byte = 0; byte < 2;
                                             ++byte) {
                                            const int64_t slice =
                                                slices[packedStream + byte];
                                            const int64_t cycle = writeCycle
                                                + slice
                                                    / target_.streams()
                                                          .mem_slices_per_register_group;
                                            emitMem(rewriter_, op_.getLoc(),
                                                cycle,
                                                hemisphere
                                                        * target_.memory()
                                                              .slices_per_hemisphere
                                                    + slice,
                                                "write",
                                                layout.valuePackAddress(head,
                                                    half, tokenBlock, row),
                                                half * 2 + byte, 1, 1, 0);
                                            rawWriteEnd = std::max(
                                                rawWriteEnd, cycle + 1);
                                        }
                                    }
                                }
                            }
                        }
                        phaseStart = firstCompute
                            + tokenBlocks * computeSpacing
                            + tile;
                        if (finalReduction)
                            phaseStart = std::max(phaseStart, rawWriteEnd + 1);
                        ++projectionBlock;
                    }
                }

                if (kind == AttentionProjectionKind::Value) continue;
                int64_t ropeCycle = std::max(phaseStart, rawWriteEnd + 1);
                for (int64_t token = 0; token < op_.getSeqLen();
                     ++token, ++ropeCycle) {
                    const int64_t tokenBlock = token / tile;
                    for (int64_t hemisphere = 0;
                         hemisphere < target_.memory().hemispheres;
                         ++hemisphere) {
                        const int64_t head = headBase + hemisphere;
                        if (head >= projectionHeads[projection]) continue;
                        for (int64_t half = 0; half < 2; ++half) {
                            for (int64_t byte = 0; byte < 4; ++byte) {
                                const int64_t slice =
                                    44 + half * 4 + byte;
                                emitMem(rewriter_, op_.getLoc(),
                                    ropeCycle - readLatency(slice),
                                    hemisphere
                                            * target_.memory()
                                                  .slices_per_hemisphere
                                        + slice,
                                    "read",
                                    head * op_.getSeqLen() + token,
                                    32 + half * 4 + byte, 1, 1, 0);
                            }
                        }
                        for (int64_t byte = 0; byte < 4; ++byte) {
                            const int64_t slice = layout.ropeSlices()[byte];
                            emitMem(rewriter_, op_.getLoc(),
                                ropeCycle - readLatency(slice),
                                hemisphere
                                        * target_.memory()
                                              .slices_per_hemisphere
                                    + slice,
                                "read", layout.ropeAddress(token),
                                40 + byte, 1, 1, 0);
                        }
                        emitStagedRope(ropeCycle, hemisphere,
                            projectionValues[projection]);
                        const int64_t writeCycle = ropeCycle + 2;
                        if (kind == AttentionProjectionKind::Query) {
                            const int64_t localColumn = token % 8;
                            for (int64_t half = 0; half < 2; ++half) {
                                const auto& slices =
                                    target_.attention_query_iw_slices(half);
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    const int64_t slice =
                                        slices[localColumn * 2 + byte];
                                    const int64_t cycle =
                                        writeCycle + slice / 4;
                                    emitMem(rewriter_, op_.getLoc(), cycle,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + slice,
                                        "write",
                                        layout.queryIwAddress(head,
                                            half, tokenBlock,
                                            (token % tile) / 8),
                                        half * 2 + byte, 1, 1, 0);
                                    rawWriteEnd = std::max(
                                        rawWriteEnd, cycle + 1);
                                }
                            }
                        } else {
                            const int64_t outputAddress =
                                layout.projectionAddress(
                                    kind, head, tokenBlock);
                            for (int64_t byte = 0; byte < 4; ++byte) {
                                emitMem(rewriter_, op_.getLoc(), writeCycle,
                                    hemisphere
                                            * target_.memory()
                                                  .slices_per_hemisphere
                                        + byte,
                                    "write", outputAddress + token % tile,
                                    byte, 1, 1, 0);
                            }
                            rawWriteEnd = std::max(
                                rawWriteEnd, writeCycle + 1);
                        }
                    }
                }
                phaseStart = std::max(ropeCycle + 1, rawWriteEnd + 1);
            }
        }
        const auto hemisphereName = [](int64_t hemisphere) {
            return hemisphere == 0 ? "east" : "west";
        };
        const int64_t registerGroup =
            target_.streams().mem_slices_per_register_group;
        const int64_t queryHeadsPerKv =
            op_.getQueryHeads() / op_.getKvHeads();
        int64_t copyCycle = phaseStart + 16;
        for (int64_t queryHead = 0;
             queryHead < op_.getQueryHeads(); ++queryHead) {
            const int64_t kvHead = queryHead / queryHeadsPerKv;
            const int64_t source =
                queryHead % target_.memory().hemispheres;
            const int64_t destination =
                kvHead % target_.memory().hemispheres;
            if (source == destination) continue;
            for (int64_t queryBlock = 0;
                 queryBlock < tokenBlocks; ++queryBlock) {
                for (int64_t reduction = 0; reduction < 2;
                     ++reduction) {
                    const auto& slices =
                        target_.attention_query_iw_slices(reduction);
                    for (int64_t phase = 0;
                         phase < target_.throughput().tile_rows;
                         ++phase, ++copyCycle) {
                        for (int64_t stream = 0;
                             stream
                                 < static_cast<int64_t>(slices.size());
                             ++stream) {
                            const int64_t slice = slices[stream];
                            emitMem(rewriter_, op_.getLoc(),
                                copyCycle - readLatency(slice),
                                source
                                        * target_.memory()
                                              .slices_per_hemisphere
                                    + slice,
                                "read",
                                layout.queryIwAddress(queryHead,
                                    reduction, queryBlock, phase),
                                32 + stream, 1, 1, 0);
                            emitVxm(rewriter_, op_.getLoc(),
                                op_.getInput(), copyCycle, stream,
                                "pass", "stream_i8", 32 + stream,
                                0.0f, "immediate", 0, 0.0f, "i8",
                                stream, hemisphereName(source),
                                hemisphereName(destination));
                            emitMem(rewriter_, op_.getLoc(),
                                copyCycle + 1
                                    + slice / registerGroup,
                                destination
                                        * target_.memory()
                                              .slices_per_hemisphere
                                    + slice,
                                "write",
                                layout.queryIwAddress(queryHead,
                                    reduction, queryBlock, phase),
                                stream, 1, 1, 0);
                        }
                    }
                }
            }
            copyCycle += 20;
        }
        return copyCycle + 16;
    }

    const int64_t weightLoadLead =
        (target_.memory().hemispheres - 1) * 8 + 7 + weightToIw + 1;
    int64_t projectionBlock = 0;
    for (int64_t projection = 0; projection < 3; ++projection) {
        const auto kind = projectionKind(projection);
        for (int64_t headBase = 0; headBase < projectionHeads[projection]; headBase += 2) {
            for (int64_t reductionBlock = 0; reductionBlock < hiddenBlocks; ++reductionBlock) {
                const int64_t firstCompute = reductionBlock == 0
                    ? phaseStart + readLatency(layout.weightSlices().back())
                        + weightLoadLead
                    : phaseStart;
                const int64_t dequantStart = firstCompute - weightLoadLead;
                const int64_t weightBuffer =
                    projectionBlock % target_.throughput().mxm_weight_buffers;
                for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                    const int64_t head = headBase + hemisphere;
                    if (head >= projectionHeads[projection]) continue;
                    for (int64_t pulse = 0; pulse < 8; ++pulse) {
                        const int64_t localMxm = pulse / 4;
                        const int64_t column = 3 - pulse % 4;
                        const int64_t cycle = dequantStart + hemisphere * 8 + pulse;
                        const int64_t address = layout.weightAddress(kind,
                            head * projectionHeadBlocks + localMxm,
                            reductionBlock, localMxm, pulse % 4);
                        for (int64_t stream = 0; stream < 8; ++stream) {
                            const int64_t slice = layout.weightSlices()[stream];
                            emitMem(rewriter_, op_.getLoc(), cycle - readLatency(slice),
                                hemisphere * target_.memory().slices_per_hemisphere + slice,
                                "read", address, 32 + stream, 1, 1, 0,
                                "sram",
                                functionArgumentIndex(
                                    projectionValues[projection]));
                        }
                        emitDequant(cycle, hemisphere, localMxm,
                            projectionValues[projection],
                            projectionScales[projection]);
                        emitMxm(rewriter_, op_.getLoc(), cycle + weightToIw,
                            hemisphere * 2 + localMxm, "iw", weightBuffer, column,
                            0, 0, 1, 1, 0, 1, "stream", true,
                            "supercell", 0, dataFormat);
                    }
                }

                const bool finalReduction = reductionBlock + 1 == hiddenBlocks;
                int64_t finalWriteEnd = firstCompute;
                // A final-reduction tile reaches MEM after ACC -> VXM and
                // projection-specific transport. The next token tile may
                // reuse those MEM slice ports for its activation read.
                int64_t writebackDelay = 0;
                if (kind == AttentionProjectionKind::Query) {
                    for (int64_t reduction = 0; reduction < 2;
                         ++reduction) {
                        for (int64_t slice :
                             target_.attention_query_iw_slices(
                                 reduction))
                            writebackDelay = std::max(
                                writebackDelay, 2 + slice / 4);
                    }
                } else if (kind
                           == AttentionProjectionKind::Key) {
                    writebackDelay = 2;
                } else {
                    for (int64_t reduction = 0; reduction < 2;
                         ++reduction) {
                        for (int64_t slice :
                             layout.valuePackSlices(reduction))
                            writebackDelay = std::max(
                                writebackDelay,
                                1 + slice
                                    / target_.streams()
                                          .mem_slices_per_register_group);
                    }
                }
                const int64_t computeBlockCycles =
                    finalReduction
                    ? std::max(
                          target_.mxm_block_issue_interval() + tile,
                          target_.throughput()
                                  .accumulator_to_vxm_latency
                              + tile + writebackDelay
                              + activationLatency)
                    : target_.mxm_block_issue_interval();
                for (int64_t tokenBlock = 0; tokenBlock < tokenBlocks; ++tokenBlock) {
                    for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
                        const int64_t head = headBase + hemisphere;
                        if (head >= projectionHeads[projection]) continue;
                        const int64_t computeCycle = firstCompute
                            + tokenBlock * computeBlockCycles;
                        const int64_t inputAddress =
                            layout.activationAddress(
                                reductionBlock, tokenBlock)
                            + (inputDistributed16
                                    ? inputStagingPlacement
                                          .getAs<mlir::IntegerAttr>(
                                              "base_row")
                                          .getInt()
                                    : 0);
                        const int64_t outputAddress = layout.projectionAddress(
                            kind, head, tokenBlock);
                        llvm::SmallVector<int64_t> segmentRows;
                        llvm::SmallVector<int64_t> segmentStreams;
                        const bool prefetchNextWeight = !finalReduction
                            && tokenBlock + 1 == tokenBlocks;
                        if (prefetchNextWeight) {
                            const int64_t nextFirstCompute =
                                firstCompute + tokenBlocks * computeBlockCycles;
                            const int64_t nextDequantStart =
                                nextFirstCompute - weightLoadLead;
                            const int64_t switchRow = nextDequantStart
                                + hemisphere * 8 + weightToIw - computeCycle;
                            if (switchRow > 0) {
                                segmentRows.push_back(switchRow);
                                segmentStreams.push_back(0);
                            }
                            const int64_t switchedRows = std::min<int64_t>(
                                target_.throughput().tile_rows,
                                tile - switchRow);
                            segmentRows.push_back(switchedRows);
                            segmentStreams.push_back(
                                target_.throughput().mxm_load_streams_per_cycle);
                            if (switchRow + switchedRows < tile) {
                                segmentRows.push_back(
                                    tile - switchRow - switchedRows);
                                segmentStreams.push_back(0);
                            }
                        } else {
                            segmentRows.push_back(tile);
                            segmentStreams.push_back(0);
                        }
                        const char* destination =
                            finalReduction ? "stream" : "sram";
                        int64_t rowOffset = 0;
                        for (std::size_t segment = 0;
                             segment < segmentRows.size(); ++segment) {
                            const int64_t rows = segmentRows[segment];
                            const int64_t streamBase = segmentStreams[segment];
                            const int64_t segmentCycle = computeCycle + rowOffset;
                            if (inputDistributed16) {
                                for (int64_t byte = 0; byte < 2; ++byte) {
                                    emitMem(rewriter_, op_.getLoc(),
                                        segmentCycle - activationLatency,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + projectionActivationSlices[byte],
                                        "read", inputAddress + rowOffset,
                                        streamBase + byte, rows, 1, 1);
                                }
                            } else {
                                for (int64_t byte = 0; byte < 4; ++byte) {
                                    emitMem(rewriter_, op_.getLoc(),
                                        segmentCycle - activationLatency,
                                        hemisphere
                                                * target_.memory()
                                                      .slices_per_hemisphere
                                            + layout.activationSlices()[byte],
                                        "read", inputAddress + rowOffset,
                                        streamBase + byte, rows, 1, 1);
                                }
                            }
                            emitMxm(rewriter_, op_.getLoc(), segmentCycle,
                                hemisphere * 2, "compute", weightBuffer, 0,
                                streamBase, 0, rows, 1, outputAddress, 1,
                                destination, true, "supercell", 0,
                                dataFormat);
                            emitMxm(rewriter_, op_.getLoc(), segmentCycle,
                                hemisphere * 2 + 1, "compute", weightBuffer, 0,
                                streamBase
                                    + (inputDistributed16 ? 0 : 2),
                                4, rows, 1,
                                outputAddress, 1, destination, true,
                                "supercell", 0, dataFormat);
                            rowOffset += rows;
                        }
                        if (!finalReduction) continue;

                        for (int64_t offset = 0; offset < tile; ++offset) {
                            const int64_t token = tokenBlock * tile + offset;
                            const int64_t vxmCycle = computeCycle
                                + target_.throughput().accumulator_to_vxm_latency + offset;
                            if (kind != AttentionProjectionKind::Value) {
                                for (int64_t byte = 0; byte < 4; ++byte) {
                                    const int64_t slice = layout.ropeSlices()[byte];
                                    emitMem(rewriter_, op_.getLoc(), vxmCycle - readLatency(slice),
                                        hemisphere * target_.memory().slices_per_hemisphere + slice,
                                        "read", layout.ropeAddress(token), 40 + byte, 1, 1, 0);
                                }
                            }
                            emitRopeOrCast(vxmCycle, hemisphere,
                                kind != AttentionProjectionKind::Value,
                                projectionValues[projection]);
                            const int64_t writeCycle = vxmCycle
                                + (kind == AttentionProjectionKind::Value ? 1 : 2);
                            if (kind == AttentionProjectionKind::Query) {
                                const int64_t phase = (token % tile) / 8;
                                const int64_t localColumn = token % 8;
                                for (int64_t reduction = 0; reduction < 2; ++reduction) {
                                    const auto& slices = target_.attention_query_iw_slices(reduction);
                                    for (int64_t byte = 0; byte < 2; ++byte) {
                                        const int64_t stream = reduction * 2 + byte;
                                        const int64_t slice = slices[localColumn * 2 + byte];
                                        emitMem(rewriter_, op_.getLoc(), writeCycle + slice / 4,
                                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                                            "write", layout.queryIwAddress(head, reduction, tokenBlock, phase),
                                            stream, 1, 1, 0);
                                        finalWriteEnd = std::max(
                                            finalWriteEnd,
                                            writeCycle + slice / 4 + 1);
                                    }
                                }
                            } else if (kind == AttentionProjectionKind::Key) {
                                for (int64_t byte = 0; byte < 4; ++byte) {
                                    emitMem(rewriter_, op_.getLoc(), writeCycle,
                                        hemisphere * target_.memory().slices_per_hemisphere + byte,
                                        "write", outputAddress + offset, byte, 1, 1, 0);
                                    finalWriteEnd =
                                        std::max(finalWriteEnd,
                                            writeCycle + 1);
                                }
                            } else {
                                const int64_t packedStream = (token % 8) * 2;
                                const int64_t row = (token % tile) / 8;
                                for (int64_t reduction = 0; reduction < 2; ++reduction) {
                                    const auto slices = layout.valuePackSlices(reduction);
                                    for (int64_t byte = 0; byte < 2; ++byte) {
                                        const int64_t slice = slices[packedStream + byte];
                                        emitMem(rewriter_, op_.getLoc(), writeCycle
                                                + slice / target_.streams().mem_slices_per_register_group,
                                            hemisphere * target_.memory().slices_per_hemisphere + slice,
                                            "write", layout.valuePackAddress(head, reduction,
                                                tokenBlock, row), reduction * 2 + byte,
                                            1, 1, 0);
                                        finalWriteEnd = std::max(
                                            finalWriteEnd,
                                            writeCycle
                                                + slice
                                                    / target_.streams()
                                                          .mem_slices_per_register_group
                                                + 1);
                                    }
                                }
                            }
                        }
                    }
                }
                phaseStart = firstCompute + tokenBlocks * computeBlockCycles;
                if (finalReduction)
                    phaseStart = std::max(
                        phaseStart, finalWriteEnd);
                ++projectionBlock;
            }
        }
    }

    const auto hemisphereName = [](int64_t hemisphere) {
        return hemisphere == 0 ? "east" : "west";
    };
    const int64_t groups = target_.streams().mem_slices_per_register_group;
    const int64_t headBlocks = op_.getHeadDim() / tile;

    // Projection alternates logical heads across hemispheres. GQA QK work is
    // placed beside the shared KV head, so copy only Q heads whose projection
    // home differs from that KV home.
    int64_t copyCycle = phaseStart + 16;
    const int64_t queryHeadsPerKv = op_.getQueryHeads() / op_.getKvHeads();
    for (int64_t queryHead = 0; queryHead < op_.getQueryHeads(); ++queryHead) {
        const int64_t kvHead = queryHead / queryHeadsPerKv;
        const int64_t source =
            queryHead % target_.memory().hemispheres;
        const int64_t destination =
            kvHead % target_.memory().hemispheres;
        if (source == destination) continue;
        for (int64_t queryBlock = 0; queryBlock < tokenBlocks; ++queryBlock) {
                for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
                    const auto& slices =
                    layout.queryIwSlices(reduction);
                for (int64_t phase = 0;
                     phase < target_.throughput().tile_rows;
                     ++phase, ++copyCycle) {
                    for (int64_t stream = 0;
                         stream < static_cast<int64_t>(slices.size()); ++stream) {
                        const int64_t slice = slices[stream];
                        emitMem(rewriter_, op_.getLoc(),
                            copyCycle - readLatency(slice),
                            source * target_.memory().slices_per_hemisphere
                                + slice,
                            "read",
                            layout.queryIwAddress(
                                queryHead, reduction, queryBlock, phase),
                            32 + stream, 1, 1, 0);
                        emitVxm(rewriter_, op_.getLoc(), op_.getInput(),
                            copyCycle, stream, "pass",
                            "stream_i8", 32 + stream, 0.0f,
                            "immediate", 0, 0.0f, "i8", stream,
                            hemisphereName(source),
                            hemisphereName(destination));
                        emitMem(rewriter_, op_.getLoc(),
                            copyCycle + 1 + slice / groups,
                            destination
                                    * target_.memory().slices_per_hemisphere
                                + slice,
                            "write",
                            layout.queryIwAddress(
                                queryHead, reduction, queryBlock, phase),
                            stream, 1, 1, 0);
                    }
                }
            }
        }
        copyCycle += 20;
    }

    // Each hemisphere has two MXMs. Duplicate K's two 16-bit pairs onto slices
    // 4..7 so MXM0 and MXM1 can consume the same KV head concurrently.
    int64_t keyCopyCycle = copyCycle + 16;
    for (int64_t keyHead = 0; keyHead < op_.getKvHeads(); ++keyHead) {
        const int64_t hemisphere =
            keyHead % target_.memory().hemispheres;
        const char* hemi = hemisphereName(hemisphere);
        for (int64_t token = 0; token < op_.getSeqLen();
             ++token, ++keyCopyCycle) {
            for (int64_t byte = 0; byte < 4; ++byte) {
                emitMem(rewriter_, op_.getLoc(),
                    keyCopyCycle - readLatency(byte),
                    hemisphere * target_.memory().slices_per_hemisphere
                        + byte,
                    "read", layout.keyAddress(keyHead, 0, token / tile)
                            + token % tile,
                    32 + byte, 1, 1, 0);
            }
            emitVxm(rewriter_, op_.getLoc(), op_.getKeyWeight(),
                keyCopyCycle, 0, "pass",
                streamKind, 32, 0.0f,
                "immediate", 0, 0.0f, dataFormat, 0, hemi, hemi);
            emitVxm(rewriter_, op_.getLoc(), op_.getKeyWeight(),
                keyCopyCycle, 1, "pass",
                streamKind, 34, 0.0f,
                "immediate", 0, 0.0f, dataFormat, 2, hemi, hemi);
            for (int64_t byte = 0; byte < 4; ++byte) {
                const int64_t slice = 4 + byte;
                emitMem(rewriter_, op_.getLoc(),
                    keyCopyCycle + 1 + slice / groups,
                    hemisphere * target_.memory().slices_per_hemisphere
                        + slice,
                    "write", layout.keyAddress(keyHead, 0, token / tile)
                            + token % tile,
                    byte, 1, 1, 0);
            }
        }
        keyCopyCycle += 12;
    }
    return keyCopyCycle + 16;
}

void AttentionScheduleEmitter::emitQk(int64_t qkStart,
    int64_t qkWaveCycles, int64_t qkIwToComputeCycles,
    bool fusedSoftmax)
{
    const AttentionMemoryLayout layout(op_, target_);
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op_.getInput().getType())
            .getElementType();
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(elementType);
    const int64_t tile = target_.throughput().mxm_rows;
    const int64_t tokenBlocks = op_.getSeqLen() / tile;
    const int64_t headBlocks = op_.getHeadDim() / tile;
    const int64_t issue = target_.mxm_block_issue_interval();

    for (std::size_t waveIndex = 0;
         waveIndex < stage_plan_.qk_waves.size(); ++waveIndex) {
        const int64_t waveStart = qkStart + static_cast<int64_t>(waveIndex) * qkWaveCycles;
        const int64_t firstIwCycle = waveStart + target_.throughput().mxm_earliest_iw_cycle
            + *target_.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, 0);
        const int64_t firstComputeCycle = firstIwCycle + qkIwToComputeCycles;
        for (const auto& work : stage_plan_.qk_waves[waveIndex].slots) {
            if (!work) continue;
            const int64_t mxm = work->hemisphere * target_.throughput().mxms_per_hemisphere
                + work->local_mxm;
            // Direct16 Q weights occupy E0..E15. Keep K activations on the
            // upper east streams so the next Q block can be loaded while the
            // current block computes.
            const int64_t activationStream = 16 + work->local_mxm * 2;
            const int64_t outputStream = work->local_mxm * 4;
            const bool wavefront =
                target_.supports_mxm_weight_activation_overlap()
                && target_.throughput().mxm_weight_buffers >= 2
                && target_.throughput().mxms_per_hemisphere == 1;
            for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
                const auto iwSlices = layout.queryIwSlices(reduction);
                const int64_t reductionComputeCycle = firstComputeCycle
                    + reduction * tokenBlocks * issue;
                const int64_t reductionIwCycle = wavefront
                    ? reductionComputeCycle - qkIwToComputeCycles
                        - work->local_mxm
                            * target_.throughput().tile_rows
                    : firstIwCycle + work->local_mxm * 8
                        + reduction * tile / 8;
                for (int64_t phase = 0; phase < tile / 8; ++phase) {
                    const int64_t iwCycle = reductionIwCycle + phase;
                    const int64_t sourcePhase = tile / 8 - 1 - phase;
                    const int64_t queryAddress = layout.queryIwAddress(
                        work->query_head, reduction, work->query_block,
                        sourcePhase);
                    for (int64_t stream = 0; stream < static_cast<int64_t>(iwSlices.size()); ++stream) {
                        const int64_t slice = iwSlices[stream];
                        const int64_t latency = *target_.transport_latency(
                            target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                            target::StreamDirection::East, slice);
                        emitMem(rewriter_, op_.getLoc(), iwCycle - latency,
                            work->hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", queryAddress,
                            work->local_mxm * static_cast<int64_t>(iwSlices.size()) + stream,
                            1, 1, 0);
                    }
                    emitMxm(rewriter_, op_.getLoc(), iwCycle, mxm, "iw",
                        reduction
                            % target_.throughput().mxm_weight_buffers,
                        tile / 8 - 1 - phase, 0, 0, 1, 1,
                        0, 1, "stream", true, "supercell", 0,
                        dataFormat);
                }
            }
            for (int64_t reduction = 0; reduction < headBlocks; ++reduction) {
                const bool finalReduction = reduction + 1 == headBlocks;
                for (int64_t keyBlock = 0; keyBlock < tokenBlocks; ++keyBlock) {
                    const int64_t computeCycle = firstComputeCycle
                        + (reduction * tokenBlocks + keyBlock) * issue;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice =
                            target_.throughput().mxms_per_hemisphere == 1
                            ? layout.keySlices(reduction)[byte]
                            : work->local_mxm * 4
                                + (reduction % 2) * 2 + byte;
                        const int64_t latency = *target_.transport_latency(
                            target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
                            target::StreamDirection::East, slice);
                        emitMem(rewriter_, op_.getLoc(), computeCycle - latency,
                            work->hemisphere * target_.memory().slices_per_hemisphere + slice,
                            "read", layout.keyAddress(work->kv_head,
                                reduction, keyBlock),
                            activationStream + byte, tile, 1, 1);
                    }
                    emitMxm(rewriter_, op_.getLoc(), computeCycle, mxm, "compute",
                        reduction
                            % target_.throughput().mxm_weight_buffers,
                        0, activationStream, outputStream, tile, 1,
                        layout.scoreAccumulatorAddress(work->query_head,
                            work->query_block, keyBlock),
                        1,
                        finalReduction ? "stream" : "sram",
                        finalReduction,
                        "supercell", 0, dataFormat);
                    if (!fusedSoftmax && finalReduction) {
                        // Tail softmax runs after all QK waves, so completed
                        // FP32 scores must leave the finite MXM accumulator.
                        // Write the four result byte streams directly to the
                        // score plane and clear the reusable accumulator rows.
                        for (int64_t byte = 0; byte < 4; ++byte) {
                            const int64_t slice =
                                layout.scaledScoreSlices(work->local_mxm)[byte];
                            const auto latency = target_.transport_latency(
                                target::StreamEndpoint::MxmResult,
                                target::StreamEndpoint::Mem,
                                target::StreamDirection::West, slice);
                            if (!latency) continue;
                            emitMem(rewriter_, op_.getLoc(),
                                computeCycle
                                    + target_.mxm_first_result_latency()
                                    + *latency,
                                work->hemisphere
                                        * target_.memory().slices_per_hemisphere
                                    + slice,
                                "write", layout.scoreAddress(
                                    work->query_head, work->query_block,
                                    keyBlock * tile),
                                32 + outputStream + byte, tile, 1, 1);
                        }
                    }
                }
            }
        }
    }
}

} // namespace ftlpu::compiler::schedule
