#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {

std::pair<VxmOp, VxmOp> emitFfnSwishAlu(
    mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Type resultType, mlir::Value gateValue, mlir::Value upValue,
    const target::LPUTargetModel& target, FfnScheduleStrategy strategy,
    int64_t cycle, int64_t hemisphere, int64_t outputStream,
    int64_t repeatCount, int64_t repeatInterval)
{
    const int64_t inputStream = strategy == FfnScheduleStrategy::Fused
        ? 8 + hemisphere * 8
        : 0;
    const int64_t encodedInput =
        target.streams().streams_per_direction + inputStream;
    const auto hemi = hemisphere_name(hemisphere);
    const auto dataFormat = lpu_16bit_data_format(
        llvm::cast<mlir::RankedTensorType>(
            resultType).getElementType());
    auto head = create_vxm(rewriter, location, gateValue, upValue,
        resultType, cycle, 0, "negate", "stream_bf16",
        encodedInput, 0, "stream_bf16",
        encodedInput + 2, 0, "fp32", -1, repeatCount, repeatInterval,
        hemi, hemi);
    if (strategy == FfnScheduleStrategy::Fused) {
        // A fused task feeds one completed projection tile from one
        // hemisphere. Feed both mirrored VXM chains so neither physical
        // output is left on the passive SR path. Only the owner chain is the
        // authoritative result; the fused stage repairs the sink copy before
        // Down projection consumes it.
        const auto source = rewriter.getStringAttr(hemi);
        head->setAttr("lhs_stream_source", source);
        head->setAttr("rhs_stream_source", source);
    }
    mlir::Value value = head.getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 1, "exp", "previous", 0, 0,
        "immediate", 0, 0, "fp32", -1, repeatCount, repeatInterval,
        hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 2, "add", "previous", 0, 0,
        "immediate", 0, 1, "fp32", -1, repeatCount, repeatInterval,
        hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 3, "reciprocal", "previous",
        0, 0, "immediate", 0, 0, "fp32", -1,
        repeatCount, repeatInterval,
        hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 4, "multiply", "previous", 0, 0,
        "original", 0, 0, "fp32", -1, repeatCount, repeatInterval,
        hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 5, "multiply", "previous", 0, 0,
        "auxiliary", 0, 0, "fp32", -1, repeatCount, repeatInterval,
        hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 6, "bypass", "previous", 0, 0,
        "immediate", 0, 0, "fp32", -1, repeatCount, repeatInterval,
        hemi, hemi).getResult();
    const int64_t peer = 1 - hemisphere;
    auto output = create_vxm(rewriter, location, value, upValue,
        resultType, cycle, 7, "bypass", "previous", 0, 0,
        "immediate", 0, 0, dataFormat, outputStream,
        repeatCount, repeatInterval, hemi, hemisphere_name(peer));
    return {output, output};
}

mlir::Value emitFfnSwishResultRow(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& plan, const target::LPUTargetModel& target,
    llvm::ArrayRef<int64_t> hiddenSlices, mlir::Value output,
    int64_t inputCycle, int64_t mTile, int64_t pair, int64_t row,
    int64_t sourceHemisphere, bool mirroredBroadcast)
{
    constexpr int64_t kVxmSwishLatency = 17;
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t destination = 1 - sourceHemisphere;
    const int64_t destinationStream = sourceHemisphere == 0 ? 6 : 14;
    const auto hiddenKind =
        plan.getHidden0Placement().getAs<mlir::StringAttr>("kind");
    const bool singleMxmVector =
        target.throughput().mxms_per_hemisphere == 1;
    const int64_t nblock = singleMxmVector
        ? (pair / 2) * 4 + sourceHemisphere * 2 + pair % 2
        : pair;
    const int64_t hiddenBaseRow =
        get_base_row(plan.getHidden0Placement());
    const int64_t hiddenBank = plan.getHidden0Placement()
        .getAs<mlir::IntegerAttr>("bank").getInt();
    const bool distributed16 = hiddenKind
        && hiddenKind.getValue() == "fp16_mxm_distributed_16";

    mlir::Value lastHidden;
    for (int64_t byte = 0; byte < 2; ++byte) {
        int64_t slice = hiddenSlices[2 * (nblock % 2) + byte];
        int64_t address = hiddenBaseRow
            + (nblock / 2) * plan.getM() + mTile * tile + row;
        llvm::StringRef kind = "fp16_mxm_activation_planar";
        if (distributed16) {
            const int64_t token = mTile * tile + row;
            const int64_t tokenWithinBlock = token % tile;
            const int64_t tokenWave =
                tokenWithinBlock / target.throughput().mxm_block_rows;
            const int64_t tokenLane =
                tokenWithinBlock % target.throughput().mxm_block_rows;
            const int64_t reductionBlocks = plan.getHidden() / tile;
            slice = hiddenSlices[2 * tokenLane + byte];
            address = hiddenBaseRow
                + ((token / tile) * reductionBlocks + nblock)
                    * target.throughput().tile_rows
                + tokenWave;
            kind = "fp16_mxm_distributed_16";
        }
        const auto latency = target.transport_latency(
            target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice);
        if (!latency) return {};
        const auto emitWrite = [&](int64_t hemisphere, int64_t stream) {
            auto placement = schedule_placement(rewriter, {slice}, address,
                1, 1, hemisphere_name(hemisphere), kind, hiddenBank);
            auto write = rewriter.create<MemWriteOp>(plan.getLoc(), output,
                inputCycle + kVxmSwishLatency + *latency,
                1, stream + byte, 1, 0,
                rewriter.getStringAttr("east"), plan.getHidden0Address(),
                placement, tile);
            lastHidden = write.getOutput();
        };
        emitWrite(destination, destinationStream);
        if (mirroredBroadcast) {
            // Compact queue 7 controls physical C7 and C15. Their fixed
            // outputs are W6/W7 and E14/E15. Consume the non-owner result in
            // the peer hidden slot before it can drift into the MXM weight
            // stream window. The stage later overwrites this sink copy from
            // the authoritative owner result.
            const int64_t mirroredDestination = sourceHemisphere;
            const int64_t mirroredStream =
                sourceHemisphere == 0 ? 14 : 6;
            emitWrite(mirroredDestination, mirroredStream);
        }
    }
    return lastHidden;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
