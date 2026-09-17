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

mlir::Value emitFfnSwishResultTile(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& plan, const target::LPUTargetModel& target,
    llvm::ArrayRef<int64_t> hiddenSlices, mlir::Value output,
    int64_t inputCycle, int64_t mTile, int64_t pair,
    int64_t sourceHemisphere, int64_t rowCount, bool mirroredBroadcast,
    FfnLoopDomain3D outerDomain)
{
    constexpr int64_t kVxmSwishLatency = 17;
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t blockRows = target.throughput().mxm_block_rows;
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

    const auto emitWrite = [&](int64_t slice, int64_t address,
                               int64_t firstRow, int64_t byte,
                               int64_t hemisphere, int64_t stream,
                               FfnLoopDomain3D domain,
                               llvm::StringRef kind) {
        const auto latency = target.transport_latency(
            target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice);
        if (!latency) return;
        const int64_t innerCount = distributed16 ? 1 : rowCount;
        auto placement = schedule_placement(rewriter, {slice}, address,
            innerCount, 1, hemisphere_name(hemisphere), kind, hiddenBank);
        auto write = rewriter.create<MemWriteOp>(plan.getLoc(), output,
            inputCycle + firstRow + kVxmSwishLatency + *latency,
            innerCount, stream + byte, 1, 0,
            rewriter.getStringAttr("east"), plan.getHidden0Address(),
            placement, innerCount * tile);
        setFfnLoopDomain3D(write.getOperation(), rewriter, domain);
        lastHidden = write.getOutput();
    };

    if (!distributed16) {
        FfnLoopDomain3D domain = outerDomain;
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = hiddenSlices[2 * (nblock % 2) + byte];
            const int64_t address = hiddenBaseRow
                + (nblock / 2) * plan.getM() + mTile * tile;
            emitWrite(slice, address, 0, byte, destination,
                destinationStream, domain, "fp16_mxm_activation_planar");
            if (mirroredBroadcast) {
                emitWrite(slice, address, 0, byte, sourceHemisphere,
                    sourceHemisphere == 0 ? 14 : 6, domain,
                    "fp16_mxm_activation_planar");
            }
        }
        return lastHidden;
    }

    const int64_t hiddenBlocks = plan.getHidden() / tile;
    for (int64_t tokenLane = 0; tokenLane < blockRows; ++tokenLane) {
        if (tokenLane >= rowCount) break;
        const int64_t occurrenceCount = 1
            + (rowCount - 1 - tokenLane) / blockRows;
        const int64_t token = mTile * tile + tokenLane;
        const int64_t tokenWave = (token % tile) / blockRows;
        const int64_t address = hiddenBaseRow
            + ((token / tile) * hiddenBlocks + nblock)
                * target.throughput().tile_rows
            + tokenWave;
        FfnLoopDomain3D domain = outerDomain;
        domain.wave_count = occurrenceCount;
        domain.wave_interval = blockRows;
        domain.wave_address_stride = 1;
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = hiddenSlices[2 * tokenLane + byte];
            emitWrite(slice, address, tokenLane, byte, destination,
                destinationStream, domain, "fp16_mxm_distributed_16");
            if (mirroredBroadcast) {
                emitWrite(slice, address, tokenLane, byte, sourceHemisphere,
                    sourceHemisphere == 0 ? 14 : 6, domain,
                    "fp16_mxm_distributed_16");
            }
        }
    }
    return lastHidden;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
