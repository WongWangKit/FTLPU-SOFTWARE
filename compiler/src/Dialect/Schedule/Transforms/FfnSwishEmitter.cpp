#include "FfnEmitterUtils.hpp"

namespace ftlpu::compiler::schedule::ffn_detail {

std::pair<VxmOp, VxmOp> emitFfnSwishAlu(
    mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::Type resultType, mlir::Value gateValue, mlir::Value upValue,
    const target::LPUTargetModel& target, FfnScheduleStrategy strategy,
    int64_t cycle, int64_t hemisphere)
{
    const auto& throughput = target.throughput();
    const int64_t inputStream = strategy == FfnScheduleStrategy::Fused
        ? 8 + hemisphere * 8
        : 0;
    const int64_t outputStream = strategy == FfnScheduleStrategy::Fused
        ? target.streams().streams_per_direction - 2
        : 0;
    const int64_t encodedInput =
        target.streams().streams_per_direction + inputStream;
    const auto hemi = hemisphere_name(hemisphere);
    mlir::Value value;

    value = create_vxm(rewriter, location, gateValue, upValue,
        resultType, cycle, 0, "negate", "stream_f32",
        encodedInput, 0, "immediate", 0, 0, "fp32", -1, 1, 1,
        hemi, hemi).getResult();
    value = create_vxm(rewriter, location, gateValue, upValue,
        resultType, cycle, 1, "multiply", "stream_f32",
        encodedInput, 0, "stream_f32",
        encodedInput + throughput.mxm_result_streams,
        0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 1, 2, "exp", "alu", 0, 0,
        "immediate", 0, 0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 1, 5, "pass", "alu", 1, 0,
        "immediate", 0, 0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 2, 3, "add", "alu", 2, 0,
        "immediate", 0, 1, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 2, 6, "pass", "alu", 5, 0,
        "immediate", 0, 0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 3, 4, "divide", "immediate",
        0, 1, "alu", 3, 0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 3, 7, "pass", "alu", 6, 0,
        "immediate", 0, 0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    value = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 4, 8, "multiply", "alu", 7, 0,
        "alu", 4, 0, "fp32", -1, 1, 1, hemi, hemi).getResult();
    auto localCast = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 5, 9, "cast", "alu", 8, 0,
        "immediate", 0, 0, "fp16", outputStream, 1, 1, hemi, hemi);
    const int64_t peer = 1 - hemisphere;
    auto peerCast = create_vxm(rewriter, location, value, upValue,
        resultType, cycle + 5, 11, "cast", "alu", 8, 0,
        "immediate", 0, 0, "fp16", outputStream, 1, 1, hemi,
        hemisphere_name(peer));
    return {localCast, peerCast};
}

mlir::Value emitFfnSwishRow(mlir::IRRewriter& rewriter,
    PrimitiveFfnSchedulePlan& plan, const target::LPUTargetModel& target,
    FfnScheduleStrategy strategy, llvm::ArrayRef<int64_t> hiddenSlices,
    mlir::Value gateValue, mlir::Value upValue, int64_t cycle,
    int64_t mTile, int64_t pair, int64_t row, int64_t hemisphere)
{
    const auto& memory = target.memory();
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t nblock = pair * memory.hemispheres + hemisphere;
    const int64_t outputStream = strategy == FfnScheduleStrategy::Fused
        ? target.streams().streams_per_direction - 2
        : 0;
    auto [localCast, peerCast] = emitFfnSwishAlu(rewriter, plan.getLoc(),
        plan.getResult().getType(), gateValue, upValue, target, strategy,
        cycle, hemisphere);

    mlir::Value lastHidden;
    for (int64_t destination = 0; destination < memory.hemispheres;
         ++destination) {
        for (int64_t byte = 0; byte < 2; ++byte) {
            auto placement = schedule_placement(rewriter,
                {hiddenSlices[byte]}, nblock * plan.getM() + mTile * tile + row,
                1, 1, hemisphere_name(destination),
                "fp16_mxm_activation_planar");
            mlir::Value output =
                destination == hemisphere ? localCast.getResult()
                                          : peerCast.getResult();
            auto write = rewriter.create<MemWriteOp>(plan.getLoc(), output,
                cycle + 6
                    + hiddenSlices[byte]
                        / target.streams().mem_slices_per_register_group,
                1, outputStream + byte, 1, 0,
                rewriter.getStringAttr("east"), plan.getHidden0Address(),
                placement, tile);
            lastHidden = write.getOutput();
        }
    }
    return lastHidden;
}

} // namespace ftlpu::compiler::schedule::ffn_detail
