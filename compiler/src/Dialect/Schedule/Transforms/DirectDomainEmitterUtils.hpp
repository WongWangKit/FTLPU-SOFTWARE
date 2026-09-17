#pragma once

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/PatternMatch.h"

namespace ftlpu::compiler::schedule::direct_domain_detail {

// Materialize exactly one operator-owned MEM domain.  Unlike emitMem(), this
// helper never inspects an earlier operation and therefore cannot choose a
// loop dimension from incidental emission order.
inline void emitMem3D(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle, int64_t queue,
    llvm::StringRef opcode, int64_t address, int64_t packedStream,
    int64_t repeatCount, int64_t repeatInterval, int64_t addressStride,
    int64_t waveCount = 1, int64_t waveInterval = 1,
    int64_t waveAddressStride = 0, int64_t groupCount = 1,
    int64_t groupInterval = 1, int64_t groupAddressStride = 0,
    int64_t addressBinding = -1, int64_t bank = -1,
    int64_t weightPage = -1, int64_t logicalBaseRow = -1)
{
    const int64_t hemisphere =
        queue / target.memory().slices_per_hemisphere;
    const int64_t slice =
        queue % target.memory().slices_per_hemisphere;
    mlir::OperationState state(location, MemTransferOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr(
            "hemisphere", rewriter.getI64IntegerAttr(hemisphere)),
        rewriter.getNamedAttr("slice", rewriter.getI64IntegerAttr(slice)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr(
            "address", rewriter.getI64IntegerAttr(address)),
        rewriter.getNamedAttr(
            "packed_stream", rewriter.getI64IntegerAttr(packedStream)),
        rewriter.getNamedAttr(
            "repeat_count", rewriter.getI64IntegerAttr(repeatCount)),
        rewriter.getNamedAttr(
            "repeat_interval", rewriter.getI64IntegerAttr(repeatInterval)),
        rewriter.getNamedAttr(
            "address_stride", rewriter.getI64IntegerAttr(addressStride)),
        rewriter.getNamedAttr(
            "wave_count", rewriter.getI64IntegerAttr(waveCount)),
        rewriter.getNamedAttr(
            "wave_interval", rewriter.getI64IntegerAttr(waveInterval)),
        rewriter.getNamedAttr("wave_address_stride",
            rewriter.getI64IntegerAttr(waveAddressStride)),
        rewriter.getNamedAttr(
            "group_count", rewriter.getI64IntegerAttr(groupCount)),
        rewriter.getNamedAttr(
            "group_interval", rewriter.getI64IntegerAttr(groupInterval)),
        rewriter.getNamedAttr("group_address_stride",
            rewriter.getI64IntegerAttr(groupAddressStride)),
    });
    if (addressBinding >= 0)
        state.addAttribute(
            "address_binding", rewriter.getI64IntegerAttr(addressBinding));
    if (bank >= 0)
        state.addAttribute("bank", rewriter.getI64IntegerAttr(bank));
    if (weightPage >= 0)
        state.addAttribute(
            "weight_page", rewriter.getI64IntegerAttr(weightPage));
    if (logicalBaseRow >= 0)
        state.addAttribute(
            "logical_base_row", rewriter.getI64IntegerAttr(logicalBaseRow));
    rewriter.create(state);
}

} // namespace ftlpu::compiler::schedule::direct_domain_detail
