#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <array>

namespace ftlpu::compiler::schedule {
namespace {

using attention_detail::emitMem;
using ffn_detail::create_vxm;

mlir::DictionaryAttr allocationPlacement(
    mlir::ArrayAttr allocations, int64_t index)
{
    return llvm::cast<mlir::DictionaryAttr>(allocations[index])
        .getAs<mlir::DictionaryAttr>("placement");
}

llvm::SmallVector<int64_t> placementSlices(
    mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute value :
        placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(value).getInt());
    return result;
}

struct TileAddress {
    std::array<int64_t, 2> slices;
    int64_t row;
    int64_t hemisphere;
};

bool isDistributed16(mlir::DictionaryAttr placement)
{
    const auto kind = placement.getAs<mlir::StringAttr>("kind");
    return kind && kind.getValue() == "fp16_mxm_distributed_16";
}

mlir::FailureOr<TileAddress> distributedAddress(
    mlir::DictionaryAttr placement, int64_t block, int64_t row,
    int64_t columnBlocks, int64_t hemisphere)
{
    const auto slices = placementSlices(placement);
    if (!isDistributed16(placement) || slices.size() != 16)
        return mlir::failure();
    const int64_t tile = 32;
    const int64_t tokenBlock = row / tile;
    const int64_t tokenWithinBlock = row % tile;
    const int64_t tokenWave = tokenWithinBlock / 8;
    const int64_t tokenLane = tokenWithinBlock % 8;
    const int64_t base =
        placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    return TileAddress {
        {slices[2 * tokenLane], slices[2 * tokenLane + 1]},
        base + (tokenBlock * columnBlocks + block) * 4 + tokenWave,
        hemisphere};
}

mlir::FailureOr<TileAddress> tileAddress(
    mlir::DictionaryAttr placement, int64_t block, int64_t rows,
    int64_t preferredHemisphere)
{
    const auto kind = placement.getAs<mlir::StringAttr>("kind");
    const auto slices = placementSlices(placement);
    const int64_t base =
        placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    if (!kind || slices.size() < 2) return mlir::failure();
    if (kind.getValue() == "fp16_mxm_activation_planar")
        return TileAddress {{slices[0], slices[1]},
            base + block * rows, preferredHemisphere};
    if (kind.getValue() == "fp16_pair_planar") {
        const int64_t pair = block % 2;
        if (slices.size() < static_cast<std::size_t>(2 * pair + 2))
            return mlir::failure();
        const auto hemisphereAttr =
            placement.getAs<mlir::StringAttr>("hemisphere");
        const bool dual = hemisphereAttr
            && hemisphereAttr.getValue() == "both";
        return TileAddress {{slices[2 * pair], slices[2 * pair + 1]},
            base + (dual ? block / 4 : block / 2) * rows,
            dual ? (block / 2) % 2 : 0};
    }
    if (kind.getValue() == "fp16_mxm_distributed_16")
        return distributedAddress(
            placement, block, 0, 1, preferredHemisphere);
    return mlir::failure();
}

BindingOp createOutputBinding(mlir::IRRewriter& rewriter,
    stream::ElementwiseTaskOp op, int64_t index,
    mlir::DictionaryAttr placement)
{
    const auto type =
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("output")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr("result")),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * 2)),
        rewriter.getNamedAttr("placement", placement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

void createTimeline(mlir::IRRewriter& rewriter,
    stream::ElementwiseTaskOp op, int64_t start, int64_t end)
{
    mlir::OperationState state(
        op.getLoc(), TimelineOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("elementwise.add")),
        rewriter.getNamedAttr(
            "start", rewriter.getI64IntegerAttr(start)),
        rewriter.getNamedAttr(
            "end", rewriter.getI64IntegerAttr(end)),
    });
    rewriter.create(state);
}

mlir::LogicalResult lowerElementwise(mlir::IRRewriter& rewriter,
    stream::ElementwiseTaskOp op,
    const target::LPUTargetModel& target, int64_t outputIndex)
{
    const auto type =
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType());
    const int64_t rows = type.getDimSize(0);
    const int64_t columns = type.getDimSize(1);
    const int64_t tile = target.throughput().mxm_rows;
    if (op.getKind() != "add" || type.getRank() != 2
        || !type.getElementType().isF16() || rows % tile != 0
        || columns % tile != 0) {
        return op.emitError(
            "elementwise schedule currently supports tile-aligned FP16 add");
    }

    const auto lhsPlacement =
        allocationPlacement(op.getLhsAllocations(), 0);
    const auto rhsPlacement =
        allocationPlacement(op.getRhsAllocations(), 0);
    const auto resultPlacement =
        allocationPlacement(op.getResultAllocations(), 0);
    const auto resultKind =
        resultPlacement.getAs<mlir::StringAttr>("kind").getValue();
    const auto resultSlices = placementSlices(resultPlacement);
    const bool resultDistributed = isDistributed16(resultPlacement);
    if ((!resultDistributed && resultSlices.size() < 4)
        || (resultDistributed && resultSlices.size() != 16)
        || (resultKind != "fp16_pair_planar"
            && resultKind != "fp16_mxm_activation_planar"
            && resultKind != "fp16_mxm_distributed_16"))
        return op.emitError("unsupported elementwise result layout");

    const auto westLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice).value();
    };
    const auto eastLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice).value();
    };

    rewriter.setInsertionPoint(op);
    int64_t cycle = 10;
    const int64_t start = cycle;
    mlir::Value finalValue = op.getLhs();
    const int64_t columnBlocks = columns / tile;
    for (int64_t block = 0; block < columnBlocks; ++block) {
        auto probeLhs = tileAddress(lhsPlacement, block, rows, 0);
        auto probeRhs = tileAddress(rhsPlacement, block, rows, 0);
        if (mlir::failed(probeLhs) || mlir::failed(probeRhs))
            return op.emitError("unsupported elementwise operand layout");
        const int64_t hemisphere = std::max(
            probeLhs->hemisphere, probeRhs->hemisphere);
        auto lhs = tileAddress(
            lhsPlacement, block, rows, hemisphere);
        auto rhs = tileAddress(
            rhsPlacement, block, rows, hemisphere);
        auto result = tileAddress(
            resultPlacement, block, rows, hemisphere);
        if (mlir::failed(lhs) || mlir::failed(rhs)
            || mlir::failed(result))
            return op.emitError("unsupported elementwise operand layout");

        const int64_t vxmCycle = cycle;
        const auto emitOperand = [&](mlir::DictionaryAttr placement,
                                     const TileAddress& address,
                                     int64_t streamBase) {
            if (isDistributed16(placement)) {
                for (int64_t row = 0; row < rows; ++row) {
                    auto distributed = distributedAddress(placement,
                        block, row, columnBlocks, hemisphere);
                    if (mlir::failed(distributed)) return false;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        emitMem(rewriter, op.getLoc(),
                            vxmCycle + row
                                - westLatency(
                                    distributed->slices[byte]),
                            distributed->hemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + distributed->slices[byte],
                            "read", distributed->row,
                            streamBase + byte, 1, 1, 1);
                    }
                }
                return true;
            }
            for (int64_t byte = 0; byte < 2; ++byte) {
                emitMem(rewriter, op.getLoc(),
                    vxmCycle - westLatency(address.slices[byte]),
                    address.hemisphere
                            * target.memory().slices_per_hemisphere
                        + address.slices[byte],
                    "read", address.row, streamBase + byte,
                    rows, 1, 1);
            }
            return true;
        };
        if (!emitOperand(lhsPlacement, *lhs, 32)
            || !emitOperand(rhsPlacement, *rhs, 34))
            return op.emitError("invalid distributed elementwise address");
        auto sum = create_vxm(rewriter, op.getLoc(),
            op.getLhs(), op.getRhs(), type, vxmCycle, 0, "add",
            "stream_f16", 32, 0.0f, "stream_f16", 34, 0.0f,
            "fp32", -1, rows, 1,
            hemisphere == 0 ? "east" : "west",
            hemisphere == 0 ? "east" : "west");
        auto cast = create_vxm(rewriter, op.getLoc(),
            sum.getResult(), sum.getResult(), type, vxmCycle + 1, 1,
            "cast", "alu", 0, 0.0f, "immediate", 0, 0.0f,
            "fp16", 0, rows, 1,
            hemisphere == 0 ? "east" : "west",
            hemisphere == 0 ? "east" : "west");
        finalValue = cast.getResult();

        int64_t outputSliceCount = 2;
        if (resultKind == "fp16_mxm_activation_planar") {
            create_vxm(rewriter, op.getLoc(),
                sum.getResult(), sum.getResult(), type,
                vxmCycle + 1, 2, "cast", "alu", 0, 0.0f,
                "immediate", 0, 0.0f, "fp16", 2,
                rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west");
            create_vxm(rewriter, op.getLoc(),
                sum.getResult(), sum.getResult(), type,
                vxmCycle + 1, 3, "cast", "alu", 0, 0.0f,
                "immediate", 0, 0.0f, "fp16", 0,
                rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "west" : "east");
            create_vxm(rewriter, op.getLoc(),
                sum.getResult(), sum.getResult(), type,
                vxmCycle + 1, 4, "cast", "alu", 0, 0.0f,
                "immediate", 0, 0.0f, "fp16", 2,
                rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "west" : "east");
            outputSliceCount = 4;
        } else if (resultDistributed) {
            create_vxm(rewriter, op.getLoc(),
                sum.getResult(), sum.getResult(), type,
                vxmCycle + 1, 2, "cast", "alu", 0, 0.0f,
                "immediate", 0, 0.0f, "fp16", 0,
                rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "west" : "east");
        }
        if (resultDistributed) {
            for (int64_t outputHemisphere = 0;
                 outputHemisphere < target.memory().hemispheres;
                 ++outputHemisphere) {
                const int64_t streamBase =
                    outputHemisphere == hemisphere ? 0 : 0;
                for (int64_t row = 0; row < rows; ++row) {
                    auto output = distributedAddress(resultPlacement,
                        block, row, columnBlocks, outputHemisphere);
                    if (mlir::failed(output))
                        return op.emitError(
                            "invalid distributed elementwise result");
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        emitMem(rewriter, op.getLoc(),
                            vxmCycle + 1 + row
                                + eastLatency(output->slices[byte]),
                            outputHemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + output->slices[byte],
                            "write", output->row,
                            streamBase + byte, 1, 1, 1);
                    }
                }
            }
            cycle += rows + 2 + eastLatency(resultSlices.back());
            continue;
        }
        const int64_t firstOutputHemisphere =
            resultKind == "fp16_mxm_activation_planar"
            ? 0 : result->hemisphere;
        const int64_t outputHemispheres =
            resultKind == "fp16_mxm_activation_planar" ? 2 : 1;
        for (int64_t outputHemisphere = firstOutputHemisphere;
             outputHemisphere
                < firstOutputHemisphere + outputHemispheres;
             ++outputHemisphere) {
            for (int64_t byte = 0; byte < outputSliceCount; ++byte) {
                const int64_t slice =
                    resultKind == "fp16_pair_planar"
                    ? result->slices[byte] : resultSlices[byte];
                emitMem(rewriter, op.getLoc(),
                    vxmCycle + 1 + eastLatency(slice),
                    outputHemisphere
                            * target.memory().slices_per_hemisphere
                        + slice,
                    "write", result->row, byte, rows, 1, 1);
            }
        }
        cycle += rows + 2
            + eastLatency(resultSlices[outputSliceCount - 1]);
    }
    createTimeline(rewriter, op, start, cycle);
    auto output =
        createOutputBinding(rewriter, op, outputIndex, resultPlacement);
    rewriter.replaceOp(op, output.getValue());
    return mlir::success();
}

} // namespace

mlir::LogicalResult lowerElementwiseSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::ElementwiseTaskOp> operations;
    function.walk(
        [&](stream::ElementwiseTaskOp op) { operations.push_back(op); });
    int64_t outputIndex = 0;
    for (stream::ElementwiseTaskOp op : operations) {
        if (mlir::failed(
                lowerElementwise(rewriter, op, target, outputIndex++)))
            return mlir::failure();
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
