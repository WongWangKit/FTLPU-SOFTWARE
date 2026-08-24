#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
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
    int64_t bank;
};

int64_t placementBank(mlir::DictionaryAttr placement)
{
    if (const auto bank = placement.getAs<mlir::IntegerAttr>("bank"))
        return bank.getInt();
    return 0;
}

bool isDistributed16(mlir::DictionaryAttr placement)
{
    const auto kind = placement.getAs<mlir::StringAttr>("kind");
    return kind
        && (kind.getValue() == "fp16_mxm_distributed_16"
            || kind.getValue()
                == "fp16_mxm_block8_distributed_16");
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
        hemisphere, placementBank(placement)};
}

mlir::FailureOr<TileAddress> tileAddress(
    mlir::DictionaryAttr placement, int64_t block, int64_t rows,
    int64_t preferredHemisphere,
    const target::LPUTargetModel& target)
{
    const auto kind = placement.getAs<mlir::StringAttr>("kind");
    const auto slices = placementSlices(placement);
    const int64_t base =
        placement.getAs<mlir::IntegerAttr>("base_row").getInt();
    if (!kind || slices.size() < 2) return mlir::failure();
    if (kind.getValue() == "fp16_mxm_activation_planar")
        return TileAddress {{slices[0], slices[1]},
            base + block * rows, preferredHemisphere,
            placementBank(placement)};
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
            dual ? (block / 2) % 2 : 0, placementBank(placement)};
    }
    if (isDistributed16(placement)) {
        if (kind.getValue()
            == "fp16_mxm_block8_distributed_16")
            preferredHemisphere =
                (block / target.throughput().mxms_per_hemisphere)
                % target.memory().hemispheres;
        return distributedAddress(
            placement, block, 0, 1, preferredHemisphere);
    }
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
    const auto streamKind =
        lpu_16bit_stream_kind(type.getElementType());
    const auto dataFormat =
        lpu_16bit_data_format(type.getElementType());
    if (op.getKind() != "add" || type.getRank() != 2
        || !is_lpu_16bit_float(type.getElementType())
        || rows % tile != 0
        || columns % tile != 0) {
        return op.emitError(
            "elementwise schedule currently supports tile-aligned "
            "16-bit float add");
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
    const auto layoutKind = [](mlir::DictionaryAttr placement) {
        const auto kind = placement.getAs<mlir::StringAttr>("kind");
        return kind ? kind.getValue() : llvm::StringRef("<missing>");
    };
    if ((!resultDistributed && resultSlices.size() < 4)
        || (resultDistributed && resultSlices.size() != 16)
        || (resultKind != "fp16_pair_planar"
            && resultKind != "fp16_mxm_activation_planar"
            && resultKind != "fp16_mxm_distributed_16"
            && resultKind
                != "fp16_mxm_block8_distributed_16"))
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
        auto probeLhs = tileAddress(
            lhsPlacement, block, rows, 0, target);
        auto probeRhs = tileAddress(
            rhsPlacement, block, rows, 0, target);
        if (mlir::failed(probeLhs) || mlir::failed(probeRhs))
            return op.emitError("unsupported elementwise operand layout")
                << " during hemisphere probe: lhs="
                << layoutKind(lhsPlacement) << ", rhs="
                << layoutKind(rhsPlacement);
        const int64_t hemisphere = std::max(
            probeLhs->hemisphere, probeRhs->hemisphere);
        auto lhs = tileAddress(
            lhsPlacement, block, rows, hemisphere, target);
        auto rhs = tileAddress(
            rhsPlacement, block, rows, hemisphere, target);
        auto result = tileAddress(
            resultPlacement, block, rows, hemisphere, target);
        if (mlir::failed(lhs) || mlir::failed(rhs)
            || mlir::failed(result))
            return op.emitError("unsupported elementwise operand layout")
                << " during tile addressing: lhs="
                << layoutKind(lhsPlacement) << ", rhs="
                << layoutKind(rhsPlacement) << ", result="
                << layoutKind(resultPlacement) << ", block=" << block
                << ", hemisphere=" << hemisphere;

        const int64_t vxmCycle = cycle;
        const auto emitOperand = [&](mlir::DictionaryAttr placement,
                                     const TileAddress& address,
                                     int64_t inputCycle,
                                     int64_t streamBase,
                                     int64_t sourceHemisphere) {
            if (isDistributed16(placement)) {
                for (int64_t row = 0; row < rows; ++row) {
                    auto distributed = distributedAddress(placement,
                        block, row, columnBlocks, sourceHemisphere);
                    if (mlir::failed(distributed)) return false;
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        emitMem(rewriter, op.getLoc(),
                            inputCycle + row
                                - westLatency(
                                    distributed->slices[byte]),
                            distributed->hemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + distributed->slices[byte],
                            "read", distributed->row,
                            streamBase + byte, 1, 1, 1,
                            "sram", -1, distributed->bank);
                    }
                }
                return true;
            }
            for (int64_t byte = 0; byte < 2; ++byte) {
                emitMem(rewriter, op.getLoc(),
                    inputCycle - westLatency(address.slices[byte]),
                    sourceHemisphere
                            * target.memory().slices_per_hemisphere
                        + address.slices[byte],
                    "read", address.row, streamBase + byte,
                    rows, 1, 1, "sram", -1, address.bank);
            }
            return true;
        };
        for (int64_t sourceHemisphere = 0;
             sourceHemisphere < target.memory().hemispheres;
             ++sourceHemisphere) {
            const int64_t mirroredStreamOffset =
                sourceHemisphere * 16;
            if (!emitOperand(lhsPlacement, *lhs, vxmCycle,
                    32 + mirroredStreamOffset, sourceHemisphere)
                || !emitOperand(rhsPlacement, *rhs, vxmCycle,
                    34 + mirroredStreamOffset, sourceHemisphere))
                return op.emitError(
                    "invalid distributed elementwise address");
        }
        auto sum = create_vxm(rewriter, op.getLoc(),
            op.getLhs(), op.getRhs(), type, vxmCycle - 1, 0, "add",
            streamKind, 32, 0.0f, streamKind, 34, 0.0f,
            "fp32", -1, rows, 1,
            hemisphere == 0 ? "east" : "west",
            hemisphere == 0 ? "east" : "west",
            -1, false, false, true, false, 2);
        auto cast = create_vxm(rewriter, op.getLoc(),
            sum.getResult(), sum.getResult(), type, vxmCycle - 1, 1,
            "cast", "alu", 0, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 0, rows, 1,
            hemisphere == 0 ? "east" : "west",
            hemisphere == 0 ? "east" : "west",
            -1, false, false, true, false, 2);
        finalValue = cast.getResult();

        int64_t outputSliceCount = 2;
        int64_t secondOutputCycle = -1;
        if (resultKind == "fp16_mxm_activation_planar") {
            const int64_t secondVxmCycle = vxmCycle + rows + 2;
            for (int64_t sourceHemisphere = 0;
                 sourceHemisphere < target.memory().hemispheres;
                 ++sourceHemisphere) {
                const int64_t mirroredStreamOffset =
                    sourceHemisphere * 16;
                if (!emitOperand(lhsPlacement, *lhs, secondVxmCycle,
                        36 + mirroredStreamOffset, sourceHemisphere)
                    || !emitOperand(rhsPlacement, *rhs, secondVxmCycle,
                        38 + mirroredStreamOffset, sourceHemisphere))
                    return op.emitError(
                        "invalid second planar elementwise input pass");
            }
            auto secondSum = create_vxm(rewriter, op.getLoc(),
                op.getLhs(), op.getRhs(), type,
                secondVxmCycle - 1, 2, "add",
                streamKind, 36, 0.0f, streamKind, 38, 0.0f,
                "fp32", -1, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 2);
            create_vxm(rewriter, op.getLoc(),
                secondSum.getResult(), secondSum.getResult(), type,
                secondVxmCycle - 1, 3, "cast", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, dataFormat, 2, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 2);
            outputSliceCount = 4;
            secondOutputCycle = secondVxmCycle + 1;
        }
        if (resultDistributed) {
            for (int64_t outputHemisphere = 0;
                 outputHemisphere < target.memory().hemispheres;
                 ++outputHemisphere) {
                const int64_t streamBase =
                    outputHemisphere == 0 ? 8 : 0;
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
                            streamBase + byte, 1, 1, 1,
                            "sram", -1, output->bank);
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
                const int64_t pairCycle = byte < 2
                    ? vxmCycle + 1 : secondOutputCycle;
                const int64_t physicalStreamBase =
                    outputHemisphere == 0 ? 8 : 0;
                emitMem(rewriter, op.getLoc(),
                    pairCycle + eastLatency(slice),
                    outputHemisphere
                            * target.memory().slices_per_hemisphere
                        + slice,
                    "write", result->row,
                    physicalStreamBase + byte, rows, 1, 1,
                    "sram", -1, result->bank);
            }
        }
        cycle = std::max(cycle,
            (secondOutputCycle >= 0
                 ? secondOutputCycle + rows + 1
                 : vxmCycle + rows + 2)
            + eastLatency(resultSlices[outputSliceCount - 1]));
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
