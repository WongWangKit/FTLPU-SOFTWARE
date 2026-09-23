#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
#include <array>

namespace ftlpu::compiler::schedule {
namespace {

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

bool sharesMemQueue(mlir::DictionaryAttr lhs,
    mlir::DictionaryAttr rhs)
{
    if (placementBank(lhs) != placementBank(rhs))
        return false;
    const auto lhsSlices = placementSlices(lhs);
    const auto rhsSlices = placementSlices(rhs);
    return std::any_of(lhsSlices.begin(), lhsSlices.end(),
        [&](int64_t lhsSlice) {
            return std::find(rhsSlices.begin(), rhsSlices.end(), lhsSlice)
                != rhsSlices.end();
        });
}

bool usesMemQueue(mlir::DictionaryAttr placement,
    int64_t bank, int64_t slice)
{
    if (placementBank(placement) != bank)
        return false;
    const auto slices = placementSlices(placement);
    return std::find(slices.begin(), slices.end(), slice) != slices.end();
}

void emitMemColumnDomain(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    int64_t cycle, int64_t queue, llvm::StringRef opcode,
    int64_t address, int64_t packedStream,
    int64_t repeatCount, int64_t repeatInterval, int64_t addressStride,
    int64_t waveCount, int64_t waveInterval, int64_t waveAddressStride,
    int64_t columnCount, int64_t columnInterval,
    int64_t columnAddressStride, int64_t bank, bool splitColumns)
{
    if (!splitColumns || columnCount == 1) {
        direct_domain_detail::emitMem3D(rewriter, location, target,
            cycle, queue, opcode, address, packedStream,
            repeatCount, repeatInterval, addressStride,
            waveCount, waveInterval, waveAddressStride,
            columnCount, columnInterval, columnAddressStride, -1, bank);
        return;
    }
    // This loop is the operator's direct lowering for queues that must
    // alternate READ_3D and WRITE_3D.  Each emitted command is already a
    // closed hardware domain; no later pass splits or moves it.
    for (int64_t column = 0; column < columnCount; ++column) {
        direct_domain_detail::emitMem3D(rewriter, location, target,
            cycle + column * columnInterval, queue, opcode,
            address + column * columnAddressStride, packedStream,
            repeatCount, repeatInterval, addressStride,
            waveCount, waveInterval, waveAddressStride,
            1, 1, 0, -1, bank);
    }
}

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
    std::string bindingName = "elementwise.";
    bindingName.append(op.getKind());
    bindingName.push_back('.');
    bindingName.append(std::to_string(index));
    mlir::OperationState state(op.getLoc(), BindingOp::getOperationName());
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr("access", rewriter.getStringAttr("output")),
        rewriter.getNamedAttr("role", rewriter.getStringAttr("result")),
        rewriter.getNamedAttr(
            "name", rewriter.getStringAttr(bindingName)),
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
    const int64_t logicalRows = type.getDimSize(0);
    const int64_t columns = type.getDimSize(1);
    const int64_t tile = target.throughput().mxm_rows;
    // One decode token occupies one SRAM vector.  Its four tile segments are
    // presented to VXM on four cycles while the SRAM row address stays fixed.
    // Treating it as a 32-token MXM tile walks into the following column
    // blocks of pair-planar storage after the first 64 features.
    const bool singleRowDecode = logicalRows == 1;
    const int64_t rows = singleRowDecode
        ? target.throughput().tile_rows : logicalRows;
    const auto streamKind =
        lpu_16bit_stream_kind(type.getElementType());
    const auto dataFormat =
        lpu_16bit_data_format(type.getElementType());
    if (op.getKind() != "add" || type.getRank() != 2
        || !is_lpu_16bit_float(type.getElementType())
        || (logicalRows != 1 && logicalRows % tile != 0)
        || columns % tile != 0) {
        return op.emitError(
            "elementwise schedule currently supports tile-aligned "
            "16-bit float add");
    }

    const auto lhsPlacement =
        allocationPlacement(op.getLhsAllocations(),
            op.getLhsAllocations().size() > 1
                ? op.getLhsAllocations().size() - 1 : 0);
    const auto rhsPlacement =
        allocationPlacement(op.getRhsAllocations(),
            op.getRhsAllocations().size() > 1
                ? op.getRhsAllocations().size() - 1 : 0);
    const auto resultPlacement =
        allocationPlacement(op.getResultAllocations(), 0);
    const auto persistentResultPlacement =
        op.getResultAllocations().size() > 1
        ? allocationPlacement(op.getResultAllocations(), 1)
        : mlir::DictionaryAttr {};
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
    const auto bridgeWriteLatency = [&](int64_t slice) {
        return target.transport_latency(
            target::StreamEndpoint::VxmBridgeResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice).value();
    };
    const auto passiveReadLatency = [&](int64_t slice) {
        return westLatency(slice) + 1;
    };

    rewriter.setInsertionPoint(op);
    int64_t cycle = 10;
    const int64_t start = cycle;
    mlir::Value finalValue = op.getLhs();
    const int64_t columnBlocks = columns / tile;
    // Every supported layout repeats its physical command body after four
    // column blocks.  pair_planar uses block%2 for the slice pair and
    // (block/2)%2 for the hemisphere; distributed16 and activation_planar
    // have an affine address at that same period.  Compute the four physical
    // body durations once, then make column-block progression an ICU domain
    // instead of materializing one copy of every command per model block.
    // Run residue domains consecutively: a physical queue completes every
    // group of one coarse command before the next residue starts.
    constexpr int64_t kColumnBlockResidues = 4;
    // A distributed result can reuse an operand's physical bank/slice queue.
    // In that case a token-major READ_3D stays live across the interleaved
    // result WRITE_3D launches, and a column domain would keep both contexts
    // live across every block as well.  Visit the four rows owned by each
    // slice pair consecutively and retire both MEM descriptors inside one
    // column block.  Only those aliasing queue domains are emitted once per
    // block; non-alias MEM queues and the VXM keep the 12-column domain.  The
    // VXM still consumes one element per cycle, and all operand/result
    // addresses use the same pair-major permutation.
    const bool pairMajorMemOrder = resultDistributed
        && (rows == tile || singleRowDecode)
        && (sharesMemQueue(lhsPlacement, resultPlacement)
            || sharesMemQueue(rhsPlacement, resultPlacement));
    const bool closedColumnDomain = resultDistributed
        && (rows == tile || singleRowDecode);
    // Depth four adds two pass stages after the add.  This lets an aliased
    // READ_3D retire before the corresponding result WRITE_3D begins without
    // presenting input data before the VXM has configured its Bundle.
    const int64_t outputPipelineLatency = pairMajorMemOrder ? 3 : 1;
    const int64_t emittedColumnBlocks = closedColumnDomain
        ? std::min<int64_t>(columnBlocks, kColumnBlockResidues)
        : columnBlocks;
    struct ResidueDuration {
        int64_t active = 0;
        int64_t bridge = 0;
    };
    std::array<ResidueDuration, kColumnBlockResidues> residueDurations {};
    const bool operandsMirrored = isDistributed16(lhsPlacement)
        && isDistributed16(rhsPlacement);
    if (closedColumnDomain) {
        const auto blockDuration = [&](int64_t block)
            -> mlir::FailureOr<ResidueDuration> {
            auto probeLhs = tileAddress(
                lhsPlacement, block, logicalRows, 0, target);
            auto probeRhs = tileAddress(
                rhsPlacement, block, logicalRows, 0, target);
            if (mlir::failed(probeLhs) || mlir::failed(probeRhs))
                return mlir::failure();
            const int64_t hemisphere = std::max(
                probeLhs->hemisphere, probeRhs->hemisphere);
            auto result = tileAddress(
                resultPlacement, block, logicalRows, hemisphere, target);
            if (mlir::failed(result)) return mlir::failure();

            const int64_t validOutputHemisphere = 1 - hemisphere;
            const int64_t mirrorOutputHemisphere = hemisphere;
            auto persistent = persistentResultPlacement
                ? tileAddress(persistentResultPlacement, block, logicalRows,
                      validOutputHemisphere, target)
                : mlir::FailureOr<TileAddress>(mlir::failure());
            if (persistentResultPlacement && mlir::failed(persistent))
                return mlir::failure();
            const bool persistentOnValidOutput =
                persistentResultPlacement
                && persistent->hemisphere == validOutputHemisphere;
            const bool persistentOnMirrorOutput =
                persistentResultPlacement
                && persistent->hemisphere == mirrorOutputHemisphere;
            if (persistentResultPlacement
                && !persistentOnValidOutput
                && !persistentOnMirrorOutput)
                return mlir::failure();

            int64_t activeWriteEnd = 0;
            const int64_t tokenBlocks = singleRowDecode ? 1 : rows / tile;
            const int64_t tileRows = target.throughput().tile_rows;
            const int64_t lanes = target.throughput().lanes_per_tile;
            const int64_t physicalPairs = singleRowDecode ? 1 : lanes;
            for (int64_t outputHemisphere = 0;
                 outputHemisphere < target.memory().hemispheres;
                 ++outputHemisphere) {
                const bool preserveValidOutput = persistentOnValidOutput
                    && outputHemisphere == validOutputHemisphere;
                for (int64_t pair = 0; pair < physicalPairs; ++pair) {
                    auto output = distributedAddress(resultPlacement,
                        block, pair, columnBlocks, outputHemisphere);
                    if (mlir::failed(output)) return mlir::failure();
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t pairCycle = pairMajorMemOrder
                            ? pair * tileRows : pair;
                        const int64_t writeCycle = outputPipelineLatency
                            + pairCycle
                            + eastLatency(output->slices[byte]);
                        activeWriteEnd = std::max(activeWriteEnd,
                            writeCycle + (tileRows - 1)
                                    * (singleRowDecode ? 1
                                        : (pairMajorMemOrder ? 1 : lanes))
                                + (tokenBlocks - 1) * tile + 1);
                    }
                }
                if (preserveValidOutput) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t persistentWriteCycle =
                            outputPipelineLatency
                            + eastLatency(persistent->slices[byte]);
                        activeWriteEnd = std::max(activeWriteEnd,
                            persistentWriteCycle + rows);
                    }
                }
            }

            const int64_t activeDuration = activeWriteEnd + 1;
            if (operandsMirrored)
                return ResidueDuration {activeDuration, 0};

            int64_t maximumReadLatency = 0;
            for (int64_t slice : resultSlices)
                maximumReadLatency = std::max(
                    maximumReadLatency, passiveReadLatency(slice));
            const int64_t bridgeStart = activeDuration;
            const int64_t bridgeCycle =
                bridgeStart + maximumReadLatency;
            int64_t bridgeEnd = bridgeCycle;
            for (int64_t pair = 0; pair < physicalPairs; ++pair) {
                auto mirror = distributedAddress(resultPlacement,
                    block, pair, columnBlocks, mirrorOutputHemisphere);
                if (mlir::failed(mirror)) return mlir::failure();
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t mirrorWriteCycle = bridgeCycle + pair
                        + bridgeWriteLatency(mirror->slices[byte]);
                    bridgeEnd = std::max(bridgeEnd,
                        mirrorWriteCycle + (tileRows - 1)
                                * (singleRowDecode ? 1 : lanes)
                            + (tokenBlocks - 1) * tile + 1);
                }
            }
            if (persistentOnMirrorOutput) {
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t persistentWriteCycle = bridgeCycle
                        + bridgeWriteLatency(persistent->slices[byte]);
                    bridgeEnd = std::max(
                        bridgeEnd, persistentWriteCycle + rows);
                }
            }
            return ResidueDuration {
                activeDuration, bridgeEnd + 1 - bridgeStart};
        };

        for (int64_t residue = 0; residue < emittedColumnBlocks;
             ++residue) {
            auto duration = blockDuration(residue);
            if (mlir::failed(duration))
                return op.emitError(
                    "cannot form closed elementwise column domain");
            residueDurations[static_cast<std::size_t>(residue)] = *duration;
        }
    }

    for (int64_t block = 0; block < emittedColumnBlocks; ++block) {
        const int64_t columnDomainCount = closedColumnDomain
            ? 1 + (columnBlocks - 1 - block) / kColumnBlockResidues
            : 1;
        const ResidueDuration residueDuration = closedColumnDomain
            ? residueDurations[static_cast<std::size_t>(block)]
            : ResidueDuration {1, 1};
        const int64_t activeColumnInterval = closedColumnDomain
            ? residueDuration.active : 1;
        const int64_t bridgeColumnInterval = closedColumnDomain
            ? residueDuration.bridge : 1;
        auto probeLhs = tileAddress(
            lhsPlacement, block, logicalRows, 0, target);
        auto probeRhs = tileAddress(
            rhsPlacement, block, logicalRows, 0, target);
        if (mlir::failed(probeLhs) || mlir::failed(probeRhs))
            return op.emitError("unsupported elementwise operand layout")
                << " during hemisphere probe: lhs="
                << layoutKind(lhsPlacement) << ", rhs="
                << layoutKind(rhsPlacement);
        const int64_t hemisphere = std::max(
            probeLhs->hemisphere, probeRhs->hemisphere);
        auto lhs = tileAddress(
            lhsPlacement, block, logicalRows, hemisphere, target);
        auto rhs = tileAddress(
            rhsPlacement, block, logicalRows, hemisphere, target);
        auto result = tileAddress(
            resultPlacement, block, logicalRows, hemisphere, target);
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
            const int64_t tileRows = target.throughput().tile_rows;
            const int64_t lanes = target.throughput().lanes_per_tile;
            if (isDistributed16(placement)) {
                const int64_t tokenBlocks = singleRowDecode ? 1 : rows / tile;
                const int64_t physicalPairs = singleRowDecode ? 1 : lanes;
                for (int64_t pair = 0; pair < physicalPairs; ++pair) {
                    auto distributed = distributedAddress(placement,
                        block, pair, columnBlocks, sourceHemisphere);
                    if (mlir::failed(distributed)) return false;
                    int64_t columnAddressStride = 0;
                    if (columnDomainCount > 1) {
                        auto next = distributedAddress(placement,
                            block + kColumnBlockResidues, pair,
                            columnBlocks, sourceHemisphere);
                        if (mlir::failed(next)) return false;
                        columnAddressStride =
                            next->row - distributed->row;
                    }
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t pairCycle = pairMajorMemOrder
                            ? pair * tileRows : pair;
                        const int64_t slice = distributed->slices[byte];
                        const bool splitColumnDomain = pairMajorMemOrder
                            && usesMemQueue(resultPlacement,
                                distributed->bank, slice);
                        emitMemColumnDomain(
                            rewriter, op.getLoc(), target,
                            inputCycle + pairCycle
                                - westLatency(slice),
                            distributed->hemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + slice,
                            "read", distributed->row,
                            streamBase + byte,
                            tileRows,
                            singleRowDecode ? 1
                                : (pairMajorMemOrder ? 1 : lanes),
                            singleRowDecode ? 0 : 1,
                            tokenBlocks, tile, columnBlocks * tileRows,
                            columnDomainCount, activeColumnInterval,
                            columnAddressStride, distributed->bank,
                            splitColumnDomain);
                    }
                }
                return true;
            }
            int64_t columnAddressStride = 0;
            if (columnDomainCount > 1) {
                auto next = tileAddress(placement,
                    block + kColumnBlockResidues, logicalRows,
                    sourceHemisphere, target);
                if (mlir::failed(next)) return false;
                columnAddressStride = next->row - address.row;
            }
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = address.slices[byte];
                const bool splitColumnDomain = pairMajorMemOrder
                    && usesMemQueue(
                        resultPlacement, address.bank, slice);
                if (pairMajorMemOrder) {
                    emitMemColumnDomain(
                        rewriter, op.getLoc(), target,
                        inputCycle - westLatency(slice),
                        sourceHemisphere
                                * target.memory().slices_per_hemisphere
                            + slice,
                        "read", address.row, streamBase + byte,
                        tileRows, 1,
                        singleRowDecode ? 0 : lanes,
                        singleRowDecode ? 1 : lanes, tileRows, 1,
                        columnDomainCount, activeColumnInterval,
                        columnAddressStride,
                        address.bank, splitColumnDomain);
                } else {
                    emitMemColumnDomain(
                        rewriter, op.getLoc(), target,
                        inputCycle - westLatency(slice),
                        sourceHemisphere
                                * target.memory().slices_per_hemisphere
                            + slice,
                        "read", address.row, streamBase + byte,
                        rows, 1, singleRowDecode ? 0 : 1,
                        1, 1, 0,
                        columnDomainCount, activeColumnInterval,
                        columnAddressStride,
                        address.bank, splitColumnDomain);
                }
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
            -1, false, false, true, false,
            pairMajorMemOrder ? 4 : 2);
        const auto setColumnDomain = [&](VxmOp instruction) {
            if (columnDomainCount <= 1)
                return;
            instruction->setAttr("wave_count",
                rewriter.getI64IntegerAttr(columnDomainCount));
            instruction->setAttr("wave_interval",
                rewriter.getI64IntegerAttr(activeColumnInterval));
        };
        setColumnDomain(sum);
        if (pairMajorMemOrder) {
            auto pass1 = create_vxm(rewriter, op.getLoc(),
                sum.getResult(), sum.getResult(), type,
                vxmCycle - 1, 1, "pass",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 4);
            auto pass2 = create_vxm(rewriter, op.getLoc(),
                pass1.getResult(), pass1.getResult(), type,
                vxmCycle - 1, 2, "pass",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 4);
            auto cast = create_vxm(rewriter, op.getLoc(),
                pass2.getResult(), pass2.getResult(), type,
                vxmCycle - 1, 3, "cast",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                dataFormat, 2, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 4);
            setColumnDomain(pass1);
            setColumnDomain(pass2);
            setColumnDomain(cast);
            finalValue = cast.getResult();
        } else {
            auto cast = create_vxm(rewriter, op.getLoc(),
                sum.getResult(), sum.getResult(), type,
                vxmCycle - 1, 1, "cast",
                "alu", 0, 0.0f, "immediate", 0, 0.0f,
                dataFormat, 0, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 2);
            setColumnDomain(cast);
            finalValue = cast.getResult();
        }

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
            if (columnDomainCount > 1) {
                secondSum->setAttr("wave_count",
                    rewriter.getI64IntegerAttr(columnDomainCount));
                secondSum->setAttr("wave_interval",
                    rewriter.getI64IntegerAttr(activeColumnInterval));
            }
            auto secondCast = create_vxm(rewriter, op.getLoc(),
                secondSum.getResult(), secondSum.getResult(), type,
                secondVxmCycle - 1, 3, "cast", "previous", 0, 0.0f,
                "immediate", 0, 0.0f, dataFormat, 2, rows, 1,
                hemisphere == 0 ? "east" : "west",
                hemisphere == 0 ? "east" : "west",
                -1, false, false, true, false, 2);
            if (columnDomainCount > 1) {
                secondCast->setAttr("wave_count",
                    rewriter.getI64IntegerAttr(columnDomainCount));
                secondCast->setAttr("wave_interval",
                    rewriter.getI64IntegerAttr(activeColumnInterval));
            }
            outputSliceCount = 4;
            secondOutputCycle = secondVxmCycle + 1;
        }
        if (resultDistributed) {
            const int64_t validOutputHemisphere = 1 - hemisphere;
            const int64_t mirrorOutputHemisphere = hemisphere;
            auto persistent = persistentResultPlacement
                ? tileAddress(persistentResultPlacement, block, logicalRows,
                      validOutputHemisphere, target)
                : mlir::FailureOr<TileAddress>(mlir::failure());
            if (persistentResultPlacement && mlir::failed(persistent))
                return op.emitError(
                    "invalid persistent elementwise result layout");
            const bool persistentOnValidOutput =
                persistentResultPlacement
                && persistent->hemisphere == validOutputHemisphere;
            const bool persistentOnMirrorOutput =
                persistentResultPlacement
                && persistent->hemisphere == mirrorOutputHemisphere;
            if (persistentResultPlacement
                && !persistentOnValidOutput
                && !persistentOnMirrorOutput)
                return op.emitError(
                    "persistent elementwise result hemisphere is not "
                    "reachable from either VXM output");
            int64_t persistentColumnAddressStride = 0;
            if (persistentResultPlacement && columnDomainCount > 1) {
                auto nextPersistent = tileAddress(persistentResultPlacement,
                    block + kColumnBlockResidues, logicalRows,
                    validOutputHemisphere, target);
                if (mlir::failed(nextPersistent))
                    return op.emitError(
                        "invalid persistent elementwise column domain");
                persistentColumnAddressStride =
                    nextPersistent->row - persistent->row;
            }
            int64_t activeWriteEnd = vxmCycle;
            const int64_t tokenBlocks = singleRowDecode ? 1 : rows / tile;
            const int64_t tileRows = target.throughput().tile_rows;
            const int64_t lanes = target.throughput().lanes_per_tile;
            const int64_t physicalPairs = singleRowDecode ? 1 : lanes;
            for (int64_t outputHemisphere = 0;
                 outputHemisphere < target.memory().hemispheres;
                 ++outputHemisphere) {
                const int64_t fixedOutputStream =
                    pairMajorMemOrder ? 2 : 0;
                const int64_t streamBase =
                    (outputHemisphere == 0 ? 8 : 0)
                    + fixedOutputStream;
                const bool preserveValidOutput =
                    persistentOnValidOutput
                    && outputHemisphere == validOutputHemisphere;
                for (int64_t pair = 0; pair < physicalPairs; ++pair) {
                    auto output = distributedAddress(resultPlacement,
                        block, pair, columnBlocks, outputHemisphere);
                    if (mlir::failed(output))
                        return op.emitError(
                            "invalid distributed elementwise result");
                    int64_t outputColumnAddressStride = 0;
                    if (columnDomainCount > 1) {
                        auto nextOutput = distributedAddress(resultPlacement,
                            block + kColumnBlockResidues, pair,
                            columnBlocks, outputHemisphere);
                        if (mlir::failed(nextOutput))
                            return op.emitError(
                                "invalid distributed elementwise column domain");
                        outputColumnAddressStride =
                            nextOutput->row - output->row;
                    }
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t pairCycle = pairMajorMemOrder
                            ? pair * tileRows : pair;
                        const int64_t slice = output->slices[byte];
                        const int64_t writeCycle = vxmCycle
                            + outputPipelineLatency + pairCycle
                            + eastLatency(slice);
                        const bool splitColumnDomain = pairMajorMemOrder
                            && (usesMemQueue(
                                    lhsPlacement, output->bank, slice)
                                || usesMemQueue(
                                    rhsPlacement, output->bank, slice));
                        emitMemColumnDomain(
                            rewriter, op.getLoc(), target,
                            writeCycle,
                            outputHemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + slice,
                            preserveValidOutput ? "write_tap" : "write",
                            output->row,
                            streamBase + byte,
                            tileRows,
                            singleRowDecode ? 1
                                : (pairMajorMemOrder ? 1 : lanes),
                            singleRowDecode ? 0 : 1,
                            tokenBlocks, tile, columnBlocks * tileRows,
                            columnDomainCount, activeColumnInterval,
                            outputColumnAddressStride, output->bank,
                            splitColumnDomain);
                        activeWriteEnd = std::max(activeWriteEnd,
                            writeCycle + (tileRows - 1)
                                    * (singleRowDecode ? 1
                                        : (pairMajorMemOrder ? 1 : lanes))
                                + (tokenBlocks - 1) * tile + 1);
                    }
                }
                if (preserveValidOutput) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = persistent->slices[byte];
                        const int64_t persistentWriteCycle =
                            vxmCycle + outputPipelineLatency
                            + eastLatency(slice);
                        const bool splitColumnDomain = pairMajorMemOrder
                            && (usesMemQueue(lhsPlacement,
                                    persistent->bank, slice)
                                || usesMemQueue(rhsPlacement,
                                    persistent->bank, slice));
                        emitMemColumnDomain(
                            rewriter, op.getLoc(), target,
                            persistentWriteCycle,
                            validOutputHemisphere
                                    * target.memory().slices_per_hemisphere
                                + slice,
                            "write", persistent->row,
                            streamBase + byte,
                            pairMajorMemOrder ? tileRows : rows,
                            1, singleRowDecode ? 0
                                    : (pairMajorMemOrder ? lanes : 1),
                            pairMajorMemOrder
                                ? (singleRowDecode ? 1 : lanes) : 1,
                            pairMajorMemOrder ? tileRows : 1,
                            pairMajorMemOrder ? 1 : 0,
                            columnDomainCount, activeColumnInterval,
                            persistentColumnAddressStride,
                            persistent->bank, splitColumnDomain);
                        activeWriteEnd = std::max(activeWriteEnd,
                            persistentWriteCycle + rows);
                    }
                }
            }

            const int64_t oneActiveEnd = activeWriteEnd + 1;
            const int64_t bridgePhaseStart = closedColumnDomain
                ? vxmCycle + columnDomainCount * residueDuration.active
                : activeWriteEnd;
            if (!operandsMirrored) {
                // A logical VXM instruction drives both physical chains. If
                // one operand is hemisphere-local, only that hemisphere's
                // chain produces a valid result. Bridge it to the other side
                // so distributed16 remains physically mirrored.  For a
                // closed column domain, finish every active-result group
                // before launching the bridge domain; the two write bodies
                // can share a physical MEM queue and one ICU context cannot
                // interleave them.
                constexpr int64_t bridgeStream = 20;
                int64_t maximumReadLatency = 0;
                for (int64_t slice : resultSlices)
                    maximumReadLatency = std::max(
                        maximumReadLatency, passiveReadLatency(slice));
                const int64_t bridgeCycle =
                    bridgePhaseStart + maximumReadLatency;
                int64_t bridgeEnd = bridgeCycle;
                for (int64_t pair = 0; pair < physicalPairs; ++pair) {
                    auto source = distributedAddress(resultPlacement,
                        block, pair, columnBlocks,
                        validOutputHemisphere);
                    auto mirror = distributedAddress(resultPlacement,
                        block, pair, columnBlocks,
                        mirrorOutputHemisphere);
                    if (mlir::failed(source) || mlir::failed(mirror))
                        return op.emitError(
                            "invalid distributed elementwise mirror");
                    int64_t sourceColumnAddressStride = 0;
                    int64_t mirrorColumnAddressStride = 0;
                    if (columnDomainCount > 1) {
                        auto nextSource = distributedAddress(resultPlacement,
                            block + kColumnBlockResidues, pair,
                            columnBlocks, validOutputHemisphere);
                        auto nextMirror = distributedAddress(resultPlacement,
                            block + kColumnBlockResidues, pair,
                            columnBlocks, mirrorOutputHemisphere);
                        if (mlir::failed(nextSource)
                            || mlir::failed(nextMirror))
                            return op.emitError(
                                "invalid distributed elementwise mirror domain");
                        sourceColumnAddressStride =
                            nextSource->row - source->row;
                        mirrorColumnAddressStride =
                            nextMirror->row - mirror->row;
                    }
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t readCycle = bridgeCycle + pair
                                - passiveReadLatency(
                                    source->slices[byte]);
                        direct_domain_detail::emitMem3D(
                            rewriter, op.getLoc(), target, readCycle,
                            validOutputHemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + source->slices[byte],
                            "read", source->row,
                            32 + bridgeStream + byte,
                            tileRows, singleRowDecode ? 1 : lanes,
                            singleRowDecode ? 0 : 1,
                            tokenBlocks, tile, columnBlocks * tileRows,
                            columnDomainCount, bridgeColumnInterval,
                            sourceColumnAddressStride, -1, source->bank);
                        const int64_t mirrorWriteCycle = bridgeCycle + pair
                            + bridgeWriteLatency(mirror->slices[byte]);
                        direct_domain_detail::emitMem3D(
                            rewriter, op.getLoc(), target, mirrorWriteCycle,
                            mirrorOutputHemisphere
                                    * target.memory()
                                          .slices_per_hemisphere
                                + mirror->slices[byte],
                            persistentOnMirrorOutput
                                ? "write_tap" : "write",
                            mirror->row,
                            bridgeStream + byte,
                            tileRows, singleRowDecode ? 1 : lanes,
                            singleRowDecode ? 0 : 1,
                            tokenBlocks, tile, columnBlocks * tileRows,
                            columnDomainCount, bridgeColumnInterval,
                            mirrorColumnAddressStride, -1, mirror->bank);
                        bridgeEnd = std::max(
                            bridgeEnd,
                            mirrorWriteCycle + (tileRows - 1)
                                    * (singleRowDecode ? 1 : lanes)
                                + (tokenBlocks - 1) * tile + 1);
                    }
                }
                if (persistentOnMirrorOutput) {
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t persistentWriteCycle = bridgeCycle
                            + bridgeWriteLatency(persistent->slices[byte]);
                        direct_domain_detail::emitMem3D(
                            rewriter, op.getLoc(), target,
                            persistentWriteCycle,
                            mirrorOutputHemisphere
                                    * target.memory().slices_per_hemisphere
                                + persistent->slices[byte],
                            "write", persistent->row,
                            bridgeStream + byte, rows, 1, 1,
                            1, 1, 0,
                            columnDomainCount, bridgeColumnInterval,
                            persistentColumnAddressStride, -1,
                            persistent->bank);
                        bridgeEnd = std::max(
                            bridgeEnd, persistentWriteCycle + rows);
                    }
                }
                cycle = bridgeEnd + 1;
            } else {
                cycle = oneActiveEnd;
            }
            if (closedColumnDomain) {
                if (oneActiveEnd
                    != vxmCycle + residueDuration.active)
                    return op.emitError(
                        "elementwise active residue duration does not match emitted body");
                if (!operandsMirrored
                    && cycle
                        != bridgePhaseStart + residueDuration.bridge)
                    return op.emitError(
                        "elementwise bridge residue duration does not match emitted body");
                cycle = vxmCycle
                    + columnDomainCount
                        * (residueDuration.active + residueDuration.bridge);
            }
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
                direct_domain_detail::emitMem3D(
                    rewriter, op.getLoc(), target,
                    pairCycle + eastLatency(slice),
                    outputHemisphere
                            * target.memory().slices_per_hemisphere
                        + slice,
                    "write", result->row,
                    physicalStreamBase + byte, rows, 1, 1,
                    1, 1, 0, 1, 1, 0, -1, result->bank);
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
