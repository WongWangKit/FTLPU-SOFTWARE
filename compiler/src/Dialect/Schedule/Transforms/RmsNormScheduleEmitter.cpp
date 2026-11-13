#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "DirectDomainEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ftlpu::compiler::schedule {
namespace {

using attention_detail::emitMem;
using direct_domain_detail::emitMem3D;
using attention_detail::emitSxm;
using attention_detail::emitWavefrontBeat;
using attention_detail::emitWavefrontTail;
using attention_detail::blockDiagonalMap;
using ffn_detail::create_vxm;
using ffn_detail::schedule_placement;

// Hemispheres own independently numbered stream fabrics. Directions are local
// to each fabric, so SXM ingress is eastbound and egress is westbound on both.
constexpr auto kMemToSxmDirection = target::StreamDirection::East;
constexpr auto kSxmToMemDirection = target::StreamDirection::West;
constexpr int64_t kSxmInputStream = 0;
constexpr int64_t kSxmTransposeStream = 16;
constexpr int64_t kSxmOutputStream = 32;

mlir::DictionaryAttr allocationPlacement(mlir::ArrayAttr allocations,
    int64_t index)
{
    return llvm::cast<mlir::DictionaryAttr>(allocations[index])
        .getAs<mlir::DictionaryAttr>("placement");
}

llvm::SmallVector<int64_t> slices(mlir::DictionaryAttr placement)
{
    llvm::SmallVector<int64_t> result;
    for (mlir::Attribute value :
        placement.getAs<mlir::ArrayAttr>("slices"))
        result.push_back(llvm::cast<mlir::IntegerAttr>(value).getInt());
    return result;
}

int64_t elementBytes(mlir::RankedTensorType type)
{
    if (is_lpu_16bit_float(type.getElementType())) return 2;
    if (type.getElementType().isF32()) return 4;
    return 0;
}

int64_t inputBindingIndex(mlir::Value value)
{
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
        return argument.getArgNumber();
    if (auto binding =
            value.getDefiningOp<schedule::BindingOp>())
        return binding.getAccess() == "input"
            ? binding.getIndex() : -1;
    return -1;
}

int64_t baseRow(mlir::DictionaryAttr placement)
{
    return placement.getAs<mlir::IntegerAttr>("base_row").getInt();
}

int64_t bank(mlir::DictionaryAttr placement)
{
    if (auto value = placement.getAs<mlir::IntegerAttr>("bank"))
        return value.getInt();
    return 0;
}

BindingOp createBinding(mlir::IRRewriter& rewriter, mlir::Location location,
    mlir::ValueRange source, int64_t index, llvm::StringRef access,
    llvm::StringRef role, mlir::RankedTensorType type,
    mlir::DictionaryAttr placement)
{
    mlir::OperationState state(location, BindingOp::getOperationName());
    state.addOperands(source);
    state.addTypes(type);
    state.addAttributes({
        rewriter.getNamedAttr("index", rewriter.getI64IntegerAttr(index)),
        rewriter.getNamedAttr("access", rewriter.getStringAttr(access)),
        rewriter.getNamedAttr("role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(
            type.getNumElements() * elementBytes(type))),
        rewriter.getNamedAttr("placement", placement),
    });
    return llvm::cast<BindingOp>(rewriter.create(state));
}

void createTimeline(mlir::IRRewriter& rewriter, mlir::Location location,
    llvm::StringRef name, int64_t start, int64_t end)
{
    mlir::OperationState state(location, TimelineOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("name", rewriter.getStringAttr(name)),
        rewriter.getNamedAttr("start", rewriter.getI64IntegerAttr(start)),
        rewriter.getNamedAttr("end", rewriter.getI64IntegerAttr(end)),
    });
    rewriter.create(state);
}

int64_t serialFeedbackAddress(mlir::DictionaryAttr placement,
    int64_t tokenBlock, int64_t hiddenBlock, int64_t wave,
    int64_t row, int64_t rows, int64_t hidden, bool feedbackLayout)
{
    if (feedbackLayout)
        return baseRow(placement) + tokenBlock * hidden
            + hiddenBlock * 32 + wave * 8 + row;
    return baseRow(placement) + hiddenBlock * rows
        + tokenBlock * 32 + wave * 8 + row;
}

int64_t rowParallelAddress(mlir::DictionaryAttr placement,
    int64_t tokenBlock, int64_t hiddenBlock, int64_t wave,
    int64_t row, int64_t hidden)
{
    return baseRow(placement) + (tokenBlock / 8) * hidden
        + hiddenBlock * 32 + wave * 8 + row;
}

int64_t emitSerialFeedbackTranspose(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr outputPlacement,
    int64_t rows, int64_t hidden, int64_t start,
    bool inputFeedback, bool outputFeedback, bool broadcastInput)
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t tileRows = target.throughput().tile_rows;
    const int64_t tokenBlocks = broadcastInput ? 1 : rows / tile;
    int64_t cycle = start;

    for (int64_t tokenBlock = 0;
         tokenBlock < tokenBlocks; ++tokenBlock) {
        for (int64_t hiddenBlock = 0;
             hiddenBlock < hidden / tile; ++hiddenBlock) {
            for (int64_t wave = 0; wave < tileRows; ++wave) {
                const int64_t capture = cycle
                    + target.throughput().mem_to_sxm_latency;
                for (int64_t row = 0; row < lanes; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < target.memory().hemispheres;
                         ++hemisphere) {
                        const std::array<int64_t, 2> sourceStreams {
                            kSxmInputStream,
                            kSxmInputStream + 1,
                        };
                        const std::array<int64_t, 2> transposeStreams {
                            kSxmTransposeStream,
                            kSxmTransposeStream + 1,
                        };
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = inputSlices[byte];
                            const int64_t latency =
                                *target.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::SxmInput,
                                    kMemToSxmDirection, slice);
                            const int64_t address = broadcastInput
                                ? baseRow(inputPlacement) + hiddenBlock
                                : serialFeedbackAddress(inputPlacement,
                                    tokenBlock, hiddenBlock, wave, row,
                                    rows, hidden, inputFeedback);
                            emitMem(rewriter, location,
                                capture + row - latency,
                                hemisphere
                                        * target.memory().
                                            slices_per_hemisphere
                                    + slice,
                                "read", address,
                                kSxmInputStream + byte,
                                1, 1, 0);
                        }
                        emitSxm(rewriter, location, capture + row,
                            hemisphere, "transpose", sourceStreams,
                            transposeStreams,
                            attention_detail::identityMap(target));
                    }
                }

                const int64_t permute = capture + lanes
                    + target.throughput().tile_rows;
                const int64_t diagonal = inputFeedback && !outputFeedback
                    ? (tileRows - wave) % tileRows
                    : wave;
                const auto map = blockDiagonalMap(diagonal, target);
                for (int64_t row = 0; row < lanes; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < target.memory().hemispheres;
                         ++hemisphere) {
                        const std::array<int64_t, 2> transposeStreams {
                            kSxmTransposeStream,
                            kSxmTransposeStream + 1,
                        };
                        const std::array<int64_t, 2> outputStreams {
                            kSxmOutputStream,
                            kSxmOutputStream + 1,
                        };
                        emitSxm(rewriter, location, permute + row,
                            hemisphere, "permute", transposeStreams,
                            outputStreams, map);
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t address =
                                serialFeedbackAddress(outputPlacement,
                                    tokenBlock, hiddenBlock, wave, row,
                                    rows, hidden, outputFeedback);
                            for (int64_t duplicate = byte;
                                 duplicate
                                     < static_cast<int64_t>(
                                         outputSlices.size());
                                 duplicate += 2) {
                                const int64_t slice =
                                    outputSlices[duplicate];
                                const int64_t latency =
                                    *target.transport_latency(
                                        target::StreamEndpoint::SxmResult,
                                        target::StreamEndpoint::Mem,
                                        kSxmToMemDirection, slice);
                                emitMem(rewriter, location,
                                    permute + row + latency,
                                    hemisphere
                                            * target.memory().
                                                slices_per_hemisphere
                                        + slice,
                                    "write", address,
                                    kSxmOutputStream + byte,
                                    1, 1, 0);
                            }
                        }
                    }
                }
                cycle = permute + lanes;
            }
        }
    }
    return cycle + target.streams().system_register_columns;
}

int64_t emitMatrixTranspose(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr outputPlacement,
    int64_t rows, int64_t hidden, int64_t start,
    bool inputFeedback, bool outputFeedback, bool broadcastInput)
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t tokenBlocks = broadcastInput ? 1 : rows / tile;
    int64_t cycle = start;

    for (int64_t tokenBlock = 0;
         tokenBlock < tokenBlocks; ++tokenBlock) {
        for (int64_t hiddenBlock = 0;
             hiddenBlock < hidden / tile; ++hiddenBlock) {
            for (int64_t hemisphere = 0;
                 hemisphere < target.memory().hemispheres;
                 ++hemisphere) {
                const int64_t capture = cycle
                    + target.throughput().mem_to_sxm_latency;
                for (int64_t row = 0; row < tile; ++row) {
                    const std::array<int64_t, 2> sourceStreams {
                        kSxmInputStream,
                        kSxmInputStream + 1,
                    };
                    const std::array<int64_t, 2> transposeStreams {
                        kSxmTransposeStream,
                        kSxmTransposeStream + 1,
                    };
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = inputSlices[byte];
                        const int64_t latency =
                            *target.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::SxmInput,
                                kMemToSxmDirection, slice);
                        const int64_t address = broadcastInput
                            ? baseRow(inputPlacement) + hiddenBlock
                            : inputFeedback
                                ? baseRow(inputPlacement)
                                    + tokenBlock * hidden
                                    + hiddenBlock * tile + row
                                : baseRow(inputPlacement)
                                    + hiddenBlock * rows
                                    + tokenBlock * tile + row;
                        emitMem(rewriter, location,
                            capture + row - latency,
                            hemisphere
                                    * target.memory().slices_per_hemisphere
                                + slice,
                            "read", address,
                            kSxmInputStream + byte,
                            1, 1, 0);
                    }
                    emitSxm(rewriter, location, capture + row,
                        hemisphere, "transpose", sourceStreams,
                        transposeStreams,
                        attention_detail::identityMap(target),
                        "matrix_columns");
                }
                const int64_t emit = capture + tile
                    + target.throughput().tile_rows + 1;
                for (int64_t row = 0; row < tile; ++row) {
                    const std::array<int64_t, 2> transposeStreams {
                        kSxmTransposeStream,
                        kSxmTransposeStream + 1,
                    };
                    const std::array<int64_t, 2> outputStreams {
                        kSxmOutputStream,
                        kSxmOutputStream + 1,
                    };
                    emitSxm(rewriter, location, emit + row,
                        hemisphere, "permute", transposeStreams,
                        outputStreams, attention_detail::identityMap(target),
                        "matrix_columns");
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        for (int64_t duplicate = byte;
                             duplicate
                                 < static_cast<int64_t>(
                                     outputSlices.size());
                             duplicate += 2) {
                            const int64_t slice = outputSlices[duplicate];
                            const int64_t latency =
                                *target.transport_latency(
                                    target::StreamEndpoint::SxmResult,
                                    target::StreamEndpoint::Mem,
                                    kSxmToMemDirection, slice);
                            const int64_t address = outputFeedback
                                ? baseRow(outputPlacement)
                                    + tokenBlock * hidden
                                    + hiddenBlock * tile + row
                                : baseRow(outputPlacement)
                                    + hiddenBlock * rows
                                    + tokenBlock * tile + row;
                            emitMem(rewriter, location,
                                emit + row + latency,
                                hemisphere
                                        * target.memory().
                                            slices_per_hemisphere
                                    + slice,
                                "write", address,
                                kSxmOutputStream + byte,
                                1, 1, 0);
                        }
                    }
                }
                cycle = emit + tile
                    + target.streams().system_register_columns;
            }
        }
    }
    return cycle;
}

llvm::SmallVector<int64_t> streamRange(int64_t first, int64_t count)
{
    llvm::SmallVector<int64_t> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int64_t index = 0; index < count; ++index)
        result.push_back(first + index);
    return result;
}

llvm::SmallVector<int64_t, 64> rotateTileMap(int64_t distance,
    const target::LPUTargetModel& target)
{
    const int64_t tileRows = target.throughput().tile_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    auto map = attention_detail::identityMap(tileRows * lanes);
    for (int64_t destination = 0; destination < tileRows;
         ++destination) {
        const int64_t source =
            (destination + distance) % tileRows;
        for (int64_t lane = 0; lane < lanes; ++lane)
            map[static_cast<std::size_t>(
                destination * lanes + lane)] =
                    source * lanes + lane;
    }
    return map;
}

int64_t planarAddress(mlir::DictionaryAttr placement,
    int64_t tokenBlock, int64_t hiddenBlock, int64_t wave,
    int64_t row, int64_t rows)
{
    return baseRow(placement) + hiddenBlock * rows
        + tokenBlock * 32 + wave * 8 + row;
}

int64_t packedAddress(mlir::DictionaryAttr placement,
    int64_t tokenBlock, int64_t hiddenBlock, int64_t wave,
    int64_t hiddenBlocks, int64_t tileRows)
{
    return baseRow(placement)
        + (tokenBlock * hiddenBlocks + hiddenBlock) * tileRows + wave;
}

int64_t emitDistributedMatrixTranspose(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr outputPlacement,
    int64_t rows, int64_t hidden, int64_t start,
    int64_t selectedHemisphere = -1)
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t width = 2 * lanes;
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t tileRows = target.throughput().tile_rows;
    if (inputSlices.size() != static_cast<std::size_t>(width)
        || outputSlices.size() != static_cast<std::size_t>(width))
        return start;

    llvm::SmallVector<int64_t> sourceStreams(
        static_cast<std::size_t>(width));
    llvm::SmallVector<int64_t> transposeStreams(
        static_cast<std::size_t>(width));
    llvm::SmallVector<int64_t> outputStreams(
        static_cast<std::size_t>(width));
    const int64_t sourceBase =
        target.streams().streams_per_direction - width;
    const int64_t outputBase = target.streams().streams_per_direction;
    for (int64_t stream = 0; stream < width; ++stream) {
        sourceStreams[static_cast<std::size_t>(stream)] = sourceBase + stream;
        transposeStreams[static_cast<std::size_t>(stream)] = stream;
        outputStreams[static_cast<std::size_t>(stream)] = outputBase + stream;
    }
    const auto readLatency = [&](int64_t slice) {
        return *target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::SxmInput,
            target::StreamDirection::East, slice);
    };
    const auto writeLatency = [&](int64_t slice) {
        return *target.transport_latency(target::StreamEndpoint::SxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::West, slice);
    };
    int64_t maxReadLatency = 0;
    int64_t maxWriteLatency = 0;
    for (int64_t slice : inputSlices)
        maxReadLatency = std::max(maxReadLatency, readLatency(slice));
    for (int64_t slice : outputSlices)
        maxWriteLatency = std::max(maxWriteLatency, writeLatency(slice));

    const int64_t hiddenBlocks = hidden / tile;
    const int64_t blockCount = (rows / tile) * hiddenBlocks;
    const int64_t inputBeats = blockCount * tileRows;
    const int64_t captureStart = start + maxReadLatency;
    const auto module = rewriter.getInsertionBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    const auto wide = module ? module->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide") : mlir::BoolAttr{};
    if (wide && wide.getValue()) {
        // Serialize distributed row pairs onto one BF16 vector stream.
        // Scatter transposed columns back with MEM addresses, no permute.
        for (int64_t h=0;h<target.memory().hemispheres;++h) {
            if(selectedHemisphere>=0 && h!=selectedHemisphere) continue;
            for(int64_t pair=0;pair<lanes;++pair) for(int64_t byte=0;byte<2;++byte) {
                const auto src=inputSlices[2*pair+byte], dst=outputSlices[2*pair+byte];
                emitMem3D(rewriter,location,target,captureStart+pair-readLatency(src),
                    h*target.memory().slices_per_hemisphere+src,"read",baseRow(inputPlacement),
                    sourceBase+byte,tileRows,lanes,1,blockCount,tile,tileRows,1,1,0,-1,bank(inputPlacement));
                emitMem3D(rewriter,location,target,captureStart+tile+tileRows-1+pair+writeLatency(dst),
                    h*target.memory().slices_per_hemisphere+dst,"write",baseRow(outputPlacement),
                    outputBase+byte,tileRows,lanes,1,blockCount,tile,tileRows,1,1,0,-1,bank(outputPlacement));
            }
            emitSxm(rewriter,location,captureStart,h,"transpose",
                {sourceBase,sourceBase+1},{outputBase,outputBase+1},
                attention_detail::identityMap(target),"wide_matrix",-1,-1,-1,blockCount,tile);
        }
        return captureStart+blockCount*tile+tile+tileRows-1+maxWriteLatency+tileRows;
    }
    for (int64_t hemisphere = 0;
         hemisphere < target.memory().hemispheres; ++hemisphere) {
        if (selectedHemisphere >= 0 && hemisphere != selectedHemisphere)
            continue;
        for (int64_t stream = 0; stream < width; ++stream) {
            const int64_t slice =
                inputSlices[static_cast<std::size_t>(stream)];
            emitMem3D(rewriter, location, target,
                captureStart - readLatency(slice),
                hemisphere * target.memory().slices_per_hemisphere + slice,
                "read", baseRow(inputPlacement), sourceBase + stream,
                inputBeats, 1, 1, 1, 1, 0, 1, 1, 0, -1,
                bank(inputPlacement));
        }

        // The transpose body is invariant.  The permute body advances by one
        // physical lane group on every beat, which is exactly the native SXM
        // map-stride induction field.
        emitSxm(rewriter, location, captureStart, hemisphere, "transpose",
            sourceStreams, transposeStreams,
            attention_detail::identityMap(target), "vector_columns", -1, -1,
            -1, inputBeats, 1);
        emitSxm(rewriter, location, captureStart + 1, hemisphere, "permute",
            transposeStreams, outputStreams,
            blockDiagonalMap(0, target), "vector_columns", -1, -1,
            -1, inputBeats + tileRows - 1, 1, 1, 1, lanes);

        for (int64_t stream = 0; stream < width; ++stream) {
            const int64_t slice =
                outputSlices[static_cast<std::size_t>(stream)];
            emitMem3D(rewriter, location, target,
                captureStart + 1 + writeLatency(slice),
                hemisphere * target.memory().slices_per_hemisphere + slice,
                "write", baseRow(outputPlacement), outputBase + stream,
                inputBeats, 1, 1, 1, 1, 0, 1, 1, 0, -1,
                bank(outputPlacement));
        }

    }
    return captureStart + inputBeats
        + std::max(tileRows, maxWriteLatency + 1);
}

int64_t emitPairToPackedTranspose(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr outputPlacement,
    int64_t rows, int64_t hidden, int64_t start,
    bool broadcastInput)
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t tileRows = target.throughput().tile_rows;
    const int64_t packedStreams = 2 * lanes;
    const int64_t hiddenBlocks = hidden / tile;
    const int64_t tokenBlocks = broadcastInput ? 1 : rows / tile;
    const auto inputKind = inputPlacement.getAs<mlir::StringAttr>("kind");
    const bool rowParallel = inputKind
        && inputKind.getValue() == "fp16_vxm_row_parallel_8";
    int64_t cycle = start;

    for (int64_t tokenBlock = 0;
         tokenBlock < tokenBlocks; ++tokenBlock) {
        for (int64_t featureWave = 0;
             featureWave < tileRows; ++featureWave) {
            for (int64_t hiddenBlock = 0;
                 hiddenBlock < hiddenBlocks; ++hiddenBlock) {
                const int64_t fill = cycle
                    + target.throughput().mem_to_sxm_latency;
                for (int64_t hemisphere = 0;
                     hemisphere < target.memory().hemispheres;
                     ++hemisphere) {
                    const auto transposeStreams = streamRange(
                        kSxmTransposeStream, packedStreams);
                    for (int64_t row = 0; row < lanes; ++row) {
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t pair = rowParallel
                                ? tokenBlock % 8 : 0;
                            const int64_t slice =
                                inputSlices[2 * pair + byte];
                            const int64_t latency =
                                *target.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::SxmInput,
                                    kMemToSxmDirection, slice);
                            const int64_t address = broadcastInput
                                ? baseRow(inputPlacement) + hiddenBlock
                                : rowParallel
                                ? rowParallelAddress(inputPlacement,
                                    tokenBlock, hiddenBlock, featureWave, row,
                                    hidden)
                                : planarAddress(inputPlacement,
                                    tokenBlock, hiddenBlock, featureWave, row,
                                    rows);
                            emitMem(rewriter, location,
                                fill + row - latency,
                                hemisphere
                                        * target.memory().
                                            slices_per_hemisphere
                                    + slice,
                                "read", address,
                                kSxmInputStream + 2 * row + byte,
                                1, 1, 0, "sram", -1,
                                bank(inputPlacement));
                        }
                        const std::array<int64_t, 2> sourceStreams {
                            kSxmInputStream + 2 * row,
                            kSxmInputStream + 2 * row + 1,
                        };
                        emitSxm(rewriter, location, fill + row,
                            hemisphere, "transpose", sourceStreams,
                            transposeStreams,
                            attention_detail::identityMap(target),
                            "vector_columns", -1, row);
                    }
                }

                const int64_t permuteBegin =
                    fill + lanes + tileRows - 1;
                for (int64_t tokenWave = 0;
                     tokenWave < tileRows; ++tokenWave) {
                    const int64_t permute = permuteBegin + tokenWave;
                    const auto map = blockDiagonalMap(
                        (tokenWave + featureWave) % tileRows, target);
                    for (int64_t hemisphere = 0;
                         hemisphere < target.memory().hemispheres;
                         ++hemisphere) {
                        const auto transposeStreams = streamRange(
                            kSxmTransposeStream, packedStreams);
                        const auto outputStreams = streamRange(
                            kSxmOutputStream, packedStreams);
                        emitSxm(rewriter, location, permute, hemisphere,
                            "permute", transposeStreams, outputStreams, map,
                            "vector_columns", -1, -1, featureWave);
                        for (int64_t stream = 0;
                             stream < packedStreams; ++stream) {
                            const int64_t slice = outputSlices[stream];
                            const int64_t latency = *target.transport_latency(
                                target::StreamEndpoint::SxmResult,
                                target::StreamEndpoint::Mem,
                                kSxmToMemDirection,
                                slice);
                            emitMem(rewriter, location,
                                permute + latency - featureWave,
                                hemisphere
                                        * target.memory().
                                            slices_per_hemisphere
                                    + slice,
                                "write_tap", packedAddress(outputPlacement,
                                    tokenBlock, hiddenBlock, tokenWave,
                                    hiddenBlocks, tileRows),
                                kSxmOutputStream + stream,
                                1, 1, 0, "sram", -1,
                                bank(outputPlacement));
                        }
                    }
                }
                cycle = permuteBegin + tileRows
                    + target.streams().system_register_columns;
            }
        }
    }
    return cycle;
}

int64_t emitPackedToPairTranspose(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr outputPlacement,
    int64_t rows, int64_t hidden, int64_t start,
    bool outputFeedback,
    mlir::DictionaryAttr duplicateOutputPlacement = {})
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const auto duplicateOutputSlices = duplicateOutputPlacement
        ? slices(duplicateOutputPlacement)
        : llvm::SmallVector<int64_t>{};
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t tileRows = target.throughput().tile_rows;
    const int64_t packedStreams = 2 * lanes;
    const int64_t hiddenBlocks = hidden / tile;
    int64_t cycle = start;

    for (int64_t tokenBlock = 0; tokenBlock < rows / tile; ++tokenBlock) {
        for (int64_t tokenWave = 0;
             tokenWave < tileRows; ++tokenWave) {
            for (int64_t hiddenBlock = 0;
                 hiddenBlock < hiddenBlocks; ++hiddenBlock) {
                const int64_t capture = cycle
                    + target.throughput().mem_to_sxm_latency;
                for (int64_t hemisphere = 0;
                     hemisphere < target.memory().hemispheres;
                     ++hemisphere) {
                    const auto sourceStreams = streamRange(
                        kSxmInputStream, packedStreams);
                    const auto transposeStreams = streamRange(
                        kSxmTransposeStream, packedStreams);
                    for (int64_t stream = 0;
                         stream < packedStreams; ++stream) {
                        const int64_t slice = inputSlices[stream];
                        const int64_t latency = *target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::SxmInput,
                            kMemToSxmDirection, slice);
                        emitMem(rewriter, location, capture - latency,
                            hemisphere
                                    * target.memory().slices_per_hemisphere
                                + slice,
                            "read", packedAddress(inputPlacement,
                                tokenBlock, hiddenBlock, tokenWave,
                                hiddenBlocks, tileRows),
                            kSxmInputStream + stream,
                            1, 1, 0, "sram", -1,
                            bank(inputPlacement));
                    }
                    emitSxm(rewriter, location, capture, hemisphere,
                        "transpose", sourceStreams, transposeStreams,
                        attention_detail::identityMap(target));
                }

                const int64_t permuteBegin = capture
                    + target.throughput().tile_rows + 1;
                for (int64_t featureWave = 0;
                     featureWave < tileRows; ++featureWave) {
                    const auto map = blockDiagonalMap(
                        (tokenWave + featureWave) % tileRows, target);
                    for (int64_t row = 0; row < lanes; ++row) {
                        const int64_t permute = permuteBegin
                            + featureWave * lanes + row;
                        for (int64_t hemisphere = 0;
                             hemisphere < target.memory().hemispheres;
                             ++hemisphere) {
                            const auto transposeStreams = streamRange(
                                kSxmTransposeStream, packedStreams);
                            const auto outputStreams = streamRange(
                                kSxmOutputStream, packedStreams);
                            emitSxm(rewriter, location, permute,
                                hemisphere, "permute", transposeStreams,
                                outputStreams, map, "vector_columns", row,
                                -1, tokenWave);
                            for (int64_t byte = 0; byte < 2; ++byte) {
                            llvm::SmallVector<int64_t, 8> sliceIndices;
                            if (outputFeedback) {
                                const int64_t tokenBlocks = rows / tile;
                                const int64_t batch = tokenBlock / 8;
                                const int64_t active =
                                    std::min<int64_t>(8,
                                        tokenBlocks - batch * 8);
                                for (int64_t pair = tokenBlock % 8;
                                     pair < 8; pair += active)
                                    sliceIndices.push_back(2 * pair + byte);
                            } else {
                                for (int64_t index = byte;
                                     index < static_cast<int64_t>(
                                         outputSlices.size());
                                     index += 2)
                                    sliceIndices.push_back(index);
                            }
                            llvm::SmallVector<mlir::DictionaryAttr, 2>
                                destinations {outputPlacement};
                            if (duplicateOutputPlacement)
                                destinations.push_back(
                                    duplicateOutputPlacement);
                            for (mlir::DictionaryAttr destination :
                                 destinations) {
                                const auto destinationSlices =
                                    destination == outputPlacement
                                    ? llvm::ArrayRef<int64_t>(outputSlices)
                                    : llvm::ArrayRef<int64_t>(
                                          duplicateOutputSlices);
                                for (int64_t duplicate : sliceIndices) {
                                    const int64_t slice =
                                        destinationSlices[duplicate];
                                    const int64_t latency =
                                        *target.transport_latency(
                                            target::StreamEndpoint::SxmResult,
                                            target::StreamEndpoint::Mem,
                                            kSxmToMemDirection, slice);
                                    emitMem(rewriter, location,
                                        permute + latency - tokenWave,
                                        hemisphere
                                                * target.memory().
                                                    slices_per_hemisphere
                                            + slice,
                                        "write_tap", outputFeedback
                                            ? rowParallelAddress(
                                                  destination, tokenBlock,
                                                  hiddenBlock, featureWave,
                                                  row,
                                                  hidden)
                                            : planarAddress(
                                                  destination, tokenBlock,
                                                  hiddenBlock, featureWave,
                                                  row,
                                                  rows),
                                        kSxmOutputStream + 2 * row + byte,
                                        1, 1, 0, "sram", -1,
                                        bank(destination));
                                }
                            }
                            }
                        }
                    }
                }
                cycle = permuteBegin + tileRows * lanes
                    + target.streams().system_register_columns;
            }
        }
    }
    return cycle;
}

int64_t emitLegacyVxmFeedback(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr weightPlacement,
    mlir::DictionaryAttr outputPlacement, int64_t start)
{
    const auto inputSlices = slices(inputPlacement);
    const auto weightSlices = slices(weightPlacement);
    const auto outputSlices = slices(outputPlacement);
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(inputType.getElementType());
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(inputType.getElementType());
    const int64_t rows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t waves = target.throughput().tile_rows;
    const auto westReadLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice).value();
    };
    const auto eastWriteLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice).value();
    };

    const int64_t hiddenBlocks = hidden / tile;
    const int64_t featuresPerBeat = target.throughput().lanes_per_tile;
    int64_t cycle = start;
    for (int64_t tokenBlock = 0;
         tokenBlock < rows / tile; ++tokenBlock) {
        const int64_t reductionStart = cycle
            + westReadLatency(inputSlices.back());
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, reductionStart - 1, 15, "pass",
            "immediate", 0, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, 1, 1, "east", "east");
        int64_t beat = 0;
        for (int64_t hiddenBlock = 0;
             hiddenBlock < hiddenBlocks; ++hiddenBlock) {
            for (int64_t featureWave = 0;
                 featureWave < waves; ++featureWave, ++beat) {
                const int64_t square = reductionStart + beat;
                const int64_t address = packedAddress(inputPlacement,
                    tokenBlock, hiddenBlock, featureWave, hiddenBlocks, waves);
                for (int64_t stream = 0; stream < 16; ++stream)
                    emitMem(rewriter, op.getLoc(),
                        square - westReadLatency(inputSlices[stream]),
                        inputSlices[stream], "read", address,
                        32 + stream, 1, 1, 0);
                for (int64_t feature = 0;
                     feature < featuresPerBeat; ++feature)
                    create_vxm(rewriter, op.getLoc(),
                        op.getInput(), op.getInput(), inputType,
                        square, feature, "square",
                        streamKind, 32 + 2 * feature, 0.0f,
                        "immediate", 0, 0.0f, "fp32", -1,
                        1, 1, "east", "east");
                for (int64_t pair = 0; pair < 4; ++pair)
                    create_vxm(rewriter, op.getLoc(),
                        op.getInput(), op.getInput(), inputType,
                        square + 1, 8 + pair, "add",
                        "alu", 2 * pair, 0.0f,
                        "alu", 2 * pair + 1, 0.0f, "fp32", -1,
                        1, 1, "east", "east");
                for (int64_t pair = 0; pair < 2; ++pair)
                    create_vxm(rewriter, op.getLoc(),
                        op.getInput(), op.getInput(), inputType,
                        square + 2, 12 + pair, "add",
                        "alu", 8 + 2 * pair, 0.0f,
                        "alu", 9 + 2 * pair, 0.0f, "fp32", -1,
                        1, 1, "east", "east");
                create_vxm(rewriter, op.getLoc(),
                    op.getInput(), op.getInput(), inputType,
                    square + 3, 14, "add",
                    "alu", 12, 0.0f, "alu", 13, 0.0f,
                    "fp32", -1, 1, 1, "east", "east");
                create_vxm(rewriter, op.getLoc(),
                    op.getInput(), op.getInput(), inputType,
                    square + 4, 15, "add",
                    "alu", 15, 0.0f, "alu", 14, 0.0f,
                    "fp32", -1, 1, 1, "east", "east");
            }
        }

        // Every physical VXM lane owns one logical row. ALU15 therefore
        // contains 32 independent sum(x^2) values after all feature waves.
        const int64_t factor = reductionStart + beat + 4;
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, factor, 8, "divide",
            "alu", 15, 0.0f, "immediate", 0,
            static_cast<float>(hidden), "fp32", -1, 1, 1,
            "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, factor + 1, 9, "add",
            "alu", 8, 0.0f, "immediate", 0,
            static_cast<float>(op.getEpsilon().convertToDouble()),
            "fp32", -1, 1, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, factor + 2, 10, "sqrt",
            "alu", 9, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, 1, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, factor + 3, 11, "divide",
            "immediate", 0, 1.0f, "alu", 10, 0.0f,
            "fp32", -1, 1, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, factor + 4, 15, "pass",
            "alu", 11, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, 1, 1, "east", "east");

        int64_t normalize = factor + 5;
        for (int64_t hiddenBlock = 0;
             hiddenBlock < hiddenBlocks; ++hiddenBlock) {
            for (int64_t featureWave = 0;
                 featureWave < waves; ++featureWave) {
                const int64_t inputAddress = packedAddress(inputPlacement,
                    tokenBlock, hiddenBlock, featureWave, hiddenBlocks, waves);
                const int64_t weightAddress = baseRow(weightPlacement)
                    + hiddenBlock * waves + featureWave;
                const int64_t outputAddress = packedAddress(outputPlacement,
                    tokenBlock, hiddenBlock, featureWave, hiddenBlocks, waves);
                for (int64_t stream = 0; stream < 16; ++stream) {
                    emitMem(rewriter, op.getLoc(),
                        normalize - westReadLatency(inputSlices[stream]),
                        inputSlices[stream], "read", inputAddress,
                        32 + stream, 1, 1, 0);
                    emitMem(rewriter, op.getLoc(),
                        normalize - westReadLatency(weightSlices[stream]),
                        weightSlices[stream], "read", weightAddress,
                        48 + stream, 1, 1, 0, "sram",
                        inputBindingIndex(op.getWeight()));
                }
                for (int64_t feature = 0;
                     feature < featuresPerBeat; ++feature) {
                    const int64_t featureCycle = normalize;
                    create_vxm(rewriter, op.getLoc(),
                        op.getInput(), op.getInput(), inputType,
                        featureCycle, feature, "multiply",
                        streamKind, 32 + 2 * feature, 0.0f,
                        "alu", 15, 0.0f, "fp32", -1,
                        1, 1, "east", "east");
                    create_vxm(rewriter, op.getLoc(),
                        op.getInput(), op.getWeight(), inputType,
                        featureCycle + 1, feature, "multiply",
                        "alu", feature, 0.0f,
                        streamKind, 48 + 2 * feature, 0.0f,
                        dataFormat, 2 * feature, 1, 1,
                        "east", "east");
                    create_vxm(rewriter, op.getLoc(),
                        op.getInput(), op.getWeight(), inputType,
                        featureCycle + 2, feature, "pass",
                        "alu", feature, 0.0f,
                        "immediate", 0, 0.0f,
                        dataFormat, 16 + 2 * feature, 1, 1,
                        "east", "west");
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice =
                            outputSlices[2 * feature + byte];
                        emitMem(rewriter, op.getLoc(),
                            featureCycle + 1
                                + eastWriteLatency(slice),
                            slice, "write", outputAddress,
                            2 * feature + byte, 1, 1, 0);
                        emitMem(rewriter, op.getLoc(),
                            featureCycle + 2
                                + eastWriteLatency(slice),
                            target.memory().slices_per_hemisphere
                                + slice,
                            "write", outputAddress,
                            16 + 2 * feature + byte, 1, 1, 0);
                    }
                }
                normalize += 4
                    + target.streams().system_register_columns;
            }
        }
        cycle = normalize;
    }
    return cycle;
}

int64_t emitVxmFeedbackImpl(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::Value input, mlir::Value weight,
    mlir::RankedTensorType inputType, double epsilon,
    const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr weightPlacement,
    mlir::DictionaryAttr outputPlacement, int64_t start,
    bool l2Norm = false, float outputScale = 1.0f,
    llvm::ArrayRef<int64_t> streamedReduction = {}, bool independentHeads = false,
    int64_t* normalizedStreamStart = nullptr)
{
    const auto inputSlices = slices(inputPlacement);
    const auto weightSlices = weightPlacement ? slices(weightPlacement)
        : llvm::SmallVector<int64_t>();
    const auto outputSlices = slices(outputPlacement);
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(inputType.getElementType());
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(inputType.getElementType());
    const int64_t logicalRows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const int64_t rows = logicalRows == 1
        ? target.throughput().mxm_rows : logicalRows;
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t hiddenBlocks = hidden / tile;
    const int64_t tokenBlocks = rows / tile;
    const int64_t distributedRows =
        rows * hidden / (tile * lanes);
    const bool planarReplay = inputSlices.size() == 2 && outputSlices.size() == 2
        && inputPlacement.getAs<mlir::StringAttr>("kind").getValue() == "fp16_pair_planar"
        && normalizedStreamStart && !streamedReduction.empty() && tokenBlocks == 1;
    // If epsilon and unit variance cannot share even the full FP16 normal
    // range, a single scaled RSQRT can overflow on ordinary activations.
    // Retain the sum in the FP32 feedback chain for the multi-range lookup.
    const bool rangeSafeFactor = !l2Norm && inputType.getElementType().isBF16()
        && epsilon > 0.0 && epsilon <= 0x1p-30;
    llvm::SmallVector<int> lookupExponents;
    if (rangeSafeFactor) {
        // FP16 normal exponents span [-14,15]. A stride of 28 gives
        // overlapping bands; even exponents make the correction exact.
        int exponent = -114;
        const double minimumScale = 0x1p-14 / epsilon;
        // The smallest positive binary32 epsilon is 2^-149, requiring
        // a final even scale exponent of at most 136.
        while (std::ldexp(1.0, exponent) < minimumScale && exponent < 136) {
            lookupExponents.push_back(exponent);
            exponent = std::min(exponent + 28, 136);
        }
        lookupExponents.push_back(exponent);
    }
    if ((!planarReplay && (inputSlices.size() != 16 || outputSlices.size() != 16))
        || (!l2Norm && weightSlices.size() != 2 && weightSlices.size() != 4
            && weightSlices.size() != 16))
        return start;

    const auto readLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice).value();
    };
    const auto writeLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice).value();
    };
    const auto maxReadLatency = [&](llvm::ArrayRef<int64_t> memorySlices) {
        int64_t latency = 0;
        for (int64_t slice : memorySlices)
            latency = std::max(latency, readLatency(slice));
        return latency;
    };
    const auto maxWriteLatency = [&](llvm::ArrayRef<int64_t> memorySlices) {
        int64_t latency = 0;
        for (int64_t slice : memorySlices)
            latency = std::max(latency, writeLatency(slice));
        return latency;
    };
    const auto vxm = [&](int64_t cycle, int64_t queue,
                         llvm::StringRef opcode,
                         llvm::StringRef lhsKind, int64_t lhsIndex,
                         float lhsImmediate,
                         llvm::StringRef rhsKind, int64_t rhsIndex,
                         float rhsImmediate,
                         llvm::StringRef castTarget, int64_t outputStream,
                         int64_t repeatCount, int64_t chainDepth,
                         bool accumulatorReset = false,
                         bool accumulatorWrite = false,
                         bool accumulatorEmit = true,
                         bool localScalarWrite = false) {
        auto primary = create_vxm(rewriter, location, input,
            weight, inputType, cycle, queue, opcode,
            lhsKind, lhsIndex, lhsImmediate,
            rhsKind, rhsIndex, rhsImmediate,
            castTarget, outputStream, repeatCount, 1,
            "east", "east", -1, accumulatorReset,
            accumulatorWrite, accumulatorEmit, localScalarWrite,
            chainDepth);
        return primary;
    };
    const auto emitMirroredPairRead =
        [&](llvm::ArrayRef<int64_t> memorySlices, int64_t pair,
            int64_t address, int64_t stream, int64_t inputCycle,
            int64_t memoryBank, int64_t addressBinding = -1,
            int64_t repeatCount = 1, int64_t repeatInterval = 1,
            int64_t addressStride = 0, int64_t waveCount = 1,
            int64_t waveInterval = 1,
            int64_t waveAddressStride = 0,
            int64_t groupCount = 1,
            int64_t groupInterval = 1) {
        for (int64_t hemisphere = 0;
             hemisphere < target.memory().hemispheres; ++hemisphere) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = memorySlices[2 * pair + byte];
                emitMem3D(rewriter, location, target,
                    inputCycle - readLatency(slice),
                    hemisphere * target.memory().slices_per_hemisphere
                        + slice,
                    "read", address,
                    32 + hemisphere * 16 + stream + byte,
                    repeatCount, repeatInterval, addressStride,
                    waveCount, waveInterval, waveAddressStride,
                    groupCount, groupInterval, 0,
                    addressBinding, memoryBank);
            }
        }
    };
    const auto emitMirroredWrite =
        [&](llvm::ArrayRef<int64_t> memorySlices, int64_t address,
            int64_t originalOutputStream, int64_t outputCycle,
            int64_t byteCount, int64_t memoryBank,
            int64_t repeatCount = 1, int64_t repeatInterval = 1,
            int64_t addressStride = 0, int64_t waveCount = 1,
            int64_t waveInterval = 1,
            int64_t waveAddressStride = 0) {
        for (int64_t destination = 0;
             destination < target.memory().hemispheres; ++destination) {
            const int64_t source = 1 - destination;
            for (int64_t byte = 0; byte < byteCount; ++byte) {
                const int64_t slice = memorySlices[byte];
                const int64_t startCycle = outputCycle
                    + writeLatency(slice);
                emitMem3D(rewriter, location, target,
                    startCycle,
                    destination * target.memory().slices_per_hemisphere
                        + slice,
                    "write", address,
                    source * 8 + originalOutputStream + byte,
                    repeatCount, repeatInterval, addressStride,
                    waveCount, waveInterval, waveAddressStride,
                    1, 1, 0, -1, memoryBank);
            }
        }
    };

    int64_t cycle = start;
    for (int64_t tokenBlock = 0;
         tokenBlock < tokenBlocks; ++tokenBlock) {
        const int64_t reductionInput = streamedReduction.empty() ? cycle
            + maxReadLatency(inputSlices) + 1 : streamedReduction.front();
        const int64_t reductionConfig = reductionInput - 1;
        if(streamedReduction.empty()) {
        // Preserve x in the chain token; C1 squares Previous * Original.
        // This avoids asking both fixed input ports to read the same stream.
        vxm(reductionConfig, 0, "pass",
            streamKind, 32, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, hidden, 4);
        vxm(reductionConfig, 1, "multiply",
            "previous", 0, 0.0f, "original", 0, 0.0f,
            "fp32", -1, hidden, 4);
        vxm(reductionConfig, 2, "pass",
            "previous", 0, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, hidden, 4);
        vxm(reductionConfig, 3, "add",
            "previous", 0, 0.0f, "accumulator", 0, 0.0f,
            "fp32", -1, 1, 4, true, true, false);
        if (hidden > 2)
            vxm(reductionConfig + 1, 3, "add",
                "previous", 0, 0.0f, "accumulator", 0, 0.0f,
                "fp32", -1, hidden - 2, 4, false, true, false);
        vxm(reductionConfig + 2, 3, "add",
            "previous", 0, 0.0f, "accumulator", 0, 0.0f,
            rangeSafeFactor ? "fp32" : dataFormat, rangeSafeFactor ? -1 : 2,
            1, 4, false, true, true);

        } else {
            // Each SXM tile supplies 32 consecutive feature vectors directly.
            // Keep the same accumulator across gaps between tiles.
            for(size_t segment=0;segment<streamedReduction.size();++segment) {
                const auto config=streamedReduction[segment]-1;
                vxm(config,0,"pass",streamKind,32,0.f,"immediate",0,0.f,"fp32",-1,32,4);
                vxm(config,1,"multiply","previous",0,0.f,"original",0,0.f,"fp32",-1,32,4);
                vxm(config,2,"pass","previous",0,0.f,"immediate",0,0.f,"fp32",-1,32,4);
                const bool first=segment==0,last=segment+1==streamedReduction.size();
                if(first) vxm(config,3,"add","previous",0,0.f,"accumulator",0,0.f,"fp32",-1,1,4,true,true,false);
                vxm(config+(first?1:0),3,"add","previous",0,0.f,"accumulator",0,0.f,
                    "fp32",-1,32-(first?1:0)-(last?1:0),4,false,true,false);
                if(last) vxm(config+(first?2:1),3,"add","previous",0,0.f,"accumulator",0,0.f,
                    rangeSafeFactor ? "fp32" : dataFormat,rangeSafeFactor ? -1 : 2,1,4,false,true,true);
            }
        }
        const int64_t featuresPerPair = hidden / lanes;
        const int64_t scalarAddress = baseRow(inputPlacement)
            + (planarReplay ? hidden : distributedRows) + tokenBlock;
        const int64_t sumOutput = (streamedReduction.empty() ? reductionInput+hidden : streamedReduction.back()+32) + 3;
        const auto scalarSlices =
            llvm::ArrayRef<int64_t>(inputSlices).take_front(2);
        const int64_t scalarWriteEnd = sumOutput
            + maxWriteLatency(scalarSlices)
            + 2;
        int64_t scalarReady = scalarWriteEnd;
        if (independentHeads && !rangeSafeFactor) {
            // Fixed VXM outputs cross to the opposite MEM hemisphere. Return
            // only the two-byte reduction scalar to its owning head; the
            // retained feature tile never moves between hemispheres.
            for (int64_t side = 0; side < 2; ++side)
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const auto slice = scalarSlices[byte];
                    const auto latency = 2*(slice/target.streams().mem_slices_per_register_group)+2;
                    emitMem3D(rewriter,location,target,scalarWriteEnd,
                        side*target.memory().slices_per_hemisphere+slice,"read",scalarAddress,48+byte,
                        1,1,0,1,1,0,1,1,0,-1,bank(inputPlacement));
                    emitMem3D(rewriter,location,target,scalarWriteEnd+latency,
                        (1-side)*target.memory().slices_per_hemisphere+slice,"write",scalarAddress,16+byte,
                        1,1,0,1,1,0,1,1,0,-1,bank(inputPlacement));
                    scalarReady=std::max(scalarReady,scalarWriteEnd+latency+2);
                }
        }
        // Basic ALU, multiply and LUT latencies are 1, 2 and 5 clocks
        // (VxmAlu and VxmSpecialAlu). Four cells are traversed per pass.
        // Feedback clocks the result back into the head input register.
        constexpr int64_t basic = 1, mul = 2, lut = 5, feedbackEdge = 1;
        constexpr int64_t lookupLatency = basic + basic + mul + lut;
        constexpr int64_t correctionLatency = mul + basic + basic + basic;
        constexpr int64_t bandInterval = lookupLatency + feedbackEdge
            + correctionLatency + feedbackEdge;
        constexpr int64_t extraScalePass = basic + basic + mul + mul + feedbackEdge;
        const bool splitFinalScale = rangeSafeFactor && lookupExponents.back() > 126;
        const int64_t factorInput = rangeSafeFactor
            ? sumOutput + feedbackEdge + correctionLatency + feedbackEdge
            : scalarReady + maxReadLatency(scalarSlices) + 1;
        const int64_t normalizeInput = rangeSafeFactor
            ? factorInput + bandInterval * lookupExponents.size()
                + (splitFinalScale ? extraScalePass : 0) + 1
            : factorInput + 10;
        // The first input pair also holds the reduction scalar. A later pair
        // may span both passes only when no other memory role can insert a
        // command into its physical (hemisphere, slice, bank) ICU queue.
        const auto canGroupInputPair = [&](int64_t pair) {
            if (!streamedReduction.empty() || pair == 0 || normalizeInput - reductionInput
                    <= (featuresPerPair - 1) * lanes)
                return false;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = inputSlices[2 * pair + byte];
                if (std::count(inputSlices.begin(), inputSlices.end(), slice)
                        != 1
                    || (!l2Norm && bank(inputPlacement) == bank(weightPlacement)
                        && std::find(weightSlices.begin(),
                            weightSlices.end(), slice)
                            != weightSlices.end())
                    || (bank(inputPlacement) == bank(outputPlacement)
                        && std::find(outputSlices.begin(),
                            outputSlices.end(), slice)
                            != outputSlices.end()))
                    return false;
            }
            return true;
        };
        if(streamedReduction.empty()) for (int64_t pair = 0; pair < lanes; ++pair) {
            emitMirroredPairRead(inputSlices, pair,
                packedAddress(inputPlacement, tokenBlock,
                    0, 0, hiddenBlocks, target.throughput().tile_rows),
                0, reductionInput + pair, bank(inputPlacement), -1,
                featuresPerPair, lanes, 1, 1, 1, 0,
                canGroupInputPair(pair) ? 2 : 1,
                normalizeInput - reductionInput);
        }

        if (!rangeSafeFactor)
            emitMirroredWrite(llvm::ArrayRef<int64_t>(inputSlices).take_front(2),
                scalarAddress, 2, sumOutput, 2, bank(inputPlacement));

        const int64_t factorConfig = factorInput - 1;
        // The LUT ingress is FP16 with flush-to-zero. Scale the positive
        // sum+epsilon before RSQRT, then undo that scale in normalization.
        const float lookupScale = !rangeSafeFactor && epsilon > 0.0 && epsilon < 0x1p-14
            ? std::exp2(std::ceil(std::log2(0x1p-14 / epsilon))) : 1.0f;
        if (rangeSafeFactor) {
            // Arm the successor after the reduction's reset, repeat and
            // emit descriptors are queued. The head waits for the tail's
            // actual data-valid, independently of the number of features.
            const int64_t lastReductionConfig = streamedReduction.empty()
                ? reductionConfig : streamedReduction.back() - 1;
            const int64_t sumFeedback = lastReductionConfig + 3;
            const auto pass = [&](int64_t time, int64_t column) {
                vxm(time,column,"pass","previous",0,0.f,"immediate",0,0.f,"fp32",-1,1,4);
            };
            vxm(sumFeedback,0,"multiply","feedback",0,0.f,"immediate",0,
                1.f / static_cast<float>(hidden),"fp32",-1,1,4);
            vxm(sumFeedback,1,"add","previous",0,0.f,"immediate",0,
                static_cast<float>(epsilon),"fp32",-1,1,4,false,false,true,true);
            pass(sumFeedback,2);
            pass(sumFeedback,3);
            for (auto [index, exponent] : llvm::enumerate(lookupExponents)) {
                const int64_t issue = factorInput + index * bandInterval;
                const int firstExponent = std::min(exponent, 126);
                const float scale = std::ldexp(1.f, firstExponent);
                const float minimum = std::max(std::ldexp(1.f, -14 - exponent),
                                               std::numeric_limits<float>::denorm_min());
                // f_s(y) = sqrt(s) * rsqrt(s * max(y, 2^-14/s)).
                // Below its band f_s is a lower bound, above it the FP16
                // ingress returns +inf and RSQRT returns zero.
                vxm(issue,0,"pass","immediate",0,minimum,"immediate",0,minimum,"fp32",-1,1,4);
                vxm(issue,1,"max","previous",0,0.f,"accumulator",0,0.f,"fp32",-1,1,4);
                vxm(issue,2,"multiply","previous",0,0.f,"immediate",0,scale,"fp32",-1,1,4);
                int64_t lookup = issue;
                if (exponent > firstExponent) {
                    // Represent the final scale as two finite FP32 powers
                    // when epsilon itself is in the binary32 subnormal range.
                    vxm(issue,3,"multiply","previous",0,0.f,"immediate",0,
                        std::ldexp(1.f, exponent - firstExponent),"fp32",-1,1,4);
                    lookup = issue + 1;
                    vxm(lookup,0,"pass","feedback",0,0.f,"immediate",0,0.f,"fp32",-1,1,4);
                    pass(lookup,1);
                    pass(lookup,2);
                }
                vxm(lookup,3,"rsqrt","previous",0,0.f,"immediate",0,0.f,"fp32",-1,1,4);
                // One control word per queue per clock. Enqueue the next
                // selector immediately; it waits for the five-cycle LUT.
                const int64_t feedback = lookup + 1;
                vxm(feedback,0,"multiply","feedback",0,0.f,"immediate",0,
                    std::ldexp(1.f, exponent / 2),"fp32",-1,1,4);
                pass(feedback,1);
                pass(feedback,2);
                vxm(feedback,3,"max","previous",0,0.f,"accumulator",0,0.f,
                    "fp32",-1,1,4,index == 0,true,false);
            }
        } else if (l2Norm) {
            // L2 uses sum+epsilon rather than mean+epsilon. Keep the
            // lookup result in C3/C11; normalize compensates lookupScale
            // and applies the requested output scale as an FP32 immediate.
            vxm(factorConfig, 0, "add",
                streamKind, 32, 0.0f, "immediate", 0,
                static_cast<float>(epsilon), "fp32", -1, 1, 4);
            vxm(factorConfig, 1, "multiply",
                "previous", 0, 0.0f, "immediate", 0, lookupScale,
                "fp32", -1, 1, 4);
            vxm(factorConfig, 2, "pass",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 1, 4);
            vxm(factorConfig, 3, "rsqrt",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 1, 4, false, false, true, true);
        } else {
            vxm(factorConfig, 0, "multiply",
                streamKind, 32, 0.0f, "immediate", 0,
                lookupScale / static_cast<float>(hidden), "fp32", -1, 1, 4);
            vxm(factorConfig, 1, "add",
                "previous", 0, 0.0f, "immediate", 0,
                static_cast<float>(epsilon) * lookupScale,
                "fp32", -1, 1, 4);
            vxm(factorConfig, 2, "pass",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 1, 4);
            vxm(factorConfig, 3, "rsqrt",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 1, 4, false, false, true, true);
        }
        if (!rangeSafeFactor)
            emitMirroredPairRead(scalarSlices, 0, scalarAddress, 0,
                factorInput, bank(inputPlacement));

        const int64_t normalizeConfig = normalizeInput - 1;
        if (normalizedStreamStart) *normalizedStreamStart = normalizeInput + (l2Norm ? 5 : 6);
        vxm(normalizeConfig, 0, "multiply",
            streamKind, 32, 0.0f, l2Norm ? "immediate" : streamKind,
            l2Norm ? 0 : 34, l2Norm ? outputScale * std::sqrt(lookupScale) : 0.0f,
            "fp32", -1, hidden, 4);
        vxm(normalizeConfig, 1, "pass",
            "previous", 0, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, hidden, 4);
        vxm(normalizeConfig, 2, l2Norm ? "pass" : "multiply",
            "previous", 0, 0.0f, "immediate", 0, std::sqrt(lookupScale),
            "fp32", -1, hidden, 4);
        vxm(normalizeConfig, 3, "multiply",
            "previous", 0, 0.0f, "accumulator", 0, 0.0f,
            dataFormat, 2, hidden, 4, false, false, true, false);

        int64_t normalizeWriteEnd = normalizeInput;
        const bool broadcastWeight = weightSlices.size() == 2
            || weightSlices.size() == 4;
        if (planarReplay) {
            emitMirroredPairRead(inputSlices, 0, baseRow(inputPlacement), 0,
                normalizeInput, bank(inputPlacement), -1, hidden, 1, 1);
            normalizeWriteEnd = normalizeInput + hidden + (l2Norm ? 5 : 6) + 1;
        }
        for (int64_t pair = 0; !planarReplay && pair < lanes; ++pair) {
            const int64_t inputAddress = packedAddress(
                inputPlacement, tokenBlock, 0, 0, hiddenBlocks,
                target.throughput().tile_rows);
            const int64_t outputAddress = packedAddress(
                outputPlacement, tokenBlock, 0, 0, hiddenBlocks,
                target.throughput().tile_rows);
            if (!canGroupInputPair(pair))
                emitMirroredPairRead(inputSlices, pair, inputAddress, 0,
                    normalizeInput + pair, bank(inputPlacement), -1,
                    featuresPerPair, lanes, 1);
            if (!l2Norm && !broadcastWeight)
                emitMirroredPairRead(weightSlices, pair,
                    baseRow(weightPlacement), 2,
                    normalizeInput + pair, bank(weightPlacement),
                    inputBindingIndex(weight), featuresPerPair, lanes, 1);
            // C0 multiplies by gamma for RMSNorm or the immediate output
            // scale for L2Norm; both have the same two-cycle latency.
            const int64_t outputCycle = normalizeInput + pair + (l2Norm ? 5 : 6);
            const auto outputPair =
                llvm::ArrayRef<int64_t>(outputSlices).slice(2 * pair, 2);
            if (!normalizedStreamStart)
                emitMirroredWrite(outputPair, outputAddress, 2,
                    outputCycle, 2, bank(outputPlacement),
                    featuresPerPair, lanes, 1, 1, 1, 0);
            normalizeWriteEnd = std::max(normalizeWriteEnd,
                outputCycle + (featuresPerPair - 1) * lanes
                    + (normalizedStreamStart ? 0 : maxWriteLatency(outputPair))
                    + 2);
        }
        if (broadcastWeight) {
            const int64_t segmentRows = weightPlacement
                .getAs<mlir::IntegerAttr>("instruction_count").getInt();
            for (int64_t segment = 0;
                 segment < static_cast<int64_t>(weightSlices.size() / 2);
                 ++segment) {
                const int64_t feature = segment * segmentRows;
                if (feature >= hidden) break;
                emitMirroredPairRead(weightSlices, segment,
                    baseRow(weightPlacement), 2,
                    normalizeInput + feature, bank(weightPlacement),
                    inputBindingIndex(weight),
                    std::min(segmentRows, hidden - feature), 1, 1);
            }
        }
        cycle = normalizeWriteEnd + 1;
    }
    return cycle;
}

int64_t emitRowParallelVxmFeedback(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr weightPlacement,
    mlir::DictionaryAttr outputPlacement, int64_t start)
{
    const auto inputSlices = slices(inputPlacement);
    const auto weightSlices = slices(weightPlacement);
    const auto outputSlices = slices(outputPlacement);
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(inputType.getElementType());
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(inputType.getElementType());
    const int64_t rows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const int64_t rowWidth = target.throughput().mxm_rows;
    const int64_t tokenBlocks = rows / rowWidth;
    const int64_t batchCount = (tokenBlocks + 7) / 8;
    const int64_t scalarBase = baseRow(outputPlacement)
        + batchCount * hidden;
    const auto readLatency = [&](int64_t slice) {
        return *target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, slice);
    };
    const auto writeLatency = [&](int64_t slice) {
        return *target.transport_latency(target::StreamEndpoint::VxmResult,
            target::StreamEndpoint::Mem,
            target::StreamDirection::East, slice);
    };
    const auto address = [&](mlir::DictionaryAttr placement,
                             int64_t batch, int64_t column) {
        return baseRow(placement) + batch * hidden + column;
    };
    // One shared control queue drives an east physical chain and its west
    // mirror. Fixed VXM groups 0..7 consume East while groups 8..15 consume
    // West, matching the executable-wide fabric mux used by FFN/attention.
    const auto inputHemisphere = [](int64_t pair) {
        return pair < 4 ? int64_t {0} : int64_t {1};
    };
    const auto outputHemisphere = [&](int64_t pair) {
        return 1 - inputHemisphere(pair);
    };
    const auto emitReadPair = [&](llvm::ArrayRef<int64_t> memorySlices,
                                  int64_t pair, int64_t memoryAddress,
                                  int64_t stream, int64_t inputCycle,
                                  int64_t count, int64_t stride,
                                  int64_t memoryBank,
                                  int64_t addressBinding = -1,
                                  int64_t sourceHemisphere = -1) {
        if (sourceHemisphere < 0)
            sourceHemisphere = inputHemisphere(pair);
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = memorySlices[2 * pair + byte];
            emitMem(rewriter, op.getLoc(),
                inputCycle - readLatency(slice),
                sourceHemisphere
                        * target.memory().slices_per_hemisphere
                    + slice,
                "read",
                memoryAddress, 32 + stream + byte, count, 1, stride,
                "sram", addressBinding, memoryBank);
        }
    };
    const auto emitWritePair = [&](int64_t pair, int64_t memoryAddress,
                                   int64_t stream, int64_t outputCycle,
                                   int64_t count, int64_t stride,
                                   int64_t destinationHemisphere = -1) {
        if (destinationHemisphere < 0)
            destinationHemisphere = outputHemisphere(pair);
        for (int64_t byte = 0; byte < 2; ++byte) {
            const int64_t slice = outputSlices[2 * pair + byte];
            emitMem(rewriter, op.getLoc(),
                outputCycle + writeLatency(slice),
                destinationHemisphere
                        * target.memory().slices_per_hemisphere
                    + slice,
                "write",
                memoryAddress, stream + byte, count, 1, stride,
                "sram", -1, bank(outputPlacement));
        }
    };

    int64_t cycle = start;
    for (int64_t batch = 0; batch < batchCount; ++batch) {
        const int64_t squareConfig = cycle + 8;
        const int64_t squareInput = squareConfig + 1;
        for (int64_t head = 0; head < 8; head += 2) {
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, squareConfig, head, "multiply",
                streamKind, 32 + head * 2, 0.0f,
                streamKind, 32 + head * 2, 0.0f,
                "fp32", -1, hidden, 1, "east", "east",
                -1, false, false, true, false, 2);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, squareConfig, head + 1, "add",
                "previous", 0, 0.0f, "accumulator", 0, 0.0f,
                "fp32", -1, 1, 1, "east", "east",
                -1, true, true, false, false, 2);
            if (hidden > 2)
                create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                    inputType, squareConfig + 1, head + 1, "add",
                    "previous", 0, 0.0f, "accumulator", 0, 0.0f,
                    "fp32", -1, hidden - 2, 1, "east", "east",
                    -1, false, true, false, false, 2);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, squareConfig + 2, head + 1, "add",
                "previous", 0, 0.0f, "accumulator", 0, 0.0f,
                dataFormat, head, 1, 1, "east", "east",
                -1, false, true, true, false, 2);
        }
        for (int64_t pair = 0; pair < 8; ++pair) {
            emitReadPair(inputSlices, pair,
                address(inputPlacement, batch, 0), pair * 4,
                squareInput, hidden, 1, bank(inputPlacement));
        }
        const int64_t squareOutput = squareInput + hidden + 1;
        for (int64_t pair = 0; pair < 8; ++pair)
            emitWritePair(pair, scalarBase + batch * 2,
                pair * 2, squareOutput, 1, 0);

        int64_t maxScalarWrite = squareOutput;
        for (int64_t pair = 0; pair < 8; ++pair)
            maxScalarWrite = std::max(maxScalarWrite,
                squareOutput + writeLatency(outputSlices[2 * pair + 1]));
        const int64_t rsqrtConfig = maxScalarWrite + 8;
        const int64_t rsqrtInput = rsqrtConfig + 1;
        for (int64_t head : {int64_t{0}, int64_t{4}}) {
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, rsqrtConfig, head, "multiply",
                streamKind, 32 + head * 2, 0.0f,
                "immediate", 0, 1.0f / static_cast<float>(hidden),
                "fp32", -1, 2, 1, "east", "east",
                -1, false, false, true, false, 4);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, rsqrtConfig, head + 1, "add",
                "previous", 0, 0.0f, "immediate", 0,
                static_cast<float>(op.getEpsilon().convertToDouble()),
                "fp32", -1, 2, 1, "east", "east",
                -1, false, false, true, false, 4);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, rsqrtConfig, head + 2, "pass",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 2, 1, "east", "east",
                -1, false, false, true, false, 4);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, rsqrtConfig, head + 3, "rsqrt",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                dataFormat, head + 2, 2, 1, "east", "east",
                -1, false, false, true, false, 4);
        }
        constexpr std::array<int64_t, 4> rsqrtInputs {0, 8, 16, 24};
        constexpr std::array<int64_t, 4> rsqrtOutputs {2, 6, 10, 14};
        // Square results cross the VXM: pairs 4/5 are now on East and pairs
        // 0/1 are on West. Feed those to C0/C4/C8/C12 respectively; wave 1
        // repeats the same permutation for pairs 6/7/2/3.
        constexpr std::array<int64_t, 8> rsqrtPairs {
            4, 5, 0, 1, 6, 7, 2, 3,
        };
        for (int64_t wave = 0; wave < 2; ++wave) {
            for (int64_t chain = 0; chain < 4; ++chain) {
                const int64_t pair = rsqrtPairs[wave * 4 + chain];
                emitReadPair(outputSlices, pair,
                    scalarBase + batch * 2, rsqrtInputs[chain],
                    rsqrtInput + wave, 1, 0, bank(outputPlacement), -1,
                    outputHemisphere(pair));
            }
        }
        const int64_t rsqrtOutput = rsqrtInput + 8;
        for (int64_t wave = 0; wave < 2; ++wave) {
            for (int64_t chain = 0; chain < 4; ++chain) {
                const int64_t pair = rsqrtPairs[wave * 4 + chain];
                emitWritePair(pair, scalarBase + batch * 2 + 1,
                    rsqrtOutputs[chain], rsqrtOutput + wave, 1, 0,
                    inputHemisphere(pair));
            }
        }

        int64_t maxRsqrtWrite = rsqrtOutput + 1;
        for (int64_t pair = 0; pair < 8; ++pair)
            maxRsqrtWrite = std::max(maxRsqrtWrite,
                rsqrtOutput + (pair / 4)
                    + writeLatency(outputSlices[2 * pair + 1]));
        const int64_t scalarConfig = maxRsqrtWrite + 8;
        const int64_t scalarInput = scalarConfig + 1;
        for (int64_t head = 0; head < 8; head += 2) {
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, scalarConfig, head, "pass",
                streamKind, 32 + head * 2, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1,
                1, 1, "east", "east",
                -1, false, false, true, false, 2);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, scalarConfig, head + 1, "pass",
                "previous", 0, 0.0f, "immediate", 0, 0.0f,
                "fp32", -1, 1, 1, "east", "east",
                -1, false, false, true, true, 2);
        }
        for (int64_t pair = 0; pair < 8; ++pair)
            emitReadPair(outputSlices, pair,
                scalarBase + batch * 2 + 1, pair * 4,
                scalarInput, 1, 0, bank(outputPlacement));

        const int64_t normalizeConfig = scalarInput + 20;
        const int64_t normalizeInput = normalizeConfig + 1;
        for (int64_t head = 0; head < 8; head += 2) {
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getWeight(),
                inputType, normalizeConfig, head, "multiply",
                streamKind, 32 + head * 2, 0.0f,
                streamKind, 34, 0.0f,
                "fp32", -1, hidden, 1, "east", "east",
                -1, false, false, true, false, 2);
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getWeight(),
                inputType, normalizeConfig, head + 1, "multiply",
                "previous", 0, 0.0f, "accumulator", 0, 0.0f,
                dataFormat, head, hidden, 1, "east", "east",
                -1, false, false, true, false, 2);
        }
        for (int64_t pair = 0; pair < 8; ++pair) {
            emitReadPair(inputSlices, pair,
                address(inputPlacement, batch, 0), pair * 4,
                normalizeInput, hidden, 1, bank(inputPlacement));
        }
        for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere)
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = weightSlices[byte];
                emitMem(rewriter, op.getLoc(),
                    normalizeInput - readLatency(slice),
                    hemisphere * target.memory().slices_per_hemisphere
                        + slice,
                    "read", baseRow(weightPlacement),
                    34 + hemisphere * 16 + byte,
                    hidden, 1, 1, "sram",
                    inputBindingIndex(op.getWeight()),
                    bank(weightPlacement));
            }
        const int64_t normalizeOutput = normalizeInput + 3;
        for (int64_t pair = 0; pair < 8; ++pair)
            emitWritePair(pair, address(outputPlacement, batch, 0),
                pair * 2, normalizeOutput, hidden, 1);

        // Each normalized physical chain writes to the hemisphere opposite
        // its input. Restore-layout SXMs run in both hemispheres, so multicast
        // the owner copy through the passive VXM bridge after normalization.
        int64_t maxNormalizeWrite = normalizeOutput + hidden;
        int64_t maxReadLatency = 0;
        for (int64_t pair = 0; pair < 8; ++pair) {
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = outputSlices[2 * pair + byte];
                maxNormalizeWrite = std::max(maxNormalizeWrite,
                    normalizeOutput + hidden - 1
                        + writeLatency(slice));
                maxReadLatency = std::max(
                    maxReadLatency, readLatency(slice));
            }
        }
        const int64_t bridgeInput = maxNormalizeWrite
            + maxReadLatency + 2;
        int64_t bridgeEnd = bridgeInput + hidden;
        for (int64_t pair = 0; pair < 8; ++pair) {
            const int64_t owner = outputHemisphere(pair);
            const int64_t peer = 1 - owner;
            for (int64_t byte = 0; byte < 2; ++byte) {
                const int64_t slice = outputSlices[2 * pair + byte];
                emitMem(rewriter, op.getLoc(),
                    bridgeInput - readLatency(slice),
                    owner * target.memory().slices_per_hemisphere
                        + slice,
                    "read", address(outputPlacement, batch, 0),
                    32 + pair * 2 + byte, hidden, 1, 1,
                    "sram", -1, bank(outputPlacement));
                const int64_t bridgeWriteLatency =
                    *target.transport_latency(
                        target::StreamEndpoint::VxmBridgeResult,
                        target::StreamEndpoint::Mem,
                        target::StreamDirection::East, slice);
                const int64_t writeCycle = bridgeInput
                    + bridgeWriteLatency;
                emitMem(rewriter, op.getLoc(), writeCycle,
                    peer * target.memory().slices_per_hemisphere
                        + slice,
                    "write", address(outputPlacement, batch, 0),
                    pair * 2 + byte, hidden, 1, 1,
                    "sram", -1, bank(outputPlacement));
                bridgeEnd = std::max(bridgeEnd,
                    writeCycle + hidden);
            }
        }
        cycle = bridgeEnd + 8;
    }
    return cycle;
}

// Shared planar ingress/egress for RMSNorm and L2Norm. The arithmetic remains
// distinct in emitVxmFeedbackImpl (mean+epsilon+gamma versus sum+epsilon).
mlir::FailureOr<mlir::Value> emitDirectPlanarNorm(mlir::IRRewriter& rewriter,
    mlir::Operation* op, mlir::Value inputValue, mlir::Value weight,
    mlir::RankedTensorType inputType, const target::LPUTargetModel& target,
    mlir::DictionaryAttr feedbackInput, mlir::DictionaryAttr feedbackOutput,
    mlir::DictionaryAttr result, mlir::DictionaryAttr weightPlacement,
    double epsilon, bool l2Norm, float outputScale, int64_t outputIndex)
{
    const std::string prefix=l2Norm?"l2norm":"rmsnorm";
    const auto words=op->getAttrOfType<mlir::ArrayAttr>("native_planar_words");
    const auto partitionedWords=op->getAttrOfType<mlir::ArrayAttr>("native_partitioned_planar_words");
    const int64_t blocks=inputType.getDimSize(1)/32;
    const auto stageSlices=slices(feedbackOutput), feedbackSlices=slices(feedbackInput);
    struct Word { mlir::DictionaryAttr p; int64_t address; };
    llvm::SmallVector<llvm::SmallVector<Word>> tileSources[2];
    int64_t gatherIssue[2]={32,32}, gatherEnd=32;
    // Shared VXM controls execute on both fixed chains. Prepare only
    // missing source words on each fabric, before either SXM starts.
    // Padding rows alias their real head; do not copy them twice.
    for(int64_t side=0;side<2;++side) for(int64_t block=0;block<blocks;++block) {
        llvm::SmallVector<Word> sources;
        for(int64_t row=0;row<32;++row) {
            const auto d=llvm::cast<mlir::DictionaryAttr>(partitionedWords
                ? partitionedWords[(side*32+row)*blocks+block] : words[row*blocks+block]);
            auto p=d.getAs<mlir::DictionaryAttr>("placement");
            const int64_t address=d.getAs<mlir::IntegerAttr>("address").getInt();
            const int64_t sourceSide=p.getAs<mlir::StringAttr>("hemisphere").getValue()=="west"?1:0;
            if(partitionedWords && sourceSide!=side)
                { op->emitError("partitioned norm head must remain on its producer hemisphere"); return mlir::failure(); }
            if(sourceSide==side) { sources.push_back({p,address}); continue; }
            int64_t previous=0;
            for(;previous<row;++previous)
                if(words[previous*blocks+block]==words[row*blocks+block]) break;
            if(previous<row) { sources.push_back(sources[previous]); continue; }
            const auto ids=slices(p);
            const int64_t pair=row%8, issue=gatherIssue[side]++;
            for(int64_t byte=0;byte<2;++byte) {
                const int64_t dst=stageSlices[2*pair+byte],latency=ids[byte]/4+dst/4+2;
                emitMem3D(rewriter,op->getLoc(),target,issue,
                    sourceSide*target.memory().slices_per_hemisphere+ids[byte],"read",address,52+byte,
                    1,1,0,1,1,0,1,1,0,-1,bank(p));
                emitMem3D(rewriter,op->getLoc(),target,issue+latency,
                    side*target.memory().slices_per_hemisphere+dst,"write",baseRow(feedbackOutput)+block*4+row/8,20+byte,
                    1,1,0,1,1,0,1,1,0,-1,bank(feedbackOutput));
                gatherEnd=std::max(gatherEnd,issue+latency+4);
            }
            mlir::NamedAttrList staged(feedbackOutput);
            staged.set("slices",rewriter.getArrayAttr({rewriter.getI64IntegerAttr(stageSlices[2*pair]),rewriter.getI64IntegerAttr(stageSlices[2*pair+1])}));
            staged.set("hemisphere",rewriter.getStringAttr(side?"west":"east"));
            sources.push_back({staged.getDictionary(rewriter.getContext()),baseRow(feedbackOutput)+block*4+row/8});
        }
        tileSources[side].push_back(std::move(sources));
    }
    if(!partitionedWords)
        createTimeline(rewriter,op->getLoc(),"l2norm.prepare_sxm_pair",32,gatherEnd);
    // Both wide SXMs feed the fixed VXM groups (East 0/1, West 16/17).
    // Tap the transposed x along the SR route for normalization replay;
    // never consume it before it reaches VXM.
    const int64_t firstCapture=gatherEnd+32;
    int64_t transposeEnd=firstCapture;
    for(int64_t side=0;side<2;++side) {
        emitSxm(rewriter,op->getLoc(),firstCapture,side,"transpose",{16,17},{32+side*16,33+side*16},
            attention_detail::identityMap(target),"wide_matrix",-1,-1,-1,blocks,32);
        for(int64_t block=0;block<blocks;++block) {
            const auto &sources=tileSources[side][block];
            const int64_t capture=firstCapture+block*32;
            for(int64_t row=0;row<32;) {
                const auto w=sources[row];int64_t count=1,stride=0;
                if(row+1<32 && sources[row+1].p==w.p) stride=sources[row+1].address-w.address;
                while(row+count<32 && sources[row+count].p==w.p && sources[row+count].address==w.address+count*stride) ++count;
                const auto ids=slices(w.p);
                for(int64_t byte=0;byte<2;++byte) {
                    const int64_t latency=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,ids[byte]);
                    emitMem3D(rewriter,op->getLoc(),target,capture+row-latency,
                        side*target.memory().slices_per_hemisphere+ids[byte],"read",w.address,16+byte,
                        count,1,stride,1,1,0,1,1,0,-1,bank(w.p));
                }
                row+=count;
            }
            for(int64_t pair=0;pair<8;++pair) for(int64_t byte=0;byte<2;++byte) {
                const int64_t dst=feedbackSlices[2*pair+byte];
                const int64_t latency=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,dst);
                emitMem3D(rewriter,op->getLoc(),target,capture+35+pair+latency,
                    side*target.memory().slices_per_hemisphere+dst,"write_tap",baseRow(feedbackInput)+block*4,32+side*16+byte,
                    4,8,1,1,1,0,1,1,0,-1,bank(feedbackInput));
                transposeEnd=std::max(transposeEnd,capture+35+pair+latency+28);
            }
        }
    }
    llvm::SmallVector<int64_t> arrivals;
    const int64_t routeLatency=*target.transport_latency(target::StreamEndpoint::SxmResult,
        target::StreamEndpoint::VxmInput,target::StreamDirection::West,0);
    for(int64_t block=0;block<blocks;++block)
        arrivals.push_back(firstCapture+block*32+35+routeLatency);
    createTimeline(rewriter,op->getLoc(),prefix+".direct_planar_ingress",0,transposeEnd);
    int64_t normalizedStreamStart=0;
    int64_t cycle=emitVxmFeedbackImpl(rewriter,op->getLoc(),inputValue,weight,inputType,
        epsilon,target,feedbackInput,weightPlacement,feedbackOutput,transposeEnd,l2Norm,
        outputScale,arrivals,bool(partitionedWords),&normalizedStreamStart);
    createTimeline(rewriter,op->getLoc(),prefix+".sxm_vxm_streaming",arrivals.front(),cycle);
    // Capture normalized vectors directly on the forward SR fabric. VXM
    // emits 32 consecutive feature vectors per wide-transpose tile, with
    // the existing superlane skew; no normalized-output MEM spill is needed.
    // The input tap/replay above is still needed for the reduction dependency.
    const int64_t firstSide=partitionedWords?0:(result.getAs<mlir::StringAttr>("hemisphere").getValue()=="west"?1:0);
    const int64_t capture=normalizedStreamStart+*target.transport_latency(
        target::StreamEndpoint::VxmResult,target::StreamEndpoint::SxmInput,
        target::StreamDirection::East,0);
    int64_t restored=cycle;
    for(int64_t side=firstSide;side<(partitionedWords?2:firstSide+1);++side) {
    const int64_t stream=(1-side)*8+2;
    emitSxm(rewriter,op->getLoc(),capture,side,"transpose",{stream,stream+1},{32,33},
        attention_detail::identityMap(target),"wide_matrix",-1,-1,-1,blocks,32);
    const auto outputSlices=slices(result);
    for(int64_t byte=0;byte<2;++byte) {
        const auto dst=outputSlices[byte];
        const int64_t latency=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,dst);
        emitMem3D(rewriter,op->getLoc(),target,capture+35+latency,
            side*target.memory().slices_per_hemisphere+dst,"write",baseRow(result),32+byte,
            blocks*32,1,1,1,1,0,1,1,0,-1,bank(result));
        restored=std::max(restored,capture+35+latency+blocks*32+4);
    }
    }
    createTimeline(rewriter,op->getLoc(),prefix+".vxm_sxm_direct",normalizedStreamStart,restored);
    createTimeline(rewriter,op->getLoc(),prefix+".direct_planar_output",capture,restored);
    auto output=createBinding(rewriter,op->getLoc(),{},outputIndex,"output","result",inputType,result);
    output->setAttr("name",rewriter.getStringAttr(prefix+".result."+std::to_string(outputIndex)));
    return output.getValue();
}

// A planar 32-token tile feeds both passes without distributed16 scratch.
// SXM -> VXM is tapped only to retain x across the full-width reduction;
// the normalized stream goes straight back through SXM into its final home.
mlir::Value emitStreamingPlanarRmsNorm(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    mlir::DictionaryAttr input, mlir::DictionaryAttr weight,
    mlir::DictionaryAttr replay, mlir::DictionaryAttr result, int64_t outputIndex)
{
    const auto type=llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const int64_t hidden=type.getDimSize(1), blocks=hidden/32;
    const auto inputIds=slices(input), replayIds=slices(replay), resultIds=slices(result);
    const auto latency=[&](target::StreamEndpoint src, target::StreamEndpoint dst,
                           target::StreamDirection direction,int64_t slice=0) {
        return *target.transport_latency(src,dst,direction,slice);
    };
    using E=target::StreamEndpoint;
    using D=target::StreamDirection;
    int64_t firstCapture=32;
    for(auto id:inputIds) firstCapture=std::max(firstCapture,latency(E::Mem,E::SxmInput,D::East,id)+32);
    int64_t replayEnd=0;
    for(int64_t side=0;side<2;++side) {
        emitSxm(rewriter,op.getLoc(),firstCapture,side,"transpose",{16,17},
            {32+side*16,33+side*16},attention_detail::identityMap(target),
            "wide_matrix",-1,-1,-1,blocks,32);
        for(int64_t byte=0;byte<2;++byte) {
            emitMem3D(rewriter,op.getLoc(),target,
                firstCapture-latency(E::Mem,E::SxmInput,D::East,inputIds[byte]),
                side*target.memory().slices_per_hemisphere+inputIds[byte],
                "read",baseRow(input),16+byte,hidden,1,1,1,1,0,1,1,0,
                inputBindingIndex(op.getInput()),bank(input));
            const int64_t tap=firstCapture+35+latency(E::SxmResult,E::Mem,D::West,replayIds[byte]);
            emitMem3D(rewriter,op.getLoc(),target,tap,
                side*target.memory().slices_per_hemisphere+replayIds[byte],
                "write_tap",baseRow(replay),32+side*16+byte,
                hidden,1,1,1,1,0,1,1,0,-1,bank(replay));
            replayEnd=std::max(replayEnd,tap+hidden+4);
        }
    }
    llvm::SmallVector<int64_t> arrivals;
    for(int64_t block=0;block<blocks;++block)
        arrivals.push_back(firstCapture+block*32+35+latency(E::SxmResult,E::VxmInput,D::West));
    int64_t normalized=0;
    const auto feedbackEnd=emitVxmFeedbackImpl(rewriter,op.getLoc(),op.getInput(),op.getWeight(),
        type,op.getEpsilon().convertToDouble(),target,replay,weight,result,replayEnd,
        false,1.0f,arrivals,false,&normalized);
    const int64_t capture=normalized+latency(E::VxmResult,E::SxmInput,D::East);
    int64_t end=feedbackEnd;
    for(int64_t side=0;side<2;++side) {
        const int64_t source=(1-side)*8+2;
        emitSxm(rewriter,op.getLoc(),capture,side,"transpose",{source,source+1},{32,33},
            attention_detail::identityMap(target),"wide_matrix",-1,-1,-1,blocks,32);
        for(int64_t byte=0;byte<2;++byte) {
            const auto write=capture+35+latency(E::SxmResult,E::Mem,D::West,resultIds[byte]);
            emitMem3D(rewriter,op.getLoc(),target,write,
                side*target.memory().slices_per_hemisphere+resultIds[byte],
                "write",baseRow(result),32+byte,hidden,1,1,1,1,0,1,1,0,-1,bank(result));
            end=std::max(end,write+hidden+4);
        }
    }
    createTimeline(rewriter,op.getLoc(),"rmsnorm.direct_planar_ingress",0,replayEnd);
    createTimeline(rewriter,op.getLoc(),"rmsnorm.sxm_vxm_streaming",arrivals.front(),arrivals.back()+32+4);
    createTimeline(rewriter,op.getLoc(),"rmsnorm.vxm_sxm_direct",normalized,end);
    createTimeline(rewriter,op.getLoc(),"rmsnorm.direct_planar_output",capture,end);
    auto output=createBinding(rewriter,op.getLoc(),{},outputIndex,"output","result",type,result);
    output->setAttr("name",rewriter.getStringAttr("rmsnorm.result."+std::to_string(outputIndex)));
    return output.getValue();
}

mlir::LogicalResult lowerRmsNormFeedback(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    int64_t outputIndex)
{
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const int64_t logicalRows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const int64_t rows = logicalRows == 1
        ? target.throughput().mxm_rows : logicalRows;
    const auto executionInputType = logicalRows == rows
        ? inputType
        : mlir::RankedTensorType::get(
            {rows, hidden}, inputType.getElementType());
    const auto inputPlacement =
        allocationPlacement(op.getInputAllocations(), 0);
    const auto weightPlacement =
        allocationPlacement(op.getWeightAllocations(), 0);
    const auto feedbackInputPlacement =
        allocationPlacement(op.getScratchAllocations(), 0);
    const auto resultPlacement =
        allocationPlacement(op.getResultAllocations(), 0);
    const auto planarAttr=op.getConfig().getAs<mlir::BoolAttr>("planar_streaming");
    const bool planar=planarAttr && planarAttr.getValue();
    const auto feedbackOutputPlacement = planar ? resultPlacement
        : allocationPlacement(op.getScratchAllocations(), 1);

    rewriter.setInsertionPoint(op);
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(op.getInput()))
        createBinding(rewriter, op.getLoc(), op.getInput(),
            argument.getArgNumber(), "input", "activation",
            inputType, inputPlacement);
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(op.getWeight()))
        createBinding(rewriter, op.getLoc(), op.getWeight(),
            argument.getArgNumber(), "input", "weight",
            llvm::cast<mlir::RankedTensorType>(op.getWeight().getType()),
            weightPlacement);

    if(planar) {
        rewriter.replaceOp(op,emitStreamingPlanarRmsNorm(rewriter,op,target,
            inputPlacement,weightPlacement,feedbackInputPlacement,resultPlacement,outputIndex));
        return mlir::success();
    }
    if(op->hasAttr("native_planar_words")) {
        auto value=emitDirectPlanarNorm(rewriter,op,op.getInput(),op.getWeight(),executionInputType,
            target,feedbackInputPlacement,feedbackOutputPlacement,resultPlacement,weightPlacement,
            op.getEpsilon().convertToDouble(),false,1.0f,outputIndex);
        if(mlir::failed(value)) return mlir::failure();
        rewriter.replaceOp(op,*value);return mlir::success();
    }
    const auto inputKind =
        inputPlacement.getAs<mlir::StringAttr>("kind");
    const bool vxmInput = inputKind
        && inputKind.getValue() == "fp16_vxm_distributed_16";
    const bool mxmInput = inputKind
        && inputKind.getValue() == "fp16_mxm_distributed_16";
    if (!vxmInput && !mxmInput)
        return op.emitError(
            "feedback RMSNorm requires MXM- or VXM-oriented distributed16 input");
    int64_t gammaCopyEnd = 0;
    auto executionWeightPlacement = weightPlacement;
    if (op.getScratchAllocations().size() > 2) {
        executionWeightPlacement = allocationPlacement(op.getScratchAllocations(), 2);
        const auto sourceSlices = slices(weightPlacement);
        const auto destinationSlices = slices(executionWeightPlacement);
        const int64_t segmentRows = weightPlacement
            .getAs<mlir::IntegerAttr>("instruction_count").getInt();
        for (int64_t segment = 0; segment * segmentRows < hidden; ++segment) {
            const int64_t count = std::min(segmentRows, hidden - segment * segmentRows);
            int64_t end = gammaCopyEnd;
            for (int64_t hemisphere = 0; hemisphere < target.memory().hemispheres;
                 ++hemisphere)
                for (int64_t byte = 0; byte < 2; ++byte) {
                    const int64_t src = sourceSlices[2 * segment + byte];
                    const int64_t dst = destinationSlices[byte];
                    const int64_t groups = target.streams().mem_slices_per_register_group;
                    const int64_t latency = dst / groups - src / groups;
                    if (latency <= 0)
                        return op.emitError("compact gamma copy requires a later MEM group");
                    emitMem(rewriter, op.getLoc(), gammaCopyEnd,
                        hemisphere * target.memory().slices_per_hemisphere + src,
                        "read", baseRow(weightPlacement), byte, count, 1, 1,
                        "sram", inputBindingIndex(op.getWeight()), bank(weightPlacement));
                    emitMem(rewriter, op.getLoc(), gammaCopyEnd + latency,
                        hemisphere * target.memory().slices_per_hemisphere + dst,
                        "write", baseRow(executionWeightPlacement) + segment * segmentRows,
                        byte, count, 1, 1, "sram", -1, bank(executionWeightPlacement));
                    end = std::max(end, gammaCopyEnd + latency + count);
                }
            gammaCopyEnd = end;
        }
        // MEM instructions launch at tile zero; the last vector must drain
        // through every physical tile before this materialization is ready.
        gammaCopyEnd += target.throughput().tile_rows - 1;
        createTimeline(rewriter, op.getLoc(), "rmsnorm.materialize_gamma", 0, gammaCopyEnd);
    }
    const int64_t inputTransposeEnd = vxmInput
        ? gammaCopyEnd
        : emitDistributedMatrixTranspose(
              rewriter, op.getLoc(), target, inputPlacement,
              feedbackInputPlacement, rows, hidden, gammaCopyEnd);
    const auto feedbackInput = vxmInput
        ? inputPlacement : feedbackInputPlacement;
    const auto weightKind =
        weightPlacement.getAs<mlir::StringAttr>("kind");
    const bool distributedWeight = weightKind
        && (weightKind.getValue() == "fp16_vxm_distributed_16"
            || weightKind.getValue() == "fp16_vxm_gamma_broadcast");
    if (!distributedWeight)
        return op.emitError(
            "feedback RMSNorm requires VXM-oriented distributed16 gamma");
    const int64_t weightTransposeEnd = inputTransposeEnd;
    const int64_t feedbackEnd = emitVxmFeedbackRmsNorm(
        rewriter, op.getLoc(), op.getInput(), op.getWeight(),
        executionInputType, op.getEpsilon().convertToDouble(), target,
        feedbackInput,
        executionWeightPlacement, feedbackOutputPlacement,
        weightTransposeEnd);
    const auto resultKind =
        resultPlacement.getAs<mlir::StringAttr>("kind");
    const bool vxmResult = resultKind
        && resultKind.getValue() == "fp16_vxm_distributed_16";
    const bool mxmResult = resultKind
        && resultKind.getValue() == "fp16_mxm_distributed_16";
    if (!vxmResult && !mxmResult)
        return op.emitError(
            "feedback RMSNorm requires MXM- or VXM-oriented distributed16 result");
    const int64_t restoreEnd = vxmResult
        ? feedbackEnd
        : emitDistributedMatrixTranspose(
              rewriter, op.getLoc(), target,
              feedbackOutputPlacement, resultPlacement,
              rows, hidden, feedbackEnd);

    if (!vxmInput)
        createTimeline(rewriter, op.getLoc(), "rmsnorm.transpose_input",
            gammaCopyEnd, inputTransposeEnd);
    createTimeline(rewriter, op.getLoc(), "rmsnorm.feedback",
        weightTransposeEnd, feedbackEnd);
    if (!vxmResult)
        createTimeline(rewriter, op.getLoc(), "rmsnorm.restore_layout",
            feedbackEnd, restoreEnd);
    auto output = createBinding(rewriter, op.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType()),
        resultPlacement);
    output->setAttr("name", rewriter.getStringAttr(
        "rmsnorm.result." + std::to_string(outputIndex)));
    rewriter.replaceOp(op, output.getValue());
    return mlir::success();
}

mlir::LogicalResult lowerRmsNorm(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    int64_t outputIndex)
{
    return lowerRmsNormFeedback(rewriter, op, target, outputIndex);
}

} // namespace

int64_t emitVxmFeedbackRmsNorm(mlir::IRRewriter& rewriter,
    mlir::Location location, mlir::Value input, mlir::Value weight,
    mlir::RankedTensorType inputType, double epsilon,
    const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr weightPlacement,
    mlir::DictionaryAttr outputPlacement, int64_t start)
{
    return emitVxmFeedbackImpl(rewriter, location, input, weight, inputType,
        epsilon, target, inputPlacement, weightPlacement, outputPlacement,
        start);
}

mlir::LogicalResult lowerL2NormSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::L2NormTaskOp> operations;
    function.walk([&](stream::L2NormTaskOp op) { operations.push_back(op); });
    int64_t outputIndex = 0;
    for (auto op : operations) {
        const auto inputType = op.getInput().getType();
        const auto input = allocationPlacement(op.getInputAllocations(), 0);
        const auto result = allocationPlacement(op.getResultAllocations(), 0);
        const auto feedbackInput = allocationPlacement(op.getScratchAllocations(), 0);
        const auto feedbackOutput = allocationPlacement(op.getScratchAllocations(), 1);
        if (input.getAs<mlir::StringAttr>("kind").getValue() != "fp16_mxm_distributed_16"
            || (result.getAs<mlir::StringAttr>("kind").getValue() != "fp16_mxm_distributed_16"
                && !(op->hasAttr("native_planar_words") &&
                    (result.getAs<mlir::StringAttr>("kind").getValue()=="fp16_pair_planar"
                     || result.getAs<mlir::StringAttr>("kind").getValue()=="token_partition_planar"))))
            return op.emitError("L2Norm requires MXM distributed16 input and result");
        rewriter.setInsertionPoint(op);
        if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(op.getInput()))
            createBinding(rewriter, op.getLoc(), op.getInput(), arg.getArgNumber(),
                "input", "activation", inputType, input);
        if (op->hasAttr("native_planar_words")) {
            auto value=emitDirectPlanarNorm(rewriter,op,op.getInput(),op.getInput(),inputType,
                target,feedbackInput,feedbackOutput,result,{},op.getEpsilon().convertToDouble(),
                true,op.getScale().convertToFloat(),outputIndex++);
            if(mlir::failed(value)) return mlir::failure();
            rewriter.replaceOp(op,*value);continue;
        }
        if (auto groups = op->getAttrOfType<mlir::ArrayAttr>("native_input_groups")) {
            const int64_t rows = op->getAttrOfType<mlir::IntegerAttr>("native_input_rows").getInt();
            const int64_t hidden = inputType.getDimSize(1);
            const int64_t tile = target.throughput().mxm_rows;
            const int64_t words = hidden / target.throughput().lanes_per_tile;
            const auto tileType = mlir::RankedTensorType::get(
                {tile, hidden}, inputType.getElementType());
            const auto offset = [&](mlir::DictionaryAttr p, int64_t n) {
                mlir::NamedAttrList attrs(p);
                attrs.set("base_row", rewriter.getI64IntegerAttr(baseRow(p) + n));
                return attrs.getDictionary(rewriter.getContext());
            };
            bool nativePartition = op->getParentOfType<mlir::func::FuncOp>()->hasAttr("ftlpu.l2_native_partition_output") && hidden==128;
            const auto wideOutput=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
            const auto headsOutput=op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.l2_head_partition");
            nativePartition &= wideOutput && wideOutput.getValue() && headsOutput && headsOutput.getValue();
            for(int64_t g=0; nativePartition && g<static_cast<int64_t>(groups.size()); ++g) {
                auto d=llvm::cast<mlir::DictionaryAttr>(groups[g]);
                auto p=d.getAs<mlir::DictionaryAttr>("placement");
                nativePartition = p && !d.getAs<mlir::ArrayAttr>("chunks") &&
                    (p.getAs<mlir::StringAttr>("hemisphere").getValue()=="east") == (g%2==0);
            }
            int64_t cycle = 0;
            // The physical output remains [tokens, groups * hidden]. Logical
            // [tokens * groups, hidden] is a reshape of the same flat tensor.
            for (int64_t block = 0; block < rows / tile; ++block)
                for (int64_t g = 0; g < static_cast<int64_t>(groups.size()); ++g) {
                    auto descriptor = llvm::cast<mlir::DictionaryAttr>(groups[g]);
                    struct FeedbackCopy { int64_t side, destination, words; };
                    llvm::SmallVector<FeedbackCopy> copies;
                    const auto widePolicy = op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.sxm_wide");
                    const bool pipeline = widePolicy && widePolicy.getValue();
                    const auto headPolicy = op->getParentOfType<mlir::ModuleOp>()->getAttrOfType<mlir::BoolAttr>("ftlpu.l2_head_partition");
                    if (pipeline && headPolicy && headPolicy.getValue()
                        && !descriptor.getAs<mlir::ArrayAttr>("chunks") && g + 1 < static_cast<int64_t>(groups.size())) {
                        auto next = llvm::cast<mlir::DictionaryAttr>(groups[g + 1]);
                        auto left = descriptor.getAs<mlir::DictionaryAttr>("placement");
                        auto right = next.getAs<mlir::DictionaryAttr>("placement");
                        if (next.getAs<mlir::ArrayAttr>("chunks") || !left || !right
                            || left.getAs<mlir::StringAttr>("hemisphere") == right.getAs<mlir::StringAttr>("hemisphere"))
                            return op.emitError("head-partitioned L2 requires paired independent hemispheres");
                        const int64_t firstCapture = cycle + target.throughput().mem_to_sxm_latency;
                        const auto saved = slices(feedbackInput);
                        int64_t allWrites = cycle;
                        for (int64_t member = 0; member < 2; ++member) {
                            auto d = member ? next : descriptor;
                            auto source = d.getAs<mlir::DictionaryAttr>("placement");
                            const int64_t side = source.getAs<mlir::StringAttr>("hemisphere").getValue() == "west" ? 1 : 0;
                            source = offset(source, (block * d.getAs<mlir::IntegerAttr>("columns").getInt()
                                + d.getAs<mlir::IntegerAttr>("column").getInt()) / 8);
                            if (bank(source) == bank(feedbackInput))
                                return op.emitError("streaming L2 source and retention require different banks");
                            const auto src = slices(source);
                            for (int64_t pair = 0; pair < 8; ++pair) for (int64_t byte = 0; byte < 2; ++byte) {
                                const auto rlat = *target.transport_latency(target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::SxmInput,target::StreamDirection::East,src[2*pair+byte]);
                                emitMem3D(rewriter,op.getLoc(),target,firstCapture+pair-rlat,
                                    side*target.memory().slices_per_hemisphere+src[2*pair+byte],"read",baseRow(source),16+byte,
                                    4,8,1,hidden/32,32,4,1,1,0,-1,bank(source));
                                const auto dst = saved[2*pair+byte];
                                const auto wlat = *target.transport_latency(target::StreamEndpoint::SxmResult,
                                    target::StreamEndpoint::Mem,target::StreamDirection::West,dst);
                                emitMem3D(rewriter,op.getLoc(),target,firstCapture+35+pair+wlat,
                                    side*target.memory().slices_per_hemisphere+dst,"write_tap",baseRow(feedbackInput),32+side*16+byte,
                                    4,8,1,hidden/32,32,4,1,1,0,-1,bank(feedbackInput));
                                allWrites = std::max(allWrites, firstCapture+35+pair+wlat+hidden-4);
                            }
                            emitSxm(rewriter,op.getLoc(),firstCapture,side,"transpose",{16,17},{32+side*16,33+side*16},
                                attention_detail::identityMap(target),"wide_matrix",-1,-1,-1,hidden/32,32);
                        }
                        llvm::SmallVector<int64_t> arrivals;
                        for (int64_t c = 0; c < hidden / 32; ++c)
                            arrivals.push_back(firstCapture+c*32+35+target.streams().system_register_columns-1);
                        const auto begin = cycle;
                        int64_t normalizedStreamStart = 0;
                        cycle = emitVxmFeedbackImpl(rewriter,op.getLoc(),op.getInput(),op.getInput(),tileType,
                            op.getEpsilon().convertToDouble(),target,feedbackInput,{},feedbackOutput,allWrites,
                            true,op.getScale().convertToFloat(),arrivals,true,&normalizedStreamStart);
                        createTimeline(rewriter,op.getLoc(),"l2norm.sxm_vxm_streaming",begin,cycle);
                        // The normalized vectors already travel forward from
                        // VXM to SXM. Capture them in-place on the SR fabric,
                        // without writing/reading the feedbackOutput scratch.
                        const auto restoreBegin = normalizedStreamStart;
                        const auto capture = normalizedStreamStart + *target.transport_latency(
                            target::StreamEndpoint::VxmResult,target::StreamEndpoint::SxmInput,
                            target::StreamDirection::East,0);
                        int64_t restoreEnd = cycle;
                        for (int64_t member = 0; member < 2; ++member) {
                            auto d = member ? next : descriptor;
                            const int64_t side = d.getAs<mlir::DictionaryAttr>("placement").getAs<mlir::StringAttr>("hemisphere").getValue() == "west" ? 0 : 1;
                            const int64_t stream=(1-side)*8+2;
                            auto destination=offset(result,(block*groups.size()+g+member)*words);
                            const auto dst=slices(destination);
                            emitSxm(rewriter,op.getLoc(),capture,side,"transpose",
                                {stream,stream+1},{32,33},attention_detail::identityMap(target),
                                "wide_matrix",-1,-1,-1,hidden/32,32);
                            for(int64_t pair=0;pair<8;++pair)for(int64_t byte=0;byte<2;++byte) {
                                const auto slice=dst[2*pair+byte];
                                const auto latency=*target.transport_latency(target::StreamEndpoint::SxmResult,
                                    target::StreamEndpoint::Mem,target::StreamDirection::West,slice);
                                if(auto stripe=destination.getAs<mlir::IntegerAttr>("bank_tile_rows")) {
                                    if(!nativePartition || stripe.getInt()!=4 || hidden!=128)
                                        return op.emitError("bank-striped L2 requires native paired D128 heads");
                                    for(int64_t parity=0;parity<2;++parity)
                                        emitMem3D(rewriter,op.getLoc(),target,capture+35+pair+latency+parity*32,
                                            side*target.memory().slices_per_hemisphere+slice,"write",baseRow(destination)+parity*4,32+byte,
                                            4,8,1,hidden/64,64,8,1,1,0,-1,bank(destination)^parity);
                                } else {
                                emitMem3D(rewriter,op.getLoc(),target,capture+35+pair+latency,
                                    side*target.memory().slices_per_hemisphere+slice,"write",baseRow(destination),32+byte,
                                    4,8,1,hidden/32,32,4,1,1,0,-1,bank(destination));
                                }
                                restoreEnd=std::max(restoreEnd,capture+35+pair+latency+hidden-8+1);
                            }
                        }
                        cycle = restoreEnd;
                        createTimeline(rewriter,op.getLoc(),"l2norm.vxm_sxm_direct",restoreBegin,cycle);
                        // The public distributed16 binding currently promises both
                        // hemispheres. Materialize that contract only at the output,
                        // never before or during the independent head reductions.
                        if (!nativePartition) {
                        const auto exportBegin = cycle;
                        const auto ids = slices(result);
                        for (int64_t member = 0; member < 2; ++member) {
                            auto d = member ? next : descriptor;
                            const int64_t side = d.getAs<mlir::DictionaryAttr>("placement").getAs<mlir::StringAttr>("hemisphere").getValue() == "west" ? 0 : 1;
                            int64_t drain = cycle;
                            for (int64_t plane = 0; plane < static_cast<int64_t>(ids.size()); ++plane) {
                                const int64_t latency = 2*(ids[plane]/target.streams().mem_slices_per_register_group)+2;
                                const int64_t row = baseRow(result)+(block*groups.size()+g+member)*words;
                                emitMem3D(rewriter,op.getLoc(),target,cycle,side*target.memory().slices_per_hemisphere+ids[plane],
                                    "read",row,48+plane,words,1,1,1,1,0,1,1,0,-1,bank(result));
                                emitMem3D(rewriter,op.getLoc(),target,cycle+latency,(1-side)*target.memory().slices_per_hemisphere+ids[plane],
                                    "write",row,16+plane,words,1,1,1,1,0,1,1,0,-1,bank(result));
                                drain = std::max(drain,cycle+latency+words+1);
                            }
                            cycle = drain;
                        }
                        createTimeline(rewriter,op.getLoc(),"l2norm.output_binding_mirror",exportBegin,cycle);
                        }
                        ++g;
                        continue;
                    }
                    if(pipeline && descriptor.getAs<mlir::ArrayAttr>("chunks")) {
                        const auto chunks=descriptor.getAs<mlir::ArrayAttr>("chunks");
                        struct Source { mlir::DictionaryAttr p; int64_t side; };
                        llvm::SmallVector<Source> sources;
                        for(auto attr:chunks) {
                            auto d=llvm::cast<mlir::DictionaryAttr>(attr);
                            auto p=d.getAs<mlir::DictionaryAttr>("placement");
                            p=offset(p,(block*d.getAs<mlir::IntegerAttr>("columns").getInt()+d.getAs<mlir::IntegerAttr>("column").getInt())/8);
                            sources.push_back({p,p.getAs<mlir::StringAttr>("hemisphere").getValue()=="west"?1:0});
                        }
                        const auto staged=slices(feedbackOutput), saved=slices(feedbackInput);
                        const int64_t gatherBegin=cycle;
                        // Move only the missing source half before transpose.
                        // The original half remains at its producer address.
                        for(int64_t side=0;side<2;++side) {
                            int64_t issue=cycle,drain=cycle;
                            for(size_t c=0;c<sources.size();++c) if(sources[c].side==side) {
                                const auto src=slices(sources[c].p);
                                for(int64_t plane=0;plane<16;++plane) {
                                    const auto latency=src[plane]/4+staged[plane]/4+2;
                                    emitMem3D(rewriter,op.getLoc(),target,issue,
                                        side*target.memory().slices_per_hemisphere+src[plane],"read",baseRow(sources[c].p),48+plane,
                                        4,1,1,1,1,0,1,1,0,-1,bank(sources[c].p));
                                    emitMem3D(rewriter,op.getLoc(),target,issue+latency,
                                        (1-side)*target.memory().slices_per_hemisphere+staged[plane],"write",baseRow(feedbackOutput)+c*4,16+plane,
                                        4,1,1,1,1,0,1,1,0,-1,bank(feedbackOutput));
                                    drain=std::max(drain,issue+latency+5);
                                }
                                issue+=4;
                            }
                            cycle=drain;
                        }
                        createTimeline(rewriter,op.getLoc(),"l2norm.prepare_sxm_pair",gatherBegin,cycle);
                        llvm::SmallVector<int64_t> arrivals;
                        int64_t launch=cycle,allWrites=cycle;
                        for(size_t c=0;c<sources.size();++c) {
                            // A full tile finishes writing before a reused MEM
                            // bank queue reads the following source tile.
                            const int64_t capture=launch+target.throughput().mem_to_sxm_latency;
                            int64_t drain=capture+67;
                            for(int64_t side=0;side<2;++side) {
                                const auto source=side==sources[c].side?sources[c].p:offset(feedbackOutput,c*4);
                                const auto src=slices(source);
                                for(int64_t pair=0;pair<8;++pair) for(int64_t byte=0;byte<2;++byte) {
                                    const auto rlat=*target.transport_latency(target::StreamEndpoint::Mem,target::StreamEndpoint::SxmInput,target::StreamDirection::East,src[2*pair+byte]);
                                    emitMem3D(rewriter,op.getLoc(),target,capture+pair-rlat,
                                        side*target.memory().slices_per_hemisphere+src[2*pair+byte],"read",baseRow(source),16+byte,
                                        4,8,1,1,1,0,1,1,0,-1,bank(source));
                                    const auto dst=saved[2*pair+byte];
                                    const auto wlat=*target.transport_latency(target::StreamEndpoint::SxmResult,target::StreamEndpoint::Mem,target::StreamDirection::West,dst);
                                    emitMem3D(rewriter,op.getLoc(),target,capture+35+pair+wlat,
                                        side*target.memory().slices_per_hemisphere+dst,"write_tap",baseRow(feedbackInput)+c*4,32+side*16+byte,
                                        4,8,1,1,1,0,1,1,0,-1,bank(feedbackInput));
                                    drain=std::max(drain,capture+35+pair+wlat+28);
                                }
                                emitSxm(rewriter,op.getLoc(),capture,side,"transpose",{16,17},{32+side*16,33+side*16},
                                    attention_detail::identityMap(target),"wide_matrix");
                            }
                            arrivals.push_back(capture+35+target.streams().system_register_columns-1);
                            allWrites=std::max(allWrites,drain);launch=drain;
                        }
                        const auto begin=arrivals.front();
                        cycle=emitVxmFeedbackImpl(rewriter,op.getLoc(),op.getInput(),op.getInput(),tileType,
                            op.getEpsilon().convertToDouble(),target,feedbackInput,{},feedbackOutput,allWrites,
                            true,op.getScale().convertToFloat(),arrivals);
                        createTimeline(rewriter,op.getLoc(),"l2norm.sxm_vxm_streaming",begin,cycle);
                        const auto restore=cycle;
                        cycle=emitDistributedMatrixTranspose(rewriter,op.getLoc(),target,feedbackOutput,
                            offset(result,(block*groups.size()+g)*words),tile,hidden,cycle);
                        createTimeline(rewriter,op.getLoc(),"l2norm.restore_layout",restore,cycle);
                        continue;
                    }
                    int64_t transposeLaunch=cycle, transposeEnd=cycle;
                    int64_t sideDrain[2]={cycle,cycle};
                    const auto transposePart = [&](mlir::DictionaryAttr part, int64_t destination, int64_t partWidth) {
                    const int64_t partWords = partWidth / target.throughput().lanes_per_tile;
                    auto source = part.getAs<mlir::DictionaryAttr>("placement");
                    const int64_t columns = part.getAs<mlir::IntegerAttr>("columns").getInt();
                    const int64_t column = part.getAs<mlir::IntegerAttr>("column").getInt();
                    source = offset(source, (block * columns + column) /
                        target.throughput().lanes_per_tile);
                    const auto hemisphere = source.getAs<mlir::StringAttr>("hemisphere").getValue();
                    const int64_t side = hemisphere == "both" ? -1 : hemisphere == "west" ? 1 : 0;
                    int64_t begin = pipeline ? transposeLaunch : cycle;
                    // A MEM bank ICU cannot interleave the next input read
                    // with the previous output domain on the same hemisphere.
                    if(side>=0 && bank(source)==bank(feedbackInput)) {
                        const auto srcSlices=slices(source), dstSlices=slices(feedbackInput);
                        bool shared=false;
                        for(auto src:srcSlices) shared |= llvm::is_contained(dstSlices,src);
                        if(shared) begin=std::max(begin,sideDrain[side]);
                    }
                    const int64_t end = emitDistributedMatrixTranspose(rewriter, op.getLoc(),
                        target, source, offset(feedbackInput, destination), tile, partWidth, begin, side);
                    transposeEnd=std::max(transposeEnd,end);
                    if(side>=0) sideDrain[side]=end;
                    else sideDrain[0]=sideDrain[1]=end;
                    transposeLaunch=begin+partWidth;
                    cycle=end;
                    createTimeline(rewriter, op.getLoc(), "l2norm.transpose_native_group", begin, end);
                    if(side>=0) copies.push_back({side,destination,partWords});
                    };
                    if (auto chunks = descriptor.getAs<mlir::ArrayAttr>("chunks")) {
                        for (int64_t c = 0; c < static_cast<int64_t>(chunks.size()); ++c)
                            transposePart(llvm::cast<mlir::DictionaryAttr>(chunks[c]), c * 4, 32);
                    } else transposePart(descriptor, 0, hidden);
                    cycle=transposeEnd;
                    // Copy the missing hemisphere words as one streaming phase,
                    // after all transposes drain. Do not drain the SR fabric
                    // separately after every 32-feature chunk.
                    const int64_t copyBegin=cycle;
                    int64_t copyCursor=cycle,copyEnd=cycle;
                    const auto ids=slices(feedbackInput);
                    std::stable_sort(copies.begin(),copies.end(),[](const auto &a,const auto &b){return a.side<b.side;});
                    int64_t previousSide=-1;
                    for(const auto &copy:copies) {
                        if(previousSide>=0 && previousSide!=copy.side) copyCursor=copyEnd;
                        previousSide=copy.side;
                        for(int64_t p=0;p<static_cast<int64_t>(ids.size());++p) {
                            const int64_t slice=ids[p], latency=2*(slice/target.streams().mem_slices_per_register_group)+2;
                            emitMem3D(rewriter,op.getLoc(),target,copyCursor,
                                copy.side*target.memory().slices_per_hemisphere+slice,"read",
                                baseRow(feedbackInput)+copy.destination,48+p,copy.words,1,1,1,1,0,1,1,0,-1,bank(feedbackInput));
                            emitMem3D(rewriter,op.getLoc(),target,copyCursor+latency,
                                (1-copy.side)*target.memory().slices_per_hemisphere+slice,"write",
                                baseRow(feedbackInput)+copy.destination,16+p,copy.words,1,1,1,1,0,1,1,0,-1,bank(feedbackInput));
                            copyEnd=std::max(copyEnd,copyCursor+latency+copy.words+1);
                        }
                        copyCursor+=copy.words;
                    }
                    cycle=copyEnd;
                    if(!copies.empty()) createTimeline(rewriter,op.getLoc(),"l2norm.replicate_feedback_tile",copyBegin,cycle);
                    const int64_t feedbackBegin = cycle;
                    cycle = emitVxmFeedbackImpl(rewriter, op.getLoc(), op.getInput(),
                        op.getInput(), tileType, op.getEpsilon().convertToDouble(),
                        target, feedbackInput, {}, feedbackOutput, cycle,
                        true, op.getScale().convertToFloat());
                    createTimeline(rewriter, op.getLoc(), "l2norm.feedback", feedbackBegin, cycle);
                    const int64_t restoreBegin = cycle;
                    cycle = emitDistributedMatrixTranspose(rewriter, op.getLoc(), target,
                        feedbackOutput, offset(result, (block * groups.size() + g) * words),
                        tile, hidden, cycle);
                    createTimeline(rewriter, op.getLoc(), "l2norm.restore_layout", restoreBegin, cycle);
                }
            auto resultLayout=result;
            if(nativePartition) {
                mlir::NamedAttrList attrs(result);
                attrs.set("kind",rewriter.getStringAttr("fp16_mxm_block4_partition_distributed_16"));
                resultLayout=attrs.getDictionary(rewriter.getContext());
            }
            auto output = createBinding(rewriter, op.getLoc(), {}, outputIndex,
                "output", "result", inputType, resultLayout);
            output->setAttr("name", rewriter.getStringAttr("l2norm.result." + std::to_string(outputIndex++)));
            rewriter.replaceOp(op, output.getValue());
            continue;
        }
        const int64_t transposeEnd = emitDistributedMatrixTranspose(rewriter,
            op.getLoc(), target, input, feedbackInput,
            inputType.getDimSize(0), inputType.getDimSize(1), 0);
        // The second SSA operand of VXM schedule ops is bookkeeping only for
        // immediate RHS instructions. L2 has one input and no gamma binding.
        const int64_t end = emitVxmFeedbackImpl(rewriter, op.getLoc(),
            op.getInput(), op.getInput(), inputType, op.getEpsilon().convertToDouble(),
            target, feedbackInput, {}, feedbackOutput, transposeEnd,
            true, op.getScale().convertToFloat());
        if (end <= 0) return op.emitError("invalid L2Norm feedback placement");
        const int64_t restoreEnd = emitDistributedMatrixTranspose(rewriter,
            op.getLoc(), target, feedbackOutput, result,
            inputType.getDimSize(0), inputType.getDimSize(1), end);
        createTimeline(rewriter, op.getLoc(), "l2norm.transpose_input", 0, transposeEnd);
        createTimeline(rewriter, op.getLoc(), "l2norm.feedback", transposeEnd, end);
        createTimeline(rewriter, op.getLoc(), "l2norm.restore_layout", end, restoreEnd);
        auto output = createBinding(rewriter, op.getLoc(), {}, outputIndex,
            "output", "result", inputType, result);
        output->setAttr("name", rewriter.getStringAttr("l2norm.result." + std::to_string(outputIndex++)));
        rewriter.replaceOp(op, output.getValue());
    }
    return mlir::success();
}

mlir::LogicalResult lowerRmsNormSchedules(mlir::IRRewriter& rewriter,
    mlir::func::FuncOp function, const target::LPUTargetModel& target)
{
    llvm::SmallVector<stream::RmsNormTaskOp> operations;
    function.walk(
        [&](stream::RmsNormTaskOp op) { operations.push_back(op); });
    int64_t outputIndex = 0;
    for (stream::RmsNormTaskOp op : operations) {
        if (mlir::failed(
                lowerRmsNorm(rewriter, op, target, outputIndex++)))
            return mlir::failure();
    }
    return mlir::success();
}

} // namespace ftlpu::compiler::schedule
