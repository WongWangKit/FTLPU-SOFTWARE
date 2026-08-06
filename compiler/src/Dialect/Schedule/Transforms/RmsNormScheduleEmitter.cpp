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
using attention_detail::emitMxm;
using attention_detail::emitSxm;
using attention_detail::emitWavefrontBeat;
using attention_detail::emitWavefrontTail;
using attention_detail::blockDiagonalMap;
using ffn_detail::create_vxm;
using ffn_detail::schedule_placement;

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

struct TileAddress {
    std::array<int64_t, 2> slices;
    int64_t row;
};

int64_t baseRow(mlir::DictionaryAttr placement)
{
    return placement.getAs<mlir::IntegerAttr>("base_row").getInt();
}

mlir::FailureOr<TileAddress> tileAddress(
    mlir::DictionaryAttr placement, int64_t block, int64_t rows)
{
    const auto layout = placement.getAs<mlir::StringAttr>("kind");
    const auto memorySlices = slices(placement);
    if (!layout || memorySlices.size() < 2) return mlir::failure();

    if (layout.getValue() == "fp16_mxm_activation_planar") {
        return TileAddress {
            {memorySlices[0], memorySlices[1]},
            baseRow(placement) + block * rows,
        };
    }
    if (layout.getValue() == "fp16_pair_planar") {
        const int64_t pair = block % 2;
        if (memorySlices.size() < static_cast<std::size_t>(2 * pair + 2))
            return mlir::failure();
        return TileAddress {
            {memorySlices[2 * pair], memorySlices[2 * pair + 1]},
            baseRow(placement) + (block / 2) * rows,
        };
    }
    return mlir::failure();
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
                            hemisphere == 0 ? 0 : 32,
                            hemisphere == 0 ? 1 : 33,
                        };
                        const std::array<int64_t, 2> transposeStreams {
                            hemisphere == 0 ? 16 : 48,
                            hemisphere == 0 ? 17 : 49,
                        };
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = inputSlices[byte];
                            const auto direction = hemisphere == 0
                                ? target::StreamDirection::East
                                : target::StreamDirection::West;
                            const int64_t latency =
                                *target.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::SxmInput,
                                    direction, slice);
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
                                hemisphere == 0 ? byte : 32 + byte,
                                1, 1, 0);
                        }
                        emitSxm(rewriter, location, capture + row,
                            hemisphere, "transpose", sourceStreams,
                            transposeStreams,
                            attention_detail::identityMap());
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
                            hemisphere == 0 ? 16 : 48,
                            hemisphere == 0 ? 17 : 49,
                        };
                        const std::array<int64_t, 2> outputStreams {
                            hemisphere == 0 ? 32 : 0,
                            hemisphere == 0 ? 33 : 1,
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
                                const auto direction = hemisphere == 0
                                    ? target::StreamDirection::West
                                    : target::StreamDirection::East;
                                const int64_t latency =
                                    *target.transport_latency(
                                        target::StreamEndpoint::SxmResult,
                                        target::StreamEndpoint::Mem,
                                        direction, slice);
                                emitMem(rewriter, location,
                                    permute + row + latency,
                                    hemisphere
                                            * target.memory().
                                                slices_per_hemisphere
                                        + slice,
                                    "write", address,
                                    hemisphere == 0 ? 32 + byte : byte,
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
                        hemisphere == 0 ? 0 : 32,
                        hemisphere == 0 ? 1 : 33,
                    };
                    const std::array<int64_t, 2> transposeStreams {
                        hemisphere == 0 ? 16 : 48,
                        hemisphere == 0 ? 17 : 49,
                    };
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        const int64_t slice = inputSlices[byte];
                        const auto direction = hemisphere == 0
                            ? target::StreamDirection::East
                            : target::StreamDirection::West;
                        const int64_t latency =
                            *target.transport_latency(
                                target::StreamEndpoint::Mem,
                                target::StreamEndpoint::SxmInput,
                                direction, slice);
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
                            hemisphere == 0 ? byte : 32 + byte,
                            1, 1, 0);
                    }
                    emitSxm(rewriter, location, capture + row,
                        hemisphere, "transpose", sourceStreams,
                        transposeStreams,
                        attention_detail::identityMap(),
                        "matrix_columns");
                }
                const int64_t emit = capture + tile
                    + target.throughput().tile_rows + 1;
                for (int64_t row = 0; row < tile; ++row) {
                    const std::array<int64_t, 2> transposeStreams {
                        hemisphere == 0 ? 16 : 48,
                        hemisphere == 0 ? 17 : 49,
                    };
                    const std::array<int64_t, 2> outputStreams {
                        hemisphere == 0 ? 32 : 0,
                        hemisphere == 0 ? 33 : 1,
                    };
                    emitSxm(rewriter, location, emit + row,
                        hemisphere, "permute", transposeStreams,
                        outputStreams, attention_detail::identityMap(),
                        "matrix_columns");
                    for (int64_t byte = 0; byte < 2; ++byte) {
                        for (int64_t duplicate = byte;
                             duplicate
                                 < static_cast<int64_t>(
                                     outputSlices.size());
                             duplicate += 2) {
                            const int64_t slice = outputSlices[duplicate];
                            const auto direction = hemisphere == 0
                                ? target::StreamDirection::West
                                : target::StreamDirection::East;
                            const int64_t latency =
                                *target.transport_latency(
                                    target::StreamEndpoint::SxmResult,
                                    target::StreamEndpoint::Mem,
                                    direction, slice);
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
                                hemisphere == 0 ? 32 + byte : byte,
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

std::array<int64_t, 32> rotateTileMap(int64_t distance,
    const target::LPUTargetModel& target)
{
    auto map = attention_detail::identityMap();
    const int64_t tileRows = target.throughput().tile_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
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
    int64_t hiddenBlocks)
{
    return baseRow(placement)
        + (tokenBlock * hiddenBlocks + hiddenBlock) * 4 + wave;
}

int64_t emitDistributedMatrixTranspose(mlir::IRRewriter& rewriter,
    mlir::Location location, const target::LPUTargetModel& target,
    mlir::DictionaryAttr inputPlacement,
    mlir::DictionaryAttr outputPlacement,
    int64_t rows, int64_t hidden, int64_t start)
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const int64_t width = 2 * target.throughput().lanes_per_tile;
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t tileRows = target.throughput().tile_rows;
    if (inputSlices.size() != static_cast<std::size_t>(width)
        || outputSlices.size() != static_cast<std::size_t>(width))
        return start;

    std::array<int64_t, 16> sourceStreams {};
    std::array<int64_t, 16> transposeStreams {};
    std::array<int64_t, 16> outputStreams {};
    for (int64_t stream = 0; stream < width; ++stream) {
        sourceStreams[static_cast<std::size_t>(stream)] = stream;
        transposeStreams[static_cast<std::size_t>(stream)] = 16 + stream;
        outputStreams[static_cast<std::size_t>(stream)] = 32 + stream;
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
    std::array<int64_t, 2> captureReady {
        start + maxReadLatency, start + maxReadLatency};
    for (int64_t block = 0; block < blockCount; ++block) {
        for (int64_t hemisphere = 0;
             hemisphere < target.memory().hemispheres; ++hemisphere) {
            const int64_t capture =
                captureReady[static_cast<std::size_t>(hemisphere)];
            for (int64_t beat = 0; beat < tileRows; ++beat) {
                for (int64_t stream = 0; stream < width; ++stream) {
                    const int64_t slice =
                        inputSlices[static_cast<std::size_t>(stream)];
                    emitMem(rewriter, location,
                        capture + beat - readLatency(slice),
                        hemisphere * target.memory().slices_per_hemisphere
                            + slice,
                        "read", baseRow(inputPlacement)
                            + block * tileRows + beat,
                        stream, 1, 1, 0);
                }
                emitWavefrontBeat(rewriter, location, target,
                    capture + beat, hemisphere, beat,
                    sourceStreams, transposeStreams, outputStreams);
                for (int64_t stream = 0; stream < width; ++stream) {
                    const int64_t slice =
                        outputSlices[static_cast<std::size_t>(stream)];
                    emitMem(rewriter, location,
                        capture + beat + 1 + writeLatency(slice),
                        hemisphere * target.memory().slices_per_hemisphere
                            + slice,
                        "write", baseRow(outputPlacement)
                            + block * tileRows + beat,
                        32 + stream, 1, 1, 0);
                }
            }
            captureReady[static_cast<std::size_t>(hemisphere)] += tileRows;
        }
    }
    for (int64_t hemisphere = 0;
         hemisphere < target.memory().hemispheres; ++hemisphere) {
        for (int64_t tail = 0; tail < tileRows - 1; ++tail)
            emitWavefrontTail(rewriter, location, target,
                captureReady[static_cast<std::size_t>(hemisphere)] + tail,
                hemisphere, tail, transposeStreams, outputStreams);
    }
    return std::max(captureReady[0], captureReady[1])
        + tileRows - 1 + maxWriteLatency + 2;
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
    int64_t cycle = start;

    for (int64_t tokenBlock = 0;
         tokenBlock < tokenBlocks; ++tokenBlock) {
        for (int64_t hiddenBlock = 0;
             hiddenBlock < hiddenBlocks; ++hiddenBlock) {
            for (int64_t wave = 0; wave < tileRows; ++wave) {
                const int64_t capture = cycle
                    + target.throughput().mem_to_sxm_latency;
                for (int64_t row = 0; row < lanes; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < target.memory().hemispheres;
                         ++hemisphere) {
                        const std::array<int64_t, 2> sourceStreams {
                            hemisphere == 0 ? 0 : 32,
                            hemisphere == 0 ? 1 : 33,
                        };
                        const auto transposeStreams = streamRange(
                            hemisphere == 0 ? 16 : 48, packedStreams);
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            const int64_t slice = inputSlices[byte];
                            const int64_t latency =
                                *target.transport_latency(
                                    target::StreamEndpoint::Mem,
                                    target::StreamEndpoint::SxmInput,
                                    target::StreamDirection::East, slice);
                            const int64_t address = broadcastInput
                                ? baseRow(inputPlacement) + hiddenBlock
                                : planarAddress(inputPlacement,
                                    tokenBlock, hiddenBlock, wave, row,
                                    rows);
                            emitMem(rewriter, location,
                                capture + row - latency,
                                hemisphere
                                        * target.memory().
                                            slices_per_hemisphere
                                    + slice,
                                "read", address,
                                hemisphere == 0 ? byte : 32 + byte,
                                1, 1, 0);
                        }
                        emitSxm(rewriter, location, capture + row,
                            hemisphere, "transpose", sourceStreams,
                            transposeStreams,
                            attention_detail::identityMap());
                    }
                }

                const int64_t permute = capture + lanes
                    + target.throughput().tile_rows;
                const auto map = blockDiagonalMap(wave, target);
                for (int64_t hemisphere = 0;
                     hemisphere < target.memory().hemispheres;
                     ++hemisphere) {
                    const auto transposeStreams = streamRange(
                        hemisphere == 0 ? 16 : 48, packedStreams);
                    const auto outputStreams = streamRange(
                        hemisphere == 0 ? 32 : 0, packedStreams);
                    emitSxm(rewriter, location, permute, hemisphere,
                        "permute", transposeStreams, outputStreams, map);
                    for (int64_t stream = 0;
                         stream < packedStreams; ++stream) {
                        const int64_t slice = outputSlices[stream];
                        const auto direction = hemisphere == 0
                            ? target::StreamDirection::West
                            : target::StreamDirection::East;
                        const int64_t latency = *target.transport_latency(
                            target::StreamEndpoint::SxmResult,
                            target::StreamEndpoint::Mem, direction, slice);
                        emitMem(rewriter, location, permute + latency,
                            hemisphere
                                    * target.memory().slices_per_hemisphere
                                + slice,
                            "write", packedAddress(outputPlacement,
                                tokenBlock, hiddenBlock, wave,
                                hiddenBlocks),
                            hemisphere == 0 ? 32 + stream : stream,
                            1, 1, 0);
                    }
                }
                cycle = permute + target.streams().system_register_columns;
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
    bool outputFeedback)
{
    const auto inputSlices = slices(inputPlacement);
    const auto outputSlices = slices(outputPlacement);
    const int64_t tile = target.throughput().mxm_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    const int64_t tileRows = target.throughput().tile_rows;
    const int64_t packedStreams = 2 * lanes;
    const int64_t hiddenBlocks = hidden / tile;
    int64_t cycle = start;

    for (int64_t tokenBlock = 0; tokenBlock < rows / tile; ++tokenBlock) {
        for (int64_t hiddenBlock = 0;
             hiddenBlock < hiddenBlocks; ++hiddenBlock) {
            for (int64_t wave = 0; wave < tileRows; ++wave) {
                const int64_t capture = cycle
                    + target.throughput().mem_to_sxm_latency;
                for (int64_t hemisphere = 0;
                     hemisphere < target.memory().hemispheres;
                     ++hemisphere) {
                    const auto sourceStreams = streamRange(
                        hemisphere == 0 ? 0 : 32, packedStreams);
                    const auto transposeStreams = streamRange(
                        hemisphere == 0 ? 16 : 48, packedStreams);
                    for (int64_t stream = 0;
                         stream < packedStreams; ++stream) {
                        const int64_t slice = inputSlices[stream];
                        const auto direction = hemisphere == 0
                            ? target::StreamDirection::East
                            : target::StreamDirection::West;
                        const int64_t latency = *target.transport_latency(
                            target::StreamEndpoint::Mem,
                            target::StreamEndpoint::SxmInput,
                            direction, slice);
                        emitMem(rewriter, location, capture - latency,
                            hemisphere
                                    * target.memory().slices_per_hemisphere
                                + slice,
                            "read", packedAddress(inputPlacement,
                                tokenBlock, hiddenBlock, wave,
                                hiddenBlocks),
                            hemisphere == 0 ? stream : 32 + stream,
                            1, 1, 0);
                    }
                    emitSxm(rewriter, location, capture, hemisphere,
                        "transpose", sourceStreams, transposeStreams,
                        attention_detail::identityMap());
                }

                const int64_t permute = capture
                    + target.throughput().tile_rows + 1;
                const auto map = blockDiagonalMap(wave, target);
                for (int64_t row = 0; row < lanes; ++row) {
                    for (int64_t hemisphere = 0;
                         hemisphere < target.memory().hemispheres;
                         ++hemisphere) {
                        const auto transposeStreams = streamRange(
                            hemisphere == 0 ? 16 : 48, packedStreams);
                        const std::array<int64_t, 2> outputStreams {
                            hemisphere == 0 ? 32 : 0,
                            hemisphere == 0 ? 33 : 1,
                        };
                        emitSxm(rewriter, location, permute + row,
                            hemisphere, "permute", transposeStreams,
                            outputStreams, map);
                        for (int64_t byte = 0; byte < 2; ++byte) {
                            for (int64_t duplicate = byte;
                                 duplicate
                                     < static_cast<int64_t>(
                                         outputSlices.size());
                                 duplicate += 2) {
                                const int64_t slice =
                                    outputSlices[duplicate];
                                const auto direction = hemisphere == 0
                                    ? target::StreamDirection::West
                                    : target::StreamDirection::East;
                                const int64_t latency =
                                    *target.transport_latency(
                                        target::StreamEndpoint::SxmResult,
                                        target::StreamEndpoint::Mem,
                                        direction, slice);
                                emitMem(rewriter, location,
                                    permute + row + latency,
                                    hemisphere
                                            * target.memory().
                                                slices_per_hemisphere
                                        + slice,
                                    "write", outputFeedback
                                        ? serialFeedbackAddress(
                                              outputPlacement, tokenBlock,
                                              hiddenBlock, wave, row,
                                              rows, hidden, true)
                                        : planarAddress(
                                              outputPlacement, tokenBlock,
                                              hiddenBlock, wave, row, rows),
                                    hemisphere == 0 ? 32 + byte : byte,
                                    1, 1, 0);
                            }
                        }
                    }
                }
                cycle = permute + lanes
                    + target.streams().system_register_columns;
            }
        }
    }
    return cycle;
}

int64_t emitSerialVxmFeedback(mlir::IRRewriter& rewriter,
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

    int64_t cycle = start;
    for (int64_t tokenBlock = 0;
         tokenBlock < rows / tile; ++tokenBlock) {
        const int64_t square = cycle
            + std::max(westReadLatency(inputSlices[0]),
                westReadLatency(inputSlices[1]));
        const int64_t inputAddress =
            baseRow(inputPlacement) + tokenBlock * hidden;
        for (int64_t byte = 0; byte < 2; ++byte)
            emitMem(rewriter, op.getLoc(),
                square - westReadLatency(inputSlices[byte]),
                inputSlices[byte], "read", inputAddress,
                32 + byte, hidden, 1, 1);
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, square, 0, "square",
            streamKind, 32, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, hidden, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, square, 1, "pass",
            "immediate", 0, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, 1, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, square + 1, 1, "add",
            "alu", 0, 0.0f, "alu", 1, 0.0f,
            "fp32", -1, hidden, 1, "east", "east");

        const int64_t normalize = square + hidden + 1;
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, normalize, 2, "divide",
            "alu", 1, 0.0f, "immediate", 0,
            static_cast<float>(hidden), "fp32", -1, 1, 1,
            "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, normalize + 1, 3, "add",
            "alu", 2, 0.0f, "immediate", 0,
            static_cast<float>(op.getEpsilon().convertToDouble()),
            "fp32", -1, 1, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, normalize + 2, 4, "sqrt",
            "alu", 3, 0.0f, "immediate", 0, 0.0f,
            "fp32", -1, 1, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, normalize + 3, 5, "divide",
            "immediate", 0, 1.0f, "alu", 4, 0.0f,
            "fp32", -1, 1, 1, "east", "east");

        for (int64_t byte = 0; byte < 2; ++byte) {
            emitMem(rewriter, op.getLoc(),
                normalize - westReadLatency(inputSlices[byte]),
                inputSlices[byte], "read", inputAddress,
                32 + byte, 1, 1, 0);
            emitMem(rewriter, op.getLoc(),
                normalize - westReadLatency(inputSlices[byte]) + 1,
                inputSlices[byte], "read", inputAddress,
                32 + byte, hidden, 1, 1);
            emitMem(rewriter, op.getLoc(),
                normalize - westReadLatency(weightSlices[byte]),
                weightSlices[byte], "read",
                baseRow(weightPlacement), 34 + byte,
                1, 1, 0, "sram",
                inputBindingIndex(op.getWeight()));
            emitMem(rewriter, op.getLoc(),
                normalize - westReadLatency(weightSlices[byte]) + 2,
                weightSlices[byte], "read",
                baseRow(weightPlacement), 34 + byte,
                hidden, 1, 1, "sram",
                inputBindingIndex(op.getWeight()));
        }
        for (int64_t queue = 6; queue <= 9; ++queue)
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, normalize + queue - 6, queue, "pass",
                queue == 6 ? streamKind : llvm::StringRef("alu"),
                queue == 6 ? 32 : queue - 1, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1,
                hidden, 1, "east", "east");
        for (int64_t queue = 10; queue <= 13; ++queue)
            create_vxm(rewriter, op.getLoc(), op.getWeight(), op.getWeight(),
                inputType, normalize + queue - 10, queue, "pass",
                queue == 10 ? streamKind : llvm::StringRef("alu"),
                queue == 10 ? 34 : queue - 1, 0.0f,
                "immediate", 0, 0.0f, "fp32", -1,
                hidden, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
            inputType, normalize + 5, 14, "multiply",
            "alu", 9, 0.0f, "alu", 5, 0.0f,
            "fp32", -1, hidden, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getWeight(),
            inputType, normalize + 6, 15, "multiply",
            "alu", 14, 0.0f, "alu", 13, 0.0f,
            dataFormat, 0, hidden, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), op.getInput(), op.getWeight(),
            inputType, normalize + 7, 0, "pass",
            "alu", 15, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 16, hidden, 1, "east", "west");

        for (int64_t byte = 0; byte < 2; ++byte) {
            emitMem(rewriter, op.getLoc(),
                normalize + 6 + eastWriteLatency(outputSlices[byte]),
                outputSlices[byte], "write",
                baseRow(outputPlacement) + tokenBlock * hidden,
                byte, hidden, 1, 1);
            emitMem(rewriter, op.getLoc(),
                normalize + 7 + eastWriteLatency(outputSlices[byte]),
                target.memory().slices_per_hemisphere
                    + outputSlices[byte],
                "write",
                baseRow(outputPlacement) + tokenBlock * hidden,
                16 + byte, hidden, 1, 1);
        }

        const int64_t tailStart = normalize + hidden + 12;
        for (int64_t tail = 0; tail < 2; ++tail) {
            const int64_t column = hidden - 2 + tail;
            const int64_t xCycle = tailStart + tail;
            const int64_t gammaCycle = xCycle + 1;
            for (int64_t byte = 0; byte < 2; ++byte) {
                emitMem(rewriter, op.getLoc(),
                    xCycle - westReadLatency(inputSlices[byte]),
                    inputSlices[byte], "read",
                    inputAddress + column, 32 + byte, 1, 1, 0);
                emitMem(rewriter, op.getLoc(),
                    gammaCycle - westReadLatency(weightSlices[byte]),
                    weightSlices[byte], "read",
                    baseRow(weightPlacement) + column,
                    34 + byte, 1, 1, 0, "sram",
                    inputBindingIndex(op.getWeight()));
            }
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getInput(),
                inputType, xCycle, 6, "multiply",
                streamKind, 32, 0.0f, "alu", 5, 0.0f,
                "fp32", -1, 1, 1, "east", "east");
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getWeight(),
                inputType, gammaCycle, 7, "multiply",
                "alu", 6, 0.0f, streamKind, 34, 0.0f,
                dataFormat, 0, 1, 1, "east", "east");
            create_vxm(rewriter, op.getLoc(), op.getInput(), op.getWeight(),
                inputType, gammaCycle + 1, 8, "pass",
                "alu", 7, 0.0f, "immediate", 0, 0.0f,
                dataFormat, 16, 1, 1, "east", "west");
            for (int64_t byte = 0; byte < 2; ++byte) {
                emitMem(rewriter, op.getLoc(),
                    gammaCycle + eastWriteLatency(outputSlices[byte]),
                    outputSlices[byte], "write",
                    baseRow(outputPlacement)
                        + tokenBlock * hidden + column,
                    byte, 1, 1, 0);
                emitMem(rewriter, op.getLoc(),
                    gammaCycle + 1
                        + eastWriteLatency(outputSlices[byte]),
                    target.memory().slices_per_hemisphere
                        + outputSlices[byte],
                    "write",
                    baseRow(outputPlacement)
                        + tokenBlock * hidden + column,
                    16 + byte, 1, 1, 0);
            }
        }
        cycle = tailStart + 4
            + std::max(eastWriteLatency(outputSlices[0]),
                eastWriteLatency(outputSlices[1]));
    }
    return cycle;
}

int64_t emitVxmFeedback(mlir::IRRewriter& rewriter,
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
    const int64_t waves = target.throughput().tile_rows;
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
                    tokenBlock, hiddenBlock, featureWave, hiddenBlocks);
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
                    tokenBlock, hiddenBlock, featureWave, hiddenBlocks);
                const int64_t weightAddress = baseRow(weightPlacement)
                    + hiddenBlock * waves + featureWave;
                const int64_t outputAddress = packedAddress(outputPlacement,
                    tokenBlock, hiddenBlock, featureWave, hiddenBlocks);
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

mlir::LogicalResult lowerRmsNormFeedback(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    int64_t outputIndex)
{
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const int64_t rows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const auto inputPlacement =
        allocationPlacement(op.getInputAllocations(), 0);
    const auto weightPlacement =
        allocationPlacement(op.getWeightAllocations(), 0);
    const auto feedbackInputPlacement =
        allocationPlacement(op.getScratchAllocations(), 0);
    const auto feedbackOutputPlacement =
        allocationPlacement(op.getScratchAllocations(), 2);
    const auto resultPlacement =
        allocationPlacement(op.getResultAllocations(), 0);

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

    const auto inputKind =
        inputPlacement.getAs<mlir::StringAttr>("kind");
    const bool vxmInput = inputKind
        && inputKind.getValue() == "fp16_vxm_distributed_16";
    const bool mxmInput = inputKind
        && inputKind.getValue() == "fp16_mxm_distributed_16";
    if (!vxmInput && !mxmInput)
        return op.emitError(
            "feedback RMSNorm requires MXM- or VXM-oriented distributed16 input");
    const int64_t inputTransposeEnd = vxmInput
        ? 0
        : emitDistributedMatrixTranspose(
              rewriter, op.getLoc(), target, inputPlacement,
              feedbackInputPlacement, rows, hidden, 0);
    const auto feedbackInput = vxmInput
        ? inputPlacement : feedbackInputPlacement;
    const auto weightKind =
        weightPlacement.getAs<mlir::StringAttr>("kind");
    const bool distributedWeight = weightKind
        && weightKind.getValue() == "fp16_vxm_distributed_16";
    if (!distributedWeight)
        return op.emitError(
            "feedback RMSNorm requires VXM-oriented distributed16 gamma");
    const int64_t weightTransposeEnd = inputTransposeEnd;
    const int64_t feedbackEnd = emitVxmFeedback(
        rewriter, op, target, feedbackInput,
        weightPlacement, feedbackOutputPlacement,
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
            0, inputTransposeEnd);
    createTimeline(rewriter, op.getLoc(), "rmsnorm.feedback",
        weightTransposeEnd, feedbackEnd);
    if (!vxmResult)
        createTimeline(rewriter, op.getLoc(), "rmsnorm.restore_layout",
            feedbackEnd, restoreEnd);
    auto output = createBinding(rewriter, op.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType()),
        resultPlacement);
    rewriter.replaceOp(op, output.getValue());
    return mlir::success();
}

mlir::LogicalResult lowerRmsNormMxm(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    int64_t outputIndex)
{
    const auto inputType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    const auto weightType =
        llvm::cast<mlir::RankedTensorType>(op.getWeight().getType());
    const llvm::StringRef streamKind =
        lpu_16bit_stream_kind(inputType.getElementType());
    const llvm::StringRef dataFormat =
        lpu_16bit_data_format(inputType.getElementType());
    const int64_t rows = inputType.getDimSize(0);
    const int64_t hidden = inputType.getDimSize(1);
    const int64_t tile = target.throughput().mxm_rows;
    if (inputType.getRank() != 2 || weightType.getRank() != 1
        || rows % tile != 0 || hidden % tile != 0) {
        return op.emitError(
            "RMSNorm schedule requires [M,K] and [K] tile-aligned tensors");
    }

    const auto inputPlacement =
        allocationPlacement(op.getInputAllocations(), 0);
    const auto weightPlacement =
        allocationPlacement(op.getWeightAllocations(), 0);
    const auto squarePlacement =
        allocationPlacement(op.getScratchAllocations(), 0);
    const auto factorPlacement =
        allocationPlacement(op.getScratchAllocations(), 1);
    const auto resultPlacement =
        allocationPlacement(op.getResultAllocations(), 0);
    const auto inputSlices = slices(inputPlacement);
    const auto weightSlices = slices(weightPlacement);
    const auto squareSlices = slices(squarePlacement);
    const auto factorSlices = slices(factorPlacement);
    const auto resultSlices = slices(resultPlacement);
    if (inputSlices.size() < 2 || weightSlices.size() < 2
        || squareSlices.size() < 2 || factorSlices.size() < 2
        || resultSlices.size() < 4) {
        return op.emitError(
            "RMSNorm schedule requires 16-bit float byte-pair slices");
    }

    rewriter.setInsertionPoint(op);
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(op.getInput()))
        createBinding(rewriter, op.getLoc(), op.getInput(),
            argument.getArgNumber(), "input", "activation",
            inputType, inputPlacement);
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(op.getWeight()))
        createBinding(rewriter, op.getLoc(), op.getWeight(),
            argument.getArgNumber(), "input", "weight",
            weightType, weightPlacement);

    const auto eastReadLatency = [&](int64_t slice) {
        return target.transport_latency(target::StreamEndpoint::Mem,
            target::StreamEndpoint::MxmActivation,
            target::StreamDirection::East, slice).value();
    };
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

    // Build one reusable 32x32 matrix whose entries are 1 / hidden.
    // Four VXM pulses provide the 16 FP16 weight streams consumed by IW.
    const int64_t weightCastStart = 0;
    const int64_t weightToIw =
        target.throughput().vxm_weight_to_iw_latency;
    for (int64_t pulse = 0; pulse < target.throughput().tile_rows; ++pulse) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            auto constant = create_vxm(rewriter, op.getLoc(),
                op.getInput(), op.getInput(), inputType,
                weightCastStart + pulse, lane, "pass",
                "immediate", 0, 1.0f / static_cast<float>(hidden),
                "immediate", 0, 0.0f, "fp32", -1, 1, 1,
                "east", "east");
            create_vxm(rewriter, op.getLoc(), constant.getResult(),
                constant.getResult(), inputType,
                weightCastStart + pulse + 1, 8 + lane, "cast",
                "alu", lane, 0.0f, "immediate", 0, 0.0f,
                dataFormat, lane * 2, 1, 1, "east", "east");
        }
        emitMxm(rewriter, op.getLoc(), weightCastStart + pulse + weightToIw,
            0, "iw", 0, 3 - pulse, 0, 0, 1, 1);
    }
    const int64_t weightReady =
        weightCastStart + weightToIw + target.throughput().tile_rows;

    // Square each activation tile and retain the original MXM activation
    // layout, so the following reduction and downstream layers compose.
    const int64_t squareStart = std::max<int64_t>(
        weightReady, *std::max_element(inputSlices.begin(), inputSlices.end())
            / target.streams().mem_slices_per_register_group + 2);
    int64_t cycle = squareStart;
    for (int64_t block = 0; block < hidden / tile; ++block) {
        auto inputTile = tileAddress(inputPlacement, block, rows);
        auto squareTile = tileAddress(squarePlacement, block, rows);
        if (mlir::failed(inputTile) || mlir::failed(squareTile))
            return op.emitError("unsupported RMSNorm square layout");
        const int64_t vxmCycle = cycle;
        for (int64_t byte = 0; byte < 2; ++byte)
            emitMem(rewriter, op.getLoc(),
                vxmCycle - westReadLatency(inputTile->slices[byte]),
                inputTile->slices[byte], "read", inputTile->row, 32 + byte,
                rows, 1, 1);
        auto squared = create_vxm(rewriter, op.getLoc(),
            op.getInput(), op.getInput(), inputType, vxmCycle, 8,
            "square", streamKind, 32, 0.0f,
            "immediate", 0, 0.0f, "fp32", -1,
            rows, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(),
            op.getInput(), op.getInput(), inputType, vxmCycle, 10,
            "pass", streamKind, 32, 0.0f,
            "immediate", 0, 0.0f, dataFormat, 2,
            rows, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(), squared.getResult(),
            squared.getResult(), inputType, vxmCycle + 1, 9, "cast",
            "alu", 8, 0.0f, "immediate", 0, 0.0f,
            dataFormat, 0, rows, 1, "east", "east");
        for (int64_t byte = 0; byte < 2; ++byte)
            emitMem(rewriter, op.getLoc(),
                vxmCycle + 1 + eastWriteLatency(squareTile->slices[byte]),
                squareTile->slices[byte], "write", squareTile->row, byte,
                rows, 1, 1);
        for (int64_t byte = 0; byte < 2; ++byte)
            emitMem(rewriter, op.getLoc(),
                vxmCycle + eastWriteLatency(squareSlices[2 + byte]),
                squareSlices[2 + byte], "write", squareTile->row,
                2 + byte, rows, 1, 1);
        cycle += rows + 1 + std::max(
            eastWriteLatency(squareSlices[1]),
            eastWriteLatency(squareSlices[3])) + 1;
    }
    const int64_t squareEnd = cycle;

    // Accumulate all hidden tiles in MXM0. A compute command may cover at
    // most one physical 32-row MXM wave: the CModel wraps its internal row
    // selector after that point. Keep each token wave in a distinct external
    // accumulator address range, then stream the four final waves to VXM.
    const int64_t tokenWaves = rows / tile;
    const int64_t clearStart = squareEnd;
    for (int64_t row = 0; row < rows; ++row)
        emitMxm(rewriter, op.getLoc(), clearStart + row,
            0, "accumulator_read", 0, 0, 0, 0, 1, 1,
            row, 1, "sram", true, "supercell", 0, dataFormat);
    const int64_t reduceStart = std::max(
        clearStart + rows
            + target.throughput().accumulator_read_to_vxm_latency + 1,
        squareEnd + std::max(
        eastReadLatency(squareSlices[0]),
        eastReadLatency(squareSlices[1])));
    int64_t finalCompute = 0;
    const int64_t issue = target.mxm_block_issue_interval();
    for (int64_t block = 0; block < hidden / tile; ++block) {
        const bool final = block + 1 == hidden / tile;
        for (int64_t wave = 0; wave < tokenWaves; ++wave) {
            const int64_t ordinal = block * tokenWaves + wave;
            const int64_t computeCycle = reduceStart + ordinal * issue;
            const int64_t tokenBase = wave * tile;
            auto squareTile = tileAddress(squarePlacement, block, rows);
            if (mlir::failed(squareTile))
                return op.emitError("unsupported RMSNorm reduction layout");
            for (int64_t byte = 0; byte < 2; ++byte)
                emitMem(rewriter, op.getLoc(),
                    computeCycle - eastReadLatency(
                        squareTile->slices[byte]),
                    squareTile->slices[byte], "read",
                    squareTile->row + tokenBase, byte, tile, 1, 1);
            emitMxm(rewriter, op.getLoc(), computeCycle, 0, "compute",
                0, 0, 0, 0, tile, 1, tokenBase, 1,
                final ? "stream" : "sram", final, "supercell", 0,
                dataFormat);
            if (final && wave == 0) finalCompute = computeCycle;
        }
    }
    const int64_t factorVxm =
        finalCompute + target.throughput().accumulator_to_vxm_latency;
    auto meanPlusEpsilon = create_vxm(rewriter, op.getLoc(),
        op.getInput(), op.getInput(), inputType, factorVxm, 0, "add",
        "stream_f32", 32, 0.0f, "immediate", 0,
        static_cast<float>(op.getEpsilon().convertToDouble()),
        "fp32", -1, rows, 1, "east", "east");
    auto root = create_vxm(rewriter, op.getLoc(),
        meanPlusEpsilon.getResult(), meanPlusEpsilon.getResult(), inputType,
        factorVxm + 1, 1, "sqrt", "alu", 0, 0.0f,
        "immediate", 0, 0.0f, "fp32", -1,
        rows, 1, "east", "east");
    auto inverse = create_vxm(rewriter, op.getLoc(),
        root.getResult(), root.getResult(), inputType, factorVxm + 2, 2,
        "divide", "immediate", 0, 1.0f, "alu", 1, 0.0f,
        "fp32", -1, rows, 1, "east", "east");
    create_vxm(rewriter, op.getLoc(), inverse.getResult(),
        inverse.getResult(), inputType, factorVxm + 3, 3, "cast",
        "alu", 2, 0.0f, "immediate", 0, 0.0f,
        dataFormat, 0, rows, 1, "east", "east");
    for (int64_t byte = 0; byte < 2; ++byte)
        emitMem(rewriter, op.getLoc(),
            factorVxm + 3 + eastWriteLatency(factorSlices[byte]),
            factorSlices[byte], "write", baseRow(factorPlacement), byte,
            rows, 1, 1);
    const int64_t factorEnd =
        factorVxm + 3 + rows + eastWriteLatency(factorSlices[1]);

    // Broadcast the stored row factor over each 32-column activation tile,
    // multiply by gamma, and write a standard activation-planar result.
    cycle = factorEnd + std::max(
        westReadLatency(factorSlices[0]),
        westReadLatency(factorSlices[1]));
    mlir::Value finalValue = op.getInput();
    for (int64_t block = 0; block < hidden / tile; ++block) {
        auto inputTile = tileAddress(squarePlacement, block, rows);
        auto resultTile = tileAddress(resultPlacement, block, rows);
        if (mlir::failed(inputTile) || mlir::failed(resultTile))
            return op.emitError("unsupported RMSNorm scale layout");
        inputTile->slices = {squareSlices[2], squareSlices[3]};
        const int64_t vxmCycle = cycle;
        for (int64_t byte = 0; byte < 2; ++byte) {
            emitMem(rewriter, op.getLoc(),
                vxmCycle - westReadLatency(inputTile->slices[byte]),
                inputTile->slices[byte], "read", inputTile->row,
                32 + byte, rows, 1, 1);
            emitMem(rewriter, op.getLoc(),
                vxmCycle - westReadLatency(factorSlices[byte]),
                factorSlices[byte], "read", baseRow(factorPlacement),
                34 + byte, rows, 1, 1);
            emitMem(rewriter, op.getLoc(),
                vxmCycle - westReadLatency(weightSlices[byte]),
                weightSlices[byte], "read",
                baseRow(weightPlacement) + block,
                36 + byte, rows, 1, 0, "sram",
                inputBindingIndex(op.getWeight()));
        }
        auto normalized = create_vxm(rewriter, op.getLoc(),
            op.getInput(), op.getInput(), inputType, vxmCycle, 4,
            "multiply", streamKind, 32, 0.0f,
            streamKind, 34, 0.0f, "fp32", -1,
            rows, 1, "east", "east");
        auto scaled = create_vxm(rewriter, op.getLoc(),
            normalized.getResult(), op.getWeight(), inputType,
            vxmCycle + 1, 5, "multiply", "alu", 4, 0.0f,
            streamKind, 36, 0.0f, "fp32", -1,
            rows, 1, "east", "east");
        auto cast = create_vxm(rewriter, op.getLoc(),
            scaled.getResult(), scaled.getResult(), inputType,
            vxmCycle + 2, 6, "cast", "alu", 5, 0.0f,
            "immediate", 0, 0.0f, dataFormat, 0,
            rows, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(),
            scaled.getResult(), scaled.getResult(), inputType,
            vxmCycle + 2, 7, "cast", "alu", 5, 0.0f,
            "immediate", 0, 0.0f, dataFormat, 2,
            rows, 1, "east", "east");
        create_vxm(rewriter, op.getLoc(),
            scaled.getResult(), scaled.getResult(), inputType,
            vxmCycle + 2, 8, "cast", "alu", 5, 0.0f,
            "immediate", 0, 0.0f, dataFormat, 0,
            rows, 1, "east", "west");
        create_vxm(rewriter, op.getLoc(),
            scaled.getResult(), scaled.getResult(), inputType,
            vxmCycle + 2, 9, "cast", "alu", 5, 0.0f,
            "immediate", 0, 0.0f, dataFormat, 2,
            rows, 1, "east", "west");
        finalValue = cast.getResult();
        for (int64_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (int64_t byte = 0; byte < 4; ++byte)
                emitMem(rewriter, op.getLoc(),
                    vxmCycle + 2
                        + eastWriteLatency(resultSlices[byte]),
                    hemisphere
                            * target.memory().slices_per_hemisphere
                        + resultSlices[byte],
                    "write", resultTile->row, byte,
                    rows, 1, 1);
        }
        cycle += rows + 2 + eastWriteLatency(resultSlices[1]) + 1;
    }

    createTimeline(rewriter, op.getLoc(), "rmsnorm.square",
        squareStart, squareEnd);
    createTimeline(rewriter, op.getLoc(), "rmsnorm.reduce",
        reduceStart, factorEnd);
    createTimeline(rewriter, op.getLoc(), "rmsnorm.scale",
        factorEnd, cycle);
    auto output = createBinding(rewriter, op.getLoc(), {},
        outputIndex, "output", "result",
        llvm::cast<mlir::RankedTensorType>(op.getResult().getType()),
        resultPlacement);
    rewriter.replaceOp(op, output.getValue());
    return mlir::success();
}

mlir::LogicalResult lowerRmsNorm(mlir::IRRewriter& rewriter,
    stream::RmsNormTaskOp op, const target::LPUTargetModel& target,
    int64_t outputIndex)
{
    const auto strategy =
        op.getConfig().getAs<mlir::StringAttr>("strategy");
    if (strategy && strategy.getValue() == "vxm_feedback")
        return lowerRmsNormFeedback(
            rewriter, op, target, outputIndex);
    return lowerRmsNormMxm(rewriter, op, target, outputIndex);
}

} // namespace

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
