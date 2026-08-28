#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"

#include "AttentionEmitterUtils.hpp"
#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Support/float_format.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
#include <array>
#include <string>

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

struct TileAddress {
    std::array<int64_t, 2> slices;
    int64_t row;
};

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
                        attention_detail::identityMap(),
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
                        outputStreams, attention_detail::identityMap(),
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
                const int64_t beatCycle = capture
                    + beat * (tileRows + 1);
                for (int64_t stream = 0; stream < width; ++stream) {
                    const int64_t slice =
                        inputSlices[static_cast<std::size_t>(stream)];
                    emitMem(rewriter, location,
                        beatCycle - readLatency(slice),
                        hemisphere * target.memory().slices_per_hemisphere
                            + slice,
                        "read", baseRow(inputPlacement)
                            + block * tileRows + beat,
                        stream, 1, 1, 0);
                }
                sxm_detail::emitBufferedWavefrontBeat(rewriter, location,
                    target, beatCycle, hemisphere, beat,
                    sourceStreams, transposeStreams, outputStreams);
                for (int64_t stream = 0; stream < width; ++stream) {
                    const int64_t slice =
                        outputSlices[static_cast<std::size_t>(stream)];
                    emitMem(rewriter, location,
                        beatCycle + 1 + writeLatency(slice),
                        hemisphere * target.memory().slices_per_hemisphere
                            + slice,
                        "write", baseRow(outputPlacement)
                            + block * tileRows + beat,
                        32 + stream, 1, 1, 0);
                }
            }
            captureReady[static_cast<std::size_t>(hemisphere)] +=
                tileRows * (tileRows + 1);
        }
    }
    return std::max(captureReady[0], captureReady[1])
        + maxWriteLatency + tileRows + 2;
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
                            attention_detail::identityMap(),
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
                                    hiddenBlocks),
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
                                hiddenBlocks),
                            kSxmInputStream + stream,
                            1, 1, 0, "sram", -1,
                            bank(inputPlacement));
                    }
                    emitSxm(rewriter, location, capture, hemisphere,
                        "transpose", sourceStreams, transposeStreams,
                        attention_detail::identityMap());
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
                const int64_t writeCycle = bridgeInput
                    + writeLatency(slice);
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
        allocationPlacement(op.getScratchAllocations(), 1);
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
        && inputKind.getValue() == "fp16_vxm_row_parallel_8";
    const bool mxmInput = inputKind
        && inputKind.getValue() == "fp16_mxm_distributed_16";
    if (!vxmInput && !mxmInput)
        return op.emitError(
            "feedback RMSNorm requires MXM- or VXM-oriented distributed16 input");
    const int64_t inputTransposeEnd = vxmInput
        ? 0
        : emitPackedToPairTranspose(
              rewriter, op.getLoc(), target, inputPlacement,
              feedbackInputPlacement, rows, hidden, 0, true);
    const auto feedbackInput = vxmInput
        ? inputPlacement : feedbackInputPlacement;
    const auto weightKind =
        weightPlacement.getAs<mlir::StringAttr>("kind");
    const bool distributedWeight = weightKind
        && (weightKind.getValue() == "fp16_vxm_row_parallel_8"
            || weightKind.getValue()
                == "fp16_vxm_gamma_broadcast");
    if (!distributedWeight)
        return op.emitError(
            "feedback RMSNorm requires VXM-oriented distributed16 gamma");
    const int64_t weightTransposeEnd = inputTransposeEnd;
    const int64_t feedbackEnd = emitRowParallelVxmFeedback(
        rewriter, op, target, feedbackInput,
        weightPlacement, feedbackOutputPlacement,
        weightTransposeEnd);
    const auto resultKind =
        resultPlacement.getAs<mlir::StringAttr>("kind");
    const bool vxmResult = resultKind
        && resultKind.getValue() == "fp16_vxm_row_parallel_8";
    const bool mxmResult = resultKind
        && resultKind.getValue() == "fp16_mxm_distributed_16";
    if (!vxmResult && !mxmResult)
        return op.emitError(
            "feedback RMSNorm requires MXM- or VXM-oriented distributed16 result");
    const int64_t restoreEnd = vxmResult
        ? feedbackEnd
        : emitPairToPackedTranspose(
              rewriter, op.getLoc(), target,
              feedbackOutputPlacement, resultPlacement,
              rows, hidden, feedbackEnd, false);

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
    output->setAttr("name", rewriter.getStringAttr(
        "rmsnorm.result." + std::to_string(outputIndex)));
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
    output->setAttr("name", rewriter.getStringAttr(
        "rmsnorm.result." + std::to_string(outputIndex)));
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
