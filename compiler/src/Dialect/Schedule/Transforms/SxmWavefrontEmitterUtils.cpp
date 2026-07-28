#include "SxmWavefrontEmitterUtils.hpp"

#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace ftlpu::compiler::schedule::sxm_detail {
namespace {

void requirePhysicalWidth(const target::LPUTargetModel& target,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams)
{
    const auto width = static_cast<std::size_t>(
        2 * target.throughput().lanes_per_tile);
    assert(sourceStreams.size() == width
        && destinationStreams.size() == width
        && "SXM requires one FP16 byte-stream pair per lane");
}

} // namespace

void emitSxm(mlir::IRRewriter& rewriter, mlir::Location location,
    int64_t cycle, int64_t hemisphere, llvm::StringRef opcode,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::ArrayRef<int64_t> permuteMap,
    llvm::StringRef weightLayout)
{
    const auto integers = [&](llvm::ArrayRef<int64_t> values) {
        llvm::SmallVector<mlir::Attribute> attributes;
        attributes.reserve(values.size());
        for (int64_t value : values)
            attributes.push_back(rewriter.getI64IntegerAttr(value));
        return rewriter.getArrayAttr(attributes);
    };
    mlir::OperationState state(location, SxmOp::getOperationName());
    state.addAttributes({
        rewriter.getNamedAttr("cycle", rewriter.getI64IntegerAttr(cycle)),
        rewriter.getNamedAttr("hemisphere", rewriter.getI64IntegerAttr(hemisphere)),
        rewriter.getNamedAttr("opcode", rewriter.getStringAttr(opcode)),
        rewriter.getNamedAttr("source_streams", integers(sourceStreams)),
        rewriter.getNamedAttr("destination_streams", integers(destinationStreams)),
        rewriter.getNamedAttr("permute_map", integers(permuteMap)),
        rewriter.getNamedAttr("weight_layout", rewriter.getStringAttr(weightLayout)),
    });
    rewriter.create(state);
}

std::array<int64_t, 32> identityMap()
{
    std::array<int64_t, 32> map {};
    for (int64_t lane = 0; lane < static_cast<int64_t>(map.size()); ++lane)
        map[static_cast<std::size_t>(lane)] = lane;
    return map;
}

std::array<int64_t, 32> blockDiagonalMap(int64_t diagonal,
    const target::LPUTargetModel& target)
{
    auto map = identityMap();
    const int64_t rows = target.throughput().tile_rows;
    const int64_t lanes = target.throughput().lanes_per_tile;
    for (int64_t destination = 0; destination < rows; ++destination) {
        const int64_t source = (diagonal + rows - destination) % rows;
        for (int64_t lane = 0; lane < lanes; ++lane)
            map[static_cast<std::size_t>(destination * lanes + lane)]
                = source * lanes + lane;
    }
    return map;
}

void emitWavefrontBeat(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle,
    int64_t hemisphere, int64_t diagonal,
    llvm::ArrayRef<int64_t> sourceStreams,
    llvm::ArrayRef<int64_t> transposeStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::StringRef weightLayout)
{
    requirePhysicalWidth(target, sourceStreams, transposeStreams);
    requirePhysicalWidth(target, transposeStreams, destinationStreams);
    emitSxm(rewriter, location, cycle, hemisphere, "transpose",
        sourceStreams, transposeStreams, identityMap());
    emitSxm(rewriter, location, cycle + 1, hemisphere, "permute",
        transposeStreams, destinationStreams,
        blockDiagonalMap(diagonal, target), weightLayout);
}

void emitWavefrontTail(mlir::IRRewriter& rewriter, mlir::Location location,
    const target::LPUTargetModel& target, int64_t cycle,
    int64_t hemisphere, int64_t diagonal,
    llvm::ArrayRef<int64_t> transposeStreams,
    llvm::ArrayRef<int64_t> destinationStreams,
    llvm::StringRef weightLayout)
{
    requirePhysicalWidth(target, transposeStreams, destinationStreams);
    emitSxm(rewriter, location, cycle + 1, hemisphere, "permute",
        transposeStreams, destinationStreams,
        blockDiagonalMap(diagonal, target), weightLayout);
}

} // namespace ftlpu::compiler::schedule::sxm_detail
