#include "KernelToTensorLowering.hpp"

#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace ftlpu::compiler::tensor_lowering {

mlir::FailureOr<int64_t> RowAllocator::allocate(int64_t rows)
{
    for (size_t index = 0; index < free_blocks_.size(); ++index) {
        FreeBlock& block = free_blocks_[index];
        if (block.rows < rows) continue;
        const int64_t offset = block.offset;
        block.offset += rows;
        block.rows -= rows;
        if (block.rows == 0)
            free_blocks_.erase(free_blocks_.begin() + index);
        return offset;
    }
    if (rows > 8192 - next_row_) return mlir::failure();
    const int64_t offset = next_row_;
    next_row_ += rows;
    return offset;
}

void RowAllocator::release(int64_t offset, int64_t rows)
{
    FreeBlock released {offset, rows};
    auto position = llvm::lower_bound(free_blocks_, released.offset,
        [](const FreeBlock& block, int64_t value) {
            return block.offset < value;
        });
    free_blocks_.insert(position, released);

    llvm::SmallVector<FreeBlock> merged;
    for (const FreeBlock& block : free_blocks_) {
        if (!merged.empty()
            && merged.back().offset + merged.back().rows == block.offset)
            merged.back().rows += block.rows;
        else
            merged.push_back(block);
    }
    free_blocks_ = std::move(merged);
    if (!free_blocks_.empty()
        && free_blocks_.back().offset + free_blocks_.back().rows == next_row_) {
        next_row_ = free_blocks_.back().offset;
        free_blocks_.pop_back();
    }
}

mlir::FailureOr<Allocation> EastMemoryAllocator::allocate(
    PlacementKind kind, int64_t bytes)
{
    constexpr int64_t vector_bytes = 320;
    const int64_t slice_count = kind == PlacementKind::Weight ? 16
        : kind == PlacementKind::Result ? 4
                                          : 1;
    const int64_t instruction_count =
        (bytes + vector_bytes * slice_count - 1)
        / (vector_bytes * slice_count);
    const int64_t row_span = (instruction_count - 1) * 16 + 1;
    const auto base_row = allocator(kind).allocate(row_span);
    if (mlir::failed(base_row)) return mlir::failure();

    llvm::SmallVector<int64_t, 16> slices;
    const int64_t first_slice = kind == PlacementKind::Weight ? 0
        : kind == PlacementKind::Result ? 40
        : kind == PlacementKind::VxmResult ? 40
        : kind == PlacementKind::VxmResult1 ? 41
        : kind == PlacementKind::FinalResult ? 42
                                              : 32;
    for (int64_t index = 0; index < slice_count; ++index)
        slices.push_back(first_slice + index);
    return Allocation {kind, std::move(slices), *base_row,
        instruction_count, kind == PlacementKind::Weight ? -16 : 16,
        row_span, bytes};
}

void EastMemoryAllocator::release(const Allocation& allocation)
{
    allocator(allocation.kind)
        .release(allocation.base_row, allocation.row_span);
}

RowAllocator& EastMemoryAllocator::allocator(PlacementKind kind)
{
    if (kind == PlacementKind::Weight) return weight_;
    if (kind == PlacementKind::Result) return result_;
    if (kind == PlacementKind::VxmResult) return vxm_result_;
    if (kind == PlacementKind::VxmResult1) return vxm_result1_;
    if (kind == PlacementKind::FinalResult) return final_result_;
    return activation_;
}

mlir::FailureOr<int64_t> get_static_tensor_bytes(
    mlir::RankedTensorType type)
{
    if (!type || !type.hasStaticShape()) return mlir::failure();
    int64_t element_bits = 0;
    if (auto integer =
            llvm::dyn_cast<mlir::IntegerType>(type.getElementType()))
        element_bits = integer.getWidth();
    else if (auto floating =
                 llvm::dyn_cast<mlir::FloatType>(type.getElementType()))
        element_bits = floating.getWidth();
    if (element_bits <= 0 || element_bits % 8 != 0)
        return mlir::failure();

    int64_t bytes = element_bits / 8;
    for (int64_t dimension : type.getShape()) {
        if (dimension <= 0
            || bytes > std::numeric_limits<int64_t>::max() / dimension)
            return mlir::failure();
        bytes *= dimension;
    }
    return bytes;
}

mlir::DictionaryAttr make_address_attr(
    mlir::OpBuilder& builder, const Allocation& allocation)
{
    return builder.getDictionaryAttr({
        builder.getNamedAttr("device", builder.getI64IntegerAttr(0)),
        builder.getNamedAttr(
            "hemisphere", builder.getStringAttr("east")),
        builder.getNamedAttr(
            "slice", builder.getI64IntegerAttr(allocation.slices.front())),
        builder.getNamedAttr(
            "bank", builder.getI64IntegerAttr(allocation.base_row / 4096)),
        builder.getNamedAttr(
            "word", builder.getI64IntegerAttr(allocation.base_row % 4096)),
        builder.getNamedAttr("byte", builder.getI64IntegerAttr(0)),
    });
}

mlir::DictionaryAttr make_placement_attr(
    mlir::OpBuilder& builder, const Allocation& allocation)
{
    llvm::SmallVector<mlir::Attribute> slices;
    for (int64_t slice : allocation.slices)
        slices.push_back(builder.getI64IntegerAttr(slice));
    const char* kind = allocation.kind == PlacementKind::Weight
        ? "mxm_weight_striped"
        : allocation.kind == PlacementKind::Result
        ? "int32_byte_planar"
        : "vector";
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("slices", builder.getArrayAttr(slices)),
        builder.getNamedAttr(
            "base_row", builder.getI64IntegerAttr(allocation.base_row)),
        builder.getNamedAttr("instruction_count",
            builder.getI64IntegerAttr(allocation.instruction_count)),
        builder.getNamedAttr("address_stride",
            builder.getI64IntegerAttr(allocation.address_stride)),
    });
}

Allocation fixed_allocation(PlacementKind kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, int64_t bytes)
{
    return Allocation {kind, llvm::SmallVector<int64_t, 16>(slices),
        base_row, instruction_count, 1, instruction_count, bytes};
}

bool is_w8a16_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target)
{
    return graph.gate.getLhs().getType().getElementType().isF16()
        && graph.gate.getRhs().getType().getElementType().isInteger(8)
        && graph.up.getRhs().getType().getElementType().isInteger(8)
        && graph.output.getRhs().getType().getElementType().isInteger(8)
        && graph.output.getResult().getType().getElementType().isF16()
        && target.supports_w8a16_ffn_shape(graph.gate.getM(),
            graph.gate.getK(), graph.gate.getN(), graph.output.getN());
}

mlir::DictionaryAttr make_profile_placement(
    mlir::OpBuilder& builder, const Allocation& allocation,
    llvm::StringRef kind, llvm::StringRef hemisphere)
{
    llvm::SmallVector<mlir::Attribute> slices;
    for (int64_t slice : allocation.slices)
        slices.push_back(builder.getI64IntegerAttr(slice));
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr(
            "hemisphere", builder.getStringAttr(hemisphere)),
        builder.getNamedAttr("slices", builder.getArrayAttr(slices)),
        builder.getNamedAttr(
            "base_row", builder.getI64IntegerAttr(allocation.base_row)),
        builder.getNamedAttr("instruction_count",
            builder.getI64IntegerAttr(allocation.instruction_count)),
        builder.getNamedAttr("address_stride",
            builder.getI64IntegerAttr(allocation.address_stride)),
    });
}

mlir::DictionaryAttr make_task_allocation(
    mlir::OpBuilder& builder, const Allocation& allocation,
    mlir::DictionaryAttr placement)
{
    return builder.getDictionaryAttr({
        builder.getNamedAttr(
            "address", make_address_attr(builder, allocation)),
        builder.getNamedAttr("placement", placement),
        builder.getNamedAttr(
            "bytes", builder.getI64IntegerAttr(allocation.bytes)),
    });
}

mlir::ArrayAttr make_task_allocations(mlir::OpBuilder& builder,
    std::initializer_list<mlir::DictionaryAttr> allocations)
{
    llvm::SmallVector<mlir::Attribute> attributes(
        allocations.begin(), allocations.end());
    return builder.getArrayAttr(attributes);
}

mlir::DictionaryAttr make_attention_placement(
    mlir::OpBuilder& builder, llvm::StringRef kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, llvm::StringRef hemisphere)
{
    llvm::SmallVector<mlir::Attribute> attrs;
    for (int64_t slice : slices)
        attrs.push_back(builder.getI64IntegerAttr(slice));
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr(
            "hemisphere", builder.getStringAttr(hemisphere)),
        builder.getNamedAttr("slices", builder.getArrayAttr(attrs)),
        builder.getNamedAttr(
            "base_row", builder.getI64IntegerAttr(base_row)),
        builder.getNamedAttr("instruction_count",
            builder.getI64IntegerAttr(instruction_count)),
        builder.getNamedAttr(
            "address_stride", builder.getI64IntegerAttr(1)),
    });
}

} // namespace ftlpu::compiler::tensor_lowering
