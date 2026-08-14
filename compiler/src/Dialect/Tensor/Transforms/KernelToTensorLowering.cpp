#include "KernelToTensorLowering.hpp"

#include "ftlpu/compiler/Support/float_format.hpp"

#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace ftlpu::compiler::tensor_lowering {

FunctionMemoryPlanner::FunctionMemoryPlanner(
    mlir::func::FuncOp function, EastMemoryAllocator& allocator)
    : allocator_(allocator)
{
    int64_t ordinal = 0;
    for (mlir::Operation& operation : function.getBody().front()) {
        ordinals_[&operation] = ordinal;
        for (mlir::Value operand : operation.getOperands())
            last_uses_[operand] = ordinal;
        ++ordinal;
    }
}

mlir::FailureOr<Allocation> FunctionMemoryPlanner::allocate(
    mlir::Value value, PlacementKind kind)
{
    if (const auto found = allocations_.find(value);
        found != allocations_.end())
        return found->second.kind == kind
            ? mlir::FailureOr<Allocation>(found->second)
            : mlir::FailureOr<Allocation>(mlir::failure());
    const auto type =
        llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
    const auto bytes = get_static_tensor_bytes(type);
    if (mlir::failed(bytes)) return mlir::failure();
    auto allocation = allocator_.allocate(kind, *bytes);
    if (mlir::failed(allocation)) return mlir::failure();
    allocations_.try_emplace(value, *allocation);
    return *allocation;
}

mlir::FailureOr<Allocation> FunctionMemoryPlanner::lookup(
    mlir::Value value) const
{
    const auto found = allocations_.find(value);
    if (found == allocations_.end()) return mlir::failure();
    return found->second;
}

mlir::LogicalResult FunctionMemoryPlanner::bind(
    mlir::Value value, const Allocation& allocation)
{
    if (const auto found = allocations_.find(value);
        found != allocations_.end())
        return found->second.slices == allocation.slices
                && found->second.base_row == allocation.base_row
            ? mlir::success()
            : mlir::failure();
    allocations_.try_emplace(value, allocation);
    externally_managed_.insert(value);
    return mlir::success();
}

void FunctionMemoryPlanner::release_before(mlir::Operation* operation)
{
    const int64_t current = ordinal(operation);
    llvm::SmallVector<mlir::Value> expired;
    for (const auto& [value, allocation] : allocations_) {
        const auto use = last_uses_.find(value);
        if (use == last_uses_.end() || use->second < current)
            expired.push_back(value);
    }
    for (mlir::Value value : expired) {
        if (!externally_managed_.contains(value))
            allocator_.release(allocations_.lookup(value));
        externally_managed_.erase(value);
        allocations_.erase(value);
    }
}

void FunctionMemoryPlanner::replace_value(
    mlir::Value old_value, mlir::Value new_value)
{
    if (auto allocation = allocations_.find(old_value);
        allocation != allocations_.end()) {
        allocations_.try_emplace(new_value, allocation->second);
        if (externally_managed_.erase(old_value))
            externally_managed_.insert(new_value);
        allocations_.erase(allocation);
    }
    if (auto use = last_uses_.find(old_value);
        use != last_uses_.end()) {
        last_uses_[new_value] = use->second;
        last_uses_.erase(use);
    }
}

int64_t FunctionMemoryPlanner::ordinal(
    mlir::Operation* operation) const
{
    const auto found = ordinals_.find(operation);
    return found == ordinals_.end() ? 0 : found->second;
}

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
    if (rows > capacity_rows_ - next_row_) return mlir::failure();
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
        row_span, bytes, "", "east", 0};
}

EastMemoryAllocator::EastMemoryAllocator(
    const target::LPUTargetModel& target)
    : activation_(target.memory().words_per_bank),
      weight_(target.memory().words_per_bank),
      result_(target.memory().words_per_bank),
      vxm_result_(target.memory().words_per_bank),
      vxm_result1_(target.memory().words_per_bank),
      final_result_(target.memory().words_per_bank)
{
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
            "bank", builder.getI64IntegerAttr(allocation.bank)),
        builder.getNamedAttr(
            "word", builder.getI64IntegerAttr(allocation.base_row)),
        builder.getNamedAttr("byte", builder.getI64IntegerAttr(0)),
    });
}

mlir::DictionaryAttr make_placement_attr(
    mlir::OpBuilder& builder, const Allocation& allocation)
{
    llvm::SmallVector<mlir::Attribute> slices;
    for (int64_t slice : allocation.slices)
        slices.push_back(builder.getI64IntegerAttr(slice));
    const std::string kind = !allocation.layout.empty()
        ? allocation.layout
        : allocation.kind == PlacementKind::Weight
        ? "mxm_weight_striped"
        : allocation.kind == PlacementKind::Result
        ? "int32_byte_planar"
        : "vector";
    llvm::SmallVector<mlir::NamedAttribute> attributes {
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("slices", builder.getArrayAttr(slices)),
        builder.getNamedAttr(
            "base_row", builder.getI64IntegerAttr(allocation.base_row)),
        builder.getNamedAttr("instruction_count",
            builder.getI64IntegerAttr(allocation.instruction_count)),
        builder.getNamedAttr("address_stride",
            builder.getI64IntegerAttr(allocation.address_stride)),
        builder.getNamedAttr(
            "bank", builder.getI64IntegerAttr(allocation.bank)),
    };
    if (!allocation.hemisphere.empty())
        attributes.push_back(builder.getNamedAttr(
            "hemisphere", builder.getStringAttr(allocation.hemisphere)));
    return builder.getDictionaryAttr(attributes);
}

Allocation fixed_allocation(PlacementKind kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, int64_t bytes)
{
    return fixed_allocation(kind, slices, base_row,
        instruction_count, bytes, {}, "east");
}

Allocation fixed_allocation(PlacementKind kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, int64_t bytes,
    llvm::StringRef layout, llvm::StringRef hemisphere, int64_t bank)
{
    return Allocation {kind, llvm::SmallVector<int64_t, 16>(slices),
        base_row, instruction_count, 1, instruction_count, bytes,
        layout.str(), hemisphere.str(), bank};
}

bool is_w8a16_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target)
{
    return is_lpu_16bit_float(
               graph.gate.getLhs().getType().getElementType())
        && graph.gate.getRhs().getType().getElementType().isInteger(8)
        && graph.up.getRhs().getType().getElementType().isInteger(8)
        && graph.output.getRhs().getType().getElementType().isInteger(8)
        && is_lpu_16bit_float(
            graph.output.getResult().getType().getElementType())
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
        builder.getNamedAttr(
            "bank", builder.getI64IntegerAttr(allocation.bank)),
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

mlir::FailureOr<mlir::ArrayAttr> get_value_task_allocations(
    mlir::Value value)
{
    if (auto op = value.getDefiningOp<tensor::MatmulTaskOp>())
        return op.getResultAllocations();
    if (auto op = value.getDefiningOp<tensor::RmsNormTaskOp>())
        return op.getResultAllocations();
    if (auto op = value.getDefiningOp<tensor::ElementwiseTaskOp>())
        return op.getResultAllocations();
    if (auto op = value.getDefiningOp<tensor::ProjectionTaskOp>()) {
        auto placement =
            op.getMemoryPlan().getAs<mlir::DictionaryAttr>("result");
        const auto type =
            llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
        if (!placement || !type) return mlir::failure();
        auto bytes = get_static_tensor_bytes(type);
        if (mlir::failed(bytes)) return mlir::failure();
        llvm::SmallVector<int64_t> slices;
        for (mlir::Attribute slice :
            placement.getAs<mlir::ArrayAttr>("slices"))
            slices.push_back(
                llvm::cast<mlir::IntegerAttr>(slice).getInt());
        const int64_t baseRow =
            placement.getAs<mlir::IntegerAttr>("base_row").getInt();
        const int64_t instructionCount =
            placement.getAs<mlir::IntegerAttr>(
                "instruction_count").getInt();
        const auto kind =
            placement.getAs<mlir::StringAttr>("kind").getValue();
        const auto hemisphere =
            placement.getAs<mlir::StringAttr>("hemisphere").getValue();
        const Allocation allocation = fixed_allocation(
            PlacementKind::FinalResult, slices, baseRow,
            instructionCount, *bytes, kind, hemisphere);
        mlir::OpBuilder builder(value.getContext());
        return make_task_allocations(builder,
            {make_task_allocation(builder, allocation, placement)});
    }
    return mlir::failure();
}

mlir::FailureOr<mlir::DictionaryAttr> get_value_placement(
    mlir::Value value)
{
    auto allocations = get_value_task_allocations(value);
    if (mlir::failed(allocations) || allocations->empty())
        return mlir::failure();
    auto allocation =
        llvm::dyn_cast<mlir::DictionaryAttr>((*allocations)[0]);
    if (!allocation) return mlir::failure();
    auto placement =
        allocation.getAs<mlir::DictionaryAttr>("placement");
    if (!placement) return mlir::failure();
    return placement;
}

mlir::DictionaryAttr make_attention_placement(
    mlir::OpBuilder& builder, llvm::StringRef kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, llvm::StringRef hemisphere, int64_t bank)
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
        builder.getNamedAttr("bank", builder.getI64IntegerAttr(bank)),
    });
}

} // namespace ftlpu::compiler::tensor_lowering
