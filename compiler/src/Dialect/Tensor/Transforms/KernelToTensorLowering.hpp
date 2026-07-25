#pragma once

#include "ftlpu/compiler/Dialect/Kernel/Analysis/attention_graph.hpp"
#include "ftlpu/compiler/Dialect/Kernel/Analysis/ffn_graph.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/PatternMatch.h"

#include <initializer_list>

namespace ftlpu::compiler::tensor_lowering {

enum class PlacementKind {
    Activation,
    Weight,
    Result,
    VxmResult,
    VxmResult1,
    FinalResult
};

struct Allocation {
    PlacementKind kind;
    llvm::SmallVector<int64_t, 16> slices;
    int64_t base_row;
    int64_t instruction_count;
    int64_t address_stride;
    int64_t row_span;
    int64_t bytes;
};

class RowAllocator {
public:
    mlir::FailureOr<int64_t> allocate(int64_t rows);
    void release(int64_t offset, int64_t rows);

private:
    struct FreeBlock {
        int64_t offset;
        int64_t rows;
    };
    int64_t next_row_ = 0;
    llvm::SmallVector<FreeBlock> free_blocks_;
};

class EastMemoryAllocator {
public:
    mlir::FailureOr<Allocation> allocate(PlacementKind kind, int64_t bytes);
    void release(const Allocation& allocation);

private:
    RowAllocator& allocator(PlacementKind kind);

    RowAllocator activation_;
    RowAllocator weight_;
    RowAllocator result_;
    RowAllocator vxm_result_;
    RowAllocator vxm_result1_;
    RowAllocator final_result_;
};

using AllocateValueFn =
    llvm::function_ref<mlir::FailureOr<Allocation>(
        mlir::Value, PlacementKind)>;

mlir::FailureOr<int64_t> get_static_tensor_bytes(
    mlir::RankedTensorType type);
mlir::DictionaryAttr make_address_attr(
    mlir::OpBuilder& builder, const Allocation& allocation);
mlir::DictionaryAttr make_placement_attr(
    mlir::OpBuilder& builder, const Allocation& allocation);
Allocation fixed_allocation(PlacementKind kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, int64_t bytes);
bool is_w8a16_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target);
mlir::DictionaryAttr make_profile_placement(
    mlir::OpBuilder& builder, const Allocation& allocation,
    llvm::StringRef kind, llvm::StringRef hemisphere);
mlir::DictionaryAttr make_task_allocation(
    mlir::OpBuilder& builder, const Allocation& allocation,
    mlir::DictionaryAttr placement);
mlir::ArrayAttr make_task_allocations(mlir::OpBuilder& builder,
    std::initializer_list<mlir::DictionaryAttr> allocations);
mlir::DictionaryAttr make_attention_placement(
    mlir::OpBuilder& builder, llvm::StringRef kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, llvm::StringRef hemisphere);

mlir::LogicalResult lower_attention(kernel::AttentionGraph& graph,
    const target::LPUTargetModel& target, mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target, EastMemoryAllocator& allocator,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_swiglu(kernel::SwigluOp op,
    EastMemoryAllocator& allocator, AllocateValueFn allocate_value,
    mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_matmul(kernel::MatmulOp op,
    EastMemoryAllocator& allocator,
    llvm::DenseMap<mlir::Value, Allocation>& allocations,
    llvm::DenseMap<mlir::Value, int64_t>& last_uses,
    const llvm::DenseMap<mlir::Operation*, int64_t>& ordinals,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter);

} // namespace ftlpu::compiler::tensor_lowering
