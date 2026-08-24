#pragma once

#include "ftlpu/compiler/Dialect/Kernel/Analysis/attention_graph.hpp"
#include "ftlpu/compiler/Dialect/Kernel/Analysis/ffn_graph.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <initializer_list>
#include <string>

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
    std::string layout;
    std::string hemisphere;
    int64_t bank = 0;
};

class RowAllocator {
public:
    explicit RowAllocator(int64_t capacity_rows = 8192)
        : capacity_rows_(capacity_rows)
    {
    }

    mlir::FailureOr<int64_t> allocate(int64_t rows);
    void release(int64_t offset, int64_t rows);

private:
    struct FreeBlock {
        int64_t offset;
        int64_t rows;
    };
    int64_t capacity_rows_;
    int64_t next_row_ = 0;
    llvm::SmallVector<FreeBlock> free_blocks_;
};

class EastMemoryAllocator {
public:
    explicit EastMemoryAllocator(const target::LPUTargetModel& target);

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

class FunctionMemoryPlanner {
public:
    FunctionMemoryPlanner(
        mlir::func::FuncOp function, EastMemoryAllocator& allocator);

    mlir::FailureOr<Allocation> allocate(
        mlir::Value value, PlacementKind kind);
    mlir::FailureOr<Allocation> lookup(mlir::Value value) const;
    mlir::LogicalResult bind(
        mlir::Value value, const Allocation& allocation);
    void release_before(mlir::Operation* operation);
    void replace_value(mlir::Value old_value, mlir::Value new_value);
    int64_t ordinal(mlir::Operation* operation) const;

private:
    EastMemoryAllocator& allocator_;
    llvm::DenseMap<mlir::Value, Allocation> allocations_;
    llvm::DenseSet<mlir::Value> externally_managed_;
    llvm::DenseMap<mlir::Value, int64_t> last_uses_;
    llvm::DenseMap<mlir::Operation*, int64_t> ordinals_;
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
Allocation fixed_allocation(PlacementKind kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, int64_t bytes,
    llvm::StringRef layout, llvm::StringRef hemisphere,
    int64_t bank = 0);
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
mlir::FailureOr<mlir::ArrayAttr> get_value_task_allocations(
    mlir::Value value);
mlir::FailureOr<mlir::DictionaryAttr> get_value_placement(
    mlir::Value value);
mlir::DictionaryAttr make_attention_placement(
    mlir::OpBuilder& builder, llvm::StringRef kind,
    llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, llvm::StringRef hemisphere,
    int64_t bank = 0);

mlir::LogicalResult lower_attention(kernel::AttentionGraph& graph,
    const target::LPUTargetModel& target, int64_t weight_bank,
    mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_ffn(kernel::FfnGraph& graph,
    const target::LPUTargetModel& target, EastMemoryAllocator& allocator,
    AllocateValueFn allocate_value, int64_t weight_bank,
    mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_swiglu(kernel::SwigluOp op,
    EastMemoryAllocator& allocator, AllocateValueFn allocate_value,
    mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_rms_norm(kernel::RmsNormOp op,
    const target::LPUTargetModel& target,
    RmsNormLoweringStrategy strategy, int64_t feedback_weight_base_row,
    int64_t feedback_weight_slice_base,
    int64_t weight_bank, FunctionMemoryPlanner& planner,
    mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_elementwise(kernel::ElementwiseOp op,
    const target::LPUTargetModel& target,
    EastMemoryAllocator& allocator, AllocateValueFn allocate_value,
    RmsNormLoweringStrategy rmsnorm_strategy, int64_t weight_bank,
    mlir::IRRewriter& rewriter);
mlir::LogicalResult lower_matmul(kernel::MatmulOp op,
    const target::LPUTargetModel& target,
    FunctionMemoryPlanner& planner,
    AllocateValueFn allocate_value, mlir::IRRewriter& rewriter);

} // namespace ftlpu::compiler::tensor_lowering
