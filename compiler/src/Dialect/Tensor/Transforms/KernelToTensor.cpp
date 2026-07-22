#include "ftlpu/compiler/Dialect/Kernel/IR/kernel_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <limits>

namespace ftlpu::compiler {
namespace {

enum class PlacementKind { Activation, Weight, Result, VxmResult, VxmResult1, FinalResult };

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
    mlir::FailureOr<int64_t> allocate(int64_t rows)
    {
        for (size_t index = 0; index < free_blocks_.size(); ++index) {
            FreeBlock& block = free_blocks_[index];
            if (block.rows < rows) continue;
            const int64_t offset = block.offset;
            block.offset += rows;
            block.rows -= rows;
            if (block.rows == 0) free_blocks_.erase(free_blocks_.begin() + index);
            return offset;
        }
        if (rows > 8192 - next_row_) return mlir::failure();
        const int64_t offset = next_row_;
        next_row_ += rows;
        return offset;
    }

    void release(int64_t offset, int64_t rows)
    {
        FreeBlock released{offset, rows};
        auto position = llvm::lower_bound(free_blocks_, released.offset,
            [](const FreeBlock& block, int64_t offset) { return block.offset < offset; });
        free_blocks_.insert(position, released);

        llvm::SmallVector<FreeBlock> merged;
        for (const FreeBlock& block : free_blocks_) {
            if (!merged.empty() && merged.back().offset + merged.back().rows == block.offset)
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
    mlir::FailureOr<Allocation> allocate(PlacementKind kind, int64_t bytes)
    {
        constexpr int64_t vector_bytes = 320;
        const int64_t slice_count = kind == PlacementKind::Weight ? 16
            : kind == PlacementKind::Result ? 4 : 1;
        const int64_t instruction_count =
            (bytes + vector_bytes * slice_count - 1) / (vector_bytes * slice_count);
        const int64_t row_span = (instruction_count - 1) * 16 + 1;
        RowAllocator& pool = allocator(kind);
        const auto base_row = pool.allocate(row_span);
        if (mlir::failed(base_row)) return mlir::failure();
        llvm::SmallVector<int64_t, 16> slices;
        const int64_t first_slice = kind == PlacementKind::Weight ? 0
            : kind == PlacementKind::Result ? 40
            : kind == PlacementKind::VxmResult ? 40
            : kind == PlacementKind::VxmResult1 ? 41
            : kind == PlacementKind::FinalResult ? 42 : 32;
        for (int64_t index = 0; index < slice_count; ++index)
            slices.push_back(first_slice + index);
        return Allocation{kind, std::move(slices), *base_row, instruction_count,
            kind == PlacementKind::Weight ? -16 : 16, row_span, bytes};
    }

    void release(const Allocation& allocation)
    {
        allocator(allocation.kind).release(allocation.base_row, allocation.row_span);
    }

private:
    RowAllocator& allocator(PlacementKind kind)
    {
        if (kind == PlacementKind::Weight) return weight_;
        if (kind == PlacementKind::Result) return result_;
        if (kind == PlacementKind::VxmResult) return vxm_result_;
        if (kind == PlacementKind::VxmResult1) return vxm_result1_;
        if (kind == PlacementKind::FinalResult) return final_result_;
        return activation_;
    }

    RowAllocator activation_;
    RowAllocator weight_;
    RowAllocator result_;
    RowAllocator vxm_result_;
    RowAllocator vxm_result1_;
    RowAllocator final_result_;
};

mlir::FailureOr<int64_t> get_static_tensor_bytes(mlir::RankedTensorType type)
{
    if (!type || !type.hasStaticShape()) return mlir::failure();

    int64_t element_bits = 0;
    if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type.getElementType()))
        element_bits = integer.getWidth();
    else if (auto floating = llvm::dyn_cast<mlir::FloatType>(type.getElementType()))
        element_bits = floating.getWidth();
    if (element_bits <= 0 || element_bits % 8 != 0) return mlir::failure();

    int64_t bytes = element_bits / 8;
    for (int64_t dimension : type.getShape()) {
        if (dimension <= 0 || bytes > std::numeric_limits<int64_t>::max() / dimension)
            return mlir::failure();
        bytes *= dimension;
    }
    return bytes;
}

mlir::DictionaryAttr make_address_attr(mlir::OpBuilder& builder, const Allocation& allocation)
{
    return builder.getDictionaryAttr({
        builder.getNamedAttr("device", builder.getI64IntegerAttr(0)),
        builder.getNamedAttr("hemisphere", builder.getStringAttr("east")),
        builder.getNamedAttr("slice", builder.getI64IntegerAttr(allocation.slices.front())),
        builder.getNamedAttr("bank", builder.getI64IntegerAttr(allocation.base_row / 4096)),
        builder.getNamedAttr("word", builder.getI64IntegerAttr(allocation.base_row % 4096)),
        builder.getNamedAttr("byte", builder.getI64IntegerAttr(0)),
    });
}

mlir::DictionaryAttr make_placement_attr(mlir::OpBuilder& builder, const Allocation& allocation)
{
    llvm::SmallVector<mlir::Attribute> slices;
    for (int64_t slice : allocation.slices)
        slices.push_back(builder.getI64IntegerAttr(slice));
    const char* kind = allocation.kind == PlacementKind::Weight ? "mxm_weight_striped"
        : allocation.kind == PlacementKind::Result ? "int32_byte_planar" : "vector";
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("slices", builder.getArrayAttr(slices)),
        builder.getNamedAttr("base_row", builder.getI64IntegerAttr(allocation.base_row)),
        builder.getNamedAttr("instruction_count", builder.getI64IntegerAttr(allocation.instruction_count)),
        builder.getNamedAttr("address_stride", builder.getI64IntegerAttr(allocation.address_stride)),
    });
}

Allocation fixed_allocation(PlacementKind kind, llvm::ArrayRef<int64_t> slices,
    int64_t base_row, int64_t instruction_count, int64_t bytes)
{
    return Allocation {kind, llvm::SmallVector<int64_t, 16>(slices), base_row,
        instruction_count, 1, instruction_count, bytes};
}

bool is_w8a16_ffn(kernel::FfnOp op, const target::LPUTargetModel& target)
{
    return op.getInput().getType().getElementType().isF16()
        && op.getGateWeight().getType().getElementType().isInteger(8)
        && op.getUpWeight().getType().getElementType().isInteger(8)
        && op.getDownWeight().getType().getElementType().isInteger(8)
        && op.getResult().getType().getElementType().isF16()
        && target.supports_w8a16_ffn_shape(
            op.getM(), op.getK(), op.getHidden(), op.getN());
}

mlir::DictionaryAttr make_profile_placement(mlir::OpBuilder& builder,
    const Allocation& allocation, llvm::StringRef kind, llvm::StringRef hemisphere)
{
    llvm::SmallVector<mlir::Attribute> slices;
    for (int64_t slice : allocation.slices) slices.push_back(builder.getI64IntegerAttr(slice));
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("hemisphere", builder.getStringAttr(hemisphere)),
        builder.getNamedAttr("slices", builder.getArrayAttr(slices)),
        builder.getNamedAttr("base_row", builder.getI64IntegerAttr(allocation.base_row)),
        builder.getNamedAttr("instruction_count", builder.getI64IntegerAttr(allocation.instruction_count)),
        builder.getNamedAttr("address_stride", builder.getI64IntegerAttr(allocation.address_stride)),
    });
}

mlir::DictionaryAttr make_attention_placement(mlir::OpBuilder& builder,
    llvm::StringRef kind, llvm::ArrayRef<int64_t> slices, int64_t base_row,
    int64_t instruction_count, llvm::StringRef hemisphere)
{
    llvm::SmallVector<mlir::Attribute> attrs;
    for (const int64_t slice : slices) attrs.push_back(builder.getI64IntegerAttr(slice));
    return builder.getDictionaryAttr({
        builder.getNamedAttr("kind", builder.getStringAttr(kind)),
        builder.getNamedAttr("hemisphere", builder.getStringAttr(hemisphere)),
        builder.getNamedAttr("slices", builder.getArrayAttr(attrs)),
        builder.getNamedAttr("base_row", builder.getI64IntegerAttr(base_row)),
        builder.getNamedAttr("instruction_count", builder.getI64IntegerAttr(instruction_count)),
        builder.getNamedAttr("address_stride", builder.getI64IntegerAttr(1)),
    });
}

class LowerKernelToTensorPass final
    : public mlir::PassWrapper<LowerKernelToTensorPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKernelToTensorPass)

    llvm::StringRef getArgument() const final { return "ftlpu-kernel-to-tensor"; }
    llvm::StringRef getDescription() const final
    {
        return "Assigns physical LPU MEM storage to Kernel IR tensors";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError("MEM lifetime allocation currently requires a single-block function");
            signalPassFailure();
            return;
        }

        EastMemoryAllocator allocator;
        const target::LPUTargetModel target;
        llvm::DenseMap<mlir::Value, Allocation> allocations;
        llvm::DenseMap<mlir::Value, int64_t> last_uses;
        llvm::DenseMap<mlir::Operation*, int64_t> ordinals;
        llvm::SmallVector<kernel::MatmulOp> matmuls;
        llvm::SmallVector<kernel::SwigluOp> swiglus;
        llvm::SmallVector<kernel::FfnOp> ffns;
        llvm::SmallVector<kernel::AttentionOp> attentions;
        int64_t ordinal = 0;
        for (mlir::Operation& operation : function.getBody().front()) {
            ordinals[&operation] = ordinal;
            for (mlir::Value operand : operation.getOperands()) last_uses[operand] = ordinal;
            if (auto matmul = llvm::dyn_cast<kernel::MatmulOp>(&operation)) matmuls.push_back(matmul);
            if (auto swiglu = llvm::dyn_cast<kernel::SwigluOp>(&operation)) swiglus.push_back(swiglu);
            if (auto ffn = llvm::dyn_cast<kernel::FfnOp>(&operation)) ffns.push_back(ffn);
            if (auto attention = llvm::dyn_cast<kernel::AttentionOp>(&operation)) attentions.push_back(attention);
            ++ordinal;
        }

        auto allocate_value = [&](mlir::Value value, PlacementKind kind) -> mlir::FailureOr<Allocation> {
            if (const auto found = allocations.find(value); found != allocations.end())
                return found->second.kind == kind ? mlir::FailureOr<Allocation>(found->second)
                                                  : mlir::FailureOr<Allocation>(mlir::failure());
            const auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
            const auto bytes = get_static_tensor_bytes(type);
            if (mlir::failed(bytes)) return mlir::failure();
            const auto allocation = allocator.allocate(kind, *bytes);
            if (mlir::failed(allocation)) return mlir::failure();
            allocations.try_emplace(value, *allocation);
            return *allocation;
        };

        auto argument_kind = [](mlir::BlockArgument argument) {
            for (mlir::OpOperand& use : argument.getUses()) {
                if (auto matmul = llvm::dyn_cast<kernel::MatmulOp>(use.getOwner())) {
                    if (use.getOperandNumber() == 1) return PlacementKind::Weight;
                }
                if (auto swiglu = llvm::dyn_cast<kernel::SwigluOp>(use.getOwner())) {
                    if (use.getOperandNumber() == 1 || use.getOperandNumber() == 2)
                        return PlacementKind::Weight;
                }
                if (auto ffn = llvm::dyn_cast<kernel::FfnOp>(use.getOwner())) {
                    if (use.getOperandNumber() >= 1 && use.getOperandNumber() <= 3)
                        return PlacementKind::Weight;
                }
                if (auto attention = llvm::dyn_cast<kernel::AttentionOp>(use.getOwner())) {
                    if (use.getOperandNumber() >= 1) return PlacementKind::Weight;
                }
            }
            return PlacementKind::Activation;
        };

        // Function arguments are model inputs and coexist in MEM at entry.
        for (mlir::BlockArgument argument : function.getArguments()) {
            if (!last_uses.contains(argument)) continue;
            bool fixed_w8a16_operand = false;
            for (mlir::OpOperand& use : argument.getUses()) {
                if (auto ffn = llvm::dyn_cast<kernel::FfnOp>(use.getOwner()))
                    fixed_w8a16_operand |= is_w8a16_ffn(ffn, target);
                if (llvm::isa<kernel::AttentionOp>(use.getOwner())) fixed_w8a16_operand = true;
            }
            if (fixed_w8a16_operand) continue;
            if (mlir::failed(allocate_value(argument, argument_kind(argument)))) {
                function.emitError("cannot allocate function inputs in the east MEM hemisphere");
                signalPassFailure();
                return;
            }
        }

        mlir::IRRewriter rewriter(&getContext());
        for (kernel::AttentionOp op : attentions) {
            const int64_t tile = target.throughput().mxm_rows;
            const int64_t blocks = op.getSeqLen() / tile;
            const int64_t query_width = op.getQueryHeads() * op.getHeadDim();
            const int64_t kv_width = op.getKvHeads() * op.getHeadDim();
            const auto attention_weight_rows = [&](int64_t columns) {
                const int64_t head_groups = (columns + 2 * op.getHeadDim() - 1)
                    / (2 * op.getHeadDim());
                return head_groups * (op.getHidden() / tile) * 8;
            };
            llvm::SmallVector<int64_t> weight_slices;
            for (int64_t index = 0; index < target.memory().w8a16_weight_slice_count; ++index)
                weight_slices.push_back(index * target.memory().w8a16_weight_slice_stride);
            const llvm::SmallVector<int64_t> activation_slices {32, 33, 34, 35};
            const llvm::SmallVector<int64_t> output_slices {0, 1, 2, 3};
            const int64_t q_weight_rows = attention_weight_rows(query_width);
            const int64_t k_weight_rows = attention_weight_rows(kv_width);
            const int64_t v_weight_rows = attention_weight_rows(kv_width);
            const int64_t o_weight_rows = op.getHidden() * query_width
                / (target.memory().hemispheres * target.memory().w8a16_weight_slice_count * tile);
            const int64_t query_rows = op.getQueryHeads() * blocks * target.throughput().tile_rows;
            const int64_t score_rows = op.getQueryHeads() * blocks * op.getSeqLen();
            const int64_t context_rows = op.getQueryHeads() * op.getSeqLen();
            rewriter.setInsertionPoint(op);
            const auto plan = rewriter.getDictionaryAttr({
                rewriter.getNamedAttr("input", make_attention_placement(rewriter,
                    "fp16_mxm_activation_planar", activation_slices, 0,
                    op.getSeqLen() * op.getHidden() / tile, "both")),
                rewriter.getNamedAttr("query_weight", make_attention_placement(rewriter,
                    "w8a16_attention_weight_striped", weight_slices, 0, q_weight_rows, "both")),
                rewriter.getNamedAttr("key_weight", make_attention_placement(rewriter,
                    "w8a16_attention_weight_striped", weight_slices, q_weight_rows, k_weight_rows, "both")),
                rewriter.getNamedAttr("value_weight", make_attention_placement(rewriter,
                    "w8a16_attention_weight_striped", weight_slices, q_weight_rows + k_weight_rows, v_weight_rows, "both")),
                rewriter.getNamedAttr("output_weight", make_attention_placement(rewriter,
                    "w8a16_mxm_weight_striped", weight_slices, q_weight_rows + k_weight_rows + v_weight_rows,
                    o_weight_rows, "east")),
                rewriter.getNamedAttr("query", make_attention_placement(rewriter,
                    "fp16_query_iw", output_slices, 7600, query_rows, "both")),
                rewriter.getNamedAttr("key", make_attention_placement(rewriter,
                    "fp16_head_planar", output_slices, 0, op.getKvHeads() * op.getSeqLen(), "both")),
                rewriter.getNamedAttr("value", make_attention_placement(rewriter,
                    "fp16_head_planar", output_slices, op.getKvHeads() * op.getSeqLen(), op.getKvHeads() * op.getSeqLen(), "both")),
                rewriter.getNamedAttr("score", make_attention_placement(rewriter,
                    "fp16_score_block", llvm::ArrayRef<int64_t>({8, 9, 10, 11}), 3000, score_rows, "both")),
                rewriter.getNamedAttr("exp", make_attention_placement(rewriter,
                    "fp16_score_block", llvm::ArrayRef<int64_t>({12, 13, 14, 15}), 3000, score_rows, "both")),
                rewriter.getNamedAttr("probability", make_attention_placement(rewriter,
                    "fp16_score_block", llvm::ArrayRef<int64_t>({16, 17}), 0, score_rows, "both")),
                rewriter.getNamedAttr("rope", make_attention_placement(rewriter,
                    "fp16_rope_table", llvm::ArrayRef<int64_t>({4, 5, 6, 7}), 7000,
                    op.getSeqLen(), "both")),
                rewriter.getNamedAttr("context", make_attention_placement(rewriter,
                    "fp16_head_planar", llvm::ArrayRef<int64_t>({20, 21, 22, 23, 24, 25, 26, 27}),
                    2000, context_rows, "both")),
                rewriter.getNamedAttr("result", make_attention_placement(rewriter,
                    "fp16_pair_planar", llvm::ArrayRef<int64_t>({28, 29, 30, 31}), 0,
                    op.getSeqLen() * op.getHidden() / (tile * 2), "east")),
            });
            mlir::OperationState state(op.getLoc(), tensor::AttentionOp::getOperationName());
            state.addOperands(op->getOperands());
            state.addTypes(op.getResult().getType());
            state.addAttributes({
                rewriter.getNamedAttr("seq_len", op.getSeqLenAttr()),
                rewriter.getNamedAttr("hidden", op.getHiddenAttr()),
                rewriter.getNamedAttr("query_heads", op.getQueryHeadsAttr()),
                rewriter.getNamedAttr("kv_heads", op.getKvHeadsAttr()),
                rewriter.getNamedAttr("head_dim", op.getHeadDimAttr()),
                rewriter.getNamedAttr("rope_theta", op.getRopeThetaAttr()),
                rewriter.getNamedAttr("causal", op.getCausalAttr()),
                rewriter.getNamedAttr("memory_plan", plan),
            });
            auto lowered = llvm::cast<tensor::AttentionOp>(rewriter.create(state));
            rewriter.replaceOp(op, lowered.getResult());
        }
        for (kernel::FfnOp op : ffns) {
            const bool w8a16 = is_w8a16_ffn(op, target);
            const auto& memory = target.memory();
            const auto& throughput = target.throughput();
            llvm::SmallVector<int64_t> weight_slices;
            for (int64_t index = 0; index < memory.w8a16_weight_slice_count; ++index)
                weight_slices.push_back(index * memory.w8a16_weight_slice_stride);
            llvm::SmallVector<int64_t> activation_slices;
            llvm::SmallVector<int64_t> hidden_slices;
            llvm::SmallVector<int64_t> result_slices;
            for (int64_t index = 0; index < throughput.mxm_activation_streams; ++index) {
                activation_slices.push_back(memory.w8a16_activation_slice_base + index);
                hidden_slices.push_back(memory.w8a16_hidden_slice_base
                    + index + (index == 3 ? 5 : 0));
                result_slices.push_back(memory.w8a16_result_slice_base + index);
            }
            const int64_t hidden_pass_bytes = w8a16
                ? op.getM() * (op.getHidden() / target.memory().hemispheres) * 2
                : op.getM() * 320;
            const int64_t gate_rows = w8a16
                ? op.getK() * op.getHidden()
                    / (memory.hemispheres * memory.w8a16_weight_slice_count
                        * throughput.tile_rows * throughput.lanes_per_tile)
                : 0;
            const int64_t down_rows = w8a16
                ? op.getHidden() * op.getN()
                    / (memory.w8a16_weight_slice_count
                        * throughput.tile_rows * throughput.lanes_per_tile)
                : 0;
            auto input = w8a16
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Activation,
                    activation_slices, 0, op.getM() * op.getK() / throughput.mxm_rows,
                    op.getM() * op.getK() * 2))
                : allocate_value(op.getInput(), PlacementKind::Activation);
            auto gate = w8a16
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
                    weight_slices, 0, gate_rows, op.getK() * op.getHidden()))
                : allocate_value(op.getGateWeight(), PlacementKind::Weight);
            auto up = w8a16
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
                    weight_slices, gate_rows, gate_rows, op.getK() * op.getHidden()))
                : allocate_value(op.getUpWeight(), PlacementKind::Weight);
            auto down = w8a16
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::Weight,
                    weight_slices, 2 * gate_rows, down_rows, op.getHidden() * op.getN()))
                : allocate_value(op.getDownWeight(), PlacementKind::Weight);
            auto hidden0 = w8a16
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::VxmResult,
                    hidden_slices, 0,
                    op.getM() * (op.getHidden() / memory.hemispheres)
                        / throughput.mxm_rows,
                    hidden_pass_bytes))
                : allocator.allocate(PlacementKind::VxmResult, hidden_pass_bytes);
            auto hidden1 = w8a16
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::VxmResult1,
                    hidden_slices, 0,
                    op.getM() * (op.getHidden() / memory.hemispheres)
                        / throughput.mxm_rows,
                    hidden_pass_bytes))
                : allocator.allocate(PlacementKind::VxmResult1, hidden_pass_bytes);
            const auto result_bytes = get_static_tensor_bytes(op.getResult().getType());
            const auto result = w8a16 && mlir::succeeded(result_bytes)
                ? mlir::FailureOr<Allocation>(fixed_allocation(PlacementKind::FinalResult,
                    result_slices, 0,
                    op.getM() * op.getN()
                        / (throughput.mxm_rows * throughput.mxms_per_hemisphere),
                    *result_bytes))
                : mlir::succeeded(result_bytes)
                ? allocator.allocate(PlacementKind::FinalResult, *result_bytes)
                : mlir::FailureOr<Allocation>(mlir::failure());
            if (mlir::failed(input) || mlir::failed(gate) || mlir::failed(up)
                || mlir::failed(down) || mlir::failed(hidden0) || mlir::failed(hidden1)
                || mlir::failed(result)) {
                op.emitError("cannot allocate complete FFN storage");
                signalPassFailure(); return;
            }
            rewriter.setInsertionPoint(op);
            mlir::OperationState state(op.getLoc(), tensor::FfnOp::getOperationName());
            state.addOperands({op.getInput(), op.getGateWeight(), op.getUpWeight(), op.getDownWeight()});
            state.addTypes(op.getResult().getType());
            state.addAttributes({
                rewriter.getNamedAttr("m", op.getMAttr()),
                rewriter.getNamedAttr("k", op.getKAttr()),
                rewriter.getNamedAttr("hidden", op.getHiddenAttr()),
                rewriter.getNamedAttr("n", op.getNAttr()),
                rewriter.getNamedAttr("gate_scale", op.getGateScaleAttr()),
                rewriter.getNamedAttr("up_scale", op.getUpScaleAttr()),
                rewriter.getNamedAttr("hidden_scale", op.getHiddenScaleAttr()),
                rewriter.getNamedAttr("hidden_zero_point", op.getHiddenZeroPointAttr()),
                rewriter.getNamedAttr("down_lhs_scale", op.getDownLhsScaleAttr()),
                rewriter.getNamedAttr("down_rhs_scale", op.getDownRhsScaleAttr()),
                rewriter.getNamedAttr("output_scale", op.getOutputScaleAttr()),
                rewriter.getNamedAttr("output_zero_point", op.getOutputZeroPointAttr()),
                rewriter.getNamedAttr("input_address", make_address_attr(rewriter, *input)),
                rewriter.getNamedAttr("input_placement", w8a16
                    ? make_profile_placement(rewriter, *input, "fp16_mxm_activation_planar", "both")
                    : make_placement_attr(rewriter, *input)),
                rewriter.getNamedAttr("gate_weight_address", make_address_attr(rewriter, *gate)),
                rewriter.getNamedAttr("gate_weight_placement", w8a16
                    ? make_profile_placement(rewriter, *gate, "w8a16_mxm_weight_striped", "both")
                    : make_placement_attr(rewriter, *gate)),
                rewriter.getNamedAttr("up_weight_address", make_address_attr(rewriter, *up)),
                rewriter.getNamedAttr("up_weight_placement", w8a16
                    ? make_profile_placement(rewriter, *up, "w8a16_mxm_weight_striped", "both")
                    : make_placement_attr(rewriter, *up)),
                rewriter.getNamedAttr("down_weight_address", make_address_attr(rewriter, *down)),
                rewriter.getNamedAttr("down_weight_placement", w8a16
                    ? make_profile_placement(rewriter, *down, "w8a16_mxm_weight_striped", "east")
                    : make_placement_attr(rewriter, *down)),
                rewriter.getNamedAttr("hidden0_address", make_address_attr(rewriter, *hidden0)),
                rewriter.getNamedAttr("hidden0_placement", w8a16
                    ? make_profile_placement(rewriter, *hidden0, "fp16_mxm_activation_planar", "west")
                    : make_placement_attr(rewriter, *hidden0)),
                rewriter.getNamedAttr("hidden1_address", make_address_attr(rewriter, *hidden1)),
                rewriter.getNamedAttr("hidden1_placement", w8a16
                    ? make_profile_placement(rewriter, *hidden1, "fp16_mxm_activation_planar", "east")
                    : make_placement_attr(rewriter, *hidden1)),
                rewriter.getNamedAttr("result_address", make_address_attr(rewriter, *result)),
                rewriter.getNamedAttr("result_placement", w8a16
                    ? make_profile_placement(rewriter, *result, "fp16_pair_planar", "east")
                    : make_placement_attr(rewriter, *result)),
                rewriter.getNamedAttr("input_bytes", rewriter.getI64IntegerAttr(input->bytes)),
                rewriter.getNamedAttr("gate_weight_bytes", rewriter.getI64IntegerAttr(gate->bytes)),
                rewriter.getNamedAttr("up_weight_bytes", rewriter.getI64IntegerAttr(up->bytes)),
                rewriter.getNamedAttr("down_weight_bytes", rewriter.getI64IntegerAttr(down->bytes)),
                rewriter.getNamedAttr("hidden_pass_bytes", rewriter.getI64IntegerAttr(hidden_pass_bytes)),
                rewriter.getNamedAttr("result_bytes", rewriter.getI64IntegerAttr(result->bytes)),
            });
            auto lowered = llvm::cast<tensor::FfnOp>(rewriter.create(state));
            rewriter.replaceOp(op, lowered.getResult());
        }

        for (kernel::SwigluOp op : swiglus) {
            const auto input = allocate_value(op.getInput(), PlacementKind::Activation);
            const auto gate = allocate_value(op.getGateWeight(), PlacementKind::Weight);
            const auto up = allocate_value(op.getUpWeight(), PlacementKind::Weight);
            const auto result_bytes = get_static_tensor_bytes(op.getResult().getType());
            const auto result = mlir::succeeded(result_bytes)
                ? allocator.allocate(PlacementKind::VxmResult, *result_bytes)
                : mlir::FailureOr<Allocation>(mlir::failure());
            if (mlir::failed(input) || mlir::failed(gate) || mlir::failed(up)
                || mlir::failed(result)) {
                op.emitError("cannot allocate dual-MXM SwiGLU storage");
                signalPassFailure();
                return;
            }
            rewriter.setInsertionPoint(op);
            mlir::OperationState state(op.getLoc(), tensor::SwigluOp::getOperationName());
            state.addOperands({op.getInput(), op.getGateWeight(), op.getUpWeight()});
            state.addTypes(op.getResult().getType());
            state.addAttributes({
                rewriter.getNamedAttr("m", op.getMAttr()),
                rewriter.getNamedAttr("n", op.getNAttr()),
                rewriter.getNamedAttr("k", op.getKAttr()),
                rewriter.getNamedAttr("gate_scale", op.getGateScaleAttr()),
                rewriter.getNamedAttr("up_scale", op.getUpScaleAttr()),
                rewriter.getNamedAttr("output_scale", op.getOutputScaleAttr()),
                rewriter.getNamedAttr("output_zero_point", op.getOutputZeroPointAttr()),
                rewriter.getNamedAttr("input_address", make_address_attr(rewriter, *input)),
                rewriter.getNamedAttr("input_placement", make_placement_attr(rewriter, *input)),
                rewriter.getNamedAttr("gate_weight_address", make_address_attr(rewriter, *gate)),
                rewriter.getNamedAttr("gate_weight_placement", make_placement_attr(rewriter, *gate)),
                rewriter.getNamedAttr("up_weight_address", make_address_attr(rewriter, *up)),
                rewriter.getNamedAttr("up_weight_placement", make_placement_attr(rewriter, *up)),
                rewriter.getNamedAttr("result_address", make_address_attr(rewriter, *result)),
                rewriter.getNamedAttr("result_placement", make_placement_attr(rewriter, *result)),
                rewriter.getNamedAttr("input_bytes", rewriter.getI64IntegerAttr(input->bytes)),
                rewriter.getNamedAttr("gate_weight_bytes", rewriter.getI64IntegerAttr(gate->bytes)),
                rewriter.getNamedAttr("up_weight_bytes", rewriter.getI64IntegerAttr(up->bytes)),
                rewriter.getNamedAttr("result_bytes", rewriter.getI64IntegerAttr(result->bytes)),
            });
            auto lowered = llvm::cast<tensor::SwigluOp>(rewriter.create(state));
            rewriter.replaceOp(op, lowered.getResult());
        }

        for (kernel::MatmulOp op : matmuls) {
            const int64_t current_ordinal = ordinals.lookup(op.getOperation());
            llvm::SmallVector<mlir::Value> expired;
            for (const auto& [value, allocation] : allocations) {
                const auto use = last_uses.find(value);
                if (use == last_uses.end() || use->second < current_ordinal) expired.push_back(value);
            }
            for (mlir::Value value : expired) {
                allocator.release(allocations.lookup(value));
                allocations.erase(value);
            }

            const auto lhs = allocate_value(op.getLhs(), PlacementKind::Activation);
            const auto rhs = allocate_value(op.getRhs(), PlacementKind::Weight);
            const auto result_bytes = get_static_tensor_bytes(op.getResult().getType());
            const auto result = mlir::succeeded(result_bytes) ? allocator.allocate(PlacementKind::Result, *result_bytes)
                                                              : mlir::FailureOr<Allocation>(mlir::failure());
            if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(result)) {
                op.emitError("cannot allocate static tensor storage in the east MEM hemisphere");
                signalPassFailure();
                return;
            }

            const mlir::Value old_result = op.getResult();
            rewriter.setInsertionPoint(op);
            auto lowered = rewriter.create<tensor::MatmulOp>(op.getLoc(), op.getLhs(), op.getRhs(),
                op.getResult().getType(), op.getM(), op.getN(), op.getK(), op.getUnitAttr(),
                make_address_attr(rewriter, *lhs), make_placement_attr(rewriter, *lhs),
                make_address_attr(rewriter, *rhs), make_placement_attr(rewriter, *rhs),
                make_address_attr(rewriter, *result), make_placement_attr(rewriter, *result),
                lhs->bytes, rhs->bytes, result->bytes);
            allocations.try_emplace(lowered.getResult(), *result);
            if (const auto use = last_uses.find(old_result); use != last_uses.end())
                last_uses[lowered.getResult()] = use->second;
            rewriter.replaceOp(op, lowered.getResult());

            llvm::SmallDenseSet<mlir::Value> candidates;
            candidates.insert(lowered.getLhs());
            candidates.insert(lowered.getRhs());
            candidates.insert(lowered.getResult());
            for (mlir::Value value : candidates) {
                const auto use = last_uses.find(value);
                if (use != last_uses.end() && use->second > current_ordinal) continue;
                const auto allocation = allocations.find(value);
                if (allocation == allocations.end()) continue;
                allocator.release(allocation->second);
                allocations.erase(allocation);
            }
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_kernel_to_tensor_pass()
{
    return std::make_unique<LowerKernelToTensorPass>();
}

} // namespace ftlpu::compiler
