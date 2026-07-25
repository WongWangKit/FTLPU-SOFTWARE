#pragma once

#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_allocator.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/Analysis/attention_task_graph.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"

#include "mlir/IR/PatternMatch.h"

namespace ftlpu::compiler::stream_lowering {

inline mlir::FailureOr<int64_t> get_mem_slice(mlir::DictionaryAttr address)
{
    const auto slice = address.getAs<mlir::IntegerAttr>("slice");
    if (!slice) return mlir::failure();
    return slice.getInt();
}

struct TaskAllocation {
    mlir::DictionaryAttr address;
    mlir::DictionaryAttr placement;
    int64_t bytes;
};

inline mlir::FailureOr<TaskAllocation> get_task_allocation(
    mlir::ArrayAttr allocations, size_t index)
{
    if (index >= allocations.size()) return mlir::failure();
    const auto dictionary =
        llvm::dyn_cast<mlir::DictionaryAttr>(allocations[index]);
    if (!dictionary) return mlir::failure();
    const auto address =
        dictionary.getAs<mlir::DictionaryAttr>("address");
    const auto placement =
        dictionary.getAs<mlir::DictionaryAttr>("placement");
    const auto bytes = dictionary.getAs<mlir::IntegerAttr>("bytes");
    if (!address || !placement || !bytes) return mlir::failure();
    return TaskAllocation{address, placement, bytes.getInt()};
}

struct FfnTaskGraph {
    tensor::MatmulTaskOp gate;
    tensor::MatmulTaskOp up;
    tensor::SwishTaskOp swish;
    tensor::ElementwiseTaskOp multiply;
    tensor::MatmulTaskOp down;
    TaskAllocation input;
    TaskAllocation gate_weight;
    TaskAllocation up_weight;
    TaskAllocation down_weight;
    TaskAllocation hidden0;
    TaskAllocation hidden1;
    TaskAllocation result;
    mlir::FloatAttr gate_scale;
    mlir::FloatAttr up_scale;
    mlir::FloatAttr hidden_scale;
    mlir::IntegerAttr hidden_zero_point;
    mlir::FloatAttr down_lhs_scale;
    mlir::FloatAttr down_rhs_scale;
    mlir::FloatAttr output_scale;
    mlir::IntegerAttr output_zero_point;

    mlir::Value getInput() { return gate.getLhs(); }
    mlir::Value getGateWeight() { return gate.getRhs(); }
    mlir::Value getUpWeight() { return up.getRhs(); }
    mlir::Value getDownWeight() { return down.getRhs(); }
    mlir::Value getResult() { return down.getResult(); }
    mlir::Location getLoc() { return down.getLoc(); }
    uint64_t getM() { return down.getM(); }
    uint64_t getK() { return gate.getK(); }
    uint64_t getHidden() { return gate.getN(); }
    uint64_t getN() { return down.getN(); }
    mlir::DictionaryAttr getInputAddress() { return input.address; }
    mlir::DictionaryAttr getInputPlacement() { return input.placement; }
    int64_t getInputBytes() { return input.bytes; }
    mlir::DictionaryAttr getGateWeightAddress()
    {
        return gate_weight.address;
    }
    mlir::DictionaryAttr getGateWeightPlacement()
    {
        return gate_weight.placement;
    }
    int64_t getGateWeightBytes() { return gate_weight.bytes; }
    mlir::DictionaryAttr getUpWeightAddress()
    {
        return up_weight.address;
    }
    mlir::DictionaryAttr getUpWeightPlacement()
    {
        return up_weight.placement;
    }
    int64_t getUpWeightBytes() { return up_weight.bytes; }
    mlir::DictionaryAttr getDownWeightAddress()
    {
        return down_weight.address;
    }
    mlir::DictionaryAttr getDownWeightPlacement()
    {
        return down_weight.placement;
    }
    int64_t getDownWeightBytes() { return down_weight.bytes; }
    mlir::DictionaryAttr getHidden0Address() { return hidden0.address; }
    mlir::DictionaryAttr getHidden0Placement()
    {
        return hidden0.placement;
    }
    mlir::DictionaryAttr getHidden1Address() { return hidden1.address; }
    mlir::DictionaryAttr getHidden1Placement()
    {
        return hidden1.placement;
    }
    int64_t getHiddenPassBytes() { return hidden0.bytes; }
    mlir::DictionaryAttr getResultAddress() { return result.address; }
    mlir::DictionaryAttr getResultPlacement()
    {
        return result.placement;
    }
    int64_t getResultBytes() { return result.bytes; }
    llvm::APFloat getGateScale() { return gate_scale.getValue(); }
    llvm::APFloat getUpScale() { return up_scale.getValue(); }
    llvm::APFloat getDownRhsScale()
    {
        return down_rhs_scale.getValue();
    }
    mlir::FloatAttr getGateScaleAttr() { return gate_scale; }
    mlir::FloatAttr getUpScaleAttr() { return up_scale; }
    mlir::FloatAttr getHiddenScaleAttr() { return hidden_scale; }
    mlir::IntegerAttr getHiddenZeroPointAttr()
    {
        return hidden_zero_point;
    }
    mlir::FloatAttr getDownLhsScaleAttr() { return down_lhs_scale; }
    mlir::FloatAttr getDownRhsScaleAttr() { return down_rhs_scale; }
    mlir::FloatAttr getOutputScaleAttr() { return output_scale; }
    mlir::IntegerAttr getOutputZeroPointAttr()
    {
        return output_zero_point;
    }
    mlir::InFlightDiagnostic emitError(llvm::StringRef message)
    {
        return down.emitError(message);
    }
};

struct LoweringContext {
    const target::LPUTargetModel& target;
    stream::StreamAllocator& allocator;
    mlir::IRRewriter& rewriter;
    int64_t& stage;
};

mlir::LogicalResult lower_attention(
    tensor::AttentionTaskGraph graph, LoweringContext& context);
mlir::LogicalResult lower_ffn(
    FfnTaskGraph& graph, LoweringContext& context);
mlir::LogicalResult lower_swiglu(
    tensor::SwigluOp op, LoweringContext& context);
mlir::LogicalResult lower_matmul(
    tensor::MatmulOp op, LoweringContext& context);

} // namespace ftlpu::compiler::stream_lowering
