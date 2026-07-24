#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_allocator.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "ftlpu/compiler/Dialect/Tensor/IR/tensor_dialect.hpp"
#include "ftlpu/compiler/Target/lpu_target_model.hpp"
#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {

mlir::FailureOr<int64_t> get_mem_slice(mlir::DictionaryAttr address)
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

mlir::FailureOr<TaskAllocation> get_task_allocation(
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

struct TensorFfnPlan {
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

class LowerTensorToStreamPass final
    : public mlir::PassWrapper<LowerTensorToStreamPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorToStreamPass)

    llvm::StringRef getArgument() const final { return "ftlpu-tensor-to-stream"; }
    llvm::StringRef getDescription() const final
    {
        return "Routes MEM tensors through allocated LPU streams and MXM endpoints";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError("stream allocation currently requires a single-block function");
            signalPassFailure();
            return;
        }

        auto target_model =
            target::LPUTargetModel::from_operation(function);
        if (mlir::failed(target_model)) {
            signalPassFailure();
            return;
        }
        const target::LPUTargetModel& target = *target_model;
        stream::StreamAllocator allocator(target);
        mlir::IRRewriter rewriter(&getContext());

        llvm::SmallVector<TensorFfnPlan, 2> ffns;
        llvm::SmallVector<tensor::MatmulTaskOp> task_matmuls;
        function.walk(
            [&](tensor::MatmulTaskOp op) { task_matmuls.push_back(op); });
        for (tensor::MatmulTaskOp down : task_matmuls) {
            auto multiply =
                down.getLhs().getDefiningOp<tensor::ElementwiseTaskOp>();
            if (!multiply || multiply.getKind() != "multiply") continue;
            auto swish =
                multiply.getLhs().getDefiningOp<tensor::SwishTaskOp>();
            auto gate = swish
                ? swish.getInput().getDefiningOp<tensor::MatmulTaskOp>()
                : tensor::MatmulTaskOp{};
            auto up =
                multiply.getRhs().getDefiningOp<tensor::MatmulTaskOp>();
            if (!swish || !gate || !up || gate.getLhs() != up.getLhs())
                continue;

            const auto input =
                get_task_allocation(gate.getLhsAllocations(), 0);
            const auto gate_weight =
                get_task_allocation(gate.getRhsAllocations(), 0);
            const auto up_weight =
                get_task_allocation(up.getRhsAllocations(), 0);
            const auto down_weight =
                get_task_allocation(down.getRhsAllocations(), 0);
            const auto hidden0 =
                get_task_allocation(multiply.getResultAllocations(), 0);
            const auto hidden1 =
                get_task_allocation(multiply.getResultAllocations(), 1);
            const auto result =
                get_task_allocation(down.getResultAllocations(), 0);
            const auto gate_scale =
                gate.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
            const auto up_scale =
                up.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
            const auto hidden_scale =
                multiply.getConfig().getAs<mlir::FloatAttr>("output_scale");
            const auto hidden_zero_point =
                multiply.getConfig().getAs<mlir::IntegerAttr>(
                    "output_zero_point");
            const auto down_lhs_scale =
                down.getConfig().getAs<mlir::FloatAttr>("lhs_scale");
            const auto down_rhs_scale =
                down.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
            const auto output_scale =
                down.getConfig().getAs<mlir::FloatAttr>("output_scale");
            const auto output_zero_point =
                down.getConfig().getAs<mlir::IntegerAttr>(
                    "output_zero_point");
            if (mlir::failed(input) || mlir::failed(gate_weight)
                || mlir::failed(up_weight) || mlir::failed(down_weight)
                || mlir::failed(hidden0) || mlir::failed(hidden1)
                || mlir::failed(result) || !gate_scale || !up_scale
                || !hidden_scale || !hidden_zero_point || !down_lhs_scale
                || !down_rhs_scale || !output_scale || !output_zero_point) {
                down.emitError("incomplete physical FFN task plan");
                signalPassFailure();
                return;
            }

            ffns.push_back(TensorFfnPlan{
                gate, up, swish, multiply, down, *input, *gate_weight,
                *up_weight, *down_weight, *hidden0, *hidden1, *result,
                gate_scale, up_scale, hidden_scale, hidden_zero_point,
                down_lhs_scale, down_rhs_scale, output_scale,
                output_zero_point,
            });
        }

        llvm::SmallVector<tensor::MatmulOp> matmuls;
        llvm::SmallVector<tensor::SwigluOp> swiglus;
        llvm::SmallVector<tensor::AttentionOp> attentions;
        function.walk([&](tensor::MatmulOp op) { matmuls.push_back(op); });
        function.walk([&](tensor::SwigluOp op) { swiglus.push_back(op); });
        function.walk([&](tensor::AttentionOp op) { attentions.push_back(op); });

        int64_t stage = 0;
        for (tensor::AttentionOp op : attentions) {
            llvm::SmallVector<mlir::Attribute> routes;
            auto placement = [&](llvm::StringRef name) {
                return op.getMemoryPlan().getAs<mlir::DictionaryAttr>(name);
            };
            auto add_route = [&](llvm::StringRef phase, llvm::StringRef role,
                                 target::StreamEndpoint source, target::StreamEndpoint destination,
                                 target::StreamDirection direction, llvm::StringRef buffer,
                                 int64_t begin, int64_t end) -> bool {
                const auto buffer_placement = placement(buffer);
                const auto slices = buffer_placement.getAs<mlir::ArrayAttr>("slices");
                if (!buffer_placement || !slices || slices.empty()) return false;
                const auto slice = llvm::cast<mlir::IntegerAttr>(slices[0]).getInt();
                const auto binding = allocator.allocate(source, destination, direction, slice, begin, end);
                const auto latency = target.transport_latency(source, destination, direction, slice);
                if (mlir::failed(static_cast<mlir::LogicalResult>(binding)) || !latency)
                    return false;
                routes.push_back(rewriter.getDictionaryAttr({
                    rewriter.getNamedAttr("phase", rewriter.getStringAttr(phase)),
                    rewriter.getNamedAttr("role", rewriter.getStringAttr(role)),
                    rewriter.getNamedAttr("buffer", rewriter.getStringAttr(buffer)),
                    rewriter.getNamedAttr("source", rewriter.getStringAttr(llvm::StringRef(target::LPUTargetModel::endpoint_name(source)))),
                    rewriter.getNamedAttr("destination", rewriter.getStringAttr(llvm::StringRef(target::LPUTargetModel::endpoint_name(destination)))),
                    rewriter.getNamedAttr("direction", rewriter.getStringAttr(llvm::StringRef(target::LPUTargetModel::direction_name(direction)))),
                    rewriter.getNamedAttr("stream_base", rewriter.getI64IntegerAttr(binding->stream_base)),
                    rewriter.getNamedAttr("stream_count", rewriter.getI64IntegerAttr(binding->stream_count)),
                    rewriter.getNamedAttr("register_id", rewriter.getI64IntegerAttr(binding->register_id)),
                    rewriter.getNamedAttr("producer_stage", rewriter.getI64IntegerAttr(begin)),
                    rewriter.getNamedAttr("consumer_stage", rewriter.getI64IntegerAttr(end)),
                    rewriter.getNamedAttr("transport_latency", rewriter.getI64IntegerAttr(*latency)),
                }));
                return true;
            };
            const bool allocated =
                add_route("qkv", "query_weight", target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput, target::StreamDirection::East, "query_weight", stage, stage + 2)
                && add_route("qkv", "query_weight_dequant", target::StreamEndpoint::VxmResult, target::StreamEndpoint::MxmWeight, target::StreamDirection::East, "query_weight", stage + 2, stage + 4)
                && add_route("qkv", "activation", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation, target::StreamDirection::East, "input", stage + 4, stage + 8)
                && add_route("qkv", "qkv_result", target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem, target::StreamDirection::West, "query", stage + 8, stage + 10)
                && add_route("qkv", "key_weight", target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput, target::StreamDirection::East, "key_weight", stage + 10, stage + 12)
                && add_route("qkv", "key_weight_dequant", target::StreamEndpoint::VxmResult, target::StreamEndpoint::MxmWeight, target::StreamDirection::East, "key_weight", stage + 12, stage + 14)
                && add_route("qkv", "key_activation", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation, target::StreamDirection::East, "input", stage + 14, stage + 18)
                && add_route("qkv", "key_result", target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem, target::StreamDirection::West, "key", stage + 18, stage + 20)
                && add_route("qkv", "value_weight", target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput, target::StreamDirection::East, "value_weight", stage + 20, stage + 22)
                && add_route("qkv", "value_weight_dequant", target::StreamEndpoint::VxmResult, target::StreamEndpoint::MxmWeight, target::StreamDirection::East, "value_weight", stage + 22, stage + 24)
                && add_route("qkv", "value_activation", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation, target::StreamDirection::East, "input", stage + 24, stage + 28)
                && add_route("qkv", "value_result", target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem, target::StreamDirection::West, "value", stage + 28, stage + 30)
                && add_route("rope", "qk_to_vxm", target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput, target::StreamDirection::East, "query", stage + 30, stage + 32)
                && add_route("qk", "query_activation", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation, target::StreamDirection::East, "query", stage + 32, stage + 36)
                && add_route("qk", "key_weight", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight, target::StreamDirection::East, "key", stage + 32, stage + 36)
                && add_route("qk", "score_result", target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem, target::StreamDirection::West, "score", stage + 36, stage + 38)
                && add_route("softmax", "score_to_vxm", target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput, target::StreamDirection::East, "score", stage + 38, stage + 41)
                && add_route("pv", "probability_activation", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation, target::StreamDirection::East, "probability", stage + 41, stage + 45)
                && add_route("pv", "value_weight", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight, target::StreamDirection::East, "value", stage + 41, stage + 45)
                && add_route("pv", "context_result", target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem, target::StreamDirection::West, "context", stage + 45, stage + 47)
                && add_route("o_proj", "context_activation", target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation, target::StreamDirection::East, "context", stage + 47, stage + 51)
                && add_route("o_proj", "output_weight", target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput, target::StreamDirection::East, "output_weight", stage + 47, stage + 49)
                && add_route("o_proj", "output_weight_dequant", target::StreamEndpoint::VxmResult, target::StreamEndpoint::MxmWeight, target::StreamDirection::East, "output_weight", stage + 49, stage + 51)
                && add_route("o_proj", "result", target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem, target::StreamDirection::West, "result", stage + 51, stage + 53);
            if (!allocated) {
                op.emitError("cannot allocate generic attention stream topology");
                signalPassFailure(); return;
            }
            rewriter.setInsertionPoint(op);
            mlir::OperationState state(op.getLoc(), stream::AttentionOp::getOperationName());
            state.addOperands(op->getOperands());
            state.addTypes(op.getResult().getType());
            for (llvm::StringRef name : {"seq_len", "hidden", "query_heads", "kv_heads", "head_dim", "rope_theta", "causal", "memory_plan"})
                state.addAttribute(name, op->getAttr(name));
            state.addAttribute("routes", rewriter.getArrayAttr(routes));
            auto lowered = llvm::cast<stream::AttentionOp>(rewriter.create(state));
            rewriter.replaceOp(op, lowered.getResult());
            stage += 54;
        }
        for (TensorFfnPlan& op : ffns) {
            const auto element_type = [](mlir::Value value) {
                return llvm::cast<mlir::RankedTensorType>(value.getType())
                    .getElementType();
            };
            const bool w8a16 = element_type(op.getInput()).isF16()
                && element_type(op.getGateWeight()).isInteger(8)
                && element_type(op.getUpWeight()).isInteger(8)
                && element_type(op.getDownWeight()).isInteger(8)
                && element_type(op.getResult()).isF16()
                && target.supports_w8a16_ffn_shape(
                    op.getM(), op.getK(), op.getHidden(), op.getN());
            const auto input_slice = get_mem_slice(op.getInputAddress());
            const auto gate_slice = get_mem_slice(op.getGateWeightAddress());
            const auto up_slice = get_mem_slice(op.getUpWeightAddress());
            const auto down_slice = get_mem_slice(op.getDownWeightAddress());
            const auto hidden0_slice = get_mem_slice(op.getHidden0Address());
            const auto hidden1_slice = get_mem_slice(op.getHidden1Address());
            const auto result_slice = get_mem_slice(op.getResultAddress());
            if (mlir::failed(input_slice) || mlir::failed(gate_slice)
                || mlir::failed(up_slice) || mlir::failed(down_slice)
                || mlir::failed(hidden0_slice) || mlir::failed(hidden1_slice)
                || mlir::failed(result_slice)) {
                op.emitError("requires valid complete-FFN MEM addresses");
                signalPassFailure(); return;
            }
            auto allocate = [&](target::StreamEndpoint destination, int64_t slice,
                                int64_t begin, int64_t end) {
                return allocator.allocate(target::StreamEndpoint::Mem, destination,
                    target::StreamDirection::East, slice, begin, end);
            };
            const auto weight_endpoint = w8a16 ? target::StreamEndpoint::VxmInput
                                                  : target::StreamEndpoint::MxmWeight;
            const auto gate_binding = allocate(weight_endpoint,
                *gate_slice, stage, stage + 2);
            const auto up_binding = allocate(weight_endpoint,
                *up_slice, stage, stage + 2);
            const auto activation_binding = allocate(target::StreamEndpoint::MxmActivation,
                *input_slice, stage + (w8a16 ? 4 : 2), stage + (w8a16 ? 8 : 6));
            const auto down0_binding = allocate(weight_endpoint,
                *down_slice, stage + 7, stage + 9);
            const auto down1_binding = allocate(weight_endpoint,
                *down_slice, stage + 7, stage + 9);
            const auto hidden0_binding = allocate(
                target::StreamEndpoint::MxmActivation,
                *hidden0_slice, stage + 11, stage + 13);
            const auto hidden1_binding = allocate(
                target::StreamEndpoint::MxmActivation,
                *hidden1_slice, stage + 11, stage + 13);
            if (mlir::failed(gate_binding) || mlir::failed(up_binding)
                || mlir::failed(activation_binding) || mlir::failed(down0_binding)
                || mlir::failed(down1_binding)
                || mlir::failed(hidden0_binding)
                || mlir::failed(hidden1_binding)) {
                auto diagnostic = op.emitError("cannot allocate complete-FFN stream ranges:");
                if (mlir::failed(gate_binding)) diagnostic << " gate";
                if (mlir::failed(up_binding)) diagnostic << " up";
                if (mlir::failed(activation_binding)) diagnostic << " activation";
                if (mlir::failed(down0_binding)) diagnostic << " down0";
                if (mlir::failed(down1_binding)) diagnostic << " down1";
                if (mlir::failed(hidden0_binding)) diagnostic << " hidden0";
                if (mlir::failed(hidden1_binding)) diagnostic << " hidden1";
                signalPassFailure(); return;
            }
            auto latency = [&](target::StreamEndpoint endpoint, int64_t slice) {
                return target.transport_latency(target::StreamEndpoint::Mem, endpoint,
                    target::StreamDirection::East, slice);
            };
            const auto gate_latency = latency(weight_endpoint, *gate_slice);
            const auto up_latency = latency(weight_endpoint, *up_slice);
            const auto input_latency = latency(target::StreamEndpoint::MxmActivation, *input_slice);
            const auto down_latency = latency(weight_endpoint, *down_slice);
            const auto hidden0_latency = latency(
                target::StreamEndpoint::MxmActivation, *hidden0_slice);
            const auto hidden1_latency = latency(
                target::StreamEndpoint::MxmActivation, *hidden1_slice);
            if (!gate_latency || !up_latency || !input_latency || !down_latency
                || !hidden0_latency || !hidden1_latency) {
                op.emitError("complete-FFN route is unsupported by the target");
                signalPassFailure(); return;
            }
            rewriter.setInsertionPoint(op.down);
            auto route = [&](mlir::Value value, const stream::StreamBinding& binding,
                             target::StreamEndpoint destination, int64_t unit,
                             mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
                             int64_t bytes, int64_t route_latency) {
                return rewriter.create<stream::RouteOp>(op.getLoc(), value,
                    binding.stream_base, binding.stream_count, binding.register_id,
                    rewriter.getStringAttr("east"), rewriter.getStringAttr("MEM"),
                    rewriter.getStringAttr(target::LPUTargetModel::endpoint_name(destination)),
                    -1, unit, address, placement, bytes, route_latency);
            };
            auto gate_route = route(op.getGateWeight(), *gate_binding,
                weight_endpoint, 0, op.getGateWeightAddress(),
                op.getGateWeightPlacement(), op.getGateWeightBytes(), *gate_latency);
            auto up_route = route(op.getUpWeight(), *up_binding,
                weight_endpoint, w8a16 ? 0 : 1, op.getUpWeightAddress(),
                op.getUpWeightPlacement(), op.getUpWeightBytes(), *up_latency);
            auto activation_route = route(op.getInput(), *activation_binding,
                target::StreamEndpoint::MxmActivation, 0, op.getInputAddress(),
                op.getInputPlacement(), op.getInputBytes(), *input_latency);
            auto down0_route = route(op.getDownWeight(), *down0_binding,
                weight_endpoint, 0, op.getDownWeightAddress(),
                op.getDownWeightPlacement(), op.getDownWeightBytes(), *down_latency);
            auto down1_route = route(op.getDownWeight(), *down1_binding,
                weight_endpoint, w8a16 ? 0 : 1, op.getDownWeightAddress(),
                op.getDownWeightPlacement(), op.getDownWeightBytes(), *down_latency);
            if (w8a16) {
                auto connect_dequant = [&](stream::RouteOp raw, int64_t slice,
                                           int64_t unit, int64_t begin, float scale) {
                    auto input_type = llvm::cast<mlir::RankedTensorType>(raw.getOutput().getType());
                    auto fp16_type = mlir::RankedTensorType::get(input_type.getShape(),
                        rewriter.getF16Type());
                    mlir::OperationState dequant_state(op.getLoc(),
                        stream::DequantizeOp::getOperationName());
                    dequant_state.addOperands(raw.getOutput());
                    dequant_state.addTypes(fp16_type);
                    dequant_state.addAttributes({
                        rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)),
                        rewriter.getNamedAttr("input_stream_base", rewriter.getI64IntegerAttr(raw.getStreamBase())),
                        rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(0)),
                        rewriter.getNamedAttr("input_hemisphere", rewriter.getStringAttr("east")),
                        rewriter.getNamedAttr("output_hemisphere", rewriter.getStringAttr("east")),
                    });
                    auto dequant = llvm::cast<stream::DequantizeOp>(rewriter.create(dequant_state));
                    auto binding = allocator.allocate(target::StreamEndpoint::VxmResult,
                        target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
                        slice, begin, begin + 2);
                    if (mlir::failed(binding)) return stream::RouteOp{};
                    const auto latency = target.transport_latency(target::StreamEndpoint::VxmResult,
                        target::StreamEndpoint::MxmWeight, target::StreamDirection::East, slice);
                    return rewriter.create<stream::RouteOp>(op.getLoc(), dequant.getResult(),
                        binding->stream_base, binding->stream_count, binding->register_id,
                        rewriter.getStringAttr("east"), rewriter.getStringAttr("VXM.result"),
                        rewriter.getStringAttr("MXM.weight"), 0, unit, raw.getAddress(),
                        raw.getPlacement(), raw.getBytes(), *latency);
                };
                gate_route = connect_dequant(gate_route, *gate_slice, 0, stage + 2,
                    op.getGateScale().convertToFloat());
                up_route = connect_dequant(up_route, *up_slice, 1, stage + 2,
                    op.getUpScale().convertToFloat());
                down0_route = connect_dequant(down0_route, *down_slice, 0, stage + 9,
                    op.getDownRhsScale().convertToFloat());
                down1_route = connect_dequant(down1_route, *down_slice, 1, stage + 9,
                    op.getDownRhsScale().convertToFloat());
                if (!gate_route || !up_route || !down0_route || !down1_route) {
                    op.emitError("cannot allocate W8A16 VXM-to-MXM streams");
                    signalPassFailure(); return;
                }
            }
            const auto empty_allocations = rewriter.getArrayAttr({});
            const auto i64_array = [&](std::initializer_list<int64_t> values) {
                llvm::SmallVector<mlir::Attribute> attributes;
                for (int64_t value : values)
                    attributes.push_back(rewriter.getI64IntegerAttr(value));
                return rewriter.getArrayAttr(attributes);
            };
            const auto allocation = [&](mlir::DictionaryAttr address,
                                        mlir::DictionaryAttr placement,
                                        int64_t bytes) {
                return rewriter.getDictionaryAttr({
                    rewriter.getNamedAttr("address", address),
                    rewriter.getNamedAttr("placement", placement),
                    rewriter.getNamedAttr(
                        "bytes", rewriter.getI64IntegerAttr(bytes)),
                });
            };
            const auto hidden_allocations = rewriter.getArrayAttr({
                allocation(op.getHidden0Address(), op.getHidden0Placement(),
                    op.getHiddenPassBytes()),
                allocation(op.getHidden1Address(), op.getHidden1Placement(),
                    op.getHiddenPassBytes()),
            });
            const auto result_allocations = rewriter.getArrayAttr({
                allocation(op.getResultAddress(), op.getResultPlacement(),
                    op.getResultBytes()),
            });
            const auto projection_type = mlir::RankedTensorType::get(
                {static_cast<int64_t>(op.getM()),
                    static_cast<int64_t>(op.getHidden())},
                rewriter.getF32Type());
            const auto hidden_type = mlir::RankedTensorType::get(
                {static_cast<int64_t>(op.getM()),
                    static_cast<int64_t>(op.getHidden())},
                w8a16 ? mlir::Type(rewriter.getF16Type())
                      : mlir::Type(rewriter.getI8Type()));
            const auto down_partial_type = mlir::RankedTensorType::get(
                {static_cast<int64_t>(op.getM()),
                    static_cast<int64_t>(op.getN())},
                rewriter.getF32Type());
            const int64_t result_stream_count =
                target.throughput().mxm_result_streams;
            const int64_t second_result_stream = result_stream_count;

            auto create_matmul_task =
                [&](mlir::ValueRange lhs, mlir::ValueRange rhs,
                    mlir::Type result_type, int64_t m, int64_t n, int64_t k,
                    mlir::ArrayAttr unit_ids, mlir::ArrayAttr buffers,
                    mlir::ArrayAttr stream_bases,
                    mlir::ArrayAttr stream_counts,
                    mlir::ArrayAttr result_plan,
                    mlir::DictionaryAttr config) {
                    mlir::OperationState task_state(
                        op.getLoc(), stream::MatmulTaskOp::getOperationName());
                    task_state.addOperands(lhs);
                    task_state.addOperands(rhs);
                    task_state.addTypes(result_type);
                    task_state.addAttributes({
                        rewriter.getNamedAttr("m",
                            rewriter.getI64IntegerAttr(m)),
                        rewriter.getNamedAttr("n",
                            rewriter.getI64IntegerAttr(n)),
                        rewriter.getNamedAttr("k",
                            rewriter.getI64IntegerAttr(k)),
                        rewriter.getNamedAttr("unit_ids", unit_ids),
                        rewriter.getNamedAttr("weight_buffers", buffers),
                        rewriter.getNamedAttr(
                            "result_stream_bases", stream_bases),
                        rewriter.getNamedAttr(
                            "result_stream_counts", stream_counts),
                        rewriter.getNamedAttr(
                            "result_allocations", result_plan),
                        rewriter.getNamedAttr("config", config),
                        rewriter.getNamedAttr("operandSegmentSizes",
                            rewriter.getDenseI32ArrayAttr(
                                {static_cast<int32_t>(lhs.size()),
                                    static_cast<int32_t>(rhs.size())})),
                    });
                    return llvm::cast<stream::MatmulTaskOp>(
                        rewriter.create(task_state));
                };

            auto gate_task = create_matmul_task(
                mlir::ValueRange{activation_route.getOutput()},
                mlir::ValueRange{gate_route.getOutput()}, projection_type,
                op.getM(), op.getHidden(), op.getK(), i64_array({0}),
                i64_array({0}), i64_array({0}),
                i64_array({result_stream_count}),
                empty_allocations, rewriter.getDictionaryAttr({
                    rewriter.getNamedAttr(
                        "rhs_scale", op.getGateScaleAttr()),
                }));
            auto up_task = create_matmul_task(
                mlir::ValueRange{activation_route.getOutput()},
                mlir::ValueRange{up_route.getOutput()}, projection_type,
                op.getM(), op.getHidden(), op.getK(), i64_array({1}),
                i64_array({0}), i64_array({second_result_stream}),
                i64_array({result_stream_count}),
                empty_allocations, rewriter.getDictionaryAttr({
                    rewriter.getNamedAttr(
                        "rhs_scale", op.getUpScaleAttr()),
                }));

            mlir::OperationState swish_state(
                op.getLoc(), stream::SwishTaskOp::getOperationName());
            swish_state.addOperands(gate_task.getResult());
            swish_state.addTypes(projection_type);
            swish_state.addAttributes({
                rewriter.getNamedAttr(
                    "input_stream_base", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr(
                    "output_stream_base", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr(
                    "config", rewriter.getDictionaryAttr({})),
            });
            auto swish =
                llvm::cast<stream::SwishTaskOp>(rewriter.create(swish_state));

            mlir::OperationState multiply_state(
                op.getLoc(), stream::ElementwiseTaskOp::getOperationName());
            multiply_state.addOperands(
                {swish.getResult(), up_task.getResult()});
            multiply_state.addTypes(hidden_type);
            multiply_state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr("multiply")),
                rewriter.getNamedAttr(
                    "input_stream_bases",
                    i64_array({0, second_result_stream})),
                rewriter.getNamedAttr(
                    "output_stream_base", rewriter.getI64IntegerAttr(
                        w8a16 ? 0
                              : target.streams().streams_per_direction - 1)),
                rewriter.getNamedAttr(
                    "result_allocations", hidden_allocations),
                rewriter.getNamedAttr("config",
                    rewriter.getDictionaryAttr({
                        rewriter.getNamedAttr(
                            "output_scale", op.getHiddenScaleAttr()),
                        rewriter.getNamedAttr("output_zero_point",
                            op.getHiddenZeroPointAttr()),
                    })),
            });
            auto multiply = llvm::cast<stream::ElementwiseTaskOp>(
                rewriter.create(multiply_state));

            auto hidden0_route = route(multiply.getResult(), *hidden0_binding,
                target::StreamEndpoint::MxmActivation, 0,
                op.getHidden0Address(), op.getHidden0Placement(),
                op.getHiddenPassBytes(), *hidden0_latency);
            auto hidden1_route = route(multiply.getResult(), *hidden1_binding,
                target::StreamEndpoint::MxmActivation, 1,
                op.getHidden1Address(), op.getHidden1Placement(),
                op.getHiddenPassBytes(), *hidden1_latency);
            auto down0_task = create_matmul_task(
                mlir::ValueRange{hidden0_route.getOutput()},
                mlir::ValueRange{down0_route.getOutput()}, down_partial_type,
                op.getM(), op.getN(), op.getHidden(), i64_array({0}),
                i64_array({0}), i64_array({0}),
                i64_array({result_stream_count}),
                empty_allocations, rewriter.getDictionaryAttr({
                    rewriter.getNamedAttr(
                        "lhs_scale", op.getDownLhsScaleAttr()),
                    rewriter.getNamedAttr(
                        "rhs_scale", op.getDownRhsScaleAttr()),
                }));
            auto down1_task = create_matmul_task(
                mlir::ValueRange{hidden1_route.getOutput()},
                mlir::ValueRange{down1_route.getOutput()}, down_partial_type,
                op.getM(), op.getN(), op.getHidden(), i64_array({1}),
                i64_array({0}), i64_array({second_result_stream}),
                i64_array({result_stream_count}),
                empty_allocations, rewriter.getDictionaryAttr({
                    rewriter.getNamedAttr(
                        "lhs_scale", op.getDownLhsScaleAttr()),
                    rewriter.getNamedAttr(
                        "rhs_scale", op.getDownRhsScaleAttr()),
                }));

            mlir::OperationState add_state(
                op.getLoc(), stream::ElementwiseTaskOp::getOperationName());
            add_state.addOperands(
                {down0_task.getResult(), down1_task.getResult()});
            add_state.addTypes(op.getResult().getType());
            add_state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr("add_quant")),
                rewriter.getNamedAttr(
                    "input_stream_bases",
                    i64_array({0, second_result_stream})),
                rewriter.getNamedAttr(
                    "output_stream_base", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr(
                    "result_allocations", result_allocations),
                rewriter.getNamedAttr("config",
                    rewriter.getDictionaryAttr({
                        rewriter.getNamedAttr(
                            "output_scale", op.getOutputScaleAttr()),
                        rewriter.getNamedAttr("output_zero_point",
                            op.getOutputZeroPointAttr()),
                    })),
            });
            auto add = llvm::cast<stream::ElementwiseTaskOp>(
                rewriter.create(add_state));
            rewriter.replaceOp(op.down, add.getResult());
            rewriter.eraseOp(op.multiply);
            rewriter.eraseOp(op.swish);
            rewriter.eraseOp(op.up);
            rewriter.eraseOp(op.gate);
            stage += 14;
        }

        for (tensor::SwigluOp op : swiglus) {
            const auto input_slice = get_mem_slice(op.getInputAddress());
            const auto gate_slice = get_mem_slice(op.getGateWeightAddress());
            const auto up_slice = get_mem_slice(op.getUpWeightAddress());
            const auto result_slice = get_mem_slice(op.getResultAddress());
            if (mlir::failed(input_slice) || mlir::failed(gate_slice)
                || mlir::failed(up_slice) || mlir::failed(result_slice)) {
                op.emitError("requires valid MEM addresses before stream lowering");
                signalPassFailure(); return;
            }
            const auto gate_binding = allocator.allocate(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
                *gate_slice, stage, stage + 2);
            const auto up_binding = allocator.allocate(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
                *up_slice, stage, stage + 2);
            const auto activation_binding = allocator.allocate(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation, target::StreamDirection::East,
                *input_slice, stage + 2, stage + 4);
            const auto output_binding = allocator.allocate(target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem, target::StreamDirection::East,
                *result_slice, stage + 4, stage + 6);
            const auto gate_latency = target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, *gate_slice);
            const auto up_latency = target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, *up_slice);
            const auto activation_latency = target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation, target::StreamDirection::East, *input_slice);
            const auto output_latency = target.transport_latency(target::StreamEndpoint::VxmResult,
                target::StreamEndpoint::Mem, target::StreamDirection::East, *result_slice);
            if (mlir::failed(gate_binding) || mlir::failed(up_binding)
                || mlir::failed(activation_binding) || mlir::failed(output_binding)
                || !gate_latency || !up_latency || !activation_latency || !output_latency) {
                op.emitError("cannot allocate the dual-MXM/VXM stream topology");
                signalPassFailure(); return;
            }
            rewriter.setInsertionPoint(op);
            auto route = [&](mlir::Value value, const stream::StreamBinding& binding,
                             target::StreamEndpoint destination, int64_t unit,
                             mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
                             int64_t bytes, int64_t latency) {
                return rewriter.create<stream::RouteOp>(op.getLoc(), value,
                    binding.stream_base, binding.stream_count, binding.register_id,
                    rewriter.getStringAttr("east"), rewriter.getStringAttr("MEM"),
                    rewriter.getStringAttr(target::LPUTargetModel::endpoint_name(destination)),
                    -1, unit, address, placement, bytes, latency);
            };
            auto gate_route = route(op.getGateWeight(), *gate_binding,
                target::StreamEndpoint::MxmWeight, 0, op.getGateWeightAddress(),
                op.getGateWeightPlacement(), op.getGateWeightBytes(), *gate_latency);
            auto up_route = route(op.getUpWeight(), *up_binding,
                target::StreamEndpoint::MxmWeight, 1, op.getUpWeightAddress(),
                op.getUpWeightPlacement(), op.getUpWeightBytes(), *up_latency);
            auto activation_route = route(op.getInput(), *activation_binding,
                target::StreamEndpoint::MxmActivation, 0, op.getInputAddress(),
                op.getInputPlacement(), op.getInputBytes(), *activation_latency);
            mlir::OperationState state(op.getLoc(), stream::SwigluOp::getOperationName());
            state.addOperands({activation_route.getOutput(), gate_route.getOutput(), up_route.getOutput()});
            state.addTypes(op.getResult().getType());
            state.addAttributes({
                rewriter.getNamedAttr("m", op.getMAttr()), rewriter.getNamedAttr("n", op.getNAttr()),
                rewriter.getNamedAttr("k", op.getKAttr()),
                rewriter.getNamedAttr("gate_unit_id", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("up_unit_id", rewriter.getI64IntegerAttr(1)),
                rewriter.getNamedAttr("gate_weight_buffer", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("up_weight_buffer", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("gate_output_stream_base", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("up_output_stream_base", rewriter.getI64IntegerAttr(4)),
                rewriter.getNamedAttr("vxm_output_stream", rewriter.getI64IntegerAttr(31)),
                rewriter.getNamedAttr("output_register_id", rewriter.getI64IntegerAttr(output_binding->register_id)),
                rewriter.getNamedAttr("output_transport_latency", rewriter.getI64IntegerAttr(*output_latency)),
                rewriter.getNamedAttr("gate_scale", op.getGateScaleAttr()),
                rewriter.getNamedAttr("up_scale", op.getUpScaleAttr()),
                rewriter.getNamedAttr("output_scale", op.getOutputScaleAttr()),
                rewriter.getNamedAttr("output_zero_point", op.getOutputZeroPointAttr()),
                rewriter.getNamedAttr("result_address", op.getResultAddressAttr()),
                rewriter.getNamedAttr("result_placement", op.getResultPlacementAttr()),
                rewriter.getNamedAttr("result_bytes", op.getResultBytesAttr()),
            });
            auto lowered = llvm::cast<stream::SwigluOp>(rewriter.create(state));
            rewriter.replaceOp(op, lowered.getResult());
            stage += 6;
        }

        for (tensor::MatmulOp op : matmuls) {
            const auto lhs_slice = get_mem_slice(op.getLhsAddress());
            const auto rhs_slice = get_mem_slice(op.getRhsAddress());
            const auto result_slice = get_mem_slice(op.getResultAddress());
            if (mlir::failed(lhs_slice) || mlir::failed(rhs_slice) || mlir::failed(result_slice)) {
                op.emitError("requires valid MEM addresses before stream lowering");
                signalPassFailure();
                return;
            }

            const auto rhs_binding = allocator.allocate(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East,
                *rhs_slice, stage, stage + 4);
            const auto lhs_binding = allocator.allocate(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation, target::StreamDirection::East,
                *lhs_slice, stage + 2, stage + 4);
            const auto result_binding = allocator.allocate(target::StreamEndpoint::MxmResult,
                target::StreamEndpoint::Mem, target::StreamDirection::West,
                *result_slice, stage + 4, stage + 6);
            const auto lhs_latency = target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmActivation, target::StreamDirection::East, *lhs_slice);
            const auto rhs_latency = target.transport_latency(target::StreamEndpoint::Mem,
                target::StreamEndpoint::MxmWeight, target::StreamDirection::East, *rhs_slice);
            const auto result_latency = target.transport_latency(target::StreamEndpoint::MxmResult,
                target::StreamEndpoint::Mem, target::StreamDirection::West, *result_slice);
            if (mlir::failed(lhs_binding) || mlir::failed(rhs_binding) || mlir::failed(result_binding)
                || !lhs_latency || !rhs_latency || !result_latency) {
                op.emitError("cannot allocate a legal stream route for the LPU target");
                signalPassFailure();
                return;
            }

            rewriter.setInsertionPoint(op);
            auto make_route = [&](mlir::Value input, const stream::StreamBinding& binding,
                                  target::StreamEndpoint source, target::StreamEndpoint destination,
                                  int64_t source_unit_id, int64_t destination_unit_id,
                                  mlir::DictionaryAttr address, mlir::DictionaryAttr placement,
                                  int64_t bytes, int64_t latency) {
                return rewriter.create<stream::RouteOp>(op.getLoc(), input,
                    binding.stream_base, binding.stream_count, binding.register_id,
                    rewriter.getStringAttr(target::LPUTargetModel::direction_name(binding.direction)),
                    rewriter.getStringAttr(target::LPUTargetModel::endpoint_name(source)),
                    rewriter.getStringAttr(target::LPUTargetModel::endpoint_name(destination)),
                    source_unit_id, destination_unit_id,
                    address, placement, bytes, latency);
            };

            auto rhs_route = make_route(op.getRhs(), *rhs_binding,
                target::StreamEndpoint::Mem, target::StreamEndpoint::MxmWeight,
                -1, 0,
                op.getRhsAddress(), op.getRhsPlacement(), op.getRhsBytes(), *rhs_latency);
            auto lhs_route = make_route(op.getLhs(), *lhs_binding,
                target::StreamEndpoint::Mem, target::StreamEndpoint::MxmActivation,
                -1, 0,
                op.getLhsAddress(), op.getLhsPlacement(), op.getLhsBytes(), *lhs_latency);
            auto matmul = rewriter.create<stream::MatmulOp>(op.getLoc(),
                lhs_route.getOutput(), rhs_route.getOutput(), op.getResult().getType(),
                op.getM(), op.getN(), op.getK(), 0, 0);
            auto result_route = make_route(matmul.getResult(), *result_binding,
                target::StreamEndpoint::MxmResult, target::StreamEndpoint::Mem,
                0, -1,
                op.getResultAddress(), op.getResultPlacement(), op.getResultBytes(), *result_latency);
            rewriter.replaceOp(op, result_route.getOutput());
            stage += 6;
        }

        mlir::Operation* unlowered_task = nullptr;
        function.walk([&](mlir::Operation* operation) {
            if (!unlowered_task
                && (llvm::isa<tensor::MatmulTaskOp>(operation)
                    || llvm::isa<tensor::SwishTaskOp>(operation)
                    || llvm::isa<tensor::ElementwiseTaskOp>(operation)))
                unlowered_task = operation;
        });
        if (unlowered_task) {
            unlowered_task->emitError(
                "tensor task graph is not supported by stream lowering");
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_tensor_to_stream_pass()
{
    return std::make_unique<LowerTensorToStreamPass>();
}

} // namespace ftlpu::compiler
