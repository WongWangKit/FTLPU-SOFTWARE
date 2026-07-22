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

        const target::LPUTargetModel target;
        stream::StreamAllocator allocator(target);
        llvm::SmallVector<tensor::MatmulOp> matmuls;
        llvm::SmallVector<tensor::SwigluOp> swiglus;
        llvm::SmallVector<tensor::FfnOp> ffns;
        llvm::SmallVector<tensor::AttentionOp> attentions;
        function.walk([&](tensor::MatmulOp op) { matmuls.push_back(op); });
        function.walk([&](tensor::SwigluOp op) { swiglus.push_back(op); });
        function.walk([&](tensor::FfnOp op) { ffns.push_back(op); });
        function.walk([&](tensor::AttentionOp op) { attentions.push_back(op); });

        mlir::IRRewriter rewriter(&getContext());
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
        for (tensor::FfnOp op : ffns) {
            const bool w8a16 = op.getInput().getType().getElementType().isF16()
                && op.getGateWeight().getType().getElementType().isInteger(8)
                && op.getUpWeight().getType().getElementType().isInteger(8)
                && op.getDownWeight().getType().getElementType().isInteger(8)
                && op.getResult().getType().getElementType().isF16()
                && target.supports_w8a16_ffn_shape(
                    op.getM(), op.getK(), op.getHidden(), op.getN());
            const auto input_slice = get_mem_slice(op.getInputAddress());
            const auto gate_slice = get_mem_slice(op.getGateWeightAddress());
            const auto up_slice = get_mem_slice(op.getUpWeightAddress());
            const auto down_slice = get_mem_slice(op.getDownWeightAddress());
            const auto result_slice = get_mem_slice(op.getResultAddress());
            if (mlir::failed(input_slice) || mlir::failed(gate_slice)
                || mlir::failed(up_slice) || mlir::failed(down_slice)
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
            if (mlir::failed(gate_binding) || mlir::failed(up_binding)
                || mlir::failed(activation_binding) || mlir::failed(down0_binding)
                || mlir::failed(down1_binding)) {
                op.emitError("cannot allocate complete-FFN stream ranges");
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
            if (!gate_latency || !up_latency || !input_latency || !down_latency) {
                op.emitError("complete-FFN route is unsupported by the target");
                signalPassFailure(); return;
            }
            rewriter.setInsertionPoint(op);
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
            mlir::OperationState state(op.getLoc(), stream::FfnOp::getOperationName());
            state.addOperands({activation_route.getOutput(), gate_route.getOutput(),
                up_route.getOutput(), down0_route.getOutput(), down1_route.getOutput()});
            state.addTypes(op.getResult().getType());
            state.addAttributes({
                rewriter.getNamedAttr("m", op.getMAttr()), rewriter.getNamedAttr("k", op.getKAttr()),
                rewriter.getNamedAttr("hidden", op.getHiddenAttr()), rewriter.getNamedAttr("n", op.getNAttr()),
                rewriter.getNamedAttr("gate_output_stream_base", rewriter.getI64IntegerAttr(0)),
                rewriter.getNamedAttr("up_output_stream_base", rewriter.getI64IntegerAttr(4)),
                rewriter.getNamedAttr("vxm_output_stream", rewriter.getI64IntegerAttr(
                    w8a16 ? 0 : 31)),
                rewriter.getNamedAttr("gate_scale", op.getGateScaleAttr()),
                rewriter.getNamedAttr("up_scale", op.getUpScaleAttr()),
                rewriter.getNamedAttr("hidden_scale", op.getHiddenScaleAttr()),
                rewriter.getNamedAttr("hidden_zero_point", op.getHiddenZeroPointAttr()),
                rewriter.getNamedAttr("down_lhs_scale", op.getDownLhsScaleAttr()),
                rewriter.getNamedAttr("down_rhs_scale", op.getDownRhsScaleAttr()),
                rewriter.getNamedAttr("output_scale", op.getOutputScaleAttr()),
                rewriter.getNamedAttr("output_zero_point", op.getOutputZeroPointAttr()),
                rewriter.getNamedAttr("hidden0_address", op.getHidden0AddressAttr()),
                rewriter.getNamedAttr("hidden0_placement", op.getHidden0PlacementAttr()),
                rewriter.getNamedAttr("hidden1_address", op.getHidden1AddressAttr()),
                rewriter.getNamedAttr("hidden1_placement", op.getHidden1PlacementAttr()),
                rewriter.getNamedAttr("result_address", op.getResultAddressAttr()),
                rewriter.getNamedAttr("result_placement", op.getResultPlacementAttr()),
                rewriter.getNamedAttr("hidden_pass_bytes", op.getHiddenPassBytesAttr()),
                rewriter.getNamedAttr("result_bytes", op.getResultBytesAttr()),
            });
            auto lowered = llvm::cast<stream::FfnOp>(rewriter.create(state));
            rewriter.replaceOp(op, lowered.getResult());
            stage += 10;
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
    }
};

} // namespace

std::unique_ptr<mlir::Pass> create_lower_tensor_to_stream_pass()
{
    return std::make_unique<LowerTensorToStreamPass>();
}

} // namespace ftlpu::compiler
