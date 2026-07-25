#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {

mlir::LogicalResult lower_matmul(
    tensor::MatmulOp op, LoweringContext& context)
{
    const target::LPUTargetModel& target = context.target;
    stream::StreamAllocator& allocator = context.allocator;
    mlir::IRRewriter& rewriter = context.rewriter;
    int64_t& stage = context.stage;

    const auto lhs_slice = get_mem_slice(op.getLhsAddress());
    const auto rhs_slice = get_mem_slice(op.getRhsAddress());
    const auto result_slice = get_mem_slice(op.getResultAddress());
    if (mlir::failed(lhs_slice) || mlir::failed(rhs_slice) || mlir::failed(result_slice)) {
        op.emitError("requires valid MEM addresses before stream lowering");
        return mlir::failure();
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
        return mlir::failure();
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
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
