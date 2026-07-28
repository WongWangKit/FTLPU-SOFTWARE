#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {
namespace {

int64_t first_slice(mlir::DictionaryAttr placement)
{
    return llvm::cast<mlir::IntegerAttr>(
        placement.getAs<mlir::ArrayAttr>("slices")[0]).getInt();
}

} // namespace

mlir::LogicalResult lower_elementwise(
    tensor::ElementwiseTaskOp op, LoweringContext& context)
{
    auto lhs = get_task_allocation(op.getLhsAllocations(), 0);
    auto rhs = get_task_allocation(op.getRhsAllocations(), 0);
    auto result = get_task_allocation(op.getResultAllocations(), 0);
    if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(result)) {
        op.emitError("requires physical lhs, rhs, and result allocations");
        return mlir::failure();
    }

    const int64_t begin = context.stage;
    auto lhsBinding = context.allocator.allocate(
        target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
        target::StreamDirection::West, first_slice(lhs->placement),
        begin, begin + 2);
    auto rhsBinding = context.allocator.allocate(
        target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
        target::StreamDirection::West, first_slice(rhs->placement),
        begin, begin + 2);
    auto resultBinding = context.allocator.allocate(
        target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
        target::StreamDirection::East, first_slice(result->placement),
        begin + 2, begin + 4);
    if (mlir::failed(lhsBinding) || mlir::failed(rhsBinding)
        || mlir::failed(resultBinding)) {
        op.emitError("cannot allocate elementwise stream routes");
        return mlir::failure();
    }

    const auto integers = [&](std::initializer_list<int64_t> values) {
        llvm::SmallVector<mlir::Attribute> attributes;
        for (int64_t value : values)
            attributes.push_back(
                context.rewriter.getI64IntegerAttr(value));
        return context.rewriter.getArrayAttr(attributes);
    };

    context.rewriter.setInsertionPoint(op);
    mlir::OperationState state(
        op.getLoc(), stream::ElementwiseTaskOp::getOperationName());
    state.addOperands({op.getLhs(), op.getRhs()});
    state.addTypes(op.getResult().getType());
    state.addAttributes({
        context.rewriter.getNamedAttr("kind", op.getKindAttr()),
        context.rewriter.getNamedAttr("input_stream_bases",
            integers({lhsBinding->stream_base, rhsBinding->stream_base})),
        context.rewriter.getNamedAttr("output_stream_base",
            context.rewriter.getI64IntegerAttr(
                resultBinding->stream_base)),
        context.rewriter.getNamedAttr(
            "lhs_allocations", op.getLhsAllocations()),
        context.rewriter.getNamedAttr(
            "rhs_allocations", op.getRhsAllocations()),
        context.rewriter.getNamedAttr(
            "result_allocations", op.getResultAllocations()),
        context.rewriter.getNamedAttr("config", op.getConfig()),
    });
    auto lowered = llvm::cast<stream::ElementwiseTaskOp>(
        context.rewriter.create(state));
    context.rewriter.replaceOp(op, lowered.getResult());
    context.stage += 4;
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
