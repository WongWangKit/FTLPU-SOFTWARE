#include "TensorToStreamLowering.hpp"

namespace ftlpu::compiler::stream_lowering {
namespace {

int64_t firstSlice(mlir::ArrayAttr allocations)
{
    const auto allocation = llvm::cast<mlir::DictionaryAttr>(allocations[0]);
    const auto placement =
        allocation.getAs<mlir::DictionaryAttr>("placement");
    return llvm::cast<mlir::IntegerAttr>(
        placement.getAs<mlir::ArrayAttr>("slices")[0]).getInt();
}

} // namespace

mlir::LogicalResult lower_gated_delta_rule(
    tensor::GatedDeltaRuleTaskOp op, LoweringContext& context)
{
    const int64_t begin = context.stage;
    const auto input = [&](mlir::ArrayAttr allocations, int64_t width) {
        return context.allocator.allocate(
            target::StreamEndpoint::Mem, target::StreamEndpoint::VxmInput,
            target::StreamDirection::West, firstSlice(allocations),
            begin, begin + 8, width);
    };
    auto query = input(op.getQueryAllocations(), 2);
    auto key = input(op.getKeyAllocations(), 2);
    auto value = input(op.getValueAllocations(), 2);
    auto decay = input(op.getDecayAllocations(), 4);
    auto beta = input(op.getBetaAllocations(), 4);
    auto state = input(op.getStateAllocations(), 4);
    auto output = context.allocator.allocate(
        target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
        target::StreamDirection::East,
        firstSlice(op.getOutputAllocations()), begin, begin + 8, 2);
    auto finalState = context.allocator.allocate(
        target::StreamEndpoint::VxmResult, target::StreamEndpoint::Mem,
        target::StreamDirection::East,
        firstSlice(op.getFinalStateAllocations()), begin, begin + 8, 4);
    if (mlir::failed(query) || mlir::failed(key) || mlir::failed(value)
        || mlir::failed(decay) || mlir::failed(beta)
        || mlir::failed(state) || mlir::failed(output)
        || mlir::failed(finalState))
        return op.emitError(
            "cannot allocate gated delta-rule stream routes");

    const auto integers = [&](std::initializer_list<int64_t> values) {
        llvm::SmallVector<mlir::Attribute> attributes;
        for (int64_t value : values)
            attributes.push_back(
                context.rewriter.getI64IntegerAttr(value));
        return context.rewriter.getArrayAttr(attributes);
    };
    context.rewriter.setInsertionPoint(op);
    mlir::OperationState task(
        op.getLoc(), stream::GatedDeltaRuleTaskOp::getOperationName());
    task.addOperands({op.getQuery(), op.getKey(), op.getValue(),
        op.getDecay(), op.getBeta(), op.getInitialState()});
    task.addTypes({op.getOutput().getType(),
        op.getFinalState().getType()});
    task.addAttributes({
        context.rewriter.getNamedAttr("heads", op.getHeadsAttr()),
        context.rewriter.getNamedAttr("key_dim", op.getKeyDimAttr()),
        context.rewriter.getNamedAttr("value_dim", op.getValueDimAttr()),
        context.rewriter.getNamedAttr("input_stream_bases", integers({
            query->stream_base, key->stream_base, value->stream_base,
            decay->stream_base, beta->stream_base, state->stream_base})),
        context.rewriter.getNamedAttr("output_stream_bases", integers({
            output->stream_base, finalState->stream_base})),
        context.rewriter.getNamedAttr(
            "query_allocations", op.getQueryAllocations()),
        context.rewriter.getNamedAttr(
            "key_allocations", op.getKeyAllocations()),
        context.rewriter.getNamedAttr(
            "value_allocations", op.getValueAllocations()),
        context.rewriter.getNamedAttr(
            "decay_allocations", op.getDecayAllocations()),
        context.rewriter.getNamedAttr(
            "beta_allocations", op.getBetaAllocations()),
        context.rewriter.getNamedAttr(
            "state_allocations", op.getStateAllocations()),
        context.rewriter.getNamedAttr(
            "scratch_allocations", op.getScratchAllocations()),
        context.rewriter.getNamedAttr(
            "output_allocations", op.getOutputAllocations()),
        context.rewriter.getNamedAttr(
            "final_state_allocations", op.getFinalStateAllocations()),
        context.rewriter.getNamedAttr("config", op.getConfig()),
    });
    auto lowered = llvm::cast<stream::GatedDeltaRuleTaskOp>(
        context.rewriter.create(task));
    context.rewriter.replaceOp(op,
        {lowered.getOutput(), lowered.getFinalState()});
    context.stage += 8;
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
