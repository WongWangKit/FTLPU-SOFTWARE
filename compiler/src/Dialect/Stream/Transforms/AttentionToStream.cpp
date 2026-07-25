#include "TensorToStreamLowering.hpp"

#include "ftlpu/compiler/Dialect/Stream/Analysis/stream_route_plan.hpp"

#include "llvm/ADT/STLExtras.h"

namespace ftlpu::compiler::stream_lowering {

mlir::LogicalResult lower_attention(
    tensor::AttentionTaskGraph op, LoweringContext& context)
{
    const target::LPUTargetModel& target = context.target;
    stream::StreamAllocator& allocator = context.allocator;
    mlir::IRRewriter& rewriter = context.rewriter;
    int64_t& stage = context.stage;

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
    const stream::StreamRoutePlan route_plan =
        stream::plan_attention_routes();
    bool allocated = route_plan.valid();
    for (const stream::RouteLifetime& route : route_plan.routes()) {
        allocated = allocated
            && add_route(route.phase, route.role, route.source,
                route.destination, route.direction, route.buffer,
                stage + route.producer_offset,
                stage + route.consumer_offset);
    }
    if (!allocated) {
        op.emitError("cannot allocate generic attention stream topology");
        return mlir::failure();
    }
    rewriter.setInsertionPoint(op.output);
    const auto config = op.config();
    const auto routesForPhase = [&](llvm::StringRef phase) {
        llvm::SmallVector<mlir::Attribute> selected;
        for (mlir::Attribute attribute : routes) {
            auto route = llvm::cast<mlir::DictionaryAttr>(attribute);
            if (route.getAs<mlir::StringAttr>("phase").getValue()
                == phase)
                selected.push_back(route);
        }
        return rewriter.getArrayAttr(selected);
    };
    const auto routesForPhaseAndRoles =
        [&](llvm::StringRef phase,
            std::initializer_list<llvm::StringRef> roles) {
            llvm::SmallVector<mlir::Attribute> selected;
            for (mlir::Attribute attribute : routes) {
                auto route =
                    llvm::cast<mlir::DictionaryAttr>(attribute);
                if (route.getAs<mlir::StringAttr>("phase").getValue()
                    != phase)
                    continue;
                const llvm::StringRef role =
                    route.getAs<mlir::StringAttr>("role").getValue();
                for (llvm::StringRef candidate : roles)
                    if (candidate == role) {
                        selected.push_back(route);
                        break;
                    }
            }
            return rewriter.getArrayAttr(selected);
        };
    const auto elementType =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType())
            .getElementType();
    const auto matrixType = [&](int64_t rows, int64_t columns) {
        return mlir::RankedTensorType::get(
            {rows, columns}, elementType);
    };
    const auto scoreType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(op.getQueryHeads()),
            static_cast<int64_t>(op.getSeqLen()),
            static_cast<int64_t>(op.getSeqLen())},
        elementType);
    const auto createProjection =
        [&](mlir::Value input, mlir::Value weight,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::ArrayAttr taskRoutes, bool ownsMemoryPlan = false) {
            mlir::OperationState state(
                op.getLoc(),
                stream::ProjectionTaskOp::getOperationName());
            state.addOperands({input, weight});
            state.addTypes(resultType);
            state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr(kind)),
                rewriter.getNamedAttr("config", config),
                rewriter.getNamedAttr("routes", taskRoutes),
            });
            if (ownsMemoryPlan)
                state.addAttribute("memory_plan", op.getMemoryPlan());
            return llvm::cast<stream::ProjectionTaskOp>(
                rewriter.create(state));
        };
    const auto createUnary =
        [&](llvm::StringRef operationName, mlir::Value input,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::ArrayAttr taskRoutes) {
            mlir::OperationState state(op.getLoc(), operationName);
            state.addOperands(input);
            state.addTypes(resultType);
            state.addAttribute("config", config);
            state.addAttribute("routes", taskRoutes);
            if (!kind.empty())
                state.addAttribute(
                    "kind", rewriter.getStringAttr(kind));
            return rewriter.create(state)->getResult(0);
        };
    const auto createBatchMatmul =
        [&](mlir::Value lhs, mlir::Value rhs,
            llvm::StringRef kind, mlir::Type resultType,
            mlir::ArrayAttr taskRoutes) {
            mlir::OperationState state(
                op.getLoc(),
                stream::BatchMatmulTaskOp::getOperationName());
            state.addOperands({lhs, rhs});
            state.addTypes(resultType);
            state.addAttributes({
                rewriter.getNamedAttr(
                    "kind", rewriter.getStringAttr(kind)),
                rewriter.getNamedAttr("config", config),
                rewriter.getNamedAttr("routes", taskRoutes),
            });
            return llvm::cast<stream::BatchMatmulTaskOp>(
                rewriter.create(state));
        };

    auto query = createProjection(op.getInput(), op.getQueryWeight(),
        "query", matrixType(op.getSeqLen(),
                     op.getQueryHeads() * op.getHeadDim()),
        routesForPhaseAndRoles("qkv", {"query_weight",
            "query_weight_dequant", "activation", "qkv_result"}));
    auto key = createProjection(op.getInput(), op.getKeyWeight(),
        "key", matrixType(op.getSeqLen(),
                   op.getKvHeads() * op.getHeadDim()),
        routesForPhaseAndRoles("qkv", {"key_weight",
            "key_weight_dequant", "key_activation", "key_result"}));
    auto value = createProjection(op.getInput(), op.getValueWeight(),
        "value", matrixType(op.getSeqLen(),
                     op.getKvHeads() * op.getHeadDim()),
        routesForPhaseAndRoles("qkv", {"value_weight",
            "value_weight_dequant", "value_activation",
            "value_result"}));
    const mlir::Value rotatedQuery = createUnary(
        stream::RopeTaskOp::getOperationName(), query.getResult(),
        "query", query.getResult().getType(),
        routesForPhase("rope"));
    const mlir::Value rotatedKey = createUnary(
        stream::RopeTaskOp::getOperationName(), key.getResult(),
        "key", key.getResult().getType(),
        rewriter.getArrayAttr({}));
    auto qk = createBatchMatmul(
        rotatedQuery, rotatedKey, "qk", scoreType,
        routesForPhase("qk"));
    const mlir::Value probability = createUnary(
        stream::SoftmaxTaskOp::getOperationName(), qk.getResult(),
        "", scoreType, routesForPhase("softmax"));
    const mlir::Value transposedProbability = createUnary(
        stream::TransposeTaskOp::getOperationName(), probability,
        "probability", scoreType, rewriter.getArrayAttr({}));
    const mlir::Value transposedValue = createUnary(
        stream::TransposeTaskOp::getOperationName(), value.getResult(),
        "value", value.getResult().getType(),
        rewriter.getArrayAttr({}));
    auto pv = createBatchMatmul(
        transposedProbability, transposedValue, "pv",
        matrixType(op.getSeqLen(),
            op.getQueryHeads() * op.getHeadDim()),
        routesForPhase("pv"));
    auto output = createProjection(pv.getResult(),
        op.getOutputWeight(), "output", op.getResult().getType(),
        routesForPhase("o_proj"), true);
    rewriter.replaceOp(op.output, output.getResult());
    rewriter.eraseOp(op.pv);
    rewriter.eraseOp(op.value_transpose);
    rewriter.eraseOp(op.probability_transpose);
    rewriter.eraseOp(op.softmax);
    rewriter.eraseOp(op.qk);
    rewriter.eraseOp(op.key_rope);
    rewriter.eraseOp(op.query_rope);
    rewriter.eraseOp(op.value);
    rewriter.eraseOp(op.key);
    rewriter.eraseOp(op.query);
    stage += route_plan.duration();
    return mlir::success();
}

} // namespace ftlpu::compiler::stream_lowering
