#include "ftlpu/compiler/Dialect/Schedule/Analysis/ffn_schedule_planner.hpp"

namespace ftlpu::compiler::schedule {
namespace {

mlir::FailureOr<TaskAllocation> getTaskAllocation(
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

} // namespace

mlir::FailureOr<llvm::SmallVector<PrimitiveFfnSchedulePlan, 2>>
collectPrimitiveFfnSchedulePlans(mlir::func::FuncOp function)
{
    auto target = target::LPUTargetModel::from_operation(function);
    if (mlir::failed(target)) return mlir::failure();
    llvm::SmallVector<PrimitiveFfnSchedulePlan, 2> plans;
    llvm::SmallVector<stream::ElementwiseTaskOp> additions;
    function.walk([&](stream::ElementwiseTaskOp op) {
        if (op.getKind() == "add_quant") additions.push_back(op);
    });

    for (stream::ElementwiseTaskOp add : additions) {
        auto down0 = add.getLhs().getDefiningOp<stream::MatmulTaskOp>();
        auto down1 = add.getRhs().getDefiningOp<stream::MatmulTaskOp>();
        if (!down0 || !down1 || down0.getLhs().size() != 1
            || down1.getLhs().size() != 1 || down0.getRhs().size() != 1
            || down1.getRhs().size() != 1)
            continue;

        auto hidden0Route = down0.getLhs()[0].getDefiningOp<stream::RouteOp>();
        auto hidden1Route = down1.getLhs()[0].getDefiningOp<stream::RouteOp>();
        auto multiply = hidden0Route
            ? hidden0Route.getInput().getDefiningOp<stream::ElementwiseTaskOp>()
            : stream::ElementwiseTaskOp{};
        if (!hidden0Route || !hidden1Route || !multiply
            || multiply.getKind() != "multiply"
            || hidden1Route.getInput() != multiply.getResult())
            continue;

        auto swish = multiply.getLhs().getDefiningOp<stream::SwishTaskOp>();
        auto gate = swish
            ? swish.getInput().getDefiningOp<stream::MatmulTaskOp>()
            : stream::MatmulTaskOp{};
        auto up = multiply.getRhs().getDefiningOp<stream::MatmulTaskOp>();
        if (!swish || !gate || !up || gate.getLhs().size() != 1
            || up.getLhs().size() != 1 || gate.getRhs().size() != 1
            || up.getRhs().size() != 1 || gate.getLhs()[0] != up.getLhs()[0])
            continue;

        auto activationRoute =
            gate.getLhs()[0].getDefiningOp<stream::RouteOp>();
        auto gateRoute = gate.getRhs()[0].getDefiningOp<stream::RouteOp>();
        auto upRoute = up.getRhs()[0].getDefiningOp<stream::RouteOp>();
        auto down0Route = down0.getRhs()[0].getDefiningOp<stream::RouteOp>();
        auto down1Route = down1.getRhs()[0].getDefiningOp<stream::RouteOp>();
        const auto hidden0 =
            getTaskAllocation(multiply.getResultAllocations(), 0);
        const auto hidden1 =
            getTaskAllocation(multiply.getResultAllocations(), 1);
        const auto result = getTaskAllocation(add.getResultAllocations(), 0);
        const auto gateScale =
            gate.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
        const auto upScale =
            up.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
        const auto hiddenScale =
            multiply.getConfig().getAs<mlir::FloatAttr>("output_scale");
        const auto hiddenZeroPoint =
            multiply.getConfig().getAs<mlir::IntegerAttr>("output_zero_point");
        const auto downLhsScale =
            down0.getConfig().getAs<mlir::FloatAttr>("lhs_scale");
        const auto downRhsScale =
            down0.getConfig().getAs<mlir::FloatAttr>("rhs_scale");
        const auto outputScale =
            add.getConfig().getAs<mlir::FloatAttr>("output_scale");
        const auto outputZeroPoint =
            add.getConfig().getAs<mlir::IntegerAttr>("output_zero_point");
        if (!activationRoute || !gateRoute || !upRoute || !down0Route
            || !down1Route || mlir::failed(hidden0) || mlir::failed(hidden1)
            || mlir::failed(result) || !gateScale || !upScale || !hiddenScale
            || !hiddenZeroPoint || !downLhsScale || !downRhsScale
            || !outputScale || !outputZeroPoint) {
            add.emitError("incomplete primitive FFN stream graph");
            return mlir::failure();
        }

        plans.push_back(PrimitiveFfnSchedulePlan{
            add, down0, down1, hidden0Route, hidden1Route, multiply, swish,
            gate, up, activationRoute, gateRoute, upRoute, down0Route,
            down1Route, *hidden0, *hidden1, *result, gateScale, upScale,
            downRhsScale,
            buildFfnTaskPlan(
                {static_cast<int64_t>(down0.getM()),
                    static_cast<int64_t>(gate.getK()),
                    static_cast<int64_t>(gate.getN()),
                    static_cast<int64_t>(down0.getN())},
                *target),
        });
    }
    return plans;
}

} // namespace ftlpu::compiler::schedule
