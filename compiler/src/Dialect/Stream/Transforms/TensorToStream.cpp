#include "TensorToStreamLowering.hpp"

#include "ftlpu/compiler/Transforms/passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace ftlpu::compiler {
namespace {
using namespace stream_lowering;

mlir::FailureOr<llvm::SmallVector<FfnTaskGraph, 2>>
collect_ffn_task_graphs(mlir::func::FuncOp function)
{
    llvm::SmallVector<FfnTaskGraph, 2> ffns;
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
            return mlir::failure();
        }

        ffns.push_back(FfnTaskGraph{
            gate, up, swish, multiply, down, *input, *gate_weight,
            *up_weight, *down_weight, *hidden0, *hidden1, *result,
            gate_scale, up_scale, hidden_scale, hidden_zero_point,
            down_lhs_scale, down_rhs_scale, output_scale,
            output_zero_point,
        });
    }
    return ffns;
}

class LowerTensorToStreamPass final
    : public mlir::PassWrapper<LowerTensorToStreamPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorToStreamPass)

    llvm::StringRef getArgument() const final
    {
        return "ftlpu-tensor-to-stream";
    }
    llvm::StringRef getDescription() const final
    {
        return "Routes MEM tensors through allocated LPU streams and MXM endpoints";
    }

    void runOnOperation() final
    {
        mlir::func::FuncOp function = getOperation();
        if (!function.getBody().hasOneBlock()) {
            function.emitError(
                "stream allocation currently requires a single-block function");
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

        auto ffns = collect_ffn_task_graphs(function);
        auto attentions = tensor::collectAttentionTaskGraphs(function);
        if (mlir::failed(ffns) || mlir::failed(attentions)) {
            signalPassFailure();
            return;
        }

        llvm::SmallVector<tensor::MatmulOp> matmuls;
        llvm::SmallVector<tensor::SwigluOp> swiglus;
        llvm::SmallVector<tensor::RmsNormTaskOp> rmsNorms;
        function.walk(
            [&](tensor::MatmulOp op) { matmuls.push_back(op); });
        function.walk(
            [&](tensor::SwigluOp op) { swiglus.push_back(op); });
        function.walk(
            [&](tensor::RmsNormTaskOp op) { rmsNorms.push_back(op); });

        int64_t stage = 0;
        LoweringContext context{target, allocator, rewriter, stage};
        for (tensor::AttentionTaskGraph graph : *attentions) {
            if (mlir::failed(lower_attention(graph, context))) {
                signalPassFailure();
                return;
            }
        }
        for (FfnTaskGraph& graph : *ffns) {
            if (mlir::failed(lower_ffn(graph, context))) {
                signalPassFailure();
                return;
            }
        }
        for (tensor::RmsNormTaskOp op : rmsNorms) {
            if (mlir::failed(lower_rms_norm(op, context))) {
                signalPassFailure();
                return;
            }
        }
        llvm::SmallVector<tensor::ElementwiseTaskOp> elementwiseOps;
        function.walk([&](tensor::ElementwiseTaskOp op) {
            elementwiseOps.push_back(op);
        });
        for (tensor::ElementwiseTaskOp op : elementwiseOps) {
            if (mlir::failed(lower_elementwise(op, context))) {
                signalPassFailure();
                return;
            }
        }
        for (tensor::SwigluOp op : swiglus) {
            if (mlir::failed(lower_swiglu(op, context))) {
                signalPassFailure();
                return;
            }
        }
        for (tensor::MatmulOp op : matmuls) {
            if (mlir::failed(lower_matmul(op, context))) {
                signalPassFailure();
                return;
            }
        }

        mlir::Operation* unlowered_task = nullptr;
        function.walk([&](mlir::Operation* operation) {
            if (!unlowered_task
                && (llvm::isa<tensor::MatmulTaskOp>(operation)
                    || llvm::isa<tensor::SwishTaskOp>(operation)
                    || llvm::isa<tensor::ElementwiseTaskOp>(operation)
                    || llvm::isa<tensor::ProjectionTaskOp>(operation)
                    || llvm::isa<tensor::RopeTaskOp>(operation)
                    || llvm::isa<tensor::BatchMatmulTaskOp>(operation)
                    || llvm::isa<tensor::SoftmaxTaskOp>(operation)
                    || llvm::isa<tensor::RmsNormTaskOp>(operation)
                    || llvm::isa<tensor::TransposeTaskOp>(operation)))
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
