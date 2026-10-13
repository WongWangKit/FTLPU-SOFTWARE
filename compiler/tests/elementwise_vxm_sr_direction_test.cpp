#include "ftlpu/compiler/Dialect/Schedule/Transforms/stream_schedule_emitters.hpp"
#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"
#include "ftlpu/compiler/Dialect/Stream/IR/stream_dialect.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include <stdexcept>

void test_direction(bool scalarMultiply)
{
    using namespace ftlpu::compiler;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<schedule::ScheduleDialect>();
    context.getOrLoadDialect<stream::StreamDialect>();
    mlir::IRRewriter rewriter(&context);
    const auto location = rewriter.getUnknownLoc();
    auto module = mlir::ModuleOp::create(location);
    rewriter.setInsertionPointToStart(module.getBody());
    const auto tensor = mlir::RankedTensorType::get({32, 128}, rewriter.getBF16Type());
    auto function = rewriter.create<mlir::func::FuncOp>(location, "residual",
        rewriter.getFunctionType({tensor, tensor}, {tensor}));
    auto* entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);
    const auto allocation = [&](llvm::StringRef kind, int64_t sliceBase,
                                int64_t sliceCount, int64_t bank) {
        llvm::SmallVector<mlir::Attribute> slices;
        for (int64_t slice = 0; slice < sliceCount; ++slice)
            slices.push_back(rewriter.getI64IntegerAttr(sliceBase + slice));
        auto placement = rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("kind", rewriter.getStringAttr(kind)),
            rewriter.getNamedAttr("slices", rewriter.getArrayAttr(slices)),
            rewriter.getNamedAttr("hemisphere", rewriter.getStringAttr("both")),
            rewriter.getNamedAttr("base_row", rewriter.getI64IntegerAttr(0)),
            rewriter.getNamedAttr("bank", rewriter.getI64IntegerAttr(bank))});
        return rewriter.getArrayAttr({rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("placement", placement)})});
    };
    mlir::OperationState state(location, stream::ElementwiseTaskOp::getOperationName());
    state.addOperands({entry->getArgument(0), entry->getArgument(1)});
    state.addTypes(tensor);
    state.addAttributes({
        rewriter.getNamedAttr("kind", rewriter.getStringAttr(scalarMultiply ? "multiply" : "add")),
        rewriter.getNamedAttr("input_stream_bases", rewriter.getArrayAttr({})),
        rewriter.getNamedAttr("output_stream_base", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("lhs_allocations", allocation("fp16_pair_planar", 0, 4, 0)),
        rewriter.getNamedAttr("rhs_allocations", allocation("fp16_mxm_distributed_16", 0, 16, 1)),
        rewriter.getNamedAttr("result_allocations", allocation("fp16_mxm_distributed_16", 16, 16, 0)),
        rewriter.getNamedAttr("config", scalarMultiply
            ? rewriter.getDictionaryAttr({rewriter.getNamedAttr("rhs_immediate", rewriter.getF32FloatAttr(0.25f))})
            : rewriter.getDictionaryAttr({}))});
    auto* task = rewriter.create(state);
    rewriter.create<mlir::func::ReturnOp>(location, task->getResults());
    target::LPUTargetModel target;
    if (mlir::failed(schedule::lowerElementwiseSchedules(rewriter, function, target)))
        throw std::logic_error("residual SR lowering failed");
    int heads = 0;
    function.walk([&](schedule::VxmOp op) {
        if (op.getOpcode() != (scalarMultiply ? "multiply" : "add")) return;
        ++heads;
        const int64_t expected = op.getInputHemisphere() == "west" ? 48 : 32;
        if (op->getAttrOfType<mlir::IntegerAttr>("lhs_index").getInt() != expected
            || (!scalarMultiply && op->getAttrOfType<mlir::IntegerAttr>("rhs_index").getInt() != expected + 2))
            throw std::logic_error("residual head selected the other hemisphere's input SRs");
        if (scalarMultiply && (op->getAttrOfType<mlir::StringAttr>("rhs_kind").getValue() != "immediate"
                || op->getAttrOfType<mlir::FloatAttr>("rhs_immediate").getValueAsDouble() != 0.25))
            throw std::logic_error("scalar residual must use an immediate operand");
    });
    if (heads != 8)
        throw std::logic_error("residual test did not cover all four layout residues");
}

int main()
{
    test_direction(false);
    test_direction(true);
}
