#include "AttentionEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <iostream>

int main()
{
    using namespace ftlpu::compiler;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<schedule::ScheduleDialect>();

    mlir::IRRewriter rewriter(&context);
    auto module = mlir::ModuleOp::create(rewriter.getUnknownLoc());
    rewriter.setInsertionPointToStart(module.getBody());
    const auto tensor =
        mlir::RankedTensorType::get({32, 32}, rewriter.getF32Type());
    auto function = rewriter.create<mlir::func::FuncOp>(
        rewriter.getUnknownLoc(), "rope_emitter",
        rewriter.getFunctionType({tensor}, {tensor}));
    mlir::Block* entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    target::LPUTargetModel target;
    schedule::attention_detail::emitRopeOrCast(
        rewriter, rewriter.getUnknownLoc(), target, 10, 0, true,
        entry->getArgument(0), rewriter.getBF16Type());
    schedule::attention_detail::emitRopeOrCast(
        rewriter, rewriter.getUnknownLoc(), target, 20, 1, false,
        entry->getArgument(0), rewriter.getBF16Type());

    int ropeInstructions = 0;
    int castInstructions = 0;
    function.walk([&](schedule::VxmOp op) {
        if (op.getCycle() == 10 || op.getCycle() == 11)
            ++ropeInstructions;
        if (op.getCycle() == 20) ++castInstructions;
    });
    if (ropeInstructions != 6 || castInstructions != 2) {
        std::cerr << "unexpected VXM instruction counts: rope="
                  << ropeInstructions << ", cast=" << castInstructions << '\n';
        return 1;
    }
    return 0;
}
