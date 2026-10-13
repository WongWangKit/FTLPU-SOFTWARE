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
    schedule::attention_detail::emitRopeOrCast(
        rewriter, rewriter.getUnknownLoc(), target, 30, 1, true,
        entry->getArgument(0), rewriter.getBF16Type());

    int ropeInstructions = 0;
    int castInstructions = 0;
    bool sawWestCast15 = false;
    bool sawWestCast14 = false;
    bool sawWestRopeHead = false;
    bool sawWestRopeLowTail = false;
    bool sawWestRopeHighTail = false;
    function.walk([&](schedule::VxmOp op) {
        if (op.getCycle() == 10 || op.getCycle() == 11)
            ++ropeInstructions;
        if (op.getCycle() == 20) ++castInstructions;
        if (op.getCycle() == 20 && op.getQueue() == 15)
            sawWestCast15 = true;
        if (op.getCycle() == 20 && op.getQueue() == 14)
            sawWestCast14 = true;
        if (op.getCycle() == 30 && op.getQueue() == 15)
            sawWestRopeHead = true;
        if (op.getCycle() == 31 && op.getQueue() == 13
            && op.getLhsKind() == "alu" && op.getLhsIndex() == 15
            && op.getRhsKind() == "alu" && op.getRhsIndex() == 14)
            sawWestRopeLowTail = true;
        if (op.getCycle() == 31 && op.getQueue() == 10
            && op.getLhsKind() == "alu" && op.getLhsIndex() == 12
            && op.getRhsKind() == "alu" && op.getRhsIndex() == 11)
            sawWestRopeHighTail = true;
    });
    if (ropeInstructions != 6 || castInstructions != 2
        || !sawWestCast15 || !sawWestCast14) {
        std::cerr << "unexpected VXM instruction counts: rope="
                  << ropeInstructions << ", cast=" << castInstructions << '\n';
        return 1;
    }
    if (!sawWestRopeHead || !sawWestRopeLowTail
        || !sawWestRopeHighTail) {
        std::cerr << "west RoPE does not follow C15-to-C10 propagation\n";
        return 1;
    }
    return 0;
}
