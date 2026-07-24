#include "FfnEmitterUtils.hpp"

#include "ftlpu/compiler/Dialect/Schedule/IR/schedule_dialect.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

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
        rewriter.getUnknownLoc(), "swish_emitter",
        rewriter.getFunctionType({tensor, tensor}, {tensor}));
    mlir::Block* entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    target::LPUTargetModel target;
    auto [local, peer] = schedule::ffn_detail::emitFfnSwishAlu(
        rewriter, rewriter.getUnknownLoc(), tensor,
        entry->getArgument(0), entry->getArgument(1), target,
        FfnScheduleStrategy::Fused, 10, 0);

    int instructions = 0;
    int expInstructions = 0;
    int divideInstructions = 0;
    int castInstructions = 0;
    int64_t lastCycle = 0;
    function.walk([&](schedule::VxmOp op) {
        ++instructions;
        lastCycle = std::max(lastCycle, static_cast<int64_t>(op.getCycle()));
        if (op.getOpcode() == "exp") ++expInstructions;
        if (op.getOpcode() == "divide") ++divideInstructions;
        if (op.getOpcode() == "cast") ++castInstructions;
    });

    require(instructions == 11, "Swish emitter must issue eleven VXM ops");
    require(expInstructions == 1, "Swish emitter must issue one exp");
    require(divideInstructions == 1, "Swish emitter must issue one divide");
    require(castInstructions == 2, "Swish emitter must fan out two casts");
    require(lastCycle == 15, "Swish VXM pipeline must occupy six cycles");
    require(local.getOutputHemisphere() == "east",
        "local Swish output must remain in the source hemisphere");
    require(peer.getOutputHemisphere() == "west",
        "peer Swish output must cross to the other hemisphere");
}
